"""Bayesian amplitude calibration of the stations (`bayes_gscale`).

`gscale` solves one amplitude correction per station and IF for the whole observation, against *a* model - and that model was built from the very data being calibrated, so a station with a wrong amplitude scale has already pulled the model towards its own error. `bayes_gscale` removes that circularity and puts error bars on the answer:

1. **Leave one station out.** For every station `a` (and once with the full array) the source model is rebuilt with `a` ignored, phase self-calibrating as it goes. `a` is then brought back, phase self-calibrated against that fixed model, and the whole array `gscale`d. The correction this gives `a` comes from a model it had no say in. The runs that leave out the *other* stations give a jackknife spread for it; and where leaving `a` out moves its own estimate, that shift is counted as systematic uncertainty too, since it can mean either that the full-array model had absorbed the station's error or that the station's baselines reach spatial frequencies no other station constrains.
2. **Compare source models.** The same is done for several model families - one to a few Gaussians or point sources fitted with `modelfit` (the default: one to three Gaussians), or CLEAN - and the full-array runs are weighed against each other by the Bayesian information criterion, ``BIC = chisq / s^2 + k ln(n)``, with `k` the model's parameter count and `s^2` the reduced chi-squared of the best of them (the data weights of VLBI are rarely on an absolute scale, and this is the usual way of not trusting them for it). ``exp(-BIC/2)``, normalised, is each model's posterior probability, and the corrections are averaged over models with those weights.
3. **Correct only what is off.** Each station and IF is either *as calibrated* - its true log-amplitude error drawn from a Gaussian of width `prior_sigma`, the a-priori calibration accuracy (~10% for the EVN) - or *off* - drawn from a much wider one, `outlier_sigma` (a factor of a few: a wrong SEFD, a dead receptor averaged in, a pointing problem). The Bayes factor between the two gives the probability that it is off. An error within the a-priori accuracy is left alone, since the data cannot tell it from the calibration's own scatter and correcting it would only move the good stations; one clearly beyond it gets its estimate almost in full, since the wide prior hardly shrinks it. The correction applied is the posterior average over both: zero if as calibrated, the posterior mean if off.
4. **Keep the flux scale with the median station.** `gscale` keeps the data's flux scale by setting the mean gain amplitude of every IF to one (difmap's ``norm_cors``), so one station far off its a-priori calibration drags every other station of its IFs with it - and leaving it out of the model cannot show it, since it is still in the normalisation. Each run therefore solves with a floating scale and then sets the correction of the median station of every IF to one (``norm="mean"`` restores difmap's behaviour); the findings name a station that would have dragged the mean.

The likelihoods are Gaussian throughout (in log amplitude for the gains), which is what makes each step closed-form and the whole thing cheap enough to run on every station: every run is independent, works on its own copy of the observation, and the heavy lifting (imaging, CLEAN, model fitting, self-calibration) releases the GIL, so the runs go in parallel threads.

The result (`BayesGainResult`) prints a summary, plots its diagnostics, and writes a JSON report and a calibration table of the constant corrections applied - which, having been derived on a well-modelled calibrator, is what can be carried over to other sources of the same observation.
"""

from __future__ import annotations

import math
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from difmapy.report import write_json

__all__ = ["bayes_gscale", "BayesGainResult", "DEFAULT_MODELS"]

#: The source-model families compared by default. CLEAN is left out:
#: with its many degrees of freedom it was often favoured by the model
#: comparison where it should not have been (why is still to be
#: understood), so it has to be asked for (``models=("clean", ...)``).
DEFAULT_MODELS = ("gauss1", "gauss2", "gauss3")
DEFAULT_CLEAN_NITER = 200

# Free parameters that each difmap free-parameter bit stands for
# (flux, x+y, major, ratio, phi, spectral index).
_BIT_NPAR = {1: 1, 2: 2, 4: 1, 8: 1, 16: 1, 32: 1}


def _nfree(mask) -> int:
    return sum(n for bit, n in _BIT_NPAR.items() if int(mask) & bit)


# ----------------------------------------------------------------------
# source models
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """How to build one family of source models."""

    name: str
    kind: str   # "clean", "gauss", "point" or "current"
    n: int      # CLEAN iterations per round, or number of components

    @classmethod
    def parse(cls, spec) -> "ModelSpec":
        if isinstance(spec, ModelSpec):
            return spec
        key = str(spec).strip().lower().replace(" ", "").replace("_", "")
        if key == "current":
            return cls("current", "current", 0)
        m = re.fullmatch(r"clean(\d*)", key)
        if m:
            n = int(m.group(1) or DEFAULT_CLEAN_NITER)
            return cls(f"clean{n}" if m.group(1) else "clean", "clean", n)
        m = re.fullmatch(r"(gauss|gaussian|point|delta)s?(\d+)", key) or \
            re.fullmatch(r"(\d+)(gauss|gaussian|point|delta)s?", key)
        if m:
            a, b = m.groups()
            word, n = (a, b) if a[0].isalpha() else (b, a)
            kind = "gauss" if word.startswith("gauss") else "point"
            n = int(n)
            if n < 1:
                raise ValueError(f"model {spec!r} has no components")
            return cls(f"{kind}{n}", kind, n)
        raise ValueError(
            f"unknown source model {spec!r}; use 'clean' (or 'clean500' for "
            "500 iterations a round), 'gaussN', 'pointN' or 'current'"
        )

    def nparam(self, obs) -> int:
        """Parameters the model spent on fitting the data: the free ones of fitted components, and three (flux, x, y) per CLEAN position."""
        model = obs.model
        if self.kind == "clean" or (self.kind == "current" and
                                    not any(c["freepar"] for c in model)):
            pos = {(round(c["x"], 6), round(c["y"], 6)) for c in model}
            return 3 * len(pos)
        return sum(_nfree(c["freepar"]) for c in model)


def _build_model(o, spec, nloop, solint, clean_gain):
    """Build `spec` on `o`, phase self-calibrating after each round."""
    if spec.kind != "current":
        o.clrmod()
    free_g = ("flux", "pos", "major")
    free_p = ("flux", "pos")
    for r in range(nloop):
        if spec.kind == "clean":
            o.clean(spec.n, clean_gain, verbose=False)
        elif spec.kind in ("gauss", "point"):
            if r == 0:
                for _ in range(spec.n):
                    if spec.kind == "gauss":
                        o.seed_model(type="gauss", free=free_g)
                    else:
                        o.seed_model(type="delta", free=free_p)
                    o.modelfit(verbose=False)
            else:
                o.modelfit(verbose=False)
        elif o.nvariable:
            o.modelfit(verbose=False)
        if not o.model or not np.isfinite(o.model_flux) or o.model_flux <= 0:
            raise RuntimeError("the model came out empty")
        o.selfcal(phase=True, solint=solint, verbose=False, mapstats=False)


def _antenna_chisq(o):
    """Reduced chi-squared of the fit per station, over every baseline it is on (difmap's convention: real and imaginary parts count separately), and the number of visibilities behind it."""
    vis, wt = (np.asarray(x) for x in o._core.stream_vis())
    model = np.asarray(o._core.stream_model())
    sel = o._core.selection()
    good = (wt > 0) & np.asarray(sel["if_used"], dtype=bool)[None, :]
    r2 = np.where(good, wt * np.abs(vis - model) ** 2, 0.0).sum(axis=1)
    n = good.sum(axis=1)
    _, a1, a2, *_ = o._core.rows()
    nant = len(o.antennas)
    chi = np.bincount(a1, r2, nant) + np.bincount(a2, r2, nant)
    cnt = np.bincount(a1, n, nant) + np.bincount(a2, n, nant)
    with np.errstate(invalid="ignore", divide="ignore"):
        return chi / (2.0 * cnt), cnt.astype(int)


# ----------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------


def _median_normalised(logg):
    """Log-gains ``[nif, nant]`` shifted so that the median station of each IF has no correction.

    gscale keeps the data's flux scale by setting the *mean* gain amplitude of every IF to one (difmap's ``norm_cors``), so a single station that is far off - one that needs x4 in the two IFs it has data in - pulls every other station of those IFs down with it (to x0.57 for eight stations), and leaving it out of the model does not help: it is still in the normalisation. The median station is not moved by it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(logg, axis=1, keepdims=True)
    return logg - np.where(np.isfinite(centre), centre, 0.0)


def _model_variance(model_sigma, names, nif):
    """``[nif, nant]`` variances from `bayes_gscale`'s ``model_sigma`` (zero where a station has none)."""
    extra = np.zeros((nif, len(names)))
    for name, value in (model_sigma or {}).items():
        if name not in names or value is None:
            continue
        v = np.asarray(value, float)
        if v.ndim == 0:
            v = np.full(nif, float(v))
        elif v.shape != (nif,):
            raise ValueError(f"model_sigma of {name}: {v.size} values for {nif} IFs")
        extra[:, names.index(name)] = np.where(np.isfinite(v), v, 0.0) ** 2
    return extra


def _one_run(base, spec, exclude, st):
    """Build `spec` without station `exclude` (None: the full array), then gscale everyone against it. Returns the per-(IF, station) log-amplitude corrections and the fit statistics on the full data."""
    t0 = time.perf_counter()
    o = base.copy()
    out = {"model": spec.name, "exclude": exclude, "ok": False}
    try:
        if exclude is not None:
            o.ignore(exclude)
        _build_model(o, spec, st["nloop"], st["solint"], st["clean_gain"])
        if exclude is not None:
            o.unignore(exclude)
            # Phase-calibrate the station just brought back against the
            # model it did not shape, so its amplitudes average coherently.
            o.selfcal(phase=True, solint=st["solint"], verbose=False,
                      mapstats=False)
        keep = o.copy() if (exclude is None and st["keep"]) else None
        before = o.moddif()
        chi_before, _ = _antenna_chisq(o)
        # The median renormalisation is done here, on the floating-scale
        # solution; difmap's own (the mean) is gscale's.
        median_norm = not st["float_scale"] and st["norm"] == "median"
        g = o.gscale(float_scale=st["float_scale"] or median_norm, quiet=False,
                     verbose=False, mapstats=False)
        after = o.moddif()
        chi_after, nvis_ant = _antenna_chisq(o)
        amp = np.array([g["gains_per_if"][n] for n in o.antennas], float).T
        with np.errstate(invalid="ignore", divide="ignore"):
            logg = np.where(amp > 0, np.log(amp), np.nan)
        if median_norm:
            logg = _median_normalised(logg)
        if not np.isfinite(after["chisq"]):
            raise RuntimeError("the fit statistics are not finite")
        out.update(
            ok=True,
            logg=logg,                       # [nif, nant]
            chisq=float(after["chisq"]),
            chisq_before=float(before["chisq"]),
            ndata=int(after["ndata"]),
            nparam=spec.nparam(o),
            ncomp=len(o.model),
            model_flux=float(o.model_flux),
            antenna_rchisq_before=chi_before,
            antenna_rchisq=chi_after,
            antenna_nvis=nvis_ant,
            state=keep,
        )
    except Exception as exc:  # noqa: BLE001 - a failed run is reported
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["seconds"] = time.perf_counter() - t0
    return out


# ----------------------------------------------------------------------
# the statistics
# ----------------------------------------------------------------------


def _model_posterior(full_runs, model_names):
    """BIC-based posterior probability of each model family, from the full-array runs, with the chi-squared rescaled by the best model's reduced chi-squared."""
    rows = []
    for name in model_names:
        r = full_runs.get(name)
        if r is None or not r["ok"]:
            rows.append(None)
            continue
        rows.append((r["chisq"], r["ndata"], r["nparam"]))
    red = [c / max(n - k, 1) for (c, n, k) in (x for x in rows if x)]
    if not red:
        raise RuntimeError("no source model could be built from the full array")
    s2 = min(red)
    bic = np.array([np.nan if x is None else x[0] / s2 + x[2] * math.log(x[1])
                    for x in rows])
    ok = np.isfinite(bic)
    d = bic - np.nanmin(bic)
    post = np.where(ok, np.exp(-0.5 * np.where(ok, d, 0.0)), 0.0)
    post /= post.sum()
    return post, bic, d, s2, red


@dataclass
class BayesGainResult:
    """What `bayes_gscale` found; see the module docstring for the method. Log-gain arrays are natural logs of the amplitude corrections (which multiply the data), ``[nif, nant]`` unless said otherwise."""

    source: str
    antennas: list
    if_freqs: list
    models: list
    settings: dict
    model_prob: np.ndarray        # [nmodel]
    model_bic: np.ndarray         # [nmodel]
    model_dbic: np.ndarray
    model_rchisq: list            # reduced chisq per model (full array)
    noise_scale: float            # s^2 used to rescale chisq
    runs: list                    # per-run summaries (no arrays)
    logg: np.ndarray              # [nmodel, 1 + nant, nif, nant]; NaN = none
    loo: np.ndarray               # [nmodel, nif, nant] leave-own-out estimate
    loo_sigma: np.ndarray         # [nmodel, nif, nant] jackknife sigma
    naive: np.ndarray             # [nif, nant] gscale of the best full model
    mean: np.ndarray              # model-averaged estimate
    sigma: np.ndarray
    post_mean: np.ndarray         # if the station is off (the wide prior)
    post_sigma: np.ndarray
    log10_bf: np.ndarray          # log10 Bayes factor, needed : not needed
    p_correction: np.ndarray      # posterior probability it is off, so corrected
    applied_log: np.ndarray       # the log corrections applied
    applied_sigma: np.ndarray
    influence: np.ndarray         # [nant(excluded), nif, nant] shift, best model
    antenna_rchisq: dict          # before/after/naive, [nant] each
    fit: dict                     # whole-data fit statistics
    applied: bool = False
    elapsed: float = 0.0
    files: dict = field(default_factory=dict)

    # -- derived views ----------------------------------------------------

    @property
    def factors(self) -> np.ndarray:
        """The amplitude corrections applied, ``[nif, nant]``."""
        return np.exp(np.where(np.isfinite(self.applied_log),
                               self.applied_log, 0.0))

    @property
    def best_model(self) -> str:
        return self.models[int(np.argmax(self.model_prob))]

    def _if_label(self, i):
        return f"IF{i + 1}"

    def _per_if(self, a):
        """The per-IF numbers of station `a` (None where it has no data in that IF)."""
        bi = int(np.argmax(self.model_prob))
        out = []
        for i, freq in enumerate(self.if_freqs):
            lg = self.applied_log[i, a]
            if not np.isfinite(lg):
                out.append({"if": i + 1, "freq_hz": freq, "correction": None})
                continue

            def fac(x):
                return math.exp(x) if np.isfinite(x) else None

            out.append({
                "if": i + 1,
                "freq_hz": freq,
                "correction": math.exp(lg),
                "sigma": float(self.applied_sigma[i, a]),
                "p_correction": float(self.p_correction[i, a]),
                "log10_bayes_factor": float(self.log10_bf[i, a]),
                "naive_gscale": fac(self.naive[i, a]),
                "loo_estimate": fac(self.mean[i, a]),
                "loo_shift": fac(self.loo[bi, i, a] - self.logg[bi, 0, i, a]),
            })
        return out

    def station_table(self) -> list[dict]:
        """One row per station. `per_if` holds, for every IF, the correction applied, its uncertainty, the probability that it is needed (and the log10 Bayes factor), the plain gscale value, the leave-one-out estimate and the factor by which leaving the station out moved it. With ``per_if=False`` the IFs share one correction; otherwise the station-level `correction`, `sigma` and `p_correction` are only medians over IFs, for sorting - the per-IF values are the ones applied."""
        rows = []
        for a, name in enumerate(self.antennas):
            per_if = self._per_if(a)
            ok = [d for d in per_if if d["correction"] is not None]

            def med(key, log=False):
                v = [d[key] for d in ok if d.get(key) is not None]
                if not v:
                    return float("nan")
                v = np.log(v) if log else np.asarray(v)
                m = float(np.median(v))
                return math.exp(m) if log else m

            rows.append({
                "station": name,
                "has_data": bool(ok),
                "correction": med("correction", log=True),
                "sigma": med("sigma"),
                "p_correction": med("p_correction"),
                "naive_gscale": med("naive_gscale", log=True),
                "loo_estimate": med("loo_estimate", log=True),
                "loo_shift": med("loo_shift", log=True),
                "rchisq_before": float(self.antenna_rchisq["before"][a]),
                "rchisq_after": float(self.antenna_rchisq["after"][a]),
                "per_if": per_if,
            })
        return rows

    @staticmethod
    def _mark(p):
        """* for a correction needed (P >= 0.95), ? for probably needed (P >= 0.75)."""
        return "*" if p >= 0.95 else "?" if p >= 0.75 else " "

    def _normalisation_findings(self) -> list[str]:
        """The stations that would have dragged a mean renormalisation of the flux scale (difmap's) by more than 5% in some IF on their own: in each such IF, the station without which the drag would be less than half, with its difmap gscale there, what every other station would have been scaled by, and what was applied to it instead (a station the data do not constrain gets a wild gscale and is left alone)."""
        if self.settings.get("float_scale") or self.settings.get("norm", "mean") != "median":
            return []
        dragged: dict = {}
        for i, row in enumerate(np.asarray(self.naive, float)):
            ok = np.flatnonzero(np.isfinite(row))
            if ok.size < 3:
                continue
            drag = math.log(float(np.mean(np.exp(row[ok]))))
            if abs(drag) < 0.05:
                continue
            without = {a: abs(math.log(float(np.mean(np.exp(row[ok[ok != a]]))))) for a in ok}
            a = min(without, key=without.get)
            if without[a] >= 0.5 * abs(drag):
                continue
            dragged.setdefault(int(a), []).append((i, math.exp(row[a]), math.exp(-drag)))
        return [f"{self.antennas[a]}: difmap's gscale gives it "
                + ", ".join(f"x{c:.2f} in {self._if_label(i)}" for i, c, _ in items)
                + ", and its mean renormalisation of the flux scale would have scaled every other station "
                "there by " + ", ".join(f"x{d:.2f}" for _, _, d in items)
                + " - the median station holds the scale instead; applied to it: "
                + ", ".join(f"x{math.exp(self.applied_log[i, a]):.2f} (P = {self.p_correction[i, a]:.2f})"
                            for i, _, _ in items)
                for a, items in dragged.items()]

    def findings(self) -> list[str]:
        """The notable conclusions, in words."""
        out = []
        bi = int(np.argmax(self.model_prob))
        out.append(
            f"Best source model: {self.models[bi]} "
            f"(posterior probability {self.model_prob[bi]:.3f})"
        )
        shared = not self.settings.get("per_if", True) or len(self.if_freqs) == 1
        for row in self.station_table():
            if not row["has_data"]:
                continue
            name = row["station"]
            ok = [d for d in row["per_if"] if d["correction"] is not None]
            for level, how in ((0.95, "needed"), (0.75, "probably needed")):
                hi = math.inf if level == 0.95 else 0.95
                sel = [d for d in ok if level <= d["p_correction"] < hi]
                if not sel:
                    continue
                if shared:
                    d = sel[0]
                    out.append(
                        f"{name}: a correction is {how} (P = "
                        f"{d['p_correction']:.3f}); applied x"
                        f"{d['correction']:.3f} +- {100 * d['sigma']:.1f}% "
                        "in every IF")
                else:
                    ifs = ", ".join(
                        f"{self._if_label(d['if'] - 1)} x{d['correction']:.3f}"
                        f" +- {100 * d['sigma']:.1f}% (P = "
                        f"{d['p_correction']:.3f})" for d in sel)
                    out.append(f"{name}: a correction is {how} in {ifs}")
            shift, sig = row["loo_shift"], row["sigma"]
            if np.isfinite(shift) and abs(math.log(shift)) > max(3 * sig, 0.02):
                out.append(
                    f"{name}: leaving it out of the model moves its gain by "
                    f"{100 * (shift - 1):+.1f}% (median over IFs) - either "
                    "the full-array model had absorbed part of its error, "
                    "or its baselines reach spatial frequencies the other "
                    "stations do not constrain; the shift is counted in its "
                    "uncertainty"
                )
        out += self._normalisation_findings()
        worst = int(np.nanargmax(np.nan_to_num(
            self.antenna_rchisq["after"], nan=-np.inf)))
        out.append(
            f"Worst-fitting station after calibration: "
            f"{self.antennas[worst]} (reduced chi-squared "
            f"{self.antenna_rchisq['after'][worst]:.3g} vs a median of "
            f"{np.nanmedian(self.antenna_rchisq['after']):.3g})"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            net = np.nanmean(self.applied_log, axis=1)
        out.append("Net flux-scale change: " + ", ".join(
            f"{self._if_label(i)} {100 * (math.exp(v) - 1):+.2f}%"
            for i, v in enumerate(net) if np.isfinite(v)))
        nfail = sum(not r["ok"] for r in self.runs)
        if nfail:
            out.append(f"{nfail} of {len(self.runs)} runs failed and were "
                       "left out (see `runs`)")
        return out

    def summary(self) -> str:
        """A plain-text report of the findings: the model comparison, each station's correction in every IF, and the findings."""
        lines = [
            f"bayes_gscale: {self.source}, {len(self.antennas)} stations, "
            f"{len(self.if_freqs)} IFs; {len(self.runs)} runs in "
            f"{self.elapsed:.1f} s",
            "",
            "Source models (full array):",
            f"  {'model':<10}{'P(model)':>10}{'dBIC':>12}{'rchisq':>10}"
            f"{'params':>8}",
        ]
        full = {r["model"]: r for r in self.runs if r["exclude"] is None}
        for i, m in enumerate(self.models):
            r = full.get(m, {})
            lines.append(
                f"  {m:<10}{self.model_prob[i]:>10.3g}"
                f"{self.model_dbic[i]:>12.4g}{self.model_rchisq[i]:>10.4g}"
                f"{r.get('nparam', 0):>8}"
            )
        shared = not self.settings.get("per_if", True) or len(self.if_freqs) == 1
        cols = ([("all IFs", None)] if shared else
                [(f"{self._if_label(i)} {f / 1e9:.3f}G", i)
                 for i, f in enumerate(self.if_freqs)])
        width = 17
        table = self.station_table()
        lines += [
            "",
            "Station corrections (factors multiply the data; +- 1 sigma; "
            "* off its calibration, P >= 0.95; ? probably, P >= 0.75;",
            f"a-priori accuracy {100 * self.settings['prior_sigma']:.0f}%, left alone within it):",
            f"  {'station':<9}" + "".join(f"{c:>{width}}" for c, _ in cols)
            + f"{'rchi2 before':>14}{'after':>8}",
        ]
        for row in table:
            if not row["has_data"]:
                lines.append(f"  {row['station']:<9}{'-':>{width}}  (no usable "
                             "data; left at 1)")
                continue
            cells = []
            for _, i in cols:
                d = row["per_if"][0 if i is None else i]
                if d["correction"] is None:
                    cells.append(f"{'-':>{width}}")
                else:
                    cell = (f"{d['correction']:.3f}+-{100 * d['sigma']:.1f}%"
                            f"{self._mark(d['p_correction'])}")
                    cells.append(f"{cell:>{width}}")
            lines.append(
                f"  {row['station']:<9}" + "".join(cells)
                + f"{row['rchisq_before']:>14.4g}{row['rchisq_after']:>8.4g}")
        # How the estimates behind the corrections compare.
        lines += [
            "",
            "Estimates behind them (plain gscale / leave-one-out, before "
            "the prior):",
            f"  {'station':<9}" + "".join(f"{c:>{width}}" for c, _ in cols),
        ]
        for row in table:
            if not row["has_data"]:
                continue
            cells = []
            for _, i in cols:
                # One shared correction: the IF-combined estimates.
                d = row if i is None else row["per_if"][i]
                naive, loo = d.get("naive_gscale"), d.get("loo_estimate")
                if d.get("correction") is None or not naive or \
                        not np.isfinite(naive):
                    cells.append(f"{'-':>{width}}")
                else:
                    cell = (f"{naive:.3f} / "
                            + (f"{loo:.3f}" if loo and np.isfinite(loo)
                               else "-"))
                    cells.append(f"{cell:>{width}}")
            lines.append(f"  {row['station']:<9}" + "".join(cells))
        lines += ["", "Findings:"] + [f"  - {f}" for f in self.findings()]
        if self.applied:
            lines.append("")
            lines.append("The corrections have been applied to the data"
                         + (", per IF." if not shared else "."))
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def to_dict(self) -> dict:
        """Everything, as plain data (what `write` saves as JSON)."""
        return {
            "source": self.source,
            "method": "bayes_gscale (leave-one-station-out gscale, BIC "
                      "model averaging, Gaussian log-gain prior)",
            "settings": self.settings,
            "antennas": self.antennas,
            "if_freqs_hz": self.if_freqs,
            "models": [
                {"name": m, "probability": self.model_prob[i],
                 "bic": self.model_bic[i], "dbic": self.model_dbic[i],
                 "rchisq": self.model_rchisq[i]}
                for i, m in enumerate(self.models)
            ],
            "best_model": self.best_model,
            "noise_scale": self.noise_scale,
            "stations": self.station_table(),
            "findings": self.findings(),
            "fit": self.fit,
            "corrections_per_if": {
                name: [float(v) for v in self.factors[:, a]]
                for a, name in enumerate(self.antennas)
            },
            "log_gain": {
                "loo": self.loo, "loo_sigma": self.loo_sigma,
                "naive": self.naive, "mean": self.mean, "sigma": self.sigma,
                "post_mean": self.post_mean, "post_sigma": self.post_sigma,
                "log10_bayes_factor": self.log10_bf,
                "p_correction": self.p_correction,
                "applied": self.applied_log, "applied_sigma": self.applied_sigma,
                "influence_best_model": self.influence,
            },
            "antenna_rchisq": self.antenna_rchisq,
            "runs": self.runs,
            "applied": self.applied,
            "elapsed_s": self.elapsed,
            "files": self.files,
        }

    def plot(self, path=None, show=None):
        """The diagnostic figure (see `difmapy.bayesplot`); saved to `path` if given. Returns the matplotlib figure."""
        from difmapy.bayesplot import plot_bayes_gscale

        return plot_bayes_gscale(self, path=path, show=show)

    def write(self, prefix, obs=None, outformat=None, verbose=True, **kwargs):
        """Write ``<prefix>.json`` (the report), ``<prefix>.png`` (the diagnostics) and, given the observation, the calibration table of the corrections (`savecaltable` with these constant gains; extra keyword arguments go to it). Returns the paths written."""
        files = {}
        self.files = files
        parent = os.path.dirname(os.fspath(prefix))
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            self.plot(path=f"{prefix}.png", show=False)
            files["plot"] = f"{prefix}.png"
        except ImportError as exc:
            if verbose:
                print(f"bayes_gscale: no diagnostic plot ({exc})")
        # The report first, so that a table that cannot be written (no
        # MS for a CASA table, say) does not lose it.
        files["json"] = f"{prefix}.json"
        write_json(files["json"], self.to_dict())
        if obs is not None:
            files["caltable"] = self.save_caltable(
                obs, prefix, outformat=outformat, verbose=verbose, **kwargs)
            write_json(files["json"], self.to_dict())
        return files

    def save_caltable(self, obs, path, outformat=None, verbose=True, **kwargs):
        """The constant corrections as a CASA and/or AIPS table (see `Observation.savecaltable`), for applying to other sources."""
        from difmapy.io.solutions import constant

        fmts = obs.caltable_formats(outformat, ms=kwargs.get("ms"),
                                    uvfits=kwargs.get("uvfits"))
        paths = obs.caltable_paths(path, fmts)
        if not str(path).lower().endswith(".fits"):
            # A prefix: name each table by its kind.
            paths = {f: f"{path}.G" if f == "casa" else f"{path}.TASAV.FITS"
                     for f in fmts}
        has = np.isfinite(self.applied_log)
        sol = constant(obs, np.exp(np.where(has, self.applied_log, 0.0)),
                       used=has)
        written = []
        for fmt in fmts:
            info = obs.savecaltable(paths[fmt], outformat=fmt, solutions=sol,
                                    quiet=False, verbose=verbose, **kwargs)
            written.append(info["path"])
        return written


def _combine_ifs(mean, var):
    """One estimate per station from its per-IF ones: the inverse-variance mean, with the variance of fully correlated estimates (they share a model) plus the scatter between IFs."""
    w = np.where(np.isfinite(mean) & np.isfinite(var) & (var > 0), 1 / var, 0.0)
    sw = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(sw > 0, (w * np.nan_to_num(mean)).sum(axis=0) / sw, np.nan)
        v_corr = np.where(sw > 0, (w * np.nan_to_num(var)).sum(axis=0) / sw, np.nan)
        scatter = np.where(
            sw > 0, (w * (np.nan_to_num(mean) - m) ** 2).sum(axis=0) / sw, 0.0)
        nif = np.maximum((w > 0).sum(axis=0), 1)
    v = v_corr + scatter / nif
    shape = mean.shape
    return np.broadcast_to(m, shape).copy(), np.broadcast_to(v, shape).copy()


# ----------------------------------------------------------------------
# the driver
# ----------------------------------------------------------------------


def bayes_gscale(obs, models=DEFAULT_MODELS, prior_sigma=0.10, outlier_sigma=1.0, jackknife=True,
                 per_if=True, float_scale=False, norm="median", nloop=2, solint=0.0,
                 clean_gain=0.05, sigma_floor=0.01, model_sigma=None, timeavg=None,
                 workers=None, apply=True, prefix=None, outformat=None,
                 ms=None, uvfits=None, plot=None,
                 quiet=True, verbose=True) -> BayesGainResult:
    """Bayesian whole-observation amplitude calibration of every station; the method is described in `difmapy.bayescal`.

    Parameters
    ----------
    models : sequence of str
        Source-model families to compare: ``"clean"`` (``"clean500"`` for 500 iterations a round), ``"gaussN"``/``"pointN"`` (N circular Gaussians / point sources, seeded at the residual peak and fitted with `modelfit`), and ``"current"`` (the model in the observation now, refitted where it has free parameters; being built from all stations, its leave-one-out estimates are not independent).
    prior_sigma : float
        The fractional accuracy of the a-priori amplitude calibration: width of the log-amplitude errors of a station *as calibrated*, which are left alone.
    outlier_sigma : float
        Width of the log-amplitude errors of a station that is *off* (1.0: a factor of a few), which are corrected almost in full.
    jackknife : bool
        Do the leave-one-station-out runs. Without them each station's estimate comes from the full-array model and its uncertainty from the spread between models only.
    per_if : bool
        True (the default): every IF gets its own correction, as `gscale` solves them, with its own uncertainty and P(needed) - the summary, the figure and the tables show and apply one per station and IF. False: the IFs' estimates are combined into one correction per station, applied to every IF.
    float_scale : bool
        False keeps the data's flux scale (see `norm`); True lets it float onto the model's.
    norm : {"median", "mean"}
        How the flux scale is kept with ``float_scale=False``: the median station of each IF gets no correction (the default), or the mean gain amplitude of each IF is one, as difmap's `gscale` does. One station far off its a-priori calibration drags the mean, and so every other station of its IFs, with it - and leaving it out of the model does not help, since it is still in the normalisation. It does not move the median.
    nloop : int
        Rounds of model building and phase self-calibration per run.
    solint : float | str
        Phase self-calibration interval (as for `selfcal`).
    clean_gain : float
        CLEAN loop gain for the ``clean`` models.
    sigma_floor : float
        Smallest uncertainty assumed for a log-gain (systematics the jackknife cannot see).
    model_sigma : dict | None
        Station -> uncertainty of its log-gain that comes from the source model (one value, or one per IF; NaN or None for none), added in quadrature. For a model brought from elsewhere (``"current"``) the jackknife cannot measure it: the model is not rebuilt without the station, so one whose baselines reach spatial frequencies no other station samples gets a confident correction that only reflects the model there. The ``sigma`` an earlier run with the station's own models found is that measure.
    timeavg : float | str | None
        Average the working data in time first (as `uvaver`); the corrections are constant, so this costs little and speeds up long observations.
    workers : int | None
        Threads to use (default: all CPUs).
    apply : bool
        Apply the corrections to `obs` (as `gscale` would).
    prefix : str | None
        Write ``<prefix>.json``, ``<prefix>.png`` and the calibration table (in `outformat`, default the data's native one; `ms` and `uvfits` are the files it refers to, as for `savecaltable`).
    plot : bool | None
        Show the diagnostic figure (default: only when nothing is written and `verbose`).
    quiet : bool
        Return None (the default). ``False`` returns the `BayesGainResult`.
    verbose : bool
        Print the progress and the summary.

    Returns
    -------
    BayesGainResult | None
        The result, with ``quiet=False``.
    """
    from difmapy.observation import MAS

    t_start = time.perf_counter()
    specs = [ModelSpec.parse(m) for m in models]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("the same model is listed twice")
    if not specs:
        raise ValueError("no source models to compare")
    if prior_sigma <= 0:
        raise ValueError("prior_sigma must be positive")
    if outlier_sigma <= prior_sigma:
        raise ValueError("outlier_sigma must be larger than prior_sigma")
    if norm not in ("median", "mean"):
        raise ValueError(f"norm must be 'median' or 'mean', got {norm!r}")

    base = obs.copy()
    if timeavg:
        base = base.uvaver(timeavg)
    if not base._mapsize_set:
        # The model only needs the source, not a survey-size field.
        base.mapsize(512, base.estimated_resolution() / 4.0)
    names = list(base.antennas)
    nant, nif = len(names), base.nif

    # Stations with data (and not already ignored) are the ones to leave
    # out in turn.
    _, a1, a2, *_ = base._core.rows()
    flags = np.asarray(base.flags)
    usable = ~flags.all(axis=(1, 2))
    present = np.zeros(nant, bool)
    present[np.asarray(a1)[usable]] = True
    present[np.asarray(a2)[usable]] = True
    excluded = [n for i, n in enumerate(names)
                if present[i] and n not in base.ignored]
    subsets = [None] + (excluded if jackknife else [])

    st = {"nloop": int(nloop), "solint": solint, "clean_gain": float(clean_gain),
          "float_scale": bool(float_scale), "norm": norm, "keep": True}
    jobs = [(s, e) for s in specs for e in subsets]
    nworkers = max(1, min(int(workers or os.cpu_count() or 1), len(jobs)))
    if verbose:
        print(f"bayes_gscale: {len(jobs)} runs ({len(specs)} models x "
              f"{len(subsets)} station subsets) on {nworkers} threads")
    results = []
    with ThreadPoolExecutor(max_workers=nworkers) as pool:
        futs = [pool.submit(_one_run, base, s, e, st) for s, e in jobs]
        for f in as_completed(futs):
            results.append(f.result())

    model_names = [s.name for s in specs]
    mi = {m: i for i, m in enumerate(model_names)}
    ai = {n: i for i, n in enumerate(names)}
    logg = np.full((len(specs), 1 + nant, nif, nant), np.nan)
    full_runs = {}
    for r in results:
        if r["exclude"] is None:
            full_runs[r["model"]] = r
        if r["ok"]:
            e = 0 if r["exclude"] is None else 1 + ai[r["exclude"]]
            logg[mi[r["model"]], e] = r["logg"]

    prob, bic, dbic, s2, red = _model_posterior(full_runs, model_names)
    rchisq = []
    for m in model_names:
        r = full_runs.get(m)
        rchisq.append(r["chisq"] / max(r["ndata"] - r["nparam"], 1)
                      if r and r["ok"] else float("nan"))

    # Per model: each station's leave-own-out estimate and its jackknife
    # spread over the runs that left out someone else.
    floor2 = float(sigma_floor) ** 2
    loo = np.full((len(specs), nif, nant), np.nan)
    loo_var = np.full((len(specs), nif, nant), np.nan)
    for m in range(len(specs)):
        for a in range(nant):
            own = logg[m, 1 + a, :, a]
            loo[m, :, a] = own if np.isfinite(own).any() else logg[m, 0, :, a]
            others = np.delete(logg[m, 1:, :, a], a, axis=0)   # [nant-1, nif]
            n = np.isfinite(others).sum(axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mu = np.nanmean(others, axis=0)
            ss = np.nansum((others - mu) ** 2, axis=0)
            jk = np.where(n >= 2, (n - 1) / np.maximum(n, 1) * ss, np.nan)
            # Where leaving the station out moves its own estimate, the
            # data cannot say whether the full-array model absorbed its
            # error or the station samples spatial frequencies nobody
            # else constrains; count the disagreement as systematic
            # uncertainty (the full-array value then sits at 2 sigma).
            disc = loo[m, :, a] - logg[m, 0, :, a]
            disc2 = np.where(np.isfinite(disc), 0.25 * disc ** 2, 0.0)
            loo_var[m, :, a] = np.where(np.isfinite(jk), jk, 0.0) + floor2 + disc2

    # Model averaging, over the models that have an estimate.
    w = prob[:, None, None] * np.isfinite(loo)
    sw = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(sw > 0, (w * np.nan_to_num(loo)).sum(axis=0) / sw, np.nan)
        var = np.where(sw > 0, (w * (np.nan_to_num(loo_var)
                                     + (np.nan_to_num(loo) - mean) ** 2)
                                ).sum(axis=0) / sw, np.nan)
    var = var + _model_variance(model_sigma, names, nif)
    if not per_if:
        mean, var = _combine_ifs(mean, var)

    # The two hypotheses: as calibrated (errors within the a-priori
    # accuracy, left alone) or off (a wide prior, corrected).
    tau0 = float(prior_sigma) ** 2
    tau1 = float(outlier_sigma) ** 2
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        post_mean = mean * tau1 / (tau1 + var)
        post_var = tau1 * var / (tau1 + var)
        # ln BF = ln N(mean; 0, var + tau1) - ln N(mean; 0, var + tau0)
        ln_bf = 0.5 * np.log((var + tau0) / (var + tau1)) \
            + 0.5 * mean ** 2 * (1.0 / (var + tau0) - 1.0 / (var + tau1))
        p1 = 0.5 * (1.0 + np.tanh(0.5 * ln_bf))   # logistic, overflow-free
        applied = p1 * post_mean
        applied_var = p1 * (post_var + post_mean ** 2) - applied ** 2

    best = int(np.argmax(prob))
    naive = logg[best, 0]
    influence = logg[best, 1:] - logg[best, 0][None]
    for a in range(nant):
        influence[a, :, a] = np.nan  # its own gain: see loo instead

    # Station fits: before gscale, after the plain gscale, and after the
    # Bayesian corrections, all against the best model.
    ref = full_runs[model_names[best]]
    state = ref.get("state")
    fit = {"model": model_names[best],
           "before": {"chisq": ref["chisq_before"], "ndata": ref["ndata"]},
           "naive_gscale": {"chisq": ref["chisq"], "ndata": ref["ndata"]}}
    ant_after = ref["antenna_rchisq"]
    if state is not None:
        fac = np.exp(np.where(np.isfinite(applied), applied, 0.0))
        state._core.apply_gain_factors(np.ascontiguousarray(fac, np.float32))
        state._dirty()
        d = state.moddif()
        fit["bayes"] = {"chisq": d["chisq"], "ndata": d["ndata"]}
        ant_after, _ = _antenna_chisq(state)
    antenna_rchisq = {"before": ref["antenna_rchisq_before"],
                      "naive": ref["antenna_rchisq"], "after": ant_after}

    runs = [{k: v for k, v in r.items()
             if k not in ("logg", "state", "antenna_rchisq_before",
                          "antenna_rchisq", "antenna_nvis")}
            for r in sorted(results, key=lambda r: (
                mi[r["model"]], -1 if r["exclude"] is None else ai[r["exclude"]]))]

    res = BayesGainResult(
        source=obs.source, antennas=names,
        if_freqs=[f for (f, _, _) in obs._core.ifs], models=model_names,
        settings={"models": model_names, "prior_sigma": float(prior_sigma),
                  "outlier_sigma": float(outlier_sigma),
                  "jackknife": bool(jackknife), "per_if": bool(per_if),
                  "float_scale": bool(float_scale), "norm": norm, "nloop": int(nloop),
                  "solint": solint, "clean_gain": float(clean_gain),
                  "sigma_floor": float(sigma_floor),
                  "model_sigma": {str(k): (None if v is None else np.asarray(v, float).tolist())
                                  for k, v in (model_sigma or {}).items()},
                  "timeavg": timeavg,
                  "workers": nworkers,
                  "mapsize": [base._nx, base._xinc / MAS]},
        model_prob=prob, model_bic=bic, model_dbic=dbic, model_rchisq=rchisq,
        noise_scale=float(s2), runs=runs, logg=logg, loo=loo,
        loo_sigma=np.sqrt(loo_var), naive=naive, mean=mean, sigma=np.sqrt(var),
        post_mean=post_mean, post_sigma=np.sqrt(post_var), log10_bf=ln_bf / math.log(10.0),
        p_correction=p1, applied_log=applied,
        applied_sigma=np.sqrt(np.maximum(applied_var, 0.0)),
        influence=influence, antenna_rchisq=antenna_rchisq, fit=fit,
    )
    if apply:
        fac = np.exp(np.where(np.isfinite(applied), applied, 0.0))
        obs._forget_solution_flags()
        obs._core.apply_gain_factors(np.ascontiguousarray(fac, np.float32))
        obs._dirty()
        res.applied = True
    res.elapsed = time.perf_counter() - t_start
    if prefix:
        extra = {k: v for k, v in (("ms", ms), ("uvfits", uvfits)) if v}
        res.write(prefix, obs=obs, outformat=outformat, verbose=verbose, **extra)
    if verbose:
        print(res.summary())
        if res.files:
            print("Wrote " + ", ".join(
                p if isinstance(p, str) else ", ".join(p)
                for p in res.files.values()))
    if plot or (plot is None and not prefix and verbose):
        try:
            res.plot(show=True)
        except ImportError as exc:
            if verbose:
                print(f"bayes_gscale: no diagnostic plot ({exc})")
    return None if quiet else res
