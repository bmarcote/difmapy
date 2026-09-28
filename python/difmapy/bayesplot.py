"""The diagnostic figure of `bayes_gscale`, drawn with matplotlib.

It is a static report figure (saved next to the JSON report), not one
of the interactive pyqtgraph windows, so it needs neither Qt nor a
display. Five panels:

* the correction per station: the plain full-array `gscale`, the
  leave-one-out estimate, and the Bayesian value applied with its
  uncertainty, against the prior band;
* the probability that each station needs a correction;
* the evidence for each source model (BIC difference, log scale);
* the leave-one-out influence matrix: how much each station's gain
  moves when another station is left out of the model;
* each station's fit to the model before and after calibration.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["plot_bayes_gscale"]

# Reference palette (light chart surface), categorical slots 1-3 in
# order, which validate all-pairs for colour-vision deficiency.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL = "#f0efec"
DIVERGING = ["#104281", "#2a78d6", "#86b6ef", NEUTRAL,
             "#f4a3a2", "#e34948", "#9f1f1f"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=AXIS)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)


def _median_ifs(x):
    """Median over IFs ([nif, nant] -> [nant]), NaN where none."""
    out = np.full(x.shape[-1], np.nan)
    for a in range(x.shape[-1]):
        v = x[..., a][np.isfinite(x[..., a])]
        if v.size:
            out[a] = np.median(v)
    return out


#: Settings the figure is drawn with whatever the user's matplotlibrc
#: says: LaTeX text would read every "%" as a comment, and ticks on all
#: four sides clutter a multi-panel report.
RC = {
    "text.usetex": False,
    "font.family": "sans-serif",
    "mathtext.default": "regular",
    "xtick.top": False, "ytick.right": False,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "xtick.direction": "out", "ytick.direction": "out",
}


def plot_bayes_gscale(res, path=None, show=None):
    """Draw the diagnostics of a `BayesGainResult`; save to `path` if
    given, show it if `show` (default: when not saving). Returns the
    figure."""
    import matplotlib

    with matplotlib.rc_context(RC):
        return _draw(res, path, show)


def _draw(res, path, show):
    try:
        import matplotlib
        if show is False or (path and show is None):
            matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "the bayes_gscale figure needs matplotlib (pip install matplotlib)"
        ) from exc

    has = np.isfinite(res.applied_log).any(axis=0)
    idx = np.nonzero(has)[0]
    names = [res.antennas[i] for i in idx]
    x = np.arange(len(idx))
    bi = int(np.argmax(res.model_prob))

    fig = plt.figure(figsize=(12, 12), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1, 1.1], hspace=0.42,
                          wspace=0.24, left=0.07, right=0.97, top=0.93,
                          bottom=0.06)
    fig.suptitle(
        f"bayes_gscale - {res.source}: best model {res.best_model} "
        f"(P = {res.model_prob[bi]:.3f}), prior sigma "
        f"{100 * res.settings['prior_sigma']:.0f}%",
        color=INK, fontsize=12, x=0.07, ha="left")

    # Every IF of a station side by side (IF1 on the left), unless the
    # IFs share one correction.
    per_if = bool(res.settings.get("per_if", True)) and len(res.if_freqs) > 1
    ifs = list(range(len(res.if_freqs))) if per_if else [0]
    nslot = len(ifs)
    step = 0.8 / nslot
    xs = x[:, None] + (np.arange(nslot)[None, :] - (nslot - 1) / 2) * step

    def pick(arr):
        """[len(idx), nslot] of a [nif, nant] array."""
        return arr[ifs][:, idx].T

    # -- corrections --------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    ax.grid(False, axis="x")
    tau = res.settings["prior_sigma"]
    ax.axhspan(math.exp(-tau), math.exp(tau), color=NEUTRAL, zorder=0,
               label=f"prior +-1 sigma ({100 * tau:.0f}%)")
    ax.axhline(1.0, color=AXIS, linewidth=1, zorder=1)
    for k in range(1, len(idx)):
        ax.axvline(k - 0.5, color=GRID, linewidth=0.8, zorder=0)
    naive, loo = np.exp(pick(res.naive)), np.exp(pick(res.mean))
    app, sig = pick(res.applied_log), pick(res.applied_sigma)
    off = 0.28 * step
    ms = 8 if nslot <= 2 else 6
    ax.plot((xs - off).ravel(), naive.ravel(), "o", ms=ms, mfc="none",
            mec=ORANGE, mew=1.8, label="gscale, full-array model", zorder=3)
    ax.plot(xs.ravel(), loo.ravel(), "s", ms=ms, color=AQUA, mec=SURFACE,
            mew=1, label="leave-one-out (model-averaged)", zorder=3)
    ax.errorbar((xs + off).ravel(), np.exp(app).ravel(),
                yerr=[(np.exp(app) - np.exp(app - sig)).ravel(),
                      (np.exp(app + sig) - np.exp(app)).ravel()],
                fmt="D", ms=ms, color=BLUE, mec=SURFACE, mew=1, elinewidth=2,
                capsize=0, label="applied (Bayesian) +-1 sigma", zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(x, names)
    ax.set_xlim(-0.5, len(idx) - 0.5)
    ax.set_ylabel("amplitude correction (multiplies the data)", color=INK2,
                  fontsize=9)
    what = (f"per IF - IF1 ... IF{nslot} left to right in each station"
            if per_if else "one per station, all IFs")
    ax.set_title(f"Station corrections ({what})", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left",
              labelcolor=INK2)
    lo = np.nanmin(np.concatenate([naive.ravel(), loo.ravel(),
                                   np.exp(app - sig).ravel()]))
    hi = np.nanmax(np.concatenate([naive.ravel(), loo.ravel(),
                                   np.exp(app + sig).ravel()]))
    ax.set_ylim(min(lo, math.exp(-tau)) / 1.08, max(hi, math.exp(tau)) * 1.2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    # -- probability a correction is needed -------------------------
    ax = fig.add_subplot(gs[1, 0])
    _style(ax)
    p = pick(res.p_correction)
    ax.bar(xs.ravel(), np.nan_to_num(p.ravel()), width=0.9 * step,
           color=BLUE, edgecolor=SURFACE, linewidth=1 if nslot > 2 else 2)
    for level, text in ((0.5, "no preference"), (0.95, "needed")):
        ax.axhline(level, color=MUTED, linewidth=1, linestyle="--")
        ax.text(len(x) - 0.5, level + 0.015, text, color=MUTED, fontsize=7,
                ha="right", va="bottom", zorder=5,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(idx) - 0.5)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("P(correction needed)", color=INK2, fontsize=9)
    ax.set_title("Is a correction warranted?"
                 + (" (bars: IF1 ... IF%d)" % nslot if per_if else ""),
                 fontsize=10, loc="left")

    # -- model evidence -----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    ym = np.arange(len(res.models))
    d = np.where(np.isfinite(res.model_dbic), res.model_dbic, np.nan)
    ax.barh(ym, np.maximum(np.nan_to_num(d), 0.0) + 1.0, height=0.6,
            color=[BLUE if i == bi else AXIS for i in ym], edgecolor=SURFACE,
            linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks(ym, res.models, fontsize=8)
    ax.invert_yaxis()
    for i in ym:
        label = ("failed" if not np.isfinite(d[i]) else
                 f"P = {res.model_prob[i]:.3g}, rchisq {res.model_rchisq[i]:.3g}")
        ax.text(max(np.nan_to_num(d[i]), 0.0) + 1.0, i, "  " + label,
                va="center", fontsize=7, color=INK2)
    ax.set_xlabel("1 + delta BIC (log)", color=INK2, fontsize=9)
    ax.set_title("Source-model evidence (full array)", fontsize=10, loc="left")
    ax.set_xlim(right=max(10.0, np.nanmax(np.nan_to_num(d)) + 1.0) * 30)

    # -- influence matrix ---------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.set_facecolor(SURFACE)
    infl = res.influence[idx][:, :, idx]                # [excl, nif, ant]
    m = np.full((len(idx), len(idx)), np.nan)
    for i in range(len(idx)):
        m[i] = _median_ifs(infl[i])
    pct = 100.0 * (np.exp(m) - 1.0)
    lim = np.nanmax(np.abs(pct)) if np.isfinite(pct).any() else 1.0
    lim = max(lim, 1.0)
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING).with_extremes(
        bad=SURFACE)
    im = ax.imshow(pct, cmap=cmap, norm=TwoSlopeNorm(0.0, -lim, lim),
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(idx)), names, fontsize=7, rotation=90)
    ax.set_yticks(np.arange(len(idx)), names, fontsize=7)
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("station whose gain moves", color=INK2, fontsize=9)
    ax.set_ylabel("station left out of the model", color=INK2, fontsize=9)
    ax.set_title(f"Leave-one-out influence ({res.best_model}), % change",
                 fontsize=10, loc="left", color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)

    # -- station fit before / after -----------------------------------
    ax = fig.add_subplot(gs[2, 1])
    _style(ax)
    before = np.asarray(res.antenna_rchisq["before"], float)[idx]
    after = np.asarray(res.antenna_rchisq["after"], float)[idx]
    w = 0.36
    ax.bar(x - w / 2, before, width=w, color=ORANGE, edgecolor=SURFACE,
           linewidth=2, label="before")
    ax.bar(x + w / 2, after, width=w, color=BLUE, edgecolor=SURFACE,
           linewidth=2, label="after (Bayesian)")
    ax.set_yscale("log")
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("reduced chi-squared", color=INK2, fontsize=9)
    ax.set_title(f"Station fit to the {res.best_model} model", fontsize=10,
                 loc="left")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2)

    if path:
        fig.savefig(path, dpi=130, facecolor=SURFACE)
    if show or (show is None and not path):
        plt.show()
    elif path:
        plt.close(fig)
    return fig
