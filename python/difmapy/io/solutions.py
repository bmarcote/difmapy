"""Antenna gain solutions, in the form the calibration-table writers take.

Both the CASA ("G Jones") and the AIPS (SN) writers export the same
thing: a set of antenna-based amplitude and phase corrections on a grid
of solution times. `GainSolutions` holds it in difmapy's own convention -
the *correction* that multiplies the data,

    V_corrected = V_raw * c_p * conj(c_q),    c_a = amp_a * exp(i phs_a)

- and each writer converts to its package's convention on the way out.

`from_gain_table` takes the solutions from the observation's accumulated
gain table (what `selfcal`/`gscale` built up); `constant` builds a
time-independent set, which is what a whole-observation amplitude
calibration such as `bayes_gscale` produces and what can be carried over
to other sources of the same observation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["GainSolutions", "from_gain_table", "constant", "export_warnings"]


@dataclass
class GainSolutions:
    """Antenna gain corrections on a grid of solution times.

    Arrays are ``[ntime, nif, nant]`` over the observation's IFs and
    antenna entries; `times` are the solution centres in seconds since
    `ref_mjd` and `interval` their durations in seconds.
    """

    times: np.ndarray
    interval: np.ndarray
    amp: np.ndarray
    phs: np.ndarray
    bad: np.ndarray
    used: np.ndarray
    ref_mjd: float
    warnings: list = field(default_factory=list)

    @property
    def shape(self):
        return self.amp.shape

    def correction(self):
        """The complex corrections, ``amp * exp(i phs)``."""
        return self.amp * np.exp(1j * self.phs)

    def flags(self, flag_uncalibrated=False):
        """Which solutions to write as flagged."""
        flag = self.bad.copy()
        if flag_uncalibrated:
            flag |= ~self.used
        return flag


def export_warnings(obs) -> list[str]:
    """What an antenna-based table cannot carry, said out loud rather
    than silently dropped."""
    core = obs._core
    warnings = []
    if any(abs(v) > 0 for v in core.shift_total):
        e, n = obs.total_shift
        warnings.append(
            f"a phase-centre shift of ({e:.4g}, {n:.4g}) mas is in effect; "
            "a calibration table cannot express it (use CASA's phaseshift "
            "or AIPS UVFIX, or wobs(freeze_shift=True) to bake it into the "
            "data)"
        )
    bls, bamp, bphs = core.baseline_corrections()
    if len(bls) and not (np.allclose(bamp, 1.0) and np.allclose(bphs, 0.0)):
        warnings.append(
            "baseline-based corrections from resoff() are in effect; these "
            "have no antenna-based equivalent and are NOT included in the "
            "table (use wobs() to write data with them applied)"
        )
    return warnings


def from_gain_table(obs, since=None) -> GainSolutions:
    """The accumulated gain table, one solution per integration.

    `since` (from `Observation.gain_snapshot()`) exports only what has
    been accumulated after that snapshot, by dividing out the
    corrections that were already in place.
    """
    core = obs._core
    nant = len(core.antenna_names)
    nif = core.nif
    ntimes = core.ntimes
    amp, phs, bad = (np.asarray(x) for x in core.gains())
    amp = amp.reshape(ntimes, nif, nant).astype(np.float64)
    phs = phs.reshape(ntimes, nif, nant).astype(np.float64)
    bad = bad.reshape(ntimes, nif, nant).copy()
    used = np.asarray(core.gains_used()).reshape(ntimes, nif, nant).copy()

    if since is not None:
        amp0, phs0 = since["amp"], since["phs"]
        if amp0.shape != amp.shape:
            raise ValueError(
                "the gain snapshot does not match this observation "
                f"(snapshot {amp0.shape}, now {amp.shape}); it must come "
                "from the same observation without a reselection"
            )
        amp = np.where(amp0 != 0, amp / np.where(amp0 != 0, amp0, 1.0), amp)
        phs = phs - phs0
        # `used` still means "a solution was applied here".

    # Solution interval: the integration length, from the data.
    inttime = np.asarray(core.inttimes(), dtype=np.float64)
    tidx = np.asarray(core.time_index())
    interval = np.zeros(ntimes)
    np.maximum.at(interval, tidx, inttime)
    return GainSolutions(
        times=np.asarray(core.times(), dtype=np.float64),
        interval=interval,
        amp=amp,
        phs=phs,
        bad=bad,
        used=used,
        ref_mjd=float(core.ref_mjd),
        warnings=export_warnings(obs),
    )


def constant(obs, amp, phs=None, bad=None, used=None) -> GainSolutions:
    """Time-independent corrections: `amp`/`phs` (radians) are
    ``[nif, nant]``; `bad` optionally marks entries whose data should be
    flagged, and `used` those actually solved for (default: all that
    are not bad), which is what `flag_uncalibrated` goes by.

    The table gets two identical solutions per antenna, at the first and
    the last integration, each valid over the whole observation, so
    that any interpolation mode - nearest, linear, AIPS 2PT - gives the
    same constant value throughout, and the nearest one beyond it.
    """
    core = obs._core
    nif, nant = core.nif, len(core.antenna_names)
    amp = np.asarray(amp, dtype=np.float64)
    if amp.shape != (nif, nant):
        raise ValueError(f"amp must be [nif, nant] = [{nif}, {nant}]")
    phs = np.zeros_like(amp) if phs is None else np.asarray(phs, dtype=np.float64)
    bad = np.zeros(amp.shape, dtype=bool) if bad is None else np.asarray(bad, bool)
    used = ~bad if used is None else np.asarray(used, bool)
    t = np.asarray(core.times(), dtype=np.float64)
    t0, t1 = float(t.min()), float(t.max())
    span = max(t1 - t0, float(np.max(core.inttimes())), 1.0)
    times = np.array([t0, t1]) if t1 > t0 else np.array([t0])
    n = len(times)
    return GainSolutions(
        times=times,
        interval=np.full(n, span),
        amp=np.repeat(amp[None], n, axis=0),
        phs=np.repeat(phs[None], n, axis=0),
        bad=np.repeat(bad[None], n, axis=0),
        used=np.repeat(used[None], n, axis=0),
        ref_mjd=float(core.ref_mjd),
        warnings=export_warnings(obs),
    )
