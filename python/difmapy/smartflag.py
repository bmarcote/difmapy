"""Smart flagging of a selection (Shift+S, then a drag, on a visibility plot): :mod:`difmapy.outliers` on the data.

The whole stream of the observation (every baseline and IF, the selected Stokes product) becomes one
:class:`difmapy.outliers.VisGrid` - one channel block, one correlation, one time bin per integration - so the
trends and the noise are those of the whole source and every antenna has all its baselines to be judged on. The
selection only decides where a flag may *start*: outliers among the selected points seed it, and the region then
grows to the time range of the fault and to every baseline of the antenna responsible, inside or outside the box.
That is the difference with vlbipy's ``flag smart-outliers``, which seeds anywhere.

Scans are runs of integrations without a gap longer than ``max(SCAN_GAP_FACTOR x the typical integration
spacing, MIN_SCAN_GAP)``.
"""

from __future__ import annotations

import numpy as np

from difmapy.outliers import OutlierSettings, VisGrid, find_outliers

__all__ = ["stream_grid", "smart_flag_ops"]

#: A gap longer than this many typical integration spacings starts a new scan ...
SCAN_GAP_FACTOR = 4.0
#: ... and never one shorter than this [s].
MIN_SCAN_GAP = 60.0


def _scans(times: np.ndarray) -> np.ndarray:
    """Scan number of every (sorted, unique) integration time, from the gaps between them."""
    if times.size < 2:
        return np.zeros(times.size, dtype=np.int64)
    steps = np.diff(times)
    gap = max(SCAN_GAP_FACTOR * float(np.median(steps)), MIN_SCAN_GAP)
    return np.concatenate([[0], np.cumsum(steps > gap)]).astype(np.int64)


def stream_grid(obs, selected=None) -> tuple[VisGrid, dict]:
    """The observation's stream as a :class:`VisGrid`, and how to map its cells back to rows.

    Parameters
    ----------
    obs : difmapy.Observation
        The observation (its current stream: calibration, selection and flags as shown).
    selected : (numpy.ndarray, numpy.ndarray), optional
        ``(rows, ifs)`` of the selected points: only these cells may seed a flag (``VisGrid.restrict``).

    Returns
    -------
    (VisGrid, dict)
        The grid (``count`` is the weight relative to the median good weight of the IF, so ``sqrt(count)``
        scales the noise as the weights say) and ``baseline``/``time`` (per row: its grid baseline and time bin),
        ``antennas`` (grid antenna -> station index), ``used`` (IFs in the selection) and ``weights``.
    """
    core = obs._core
    vis, wt = core.stream_vis()
    vis, wt = np.asarray(vis), np.asarray(wt, dtype=np.float64)
    time, a1, a2, *_ = core.rows()
    time, a1, a2 = np.asarray(time, dtype=np.float64), np.asarray(a1), np.asarray(a2)
    used = np.asarray(core.selection()["if_used"], dtype=bool)
    nrow, nif = vis.shape
    stations = np.unique(np.concatenate([a1, a2]))
    low = np.searchsorted(stations, np.minimum(a1, a2))
    high = np.searchsorted(stations, np.maximum(a1, a2))
    pairs, baseline = np.unique(low * len(stations) + high, return_inverse=True)
    times, tbin = np.unique(time, return_inverse=True)
    good = (wt > 0) & used[None, :] & (low != high)[:, None]
    reference = np.array([np.median(wt[good[:, k], k]) if good[:, k].any() else 1.0 for k in range(nif)])
    weight = np.where(good, wt / reference[None, :], 0.0)
    shape = (pairs.size, nif, 1, 1, times.size)
    sums = np.zeros(shape, dtype=np.complex128)
    counts = np.zeros(shape, dtype=np.float64)
    for k in range(nif):
        np.add.at(sums[:, k, 0, 0], (baseline, tbin), np.where(good[:, k], vis[:, k] * weight[:, k], 0))
        np.add.at(counts[:, k, 0, 0], (baseline, tbin), weight[:, k])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, sums / np.maximum(counts, 1e-30), 0)
    restrict = None
    if selected is not None:
        rows, ifs = (np.asarray(v, dtype=np.int64) for v in selected)
        restrict = np.zeros(shape, dtype=bool)
        restrict[baseline[rows], ifs, 0, 0, tbin[rows]] = True
    grid = VisGrid(mean, counts, pairs // len(stations), pairs % len(stations), times, _scans(times),
                   len(stations), restrict=restrict)
    return grid, {"baseline": baseline, "time": tbin, "antennas": stations, "used": used, "weights": wt}


def smart_flag_ops(obs, rows, ifs, *, use_phases: bool = True, settings=None) -> tuple[list, dict]:
    """The flag edits that remove the outliers a selection points at.

    Parameters
    ----------
    obs : difmapy.Observation
        The observation.
    rows, ifs : array-like
        The selected points (stream row and IF of each).
    use_phases : bool
        Test the phases too (the box was drawn on a phase panel, or the data are phase self-calibrated);
        False: the amplitudes only.
    settings : dict, optional
        Further :class:`difmapy.outliers.OutlierSettings` fields.

    Returns
    -------
    (list, dict)
        ``[(rows, if_index, True), ...]`` for :class:`difmapy.plots.base.EditHistory` (only samples not flagged
        yet), and a summary: ``samples``, ``antennas`` (station indices flagged as antennas), ``baselines``.
    """
    grid, index = stream_grid(obs, selected=(rows, ifs))
    result = find_outliers(grid, OutlierSettings(**{"use_phases": bool(use_phases), **dict(settings or {})}))
    flagged = result.cell_flags(grid)[:, :, 0, 0, :]                       # (nbl, nif, ntime)
    ops, samples = [], 0
    for k in np.flatnonzero(index["used"]):
        hit = flagged[index["baseline"], k, index["time"]] & (index["weights"][:, k] > 0)
        if hit.any():
            chosen = np.flatnonzero(hit)
            ops.append((chosen, int(k), True))
            samples += int(chosen.size)
    antennas = sorted(int(index["antennas"][a]) for a in np.flatnonzero(result.antenna_flags.any(axis=(1, 2, 3, 4))))
    baselines = int(result.baseline_flags.any(axis=(1, 2, 3, 4)).sum())
    return ops, {"samples": samples, "antennas": antennas, "baselines": baselines}
