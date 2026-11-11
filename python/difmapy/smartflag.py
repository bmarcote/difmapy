"""Smart flagging of the data shown: :mod:`difmapy.outliers` on the stream of an observation.

The whole stream of the observation (every baseline and IF, the selected Stokes product) becomes one
:class:`difmapy.outliers.VisGrid` - one channel block, one correlation, one time bin per integration - so the
trends and the noise are those of the whole source and every antenna has all its baselines to be judged on.

Two uses:

* ``Observation.smartflag()`` (the dashboard's Smart flag button) searches all of it: the antennas whose
  baselines depart together beyond ``antenna_sigma`` are flagged over the time range of the fault, and single
  baselines beyond ``baseline_sigma``.
* A selection (Shift+S, then a drag, on a visibility plot) says which points are outliers: they are flagged
  whatever the model makes of them, and the antenna behind them - of the two of each selected baseline, the one
  whose other baselines depart with it - is flagged on every baseline, inside or outside the box, over the time
  range of the fault. A selected point no antenna explains is flagged on its own.

Scans are runs of integrations without a gap longer than ``max(SCAN_GAP_FACTOR x the typical integration
spacing, MIN_SCAN_GAP)``.
"""

from __future__ import annotations

import numpy as np

from difmapy.outliers import OutlierSettings, VisGrid, find_outliers

__all__ = ["stream_grid", "smart_flag_ops", "apply_ops", "describe"]

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
        ``(rows, ifs)`` of the points picked out as outliers (``VisGrid.selected``); None searches all the data.

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
    picked = None
    if selected is not None:
        rows, ifs = (np.asarray(v, dtype=np.int64) for v in selected)
        picked = np.zeros(shape, dtype=bool)
        picked[baseline[rows], ifs, 0, 0, tbin[rows]] = True
    grid = VisGrid(mean, counts, pairs // len(stations), pairs % len(stations), times, _scans(times),
                   len(stations), selected=picked)
    return grid, {"baseline": baseline, "time": tbin, "antennas": stations, "used": used, "weights": wt}


def smart_flag_ops(obs, rows=None, ifs=None, *, use_phases: bool = True, settings=None) -> tuple[list, dict]:
    """The flag edits the smart flagger asks for, on all the data or from a selection of outliers.

    Parameters
    ----------
    obs : difmapy.Observation
        The observation.
    rows, ifs : array-like, optional
        The points picked out as outliers (stream row and IF of each): they are flagged, and so is the antenna
        behind them over the time range of the fault. None (both): a search of all the data.
    use_phases : bool
        Test the phases too (a box drawn on a phase panel, phase-calibrated data); False: the amplitudes only.
    settings : dict, optional
        Further :class:`difmapy.outliers.OutlierSettings` fields.

    Returns
    -------
    (list, dict)
        ``[(rows, if_index, True), ...]`` for :func:`apply_ops` or :class:`difmapy.plots.base.EditHistory`
        (only samples not flagged yet), and a summary: ``samples`` (samples to flag), ``antennas`` (station index
        -> samples flagged on its baselines because the station was blamed, for the stations with any) and
        ``alone`` (samples flagged on their own baseline).
    """
    if (rows is None) != (ifs is None):
        raise ValueError("smart_flag_ops: give both rows and ifs, or neither")
    grid, index = stream_grid(obs, selected=None if rows is None else (rows, ifs))
    result = find_outliers(grid, OutlierSettings(**{"use_phases": bool(use_phases), **dict(settings or {})}))
    flagged = result.cell_flags(grid)[:, :, 0, 0, :]                       # (nbl, nif, ntime)
    by_antenna = result.antenna_flags[:, :, 0, 0, :]                       # (nant, nif, ntime)
    baseline, tbin = index["baseline"], index["time"]
    first, second = grid.ant1[baseline], grid.ant2[baseline]
    per_antenna = np.zeros(grid.nant, dtype=np.int64)
    ops, samples, alone = [], 0, 0
    for k in np.flatnonzero(index["used"]):
        hit = flagged[baseline, k, tbin] & (index["weights"][:, k] > 0)
        if not hit.any():
            continue
        chosen = np.flatnonzero(hit)
        ops.append((chosen, int(k), True))
        samples += int(chosen.size)
        on_first = hit & by_antenna[first, k, tbin]
        on_second = hit & by_antenna[second, k, tbin]
        np.add.at(per_antenna, first[on_first], 1)
        np.add.at(per_antenna, second[on_second], 1)
        alone += int((hit & ~on_first & ~on_second).sum())
    antennas = {int(index["antennas"][a]): int(per_antenna[a]) for a in np.flatnonzero(per_antenna)}
    return ops, {"samples": samples, "antennas": antennas, "alone": alone}


def apply_ops(obs, ops) -> None:
    """Flag what :func:`smart_flag_ops` returned (the selected channels of each IF), outside any undo history."""
    core = obs._core
    for rows, cif, flag in ops:
        core.edit_rows([int(r) for r in rows], bool(flag), if_index=int(cif), sel_chan=True)
    obs._reapply_ignores()
    obs._dirty()


def describe(obs, summary) -> str:
    """One line on what a smart flag did: the stations blamed (samples on their baselines) and the rest."""
    names = obs.antennas
    who = ", ".join(f"{names[a] if 0 <= a < len(names) else a} ({n})" for a, n in summary["antennas"].items())
    parts = [f"{summary['samples']} samples"]
    if who:
        parts.append(f"antennas {who}")
    if summary["alone"]:
        parts.append(f"{summary['alone']} on their own baselines")
    return "; ".join(parts)
