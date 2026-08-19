"""Time averaging of UV data (difmap `uvaver`).

Averages the *calibrated* visibilities into coarser integrations,
returning a new observation. Flagged samples are excluded from the
average; an output sample is flagged only if none of its inputs were
usable.
"""

from __future__ import annotations

import numpy as np

from difmapy._core import CoreObservation

__all__ = ["uvaver"]


def uvaver(core: CoreObservation, aver_time: float, doscatter: bool = False):
    """Return a new CoreObservation with the data averaged into
    `aver_time`-second bins.

    Parameters
    ----------
    aver_time : float
        Duration of the output integrations (seconds).
    doscatter : bool
        If True, derive the output weights from the scatter of the
        averaged samples instead of summing the input weights.
    """
    if aver_time <= 0:
        raise ValueError("aver_time must be positive")

    vis, wt = core.calibrated_cube()  # signed weights: <0 flagged
    vis = np.asarray(vis)
    wt = np.asarray(wt)
    good = wt > 0
    w = np.where(good, wt, 0.0).astype(np.float64)

    time, a1, a2, us, vs, ws = core.rows()
    time = np.asarray(time)
    uvw = np.column_stack([us, vs, ws])
    inttime = np.asarray(core.inttimes(), dtype=np.float64)

    # Bin index of each row, and the (bin, baseline) grouping.
    tbin = np.floor(time / aver_time).astype(np.int64)
    keys = np.stack([tbin, np.asarray(a1, np.int64), np.asarray(a2, np.int64)], axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_s = keys[order]
    # Start of each group in the sorted order.
    new_group = np.ones(len(keys_s), dtype=bool)
    new_group[1:] = np.any(keys_s[1:] != keys_s[:-1], axis=1)
    gidx = np.cumsum(new_group) - 1
    ngroup = int(gidx[-1]) + 1 if len(gidx) else 0
    if ngroup == 0:
        raise ValueError("no data to average")

    # Sort the data into group order once.
    vis_s = vis[order]
    w_s = w[order]
    good_s = good[order]
    uvw_s = uvw[order]
    time_s = time[order]
    it_s = inttime[order]

    shape = (ngroup,) + vis.shape[1:]
    sum_wv = np.zeros(shape, dtype=np.complex128)
    sum_w = np.zeros(shape, dtype=np.float64)
    sum_n = np.zeros(shape, dtype=np.int64)
    # Weighted vector average per (channel, pol).
    np.add.at(sum_wv, gidx, w_s * vis_s)
    np.add.at(sum_w, gidx, w_s)
    np.add.at(sum_n, gidx, good_s.astype(np.int64))

    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(sum_w > 0, sum_wv / np.maximum(sum_w, 1e-300), 0.0)
    out_vis = avg.astype(np.complex64)
    out_flag = sum_w <= 0

    if doscatter:
        # Weight from the scatter about the mean: wt = n / variance.
        sum_sq = np.zeros(shape, dtype=np.float64)
        resid = np.abs(vis_s - avg[gidx]) ** 2
        np.add.at(sum_sq, gidx, np.where(good_s, resid, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(sum_n > 1, sum_sq / np.maximum(sum_n - 1, 1), np.nan)
            out_wt = np.where(
                (sum_n > 1) & (var > 0), sum_n / np.maximum(var, 1e-300), sum_w
            )
    else:
        out_wt = sum_w  # sum of input weights (difmap default)
    out_wt = np.where(out_flag, 0.0, out_wt).astype(np.float32)

    # Row metadata: mean uvw/time weighted by data weight (falling back
    # to a plain mean where everything was flagged).
    rw = np.maximum(w_s.sum(axis=(1, 2)), 0.0)
    sum_rw = np.zeros(ngroup)
    np.add.at(sum_rw, gidx, rw)
    sum_cnt = np.zeros(ngroup)
    np.add.at(sum_cnt, gidx, 1.0)

    def _avg(col, weights):
        acc = np.zeros((ngroup,) + col.shape[1:], dtype=np.float64)
        np.add.at(acc, gidx, col * weights.reshape((-1,) + (1,) * (col.ndim - 1)))
        return acc

    wsum = np.where(sum_rw > 0, sum_rw, sum_cnt)
    wcol = np.where(rw > 0, rw, np.where(sum_rw[gidx] > 0, 0.0, 1.0))
    out_uvw = _avg(uvw_s, wcol) / wsum[:, None]
    out_time = _avg(time_s, wcol) / wsum
    out_int = np.zeros(ngroup)
    np.add.at(out_int, gidx, it_s)  # integration times add

    # Antennas of each group, from its first row.
    first = np.nonzero(new_group)[0]
    out_a1 = np.asarray(a1)[order][first].astype(np.uint32)
    out_a2 = np.asarray(a2)[order][first].astype(np.uint32)

    # Sort the output rows by time (the core requires time-sorted rows).
    rorder = np.lexsort((out_a2, out_a1, out_time))
    ifs = core.ifs
    new = CoreObservation(
        core.source_name,
        core.ra,
        core.dec,
        2000.0,
        core.antenna_names,
        np.asarray(core.antenna_xyz),
        list(core.antenna_subarrays),
        [f for (f, _, _) in ifs],
        [d for (_, d, _) in ifs],
        [n for (_, _, n) in ifs],
        list(core.pols),
        np.ascontiguousarray(out_time[rorder]),
        np.ascontiguousarray(out_int[rorder].astype(np.float32)),
        np.ascontiguousarray(out_a1[rorder]),
        np.ascontiguousarray(out_a2[rorder]),
        np.ascontiguousarray(out_uvw[rorder]),
        np.ascontiguousarray(out_vis[rorder]),
        np.ascontiguousarray(out_wt[rorder]),
        core.ref_mjd,
        flag=np.ascontiguousarray(out_flag[rorder]),
    )
    return new
