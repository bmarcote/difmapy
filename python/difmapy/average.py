"""Time and frequency averaging of UV data.

`uvaver` (difmap's own command) averages the *calibrated* visibilities
into coarser integrations, and `chanaver` averages adjacent channels of
each IF (AIPS AVSPC, CASA split's `width`); both return a new
observation. Flagged samples are excluded from the average; an output
sample is flagged only if none of its inputs were usable.
"""

from __future__ import annotations

import numpy as np

from difmapy._core import CoreObservation

__all__ = ["uvaver", "chanaver"]


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

    # Average what the user currently sees: gains, baseline corrections
    # and any accumulated shift are all applied.
    vis, wt = core.calibrated_cube(apply_shift=True)  # signed wt: <0 flagged
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

    # The rows are sorted into group order, so all the group sums are
    # segmented reductions. reduceat costs about the same time as
    # np.add.at here (sorting the cube dominates) but avoids its
    # float64 upcast, halving the peak temporary.
    starts = np.nonzero(new_group)[0]

    def _sum(arr):
        return np.add.reduceat(arr, starts, axis=0)

    sum_w = _sum(w_s)
    sum_n = _sum(good_s.astype(np.int32))
    sum_wv = _sum(w_s.astype(np.float32) * vis_s)  # complex64 temporary

    with np.errstate(invalid="ignore", divide="ignore"):
        avg = np.where(sum_w > 0, sum_wv / np.maximum(sum_w, 1e-300), 0.0)
    out_vis = avg.astype(np.complex64)
    out_flag = sum_w <= 0

    if doscatter:
        # Weight from the scatter about the mean: wt = n / variance.
        resid = np.abs(vis_s - avg[gidx].astype(np.complex64)) ** 2
        sum_sq = _sum(np.where(good_s, resid, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            var = np.where(sum_n > 1, sum_sq / np.maximum(sum_n - 1, 1), np.nan)
            out_wt = np.where(
                (sum_n > 1) & (var > 0), sum_n / np.maximum(var, 1e-300), sum_w
            )
    else:
        out_wt = sum_w  # sum of input weights (difmap default)
    out_wt = np.where(out_flag, 0.0, out_wt).astype(np.float32)

    # Row metadata: uvw is averaged with the row's total data weight,
    # falling back to a plain mean for fully flagged groups.
    rw = np.maximum(w_s.sum(axis=(1, 2)), 0.0)
    sum_rw = _sum(rw)
    sum_cnt = _sum(np.ones(len(rw)))
    wsum = np.where(sum_rw > 0, sum_rw, sum_cnt)
    wcol = np.where(rw > 0, rw, np.where(sum_rw[gidx] > 0, 0.0, 1.0))
    out_uvw = _sum(uvw_s * wcol[:, None]) / wsum[:, None]
    out_int = _sum(it_s)  # integration times add

    # Every baseline in a time bin must end up with *exactly* the same
    # timestamp: the core identifies integrations by equal times, and a
    # per-baseline weighted mean would differ in the last bits, turning
    # each baseline into its own integration (which would quietly break
    # per-integration self-calibration and closure phases).
    tbin_s = tbin[order]
    ubin, bin_of_row = np.unique(tbin_s, return_inverse=True)
    nb = len(ubin)
    num = np.bincount(bin_of_row, weights=time_s * rw, minlength=nb)
    den = np.bincount(bin_of_row, weights=rw, minlength=nb)
    plain = np.bincount(bin_of_row, weights=time_s, minlength=nb) / np.bincount(
        bin_of_row, minlength=nb
    )
    canon_time = np.where(den > 0, num / np.maximum(den, 1e-300), plain)
    out_time = canon_time[bin_of_row[starts]]

    # Antennas of each group, from its first row.
    out_a1 = np.asarray(a1)[order][starts].astype(np.uint32)
    out_a2 = np.asarray(a2)[order][starts].astype(np.uint32)

    # Sort the output rows by time (the core requires time-sorted rows).
    rorder = np.lexsort((out_a2, out_a1, out_time))
    ifs = core.ifs
    # Which output row each input row went into, for tracing an averaged
    # sample back to the rows it was made of (`save_flags`).
    new_of_group = np.empty(ngroup, dtype=np.int64)
    new_of_group[rorder] = np.arange(ngroup)
    row_map = np.empty(len(order), dtype=np.int64)
    row_map[order] = new_of_group[gidx]
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
    new._avg_row_map = row_map
    return new


def chanaver(core: CoreObservation, nchan=None, chlist=None):
    """Return a new CoreObservation with every `nchan` adjacent channels
    of each IF averaged into one.

    Parameters
    ----------
    nchan : int | str | None
        Channels per output channel. It must divide every IF's number
        of channels, since an IF's channels are equally spaced by
        construction. ``None``, ``0`` or ``"all"`` average each IF down
        to a single channel.
    chlist : list[(int, int)] | None
        Inclusive global channel ranges to use (as for `select`); the
        other channels are left out of the averages, which is how band
        edges are dropped before averaging. Default: all channels.

    As with `uvaver`, what is averaged is what the user currently sees -
    gains, baseline corrections and any shift applied, the shift at each
    channel's own frequency - and the result starts uncalibrated.

    The weights of the inputs add. A sample is good if any of its inputs
    was; a bin whose inputs were all flagged keeps the average of those
    (flagged, so `unflag` can bring it back), and only a bin with no
    data at all is deleted.
    """
    vis, wt = core.calibrated_cube(apply_shift=True)  # signed wt: <0 flagged
    vis = np.asarray(vis)
    wt = np.asarray(wt)
    nrow, nctotal, npol = vis.shape
    use = np.ones(nctotal, dtype=bool)
    if chlist:
        use[:] = False
        for a, b in chlist:
            a, b = int(a), int(b)
            if a > b or a < 0 or b >= nctotal:
                raise ValueError(f"invalid channel range {a}-{b} "
                                 f"(there are {nctotal} channels)")
            use[a:b + 1] = True
    good = (wt > 0) & use[None, :, None]
    present = (wt != 0) & use[None, :, None]

    out_vis, out_wt, out_flag = [], [], []
    freqs, widths, counts = [], [], []
    # Which output channel each input channel went into (-1: left out),
    # for tracing an averaged sample back to its channels.
    chan_map = np.full(nctotal, -1, dtype=np.int64)
    out_coff = 0
    coff = 0
    for cif, (f0, df, nch) in enumerate(core.ifs):
        if nchan in (None, 0) or str(nchan).lower() == "all":
            n = nch
        else:
            n = int(nchan)
            if n < 1:
                raise ValueError("nchan must be a positive number of channels")
            n = min(n, nch)
        if nch % n:
            raise ValueError(
                f"IF {cif + 1} has {nch} channels, which cannot be averaged "
                f"in groups of {n}: the averaged channels must stay equally "
                "spaced, so use a divisor of the number of channels"
            )
        nout = nch // n
        sl = slice(coff, coff + nch)
        shape = (nrow, nout, n, npol)
        v = vis[:, sl].reshape(shape)
        w = np.abs(wt[:, sl]).reshape(shape)
        g = good[:, sl].reshape(shape)
        p = present[:, sl].reshape(shape)
        # Good inputs if there are any; otherwise the flagged ones.
        anygood = g.any(axis=2, keepdims=True)
        take = np.where(anygood, g, p)
        wsum = np.where(take, w, 0.0).sum(axis=2, dtype=np.float64)
        vsum = np.where(take, w * v, 0.0).sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            avg = np.where(wsum > 0, vsum / np.maximum(wsum, 1e-300), 0.0)
        out_vis.append(avg.astype(np.complex64))
        out_wt.append(wsum.astype(np.float32))
        out_flag.append(~anygood[:, :, 0, :])
        # The centre of the first output channel, and the new spacing.
        freqs.append(float(f0 + 0.5 * (n - 1) * df))
        widths.append(float(df * n))
        counts.append(nout)
        chan_map[sl] = np.where(use[sl], out_coff + np.arange(nch) // n, -1)
        out_coff += nout
        coff += nch

    out_vis = np.concatenate(out_vis, axis=1)
    out_wt = np.concatenate(out_wt, axis=1)
    out_flag = np.concatenate(out_flag, axis=1) | (out_wt <= 0)
    time, a1, a2, us, vs, ws = core.rows()
    new = CoreObservation(
        core.source_name,
        core.ra,
        core.dec,
        2000.0,
        core.antenna_names,
        np.asarray(core.antenna_xyz),
        list(core.antenna_subarrays),
        freqs,
        widths,
        counts,
        list(core.pols),
        np.ascontiguousarray(np.asarray(time, dtype=np.float64)),
        np.ascontiguousarray(np.asarray(core.inttimes(), dtype=np.float32)),
        np.ascontiguousarray(np.asarray(a1, dtype=np.uint32)),
        np.ascontiguousarray(np.asarray(a2, dtype=np.uint32)),
        np.ascontiguousarray(np.column_stack([us, vs, ws]).astype(np.float64)),
        np.ascontiguousarray(out_vis),
        np.ascontiguousarray(out_wt),
        core.ref_mjd,
        flag=np.ascontiguousarray(out_flag),
    )
    new._avg_chan_map = chan_map
    return new
