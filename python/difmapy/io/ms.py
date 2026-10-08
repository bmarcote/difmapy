"""Measurement Set reader (via casatools).

Loads a single-field MS into the in-memory core observation. All
spectral windows become IFs (each may have a different number of
channels); rows of the same (time, array, baseline) are merged into a
single visibility row covering all IFs, as difmap does for UVFITS.
"""

from __future__ import annotations

import os

import numpy as np

from difmapy._core import CoreObservation

__all__ = ["load_ms", "save_flags", "save_ms"]

C = 299792458.0

# CASA Stokes enumeration -> AIPS FITS convention codes.
_CASA_TO_AIPS = {
    1: 1, 2: 2, 3: 3, 4: 4,  # I Q U V
    5: -1, 6: -3, 7: -4, 8: -2,  # RR RL LR LL
    9: -5, 10: -7, 11: -8, 12: -6,  # XX XY YX YY
}


def load_ms(path, field=None, data_column="DATA", wtscale=1.0):
    """Read a CASA Measurement Set and return a CoreObservation.

    Parameters
    ----------
    path : str
        Path of the MS directory.
    field : str | int | None
        Field name or FIELD_ID to load. Mandatory if the MS contains
        several fields (difmapy is single-source, like difmap).
    data_column : str
        "DATA" or "CORRECTED_DATA".
    wtscale : float
        Optional scale factor applied to the weights.
    """
    try:
        import casatools
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "reading Measurement Sets requires casatools: pip install casatools"
        ) from exc

    tb = casatools.table()

    # ---- subtables ----
    tb.open(f"{path}/FIELD", nomodify=True)
    field_names = [str(n) for n in tb.getcol("NAME")]
    phase_dir = tb.getcol("PHASE_DIR")  # [2, npoly, nfield] radians
    tb.close()

    if field is None:
        if len(field_names) > 1:
            raise ValueError(
                f"MS contains several fields {field_names}; difmapy is "
                "single-source - pass field=<name or id>"
            )
        field_id = 0
    elif isinstance(field, int):
        field_id = field
    else:
        if str(field) not in field_names:
            raise ValueError(f"field {field!r} not in {field_names}")
        field_id = field_names.index(str(field))
    source = field_names[field_id]
    ra = float(phase_dir[0, 0, field_id]) % (2 * np.pi)
    dec = float(phase_dir[1, 0, field_id])

    tb.open(f"{path}/ANTENNA", nomodify=True)
    ant_names_tab = [str(n) for n in tb.getcol("NAME")]
    ant_pos = np.asarray(tb.getcol("POSITION")).T.astype(np.float64)  # [nant, 3]
    tb.close()

    tb.open(f"{path}/DATA_DESCRIPTION", nomodify=True)
    dd_spw = np.atleast_1d(tb.getcol("SPECTRAL_WINDOW_ID")).astype(int)
    dd_pol = np.atleast_1d(tb.getcol("POLARIZATION_ID")).astype(int)
    tb.close()

    tb.open(f"{path}/SPECTRAL_WINDOW", nomodify=True)
    spw_freqs = [np.atleast_1d(tb.getcell("CHAN_FREQ", i)) for i in range(tb.nrows())]
    spw_width = [np.atleast_1d(tb.getcell("CHAN_WIDTH", i)) for i in range(tb.nrows())]
    tb.close()

    tb.open(f"{path}/POLARIZATION", nomodify=True)
    pol_types = [np.atleast_1d(tb.getcell("CORR_TYPE", i)) for i in range(tb.nrows())]
    tb.close()

    # ---- main table ----
    tb.open(path, nomodify=True)
    if data_column not in tb.colnames():
        raise ValueError(f"MS has no {data_column} column")
    has_wtsp = "WEIGHT_SPECTRUM" in tb.colnames()

    query = f"FIELD_ID=={field_id} && ANTENNA1!=ANTENNA2"
    sel = tb.query(query)
    nsel = sel.nrows()
    if nsel == 0:
        tb.close()
        raise ValueError("no cross-correlation rows for the selected field")

    ms_rownr = np.asarray(sel.rownumbers(), dtype=np.int64)
    time = sel.getcol("TIME")  # seconds (MJD epoch)
    a1 = sel.getcol("ANTENNA1").astype(int)
    a2 = sel.getcol("ANTENNA2").astype(int)
    arr = sel.getcol("ARRAY_ID").astype(int)
    ddid = sel.getcol("DATA_DESC_ID").astype(int)
    uvw_m = np.asarray(sel.getcol("UVW")).T  # [n, 3] metres
    exposure = sel.getcol("EXPOSURE").astype(np.float32)
    flag_row = sel.getcol("FLAG_ROW").astype(bool)
    sel.close()

    # The IFs used, in DATA_DESC_ID order.
    used_dd = np.unique(ddid)
    corr0 = pol_types[dd_pol[used_dd[0]]]
    for d in used_dd:
        if not np.array_equal(pol_types[dd_pol[d]], corr0):
            raise ValueError("spectral windows with differing polarizations")
    pols = [_CASA_TO_AIPS.get(int(c)) for c in corr0]
    if any(p is None for p in pols):
        raise ValueError(f"unsupported CASA correlation types {corr0}")
    npol = len(pols)

    if_freq, if_df, if_nchan, coff = [], [], [], {}
    for d in used_dd:
        freqs = spw_freqs[dd_spw[d]]
        width = spw_width[dd_spw[d]]
        coff[int(d)] = int(np.sum(if_nchan))
        if_freq.append(float(freqs[0]))
        if_df.append(float(width[0] if len(freqs) < 2 else freqs[1] - freqs[0]))
        if_nchan.append(len(freqs))
    nctotal = int(np.sum(if_nchan))
    nif = len(if_nchan)

    # ---- merge rows of the same (time, array, baseline) ----
    key = np.rec.fromarrays([time, arr, a1, a2], names="t,s,a,b")
    ukey, row_of = np.unique(key, return_inverse=True)
    nrow = len(ukey)
    order = np.argsort(ukey, order=("t", "s", "a", "b"), kind="stable")
    rank = np.empty(nrow, dtype=int)
    rank[order] = np.arange(nrow)
    row_of = rank[row_of]
    ukey = ukey[order]

    ref_mjd = float(np.floor(ukey.t.min() / 86400.0))
    tsec = ukey.t - ref_mjd * 86400.0

    # Antenna entries per (array, antenna-in-table).
    arrays = np.unique(arr)
    nant_tab = len(ant_names_tab)
    ant_names, ant_xyz, ant_sub = [], [], []
    index_of = {}
    for isub, a in enumerate(arrays):
        for k in range(nant_tab):
            index_of[(int(a), k)] = len(ant_names)
            ant_names.append(ant_names_tab[k])
            ant_xyz.append(ant_pos[k])
            ant_sub.append(isub)

    ant1 = np.zeros(nrow, dtype=np.uint32)
    ant2 = np.zeros(nrow, dtype=np.uint32)
    uvw = np.zeros((nrow, 3), dtype=np.float64)
    inttime = np.zeros(nrow, dtype=np.float32)
    for k in range(nsel):
        r = row_of[k]
        ant1[r] = index_of[(int(arr[k]), int(a1[k]))]
        ant2[r] = index_of[(int(arr[k]), int(a2[k]))]
        uvw[r] = uvw_m[k] / C
        inttime[r] = exposure[k]

    vis = np.zeros((nrow, nctotal, npol), dtype=np.complex64)
    wt = np.zeros((nrow, nctotal, npol), dtype=np.float32)  # 0 = absent
    flag = np.ones((nrow, nctotal, npol), dtype=bool)  # absent = flagged
    # Provenance for writing flags back: MS main-table row number of
    # each (merged row, IF), or -1 where the IF was not observed.
    ms_row = np.full((nrow, nif), -1, dtype=np.int64)

    # ---- per-DDID bulk reads ----
    for d in used_dd:
        dsel = tb.query(f"{query} && DATA_DESC_ID=={int(d)}")
        n_d = dsel.nrows()
        if n_d == 0:
            dsel.close()
            continue
        data = np.asarray(dsel.getcol(data_column))  # [ncorr, nchan, n]
        flags = np.asarray(dsel.getcol("FLAG")).astype(bool)
        if has_wtsp:
            try:
                w = np.asarray(dsel.getcol("WEIGHT_SPECTRUM"))
            except RuntimeError:
                w = None
        else:
            w = None
        if w is None:
            w0 = np.asarray(dsel.getcol("WEIGHT"))  # [ncorr, n]
            w = np.broadcast_to(w0[:, np.newaxis, :], data.shape).copy()
        # Row indexes of this DDID within the merged table.
        dmask = ddid == d
        rows_d = row_of[dmask]
        dsel.close()

        data = np.transpose(data, (2, 1, 0))  # [n, nchan, ncorr]
        flags = np.transpose(flags, (2, 1, 0))
        w = np.abs(np.transpose(w, (2, 1, 0)).astype(np.float32))
        if wtscale != 1.0:
            w = w * np.float32(wtscale)
        # The FLAG column is carried through as-is (FLAG_ROW folded in);
        # weights stay non-negative.
        allflag = flags | flag_row[dmask][:, np.newaxis, np.newaxis]
        c0 = coff[int(d)]
        nch = data.shape[1]
        vis[rows_d, c0 : c0 + nch, :] = data.astype(np.complex64)
        wt[rows_d, c0 : c0 + nch, :] = w
        flag[rows_d, c0 : c0 + nch, :] = allflag
        ms_row[rows_d, int(np.nonzero(used_dd == d)[0][0])] = ms_rownr[dmask]
    tb.close()

    core = CoreObservation(
        source,
        ra,
        dec,
        2000.0,
        ant_names,
        np.asarray(ant_xyz, dtype=np.float64),
        [int(s) for s in ant_sub],
        if_freq,
        if_df,
        if_nchan,
        [int(p) for p in pols],
        np.ascontiguousarray(tsec),
        np.ascontiguousarray(inttime),
        np.ascontiguousarray(ant1),
        np.ascontiguousarray(ant2),
        np.ascontiguousarray(uvw),
        np.ascontiguousarray(vis),
        np.ascontiguousarray(wt),
        ref_mjd,
        flag=np.ascontiguousarray(flag),
    )
    # Provenance needed to write the FLAG column back (save_flags) and to
    # export CASA calibration tables (savecaltable).
    core._ms_origin = {
        "path": path,
        "data_column": data_column,
        "ms_row": ms_row,
        "if_nchan": list(if_nchan),
        "field_id": int(field_id),
        # SPECTRAL_WINDOW_ID of each difmapy IF, in IF order.
        "spw_ids": [int(dd_spw[d]) for d in used_dd],
        # Antenna table row of each difmapy antenna entry.
        "ant_ids": [int(k) % nant_tab for k in range(len(ant_names))],
    }
    # The part of that which survives averaging (the rows change, the
    # antennas, spectral windows and field do not), for savecaltable.
    # AIPS numbers antennas from 1 in ANTENNA-table order, which is what
    # CASA's exportuvfits writes.
    core._cal_origin = {
        "format": "ms",
        "path": path,
        "field_id": int(field_id),
        "spw_ids": list(core._ms_origin["spw_ids"]),
        "ant_numbers": [a + 1 for a in core._ms_origin["ant_ids"]],
        # What `save_averaged_ms` needs to lay out new main-table rows:
        # the DATA_DESC_ID of each IF, the ANTENNA row of each antenna
        # entry and the ARRAY_ID of each subarray.
        "dd_ids": [int(d) for d in used_dd],
        "ant_ids": list(core._ms_origin["ant_ids"]),
        "array_ids": [int(a) for a in arrays],
        "data_column": data_column,
    }
    return core


def save_flags(core, path=None, flag_row=True):
    """Write difmapy's FLAG column back into a Measurement Set.

    This is the standard MS way of recording flags (as opposed to
    difmap's practice of writing a new UV file). Only the FLAG (and
    optionally FLAG_ROW) column is modified; data and weights are left
    untouched.

    Parameters
    ----------
    core : CoreObservation
        Observation loaded from an MS (or `path` must be given).
    path : str | None
        Target MS; defaults to the MS the data were loaded from.
    flag_row : bool
        Also update FLAG_ROW (set when every correlation of a row is
        flagged), which many CASA tasks use as a fast path.
    """
    try:
        import casatools
    except ImportError as exc:  # pragma: no cover
        raise ImportError("writing MS flags requires casatools") from exc

    origin = getattr(core, "_ms_origin", None)
    averaged = origin is None
    if averaged:
        origin = getattr(core, "_ms_avg_origin", None)
    if origin is None:
        raise ValueError(
            "this observation was not loaded from a Measurement Set; "
            "use wobs() to write UVFITS instead"
        )
    if path is not None and os.path.abspath(path) != os.path.abspath(origin["path"]):
        raise ValueError(
            f"flags can only be written back to the originating MS "
            f"({origin['path']}); got {path}"
        )
    target = origin["path"]
    if averaged:
        return _save_flags_averaged(core, origin, flag_row)

    flags = np.asarray(core.flags())  # [nrow, nctotal, npol]
    ms_row = origin["ms_row"]  # [nrow, nif]
    if_nchan = origin["if_nchan"]

    tb = casatools.table()
    tb.open(target, nomodify=False)
    try:
        nwritten = _put_cube(
            tb, "FLAG", flags, ms_row, if_nchan,
            extra=("FLAG_ROW", lambda c: c.all(axis=(1, 2))) if flag_row else None,
        )
        tb.flush()
    finally:
        tb.close()
    return nwritten


def _save_flags_averaged(core, origin, flag_row):
    """`save_flags` for time- and/or channel-averaged data.

    An averaged sample stands for several samples of the MS, so its flag
    cannot simply be copied back. What can be said is: a flagged
    averaged sample means none of the samples behind it should be used,
    so they are all flagged; an unflagged one says nothing about each of
    them (some may have been flagged in the file, and were left out of
    the average), so they are left as they are. Flags are therefore only
    ever *added* to the MS from averaged data.
    """
    import casatools

    flags = np.asarray(core.flags())           # [nrow_avg, nc_avg, npol]
    ms_row = origin["ms_row"]                  # [nrow_orig, nif]
    row_map = np.asarray(origin["row_map"])    # orig row -> averaged row
    chan_map = np.asarray(origin["chan_map"])  # orig channel -> averaged
    tb = casatools.table()
    tb.open(origin["path"], nomodify=False)
    nwritten = 0
    try:
        coff = 0
        for i, nch in enumerate(origin["if_nchan"]):
            cmap = chan_map[coff : coff + nch]
            coff += nch
            rows = ms_row[:, i]
            valid = np.nonzero(rows >= 0)[0]
            used = cmap >= 0
            if valid.size == 0 or not used.any():
                continue
            order = np.argsort(rows[valid])
            src = valid[order]                 # difmapy rows, in MS order
            r = rows[src]
            # The averaged flags as seen from the original samples:
            # [n, nch, npol], False for channels left out of the average.
            add = np.zeros((len(src), nch, flags.shape[2]), dtype=bool)
            add[:, used, :] = flags[row_map[src]][:, cmap[used], :]
            breaks = np.nonzero(np.diff(r) != 1)[0] + 1
            for cr, ca in zip(np.split(r, breaks), np.split(add, breaks)):
                if not ca.any():
                    continue
                start, n = int(cr[0]), len(cr)
                old = np.asarray(tb.getcol("FLAG", startrow=start, nrow=n),
                                 dtype=bool)     # [npol, nch, n]
                new = old | np.transpose(ca, (2, 1, 0))
                if np.array_equal(new, old):
                    continue
                tb.putcol("FLAG", new, startrow=start, nrow=n)
                if flag_row:
                    tb.putcol("FLAG_ROW", new.all(axis=(0, 1)),
                              startrow=start, nrow=n)
                # Count the rows that gained a flag, not the block.
                nwritten += int((new != old).any(axis=(0, 1)).sum())
        tb.flush()
    finally:
        tb.close()
    return nwritten


def _raw_cube(core):
    """The observation's visibilities with none of its calibration
    applied (gains, gain flags and baseline corrections removed on a
    copy)."""
    raw = core.copy()
    raw.uncalib(True, True, True)
    raw.clroff()
    return np.asarray(raw.calibrated_cube()[0])


def _nearest_metadata(tb, src, field_id, times_mjds, columns):
    """The value of per-row bookkeeping columns (SCAN_NUMBER, ...) of the
    source MS at the row nearest in time to each of `times_mjds`."""
    tb.open(src)
    try:
        sel = tb.query(f"FIELD_ID=={int(field_id)}")
        try:
            t = np.asarray(sel.getcol("TIME"), dtype=np.float64)
            cols = {c: np.asarray(sel.getcol(c))
                    for c in columns if c in sel.colnames()}
        finally:
            sel.close()
    finally:
        tb.close()
    t_u, first = np.unique(t, return_index=True)
    idx = np.clip(np.searchsorted(t_u, times_mjds), 0, len(t_u) - 1)
    left = np.clip(idx - 1, 0, len(t_u) - 1)
    idx = np.where(np.abs(t_u[left] - times_mjds)
                   <= np.abs(t_u[idx] - times_mjds), left, idx)
    return {c: v[first][idx] for c, v in cols.items()}


def save_averaged_ms(core, path, overwrite=False, flag_row=True,
                     by_window=None):
    """Write time- and/or channel-averaged data as a new Measurement Set.

    Averaged rows no longer correspond to rows of the MS they came from,
    so the copy-and-fill of `save_ms` cannot be used. Instead the new MS
    takes its *structure* and every subtable (ANTENNA, FIELD,
    SPECTRAL_WINDOW, SOURCE, ...) from the originating MS, and its main
    table is built from the observation: one row per integration,
    baseline and spectral window that has data, carrying

    * ``DATA`` - the averaged visibilities as loaded (no difmapy
      calibration), ``CORRECTED_DATA`` - the same with the session's
      gains and baseline corrections applied;
    * ``FLAG``/``FLAG_ROW``, ``WEIGHT`` (the mean channel weight, which
      is what the loader spreads over the channels again), ``SIGMA``,
      and ``WEIGHT_SPECTRUM`` when the original had one;
    * ``TIME``/``TIME_CENTROID``, ``INTERVAL``/``EXPOSURE`` (the summed
      integration time), ``UVW`` in metres, the antenna, array, field
      and data-description ids, and the scan number, observation, state,
      processor and feed ids of the original row nearest in time.

    If channels were averaged, the SPECTRAL_WINDOW rows are rewritten
    with the new channel frequencies and widths. Autocorrelations are
    not carried over (difmapy never loads them), and neither is a
    MODEL_DATA column, which would be stale.

    `by_window` forces the cubes to be written one spectral window at a
    time (True) rather than each column in one piece; that is only
    needed, and chosen by default, when the windows have different
    numbers of channels.

    Returns the path written.
    """
    import shutil

    try:
        import casatools
    except ImportError as exc:  # pragma: no cover
        raise ImportError("writing a Measurement Set requires casatools") from exc

    origin = getattr(core, "_cal_origin", None) or {}
    if origin.get("format") != "ms" or "dd_ids" not in origin:
        raise ValueError(
            "this observation did not come from a Measurement Set, and "
            "difmapy writes one from the structure of the MS it came "
            "from; use wobs() to write UVFITS instead"
        )
    src = origin["path"]
    if not os.path.isdir(src):
        raise ValueError(
            f"the Measurement Set the data came from ({src}) is not there "
            "any more; its structure and subtables are needed to write a "
            "new one"
        )
    if os.path.abspath(path) == os.path.abspath(src):
        raise ValueError(f"refusing to overwrite the originating MS {src}")
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        shutil.rmtree(path)

    ifs = core.ifs
    nif, npol = len(ifs), core.npol
    cal, wt = (np.asarray(x) for x in core.calibrated_cube())
    raw = _raw_cube(core)
    flags = np.asarray(core.flags())
    time, a1, a2, us, vs, ws = (np.asarray(x) for x in core.rows())
    inttime = np.asarray(core.inttimes(), dtype=np.float64)
    coffs = np.concatenate([[0], np.cumsum([n for (_, _, n) in ifs])])

    # One MS row per (difmapy row, IF) that holds any data.
    present = np.stack(
        [(wt[:, coffs[i] : coffs[i + 1], :] != 0).any(axis=(1, 2))
         for i in range(nif)], axis=1)
    rows, cifs = np.nonzero(present)   # row-major: time order is kept
    n = len(rows)
    if n == 0:
        raise ValueError("there is no data to write")

    t_mjds = core.ref_mjd * 86400.0 + time[rows]
    tb = casatools.table()
    meta = _nearest_metadata(
        tb, src, origin.get("field_id", 0), t_mjds,
        ("SCAN_NUMBER", "OBSERVATION_ID", "STATE_ID", "PROCESSOR_ID",
         "FEED1", "FEED2"))

    # The structure, without rows - which also empties the subtables, so
    # those are then copied over in full.
    tb.open(src)
    try:
        subtables = [k for k, v in tb.getkeywords().items()
                     if isinstance(v, str) and v.startswith("Table:")]
        had_wtsp = "WEIGHT_SPECTRUM" in tb.colnames()
        out = tb.copy(path, deep=True, valuecopy=True, norows=True,
                      returnobject=True)
        out.close()
    finally:
        tb.close()
    for sub in subtables:
        if not os.path.isdir(os.path.join(src, sub)):
            continue
        dest = os.path.join(path, sub)
        shutil.rmtree(dest, ignore_errors=True)
        tb.open(os.path.join(src, sub))
        try:
            copy = tb.copy(dest, deep=True, valuecopy=True, returnobject=True)
            copy.close()
        finally:
            tb.close()

    ant_ids = np.asarray(origin["ant_ids"], dtype=np.int32)
    sub_of = np.asarray(core.antenna_subarrays, dtype=int)
    array_ids = np.asarray(origin["array_ids"], dtype=np.int32)
    dd_ids = np.asarray(origin["dd_ids"], dtype=np.int32)

    tb.open(path, nomodify=False)
    try:
        stale = [c for c in ("MODEL_DATA", "SIGMA_SPECTRUM")
                 if c in tb.colnames()]
        if stale:
            tb.removecols(stale)
        if "CORRECTED_DATA" not in tb.colnames():
            _add_data_column(tb, "CORRECTED_DATA")
        tb.addrows(n)
        tb.putcol("TIME", t_mjds)
        tb.putcol("TIME_CENTROID", t_mjds)
        tb.putcol("INTERVAL", inttime[rows])
        tb.putcol("EXPOSURE", inttime[rows])
        tb.putcol("UVW", np.ascontiguousarray(
            np.stack([us[rows], vs[rows], ws[rows]]) * C))
        tb.putcol("ANTENNA1", ant_ids[a1[rows]])
        tb.putcol("ANTENNA2", ant_ids[a2[rows]])
        tb.putcol("ARRAY_ID", array_ids[sub_of[a1[rows]]])
        tb.putcol("DATA_DESC_ID", dd_ids[cifs])
        tb.putcol("FIELD_ID",
                  np.full(n, int(origin.get("field_id", 0)), dtype=np.int32))
        for col, values in meta.items():
            tb.putcol(col, np.ascontiguousarray(values))

        # The cubes: [n, nchan, npol] -> the MS's [npol, nchan, n].
        def write(target, cells):
            w = np.abs(cells(wt)).astype(np.float32)
            f = cells(flags).astype(bool)
            cnt = np.maximum((w > 0).sum(axis=1), 1)
            weight = (w.sum(axis=1) / cnt).astype(np.float32)   # [npol, n]
            with np.errstate(divide="ignore"):
                sigma = np.where(weight > 0, 1.0 / np.sqrt(weight), 0.0)
            target.putcol("DATA", cells(raw))
            target.putcol("CORRECTED_DATA", cells(cal))
            target.putcol("FLAG", f)
            target.putcol("FLAG_ROW", f.all(axis=(0, 1)) if flag_row
                          else np.zeros(f.shape[2], bool))
            target.putcol("WEIGHT", weight)
            target.putcol("SIGMA", sigma.astype(np.float32))
            if had_wtsp:
                target.putcol("WEIGHT_SPECTRUM", w)

        nchans = {n for (_, _, n) in ifs}
        if len(nchans) == 1 and not by_window:
            # Every IF has the same shape: each column in one piece,
            # which is some twenty times faster than window by window.
            chan = coffs[cifs][:, None] + np.arange(nchans.pop())[None, :]

            def cells(cube):
                return np.ascontiguousarray(
                    np.transpose(cube[rows[:, None], chan, :], (2, 1, 0)))

            write(tb, cells)
        else:
            # Channel counts differ, so one spectral window at a time. A
            # query result is a view of those rows, ascending like `sel`.
            for i in range(nif):
                sel = np.nonzero(cifs == i)[0]
                if sel.size == 0:
                    continue

                def cells(cube, sel=sel, i=i):
                    block = cube[rows[sel], coffs[i] : coffs[i + 1], :]
                    return np.ascontiguousarray(np.transpose(block, (2, 1, 0)))

                view = tb.query(f"DATA_DESC_ID=={int(dd_ids[i])}")
                try:
                    write(view, cells)
                finally:
                    view.close()
        tb.flush()
    finally:
        tb.close()

    _update_spectral_windows(tb, path, origin["spw_ids"], ifs)
    return path


def _update_spectral_windows(tb, path, spw_ids, ifs):
    """Make the SPECTRAL_WINDOW rows describe the (possibly averaged)
    channels of each IF; rows that already do are left untouched."""
    tb.open(os.path.join(path, "SPECTRAL_WINDOW"), nomodify=False)
    try:
        names = tb.colnames()
        for spw, (f0, df, nch) in zip(spw_ids, ifs):
            spw = int(spw)
            freqs = f0 + df * np.arange(nch)
            old = np.atleast_1d(tb.getcell("CHAN_FREQ", spw))
            if len(old) == nch and np.allclose(old, freqs, rtol=0, atol=1e-3):
                continue
            # The sign convention of CHAN_WIDTH is the file's own.
            old_w = np.atleast_1d(tb.getcell("CHAN_WIDTH", spw))
            width = np.full(nch, abs(df) if old_w[0] >= 0 else -abs(df))
            tb.putcell("NUM_CHAN", spw, int(nch))
            tb.putcell("CHAN_FREQ", spw, freqs)
            tb.putcell("CHAN_WIDTH", spw, width)
            for col in ("EFFECTIVE_BW", "RESOLUTION"):
                if col in names:
                    tb.putcell(col, spw, np.full(nch, abs(df)))
            if "TOTAL_BANDWIDTH" in names:
                tb.putcell("TOTAL_BANDWIDTH", spw, float(nch * abs(df)))
        tb.flush()
    finally:
        tb.close()


def _put_cube(tb, column, cube, ms_row, if_nchan, extra=None):
    """Write a difmapy cube `[nrow, nctotal, npol]` into an MS column.

    difmapy merges the MS rows of one (time, baseline) across spectral
    windows into a single row covering every IF, so writing back means
    undoing that: each IF's channels go to its own MS row, given by
    `ms_row[:, if]` (negative where the MS had no such row).

    `extra` is an optional ``(name, func)`` writing a second, per-row
    column derived from the same block (FLAG_ROW from FLAG).

    Returns the number of MS rows written.
    """
    nwritten = 0
    coff = 0
    for i, nch in enumerate(if_nchan):
        rows = ms_row[:, i]
        valid = rows >= 0
        if not valid.any():
            coff += nch
            continue
        r = rows[valid]
        # [n, nchan, npol] -> MS cell order [npol, nchan]
        block = cube[valid, coff : coff + nch, :]
        cells = np.transpose(block, (0, 2, 1))
        order = np.argsort(r)
        r_sorted = r[order]
        cells = cells[order]
        # Contiguous runs can be written in one putcol call.
        breaks = np.nonzero(np.diff(r_sorted) != 1)[0] + 1
        for chunk_r, chunk_c in zip(
            np.split(r_sorted, breaks), np.split(cells, breaks)
        ):
            # chunk_c is [n, npol, nchan]; the MS column wants
            # [npol, nchan, n].
            arr = np.ascontiguousarray(np.transpose(chunk_c, (1, 2, 0)))
            tb.putcol(column, arr, startrow=int(chunk_r[0]), nrow=len(chunk_r))
            if extra is not None:
                name, func = extra
                tb.putcol(
                    name, func(chunk_c), startrow=int(chunk_r[0]),
                    nrow=len(chunk_r),
                )
            nwritten += len(chunk_r)
        coff += nch
    return nwritten


def save_ms(core, path, data_column="CORRECTED_DATA", overwrite=False,
            flag_row=True):
    """Write the observation out as a Measurement Set at `path`.

    difmapy has no MS writer of its own: an MS is a directory of
    reference-counted CASA tables with subtables (ANTENNA, FIELD,
    SPECTRAL_WINDOW, SOURCE, ...) that no difmapy session holds enough
    of to rebuild faithfully. So the MS the data were *loaded from* is
    copied and the session's own results are written into the copy -
    the calibrated visibilities into `data_column` and the current
    flags into FLAG. Everything else (metadata, subtables, the
    untouched DATA column) is preserved exactly, which is what makes
    the result usable by CASA.

    The original MS is never modified; see `save_flags` for writing
    flags back into it in place.

    Parameters
    ----------
    core : CoreObservation
        Observation loaded from an MS.
    path : str
        Path of the MS to create.
    data_column : str | None
        Column the calibrated visibilities go to, created if the MS
        has none. None writes only the flags.
    overwrite : bool
        Replace `path` if it already exists.
    flag_row : bool
        Also maintain FLAG_ROW, as `save_flags` does.

    Returns
    -------
    str
        The path written.
    """
    import shutil

    try:
        import casatools
    except ImportError as exc:  # pragma: no cover
        raise ImportError("writing a Measurement Set requires casatools") from exc

    origin = getattr(core, "_ms_origin", None)
    if origin is None:
        raise ValueError(
            "this observation was not loaded from a Measurement Set, and "
            "difmapy can only write one by copying the MS it came from; "
            "use wobs() to write UVFITS instead"
        )
    src = origin["path"]
    if os.path.abspath(path) == os.path.abspath(src):
        raise ValueError(
            f"refusing to overwrite the originating MS {src}; "
            "use save_flags() to write flags back into it"
        )
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True")
        shutil.rmtree(path)
    shutil.copytree(src, path)

    ms_row = origin["ms_row"]  # [nrow, nif]
    if_nchan = origin["if_nchan"]
    flags = np.asarray(core.flags())  # [nrow, nctotal, npol]

    tb = casatools.table()
    tb.open(path, nomodify=False)
    try:
        _put_cube(
            tb, "FLAG", flags, ms_row, if_nchan,
            extra=("FLAG_ROW", lambda c: c.all(axis=(1, 2))) if flag_row else None,
        )
        if data_column:
            vis, _ = core.calibrated_cube()
            vis = np.asarray(vis)
            if data_column not in tb.colnames():
                _add_data_column(tb, data_column)
            _put_cube(tb, data_column, vis, ms_row, if_nchan)
        tb.flush()
    finally:
        tb.close()
    return path


def _add_data_column(tb, name, like="DATA"):
    """Add a complex visibility column shaped like an existing one."""
    desc = tb.getcoldesc(like)
    desc.pop("comment", None)
    tb.addcols({name: desc})

