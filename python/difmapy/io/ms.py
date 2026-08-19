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

__all__ = ["load_ms", "save_flags"]

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
    # Provenance needed to write the FLAG column back (save_flags).
    core._ms_origin = {
        "path": path,
        "data_column": data_column,
        "ms_row": ms_row,
        "if_nchan": list(if_nchan),
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

    flags = np.asarray(core.flags())  # [nrow, nctotal, npol]
    ms_row = origin["ms_row"]  # [nrow, nif]
    if_nchan = origin["if_nchan"]

    tb = casatools.table()
    tb.open(target, nomodify=False)
    try:
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
            block = flags[valid, coff : coff + nch, :]
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
                tb.putcol("FLAG", arr, startrow=int(chunk_r[0]), nrow=len(chunk_r))
                if flag_row:
                    tb.putcol(
                        "FLAG_ROW",
                        chunk_c.all(axis=(1, 2)),
                        startrow=int(chunk_r[0]),
                        nrow=len(chunk_r),
                    )
                nwritten += len(chunk_r)
            coff += nch
        tb.flush()
    finally:
        tb.close()
    return nwritten
