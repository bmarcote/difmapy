"""Export difmapy's accumulated calibration as a CASA calibration table.

difmapy keeps calibration out of the data: the antenna gains built up by
`selfcal`, `gscale` and friends live in a separate table in memory and
are applied on the fly. That maps directly onto CASA's model, where a
calibration table is applied with `applycal` to produce
CORRECTED_DATA - so the gains can be exported and reused, for instance
on the full-resolution Measurement Set that the imaging was not done on.

Conventions
-----------
difmapy (like difmap) stores the *correction* applied to the data:

    V_corrected = V_raw * c_p * conj(c_q),    c_a = amp_a * exp(i phi_a)

CASA's `applycal` instead divides by the antenna *gains*:

    CORRECTED_DATA = DATA / (G_p * conj(G_q))

so the exported table holds the reciprocal, ``G_a = 1 / c_a``. This is
verified against CASA itself in the test suite: applying the exported
table with `applycal` reproduces difmapy's corrected visibilities.

What can and cannot be exported
-------------------------------
Antenna-based amplitude and phase corrections become a standard
"G Jones" table. Baseline-based corrections (`resoff`) have no
antenna-based equivalent, and a phase-centre `shift` is a coordinate
change rather than a calibration; both are reported rather than
silently dropped. Use `wobs()` to write data with everything applied.
"""

from __future__ import annotations

import os
import shutil

import numpy as np

__all__ = ["save_caltable"]

# Column layout of a CASA NewCalTable ("G Jones"), as produced by
# gaincal: eight scalar columns and five variable-shaped array columns.
_SCALARS = {
    "TIME": "double",
    "FIELD_ID": "int",
    "SPECTRAL_WINDOW_ID": "int",
    "ANTENNA1": "int",
    "ANTENNA2": "int",
    "INTERVAL": "double",
    "SCAN_NUMBER": "int",
    "OBSERVATION_ID": "int",
}
_ARRAYS = {
    "CPARAM": "complex",
    "PARAMERR": "float",
    "FLAG": "boolean",
    "SNR": "float",
    "WEIGHT": "float",
}
_SUBTABLES = ("ANTENNA", "FIELD", "OBSERVATION", "SPECTRAL_WINDOW", "HISTORY")


def _table_description():
    """The table description CASA's own G tables use."""
    desc = {}
    for name, vtype in _SCALARS.items():
        desc[name] = {
            "valueType": vtype,
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "MSMTAB",
            "option": 5,
            "maxlen": 0,
            "comment": "",
        }
    for name, vtype in _ARRAYS.items():
        desc[name] = {
            "valueType": vtype,
            "dataManagerType": "StandardStMan",
            "dataManagerGroup": "MSMTAB",
            "option": 0,
            "ndim": -1,  # shape varies per row
            "maxlen": 0,
            "comment": "",
        }
    return desc


def _antenna_index_by_name(tb, ms_path, names):
    """Map difmapy antenna names onto ANTENNA table rows of `ms_path`.

    Matching by name (not position) is what lets a table exported from
    one Measurement Set be applied to another with the same stations.
    """
    tb.open(os.path.join(ms_path, "ANTENNA"))
    try:
        ms_names = [str(n).strip() for n in tb.getcol("NAME")]
    finally:
        tb.close()
    lookup = {n.upper(): i for i, n in enumerate(ms_names)}
    out = []
    for name in names:
        key = str(name).strip().upper()
        if key not in lookup:
            raise ValueError(
                f"antenna {name!r} is not in {ms_path}/ANTENNA "
                f"(it has {ms_names}); the table cannot be applied there"
            )
        out.append(lookup[key])
    return out


def _scan_and_observation(tb, ms_path, times_mjds, field_id):
    """The SCAN_NUMBER and OBSERVATION_ID in force at each solution time.

    Taken from the Measurement Set so that scan-aware interpolation in
    `applycal` behaves as it would for a table CASA wrote itself.
    """
    tb.open(ms_path)
    try:
        sel = tb.query(f"FIELD_ID=={int(field_id)}")
        try:
            t = np.asarray(sel.getcol("TIME"), dtype=np.float64)
            scan = np.asarray(sel.getcol("SCAN_NUMBER"), dtype=np.int32)
            obs = np.asarray(sel.getcol("OBSERVATION_ID"), dtype=np.int32)
        finally:
            sel.close()
    finally:
        tb.close()
    order = np.argsort(t)
    t, scan, obs = t[order], scan[order], obs[order]
    # Nearest MS row in time for each solution.
    idx = np.searchsorted(t, times_mjds)
    idx = np.clip(idx, 0, len(t) - 1)
    left = np.clip(idx - 1, 0, len(t) - 1)
    take_left = np.abs(t[left] - times_mjds) <= np.abs(t[idx] - times_mjds)
    idx = np.where(take_left, left, idx)
    return scan[idx], obs[idx]


def save_caltable(obs, path, ms=None, spw_ids=None, flag_uncalibrated=False,
                  overwrite=True, since=None):
    """Write the accumulated antenna gains as a CASA "G Jones" table.

    Parameters
    ----------
    obs : difmapy.Observation
        The observation whose gains are exported. The table is a
        snapshot of everything applied so far (the gains accumulate as
        `selfcal`/`gscale` are run), so exporting again later gives a
        table that supersedes, rather than complements, this one.
    path : str
        Output table (a directory, as CASA tables are).
    ms : str | None
        Measurement Set whose ANTENNA/FIELD/SPECTRAL_WINDOW/OBSERVATION
        metadata the table refers to; defaults to the MS the data were
        loaded from. Required for UVFITS-loaded data. Antennas are
        matched by *name*, so a compatible MS other than the original
        can be used.
    spw_ids : list[int] | None
        SPECTRAL_WINDOW_ID for each difmapy IF, in order. Defaults to
        the ids the data came from, or 0..nif-1 for UVFITS input.
    flag_uncalibrated : bool
        If true, mark solutions that were never actually solved for as
        flagged, so `applycal` flags the corresponding data. By default
        they are written as unity and left unflagged, which leaves such
        data untouched rather than discarding it.
    since : gain snapshot | None
        If given (from `Observation.gain_snapshot()`), export only the
        calibration accumulated *since* that point. Applying the
        resulting chain of tables together, as CASA does with
        ``gaintable=[t1, t2, ...]``, is equivalent to applying one
        cumulative table.
    """
    try:
        import casatools
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "writing CASA calibration tables requires casatools"
        ) from exc

    core = obs._core
    origin = getattr(core, "_ms_origin", None)
    ms_path = ms or (origin or {}).get("path")
    if not ms_path:
        raise ValueError(
            "a Measurement Set is needed to write a calibration table "
            "(its antenna, spectral window and field metadata are "
            "referenced); pass ms=... for UVFITS-loaded data"
        )
    if not os.path.isdir(ms_path):
        raise ValueError(f"not a Measurement Set: {ms_path}")

    warnings = []
    if any(abs(v) > 0 for v in core.shift_total):
        e, n = obs.total_shift
        warnings.append(
            f"a phase-centre shift of ({e:.4g}, {n:.4g}) mas is in effect; "
            "a calibration table cannot express it (use CASA's phaseshift, "
            "or wobs(freeze_shift=True) to bake it into the data)"
        )
    bls, bamp, bphs = core.baseline_corrections()
    if len(bls) and not (np.allclose(bamp, 1.0) and np.allclose(bphs, 0.0)):
        warnings.append(
            "baseline-based corrections from resoff() are in effect; these "
            "have no antenna-based equivalent and are NOT included in the "
            "table (use wobs() to write data with them applied)"
        )

    nant = len(core.antenna_names)
    nif = core.nif
    ntimes = core.ntimes
    amp, phs, bad = (np.asarray(x) for x in core.gains())
    amp = amp.reshape(ntimes, nif, nant).astype(np.float64)
    phs = phs.reshape(ntimes, nif, nant).astype(np.float64)
    bad = bad.reshape(ntimes, nif, nant)
    used = np.asarray(core.gains_used()).reshape(ntimes, nif, nant)

    if since is not None:
        # Export only what has been accumulated since the snapshot, by
        # dividing out the corrections that were already in place.
        amp0, phs0, used0 = since["amp"], since["phs"], since["used"]
        if amp0.shape != amp.shape:
            raise ValueError(
                "the gain snapshot does not match this observation "
                f"(snapshot {amp0.shape}, now {amp.shape}); it must come "
                "from the same observation without a reselection"
            )
        amp = np.where(amp0 != 0, amp / np.where(amp0 != 0, amp0, 1.0), amp)
        phs = phs - phs0
        del used0  # `used` still means "a solution was applied here"

    if spw_ids is None:
        spw_ids = (origin or {}).get("spw_ids") or list(range(nif))
    spw_ids = [int(s) for s in spw_ids]
    if len(spw_ids) != nif:
        raise ValueError(f"spw_ids has {len(spw_ids)} entries for {nif} IFs")
    field_id = int((origin or {}).get("field_id", 0))

    tb = casatools.table()
    ant_ids = _antenna_index_by_name(tb, ms_path, core.antenna_names)

    # Solution times as MJD seconds, matching the Measurement Set.
    times_mjds = core.ref_mjd * 86400.0 + np.asarray(core.times())
    # Solution interval: the integration length, from the data.
    inttime = np.asarray(core.inttimes(), dtype=np.float64)
    tidx = np.asarray(core.time_index())
    interval = np.zeros(ntimes)
    np.maximum.at(interval, tidx, inttime)
    scan, obsid = _scan_and_observation(tb, ms_path, times_mjds, field_id)

    # One row per (time, IF, antenna), as CASA's own G tables have.
    n = ntimes * nif * nant
    t_col = np.repeat(times_mjds, nif * nant)
    iv_col = np.repeat(interval, nif * nant)
    scan_col = np.repeat(scan, nif * nant).astype(np.int32)
    obs_col = np.repeat(obsid, nif * nant).astype(np.int32)
    spw_col = np.tile(np.repeat(np.asarray(spw_ids, dtype=np.int32), nant), ntimes)
    a1_col = np.tile(np.asarray(ant_ids, dtype=np.int32), ntimes * nif)

    # The gain is the reciprocal of the correction difmapy applies.
    corr = amp * np.exp(1j * phs)
    gain = np.where(np.abs(corr) > 0, 1.0 / corr, 1.0 + 0j).ravel()
    flag = bad.ravel().copy()
    if flag_uncalibrated:
        flag |= ~used.ravel()
    # A flagged solution must not carry a meaningful value.
    gain = np.where(flag, 1.0 + 0j, gain)

    # CPARAM/FLAG/... are [ncorr, nchan, nrow]; the gains are
    # polarization-independent (they are solved on the total-intensity
    # stream), so both parallel hands get the same value.
    ncorr, nchan = 2, 1
    cparam = np.tile(gain.astype(np.complex64), (ncorr, nchan, 1))
    flagcol = np.tile(flag, (ncorr, nchan, 1))
    snr = np.where(flagcol, 0.0, 1.0).astype(np.float32)
    paramerr = np.zeros((ncorr, nchan, n), dtype=np.float32)

    if os.path.exists(path):
        if not overwrite:
            raise ValueError(f"{path} exists")
        shutil.rmtree(path)

    tb.create(path, _table_description())
    try:
        tb.addrows(n)
        tb.putcol("TIME", t_col)
        tb.putcol("INTERVAL", iv_col)
        tb.putcol("ANTENNA1", a1_col)
        # -1 marks an antenna-based (as opposed to baseline-based) table.
        tb.putcol("ANTENNA2", np.full(n, -1, dtype=np.int32))
        tb.putcol("SPECTRAL_WINDOW_ID", spw_col)
        tb.putcol("FIELD_ID", np.full(n, field_id, dtype=np.int32))
        tb.putcol("SCAN_NUMBER", scan_col)
        tb.putcol("OBSERVATION_ID", obs_col)
        tb.putcol("CPARAM", cparam)
        tb.putcol("PARAMERR", paramerr)
        tb.putcol("FLAG", flagcol)
        tb.putcol("SNR", snr)
        # WEIGHT is left unwritten, as in CASA's own tables.
        tb.putkeyword("VisCal", "G Jones")
        tb.putkeyword("ParType", "Complex")
        tb.putkeyword("PolBasis", "unknown")
        tb.putkeyword("MSName", os.path.basename(os.path.normpath(ms_path)))
        tb.putkeyword("difmapy", "written by difmapy")
        # The table *info* is what CASA checks to recognise a caltable;
        # it is distinct from the keywords above.
        tb.putinfo({"type": "Calibration", "subType": "G Jones", "readme": ""})
        # CASA's conformance check requires units on the time columns,
        # and reads TIME as a UTC epoch.
        tb.putcolkeyword("TIME", "QuantumUnits", ["s"])
        tb.putcolkeyword("TIME", "MEASINFO", {"Ref": "UTC", "type": "epoch"})
        tb.putcolkeyword("INTERVAL", "QuantumUnits", ["s"])
    finally:
        tb.close()

    # The subtables are copies of the Measurement Set's, which is how
    # CASA builds them.
    for sub in _SUBTABLES:
        src = os.path.join(ms_path, sub)
        if not os.path.isdir(src):
            continue
        dest = os.path.join(path, sub)
        tb.open(src)
        try:
            copy = tb.copy(dest, deep=True, valuecopy=True)
            if copy is not None:
                copy.close()
        finally:
            tb.close()
        tb.open(path, nomodify=False)
        try:
            tb.putkeyword(sub, f"Table: {dest}")
        finally:
            tb.close()

    return {"path": path, "nrows": n, "nflagged": int(flag.sum()),
            "warnings": warnings}
