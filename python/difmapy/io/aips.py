"""Export difmapy's calibration as an AIPS SN table, in a TASAV file.

AIPS keeps calibration in tables attached to the UV data - SN
("solution") tables are what CALIB writes. The way to carry tables
between AIPS catalog entries, or between installations, is TASAV: a
catalog entry holding a single dummy visibility and the tables, written
to disk with FITTP. `save_sntable` writes such a file directly, with the
antenna (AN) and frequency (FQ) tables alongside the SN table, laid out
as AIPS 31DEC24's own TASAV/FITTP output is (SN revision 11).

In AIPS::

    FITLD  the file                      -> xxx.TASAV.1
    TACOP  INEXT 'SN', from the TASAV entry to the UV data
    then, for single-source data, DOCAL 1 with GAINUSE = that SN
    version applies it directly (SPLIT, IMAGR, ...); for multi-source
    data, CLCAL turns it into a new CL table first.

Conventions
-----------
AIPS SN gains multiply the data, as difmapy's corrections do - but with
the opposite phase convention:

    AIPS:     V_corrected = V_obs * conj(g_p) * g_q
    difmapy:  V_corrected = V_obs * c_p * conj(c_q)

so the table holds ``g = conj(c)``: difmapy's amplitude as it is, and
its phase negated. (A CASA table instead holds the reciprocal, 1/c.)
Both were established, and are checked by the test suite, by applying a
written table in AIPS itself (TACOP, then SPLIT with DOCAL) and
comparing the result with difmapy's own corrected visibilities.

Antennas in an SN table are AIPS station numbers, not names, so they
are matched by name against the AN table of the UV data the table is
meant for: the file the observation was loaded from, or another one
given as `uvfits=`. Times are days from that file's reference date.
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits
from astropy.time import Time

__all__ = ["save_sntable", "save_fgtable"]

#: SN table revision written by AIPS 31DEC24 (with the dispersive-delay
#: columns DISP/DDISP).
SN_REVISION = 11


def _target_tables(path):
    """AN tables, FQ table, reference JD and primary-header axes of an
    existing UVFITS file, which the SN table must be consistent with."""
    with fits.open(path, memmap=True) as hdul:
        ghdu = hdul[0]
        hdr = ghdu.header
        an = sorted(
            (h for h in hdul[1:] if h.name in ("AIPS AN", "AN")),
            key=lambda h: h.header.get("EXTVER", 1),
        )
        fq = next((h for h in hdul[1:] if h.name in ("AIPS FQ", "FQ")), None)
        if not an:
            raise ValueError(f"{path} has no AIPS AN antenna table")
        an = [fits.BinTableHDU(data=h.data.copy(), header=h.header.copy())
              for h in an]
        fq = (None if fq is None else
              fits.BinTableHDU(data=fq.data.copy(), header=fq.header.copy()))
        axes = _axes_of(hdr)
        ref_jd = _reference_jd(ghdu, an[0].header)
        keep = {k: hdr[k] for k in ("OBJECT", "TELESCOP", "INSTRUME",
                                    "OBSERVER", "EQUINOX") if k in hdr}
    return an, fq, ref_jd, axes, keep


def _axes_of(hdr):
    """(CTYPE, NAXIS, CRVAL, CDELT, CRPIX) of the data axes after
    COMPLEX (axis 1 is the empty random-groups axis, 2 is COMPLEX)."""
    out = []
    for i in range(3, hdr["NAXIS"] + 1):
        out.append((
            str(hdr.get(f"CTYPE{i}", "")).strip(),
            int(hdr[f"NAXIS{i}"]),
            float(hdr.get(f"CRVAL{i}", 1.0)),
            float(hdr.get(f"CDELT{i}", 1.0)),
            float(hdr.get(f"CRPIX{i}", 1.0)),
        ))
    return out


def _reference_jd(ghdu, an_header):
    """The file's reference date as a JD at 0h: the AN table's RDATE
    (which is what AIPS counts SN times from), else DATE-OBS, else the
    0h of the day of the first visibility, which is the loader's
    reference too."""
    for value in (an_header.get("RDATE"), ghdu.header.get("DATE-OBS")):
        if value:
            day = str(value).strip()[:10]
            try:
                return float(Time(day, format="iso", scale="utc").jd)
            except ValueError:
                pass
    # par() sums a parameter that is split in two (DATE, DATE).
    jd = np.asarray(ghdu.data.par("DATE"), dtype=np.float64)
    return float(np.floor(jd.min() - 2400000.5) + 2400000.5)


def _match_antennas(core, an_tables):
    """(subarray, station number) in the target file of each difmapy
    antenna entry, by name."""
    names = core.antenna_names
    subs = list(core.antenna_subarrays)
    if max(subs) + 1 > len(an_tables):
        raise ValueError(
            f"the observation has {max(subs) + 1} subarrays but the target "
            f"UV data has AN tables for {len(an_tables)}"
        )
    lookup = []
    for an in an_tables:
        lookup.append({
            str(n).strip().upper(): int(k)
            for n, k in zip(an.data["ANNAME"], an.data["NOSTA"])
        })
    out = []
    for name, s in zip(names, subs):
        key = str(name).strip().upper()
        if key not in lookup[s]:
            raise ValueError(
                f"antenna {name!r} is not in the target's AN table for "
                f"subarray {s + 1} (it has {sorted(lookup[s])}); the table "
                "could not be applied there"
            )
        out.append((s + 1, lookup[s][key]))
    return out


def _synthesised_tables(core, numbers):
    """AN and FQ tables for an observation with no UVFITS file behind it
    (loaded from a Measurement Set), numbered as `numbers` say."""
    names = core.antenna_names
    subs = list(core.antenna_subarrays)
    xyz = np.asarray(core.antenna_xyz)
    ifs = core.ifs
    ref_jd = core.ref_mjd + 2400000.5
    rdate = Time(ref_jd, format="jd", scale="utc").strftime("%Y-%m-%d")
    an_tables = []
    for s in range(max(subs) + 1):
        idx = [g for g in range(len(names)) if subs[g] == s]
        an = fits.BinTableHDU.from_columns([
            fits.Column(name="ANNAME", format="8A",
                        array=np.array([names[g] for g in idx])),
            fits.Column(name="STABXYZ", format="3D", unit="METERS",
                        array=xyz[idx]),
            fits.Column(name="NOSTA", format="1J",
                        array=np.array([numbers[g] for g in idx], np.int32)),
            fits.Column(name="MNTSTA", format="1J",
                        array=np.zeros(len(idx), dtype=np.int32)),
            fits.Column(name="STAXOF", format="1E", unit="METERS",
                        array=np.zeros(len(idx), dtype=np.float32)),
            fits.Column(name="POLTYA", format="1A",
                        array=np.array(["R"] * len(idx))),
            fits.Column(name="POLTYB", format="1A",
                        array=np.array(["L"] * len(idx))),
        ])
        an.name = "AIPS AN"
        h = an.header
        h["EXTVER"] = s + 1
        h["ARRAYX"] = h["ARRAYY"] = h["ARRAYZ"] = 0.0
        h["FREQ"] = ifs[0][0]
        h["RDATE"] = rdate
        h["TIMSYS"] = "UTC"
        h["NO_IF"] = len(ifs)
        an_tables.append(an)
    nif = len(ifs)
    fq = fits.BinTableHDU.from_columns([
        fits.Column(name="FRQSEL", format="1J", array=np.array([1], np.int32)),
        fits.Column(name="IF FREQ", format=f"{nif}D", unit="HZ",
                    array=np.array([[f - ifs[0][0] for (f, _, _) in ifs]])),
        fits.Column(name="CH WIDTH", format=f"{nif}E", unit="HZ",
                    array=np.array([[df for (_, df, _) in ifs]], np.float32)),
        fits.Column(name="TOTAL BANDWIDTH", format=f"{nif}E", unit="HZ",
                    array=np.array([[abs(df) * n for (_, df, n) in ifs]],
                                   np.float32)),
        fits.Column(name="SIDEBAND", format=f"{nif}J",
                    array=np.array([[1 if df >= 0 else -1 for (_, df, _) in ifs]],
                                   np.int32)),
    ])
    fq.name = "AIPS FQ"
    fq.header["NO_IF"] = nif
    return an_tables, fq, ref_jd


def _synthesised_axes(core):
    """Primary-header data axes describing the observation itself."""
    ifs = core.ifs
    pols = sorted(core.pols, key=abs) if all(p < 0 for p in core.pols) \
        else sorted(core.pols)
    steps = np.diff(pols)
    cdelt_s = float(steps[0]) if len(pols) > 1 and np.all(steps == steps[0]) \
        else (-1.0 if pols[0] < 0 else 1.0)
    return [
        ("STOKES", len(pols), float(pols[0]), cdelt_s, 1.0),
        ("FREQ", max(n for (_, _, n) in ifs), ifs[0][0], ifs[0][1], 1.0),
        ("IF", len(ifs), 1.0, 1.0, 1.0),
        ("RA", 1, float(np.rad2deg(core.ra)), 1.0, 1.0),
        ("DEC", 1, float(np.rad2deg(core.dec)), 1.0, 1.0),
    ]


def _dummy_groups(axes, ref_jd, keep, date_obs):
    """The single all-zero visibility a TASAV entry carries, with the
    data axes of the UV data it belongs to (TACOP checks them)."""
    shape = [1] + [n for (_, n, _, _, _) in reversed(axes)] + [3]
    data = np.zeros(shape, dtype=np.float32)
    parnames = ["UU---SIN", "VV---SIN", "WW---SIN", "DATE", "DATE",
                "BASELINE", "FREQSEL"]
    pardata = [np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
               np.zeros(1), np.full(1, -0.01), np.zeros(1)]
    groups = fits.GroupData(
        data, parnames=parnames, pardata=pardata, bitpix=-32,
        parbzeros=[0.0, 0.0, 0.0, ref_jd, 0.0, 0.0, 0.0],
    )
    ghdu = fits.GroupsHDU(groups)
    hdr = ghdu.header
    hdr["CTYPE2"], hdr["CRVAL2"], hdr["CDELT2"], hdr["CRPIX2"] = (
        "COMPLEX", 1.0, 1.0, 1.0)
    for i, (ctype, _, crval, cdelt, crpix) in enumerate(axes, start=3):
        hdr[f"CTYPE{i}"] = ctype
        hdr[f"CRVAL{i}"] = crval
        hdr[f"CDELT{i}"] = cdelt
        hdr[f"CRPIX{i}"] = crpix
    for k, v in keep.items():
        hdr[k] = v
    hdr["DATE-OBS"] = date_obs
    hdr["BUNIT"] = "UNCALIB"
    hdr["ORIGIN"] = "difmapy"
    return ghdu


def _sn_table(sol, ant_ref, nif, npol_sn, ref_jd, flag_uncalibrated,
              source_id, extver):
    """The SN binary table: one row per (solution time, antenna), with
    per-IF arrays, and the same gains for both polarizations."""
    ntime, _, nant = sol.shape
    # AIPS applies conj(g_p) g_q, difmapy c_p conj(c_q): g = conj(c).
    corr = np.conj(sol.correction())               # [nt, nif, nant]
    flag = sol.flags(flag_uncalibrated)
    # Solution times as days from the target's reference date.
    t_days = (sol.ref_mjd + 2400000.5 - ref_jd) + sol.times / 86400.0
    iv_days = sol.interval / 86400.0

    nrow = ntime * nant
    time = np.repeat(t_days, nant)
    interval = np.repeat(iv_days, nant).astype(np.float32)
    sub = np.tile([s for (s, _) in ant_ref], ntime).astype(np.int32)
    ant = np.tile([a for (_, a) in ant_ref], ntime).astype(np.int32)

    # [nt, nif, nant] -> rows of [nif], row order (time, antenna).
    c = np.transpose(corr, (0, 2, 1)).reshape(nrow, nif)
    f = np.transpose(flag, (0, 2, 1)).reshape(nrow, nif)
    real = np.where(f, np.nan, c.real).astype(np.float32)
    imag = np.where(f, np.nan, c.imag).astype(np.float32)
    weight = np.where(f, 0.0, 1.0).astype(np.float32)
    zeros1 = np.zeros(nrow, dtype=np.float32)
    zerosn = np.zeros((nrow, nif), dtype=np.float32)
    refant = np.zeros((nrow, nif), dtype=np.int32)

    cols = [
        fits.Column(name="TIME", format="1D", unit="DAYS", array=time),
        fits.Column(name="TIME INTERVAL", format="1E", unit="DAYS",
                    array=interval),
        fits.Column(name="SOURCE ID", format="1J",
                    array=np.full(nrow, source_id, np.int32)),
        fits.Column(name="ANTENNA NO.", format="1J", array=ant),
        fits.Column(name="SUBARRAY", format="1J", array=sub),
        fits.Column(name="FREQ ID", format="1J", array=np.ones(nrow, np.int32)),
        fits.Column(name="I.FAR.ROT", format="1E", unit="RAD/M**2",
                    array=zeros1),
        fits.Column(name="NODE NO.", format="1J", array=np.zeros(nrow, np.int32)),
    ]
    for p in range(1, npol_sn + 1):
        cols += [
            fits.Column(name=f"MBDELAY{p}", format="1E", unit="SECONDS",
                        array=zeros1),
            fits.Column(name=f"DISP {p}", format="1E", unit="SEC/M**2",
                        array=zeros1),
            fits.Column(name=f"DDISP {p}", format="1E", unit="S/S/M**2",
                        array=zeros1),
            fits.Column(name=f"REAL{p}", format=f"{nif}E", array=real),
            fits.Column(name=f"IMAG{p}", format=f"{nif}E", array=imag),
            fits.Column(name=f"DELAY {p}", format=f"{nif}E", unit="SECONDS",
                        array=zerosn),
            fits.Column(name=f"RATE {p}", format=f"{nif}E", unit="SEC/SEC",
                        array=zerosn),
            fits.Column(name=f"WEIGHT {p}", format=f"{nif}E", array=weight),
            fits.Column(name=f"REFANT {p}", format=f"{nif}J", array=refant),
        ]
    sn = fits.BinTableHDU.from_columns(cols)
    sn.name = "AIPS SN"
    h = sn.header
    h["EXTVER"] = int(extver)
    h["NO_ANT"] = int(max(a for (_, a) in ant_ref))
    h["NO_POL"] = int(npol_sn)
    h["NO_IF"] = int(nif)
    h["NO_NODES"] = 0
    h["MGMOD"] = 1.0
    h["APPLIED"] = False
    h["REVISION"] = SN_REVISION
    # Rows are in time order (AIPS 'TB' sort code 1).
    h["ISORTORD"] = 1
    return sn


def _tasav_frame(core, uvfits, nif, what):
    """The part of a TASAV file every table shares: the dummy
    visibility and the FQ and AN tables of the UV data it is for (the
    file the observation was loaded from, `uvfits`, or - for data with
    no UVFITS file behind them - tables synthesised from the
    observation).

    Returns ``(hdus, ant_ref, ref_jd, target)``: the HDUs so far, the
    (subarray, AIPS station number) of each difmapy antenna entry, the
    JD that AIPS times are counted from, and the target file used.
    """
    origin = getattr(core, "_cal_origin", None) or {}
    nant = len(core.antenna_names)
    target = uvfits
    if target is None and origin.get("format") == "uvfits" and \
            os.path.isfile(origin.get("path", "")):
        target = origin["path"]

    if target is not None:
        an_tables, fq, ref_jd, axes, keep = _target_tables(target)
        ant_ref = _match_antennas(core, an_tables)
        no_if = fq.header.get("NO_IF") if fq is not None else None
        if no_if is None and fq is not None:
            no_if = np.atleast_1d(fq.data["IF FREQ"][0]).size
        if no_if is not None and int(no_if) != nif:
            raise ValueError(
                f"{target} has {no_if} IFs but the observation has {nif}; "
                f"{what} must refer to the IFs of the data it is applied to"
            )
    else:
        numbers = origin.get("ant_numbers")
        if not numbers or len(numbers) != nant:
            # Number each subarray's antennas from 1, as wobs() does.
            numbers, seen = [], {}
            for s in core.antenna_subarrays:
                seen[s] = seen.get(s, 0) + 1
                numbers.append(seen[s])
        an_tables, fq, ref_jd = _synthesised_tables(core, numbers)
        ant_ref = [(int(s) + 1, int(n))
                   for s, n in zip(core.antenna_subarrays, numbers)]
        axes = _synthesised_axes(core)
        keep = {"OBJECT": core.source_name, "EQUINOX": 2000.0}

    date_obs = Time(ref_jd, format="jd", scale="utc").strftime("%Y-%m-%d")
    hdus = [_dummy_groups(axes, ref_jd, keep, date_obs)]
    if fq is not None:
        hdus.append(fq)
    hdus.extend(an_tables)
    return hdus, ant_ref, ref_jd, target


def save_fgtable(obs, path, entries, uvfits=None, overwrite=True,
                 reason="difmapy", fgver=1):
    """Write flags as an AIPS FG table in a TASAV FITS file.

    `entries` are the flag entries of `difmapy.io.flags.flag_entries`.
    The file is laid out like the SN one (`save_sntable`), and used the
    same way: FITLD it, TACOP ``INEXT 'FG'`` onto the UV data, and the
    flags apply wherever ``FLAGVER`` selects that table. The FG columns
    are those AIPS 31DEC24's UVFLG writes; antennas are AIPS station
    numbers and times days from the reference date of the target UV
    file, exactly as for the SN table.

    Returns a summary dict (`path`, `nrows`, `target`).
    """
    core = obs._core
    hdus, ant_ref, ref_jd, target = _tasav_frame(
        core, uvfits, entries["nif"], "an FG table")
    rows = entries["rows"]
    n = len(rows)
    day0 = core.ref_mjd + 2400000.5 - ref_jd
    # Which PFLAGS bit each recorded polarization is: the position on
    # the FITS STOKES axis (RR, LL, RL, LR or XX, YY, XY, YX).
    bit = {c: (abs(c) - 1) % 4 for c in core.pols if c < 0}
    ants = np.zeros((n, 2), np.int32)
    sub = np.zeros(n, np.int32)
    trange = np.zeros((n, 2), np.float32)
    ifs = np.zeros((n, 2), np.int32)
    chans = np.zeros((n, 2), np.int32)
    pflags = np.zeros((n, 4), bool)
    for k, e in enumerate(rows):
        s1, n1 = ant_ref[e["ant1"]]
        n2 = 0 if e["ant2"] is None else ant_ref[e["ant2"]][1]
        ants[k] = (n1, 0) if n2 == 0 else sorted((n1, n2))
        sub[k] = s1
        # float32 days resolve ~0.01 s; round outwards so that an edge
        # never excludes the sample it was drawn around.
        t0 = np.float32(day0 + e["t0"] / 86400.0)
        t1 = np.float32(day0 + e["t1"] / 86400.0)
        trange[k] = (np.nextafter(t0, np.float32(-np.inf)),
                     np.nextafter(t1, np.float32(np.inf)))
        ifs[k] = (e["if0"] + 1, e["if1"] + 1)
        chans[k] = (1, 0) if e["chans"] is None else (
            e["chans"][0] + 1, e["chans"][1] + 1)
        if e["pols"] is None or not bit:
            pflags[k] = True
        else:
            for code, on in zip(core.pols, e["pols"]):
                if on and code in bit:
                    pflags[k, bit[code]] = True
    fg = fits.BinTableHDU.from_columns([
        fits.Column(name="SOURCE", format="1J", array=np.zeros(n, np.int32)),
        fits.Column(name="SUBARRAY", format="1J", array=sub),
        fits.Column(name="FREQ ID", format="1J",
                    array=np.full(n, -1, np.int32)),
        fits.Column(name="ANTS", format="2J", array=ants),
        fits.Column(name="TIME RANGE", format="2E", unit="DAYS",
                    array=trange),
        fits.Column(name="IFS", format="2J", array=ifs),
        fits.Column(name="CHANS", format="2J", array=chans),
        fits.Column(name="PFLAGS", format="4X", array=pflags),
        fits.Column(name="REASON", format="24A",
                    array=np.array([str(reason)[:24]] * n)),
    ])
    fg.name = "AIPS FG"
    fg.header["EXTVER"] = int(fgver)
    hdus.append(fg)
    if os.path.exists(path) and not overwrite:
        raise ValueError(f"{path} exists")
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return {"path": path, "nrows": n, "target": target}


def save_sntable(obs, path, uvfits=None, flag_uncalibrated=False,
                 overwrite=True, since=None, solutions=None, source_id=0,
                 snver=1):
    """Write antenna gains as an AIPS SN table in a TASAV FITS file.

    Parameters
    ----------
    obs : difmapy.Observation
        The observation whose gains are exported (the accumulated gain
        table, or `solutions`).
    path : str
        Output FITS file (conventionally ``*.TASAV.FITS``).
    uvfits : str | None
        The UVFITS file the table is for. Its AN tables give the AIPS
        antenna numbers (matched by name), its reference date the time
        origin, and its data axes the dummy visibility's. Defaults to
        the file the data were loaded from; for data loaded from a
        Measurement Set, antennas are numbered in ANTENNA-table order
        from 1, as CASA's exportuvfits does.
    flag_uncalibrated : bool
        Write solutions that were never solved for as flagged, which
        makes AIPS flag that data, instead of as unity.
    since : gain snapshot | None
        Export only the calibration accumulated since
        `Observation.gain_snapshot()` was taken.
    solutions : difmapy.io.solutions.GainSolutions | None
        Write these instead of the observation's gain table.
    source_id : int
        SOURCE ID of every row. 0 (the default) does not tie the table
        to a source, so it is used for every source when copied onto a
        multi-source file; give the source's SU-table number to restrict
        it.
    snver : int
        SN table version (EXTVER) in the file.
    """
    from difmapy.io.solutions import from_gain_table

    core = obs._core
    sol = solutions if solutions is not None else from_gain_table(obs, since)
    ntime, nif, nant = sol.shape
    hdus, ant_ref, ref_jd, target = _tasav_frame(core, uvfits, nif,
                                                 "an SN table")

    # Both parallel hands get the same gain, as in the CASA table; a
    # single-polarization dataset gets a single-polarization table.
    parallel = [p for p in core.pols if p in (-1, -2, -5, -6)]
    npol_sn = 2 if len(parallel) >= 2 else 1

    sn = _sn_table(sol, ant_ref, nif, npol_sn, ref_jd, flag_uncalibrated,
                   int(source_id), snver)
    hdus.append(sn)

    if os.path.exists(path) and not overwrite:
        raise ValueError(f"{path} exists")
    fits.HDUList(hdus).writeto(path, overwrite=True)
    nflag = int(sol.flags(flag_uncalibrated).sum())
    return {"path": path, "nrows": int(sn.header["NAXIS2"]),
            "nflagged": nflag, "warnings": list(sol.warnings),
            "target": target}
