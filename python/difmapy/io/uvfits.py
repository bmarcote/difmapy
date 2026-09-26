"""Random-groups UVFITS reader.

Loads a single-source UVFITS file (as written by AIPS FITTP, difmap,
CASA exportuvfits, ...) into the arrays expected by the Rust core:
all IFs/channels/polarizations in RAM.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits

from difmapy._core import CoreObservation

__all__ = ["load_uvfits", "save_uvfits"]


def _group_parameter(hdu, names):
    """Sum all group parameters whose name matches one of `names`
    (UVFITS allows split parameters, e.g. two DATE entries)."""
    parnames = [p.upper().rstrip("-") for p in hdu.data.parnames]
    total = None
    for i, pname in enumerate(parnames):
        base = pname.split("--")[0].split("---")[0]
        if base in names or pname in names:
            par = hdu.data.par(i).astype(np.float64)
            total = par if total is None else total + par
    if total is None:
        raise ValueError(f"UVFITS file lacks required group parameter {names[0]}")
    return total


def _axis_index(header, ctype):
    """1-based FITS axis number whose CTYPE starts with `ctype`."""
    for i in range(1, header["NAXIS"] + 1):
        if header.get(f"CTYPE{i}", "").upper().startswith(ctype):
            return i
    return None


def load_uvfits(path, wtscale=1.0):
    """Read a random-groups UVFITS file and return a CoreObservation.

    Parameters
    ----------
    path : str
        Path of the UVFITS file.
    wtscale : float
        Optional scale factor applied to the raw visibility weights.
    """
    with fits.open(path, memmap=True) as hdul:
        ghdu = hdul[0]
        if not isinstance(ghdu, fits.GroupsHDU) or ghdu.data is None:
            raise ValueError(f"{path} is not a random-groups UVFITS file")
        hdr = ghdu.header

        parnames = [p.upper() for p in ghdu.data.parnames]
        basenames = [p.split("--")[0].split("---")[0] for p in parnames]
        if "SOURCE" in basenames:
            src = _group_parameter(ghdu, ("SOURCE",))
            if len(np.unique(src.astype(int))) > 1:
                raise ValueError(
                    "multi-source UVFITS files are not supported; "
                    "split the file into single sources first"
                )

        # ---- axes ----
        iax_complex = _axis_index(hdr, "COMPLEX")
        iax_stokes = _axis_index(hdr, "STOKES")
        iax_freq = _axis_index(hdr, "FREQ")
        iax_if = _axis_index(hdr, "IF")
        if iax_complex is None or iax_stokes is None or iax_freq is None:
            raise ValueError("UVFITS file lacks COMPLEX/STOKES/FREQ axes")
        nchan = hdr[f"NAXIS{iax_freq}"]
        npol = hdr[f"NAXIS{iax_stokes}"]
        nif = hdr[f"NAXIS{iax_if}"] if iax_if is not None else 1

        # Stokes codes (AIPS convention).
        crval_s = int(hdr[f"CRVAL{iax_stokes}"])
        cdelt_s = int(hdr.get(f"CDELT{iax_stokes}", -1 if crval_s < 0 else 1))
        crpix_s = hdr.get(f"CRPIX{iax_stokes}", 1.0)
        pols = [int(crval_s + (i + 1 - crpix_s) * cdelt_s) for i in range(npol)]

        # Reference frequency / channel width.
        ref_freq = float(hdr[f"CRVAL{iax_freq}"])
        cdelt_f = float(hdr.get(f"CDELT{iax_freq}", 0.0))
        crpix_f = float(hdr.get(f"CRPIX{iax_freq}", 1.0))

        # ---- FQ table: per-IF frequency offsets ----
        if_offset = np.zeros(nif)
        if_chwidth = np.full(nif, cdelt_f)
        for hdu in hdul[1:]:
            if hdu.name in ("AIPS FQ", "FQ"):
                if len(hdu.data) > 1:
                    raise ValueError("multiple FQ frequency setups are not supported")
                row = hdu.data[0]
                if_offset = np.atleast_1d(np.asarray(row["IF FREQ"], dtype=np.float64))
                chw = np.atleast_1d(np.asarray(row["CH WIDTH"], dtype=np.float64))
                if chw.size == nif and np.all(chw != 0):
                    if_chwidth = chw
                break
        if if_offset.size != nif:
            raise ValueError(f"FQ table has {if_offset.size} IFs, header has {nif}")
        # Frequency of channel 0 of each IF.
        if_freq = ref_freq + if_offset + (1.0 - crpix_f) * if_chwidth
        if_df = if_chwidth
        if_nchan = [nchan] * nif

        # ---- AN tables: antennas per subarray ----
        an_tables = sorted(
            (h for h in hdul[1:] if h.name in ("AIPS AN", "AN")),
            key=lambda h: h.header.get("EXTVER", 1),
        )
        if not an_tables:
            raise ValueError("UVFITS file lacks an AIPS AN antenna table")
        ant_names, ant_xyz, ant_sub, ant_nosta = [], [], [], []
        # Map (subarray, antenna_number) -> global index.
        index_of = {}
        for isub, an in enumerate(an_tables):
            nosta = np.asarray(an.data["NOSTA"], dtype=int)
            names = [str(n).strip() for n in an.data["ANNAME"]]
            xyz = np.asarray(an.data["STABXYZ"], dtype=np.float64)
            for k in range(len(nosta)):
                index_of[(isub, int(nosta[k]))] = len(ant_names)
                ant_names.append(names[k] or f"ANT{int(nosta[k])}")
                ant_xyz.append(xyz[k])
                ant_sub.append(isub)
                ant_nosta.append(int(nosta[k]))
        ant_xyz = np.asarray(ant_xyz, dtype=np.float64)

        # ---- group parameters ----
        uu = _group_parameter(ghdu, ("UU",))
        vv = _group_parameter(ghdu, ("VV",))
        ww = _group_parameter(ghdu, ("WW",))
        baseline = _group_parameter(ghdu, ("BASELINE",))
        jd = _group_parameter(ghdu, ("DATE",))
        try:
            inttim = _group_parameter(ghdu, ("INTTIM",)).astype(np.float32)
        except ValueError:
            inttim = np.zeros(len(uu), dtype=np.float32)

        # Decode baseline numbers: 256*a1 + a2 + (subarray-1)/100.
        ibl = np.floor(baseline + 0.005).astype(int)
        sub0 = np.clip(np.rint((baseline - ibl) * 100).astype(int), 0, None)
        a1n = ibl // 256
        a2n = ibl % 256

        # Drop autocorrelations (difmap uses cross-correlations only).
        cross = a1n != a2n
        if not cross.all():
            uu, vv, ww = uu[cross], vv[cross], ww[cross]
            jd, inttim = jd[cross], inttim[cross]
            a1n, a2n, sub0 = a1n[cross], a2n[cross], sub0[cross]

        mjd = jd - 2400000.5
        ref_mjd = np.floor(mjd.min())
        tsec = (mjd - ref_mjd) * 86400.0

        # ---- visibilities ----
        # Group data shape: (ngroups, ..., IF, FREQ, STOKES, COMPLEX);
        # reshape to [ngroups, nif*nchan, npol, >=2].
        gdata = ghdu.data.data
        if not cross.all():
            gdata = gdata[cross]
        ngroups = gdata.shape[0]
        ncplx = gdata.shape[-1]
        gdata = gdata.reshape(ngroups, nif, nchan, npol, ncplx)
        re = gdata[..., 0].astype(np.float32)
        im = gdata[..., 1].astype(np.float32)
        wt = (
            gdata[..., 2].astype(np.float32)
            if ncplx >= 3
            else np.ones_like(re, dtype=np.float32)
        )
        if wtscale != 1.0:
            wt = wt * np.float32(wtscale)

        # ---- sort rows by (time, subarray, baseline) ----
        order = np.lexsort((a2n, a1n, sub0, tsec))
        tsec = tsec[order]
        uu, vv, ww = uu[order], vv[order], ww[order]
        a1n, a2n, sub0 = a1n[order], a2n[order], sub0[order]
        inttim = inttim[order]
        re, im, wt = re[order], im[order], wt[order]

        # Global antenna indices.
        try:
            ant1 = np.fromiter(
                (index_of[(s, a)] for s, a in zip(sub0, a1n)),
                dtype=np.uint32,
                count=ngroups,
            )
            ant2 = np.fromiter(
                (index_of[(s, a)] for s, a in zip(sub0, a2n)),
                dtype=np.uint32,
                count=ngroups,
            )
        except KeyError as exc:
            raise ValueError(f"baseline references unknown antenna {exc}") from exc

        vis = (re + 1j * im).astype(np.complex64).reshape(ngroups, nif * nchan, npol)
        wt = wt.reshape(ngroups, nif * nchan, npol)
        # UVFITS encodes flags as negative weights; convert to an
        # explicit FLAG column (MS convention) and keep weights >= 0.
        flag = wt <= 0.0
        wt = np.abs(wt)
        uvw = np.column_stack([uu, vv, ww]).astype(np.float64)

        # ---- source ----
        source = str(hdr.get("OBJECT", "unknown")).strip()
        iax_ra = _axis_index(hdr, "RA")
        iax_dec = _axis_index(hdr, "DEC")
        ra = np.deg2rad(float(hdr[f"CRVAL{iax_ra}"])) if iax_ra else 0.0
        dec = np.deg2rad(float(hdr[f"CRVAL{iax_dec}"])) if iax_dec else 0.0
        epoch = float(hdr.get("EQUINOX", hdr.get("EPOCH", 2000.0)))

        core = CoreObservation(
            source,
            ra,
            dec,
            epoch,
            ant_names,
            ant_xyz,
            [int(s) for s in ant_sub],
            [float(f) for f in if_freq],
            [float(d) for d in if_df],
            if_nchan,
            pols,
            np.ascontiguousarray(tsec),
            np.ascontiguousarray(inttim),
            np.ascontiguousarray(ant1),
            np.ascontiguousarray(ant2),
            np.ascontiguousarray(uvw),
            np.ascontiguousarray(vis),
            np.ascontiguousarray(wt),
            float(ref_mjd),
            flag=np.ascontiguousarray(flag),
        )
    # What a calibration table needs to refer back to this file: the
    # AIPS station numbers (an SN table is keyed by them, not by name).
    core._cal_origin = {
        "format": "uvfits",
        "path": path,
        "ant_numbers": ant_nosta,
    }
    return core


def save_uvfits(core, path, overwrite=True, freeze_shift=False):
    """Write a CoreObservation to a random-groups UVFITS file
    (difmap wobs). Antenna gains and baseline corrections are applied
    to the data; flags are preserved as negative weights.

    As in difmap, accumulated phase-center shifts are *not* written
    unless `freeze_shift` is true - `Observation.save()` instead records
    the shift so that `get()` re-applies it.
    """
    vis, wt = core.calibrated_cube(apply_shift=freeze_shift)
    time, ant1, ant2, us, vs, ws = core.rows()
    inttim = core.inttimes()
    ifs = core.ifs
    nif = len(ifs)
    nchan = max(n for (_, _, n) in ifs)
    npol = core.npol
    nrow = core.nrow

    # The FITS STOKES axis must be a regular sequence, but MS
    # correlations may come in another order (CASA uses RR, RL, LR, LL),
    # so sort the polarization axis into a FITS-expressible order.
    pols = list(core.pols)
    if all(p < 0 for p in pols):
        porder = np.argsort(np.abs(pols))  # RR, LL, RL, LR ...
    else:
        porder = np.argsort(pols)  # I, Q, U, V
    pols = [pols[i] for i in porder]
    if len(pols) > 1:
        steps = np.diff(pols)
        if not np.all(steps == steps[0]):
            raise ValueError(f"cannot express polarizations {pols} on a FITS axis")
        cdelt_s = int(steps[0])
    else:
        cdelt_s = -1 if pols[0] < 0 else 1
    if not np.array_equal(porder, np.arange(len(pols))):
        vis = vis[:, :, porder]
        wt = wt[:, :, porder]

    # Data cube [ngroups, 1, 1, nif, nchan, npol, 3]; IFs with fewer
    # channels than nchan are padded with zero-weight visibilities.
    gdata = np.zeros((nrow, 1, 1, nif, nchan, npol, 3), dtype=np.float32)
    for i, (_, _, nch) in enumerate(ifs):
        coff = sum(n for (_, _, n) in ifs[:i])
        block = vis[:, coff : coff + nch, :]
        gdata[:, 0, 0, i, :nch, :, 0] = block.real
        gdata[:, 0, 0, i, :nch, :, 1] = block.imag
        gdata[:, 0, 0, i, :nch, :, 2] = wt[:, coff : coff + nch, :]

    # Antennas: renumber per subarray (1-based NOSTA).
    names = core.antenna_names
    subs = core.antenna_subarrays
    nsub = max(subs) + 1
    # Global index -> (subarray, station number)
    nosta_of = {}
    per_sub = [[] for _ in range(nsub)]
    for g, s in enumerate(subs):
        per_sub[s].append(g)
        nosta_of[g] = (s, len(per_sub[s]))

    bl = np.zeros(nrow)
    for k in range(nrow):
        s1, n1 = nosta_of[int(ant1[k])]
        _, n2 = nosta_of[int(ant2[k])]
        bl[k] = 256 * n1 + n2 + 0.01 * s1

    jd0 = core.ref_mjd + 2400000.5
    pdata = [
        us,
        vs,
        ws,
        bl,
        np.full(nrow, jd0),
        time / 86400.0,
        np.asarray(inttim, dtype=np.float64),
    ]
    parnames = ["UU", "VV", "WW", "BASELINE", "DATE", "DATE", "INTTIM"]
    groups = fits.GroupData(gdata, parnames=parnames, pardata=pdata, bitpix=-32)
    ghdu = fits.GroupsHDU(groups)
    hdr = ghdu.header
    hdr["OBJECT"] = core.source_name
    hdr["EQUINOX"] = 2000.0
    hdr["BUNIT"] = "JY"
    hdr["OBSRA"] = np.rad2deg(core.ra)
    hdr["OBSDEC"] = np.rad2deg(core.dec)
    hdr["ORIGIN"] = "difmapy"
    for i, (ctype, crval, cdelt) in enumerate(
        [
            ("COMPLEX", 1.0, 1.0),
            ("STOKES", float(pols[0]), float(cdelt_s)),
            ("FREQ", ifs[0][0], ifs[0][1]),
            ("IF", 1.0, 1.0),
            ("RA", np.rad2deg(core.ra), 1.0),
            ("DEC", np.rad2deg(core.dec), 1.0),
        ],
        start=2,
    ):
        hdr[f"CTYPE{i}"] = ctype
        hdr[f"CRVAL{i}"] = crval
        hdr[f"CRPIX{i}"] = 1.0
        hdr[f"CDELT{i}"] = cdelt

    hdus = [ghdu]
    ant_xyz = np.asarray(core.antenna_xyz)
    for s in range(nsub):
        idx = per_sub[s]
        an = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="ANNAME", format="8A",
                            array=np.array([names[g] for g in idx])),
                fits.Column(name="STABXYZ", format="3D", unit="METERS",
                            array=ant_xyz[idx]),
                fits.Column(name="NOSTA", format="1J",
                            array=np.arange(1, len(idx) + 1, dtype=np.int32)),
                fits.Column(name="MNTSTA", format="1J",
                            array=np.zeros(len(idx), dtype=np.int32)),
            ]
        )
        an.name = "AIPS AN"
        an.header["EXTVER"] = s + 1
        an.header["ARRAYX"] = 0.0
        an.header["ARRAYY"] = 0.0
        an.header["ARRAYZ"] = 0.0
        an.header["FREQ"] = ifs[0][0]
        hdus.append(an)

    fq = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="FRQSEL", format="1J",
                        array=np.array([1], dtype=np.int32)),
            fits.Column(name="IF FREQ", format=f"{nif}D", unit="HZ",
                        array=np.array([[f - ifs[0][0] for (f, _, _) in ifs]])),
            fits.Column(name="CH WIDTH", format=f"{nif}E", unit="HZ",
                        array=np.array([[df for (_, df, _) in ifs]],
                                       dtype=np.float32)),
            fits.Column(name="TOTAL BANDWIDTH", format=f"{nif}E", unit="HZ",
                        array=np.array([[abs(df) * n for (_, df, n) in ifs]],
                                       dtype=np.float32)),
            fits.Column(name="SIDEBAND", format=f"{nif}J",
                        array=np.array([[1 if df >= 0 else -1
                                         for (_, df, _) in ifs]],
                                       dtype=np.int32)),
        ]
    )
    fq.name = "AIPS FQ"
    fq.header["NO_IF"] = nif
    hdus.append(fq)

    fits.HDUList(hdus).writeto(path, overwrite=overwrite)
