"""Shared fixtures: a synthetic random-groups UVFITS file containing a
point source of known flux and position, with known gain corruptions
optionally applied."""

import numpy as np
import pytest
from astropy.io import fits

C = 299792458.0
MAS = np.pi / (180.0 * 3600.0 * 1000.0)

NANT = 5
NTIME = 60
NIF = 2
NCHAN = 4
REF_FREQ = 4.9e9
IF_OFFSET = 1.0e8
CH_WIDTH = 1.0e6
FLUX = 2.5
X0_MAS = 4.0  # east offset (mas)
Y0_MAS = -2.5  # north offset (mas)
RA_DEG = 30.0
DEC_DEG = 40.0
REF_JD = 2460000.5  # MJD 59999.0... start of an arbitrary day


def make_uvfits(path, gerr=None, flux=FLUX, x0_mas=X0_MAS, y0_mas=Y0_MAS):
    """Write a synthetic single-source random-groups UVFITS file."""
    x0, y0 = x0_mas * MAS, y0_mas * MAS
    if_freqs = REF_FREQ + IF_OFFSET * np.arange(NIF)

    rows = []
    for it in range(NTIME):
        tday = (it * 60.0) / 86400.0
        for a in range(NANT):
            for b in range(a + 1, NANT):
                blen = 1.0e6 * ((a + b) + 1.0)
                theta = 0.7 * (it / NTIME) * np.pi + (a * NANT + b)
                u_m, v_m = blen * np.cos(theta), blen * np.sin(theta)
                u_s, v_s = u_m / C, v_m / C
                gamp, gphs = 1.0, 0.0
                if gerr is not None:
                    gamp = gerr[a][0] * gerr[b][0]
                    gphs = gerr[a][1] - gerr[b][1]
                # vis[IF, chan, pol, 3]
                vis = np.zeros((NIF, NCHAN, 2, 3), dtype=np.float32)
                for i in range(NIF):
                    for ch in range(NCHAN):
                        f = if_freqs[i] + ch * CH_WIDTH
                        phs = 2 * np.pi * (u_s * f * x0 + v_s * f * y0) + gphs
                        vis[i, ch, :, 0] = gamp * flux * np.cos(phs)
                        vis[i, ch, :, 1] = gamp * flux * np.sin(phs)
                        vis[i, ch, :, 2] = 1.0
                rows.append((u_s, v_s, 0.0, 256 * (a + 1) + (b + 1), tday, vis))

    ngroups = len(rows)
    gdata = np.zeros((ngroups, 1, 1, NIF, NCHAN, 2, 3), dtype=np.float32)
    uu = np.zeros(ngroups)
    vv = np.zeros(ngroups)
    ww = np.zeros(ngroups)
    bl = np.zeros(ngroups)
    dd = np.zeros(ngroups)
    for k, (u, v, w, b, t, vis) in enumerate(rows):
        uu[k], vv[k], ww[k], bl[k], dd[k] = u, v, w, b, t
        gdata[k, 0, 0] = vis

    pdata = [uu, vv, ww, bl, np.full(ngroups, REF_JD), dd]
    parnames = ["UU", "VV", "WW", "BASELINE", "DATE", "DATE"]
    groups = fits.GroupData(
        gdata, parnames=parnames, pardata=pdata, bitpix=-32
    )
    ghdu = fits.GroupsHDU(groups)
    hdr = ghdu.header
    hdr["OBJECT"] = "SYNTH"
    hdr["TELESCOP"] = "VLBA"
    hdr["EQUINOX"] = 2000.0
    hdr["BUNIT"] = "JY"
    # Axes: 1=COMPLEX(defined by data), 2..: set CTYPEs.
    hdr["CTYPE2"] = "COMPLEX"
    hdr["CRVAL2"] = 1.0
    hdr["CRPIX2"] = 1.0
    hdr["CDELT2"] = 1.0
    hdr["CTYPE3"] = "STOKES"
    hdr["CRVAL3"] = -1.0  # RR
    hdr["CRPIX3"] = 1.0
    hdr["CDELT3"] = -1.0  # RR, LL
    hdr["CTYPE4"] = "FREQ"
    hdr["CRVAL4"] = REF_FREQ
    hdr["CRPIX4"] = 1.0
    hdr["CDELT4"] = CH_WIDTH
    hdr["CTYPE5"] = "IF"
    hdr["CRVAL5"] = 1.0
    hdr["CRPIX5"] = 1.0
    hdr["CDELT5"] = 1.0
    hdr["CTYPE6"] = "RA"
    hdr["CRVAL6"] = RA_DEG
    hdr["CTYPE7"] = "DEC"
    hdr["CRVAL7"] = DEC_DEG

    # AN table.
    anames = np.array([f"AN{i}" for i in range(NANT)])
    xyz = np.array(
        [[6.0e6 + 1.0e5 * i, 1.0e5 * i, 1.0e5] for i in range(NANT)], dtype=np.float64
    )
    nosta = np.arange(1, NANT + 1, dtype=np.int32)
    an = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ANNAME", format="8A", array=anames),
            fits.Column(name="STABXYZ", format="3D", unit="METERS", array=xyz),
            fits.Column(name="NOSTA", format="1J", array=nosta),
            fits.Column(name="MNTSTA", format="1J", array=np.zeros(NANT, dtype=np.int32)),
        ]
    )
    an.name = "AIPS AN"
    an.header["EXTVER"] = 1
    an.header["ARRAYX"] = 0.0
    an.header["ARRAYY"] = 0.0
    an.header["ARRAYZ"] = 0.0
    an.header["FREQ"] = REF_FREQ

    # FQ table.
    fq = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="FRQSEL", format="1J", array=np.array([1], dtype=np.int32)),
            fits.Column(
                name="IF FREQ",
                format=f"{NIF}D",
                unit="HZ",
                array=(IF_OFFSET * np.arange(NIF))[np.newaxis],
            ),
            fits.Column(
                name="CH WIDTH",
                format=f"{NIF}E",
                unit="HZ",
                array=np.full((1, NIF), CH_WIDTH, dtype=np.float32),
            ),
            fits.Column(
                name="TOTAL BANDWIDTH",
                format=f"{NIF}E",
                unit="HZ",
                array=np.full((1, NIF), CH_WIDTH * NCHAN, dtype=np.float32),
            ),
            fits.Column(
                name="SIDEBAND",
                format=f"{NIF}J",
                array=np.ones((1, NIF), dtype=np.int32),
            ),
        ]
    )
    fq.name = "AIPS FQ"
    fq.header["NO_IF"] = NIF

    fits.HDUList([ghdu, an, fq]).writeto(path, overwrite=True)
    return path


@pytest.fixture(scope="session")
def uvfits_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synth.uvf"
    return str(make_uvfits(path))


@pytest.fixture(scope="session")
def corrupted_uvfits_file(tmp_path_factory):
    gerr = [(1.30, 0.40), (0.75, -0.30), (1.10, 0.15), (0.90, -0.50), (1.05, 0.25)]
    path = tmp_path_factory.mktemp("data") / "synth_gains.uvf"
    return str(make_uvfits(path, gerr=gerr)), gerr
