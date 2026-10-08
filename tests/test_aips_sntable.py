"""AIPS SN-table export (`savecaltable(outformat="AIPS")`).

As for the CASA tables, the decisive test is AIPS itself: the TASAV
file is loaded with FITLD, its SN table copied onto the original data
with TACOP and applied by SPLIT, and the result must equal difmapy's own
corrected visibilities. That pins down AIPS's gain convention (it
multiplies by conj(g_p) g_q, the opposite phase sign to difmapy), the
antenna numbering and the time origin. It runs where AIPS is installed
(``/opt/aips/LOGIN.SH``, or ``$DIFMAPY_AIPS_LOGIN``), under AIPS user
number ``$DIFMAPY_AIPS_USER`` (default 7301), whose catalog it empties.
"""

import os
import shutil
import subprocess

import numpy as np
import pytest
from astropy.io import fits

import difmapy
from conftest import NANT, NIF, REF_JD, make_uvfits

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
MS = os.path.join(HERE, "rsm07_3C345.ms")
AIPS_LOGIN = os.environ.get("DIFMAPY_AIPS_LOGIN", "/opt/aips/LOGIN.SH")
AIPS_USER = int(os.environ.get("DIFMAPY_AIPS_USER", "7301"))

#: The SN columns AIPS 31DEC24 writes (CALIB -> TASAV -> FITTP), in order.
AIPS_SN_COLUMNS = [
    "TIME", "TIME INTERVAL", "SOURCE ID", "ANTENNA NO.", "SUBARRAY",
    "FREQ ID", "I.FAR.ROT", "NODE NO.",
    "MBDELAY1", "DISP 1", "DDISP 1", "REAL1", "IMAG1", "DELAY 1", "RATE 1",
    "WEIGHT 1", "REFANT 1",
    "MBDELAY2", "DISP 2", "DDISP 2", "REAL2", "IMAG2", "DELAY 2", "RATE 2",
    "WEIGHT 2", "REFANT 2",
]


@pytest.fixture()
def calibrated(corrupted_uvfits_file):
    path, _ = corrupted_uvfits_file
    o = difmapy.load(path)
    o.mapsize(256, 0.25)
    o.addcmp(2.5, 4.0, -2.5)
    o.selfcal(amp=True, phase=True, quiet=True)
    return o


def _sn(path):
    with fits.open(path) as h:
        return h["AIPS SN"].header.copy(), h["AIPS SN"].data.copy(), \
            [x.name for x in h]


def test_default_format_follows_the_data(calibrated, tmp_path):
    assert calibrated.caltable_formats() == ("aips",)
    out = str(tmp_path / "cal.TASAV.FITS")
    info = calibrated.savecaltable(out, quiet=True)
    assert info["format"] == "aips" and os.path.isfile(out)
    assert info["nrows"] == calibrated._core.ntimes * NANT


def test_outformat_names(calibrated):
    f = calibrated.caltable_formats
    assert f("CASA") == ("casa",) and f("aips") == ("aips",)
    assert f("Both") == ("casa", "aips")
    # A reference file of the other kind says which table is wanted.
    assert f(ms="other.ms") == ("casa",) and f(uvfits="x.uvf") == ("aips",)
    with pytest.raises(ValueError, match="outformat"):
        f("miriad")
    p = calibrated.caltable_paths
    assert p("cal.G", ("casa",)) == {"casa": "cal.G"}
    assert p("cal", ("casa", "aips")) == {"casa": "cal",
                                         "aips": "cal.TASAV.FITS"}
    assert p("cal.TASAV.FITS", ("casa", "aips")) == {"casa": "cal",
                                                    "aips": "cal.TASAV.FITS"}


def test_tasav_layout_matches_aips(calibrated, tmp_path):
    out = str(tmp_path / "cal.TASAV.FITS")
    calibrated.savecaltable(out, outformat="AIPS", quiet=True)
    hdr, sn, names = _sn(out)
    assert names == ["PRIMARY", "AIPS FQ", "AIPS AN", "AIPS SN"]
    assert list(sn.columns.names) == AIPS_SN_COLUMNS
    assert (hdr["NO_IF"], hdr["NO_POL"], hdr["NO_ANT"]) == (NIF, 2, NANT)
    assert hdr["REVISION"] == 11 and hdr["APPLIED"] is False
    with fits.open(out) as h:
        g = h[0]
        assert g.header["GCOUNT"] == 1
        # The dummy visibility has the data axes of the file it is for.
        with fits.open(calibrated._core._cal_origin["path"]) as src:
            assert g.data.data.shape[1:] == src[0].data.data.shape[1:]


def test_sn_values_are_conjugate_corrections(calibrated, tmp_path):
    """SN = conj(difmapy correction), same amplitude, at the right times
    and antenna numbers."""
    out = str(tmp_path / "cal.TASAV.FITS")
    calibrated.savecaltable(out, outformat="aips", quiet=True)
    _, sn, _ = _sn(out)
    core = calibrated._core
    amp, phs, _ = (np.asarray(x) for x in core.gains())
    nt = core.ntimes
    corr = (amp * np.exp(1j * phs)).reshape(nt, NIF, NANT)
    expect = np.conj(np.transpose(corr, (0, 2, 1)).reshape(nt * NANT, NIF))
    got = sn["REAL1"] + 1j * sn["IMAG1"]
    np.testing.assert_allclose(got, expect, rtol=1e-6)
    np.testing.assert_array_equal(sn["REAL2"], sn["REAL1"])
    np.testing.assert_array_equal(sn["ANTENNA NO."],
                                  np.tile(np.arange(1, NANT + 1), nt))
    # Days from the file's reference date (REF_JD is its 0h).
    times = core.ref_mjd + 2400000.5 - REF_JD + np.asarray(core.times()) / 86400
    np.testing.assert_allclose(sn["TIME"], np.repeat(times, NANT), atol=1e-9)
    assert (sn["WEIGHT 1"] > 0).all()


def test_antennas_are_matched_by_name(calibrated, tmp_path):
    """Given another UV file, the SN table uses *its* station numbers."""
    other = str(tmp_path / "other.uvf")
    make_uvfits(other)
    with fits.open(other, mode="update") as h:
        an = h["AIPS AN"]
        an.data["NOSTA"] = an.data["NOSTA"][::-1] + 10
    out = str(tmp_path / "cal.TASAV.FITS")
    calibrated.savecaltable(out, outformat="aips", uvfits=other, quiet=True)
    hdr, sn, _ = _sn(out)
    nosta = np.arange(1, NANT + 1)[::-1] + 10
    np.testing.assert_array_equal(sn["ANTENNA NO."][:NANT], nosta)
    assert hdr["NO_ANT"] == nosta.max()


def test_if_mismatch_is_refused(calibrated, tmp_path):
    avg = str(tmp_path / "avg.uvf")
    calibrated.chanaver("all").wobs(avg)  # same IFs: accepted
    calibrated.savecaltable(str(tmp_path / "a.FITS"), outformat="aips",
                            uvfits=avg, quiet=True)
    one_if = str(tmp_path / "oneif.uvf")
    with fits.open(avg) as h:
        fq = h["AIPS FQ"]
        fq.header["NO_IF"] = 1
        h.writeto(one_if)
    with pytest.raises(ValueError, match="IFs"):
        calibrated.savecaltable(str(tmp_path / "b.FITS"), outformat="aips",
                                uvfits=one_if, quiet=True)


def test_flag_uncalibrated_blanks_solutions(calibrated, tmp_path):
    calibrated._core.uncalib(True, True, False)
    out = str(tmp_path / "cal.TASAV.FITS")
    calibrated.savecaltable(out, outformat="aips", flag_uncalibrated=True,
                            quiet=True)
    _, sn, _ = _sn(out)
    assert np.isnan(sn["REAL1"]).all() and (sn["WEIGHT 1"] == 0).all()


@pytest.mark.skipif(not os.path.isdir(MS), reason="real 3C345 MS not present")
def test_ms_data_default_to_casa_and_number_like_exportuvfits(tmp_path):
    pytest.importorskip("casatools")
    o = difmapy.load(MS)
    assert o.caltable_formats() == ("casa",)
    o.mapsize(512, 1.0)
    o.startmod(flux=1.0)
    out = str(tmp_path / "cal")
    info = o.savecaltable(out, outformat="both", quiet=True)
    assert os.path.isdir(info["casa"]["path"])
    hdr, sn, names = _sn(info["aips"]["path"])
    assert "AIPS AN" in names
    # ANTENNA-table order from 1, which is what the UVFITS export of the
    # same observation has too.
    assert list(sn["ANTENNA NO."][: len(o.antennas)]) == list(
        range(1, len(o.antennas) + 1))
    # Averaged data keep the provenance.
    info = o.uvaver(120).savecaltable(str(tmp_path / "avg.G"), quiet=True)
    assert info["format"] == "casa"


# ---------------------------------------------------------------------
# AIPS itself
# ---------------------------------------------------------------------


def _run_aips(workdir, commands, timeout=300):
    """Run AIPS non-interactively on `commands` (a list of POPS lines,
    each short enough for AIPS's line length), with the environment
    variable DFMPY pointing at `workdir`. Returns the session log."""
    script = "\n".join([str(AIPS_USER)] + commands + ["kleenex", ""])
    with open(os.path.join(workdir, "cmds.txt"), "w") as fh:
        fh.write(script)
    env = dict(os.environ, DFMPY=str(workdir))
    proc = subprocess.run(
        ["bash", "-c", f". {AIPS_LOGIN} >/dev/null 2>&1; aips notv < cmds.txt"],
        cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout,
    )
    # AIPS exits through a signal after KLEENEX; the log is what counts.
    return proc.stdout + proc.stderr


ZAP_ALL = ["indisk 1", "for i = 1 to 20; getn i; zap; end"]


@pytest.mark.skipif(not os.path.isfile(AIPS_LOGIN), reason="AIPS not installed")
@pytest.mark.skipif(not os.path.isfile(UVF), reason="real 3C345 data not present")
def test_aips_applies_the_table_like_difmapy(tmp_path):
    # AIPS upper-cases the file names it writes, so use upper case.
    shutil.copy(UVF, tmp_path / "IN.UVFITS")
    o = difmapy.load(str(tmp_path / "IN.UVFITS"))
    o.mapsize(1024, 1.0)
    o.startmod(flux=1.0)
    o.clean(200, 0.03, quiet=True)
    o.selfcal(phase=True, quiet=True)
    o.clean(200, 0.03, quiet=True)
    o.selfcal(amp=True, phase=True, solint=30, quiet=True)
    o.savecaltable(str(tmp_path / "CAL.TASAV.FITS"), quiet=True)

    log = _run_aips(tmp_path, ZAP_ALL + [
        "default fitld",
        "datain 'DFMPY:IN.UVFITS'",
        "outname 'DFMPY'; outclass 'UVDATA'; outdisk 1; outseq 1",
        "douvcomp -1",
        "go fitld; wait fitld",
        "datain 'DFMPY:CAL.TASAV.FITS'",
        "outclass 'TASAV'",
        "go fitld; wait fitld",
        "default tacop",
        "indisk 1; getn 2; inext 'SN'; inver 1",
        "outname 'DFMPY'; outclass 'UVDATA'; outseq 1; outdisk 1",
        "go tacop; wait tacop",
        "default split",
        "indisk 1; getn 1; docal 1; gainuse 1; stokes 'HALF'",
        "outclass 'SPLIT'; outdisk 1; douvcomp -1",
        "go split; wait split",
        "default fittp",
        "indisk 1; getn 3; dataout 'DFMPY:OUT.FITS'",
        "go fittp; wait fittp",
    ] + ZAP_ALL)
    assert "Applying SN Table version     1" in log, log[-3000:]
    out = tmp_path / "OUT.FITS"
    assert out.exists(), log[-3000:]

    from difmapy.io.uvfits import load_uvfits

    aips = load_uvfits(str(out))
    mine, wt = (np.asarray(x) for x in o._core.calibrated_cube())
    t, a1, a2, *_ = o._core.rows()
    index = {(int(round(x * 10)), int(p), int(q)): i
             for i, (x, p, q) in enumerate(zip(t, a1, a2))}
    ta, b1, b2, *_ = aips.rows()
    ta = np.asarray(ta) + (aips.ref_mjd - o._core.ref_mjd) * 86400.0
    rows = np.array([index[(int(round(x * 10)), int(p), int(q))]
                     for x, p, q in zip(ta, b1, b2)])
    vis_a, wt_a = (np.asarray(x) for x in aips.calibrated_cube())
    for pol in (-1, -2):
        ia, im = list(aips.pols).index(pol), list(o._core.pols).index(pol)
        good = (wt[rows, :, im] > 0) & (wt_a[:, :, ia] > 0)
        assert good.sum() > 10000
        ref = mine[rows, :, im][good]
        dev = np.abs(vis_a[:, :, ia][good] - ref)
        # float32 round-off; the calibration itself moves |V| by ~30%.
        assert dev.max() < 1e-5 * np.median(np.abs(ref))


@pytest.mark.skipif(not os.path.isfile(AIPS_LOGIN), reason="AIPS not installed")
@pytest.mark.skipif(not os.path.isfile(UVF), reason="real 3C345 data not present")
def test_aips_applies_the_flag_table_like_difmapy(tmp_path):
    """The FG table from `wflags`: copied onto the data in AIPS and
    applied by SPLIT, it must flag exactly the samples difmapy has."""
    shutil.copy(UVF, tmp_path / "IN.UVFITS")
    o = difmapy.load(str(tmp_path / "IN.UVFITS"), stokes=None)
    o.select("I")
    t = np.asarray(o._core.times())
    o.flag(station="EF", tmin=t[40], tmax=t[90])
    o.flag(baseline=("JB", "WB"), tmin=t[200], tmax=t[260], if_index=2)
    o.flag(station="T6", if_index=0)
    info = o.wflags(str(tmp_path / "FLG.FG.TASAV.FITS"), quiet=True)
    assert info["nrows"] == 3   # one entry per flag command above

    log = _run_aips(tmp_path, ZAP_ALL + [
        "default fitld",
        "datain 'DFMPY:IN.UVFITS'",
        "outname 'DFMPY'; outclass 'UVDATA'; outdisk 1; outseq 1",
        "douvcomp -1",
        "go fitld; wait fitld",
        "datain 'DFMPY:FLG.FG.TASAV.FITS'",
        "outclass 'TASAV'",
        "go fitld; wait fitld",
        "default tacop",
        "indisk 1; getn 2; inext 'FG'; inver 1",
        "outname 'DFMPY'; outclass 'UVDATA'; outseq 1; outdisk 1",
        "go tacop; wait tacop",
        "default split",
        "indisk 1; getn 1; docal -1; flagver 1; stokes 'FULL'",
        "outclass 'SPLIT'; outdisk 1; douvcomp -1",
        "go split; wait split",
        "default fittp",
        "indisk 1; getn 3; dataout 'DFMPY:OUT.FITS'",
        "go fittp; wait fittp",
    ] + ZAP_ALL)
    assert "Using flag table version   1" in log, log[-3000:]
    out = tmp_path / "OUT.FITS"
    assert out.exists(), log[-3000:]

    from difmapy.io.uvfits import load_uvfits

    aips = load_uvfits(str(out))
    tt, a1, a2, *_ = o._core.rows()
    index = {(int(round(x * 10)), int(p), int(q)): i
             for i, (x, p, q) in enumerate(zip(tt, a1, a2))}
    ta, b1, b2, *_ = aips.rows()
    ta = np.asarray(ta) + (aips.ref_mjd - o._core.ref_mjd) * 86400.0
    rows = np.array([index[(int(round(x * 10)), int(p), int(q))]
                     for x, p, q in zip(ta, b1, b2)])
    mine = np.asarray(o.flags)
    perm = [list(aips.pols).index(c) for c in o._core.pols]
    np.testing.assert_array_equal(np.asarray(aips.flags())[:, :, perm],
                                  mine[rows])
    # Rows AIPS dropped altogether are the ones flagged throughout.
    gone = np.setdiff1d(np.arange(mine.shape[0]), rows)
    assert mine[gone].all()
