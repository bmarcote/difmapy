"""Measurement Set loading tests, using a CASA-simulated MS containing
a point source at a known offset (ground truth incl. UVW conventions)."""

import os

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")

import difmapy

FLUX = 2.5
X0_MAS = 4.0  # eastward offset
Y0_MAS = -2.5  # northward offset
RA_DEG = 30.0
DEC_DEG = 40.0
CELL = 0.25
NX = 256


@pytest.fixture(scope="session")
def ms_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("data") / "sim.ms")
    clpath = path.replace(".ms", ".cl")

    sm = casatools.simulator()
    me = casatools.measures()
    sm.open(path)
    x = [-2112065.0, -1324009.0, -1449752.0, -1640953.0, -2409150.0]
    y = [-3705356.0, -5332181.0, -4975298.0, -5014816.0, -4478573.0]
    z = [4726813.0, 3231962.0, 3709123.0, 3575411.0, 3838617.0]
    sm.setconfig(
        telescopename="VLBA",
        x=x,
        y=y,
        z=z,
        dishdiameter=[25.0] * 5,
        mount=["alt-az"] * 5,
        antname=[f"AN{i}" for i in range(5)],
        coordsystem="global",
        referencelocation=me.observatory("VLBA"),
    )
    sm.setspwindow(spwname="IF1", freq="4.9GHz", deltafreq="1MHz",
                   freqresolution="1MHz", nchannels=4, stokes="RR LL")
    sm.setspwindow(spwname="IF2", freq="5.0GHz", deltafreq="1MHz",
                   freqresolution="1MHz", nchannels=4, stokes="RR LL")
    sm.setfield(
        sourcename="SYNTH",
        sourcedirection=me.direction("J2000", f"{RA_DEG}deg", f"{DEC_DEG}deg"),
    )
    sm.setauto(autocorrwt=0.0)
    sm.settimes(integrationtime="60s", usehourangle=True,
                referencetime=me.epoch("UTC", "2024/01/01/12:00:00"))
    sm.observe("SYNTH", "IF1", starttime="0s", stoptime="1800s")
    sm.observe("SYNTH", "IF2", starttime="0s", stoptime="1800s")
    sm.close()

    cl = casatools.componentlist()
    ra = RA_DEG + X0_MAS / 3.6e6 / np.cos(np.deg2rad(DEC_DEG))
    dec = DEC_DEG + Y0_MAS / 3.6e6
    cl.addcomponent(dir=f"J2000 {ra}deg {dec}deg", flux=FLUX, fluxunit="Jy",
                    shape="point", freq="4.9GHz", spectrumtype="constant")
    cl.rename(clpath)
    cl.close()

    sm.openfromms(path)
    sm.predict(complist=clpath)
    sm.close()
    return path


def test_ms_load_header(ms_path):
    o = difmapy.load(ms_path)
    assert o.source == "SYNTH"
    assert o.nif == 2
    assert o.nchan == [4, 4]
    assert o.npol == 2
    assert o.antennas == [f"AN{i}" for i in range(5)]
    c = o._core
    assert abs(np.rad2deg(c.ra) - RA_DEG) < 1e-9
    assert abs(np.rad2deg(c.dec) - DEC_DEG) < 1e-9
    ifs = c.ifs
    assert abs(ifs[0][0] - 4.9e9) < 1.0
    assert abs(ifs[1][0] - 5.0e9) < 1.0


def test_ms_invert_peak(ms_path):
    o = difmapy.load(ms_path)
    o.select("I").mapsize(NX, CELL)
    o.invert()
    # Only the inner quarter of the grid is valid (the outer margin is
    # amplified by the gridding correction), so search there.
    (x, y), peak = o.peak_offset()
    assert (x, y) == pytest.approx((X0_MAS, Y0_MAS), abs=CELL), (
        f"peak at ({x}, {y}) mas, expected ({X0_MAS}, {Y0_MAS}) mas"
    )
    assert abs(peak - FLUX) / FLUX < 0.03


def test_ms_clean_selfcal_smoke(ms_path):
    o = difmapy.load(ms_path)
    o.select("I").mapsize(NX, CELL)
    o.add_window(X0_MAS - 2, X0_MAS + 2, Y0_MAS - 2, Y0_MAS + 2)
    res = o.clean(300, 0.1)
    assert abs(res["cleaned_flux"] - FLUX) / FLUX < 0.03
    res = o.selfcal(phase=True)
    assert res["nbadsol"] == 0
    stats = o.imstat()
    assert max(abs(stats["min"]), abs(stats["max"])) < 0.05 * FLUX


def test_save_writes_an_ms_when_the_session_came_from_one(ms_path, tmp_path):
    """save() of an MS-loaded session must produce a .ms alongside the
    .uvf, carrying this session's flags and calibrated data, and leave
    the MS it was loaded from untouched."""
    obs = difmapy.Observation.from_ms(ms_path)
    obs.mapsize(NX, CELL)
    obs.flag(station="AN0")
    nflagged = int(np.asarray(obs._core.flags()).sum())
    assert nflagged > 0

    prefix = str(tmp_path / "session")
    obs.save(prefix)
    out = f"{prefix}.ms"
    assert os.path.isdir(out)
    assert os.path.exists(f"{prefix}.uvf")

    # The originating MS is not modified.
    src = difmapy.Observation.from_ms(ms_path)
    assert int(np.asarray(src._core.flags()).sum()) == 0

    # The flags are in the copy...
    back = difmapy.Observation.from_ms(out)
    assert int(np.asarray(back._core.flags()).sum()) == nflagged

    # ...and the calibrated visibilities are in CORRECTED_DATA, while
    # DATA still holds what CASA simulated.
    corr = difmapy.Observation.from_ms(out, data_column="CORRECTED_DATA")
    vis, _ = corr._core.calibrated_cube()
    ref, _ = obs._core.calibrated_cube()
    assert np.allclose(np.asarray(vis), np.asarray(ref), atol=1e-5)


def test_save_ms_refuses_to_clobber_the_original(ms_path):
    from difmapy.io.ms import save_ms

    obs = difmapy.Observation.from_ms(ms_path)
    with pytest.raises(ValueError, match="originating MS"):
        save_ms(obs._core, ms_path)


def test_save_ms_needs_an_ms_to_start_from(uvfits_file, tmp_path):
    """A UVFITS session has no MS to copy: saying so beats writing a
    directory that is not a valid Measurement Set."""
    obs = difmapy.load(uvfits_file)
    prefix = str(tmp_path / "u")
    obs.save(prefix)                      # no .ms, and no complaint
    assert not os.path.exists(f"{prefix}.ms")
    with pytest.raises(ValueError, match="not loaded from a Measurement Set"):
        obs.save(prefix, ms=True)



# ---------------------------------------------------------------------
# averaged data: save() and save_flags()
# ---------------------------------------------------------------------


def _match_rows(a, b):
    """Rows of `a` holding the same (time, baseline) as each row of `b`."""
    ta, pa, qa, *_ = (np.asarray(x) for x in a._core.rows())
    tb_, pb, qb, *_ = (np.asarray(x) for x in b._core.rows())
    ta = ta + (a._core.ref_mjd - b._core.ref_mjd) * 86400.0
    index = {(round(float(t) * 1e3), int(p), int(q)): i
             for i, (t, p, q) in enumerate(zip(ta, pa, qa))}
    return np.array([index[(round(float(t) * 1e3), int(p), int(q))]
                     for t, p, q in zip(tb_, pb, qb)])


@pytest.mark.parametrize("averaging", [
    {"timeavg": 180}, {"freqavg": 2}, {"timeavg": 180, "freqavg": "all"},
], ids=["time", "frequency", "both"])
def test_save_writes_an_ms_for_averaged_data(ms_path, tmp_path, averaging):
    """Averaged rows do not match the source MS's, so the .ms is built
    anew - and must read back as exactly what the session holds."""
    obs = difmapy.load(ms_path, **averaging)
    obs.mapsize(NX, CELL)
    obs.flag(station="AN1", tmin=0.0, tmax=float(obs._core.times()[2]))
    # A calibration that visibly changes the data (the simulation is
    # perfect, so self-calibration would find nothing to correct).
    gains = np.ones((obs.nif, len(obs.antennas)), np.float32)
    gains[:, 2] = 1.25
    obs._core.apply_gain_factors(gains, 0.3 * (gains - 1))
    prefix = str(tmp_path / "avg")
    files = obs.save(prefix, quiet=True)
    assert files["ms"] == f"{prefix}.ms" and os.path.isdir(files["ms"])

    back = difmapy.load(files["ms"], data_column="CORRECTED_DATA", stokes=None)
    assert back.nchan == obs.nchan
    for (f0, d0, _), (f1, d1, _) in zip(obs._core.ifs, back._core.ifs):
        assert f1 == pytest.approx(f0, abs=1.0) and d1 == pytest.approx(d0)
    rows = _match_rows(obs, back)
    assert len(rows) == obs._core.nrow
    mine, wt = (np.asarray(x) for x in obs._core.calibrated_cube())
    theirs, wt2 = (np.asarray(x) for x in back._core.calibrated_cube())
    np.testing.assert_array_equal(np.asarray(back.flags),
                                  np.asarray(obs.flags)[rows])
    np.testing.assert_allclose(theirs, mine[rows], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.abs(wt2), np.abs(wt[rows]), rtol=1e-6)
    _, _, _, u0, v0, _ = (np.asarray(x) for x in obs._core.rows())
    _, _, _, u1, v1, _ = (np.asarray(x) for x in back._core.rows())
    np.testing.assert_allclose(u1, u0[rows], rtol=1e-12)

    # DATA is the averaged data without the session's calibration.
    raw = difmapy.load(files["ms"], stokes=None)
    good = wt[rows] > 0
    ref = difmapy.load(ms_path, stokes=None, **averaging)
    np.testing.assert_allclose(
        np.asarray(raw._core.calibrated_cube()[0])[good],
        np.asarray(ref._core.calibrated_cube()[0])[rows][good], rtol=1e-6)
    assert np.abs(np.asarray(raw._core.calibrated_cube()[0]) - theirs
                  )[good].max() > 1e-4

    # Writing window by window (the path for unequal channel counts)
    # gives the same Measurement Set.
    from difmapy.io.ms import save_averaged_ms

    other = save_averaged_ms(obs._core, str(tmp_path / "bywin.ms"),
                             by_window=True)
    tb = casatools.table()
    cols = {}
    for path in (files["ms"], other):
        tb.open(path)
        cols[path] = {c: tb.getcol(c) for c in
                      ("DATA", "CORRECTED_DATA", "FLAG", "WEIGHT", "TIME",
                       "DATA_DESC_ID", "SCAN_NUMBER", "INTERVAL")}
        tb.close()
    for c, v in cols[files["ms"]].items():
        np.testing.assert_array_equal(v, cols[other][c], err_msg=c)

    # And CASA takes it for a Measurement Set.
    casatasks = pytest.importorskip("casatasks")
    summary = casatasks.listobs(vis=files["ms"])
    assert summary["nfields"] == 1


def test_save_flags_from_averaged_data(ms_path, tmp_path):
    """A flag on an averaged sample flags every MS sample behind it;
    nothing is unflagged, and nothing else is touched."""
    import shutil

    work = str(tmp_path / "work.ms")
    shutil.copytree(ms_path, work)
    # Something already flagged in the file, inside the range below.
    tb = casatools.table()
    tb.open(work, nomodify=False)
    f = tb.getcol("FLAG")
    f[:, 0, 7] = True
    tb.putcol("FLAG", f)
    tb.close()

    before = difmapy.load(work, stokes=None)
    f_before = np.array(before.flags, copy=True)
    obs = difmapy.load(work, timeavg=180, freqavg=2)
    assert not hasattr(obs._core, "_ms_origin")   # averaged: rows differ
    times = np.asarray(obs._core.times())
    obs.flag(station="AN1", tmin=float(times[1]) - 1, tmax=float(times[2]) + 1)
    n = obs.save_flags()
    assert n > 0

    after = difmapy.load(work, stokes=None)
    f_after = np.asarray(after.flags)
    t, a1, a2, *_ = (np.asarray(x) for x in after._core.rows())
    ia = after.antennas.index("AN1")
    # The averaged integrations 1 and 2 are these 180 s bins of the day.
    bins = np.floor(times[1:3] / 180.0)
    hit = ((a1 == ia) | (a2 == ia)) & np.isin(np.floor(t / 180.0), bins)
    assert hit.sum() > 0 and f_after[hit].all()
    np.testing.assert_array_equal(f_after[~hit], f_before[~hit])
    assert (f_after | ~f_before).all(), "nothing may be unflagged"
    assert obs.save_flags() == 0, "a second call has nothing to add"


def test_save_writes_caltable_and_image(uvfits_file, tmp_path):
    from astropy.io import fits

    obs = difmapy.load(uvfits_file)
    obs.mapsize(NX, CELL)
    prefix = str(tmp_path / "plain")
    files = obs.save(prefix, quiet=True)
    # Nothing calibrated or imaged yet: neither is written...
    assert "caltable" not in files and "image" not in files
    for kind in ("caltable", "image"):
        with pytest.raises(ValueError, match=f"{kind}=True"):
            obs.save(prefix, quiet=True, **{kind: True})

    obs.clean(100, 0.1, quiet=True)
    obs.selfcal(phase=True, quiet=True)
    prefix = str(tmp_path / "done")
    files = obs.save(prefix, quiet=True)
    # ...and afterwards both are, the table in the data's own format.
    assert files["caltable"] == f"{prefix}.TASAV.FITS"
    assert files["image"] == f"{prefix}.fits"
    with fits.open(files["caltable"]) as h:
        assert "AIPS SN" in [x.name for x in h]
    with fits.open(files["image"]) as h:
        assert h[0].header["BUNIT"].strip() == "JY/BEAM"
        assert np.nanmax(h[0].data) == pytest.approx(FLUX, rel=0.05)
    files = obs.save(prefix, quiet=True, caltable=False, image=False)
    assert "caltable" not in files and "image" not in files
    # get() still restores the session from the same prefix.
    back = difmapy.Observation.get(prefix)
    assert len(back.model) == len(obs.model)


def test_casa_applies_the_flag_commands_like_save_flags(ms_path, tmp_path):
    """`wflags` for a Measurement Set: CASA's flagdata, given the list,
    must flag exactly what `save_flags` writes - also from averaged
    data, where one entry stands for a whole bin."""
    import shutil

    casatasks = pytest.importorskip("casatasks")
    for averaging in ({}, {"timeavg": 180, "freqavg": 2}):
        a, b = str(tmp_path / "a.ms"), str(tmp_path / "b.ms")
        for d in (a, b):
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(ms_path, d)
        obs = difmapy.load(a, **averaging)
        assert obs.caltable_formats() == ("casa",)
        t = np.asarray(obs._core.times())
        obs.flag(station="AN1", tmin=float(t[1]) - 1, tmax=float(t[3]) + 1)
        obs.flag(baseline=("AN0", "AN2"), if_index=1)
        cmds = str(tmp_path / "flags.flagcmd")
        info = obs.wflags(cmds, quiet=True)
        assert info["format"] == "casa" and info["nrows"] == 2
        casatasks.flagdata(vis=b, mode="list", inpfile=cmds, flagbackup=False)
        obs.save_flags()
        fa = np.asarray(difmapy.load(a, stokes=None).flags)
        fb = np.asarray(difmapy.load(b, stokes=None).flags)
        assert fa.sum() > 0
        np.testing.assert_array_equal(fb, fa)
        # save() writes the same list, by default.
        files = obs.save(str(tmp_path / "sess"), quiet=True, ms=False)
        assert files["flags"].endswith("sess.flagcmd")
