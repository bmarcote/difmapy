"""CASA calibration-table export.

The decisive test is not that the file looks right, but that CASA
itself accepts it and that `applycal` reproduces difmapy's corrected
visibilities - which pins down the gain convention (CASA divides by
antenna gains, difmapy multiplies by corrections, so the table must
hold the reciprocal).
"""

import os
import shutil

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")
casatasks = pytest.importorskip("casatasks")

import difmapy

HERE = os.path.dirname(__file__)
SRC_MS = os.path.join(HERE, "rsm07_3C345.ms")
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(SRC_MS), reason="real 3C345 MS not present"
)


def _calibrated(path, amp=True, solint=0.0):
    """Load an MS and build up some real calibration."""
    o = difmapy.load(path)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.startmod(flux=1.0)
    o.clean(200, 0.03)
    o.selfcal(phase=True)
    o.clean(200, 0.03)
    o.selfcal(amp=amp, phase=True, solint=solint)
    return o


@pytest.fixture()
def ms(tmp_path):
    dst = str(tmp_path / "work.ms")
    shutil.copytree(SRC_MS, dst)
    return dst


def test_table_is_valid_for_casa(ms, tmp_path):
    o = _calibrated(ms)
    out = str(tmp_path / "cal.G")
    info = o.savecaltable(out, quiet=True)

    assert os.path.isdir(out)
    assert info["nrows"] == o._core.ntimes * o.nif * len(o.antennas)
    assert info["warnings"] == []

    tb = casatools.table()
    tb.open(out)
    try:
        # The structure CASA expects of a G Jones NewCalTable.
        assert tb.info()["type"] == "Calibration"
        assert tb.info()["subType"] == "G Jones"
        assert tb.getkeyword("VisCal") == "G Jones"
        assert tb.getkeyword("ParType") == "Complex"
        assert tb.getcol("CPARAM").shape == (2, 1, info["nrows"])
        # Antenna-based tables mark ANTENNA2 as -1.
        assert (tb.getcol("ANTENNA2") == -1).all()
        assert set(tb.getcol("SPECTRAL_WINDOW_ID")) == {0, 1, 2, 3}
        assert tb.getcolkeyword("TIME", "QuantumUnits")[0] == "s"
        for sub in ("ANTENNA", "FIELD", "SPECTRAL_WINDOW", "OBSERVATION"):
            assert os.path.isdir(os.path.join(out, sub))
    finally:
        tb.close()

    # CASA must accept it without complaint.
    casatasks.applycal(vis=ms, gaintable=[out], interp=["nearest"],
                       calwt=False, applymode="calonly", flagbackup=False)


def test_applycal_reproduces_difmapy_visibilities(ms, tmp_path):
    """The whole point: CASA's CORRECTED_DATA must equal what difmapy
    shows, which is only true if the reciprocal convention is right."""
    o = _calibrated(ms, amp=True, solint=30)
    out = str(tmp_path / "cal.G")
    o.savecaltable(out, quiet=True)
    casatasks.applycal(vis=ms, gaintable=[out], interp=["nearest"],
                       calwt=False, applymode="calonly", flagbackup=False)

    mine, wt = (np.asarray(x) for x in o._core.calibrated_cube())
    ms_row = o._core._ms_origin["ms_row"]
    tb = casatools.table()
    tb.open(ms)
    try:
        corr = tb.getcol("CORRECTED_DATA")
        raw = tb.getcol("DATA")
    finally:
        tb.close()
    pols = o._core.pols
    worst = 0.0
    scale = np.median(np.abs(mine[wt > 0]))
    for cif in range(o.nif):
        rows = ms_row[:, cif]
        ok = rows >= 0
        good = wt[ok, cif, pols.index(-1)] > 0
        for ip in (pols.index(-1), pols.index(-2)):
            casa = corr[ip, 0, rows[ok]][good]
            dif = mine[ok, cif, ip][good]
            worst = max(worst, float(np.abs(casa - dif).max()))
    assert worst < 1e-4 * scale, f"deviation {worst} on |V| ~ {scale}"
    # And the correction was not a no-op.
    assert (np.abs(corr - raw) > 1e-6).mean() > 0.1


def test_gains_actually_needed_to_match(ms, tmp_path):
    """Sanity check on the previous test: without applying the table,
    CORRECTED_DATA would *not* match difmapy."""
    o = _calibrated(ms)
    out = str(tmp_path / "cal.G")
    o.savecaltable(out, quiet=True)
    # Reset CORRECTED_DATA to the raw data, i.e. apply nothing.
    casatasks.clearcal(vis=ms, addmodel=False)
    mine, wt = (np.asarray(x) for x in o._core.calibrated_cube())
    ms_row = o._core._ms_origin["ms_row"]
    tb = casatools.table()
    tb.open(ms)
    try:
        corr = tb.getcol("CORRECTED_DATA")
    finally:
        tb.close()
    ip = o._core.pols.index(-1)
    rows = ms_row[:, 0]
    ok = rows >= 0
    good = wt[ok, 0, ip] > 0
    dev = np.abs(corr[ip, 0, rows[ok]][good] - mine[ok, 0, ip][good]).max()
    assert dev > 1e-3, "the gains must make a measurable difference"


def test_apply_to_a_different_ms(ms, tmp_path):
    """Gains from one MS applied to another with the same stations -
    the reason for having this feature at all."""
    other = str(tmp_path / "other.ms")
    shutil.copytree(SRC_MS, other)
    o = _calibrated(ms)
    out = str(tmp_path / "cal.G")
    o.savecaltable(out, ms=other, quiet=True)
    casatasks.applycal(vis=other, gaintable=[out], interp=["nearest"],
                       calwt=False, applymode="calonly", flagbackup=False)

    # Reloading the corrected MS must reproduce difmapy's own view.
    back = difmapy.load(other, data_column="CORRECTED_DATA")
    back.select("I")
    v1, w1 = (np.asarray(x) for x in o._core.stream_vis())
    v2, w2 = (np.asarray(x) for x in back._core.stream_vis())
    g = (w1 > 0) & (w2 > 0)
    assert g.sum() > 10000
    scale = np.median(np.abs(v1[g]))
    assert np.abs(v1[g] - v2[g]).max() < 1e-4 * scale


def test_phase_only_and_amp_only(ms, tmp_path):
    """A phase-only table must have unit amplitude, and gscale must give
    a time-constant amplitude table."""
    o = difmapy.load(ms)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.startmod(flux=1.0)  # phase self-cal only
    out = str(tmp_path / "ph.G")
    o.savecaltable(out, quiet=True)
    tb = casatools.table()
    tb.open(out)
    try:
        g = tb.getcol("CPARAM")[0, 0]
    finally:
        tb.close()
    assert np.allclose(np.abs(g), 1.0, atol=1e-5)
    assert np.abs(np.angle(g)).max() > 0.01  # phases were solved

    o2 = difmapy.load(ms)
    o2.select("I")
    o2.mapsize(1024, 1.0)
    o2.addcmp(1.0, 0.0, 0.0)
    o2.gscale()
    out2 = str(tmp_path / "amp.G")
    o2.savecaltable(out2, quiet=True)
    tb.open(out2)
    try:
        g2 = tb.getcol("CPARAM")[0, 0]
        a2 = tb.getcol("ANTENNA1")
        s2 = tb.getcol("SPECTRAL_WINDOW_ID")
    finally:
        tb.close()
    assert np.abs(np.abs(g2) - 1.0).max() > 1e-3  # amplitudes were solved
    # gscale gives one solution per antenna *and IF*, so within a given
    # (antenna, spw) the gain must be constant in time.
    for ant in np.unique(a2)[:4]:
        for spw in np.unique(s2):
            vals = np.abs(g2[(a2 == ant) & (s2 == spw)])
            assert np.allclose(vals, vals[0], rtol=1e-5), (ant, spw)
    # ...and it generally differs between IFs, which are solved apart.
    per_if = [np.abs(g2[(a2 == a2[0]) & (s2 == spw)])[0] for spw in np.unique(s2)]
    assert len(set(np.round(per_if, 6))) > 1


def test_reciprocal_convention_explicitly(ms, tmp_path):
    """CPARAM must be the reciprocal of difmapy's correction."""
    o = _calibrated(ms)
    out = str(tmp_path / "cal.G")
    o.savecaltable(out, quiet=True)
    amp, phs, _ = (np.asarray(x) for x in o._core.gains())
    nant, nif, nt = len(o.antennas), o.nif, o._core.ntimes
    corr = (amp * np.exp(1j * phs)).reshape(nt, nif, nant)

    tb = casatools.table()
    tb.open(out)
    try:
        cp = tb.getcol("CPARAM")[0, 0]
    finally:
        tb.close()
    # Rows are ordered (time, IF, antenna).
    assert np.allclose(cp, (1.0 / corr).ravel().astype(np.complex64),
                       rtol=1e-5, atol=1e-7)


def test_incremental_tables_compose(ms, tmp_path):
    """One table per self-cal round, applied as a chain by CASA, must be
    equivalent to a single cumulative table."""
    o = difmapy.load(ms)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.startmod(flux=1.0)
    o.clean(200, 0.03)

    # Round 1, then a snapshot, then round 2.
    o.selfcal(phase=True)
    t1 = str(tmp_path / "round1.G")
    o.savecaltable(t1, quiet=True)
    mark = o.gain_snapshot()
    o.clean(200, 0.03)
    o.selfcal(amp=True, phase=True, solint=30)

    t2 = str(tmp_path / "round2.G")  # increment only
    o.savecaltable(t2, quiet=True, since=mark)
    cum = str(tmp_path / "cumulative.G")
    o.savecaltable(cum, quiet=True)

    # The increment must differ from the cumulative table...
    tb = casatools.table()
    tb.open(t2)
    try:
        inc = tb.getcol("CPARAM")[0, 0]
    finally:
        tb.close()
    tb.open(cum)
    try:
        tot = tb.getcol("CPARAM")[0, 0]
    finally:
        tb.close()
    assert not np.allclose(inc, tot)
    # ...and the product of the chain must equal the cumulative gains.
    tb.open(t1)
    try:
        first = tb.getcol("CPARAM")[0, 0]
    finally:
        tb.close()
    assert np.allclose(first * inc, tot, rtol=1e-4, atol=1e-6)

    # Confirm with CASA: chain vs cumulative give the same data.
    casatasks.applycal(vis=ms, gaintable=[t1, t2], interp=["nearest"] * 2,
                       calwt=False, applymode="calonly", flagbackup=False)
    tb.open(ms)
    try:
        chained = tb.getcol("CORRECTED_DATA").copy()
    finally:
        tb.close()
    casatasks.applycal(vis=ms, gaintable=[cum], interp=["nearest"],
                       calwt=False, applymode="calonly", flagbackup=False)
    tb.open(ms)
    try:
        single = tb.getcol("CORRECTED_DATA")
        flag = tb.getcol("FLAG")
    finally:
        tb.close()
    # Compare relative deviations on unflagged data only: flagged
    # samples in this file reach |V| ~ 2000, where applying two tables
    # in sequence rounds twice and differs by a few float32 ulps.
    use = ~flag & (np.abs(single) > 0.01)
    assert use.sum() > 10000
    rel = np.abs(chained[use] - single[use]) / np.abs(single[use])
    assert rel.max() < 1e-5, f"max relative deviation {rel.max()}"


def test_snapshot_shape_is_checked(ms, tmp_path):
    o = _calibrated(ms)
    mark = o.gain_snapshot()
    mark["amp"] = mark["amp"][:, :, :-1]  # wrong shape
    with pytest.raises(ValueError, match="does not match"):
        o.savecaltable(str(tmp_path / "x.G"), since=mark, quiet=True)


def test_warns_about_resoff_and_shift(ms, tmp_path):
    o = _calibrated(ms)
    o.resoff()
    o.shift(3.0, -2.0)
    info = o.savecaltable(str(tmp_path / "cal.G"), quiet=True)
    joined = " ".join(info["warnings"])
    assert "baseline-based" in joined
    assert "phase-centre shift" in joined
    # The table is still written, and still valid.
    casatasks.applycal(vis=ms, gaintable=[info["path"]], interp=["nearest"],
                       calwt=False, applymode="calonly", flagbackup=False)


def test_flag_uncalibrated_option(ms, tmp_path):
    o = difmapy.load(ms)
    o.select("I")
    plain = o.savecaltable(str(tmp_path / "a.G"), quiet=True)
    assert plain["nflagged"] == 0  # nothing solved, nothing flagged
    flagged = o.savecaltable(str(tmp_path / "b.G"), flag_uncalibrated=True,
                             quiet=True)
    assert flagged["nflagged"] == flagged["nrows"]


def test_requires_a_measurement_set(tmp_path):
    """UVFITS-loaded data have no MS metadata, so one must be supplied."""
    if not os.path.exists(UVF):
        pytest.skip("uvfits not present")
    o = difmapy.load(UVF)
    o.select("I")
    with pytest.raises(ValueError, match="Measurement Set is needed"):
        o.savecaltable(str(tmp_path / "x.G"), outformat="CASA", quiet=True)
    # With a reference MS it works: antennas are matched by name.
    info = o.savecaltable(str(tmp_path / "y.G"), ms=SRC_MS, quiet=True)
    assert info["nrows"] > 0
    tb = casatools.table()
    tb.open(info["path"])
    try:
        assert tb.info()["type"] == "Calibration"
    finally:
        tb.close()


def test_unknown_antenna_is_reported(ms, tmp_path):
    o = difmapy.load(ms)
    o.select("I")
    # Rename a station so it no longer matches the MS.
    names = list(o._core.antenna_names)
    o._core.set_antenna_constraints(names[0], False, 1.0)  # sanity: exists
    tb = casatools.table()
    tb.open(os.path.join(ms, "ANTENNA"), nomodify=False)
    try:
        n = tb.getcol("NAME")
        n[0] = "ZZ"
        tb.putcol("NAME", n)
    finally:
        tb.close()
    with pytest.raises(ValueError, match="not in .*ANTENNA"):
        o.savecaltable(str(tmp_path / "z.G"), quiet=True)
