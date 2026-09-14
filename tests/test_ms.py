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

