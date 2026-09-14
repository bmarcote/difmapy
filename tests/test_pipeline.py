"""End-to-end tests of the Python API on synthetic UVFITS data."""

import numpy as np
import pytest

import difmapy
from conftest import DEC_DEG, FLUX, NCHAN, NIF, RA_DEG, X0_MAS, Y0_MAS

# 0.25 mas/pixel keeps the longest synthetic baselines (1.3e8 lambda)
# below the gridding Nyquist limit uinc*(nx/4 - 2) = 2e8 lambda.
CELL = 0.25  # mas/pixel
NX = 256


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(NX, CELL)
    return o


def test_load_header(uvfits_file):
    o = difmapy.load(uvfits_file)
    assert o.source == "SYNTH"
    assert o.nif == NIF
    assert o.nchan == [NCHAN] * NIF
    assert o.npol == 2
    assert len(o.antennas) == 5
    txt = o.header()
    assert "SYNTH" in txt and "IF 2" in txt


def test_select_channels(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I", channels=[(4, 7)])  # IF 2 only
    sel = o._core.selection()
    assert sel["if_used"] == [False, True]
    with pytest.raises(ValueError):
        o.select("Q")  # only RR/LL present


def test_invert_peak(obs):
    res = obs.invert()
    assert res["nused"] == obs._core.nrow * NIF
    beam = obs.dbeam
    assert abs(beam[NX // 2, NX // 2] - 1.0) < 0.01
    (bx, by), bpeak = obs.peak_offset(beam)
    assert (bx, by) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert abs(bpeak - 1.0) < 0.01
    (x, y), peak = obs.peak_offset()
    assert (x, y) == pytest.approx((X0_MAS, Y0_MAS), abs=CELL)
    assert abs(peak - FLUX) / FLUX < 0.02


def test_clean_restore_wmap(obs, tmp_path):
    obs.add_window(X0_MAS - 2, X0_MAS + 2, Y0_MAS - 2, Y0_MAS + 2)
    res = obs.clean(300, 0.1)
    assert abs(res["cleaned_flux"] - FLUX) / FLUX < 0.02
    assert abs(obs.model_flux - FLUX) / FLUX < 0.02

    # Residual map is nearly empty.
    stats = obs.imstat()
    assert max(abs(stats["min"]), abs(stats["max"])) < 0.05 * FLUX

    cln = obs.restore()
    (x, y), peak = obs.peak_offset(cln)
    assert (x, y) == pytest.approx((X0_MAS, Y0_MAS), abs=CELL)
    assert abs(peak - FLUX) / FLUX < 0.05

    # FITS output: the peak must be at the right sky position.
    from astropy.io import fits
    from astropy.wcs import WCS

    path = tmp_path / "clean.fits"
    obs.wmap(str(path))
    with fits.open(path) as hdul:
        img = hdul[0].data.squeeze()
        w = WCS(hdul[0].header).celestial
        iy, ix = np.unravel_index(np.argmax(img), img.shape)
        sky = w.pixel_to_world(ix, iy)
        exp_dec = DEC_DEG + Y0_MAS / 3.6e6
        exp_ra = RA_DEG + X0_MAS / 3.6e6 / np.cos(np.deg2rad(DEC_DEG))
        assert abs(sky.dec.deg - exp_dec) * 3.6e6 < CELL
        assert abs((sky.ra.deg - exp_ra) * np.cos(np.deg2rad(DEC_DEG))) * 3.6e6 < CELL


def test_model_io(obs, tmp_path):
    obs.add_window(X0_MAS - 2, X0_MAS + 2, Y0_MAS - 2, Y0_MAS + 2)
    obs.clean(300, 0.1)
    path = tmp_path / "test.mod"
    obs.wmodel(str(path))
    flux0 = obs.model_flux

    obs.clrmod()
    assert obs.model_flux == 0.0
    obs.rmodel(str(path))
    assert abs(obs.model_flux - flux0) / flux0 < 1e-4
    # Positions survive the polar-coordinate roundtrip.
    main = max(obs.model, key=lambda c: c["flux"])
    assert abs(main["x"] - X0_MAS) < 0.51 * CELL
    assert abs(main["y"] - Y0_MAS) < 0.51 * CELL


def test_selfcal(corrupted_uvfits_file):
    path, gerr = corrupted_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    o.mapsize(NX, CELL)
    # True model: point source at the phase center offset.
    o.addcmp(FLUX, X0_MAS, Y0_MAS)

    res = o.selfcal(amp=True, phase=True, float_scale=True)
    assert res["nbadsol"] == 0

    vis, wt = o._core.stream_vis()
    model = o._core.stream_model()
    good = wt > 0
    mismatch = np.abs(vis[good] - model[good]).max()
    assert mismatch < 0.01 * FLUX

    # Amplitude corrections approximate the inverse gain errors.
    amp, phs, _ = o._core.gains()
    nant = len(o.antennas)
    amp = amp.reshape(o._core.ntimes, o.nif, nant)
    for ia, (aerr, _) in enumerate(gerr):
        got = amp[10, 0, ia]
        assert abs(got - 1 / aerr) * aerr < 0.02

    # uncalib restores the corrupted data.
    o.uncalib()
    amp, phs, _ = o._core.gains()
    assert np.allclose(amp, 1.0) and np.allclose(phs, 0.0)


def test_uniform_vs_natural(obs):
    obs.uvweight(binwid=2.0, errpow=0.0)
    r_uni = dict(obs.invert())
    obs.uvweight(binwid=0.0, errpow=-1.0)
    r_nat = dict(obs.invert())
    # Uniform weighting must not broaden the beam.
    assert r_uni["e_bmaj"] <= r_nat["e_bmaj"] * 1.05
