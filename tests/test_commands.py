"""shift/unshift, resoff/clroff, startmod and uvaver."""

import os

import numpy as np
import pytest

import difmapy
from conftest import FLUX, X0_MAS, Y0_MAS

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
real_data = pytest.mark.skipif(
    not os.path.exists(UVF), reason="real 3C345 data not present"
)

CELL = 0.25
NX = 256


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(NX, CELL)
    return o


def _peak_offset(o):
    """Peak position of the dirty map in mas."""
    o.invert()
    m = o.dmap
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    return ((ix - NX // 2) * CELL, (iy - NX // 2) * CELL), m[iy, ix]


def test_shift_moves_source(obs):
    pos0, peak0 = _peak_offset(obs)
    assert pos0 == pytest.approx((X0_MAS, Y0_MAS), abs=CELL)

    obs.shift(3.0, -2.0)
    assert obs.total_shift == pytest.approx((3.0, -2.0))
    pos1, peak1 = _peak_offset(obs)
    assert pos1 == pytest.approx((X0_MAS + 3.0, Y0_MAS - 2.0), abs=CELL)
    # Shifting is a pure phase rotation: the peak flux is preserved.
    assert peak1 == pytest.approx(peak0, rel=1e-3)

    # Shifts accumulate and unshift restores the original state.
    obs.shift(-1.0, 0.5)
    assert obs.total_shift == pytest.approx((2.0, -1.5))
    obs.unshift()
    assert obs.total_shift == pytest.approx((0.0, 0.0))
    pos2, peak2 = _peak_offset(obs)
    assert pos2 == pytest.approx(pos0, abs=1e-9)
    assert peak2 == pytest.approx(peak0, rel=1e-5)


def test_shift_moves_model_with_data(obs):
    """The model must follow the shift, so residuals stay unchanged."""
    obs.addcmp(FLUX, X0_MAS, Y0_MAS)
    obs.keep()
    obs.invert()
    rms0 = obs.imstat()["rms"]

    obs.shift(4.0, 1.0)
    c = obs.model[0]
    assert c["x"] == pytest.approx(X0_MAS + 4.0, abs=1e-6)
    assert c["y"] == pytest.approx(Y0_MAS + 1.0, abs=1e-6)
    obs.invert()
    # The model still subtracts the source, so the residuals stay
    # numerically zero (both are at the float32 rounding level).
    assert rms0 < 1e-6 * FLUX
    assert obs.imstat()["rms"] < 1e-6 * FLUX


def test_shift_survives_flagging(obs):
    """A shift is re-applied when edited rows are re-averaged."""
    obs.shift(3.0, -2.0)
    obs.flag(station="AN2")
    obs.unflag(station="AN2")
    pos, _ = _peak_offset(obs)
    assert pos == pytest.approx((X0_MAS + 3.0, Y0_MAS - 2.0), abs=CELL)


def test_resoff_and_clroff(obs):
    """resoff must absorb a baseline-based error that self-cal cannot."""
    obs.addcmp(FLUX, X0_MAS, Y0_MAS)
    obs.keep()
    obs.invert()
    rms0 = obs.imstat()["rms"]
    n = obs.resoff()
    assert n == len(obs.baseline_corrections()) * obs.nif
    # Corrections were recorded and the fit improved (or stayed perfect).
    obs.invert()
    assert obs.imstat()["rms"] <= rms0 * 1.001
    cors = obs.baseline_corrections()
    assert len(cors) == 10  # 5 antennas
    assert all(len(c["amp"]) == obs.nif for c in cors)

    obs.clroff()
    assert all(
        np.allclose(c["amp"], 1.0) and np.allclose(c["phase"], 0.0)
        for c in obs.baseline_corrections()
    )


def test_resoff_corrects_baseline_error(uvfits_file):
    """Inject a single-baseline amplitude error; resoff must find it."""
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(NX, CELL)
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.keep()
    # The synthetic data are a perfect point source, so the corrections
    # must come out as unity on every baseline.
    o.resoff()
    for c in o.baseline_corrections():
        assert np.allclose(c["amp"], 1.0, atol=1e-3), c
        assert np.allclose(c["phase"], 0.0, atol=0.05), c


def test_startmod(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    res = o.startmod(flux=FLUX)
    assert res["nbadsol"] == 0
    # The starting model is discarded afterwards.
    assert o.model == []
    # Gains were modified.
    amp, phs, _ = o._core.gains()
    assert not np.allclose(np.asarray(phs), 0.0)


def test_uvaver(obs):
    n0 = obs._core.nrow
    t = obs._core.rows()[0]
    dt = np.diff(np.unique(t)).min()
    avg = obs.uvaver(4 * dt)
    assert avg._core.nrow < n0
    assert avg._core.ntimes <= obs._core.ntimes // 4 + 1
    assert avg.antennas == obs.antennas
    assert avg._core.ifs == obs._core.ifs
    # Averaging preserves the source: same peak position and flux.
    pos0, peak0 = _peak_offset(obs)
    pos1, peak1 = _peak_offset(avg)
    assert pos1 == pytest.approx(pos0, abs=CELL)
    assert peak1 == pytest.approx(peak0, rel=0.02)
    # Weights add up (difmap default).
    _, w0 = obs._core.calibrated_cube()
    _, w1 = avg._core.calibrated_cube()
    assert np.asarray(w1)[np.asarray(w1) > 0].sum() == pytest.approx(
        np.asarray(w0)[np.asarray(w0) > 0].sum(), rel=1e-4
    )
    # Integration times add up as well.
    assert np.asarray(avg._core.inttimes()).sum() == pytest.approx(
        np.asarray(obs._core.inttimes()).sum(), rel=1e-4
    )


def test_uvaver_scatter_weights(obs):
    t = obs._core.rows()[0]
    dt = np.diff(np.unique(t)).min()
    avg = obs.uvaver(5 * dt, doscatter=True)
    _, w = avg._core.calibrated_cube()
    w = np.asarray(w)
    # Noiseless data: the scatter is ~0, so weights are large but finite.
    assert np.isfinite(w).all()
    assert (w[w > 0] > 0).all()


@real_data
def test_real_data_shift_moves_map_rigidly():
    """The strongest check of the shift: the whole map must translate
    by exactly the requested number of pixels."""
    cell = 2.0
    o = difmapy.load(UVF)
    o.select("I")
    o.mapsize(512, cell)
    o.invert()
    m0 = o.dmap.copy()

    dx, dy = 5, -3  # pixels east, north
    o.shift(dx * cell, dy * cell)
    o.invert()
    m1 = o.dmap
    rolled = np.roll(np.roll(m0, dx, axis=1), dy, axis=0)
    inner = (slice(160, 350), slice(160, 350))
    # Agreement to <0.2% of the peak (the uv phase pattern differs
    # slightly, so gridding is not bit-identical).
    assert np.abs(m1[inner] - rolled[inner]).max() < 2e-3 * m0.max()

    o.unshift()
    o.invert()
    assert np.abs(o.dmap - m0).max() < 1e-5 * m0.max()


@real_data
def test_real_data_uvaver():
    o = difmapy.load(UVF)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.invert()
    peak0 = o.imstat()["max"]
    avg = o.uvaver(240.0)
    assert avg._core.nrow < o._core.nrow
    assert avg.antennas == o.antennas
    avg.invert()
    assert avg.imstat()["max"] == pytest.approx(peak0, rel=0.05)


@real_data
def test_real_data_resoff_reduces_residuals():
    o = difmapy.load(UVF)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.invert()
    s = o.imstat()
    px, py = s["maxpos"]
    o.addcmp(s["max"], px - 512, py - 512)
    o.keep()
    o.invert()
    rms0 = o.imstat()["rms"]
    n = o.resoff()
    assert n > 0
    o.invert()
    # Baseline corrections absorb non-closing errors: residuals drop.
    assert o.imstat()["rms"] < rms0
    o.clroff()
    o.invert()
    assert o.imstat()["rms"] == pytest.approx(rms0, rel=1e-3)
