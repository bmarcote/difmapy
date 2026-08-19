"""Regression tests for the calibration/geometry write paths.

Everything that modifies the data the user sees (antenna gains,
baseline corrections, phase-center shifts) must be reflected in what
gets written out or averaged - these were silently dropped once.
"""

import os

import numpy as np
import pytest

import difmapy

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
real_data = pytest.mark.skipif(
    not os.path.exists(UVF), reason="real 3C345 data not present"
)


def _matched(a, b):
    """Match two observations row-by-row on (time, ant1, ant2).

    Times are matched to the nearest second: writing them through the
    UVFITS DATE parameters (a Julian Date around 2.46e6) loses the
    sub-microsecond digits, and the EVN timestamps sit just below
    integer seconds, so finer rounding would split matching rows.
    """
    t1, a11, a21, *_ = a._core.rows()
    t2, a12, a22, *_ = b._core.rows()
    k1 = {(int(round(float(t))), x, y): i for i, (t, x, y) in enumerate(zip(t1, a11, a21))}
    k2 = {(int(round(float(t))), x, y): i for i, (t, x, y) in enumerate(zip(t2, a12, a22))}
    common = sorted(set(k1) & set(k2))
    assert common
    return (
        np.array([k1[k] for k in common]),
        np.array([k2[k] for k in common]),
    )


@pytest.fixture()
def modelled(uvfits_file):
    """An observation with a model and non-trivial gains."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.keep()
    o.selfcal(phase=True)
    return o


def test_wobs_includes_gains(modelled, tmp_path):
    v1, w1 = modelled._core.stream_vis()
    out = str(tmp_path / "g.uvf")
    modelled.wobs(out)
    back = difmapy.load(out)
    back.select("I")
    i1, i2 = _matched(modelled, back)
    v2, w2 = back._core.stream_vis()
    g = (np.asarray(w1)[i1] > 0) & (np.asarray(w2)[i2] > 0)
    assert np.abs(v2[i2][g] - v1[i1][g]).max() < 1e-4


def test_wobs_includes_baseline_corrections(modelled, tmp_path):
    """resoff corrections must survive a write/read cycle."""
    modelled.resoff()
    v1, w1 = modelled._core.stream_vis()
    out = str(tmp_path / "r.uvf")
    modelled.wobs(out)
    back = difmapy.load(out)
    back.select("I")
    i1, i2 = _matched(modelled, back)
    v2, w2 = back._core.stream_vis()
    g = (np.asarray(w1)[i1] > 0) & (np.asarray(w2)[i2] > 0)
    assert np.abs(v2[i2][g] - v1[i1][g]).max() < 1e-4


def test_wobs_shift_handling(uvfits_file, tmp_path):
    """By default shifts are NOT frozen into the output (difmap
    behaviour); freeze_shift=True writes the shifted data."""
    o = difmapy.load(uvfits_file)
    o.select("I")
    v_unshifted, w = o._core.stream_vis()
    o.shift(3.0, -2.0)
    v_shifted, _ = o._core.stream_vis()
    assert np.abs(np.asarray(v_shifted) - np.asarray(v_unshifted)).max() > 1e-3

    plain = str(tmp_path / "plain.uvf")
    frozen = str(tmp_path / "frozen.uvf")
    o.wobs(plain)
    o.wobs(frozen, freeze_shift=True)

    for path, ref in ((plain, v_unshifted), (frozen, v_shifted)):
        back = difmapy.load(path)
        back.select("I")
        i1, i2 = _matched(o, back)
        vb, wb = back._core.stream_vis()
        g = (np.asarray(w)[i1] > 0) & (np.asarray(wb)[i2] > 0)
        assert np.abs(vb[i2][g] - np.asarray(ref)[i1][g]).max() < 1e-3, path


def test_uvaver_includes_shift(uvfits_file):
    """Averaging must average what the user sees, including shifts."""
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.shift(3.0, -2.0)
    o.invert()
    m = o.dmap
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    avg = o.uvaver(120.0)
    avg.invert()
    m2 = avg.dmap
    iy2, ix2 = np.unravel_index(np.argmax(m2), m2.shape)
    assert (ix2, iy2) == (ix, iy)
    # The averaged data are already corrected, so no shift is pending.
    assert avg.total_shift == (0.0, 0.0)


def test_shift_moves_windows(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.add_window(-2.0, 2.0, -1.0, 1.0)
    o.shift(5.0, 3.0)
    assert o.windows == [(3.0, 7.0, 2.0, 4.0)]
    o.unshift()
    assert o.windows == [(-2.0, 2.0, -1.0, 1.0)]


def test_save_get_restores_shift(uvfits_file, tmp_path):
    """A saved session must come back with the same shift, model and
    windows, and image identically."""
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.shift(3.0, -2.0)
    o.add_window(-3.0, 3.0, -3.0, 3.0)
    o.invert()
    o.clean(100, 0.05)
    o.keep()
    prefix = str(tmp_path / "sess")
    o.save(prefix)

    back = difmapy.Observation.get(prefix)
    assert back.total_shift == pytest.approx((3.0, -2.0))
    assert back.windows == o.windows
    assert back.model_flux == pytest.approx(o.model_flux, rel=1e-4)
    # Model component positions are in the shifted frame in both.
    m1 = max(o.model, key=lambda c: abs(c["flux"]))
    m2 = max(back.model, key=lambda c: abs(c["flux"]))
    assert (m2["x"], m2["y"]) == pytest.approx((m1["x"], m1["y"]), abs=1e-3)
    o.invert()
    back.invert()
    assert np.abs(o.dmap - back.dmap).max() < 1e-3 * np.abs(o.dmap).max()


def test_modelfit_free_mask_binds_to_right_component(uvfits_file):
    """The free-parameter mask must follow its own component, even when
    CLEAN adds components before it."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.invert()
    o.clean(20, 0.05)  # adds tentative component(s) first
    _, tentative = o._core.get_models()
    n_clean = len(tentative)
    assert n_clean >= 1

    # Now add the component to be fitted, deliberately displaced.
    o.addcmp(0.5 * FLUX, X0_MAS + 3.0, Y0_MAS - 3.0, free=["flux", "pos"])
    masks = list(o._core.tentative_freepars)
    assert masks[:n_clean] == [0] * n_clean  # clean components are fixed
    assert masks[n_clean] == 3  # flux | pos

    before = [dict(c) for c in o.model if c["tentative"]]
    res = o.modelfit(niter=30)
    assert res["nfree"] == 3
    after = [dict(c) for c in o.model if c["tentative"]]
    # The CLEAN components must be untouched...
    for a, b in zip(before[:n_clean], after[:n_clean]):
        assert a["flux"] == b["flux"]
        assert (a["x"], a["y"]) == (b["x"], b["y"])
    # ...and only the last component may have moved.
    assert (after[n_clean]["x"], after[n_clean]["y"]) != (
        before[n_clean]["x"],
        before[n_clean]["y"],
    )


@real_data
def test_real_data_wobs_after_full_session(tmp_path):
    """After a full session (selfcal + resoff + shift), the written
    file must reproduce the corrected data."""
    o = difmapy.load(UVF)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.startmod(flux=1.0)
    o.clean(200, 0.03)
    o.selfcal(phase=True)
    o.clean(200, 0.03)
    o.resoff()
    v1, w1 = o._core.stream_vis()

    out = str(tmp_path / "session.uvf")
    o.wobs(out)
    back = difmapy.load(out)
    back.select("I")
    i1, i2 = _matched(o, back)
    v2, w2 = back._core.stream_vis()
    g = (np.asarray(w1)[i1] > 0) & (np.asarray(w2)[i2] > 0)
    assert g.sum() > 10000
    rel = np.abs(v2[i2][g] - v1[i1][g]).max() / np.median(np.abs(v1[i1][g]))
    assert rel < 1e-4, f"relative deviation {rel}"
