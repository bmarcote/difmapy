"""Cross-validation on real EVN data (3C345): the same observation in
UVFITS and MS format must load identically and image identically."""

import os

import numpy as np
import pytest

import difmapy

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
MS = os.path.join(HERE, "rsm07_3C345.ms")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(UVF) and os.path.isdir(MS)),
    reason="real 3C345 test data not present",
)


@pytest.fixture(scope="module")
def pair():
    ou = difmapy.load(UVF)
    om = difmapy.load(MS)
    ou.select("I")
    om.select("I")
    return ou, om


def test_headers_match(pair):
    ou, om = pair
    assert ou.source == om.source == "3C345"
    assert ou.antennas == om.antennas
    assert ou.nif == om.nif == 4
    assert ou._core.ifs == om._core.ifs
    # Cross-correlations only, same row count.
    assert ou._core.nrow == om._core.nrow == 19800


def test_data_identical(pair):
    ou, om = pair
    tu, a1u, a2u, uu, vu, wu = ou._core.rows()
    tm, a1m, a2m, um, vm, wm = om._core.rows()
    # Constant time offset between the two conventions (< 1 int time).
    dt = np.median(tm[:66]) - np.median(tu[:66])
    assert abs(dt) <= 1.0

    ku = {(int(round(t + dt)), a, b): i for i, (t, a, b) in enumerate(zip(tu, a1u, a2u))}
    km = {(int(round(t)), a, b): i for i, (t, a, b) in enumerate(zip(tm, a1m, a2m))}
    common = sorted(set(ku) & set(km))
    assert len(common) == 19800
    iu = np.array([ku[k] for k in common], dtype=int)
    im = np.array([km[k] for k in common], dtype=int)

    duvw = np.abs(
        np.column_stack([uu, vu, wu])[iu] - np.column_stack([um, vm, wm])[im]
    )
    assert duvw.max() < 1e-8  # uvw in seconds; sub-nanosecond agreement

    vu_, wtu = ou._core.stream_vis()
    vm_, wtm = om._core.stream_vis()
    gu, gm = wtu[iu] > 0, wtm[im] > 0
    assert (gu == gm).all()  # identical flag states
    both = gu & gm
    dv = np.abs(vu_[iu][both] - vm_[im][both])
    av = np.abs(vm_[im][both])
    assert np.percentile(dv / av, 99) < 1e-5  # identical visibilities
    assert np.allclose(wtu[iu][both], wtm[im][both], rtol=1e-4)


def test_ms_to_uvfits_conversion(tmp_path):
    """Writing MS-loaded data as UVFITS must reorder the CASA
    correlation axis (RR, RL, LR, LL) into a valid FITS STOKES axis
    without altering any visibility."""
    src = difmapy.load(MS)
    assert src._core.pols == [-1, -3, -4, -2]  # CASA order
    out = str(tmp_path / "from_ms.uvf")
    src.wobs(out)
    dst = difmapy.load(out)
    assert dst._core.pols == [-1, -2, -3, -4]  # regular FITS axis

    t1, a11, a21, *_ = src._core.rows()
    t2, a12, a22, *_ = dst._core.rows()
    k1 = {(int(round(t)), a, b): i for i, (t, a, b) in enumerate(zip(t1, a11, a21))}
    k2 = {(int(round(t)), a, b): i for i, (t, a, b) in enumerate(zip(t2, a12, a22))}
    common = sorted(set(k1) & set(k2))
    assert len(common) == src._core.nrow
    i1 = np.array([k1[k] for k in common])
    i2 = np.array([k2[k] for k in common])

    for pol in ("RR", "LL", "RL", "LR"):
        a = difmapy.load(MS)
        a.select(pol)
        b = difmapy.load(out)
        b.select(pol)
        va, wa = a._core.stream_vis()
        vb, wb = b._core.stream_vis()
        good = wa[i1] > 0
        assert np.array_equal(wa[i1] > 0, wb[i2] > 0)
        assert np.abs(va[i1][good] - vb[i2][good]).max() < 1e-6


@pytest.mark.parametrize("path", [UVF, MS])
def test_imaging_loop(path):
    o = difmapy.load(path)
    o.select("I")
    o.mapsize(1024, 1.0)
    # Natural weighting, so the beam size below is well defined (the
    # default uniform weighting sharpens it to ~8 mas).
    o.uvweight(0, -1)
    r = o.invert()
    bmaj, bmin, _ = o.estimated_beam
    assert 10 < bmaj < 30 and 3 < bmin < 12  # EVN L-band beam (mas)
    assert r["nused"] == 14548  # unflagged Stokes-I visibilities

    for _ in range(5):
        o.clean(100, 0.03)
        o.selfcal(phase=True)
    o.clean(200, 0.03)
    o.selfcal(amp=True, phase=True, solint=30)
    o.clean(200, 0.03)

    # 3C345 at 1.6 GHz: a few Jy of correlated flux.
    assert 3.0 < o.model_flux < 5.0
    stats = o.imstat()
    assert stats["rms"] < 0.02  # residual rms below 20 mJy/beam
    # Restored peak is a sane fraction of the model flux.
    cln = o.restore()
    assert 1.0 < cln.max() < 5.0


def test_full_session_reproduces_closure_phases():
    """A full startmod/clean/selfcal session must end with a model that
    reproduces the observed closure phases (which self-cal cannot
    change), to within the noise on the good data."""
    o = difmapy.load(UVF)
    o.select("I")
    o.mapsize(2048, 0.5)
    o.startmod(flux=1.0)
    for _ in range(4):
        o.clean(200, 0.03)
        o.selfcal(phase=True)
    o.clean(400, 0.02)
    o.selfcal(amp=True, phase=True, solint=30)
    o.clean(400, 0.02)
    assert o.imstat()["rms"] < 0.01  # < 10 mJy/beam residuals

    cps = [c for c in o.closure_phases() if np.isfinite(c["model"]).all()]
    resid = np.concatenate(
        [
            np.angle(np.exp(1j * (np.asarray(c["phase"]) - np.asarray(c["model"]))))
            for c in cps
        ]
    )
    err = np.concatenate([np.asarray(c["error"]) for c in cps])
    # Normalized residuals of order unity, and the high-SNR quartile
    # agrees to a few degrees.
    assert np.std(resid / err) < 5.0
    best = err < np.percentile(err, 25)
    assert np.rad2deg(np.std(resid[best])) < 10.0
