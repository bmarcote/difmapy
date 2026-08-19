"""modelfit tests using the real EVN uv coverage with injected models,
so the truth is known exactly while the sampling/flagging is realistic.
"""

import os

import numpy as np
import pytest

import difmapy
from difmapy._core import CoreObservation

MAS = difmapy.MAS
HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")

pytestmark = pytest.mark.skipif(
    not os.path.exists(UVF), reason="real 3C345 data not present"
)

TRUTH = dict(flux=1.8, x=2.5, y=-1.2, major=4.0, ratio=0.6, phi=35.0)  # mas/deg
POINT = dict(flux=0.6, x=-8.0, y=3.0)


@pytest.fixture(scope="module")
def template():
    """Metadata of the real observation, for building synthetic clones."""
    o = difmapy.load(UVF)
    c = o._core
    t, a1, a2, us, vs, ws = c.rows()
    _, wt = c.calibrated_cube()
    return {
        "core": c,
        "t": t,
        "a1": a1,
        "a2": a2,
        "uvw": np.column_stack([us, vs, ws]),
        "wt": np.abs(np.asarray(wt)),
        "flags": np.asarray(c.flags()),
        "chan_f": np.concatenate(
            [[f + i * df for i in range(n)] for (f, df, n) in c.ifs]
        ),
        "u": us,
        "v": vs,
    }


def _model_vis(tpl, gauss=None, point=None):
    uu = tpl["u"][:, None] * tpl["chan_f"]
    vv = tpl["v"][:, None] * tpl["chan_f"]
    V = np.zeros(uu.shape, dtype=np.complex128)
    if gauss:
        phi = np.deg2rad(gauss["phi"])
        sp, cp = np.sin(phi), np.cos(phi)
        ta = (uu * cp - vv * sp) * gauss["ratio"]
        tb = uu * sp + vv * cp
        r = np.pi * gauss["major"] * MAS * np.sqrt(ta**2 + tb**2)
        V += (
            gauss["flux"]
            * np.exp(-0.3606737602 * r**2)
            * np.exp(2j * np.pi * (uu * gauss["x"] * MAS + vv * gauss["y"] * MAS))
        )
    if point:
        V += point["flux"] * np.exp(
            2j * np.pi * (uu * point["x"] * MAS + vv * point["y"] * MAS)
        )
    return V


def make_obs(tpl, gauss=None, point=None):
    c = tpl["core"]
    V = _model_vis(tpl, gauss, point)
    vis = np.repeat(V[:, :, None], c.npol, axis=2).astype(np.complex64)
    core = CoreObservation(
        "SYNTH",
        c.ra,
        c.dec,
        2000.0,
        c.antenna_names,
        np.asarray(c.antenna_xyz),
        list(c.antenna_subarrays),
        [f for (f, _, _) in c.ifs],
        [d for (_, d, _) in c.ifs],
        [n for (_, _, n) in c.ifs],
        list(c.pols),
        tpl["t"],
        c.inttimes(),
        tpl["a1"],
        tpl["a2"],
        tpl["uvw"],
        np.ascontiguousarray(vis),
        tpl["wt"],
        c.ref_mjd,
        flag=tpl["flags"],
    )
    o = difmapy.Observation(core)
    o.select("I")
    return o


def test_chisq_zero_at_truth(template):
    o = make_obs(template, gauss=TRUTH)
    o.addcmp(TRUTH["flux"], TRUTH["x"], TRUTH["y"], type="gauss",
             major=TRUTH["major"], ratio=TRUTH["ratio"], phi=TRUTH["phi"],
             free=["flux", "pos", "shape"])
    res = o.modelfit(niter=0)
    assert res["rchisq"] < 1e-8
    assert res["nfree"] == 6  # flux, x, y, X, Y, Z
    assert res["ndfree"] == 2 * res["nvis"] - 6
    # A zero-iteration fit leaves the parameters untouched.
    m = o.model[0]
    assert m["flux"] == pytest.approx(TRUTH["flux"])
    assert m["major"] == pytest.approx(TRUTH["major"], rel=1e-5)


def test_recovers_elliptical_gaussian(template):
    o = make_obs(template, gauss=TRUTH)
    o.addcmp(1.5, 2.0, -1.0, type="gauss", major=3.5, ratio=0.65, phi=30.0,
             free=["flux", "pos", "shape"])
    res = o.modelfit(niter=50)
    assert res["rchisq"] < 1e-8, f"rchisq={res['rchisq']}"
    m = o.model[0]
    assert m["flux"] == pytest.approx(TRUTH["flux"], rel=1e-3)
    assert m["x"] == pytest.approx(TRUTH["x"], abs=1e-3)
    assert m["y"] == pytest.approx(TRUTH["y"], abs=1e-3)
    assert m["major"] == pytest.approx(TRUTH["major"], rel=1e-3)
    assert m["ratio"] == pytest.approx(TRUTH["ratio"], abs=1e-3)
    assert m["phi"] == pytest.approx(TRUTH["phi"], abs=0.1)
    # Formal errors are finite and small for noiseless data.
    e = res["errors"][0]
    assert 0 < e["flux"] < 0.01
    assert 0 < e["x"] < 0.01 and 0 < e["y"] < 0.01


@pytest.mark.parametrize(
    "free,keys",
    [
        (["flux"], ["flux"]),
        (["pos"], ["x", "y"]),
        (["shape"], ["major", "ratio", "phi"]),
    ],
)
def test_partial_parameter_fits(template, free, keys):
    """Fitting a subset of parameters recovers them and leaves the
    others untouched."""
    o = make_obs(template, gauss=TRUTH)
    start = dict(TRUTH)
    for k in keys:
        start[k] = {"flux": 1.0, "x": 2.0, "y": -1.0, "major": 3.5,
                    "ratio": 0.7, "phi": 28.0}[k]
    o.addcmp(start["flux"], start["x"], start["y"], type="gauss",
             major=start["major"], ratio=start["ratio"], phi=start["phi"],
             free=free)
    res = o.modelfit(niter=50)
    assert res["rchisq"] < 1e-8
    m = o.model[0]
    for k in keys:
        tol = 0.1 if k == "phi" else max(abs(TRUTH[k]) * 2e-3, 1e-3)
        assert m[k] == pytest.approx(TRUTH[k], abs=tol), f"{k}={m[k]}"
    assert res["nfree"] == {"flux": 1, "x": 2, "major": 3}[keys[0]]


def test_two_components(template):
    o = make_obs(template, gauss=TRUTH, point=POINT)
    o.addcmp(1.6, 2.2, -1.0, type="gauss", major=3.8, ratio=0.62, phi=33.0,
             free=["flux", "pos", "shape"])
    o.addcmp(0.5, -7.5, 2.7, free=["flux", "pos"])
    res = o.modelfit(niter=100)
    assert res["nfree"] == 9
    assert res["rchisq"] < 1e-6, f"rchisq={res['rchisq']}"
    g, p = o.model[0], o.model[1]
    assert g["flux"] == pytest.approx(TRUTH["flux"], rel=5e-3)
    assert p["flux"] == pytest.approx(POINT["flux"], rel=5e-3)
    assert p["x"] == pytest.approx(POINT["x"], abs=0.01)
    assert p["y"] == pytest.approx(POINT["y"], abs=0.01)


def test_fit_reduces_residuals_on_real_data():
    """On the real (complex) source, fitting must reduce the quantity it
    minimises - chi-squared over the visibilities - even though a single
    component is a poor model of 3C345.

    (The residual *map* rms is not used here: this snapshot's dirty map
    has near-unity sidelobes, so map statistics are dominated by them.)
    """
    o = difmapy.load(UVF)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.invert()
    (x, y), peak = o.peak_offset()
    o.addcmp(peak, x, y, free=["flux", "pos"])
    start = o.modelfit(niter=0)
    res = o.modelfit(niter=30)
    assert res["nvis"] > 10000
    assert res["rchisq"] < start["rchisq"]
    # The fitted component stays in the neighbourhood of the peak.
    c = o.model[0]
    assert abs(c["x"] - x) < 50 and abs(c["y"] - y) < 50


def test_modelfit_errors(template):
    o = make_obs(template, gauss=TRUTH)
    with pytest.raises(RuntimeError, match="no tentative model"):
        o.modelfit()
    o.addcmp(1.0, 0.0, 0.0)  # no free parameters
    with pytest.raises(ValueError, match="no free parameters"):
        o.modelfit()
    with pytest.raises(ValueError, match="unknown free parameter"):
        o.modelfit(free=["nonsense"])
    # uvrange restricts the data used.
    all_vis = o.modelfit(niter=1, free=["flux"])["nvis"]
    cut = o.modelfit(niter=1, free=["flux"], uvmin=2e7, uvmax=1e9)["nvis"]
    assert 0 < cut < all_vis
