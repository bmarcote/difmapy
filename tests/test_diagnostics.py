"""Closure phases, spectra, sampling and the diagnostic plots."""

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""

import difmapy

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
real_data = pytest.mark.skipif(
    not os.path.exists(UVF), reason="real 3C345 data not present"
)


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    return o


def test_closure_phase_zero_for_point_source(obs):
    """A point source at the phase center has zero closure phase, and a
    point source anywhere has zero closure phase too (the phases of a
    point source close exactly)."""
    cps = obs.closure_phases()
    assert cps, "no closure triangles found"
    worst = max(np.abs(np.asarray(c["phase"])).max() for c in cps)
    assert np.rad2deg(worst) < 1e-3, f"closure phase = {np.rad2deg(worst)} deg"
    # Errors are finite and positive.
    assert all((np.asarray(c["error"]) > 0).all() for c in cps)


def test_closure_phase_triangle_count(obs):
    n = len(obs.antennas)
    ntri = n * (n - 1) * (n - 2) // 6
    cps = obs.closure_phases()
    assert len(cps) == ntri * obs.nif  # one series per triangle and IF
    single = obs.closure_phases(triangle=("AN0", "AN1", "AN2"))
    assert len(single) == obs.nif
    assert tuple(single[0]["triangle"]) == (0, 1, 2)
    one_if = obs.closure_phases(triangle=("AN0", "AN1", "AN2"), if_index=1)
    assert len(one_if) == 1 and one_if[0]["if_index"] == 1


def test_closure_phase_matches_model(obs):
    """With a model established, the model closure phases must agree
    with the observed ones for a perfect model."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    obs.addcmp(FLUX, X0_MAS, Y0_MAS)
    obs.keep()
    cps = obs.closure_phases(triangle=("AN0", "AN1", "AN2"), if_index=0)
    c = cps[0]
    model = np.asarray(c["model"])
    assert np.isfinite(model).all()
    d = np.angle(np.exp(1j * (model - np.asarray(c["phase"]))))
    assert np.rad2deg(np.abs(d)).max() < 1e-3


def test_flagging_removes_closure_points(obs):
    n0 = sum(len(c["time"]) for c in obs.closure_phases())
    obs.flag(station="AN2")
    n1 = sum(len(c["time"]) for c in obs.closure_phases())
    assert n1 < n0
    obs.unflag(station="AN2")
    assert sum(len(c["time"]) for c in obs.closure_phases()) == n0


def test_spectrum(obs):
    from conftest import FLUX, NCHAN, NIF

    s = obs.spectrum()
    assert len(s["chan"]) == NIF * NCHAN
    assert (np.asarray(s["wt"]) > 0).all()
    # The synthetic source is flat-spectrum with |V| = FLUX.
    assert np.allclose(s["amp"], FLUX, rtol=1e-4)
    # Frequencies increase within each IF.
    f = np.asarray(s["freq"])
    assert (np.diff(f[:NCHAN]) > 0).all()
    # Per-baseline spectra work too.
    s1 = obs.spectrum(baseline=("AN0", "AN1"))
    assert np.allclose(s1["amp"], FLUX, rtol=1e-4)
    # Time selection reduces the weight sum.
    t = obs._core.rows()[0]
    half = obs.spectrum(tmax=float(np.median(t)))
    assert np.asarray(half["wt"]).sum() < np.asarray(s["wt"]).sum()


def test_sampling(obs):
    samp = np.asarray(obs._core.sampling())
    assert samp.shape == (obs._core.ntimes, len(obs.antennas))
    assert (samp > 0).all()  # nothing flagged initially
    obs.flag(station="AN0")
    samp = np.asarray(obs._core.sampling())
    assert (samp[:, 0] == 0).all()
    assert (samp[:, 1] > 0).all()


def test_diagnostic_plots_construct(obs):
    from difmapy.plots.diagnostics import CorPlot, CpPlot, SpecPlot, TPlot

    obs.mapsize(256, 0.25)
    obs.invert()
    obs.addcmp(2.0, 0.0, 0.0)
    obs.keep()
    obs.selfcal(phase=True)

    p = CpPlot(obs, nplot=3)
    assert p.npages >= 1
    p.page = 1
    p.refresh()
    p.close()

    p = TPlot(obs)
    p.close()

    p = CorPlot(obs, quantity="phase")
    p.refresh()
    p.close()
    p = CorPlot(obs, quantity="amp")
    p.close()

    p = SpecPlot(obs)
    p.close()
    p = SpecPlot(obs, baseline=("AN0", "AN1"), xaxis="chan")
    p.close()


@real_data
def test_closure_invariant_under_selfcal():
    """Closure phases are unaffected by antenna-based gain errors: the
    definitive check of both the closure code and self-cal."""
    o = difmapy.load(UVF)
    o.select("I")
    before = {
        (tuple(c["triangle"]), c["if_index"]): np.asarray(c["phase"]).copy()
        for c in o.closure_phases()
    }
    o.mapsize(1024, 1.0)
    o.invert()
    s = o.imstat()
    px, py = s["maxpos"]
    o.addcmp(s["max"], px - 512, py - 512)
    o.keep()
    o.selfcal(amp=True, phase=True)

    worst = 0.0
    for c in o.closure_phases():
        key = (tuple(c["triangle"]), c["if_index"])
        if key in before and len(before[key]) == len(c["phase"]):
            d = np.angle(np.exp(1j * (np.asarray(c["phase"]) - before[key])))
            worst = max(worst, np.abs(d).max())
    assert np.rad2deg(worst) < 1e-3, f"closure changed by {np.rad2deg(worst)} deg"


@real_data
def test_real_data_closure_and_spectrum():
    o = difmapy.load(UVF)
    o.select("I")
    cps = o.closure_phases()
    n = len(o.antennas)
    assert len(cps) <= n * (n - 1) * (n - 2) // 6 * o.nif
    # Real data has genuine structure: closure phases are not all zero.
    allphs = np.concatenate([np.asarray(c["phase"]) for c in cps])
    assert allphs.size > 1000
    assert np.rad2deg(np.std(allphs)) > 1.0

    s = o.spectrum()
    assert len(s["chan"]) == 4  # 4 IFs x 1 channel
    assert (np.asarray(s["wt"]) > 0).all()
    assert (np.asarray(s["amp"]) > 0).all()
