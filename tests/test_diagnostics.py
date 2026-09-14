"""Closure phases, spectra, sampling and the diagnostic plots."""

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""

import pyqtgraph as pg

import difmapy

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
real_data = pytest.mark.skipif(
    not os.path.exists(UVF), reason="real 3C345 data not present"
)


def _key(text):
    """A key press event carrying `text`."""
    from pyqtgraph.Qt import QtCore, QtGui

    return QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, 0,
                           QtCore.Qt.KeyboardModifier.NoModifier, text)


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


def test_spectrum_follows_the_calibration(corrupted_uvfits_file):
    """A time-averaged spectrum has to reflect the accumulated gains -
    it read the raw cube before - and must agree with the calibrated
    stream it is averaging."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    path, _ = corrupted_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    raw = np.asarray(o.spectrum(baseline=("AN0", "AN1"))["amp"]).copy()
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.selfcal(amp=True, phase=True, quiet=True)

    after = np.asarray(o.spectrum(baseline=("AN0", "AN1"))["amp"])
    assert not np.allclose(raw, after)
    # calibrated=False still shows the data as loaded.
    assert np.allclose(
        o.spectrum(baseline=("AN0", "AN1"), calibrated=False)["amp"], raw
    )

    # The vector average per IF must match the calibrated stream's own.
    vis, wt = o._core.stream_vis()
    _, a1, a2, *_ = o._core.rows()
    m = (np.asarray(a1) == 0) & (np.asarray(a2) == 1)
    vis, wt = np.asarray(vis)[m], np.asarray(wt)[m]
    s = o.spectrum(baseline=("AN0", "AN1"))
    nchan = o.nchan[0]
    for cif in range(o.nif):
        good = wt[:, cif] > 0
        stream = (np.sum(vis[good, cif] * wt[good, cif])
                  / np.sum(wt[good, cif]))
        chans = slice(cif * nchan, (cif + 1) * nchan)
        spec = np.mean(np.asarray(s["re"])[chans]
                       + 1j * np.asarray(s["im"])[chans])
        assert spec == pytest.approx(stream, rel=2e-3, abs=2e-3)


def test_corplot_shows_amplitude_and_phase_by_default(obs):
    """One antenna to a pair of panels, amplitude above phase, sharing
    the time axis - the same layout as vplot's baselines."""
    from difmapy.plots.diagnostics import CorPlot
    from conftest import FLUX, X0_MAS, Y0_MAS

    obs.mapsize(256, 0.25)
    obs.addcmp(FLUX, X0_MAS, Y0_MAS)
    obs.selfcal(amp=True, phase=True, quiet=True)

    p = CorPlot(obs, nplot=2)
    assert p.quantity == "both"
    assert len(p._plots) == 4                      # 2 antennas x 2 panels
    labels = [q.getAxis("left").labelText for q, _ in p._plots]
    assert "Amplitude" in labels[0] and "Phase" in labels[1]
    assert labels[0].split("<br>")[0] == labels[1].split("<br>")[0]

    # "z" brings a phase panel back to +-180, and an amplitude panel to
    # the range it was given - a single gain solution is constant to the
    # last bit of float32, and autoscaling to that shows only noise.
    assert p.default_y_range(p._plots[1][0].vb) == (-180, 180)
    amp_range = p.default_y_range(p._plots[0][0].vb)
    assert amp_range is not None and amp_range[1] - amp_range[0] >= 1e-3

    # One shared x axis: linked, labelled only at the bottom.
    first = p._plots[0][0]
    for q, _ in p._plots[1:]:
        assert q.vb.linkedView(q.vb.XAxis) is first.vb
    assert not any(q.getAxis("bottom").style["showValues"]
                   for q, _ in p._plots[:-1])
    assert p._plots[-1][0].getAxis("bottom").style["showValues"]
    p.close()

    # A single quantity still gives one panel per antenna.
    for quantity, key in (("phase", "Phase"), ("amp", "Amplitude")):
        q = CorPlot(obs, quantity=quantity, nplot=2)
        assert q.quantity == quantity
        assert len(q._plots) == 2
        assert key in q._plots[0][0].getAxis("left").labelText
        q.close()
    with pytest.raises(ValueError, match="unknown quantity"):
        CorPlot(obs, quantity="nonsense")


def test_corplot_leaves_out_integrations_without_a_solution(
    corrupted_uvfits_file,
):
    """A station with no solution reads 1.0 / 0 deg in the gain table.
    That is not a correction, and drawing it made a smooth run of
    interpolated corrections look like it jumped to unity and back."""
    from conftest import FLUX, X0_MAS, Y0_MAS
    from difmapy.plots.diagnostics import CorPlot

    path, _ = corrupted_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    # AN0 drops out for long enough that the interpolation cannot reach
    # across the gap, leaving integrations with no solution at all.
    t = np.asarray(o._core.times())
    o.flag(station="AN0", tmin=float(t[20]), tmax=float(t[39]))
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.selfcal(phase=True, amp=True, solint="5min", quiet=True)

    nt, nif, nant = o._core.ntimes, o.nif, len(o.antennas)
    used = np.asarray(o._core.gains_used()).reshape(nt, nif, nant)
    assert not used[:, 0, 0].all(), "expected some unsolved integrations"

    p = CorPlot(o, quantity="amp", nplot=1)
    assert p._items[0][1] == "AN0"
    curves = [it for it in p._plots[0][0].items
              if isinstance(it, pg.PlotDataItem)]
    assert len(curves) == nif                    # one per IF
    for cif, curve in enumerate(curves):
        x, y = curve.getData()
        assert np.allclose(x, t / 3600.0)
        solved = used[:, cif, 0]
        # The unsolved integrations are left out of the curve (drawn as
        # gaps, with connect="finite"), and no drawn point sits at the
        # "no correction" value of exactly 1.0.
        assert np.isfinite(y[solved]).all()
        assert np.isnan(y[~solved]).all()
        assert not np.any(y[solved] == 1.0)
    assert "without a solution" in p.statusBar().currentMessage()
    p.close()


def test_fplot_pages_baselines_against_frequency(obs):
    """fplot is vplot's frequency counterpart: amplitude and phase per
    baseline, averaged over time, on one shared frequency axis."""
    from conftest import FLUX, NCHAN, NIF
    from difmapy.plots.diagnostics import FPlot

    p = FPlot(obs, nplot=2)
    nant = len(obs.antennas)
    assert len(p._items) == nant * (nant - 1) // 2
    assert p.npages == len(p._items) // 2
    assert len(p._panels) == 4  # two baselines x (amplitude, phase)

    # Only the bottom panel keeps its tick labels; all are linked.
    assert not any(q.getAxis("bottom").style["showValues"] for q in p._panels[:-1])
    assert p._panels[-1].getAxis("bottom").style["showValues"]
    for q in p._panels[1:]:
        assert q.vb.linkedView(q.vb.XAxis) is p._panels[0].vb

    # Every channel of every IF is drawn, at its own frequency.
    amp_curves = [it for it in p._panels[0].items
                  if isinstance(it, pg.PlotDataItem)]
    assert len(p._plots) == len(p._panels)
    assert len(amp_curves) == NIF
    x, y = amp_curves[0].getData()
    assert len(x) == NCHAN
    assert np.allclose(y, FLUX, rtol=1e-4)   # flat-spectrum test source
    assert (np.diff(x) > 0).all()            # frequency increases, in GHz
    assert 4.8 < x[0] < 5.1

    first = [q.getAxis("left").labelText for q in p._panels]
    p.keyPressEvent(_key("n"))
    assert p.page == 1
    assert [q.getAxis("left").labelText for q in p._panels] != first
    p.close()


def test_fplot_amplitude_range_has_a_floor(obs):
    """A flat band is flat to the last bit of float32; autoscaling to
    that shows only rounding noise, so the amplitude panel never ranges
    tighter than a thousandth of the level."""
    from difmapy.plots.diagnostics import FPlot
    from conftest import FLUX

    p = FPlot(obs, nplot=1)
    lo, hi = p._panels[0].vb.viewRange()[1]
    assert hi - lo >= 1e-3 * FLUX
    assert lo < FLUX < hi
    p.close()


def test_fplot_restricts_to_one_station_and_time_range(obs):
    from difmapy.plots.diagnostics import FPlot

    p = FPlot(obs, reftel="AN1", nplot=3)
    names = obs.antennas
    assert all(names.index("AN1") in bl for bl in p._items)
    assert len(p._items) == len(names) - 1
    assert "AN1" in p.windowTitle()
    p.close()

    t = np.asarray(obs._core.rows()[0])
    half = FPlot(obs, nplot=1, tmax=float(np.median(t)))
    assert "averaged over" in half.statusBar().currentMessage()
    half.close()


def test_sampling(obs):
    samp = np.asarray(obs._core.sampling())
    assert samp.shape == (obs._core.ntimes, len(obs.antennas))
    assert (samp > 0).all()  # nothing flagged initially
    obs.flag(station="AN0")
    samp = np.asarray(obs._core.sampling())
    assert (samp[:, 0] == 0).all()
    assert (samp[:, 1] > 0).all()


def test_diagnostic_plots_construct(obs):
    from difmapy.plots.diagnostics import CorPlot, CpPlot, FPlot, SpecPlot, TPlot

    obs.mapsize(256, 0.25)
    obs.invert()
    obs.addcmp(2.0, 0.0, 0.0)
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

    p = FPlot(obs, nplot=2)
    p.refresh()
    p.close()
    p = FPlot(obs, baselines=[("AN0", "AN1")], calibrated=False)
    assert len(p._items) == 1
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
