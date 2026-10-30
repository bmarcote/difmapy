"""The dashboard: its panels, keeping them in step, and its controls."""

import os
import sys

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""

pg = pytest.importorskip("pyqtgraph")

import difmapy
from difmapy.plots.dashboard import Dashboard
from difmapy.plots.diagnostics import CorPlot, PossmPlot, TPlot
from difmapy.plots.mapplot import MapPlot
from difmapy.plots.points import RadPlot, VPlot


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.mapsize(256, 0.25)
    return o


@pytest.fixture()
def dash(obs):
    d = Dashboard(obs)
    yield d
    d.close()


def test_panels_hold_the_plots_in_tabs(dash):
    """vplot/tplot/possm top left, radplot/projplot/corplot top right,
    mapplot bottom left, Main/Output bottom right; each plot is built
    when its tab is first shown, and embedded rather than a window."""
    assert [dash.time_panel.tabText(i) for i in range(3)] == ["vplot", "tplot", "possm"]
    assert [dash.uv_panel.tabText(i) for i in range(3)] == ["radplot", "projplot", "corplot"]
    assert [dash.tabs.tabText(i) for i in range(2)] == ["Main", "Output"]
    built = dash.plots()
    assert [type(p) for p in built] == [VPlot, RadPlot, MapPlot]
    for p in built:
        assert p.embedded and p.parent() is not None and not p.isWindow()
    dash.time_panel.setCurrentIndex(1)
    dash.time_panel.setCurrentIndex(2)
    dash.uv_panel.setCurrentIndex(1)
    kinds = {type(p) for p in dash.plots()}
    assert {TPlot, PossmPlot} <= kinds
    assert dash.uv_panel.current_plot().projection == 0.0
    assert "synth" in dash.windowTitle()


def test_defaults_of_the_controls(dash):
    assert dash.phase_check.isChecked() and not dash.amp_check.isChecked()
    assert dash.float_check.isChecked()
    assert {k: cb.isChecked() for k, cb in dash.model_checks.items()} == {
        "clean": False, "gauss1": True, "gauss2": True, "gauss3": True}


def test_a_flag_edit_reloads_the_other_panels(dash, obs):
    """Flagging in vplot reloads the plots on show elsewhere at once, and
    the ones in hidden tabs when they are next shown."""
    dash.time_panel.setCurrentIndex(1)            # build tplot ...
    dash.time_panel.setCurrentIndex(0)            # ... and hide it again
    vplot = dash.time_panel.current_plot()
    radplot = dash.uv_panel.current_plot()
    mapplot = dash.map_panel.current_plot()
    tplot = dash.time_panel.plots[1]
    revs = {id(p): p._revision for p in (vplot, radplot, mapplot, tplot)}
    good = int((radplot._data["wt"] > 0).sum())
    d = vplot._data
    idx = np.nonzero((d["group"] == 0) & (d["wt"] > 0))[0][:5]
    vplot._edit_points(idx, True)
    assert int((radplot._data["wt"] > 0).sum()) < good
    assert radplot._revision > revs[id(radplot)]
    assert mapplot._revision > revs[id(mapplot)]
    assert tplot._revision == revs[id(tplot)]    # hidden: not yet
    assert 1 in dash.time_panel.stale
    dash.time_panel.setCurrentIndex(1)
    assert tplot._revision > revs[id(tplot)]


def test_clean_in_the_map_updates_the_model_elsewhere(dash, obs):
    radplot = dash.uv_panel.current_plot()
    assert not radplot.has_model()
    obs.add_window(-10, 10, -10, 10)
    dash.map_panel.current_plot().run_clean()
    assert radplot.has_model()


def test_calibration_buttons(dash, obs):
    """Selfcal, Gscale and Bayes refuse to run without a model (it would
    poison the gains), and run with what the rows say once there is one;
    everything printed shows in the Output tab."""
    dash.run_selfcal()
    assert "no model yet" in dash.output.toPlainText()
    assert not np.asarray(obs._core.gains_used()).any()
    obs.addcmp(2.5, 4.0, -2.5)
    dash.solint_edit.setText("scan")
    res = dash.run_selfcal()
    assert res["nbins"] > 0
    assert "selfcal: phase" in dash.output.toPlainText()
    dash.amp_check.setChecked(False)
    dash.phase_check.setChecked(False)
    assert dash.run_selfcal() is None
    assert "tick phase, amplitude or both" in dash.output.toPlainText()
    res = dash.run_gscale()
    assert set(res["gains"]) == set(obs.antennas)
    for cb in dash.model_checks.values():
        cb.setChecked(False)
    assert dash.run_bayes() is None
    assert "at least one source model" in dash.output.toPlainText()


def test_load_rebuilds_every_plot(dash, uvfits_file, scanned_uvfits_file):
    """A file loaded from the Main tab, with averaging, replaces the
    observation and every plot; on_load hears about it."""
    seen = []
    dash.on_load = seen.append
    old = dash.plots()
    dash.timeavg_edit.setText("60s")
    new = dash.load(scanned_uvfits_file[0])
    assert new is dash.obs and seen == [new]
    assert new._aver_time == 60.0
    assert all(p.obs is new for p in dash.plots())
    assert not set(map(id, dash.plots())) & set(map(id, old))
    assert dash.load("/no/such/file.uvf") is None
    assert dash.obs is new


def test_save_buttons_write_files(dash, obs, tmp_path):
    obs.addcmp(2.5, 4.0, -2.5)
    for kind, name in (("clean", "c.fits"), ("dirty", "d.fits"), ("beam", "b.fits"),
                       ("model", "m.mod"), ("uvfits", "v.uvf")):
        path = str(tmp_path / name)
        assert dash.save_file(kind, path) == path
        assert os.path.exists(path)
    dash.save_file("session", str(tmp_path / "sess"))
    assert os.path.exists(tmp_path / "sess.uvf")


def test_empty_dashboard_and_stdout_restored(uvfits_file):
    """With no observation the plots wait for one; closing the dashboard
    gives sys.stdout back."""
    before = sys.stdout
    d = Dashboard(None)
    assert sys.stdout is not before
    assert d.plots() == []
    d.load(uvfits_file)
    assert d.plots()
    d.close()
    assert sys.stdout is before


def test_q_does_not_close_an_embedded_plot(dash):
    from pyqtgraph.Qt import QtCore, QtGui

    p = dash.time_panel.current_plot()
    p.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, 0,
                                    QtCore.Qt.KeyboardModifier.NoModifier, "q"))
    assert p.isVisibleTo(dash)


def test_cli_dashboard_rebinds_the_session(uvfits_file, scanned_uvfits_file):
    """dashboard() at the prompt opens on obs; a file loaded in it becomes
    the session's obs, and the commands act on it."""
    from difmapy import cli

    ns = cli.make_namespace(difmapy.load(uvfits_file))
    d = ns["dashboard"]()
    assert d.obs is ns["obs"]
    new = d.load(scanned_uvfits_file[0])
    assert ns["obs"] is new
    assert ns["clean"].__self__ is new
    d.close()
