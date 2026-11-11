"""difmapy.app's Python side (difmapy.macapp): the commands the SwiftUI app sends, the split between the worker thread and the main thread, and the plot panes it hosts."""

import ctypes
import json
import os
import sys
import threading

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""

pytest.importorskip("pyqtgraph")

import difmapy
from difmapy import macapp
from difmapy.plots import base
from difmapy.plots.mapplot import MapPlot
from difmapy.plots.points import RadPlot, VPlot
from pyqtgraph.Qt import QtCore


# An ordinary Qt application for the whole test session: `macapp` would make it a guest of the app
# (AA_PluginApplication), and the rest of the suite runs Qt's own event loop.
base.ensure_app()


@pytest.fixture()
def session(monkeypatch):
    """A fresh session behind `macapp.dispatch`, as the app has one."""
    s = macapp.Session()
    monkeypatch.setattr(macapp, "_COMMANDS", {name: getattr(s, name) for name in macapp._COMMANDS})
    # Session.start puts the plots in window mode, as the app needs; the rest of the suite expects its own.
    monkeypatch.setattr(base, "_MODE", base._MODE)
    yield s
    for panel in s.panels.values():
        panel.close()


def call(name, **args):
    """A command on the main thread, as the app sends it."""
    return json.loads(macapp.dispatch(name, json.dumps(args)))


def on_worker(name, **args):
    """A command on another thread, as the app runs the ones that compute."""
    out = []
    t = threading.Thread(target=lambda: out.append(macapp.dispatch(name, json.dumps(args))))
    t.start()
    t.join()
    return json.loads(out[0])


def load(session, path, **args):
    """Load on the worker and let `finish` take the observation over (on a small map, to keep it fast)."""
    reply = on_worker("load", path=str(path), **args)
    assert "ok" in reply, reply
    session._loaded.mapsize(256, 0.25)
    return call("finish", command="load")["ok"]


def test_state_before_any_data(session):
    s = call("state")["ok"]
    assert s["version"] == difmapy.__version__
    assert not s["loaded"]
    assert s["panes"] == {"time": ["vplot", "tplot", "possm"], "uv": ["radplot", "projplot", "corplot"],
                          "map": ["mapplot"]}
    kinds = [kind for kind, _label, _title in s["savers"]]
    assert {"clean", "model", "uvfits", "caltable", "flags", "session"} <= set(kinds)


def test_qt_commands_are_refused_off_the_main_thread(session):
    """Only the commands that compute may run on the worker: the rest touch Qt."""
    assert "main thread" in on_worker("state")["error"]
    assert "main thread" in on_worker("attach", pane="map")["error"]
    assert "main thread" in on_worker("show", pane="map")["error"]
    assert "unknown command" in call("no_such_command")["error"]
    assert "no data loaded" in on_worker("smartflag")["error"]
    assert macapp.WORKER_COMMANDS <= set(macapp._COMMANDS)


def test_load_on_the_worker_then_finish(session, uvfits_file):
    s = load(session, uvfits_file)
    assert s["loaded"] and session.obs is not None
    assert s["filename"] == os.path.basename(uvfits_file)
    assert s["directory"] == os.path.dirname(uvfits_file)
    assert s["nif"] == session.obs.nif and s["nantennas"] == len(session.obs.antennas)
    assert s["ncomp"] == 0 and s["uv"] == {"uvmin": "", "uvmax": "", "value": "0.5", "radius": ""}
    assert s["save_names"]["model"] == os.path.splitext(s["filename"])[0] + ".mod"
    json.dumps(s)  # all plain JSON


def test_load_refusals(session, tmp_path):
    assert "choose a file" in on_worker("load", path="  ")["error"]
    assert "could not load" in on_worker("load", path=str(tmp_path / "missing.uvf"))["error"]
    assert 'not "lots"' in on_worker("load", path=str(tmp_path / "x.uvf"), freqavg="lots")["error"]
    assert session._loaded is None and session.obs is None


def test_load_with_averaging(session, scanned_uvfits_file):
    load(session, scanned_uvfits_file[0], timeavg="60s")
    assert session.obs._aver_time == 60.0


def test_panes_hold_the_plots_and_follow_the_selection(session, uvfits_file):
    """Each pane is a dashboard panel with its tab bar hidden (the app's segmented control picks the plot); a new file rebuilds every plot; a command takes the plots' input away until it finishes."""
    load(session, uvfits_file)
    time, uv, mapp = (session._panel(p) for p in ("time", "uv", "map"))
    assert time.tabBar().isHidden() and time.count() == 3
    assert isinstance(time.current_plot(), VPlot) and time.current_plot().obs is session.obs
    assert isinstance(mapp.current_plot(), MapPlot)
    assert call("select", pane="uv", index=1)["ok"]
    assert uv.currentIndex() == 1 and isinstance(uv.current_plot(), RadPlot)
    assert session.selection["uv"] == 1

    call("begin", label="selfcal")
    assert not any(p.isEnabled() for p in session.panels.values())
    call("finish", command="selfcal")
    assert all(p.isEnabled() for p in session.panels.values())

    old = time.current_plot()
    load(session, uvfits_file)
    assert time.current_plot() is not old and time.current_plot().obs is session.obs
    # A pane created later starts on the plot chosen for it.
    session.selection["time"] = 2
    del session.panels["time"]
    assert session._panel("time").currentIndex() == 2


def test_panes_are_embedded_not_parented(session, uvfits_file):
    """`attach` hands over the panel's native view without showing it; `show` shows it, `detach` hides it. The panel stays a top-level window: parented to a foreign QWindow, it would lose every key typed into it. It is a sub-window, which Qt gives no NSWindow: from Qt 6.12, showing any other top level puts its view back into a window of its own, out of the pane."""
    load(session, uvfits_file)
    view = call("attach", pane="map")["ok"]
    panel = session.panels["map"]
    assert isinstance(view, int) and view == int(panel.winId())
    sub = QtCore.Qt.WindowType.SubWindow
    assert panel.windowFlags() & sub == sub and panel.windowHandle().flags() & sub == sub
    assert not panel.isVisible() and "map" not in session.hosts
    assert call("show", pane="map", width=400, height=300)["ok"]
    assert panel.isVisible() and panel.isWindow() and panel.windowHandle().parent() is None
    assert (panel.width(), panel.height()) == (400, 300)
    assert call("detach", pane="map")["ok"] and not panel.isVisible()
    assert not call("detach", pane="map")["ok"]
    assert call("attach", pane="map")["ok"] == view


def test_calibration_needs_a_model(session, uvfits_file):
    """Selfcal and gscale (Bayesian or not) refuse an empty model (it would poison the gains) and run once there is one."""
    load(session, uvfits_file)
    for name, args in (("selfcal", {}), ("gscale", {}), ("gscale", {"bayes": True})):
        assert "no model yet" in on_worker(name, **args)["error"]
    assert "bayes" not in macapp._COMMANDS
    assert not np.asarray(session.obs._core.gains_used()).any()
    session.obs.addcmp(2.5, 4.0, -2.5)
    assert "phase, amplitude or both" in on_worker("selfcal", phase=False, amp=False)["error"]
    assert "ok" in on_worker("selfcal", phase=True, solint="scan")
    assert np.asarray(session.obs._core.gains_used()).any()
    assert "ok" in on_worker("gscale", float_scale=True)
    assert "at least one source model" in on_worker("gscale", bayes=True, models=[])["error"]
    s = call("finish", command="gscale")["ok"]
    assert s["ncomp"] == 1 and s["model_flux"] == pytest.approx(2.5)


def test_uv_range_and_taper(session, uvfits_file):
    load(session, uvfits_file)
    assert "ok" in on_worker("uvrange", uvmin="1", uvmax="")
    assert "ok" in on_worker("uvtaper", value="0.3", radius="20")
    uv = call("finish", command="uvtaper")["ok"]["uv"]
    assert uv["uvmin"] and not uv["uvmax"]
    assert uv["value"] == "0.3" and uv["radius"]
    assert "between 0 and 1" in on_worker("uvtaper", value="half")["error"]
    assert "ok" in on_worker("uvtaper", value="")
    assert call("state")["ok"]["uv"]["radius"] == ""


def test_smartflag_runs_on_the_worker(session, uvfits_file):
    load(session, uvfits_file)
    assert "ok" in on_worker("smartflag", use_phases=False, antenna_sigma="6")
    assert 'not "six"' in on_worker("smartflag", antenna_sigma="six")["error"]
    assert 'positive number, not "0"' in on_worker("smartflag", antenna_sigma="0")["error"]
    # Below the default edge_sigma (1.5): the edge comes down with it, rather than the run failing.
    assert "ok" in on_worker("smartflag", antenna_sigma="1")


def test_save_writes_the_products(session, uvfits_file, tmp_path):
    load(session, uvfits_file)
    session.obs.addcmp(2.5, 4.0, -2.5)
    for kind, name in (("model", "m.mod"), ("clean", "c.fits"), ("uvfits", "v.uvf")):
        path = str(tmp_path / name)
        assert on_worker("save", kind=kind, path=path)["ok"] == path
        assert os.path.exists(path)
    assert "nothing called" in on_worker("save", kind="nope", path=str(tmp_path / "x"))["error"]


def test_output_and_changes_reach_the_app(session, uvfits_file, monkeypatch):
    """Everything printed - from either thread - goes to the app's output, with no terminal escapes; a plot that changed the data makes the others catch up and tells the app."""
    got = []
    callback = macapp._NOTIFY(lambda kind, payload: got.append((kind.decode(), json.loads(payload))))
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setattr(macapp, "_notify", None)
    macapp.connect(ctypes.cast(callback, ctypes.c_void_p).value)
    assert isinstance(sys.stdout, macapp._Tee) and isinstance(sys.stderr, macapp._Tee)
    print("\x1b[1mbold\x1b[0m ± λ")
    on_worker("load", path="")                    # refused: the message is printed too
    texts = "".join(p["text"] for k, p in got if k == "output")
    assert "bold ± λ" in texts and "\x1b" not in texts
    assert "choose a file first" in texts

    load(session, uvfits_file)
    vplot = session._panel("time").current_plot()
    radplot = session._panel("uv").current_plot()
    reloaded = []
    monkeypatch.setattr(radplot, "reload", lambda: reloaded.append(radplot))
    got.clear()
    session._changed_in(vplot)
    assert reloaded == [radplot]
    assert ("changed", {}) in got


def test_a_guest_qt_never_blocks(monkeypatch):
    """In the app Qt is a guest, whose exec() never returns: presenting a plot window must not enter it."""
    from pyqtgraph.Qt import QtWidgets

    entered = []
    monkeypatch.setattr(base.pg, "exec", lambda: entered.append(True))
    monkeypatch.setattr(base, "get_mode", lambda: "window")
    widget = QtWidgets.QWidget()
    monkeypatch.setattr(base, "_qt_is_guest", lambda: True)
    base.run_if_needed(widget, block=True)
    assert not entered
    monkeypatch.setattr(base, "_qt_is_guest", lambda: False)
    base.run_if_needed(widget, block=True)
    assert entered
    widget.close()


def test_selftest_line(session):
    call("start")
    line = call("selftest")["ok"]
    assert line.startswith(f"difmapy {difmapy.__version__} app self-test")
    assert "native controls" in line
