"""The Python side of difmapy.app, the macOS application.

The app is a SwiftUI program (``packaging/macos/app/``): its window, toolbar, menus, inspector of controls, output console, file panels and settings are native. It embeds this Python through ``pybridge.c`` and hosts the interactive plots - the ordinary pyqtgraph plot windows, which keep their keys, flagging, CLEAN windows and status lines - in its panes.

Qt runs there as a guest (`difmapy.plots.base.ensure_app` with ``plugin=True``, i.e. ``Qt.AA_PluginApplication``): it leaves the menu bar, the application delegate and the Dock icon to the app, and its events are delivered by the app's own run loop, with no ``exec()``. Each pane is a dashboard plot panel (`difmapy.plots.dashboard._PlotPanel`) with its tab bar hidden - the pane's segmented control picks the plot. The panel stays a top-level Qt window, and the app puts its ``NSView`` (`Session.attach` hands it over) into an ``NSView`` of the SwiftUI window: Qt's *embedded* mode, in which the plot follows that view's size and takes the keys typed into it. The panel carries ``Qt.SubWindow``, a top level that Qt does not give an ``NSWindow``: from Qt 6.12 on, any other top level's view is put back into a window of its own on every ``show()``. It is not parented to a foreign ``QWindow`` (``QWindow.fromWinId``): Qt's Cocoa plugin sends every key to the top of a widget window's ``QWindow`` parents, which would then be that foreign window - and the keys would go nowhere.

The app calls `dispatch` with a command name and its arguments in JSON and gets JSON back: ``{"ok": result}`` or ``{"error": message}``. The commands in `WORKER_COMMANDS` compute (load, calibrate, save) and the app runs them on a worker thread, so that its window stays live; they must not touch Qt. The app brackets each with `Session.begin` (the plots stop taking input) and `Session.finish` (the plots catch up, take input again, and the new state comes back), both on the main thread, where every other command runs. This module talks back through `emit`: ``output`` (what the session prints, for the Output tab) and ``changed`` (a plot changed the data or the model).
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sys
import threading
import traceback

__all__ = ["connect", "dispatch", "emit", "Session", "SESSION", "UserError", "WORKER_COMMANDS"]

#: Commands that compute, run by the app on a worker thread: they must not touch Qt.
WORKER_COMMANDS = frozenset({"load", "smartflag", "selfcal", "gscale", "uvrange", "uvtaper", "save",
                             "install_command", "answer_command_offer"})
#: Commands after which every plot is reloaded (`load` rebuilds them instead).
DATA_COMMANDS = frozenset({"smartflag", "selfcal", "gscale"})
#: Commands after which only the map is (the UV range and taper change nothing else on show).
MAP_COMMANDS = frozenset({"uvrange", "uvtaper"})

#: Terminal escape codes (the bold of the modelfit report), which the Output tab shows as plain text.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_NOTIFY = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)
_notify = None


class UserError(Exception):
    """A command refused, with a message for the user rather than a traceback."""


def connect(address: int):
    """Called by ``pybridge.c`` once this module is imported: `address` is the app's C callback for `emit`. Everything printed from here on also goes to the app's Output tab."""
    global _notify
    _notify = _NOTIFY(address) if address else None
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if not isinstance(stream, _Tee):
            setattr(sys, name, _Tee(stream, name))


def emit(kind: str, payload=None):
    """Send `kind` and a JSON-able `payload` to the app (delivered on its main thread, whichever thread calls)."""
    if _notify is not None:
        _notify(kind.encode(), json.dumps(payload, default=_jsonable).encode())


class _Tee:
    """A stand-in for ``sys.stdout``/``sys.stderr`` that passes everything on to the real stream and also to the app. Printing must never fail a computation, so neither can raise."""

    def __init__(self, original, name):
        self._original = original
        self._name = name

    def write(self, s):
        if not s:
            return 0
        if self._original is not None:
            try:
                self._original.write(s)
            except Exception:
                pass
        try:
            emit("output", {"text": _ANSI.sub("", s), "stream": self._name})
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            if self._original is not None:
                self._original.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._original, name)


def _jsonable(value):
    """`json.dumps` default: numpy scalars and arrays as numbers and lists, anything else as text."""
    for attr in ("tolist", "item"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:
                pass
    return str(value)


class Session:
    """The app's session: the observation, and the plot panels hosted in the SwiftUI panes. It stands in for the dashboard window towards `_PlotPanel` (`obs`, `_reload`, `_changed_in`)."""

    def __init__(self):
        self.obs = None
        #: An observation loaded on the worker thread, taken over by `finish` on the main thread.
        self._loaded = None
        #: pane ("time", "uv", "map") -> its _PlotPanel
        self.panels = {}
        #: panes whose plots are in the app's window
        self.hosts = set()
        #: pane -> index of the plot shown
        self.selection = {}

    # ---- the Qt guest and the panes (main thread) -----------------------

    def start(self):
        """Start Qt as a guest of the app, and return the state."""
        from difmapy.plots.base import ensure_app, set_mode

        ensure_app(plugin=True)
        set_mode("window")
        return self.state()

    def _panel(self, pane):
        panel = self.panels.get(pane)
        if panel is None:
            from difmapy.plots.dashboard import _PlotPanel, _plot_factories
            from pyqtgraph.Qt import QtCore

            self.start()
            panel = _PlotPanel(self, _plot_factories()[pane])
            # Before its native view exists: a sub-window is a top level without an NSWindow of its own. Qt 6.12
            # puts any other top level's view back into an NSWindow it makes for it on every show(), which took
            # each plot out of its pane into a window of its own. Qt Widgets turns a parentless Qt.SubWindow into
            # SubWindow | Window, so the panel is still a top-level widget with no QWindow parent, and the keys
            # typed into it still reach it.
            panel.setWindowFlags(QtCore.Qt.WindowType.SubWindow)
            # The SwiftUI pane's segmented control picks the plot.
            panel.setTabBarAutoHide(False)
            panel.tabBar().hide()
            panel.setDocumentMode(True)
            panel.blockSignals(True)
            panel.setCurrentIndex(self.selection.get(pane, 0))
            panel.blockSignals(False)
            panel.build(panel.currentIndex())
            self.panels[pane] = panel
        return panel

    def attach(self, pane):
        """The native view of the plots of `pane` - the address of the panel's ``NSView`` - for the app to put into the pane's view; `show` then shows them. The panel is a ``Qt.SubWindow``, so Qt makes no ``NSWindow`` for that view: it is shown only inside the pane."""
        panel = self._panel(pane)
        return int(panel.winId())

    def show(self, pane, width=0, height=0):
        """Show the plots of `pane`, whose view the app has just put into the pane (at `width` x `height` points); from then on they follow that view's size."""
        panel = self._panel(pane)
        if width > 0 and height > 0:
            panel.setGeometry(0, 0, int(width), int(height))
        panel.show()
        self.hosts.add(pane)
        return True

    def detach(self, pane):
        """Hide the plots of `pane`, before the app takes their view out of the pane's (and lets go of that)."""
        panel = self.panels.get(pane)
        if pane not in self.hosts or panel is None:
            return False
        self.hosts.discard(pane)
        panel.hide()
        return True

    def select(self, pane, index):
        """Show plot `index` of `pane` (built the first time, reloaded if it went stale while hidden)."""
        index = int(index)
        self.selection[pane] = index
        panel = self.panels.get(pane)
        if panel is not None and panel.currentIndex() != index:
            panel.setCurrentIndex(index)
        return True

    def begin(self, label=""):
        """A worker command starts: the plots stop taking input until `finish`."""
        for panel in self.panels.values():
            panel.setEnabled(False)
        return True

    def finish(self, command=""):
        """A worker command ended: take over a newly loaded observation (every plot rebuilt for it) or bring the plots up to date, let them take input again, and return the state."""
        try:
            if self._loaded is not None:
                self.obs, self._loaded = self._loaded, None
                for panel in self.panels.values():
                    panel.rebuild()
            elif command in DATA_COMMANDS:
                self.sync()
            elif command in MAP_COMMANDS and "map" in self.panels:
                self.panels["map"].sync()
        finally:
            for panel in self.panels.values():
                panel.setEnabled(True)
        return self.state()

    def reload(self):
        """Reload every plot (the app's Reload All)."""
        self.sync()
        return self.state()

    def shutdown(self):
        """The app quits (after taking the plots' views out of its window): close the plots, before AppKit tears the views down."""
        for pane in list(self.hosts):
            self.detach(pane)
        from difmapy.plots.base import close_all_windows

        close_all_windows()
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        return True

    # ---- what _PlotPanel expects of its dashboard ------------------------

    def _reload(self, plot):
        try:
            plot.reload()
        except Exception as exc:
            print(f"difmapy: could not reload {plot.windowTitle()}: {exc}")

    def _changed_in(self, plot):
        """An embedded plot changed the data or the model: the other plots catch up, and the app reads the state again."""
        self.sync(source=plot)
        emit("changed", {})

    def sync(self, source=None):
        for panel in self.panels.values():
            panel.sync(source)

    # ---- the state the app shows ---------------------------------------

    def state(self):
        """What the app shows: the version, the panes and their plots, what can be saved, and - once data are loaded - the file, a summary, the model and the UV range and taper."""
        import difmapy
        from difmapy.plots.dashboard import DEFAULT_TAPER_VALUE, SAVERS, _plot_factories

        s = {
            "version": difmapy.__version__,
            "panes": {pane: [label for label, _ in specs] for pane, specs in _plot_factories().items()},
            "savers": [[kind, label, title] for kind, (label, title, *_rest) in SAVERS.items()],
            "loaded": self.obs is not None,
        }
        obs = self.obs
        if obs is None:
            return s
        from difmapy.units import format_uvdist

        path = os.fspath(obs.path) if obs.path else None
        taper = 0.0 < obs._gauval < 1.0 and obs._gaurad > 0.0
        s.update(
            filename=obs.filename,
            path=path,
            directory=os.path.dirname(os.path.normpath(path)) if path else None,
            source=str(obs.source),
            nif=int(obs.nif),
            nantennas=len(obs.antennas),
            pols=list(obs.pols),
            flagged=float(obs.flagged_fraction),
            ncomp=len(obs.model),
            nvariable=int(obs.nvariable),
            model_flux=float(obs.model_flux),
            uv={
                "uvmin": format_uvdist(obs._uvmin) if obs._uvmin else "",
                "uvmax": format_uvdist(obs._uvmax) if obs._uvmax else "",
                "value": f"{obs._gauval:g}" if taper else f"{DEFAULT_TAPER_VALUE:g}",
                "radius": format_uvdist(obs._gaurad) if taper else "",
            },
            save_names={kind: _default_name(obs, kind) for kind in SAVERS},
        )
        return s

    # ---- commands that compute (worker thread: no Qt) -------------------

    def _need_data(self, what):
        if self.obs is None:
            raise UserError(f"{what}: no data loaded")
        return self.obs

    def _need_model(self, what):
        obs = self._need_data(what)
        if not obs.model:
            # Calibrating against an empty model gives NaN gains, which
            # would poison the data.
            raise UserError(f"{what}: there is no model yet - CLEAN (c) or fit (m, f) one in the map first")
        return obs

    def load(self, path="", timeavg="", freqavg=""):
        """Load a UVFITS file or a Measurement Set as `difmapy.load` does; `finish` then shows it. `timeavg` is a time with a unit ("10s"), `freqavg` a channel count or "all"; empty for none."""
        import difmapy

        path = (path or "").strip()
        if not path:
            raise UserError("load: choose a file first")
        if os.path.basename(path) == "table.dat":
            path = os.path.dirname(path)
        timeavg = (timeavg or "").strip() or None
        text = str(freqavg or "").strip()
        if not text:
            freqavg = None
        elif text.lower() == "all":
            freqavg = "all"
        else:
            try:
                freqavg = int(text)
            except ValueError:
                raise UserError(f'load: the frequency average is a number of channels or "all", not "{text}"') from None
        try:
            obs = difmapy.load(path, timeavg=timeavg, freqavg=freqavg)
        except Exception as exc:
            raise UserError(f"load: could not load {path}: {exc}") from exc
        obs.header()
        self._loaded = obs
        return {"filename": obs.filename}

    def smartflag(self, use_phases=True, antenna_sigma=""):
        """`smartflag` over all the data, testing the phases too unless `use_phases` is false, at the antenna sigma given (empty: 5)."""
        obs = self._need_data("smartflag")
        text = str(antenna_sigma or "").strip()
        try:
            sigma = float(text) if text else 5.0
        except ValueError:
            sigma = None
        if sigma is None or not sigma > 0:
            raise UserError(f'smartflag: the antenna sigma is a positive number, not "{text}"')
        obs.smartflag(use_phases=bool(use_phases), antenna_sigma=sigma, quiet=False)
        return None

    def selfcal(self, phase=True, amp=False, converge=True, solint=""):
        """Self-calibrate: phase and/or amplitude, at the solution interval given (minutes as a bare number, or with a unit, or "scan"; empty for one solution per integration), alternating with `modelfit` until the fit converges when `converge` - which needs components with free parameters; a CLEAN-only model gets one selfcal round."""
        obs = self._need_model("selfcal")
        if not (amp or phase):
            raise UserError("selfcal: choose phase, amplitude or both")
        solint = str(solint or "").strip() or 0.0
        if converge and not obs.nvariable:
            print("selfcal: the model has no free parameters (CLEAN components only), so there is "
                  "nothing for modelfit to converge; running one selfcal round")
            converge = False
        obs.selfcal(amp=bool(amp), phase=bool(phase), solint=solint, converge=bool(converge), quiet=False)
        return None

    def gscale(self, float_scale=False, bayes=False, models=()):
        """`gscale`, floating the flux scale if `float_scale`; with `bayes`, the Bayesian gscale over the source models given ("clean", "gauss1", ...)."""
        if not bayes:
            self._need_model("gscale").gscale(float_scale=bool(float_scale), quiet=False)
            return None
        obs = self._need_model("gscale")
        models = tuple(models or ())
        if not models:
            raise UserError("gscale: choose at least one source model for Bayes")
        obs.gscale(float_scale=bool(float_scale), bayes=True, models=models, plot=False, quiet=False)
        return None

    def uvrange(self, uvmin="", uvmax=""):
        """Image only the visibilities between the two UV radii (M-lambda, or a number with a unit; an empty max is no limit), and re-image."""
        obs = self._need_data("uvrange")
        obs.uvrange(str(uvmin or "").strip() or 0.0, str(uvmax or "").strip() or 0.0)
        from difmapy.plots.dashboard import print_uv_report

        print_uv_report(obs, "uvrange")
        return None

    def uvtaper(self, value="", radius=""):
        """Taper the gridding weights to `value` at `radius` (M-lambda, or with a unit); an empty or zero value turns the taper off. Re-images."""
        obs = self._need_data("uvtaper")
        text = str(value or "").strip()
        try:
            value = float(text) if text else 0.0
        except ValueError:
            raise UserError(f'uvtaper: the value is a number between 0 and 1, not "{text}"') from None
        obs.uvtaper(value, (str(radius or "").strip() or 0.0) if value else 0.0)
        from difmapy.plots.dashboard import print_uv_report

        print_uv_report(obs, "uvtaper")
        return None

    def save(self, kind, path):
        """Write one product of the session (`kind`: a key of `difmapy.plots.dashboard.SAVERS`) to `path`."""
        from difmapy.plots.dashboard import SAVERS

        obs = self._need_data("save")
        if kind not in SAVERS:
            raise UserError(f"save: nothing called {kind!r} to save")
        label, _title, _filter, _suffix, write = SAVERS[kind]
        write(obs, path)
        print(f"saved {label.lower()}: {path}")
        return path

    # ---- the Terminal command --------------------------------------------

    def command_status(self):
        """Whether to offer the Terminal command (first launch), whether it is installed, and where."""
        from difmapy import app_main

        target = app_main.command_target()
        return {"offer": app_main.command_offer_pending(), "installed": app_main.command_installed(),
                "link": str(app_main.COMMAND_LINK), "target": str(target) if target else None}

    def answer_command_offer(self, install=False):
        """The first-launch question was answered (it is not asked again)."""
        from difmapy import app_main

        ok, msg = app_main.answer_command_offer(bool(install))
        if install:
            print(f"difmapy: {msg}")
        return {"installed": ok, "message": msg}

    def install_command(self):
        """Link the Terminal command now (the Settings window's button)."""
        from difmapy import app_main

        ok, msg = app_main.install_command()
        print(f"difmapy: {msg}")
        if not ok:
            raise UserError(f"The command was not installed: {msg}")
        return {"installed": ok, "message": msg}

    def selftest(self):
        """The line the build script's self-test prints: what the app found."""
        import difmapy
        from pyqtgraph.Qt import QtCore, QtWidgets

        from difmapy import app_main
        from difmapy.plots import style

        app = QtWidgets.QApplication.instance()
        guest = QtCore.QCoreApplication.testAttribute(QtCore.Qt.ApplicationAttribute.AA_PluginApplication)
        qt = f"Qt {QtCore.qVersion()} on {app.platformName()}" if app is not None else "Qt not started"
        line = (f"difmapy {difmapy.__version__} app self-test: python {sys.version.split()[0]} at {sys.prefix}; "
                f"stdout {getattr(sys.__stdout__, 'encoding', None)} (± λ); bundle {app_main._bundle()}; "
                f"{qt}, guest {guest}; native controls {style.native_controls()}; "
                f"command {app_main.command_target()}")
        print(line)
        return line


def _default_name(obs, kind):
    """The file name the save panel offers for `kind`: the data's name with the product's suffix."""
    from difmapy.plots.dashboard import SAVERS

    suffix = SAVERS[kind][3]
    if callable(suffix):
        try:
            suffix = suffix(obs)
        except Exception:
            suffix = ""
    return os.path.splitext(obs.filename or "difmapy")[0] + suffix


#: The one session of the app.
SESSION = Session()

_COMMANDS = {name: getattr(SESSION, name) for name in (
    "start", "state", "attach", "show", "detach", "select", "begin", "finish", "reload", "shutdown",
    "load", "smartflag", "selfcal", "gscale", "uvrange", "uvtaper", "save",
    "command_status", "answer_command_offer", "install_command", "selftest")}


def dispatch(name: str, args_json: str = "{}") -> str:
    """Run command `name` with the keyword arguments in `args_json`; returns ``{"ok": result}`` or ``{"error": message}`` as JSON. A refusal or failure is also printed (it shows in the Output tab). Commands outside `WORKER_COMMANDS` touch Qt and are refused off the main thread."""
    try:
        fn = _COMMANDS.get(name)
        if fn is None:
            raise UserError(f"unknown command {name!r}")
        if name not in WORKER_COMMANDS and threading.current_thread() is not threading.main_thread():
            raise RuntimeError(f"{name} must run on the main thread")
        args = json.loads(args_json or "{}")
        return json.dumps({"ok": fn(**args)}, default=_jsonable)
    except UserError as exc:
        print(exc)
        return json.dumps({"error": str(exc)})
    except Exception as exc:
        traceback.print_exc()
        return json.dumps({"error": f"{name}: {type(exc).__name__}: {exc}"})
