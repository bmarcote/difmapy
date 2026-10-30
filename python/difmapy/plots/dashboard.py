"""The dashboard: every view of a session in one window, kept in step.

Four panels: the time and frequency views of the visibilities (vplot, tplot, possm) top left, the uv-distance views and the gain corrections (radplot, projplot, corplot) top right, the map (mapplot) bottom left, and the controls bottom right - loading a file, the calibration commands, and an output tab that shows everything the session prints, with buttons to save its products.

The plots are the ordinary plot windows, embedded: they keep their keys, flagging and status lines. Whatever changes the data or the model in one of them - a flag edit, a CLEAN, a fit - or from the controls reloads the others, as ``r`` would; a plot in a tab that is not shown is reloaded when its tab is next opened.
"""

from __future__ import annotations

import os
import re
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from difmapy.plots.base import PlotWindow, run_if_needed

__all__ = ["Dashboard", "dashboard"]

#: Source models `bayes_gscale` may compare, as the Bayes row offers
#: them, and which of them are ticked to begin with.
BAYES_MODELS = (("clean", False), ("gauss1", True), ("gauss2", True), ("gauss3", True))

#: Terminal escape codes (the bold of the modelfit report), which the
#: output tab shows as plain text.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plot_factories():
    """(tab label, constructor taking the observation) of every embedded plot, by panel."""
    from difmapy.plots.diagnostics import CorPlot, PossmPlot, TPlot
    from difmapy.plots.mapplot import MapPlot
    from difmapy.plots.points import RadPlot, VPlot

    return {
        "time": (("vplot", lambda o: VPlot(o)),
                 ("tplot", lambda o: TPlot(o)),
                 ("possm", lambda o: PossmPlot(o))),
        "uv": (("radplot", lambda o: RadPlot(o)),
               ("projplot", lambda o: RadPlot(o, projection_deg=0.0)),
               ("corplot", lambda o: CorPlot(o))),
        "map": (("mapplot", lambda o: MapPlot(o)),),
    }


class _Tee(QtCore.QObject):
    """A stand-in for ``sys.stdout`` that passes everything on to the real one and also emits it, for the output tab. The signal is delivered in the GUI thread whichever thread printed."""

    text = QtCore.Signal(str)

    def __init__(self, original):
        super().__init__()
        self._original = original

    def write(self, s):
        if self._original is not None:
            self._original.write(s)
        if s:
            self.text.emit(s)
        return len(s)

    def flush(self):
        if self._original is not None:
            self._original.flush()

    def isatty(self):
        return bool(self._original is not None and self._original.isatty())

    def __getattr__(self, name):
        return getattr(self._original, name)


class _PlotPanel(QtWidgets.QTabWidget):
    """One panel of the dashboard: a tab per plot, each built the first time it is shown."""

    def __init__(self, dash, specs):
        super().__init__()
        self.dash = dash
        self.specs = list(specs)
        #: tab index -> the embedded plot, once built
        self.plots: dict[int, PlotWindow] = {}
        #: tabs whose plot is out of date (reloaded when shown)
        self.stale: set[int] = set()
        self.setTabBarAutoHide(True)
        self.setDocumentMode(True)
        for label, _ in self.specs:
            page = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(page)
            lay.setContentsMargins(0, 0, 0, 0)
            self.addTab(page, label)
        self.currentChanged.connect(self._shown)

    def _shown(self, index):
        if index < 0:
            return
        if index not in self.plots:
            self.build(index)
        elif index in self.stale:
            self.stale.discard(index)
            self.dash._reload(self.plots[index])

    def build(self, index):
        """Build the plot of tab `index` for the dashboard's observation (or a note asking for one)."""
        page = self.widget(index)
        lay = page.layout()
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        obs = self.dash.obs
        if obs is None:
            note = QtWidgets.QLabel("No data loaded: choose a file in the Main tab.")
            note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(note)
            return None
        label, make = self.specs[index]
        try:
            plot = make(obs)
        except Exception as exc:  # e.g. corplot before any calibration
            print(f"dashboard: {label} could not be drawn: {exc}")
            note = QtWidgets.QLabel(f"{label}: {exc}")
            note.setWordWrap(True)
            note.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(note)
            return None
        plot.embedded = True
        plot.on_change = self.dash._changed_in
        plot.setWindowFlags(QtCore.Qt.WindowType.Widget)
        lay.addWidget(plot)
        plot.show()
        self.plots[index] = plot
        self.stale.discard(index)
        return plot

    def current_plot(self):
        return self.plots.get(self.currentIndex())

    def sync(self, source=None):
        """Reload the plot on show (unless it is `source`, which is up to date) and mark the hidden ones stale."""
        for index, plot in self.plots.items():
            if plot is source:
                continue
            if index == self.currentIndex():
                self.dash._reload(plot)
            else:
                self.stale.add(index)

    def rebuild(self):
        """Throw every plot away and build the one on show again (a new observation)."""
        for plot in self.plots.values():
            plot.on_change = None
        self.plots.clear()
        self.stale.clear()
        for index in range(self.count()):
            lay = self.widget(index).layout()
            while lay.count():
                w = lay.takeAt(0).widget()
                if w is not None:
                    w.deleteLater()
        self.build(self.currentIndex())


class Dashboard(PlotWindow):
    """Every view of a session in one window, kept in step (see the module docstring).

    `obs` may be None: the plots then wait for a file to be loaded from the Main tab. `on_load` is called with each observation loaded there, which is how the ``difmapy`` prompt rebinds its commands to it.
    """

    DEFAULT_SIZE = (1850, 1150)

    def __init__(self, obs=None, on_load=None):
        super().__init__("difmapy dashboard")
        self.obs = obs
        self.on_load = on_load
        factories = _plot_factories()
        self.time_panel = _PlotPanel(self, factories["time"])
        self.uv_panel = _PlotPanel(self, factories["uv"])
        self.map_panel = _PlotPanel(self, factories["map"])
        self.panels = (self.time_panel, self.uv_panel, self.map_panel)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self._make_main(), "Main")
        self.tabs.addTab(self._make_output(), "Output")

        top = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        top.addWidget(self.time_panel)
        top.addWidget(self.uv_panel)
        bottom = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        bottom.addWidget(self.map_panel)
        bottom.addWidget(self.tabs)
        rows = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        rows.addWidget(top)
        rows.addWidget(bottom)
        for s in (top, bottom, rows):
            s.setChildrenCollapsible(False)
        self.setCentralWidget(rows)
        w, h = self.width(), self.height()
        top.setSizes([w // 2, w // 2])
        bottom.setSizes([w // 2, w // 2])
        rows.setSizes([h // 2, h // 2])

        # Everything the session prints also goes to the output tab -
        # commands typed at the prompt as well as the dashboard's own.
        # Whatever replaces sys.stdout later (a test harness, %capture)
        # would cut the tab off, so the tee is checked again before every
        # action and every half second, and put back on top if needed.
        self._tee = None
        self._ensure_tee()
        self._tee_timer = QtCore.QTimer(self)
        self._tee_timer.timeout.connect(self._ensure_tee)
        self._tee_timer.timeout.connect(self._sync_uv_fields)
        self._tee_timer.start(500)
        self.destroyed.connect(lambda *_, d=self: _restore_stdout(d._tee))

        self._set_title()
        for panel in self.panels:
            panel.build(panel.currentIndex())
        self.statusBar().showMessage(
            "the panels follow each other: flag, clean or fit in one and the rest reload | r: reload all")

    # ---- the controls -------------------------------------------------

    def _make_main(self):
        page = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(page)
        grid.setColumnStretch(1, 1)

        grid.addWidget(QtWidgets.QLabel("<b>File</b>"), 0, 0)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("a UVFITS file or a Measurement Set")
        if self.obs is not None and self.obs.path:
            self.file_edit.setText(os.fspath(self.obs.path))
        grid.addWidget(self.file_edit, 0, 1, 1, 3)
        browse = QtWidgets.QHBoxLayout()
        for text, ms in (("UVFITS...", False), ("MS...", True)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, ms=ms: self._browse(ms))
            browse.addWidget(b)
        grid.addLayout(browse, 0, 4)

        avg = QtWidgets.QHBoxLayout()
        avg.addWidget(QtWidgets.QLabel("timeavg"))
        self.timeavg_edit = QtWidgets.QLineEdit()
        self.timeavg_edit.setPlaceholderText('e.g. 10s, 2min')
        avg.addWidget(self.timeavg_edit)
        avg.addWidget(QtWidgets.QLabel("freqavg"))
        self.freqavg_edit = QtWidgets.QLineEdit()
        self.freqavg_edit.setPlaceholderText('channels, or "all"')
        avg.addWidget(self.freqavg_edit)
        grid.addLayout(avg, 1, 1, 1, 3)
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.clicked.connect(lambda: self.load())
        grid.addWidget(self.load_button, 1, 4)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        grid.addWidget(line, 2, 0, 1, 5)

        self.selfcal_button = QtWidgets.QPushButton("Selfcal")
        self.selfcal_button.clicked.connect(lambda: self.run_selfcal())
        grid.addWidget(self.selfcal_button, 3, 0)
        self.phase_check = QtWidgets.QCheckBox("phase")
        self.phase_check.setChecked(True)
        self.amp_check = QtWidgets.QCheckBox("amplitude")
        grid.addWidget(self.phase_check, 3, 1)
        grid.addWidget(self.amp_check, 4, 1)
        grid.addWidget(QtWidgets.QLabel("sol. interval"), 3, 2,
                       alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        self.solint_edit = QtWidgets.QLineEdit()
        self.solint_edit.setPlaceholderText('minutes, "30s", "scan"')
        grid.addWidget(self.solint_edit, 3, 3, 1, 2)

        self.gscale_button = QtWidgets.QPushButton("Gscale")
        self.gscale_button.clicked.connect(lambda: self.run_gscale())
        grid.addWidget(self.gscale_button, 5, 0)
        self.float_check = QtWidgets.QCheckBox("float scale")
        self.float_check.setChecked(True)
        grid.addWidget(self.float_check, 5, 1)

        self.bayes_button = QtWidgets.QPushButton("Bayes")
        self.bayes_button.clicked.connect(lambda: self.run_bayes())
        grid.addWidget(self.bayes_button, 6, 0)
        models = QtWidgets.QHBoxLayout()
        self.model_checks = {}
        for name, on in BAYES_MODELS:
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(on)
            models.addWidget(cb)
            self.model_checks[name] = cb
        models.addStretch(1)
        grid.addLayout(models, 6, 1, 1, 4)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        grid.addWidget(line, 7, 0, 1, 5)

        # Imaging: the UV range and the taper, in M-lambda unless a unit
        # is typed ("500k"). Like the calibration rows, the name applies.
        self.uv_edits = {}

        def uv_row(row, name, apply, fields):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(lambda: apply())
            grid.addWidget(b, row, 0)
            box = QtWidgets.QHBoxLayout()
            for key, label, placeholder, tip in fields:
                box.addWidget(QtWidgets.QLabel(label))
                e = QtWidgets.QLineEdit()
                e.setPlaceholderText(placeholder)
                e.setToolTip(tip)
                e.returnPressed.connect(lambda: apply())
                box.addWidget(e)
                self.uv_edits[key] = e
            box.addWidget(QtWidgets.QLabel("M\u03bb"))
            grid.addLayout(box, row, 1, 1, 4)
            return b

        self.uvrange_button = uv_row(8, "UV range", self.run_uvrange, (
            ("uvmin", "min", "0", "shortest UV radius imaged: M-lambda, or with a unit (500k)"),
            ("uvmax", "max", "no limit", "longest UV radius imaged; empty for no limit")))
        self.uvtaper_button = uv_row(9, "Taper", self.run_uvtaper, (
            ("value", "value", "none", "the taper at the radius, between 0 and 1; empty for no taper"),
            ("radius", "radius", "", "UV radius where the taper falls to the value: M-lambda, or with a unit")))
        self._sync_uv_fields()

        reload_all = QtWidgets.QPushButton("Reload all")
        reload_all.setToolTip("re-read the data in every panel (after changes made at the prompt)")
        reload_all.clicked.connect(lambda: self.reload())
        grid.addWidget(reload_all, 11, 4)
        grid.setRowStretch(10, 1)
        return page

    def _sync_uv_fields(self):
        """Show the observation's UV range and taper in the Main tab (fields being typed in are left alone), so that a change made in the map panel or at the prompt shows here too."""
        from difmapy.units import format_uvdist

        obs = self.obs
        if obs is None:
            return
        taper = 0.0 < obs._gauval < 1.0 and obs._gaurad > 0.0
        values = {
            "uvmin": format_uvdist(obs._uvmin) if obs._uvmin else "",
            "uvmax": format_uvdist(obs._uvmax) if obs._uvmax else "",
            "value": f"{obs._gauval:g}" if taper else "",
            "radius": format_uvdist(obs._gaurad) if taper else "",
        }
        for key, e in self.uv_edits.items():
            if not e.hasFocus() and e.text() != values[key]:
                e.setText(values[key])

    def _make_output(self):
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(self.output, 1)
        buttons = QtWidgets.QGridLayout()
        self.save_buttons = {}
        for i, (kind, (label, *_)) in enumerate(SAVERS.items()):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self.save_file(k))
            buttons.addWidget(b, i // 4, i % 4)
            self.save_buttons[kind] = b
        lay.addLayout(buttons)
        return page

    def _ensure_tee(self):
        """Put the output tab's tee on ``sys.stdout``, unless it is there."""
        if self._tee is not None and sys.stdout is self._tee:
            return
        if self._tee is not None:
            self._tee.text.disconnect(self._append_output)
        self._tee = _Tee(sys.stdout)
        self._tee.text.connect(self._append_output)
        sys.stdout = self._tee

    def _append_output(self, text):
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(_ANSI.sub("", text))
        self.output.setTextCursor(cursor)
        self.output.ensureCursorVisible()

    def _browse(self, ms):
        if ms:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, "Measurement Set")
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "UVFITS file", "", "UVFITS (*.uvf *.uvfits *.fits *.UVF *.UVFITS *.FITS);;All files (*)")
        if path:
            self.file_edit.setText(path)

    def _set_title(self):
        name = None if self.obs is None else self.obs.filename
        self.setWindowTitle("difmapy dashboard" + (f" ({name})" if name else ""))

    # ---- actions --------------------------------------------------------

    def _run(self, label, fn, data_changed=True):
        """Run `fn` with a busy cursor, report a failure instead of raising it, and bring every panel up to date afterwards (only the map when `data_changed` is False: the UV range and taper change nothing else on show). Returns what `fn` returned, or None on failure."""
        self._ensure_tee()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.statusBar().showMessage(f"{label}...")
        QtWidgets.QApplication.processEvents()
        try:
            out = fn()
        except Exception as exc:
            print(f"{label}: {exc}")
            self.statusBar().showMessage(f"{label} failed: {exc}")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage(f"{label}: done")
        if data_changed:
            self.sync()
        else:
            self.map_panel.sync()
        self._sync_uv_fields()
        return out

    def _need_model(self, what):
        self._ensure_tee()
        if self.obs is None:
            print(f"{what}: no data loaded")
            return False
        if not self.obs.model:
            # Calibrating against an empty model gives NaN gains, which
            # would poison the data.
            print(f"{what}: there is no model yet - CLEAN (c) or fit (m, f) one in the map first")
            return False
        return True

    def load(self, path=None, timeavg=None, freqavg=None):
        """Load a file into the dashboard, as `difmapy.load` does, and rebuild every plot for it. The arguments default to what the Main tab holds; `timeavg` and `freqavg` take what `load` takes (a time with a unit, a channel count or "all")."""
        import difmapy

        self._ensure_tee()
        path = (path or self.file_edit.text()).strip()
        if not path:
            print("load: choose a file first")
            return None
        timeavg = timeavg if timeavg is not None else (self.timeavg_edit.text().strip() or None)
        if freqavg is None:
            text = self.freqavg_edit.text().strip()
            freqavg = None if not text else ("all" if text.lower() == "all" else int(text))
        self.file_edit.setText(path)

        def go():
            obs = difmapy.load(path, timeavg=timeavg, freqavg=freqavg)
            obs.header()
            return obs

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            obs = go()
        except Exception as exc:
            print(f"load: could not load {path}: {exc}")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.obs = obs
        self._set_title()
        self._sync_uv_fields()
        for panel in self.panels:
            panel.rebuild()
        if self.on_load is not None:
            self.on_load(obs)
        return obs

    def _uv_text(self, key):
        return self.uv_edits[key].text().strip()

    def run_uvrange(self):
        """Image only the visibilities between the UV radii of the UV range row (`uvrange`: M-lambda, or a number with a unit; an empty max is no limit), and re-image."""
        if self.obs is None:
            print("uvrange: no data loaded")
            return None
        lo, hi = self._uv_text("uvmin") or 0.0, self._uv_text("uvmax") or 0.0

        def go():
            self.obs.uvrange(lo, hi)
            self._uv_report("uvrange")
            return True

        return self._run("uvrange", go, data_changed=False)

    def run_uvtaper(self):
        """Taper the gridding weights with the Taper row's value and radius (`uvtaper`: M-lambda, or a number with a unit; an empty value turns it off), and re-image."""
        if self.obs is None:
            print("uvtaper: no data loaded")
            return None
        text, radius = self._uv_text("value"), self._uv_text("radius") or 0.0

        def go():
            # Read inside the run, so a typo is reported rather than raised.
            value = float(text or 0.0)
            self.obs.uvtaper(value, radius if value else 0.0)
            self._uv_report("uvtaper")
            return True

        return self._run("uvtaper", go, data_changed=False)

    def _uv_report(self, what):
        """Re-image and print the beam that the UV range and taper now give."""
        from difmapy.units import format_uvdist

        obs = self.obs
        obs.invert(verbose=False)
        bmaj, bmin, bpa = obs.estimated_beam
        hi = f"{format_uvdist(obs._uvmax)} M-lambda" if obs._uvmax else "no limit"
        taper = (f"taper {obs._gauval:g} at {format_uvdist(obs._gaurad)} M-lambda"
                 if 0.0 < obs._gauval < 1.0 and obs._gaurad > 0.0 else "no taper")
        print(f"{what}: uv range {format_uvdist(obs._uvmin)} M-lambda to {hi}, {taper}; "
              f"beam {bmaj:.4g} x {bmin:.4g} mas at {bpa:.4g} deg")

    def run_selfcal(self):
        """Self-calibrate with what the Selfcal row says: phase and/or amplitude, at the solution interval typed (minutes as a bare number, or a string with a unit, or "scan"; empty for one solution per integration)."""
        if not self._need_model("selfcal"):
            return None
        amp, phase = self.amp_check.isChecked(), self.phase_check.isChecked()
        if not (amp or phase):
            print("selfcal: tick phase, amplitude or both")
            return None
        solint = self.solint_edit.text().strip() or 0.0
        return self._run("selfcal", lambda: self.obs.selfcal(
            amp=amp, phase=phase, solint=solint, quiet=False))

    def run_gscale(self):
        """Run `gscale`, floating the flux scale if the box is ticked."""
        if not self._need_model("gscale"):
            return None
        return self._run("gscale", lambda: self.obs.gscale(
            float_scale=self.float_check.isChecked(), quiet=False))

    def run_bayes(self):
        """Run `bayes_gscale` with the source models ticked in the Bayes row."""
        if not self._need_model("bayes_gscale"):
            return None
        models = [name for name, cb in self.model_checks.items() if cb.isChecked()]
        if not models:
            print("bayes_gscale: tick at least one source model")
            return None
        return self._run("bayes_gscale", lambda: self.obs.bayes_gscale(
            models=tuple(models), plot=False, quiet=False))

    def save_file(self, kind, path=None):
        """Save one product of the session (`kind` is a key of `SAVERS`). Without a `path` a file dialog asks for one. Returns the path written, or None."""
        if self.obs is None:
            print("save: no data loaded")
            return None
        label, title, filt, suffix, write = SAVERS[kind]
        if path is None:
            if callable(suffix):
                suffix = suffix(self.obs)
            base = os.path.splitext(self.obs.filename or "difmapy")[0]
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, base + suffix, filt)
            if not path:
                return None

        def go():
            write(self.obs, path)
            return True

        if not self._run(f"save {label.lower()}", go):
            return None
        print(f"saved {label.lower()}: {path}")
        return path

    # ---- keeping the panels in step ------------------------------------

    def _reload(self, plot):
        try:
            plot.reload()
        except Exception as exc:
            print(f"dashboard: could not reload {plot.windowTitle()}: {exc}")

    def _changed_in(self, plot):
        """An embedded plot changed the data or the model: bring the others up to date."""
        self.sync(source=plot)

    def sync(self, source=None):
        """Reload every panel on show except `source`; the plots in hidden tabs are reloaded when next shown."""
        for panel in self.panels:
            panel.sync(source)

    def reload(self):
        """Reload every plot (``r``, or the Reload all button)."""
        self.sync()

    def plots(self) -> list:
        """The embedded plots built so far."""
        return [p for panel in self.panels for p in panel.plots.values()]

    def key_help(self):
        return [("r", "reload every panel"),
                ("", "click into a plot to use its own keys (h there lists them)")]

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "r":
            self.reload()
        elif key == "h":
            self.toggle_help()
        # Nothing else: a stray q or z typed on a button must not close
        # the dashboard or move a plot it was not aimed at.

    def closeEvent(self, ev):
        self._tee_timer.stop()
        _restore_stdout(self._tee)
        for plot in self.plots():
            plot.on_change = None
        super().closeEvent(ev)


def _restore_stdout(tee):
    """Take `tee` off ``sys.stdout``, if it is still on top."""
    if tee is not None and sys.stdout is tee:
        sys.stdout = tee._original


def _native_suffix(casa, aips):
    """A default suffix that depends on the data's native table format."""
    return lambda obs: casa if obs.caltable_formats()[0] == "casa" else aips


def _save_session(obs, path):
    root, ext = os.path.splitext(path)
    obs.save(root if ext.lower() in (".uvf", ".fits", ".ms") else path)


#: What the output tab's buttons save: kind -> (button label, dialog title, file filter, default suffix (or a function of the observation giving it), writer(obs, path)).
SAVERS = {
    "clean": ("Restored map", "Save the restored (CLEAN) map", "FITS (*.fits)", ".clean.fits",
              lambda obs, p: obs.wmap(p)),
    "dirty": ("Dirty map", "Save the residual (dirty) map", "FITS (*.fits)", ".dirty.fits",
              lambda obs, p: obs.wdmap(p)),
    "beam": ("Dirty beam", "Save the dirty beam", "FITS (*.fits)", ".beam.fits",
             lambda obs, p: obs.wbeam(p)),
    "model": ("Model", "Save the model", "difmap model (*.mod)", ".mod",
              lambda obs, p: obs.wmodel(p)),
    "uvfits": ("Visibilities", "Save the calibrated visibilities", "UVFITS (*.uvf *.uvfits)", ".uvf",
               lambda obs, p: obs.wobs(p)),
    "caltable": ("Calibration table", "Save the calibration table", "All files (*)",
                 _native_suffix(".G", ".TASAV.FITS"), lambda obs, p: obs.savecaltable(p)),
    "flags": ("Flags", "Save the flags added in the session", "All files (*)",
              _native_suffix(".flagcmd", ".FG.TASAV.FITS"), lambda obs, p: obs.wflags(p)),
    "session": ("Whole session", "Save the whole session (prefix)", "All files (*)", "",
                _save_session),
}


def dashboard(obs=None, block=None, on_load=None):
    """Open the dashboard: every view of a session in one window, kept in step, with controls to load a file and calibrate it (see the module docstring and `Observation.dashboard`).

    Parameters
    ----------
    obs : Observation | None, default None
        The observation to show; ``None`` opens the dashboard empty, to load a file from its Main tab.
    block : bool | None, default None
        Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.
    on_load : callable | None, default None
        Called with every observation loaded from the Main tab (the ``difmapy`` prompt uses it to make that observation its `obs`).

    Returns
    -------
    Dashboard
        The dashboard window; its `obs` is the observation shown.
    """
    p = Dashboard(obs, on_load=on_load)
    run_if_needed(p, block)
    return p
