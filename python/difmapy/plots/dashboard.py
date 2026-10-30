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
        # The UV range and taper are set in the Main tab, so the map
        # panel goes without its own copy of those fields.
        "map": (("mapplot", lambda o: MapPlot(o, uv_controls=False)),),
    }


class _Tee(QtCore.QObject):
    """A stand-in for ``sys.stdout`` that passes everything on to the real one and also emits it, for the output tab. The signal is delivered in the GUI thread whichever thread printed."""

    text = QtCore.Signal(str)

    def __init__(self, original):
        super().__init__()
        self._original = original

    def write(self, s):
        # The tab first: whatever happens to the stream below, the output
        # is shown. And printing must never fail what printed: a report
        # with a "±" on an ASCII stdout (the app started from Finder, a
        # terminal without a UTF-8 locale) raised UnicodeEncodeError - a
        # ValueError - which mapplot took for a failed fit and undid it.
        if s:
            self.text.emit(s)
        if self._original is not None:
            try:
                self._original.write(s)
            except UnicodeEncodeError:
                enc = getattr(self._original, "encoding", None) or "ascii"
                self._original.write(s.encode(enc, "backslashreplace").decode(enc))
            except (OSError, ValueError):  # a closed or broken stream
                pass
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
        from difmapy.plots import style

        self.setTabBarAutoHide(True)
        # Native tabs (a segmented control) in the macOS app; the flat
        # document-mode bar, which the web style underlines, elsewhere.
        self.setDocumentMode(not style.native_controls())
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
        from difmapy.plots import style

        self.tabs.setObjectName("dashboardControls")
        self.tabs.setDocumentMode(not style.native_controls())
        self.tabs.addTab(self._make_main(), "Main")
        self.tabs.addTab(self._make_output(), "Output")
        self.apply_style()

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
        """The Main tab: cards of controls, each action's options under its button."""
        from difmapy.plots import style

        page = QtWidgets.QWidget()
        page.setObjectName("controlsPage")
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("controlsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        body = QtWidgets.QWidget()
        body.setObjectName("controlsBody")
        col = QtWidgets.QVBoxLayout(body)
        col.setContentsMargins(14, 10, 14, 14)
        col.setSpacing(10)

        # -- data --
        col.addWidget(_section("DATA"))
        data, lay = _card()
        row = QtWidgets.QHBoxLayout()
        self.file_button = QtWidgets.QPushButton("File\u2026")
        self.file_button.setObjectName("action")
        self.file_button.setToolTip("choose a UVFITS file, or a Measurement Set (a folder ending in .ms)")
        self.file_button.clicked.connect(lambda: self._browse())
        row.addWidget(self.file_button)
        self.file_edit = QtWidgets.QLineEdit()
        self.file_edit.setPlaceholderText("a UVFITS file or a Measurement Set")
        if self.obs is not None and self.obs.path:
            self.file_edit.setText(os.fspath(self.obs.path))
        row.addWidget(self.file_edit, 1)
        lay.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.timeavg_edit = QtWidgets.QLineEdit()
        self.timeavg_edit.setPlaceholderText("none (e.g. 10s, 2min)")
        self.freqavg_edit = QtWidgets.QLineEdit()
        self.freqavg_edit.setPlaceholderText('none (channels, or "all")')
        row.addLayout(_labelled("Time average", self.timeavg_edit), 1)
        row.addLayout(_labelled("Frequency average", self.freqavg_edit), 1)
        self.load_button = QtWidgets.QPushButton("Load")
        self.load_button.setObjectName("primary")
        self.load_button.setDefault(True)
        if style.native_controls():
            self.file_button.setDefault(True)
            self.file_button.setAutoDefault(False)
        self.load_button.clicked.connect(lambda: self.load())
        row.addWidget(self.load_button, 0, QtCore.Qt.AlignmentFlag.AlignBottom)
        lay.addLayout(row)
        col.addWidget(data)

        # -- calibration: one card per command, its options under it --
        col.addWidget(_section("CALIBRATION"))
        cal = QtWidgets.QHBoxLayout()
        cal.setSpacing(10)
        card, lay, self.selfcal_button = _action_card("Selfcal", self.run_selfcal)
        self.phase_check = QtWidgets.QCheckBox("Phase")
        self.phase_check.setChecked(True)
        self.amp_check = QtWidgets.QCheckBox("Amplitude")
        checks = QtWidgets.QHBoxLayout()
        checks.addWidget(self.phase_check)
        checks.addWidget(self.amp_check)
        checks.addStretch(1)
        lay.addLayout(checks)
        # selfcal(converge=True): alternate with modelfit until the fit
        # stops improving. It needs a model with free parameters; with
        # CLEAN components alone, one selfcal round is run instead.
        self.converge_check = QtWidgets.QCheckBox("Converge with modelfit")
        self.converge_check.setChecked(True)
        self.converge_check.setToolTip(
            "alternate selfcal and modelfit until the model-data fit stops improving "
            "(needs fitted components; a CLEAN-only model gets one selfcal round)")
        checks = QtWidgets.QHBoxLayout()
        checks.addWidget(self.converge_check)
        checks.addStretch(1)
        lay.addLayout(checks)
        self.solint_edit = QtWidgets.QLineEdit()
        self.solint_edit.setPlaceholderText('per integration ("30s", "scan")')
        lay.addLayout(_labelled("Solution interval", self.solint_edit))
        lay.addStretch(1)
        cal.addWidget(card, 1)
        card, lay, self.gscale_button = _action_card("Gscale", self.run_gscale)
        self.float_check = QtWidgets.QCheckBox("Float scale")
        self.float_check.setChecked(True)
        lay.addWidget(self.float_check)
        lay.addStretch(1)
        cal.addWidget(card, 1)
        card, lay, self.bayes_button = _action_card("Bayes", self.run_bayes)
        self.bayes_button.setToolTip("Bayesian gscale (bayes_gscale) over the source models ticked")
        grid = QtWidgets.QGridLayout()
        self.model_checks = {}
        for i, (name, on) in enumerate(BAYES_MODELS):
            cb = QtWidgets.QCheckBox(name.capitalize().replace("Gauss", "Gauss "))
            cb.setChecked(on)
            grid.addWidget(cb, i // 2, i % 2)
            self.model_checks[name] = cb
        lay.addLayout(grid)
        lay.addStretch(1)
        cal.addWidget(card, 1)
        col.addLayout(cal)

        # -- imaging: the UV range and the taper, in M-lambda unless a
        # unit is typed ("500k"); the button (or Enter) applies --
        col.addWidget(_section("IMAGING"))
        img = QtWidgets.QHBoxLayout()
        img.setSpacing(10)
        self.uv_edits = {}

        def uv_card(name, apply, fields):
            card, lay, button = _action_card(name, apply)
            row = QtWidgets.QHBoxLayout()
            for key, label, placeholder, tip in fields:
                e = QtWidgets.QLineEdit()
                e.setPlaceholderText(placeholder)
                e.setToolTip(tip)
                e.returnPressed.connect(lambda: apply())
                self.uv_edits[key] = e
                row.addLayout(_labelled(label, e), 1)
            lay.addLayout(row)
            hint = QtWidgets.QLabel("M\u03bb, or with a unit: 500k, 2e7 lambda")
            hint.setObjectName("hint")
            lay.addWidget(hint)
            img.addWidget(card, 1)
            return button

        self.uvrange_button = uv_card("UV range", self.run_uvrange, (
            ("uvmin", "Min", "0", "shortest UV radius imaged: M-lambda, or with a unit (500k)"),
            ("uvmax", "Max", "no limit", "longest UV radius imaged; empty for no limit")))
        self.uvtaper_button = uv_card("Taper", self.run_uvtaper, (
            ("value", "Value", "none", "the taper at the radius, between 0 and 1; empty for no taper"),
            ("radius", "Radius", "", "UV radius where the taper falls to the value: M-lambda, or with a unit")))
        col.addLayout(img)
        self._sync_uv_fields()

        col.addStretch(1)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        reload_all = QtWidgets.QPushButton("Reload all")
        reload_all.setToolTip("re-read the data in every panel (after changes made at the prompt)")
        reload_all.clicked.connect(lambda: self.reload())
        row.addWidget(reload_all)
        col.addLayout(row)
        scroll.setWidget(body)
        outer = QtWidgets.QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
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
        """The Output tab: what the session prints, and the buttons that save its products."""
        page = QtWidgets.QWidget()
        page.setObjectName("controlsPage")
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(14, 10, 14, 14)
        lay.setSpacing(8)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setObjectName("output")
        self.output.setReadOnly(True)
        self.output.setFont(QtGui.QFontDatabase.systemFont(
            QtGui.QFontDatabase.SystemFont.FixedFont))
        lay.addWidget(self.output, 1)
        lay.addWidget(_section("SAVE:"))
        card, inner = _card()
        buttons = QtWidgets.QGridLayout()
        buttons.setSpacing(8)
        self.save_buttons = {}
        for i, (kind, (label, *_)) in enumerate(SAVERS.items()):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self.save_file(k))
            buttons.addWidget(b, i // 4, i % 4)
            self.save_buttons[kind] = b
        inner.addLayout(buttons)
        lay.addWidget(card)
        return page

    def apply_style(self, glass=None):
        """Style the controls for the platform and the system's light or dark appearance (`difmapy.plots.style`). The window is opaque; on macOS ``$DIFMAPY_GLASS=1`` puts it on a glass background instead (`difmapy.plots.macglass`)."""
        from difmapy.plots import macglass, style

        if glass is None:
            glass = macglass.enable(self)
        self.glass = int(glass)
        style.apply(QtWidgets.QApplication.instance(), glass=bool(self.glass))
        return self.glass

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

    def _file_dialog(self, title, save=False, filters=(), suffix=""):
        """A native file dialog, opened asynchronously (`QFileDialog.open`) rather than through the static helpers. Those run the dialog in a nested modal loop, and under IPython's Qt input hook on macOS that loop is ended - and the native panel closed at once - whenever the hook stops for terminal input."""
        dlg = QtWidgets.QFileDialog(self, title)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave if save
                          else QtWidgets.QFileDialog.AcceptMode.AcceptOpen)
        if not save:
            dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        if filters:
            dlg.setNameFilters(list(filters))
        if suffix:
            dlg.setDefaultSuffix(suffix.lstrip("."))
        # Kept, so that the dialog is not collected while it is open.
        self._dialog = dlg
        return dlg

    def _browse(self):
        """Choose the file to load with the platform's own dialog: a UVFITS file, or a Measurement Set - entering the MS folder accepts it, as does choosing the ``table.dat`` inside it."""
        dlg = self._file_dialog(
            "Open a UVFITS file or a Measurement Set",
            filters=("Visibility data (*.uvf *.uvfits *.fits *.UVF *.UVFITS *.FITS table.dat)",
                     "All files (*)"))
        start = self.file_edit.text().strip()
        if start:
            dlg.setDirectory(os.path.dirname(os.path.normpath(start)) or start)

        def chosen(path):
            if os.path.basename(path) == "table.dat":
                path = os.path.dirname(path)
            self.file_edit.setText(path)

        def entered(directory):
            # A Measurement Set is a folder: going into one means it.
            if _is_ms(directory):
                dlg.reject()
                self.file_edit.setText(directory)

        dlg.fileSelected.connect(chosen)
        dlg.directoryEntered.connect(entered)
        dlg.open()
        return dlg

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
        """Self-calibrate with what the Selfcal card says: phase and/or amplitude, at the solution interval typed (minutes as a bare number, or a string with a unit, or "scan"; empty for one solution per integration), and - with "Converge with modelfit", the default - alternating with `modelfit` until the fit converges (`selfcal(converge=True)`). That needs components with free parameters; a model of CLEAN components alone gets one selfcal round, with a note saying so."""
        if not self._need_model("selfcal"):
            return None
        amp, phase = self.amp_check.isChecked(), self.phase_check.isChecked()
        if not (amp or phase):
            print("selfcal: tick phase, amplitude or both")
            return None
        solint = self.solint_edit.text().strip() or 0.0
        converge = self.converge_check.isChecked()
        if converge and not self.obs.nvariable:
            print("selfcal: the model has no free parameters (CLEAN components only), so there is "
                  "nothing for modelfit to converge; running one selfcal round")
            converge = False
        return self._run("selfcal", lambda: self.obs.selfcal(
            amp=amp, phase=phase, solint=solint, converge=converge, quiet=False))

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
        """Save one product of the session (`kind` is a key of `SAVERS`). Without a `path` the platform's file dialog asks for one, and the file is written once it is chosen (None is returned then). Returns the path written, or None."""
        if self.obs is None:
            print("save: no data loaded")
            return None
        label, title, filt, suffix, write = SAVERS[kind]
        if path is None:
            if callable(suffix):
                suffix = suffix(self.obs)
            base = os.path.splitext(self.obs.filename or "difmapy")[0]
            dlg = self._file_dialog(title, save=True, filters=(filt,))
            if self.obs.path:
                dlg.setDirectory(os.path.dirname(os.path.normpath(os.fspath(self.obs.path))))
            dlg.selectFile(base + suffix)
            dlg.fileSelected.connect(lambda p: self.save_file(kind, p))
            dlg.open()
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


def _is_ms(path) -> bool:
    """True for a Measurement Set: a folder holding a casacore table."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "table.dat"))


def _section(text):
    """A small capitalised heading over a group of cards."""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("sectionTitle")
    return lbl


def _card():
    """A rounded card, and the layout to fill it with."""
    from difmapy.plots import style

    frame = QtWidgets.QFrame()
    frame.setObjectName("card")
    style.add_shadow(frame)
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)
    return frame, lay


def _action_card(name, slot):
    """A card headed by the button that runs `slot`; the options go under it. Returns (card, layout, button)."""
    frame, lay = _card()
    from difmapy.plots import style

    button = QtWidgets.QPushButton(name)
    button.setObjectName("action")
    if style.native_controls():
        # macOS draws a default button in the accent colour: the actions
        # stand out as in the web style, without grey bars.
        button.setDefault(True)
        button.setAutoDefault(False)
    button.clicked.connect(lambda: slot())
    lay.addWidget(button)
    return frame, lay, button


def _labelled(text, widget):
    """`widget` with a small caption above it."""
    box = QtWidgets.QVBoxLayout()
    box.setSpacing(3)
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("hint")
    box.addWidget(lbl)
    box.addWidget(widget)
    return box


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
