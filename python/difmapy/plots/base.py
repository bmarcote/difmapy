"""Shared plotting infrastructure (Qt app handling, flag interaction).

Interaction conventions (all point-based plots):

* Shift + drag: flag the points inside the rubber band
* Ctrl + drag: unflag the points inside the rubber band
* hover + ``f`` / ``u``: flag / unflag the nearest point
* ``z`` / ``r``: undo / redo the last flag edit
* ``x``: show or hide the flagged points (hidden by default)
* ``h``: the key legend of the active plot
* ``q``: close the window

Flag edits are applied to the underlying data immediately (difmap
behaviour) and the plot refreshes in place; every edit is recorded
exactly, as a snapshot of the affected rows of the FLAG column, so that
undo restores precisely what was there before.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

IF_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34),
    (23, 190, 207), (214, 39, 40),
]
FLAG_COLOR = (220, 40, 40)
UNFLAG_COLOR = (40, 160, 40)
HIGHLIGHT_COLOR = (255, 140, 0)
MODEL_COLOR = (214, 30, 30)


def ensure_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = pg.mkQApp("difmapy")
    return app


def gradient_colors(n, name="viridis", lo=0.0, hi=0.88):
    """`n` colours sampled evenly from a perceptual colour map.

    Used to colour points by spectral window, where the ordering of the
    IFs is meaningful and a gradient reads better than arbitrary hues.
    """
    n = max(int(n), 1)
    try:
        cmap = pg.colormap.get(name)
        lut = cmap.getLookupTable(lo, hi, n, alpha=False)
        return [tuple(int(v) for v in row[:3]) for row in lut]
    except Exception:  # pragma: no cover - colormap missing
        return [IF_COLORS[i % len(IF_COLORS)] for i in range(n)]


class SelectViewBox(pg.ViewBox):
    """A ViewBox that reports Shift/Ctrl rubber-band drags.

    Plain drags keep pyqtgraph's usual pan/zoom behaviour; holding
    Shift (flag) or Ctrl (unflag) instead sweeps out a selection box
    and emits `sigSelected(x0, x1, y0, y1, flag)` on release.
    """

    sigSelected = QtCore.Signal(float, float, float, float, bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sel_origin = None
        self._sel_flag = True
        self._sel_rect = None
        #: identifies the panel this view box belongs to (set by the window)
        self.panel = None

    def mouseDragEvent(self, ev, axis=None):
        mods = ev.modifiers()
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        if not (shift or ctrl):
            self._clear_rect()
            return super().mouseDragEvent(ev, axis=axis)

        ev.accept()
        pos = self.mapToView(ev.pos())
        if ev.isStart():
            self._sel_origin = (pos.x(), pos.y())
            self._sel_flag = shift
            self._sel_rect = QtWidgets.QGraphicsRectItem()
            self._sel_rect.setPen(
                pg.mkPen(
                    FLAG_COLOR if shift else UNFLAG_COLOR,
                    style=QtCore.Qt.PenStyle.DashLine,
                )
            )
            self.addItem(self._sel_rect, ignoreBounds=True)
        if self._sel_origin is None:
            return
        x0, y0 = self._sel_origin
        x1, y1 = pos.x(), pos.y()
        if self._sel_rect is not None:
            self._sel_rect.setRect(
                QtCore.QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            )
        if ev.isFinish():
            flag = self._sel_flag
            self._clear_rect()
            self._sel_origin = None
            self.sigSelected.emit(
                min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1), flag
            )

    def _clear_rect(self):
        if self._sel_rect is not None:
            self.removeItem(self._sel_rect)
            self._sel_rect = None


def run_if_needed(widget, block):
    """Show the widget, entering the Qt event loop if nothing else is
    driving it.

    In a plain script the loop must be run or the window would vanish
    immediately; under IPython/Jupyter with the Qt event loop hook
    enabled (``%gui qt``) it must *not* be run, or the session would
    freeze. Pass `block` explicitly to override the detection.
    """
    widget.show()
    if block is None:
        block = not _qt_loop_hooked()
    if block:
        pg.exec()


def _qt_loop_hooked() -> bool:
    """True if an interactive shell is already pumping Qt events."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    # IPython records the active GUI integration here ("qt", "qt5"...).
    gui = getattr(shell, "active_eventloop", None)
    return bool(gui) and str(gui).startswith("qt")


# ----------------------------------------------------------------------
# help overlay
# ----------------------------------------------------------------------


class PlotWindow(QtWidgets.QMainWindow):
    """Base window: a key legend on ``h`` and closing on ``q``."""

    def __init__(self, title):
        ensure_app()
        super().__init__()
        self.setWindowTitle(title)
        self._help = None

    # Subclasses list their own keys first; these are appended.
    COMMON_KEYS = (
        ("h", "show or hide this help"),
        ("q", "close the window"),
    )

    def key_help(self) -> list[tuple[str, str]]:  # pragma: no cover - trivial
        """The keys this plot understands, as (key, description) pairs."""
        return []

    def help_text(self) -> str:
        rows = "".join(
            f"<tr><td style='padding-right:14px'><b>{k}</b></td><td>{d}</td></tr>"
            for k, d in list(self.key_help()) + list(self.COMMON_KEYS)
        )
        return (
            f"<div style='font-size:11pt'><b>{self.windowTitle()}</b>"
            f"<table style='margin-top:6px'>{rows}</table></div>"
        )

    def toggle_help(self):
        if self._help is not None:
            self._help.deleteLater()
            self._help = None
            return False
        lbl = QtWidgets.QLabel(self.help_text(), self)
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl.setStyleSheet(
            "background: rgba(255,255,255,238); color: #111;"
            "border: 1px solid #888; padding: 12px;"
        )
        lbl.adjustSize()
        self._help = lbl
        self._place_help()
        lbl.show()
        lbl.raise_()
        return True

    def _place_help(self):
        if self._help is None:
            return
        w, h = self._help.width(), self._help.height()
        self._help.move(max(0, (self.width() - w) // 2),
                        max(0, (self.height() - h) // 2))

    def resizeEvent(self, ev):  # pragma: no cover - Qt callback
        super().resizeEvent(ev)
        self._place_help()

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "h":
            self.toggle_help()
        elif key == "q":
            self.close()
        else:
            super().keyPressEvent(ev)


# ----------------------------------------------------------------------
# undo/redo of flag edits
# ----------------------------------------------------------------------


class EditHistory:
    """Exact undo/redo for interactive flag editing.

    Each edit is stored as before/after snapshots of the FLAG column
    rows it touched, so undo restores the previous state even where an
    edit hit data that was already partly flagged. Deleted samples
    (weight 0) stay flagged whatever happens, as everywhere else.
    """

    def __init__(self, obs, limit=100):
        self.obs = obs
        self.limit = int(limit)
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []

    def __len__(self):
        return len(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def apply(self, ops) -> int:
        """Apply `ops`, a sequence of (rows, if_index, flag) triples, as
        one undoable step. Returns the number of samples changed."""
        core = self.obs._core
        ops = [(np.unique(np.asarray(r, dtype=int)), c, bool(f)) for r, c, f in ops]
        rows = sorted({int(r) for rr, _, _ in ops for r in rr})
        if not rows:
            return 0
        before = np.array(core.flags_rows(rows), copy=True)
        for rr, cif, flag in ops:
            core.edit_rows(
                [int(r) for r in rr], flag,
                if_index=None if cif is None else int(cif), sel_chan=True,
            )
        after = np.array(core.flags_rows(rows), copy=True)
        nchanged = int((before != after).sum())
        if nchanged:
            self._undo.append((rows, before, after))
            del self._undo[: -self.limit]
            self._redo.clear()
        self.obs._dirty()
        return nchanged

    def _restore(self, rows, block):
        self.obs._core.set_flags_rows(rows, np.ascontiguousarray(block, dtype=bool))
        self.obs._dirty()

    def undo(self):
        """Undo the last edit; returns how many samples changed, or None."""
        if not self._undo:
            return None
        rows, before, after = self._undo.pop()
        self._restore(rows, before)
        self._redo.append((rows, before, after))
        return int((before != after).sum())

    def redo(self):
        """Redo the last undone edit; returns how many samples changed."""
        if not self._redo:
            return None
        rows, before, after = self._redo.pop()
        self._restore(rows, after)
        self._undo.append((rows, before, after))
        return int((before != after).sum())


# ----------------------------------------------------------------------
# colouring
# ----------------------------------------------------------------------


class ColorScheme:
    """How the points of a flag plot are split into coloured groups.

    `of(obs, name)` builds one from a user-facing name; calling it with
    a collected data dict returns ``(labels, keys, colors)`` where
    `keys` is an integer group per point.
    """

    def __init__(self, name, labels, keyfunc, colors):
        self.name = name
        self.labels = labels
        self._keyfunc = keyfunc
        self.colors = colors

    def keys(self, d):
        return self._keyfunc(d)

    @classmethod
    def of(cls, obs, name, nbins=8):
        name = (name or "none").lower()
        if name in ("spw", "if", "ifs"):
            nif = obs.nif
            return cls(
                "spw", [f"IF {i + 1}" for i in range(nif)],
                lambda d: np.asarray(d["cif"], dtype=int), gradient_colors(nif),
            )
        if name in ("baseline", "bl"):
            names = obs.antennas
            pairs = sorted({(int(a), int(b)) for a, b in
                            zip(*_baseline_arrays(obs))})
            index = {p: i for i, p in enumerate(pairs)}

            def keyfunc(d, index=index):
                a = np.asarray(d["a1"], dtype=int)
                b = np.asarray(d["a2"], dtype=int)
                return np.array([index.get((int(x), int(y)), 0)
                                 for x, y in zip(a, b)], dtype=int)

            return cls("baseline",
                       [f"{names[a]}-{names[b]}" for a, b in pairs],
                       keyfunc, gradient_colors(len(pairs), "turbo"))
        if name in ("antenna", "ant"):
            names = obs.antennas
            return cls(
                "antenna", list(names),
                lambda d: np.asarray(d["a1"], dtype=int),
                gradient_colors(len(names), "turbo"),
            )
        if name == "time":
            def keyfunc(d, nbins=nbins):
                t = np.asarray(d["time"], dtype=float)
                if t.size == 0 or t.max() == t.min():
                    return np.zeros(t.shape, dtype=int)
                f = (t - t.min()) / (t.max() - t.min())
                return np.clip((f * nbins).astype(int), 0, nbins - 1)

            return cls("time", [f"t{i}" for i in range(nbins)], keyfunc,
                       gradient_colors(nbins, "plasma"))
        return cls("none", ["data"],
                   lambda d: np.zeros(len(d["wt"]), dtype=int),
                   [IF_COLORS[0]])


def _baseline_arrays(obs):
    _, a1, a2, _, _, _ = obs._core.rows()
    return np.asarray(a1, dtype=int), np.asarray(a2, dtype=int)


# ----------------------------------------------------------------------
# the flag-plot base class
# ----------------------------------------------------------------------


class Panel:
    """One plot panel: which quantity it draws and, for paged plots,
    which subset of the points belongs to it."""

    __slots__ = ("key", "plot", "vb", "group", "label")

    def __init__(self, key, plot, vb, group=None, label=""):
        self.key = key
        self.plot = plot
        self.vb = vb
        self.group = group
        self.label = label


class FlagPlotBase(PlotWindow):
    """Stacked scatter panels with interactive flagging.

    Subclasses provide `_collect()`, returning a dict of parallel 1-D
    arrays: the per-point ``wt`` (signed weights), ``row`` and ``cif``
    indices, an ``x``, and one array per panel quantity; plus optional
    ``time``/``a1``/``a2`` used by the colour schemes. `_build_panels()`
    lays the panels out and returns them.
    """

    #: (data key, axis label) pairs of the quantities this plot can draw
    QUANTITIES = {
        "amp": "Amplitude (Jy)",
        "phase": "Phase (deg)",
    }

    def __init__(self, obs, title, colorby="spw", legend=False,
                 show_flagged=False):
        super().__init__(title)
        self.obs = obs
        self.history = EditHistory(obs)
        self.colors = ColorScheme.of(obs, colorby)
        self._show_flagged = bool(show_flagged)
        self._hidden_groups: set[int] = set()
        self._items: list = []
        self._panels: list[Panel] = []
        self._data = None
        self._mouse_pos = None
        self._active = None

        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        lay.addWidget(self.glw, 1)
        self._legend_box = None
        if legend:
            lay.addWidget(self._make_legend())
        self.setCentralWidget(central)
        self._proxy = pg.SignalProxy(
            self.glw.scene().sigMouseMoved, rateLimit=30, delay=0,
            slot=self._mouse_moved,
        )
        self.refresh()

    # ---- layout -------------------------------------------------------

    def _make_legend(self):
        """A column of check boxes toggling the colour groups."""
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(4, 4, 4, 4)
        v.addWidget(QtWidgets.QLabel(f"<b>{self.colors.name}</b>"))
        self._checks = []
        for i, label in enumerate(self.colors.labels):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            r, g, b = self.colors.colors[i % len(self.colors.colors)]
            cb.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold;")
            cb.stateChanged.connect(
                lambda state, k=i: self._toggle_group(k, bool(state))
            )
            v.addWidget(cb)
            self._checks.append(cb)
        v.addStretch(1)
        self._legend_box = box
        return box

    def _toggle_group(self, key, on):
        if on:
            self._hidden_groups.discard(key)
        else:
            self._hidden_groups.add(key)
        self.refresh()

    def _add_panel(self, row, key, group=None, label=None, ylabel=None):
        vb = SelectViewBox()
        plot = self.glw.addPlot(row=row, col=0, viewBox=vb)
        plot.setLabel("left", ylabel or self.QUANTITIES.get(key, key))
        plot.showGrid(x=True, y=True, alpha=0.15)
        if key == "phase":
            plot.setYRange(-180, 180)
        panel = Panel(key, plot, vb, group=group, label=label or "")
        vb.panel = panel
        vb.sigSelected.connect(
            lambda x0, x1, y0, y1, flag, p=panel: self._on_selected(p, x0, x1, y0, y1, flag)
        )
        plot.scene().sigMouseMoved.connect(
            lambda pos, p=panel: self._note_active(p, pos)
        )
        self._panels.append(panel)
        return panel

    def _build_panels(self):  # pragma: no cover - abstract
        raise NotImplementedError

    # ---- data plumbing (subclass API) ---------------------------------

    def _collect(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def _model(self, d, key):
        """Model values for panel `key`, or None if there is no model."""
        return None

    def refresh(self):
        self._data = self._collect()
        d = self._data
        for item, plot in self._items:
            plot.removeItem(item)
        self._items = []
        if not self._panels:
            self._build_panels()
        gkeys = self.colors.keys(d)
        good = d["wt"] > 0
        for panel in self._panels:
            sub = self._panel_mask(panel, d)
            y = d[panel.key]
            for k in np.unique(gkeys[sub]) if sub.any() else []:
                if int(k) in self._hidden_groups:
                    continue
                m = sub & good & (gkeys == k)
                if not m.any():
                    continue
                col = self.colors.colors[int(k) % len(self.colors.colors)]
                self._add_item(
                    panel,
                    pg.ScatterPlotItem(
                        d["x"][m], y[m], size=4, pen=None,
                        brush=pg.mkBrush(*col, 190),
                    ),
                )
            if self._show_flagged:
                m = sub & ~good & (d["wt"] != 0)
                if m.any():
                    self._add_item(
                        panel,
                        pg.ScatterPlotItem(
                            d["x"][m], y[m], size=6, symbol="x",
                            pen=pg.mkPen(*FLAG_COLOR), brush=None,
                        ),
                    )
            self._decorate(panel, d, sub, good)
        self._update_status()

    def _panel_mask(self, panel, d):
        if panel.group is None or "group" not in d:
            return np.ones(d["wt"].shape, dtype=bool)
        return np.asarray(d["group"]) == panel.group

    def _add_item(self, panel, item):
        panel.plot.addItem(item)
        self._items.append((item, panel.plot))

    def _decorate(self, panel, d, sub, good):
        """Hook for subclasses: model overlays, highlights, labels."""
        model = self._model(d, panel.key)
        if model is not None:
            m = sub & good & np.isfinite(model)
            if m.any():
                self._add_item(
                    panel,
                    pg.ScatterPlotItem(
                        d["x"][m], model[m], size=2.5, pen=None,
                        brush=pg.mkBrush(*MODEL_COLOR, 220),
                    ),
                )

    # ---- interaction --------------------------------------------------

    def _mouse_moved(self, evt):
        self._mouse_pos = evt[0]

    def _note_active(self, panel, pos):
        if panel.plot.sceneBoundingRect().contains(pos):
            self._active = panel

    def _view_coords(self, panel, scene_pos):
        p = panel.vb.mapSceneToView(scene_pos)
        return p.x(), p.y()

    def _on_selected(self, panel, x0, x1, y0, y1, flag):
        n = self._apply_box(panel, x0, x1, y0, y1, flag)
        self._message(f"{'flagged' if flag else 'unflagged'} {n} samples")

    def key_help(self):
        return [
            ("shift+drag", "flag the points inside the box"),
            ("ctrl+drag", "unflag the points inside the box"),
            ("f / u", "flag / unflag the point nearest the cursor"),
            ("z / r", "undo / redo the last flag edit"),
            ("x", "show or hide the flagged points"),
            ("drag / wheel", "pan / zoom"),
        ]

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "x":
            self._show_flagged = not self._show_flagged
            self.refresh()
        elif key in ("f", "u") and self._mouse_pos is not None:
            panel = self._active or (self._panels[0] if self._panels else None)
            if panel is not None:
                x, y = self._view_coords(panel, self._mouse_pos)
                self._flag_nearest(panel, x, y, flag=(key == "f"))
        elif key == "z":
            n = self.history.undo()
            self.refresh()
            self._message("nothing to undo" if n is None
                          else f"undid an edit of {n} samples")
        elif key == "r":
            n = self.history.redo()
            self.refresh()
            self._message("nothing to redo" if n is None
                          else f"redid an edit of {n} samples")
        else:
            super().keyPressEvent(ev)

    # ---- flagging -----------------------------------------------------

    def _apply_box(self, panel, x0, x1, y0, y1, flag) -> int:
        """Flag/unflag the points inside a box; returns how many."""
        d = self._data
        y = d[panel.key]
        inside = (d["x"] >= x0) & (d["x"] <= x1) & (y >= y0) & (y <= y1)
        inside &= self._panel_mask(panel, d)
        # Flag only currently-good points; unflag only flagged ones.
        inside &= (d["wt"] > 0) if flag else (d["wt"] < 0)
        idx = np.nonzero(inside)[0]
        return self._edit_points(idx, flag) if idx.size else 0

    def _flag_nearest(self, panel, x, y, flag):
        d = self._data
        cand = (d["wt"] > 0) if flag else (d["wt"] < 0)
        cand &= self._panel_mask(panel, d)
        if not cand.any():
            return 0
        (xr0, xr1), (yr0, yr1) = panel.vb.viewRange()
        dx = (d["x"] - x) / max(xr1 - xr0, 1e-30)
        dy = (d[panel.key] - y) / max(yr1 - yr0, 1e-30)
        dist = np.where(cand, dx * dx + dy * dy, np.inf)
        n = self._edit_points([int(np.argmin(dist))], flag)
        self._message(f"{'flagged' if flag else 'unflagged'} {n} samples")
        return n

    def _edit_ops(self, idx, flag):
        """The (rows, if_index, flag) edits for the selected points."""
        d = self._data
        idx = np.asarray(idx, dtype=int)
        ops = []
        for cif in np.unique(d["cif"][idx]):
            rows = d["row"][idx][d["cif"][idx] == cif]
            ops.append((np.unique(rows), int(cif), bool(flag)))
        return ops

    def _edit_points(self, idx, flag) -> int:
        n = self.history.apply(self._edit_ops(idx, flag))
        self.refresh()
        return n

    # ---- status -------------------------------------------------------

    def _message(self, text):
        self.statusBar().showMessage(text)

    def _status(self) -> str:
        hidden = "" if self._show_flagged else " | flagged data hidden (x)"
        return (f"colour: {self.colors.name}{hidden} | h: help")

    def _update_status(self):
        self._message(self._status())
