"""Shared plotting infrastructure (Qt app handling, flag interaction).

Interaction conventions (all point-based plots):

* Shift + drag: flag the points inside the rubber band
* Ctrl + drag: unflag the points inside the rubber band
* hover + ``f`` / ``u``: flag / unflag the nearest point
* flagged points are drawn as red crosses (toggle with ``x``)

Flag edits are applied to the underlying data immediately (difmap
behaviour) and the plot refreshes in place.
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


def ensure_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = pg.mkQApp("difmapy")
    return app


def run_if_needed(widget, block):
    """Show the widget; run the Qt loop when none is running."""
    widget.show()
    if block is None:
        # Block only when no event loop is pumping (plain scripts).
        block = not _loop_running()
    if block:
        pg.exec()


def _loop_running() -> bool:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    # IPython %gui qt integrates the loop; heuristics: an app with
    # visible top-level windows processing events.
    return bool(getattr(app, "_in_event_loop", False)) or hasattr(app, "_ipython_kernel")


class FlagScatterPlot(QtWidgets.QMainWindow):
    """Base window: scatter data with interactive flagging.

    Subclasses provide `_collect()` returning a dict with 1-D arrays:
    x, y, wt (signed weights), row (row index), cif (IF index),
    and set axis labels.
    """

    def __init__(self, obs, title):
        ensure_app()
        super().__init__()
        self.obs = obs
        self.setWindowTitle(title)
        self.pw = pg.PlotWidget()
        self.setCentralWidget(self.pw)
        self.pw.setBackground("w")
        self._show_flagged = True
        self._scatters = []
        self._data = None
        self._proxy = pg.SignalProxy(
            self.pw.scene().sigMouseMoved, rateLimit=30, delay=0,
            slot=self._mouse_moved,
        )
        self._mouse_pos = None
        self._rubber_origin = None
        self._rubber = None
        self._rubber_flag = True
        self.pw.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.statusBar().showMessage(
            "Shift+drag: flag box | Ctrl+drag: unflag box | f/u: nearest | x: toggle flagged"
        )
        self.refresh()

    # ---- data plumbing (subclass API) ----

    def _collect(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    def refresh(self):
        self._data = self._collect()
        d = self._data
        for s in self._scatters:
            self.pw.removeItem(s)
        self._scatters = []
        nif = self.obs.nif
        good = d["wt"] > 0
        for cif in range(nif):
            m = good & (d["cif"] == cif)
            if not m.any():
                continue
            s = pg.ScatterPlotItem(
                d["x"][m], d["y"][m], size=4, pen=None,
                brush=pg.mkBrush(*IF_COLORS[cif % len(IF_COLORS)], 180),
            )
            self.pw.addItem(s)
            self._scatters.append(s)
        if self._show_flagged and (~good).any():
            m = ~good & (d["wt"] != 0)
            s = pg.ScatterPlotItem(
                d["x"][m], d["y"][m], size=6, symbol="x",
                pen=pg.mkPen(*FLAG_COLOR), brush=None,
            )
            self.pw.addItem(s)
            self._scatters.append(s)

    # ---- interaction ----

    def _view_coords(self, scene_pos):
        vb = self.pw.getPlotItem().vb
        p = vb.mapSceneToView(scene_pos)
        return p.x(), p.y()

    def _mouse_moved(self, evt):
        self._mouse_pos = evt[0]
        if self._rubber is not None and self._rubber_origin is not None:
            x0, y0 = self._rubber_origin
            x1, y1 = self._view_coords(self._mouse_pos)
            self._rubber.setRect(
                QtCore.QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            )

    def _mouse_clicked(self, ev):
        mods = ev.modifiers()
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        if not (shift or ctrl):
            return
        if self._rubber is None:
            # Start the rubber band.
            self._rubber_origin = self._view_coords(ev.scenePos())
            self._rubber_flag = shift
            self._rubber = QtWidgets.QGraphicsRectItem()
            self._rubber.setPen(pg.mkPen(FLAG_COLOR if shift else (40, 160, 40),
                                         style=QtCore.Qt.PenStyle.DashLine))
            self.pw.getPlotItem().vb.addItem(self._rubber)
            ev.accept()
        else:
            # Finish: apply to points inside.
            x0, y0 = self._rubber_origin
            x1, y1 = self._view_coords(ev.scenePos())
            self.pw.getPlotItem().vb.removeItem(self._rubber)
            self._rubber = None
            self._rubber_origin = None
            self._apply_box(min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1),
                            self._rubber_flag)
            ev.accept()

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "x":
            self._show_flagged = not self._show_flagged
            self.refresh()
        elif key in ("f", "u") and self._mouse_pos is not None:
            x, y = self._view_coords(self._mouse_pos)
            self._flag_nearest(x, y, flag=(key == "f"))
        else:
            super().keyPressEvent(ev)

    # ---- flagging ----

    def _apply_box(self, x0, x1, y0, y1, flag):
        d = self._data
        inside = (d["x"] >= x0) & (d["x"] <= x1) & (d["y"] >= y0) & (d["y"] <= y1)
        # Flag only currently-good points; unflag only flagged ones.
        inside &= (d["wt"] > 0) if flag else (d["wt"] < 0)
        if not inside.any():
            return
        self._edit_points(np.nonzero(inside)[0], flag)

    def _flag_nearest(self, x, y, flag):
        d = self._data
        cand = (d["wt"] > 0) if flag else (d["wt"] < 0)
        if not cand.any():
            return
        vb = self.pw.getPlotItem().vb
        (xr0, xr1), (yr0, yr1) = vb.viewRange()
        dx = (d["x"] - x) / (xr1 - xr0)
        dy = (d["y"] - y) / (yr1 - yr0)
        dist = np.where(cand, dx * dx + dy * dy, np.inf)
        self._edit_points([int(np.argmin(dist))], flag)

    def _edit_points(self, idx, flag):
        d = self._data
        core = self.obs._core
        for cif in np.unique(d["cif"][idx]):
            rows = d["row"][idx][d["cif"][idx] == cif]
            core.edit_rows([int(r) for r in np.unique(rows)], bool(flag),
                           if_index=int(cif), sel_chan=True)
        self.obs._dirty()
        self.refresh()
