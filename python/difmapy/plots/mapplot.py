"""Interactive map display with CLEAN windows; difmap mapplot.

Interaction:

* double-click: add a CLEAN window centered on the cursor (drag its
  corner handles to resize; drag its body to move)
* ``d`` with the cursor over a window: delete it
* ``c``: run one clean batch (obs.clean with current defaults)
* ``i``: re-invert and redisplay
* ``m`` / ``b`` / ``r``: show residual map / beam / restored map

Windows are kept in sync with ``obs.windows`` (mas units).
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from difmapy.plots.base import ensure_app, run_if_needed

__all__ = ["mapplot", "MapPlot"]


class MapPlot(QtWidgets.QMainWindow):
    def __init__(self, obs, what="map", clean_args=None):
        ensure_app()
        super().__init__()
        self.obs = obs
        self.what = what
        self.clean_args = clean_args or {}
        self.setWindowTitle("difmapy mapplot")
        self.glw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.glw)
        self.glw.setBackground("w")
        self.plot = self.glw.addPlot()
        self.plot.setLabel("bottom", "Relative RA (mas)")
        self.plot.setLabel("left", "Relative Dec (mas)")
        self.plot.vb.setAspectLocked(True)
        self.plot.vb.invertX(True)  # RA increases leftward
        self.img = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.img)
        cbar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"))
        cbar.setImageItem(self.img)
        self.glw.addItem(cbar)
        self._cbar = cbar
        self._rois = []
        self._mouse_pos = None
        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.statusBar().showMessage(
            "double-click: add window | d: delete window | c: clean | "
            "i: invert | m/b/r: map/beam/restored"
        )
        self.refresh()

    # ------------------------------------------------------------------

    def _image_data(self):
        if self.what == "beam":
            return self.obs.dbeam
        if self.what == "clean":
            return self.obs.restored_map if self.obs._restored is not None \
                else self.obs.restore()
        return self.obs.dmap

    def refresh(self):
        data = self._image_data()
        ny, nx = data.shape
        ex, _, _, ey = (abs(v) for v in (
            self.obs.extent[0], self.obs.extent[1],
            self.obs.extent[2], self.obs.extent[3],
        ))
        self.img.setImage(data, autoLevels=False)
        # Map pixel coordinates to mas (x = east offset).
        self.img.setRect(QtCore.QRectF(-ex, -ey, 2 * ex, 2 * ey))
        lo, hi = np.percentile(data, [2.0, 99.9])
        self._cbar.setLevels((float(lo), float(hi)))
        self._sync_rois_from_obs()

    # ------------------------------------------------------------------
    # CLEAN windows <-> ROIs
    # ------------------------------------------------------------------

    def _sync_rois_from_obs(self):
        for roi in self._rois:
            self.plot.vb.removeItem(roi)
        self._rois = []
        for (x0, x1, y0, y1) in self.obs.windows:
            self._add_roi(x0, x1, y0, y1)

    def _add_roi(self, x0, x1, y0, y1):
        roi = pg.RectROI(
            [min(x0, x1), min(y0, y1)],
            [abs(x1 - x0), abs(y1 - y0)],
            pen=pg.mkPen((30, 200, 30), width=2),
            movable=True,
            rotatable=False,
        )
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 1], [0, 0])
        roi.sigRegionChangeFinished.connect(self._rois_to_obs)
        self.plot.vb.addItem(roi)
        self._rois.append(roi)

    def _rois_to_obs(self):
        wins = []
        for roi in self._rois:
            x, y = roi.pos()
            w, h = roi.size()
            wins.append((float(x), float(x + w), float(y), float(y + h)))
        self.obs.windows[:] = wins

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------

    def _mouse_moved(self, pos):
        self._mouse_pos = pos

    def _view_coords(self, scene_pos):
        p = self.plot.vb.mapSceneToView(scene_pos)
        return p.x(), p.y()

    def _mouse_clicked(self, ev):
        if ev.double():
            x, y = self._view_coords(ev.scenePos())
            # Default window: 10% of the displayed extent.
            ex = abs(self.obs.extent[0])
            half = 0.05 * 2 * ex
            self._add_roi(x - half, x + half, y - half, y + half)
            self._rois_to_obs()
            ev.accept()

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "d" and self._mouse_pos is not None:
            x, y = self._view_coords(self._mouse_pos)
            for roi in list(self._rois):
                rx, ry = roi.pos()
                w, h = roi.size()
                if rx <= x <= rx + w and ry <= y <= ry + h:
                    self.plot.vb.removeItem(roi)
                    self._rois.remove(roi)
                    self._rois_to_obs()
                    break
        elif key == "c":
            res = self.obs.clean(**self.clean_args)
            self.statusBar().showMessage(
                f"clean: {res['niter']} iterations, "
                f"{res['cleaned_flux']:.4g} Jy cleaned"
            )
            self.refresh()
        elif key == "i":
            self.obs.invert()
            self.refresh()
        elif key in ("m", "b", "r"):
            self.what = {"m": "map", "b": "beam", "r": "clean"}[key]
            self.refresh()
        else:
            super().keyPressEvent(ev)


def mapplot(obs, what="map", block=None, **clean_args):
    p = MapPlot(obs, what=what, clean_args=clean_args)
    run_if_needed(p, block)
    return p
