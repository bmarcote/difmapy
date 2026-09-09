"""Interactive map display with CLEAN windows and model editing;
difmap mapplot (also spelled ``maplot``).

Interaction:

* double-click: add a CLEAN window centered on the cursor (drag its
  corner handles to resize; drag its body to move)
* ``d`` with the cursor over a window: delete it
* ``c``: run one CLEAN batch; ``i``: re-invert and redisplay
* ``1`` / ``2`` / ``3`` / ``4``: residual map / dirty beam / restored
  map / model
* ``n``: add a Gaussian component by clicking its centre and axes
* ``M``: fit the model to the UV data (modelfit)
* ``k`` / ``C``: establish the tentative model / clear all models
* ``h``: the key legend; ``x``: close, reporting the image properties
* ``q``: close

Windows are kept in sync with ``obs.windows`` (mas units), and the
panel to the right of the colour bar shows the distribution of pixel
values over the displayed range, with a Gaussian fitted to the noise.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from difmapy.plots.base import MODEL_COLOR, PlotWindow, run_if_needed

__all__ = ["mapplot", "maplot", "MapPlot"]

#: What each number key displays.
VIEWS = {"1": "map", "2": "beam", "3": "clean", "4": "model"}
VIEW_NAMES = {
    "map": "residual map",
    "beam": "dirty beam",
    "clean": "restored map",
    "model": "model (restored, no residuals)",
}


class MapPlot(PlotWindow):
    """The interactive image display."""

    def __init__(self, obs, what="map", mapsize=None, cellsize=None,
                 uvweight=None, clean_args=None, quiet=False):
        super().__init__("difmapy mapplot")
        self.obs = obs
        self.what = what
        self.clean_args = clean_args or {}
        self.quiet = quiet
        self.result = None
        self._rois = []
        self._mouse_pos = None
        self._gauss_stage = None
        self._gauss = {}
        self._model_items = []
        self._apply_imaging(mapsize, cellsize, uvweight)

        self.glw = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.glw)
        self.glw.setBackground("w")
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "Relative RA (mas)")
        self.plot.setLabel("left", "Relative Dec (mas)")
        self.plot.vb.setAspectLocked(True)
        self.plot.vb.invertX(True)  # RA increases leftward
        self.img = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.img)
        self._cbar = pg.ColorBarItem(colorMap=pg.colormap.get("viridis"))
        self._cbar.setImageItem(self.img)
        self.glw.addItem(self._cbar, row=0, col=1)
        self._cbar.sigLevelsChanged.connect(lambda *_: self._update_histogram())

        # Pixel-value histogram over the displayed range, aligned with
        # the colour bar so the two read together.
        self.hist = self.glw.addPlot(row=0, col=2)
        self.hist.setLabel("bottom", "log10 N")
        self.hist.showGrid(x=True, alpha=0.2)
        self.hist.getAxis("left").setStyle(showValues=False)
        self.hist.setMaximumWidth(170)
        self._hist_items = []

        self.plot.scene().sigMouseMoved.connect(self._mouse_moved)
        self.plot.scene().sigMouseClicked.connect(self._mouse_clicked)
        self.refresh()
        if not self.quiet:
            self.report_beam()

    # ------------------------------------------------------------------
    # imaging setup
    # ------------------------------------------------------------------

    def _apply_imaging(self, mapsize, cellsize, uvweight):
        """Apply the mapsize/cellsize/uvweight overrides, falling back
        to `auto_mapsize()` when the map size was never set by hand."""
        obs = self.obs
        if mapsize is not None or cellsize is not None:
            npix = int(mapsize) if mapsize is not None else obs._nx
            cell = (float(cellsize) if cellsize is not None
                    else obs._xinc / (np.pi / (180.0 * 3600.0 * 1000.0)))
            obs.mapsize(npix, cell)
        else:
            obs._ensure_mapsize()
        if uvweight is not None:
            obs.uvweight(robust=float(uvweight))

    def report_beam(self):
        """Print the beam of the current image, as difmap does."""
        bmaj, bmin, bpa = (self.obs._restore_beam or self.obs.estimated_beam)
        kind = "Restoring" if self.obs._restore_beam else "Estimated"
        print(f"{kind} beam: {bmaj:.4g} x {bmin:.4g} mas at {bpa:.4g} deg "
              f"(map {self.obs._nx}x{self.obs._ny} of "
              f"{self.obs._xinc / (np.pi / (180.0 * 3600.0 * 1000.0)):.4g} mas)")

    # ------------------------------------------------------------------
    # display
    # ------------------------------------------------------------------

    def _image_data(self):
        obs = self.obs
        if self.what == "beam":
            return obs.dbeam
        if self.what == "clean":
            if obs._restored is None:
                obs.restore()
                if not self.quiet:
                    self.report_beam()
            return obs.restored_map
        if self.what == "model":
            obs._ensure_map()
            r = obs._invert_result
            return np.asarray(
                obs._core.restore(r["e_bmaj"], r["e_bmin"], r["e_bpa"],
                                  True, False, 0.0)
            )
        return obs.dmap

    def refresh(self):
        data = np.asarray(self._image_data())
        ex = abs(self.obs.extent[0])
        ey = abs(self.obs.extent[3])
        self.img.setImage(data, autoLevels=False)
        # Map pixel coordinates to mas (x = east offset).
        self.img.setRect(QtCore.QRectF(-ex, -ey, 2 * ex, 2 * ey))
        finite = data[np.isfinite(data)]
        if finite.size:
            lo, hi = np.percentile(finite, [2.0, 99.9])
            if hi <= lo:
                hi = lo + 1e-12
            self._cbar.setLevels((float(lo), float(hi)))
        self._sync_rois_from_obs()
        self._draw_model()
        self._update_histogram()
        self._update_status()

    def _update_status(self):
        extra = ""
        if self._gauss_stage == "center":
            extra = " | click the component centre"
        elif self._gauss_stage == "major":
            extra = " | click to set the major axis (d: point source)"
        elif self._gauss_stage == "minor":
            extra = " | click to set the minor axis (d: circular)"
        model = self.obs.model
        self.statusBar().showMessage(
            f"{VIEW_NAMES[self.what]} | {len(model)} components, "
            f"{self.obs.model_flux:.4g} Jy | {len(self.obs.windows)} windows"
            f"{extra} | h: help"
        )

    # ---- histogram ----------------------------------------------------

    def _update_histogram(self):
        for item in self._hist_items:
            self.hist.removeItem(item)
        self._hist_items = []
        data = np.asarray(self.img.image) if self.img.image is not None else None
        if data is None:
            return
        vals = np.asarray(self.obs.valid(data), dtype=np.float64).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size < 8:
            return
        lo, hi = (float(v) for v in self._cbar.levels())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        counts, edges = np.histogram(vals, bins=80, range=(lo, hi))
        centres = 0.5 * (edges[:-1] + edges[1:])
        logn = np.log10(counts + 1.0)
        curve = pg.PlotCurveItem(
            x=logn, y=centres, pen=pg.mkPen((60, 60, 60), width=1.5),
        )
        self.hist.addItem(curve)
        self._hist_items.append(curve)
        # A Gaussian fitted to the noise: the sigma-clipped moments of
        # the residuals, which excludes the source emission.
        noise = self.obs.noise_stats(self.obs.valid(self.obs.dmap))
        rms = noise["rms"]
        if np.isfinite(rms) and rms > 0:
            width = edges[1] - edges[0]
            amp = vals.size * width / (rms * np.sqrt(2.0 * np.pi))
            model = amp * np.exp(-0.5 * ((centres - noise["mean"]) / rms) ** 2)
            gauss = pg.PlotCurveItem(
                x=np.log10(model + 1.0), y=centres,
                pen=pg.mkPen(*MODEL_COLOR, width=1.5),
            )
            self.hist.addItem(gauss)
            self._hist_items.append(gauss)
            self.hist.setTitle(f"rms {rms:.3g}", size="8pt", color="#333")
        self.hist.setYRange(lo, hi, padding=0)

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
    # model components
    # ------------------------------------------------------------------

    def _draw_model(self):
        """Outline every extended component and mark the delta ones."""
        for item in self._model_items:
            self.plot.vb.removeItem(item)
        self._model_items = []
        deltas_x, deltas_y = [], []
        for c in self.obs.model:
            if c["type"] == "delta" or c["major"] <= 0.0:
                deltas_x.append(c["x"])
                deltas_y.append(c["y"])
                continue
            item = pg.PlotCurveItem(
                *_ellipse(c["x"], c["y"], c["major"], c["ratio"], c["phi"]),
                pen=pg.mkPen(*MODEL_COLOR, width=2),
            )
            self.plot.vb.addItem(item)
            self._model_items.append(item)
        if deltas_x:
            item = pg.ScatterPlotItem(
                deltas_x, deltas_y, size=6, symbol="+",
                pen=pg.mkPen(*MODEL_COLOR), brush=None,
            )
            self.plot.vb.addItem(item)
            self._model_items.append(item)

    def start_gaussian(self):
        """Begin placing a Gaussian component with the mouse."""
        self._gauss_stage = "center"
        self._gauss = {}
        self._update_status()

    def cancel_gaussian(self):
        self._gauss_stage = None
        self._gauss = {}
        self._update_status()

    def _gauss_click(self, x, y):
        g = self._gauss
        if self._gauss_stage == "center":
            g["x"], g["y"] = x, y
            self._gauss_stage = "major"
        elif self._gauss_stage == "major":
            dx, dy = x - g["x"], y - g["y"]
            r = float(np.hypot(dx, dy))
            if r <= 0.0:
                return self._add_gaussian(type="delta")
            # The click marks the half-width, so the FWHM is twice it;
            # phi is the usual position angle, north through east.
            g["major"] = 2.0 * r
            g["phi"] = float(np.rad2deg(np.arctan2(dx, dy)))
            self._gauss_stage = "minor"
        elif self._gauss_stage == "minor":
            dx, dy = x - g["x"], y - g["y"]
            phi = np.deg2rad(g["phi"])
            # Distance perpendicular to the major axis.
            perp = abs(dx * np.cos(phi) - dy * np.sin(phi))
            ratio = float(np.clip(2.0 * perp / g["major"], 1e-3, 1.0))
            return self._add_gaussian(type="gauss", ratio=ratio)
        self._update_status()

    def _add_gaussian(self, type="gauss", ratio=1.0):
        g = self._gauss
        major = 0.0 if type == "delta" else g.get("major", 0.0)
        flux = self._flux_guess(g["x"], g["y"], major, ratio)
        self.obs.addcmp(
            flux, g["x"], g["y"], type=type, major=major,
            ratio=ratio, phi=g.get("phi", 0.0),
            free=["flux", "pos"] if type == "delta" else ["flux", "pos", "major"],
        )
        c = self.obs.model[-1]
        if not self.quiet:
            if type == "delta":
                print(f"added point component: {c['flux']:.5g} Jy at "
                      f"({c['x']:.4g}, {c['y']:.4g}) mas")
            else:
                print(f"added Gaussian component: {c['flux']:.5g} Jy at "
                      f"({c['x']:.4g}, {c['y']:.4g}) mas, "
                      f"{c['major']:.4g} x {c['major'] * c['ratio']:.4g} mas "
                      f"at {c['phi']:.4g} deg")
        self.cancel_gaussian()
        self._draw_model()
        return c

    def _map_value(self, x, y, radius=0.0):
        """The brightest residual within `radius` mas of (x, y)."""
        obs = self.obs
        mas = np.pi / (180.0 * 3600.0 * 1000.0)
        img = obs.dmap
        ix = int(round(x * mas / obs._xinc + obs._nx / 2))
        iy = int(round(y * mas / obs._yinc + obs._ny / 2))
        rx = max(int(radius * mas / obs._xinc), 0)
        ry = max(int(radius * mas / obs._yinc), 0)
        x0, x1 = max(ix - rx, 0), min(ix + rx + 1, img.shape[1])
        y0, y1 = max(iy - ry, 0), min(iy + ry + 1, img.shape[0])
        if x0 >= x1 or y0 >= y1:
            return 0.0
        return float(np.max(img[y0:y1, x0:x1]))

    def _flux_guess(self, x, y, major, ratio):
        """A starting flux for a component of this size at (x, y).

        The map is in Jy/beam, so an extended Gaussian shows a peak
        lower than its flux by the ratio of the beam area to the
        convolved source area; undoing that gives a much better starting
        point for `modelfit` than the peak alone.
        """
        peak = self._map_value(x, y, radius=0.5 * major)
        if major <= 0.0 or peak == 0.0:
            return peak
        bmaj, bmin, _ = (self.obs._restore_beam or self.obs.estimated_beam)
        minor = major * ratio
        return peak * (
            np.hypot(bmaj, major) * np.hypot(bmin, minor) / max(bmaj * bmin, 1e-30)
        )

    # ------------------------------------------------------------------
    # interaction
    # ------------------------------------------------------------------

    def _mouse_moved(self, pos):
        self._mouse_pos = pos

    def _view_coords(self, scene_pos):
        p = self.plot.vb.mapSceneToView(scene_pos)
        return p.x(), p.y()

    def _mouse_clicked(self, ev):
        x, y = self._view_coords(ev.scenePos())
        if self._gauss_stage is not None:
            self._gauss_click(x, y)
            ev.accept()
            return
        if ev.double():
            # Default window: 10% of the displayed extent.
            ex = abs(self.obs.extent[0])
            half = 0.05 * 2 * ex
            self._add_roi(x - half, x + half, y - half, y + half)
            self._rois_to_obs()
            self._update_status()
            ev.accept()

    def key_help(self):
        return [
            ("double-click", "add a CLEAN window"),
            ("d", "delete the window under the cursor"),
            ("c", "CLEAN with the current settings"),
            ("i", "re-invert and redisplay"),
            ("1 / 2 / 3 / 4", "residual map / beam / restored map / model"),
            ("n", "add a Gaussian: click centre, major axis, minor axis"),
            ("d", "while adding: point source, then circular Gaussian"),
            ("M", "fit the model to the UV data (modelfit)"),
            ("k", "establish the tentative model (keep)"),
            ("C", "clear every model component"),
            ("x", "close and report the image properties"),
        ]

    def keyPressEvent(self, ev):
        key = ev.text()
        low = key.lower()
        if low == "d" and self._gauss_stage in ("major", "minor"):
            # Force a point source, or a circular Gaussian.
            self._add_gaussian(type="delta" if self._gauss_stage == "major"
                               else "gauss")
        elif low == "d" and self._mouse_pos is not None:
            self._delete_window_at(*self._view_coords(self._mouse_pos))
        elif low == "n":
            self.start_gaussian()
        elif key == "M":
            self.run_modelfit()
        elif low == "c" and key == "C":
            self.obs.clearmodel()
            if not self.quiet:
                print("cleared all model components")
            self.refresh()
        elif low == "c":
            self.run_clean()
        elif low == "k":
            self.obs.keep()
            self.refresh()
        elif low == "i":
            self.obs.invert()
            self.refresh()
        elif key in VIEWS:
            self.what = VIEWS[key]
            self.refresh()
        elif low == "x":
            self.result = self.obs.print_mapinfo()
            self.close()
        elif low == "\x1b":  # pragma: no cover - Escape
            self.cancel_gaussian()
        else:
            super().keyPressEvent(ev)

    def _delete_window_at(self, x, y):
        for roi in list(self._rois):
            rx, ry = roi.pos()
            w, h = roi.size()
            if rx <= x <= rx + w and ry <= y <= ry + h:
                self.plot.vb.removeItem(roi)
                self._rois.remove(roi)
                self._rois_to_obs()
                self._update_status()
                return True
        return False

    def run_clean(self):
        res = self.obs.clean(quiet=self.quiet, **self.clean_args)
        self.refresh()
        self.statusBar().showMessage(
            f"clean: {res['niter']} iterations, {res['cleaned_flux']:.4g} Jy "
            f"cleaned, residual rms {res['residual_rms']:.4g}"
        )
        return res

    def run_modelfit(self, **kwargs):
        try:
            res = self.obs.modelfit(quiet=self.quiet, **kwargs)
        except (ValueError, RuntimeError) as exc:
            self.statusBar().showMessage(f"modelfit: {exc}")
            return None
        self.refresh()
        self.statusBar().showMessage(
            f"modelfit: reduced chi-squared {res['rchisq']:.4g}, "
            f"{res['ncomp']} components, {res['total_flux']:.4g} Jy"
        )
        return res

    def closeEvent(self, ev):  # pragma: no cover - Qt callback
        if self.result is None:
            try:
                self.result = self.obs.mapinfo()
            except Exception:
                pass
        super().closeEvent(ev)


def _ellipse(x, y, major, ratio, phi_deg, n=64):
    """Points of a component's FWHM ellipse, in mas.

    `phi` is the major-axis position angle measured north through east,
    matching the model convention.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)
    a, b = 0.5 * major, 0.5 * major * ratio
    phi = np.deg2rad(phi_deg)
    # Major axis along (sin phi, cos phi) = north through east.
    ex = a * np.cos(t) * np.sin(phi) + b * np.sin(t) * np.cos(phi)
    ey = a * np.cos(t) * np.cos(phi) - b * np.sin(t) * np.sin(phi)
    return x + ex, y + ey


def mapplot(obs, what="map", mapsize=None, cellsize=None, uvweight=None,
            block=None, quiet=False, **clean_args):
    p = MapPlot(obs, what=what, mapsize=mapsize, cellsize=cellsize,
                uvweight=uvweight, clean_args=clean_args, quiet=quiet)
    run_if_needed(p, block)
    return p


#: difmap's shorter spelling of the same command.
maplot = mapplot
