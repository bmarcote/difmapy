"""Interactive map display with CLEAN windows and model editing;
difmap mapplot (also spelled ``maplot``).

Interaction:

* double-click: add a CLEAN window centered on the cursor (drag its
  corner handles to resize; drag its body to move)
* ``d`` with the cursor over a window: delete it
* ``c``: run one CLEAN batch; ``i``: re-invert and redisplay
* ``1`` / ``2`` / ``3`` / ``4``: residual map / dirty beam / restored
  map / model
* ``m``: place a model component - click its centre, then any two
  points on the component. The two points need be neither the major and
  minor axes nor 90 degrees apart: the longer one sets the major axis
  and the other is solved for the axial ratio. (``d`` at either step
  gives a point source or a circular Gaussian.)
* ``f``: fit the placed components, with the rest of the model, to the
  UV data (modelfit)
* ``C``: clear the model
* ``l``: switch the colours between a linear and a logarithmic scale
* ``g``: switch between difmap's pseudo-colour table and its grey scale
  (black and white)
* ``k``: show or hide the contours of the restored map. They start at
  three times the residual noise and go up by factors of sqrt(2);
  negative ones, from -3 sigma down, are dashed
* the "Weighting" box at the top: difmap's own ``uvweight`` scheme or
  Briggs robust -2 (uniform) ... +2 (natural); picking one re-inverts
* ``h``: the key legend; ``x``: close, reporting the image properties
* ``q``: close

The two handles on the colour bar set the displayed range; the log
scale is a redistribution of the colours only, so the levels and the
bar's axis stay in Jy/beam and negative residuals keep their place.

A component placed with ``m`` is only *drawn*: its guessed flux is held
out of the image until ``f`` fits it, so
placing one never changes the map underneath it.

Windows are kept in sync with ``obs.windows`` (mas units), and the
panel to the right of the colour bar shows the distribution of pixel
values over the displayed range, with a Gaussian fitted to the noise.
"""

from __future__ import annotations


import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from difmapy.plots.base import MODEL_COLOR, PlotWindow, run_if_needed
from difmapy.plots.contours import contour_levels, contour_segments

__all__ = ["mapplot", "maplot", "MapPlot"]

#: What each number key displays.
VIEWS = {"1": "map", "2": "beam", "3": "clean", "4": "model"}
VIEW_NAMES = {
    "map": "residual map",
    "beam": "dirty beam",
    "clean": "restored map",
    "model": "model (restored, no residuals)",
}
#: The heading over the image, naming what is being shown.
VIEW_TITLES = {
    "map": "Residual map",
    "beam": "Dirty beam",
    "clean": "Restored CLEAN map",
    "model": "Model, restored without residuals",
}
#: The footnote under the image: the keys worth knowing without ``h``.
SHORTCUTS = (
    "double-click: window \u2022 d: delete window \u2022 m: add component \u2022 "
    "f: modelfit \u2022 c: clean \u2022 i: invert \u2022 "
    "C: clear model \u2022 1/2/3/4: residual/beam/restored/model \u2022 "
    "l: log/linear \u2022 g: colour/grey \u2022 k: contours \u2022 h: help \u2022 x: close + report \u2022 q: close"
)

#: Strength of the logarithmic colour stretch (as in DS9's log scale).
LOG_STRETCH = 1000.0


def _difmap_rainbow():
    """Difmap's pseudo-colour table (``rainbow`` in its color.c, the
    one its mapplot installs for "color"): near-black blue for the
    faintest level, through blue, cyan, green, yellow and orange to red
    at the brightest. The stops are difmap's own; its table also runs
    past both ends (to black below, white above) for use with a changed
    contrast, which a display from minimum to maximum never reaches."""
    pos = [0.0, 0.17, 0.33, 0.50, 0.67, 0.83, 1.0]
    rgb = [(0.0, 0.0, 0.3), (0.0, 0.0, 0.8), (0.0, 1.0, 1.0),
           (0.6, 1.0, 0.3), (1.0, 1.0, 0.0), (1.0, 0.6, 0.0),
           (1.0, 0.0, 0.0)]
    return pg.ColorMap(pos, [tuple(int(round(255 * c)) for c in k) + (255,)
                             for k in rgb])


def _difmap_grey():
    """Difmap's grey scale: black at the faintest level, white at the
    brightest."""
    return pg.ColorMap([0.0, 1.0], [(0, 0, 0, 255), (255, 255, 255, 255)])


#: The colour maps `mapplot(cmap=...)` knows by name. "color" and "grey"
#: are difmap's two (its ``c`` and ``g`` keys); "viridis" is the map
#: this display used before.
COLOR_MAPS = {
    "color": _difmap_rainbow,
    "grey": _difmap_grey,
    "viridis": lambda: pg.colormap.get("viridis"),
}
_CMAP_ALIASES = {"colour": "color", "rainbow": "color", "gray": "grey",
                 "bw": "grey", "b&w": "grey"}


def _cmap_name(name):
    key = str(name).strip().lower()
    key = _CMAP_ALIASES.get(key, key)
    if key not in COLOR_MAPS:
        raise ValueError(
            f"unknown colour map {name!r}; use one of {sorted(COLOR_MAPS)}")
    return key


def log_stretch(cmap, a=LOG_STRETCH):
    """`cmap` with its colours redistributed logarithmically.

    The stretch goes into the colour map's stop positions, not into the
    pixel values: the image and the colour-bar axis stay in Jy/beam and
    negative residuals keep their place, while a pixel at fraction `p`
    of the displayed range takes the colour a linear map would give
    log(1 + a*p) / log(1 + a). Faint emission therefore gets most of
    the colour range, which is the point of a log display.
    """
    lut = cmap.getLookupTable(0.0, 1.0, 256, alpha=True)
    y = np.linspace(0.0, 1.0, lut.shape[0])
    # The stop positions are the inverse of the stretch, so that
    # colour(p) == cmap(stretch(p)).
    pos = (np.exp(y * np.log1p(a)) - 1.0) / a
    pos[0], pos[-1] = 0.0, 1.0  # exact ends, whatever the rounding did
    return pg.ColorMap(pos, lut)


class MapPlot(PlotWindow):
    """The interactive image display."""

    #: Wide enough for a square map beside its colour bar and histogram.
    DEFAULT_SIZE = (1250, 820)

    def __init__(self, obs, what="map", mapsize=None, cellsize=None,
                 uvweight=None, clean_args=None, quiet=False, scale="linear",
                 cmap="color"):
        super().__init__("difmapy mapplot")
        self.obs = obs
        self.what = what
        self.cmap = _cmap_name(cmap)
        self.scale = "log" if str(scale).lower().startswith("log") else "linear"
        self.clean_args = clean_args or {}
        self.quiet = quiet
        self.result = None
        self._rois = []
        self._mouse_pos = None
        self._gauss_stage = None
        self._gauss = {}
        self._model_items = []
        #: (path item, level) of every contour on the restored map.
        self._contour_items = []
        #: whether the restored map is drawn with contours ("k").
        self.contours = True
        #: the levels drawn, for inspection: (positive, negative, rms).
        self.contour_levels = (np.array([]), np.array([]), float("nan"))
        #: components placed here and not yet fitted (dicts as
        #: `obs.model` gives them): drawn, but not in the model until
        #: "f" fits them or the window closes, since their flux is only
        #: a guess and in the model it would be imaged at once.
        self._placed = []
        self._apply_imaging(mapsize, cellsize, uvweight)

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._make_controls(), 0)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw, 1)
        # The footnote: the main shortcuts, always visible, so that the
        # window explains itself without pressing "h".
        self.footer = QtWidgets.QLabel(SHORTCUTS)
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet(
            "background: #f4f4f4; color: #333; border-top: 1px solid #ccc;"
            "padding: 3px 6px; font-size: 10px;"
        )
        lay.addWidget(self.footer, 0)
        self.setCentralWidget(central)
        self.glw.setBackground("w")
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.setLabel("bottom", "Relative RA (mas)")
        self.plot.setLabel("left", "Relative Dec (mas)")
        self.plot.vb.setAspectLocked(True)
        self.plot.vb.invertX(True)  # RA increases leftward
        self.img = pg.ImageItem(axisOrder="row-major")
        self.plot.addItem(self.img)
        self._base_cmap = COLOR_MAPS[self.cmap]()
        cmap = (log_stretch(self._base_cmap) if self.scale == "log"
                else self._base_cmap)
        self._cbar = pg.ColorBarItem(colorMap=cmap)
        self._cbar.setImageItem(self.img)
        self.glw.addItem(self._cbar, row=0, col=1)
        self._cbar.sigLevelsChanged.connect(lambda *_: self._update_histogram())
        self._cbar.sigLevelsChanged.connect(lambda *_: self._style_contours())
        # The handles work in steps of `rounding`, which has to follow
        # the data; see `_tune_rounding`. A drag ends by snapping the
        # handles back, which is the moment to re-scale the step to
        # whatever range the user has just zoomed to.
        self._cbar.sigLevelsChangeFinished.connect(lambda *_: self._tune_rounding())

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

    #: The robustness values offered in the weighting box.
    ROBUST_CHOICES = (-2.0, -1.0, 0.0, 1.0, 2.0)
    _ROBUST_LABELS = {-2.0: "robust -2 (uniform)", 2.0: "robust +2 (natural)"}

    @staticmethod
    def _robust_name(r):
        return f"robust {r:+g}" if r else "robust 0"

    def _make_controls(self):
        """The row above the image: the weighting scheme."""
        bar = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(6, 3, 6, 3)
        row.addWidget(QtWidgets.QLabel("Weighting:"))
        self.weighting = QtWidgets.QComboBox()
        # Mouse only: with keyboard focus the box would swallow the
        # single-key shortcuts (and change the weighting on "c", "i"...).
        self.weighting.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._difmap_weighting = (self.obs._binwid, self.obs._errpow,
                                  self.obs._dorad)
        self.weighting.addItem("difmap uvweight", None)
        for r in self.ROBUST_CHOICES:
            self.weighting.addItem(
                self._ROBUST_LABELS.get(r, self._robust_name(r)), r)
        self._sync_weighting()
        self.weighting.currentIndexChanged.connect(self._weighting_chosen)
        row.addWidget(self.weighting)
        row.addStretch(1)
        return bar

    def _sync_weighting(self):
        """Show the weighting the observation is using, including one
        set at the prompt (a robustness not on the list is added)."""
        box = self.weighting
        r = self.obs.robust
        if r is None:
            self._difmap_weighting = (self.obs._binwid, self.obs._errpow,
                                      self.obs._dorad)
            b, e, _ = self._difmap_weighting
            text = f"difmap uvweight {b:g}, {e:g}"
        idx = box.findData(r) if r is not None else 0
        if idx < 0:
            box.blockSignals(True)
            box.addItem(self._robust_name(r), r)
            box.blockSignals(False)
            idx = box.count() - 1
        box.blockSignals(True)
        if r is None:
            box.setItemText(0, text)
        box.setCurrentIndex(idx)
        box.blockSignals(False)

    def set_weighting(self, robust):
        """Re-image with Briggs `robust` (-2 ... 2), or with difmap's own
        uvweight scheme for None (the binwid/errpow in use before a
        robustness was picked here). Returns the estimated beam."""
        obs = self.obs
        if robust is None:
            b, e, rad = self._difmap_weighting
            obs.uvweight(b, e, radial=rad)
        else:
            obs.uvweight(obs._binwid, obs._errpow, radial=obs._dorad,
                         robust=float(robust))
        obs.invert()
        self.refresh()
        bmaj, bmin, bpa = obs.estimated_beam
        name = ("difmap uvweight" if robust is None
                else self._robust_name(float(robust)))
        self._message(f"{name}: beam {bmaj:.4g} x {bmin:.4g} mas at "
                      f"{bpa:.4g} deg")
        if not self.quiet:
            self.report_beam()
        return (bmaj, bmin, bpa)

    def _weighting_chosen(self, index):
        self.set_weighting(self.weighting.itemData(index))

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

    def view_boxes(self):
        return [self.plot.vb]

    def _tune_rounding(self):
        """Match the colour-bar handles' step size to the image.

        `ColorBarItem` rounds every level it computes to a multiple of
        `rounding` and refuses a span narrower than it. Its default of
        1 is meaningless for a map in Jy/beam: the first drag snaps the
        levels to whole Janskys - typically (0, 1) - which flattens the
        image to a single colour and leaves the handles with no step
        small enough to do anything. A thousandth of the displayed span
        gives handles that move smoothly instead.
        """
        lo, hi = (float(v) for v in self._cbar.levels())
        span = abs(hi - lo)
        if not np.isfinite(span) or span <= 0.0:
            span = 1.0
        self._cbar.rounding = span / 1000.0

    def set_scale(self, scale):
        """Show the colours on a "linear" or "log" scale.

        Only the colour map changes: the levels, the image and the
        colour-bar axis stay in Jy/beam, so a log display of a residual
        map keeps its negatives.
        """
        scale = "log" if str(scale).lower().startswith("log") else "linear"
        if scale != self.scale:
            if scale == "log":
                # Stretch whatever map is showing, so a colour map
                # picked from the bar's own menu survives the toggle.
                self._base_cmap = self._cbar.colorMap()
                self._cbar.setColorMap(log_stretch(self._base_cmap))
            else:
                self._cbar.setColorMap(self._base_cmap)
            self.scale = scale
        self.refresh()
        return scale

    def set_cmap(self, name):
        """Show the image in the colour map `name`: "color" (difmap's
        pseudo-colour table), "grey" (black and white) or "viridis".
        The linear/log choice and the displayed range are kept."""
        self.cmap = _cmap_name(name)
        self._base_cmap = COLOR_MAPS[self.cmap]()
        self._cbar.setColorMap(log_stretch(self._base_cmap)
                               if self.scale == "log" else self._base_cmap)
        self._style_contours()
        self._update_histogram()
        return self.cmap

    def _title(self, data):
        """The heading over the image: which image this is, and the one
        number that characterises it."""
        title = VIEW_TITLES[self.what]
        if self.scale == "log":
            title += " (log colours)"
        if self.what == "beam":
            bmaj, bmin, bpa = self.obs.estimated_beam
            return f"{title} \u2014 {bmaj:.4g} x {bmin:.4g} mas at {bpa:.4g} deg"
        if self.what == "model":
            model = self.obs.model
            return (f"{title} \u2014 {len(model)} components, "
                    f"{self.obs.model_flux:.4g} Jy")
        valid = self.obs.valid(np.asarray(data))
        finite = valid[np.isfinite(valid)]
        peak = float(finite.max()) if finite.size else float("nan")
        return f"{title} \u2014 peak {peak:.4g} Jy/beam"

    def refresh(self):
        self._sync_weighting()
        data = np.asarray(self._image_data())
        self.plot.setTitle(self._title(data), size="11pt", color="#222")
        ex = abs(self.obs.extent[0])
        ey = abs(self.obs.extent[3])
        self.img.setImage(data, autoLevels=False)
        # Map pixel coordinates to mas (x = east offset).
        self.img.setRect(QtCore.QRectF(-ex, -ey, 2 * ex, 2 * ey))
        # The colours span the displayed map from its minimum to its
        # peak, as in difmap (setcmpar: the range of the inner, valid
        # area). The noise then sits at the dark end and the source
        # stands out; a percentile cut, on a large field around a
        # compact source, lands inside the noise and spends the whole
        # colour range on it.
        valid = np.asarray(self.obs.valid(data))
        finite = valid[np.isfinite(valid)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                hi = lo + 1e-12
            self._cbar.setLevels((lo, hi))
        self._tune_rounding()
        self._sync_rois_from_obs()
        self._draw_contours(data)
        self._draw_model()
        self._update_histogram()
        self._update_status()

    def _message(self, text):
        self.statusBar().showMessage(text)

    def _update_status(self):
        extra = ""
        if self._gauss_stage == "center":
            extra = " | click the component centre"
        elif self._gauss_stage == "first":
            extra = " | click a point on the component (d: point source)"
        elif self._gauss_stage == "second":
            extra = " | click a second point on it (d: circular)"
        elif self._placed:
            extra = (f" | {len(self._placed)} placed, not fitted "
                     "(f: modelfit)")
        model = self.obs.model
        self._message(
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
        residual = np.asarray(self.obs.dmap)
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
        noise = self.obs.noise_stats(self.obs.valid(residual))
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
        """Make the ROIs on screen match `obs.windows`.

        Existing ROIs are moved and resized rather than replaced: this
        runs on every refresh, and a discarded ROI is cyclic garbage
        that owns child items (its handles). If Python's collector
        happens to run while pyqtgraph is building the replacement, it
        destroys those in an order Qt does not survive - a segmentation
        fault that comes and goes with the interpreter's allocation
        counts. So nothing is discarded unless a window really went.
        """
        wins = list(self.obs.windows)
        for roi, (x0, x1, y0, y1) in zip(self._rois, wins):
            pos = [min(x0, x1), min(y0, y1)]
            size = [abs(x1 - x0), abs(y1 - y0)]
            if tuple(roi.pos()) != tuple(pos) or tuple(roi.size()) != tuple(size):
                # finish=False: this is the observation moving the ROI,
                # not the user, so nothing is written back.
                roi.setPos(pos, update=False, finish=False)
                roi.setSize(size, finish=False)
        for roi in self._rois[len(wins):]:
            self._retire_roi(roi)
        del self._rois[len(wins):]
        for (x0, x1, y0, y1) in wins[len(self._rois):]:
            self._add_roi(x0, x1, y0, y1)

    def _retire_roi(self, roi):
        """Take an ROI off the display for good, in an order that is safe
        whenever its wrapper is eventually collected: its handles are
        detached first (so that no child is destroyed along with it
        behind Python's back), then the signal that ties it to this
        window, then the item itself."""
        try:
            roi.sigRegionChangeFinished.disconnect(self._rois_to_obs)
        except (TypeError, RuntimeError):  # pragma: no cover
            pass
        for handle in list(roi.getHandles()):
            roi.removeHandle(handle)
        self.plot.vb.removeItem(roi)

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

    # ---- contours ----------------------------------------------------

    #: Contours start at this many times the residual noise...
    CONTOUR_NSIGMA = 3.0
    #: ...and each level is this factor above the one before.
    CONTOUR_FACTOR = float(np.sqrt(2.0))

    def _draw_contours(self, data):
        """Contour the restored map: solid from +3 sigma up in factors
        of sqrt(2), dashed from -3 sigma down, sigma being the noise of
        the residual map. Only the valid inner quarter is traced, like
        everything else measured on a map."""
        for item, _ in self._contour_items:
            self.plot.vb.removeItem(item)
        self._contour_items = []
        self.contour_levels = (np.array([]), np.array([]), float("nan"))
        if self.what != "clean" or not self.contours:
            return
        obs = self.obs
        sy, sx = obs.valid_slice
        sub = np.asarray(data)[sy, sx]
        finite = sub[np.isfinite(sub)]
        rms = float(obs.noise_stats()["rms"])
        if finite.size == 0:
            return
        pos, neg = contour_levels(rms, float(finite.max()), float(finite.min()),
                                  nsigma=self.CONTOUR_NSIGMA,
                                  factor=self.CONTOUR_FACTOR)
        self.contour_levels = (pos, neg, rms)
        # Pixel index -> mas, as the image is placed: pixel i spans
        # [i, i + 1] cells from the edge, so its centre is at i + 0.5.
        ex, ey = abs(obs.extent[0]), abs(obs.extent[3])
        dx, dy = 2 * ex / obs._nx, 2 * ey / obs._ny
        for level in list(pos) + list(neg):
            xs, ys = contour_segments(sub, level)
            if xs.size == 0:
                continue
            path = pg.arrayToQPath(-ex + (xs + sx.start + 0.5) * dx,
                                   -ey + (ys + sy.start + 0.5) * dy,
                                   connect="pairs")
            item = QtWidgets.QGraphicsPathItem(path)
            item.setZValue(5)          # over the image, under the windows
            self.plot.vb.addItem(item, ignoreBounds=True)
            self._contour_items.append((item, float(level)))
        self._style_contours()

    def _style_contours(self):
        """Pen every contour for contrast against the colour scale.

        A contour in the colour of its own level would vanish: the
        pixels it runs through have exactly that colour. So each takes
        whichever of light and dark stands out from the colour the scale
        gives its level - which follows the colour bar when its handles
        are dragged or the scale switches to log. Negative levels are
        dashed.
        """
        if not self._contour_items:
            return
        lo, hi = (float(v) for v in self._cbar.levels())
        span = hi - lo if hi > lo else 1.0
        cmap = self._cbar.colorMap()
        for item, level in self._contour_items:
            frac = float(np.clip((level - lo) / span, 0.0, 1.0))
            r, g, b = (float(v) for v in cmap.map(frac, mode="float")[:3])
            bright = 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5
            pen = pg.mkPen((30, 30, 30) if bright else (245, 245, 245),
                           width=1,
                           style=(QtCore.Qt.PenStyle.DashLine if level < 0
                                  else QtCore.Qt.PenStyle.SolidLine))
            pen.setCosmetic(True)
            item.setPen(pen)

    def set_contours(self, on=True):
        """Show or hide the contours of the restored map."""
        self.contours = bool(on)
        self.refresh()
        return self.contours

    def _draw_model(self):
        """Outline every extended component and mark the delta ones."""
        for item in self._model_items:
            self.plot.vb.removeItem(item)
        self._model_items = []
        deltas_x, deltas_y = [], []
        placed = {id(c) for c in self._placed}
        for c in self.obs.model + self._placed:
            if c["type"] == "delta" or c["major"] <= 0.0:
                deltas_x.append(c["x"])
                deltas_y.append(c["y"])
                continue
            item = pg.PlotCurveItem(
                *_ellipse(c["x"], c["y"], c["major"], c["ratio"], c["phi"]),
                pen=pg.mkPen(*MODEL_COLOR, width=2, style=(
                    QtCore.Qt.PenStyle.DashLine if id(c) in placed
                    else QtCore.Qt.PenStyle.SolidLine)),
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
            self._gauss_stage = "first"
        elif self._gauss_stage == "first":
            r = float(np.hypot(x - g["x"], y - g["y"]))
            if r <= 0.0:
                return self._add_gaussian(type="delta")
            g["p1"] = (x, y)
            self._gauss_stage = "second"
        elif self._gauss_stage == "second":
            if np.hypot(x - g["x"], y - g["y"]) <= 0.0:
                return self._add_gaussian(type="gauss", **self._circle_from(g["p1"]))
            return self._add_gaussian(type="gauss",
                                      **self._ellipse_from(g["p1"], (x, y)))
        self._update_status()

    def _circle_from(self, p1):
        """A circular Gaussian through `p1`."""
        g = self._gauss
        r = float(np.hypot(p1[0] - g["x"], p1[1] - g["y"]))
        return {"major": 2.0 * r, "ratio": 1.0, "phi": 0.0}

    def _ellipse_from(self, p1, p2):
        """The FWHM ellipse through two points clicked around a centre.

        The two clicks are just two points *on* the ellipse, at whatever
        position angles the user picked: neither is required to be the
        major axis, nor are they required to be 90 degrees apart. The
        longer of the two fixes the major axis (semi-axis `a` and its
        position angle), and the shorter must then satisfy the ellipse
        equation in that frame,

            (x'/a)^2 + (y'/b)^2 = 1,

        which gives the semi-minor axis `b` directly. Because the
        shorter radius is never longer than `a`, this always yields
        `b <= a`, i.e. a valid axial ratio.

        Returns the `major` (FWHM), `ratio` and `phi` of the component.
        """
        cx, cy = self._gauss["x"], self._gauss["y"]
        v1 = (p1[0] - cx, p1[1] - cy)
        v2 = (p2[0] - cx, p2[1] - cy)
        if np.hypot(*v2) > np.hypot(*v1):
            v1, v2 = v2, v1
        a = float(np.hypot(*v1))
        # Position angle of the major axis, north through east.
        phi = float(np.arctan2(v1[0], v1[1]))
        # The shorter radius in the ellipse frame: x' along the major
        # axis (sin phi, cos phi), y' along the perpendicular.
        sin, cos = np.sin(phi), np.cos(phi)
        xp = v2[0] * sin + v2[1] * cos
        yp = v2[0] * cos - v2[1] * sin
        denom = 1.0 - (xp / a) ** 2
        if abs(yp) <= 1e-6 * a or denom <= 1e-6:
            # The second click sits on the major axis itself, which says
            # nothing about the minor one: leave it circular rather than
            # collapsing the component to a line.
            ratio = 1.0
        else:
            b = float(abs(yp) / np.sqrt(denom))
            ratio = float(np.clip(b / a, 1e-3, 1.0))
        return {"major": 2.0 * a, "ratio": ratio,
                "phi": float(np.rad2deg(phi))}

    def _add_gaussian(self, type="gauss", major=0.0, ratio=1.0, phi=0.0):
        g = self._gauss
        if type == "delta":
            major, ratio, phi = 0.0, 1.0, 0.0
        flux = self._flux_guess(g["x"], g["y"], major, ratio)
        # An elliptical component is placed with its shape meant, so its
        # axial ratio and orientation are fitted along with its size;
        # a circular one keeps difmap's circular-Gaussian freedom.
        if type == "delta":
            free = ["flux", "pos"]
        elif ratio < 1.0:
            free = ["flux", "pos", "shape"]
        else:
            free = ["flux", "pos", "major"]
        from difmapy.observation import _free_mask

        # Drawn, not part of the model, until modelfit gives it a real
        # flux (see `_placed`).
        c = {"type": type, "flux": float(flux), "x": float(g["x"]),
             "y": float(g["y"]), "major": float(major), "ratio": float(ratio),
             "phi": float(phi), "freq0": 0.0, "spcind": 0.0,
             "freepar": _free_mask(free)}
        self._placed.append(c)
        if not self.quiet:
            if type == "delta":
                print(f"placed point component: {c['flux']:.5g} Jy at "
                      f"({c['x']:.4g}, {c['y']:.4g}) mas")
            else:
                print(f"placed Gaussian component: {c['flux']:.5g} Jy at "
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
            ("m", "add a component: click its centre, then two points on it"),
            ("d", "while adding: point source, then circular Gaussian"),
            ("f", "fit the placed components to the UV data (modelfit)"),
            ("C", "clear every model component"),
            ("l", "logarithmic or linear colour scale"),
            ("g", "difmap pseudo-colour or grey scale (black and white)"),
            ("k", "contours on the restored map: 3 sigma, x sqrt(2); "
                  "negative dashed"),
            ("drag the bar handles", "set the displayed range"),
            ("z / u", "restore the y / x axis range"),
            ("x", "close and report the image properties"),
        ]

    def keyPressEvent(self, ev):
        key = ev.text()
        low = key.lower()
        if low == "d" and self._gauss_stage in ("first", "second"):
            # Force a point source, or a circular Gaussian through the
            # one point already clicked.
            if self._gauss_stage == "first":
                self._add_gaussian(type="delta")
            else:
                self._add_gaussian(type="gauss",
                                   **self._circle_from(self._gauss["p1"]))
        elif low == "d" and self._mouse_pos is not None:
            self._delete_window_at(*self._view_coords(self._mouse_pos))
        elif low in ("m", "n"):
            self.start_gaussian()
        elif low == "f":
            self.run_modelfit()
        elif key == "C":
            self.obs.clearmodel()
            self._placed = []
            if not self.quiet:
                print("cleared all model components")
            self.refresh()
        elif low == "c":
            self.run_clean()
        elif low == "i":
            self.obs.invert()
            self.refresh()
        elif low == "k":
            on = self.set_contours(not self.contours)
            pos, neg, rms = self.contour_levels
            self._message(
                "contours off" if not on else
                f"contours from +-{self.CONTOUR_NSIGMA:g} x {rms:.3g} Jy/beam "
                f"in steps of sqrt(2): {len(pos)} positive, {len(neg)} "
                "negative (dashed)" if self.what == "clean" else
                "contours on (shown on the restored map, key 3)")
        elif low == "g":
            name = self.set_cmap("color" if self.cmap == "grey" else "grey")
            self._message("grey scale (black and white)" if name == "grey"
                          else "difmap pseudo-colour")
        elif low == "l":
            self._message(
                f"{self.set_scale('linear' if self.scale == 'log' else 'log')}"
                " colour scale"
            )
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
                self._retire_roi(roi)
                self._rois.remove(roi)
                self._rois_to_obs()
                self._update_status()
                return True
        return False

    def run_clean(self):
        res = self.obs.clean(quiet=self.quiet, **self.clean_args)
        self.refresh()
        self._message(
            f"clean: {res['niter']} iterations, {res['cleaned_flux']:.4g} Jy "
            f"cleaned, residual rms {res['residual_rms']:.4g}"
        )
        return res

    def run_modelfit(self, **kwargs):
        """Fit the placed components, together with every other component
        that has free parameters, and show the result (difmap modelfit).

        The placed components join the model here. Unlike
        `Observation.modelfit`, nothing is seeded: with no component to
        fit there is no telling where the user wanted one, so the plot
        asks for one instead of inventing it.
        """
        if not self._placed and not self.obs.nvariable:
            self._message("nothing to fit: press m to place a component")
            return None
        before, placed = self.obs.model, list(self._placed)
        self._commit_placed()
        try:
            res = self.obs.modelfit(quiet=self.quiet, **kwargs)
        except (ValueError, RuntimeError) as exc:
            # Leave the model as it was, and the placed components placed.
            self.obs._replace_model(before)
            self._placed = placed
            self._message(f"modelfit: {exc}")
            return None
        self.refresh()
        self._message(
            f"modelfit: reduced chi-squared {res['rchisq']:.4g}, "
            f"{res['ncomp']} components, {res['total_flux']:.4g} Jy"
        )
        return res

    def _commit_placed(self):
        """Add the placed components to the model."""
        for c in self._placed:
            self.obs.addcmp(c["flux"], c["x"], c["y"], type=c["type"],
                            major=c["major"], ratio=c["ratio"], phi=c["phi"],
                            free=c["freepar"])
        self._placed = []

    def closeEvent(self, ev):  # pragma: no cover - Qt callback
        if self._placed:
            # Placed but not fitted: still part of the model being
            # built, so they are added rather than thrown away.
            n = len(self._placed)
            self._commit_placed()
            if not self.quiet:
                print(f"added {n} placed, unfitted component(s) to the model; "
                      "obs.modelfit() fits them")
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
            block=None, quiet=False, scale="linear", cmap="color",
            **clean_args):
    p = MapPlot(obs, what=what, mapsize=mapsize, cellsize=cellsize,
                uvweight=uvweight, clean_args=clean_args, quiet=quiet,
                scale=scale, cmap=cmap)
    run_if_needed(p, block)
    return p


#: difmap's shorter spelling of the same command.
maplot = mapplot
