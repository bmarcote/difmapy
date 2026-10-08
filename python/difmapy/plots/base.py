"""Shared plotting infrastructure (Qt app handling, flag interaction).

Interaction conventions (all point-based plots):

* Shift + drag: flag the points inside the rubber band
* Ctrl + drag: unflag the points inside the rubber band
* hover + ``f`` / ``F``: flag / unflag the nearest point
* Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y): undo / redo the last flag edit
* ``x``: show or hide the flagged points (hidden by default)
* ``z`` / ``u``: restore the y / x axis range, as difmap's Z and U do
* ``r``: reload the plot from the data
* ``h``: the key legend of the active plot
* ``q``: close the window

Flag edits are applied to the underlying data immediately (difmap
behaviour) and the plot refreshes in place; every edit is recorded
exactly, as a snapshot of the affected rows of the FLAG column, so that
undo restores precisely what was there before.
"""

from __future__ import annotations

import atexit
import functools
import math
import os
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

IF_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34),
    (23, 190, 207), (214, 39, 40),
]
FLAG_COLOR = (220, 40, 40)
UNFLAG_COLOR = (40, 160, 40)
HIGHLIGHT_COLOR = (255, 140, 0)
MODEL_COLOR = (214, 30, 30)


# ----------------------------------------------------------------------
# window lifetime
# ----------------------------------------------------------------------

#: Every plot window whose C++ object still exists, open or merely
#: closed. These are *strong* references on purpose: a window nobody
#: kept ("mapplot()" at the prompt, with the result discarded) would
#: otherwise be collected at some arbitrary later moment, and Qt does
#: not survive having a live window destroyed from under it. Closing a
#: window does not remove it from here - a closed window is still a
#: live QMainWindow, and still crashes the interpreter on the way out
#: if nobody destroys it first.
_OPEN_WINDOWS: list = []
_ATEXIT_DONE = False


def _alive(win) -> bool:
    """False once the window's C++ half has been destroyed."""
    try:
        win.isVisible()
        return True
    except RuntimeError:  # shiboken/sip: the C++ object is gone
        return False


def _register_window(win):
    """Keep `win` alive, and arrange for a clean teardown at exit."""
    global _ATEXIT_DONE
    # Windows that have really gone can be forgotten; closed ones are
    # kept until their deferred deletion has actually happened, so that
    # the exit handler can still finish them off.
    _OPEN_WINDOWS[:] = [w for w in _OPEN_WINDOWS if _alive(w)]
    _OPEN_WINDOWS.append(win)
    if not _ATEXIT_DONE:
        # atexit handlers run last-registered-first, and Qt's own
        # module shutdown was registered when QtCore was imported -
        # before this - so ours goes first, which is what we need.
        atexit.register(close_all_windows)
        _ATEXIT_DONE = True


def close_all_windows():
    """Close every open plot window and destroy it, now.

    Left to itself, the interpreter garbage-collects a live plot window
    during shutdown: Python frees the wrappers of the items inside the
    scene, then the QMainWindow's destructor tears the scene down and
    Qt walks into ~QGraphicsItem on objects whose Python halves are
    already gone. On macOS that aborts the process with SIGTRAP - "zsh:
    trace trap" after an otherwise clean session.

    Deleting the C++ objects here, while the interpreter is still
    healthy, leaves nothing for the shutdown to destroy.
    """
    windows, _OPEN_WINDOWS[:] = list(_OPEN_WINDOWS), []
    app = QtWidgets.QApplication.instance()
    for win in windows:
        try:
            win.close()
        except Exception:  # pragma: no cover - already half gone
            pass
    for win in windows:
        try:
            win.deleteLater()
        except Exception:  # pragma: no cover
            pass
    if app is not None and windows:
        # deleteLater only posts an event; without an event loop to run
        # it, ask for the deferred deletions explicitly.
        app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
        app.processEvents()


def open_windows() -> list:
    """The plot windows that still exist, open or closed."""
    return [w for w in _OPEN_WINDOWS if _alive(w)]


def fit_to_screen(width, height, margin=0.92):
    """(width, height), reduced to what the screen can actually show."""
    try:
        geo = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        if geo.width() > 0 and geo.height() > 0:
            return (min(int(width), int(geo.width() * margin)),
                    min(int(height), int(geo.height() * margin)))
    except Exception:  # pragma: no cover - no screen (offscreen tests)
        pass
    return (int(width), int(height))


def ensure_app():
    app = QtWidgets.QApplication.instance()
    if app is None:
        if not os.environ.get("QT_QPA_PLATFORM") and _no_display_server():
            # A server or CI job with no X/Wayland display: Qt would abort
            # trying to connect to one. Draw off-screen instead, which is
            # all a headless pipeline can use anyway.
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
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


# ----------------------------------------------------------------------
# fast markers
# ----------------------------------------------------------------------

class FastScatter(pg.GraphicsObject):
    """Many identical markers, drawn in one call.

    pyqtgraph's ScatterPlotItem builds a per-point record of style and
    geometry, which takes seconds for the million-point panels of a long
    VLBI observation - and every refresh (a flag edit, a highlighted
    antenna, a new page) pays it again. Every marker here shares one
    symbol, so the points are kept as two plain arrays and painted as
    fragments of a single pre-rendered pixmap. Non-finite points are
    dropped.
    """

    def __init__(self, x, y, size=4, symbol="o", pen=None, brush=None):
        super().__init__()
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.all():
            x, y = x[ok], y[ok]
        self._x, self._y = x, y
        self._size = float(size)
        self._symbol = symbol
        self._pen = pg.mkPen(pen)
        self._brush = pg.mkBrush(brush)
        self._bounds = (((float(x.min()), float(x.max())),
                         (float(y.min()), float(y.max())))
                        if x.size else ((None, None), (None, None)))
        self._pixmap = None
        self._dpr = None
        self._frags = None

    def getData(self):
        """The (x, y) of the points drawn."""
        return self._x, self._y

    def _extent_px(self):
        return self._size + max(math.ceil(self._pen.widthF()), 1)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if self._x.size == 0:
            return (None, None)
        d, other = (self._x, self._y) if ax == 0 else (self._y, self._x)
        if orthoRange is not None:
            d = d[(other >= orthoRange[0]) & (other <= orthoRange[1])]
            if d.size == 0:
                return (None, None)
        elif frac >= 1.0:
            return self._bounds[ax]
        if frac >= 1.0:
            return (float(d.min()), float(d.max()))
        lo, hi = np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])
        return (float(lo), float(hi))

    def pixelPadding(self):
        return self._extent_px() * 0.7072

    def _pixel_size(self):
        """Length of a screen pixel along x and y, in data units."""
        vx, vy = self.pixelVectors()
        try:
            return (0.0 if vx is None else vx.length(),
                    0.0 if vy is None else vy.length())
        except OverflowError:
            return (0.0, 0.0)

    def boundingRect(self):
        (x0, x1), (y0, y1) = self._bounds
        if x0 is None:
            return QtCore.QRectF()
        px, py = self._pixel_size()
        pad = self.pixelPadding()
        px, py = px * pad, py * pad
        return QtCore.QRectF(x0 - px, y0 - py, (x1 - x0) + 2 * px, (y1 - y0) + 2 * py)

    def viewTransformChanged(self):
        self.prepareGeometryChange()
        super().viewTransformChanged()

    def _pixmap_for(self, dpr):
        if self._pixmap is None or self._dpr != dpr:
            from pyqtgraph.graphicsItems.ScatterPlotItem import renderSymbol

            img = renderSymbol(self._symbol, self._size, self._pen, self._brush,
                               dpr=dpr)
            pm = QtGui.QPixmap.fromImage(img)
            pm.setDevicePixelRatio(1.0)  # fragments address device pixels
            self._pixmap, self._dpr = pm, dpr
        return self._pixmap

    @pg.debug.warnOnException  # an exception raised in paint() kills Qt
    def paint(self, p, option, widget):
        x, y = self._x, self._y
        if x.size == 0:
            return
        # Cull to the view (padded by a marker) before mapping to pixels:
        # a zoomed-in view then only draws what it shows.
        vr = self.viewRect()
        if vr is not None:
            px, py = self._pixel_size()
            padx, pady = px * self._extent_px(), py * self._extent_px()
            l, r = sorted((vr.left(), vr.right()))
            b, t = sorted((vr.top(), vr.bottom()))
            m = (x >= l - padx) & (x <= r + padx) & (y >= b - pady) & (y <= t + pady)
            if not m.all():
                x, y = x[m], y[m]
            if x.size == 0:
                return
        tr = p.transform()
        X = np.clip(tr.m11() * x + tr.m21() * y + tr.dx(), -2.0 ** 30, 2.0 ** 30)
        Y = np.clip(tr.m12() * x + tr.m22() * y + tr.dy(), -2.0 ** 30, 2.0 ** 30)
        dev = p.device()
        dpr = float(dev.devicePixelRatioF()) if dev is not None else 1.0
        pm = self._pixmap_for(dpr)
        if self._frags is None:
            from pyqtgraph.Qt import internals

            self._frags = internals.PrimitiveArray(QtGui.QPainter.PixmapFragment, 10)
        self._frags.resize(X.size)
        f = self._frags.ndarray()
        f[:, 0] = X  # x, y: the centre of the marker
        f[:, 1] = Y
        f[:, 2:6] = (0.0, 0.0, pm.width(), pm.height())  # source rect
        f[:, 6:10] = (1.0 / dpr, 1.0 / dpr, 0.0, 1.0)     # scale, rotation, opacity
        p.resetTransform()
        p.drawPixmapFragments(*self._frags.drawargs(), pm)


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
    """Present a plot the way the current mode says (see `set_mode`).

    "window": show it, and enter the Qt event loop only if nothing else
    will pump it. Under IPython or Jupyter the loop is hooked into the
    session instead (what ``%gui qt`` does), so the prompt stays usable;
    a plain script blocks until the window is closed. `block` overrides
    that choice.

    "inline": draw it off-screen and never block; under Jupyter the image
    goes into the cell's output. The plot is returned either way, for
    `savefig` and further calls.
    """
    if get_mode() == "inline":
        _present_inline(widget)
        return
    widget.show()
    if block is None:
        block = not _hook_qt_loop()
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
# where plots go: interactive windows, or inline / off-screen images
# ----------------------------------------------------------------------

#: How plot commands present their plots; see `set_mode`.
MODES = ("auto", "window", "inline")
_MODE = os.environ.get("DIFMAPY_PLOT_MODE", "auto").strip().lower()
if _MODE not in MODES:
    _MODE = "auto"


def set_mode(mode="auto") -> str:
    """Choose how plot commands present their plots.

    - "window": an interactive Qt window, with flagging and the keys.
      Under IPython or Jupyter the Qt event loop is hooked in
      automatically, so the prompt or notebook stays usable.
    - "inline": no window. The plot is drawn off-screen, shown in the
      cell output under Jupyter, and returned for `savefig()` /
      `to_png()` - what pipelines and headless kernels need.
    - "auto" (the default, or $DIFMAPY_PLOT_MODE): "inline" in a Jupyter
      kernel or where there is no display, "window" otherwise.

    Returns the previous setting.
    """
    global _MODE
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown plot mode {mode!r}; use one of {MODES}")
    old, _MODE = _MODE, mode
    return old


def get_mode() -> str:
    """The mode plots are presented in now: "window" or "inline"."""
    if _MODE != "auto":
        return _MODE
    return "inline" if (in_jupyter() or not has_display()) else "window"


def in_jupyter() -> bool:
    """True inside a Jupyter kernel (notebook, lab, nbconvert...)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and type(shell).__name__ == "ZMQInteractiveShell"


def _no_display_server() -> bool:
    """True on X11/Wayland systems with neither display set."""
    if sys.platform.startswith(("linux", "freebsd", "openbsd", "netbsd")):
        return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return False


def has_display() -> bool:
    """Whether a window could be put on a screen at all."""
    platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    if platform.startswith(("offscreen", "minimal")):
        return False
    return not _no_display_server()


def _hook_qt_loop() -> bool:
    """Have an IPython or Jupyter session pump Qt events, as ``%gui qt``
    does; False if there is no session to do it."""
    if _qt_loop_hooked():
        return True
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    if shell is None:
        return False
    try:
        shell.enable_gui("qt")
    except Exception:
        return False
    return True


def _cell_number():
    """The IPython execution count, or None outside IPython."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    shell = get_ipython()
    return None if shell is None else shell.execution_count


def _present_inline(widget):
    png = widget.to_png()
    if in_jupyter():
        from IPython.display import Image, display

        display(Image(data=png))
        # Remembered so that returning the plot as the cell's value does
        # not show the same image twice.
        widget._shown_png = (_cell_number(), getattr(widget, "_revision", 0))


# ----------------------------------------------------------------------
# help overlay
# ----------------------------------------------------------------------


class PlotWindow(QtWidgets.QMainWindow):
    """Base window: a key legend on ``h`` and closing on ``q``.

    Also the home of the axis and reload keys every plot shares: ``z``
    and ``u`` put the y and x ranges back to their defaults, as
    difmap's ``Z`` and ``U`` do, and ``r`` rebuilds the plot from the
    data, for when it has been changed from the prompt behind the
    window's back.
    """

    #: Default window size, clipped to the screen. These plots are read
    #: in detail - a page of stacked panels, or a map beside its colour
    #: bar and histogram - so they open large.
    DEFAULT_SIZE = (1150, 800)

    def __init_subclass__(cls, **kwargs):
        """Count redraws: every `refresh()` a plot class defines bumps
        `_revision`, which is how the notebook display knows whether a
        plot has changed since it was shown."""
        super().__init_subclass__(**kwargs)
        fn = cls.__dict__.get("refresh")
        if fn is None or getattr(fn, "_counts_revisions", False):
            return

        @functools.wraps(fn)
        def refresh(self, *args, **kw):
            self._revision = getattr(self, "_revision", 0) + 1
            return fn(self, *args, **kw)

        refresh._counts_revisions = True
        cls.refresh = refresh

    def __init__(self, title):
        ensure_app()
        super().__init__()
        self.setWindowTitle(title)
        self._help = None
        self.resize(*fit_to_screen(*self.DEFAULT_SIZE))
        _register_window(self)

    # Subclasses list their own keys first; these are appended.
    COMMON_KEYS = (
        ("z / u", "restore the y / x axis range"),
        ("r", "reload the plot from the data"),
        ("h", "show or hide this help"),
        ("q", "close the window"),
    )

    # ---- output without a screen (notebooks, pipelines) --------------

    def render(self, width=None, height=None):
        """The plot drawn off-screen, as a QImage.

        Works whether or not the window is shown, and never puts one on
        the screen: an unshown plot is laid out and painted as a window
        would be, then hidden again. `width`/`height` resize it first.
        """
        if width or height:
            self.resize(int(width or self.width()), int(height or self.height()))
        was_visible = self.isVisible()
        if not was_visible:
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)
            self.show()
        try:
            app = ensure_app()
            for _ in range(2):  # the layout, then the items that follow it
                app.processEvents()
            return self.grab().toImage()
        finally:
            if not was_visible:
                self.hide()
                self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, False)

    def to_png(self, width=None, height=None) -> bytes:
        """The plot as PNG image data."""
        buf = QtCore.QBuffer()
        buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
        self.render(width, height).save(buf, "PNG")
        return bytes(buf.data())

    def savefig(self, path, width=None, height=None):
        """Write the plot to an image file, its format taken from the
        extension (.png, .jpg, ...).

        On a paged plot (vplot, cpplot, corplot, fplot) a `path`
        containing ``{page}`` writes every page, numbered from 1, and
        returns the list of files; otherwise the file written is
        returned.
        """
        path = os.fspath(path)
        if "{page}" in path and hasattr(self, "set_page"):
            start, written = self.page, []
            try:
                for i in range(self.npages):
                    self.set_page(i)
                    written.append(self._write_image(
                        path.replace("{page}", str(i + 1)), width, height))
            finally:
                self.set_page(start)
            return written
        return self._write_image(path, width, height)

    def _write_image(self, path, width, height):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if not self.render(width, height).save(path):
            raise OSError(f"could not write {path} (unsupported image format?)")
        return path

    def _repr_png_(self):
        """Jupyter's rich display of the plot."""
        shown = (_cell_number(), getattr(self, "_revision", 0))
        if getattr(self, "_shown_png", None) == shown:
            return None  # already in this cell's output, unchanged
        return self.to_png()

    def __repr__(self):
        return f"<{self.windowTitle()}>"

    # ---- axis ranges and reloading ------------------------------------

    def view_boxes(self) -> list:
        """The view boxes ``z`` and ``u`` act on."""
        return []

    def default_y_range(self, vb):
        """The y range ``z`` restores for `vb`; None to autoscale.

        Phase panels come back to +-180 rather than to the spread of
        whatever is displayed, which is their default view.
        """
        return None

    def reset_ranges(self, axis="both") -> int:
        """Put the axis ranges back to their defaults, on every panel.

        Returns how many panels were reset.
        """
        boxes = self.view_boxes()
        for vb in boxes:
            auto = False
            if axis in ("y", "both"):
                r = self.default_y_range(vb)
                if r is None:
                    vb.enableAutoRange(axis=vb.YAxis)
                    auto = True
                else:
                    vb.setYRange(*r, padding=0)
            # Stacked panels share one x axis: auto-ranging the panel
            # that owns it is what moves the page, and re-enabling it on
            # a follower only fights the link.
            if axis in ("x", "both") and vb.linkedView(vb.XAxis) is None:
                vb.enableAutoRange(axis=vb.XAxis)
                auto = True
            if auto:
                # pyqtgraph only works out what "auto" means at the next
                # paint. Do it now, so that the range a plot opens with
                # is decided here rather than by paint timing.
                vb.updateAutoRange()
        return len(boxes)

    def reload(self):
        """Rebuild the plot from the data (difmap's ``L``).

        The data can be edited from the prompt while a window is open -
        flagged, calibrated, re-imaged - and nothing tells the window
        about it; ``r`` is how you catch up.
        """
        refresh = getattr(self, "refresh", None)
        if callable(refresh):
            refresh()

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

    def closeEvent(self, ev):
        """Ask Qt to destroy the window once it is out of the way.

        The C++ object goes when the event loop next runs, rather than
        whenever Python happens to collect the wrapper - which, at
        interpreter shutdown, is too late to be safe. See
        `close_all_windows`.
        """
        super().closeEvent(ev)
        self.deleteLater()

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "h":
            self.toggle_help()
        elif key == "q":
            self.close()
        elif key == "z":
            self.reset_ranges("y")
        elif key == "u":
            self.reset_ranges("x")
        elif key == "r":
            self.reload()
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
        # Ignored stations stay ignored, even if an edit unflagged
        # part of their data; do it before the snapshot so that undo
        # restores what was really there.
        self.obs._reapply_ignores()
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
        self._scaled = False

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

    #: Width in pixels of the left axis of every stacked panel.
    LEFT_AXIS_WIDTH = 68

    def _share_x_axis(self):
        """Stack the panels on one common x axis.

        Every panel *but the first* is tied to it, and only the bottom
        one keeps its tick labels, so the panels of a page read as one
        plot with a single time (or UV radius) axis rather than as
        separate ones.

        The first panel must never be linked to itself: pyqtgraph
        answers a link by taking its range from the other view and
        switching auto-ranging off, so a self-link freezes the whole
        page at the empty default view and no data is ever shown.
        """
        for panel in self._panels[:-1]:
            panel.plot.getAxis("bottom").setStyle(showValues=False)
            panel.plot.setLabel("bottom", "")
        for panel in self._panels[1:]:
            panel.plot.setXLink(self._panels[0].plot)
        # One width for every left axis: left to itself each axis is as
        # wide as its own tick labels ("-150" against "4"), and the plot
        # areas of an amplitude and a phase panel then start at
        # different x.
        for panel in self._panels:
            panel.plot.getAxis("left").setWidth(self.LEFT_AXIS_WIDTH)

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
                    FastScatter(
                        d["x"][m], y[m], size=4, pen=None,
                        brush=pg.mkBrush(*col, 190),
                    ),
                )
            if self._show_flagged:
                m = sub & ~good & (d["wt"] != 0)
                if m.any():
                    self._add_item(
                        panel,
                        FastScatter(
                            d["x"][m], y[m], size=6, symbol="x",
                            pen=pg.mkPen(*FLAG_COLOR), brush=None,
                        ),
                    )
            self._decorate(panel, d, sub, good)
        if not self._scaled:
            # Open on the data, not on pyqtgraph's empty default view:
            # the panels are built and filled inside this first refresh,
            # so this is the earliest point at which the extent of the
            # data is known.
            self._scaled = True
            self.reset_ranges()
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
                    FastScatter(
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

    def view_boxes(self):
        return [panel.vb for panel in self._panels]

    def default_y_range(self, vb):
        panel = getattr(vb, "panel", None)
        if panel is None:
            return None
        if panel.key == "phase":
            return (-180, 180)
        if panel.key == "amp":
            return self._flat_amp_range(panel)
        return None

    def _flat_amp_range(self, panel):
        """A fixed range for an amplitude panel too flat to autoscale.

        A point source, or data scaled by one gain, can be constant to
        the last bit of float32; autoscaling that zooms onto rounding
        noise and pyqtgraph drops the tick labels altogether. Such a
        panel is ranged to a thousandth of its level instead (the floor
        corplot uses). Anything with real spread gets None, i.e. normal
        autoscaling.
        """
        d = self._data
        if d is None or panel.key not in d:
            return None
        m = self._panel_mask(panel, d) & (d["wt"] > 0)
        vals = [np.asarray(d[panel.key])[m]]
        model = self._model(d, panel.key)
        if model is not None:
            vals.append(np.asarray(model)[m])
        v = np.concatenate(vals)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        lo, hi = float(v.min()), float(v.max())
        floor = 1e-3 * abs(0.5 * (lo + hi))
        if hi - lo >= floor:
            return None
        pad = max(floor, 1e-12)
        return (lo - pad, hi + pad)

    def key_help(self):
        return [
            ("shift+drag", "flag the points inside the box"),
            ("ctrl+drag", "unflag the points inside the box"),
            ("f / F", "flag / unflag the point nearest the cursor"),
            ("ctrl+z / ctrl+shift+z", "undo / redo the last flag edit"),
            ("x", "show or hide the flagged points"),
            ("drag / wheel", "pan / zoom"),
        ]

    def _undo_redo(self, redo):
        n = self.history.redo() if redo else self.history.undo()
        what = "redo" if redo else "undo"
        self.refresh()
        self._message(f"nothing to {what}" if n is None
                      else f"{'redid' if redo else 'undid'} an edit of "
                           f"{n} samples")

    def keyPressEvent(self, ev):
        key = ev.text()
        low = key.lower()
        mods = ev.modifiers()
        ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        # Undo/redo take the modifier every other application uses,
        # which leaves z, u and r for the axis and reload keys difmap
        # binds them to.
        if ctrl and ev.key() == QtCore.Qt.Key.Key_Z:
            self._undo_redo(redo=shift)
        elif ctrl and ev.key() == QtCore.Qt.Key.Key_Y:
            self._undo_redo(redo=True)
        elif low == "x":
            self._show_flagged = not self._show_flagged
            self.refresh()
        elif key in ("f", "F") and self._mouse_pos is not None:
            panel = self._active or (self._panels[0] if self._panels else None)
            if panel is not None:
                x, y = self._view_coords(panel, self._mouse_pos)
                self._flag_nearest(panel, x, y, flag=(key == "f"))
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


# ----------------------------------------------------------------------
# time axes with the dead time cut out
# ----------------------------------------------------------------------

class TimeGaps:
    """Maps observing time onto an x axis with the long gaps cut out.

    VLBI scans are often separated by far more time than they last, and
    a continuous time axis then spends most of its width on nothing.
    Any gap between consecutive integrations longer than `min_frac` of
    the whole time range is cut down to `gap_frac` of the time that is
    kept, so the scans sit side by side on one axis.

    Inside a segment the mapping is a plain shift - an hour of data is
    still an hour wide - and the first segment is not shifted at all,
    so with nothing to cut plot coordinates are simply the times.
    Times are in whatever unit they are given in (the plots use hours).
    """

    def __init__(self, times, min_frac=0.10, gap_frac=0.02):
        t = np.unique(np.asarray(times, dtype=float))
        t = t[np.isfinite(t)]
        self.segments: list[tuple[float, float]] = []
        self.offsets: list[float] = []
        if t.size == 0:
            return
        span = float(t[-1] - t[0])
        cuts = (np.nonzero(np.diff(t) > min_frac * span)[0]
                if span > 0 else np.zeros(0, dtype=int))
        starts = np.concatenate(([0], cuts + 1))
        ends = np.concatenate((cuts, [t.size - 1]))
        segs = [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]
        kept = sum(b - a for a, b in segs)
        if len(segs) > 1 and kept <= 0.0:
            # Nothing but isolated instants: no width to scale the cuts
            # by, and nothing a cut would make easier to read.
            segs = [(float(t[0]), float(t[-1]))]
        width = gap_frac * kept
        x = segs[0][0]
        for a, b in segs:
            self.offsets.append(x - a)
            x += (b - a) + width
        self.segments = segs

    @classmethod
    def of(cls, obs, **kwargs):
        """The cut time axis of an observation, in hours."""
        return cls(np.asarray(obs._core.times()) / 3600.0, **kwargs)

    def __bool__(self):
        """True if anything was cut."""
        return len(self.segments) > 1

    @property
    def breaks(self) -> list[tuple[float, float]]:
        """The blank stretches the cuts leave, in plot coordinates."""
        return [
            (b0 + o0, a1 + o1)
            for (_, b0), o0, (a1, _), o1 in zip(
                self.segments[:-1], self.offsets[:-1],
                self.segments[1:], self.offsets[1:],
            )
        ]

    def _index(self, starts, values):
        idx = np.searchsorted(starts, values, side="right") - 1
        return np.clip(idx, 0, len(starts) - 1)

    def compress(self, t):
        """Times -> plot coordinates."""
        t = np.asarray(t, dtype=float)
        if not self.segments:
            return t.copy()
        starts = np.array([a for a, _ in self.segments])
        return t + np.asarray(self.offsets)[self._index(starts, t)]

    def expand(self, x):
        """Plot coordinates -> times (the inverse of `compress`)."""
        x = np.asarray(x, dtype=float)
        if not self.segments:
            return x.copy()
        starts = np.array([a + o for (a, _), o in zip(self.segments, self.offsets)])
        return x - np.asarray(self.offsets)[self._index(starts, x)]

    def in_break(self, x):
        """True where a plot coordinate falls in a cut."""
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape, dtype=bool)
        for lo, hi in self.breaks:
            out |= (x > lo) & (x < hi)
        return out


class TimeGapAxis(pg.AxisItem):
    """A bottom axis over `TimeGaps` coordinates, labelled with the
    real times; ticks that land in a cut are left unlabelled."""

    def __init__(self, gaps, **kwargs):
        super().__init__(orientation="bottom", **kwargs)
        self.gaps = gaps

    def tickValues(self, minVal, maxVal, size):
        """Round-number times within each scan, rather than ticks spaced
        evenly over the cut axis - which would label every scan at some
        arbitrary offset from its start."""
        levels = super().tickValues(minVal, maxVal, size)
        if not self.gaps:
            return levels
        out = []
        for spacing, _ in levels:
            ticks = []
            for (a, b), off in zip(self.gaps.segments, self.gaps.offsets):
                # The part of this scan in view, in real time.
                lo, hi = max(a, minVal - off), min(b, maxVal - off)
                if lo > hi:
                    continue
                first = np.ceil(lo / spacing - 1e-9) * spacing
                real = np.arange(first, hi + 1e-9 * spacing, spacing)
                ticks.extend(real + off)
            out.append((spacing, ticks))
        return out

    def tickStrings(self, values, scale, spacing):
        values = np.asarray(values, dtype=float)
        labels = super().tickStrings(self.gaps.expand(values), scale, spacing)
        blank = self.gaps.in_break(values)
        return ["" if b else s for s, b in zip(labels, blank)]


#: Shading of a cut in a time axis.
BREAK_BRUSH = (120, 120, 120, 45)


def install_time_axis(plot, gaps):
    """Give a PlotItem a time axis with `gaps` cut out, and shade the
    cuts so that they read as breaks rather than as missing data.

    Returns the shading items added (none when nothing was cut). The
    axis is installed either way, so tick labels are always real times.
    """
    old = plot.getAxis("bottom")
    label, grid = old.labelText, old.grid
    axis = TimeGapAxis(gaps)
    plot.setAxisItems({"bottom": axis})
    axis.setGrid(grid)
    if label:
        plot.setLabel("bottom", label)
    items = []
    for lo, hi in gaps.breaks:
        region = pg.LinearRegionItem(
            values=(lo, hi), movable=False, brush=pg.mkBrush(*BREAK_BRUSH),
            pen=pg.mkPen(120, 120, 120, 140, style=QtCore.Qt.PenStyle.DashLine),
        )
        region.setZValue(-10)
        plot.addItem(region, ignoreBounds=True)
        items.append(region)
    return items
