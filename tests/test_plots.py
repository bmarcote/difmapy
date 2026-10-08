"""Offscreen smoke tests of the pyqtgraph plot layer: construction,
refresh, programmatic flagging and window syncing."""

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""  # the gtk3 theme aborts w/o display

pg = pytest.importorskip("pyqtgraph")

import difmapy
from difmapy.observation import SHAPE_BITS
from difmapy.plots.mapplot import MapPlot
from pyqtgraph.Qt import QtCore
from difmapy.plots.points import RadPlot, UVPlot, VPlot

CELL = 0.25
NX = 256


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(NX, CELL)
    return o


class FakeDrag:
    """The subset of pyqtgraph's MouseDragEvent interface used by
    ViewBox.mouseDragEvent and SelectViewBox.

    Building a real MouseDragEvent needs scene-level Qt plumbing
    (currentItem, buttonDownScenePos), which would test Qt rather than
    our handler; this stub exercises our code path directly.
    """

    def __init__(self, vb, pos, down_pos, last_pos, modifier, start, finish):
        from pyqtgraph import Point
        from pyqtgraph.Qt import QtCore

        # ViewBox event positions live in its child-item frame, which
        # mapToView() converts back to data coordinates.
        self._vb = vb
        self._pos = Point(vb.mapFromView(QtCore.QPointF(*pos)))
        self._down = Point(vb.mapFromView(QtCore.QPointF(*down_pos)))
        self._last = Point(vb.mapFromView(QtCore.QPointF(*last_pos)))
        self._mod = modifier
        self.start = start
        self.finish = finish
        self.accepted = False
        self._btn = QtCore.Qt.MouseButton.LeftButton

    def pos(self):
        return self._pos

    def lastPos(self):
        return self._last

    def buttonDownPos(self, btn=None):
        return self._down

    def modifiers(self):
        return self._mod

    def button(self):
        return self._btn

    def buttons(self):
        return self._btn

    def isStart(self):
        return self.start

    def isFinish(self):
        return self.finish

    def accept(self):
        self.accepted = True

    def ignore(self):
        self.accepted = False


def _drag(vb, p0, p1, modifier):
    """Simulate a rubber-band drag across the view box in view coords."""
    steps = ((p0, p0, True, False), (p1, p0, False, False), (p1, p1, False, True))
    last = p0
    for pos, _, start, finish in steps:
        vb.mouseDragEvent(
            FakeDrag(vb, pos, p0, last, modifier, start, finish)
        )
        last = pos


def test_drag_selection_flags(obs):
    """A Shift-drag must flag exactly the enclosed points, and a
    Ctrl-drag must unflag them again (the documented interaction)."""
    from pyqtgraph.Qt import QtCore

    p = RadPlot(obs, quantity="amp")
    vb = p._panels[0].vb
    d = p._data
    n_good = int((d["wt"] > 0).sum())
    # Box covering the upper half of the amplitude range.
    ycut = float(np.median(d["amp"]))
    xhi, yhi = float(d["x"].max()) * 2, float(d["amp"].max()) * 2
    expect = int(((d["amp"] >= ycut) & (d["wt"] > 0)).sum())
    assert 0 < expect < n_good

    _drag(vb, (-1.0, ycut), (xhi, yhi),
          QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert int((p._data["wt"] < 0).sum()) == expect

    _drag(vb, (-1.0, -yhi), (xhi, yhi),
          QtCore.Qt.KeyboardModifier.ControlModifier)
    assert int((p._data["wt"] > 0).sum()) == n_good
    p.close()


def test_plain_drag_pans_not_flags(obs):
    """Without a modifier, dragging must pan (never flag)."""
    from pyqtgraph.Qt import QtCore

    p = RadPlot(obs, quantity="amp")
    vb = p._panels[0].vb
    n_good = int((p._data["wt"] > 0).sum())
    before = vb.viewRange()
    _drag(vb, (0.0, 0.0), (5.0, 5.0), QtCore.Qt.KeyboardModifier.NoModifier)
    assert int((p._data["wt"] > 0).sum()) == n_good  # nothing flagged
    assert vb.viewRange() != before  # the view moved
    p.close()


def test_radplot_flagging(obs):
    p = RadPlot(obs)
    panel = p._panels[0]
    d = p._data
    n0 = int((d["wt"] > 0).sum())
    assert n0 == obs._core.nrow * obs.nif
    # Programmatic box flag: everything beyond the median uv radius.
    cut = float(np.median(d["x"]))
    p._apply_box(panel, cut, 1e9, -1e9, 1e9, flag=True)
    d = p._data
    nf = int((d["wt"] < 0).sum())
    assert nf > 0
    assert (np.hypot(*_uv(obs)) / 1e6 >= cut).sum() == nf
    # Unflag box restores everything.
    p._apply_box(panel, -1e9, 1e9, -1e9, 1e9, flag=False)
    assert int((p._data["wt"] > 0).sum()) == n0
    p.close()


def test_radplot_defaults_to_amp_and_phase(obs):
    """The default is stacked amplitude and phase panels; the other
    spellings of that mode agree, and single quantities give one."""
    p = RadPlot(obs)
    assert [panel.key for panel in p._panels] == ["amp", "phase"]
    p.close()
    for spec in ("anp", "a&p", "ap"):
        q = RadPlot(obs, quantity=spec)
        assert [panel.key for panel in q._panels] == ["amp", "phase"]
        q.close()
    q = RadPlot(obs, quantity="phase")
    assert [panel.key for panel in q._panels] == ["phase"]
    q.close()
    with pytest.raises(ValueError, match="unknown quantity"):
        RadPlot(obs, quantity="nonsense")


def test_flagged_points_hidden_by_default(obs):
    """Flagged data is not drawn until "x" asks for it."""
    p = RadPlot(obs, quantity="amp")
    panel = p._panels[0]
    p._apply_box(panel, float(np.median(p._data["x"])), 1e9, -1e9, 1e9, flag=True)
    assert not p._show_flagged
    n_shown = len(p._items)
    p._show_flagged = True
    p.refresh()
    assert len(p._items) > n_shown  # the crosses appeared
    p.close()


def test_undo_redo_restores_flags_exactly(obs):
    """z/r must undo and redo a flag edit, restoring the FLAG column
    bit for bit - including data that was already flagged."""
    before = obs.flags.copy()
    p = RadPlot(obs, quantity="amp")
    panel = p._panels[0]
    cut = float(np.median(p._data["x"]))
    n = p._apply_box(panel, cut, 1e9, -1e9, 1e9, flag=True)
    assert n > 0
    flagged = obs.flags.copy()
    assert not np.array_equal(before, flagged)

    assert p.history.undo() == n
    assert np.array_equal(obs.flags, before)
    assert p.history.redo() == n
    assert np.array_equal(obs.flags, flagged)
    # A second undo has nothing left to do.
    assert p.history.undo() == n
    assert p.history.undo() is None
    assert np.array_equal(obs.flags, before)
    p.close()


def test_stacked_panels_open_on_the_data(obs):
    """The default view must show the data. The first panel used to be
    x-linked to itself, which switches pyqtgraph's auto-ranging off and
    left every stacked plot on the empty default 0-1 view until the user
    pressed "u" and "z"."""
    for plot in (RadPlot(obs), VPlot(obs, nplot=1)):
        for panel in plot._panels:
            (x0, x1), (y0, y1) = panel.vb.viewRange()
            d = plot._data
            sub = plot._panel_mask(panel, d)
            good = sub & (d["wt"] > 0)
            assert good.any()
            # Every point of this panel is inside the opening view.
            assert x0 <= d["x"][good].min() and x1 >= d["x"][good].max()
            y = d[panel.key][good]
            assert y0 <= y.min() and y1 >= y.max()
        plot.close()


def test_flat_amplitudes_are_not_ranged_onto_rounding_noise(obs):
    """A point source has the same amplitude on every baseline to the
    last bit of float32. Autoscaling onto that spread showed nothing but
    noise, with no tick labels at all; the panel keeps a floor of a
    thousandth of the level instead."""
    p = RadPlot(obs, quantity="amp")
    d = p._data
    amp = d["amp"][d["wt"] > 0]
    assert np.ptp(amp) < 1e-3 * amp.mean()        # the data really is flat
    y0, y1 = p._panels[0].vb.viewRange()[1]
    assert y1 - y0 >= 2e-3 * amp.mean() * 0.99
    assert y0 <= amp.min() and y1 >= amp.max()
    p.close()


def test_radplot_shows_a_fitted_model(obs):
    """A fitted model is drawn straight after modelfit. It used to stay
    out of the plotted model visibilities until it was kept."""
    (x, y), _ = obs.peak_offset()
    obs.addcmp(1.0, x, y, free=["flux", "pos"])
    obs.modelfit(niter=10, quiet=True)

    p = RadPlot(obs)
    assert p.has_model()
    good = p._data["wt"] > 0
    assert np.isfinite(p._data["model_amp"][good]).any()
    assert np.any(p._data["model_amp"][good] > 0)
    p.close()


def test_radplot_antenna_highlight_cycles(obs):
    """n/p walk through the antennas and label the highlighted one."""
    p = RadPlot(obs)
    assert p._highlight is None
    assert p.cycle_antenna(1) == 0
    assert p._labels["amp"].toPlainText() == obs.antennas[0]
    assert p.cycle_antenna(1) == 1
    assert p.cycle_antenna(-1) == 0
    assert p.cycle_antenna(-1) is None  # back to "all antennas"
    assert p._labels["amp"].toPlainText() == ""
    p.close()


def test_radplot_shows_the_model(obs):
    """With a model defined, its predicted visibilities are drawn."""
    p = RadPlot(obs)
    assert not p.has_model()
    p.close()
    obs.invert()
    obs.clean(50, 0.1, quiet=True)
    q = RadPlot(obs)
    assert q.has_model()
    assert np.isfinite(q._data["model_amp"]).any()
    # The model amplitude of a point-like source is near its flux.
    good = q._data["wt"] > 0
    assert np.median(q._data["model_amp"][good]) > 0
    q.close()


def _uv(obs):
    _, _, _, us, vs, _ = obs._core.rows()
    sel = obs._core.selection()
    freq = np.asarray(sel["if_freq"])
    return (us[:, None] * freq).ravel(), (vs[:, None] * freq).ravel()


def test_uvplot_and_vplot(obs):
    p = UVPlot(obs)
    assert len(p._data["x"]) == 2 * obs._core.nrow * obs.nif  # conjugates
    p.close()
    v = VPlot(obs, reftel="AN1", nplot=2)
    # AN1 participates in 4 of the 10 baselines, two to a page.
    assert len(v._baselines) == 4
    assert v.npages == 2
    # Only the displayed baselines are collected, amp and phase each.
    assert [(panel.key, panel.group) for panel in v._panels] == [
        ("amp", 0), ("phase", 0), ("amp", 1), ("phase", 1)
    ]
    assert len(v._data["x"]) == obs._core.nrow * obs.nif * 2 // 10
    panel = v._panels[0]
    v._flag_nearest(panel, float(v._data["x"][0]), float(v._data["amp"][0]),
                    flag=True)
    assert (v._data["wt"] < 0).any()
    v.close()


def test_vplot_pages_and_flag_modes(obs):
    """n/p page through the baselines; space switches flagging between
    the displayed baseline and every baseline of its first antenna."""
    v = VPlot(obs, nplot=3)
    assert len(v._baselines) == 10  # 5 antennas
    assert v.npages == 4
    first = [panel.label for panel in v._panels]
    v.page = 1
    v._relayout()
    assert [panel.label for panel in v._panels] != first

    v.page = 0
    v._relayout()
    idx = np.nonzero((v._data["group"] == 0) & (v._data["wt"] > 0))[0][:1]

    assert not v.by_antenna
    bl_ops = v._edit_ops(idx, True)
    v.by_antenna = True
    ant_ops = v._edit_ops(idx, True)
    # Antenna-based flagging reaches every baseline of that antenna at
    # the same integration, so it always touches more rows.
    assert sum(len(r) for r, _, _ in ant_ops) > sum(len(r) for r, _, _ in bl_ops)
    v.close()


def test_mapplot_windows_and_clean(obs):
    p = MapPlot(obs, quiet=True)
    assert p.img.image is not None
    # Add a window programmatically like a double-click would.
    p._add_roi(2.0, 6.0, -4.5, -0.5)
    p._rois_to_obs()
    assert len(obs.windows) == 1
    assert obs.windows[0] == (2.0, 6.0, -4.5, -0.5)
    # Clean uses the window; the peak lies inside it.
    res = obs.clean(300, 0.1, quiet=True)
    assert res["cleaned_flux"] > 0
    p.refresh()
    assert len(p._rois) == 1
    # Every display the number keys select must render.
    for what in ("beam", "clean", "model", "map"):
        p.what = what
        p.refresh()
    p.close()


def test_mapplot_weighting_box(obs):
    """The box at the top re-images with robust -2 ... 2, and goes back
    to difmap's own weighting."""
    obs.uvweight(0, -1)          # natural, difmap style
    p = MapPlot(obs, quiet=True)
    box = p.weighting
    assert box.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus
    assert [box.itemData(i) for i in range(box.count())] == \
        [None, -2.0, -1.0, 0.0, 1.0, 2.0]
    assert box.currentIndex() == 0 and "0, -1" in box.itemText(0)
    beams = {}
    for r in (-2.0, 2.0):
        box.setCurrentIndex(box.findData(r))
        assert obs.robust == r and obs._invert_result is not None
        beams[r] = obs.estimated_beam[0]
    assert beams[-2.0] < beams[2.0], "uniform gives the sharper beam"
    box.setCurrentIndex(0)
    assert obs.robust is None
    assert (obs._binwid, obs._errpow) == (0.0, -1.0)
    # A robustness set at the prompt shows up in the box.
    obs.uvweight(robust=0.5)
    p.refresh()
    assert box.itemData(box.currentIndex()) == 0.5
    p.close()


def test_contour_levels_and_tracing():
    from difmapy.plots.contours import contour_levels, contour_segments

    pos, neg = contour_levels(0.01, 1.0, -0.05)
    assert pos[0] == pytest.approx(0.03) and neg[0] == pytest.approx(-0.03)
    np.testing.assert_allclose(pos[1:] / pos[:-1], np.sqrt(2.0))
    np.testing.assert_allclose(neg[1:] / neg[:-1], np.sqrt(2.0))
    assert pos[-1] <= 1.0 < pos[-1] * np.sqrt(2.0)
    assert neg[-1] >= -0.05 > neg[-1] * np.sqrt(2.0)
    # Nothing above 3 sigma, or no usable noise: no levels.
    assert contour_levels(0.01, 0.02, -0.02)[0].size == 0
    assert contour_levels(0.0, 1.0)[0].size == 0

    # A circle of radius 40 pixels about an off-grid centre.
    yy, xx = np.mgrid[:200, :200]
    r = np.hypot(xx - 100.3, yy - 99.6)
    xs, ys = contour_segments(-r, -40.0)
    assert xs.size and xs.size % 2 == 0
    np.testing.assert_allclose(np.hypot(xs - 100.3, ys - 99.6), 40.0, atol=0.01)
    length = np.hypot(np.diff(xs)[::2], np.diff(ys)[::2]).sum()
    assert length == pytest.approx(2 * np.pi * 40.0, rel=1e-3)
    # A saddle: two crossing-free segments, not a cross.
    xs, ys = contour_segments(np.array([[1.0, 0.0], [0.0, 1.0]]), 0.5)
    assert xs.size == 4
    assert contour_segments(np.zeros((8, 8)), 1.0)[0].size == 0


def test_mapplot_contours_on_the_clean_map(obs):
    from pyqtgraph.Qt import QtGui

    obs.clean(200, 0.1, quiet=True)
    p = MapPlot(obs, what="clean", quiet=True)
    pos, neg, rms = p.contour_levels
    assert rms == pytest.approx(obs.noise_stats()["rms"])
    assert pos[0] == pytest.approx(3.0 * rms) and len(pos) > 3
    np.testing.assert_allclose(pos[1:] / pos[:-1], np.sqrt(2.0))
    drawn = [lv for _, lv in p._contour_items]
    assert drawn and set(drawn) <= set(pos) | set(neg)

    # A map with a deep negative: its contours are dashed, the positive
    # ones solid, and each is light or dark against the colour scale.
    data = np.array(obs.restored_map, copy=True)
    data[obs._ny // 2 + 20, obs._nx // 2 + 20] = -40.0 * rms
    p._draw_contours(data)
    styles = {lv < 0: item.pen().style() for item, lv in p._contour_items}
    assert styles[True] == QtCore.Qt.PenStyle.DashLine
    assert styles[False] == QtCore.Qt.PenStyle.SolidLine
    lo, hi = p._cbar.levels()
    colours = {item.pen().color().lightness() > 128
               for item, lv in p._contour_items}
    assert colours == {True, False}

    # Only the restored map is contoured, and "k" switches them off.
    for what in ("map", "beam", "model"):
        p.what = what
        p.refresh()
        assert not p._contour_items
    p.what = "clean"
    p.refresh()
    assert p._contour_items
    p.keyPressEvent(QtGui.QKeyEvent(
        QtCore.QEvent.Type.KeyPress, 0,
        QtCore.Qt.KeyboardModifier.NoModifier, "k"))
    assert not p.contours and not p._contour_items
    p.close()


def test_vplot_panels_have_identical_axes(obs):
    """Amplitude and phase panels: the same axes, no extra one on the
    right, and plot areas that start and end at the same x."""
    v = VPlot(obs, nplot=2)
    v.resize(900, 700)
    v.render(900, 700)
    assert {p.key for p in v._panels} == {"amp", "phase"}
    for panel in v._panels:
        assert not panel.plot.getAxis("right").isVisible()
        assert not panel.plot.getAxis("top").isVisible()
    left = {round(p.plot.vb.sceneBoundingRect().left(), 1) for p in v._panels}
    right = {round(p.plot.vb.sceneBoundingRect().right(), 1) for p in v._panels}
    assert len(left) == 1 and len(right) == 1
    v.close()


def test_mapplot_difmap_colour_maps(obs):
    """Difmap's pseudo-colour table by default, its grey scale on "g",
    and - as in difmap - colours spanning the map from minimum to peak."""
    from pyqtgraph.Qt import QtGui

    from difmapy.plots.mapplot import COLOR_MAPS

    obs.clean(100, 0.1, quiet=True)
    p = MapPlot(obs, what="clean", quiet=True)
    assert p.cmap == "color"
    valid = obs.valid(np.asarray(obs.restored_map))
    lo, hi = p._cbar.levels()
    assert lo == pytest.approx(float(valid.min()))
    assert hi == pytest.approx(float(valid.max()))
    # difmap's rainbow: dark blue at the bottom, red at the top, cyan
    # and yellow on the way.
    lut = COLOR_MAPS["color"]().map([0.0, 0.33, 0.67, 1.0], mode="byte")
    np.testing.assert_array_equal(
        lut[:, :3], [[0, 0, 76], [0, 255, 255], [255, 255, 0], [255, 0, 0]])

    def key(k):
        p.keyPressEvent(QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress, 0,
            QtCore.Qt.KeyboardModifier.NoModifier, k))

    key("g")
    assert p.cmap == "grey"
    grey = p._cbar.colorMap().map([0.0, 1.0], mode="byte")[:, :3]
    np.testing.assert_array_equal(grey, [[0, 0, 0], [255, 255, 255]])
    assert p._cbar.levels() == (lo, hi), "the range is kept"
    key("g")
    assert p.cmap == "color"
    # The log scale stretches whichever map is showing.
    p.set_scale("log")
    key("g")
    assert p.cmap == "grey" and p.scale == "log"
    p.close()
    q = MapPlot(obs, quiet=True, cmap="B&W")
    assert q.cmap == "grey"
    q.close()
    with pytest.raises(ValueError, match="colour map"):
        MapPlot(obs, quiet=True, cmap="jet")


def test_mapplot_number_keys_select_the_display(obs):
    from pyqtgraph.Qt import QtCore, QtGui

    p = MapPlot(obs, quiet=True)
    for key, what in (("2", "beam"), ("3", "clean"), ("4", "model"), ("1", "map")):
        p.keyPressEvent(
            QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, 0,
                            QtCore.Qt.KeyboardModifier.NoModifier, key)
        )
        assert p.what == what
    p.close()


def test_mapplot_histogram_tracks_the_colorbar(obs):
    """The panel beside the colour bar holds the pixel histogram and
    the Gaussian fitted to the noise, over the displayed range."""
    p = MapPlot(obs, quiet=True)
    assert len(p._hist_items) == 2  # histogram + noise Gaussian
    lo, hi = (float(v) for v in p._cbar.levels())
    assert p.hist.vb.viewRange()[1] == pytest.approx([lo, hi], rel=1e-6)
    counts, centres = p._hist_items[0].getData()
    assert centres.min() >= lo and centres.max() <= hi
    assert counts.max() > 0
    p.close()


def test_mapplot_gaussian_placement(obs):
    """m, then clicks for the centre and two points on the component,
    adds a Gaussian; d short-circuits to a point source or a circular
    Gaussian."""
    p = MapPlot(obs, quiet=True)
    p.start_gaussian()
    assert p._gauss_stage == "center"
    p._gauss_click(1.0, 2.0)
    assert p._gauss_stage == "first"
    p._gauss_click(1.0, 5.0)  # 3 mas due north of the centre
    assert p._gauss_stage == "second"
    p._gauss_click(2.5, 2.0)  # 1.5 mas due east: the minor axis
    c = p._placed[-1]
    assert p._gauss_stage is None
    assert c["type"] == "gauss"
    assert (c["x"], c["y"]) == pytest.approx((1.0, 2.0))
    assert c["major"] == pytest.approx(6.0)      # twice the click radius
    assert c["phi"] == pytest.approx(0.0)        # major axis due north
    assert c["ratio"] == pytest.approx(0.5)      # 3 mas minor / 6 mas major
    assert len(p._model_items) >= 1              # the ellipse is drawn

    # "d" at the first-point stage makes it a point source instead.
    p.start_gaussian()
    p._gauss_click(-2.0, 0.0)
    p._add_gaussian(type="delta")
    assert p._placed[-1]["type"] == "delta"
    assert p._placed[-1]["major"] == 0.0
    p.close()


def test_mapplot_gaussian_clicks_are_two_points_on_the_ellipse(obs):
    """The two clicks after the centre are just two points *on* the
    component: either may be the longer, and they need not be 90 degrees
    apart. Clicking the short axis first used to clip the axial ratio to
    1 and silently produce a circular Gaussian."""
    p = MapPlot(obs, quiet=True)

    def place(c, p1, p2):
        p.start_gaussian()
        p._gauss_click(*c)
        p._gauss_click(*p1)
        p._gauss_click(*p2)
        return p._placed[-1]

    # Short axis clicked first: still elliptical, and the same ellipse
    # as clicking them the other way round.
    a = place((0.0, 0.0), (1.5, 0.0), (0.0, 3.0))
    b = place((0.0, 0.0), (0.0, 3.0), (1.5, 0.0))
    assert a["ratio"] == pytest.approx(0.5) and a["ratio"] == b["ratio"]
    assert a["major"] == pytest.approx(6.0) == b["major"]
    assert a["phi"] == pytest.approx(0.0) == b["phi"]

    # Clicks 45 degrees apart: the longer sets the major axis, and the
    # ellipse still passes exactly through the shorter one.
    c = place((0.0, 0.0), (0.0, 4.0), (2.0, 2.0))
    assert c["major"] == pytest.approx(8.0)
    assert c["phi"] == pytest.approx(0.0)
    sa, sb = c["major"] / 2, c["major"] * c["ratio"] / 2
    assert (2.0 / sb) ** 2 + (2.0 / sa) ** 2 == pytest.approx(1.0)
    assert 0.0 < c["ratio"] < 1.0

    # An elliptical component is placed with its shape meant, so the
    # ratio and orientation are fitted too.
    assert p._placed[-1]["freepar"] & SHAPE_BITS == SHAPE_BITS

    # A second click along the major axis says nothing about the minor
    # one, so it stays circular rather than collapsing to a line.
    d = place((0.0, 0.0), (0.0, 4.0), (0.0, 2.0))
    assert d["ratio"] == pytest.approx(1.0)
    p.close()


def test_mapplot_modelfit_and_report(obs):
    p = MapPlot(obs, quiet=True)
    p.start_gaussian()
    p._gauss_click(*obs.peak_offset()[0])
    p._add_gaussian(type="delta")
    res = p.run_modelfit()
    assert res is not None and res["ncomp"] == 1
    assert res["converged"] in (True, False)

    info = obs.mapinfo()
    assert info["source"] == obs.source
    assert info["mapsize"] == (NX, NX)
    assert info["beam"]["bmaj"] > 0
    assert info["model"]["ncomp"] == 1
    assert np.isfinite(info["residual"]["rms"])
    p.close()


def _press(plot, text):
    """Send `text` to the plot as a key press."""
    from pyqtgraph.Qt import QtCore, QtGui

    plot.keyPressEvent(
        QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, 0,
                        QtCore.Qt.KeyboardModifier.NoModifier, text)
    )


def test_mapplot_m_waits_for_the_clicks_and_f_fits(obs):
    """m starts the interactive placement rather than dropping a
    component; f fits what has been placed, and refuses to invent one."""
    p = MapPlot(obs, quiet=True)
    _press(p, "f")
    assert obs.model == []  # nothing placed: no model is seeded

    _press(p, "m")
    assert p._gauss_stage == "center"
    assert obs.model == []  # still waiting for the centre click
    (x, y), _ = obs.peak_offset()
    p._gauss_click(x, y)
    p._gauss_click(x, y + 1.0)
    assert p._gauss_stage == "second"
    p._gauss_click(x + 0.5, y)
    assert obs.model == []  # placed, not yet in the model
    assert len(p._placed) == 1 and p._placed[0]["type"] == "gauss"

    _press(p, "f")
    assert p._placed == []
    assert len(obs.model) == 1  # fitted: in the model, nothing to keep
    p.close()


def test_mapplot_placed_component_is_drawn_not_imaged(obs):
    """A component that has only been placed carries a guessed flux, so
    it must not be subtracted from the map - not even by a re-invert -
    until modelfit has fitted it."""
    p = MapPlot(obs, quiet=True)
    before = np.array(p.img.image, copy=True)
    (x, y), peak = obs.peak_offset()

    p.start_gaussian()
    p._gauss_click(x, y)
    p._add_gaussian(type="delta")
    assert obs.model == []
    assert p._placed[0]["flux"] == pytest.approx(peak, rel=0.05)
    p.refresh()
    assert np.array_equal(np.asarray(p.img.image), before)

    _press(p, "i")  # re-invert: still the same map
    assert float(obs.valid(np.asarray(p.img.image)).max()) == pytest.approx(
        peak, rel=1e-3
    )

    # Fitting gives it a real flux, and the source leaves the residuals.
    p.run_modelfit()
    assert float(obs.valid().max()) < 0.05 * peak
    p.close()


def test_mapplot_close_adds_placed_components_to_the_model(obs):
    """A placed component that was never fitted is not thrown away when
    the window closes: it joins the model, ready for obs.modelfit()."""
    p = MapPlot(obs, quiet=True)
    (x, y), peak = obs.peak_offset()
    p.start_gaussian()
    p._gauss_click(x, y)
    p._add_gaussian(type="delta")
    assert obs.model == []
    p.close()
    assert len(obs.model) == 1
    assert obs.model[0]["flux"] == pytest.approx(peak, rel=0.05)
    assert obs.model[0]["freepar"] == 3  # flux | pos, as placed


def test_mapplot_failed_fit_leaves_the_model_and_placement_alone(obs, monkeypatch):
    p = MapPlot(obs, quiet=True)
    (x, y), _ = obs.peak_offset()
    p.start_gaussian()
    p._gauss_click(x, y)
    p._add_gaussian(type="delta")

    def boom(*a, **k):
        raise RuntimeError("no convergence")
    monkeypatch.setattr(obs, "modelfit", boom)
    assert p.run_modelfit() is None
    assert obs.model == [] and len(p._placed) == 1
    p._placed = []  # nothing to add on close
    p.close()


def test_mapplot_names_the_image_and_shows_the_shortcuts(obs):
    """A heading over the image says which image it is, and a footnote
    under it lists the main keys."""
    p = MapPlot(obs, quiet=True)
    assert "Residual map" in p.plot.titleLabel.text
    assert "peak" in p.plot.titleLabel.text
    for what, name in (("beam", "Dirty beam"), ("clean", "Restored CLEAN map"),
                       ("model", "Model")):
        p.what = what
        p.refresh()
        assert name in p.plot.titleLabel.text
    for key in ("m: add component", "f: modelfit", "c: clean", "q: close"):
        assert key in p.footer.text()
    p.close()


def test_mapplot_colorbar_handles_track_the_data_range(obs):
    """The colour-bar handles set the displayed range in steps scaled to
    the image. ColorBarItem rounds to whole units by default, which on a
    map in Jy/beam snapped every level to (0, 1) on the first drag and
    then refused to move."""
    obs.invert()
    p = MapPlot(obs, quiet=True)
    cb = p._cbar
    lo0, hi0 = (float(v) for v in cb.levels())
    assert cb.rounding == pytest.approx((hi0 - lo0) / 1000.0)

    # Drag the top handle down, as the mouse does, then release.
    cb.region.setRegion((63, 150))
    lo1, hi1 = (float(v) for v in cb.levels())
    assert hi1 < hi0 and lo1 == pytest.approx(lo0, abs=1e-3 * (hi0 - lo0))
    assert p.img.getLevels() == pytest.approx([lo1, hi1])
    cb._regionChanged()
    # The step follows the new, narrower range, so the next drag still
    # has somewhere to go.
    assert cb.rounding == pytest.approx((hi1 - lo1) / 1000.0)
    cb.region.setRegion((63, 150))
    assert float(cb.levels()[1]) < hi1
    p.close()


def test_mapplot_log_colour_scale(obs):
    """l redistributes the colours logarithmically; the levels, and so
    the colour-bar axis, stay in Jy/beam."""
    from difmapy.plots.mapplot import log_stretch

    obs.invert()
    p = MapPlot(obs, quiet=True)
    assert p.scale == "linear"
    levels = p._cbar.levels()

    _press(p, "l")
    assert p.scale == "log"
    assert p._cbar.levels() == levels          # only the colours changed
    assert "log colours" in p.plot.titleLabel.text
    # Colours are pushed towards the faint end: the map's midpoint now
    # takes a colour the linear map kept for much higher values.
    cmap = p._cbar.colorMap()
    assert cmap.pos[len(cmap.pos) // 2] < 0.1
    assert cmap.pos[0] == 0.0 and cmap.pos[-1] == 1.0

    _press(p, "l")
    assert p.scale == "linear"
    assert p._cbar.colorMap() is p._base_cmap
    p.close()


def test_mapplot_opens_on_a_log_scale_when_asked(obs):
    p = MapPlot(obs, quiet=True, scale="log")
    assert p.scale == "log"
    assert "log colours" in p.plot.titleLabel.text
    p.close()


def test_vplot_panels_share_one_x_axis(obs):
    """Amplitude and phase are read together against one time axis: the
    panels are linked and only the bottom one keeps its tick labels."""
    v = VPlot(obs, nplot=2)
    assert len(v._panels) == 4  # two baselines x (amp, phase)
    assert not any(panel.plot.getAxis("bottom").style["showValues"]
                   for panel in v._panels[:-1])
    assert v._panels[-1].plot.getAxis("bottom").style["showValues"]
    # Panning or zooming one panel carries the others with it. The
    # ranges are not identical to the pixel: pyqtgraph compensates for
    # the panels' different widths so that the *data* lines up.
    full = list(v._panels[1].vb.viewRange()[0])
    v._panels[0].vb.setXRange(1.0, 2.0, padding=0)
    for panel in v._panels[1:]:
        assert panel.vb.linkedView(panel.vb.XAxis) is v._panels[0].vb
        lo, hi = panel.vb.viewRange()[0]
        assert [lo, hi] != full      # it followed the zoom
        assert lo < 1.5 < hi         # onto the same stretch of time
    v.close()


def test_mapplot_overrides_the_imaging_setup(uvfits_file):
    o = difmapy.load(uvfits_file)
    p = MapPlot(o, mapsize=128, cellsize=0.4, uvweight=-1.0, quiet=True)
    assert (o._nx, o._ny) == (128, 128)
    assert o._xinc / difmapy.MAS == pytest.approx(0.4)
    assert o.robust == -1.0
    p.close()


def test_mapplot_auto_sizes_when_mapsize_was_never_set(uvfits_file):
    """With no mapsize() call, mapplot picks 4096 pixels of a tenth of
    the estimated resolution."""
    o = difmapy.load(uvfits_file)
    assert not o._mapsize_set
    p = MapPlot(o, quiet=True)
    assert o._nx == difmapy.observation.DEFAULT_NPIX
    assert o._xinc / difmapy.MAS == pytest.approx(o.estimated_resolution() / 10.0)
    p.close()


def test_plots_open_large_but_within_the_screen(obs):
    """These are read in detail, so they open big - clipped to whatever
    the screen can show (offscreen, that is the virtual screen)."""
    from difmapy.plots.base import fit_to_screen

    for cls, kw in ((MapPlot, dict(quiet=True)), (VPlot, dict(nplot=2))):
        p = cls(obs, **kw)
        assert (p.width(), p.height()) == fit_to_screen(*cls.DEFAULT_SIZE)
        p.close()
    assert MapPlot.DEFAULT_SIZE[0] >= 1200   # wider than the old default
    assert VPlot.DEFAULT_SIZE[1] >= 900


def test_z_and_u_restore_the_axis_ranges(obs):
    """difmap's Z and U: back to the default y and x ranges."""
    v = VPlot(obs, nplot=1)
    amp, phase = v._panels[0], v._panels[1]

    # Zooming in fixes both axes of both panels.
    amp.vb.setRange(xRange=(0.4, 0.5), yRange=(1.0, 1.1), padding=0)
    phase.vb.setRange(xRange=(0.4, 0.5), yRange=(-10, 10), padding=0)
    assert amp.vb.autoRangeEnabled() == [False, False]

    _press(v, "z")                                   # y only
    assert not amp.vb.autoRangeEnabled()[0]          # x left alone
    # The synthetic point source is flat to float32 rounding, so its
    # default is the floored range rather than autoscale (which would
    # zoom onto the noise).
    flat = v.default_y_range(amp.vb)
    assert flat is not None
    assert list(amp.vb.viewRange()[1]) == pytest.approx(list(flat))
    # A phase panel comes back to +-180, its default view, rather than
    # to the spread of whatever is displayed.
    assert list(phase.vb.viewRange()[1]) == pytest.approx([-180, 180], abs=1e-6)

    _press(v, "u")                                   # then x
    assert amp.vb.autoRangeEnabled()[0]
    v.close()


def test_r_reloads_the_plot_from_the_data(obs):
    """Data edited from the prompt while a window is open is picked up
    by r, and nothing else."""
    v = VPlot(obs, nplot=2)
    good = int((v._data["wt"] > 0).sum())
    obs.flag(station="AN0")                          # behind the plot's back
    assert int((v._data["wt"] > 0).sum()) == good    # not noticed yet
    _press(v, "r")
    assert int((v._data["wt"] > 0).sum()) < good
    v.close()

    p = MapPlot(obs, quiet=True)
    before = np.array(p.img.image, copy=True)
    obs.unflag(station="AN0")
    obs.invert()
    _press(p, "r")
    assert not np.array_equal(np.asarray(p.img.image), before)
    p.close()


def test_undo_moved_to_ctrl_z_and_unflag_to_shift_f(obs):
    """z, u and r are the axis/reload keys now, so the flag editing
    keys they used to occupy have moved."""
    from pyqtgraph.Qt import QtCore, QtGui

    def combo(text, key, ctrl=False, shift=False):
        mods = QtCore.Qt.KeyboardModifier.NoModifier
        if ctrl:
            mods |= QtCore.Qt.KeyboardModifier.ControlModifier
        if shift:
            mods |= QtCore.Qt.KeyboardModifier.ShiftModifier
        return QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, mods, text)

    v = VPlot(obs, nplot=1)
    panel = v._panels[0]
    idx = np.nonzero((v._data["group"] == 0) & (v._data["wt"] > 0))[0][:3]
    v._edit_points(idx, True)
    assert len(v.history) == 1
    flagged = int((v._data["wt"] < 0).sum())
    assert flagged > 0

    v.keyPressEvent(combo("\x1a", QtCore.Qt.Key.Key_Z, ctrl=True))     # undo
    assert int((v._data["wt"] < 0).sum()) == 0
    v.keyPressEvent(combo("\x1a", QtCore.Qt.Key.Key_Z, ctrl=True, shift=True))
    assert int((v._data["wt"] < 0).sum()) == flagged                    # redo
    v.keyPressEvent(combo("\x19", QtCore.Qt.Key.Key_Y, ctrl=True))     # redo
    assert int((v._data["wt"] < 0).sum()) == flagged

    # F unflags the nearest point; f flags it again.
    v._mouse_pos = panel.vb.mapViewToScene(
        QtCore.QPointF(float(v._data["x"][idx[0]]), float(v._data["amp"][idx[0]]))
    )
    v._active = panel
    v.keyPressEvent(combo("F", QtCore.Qt.Key.Key_F, shift=True))
    assert int((v._data["wt"] < 0).sum()) == flagged - 1
    v.close()


def test_windows_are_destroyed_before_the_interpreter_exits(obs):
    """The exit crash: Python collecting a live plot window during
    shutdown makes Qt tear down its scene after the Python halves of
    the items in it are gone, and it aborts (SIGTRAP). Windows are
    therefore kept alive deliberately and destroyed on the way out."""
    from difmapy.plots import base

    p = MapPlot(obs, quiet=True)
    v = VPlot(obs, nplot=1)
    # Tracked, so that nothing relies on the user keeping a reference,
    # and the exit handler is in place.
    assert p in base.open_windows() and v in base.open_windows()
    assert base._ATEXIT_DONE

    # Closing asks Qt to destroy the window, but it is still tracked
    # until that has actually happened: a closed window is still a live
    # QMainWindow, and would crash the exit just the same.
    p.close()
    assert base._alive(p)

    base.close_all_windows()
    assert base.open_windows() == []
    assert not base._alive(p) and not base._alive(v)

    # And a new window afterwards works, with the dead ones forgotten.
    q = MapPlot(obs, quiet=True)
    assert base.open_windows() == [q]
    base.close_all_windows()


@pytest.mark.parametrize("run", range(4))
def test_a_session_that_opened_a_plot_exits_cleanly(uvfits_file, run):
    """End to end: the reported crash was intermittent (about half of
    the runs), so this goes through the real command a few times."""
    import subprocess
    import sys

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_QPA_PLATFORMTHEME="")
    proc = subprocess.run(
        [sys.executable, "-m", "difmapy.cli", uvfits_file],
        input="mapsize(256, 0.25)\ninvert()\nmapplot(block=False)\nexit\n",
        capture_output=True, text=True, env=env, timeout=180,
    )
    assert proc.returncode == 0, (
        f"session exited {proc.returncode} "
        f"({'SIGTRAP' if proc.returncode == 133 else 'crash'})\n"
        + proc.stdout[-2000:] + proc.stderr[-2000:]
    )


def test_help_overlay_lists_the_keys(obs):
    p = MapPlot(obs, quiet=True)
    assert p._help is None
    assert p.toggle_help() is True
    assert p._help is not None
    text = p.help_text()
    for key in ("m", "f", "c", "h", "q", "z / u", "r"):
        assert f"<b>{key}</b>" in text
    assert p.toggle_help() is False
    assert p._help is None
    p.close()


# ---- time axes with the gaps between scans cut out ---------------------

def test_time_gaps_cut_only_the_long_gaps():
    from difmapy.plots.base import TimeGaps

    # Evenly sampled, or with a gap under 10% of the range: nothing cut,
    # and plot coordinates are the times themselves.
    g = TimeGaps(np.arange(0.0, 10.0, 0.1))
    assert not g and g.breaks == []
    assert np.allclose(g.compress([1.5, 3.0]), [1.5, 3.0])
    assert not TimeGaps(np.r_[np.linspace(0, 5, 50), np.linspace(5.5, 10, 50)])

    # Three 1-hour scans 5 and 10 hours apart.
    t = np.r_[np.linspace(0, 1, 61), np.linspace(6, 7, 61), np.linspace(17, 18, 61)]
    g = TimeGaps(t)
    assert g.segments == [(0.0, 1.0), (6.0, 7.0), (17.0, 18.0)]
    x = g.compress(t)
    assert np.allclose(g.expand(x), t)            # exact inverse
    assert x.min() == 0.0                         # first scan unshifted
    # Each scan keeps its width; each cut shrinks to 2% of the 3 h kept.
    assert x.max() == pytest.approx(3.0 + 2 * 0.06)
    assert len(g.breaks) == 2 and not g.in_break(x).any()
    lo, hi = g.breaks[0]
    assert g.in_break((lo + hi) / 2)


@pytest.fixture()
def scanned_obs(scanned_uvfits_file):
    o = difmapy.load(scanned_uvfits_file[0])
    o.select("I")
    o.mapsize(NX, CELL)
    return o


def test_vplot_cuts_the_gaps_between_scans(scanned_obs):
    """Scans far apart share one axis with the dead time cut out; the
    axis still reads real times, and flagging still hits the points
    under the box."""
    from difmapy.plots.base import TimeGapAxis

    v = VPlot(scanned_obs, nplot=1)
    assert len(v.gaps.segments) == 4 and len(v.gaps.breaks) == 3
    d = v._data
    hours = d["time"] / 3600.0
    # The drawn axis is narrower than the observation by the gaps cut.
    assert np.ptp(d["x"]) < 0.9 * np.ptp(hours)
    assert np.allclose(v.gaps.expand(d["x"]), hours)

    bottom = v._panels[-1].plot.getAxis("bottom")
    assert isinstance(bottom, TimeGapAxis)
    # A tick in the last scan is labelled with its real time.
    x_last = float(d["x"].max())
    label = bottom.tickStrings([x_last], 1.0, 0.1)[0]
    assert float(label) == pytest.approx(hours.max(), abs=0.1)
    # ... and one inside a cut is not labelled at all.
    lo, hi = v.gaps.breaks[0]
    assert bottom.tickStrings([(lo + hi) / 2], 1.0, 0.1) == [""]

    # Flag everything in the last scan by a box in plot coordinates.
    a, b = v.gaps.segments[-1]
    x0, x1 = v.gaps.compress([a, b])
    panel = v._panels[0]
    n = v._apply_box(panel, x0 - 1e-9, x1 + 1e-9, -1e9, 1e9, flag=True)
    in_last = (hours >= a) & (hours <= b)
    assert n > 0  # counts FLAG cells (channels x polarizations), not points
    assert (v._data["wt"][in_last] < 0).all()
    assert (v._data["wt"][~in_last] > 0).all()
    v.close()


def test_diagnostic_time_plots_share_the_cut_axis(scanned_obs):
    from difmapy.plots.base import TimeGapAxis
    from difmapy.plots.diagnostics import CorPlot, CpPlot, TPlot

    scanned_obs.addcmp(2.5, 0.0, 0.0)
    scanned_obs.selfcal(phase=True, quiet=True)
    t = TPlot(scanned_obs)
    c = CorPlot(scanned_obs, nplot=1)
    k = CpPlot(scanned_obs, nplot=1)
    for plot in (t.plot, *(p for p, _ in c._plots), *(p for p, _ in k._plots)):
        assert isinstance(plot.getAxis("bottom"), TimeGapAxis)
        assert len(plot.getAxis("bottom").gaps.breaks) == 3
    # The samples of tplot sit on the same compressed coordinates.
    xs = np.concatenate([i.getData()[0] for i in t.plot.items
                         if type(i).__name__ == "FastScatter"])
    assert np.isclose(xs.max(), TimeGapAxis(t.plot.getAxis("bottom").gaps)
                      .gaps.compress(np.asarray(scanned_obs._core.times()) / 3600).max())
    for w in (t, c, k):
        w.close()


def test_projplot_angle_keys_rotate_the_projection(obs):
    """< and > turn a projplot's projection angle, recompute the
    projected distances and rescale the x axis to them."""
    p = RadPlot(obs, projection_deg=0.0)
    u, v = p._data["u"].copy(), p._data["v"].copy()

    def projected(deg):
        phi = np.deg2rad(deg)
        return np.abs(u * np.sin(phi) + v * np.cos(phi)) / 1e6

    assert np.allclose(p._data["x"], projected(0.0))
    _press(p, ">")
    assert p.projection == pytest.approx(10.0)
    assert np.allclose(p._data["x"], projected(10.0))
    assert "PA 10°" in p._panels[-1].plot.getAxis("bottom").labelText
    _press(p, "<")
    _press(p, "<")
    assert p.projection == pytest.approx(-10.0)
    assert np.allclose(p._data["x"], projected(-10.0))

    # The x axis follows the new distances.
    good = p._data["wt"] > 0
    x0, x1 = p._panels[0].vb.viewRange()[0]
    assert x0 <= p._data["x"][good].min() and x1 >= p._data["x"][good].max()

    # Half a turn is a full turn of the projection: the angle wraps.
    p.projection = 85.0
    _press(p, ">")
    assert p.projection == pytest.approx(-85.0)
    assert np.allclose(p._data["x"], projected(95.0))
    assert any(k == "< / >" for k, _ in p.key_help())
    p.close()

    # A custom step, from the entry point.
    q = difmapy.plots.projplot(obs, angle_deg=30.0, step=2.5, block=False)
    _press(q, "<")
    assert q.projection == pytest.approx(27.5)
    q.close()

    # radplot has no angle to turn.
    r = RadPlot(obs)
    x = r._data["x"].copy()
    _press(r, ">")
    assert r.projection is None and np.array_equal(r._data["x"], x)
    assert not any(k == "< / >" for k, _ in r.key_help())
    r.close()


def test_vplot_first_argument_is_baselines_per_page(obs):
    """vplot(3) is three baselines to a page, as difmap's vplot takes it.
    It used to be read as reftel=3, which pinned the plot to one
    station's baselines on a single page."""
    v = VPlot(obs, 3)
    assert v.nplot == 3 and v.reftel is None
    assert len(v._baselines) == 10 and v.npages == 4
    assert v.set_page(5) == 1                       # wraps around
    v.close()
    w = VPlot(obs, 2, "AN1")
    assert w.nplot == 2 and w.reftel == obs.antennas.index("AN1")
    assert w.npages == 2
    w.close()
    x = VPlot(obs, "AN1")                           # a station name first
    assert x.reftel == 1 and x.nplot == 3
    x.close()
    y = VPlot(obs, 0)                               # difmap's 0
    assert y.nplot == len(obs.antennas) - 1
    y.close()


def test_fast_markers_bound_and_draw_every_point(obs):
    from difmapy.plots.base import FastScatter

    s = FastScatter([1.0, np.nan, 3.0, 2.0], [5.0, 1.0, np.inf, -1.0], size=4)
    x, y = s.getData()
    assert list(x) == [1.0, 2.0] and list(y) == [5.0, -1.0]  # non-finite dropped
    assert s.dataBounds(0) == (1.0, 2.0) and s.dataBounds(1) == (-1.0, 5.0)
    assert s.dataBounds(1, orthoRange=(1.5, 3.0)) == (-1.0, -1.0)
    assert FastScatter([], []).dataBounds(0) == (None, None)

    p = RadPlot(obs, quantity="amp")
    drawn = [it for it, _ in p._items if isinstance(it, FastScatter)]
    assert drawn
    assert sum(len(it.getData()[0]) for it in drawn) >= int((p._data["wt"] > 0).sum())
    assert not p.grab().isNull()
    p.close()


def test_cpplot_computes_only_the_page_it_shows(obs):
    """cpplot lists the triangles up front but computes closure phases a
    page at a time; computing all 3276 triangles of a 28-station array
    took seconds before the window opened."""
    from difmapy.plots.diagnostics import CpPlot

    p = CpPlot(obs, nplot=4)
    assert len(p._items) == 10                      # C(5, 3)
    assert all(t == tuple(sorted(t)) for t in p._items)
    assert p.npages == 3 and len(p._plots) == 4
    p.set_page(2)
    assert len(p._plots) == 2
    q = CpPlot(obs, triangles=[("AN2", "AN0", "AN1")])
    assert q._items == [(0, 1, 2)] and len(q._plots) == 1
    p.close()
    q.close()
