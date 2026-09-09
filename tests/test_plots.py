"""Offscreen smoke tests of the pyqtgraph plot layer: construction,
refresh, programmatic flagging and window syncing."""

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""  # the gtk3 theme aborts w/o display

pg = pytest.importorskip("pyqtgraph")

import difmapy
from difmapy.plots.mapplot import MapPlot
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
    obs.keep()
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
    """n, then clicks for the centre and the two axes, adds a Gaussian;
    d short-circuits to a point source or a circular Gaussian."""
    p = MapPlot(obs, quiet=True)
    p.start_gaussian()
    assert p._gauss_stage == "center"
    p._gauss_click(1.0, 2.0)
    assert p._gauss_stage == "major"
    p._gauss_click(1.0, 5.0)  # 3 mas due north of the centre
    assert p._gauss_stage == "minor"
    p._gauss_click(2.5, 2.0)  # 1.5 mas due east: the minor axis
    c = obs.model[-1]
    assert p._gauss_stage is None
    assert c["type"] == "gauss"
    assert (c["x"], c["y"]) == pytest.approx((1.0, 2.0))
    assert c["major"] == pytest.approx(6.0)      # twice the click radius
    assert c["phi"] == pytest.approx(0.0)        # major axis due north
    assert c["ratio"] == pytest.approx(0.5)      # 3 mas minor / 6 mas major
    assert len(p._model_items) >= 1              # the ellipse is drawn

    # "d" at the major-axis stage makes it a point source instead.
    p.start_gaussian()
    p._gauss_click(-2.0, 0.0)
    p._add_gaussian(type="delta")
    assert obs.model[-1]["type"] == "delta"
    assert obs.model[-1]["major"] == 0.0
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


def test_help_overlay_lists_the_keys(obs):
    p = MapPlot(obs, quiet=True)
    assert p._help is None
    assert p.toggle_help() is True
    assert p._help is not None
    text = p.help_text()
    for key in ("n", "M", "c", "h", "q"):
        assert f"<b>{key}</b>" in text
    assert p.toggle_help() is False
    assert p._help is None
    p.close()
