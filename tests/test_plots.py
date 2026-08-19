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

    p = RadPlot(obs)
    d = p._data
    n_good = int((d["wt"] > 0).sum())
    # Box covering the upper half of the amplitude range.
    ycut = float(np.median(d["y"]))
    xhi, yhi = float(d["x"].max()) * 2, float(d["y"].max()) * 2
    expect = int(((d["y"] >= ycut) & (d["wt"] > 0)).sum())
    assert 0 < expect < n_good

    _drag(p.vb, (-1.0, ycut), (xhi, yhi),
          QtCore.Qt.KeyboardModifier.ShiftModifier)
    assert int((p._data["wt"] < 0).sum()) == expect

    _drag(p.vb, (-1.0, -yhi), (xhi, yhi),
          QtCore.Qt.KeyboardModifier.ControlModifier)
    assert int((p._data["wt"] > 0).sum()) == n_good
    p.close()


def test_plain_drag_pans_not_flags(obs):
    """Without a modifier, dragging must pan (never flag)."""
    from pyqtgraph.Qt import QtCore

    p = RadPlot(obs)
    n_good = int((p._data["wt"] > 0).sum())
    before = p.vb.viewRange()
    _drag(p.vb, (0.0, 0.0), (5.0, 5.0), QtCore.Qt.KeyboardModifier.NoModifier)
    assert int((p._data["wt"] > 0).sum()) == n_good  # nothing flagged
    assert p.vb.viewRange() != before  # the view moved
    p.close()


def test_radplot_flagging(obs):
    p = RadPlot(obs)
    d = p._data
    n0 = int((d["wt"] > 0).sum())
    assert n0 == obs._core.nrow * obs.nif
    # Programmatic box flag: everything beyond the median uv radius.
    cut = float(np.median(d["x"]))
    p._apply_box(cut, 1e9, -1e9, 1e9, flag=True)
    d = p._data
    nf = int((d["wt"] < 0).sum())
    assert nf > 0
    assert (np.hypot(*_uv(obs)) / 1e6 >= cut).sum() == nf
    # Unflag box restores everything.
    p._apply_box(-1e9, 1e9, -1e9, 1e9, flag=False)
    assert int((p._data["wt"] > 0).sum()) == n0
    p.close()


def _uv(obs):
    _, _, _, us, vs, _ = obs._core.rows()
    sel = obs._core.selection()
    freq = np.asarray(sel["if_freq"])
    return (us[:, None] * freq).ravel(), (vs[:, None] * freq).ravel()


def test_uvplot_and_vplot(obs):
    p = UVPlot(obs)
    assert len(p._data["x"]) == 2 * obs._core.nrow * obs.nif  # conjugates
    p.close()
    v = VPlot(obs, reftel="AN1")
    # AN1 participates in 4 of 10 baselines.
    assert len(v._data["x"]) == obs._core.nrow * obs.nif * 4 // 10
    v._flag_nearest(v._data["x"][0], v._data["y"][0], flag=True)
    assert (v._data["wt"] < 0).any()
    v.close()


def test_mapplot_windows_and_clean(obs):
    p = MapPlot(obs)
    assert p.img.image is not None
    # Add a window programmatically like a double-click would.
    p._add_roi(2.0, 6.0, -4.5, -0.5)
    p._rois_to_obs()
    assert len(obs.windows) == 1
    assert obs.windows[0] == (2.0, 6.0, -4.5, -0.5)
    # Clean uses the window; the peak lies inside it.
    res = obs.clean(300, 0.1)
    assert res["cleaned_flux"] > 0
    p.refresh()
    assert len(p._rois) == 1
    # Switch displays.
    p.what = "beam"
    p.refresh()
    p.what = "clean"
    p.refresh()
    p.close()
