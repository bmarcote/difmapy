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
    a1 = p.obs._core.rows()[1]
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
