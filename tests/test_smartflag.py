"""Tests of the smart-flag selection (Shift+S, then a drag): :mod:`difmapy.smartflag` and the plot gesture.

The synthetic observation (conftest) is noise-free; antenna 2's gain collapses to a fifth for integrations
20-25. A selection touching that fault on one baseline must flag antenna 2 on every baseline over those
integrations (every IF; at most one integration more on either side, the edge a strong fault grows into) and
nothing else; a selection over clean data flags nothing.
"""
import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""

import difmapy
from conftest import NANT, make_uvfits, scan_times
from difmapy.smartflag import smart_flag_ops, stream_grid

DT = 60.0
BAD = range(20, 26)                                   # integrations where antenna 2 fails


def _gains(t):
    bad = int(round(t / DT)) in BAD
    return [(0.2, 0.0) if (antenna == 2 and bad) else (1.0, 0.0) for antenna in range(NANT)]


@pytest.fixture(scope="module")
def faulty_file(tmp_path_factory):
    return str(make_uvfits(tmp_path_factory.mktemp("smart") / "faulty.uvf", gerr=_gains))


@pytest.fixture()
def obs(faulty_file):
    o = difmapy.load(faulty_file)
    o.select("I")
    return o


def _points(obs, baseline, integrations):
    """(rows, ifs) of every IF of the rows of one baseline at the given integration indices."""
    time, a1, a2, *_ = obs._core.rows()
    times = np.unique(np.asarray(time))
    wanted = np.isin(np.asarray(time), times[list(integrations)])
    rows = np.flatnonzero(wanted & (np.minimum(a1, a2) == baseline[0]) & (np.maximum(a1, a2) == baseline[1]))
    nif = np.asarray(obs._core.stream_vis()[0]).shape[1]
    return np.repeat(rows, nif), np.tile(np.arange(nif), rows.size)


def test_stream_grid_has_one_cell_per_baseline_if_and_integration(obs):
    grid, index = stream_grid(obs)
    nbl = NANT * (NANT - 1) // 2
    assert grid.vis.shape[0] == nbl and grid.vis.shape[2:4] == (1, 1)
    assert grid.vis.shape[-1] == len(scan_times())
    assert np.all(grid.count[grid.count > 0] == pytest.approx(1.0))


def test_a_selection_on_the_fault_flags_the_antenna_on_every_baseline(obs):
    ops, summary = smart_flag_ops(obs, *_points(obs, (0, 2), [22, 23]), use_phases=False)
    time, a1, a2, *_ = obs._core.rows()
    times, mine = np.unique(np.asarray(time)), (np.asarray(a1) == 2) | (np.asarray(a2) == 2)
    fault = set(np.flatnonzero(np.isin(np.asarray(time), times[list(BAD)]) & mine).tolist())
    edges = set(np.flatnonzero(np.isin(np.asarray(time), times[[BAD[0] - 1, BAD[-1] + 1]]) & mine).tolist())
    assert summary["antennas"] == [2] and summary["baselines"] == 0
    assert len(ops) == 2                                                  # both IFs
    for rows, _, flag in ops:
        # The whole fault, and at most the integration either side (the edges a strong fault grows into).
        assert flag and fault <= set(rows.tolist()) <= fault | edges


def test_a_selection_over_clean_data_flags_nothing(obs):
    ops, summary = smart_flag_ops(obs, *_points(obs, (0, 1), range(40, 50)), use_phases=False)
    assert ops == [] and summary["samples"] == 0


def test_shift_s_then_a_drag_smart_flags_and_one_undo_restores(obs):
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtCore, QtGui
    from difmapy.plots.points import VPlot
    from test_plots import _drag

    plot = VPlot(obs, quantity="amp")
    before = np.asarray(obs._core.stream_vis()[1]).copy()
    plot.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, QtCore.Qt.Key.Key_S,
                                       QtCore.Qt.KeyboardModifier.ShiftModifier, "S"))
    assert plot._smart_armed and all(vb.armed for vb in plot.view_boxes())
    d = plot._data
    # The panel of a baseline of antenna 2: its fault is the only amplitude below half the flux.
    panel = next(p for p in plot._panels if (np.asarray(d["amp"])[plot._panel_mask(p, d)] < 1.0).any())
    x = np.asarray(d["x"])[plot._panel_mask(panel, d)]
    _drag(panel.vb, (float(x.min()) - 1.0, -1.0), (float(x.max()) + 1.0, 1.0), QtCore.Qt.KeyboardModifier.NoModifier)
    assert not plot._smart_armed
    after = np.asarray(obs._core.stream_vis()[1])
    newly = (before > 0) & (after < 0)
    assert newly.any(), plot.statusBar().currentMessage()
    assert "antennas" in plot.statusBar().currentMessage()
    plot._undo_redo(redo=False)
    np.testing.assert_array_equal(np.asarray(obs._core.stream_vis()[1]) > 0, before > 0)
