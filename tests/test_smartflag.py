"""Tests of smart flagging on an observation: :mod:`difmapy.smartflag`, `Observation.smartflag`, the plot gesture
(Shift+S, then a drag) and the dashboard's Smart flag button.

The synthetic observation (conftest) is noise-free, so the floors alone set the tolerance (5% of the amplitude).
Antenna 2's gain collapses to a fifth for integrations 20-25. A search of all the data finds the fault and flags
antenna 2 on every baseline over it (every IF; at most one integration more on either side, the edge a strong
fault grows into). A selection is taken as outliers: it is always flagged, and the antenna behind it is flagged
with it. (A mild fault, which only a selection finds, is tested on Gaussian noise in ``test_outliers.py``.)

The source is at the phase centre: off it (the conftest default is 4.7 mas), the phases of the longest baselines
turn so fast between integrations that the amplitude of the component-wise median the model takes as reference
falls short by several times the floor, and on noise-free data that is all the scatter there is.
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
    return str(make_uvfits(tmp_path_factory.mktemp("smart") / "faulty.uvf", gerr=_gains, x0_mas=0.0, y0_mas=0.0))



def _load(path):
    o = difmapy.load(path)
    o.select("I")
    return o


@pytest.fixture()
def obs(faulty_file):
    return _load(faulty_file)


def _points(obs, baseline, integrations):
    """(rows, ifs) of every IF of the rows of one baseline at the given integration indices."""
    time, a1, a2, *_ = obs._core.rows()
    times = np.unique(np.asarray(time))
    wanted = np.isin(np.asarray(time), times[list(integrations)])
    rows = np.flatnonzero(wanted & (np.minimum(a1, a2) == baseline[0]) & (np.maximum(a1, a2) == baseline[1]))
    nif = np.asarray(obs._core.stream_vis()[0]).shape[1]
    return np.repeat(rows, nif), np.tile(np.arange(nif), rows.size)


def _fault_rows(obs):
    """Rows of antenna 2 over the fault, and over the integration either side of it."""
    time, a1, a2, *_ = obs._core.rows()
    times, mine = np.unique(np.asarray(time)), (np.asarray(a1) == 2) | (np.asarray(a2) == 2)
    fault = set(np.flatnonzero(np.isin(np.asarray(time), times[list(BAD)]) & mine).tolist())
    edges = set(np.flatnonzero(np.isin(np.asarray(time), times[[BAD[0] - 1, BAD[-1] + 1]]) & mine).tolist())
    return fault, edges


def _assert_antenna_2_over_the_fault(obs, ops):
    fault, edges = _fault_rows(obs)
    assert len(ops) == 2                                                  # both IFs
    for rows, _, flag in ops:
        # The whole fault, and at most the integration either side (the edges a strong fault grows into).
        assert flag and fault <= set(rows.tolist()) <= fault | edges


def test_stream_grid_has_one_cell_per_baseline_if_and_integration(obs):
    grid, index = stream_grid(obs)
    nbl = NANT * (NANT - 1) // 2
    assert grid.vis.shape[0] == nbl and grid.vis.shape[2:4] == (1, 1)
    assert grid.vis.shape[-1] == len(scan_times())
    assert np.all(grid.count[grid.count > 0] == pytest.approx(1.0))
    assert grid.selected is None


def test_a_search_of_all_the_data_flags_the_faulty_antenna_only(obs):
    ops, summary = smart_flag_ops(obs, use_phases=False)
    assert list(summary["antennas"]) == [2] and summary["alone"] == 0
    _assert_antenna_2_over_the_fault(obs, ops)


def test_a_selection_on_the_fault_flags_the_antenna_on_every_baseline(obs):
    ops, summary = smart_flag_ops(obs, *_points(obs, (0, 2), [22, 23]), use_phases=False)
    assert list(summary["antennas"]) == [2] and summary["alone"] == 0
    _assert_antenna_2_over_the_fault(obs, ops)


def test_a_selection_over_clean_data_flags_exactly_the_selection(obs):
    rows, ifs = _points(obs, (0, 1), range(40, 50))
    ops, summary = smart_flag_ops(obs, rows, ifs, use_phases=False)
    assert summary["antennas"] == {} and summary["alone"] == rows.size == summary["samples"]
    flagged = {(int(r), k) for chosen, k, _ in ops for r in chosen}
    assert flagged == set(zip(rows.tolist(), ifs.tolist()))


def test_the_partner_antenna_of_a_selected_baseline_is_not_blamed(obs):
    """Selecting baseline 2-4 must blame 2, whose other baselines depart too, and not 4."""
    _, summary = smart_flag_ops(obs, *_points(obs, (2, 4), [21, 22, 23, 24]), use_phases=True)
    assert list(summary["antennas"]) == [2]


def test_smartflag_command_flags_and_reports(obs, capsys):
    before = np.asarray(obs._core.stream_vis()[1]) > 0
    report = obs.smartflag(use_phases=False, quiet=False)
    assert "smartflag: flagged" in capsys.readouterr().out
    assert list(report["antennas"]) == ["AN2"] and report["alone"] == 0
    after = np.asarray(obs._core.stream_vis()[1]) > 0
    assert int((before & ~after).sum()) == report["samples"] > 0
    fault, edges = _fault_rows(obs)
    assert fault <= set(np.flatnonzero((before & ~after).any(axis=1)).tolist()) <= fault | edges
    # Nothing is left to find, and quiet returns None.
    assert obs.smartflag(use_phases=False) is None
    assert "no outliers found" in capsys.readouterr().out


def test_smartflag_low_thresholds_take_the_edge_down(obs, monkeypatch):
    """A threshold below the default edge_sigma (1.5) lowers the edge with it - it used to fail OutlierSettings' check - while an edge_sigma given in `settings` is kept, and a non-positive threshold is refused."""
    import difmapy.smartflag as smartflag

    seen = []
    real = smartflag.smart_flag_ops
    monkeypatch.setattr(smartflag, "smart_flag_ops", lambda o, **kw: (seen.append(kw["settings"]), real(o, **kw))[1])
    obs.smartflag(use_phases=False, antenna_sigma=1.0, verbose=False)
    obs.smartflag(use_phases=False, sigma=1.2, verbose=False)
    obs.smartflag(use_phases=False, verbose=False, settings={"edge_sigma": 2.0})
    assert [s["edge_sigma"] for s in seen] == [1.0, 1.2, 2.0]
    with pytest.raises(ValueError, match="must be positive"):
        obs.smartflag(antenna_sigma=0)


def test_smartflag_keeps_ignored_stations_ignored(obs):
    obs.ignore("AN2")
    report = obs.smartflag(use_phases=False, quiet=False, verbose=False)
    assert report["samples"] == 0
    obs.unignore("AN2")


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
    assert "antennas AN2" in plot.statusBar().currentMessage()
    plot._undo_redo(redo=False)
    np.testing.assert_array_equal(np.asarray(obs._core.stream_vis()[1]) > 0, before > 0)


def test_a_smart_box_over_good_points_flags_them(obs):
    """The box is outliers by definition: a box over clean points flags exactly them, on their own baselines."""
    pytest.importorskip("pyqtgraph")
    from pyqtgraph.Qt import QtCore
    from difmapy.plots.points import VPlot
    from test_plots import _drag

    plot = VPlot(obs, quantity="amp")
    before = np.asarray(obs._core.stream_vis()[1]).copy()
    d = plot._data
    panel = next(p for p in plot._panels if not (np.asarray(d["amp"])[plot._panel_mask(p, d)] < 1.0).any())
    mask = plot._panel_mask(panel, d) & (np.asarray(d["wt"]) > 0)
    x = np.asarray(d["x"])[mask]
    x0, x1 = float(np.quantile(x, 0.7)), float(x.max()) + 1.0
    inside = mask & (np.asarray(d["x"]) >= x0)
    plot.arm_smart(True)
    _drag(panel.vb, (x0, -1.0), (x1, 1e3), QtCore.Qt.KeyboardModifier.NoModifier)
    after = np.asarray(obs._core.stream_vis()[1])
    newly = {(int(r), int(k)) for r, k in zip(*np.nonzero((before > 0) & (after < 0)))}
    assert newly == set(zip(np.asarray(d["row"])[inside].tolist(), np.asarray(d["cif"])[inside].tolist()))
    assert "on their own baselines" in plot.statusBar().currentMessage()


def test_dashboard_smart_flag_button(obs):
    """The button runs `smartflag` on all the data; a typo in the sigma field is reported, not raised."""
    pytest.importorskip("pyqtgraph")
    from difmapy.plots.dashboard import Dashboard

    dash = Dashboard(obs)
    try:
        assert dash.smart_phase_check.isChecked()
        dash.smart_phase_check.setChecked(False)
        report = dash.run_smartflag()
        assert list(report["antennas"]) == ["AN2"] and report["alone"] == 0
        assert "smartflag: flagged" in dash.output.toPlainText()
        dash.smart_sigma_edit.setText("five")
        assert dash.run_smartflag() is None                      # reported, not raised
        assert "smartflag" in dash.output.toPlainText()
    finally:
        dash.close()
