"""Tests of the antenna-based outlier flagger (:mod:`difmapy.outliers`) on synthetic grids.

A grid of 8 antennas, 2 subbands, 4 channel blocks, 2 hands and 30 scans of 10 bins carries a smooth
elevation-like trend, antenna gains and Gaussian noise; faults are injected on top. Clean data must stay
(almost) untouched; every fault must be traced to its antenna (or baseline), time range and channel blocks.
"""
from __future__ import annotations

import numpy as np
import pytest

from difmapy import outliers
from difmapy.outliers import OutlierSettings, VisGrid, find_outliers, flag_runs, sigma_radius

NANT, NSPW, NBLOCK, NCORR, NSCAN, NBIN = 8, 2, 4, 2, 30, 10


def _grid(*, seed: int = 1, snr: float = 20.0, rotate: bool = False) -> dict:
    """Synthetic visibilities: ``vis`` (nbl, nspw, nblock, ncorr, ntime), antenna indices, times, scans."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(NANT) for j in range(i + 1, NANT)]
    ant1, ant2 = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    times = (np.arange(NSCAN)[:, None] * 900.0 + np.arange(NBIN)[None, :] * 15.0).reshape(-1)
    segment = np.repeat(np.arange(NSCAN) + 100, NBIN)
    trend = 1.0 - 0.3 * np.sin(np.pi * times / times.max())
    gain = rng.uniform(0.7, 1.3, NANT)
    shape = (len(pairs), NSPW, NBLOCK, NCORR, NSCAN * NBIN)
    vis = (gain[ant1] * gain[ant2])[:, None, None, None, None] * trend * np.exp(0.2j) * np.ones(shape)
    if rotate:      # raw data: every scan of every series rotates at its own rate from its own phase
        rate = rng.uniform(-0.5, 0.5, (len(pairs), 1, 1, 1, NSCAN)).repeat(NBIN, axis=-1)
        start = rng.uniform(-3, 3, (len(pairs), 1, 1, 1, NSCAN)).repeat(NBIN, axis=-1)
        vis = vis * np.exp(1j * (rate * np.tile(np.arange(NBIN), NSCAN) + start))
    vis = vis + (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / snr
    return {"vis": vis, "ant1": ant1, "ant2": ant2, "times": times, "segment": segment}


def _run(data: dict, **settings) -> tuple[VisGrid, outliers.OutlierResult]:
    grid = VisGrid(data["vis"], np.ones(data["vis"].shape), data["ant1"], data["ant2"], data["times"],
                   data["segment"], NANT, restrict=data.get("restrict"))
    return grid, find_outliers(grid, OutlierSettings(**settings))


def _baselines_of(data: dict, antenna: int) -> np.ndarray:
    return (data["ant1"] == antenna) | (data["ant2"] == antenna)


# ---------------------------------------------------------------------------
# Clean data
# ---------------------------------------------------------------------------
def test_sigma_radius_matches_the_one_dimensional_tail():
    assert sigma_radius(3.0) == pytest.approx(3.4393, abs=1e-3)
    assert sigma_radius(5.0) == pytest.approx(5.3613, abs=1e-3)
    assert sigma_radius(40.0) > 40.0                        # the asymptotic branch stays finite and monotonic


def test_clean_data_with_a_trend_is_left_alone():
    grid, result = _run(_grid())
    assert result.summary(grid)["fraction"] < 1e-4


def test_rotating_raw_phases_are_aligned_before_the_model():
    data = _grid(rotate=True)
    grid, result = _run(data, align_phases=True, use_phases=False)
    assert result.summary(grid)["fraction"] < 1e-3


# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------
def test_a_burst_is_traced_to_its_antenna_blocks_and_time_range():
    data = _grid()
    mine = _baselines_of(data, 3)
    for k, factor in enumerate([1.2, 2, 3, 3, 3, 2, 1.3]):               # bins 52-58, with a rise and a decay
        data["vis"][mine, 0, 1:3, :, 52 + k] *= factor
    grid, result = _run(data)
    flagged = result.antenna_flags[3, 0]                                   # (nblock, ncorr, ntime)
    assert flagged[1:3][..., 52:59].all()                                  # the whole burst, edges included
    assert not flagged[..., :49].any() and not flagged[..., 61:].any()     # nothing far from it
    assert not result.antenna_flags[3, 1].any()                            # nor in the other subband
    others = [a for a in range(NANT) if a != 3]
    assert not result.antenna_flags[others].any()                          # its partners are never blamed


def test_a_whole_scan_dropout_is_found_against_the_trend():
    data = _grid()
    data["vis"][_baselines_of(data, 5), :, :, :, 120:130] *= 0.3           # scan 12, every subband
    grid, result = _run(data)
    assert result.antenna_flags[5][..., 120:130].all()
    assert not result.antenna_flags[5][..., :119].any()


def test_a_spike_no_antenna_explains_is_flagged_on_its_baseline_only():
    data = _grid()
    baseline = int(np.flatnonzero((data["ant1"] == 0) & (data["ant2"] == 1))[0])
    data["vis"][baseline, 1, 2, 1, 77] += 2.0
    grid, result = _run(data)
    assert result.baseline_flags[baseline, 1, 2, 1, 77]
    assert not result.antenna_flags[..., 77].any()


def test_a_phase_jump_counts_only_when_phases_are_tested():
    data = _grid()
    data["vis"][_baselines_of(data, 6), 1, :, 0, 204] *= np.exp(1.5j)
    _, with_phases = _run(data)
    assert with_phases.antenna_flags[6, 1, :, 0, 204].all()
    _, amplitude_only = _run(data, use_phases=False)
    assert not amplitude_only.antenna_flags[6, 1, :, 0, 204].any()


def test_a_selection_only_seeds_flags_inside_it():
    data = _grid()
    data["vis"][_baselines_of(data, 2), 0, :, :, 33] *= 4.0
    data["vis"][_baselines_of(data, 4), 0, :, :, 233] *= 4.0
    restrict = np.zeros(data["vis"].shape, dtype=bool)
    restrict[..., 200:260] = True
    data["restrict"] = restrict
    _, result = _run(data)
    assert result.antenna_flags[4, 0, ..., 233].all()
    assert not result.antenna_flags[2].any()


def test_compiled_and_numpy_medians_agree(monkeypatch):
    pytest.importorskip("numba")
    data = _grid()
    data["vis"][_baselines_of(data, 3), 0, 1:3, :, 52:58] *= 3.0
    grids = {}
    for value in ("1", "0"):
        monkeypatch.setenv(outliers.NUMBA_ENV, value)
        grid, result = _run(data)
        grids[value] = result.cell_flags(grid)
    np.testing.assert_array_equal(grids["1"], grids["0"])


# ---------------------------------------------------------------------------
# Helpers and validation
# ---------------------------------------------------------------------------
def test_flag_runs_never_cross_a_scan():
    mask = np.array([0, 1, 1, 1, 0, 1, 1], dtype=bool)
    segment = np.array([1, 1, 1, 2, 2, 2, 2])
    assert flag_runs(mask, segment) == [(1, 2), (3, 3), (5, 6)]


def test_settings_reject_unknown_keys_and_inconsistent_thresholds():
    with pytest.raises(ValueError, match="unknown setting"):
        OutlierSettings.from_mapping({"sigmas": 3})
    with pytest.raises(ValueError, match="edge_sigma"):
        OutlierSettings(sigma=3.0, edge_sigma=4.0)


def test_grid_rejects_scans_split_on_the_time_axis():
    data = _grid()
    data["segment"] = data["segment"].copy()
    data["segment"][5] = 101                                               # scan 101 appears twice
    with pytest.raises(ValueError, match="contiguous"):
        _run(data)
