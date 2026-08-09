"""Flagging/unflagging tests."""

import numpy as np
import pytest

import difmapy


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    return o


def _good_fraction(o):
    _, wt = o._core.stream_vis()
    return float((wt > 0).mean())


def test_flag_station_roundtrip(obs):
    assert _good_fraction(obs) == 1.0
    n = obs.flag(station="AN2")
    # AN2 appears on 4 of the 10 baselines.
    assert n == obs._core.nrow * 4 // 10
    assert abs(_good_fraction(obs) - 0.6) < 1e-9

    # Flags are reflected in the stream and reversible.
    obs.unflag(station="AN2")
    assert _good_fraction(obs) == 1.0


def test_flag_baseline_time_range(obs):
    t = obs._core.rows()[0]
    tmid = np.median(t)
    n = obs.flag(baseline=("AN0", "AN1"), tmax=tmid)
    assert 0 < n < obs._core.nrow
    vis, wt = obs._core.stream_vis()
    _, a1, a2, *_ = obs._core.rows()
    mask = ((a1 == 0) & (a2 == 1)) & (t <= tmid)
    assert (wt[mask] < 0).all()
    assert (wt[~mask] > 0).all()


def test_flag_single_if(obs):
    obs.flag(station="AN0", if_index=1)
    vis, wt = obs._core.stream_vis()
    _, a1, a2, *_ = obs._core.rows()
    mask = (a1 == 0) | (a2 == 0)
    assert (wt[mask, 1] < 0).all()
    assert (wt[mask, 0] > 0).all()  # IF 1 untouched


def test_flagging_affects_imaging(obs):
    obs.mapsize(256, 0.25)
    r0 = obs.invert()
    obs.flag(station="AN3")
    r1 = obs.invert()
    assert r1["nused"] < r0["nused"]
    # Unflag restores the original gridding count.
    obs.unflag(station="AN3")
    r2 = obs.invert()
    assert r2["nused"] == r0["nused"]
