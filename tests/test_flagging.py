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
    _, wt = obs._core.stream_vis()
    _, a1, a2, *_ = obs._core.rows()
    mask = ((a1 == 0) & (a2 == 1)) & (t <= tmid)
    assert (wt[mask] < 0).all()
    assert (wt[~mask] > 0).all()


def test_flag_single_if(obs):
    obs.flag(station="AN0", if_index=1)
    _, wt = obs._core.stream_vis()
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


# ----------------------------------------------------------------------
# ignore / unignore
# ----------------------------------------------------------------------


def _rows(o, name):
    """Rows of every baseline that includes this station."""
    _, a1, a2, *_ = o._core.rows()
    i = o.antennas.index(name)
    return np.nonzero((np.asarray(a1) == i) | (np.asarray(a2) == i))[0]


def test_ignore_sets_a_station_aside_and_puts_it_back(obs):
    """The point of ignore over flag: the station's own flag state is
    remembered, so it returns exactly as it was even though everyone
    else's data was edited while it was away."""
    obs.flag(baseline=("AN1", "AN2"), tmax=600.0)   # pre-existing flags
    before = np.array(obs.flags, copy=True)
    an1, an2, an3 = (_rows(obs, n) for n in ("AN1", "AN2", "AN3"))

    assert obs.ignored == []
    assert obs.ignore("an1") == ["AN1"]             # case-insensitive
    assert obs.flags[an1].all()                     # all of it, all channels
    assert _good_fraction(obs) == pytest.approx(0.6, abs=1e-9)

    # Edit other stations while AN1 is away, including an unflag that
    # would otherwise bring it back.
    obs.flag(station="AN3")
    obs.unflag(station="AN2")
    assert obs.ignored == ["AN1"]
    assert obs.flags[an1].all()

    assert obs.unignore() == []
    after = np.array(obs.flags, copy=True)
    assert (after[an1] == before[an1]).all()        # exactly as it was
    # AN3's flagging is kept, except where the AN2 unflag legitimately
    # undid it (the AN2-AN3 baseline).
    assert after[np.setdiff1d(an3, np.union1d(an1, an2))].all()
    rest = np.setdiff1d(np.arange(len(after)),
                        np.union1d(np.union1d(an1, an2), an3))
    assert (after[rest] == before[rest]).all()      # nothing else touched


def test_ignored_data_is_out_of_imaging_and_calibration(obs):
    from conftest import FLUX, X0_MAS, Y0_MAS

    obs.mapsize(256, 0.25)
    obs.ignore("AN0")
    # 4 of the 10 baselines involve AN0.
    assert obs.invert()["nused"] == obs._core.nrow * 6 // 10 * obs.nif
    assert obs.moddif()["nvis"] == obs._core.nrow * 6 // 10 * obs.nif

    obs.addcmp(FLUX, X0_MAS, Y0_MAS)
    obs.selfcal(phase=True, quiet=True)
    nt, nif, nant = obs._core.ntimes, obs.nif, len(obs.antennas)
    phs = np.asarray(obs._core.gains()[1]).reshape(nt, nif, nant)
    assert np.all(phs[:, :, 0] == 0.0)   # no solution for the ignored one
    assert np.any(phs[:, :, 1] != 0.0)


def test_ignoring_two_stations_that_share_a_baseline(obs):
    an0, an4 = _rows(obs, "AN0"), _rows(obs, "AN4")
    before = np.array(obs.flags, copy=True)
    obs.ignore("AN0", "AN4")
    assert obs.ignored == ["AN0", "AN4"]

    # Their shared baseline belongs to both, so releasing one must not
    # release it.
    obs.unignore("AN0")
    assert obs.ignored == ["AN4"]
    assert obs.flags[an4].all()
    assert not obs.flags[np.setdiff1d(an0, an4)].any()

    obs.unignore()
    assert obs.ignored == []
    assert (np.array(obs.flags) == before).all()


def test_ignore_needs_a_name_and_rejects_unknown_ones(obs):
    with pytest.raises(ValueError, match="at least one antenna"):
        obs.ignore()
    with pytest.raises(ValueError, match="unknown antenna"):
        obs.ignore("NOPE")
    assert obs.ignored == []
    # Unignoring something that is not ignored is a no-op, not an error.
    assert obs.unignore("AN0") == []


def test_ignore_survives_interactive_flag_edits(obs):
    """The plots edit flags through EditHistory, which must respect the
    ignore too - and undo must not resurrect the station either."""
    from difmapy.plots.base import EditHistory

    an2 = _rows(obs, "AN2")
    obs.ignore("AN2")
    history = EditHistory(obs)
    # Unflag some of the ignored station's rows the way a ctrl-drag would.
    history.apply([(an2[:5], 0, False)])
    assert obs.flags[an2].all()
    history.undo()
    assert obs.flags[an2].all()
    obs.unignore()
    assert not obs.flags[an2].any()
