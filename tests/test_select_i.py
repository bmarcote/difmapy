"""Stokes I now uses the permissive parallel-hand combination that
difmap spelled "pi": data survive when only one hand is usable, on the
assumption that the other is identical.
"""

import os

import numpy as np
import pytest

import difmapy
from difmapy._core import CoreObservation

MAS = difmapy.MAS
HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")


def make_obs(pols, flags=None, flux=2.0, values=None):
    """A one-baseline, one-channel observation with the given pol codes.

    `values`, if given, sets a distinct visibility per polarization
    (needed when testing V or Q, which vanish if the hands are equal -
    and an identically zero visibility counts as deleted, as in difmap).
    """
    npol = len(pols)
    nrow = 8
    vis = np.full((nrow, 1, npol), flux + 0j, dtype=np.complex64)
    if values is not None:
        for p, val in enumerate(values):
            vis[:, 0, p] = val
    wt = np.ones((nrow, 1, npol), dtype=np.float32)
    core = CoreObservation(
        "T", 0.0, 0.5, 2000.0, ["A", "B"], np.zeros((2, 3)), [0, 0],
        [5.0e9], [1.0e6], [1], list(pols),
        np.arange(nrow, dtype=np.float64) * 10.0,
        np.full(nrow, 10.0, np.float32),
        np.zeros(nrow, np.uint32), np.ones(nrow, np.uint32),
        np.tile([1e-4, 2e-4, 0.0], (nrow, 1)),
        vis, wt, 60000.0,
        flag=None if flags is None else np.ascontiguousarray(flags),
    )
    return difmapy.Observation(core)


def test_i_and_pi_are_identical():
    """Both spellings must give the same stream, and both report I."""
    o = make_obs([-1, -2])  # RR, LL
    o.select("I")
    vi, wi = (np.asarray(x) for x in o._core.stream_vis())
    assert o._core.selection()["stokes"] == "I"
    o.select("PI")
    vp, wp = (np.asarray(x) for x in o._core.stream_vis())
    assert o._core.selection()["stokes"] == "I"  # legacy spelling reports I
    assert np.array_equal(vi, vp) and np.array_equal(wi, wp)


def test_i_keeps_data_with_only_one_hand_recorded():
    """A single-polarization dataset must still yield Stokes I."""
    o = make_obs([-1], flux=3.0)  # RR only
    o.select("I")
    v, w = (np.asarray(x) for x in o._core.stream_vis())
    assert (w > 0).all()
    # RR alone is taken as I (the other hand assumed identical).
    assert np.allclose(v.real, 3.0)
    assert o._core.selection()["stokes"] == "I"
    # Likewise for a single linear hand.
    o2 = make_obs([-5], flux=1.5)  # XX only
    o2.select("I")
    v2, w2 = (np.asarray(x) for x in o2._core.stream_vis())
    assert (w2 > 0).all() and np.allclose(v2.real, 1.5)


def test_i_keeps_data_when_one_hand_is_flagged():
    """Per visibility: if one hand is flagged, use the other, rather
    than discarding the sample (which is what strict I did)."""
    pols = [-1, -2]
    flags = np.zeros((8, 1, 2), dtype=bool)
    flags[:4, 0, 1] = True  # LL flagged on the first four rows
    o = make_obs(pols, flags=flags, flux=2.0)
    o.select("I")
    v, w = (np.asarray(x) for x in o._core.stream_vis())
    assert (w > 0).all(), "samples with one good hand must survive"
    assert np.allclose(v.real, 2.0)  # scale preserved

    # Both hands flagged -> the sample stays flagged.
    flags[4:, 0, :] = True
    o = make_obs(pols, flags=flags, flux=2.0)
    o.select("I")
    _, w = (np.asarray(x) for x in o._core.stream_vis())
    assert (w[:4] > 0).all() and (w[4:] < 0).all()


def test_i_scale_matches_strict_average_for_equal_weights():
    """With equal weights the combination is exactly (RR+LL)/2, with the
    summed weight - so the flux scale is unchanged."""
    npol, nrow = 2, 4
    vis = np.zeros((nrow, 1, npol), dtype=np.complex64)
    vis[:, 0, 0] = 3.0 + 1.0j  # RR
    vis[:, 0, 1] = 1.0 - 1.0j  # LL
    wt = np.full((nrow, 1, npol), 4.0, dtype=np.float32)
    core = CoreObservation(
        "T", 0.0, 0.5, 2000.0, ["A", "B"], np.zeros((2, 3)), [0, 0],
        [5.0e9], [1.0e6], [1], [-1, -2],
        np.arange(nrow, dtype=np.float64), np.ones(nrow, np.float32),
        np.zeros(nrow, np.uint32), np.ones(nrow, np.uint32),
        np.tile([1e-4, 2e-4, 0.0], (nrow, 1)), vis, wt, 60000.0,
    )
    o = difmapy.Observation(core)
    o.select("I")
    v, w = (np.asarray(x) for x in o._core.stream_vis())
    assert np.allclose(v, 2.0 + 0.0j)  # ((3+i) + (1-i))/2
    assert np.allclose(w, 8.0)  # 4 + 4, as the strict form also gives


def test_i_still_needs_a_parallel_hand():
    o = make_obs([-3, -4])  # RL, LR only
    with pytest.raises(ValueError, match="unavailable"):
        o.select("I")
    # ...but the cross-hands themselves are selectable, and Q from them.
    o.select("RL")
    o.select("Q")


def test_v_and_q_remain_strict():
    """Only I is permissive: V needs both hands, and a flagged hand must
    still flag V (you cannot guess a missing hand for polarization)."""
    flags = np.zeros((8, 1, 2), dtype=bool)
    flags[:4, 0, 1] = True
    # RR != LL so that V is non-zero and its flag state is meaningful.
    o = make_obs([-1, -2], flags=flags, values=[3.0, 1.0])
    o.select("V")
    _, w = (np.asarray(x) for x in o._core.stream_vis())
    assert (w[:4] < 0).all(), "V must be flagged where a hand is missing"
    assert (w[4:] > 0).all()
    o1 = make_obs([-1])  # RR only
    with pytest.raises(ValueError, match="unavailable"):
        o1.select("V")


@pytest.mark.skipif(not os.path.exists(UVF), reason="real data not present")
def test_real_data_i_matches_pi():
    """On real dual-polarization data the two spellings agree."""
    o = difmapy.load(UVF)
    o.select("I")
    vi, wi = (np.asarray(x) for x in o._core.stream_vis())
    o.select("PI")
    vp, wp = (np.asarray(x) for x in o._core.stream_vis())
    assert np.array_equal(wi > 0, wp > 0)
    assert np.abs(vi - vp).max() < 1e-5 * np.median(np.abs(vi[wi > 0]))


@pytest.mark.skipif(not os.path.exists(UVF), reason="real data not present")
def test_output_image_declares_stokes_i(tmp_path):
    from astropy.io import fits

    for pol in ("I", "PI"):
        o = difmapy.load(UVF)
        o.select(pol)
        o.mapsize(256, 2.0)
        o.invert()
        path = str(tmp_path / f"{pol}.fits")
        o.wmap(path)
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            assert hdr["CTYPE4"] == "STOKES"
            assert hdr["CRVAL4"] == 1.0  # AIPS code for Stokes I
            assert hdr["CTYPE3"] == "FREQ"
            assert hdul[0].data.shape == (1, 1, 256, 256)
