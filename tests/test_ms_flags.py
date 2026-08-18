"""FLAG-column semantics and write-back to a Measurement Set.

difmapy stores flags in an explicit FLAG array (MS convention) rather
than in the sign of the weights, so flags can be persisted into the MS
without touching data or weights.
"""

import os
import shutil

import numpy as np
import pytest

casatools = pytest.importorskip("casatools")

import difmapy

HERE = os.path.dirname(__file__)
SRC_MS = os.path.join(HERE, "rsm07_3C345.ms")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(SRC_MS), reason="real 3C345 MS not present"
)


@pytest.fixture()
def ms_copy(tmp_path):
    dst = str(tmp_path / "flags.ms")
    shutil.copytree(SRC_MS, dst)
    return dst


def ms_columns(path):
    tb = casatools.table()
    tb.open(path)
    try:
        return {
            "flag": tb.getcol("FLAG"),
            "flag_row": tb.getcol("FLAG_ROW"),
            "data": tb.getcol("DATA"),
            "weight": tb.getcol("WEIGHT_SPECTRUM"),
            "a1": tb.getcol("ANTENNA1"),
            "a2": tb.getcol("ANTENNA2"),
        }
    finally:
        tb.close()


def test_ms_flags_read_matches_column(ms_copy):
    """The loaded FLAG state equals the MS FLAG column on the
    cross-correlations difmapy reads."""
    cols = ms_columns(ms_copy)
    cross = cols["a1"] != cols["a2"]
    assert cols["flag"][:, :, ~cross].all()  # autos are fully flagged
    o = difmapy.load(ms_copy)
    assert o.flagged_fraction == pytest.approx(
        cols["flag"][:, :, cross].mean(), abs=1e-12
    )


def test_weights_are_non_negative(ms_copy):
    """Flagging must not be encoded in weight signs any more."""
    o = difmapy.load(ms_copy)
    o.select("I")
    _, wt_raw = o._core.calibrated_cube()  # writer view: signed
    flags = np.asarray(o.flags)
    # The raw store keeps weights >= 0; the signed view is derived.
    assert (np.asarray(wt_raw)[~flags] >= 0).all()
    assert (np.asarray(wt_raw)[flags] <= 0).all()


def test_save_flags_roundtrip(ms_copy):
    before = ms_columns(ms_copy)
    o = difmapy.load(ms_copy)
    o.select("I")
    frac0 = o.flagged_fraction

    nrows = o.flag(station="EF")
    assert nrows > 0
    assert o.flagged_fraction > frac0
    written = o.save_flags()
    assert written > 0

    after = ms_columns(ms_copy)
    # Data and weights are untouched; only FLAG/FLAG_ROW changed.
    assert np.array_equal(before["data"], after["data"])
    assert np.array_equal(before["weight"], after["weight"])
    assert not np.array_equal(before["flag"], after["flag"])
    # Autocorrelations were not touched.
    auto = before["a1"] == before["a2"]
    assert np.array_equal(before["flag"][:, :, auto], after["flag"][:, :, auto])
    # FLAG_ROW is consistent with FLAG.
    assert np.array_equal(after["flag_row"], after["flag"].all(axis=(0, 1)))

    # Reloading reproduces the in-memory flags exactly.
    o2 = difmapy.load(ms_copy)
    o2.select("I")
    assert np.array_equal(np.asarray(o.flags), np.asarray(o2.flags))


def test_save_flags_unflag_persists(ms_copy):
    o = difmapy.load(ms_copy)
    o.select("I")
    o.unflag(station="EF")  # clears pre-existing EF flags too
    frac = o.flagged_fraction
    o.save_flags()
    o2 = difmapy.load(ms_copy)
    o2.select("I")
    assert o2.flagged_fraction == pytest.approx(frac, abs=1e-12)
    # Only absent (zero-weight) EF data remains flagged: deleted data
    # can never be unflagged (difmap FLAG_DEL).
    ief = o2.antennas.index("EF")
    _, a1, a2, *_ = o2._core.rows()
    rows = (a1 == ief) | (a2 == ief)
    _, wt = o2._core.calibrated_cube()
    still = np.asarray(o2.flags)[rows]
    assert (np.asarray(wt)[rows][still] == 0).all()


def test_flags_setter_and_imaging(ms_copy):
    o = difmapy.load(ms_copy)
    o.select("I")
    o.mapsize(512, 1.0)
    n0 = o.invert()["nused"]

    flags = np.asarray(o.flags).copy()
    flags[:] = True
    with pytest.raises(RuntimeError):
        o.flags = flags  # everything flagged -> nothing to invert
        o.invert()

    o.flags = np.zeros_like(flags)  # unflag everything
    n1 = o.invert()["nused"]
    assert n1 > n0


def test_save_flags_requires_ms_origin(uvfits_file):
    o = difmapy.load(uvfits_file)
    with pytest.raises(ValueError, match="not loaded from a Measurement Set"):
        o.save_flags()
