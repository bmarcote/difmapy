"""UVFITS writer (wobs) and save/get roundtrip tests."""

import numpy as np


import difmapy
from conftest import FLUX, X0_MAS, Y0_MAS

CELL = 0.25
NX = 256


def test_wobs_roundtrip(uvfits_file, tmp_path):
    o1 = difmapy.load(uvfits_file)
    o1.select("I")
    path = str(tmp_path / "rt.uvf")
    o1.wobs(path)

    o2 = difmapy.load(path)
    assert o2.source == o1.source
    assert o2.nif == o1.nif and o2.nchan == o1.nchan and o2.npol == o1.npol
    assert o2.antennas == o1.antennas
    assert o2._core.ifs == o1._core.ifs
    o2.select("I")

    v1, w1 = o1._core.stream_vis()
    v2, w2 = o2._core.stream_vis()
    assert np.allclose(v1, v2, rtol=1e-5, atol=1e-6)
    assert np.allclose(w1, w2, rtol=1e-4)
    # UVW and times survive.
    r1, r2 = o1._core.rows(), o2._core.rows()
    for a, b in zip(r1, r2):
        assert np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                           rtol=1e-9, atol=1e-9)


def test_wobs_applies_calibration(corrupted_uvfits_file, tmp_path):
    path_in, _ = corrupted_uvfits_file
    o = difmapy.load(path_in)
    o.select("I")
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.selfcal(amp=True, phase=True, float_scale=True)

    out = str(tmp_path / "cal.uvf")
    o.wobs(out)

    # Reloaded file is already calibrated: matches the model with
    # identity gains.
    o2 = difmapy.load(out)
    o2.select("I")
    o2.addcmp(FLUX, X0_MAS, Y0_MAS)
    vis, wt = o2._core.stream_vis()
    model = o2._core.stream_model()
    good = wt > 0
    assert np.abs(vis[good] - model[good]).max() < 0.02 * FLUX


def test_wobs_preserves_flags(uvfits_file, tmp_path):
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.flag(station="AN2")
    out = str(tmp_path / "flagged.uvf")
    o.wobs(out)
    o2 = difmapy.load(out)
    o2.select("I")
    _, w1 = o._core.stream_vis()
    _, w2 = o2._core.stream_vis()
    assert np.array_equal(w1 < 0, w2 < 0)


def test_save_get(uvfits_file, tmp_path):
    o = difmapy.load(uvfits_file)
    o.select("I", channels=[(0, 5)])
    o.mapsize(NX, CELL)
    o.add_window(X0_MAS - 2, X0_MAS + 2, Y0_MAS - 2, Y0_MAS + 2)
    o.clean(300, 0.1)
    prefix = str(tmp_path / "session")
    o.save(prefix)

    o2 = difmapy.Observation.get(prefix)
    assert o2._nx == NX and abs(o2._xinc / difmapy.MAS - CELL) < 1e-12
    sel = o2._core.selection()
    assert sel["stokes"] == "I" and sel["chlist"] == [(0, 5)]
    assert len(o2.windows) == 1
    assert abs(o2.model_flux - o.model_flux) / o.model_flux < 1e-4
    # The restored session images to (nearly) the same residual state;
    # the .mod text format rounds positions to 6 significant digits.
    s1, s2 = o.imstat(), o2.imstat()
    assert abs(s1["rms"] - s2["rms"]) < 1e-3 * FLUX
