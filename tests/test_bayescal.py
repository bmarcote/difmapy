"""Bayesian station amplitude calibration (`bayes_gscale`)."""

import json
import os

import numpy as np
import pytest
from astropy.io import fits

import difmapy
from difmapy.bayescal import ModelSpec
from conftest import NIF

MODELS = ("clean", "point1", "gauss1")


def _load(path):
    o = difmapy.load(path)
    o.mapsize(256, 0.25)
    return o


def _expected(gerr):
    """The corrections gscale finds for these gain errors: the inverse
    amplitudes, normalised to an arithmetic mean of one (difmap's
    norm_cors, with float_scale=False)."""
    c = 1.0 / np.array([a for a, _ in gerr])
    return c / c.mean()


def test_recovers_injected_station_gains(corrupted_uvfits_file):
    path, gerr = corrupted_uvfits_file
    o = _load(path)
    r = o.bayes_gscale(models=MODELS, quiet=True, plot=False)
    # A point source: the one-parameter-per-axis model wins.
    assert r.best_model in ("point1", "gauss1")
    assert r.model_prob[r.models.index("clean")] < 1e-6
    # The plain gscale finds the truth...
    np.testing.assert_allclose(np.exp(np.median(r.naive, axis=0)),
                               _expected(gerr), rtol=1e-3)
    # ...the leave-one-out estimates agree with it on a point source...
    np.testing.assert_allclose(r.mean, r.naive, atol=2e-3)
    # ...and the posterior pulls them towards the prior by
    # tau^2/(tau^2 + sigma^2); what is applied is that times P(needed).
    tau2 = 0.10 ** 2
    np.testing.assert_allclose(
        r.applied_log,
        r.p_correction * r.mean * tau2 / (tau2 + r.sigma ** 2), rtol=1e-9)
    assert (r.sigma < 0.02).all()
    # Every station is 5-30% off with a ~1% uncertainty: all needed.
    assert (r.p_correction > 0.99).all()
    assert r.applied
    # ...and the corrections really are in the data now.
    gains = o.station_gains()
    np.testing.assert_allclose([gains[n] for n in o.antennas],
                               np.median(r.factors, axis=0), rtol=1e-5)


@pytest.fixture()
def per_if_errors(uvfits_file):
    """Gain errors that differ between the IFs - opposite ones, for
    most stations - baked into the data (by averaging, which applies
    the gain table and starts afresh). AN2's second IF is exact."""
    err = np.array([[1.30, 0.75, 1.10, 0.90, 1.05],
                    [0.80, 1.25, 1.00, 1.15, 0.95]], np.float32)
    o = _load(uvfits_file)
    o._core.apply_gain_factors(err)
    return o.uvaver(60), err


def test_corrections_are_per_if(per_if_errors):
    o, err = per_if_errors
    r = o.bayes_gscale(models=("point1", "gauss1"), quiet=True, plot=False)
    truth = 1.0 / err
    truth /= truth.mean(axis=1, keepdims=True)   # gscale's norm, per IF
    np.testing.assert_allclose(np.exp(r.naive), truth, rtol=2e-3)
    # Each IF gets its own correction, and is applied as such.
    assert r.factors.shape == err.shape
    np.testing.assert_allclose(r.factors, truth, rtol=0.01)
    amp, _ = o._gain_amps(fill=np.nan)
    np.testing.assert_allclose(np.nanmedian(amp, axis=0), r.factors, rtol=1e-5)
    # The exact IF is left alone; the rest are needed.
    p = r.p_correction
    assert p[1, 2] < 0.75 and np.delete(p.ravel(), 7).min() > 0.99

    rows = {row["station"]: row for row in r.station_table()}
    per_if = rows["AN0"]["per_if"]
    assert [d["if"] for d in per_if] == [1, 2]
    assert per_if[0]["correction"] == pytest.approx(r.factors[0, 0])
    assert per_if[1]["correction"] == pytest.approx(r.factors[1, 0])

    # The report shows every IF, not a median over them.
    text = r.summary()
    assert "IF1" in text and "IF2" in text
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("AN0"))
    assert f"{r.factors[0, 0]:.3f}" in line and f"{r.factors[1, 0]:.3f}" in line
    assert any("AN0: a correction is needed in IF1" in f for f in r.findings())


def test_one_correction_for_all_ifs_averages_them(per_if_errors):
    """per_if=False: one correction per station; with opposite errors in
    the two IFs the combined estimate is uncertain and near unity."""
    o, _ = per_if_errors
    r = o.bayes_gscale(models=("point1",), quiet=True, plot=False,
                       per_if=False, apply=False)
    np.testing.assert_array_equal(r.factors[0], r.factors[1])
    assert (r.sigma[:, :2] > 0.05).all()
    assert "all IFs" in r.summary()


def test_leaves_well_calibrated_data_alone(uvfits_file):
    o = _load(uvfits_file)
    r = o.bayes_gscale(models=MODELS, quiet=True, plot=False)
    np.testing.assert_allclose(r.factors, 1.0, atol=2e-3)
    # With nothing to correct, the prior wins.
    assert (r.p_correction < 0.2).all()


def test_threads_do_not_change_the_answer(corrupted_uvfits_file):
    path, _ = corrupted_uvfits_file
    a = _load(path).bayes_gscale(models=MODELS, quiet=True, plot=False,
                                 workers=1)
    b = _load(path).bayes_gscale(models=MODELS, quiet=True, plot=False,
                                 workers=8)
    np.testing.assert_array_equal(a.applied_log, b.applied_log)
    np.testing.assert_array_equal(a.model_prob, b.model_prob)


def test_run_bookkeeping(corrupted_uvfits_file):
    path, _ = corrupted_uvfits_file
    o = _load(path)
    nant = len(o.antennas)
    r = o.bayes_gscale(models=MODELS, quiet=True, plot=False, apply=False)
    assert len(r.runs) == len(MODELS) * (1 + nant)
    assert all(run["ok"] for run in r.runs), r.runs
    assert r.logg.shape == (len(MODELS), 1 + nant, NIF, nant)
    amp, _, _ = o._core.gains()
    assert np.allclose(amp, 1.0), "apply=False must leave the data alone"

    r = o.bayes_gscale(models=MODELS, quiet=True, plot=False, apply=False,
                       jackknife=False)
    assert len(r.runs) == len(MODELS)


def test_one_correction_per_station(corrupted_uvfits_file):
    path, _ = corrupted_uvfits_file
    r = _load(path).bayes_gscale(models=MODELS, quiet=True, plot=False,
                                 per_if=False, apply=False)
    np.testing.assert_allclose(r.factors, np.broadcast_to(r.factors[:1], r.factors.shape), rtol=1e-12)


def test_model_names():
    assert ModelSpec.parse("clean") == ModelSpec("clean", "clean", 200)
    assert ModelSpec.parse("clean500").n == 500
    assert ModelSpec.parse("gauss2") == ModelSpec("gauss2", "gauss", 2)
    assert ModelSpec.parse("3 points").kind == "point"
    assert ModelSpec.parse("Current").kind == "current"
    for bad in ("gauss0", "wavelets"):
        with pytest.raises(ValueError):
            ModelSpec.parse(bad)


def test_writes_report_figure_and_table(corrupted_uvfits_file, tmp_path):
    path, gerr = corrupted_uvfits_file
    o = _load(path)
    prefix = str(tmp_path / "out" / "synth")
    r = o.bayes_gscale(models=MODELS, quiet=True, plot=False, prefix=prefix)

    with open(prefix + ".json") as fh:
        rep = json.load(fh)
    assert rep["best_model"] == r.best_model
    assert [s["station"] for s in rep["stations"]] == o.antennas
    assert rep["findings"] and rep["applied"] is True
    assert set(rep["corrections_per_if"]) == set(o.antennas)

    # UVFITS data: an AIPS table of the constant corrections.
    tasav = prefix + ".TASAV.FITS"
    assert r.files["caltable"] == [tasav]
    with fits.open(tasav) as h:
        sn = h["AIPS SN"].data
    # Two identical rows per station (start and end of the observation).
    assert len(sn) == 2 * len(o.antennas)
    fac = r.factors                                     # [nif, nant]
    np.testing.assert_allclose(sn["REAL1"][: len(o.antennas)], fac.T, rtol=1e-6)
    np.testing.assert_allclose(sn["IMAG1"], 0.0, atol=1e-7)

    if pytest.importorskip("matplotlib"):
        assert os.path.getsize(prefix + ".png") > 10000
