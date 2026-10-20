"""Bayesian station amplitude calibration (`bayes_gscale`)."""

import json
import os
import re

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
    """The corrections bayes_gscale finds for these gain errors: the
    inverse amplitudes, with the median station's set to one (its
    default norm="median")."""
    c = 1.0 / np.array([a for a, _ in gerr])
    return c / np.median(c)


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
    # ...what is applied is the estimate under "off" (the wide prior,
    # outlier_sigma = 1, barely shrinks it) times P(off)...
    tau1 = 1.0 ** 2
    np.testing.assert_allclose(
        r.applied_log,
        r.p_correction * r.mean * tau1 / (tau1 + r.sigma ** 2), rtol=1e-9)
    assert (r.sigma < 0.02).all()
    # ...and only a station clearly beyond the 10% a-priori accuracy is
    # off: AN1 (x1.40 against the median AN4) is corrected, AN2 (x0.95)
    # and AN4 itself are left alone.
    p = np.median(r.p_correction, axis=0)
    assert p[1] > 0.95 and p[2] < 0.25 and p[4] < 0.25
    np.testing.assert_allclose(np.median(r.factors, axis=0)[[1, 2, 4]], [1.40, 1.0, 1.0], atol=0.02)
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
    truth /= np.median(truth, axis=1, keepdims=True)   # the median station, per IF
    np.testing.assert_allclose(np.exp(r.naive), truth, rtol=2e-3)
    np.testing.assert_allclose(np.exp(r.mean), truth, rtol=0.01)
    # Each IF gets its own correction, and is applied as such.
    assert r.factors.shape == err.shape
    amp, _ = o._gain_amps(fill=1.0)          # a correction of exactly one leaves no solution
    np.testing.assert_allclose(np.median(amp, axis=0), r.factors, rtol=1e-5)
    # AN1 is 40% off in IF1 and only 20% in IF2: off in the one, not
    # clearly in the other; the median stations (exact) are left alone.
    p = r.p_correction
    assert p[0, 1] > 0.95 and p[1, 1] < 0.6 and p[0, 4] < 0.25 and p[1, 2] < 0.25

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
    assert any("AN1: a correction is needed in IF1" in f for f in r.findings())


@pytest.fixture()
def one_station_far_off(uvfits_file):
    """AN0 a factor 4 low in IF1 only; everything else exact."""
    err = np.ones((NIF, 5), np.float32)
    err[0, 0] = 0.25
    o = _load(uvfits_file)
    o._core.apply_gain_factors(err)
    return o.uvaver(60)


def test_one_station_far_off_does_not_drag_the_others(one_station_far_off):
    """The median station holds the flux scale: AN0 gets its x4 and the
    others nothing; difmap's mean renormalisation (norm="mean") would
    have scaled every other station of IF1 by 1/1.6, and the findings
    say so."""
    r = one_station_far_off.bayes_gscale(models=("point1",), quiet=True, plot=False, apply=False)
    np.testing.assert_allclose(np.exp(r.naive[0]), [4, 1, 1, 1, 1], rtol=0.01)
    np.testing.assert_allclose(np.exp(r.naive[1:]), 1.0, rtol=0.01)
    # It is corrected in full, in the IF it is off in only; nothing else moves.
    np.testing.assert_allclose(r.factors[0, 0], 4.0, rtol=0.02)
    np.testing.assert_allclose(np.delete(r.factors.ravel(), 0), 1.0, atol=0.01)
    assert any(re.match(r"AN0: difmap's gscale gives it x(3\.9\d|4\.0\d) in IF1, .* by x0\.6[23] .*"
                        r"applied to it: x(3\.9\d|4\.0\d) \(P = 1\.00\)$", f) for f in r.findings())

    mean = one_station_far_off.bayes_gscale(models=("point1",), quiet=True, plot=False, apply=False,
                                            norm="mean")
    np.testing.assert_allclose(np.exp(mean.naive[0]), np.array([4, 1, 1, 1, 1]) / 1.6, rtol=0.01)
    assert not any("mean renormalisation" in f for f in mean.findings())
    with pytest.raises(ValueError):
        one_station_far_off.bayes_gscale(models=("point1",), quiet=True, plot=False, norm="mode")
    with pytest.raises(ValueError):
        one_station_far_off.bayes_gscale(models=("point1",), quiet=True, plot=False, outlier_sigma=0.05)


def test_model_sigma_leaves_an_unconstrained_station_alone(one_station_far_off):
    """A model uncertainty as large as the offset (what the station's own
    jackknife would say of one nobody else constrains) undoes the
    confidence of a fixed-model run: AN0's x4 is mostly left alone, the
    others are untouched, and a per-IF value of the wrong length is an
    error."""
    r = one_station_far_off.bayes_gscale(models=("point1",), quiet=True, plot=False, apply=False,
                                         model_sigma={"AN0": [2.0, None], "AN9": 0.5})
    assert r.p_correction[0, 0] < 0.6
    assert r.factors[0, 0] < 1.3
    np.testing.assert_allclose(np.delete(r.factors.ravel(), 0), 1.0, atol=0.01)
    assert r.settings["model_sigma"]["AN0"][0] == 2.0
    with pytest.raises(ValueError):
        one_station_far_off.bayes_gscale(models=("point1",), quiet=True, plot=False, apply=False,
                                         model_sigma={"AN0": [2.0, 2.0, 2.0]})


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
