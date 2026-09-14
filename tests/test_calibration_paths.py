"""Regression tests for the calibration/geometry write paths.

Everything that modifies the data the user sees (antenna gains,
baseline corrections, phase-center shifts) must be reflected in what
gets written out or averaged - these were silently dropped once.
"""

import os

import numpy as np
import pytest

import difmapy

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
real_data = pytest.mark.skipif(
    not os.path.exists(UVF), reason="real 3C345 data not present"
)


def _matched(a, b):
    """Match two observations row-by-row on (time, ant1, ant2).

    Times are matched to the nearest second: writing them through the
    UVFITS DATE parameters (a Julian Date around 2.46e6) loses the
    sub-microsecond digits, and the EVN timestamps sit just below
    integer seconds, so finer rounding would split matching rows.
    """
    t1, a11, a21, *_ = a._core.rows()
    t2, a12, a22, *_ = b._core.rows()
    k1 = {(int(round(float(t))), x, y): i for i, (t, x, y) in enumerate(zip(t1, a11, a21))}
    k2 = {(int(round(float(t))), x, y): i for i, (t, x, y) in enumerate(zip(t2, a12, a22))}
    common = sorted(set(k1) & set(k2))
    assert common
    return (
        np.array([k1[k] for k in common]),
        np.array([k2[k] for k in common]),
    )


@pytest.fixture()
def modelled(uvfits_file):
    """An observation with a model and non-trivial gains."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.selfcal(phase=True)
    return o


def test_moddif_measures_how_well_the_model_fits(uvfits_file):
    """The exact model of the synthetic point source leaves nothing
    behind; moving it off the source does not."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    exact = o.moddif()
    assert exact["ndata"] == 2 * exact["nvis"]
    assert exact["rms"] < 1e-4 * FLUX

    o.clearmodel()
    o.addcmp(FLUX, X0_MAS + 5.0, Y0_MAS)
    assert o.moddif()["rms"] > 0.1 * FLUX
    # A UV range that excludes data narrows the sample.
    limited = o.moddif(uvmax=5e7)
    assert 0 < limited["nvis"] < exact["nvis"]


def test_selfcal_interpolates_binned_solutions_onto_integrations(tmp_path):
    """With a solution interval set, difmap does not apply a bin's
    solution as a step: it smooths and interpolates the bins onto the
    integration grid with a Gaussian of sigma = 0.37478 * solint
    (slfcal.c apply_solns), which is why its corplot shows a smooth
    evolution. Ours must do the same - the give-away is that the
    corrections vary *within* a solution bin."""
    from conftest import FLUX, X0_MAS, Y0_MAS, make_uvfits

    rng = np.random.default_rng(3)
    jitter = {}

    def gerr(t):
        # Gains that differ every integration, so the per-bin solutions
        # scatter and a step function would be unmistakable.
        if t not in jitter:
            jitter[t] = [(1.0 + 0.25 * rng.standard_normal(),
                          0.5 * rng.standard_normal()) for _ in range(5)]
        return jitter[t]

    path = str(make_uvfits(tmp_path / "jitter.uvf", gerr=gerr))
    solint = 5.0
    o = difmapy.load(path)
    o.select("I")
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.selfcal(phase=True, amp=True, solint=solint, float_scale=True, quiet=True)

    t = np.asarray(o._core.times())
    nt, nif, nant = o._core.ntimes, o.nif, len(o.antennas)
    amp = np.asarray(o._core.gains()[0]).reshape(nt, nif, nant)[:, 0, :]
    phs = np.asarray(o._core.gains()[1]).reshape(nt, nif, nant)[:, 0, :]

    # Every integration gets its own correction, not one per bin.
    binid = np.floor(t / (solint * 60.0)).astype(int)
    assert len(np.unique(binid)) < nt // 3          # far fewer bins
    assert len(np.unique(np.round(amp[:, 0], 7))) == nt

    # And they change inside a bin, which a step function cannot do.
    inside = max(
        float(np.ptp(amp[binid == b, ia]))
        for b in np.unique(binid) for ia in range(nant)
        if (binid == b).sum() > 1
    )
    assert inside > 0.1 * float(np.ptp(amp[:, 0]))

    # No discontinuities: consecutive integrations stay close compared
    # with the overall range the corrections cover.
    for y, span in ((amp[:, 0], np.ptp(amp[:, 0])),
                    (phs[:, 0], np.ptp(phs[:, 0]))):
        assert np.abs(np.diff(y)).max() < 0.5 * span

    # Per-integration solving has nothing to interpolate, and a single
    # solution (gscale) is constant - as in difmap.
    o2 = difmapy.load(path)
    o2.select("I")
    o2.addcmp(FLUX, X0_MAS, Y0_MAS)
    o2.gscale(float_scale=True, quiet=True)
    g = np.asarray(o2._core.gains()[0]).reshape(nt, nif, nant)[:, 0, 0]
    assert len(np.unique(np.round(g, 7))) == 1


def test_selfcal_reports_the_fit_before_and_after(corrupted_uvfits_file, capsys):
    """difmap's self-cal reports how the model-data fit changed; so does
    this one, plus the residual map statistics when a map exists."""
    path, _ = corrupted_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    o.mapsize(256, 0.25)
    o.invert()
    o.clean(200, 0.05, quiet=True)
    res = o.selfcal(phase=True)

    out = capsys.readouterr().out
    assert "fit before" in out and "fit after" in out and "map after" in out
    assert res["fit_after"]["rms"] < res["fit_before"]["rms"]
    assert res["fit_before"]["ndata"] == 2 * res["fit_before"]["nvis"]
    assert res["map_after"]["rms"] < res["map_before"]["rms"]

    o.clean(100, 0.05, quiet=True)
    res = o.selfcal(phase=True, quiet=True, mapstats=False)
    assert capsys.readouterr().out == ""
    assert "map_after" not in res and "fit_after" in res


def test_wobs_includes_gains(modelled, tmp_path):
    v1, w1 = modelled._core.stream_vis()
    out = str(tmp_path / "g.uvf")
    modelled.wobs(out)
    back = difmapy.load(out)
    back.select("I")
    i1, i2 = _matched(modelled, back)
    v2, w2 = back._core.stream_vis()
    g = (np.asarray(w1)[i1] > 0) & (np.asarray(w2)[i2] > 0)
    assert np.abs(v2[i2][g] - v1[i1][g]).max() < 1e-4


def test_wobs_includes_baseline_corrections(modelled, tmp_path):
    """resoff corrections must survive a write/read cycle."""
    modelled.resoff()
    v1, w1 = modelled._core.stream_vis()
    out = str(tmp_path / "r.uvf")
    modelled.wobs(out)
    back = difmapy.load(out)
    back.select("I")
    i1, i2 = _matched(modelled, back)
    v2, w2 = back._core.stream_vis()
    g = (np.asarray(w1)[i1] > 0) & (np.asarray(w2)[i2] > 0)
    assert np.abs(v2[i2][g] - v1[i1][g]).max() < 1e-4


def test_wobs_shift_handling(uvfits_file, tmp_path):
    """By default shifts are NOT frozen into the output (difmap
    behaviour); freeze_shift=True writes the shifted data."""
    o = difmapy.load(uvfits_file)
    o.select("I")
    v_unshifted, w = o._core.stream_vis()
    o.shift(3.0, -2.0)
    v_shifted, _ = o._core.stream_vis()
    assert np.abs(np.asarray(v_shifted) - np.asarray(v_unshifted)).max() > 1e-3

    plain = str(tmp_path / "plain.uvf")
    frozen = str(tmp_path / "frozen.uvf")
    o.wobs(plain)
    o.wobs(frozen, freeze_shift=True)

    for path, ref in ((plain, v_unshifted), (frozen, v_shifted)):
        back = difmapy.load(path)
        back.select("I")
        i1, i2 = _matched(o, back)
        vb, wb = back._core.stream_vis()
        g = (np.asarray(w)[i1] > 0) & (np.asarray(wb)[i2] > 0)
        assert np.abs(vb[i2][g] - np.asarray(ref)[i1][g]).max() < 1e-3, path


def test_uvaver_includes_shift(uvfits_file):
    """Averaging must average what the user sees, including shifts."""
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.shift(3.0, -2.0)
    o.invert()
    pos, _ = o.peak_offset()
    avg = o.uvaver(120.0)
    avg.invert()
    pos2, _ = avg.peak_offset()
    assert pos2 == pytest.approx(pos, abs=0.25)
    # The averaged data are already corrected, so no shift is pending.
    assert avg.total_shift == (0.0, 0.0)


def test_shift_moves_windows(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.add_window(-2.0, 2.0, -1.0, 1.0)
    o.shift(5.0, 3.0)
    assert o.windows == [(3.0, 7.0, 2.0, 4.0)]
    o.unshift()
    assert o.windows == [(-2.0, 2.0, -1.0, 1.0)]


def test_save_get_restores_shift(uvfits_file, tmp_path):
    """A saved session must come back with the same shift, model and
    windows, and image identically."""
    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.shift(3.0, -2.0)
    o.add_window(-3.0, 3.0, -3.0, 3.0)
    o.invert()
    o.clean(100, 0.05)
    prefix = str(tmp_path / "sess")
    o.save(prefix)

    back = difmapy.Observation.get(prefix)
    assert back.total_shift == pytest.approx((3.0, -2.0))
    assert back.windows == o.windows
    assert back.model_flux == pytest.approx(o.model_flux, rel=1e-4)
    # Model component positions are in the shifted frame in both.
    m1 = max(o.model, key=lambda c: abs(c["flux"]))
    m2 = max(back.model, key=lambda c: abs(c["flux"]))
    assert (m2["x"], m2["y"]) == pytest.approx((m1["x"], m1["y"]), abs=1e-3)
    o.invert()
    back.invert()
    assert np.abs(o.dmap - back.dmap).max() < 1e-3 * np.abs(o.dmap).max()


def test_modelfit_free_mask_binds_to_right_component(uvfits_file):
    """The free-parameter mask must follow its own component, even when
    CLEAN adds components before it."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.invert()
    o.clean(20, 0.05)  # CLEAN components come first
    n_clean = len(o.model)
    assert n_clean >= 1

    # Now add the component to be fitted, deliberately displaced.
    o.addcmp(0.5 * FLUX, X0_MAS + 3.0, Y0_MAS - 3.0, free=["flux", "pos"])
    masks = [c["freepar"] for c in o.model]
    assert masks[:n_clean] == [0] * n_clean  # clean components are fixed
    assert masks[n_clean] == 3  # flux | pos

    before = [dict(c) for c in o.model]
    res = o.modelfit(niter=30)
    assert res["nfree"] == 3
    after = [dict(c) for c in o.model]
    assert len(after) == len(before)
    assert res["ncomp"] == 1
    # The CLEAN components must be untouched: the fit works on the
    # residuals after them (difmap obvarmod).
    for a, b in zip(before[:n_clean], after[:n_clean]):
        assert a["flux"] == b["flux"]
        assert (a["x"], a["y"]) == (b["x"], b["y"])
    # ...and only the fitted component may have moved.
    assert (after[n_clean]["x"], after[n_clean]["y"]) != (
        before[n_clean]["x"],
        before[n_clean]["y"],
    )


def test_modelfit_does_not_count_fixed_components_twice(uvfits_file):
    """A fixed component beside the one being fitted is established, not
    dropped: dropping it left the fit to absorb its flux a second time,
    and putting it back into the model then doubled it."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.invert()
    o.clean(100, 0.05, quiet=True)  # CLEAN deltas, all fixed
    cleaned = o.model_flux
    assert 0.5 * FLUX < cleaned < FLUX

    o.addcmp(0.5 * FLUX, X0_MAS, Y0_MAS, type="gauss", major=1.0,
             free=["flux", "pos", "major"])
    o.modelfit(quiet=True)

    # The whole model accounts for the source once, not twice.
    assert o.model_flux == pytest.approx(FLUX, rel=0.02)
    # And the model, fitted part included, leaves no signal.
    assert o.moddif()["rms"] < 1e-3 * FLUX


def test_modelfit_refits_established_variable_components(uvfits_file):
    """A component keeps its free parameters in the model, so running
    modelfit again re-fits it rather than seeding a new one, as difmap
    does - which is what lets a fit be iterated."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    o.addcmp(0.5 * FLUX, X0_MAS + 1.0, Y0_MAS, free=["flux", "pos"])
    assert o.nvariable == 1

    res = o.modelfit(niter=50, quiet=True)
    assert res["ncomp"] == 1  # the existing one, not a fresh seed
    assert len(o.model) == 1
    c = o.model[0]
    assert c["flux"] == pytest.approx(FLUX, rel=1e-3)
    assert (c["x"], c["y"]) == pytest.approx((X0_MAS, Y0_MAS), abs=1e-3)


@real_data
def test_real_data_wobs_after_full_session(tmp_path):
    """After a full session (selfcal + resoff + shift), the written
    file must reproduce the corrected data."""
    o = difmapy.load(UVF)
    o.select("I")
    o.mapsize(1024, 1.0)
    o.startmod(flux=1.0)
    o.clean(200, 0.03)
    o.selfcal(phase=True)
    o.clean(200, 0.03)
    o.resoff()
    v1, w1 = o._core.stream_vis()

    out = str(tmp_path / "session.uvf")
    o.wobs(out)
    back = difmapy.load(out)
    back.select("I")
    i1, i2 = _matched(o, back)
    v2, w2 = back._core.stream_vis()
    g = (np.asarray(w1)[i1] > 0) & (np.asarray(w2)[i2] > 0)
    assert g.sum() > 10000
    rel = np.abs(v2[i2][g] - v1[i1][g]).max() / np.median(np.abs(v1[i1][g]))
    assert rel < 1e-4, f"relative deviation {rel}"


def test_the_model_needs_no_keep(uvfits_file):
    """Whatever addcmp, modelfit and clean produce is the model at once:
    in the model visibilities, measured by the residuals and solved
    against by selfcal, with no separate step to establish it."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    o = difmapy.load(uvfits_file)
    o.select("I")
    o.mapsize(256, 0.25)
    assert not hasattr(o, "keep")

    o.addcmp(0.5 * FLUX, X0_MAS, Y0_MAS, free=["flux", "pos"])
    assert "tentative" not in o.model[0]
    model_vis = np.abs(np.asarray(o._core.stream_model()))
    assert model_vis.max() == pytest.approx(0.5 * FLUX, rel=1e-5)

    o.modelfit(quiet=True)
    assert list(o._core.get_models()[1]) == []
    assert o.model[0]["flux"] == pytest.approx(FLUX, rel=1e-3)
    assert o.moddif()["rms"] < 1e-3 * FLUX  # against the fitted model
    o.selfcal(phase=True, quiet=True)
    assert o.moddif()["rms"] < 1e-3 * FLUX

    o.clearmodel()
    o.invert()
    o.clean(50, 0.1, quiet=True)
    assert list(o._core.get_models()[1]) == [] and len(o.model) > 0
    assert np.abs(np.asarray(o._core.stream_model())).max() > 0
