"""uvtaper and uvrange: units, what the gridding does with them, and their controls in mapplot and the dashboard."""

import os

import numpy as np
import pytest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = ""

import difmapy
from difmapy import MAS
from difmapy.units import parse_uvdist

NX, CELL = 256, 0.25


@pytest.fixture()
def obs(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.mapsize(NX, CELL)
    # Natural weighting that ignores the data weights: the gridding weight
    # of every visibility is then the taper alone.
    o.uvweight(0, 0)
    return o


def _uv(obs):
    """u, v (wavelengths) of every visibility the gridding sees."""
    core = obs._core
    _, wt = core.stream_vis()
    _, _, _, us, vs, _ = core.rows()
    sel = core.selection()
    freq = np.asarray(sel["if_freq"])
    good = (np.asarray(wt) > 0) & np.asarray(sel["if_used"])[None, :]
    u = (np.asarray(us)[:, None] * freq[None, :])[good]
    v = (np.asarray(vs)[:, None] * freq[None, :])[good]
    return u, v


def _pixels(n=200, seed=1):
    """Random pixels of the inner quarter, and their (x, y) in radians."""
    rng = np.random.default_rng(seed)
    pix = rng.integers(NX // 2 - NX // 8, NX // 2 + NX // 8, size=(n, 2))
    return pix, (pix[:, 0] - NX / 2) * CELL * MAS, (pix[:, 1] - NX / 2) * CELL * MAS


def _dft_beam(u, v, w, x, y):
    """The dirty beam of weights `w`, by direct Fourier sum."""
    return np.cos(2 * np.pi * (np.outer(x, u) + np.outer(y, v))) @ w / w.sum()


def test_parse_uvdist_units():
    assert parse_uvdist(50) == 5e7                 # M-lambda by default
    assert parse_uvdist("50M") == parse_uvdist("50 Mλ") == parse_uvdist("50 Mlambda") == 5e7
    assert parse_uvdist("500k") == parse_uvdist("500 klambda") == 5e5
    assert parse_uvdist("2e6 lambda") == 2e6
    assert parse_uvdist("1.5G") == 1.5e9
    assert parse_uvdist(5e7, "") == 5e7
    assert parse_uvdist(None) is None
    with pytest.raises(ValueError):
        parse_uvdist("50x")


def test_uvtaper_weights_the_data_with_a_gaussian(obs):
    """The dirty beam with a taper is the Fourier sum of the visibilities
    weighted by value ** ((r / radius) ** 2) - a Gaussian that is 1 at the
    centre and `value` at `radius` - and not of the untapered weights."""
    u, v = _uv(obs)
    r = np.hypot(u, v)
    pix, x, y = _pixels()
    radius = 0.5 * r.max()
    for value, spec in ((0.5, radius / 1e6), (0.1, f"{radius / 1e3:.9g}k")):
        obs.uvtaper(value, spec)
        assert obs._gaurad == pytest.approx(radius)
        obs.invert(verbose=False)
        beam = np.asarray(obs.dbeam)[pix[:, 1], pix[:, 0]]
        w = value ** ((r / radius) ** 2)
        assert np.abs(beam - _dft_beam(u, v, w, x, y)).max() < 3e-3
        assert np.abs(beam - _dft_beam(u, v, np.ones_like(r), x, y)).max() > 0.05
    # The weight function itself: 1 at the centre, `value` at the radius.
    assert 0.1 ** ((0.0 / radius) ** 2) == 1.0 and 0.1 ** ((radius / radius) ** 2) == pytest.approx(0.1)
    # A taper widens the beam; turning it off gives the old one back.
    obs.uvtaper()
    b0 = obs.invert(quiet=False, verbose=False)["e_bmin"]
    obs.uvtaper(0.1, radius / 1e6)
    assert obs.invert(quiet=False, verbose=False)["e_bmin"] > b0
    obs.uvtaper(0, 0)
    assert obs.invert(quiet=False, verbose=False)["e_bmin"] == pytest.approx(b0)


def test_uvtaper_rejects_a_radius_given_in_wavelengths(obs):
    """A bare number is M-lambda now; 5e7 of them is no baseline, so the
    old wavelength spelling fails loudly instead of tapering nothing."""
    with pytest.raises(ValueError, match="M-lambda"):
        obs.uvtaper(0.5, 5e7)
    obs.uvtaper(0.5, "5e7 lambda")
    assert obs._gaurad == 5e7
    with pytest.raises(ValueError):
        obs.uvtaper(1.5, 10)


def test_uvrange_grids_only_the_visibilities_in_range(obs):
    u, v = _uv(obs)
    r = np.hypot(u, v)
    pix, x, y = _pixels()
    mid = float(np.median(r))
    for lo, hi in ((0, mid / 1e6), (mid / 1e6, 0), (f"{mid / 1e3:.9g}k", None)):
        obs.uvrange(lo, hi)
        res = obs.invert(quiet=False, verbose=False)
        lo_w = parse_uvdist(lo)
        hi_w = parse_uvdist(hi) if hi else np.inf
        keep = (r >= lo_w) & (r <= hi_w)
        assert res["nused"] == keep.sum()
        beam = np.asarray(obs.dbeam)[pix[:, 1], pix[:, 0]]
        assert np.abs(beam - _dft_beam(u, v, keep.astype(float), x, y)).max() < 3e-3
    obs.uvrange()
    assert obs.invert(quiet=False, verbose=False)["nused"] == r.size


def test_taper_and_range_survive_save_and_get(obs, tmp_path):
    obs.uvtaper(0.3, 12)
    obs.uvrange("2M", 40)
    obs.save(str(tmp_path / "s"), ms=False, verbose=False)
    back = difmapy.Observation.get(str(tmp_path / "s"))
    assert (back._gauval, back._gaurad) == pytest.approx((0.3, 1.2e7))
    assert (back._uvmin, back._uvmax) == pytest.approx((2e6, 4e7))
    info = back.mapinfo()
    assert info["uvtaper"] == pytest.approx({"value": 0.3, "radius": 1.2e7})


def test_mapplot_takes_and_shows_the_taper_and_range(obs):
    from difmapy.plots.mapplot import MapPlot

    p = MapPlot(obs, verbose=False, uvtaper=(0.5, 20), uvrange=(1, "60M"))
    assert (obs._gauval, obs._gaurad) == (0.5, 2e7)
    assert (obs._uvmin, obs._uvmax) == (1e6, 6e7)
    assert {k: e.text() for k, e in p.uv_edits.items()} == {
        "uvmin": "1", "uvmax": "60", "value": "0.5", "radius": "20"}
    # Typing in the fields and pressing Enter applies them and re-images.
    p.uv_edits["value"].setText("0.2")
    p.uv_edits["radius"].setText("50000k")
    p.uv_edits["uvmax"].setText("")
    rev = p._revision
    p._uv_entered()
    assert (obs._gauval, obs._gaurad, obs._uvmax) == (0.2, 5e7, 0.0)
    assert p._revision > rev
    # A taper that leaves no weight on these 20-70 M-lambda baselines is
    # reported in the status bar, not raised from the key handler.
    p.uv_edits["radius"].setText("0.5")
    p._uv_entered()
    assert "leaves no weight" in p.statusBar().currentMessage()
    p.uv_edits["value"].setText("")                 # empty: no taper
    p.uv_edits["radius"].setText("")
    p._uv_entered()
    assert obs._gauval == 0.0
    # A setting made at the prompt shows up on the next refresh.
    obs.uvtaper(0.4, 8)
    p.refresh()
    assert p.uv_edits["radius"].text() == "8"
    p.close()


def test_dashboard_uv_rows(obs):
    from difmapy.plots.dashboard import Dashboard

    d = Dashboard(obs)
    d.uv_edits["uvmin"].setText("1")
    d.uv_edits["uvmax"].setText("30")
    d.uvrange_button.click()
    assert (obs._uvmin, obs._uvmax) == (1e6, 3e7)
    d.uv_edits["value"].setText("0.5")
    d.uv_edits["radius"].setText("10")
    d.uvtaper_button.click()
    assert (obs._gauval, obs._gaurad) == (0.5, 1e7)
    mp = d.map_panel.current_plot()
    assert mp.uv_edits == {}                         # set in the Main tab only
    assert "taper 0.5 at 10" in d.output.toPlainText()
    d.uv_edits["value"].setText("x")
    assert d.run_uvtaper() is None                  # a typo is reported, not raised
    assert "uvtaper" in d.output.toPlainText()
    obs.uvtaper()                                   # from the prompt ...
    d._sync_uv_fields()
    assert d.uv_edits["value"].text() == ""         # ... shows in the dashboard
    d.close()
