"""Unit-bearing arguments: "30s", "1min", "scan", "2scan".

Plain numbers must keep the unit each argument has always had, so that
existing scripts are unaffected; strings are how a different unit (or a
scan-based interval) is asked for.
"""

import numpy as np
import pytest

import difmapy
from difmapy.units import format_interval, parse_interval, parse_time


@pytest.mark.parametrize(
    "spec,seconds",
    [
        (30, 30.0), (30.0, 30.0), ("30", 30.0),
        ("30s", 30.0), ("30 s", 30.0), ("30sec", 30.0), ("30 seconds", 30.0),
        ("1m", 60.0), ("1min", 60.0), ("1 minute", 60.0), ("2.5 mins", 150.0),
        ("1h", 3600.0), ("1.5 hours", 5400.0), ("1hr", 3600.0),
        ("0.5d", 43200.0),
        ("-30s", -30.0), ("1e2s", 100.0),
    ],
)
def test_parse_time(spec, seconds):
    assert parse_time(spec) == pytest.approx(seconds)


def test_parse_time_default_unit():
    """A bare number means the argument's own unit."""
    assert parse_time(30, "min") == 1800.0
    assert parse_time("30", "min") == 1800.0
    assert parse_time("30s", "min") == 30.0  # a unit always wins
    assert parse_time(None) is None


@pytest.mark.parametrize(
    "spec,expected",
    [
        (0, (0.0, 0)), (30, (1800.0, 0)), ("30s", (30.0, 0)),
        ("scan", (0.0, 1)), ("SCAN", (0.0, 1)), (" scan ", (0.0, 1)),
        ("2scan", (0.0, 2)), ("3 scans", (0.0, 3)),
    ],
)
def test_parse_interval(spec, expected):
    assert parse_interval(spec, "min") == expected


@pytest.mark.parametrize(
    "spec", ["30 furlongs", "abc", "", "scan5", "1.5scan", "0scan", "-2scan"]
)
def test_parse_interval_rejects_nonsense(spec):
    with pytest.raises(ValueError):
        parse_interval(spec, "min")


def test_format_interval():
    assert format_interval(0.0, 0) == "per integration"
    assert format_interval(30.0, 0) == "every 30 s"
    assert format_interval(90.0, 0) == "every 90 s"
    assert format_interval(1800.0, 0) == "every 30 min"
    assert format_interval(5400.0, 0) == "every 90 min"
    assert format_interval(7200.0, 0) == "every 2 h"
    assert format_interval(0.0, 1) == "per scan"
    assert format_interval(0.0, 3) == "every 3 scans"


# ----------------------------------------------------------------------
# scans
# ----------------------------------------------------------------------


def test_scans_are_found_by_their_gaps(scanned_uvfits_file):
    path, nscan = scanned_uvfits_file
    o = difmapy.load(path)
    o.select("I")

    # 60 s integrations, so the default gap is 5 x 60 s, well under the
    # 10-minute gaps between scans.
    assert o.default_scangap == pytest.approx(300.0, rel=1e-3)
    scans = o.scans()
    assert len(scans) == nscan
    assert sum(s["nint"] for s in scans) == o._core.ntimes
    assert [s["first"] for s in scans] == [0, 15, 30, 45]
    for s in scans:
        assert s["tmax"] > s["tmin"]

    # A gap longer than the inter-scan gaps merges everything; a shorter
    # one still finds the same scans, since nothing inside one is that
    # far apart.
    assert len(o.scans(gap="30min")) == 1
    assert len(o.scans(gap="100s")) == nscan
    assert len(o.scans(gap=100)) == nscan


def test_selfcal_scan_interval_gives_one_solution_per_scan(scanned_uvfits_file):
    from conftest import FLUX, X0_MAS, Y0_MAS

    path, nscan = scanned_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    o.addcmp(FLUX, X0_MAS, Y0_MAS)

    # nbins counts solution bins per (subarray, IF).
    per_if = o.nif
    assert o.selfcal(phase=True, solint="scan", quiet=True)["nbins"] == nscan * per_if
    o.uncalib()
    assert o.selfcal(phase=True, solint="2scan", quiet=True)["nbins"] == 2 * per_if
    o.uncalib()
    assert o.selfcal(phase=True, solint="4scan", quiet=True)["nbins"] == per_if

    # A duration cannot make that guarantee: fixed bins are aligned to
    # the clock, not to the scans, so they do not come out one per scan.
    o.uncalib()
    assert o.selfcal(phase=True, solint="5min", quiet=True)["nbins"] != nscan * per_if

    # More scans per solution than there are is one solution.
    o.uncalib()
    assert o.selfcal(phase=True, solint="99scan", quiet=True)["nbins"] == per_if


def test_selfcal_scangap_overrides_the_scan_detection(scanned_uvfits_file):
    from conftest import FLUX, X0_MAS, Y0_MAS

    path, nscan = scanned_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    # With a gap longer than the ones in the data there is one scan, so
    # "scan" solves the whole observation at once.
    res = o.selfcal(phase=True, solint="scan", scangap="30min", quiet=True)
    assert res["nbins"] == o.nif
    o.uncalib()
    res = o.selfcal(phase=True, solint="scan", scangap=100, quiet=True)
    assert res["nbins"] == nscan * o.nif


def test_selfcal_reports_the_interval_it_used(scanned_uvfits_file, capsys):
    from conftest import FLUX, X0_MAS, Y0_MAS

    path, _ = scanned_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    o.selfcal(phase=True, solint="scan")
    assert "phase per scan" in capsys.readouterr().out
    o.uncalib()
    o.selfcal(amp=True, phase=True, solint="90s")
    assert "amplitude and phase every 90 s" in capsys.readouterr().out


def test_selfcal_solint_number_is_still_minutes(scanned_uvfits_file):
    """difmap's unit for a bare solint, unchanged."""
    from conftest import FLUX, X0_MAS, Y0_MAS

    path, _ = scanned_uvfits_file
    o = difmapy.load(path)
    o.select("I")
    o.addcmp(FLUX, X0_MAS, Y0_MAS)
    a = o.selfcal(phase=True, solint=5, quiet=True)["nbins"]
    o.uncalib()
    b = o.selfcal(phase=True, solint="5min", quiet=True)["nbins"]
    assert a == b


def test_selfcal_rejects_a_bad_interval(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    with pytest.raises(ValueError, match="cannot read"):
        o.selfcal(phase=True, solint="10 fortnights")


# ----------------------------------------------------------------------
# the other unit-bearing arguments
# ----------------------------------------------------------------------


def test_time_ranges_take_units(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    t = np.asarray(o._core.times())
    half = float(np.median(t))

    # The same half of the observation, spelled four ways.
    counts = []
    for tmax in (half, f"{half}s", f"{half / 60.0}min", f"{half / 3600.0}h"):
        counts.append(o.flag(tmin=0, tmax=tmax))
        o.unflag()
    assert len(set(counts)) == 1 and counts[0] > 0

    s_num = o.spectrum(tmax=half)
    s_str = o.spectrum(tmax=f"{half / 60.0}min")
    assert np.allclose(s_num["wt"], s_str["wt"])


def test_uvaver_takes_units(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.select("I")
    by_number = o.uvaver(180.0)
    by_string = o.uvaver("3min")
    assert by_number._core.ntimes == by_string._core.ntimes
    assert by_number._core.nrow == by_string._core.nrow


def test_fplot_time_range_takes_units(uvfits_file):
    import os

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_QPA_PLATFORMTHEME"] = ""
    pytest.importorskip("pyqtgraph")
    from difmapy.plots.diagnostics import FPlot

    o = difmapy.load(uvfits_file)
    o.select("I")
    p = FPlot(o, nplot=1, tmin="0s", tmax="30min")
    assert (p.tmin, p.tmax) == (0.0, 1800.0)
    p.close()
