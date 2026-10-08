"""Exporting the session's flags as an AIPS FG table or a CASA
flag-command list (`wflags`, and `save`)."""

import os

import numpy as np
import pytest
from astropy.io import fits

import difmapy
from conftest import NANT, NCHAN, NIF, REF_JD


@pytest.fixture()
def obs(uvfits_file):
    return difmapy.load(uvfits_file)


def _times(o):
    return np.asarray(o._core.times())


def test_no_new_flags_no_entries(obs, tmp_path):
    assert obs.flag_commands() == {"rows": [], "nif": NIF, "nsamples": 0}
    files = obs.save(str(tmp_path / "s"), quiet=True)
    assert "flags" not in files
    with pytest.raises(ValueError, match="flags=True"):
        obs.save(str(tmp_path / "s"), quiet=True, flags=True)


def test_flags_compact_into_station_and_baseline_entries(obs):
    t = _times(obs)
    obs.flag(station="AN1", tmin=t[3], tmax=t[9])
    obs.flag(baseline=("AN0", "AN2"), tmin=t[20], tmax=t[22], if_index=1)
    obs.flag(station="AN3", tmin=t[5], tmax=t[6])     # overlaps AN1's
    e = obs.flag_commands()
    rows = sorted(e["rows"], key=lambda r: (r["ant1"], r["t0"]))
    assert [(r["ant1"], r["ant2"], r["if0"], r["if1"]) for r in rows] == [
        (0, 2, 1, 1), (1, None, 0, NIF - 1), (3, None, 0, NIF - 1)]
    assert all(r["chans"] is None and r["pols"] is None for r in rows)
    # Each range covers its integrations and no neighbour.
    station = rows[1]
    assert t[2] < station["t0"] < t[3] and t[9] < station["t1"] < t[10]
    assert e["nsamples"] == int(np.asarray(obs.flags).sum())


def test_channel_and_polarization_flags(obs):
    flags = np.array(obs.flags, copy=True)
    _, a1, a2, *_ = (np.asarray(x) for x in obs._core.rows())
    rows = np.nonzero((a1 == 0) & (a2 == 1))[0][4:8]
    flags[rows, NCHAN + 1 : NCHAN + 3, 0] = True     # IF 2, channels 1-2, RR
    obs._core.set_flags(np.ascontiguousarray(flags))
    (entry,) = obs.flag_commands()["rows"]
    assert (entry["ant1"], entry["ant2"]) == (0, 1)
    assert (entry["if0"], entry["if1"]) == (1, 1)
    assert entry["chans"] == (1, 2) and entry["pols"] == [True, False]


def test_ignored_stations_are_not_exported(obs):
    obs.ignore("AN2")
    assert obs.flag_commands()["rows"] == []


def test_averaged_data_refer_to_the_original_bins_and_channels(uvfits_file):
    o = difmapy.load(uvfits_file, timeavg=180, freqavg=2)
    flags = np.array(o.flags, copy=True)
    _, a1, a2, *_ = (np.asarray(x) for x in o._core.rows())
    row = np.nonzero((a1 == 0) & (a2 == 1))[0][2]
    flags[row, 1, :] = True          # IF 1, second averaged channel
    o._core.set_flags(np.ascontiguousarray(flags))
    (entry,) = o.flag_commands()["rows"]
    # Averaged channel 1 of IF 1 is original channels 2-3...
    assert entry["chans"] == (2, 3) and (entry["if0"], entry["if1"]) == (0, 0)
    # ...and the time range is the whole 180 s bin.
    t = float(o._core.rows()[0][row])
    assert entry["t0"] == np.floor(t / 180) * 180
    assert entry["t1"] == entry["t0"] + 180


def test_fg_table_layout_and_values(obs, tmp_path):
    t = _times(obs)
    obs.flag(station="AN1", tmin=t[3], tmax=t[9])
    obs.flag(baseline=("AN2", "AN0"), tmin=t[20], tmax=t[22], if_index=1)
    out = str(tmp_path / "f.FG.TASAV.FITS")
    info = obs.wflags(out, quiet=True)
    assert info["format"] == "aips" and info["nrows"] == 2
    with fits.open(out) as h:
        assert [x.name for x in h] == ["PRIMARY", "AIPS FQ", "AIPS AN",
                                       "AIPS FG"]
        fg = h["AIPS FG"].data
        # The columns AIPS 31DEC24's UVFLG writes.
        assert list(fg.columns.names) == [
            "SOURCE", "SUBARRAY", "FREQ ID", "ANTS", "TIME RANGE", "IFS",
            "CHANS", "PFLAGS", "REASON"]
        assert sorted(map(tuple, fg["ANTS"])) == [(1, 3), (2, 0)]
        st = fg[[tuple(a) == (2, 0) for a in fg["ANTS"]]][0]
        assert tuple(st["IFS"]) == (1, NIF) and tuple(st["CHANS"]) == (1, 0)
        assert st["PFLAGS"][:2].all()
        # Days from the file's reference date, bracketing the samples.
        day = obs._core.ref_mjd + 2400000.5 - REF_JD
        lo, hi = st["TIME RANGE"]
        assert day + t[2] / 86400 < lo < day + t[3] / 86400
        assert day + t[9] / 86400 < hi < day + t[10] / 86400


def test_save_writes_the_flags_in_the_native_format(obs, tmp_path):
    t = _times(obs)
    obs.flag(station="AN4", tmin=t[1], tmax=t[2])
    prefix = str(tmp_path / "sess")
    files = obs.save(prefix, quiet=True)
    assert files["flags"] == f"{prefix}.FG.TASAV.FITS"
    assert os.path.isfile(files["flags"])
    assert "flags" not in obs.save(prefix, quiet=True, flags=False)
    both = obs.wflags(str(tmp_path / "x.flagcmd"), outformat="both", quiet=True)
    text = open(both["casa"]["path"]).read()
    assert "antenna='AN4'" in text and "mode='manual'" in text
    assert both["aips"]["path"].endswith("x.flagcmd.FG.TASAV.FITS")
