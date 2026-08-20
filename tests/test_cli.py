"""The `difmapy` command: argument handling, batch mode and the
difmap-style command namespace."""

import os

import pytest

from difmapy import cli

HERE = os.path.dirname(__file__)
UVF = os.path.join(HERE, "rsm07_3C345.uvfits")
real_data = pytest.mark.skipif(
    not os.path.exists(UVF), reason="real 3C345 data not present"
)


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert "difmapy" in capsys.readouterr().out


def test_missing_file_is_an_error(capsys, tmp_path):
    rc = cli.main([str(tmp_path / "nope.uvfits")])
    assert rc == 2
    assert "no such file" in capsys.readouterr().err


def test_bad_file_reports_cleanly(capsys, tmp_path):
    junk = tmp_path / "junk.uvfits"
    junk.write_text("not a FITS file")
    rc = cli.main([str(junk)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "could not load" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "spec,expect",
    [
        ("0-31", [(0, 31)]),
        ("0:31", [(0, 31)]),
        ("7", [(7, 7)]),
        ("0-3,8-11", [(0, 3), (8, 11)]),
        (None, None),
    ],
)
def test_channel_parsing(spec, expect):
    assert cli._parse_channels(spec) == expect


def test_namespace_without_data():
    ns = cli.make_namespace(None)
    assert ns["obs"] is None
    assert callable(ns["load"])
    assert "difmapy" in ns and "np" in ns
    # No bound commands until something is loaded.
    assert "clean" not in ns
    assert "No data loaded" in cli.banner(None)


@real_data
def test_batch_runs_commands(capsys, tmp_path):
    out = tmp_path / "cli.fits"
    rc = cli.main([
        UVF, "--mapsize", "256", "--cell", "2", "--uvweight", "0", "-1",
        "--batch",
        "-c", "invert()",
        "-c", "add_window(-100, 100, -100, 100)",
        "-c", "r = clean(200, 0.03)",
        "-c", "print('CLEANED %.3f' % r['cleaned_flux'])",
        "-c", f"wmap({str(out)!r})",
    ])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "3C345" in printed          # the header summary
    assert "Selection: I" in printed   # selection applied before printing
    flux = float(printed.split("CLEANED ")[1].split()[0])
    assert flux > 0.5
    assert out.exists()


@real_data
def test_batch_reports_errors(capsys):
    rc = cli.main([UVF, "--batch", "-c", "this_is_not_defined()"])
    assert rc == 1
    assert "error in -c" in capsys.readouterr().err


@real_data
def test_startup_options_are_applied(capsys):
    # --stokes pi is a legacy spelling and must resolve to I.
    rc = cli.main([UVF, "--stokes", "pi", "--channels", "0-1",
                   "--mapsize", "512", "--cell", "1.5", "--batch",
                   "-c", "print('SEL', obs._core.selection()['stokes'])",
                   "-c", "print('CH', obs._core.selection()['chlist'])",
                   "-c", "print('NX', obs._nx, 'CELL', obs._xinc)"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SEL I" in out
    assert "CH [(0, 1)]" in out
    assert "NX 512" in out


@real_data
def test_no_selection_when_requested(capsys):
    rc = cli.main([UVF, "--stokes", "none", "--batch",
                   "-c", "print('HAS', obs is not None)"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "HAS True" in out
    assert "Selection: none" in out


@real_data
def test_bound_commands_act_on_the_observation():
    """The bare commands must be the loaded observation's own methods."""
    obs = cli._load(UVF, "I", None, 256, 2.0, (0, -1))
    ns = cli.make_namespace(obs)
    assert ns["obs"] is obs
    for name in ("select", "invert", "clean", "selfcal", "mapplot", "wmap"):
        assert ns[name].__self__ is obs, name
    # Calling a bare command changes the observation.
    ns["invert"]()
    assert obs._invert_result is not None
    assert "3C345" in cli.banner(obs)
    assert "I selected" in cli.banner(obs)


@real_data
def test_load_helper_rebinds_commands():
    ns = cli.make_namespace(None)
    obs = ns["load"](UVF)
    assert ns["obs"] is obs
    assert ns["invert"].__self__ is obs
