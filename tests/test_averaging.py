"""Frequency averaging (`chanaver`, `load(freqavg=...)`), the time
averaging options of `load`, and `Observation.copy`."""

import numpy as np
import pytest

import difmapy
from conftest import CH_WIDTH, IF_OFFSET, NCHAN, NIF, REF_FREQ


def _stream(o):
    vis, wt = o._core.stream_vis()
    return np.asarray(vis), np.asarray(wt)


def test_freqavg_on_load_halves_the_channels(uvfits_file):
    o = difmapy.load(uvfits_file, freqavg=2)
    assert o.nchan == [NCHAN // 2] * NIF
    for cif, (freq, df, nchan) in enumerate(o._core.ifs):
        f0 = REF_FREQ + IF_OFFSET * cif
        # The first output channel is centred between the two it averages.
        assert freq == pytest.approx(f0 + 0.5 * CH_WIDTH)
        assert df == pytest.approx(2 * CH_WIDTH)


def test_averaged_channels_give_the_same_stream(uvfits_file):
    """The stream averages the selected channels of each IF anyway, so
    averaging them first in equal-weight groups changes nothing."""
    full = _stream(difmapy.load(uvfits_file))
    for n in (2, "all", True):
        vis, wt = _stream(difmapy.load(uvfits_file, freqavg=n))
        np.testing.assert_allclose(vis, full[0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(wt, full[1], rtol=1e-5)


def test_freqavg_only_averages_the_requested_channels(uvfits_file):
    """`channels` with `freqavg` drops the others from the averages - the
    way band edges are left out - and the result is what selecting those
    channels would give."""
    chans = [(1, 3), (NCHAN + 1, NCHAN + 3)]
    ref = _stream(difmapy.load(uvfits_file, channels=chans))
    avg = difmapy.load(uvfits_file, channels=chans, freqavg="all")
    assert avg.nchan == [1] * NIF
    vis, wt = _stream(avg)
    np.testing.assert_allclose(vis, ref[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(wt, ref[1], rtol=1e-5)


def test_freqavg_must_divide_the_channels(uvfits_file):
    with pytest.raises(ValueError, match="equally"):
        difmapy.load(uvfits_file, freqavg=3)


def test_flag_state_survives_channel_averaging(uvfits_file):
    o = difmapy.load(uvfits_file, stokes=None)
    flags = np.array(o.flags, copy=True)
    flags[0, 0, :] = True            # one of the two channels of bin 0
    flags[1, 0:2, :] = True          # both channels of bin 0
    o._core.set_flags(np.ascontiguousarray(flags))
    a = o.chanaver(2)
    af = np.asarray(a.flags)
    assert not af[0, 0].any(), "one good input keeps the sample good"
    assert af[1, 0].all(), "no good input flags it"
    # ...but it is flagged, not deleted, so unflag brings it back.
    a.unflag()
    assert not np.asarray(a.flags)[1, 0].any()


def test_timeavg_and_its_old_name(uvfits_file):
    a = difmapy.load(uvfits_file, timeavg="2min")
    b = difmapy.load(uvfits_file, average=120)
    assert a._core.nrow == b._core.nrow < difmapy.load(uvfits_file)._core.nrow
    with pytest.raises(ValueError, match="timeavg"):
        difmapy.load(uvfits_file, timeavg=60, average=120)


def test_time_and_frequency_averaging_together(uvfits_file):
    o = difmapy.load(uvfits_file, timeavg="2min", freqavg="all")
    assert o.nchan == [1] * NIF
    assert o._core.nrow < difmapy.load(uvfits_file)._core.nrow
    assert o._core.selection()["stokes"] == "I"
    # Calibration provenance survives, so tables can still be written.
    assert o._core._cal_origin["format"] == "uvfits"


def test_copy_is_independent(uvfits_file):
    o = difmapy.load(uvfits_file)
    o.mapsize(256, 0.25)
    o.addcmp(1.0, 0.0, 0.0)
    c = o.copy()
    c.flag(station=c.antennas[0])
    c.clrmod()
    c.mapsize(128, 1.0)
    c._core.apply_gain_factors(np.full((o.nif, len(o.antennas)), 2.0, np.float32))
    assert o.flagged_fraction == 0.0 < c.flagged_fraction
    assert len(o.model) == 1 and not c.model
    assert o._nx == 256
    amp, _, _ = o._core.gains()
    assert np.allclose(amp, 1.0)
    assert c._core._cal_origin == o._core._cal_origin


def test_apply_gain_factors_scales_the_data(uvfits_file):
    o = difmapy.load(uvfits_file)
    nant = len(o.antennas)
    before, _ = _stream(o)
    fac = np.ones((o.nif, nant), np.float32)
    fac[:, 1] = 1.5
    o._core.apply_gain_factors(fac)
    after, _ = _stream(o)
    _, a1, a2, *_ = o._core.rows()
    has1 = (np.asarray(a1) == 1) | (np.asarray(a2) == 1)
    np.testing.assert_allclose(np.abs(after[has1]), 1.5 * np.abs(before[has1]),
                               rtol=1e-5)
    np.testing.assert_allclose(after[~has1], before[~has1], rtol=1e-6)
    with pytest.raises(ValueError):
        o._core.apply_gain_factors(np.zeros((o.nif, nant), np.float32))


@pytest.mark.parametrize("fn", [
    difmapy.load, difmapy.observe, difmapy.Observation.from_uvfits,
    difmapy.Observation.from_ms, difmapy.uvaver, difmapy.chanaver,
    difmapy.Observation.uvaver, difmapy.Observation.chanaver,
], ids=lambda f: f.__qualname__)
def test_every_loader_parameter_is_documented(fn):
    """help() on a loader must list everything that can be passed."""
    import inspect

    doc = fn.__doc__
    assert "{" not in doc, "an unfilled placeholder"
    for name in inspect.signature(fn).parameters:
        if name in ("self", "cls"):
            continue
        assert f"{name} :" in doc, f"{fn.__qualname__}: {name} undocumented"


def test_ms_only_options_are_refused_for_uvfits(uvfits_file):
    with pytest.raises(ValueError, match="Measurement Sets only"):
        difmapy.load(uvfits_file, field=0)
    with pytest.raises(ValueError, match="Measurement Sets only"):
        difmapy.load(uvfits_file, data_column="CORRECTED_DATA")
    o = difmapy.load(uvfits_file, wtscale=2.0, stokes=None)
    ref = difmapy.load(uvfits_file, stokes=None)
    np.testing.assert_allclose(np.asarray(o._core.calibrated_cube()[1]),
                               2 * np.asarray(ref._core.calibrated_cube()[1]))


def test_loaders_document_their_shared_parameters_identically():
    """The parameter text is written out in each loader's docstring (so
    that tools reading the source, like the API reference, see it); the
    copies must not drift apart."""
    import inspect

    def entries(fn):
        doc = inspect.cleandoc(fn.__doc__)
        body = doc[doc.index("Parameters"):doc.index("Returns")]
        out, name = {}, None
        for line in body.splitlines()[2:]:
            if line and not line.startswith(" "):
                name = line.split(" :")[0]
                out[name] = [line]
            elif name:
                out[name].append(line)
        return {k: "\n".join(v).rstrip() for k, v in out.items()}

    ref = entries(difmapy.load)
    for fn in (difmapy.Observation.from_ms, difmapy.Observation.from_uvfits):
        for name, text in entries(fn).items():
            if name != "path":
                assert text == ref[name], f"{fn.__qualname__}: {name}"
