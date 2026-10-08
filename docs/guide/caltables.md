# Calibration tables for CASA and AIPS

difmapy keeps calibration out of the data: the corrections built up by
`selfcal` and `gscale` live in a separate gain table. That is the same
separation CASA and AIPS make, so the gains can be handed over as a
calibration table and applied there - to the full-resolution data, or
to the other sources of the observation.

```python
obs.savecaltable("mysource.G")                       # CASA, for an MS
obs.savecaltable("mysource.TASAV.FITS")              # AIPS, for UVFITS data
obs.savecaltable("mysource", outformat="both", ms="mysource.ms")
```

`outformat` is `"CASA"`, `"AIPS"` or `"both"` (case does not matter).
By default it is the one that matches the data: CASA for a Measurement
Set, AIPS for UVFITS - or, if you pass only `ms=` or only `uvfits=`, the
one that file is for. With `"both"`, `path` names the CASA table and
the AIPS file is `<path>.TASAV.FITS`.

The table is a snapshot of everything accumulated so far. To keep one
table per self-calibration round, mark a point and export the increment;
applying the chain is equivalent to applying the cumulative table:

```python
mark = obs.gain_snapshot()
obs.selfcal(amp=True, phase=True)
obs.savecaltable("round2.G", since=mark)
```

## CASA

```python
# in CASA, on the same or another MS with the same stations:
applycal(vis="other.ms", gaintable=["mysource.G"], interp=["nearest"])
```

A "G Jones" table. A Measurement Set is needed for the antenna, spectral
window and field metadata the table refers to; it defaults to the one
loaded, and `ms=` accepts another. Antennas are matched by **name**, so
gains derived from an averaged dataset can be applied to the
full-resolution MS.

Exporting from UVFITS-loaded data works with `ms=`, but timestamps in a
UVFITS file can differ from the MS's by a fraction of an integration, so
prefer exporting from the MS-loaded observation.

## AIPS

```
FITLD  the TASAV file                      -> MYSRC.TASAV.1
TACOP  inext 'SN' from it onto the UV data
single-source data: DOCAL 1, GAINUSE <that SN version> (SPLIT, IMAGR...)
multi-source data:  CLCAL first, to turn it into a new CL table
```

An SN table in a TASAV FITS file, laid out as AIPS's own TASAV and FITTP
write it: one dummy visibility plus the FQ, AN and SN tables.

An SN table refers to AIPS **station numbers** and counts time in days
from the file's reference date, so both are taken from the UV file the
table is meant for: the one loaded, or another given as `uvfits=`
(antennas matched by name). For data loaded from a Measurement Set,
antennas are numbered in ANTENNA-table order from 1, as CASA's
`exportuvfits` does. `SOURCE ID` is 0, so the table applies to every
source once copied onto a multi-source file.

## Conventions

difmapy stores the *correction* `c` that multiplies the data, for a
baseline between stations p and q:

```
V_corrected = V · c_p · conj(c_q)
```

| package | applies | so the table holds |
|---|---|---|
| CASA | `V / (G_p · conj(G_q))` | `G = 1/c` |
| AIPS | `V · conj(g_p) · g_q` | `g = conj(c)` |

That is: a CASA table holds the reciprocal of difmapy's corrections, and
an AIPS table the same amplitude with the phase negated.

Both are verified in the test suite against the packages themselves:
CASA's `applycal`, and AIPS's `SPLIT` with `DOCAL 1`, reproduce
difmapy's corrected visibilities to float32 precision.

## Limits

- The gains are polarization-independent - they are solved on the
  total-intensity data - and are written to both parallel hands.
- `resoff` baseline corrections and `shift` cannot be expressed in an
  antenna-based table. `savecaltable` reports them rather than dropping
  them silently; use `wobs()` to write data with everything applied.
- The CLCAL route on a multi-source AIPS file has not been tested.

[`save()`](saving.md) writes the calibration table automatically, and
[`bayes_gscale`](bayes-gscale.md) writes one of its constant
corrections.
