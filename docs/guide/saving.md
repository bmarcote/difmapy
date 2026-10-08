# Saving a session

```python
files = obs.save("mysession")
obs = difmapy.Observation.get("mysession")      # later
```

`save` keeps the whole session under one prefix, and returns (and
prints) the files it wrote.

| file | written | contents |
|---|---|---|
| `mysession.uvf` | always | the calibrated UV data, as UVFITS |
| `mysession.mod` | always | the model |
| `mysession.win` | always | the CLEAN windows |
| `mysession.par.json` | always | selection and imaging parameters |
| `mysession.ms` | for data from a Measurement Set | see below |
| `mysession.G` or `.TASAV.FITS` | once self-calibration has been applied | the [calibration table](caltables.md), in the data's own format |
| `mysession.flagcmd` or `.FG.TASAV.FITS` | once flags have been added | the [flag table](flagging.md#export-them-as-a-flag-table), in the data's own format |
| `mysession.fits` | once there is a model | the restored CLEAN map |

"The data's own format" is CASA for a Measurement Set and AIPS for
UVFITS.

`ms=`, `caltable=`, `flags=` and `image=` each take `False` to skip that
file, or `True` to make its absence an error instead of leaving it out.

`get()` restores the UV data, selection, imaging setup, model and
windows from the first four files.

## The Measurement Set

A session that started from a Measurement Set gets one back.

**Unaveraged data** go into a copy of the original MS: the calibrated
visibilities in `CORRECTED_DATA` and the current flags in `FLAG`.
Everything else - metadata, subtables, the untouched `DATA` column - is
preserved exactly. The original MS is never modified.

**Averaged data** - in time or frequency, on load or later - get a *new*
MS, built on the original's structure and subtables, with a main table
of the averaged rows:

- `DATA` holds the averaged data as loaded, `CORRECTED_DATA` the same
  with the session's calibration applied;
- flags, weights, times, integration times and UVW are those of the
  session; scan numbers are those of the original rows nearest in time;
- after channel averaging, the spectral windows are rewritten with the
  new channel frequencies and widths.

Rows that hold no data at all are not written, and autocorrelations and
`MODEL_DATA` are not carried over.

## Writing the UV data on its own

```python
obs.wobs("calibrated.uvfits")
obs.wobs("shifted.uvfits", freeze_shift=True)
```

`wobs` applies the antenna gains and baseline corrections to the data it
writes. Following Difmap, accumulated `shift`s are *not* frozen in
unless you ask; `save()` records the shift in its parameter file and
`get()` re-applies it, so a saved session round-trips exactly.

## Individual files

```python
obs.wmap("clean.fits");  obs.wdmap("dirty.fits");  obs.wbeam("beam.fits")
obs.wmodel("src.mod");   obs.rmodel("src.mod")
obs.wwins("src.win");    obs.rwins("src.win")
obs.savecaltable("src.G")
obs.wflags("src.flagcmd")
obs.save_flags()                  # into the originating Measurement Set
```
