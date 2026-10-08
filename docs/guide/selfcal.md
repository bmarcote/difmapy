# Self-calibration

Calibration never touches the data. The corrections live in a separate
gain table and are applied on the fly, so they can always be undone
(`obs.uncalib()`), inspected (`obs.corplot()`, `obs.station_gains()`) and
[exported](caltables.md).

## A starting point

```python
obs.startmod(flux=1.0)          # phase self-cal against a point source, then drop it
obs.startmod("previous.mod")    # ...or against a model file
```

## Phase and amplitude

```python
obs.selfcal(phase=True)                          # one solution per integration
obs.selfcal(phase=True, solint="30s")
obs.selfcal(amp=True, phase=True, solint=30)     # a bare number is minutes
obs.selfcal(amp=True, phase=True, solint="scan")
```

Each IF of each subarray is solved independently, against the current
model. `selfcal` reports, as Difmap does, how well the model fits the
data before and after the solution - and the residual-map statistics
too, when a map already exists:

```python
sc = obs.selfcal(phase=True)
sc["fit_before"]["rms"], sc["fit_after"]["rms"], sc["fit_after"]["sigma"]
sc["map_before"]["max"], sc["map_after"]["rms"]
```

`obs.moddif()` returns the same fit numbers on its own. The map
statistics cost nothing in an imaging loop, because self-calibration
invalidates the map and the next `clean` would re-invert anyway;
`mapstats=False` skips them.

!!! warning "Self-calibrate against a model"

    `selfcal` and `gscale` with an empty model produce meaningless
    gains. Build a model first (`startmod`, `clean`, `modelfit`).

## Solution intervals

`solint` is a duration - a bare number in **minutes**, as in Difmap, or
a string with its unit - or a number of scans:

```python
obs.selfcal(phase=True, solint="1min")
obs.selfcal(phase=True, solint="scan")      # exactly one solution per scan
obs.selfcal(phase=True, solint="2scan")     # one per two scans
```

`"scan"` bins by whole scans rather than by a duration, so each scan
gets one and only one solution however long it is. A fixed interval
cannot guarantee that: its bins are aligned to the clock and straddle
the gaps.

Two integrations more than a gap apart belong to different scans. The
default gap is five times the median integration spacing, and
`scangap=` overrides it:

```python
obs.scans()                 # [{'first': ..., 'tmin': ..., 'nint': ...}, ...]
obs.default_scangap         # the gap scans() used, in seconds
obs.selfcal(phase=True, solint="scan", scangap="4min")
```

`selfcal` reports the interval it used and how many solution bins came
out, so a per-scan solution can be checked at a glance:

```
selfcal: phase per scan; 8 solution bins, 0 unusable, 0 bad telescope solutions
```

### How solutions are applied

With a finite interval the solutions are not applied as steps. As in
Difmap, each bin's solution is smoothed and interpolated onto the
integrations with a Gaussian of σ = 0.375 × `solint`, truncated at
2.5σ, so `corplot` shows the corrections evolving smoothly rather than
jumping from one bin to the next.

Per-integration solving has nothing to interpolate, `gscale` is one
solution by definition, and a scan-based interval is a step per scan by
construction - blending across a slew would undo the point of solving
per scan. Where a station's data gap is wider than the interpolation can
reach, those integrations get no solution and are left out of the plot.

## Amplitude scale per station

```python
g = obs.gscale()
g["gains"]["EF"], g["gains_per_if"]["EF"]
obs.station_gains()             # everything accumulated so far
```

`gscale` solves one amplitude correction per station and IF for the
whole observation, and applies it.

!!! note "It can look like nothing happened"

    With the default `float_scale=False` (Difmap's behaviour) the gains
    are renormalised to preserve the *data's* flux scale rather than
    pull it onto the model. The corrections can be large while the map
    peak barely moves; the `fit_before`/`fit_after` statistics are what
    show that they were applied.

For amplitude corrections with error bars, and protection against a
station that has biased the model, see
[Bayesian amplitude calibration](bayes-gscale.md).

## Constraining stations

```python
obs.selfant()                        # list every antenna's constraints
obs.selfant("EF", fix=True)          # hold EF's gain at 1
obs.selfant("all", weight=1.0)       # or "*": the whole array
```

A fixed antenna keeps its gain while the others solve against it;
`weight` scales how much its baselines count.

## Setting a station aside

```python
obs.ignore("ef")                  # case-insensitive; several names allowed
obs.ignored                       # ['EF']
...                               # image, self-cal, fit, plot without it
obs.unignore()                    # or unignore("EF"); no argument = all
```

`ignore` flags every baseline of those stations, so imaging, model
fitting, self-calibration and the plots all skip them. Unlike `flag`, it
remembers their exact flag state first, and `unignore` puts that back
rather than unflagging wholesale - the station returns with its own
history intact, and an `unflag` in the meantime does not resurrect it.

Two consequences: flag edits made to an ignored station's own baselines
while it is away are discarded by `unignore`; and self-calibration
solved while a station is ignored has no solution for it, so it returns
uncalibrated for those intervals. Self-calibrate again, or `selfant` to
hold it fixed.

## Baseline-based corrections

`obs.resoff()` solves per-baseline amplitude and phase offsets against
the model, to absorb errors that do not close; `obs.clroff()` removes
them. They cannot be written to an antenna-based calibration table.
