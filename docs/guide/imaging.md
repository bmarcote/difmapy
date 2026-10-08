# Imaging

## Map size and cell

```python
obs.mapsize(2048, 0.5)          # pixels, mas per pixel
obs.auto_mapsize()              # 4096 pixels of a tenth of the resolution
obs.estimated_resolution()      # lambda / B_max, in mas
```

The number of pixels must be a multiple of four. Powers of two (or
products of small primes) are the fastest, but are no longer required.

!!! warning "Only the inner quarter of a map is valid"

    As in Difmap, the grid is padded: outside the central quarter the
    gridding correction amplifies pixels without bound, so values there
    are meaningless - a dirty beam can even exceed 1 out there. `clean`
    and `imstat` already restrict themselves to that area. When working
    with the arrays yourself, use `obs.valid()`, `obs.peak_offset()` or
    `obs.valid_slice` rather than scanning the whole image, and choose a
    map size that puts your source inside the inner quarter.

## Weighting

```python
obs.uvweight(2, 0)              # Difmap's default: uniform, 2-pixel bins
obs.uvweight(0, -1)             # natural, using the data weights
obs.uvweight(robust=0)          # Briggs robustness
```

The defaults are Difmap's - uniform weighting with a 2-pixel bin and no
amplitude-error weighting - so a first image reproduces what Difmap
would give. Natural weighting roughly doubles the beam on typical VLBI
data.

**Briggs robust weighting** is one number from -2 (uniform, sharpest
beam) to +2 (natural, lowest noise). It supersedes `binwid`/`errpow`
while set; `uvweight(robust=None)` goes back to Difmap's scheme. The two
ends reproduce Difmap's own uniform and natural weighting. The
"Weighting" box in the [map window](mapplot.md) is the same control.

`uvtaper`, `uvrange` and `uvzero` work as in Difmap.

## Invert, CLEAN, restore

```python
obs.invert()                              # dirty map and beam
obs.add_window(-5, 5, -5, 5)              # a CLEAN window, in mas
res = obs.clean(200, 0.03)                # iterations, loop gain
m = obs.restore()                         # the restored map, a numpy array
```

`clean` inverts first if it has to. A negative number of iterations
stops at the first negative component, and `cutoff=` stops at a residual
level. Windows can also be drawn in the map window, and read and written
as Difmap `.win` files (`rwins`, `wwins`).

```python
obs.dmap, obs.dbeam             # residual map and dirty beam
obs.imstat()                    # peak, minimum and rms of the valid area
obs.noise_stats()               # robust (sigma-clipped) noise of the residuals
obs.mapinfo()                   # beam, peak, model, total flux, noise
obs.estimated_beam              # (major, minor) in mas and position angle
```

## Output

```python
obs.wmap("clean.fits")          # restored map, with WCS and beam
obs.wdmap("dirty.fits")         # residual map
obs.wbeam("beam.fits")          # dirty beam
obs.wmodel("clean.mod")         # the model, as a Difmap .mod file
```

## The numbers each command returns

`clean`, `modelfit`, `selfcal` and `gscale` print a short report and
return it as a dict, so a scripted run can keep the numbers:

```python
res = obs.clean(200, 0.03)
res["ncomp"], res["cleaned_flux"], res["total_flux"], res["residual_rms"]
```

Every command that reports numbers also takes `outfile=`, which writes
the same result as JSON:

```python
obs.clean(200, 0.03, outfile="clean.json")
obs.mapinfo(outfile="image.json")
```

`invert`, `clean`, `modelfit`, `selfcal`, `gscale`, `imstat`,
`noise_stats`, `moddif`, `mapinfo`, `station_gains`,
`baseline_corrections`, `spectrum`, `closure_phases` and `scans` all
accept it. Arrays become lists, and NaN - which JSON cannot write, and
which means "no solution" here - becomes `null`. `quiet=True` prints
nothing.

## Shifting the phase centre

```python
obs.shift(3.0, -2.0)            # east, north, in mas
obs.unshift()
obs.total_shift
```

Shifts accumulate and are reversible. As in Difmap they are not frozen
into the UV data that `wobs` writes unless you ask for it - see
[Saving a session](saving.md).
