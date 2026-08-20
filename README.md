# difmapy

A modern reimplementation of [Difmap](ftp://ftp.astro.caltech.edu/pub/difmap/difmap.html)
(Shepherd 1997) for interactive VLBI imaging and self-calibration:
a **Rust** compute engine with a **Python** API and **pyqtgraph**
interactive plots.

Where the original paged one IF at a time through scratch files
(`uvdata.scr`, `ifdata.scr`, `modvis.scr`), difmapy keeps **everything
in RAM**. Visibilities and weights are never modified: flags live in a
separate FLAG array and calibrations in separate gain/baseline tables,
composed on the fly when the working data stream is built. Every
operation — selection, gridding, CLEAN, self-cal, flagging — therefore
runs in real time and is fully reversible.

Differences from the original by design:

* **`select("I")` is permissive**: total intensity is formed from
  whichever parallel hands are usable, so data survive when one hand is
  missing or flagged (difmap called this `pi`, which remains as a legacy
  alias). With both hands present and equal weights this is exactly
  (RR+LL)/2, so the flux scale is unchanged; where only one hand
  survives the other is assumed identical, which neglects circular
  polarization. Output images declare Stokes I. `Q`, `U` and `V` remain
  strict, since a missing hand cannot be guessed for polarization.
* reads **UVFITS and Measurement Sets** (both single-source)
* **flags are stored in an explicit FLAG column**, the MS convention,
  and can be written straight back into the MS with `save_flags()`
  instead of having to write out a new UV file
* native **multi-IF / multi-channel** handling: all subbands (IFs /
  SPWs) are gridded together (multi-frequency synthesis), and each may
  have a different number of channels
* Python API instead of the sphere command language; interactive plots
  use pyqtgraph instead of PGPLOT

The numerics are faithful ports of the difmap algorithms
(uvinvert/uvtrans gridding+FFT, Högbom mapclean, mapres restore,
modvis models, slfcal self-calibration, modfit/lmfit model fitting).

## Install

Building needs a Rust toolchain (https://rustup.rs); using difmapy does
not.

```sh
pip install maturin
maturin develop --release          # builds the Rust core into your env
pip install ".[plot]"              # pyqtgraph + PySide6 for plots
pip install ".[ms]"                # casatools for Measurement Sets
```

See `INSTALL.md` for building redistributable wheels and for publishing
to PyPI - and read `NOTICE.md` first: difmapy is a close port of Difmap,
whose licence terms must be settled before any public release.

## Usage

### The `difmapy` command

```sh
difmapy                                   # empty interactive session
difmapy mysource.uvfits                   # load and select Stokes I
difmapy data.ms --stokes I --mapsize 2048 --cell 0.5
difmapy data.ms --channels 0-31           # channel ranges
difmapy data.ms -c "clean(200, 0.03)"     # run commands on startup
difmapy data.ms --batch -c "wmap('m.fits')"   # scripted, no prompt
```

This opens an IPython session with the observation bound to `obs` and
the difmap-style commands available as bare functions, so a session
reads much like difmap:

```
In [1]: invert()
In [2]: add_window(-5, 5, -5, 5)
In [3]: clean(200, 0.03)
In [4]: selfcal(phase=True)
In [5]: radplot()          # opens; the prompt stays usable
In [6]: wmap('clean.fits')
```

The Qt event loop is driven by IPython, so plot windows stay live and
interactive while you keep typing - flagging in a plot updates `obs`
immediately. `load('other.uvfits')` switches dataset and rebinds the
commands. `--batch` makes the same commands usable from shell scripts.

### As a library

```python
import difmapy

obs = difmapy.load("mysource.uvfits")  # or a .ms directory
print(obs.header())

obs.select("I")                        # Stokes I, all channels
# obs.select("I", channels=[(0, 31)])  # or channel ranges (global axis)
# obs.select("RR") / "LL" / "Q" / "U" / "V" / "XX" ... also available

obs.mapsize(2048, 0.5)                 # pixels (power of 2), mas/pixel
obs.uvweight(0, -1)                    # natural weighting w/ data weights
                                       # (default is difmap's uniform 2, 0)
obs.startmod(flux=1.0)                 # phase selfcal to a point source

for _ in range(4):                     # the classic difmap loop
    obs.clean(200, 0.03)
    obs.selfcal(phase=True)
obs.selfcal(amp=True, phase=True, solint=30)
obs.clean(400, 0.02)

obs.mapplot()                          # interactive: double-click adds
                                       # CLEAN windows, c=clean, i=invert
obs.radplot()                          # amp vs uv-radius; Shift+drag to flag
obs.cpplot()                           # closure phases vs the model
obs.corplot()                          # self-cal gain solutions

m = obs.restore()                      # restored map (numpy array)
obs.wmap("clean.fits")                 # FITS output with WCS + beam
obs.save("mysession")                  # .uvf + .mod + .win + parameters
# later: obs = difmapy.Observation.get("mysession")
```

### Flagging

Flags are held in an explicit FLAG array and every edit is reversible:

```python
obs.flag(station="EF", if_index=2)                 # all EF baselines, IF 3
obs.flag(baseline=("EF", "JB"), tmin=0, tmax=3600)
obs.unflag(station="EF", if_index=2)
print(obs.flagged_fraction)

obs.save_flags()        # write FLAG (+FLAG_ROW) back into the source MS
```

Interactive flagging (`radplot`, `uvplot`, `vplot`): **Shift+drag**
sweeps a box to flag, **Ctrl+drag** unflags, `f`/`u` act on the nearest
point, `x` toggles display of flagged points. Plain drag/wheel keep
pyqtgraph's pan/zoom.

Data that are absent from the file (zero weight) count as permanently
flagged and cannot be unflagged, matching difmap's deleted-data flag.

### Two things worth knowing

**Only the inner quarter of a map is valid.** As in difmap, the grid is
padded: outside the central quarter the gridding correction amplifies
pixels without bound, so values there are meaningless (a dirty beam
can even exceed 1 out there). `clean` and `imstat` already restrict
themselves to that area; when working with the arrays directly use
`obs.valid()`, `obs.peak_offset()` or `obs.valid_slice` rather than
scanning the whole image.

**Weighting defaults follow difmap**, i.e. uniform with a 2-pixel bin
and no amplitude-error weighting (`invdef` in difmap.c), so a first
image reproduces what difmap would give. Use `obs.uvweight(0, -1)` for
natural weighting that uses the data weights - it roughly doubles the
beam size on typical VLBI data.

### What gets written out

`wobs` applies the antenna gains and baseline corrections to the data
it writes. Following difmap, accumulated `shift`s are *not* frozen in
unless you ask (`wobs(..., freeze_shift=True)`); `save()` records the
shift in its parameter file and `get()` re-applies it, so a saved
session round-trips exactly.

### Model fitting

```python
obs.addcmp(1.5, 2.0, -1.0, type="gauss", major=3.0, ratio=0.7, phi=30,
           free=["flux", "pos", "shape"])
res = obs.modelfit(niter=50)
print(res["rchisq"], obs.model, res["errors"])
```

Like difmap's, this is a local optimizer: it converges from a sensible
starting guess but can settle in a local minimum from a far-off one, so
start from something like the map peak. The reported `errors` are
first-order estimates from the inverse Hessian and ignore parameter
covariances.

### Other commands

`shift`/`unshift`, `resoff`/`clroff`, `uvaver`, `uvtaper`, `uvrange`,
`uvzero`, `gscale`, `uncalib`, `selfant`, `keep`, `clrmod`,
`wmodel`/`rmodel`, `wwins`/`rwins`, `wobs`, `wdmap`, `wbeam`,
`imstat`, `spectrum`, `closure_phases`, `specplot`, `tplot`.

## Layout

```
crates/difmap-core   pure-Rust engine (no Python): data model, stream
                     selection, gridding/FFT, clean, restore, selfcal,
                     modelfit, models, closure, editing, geometry
crates/difmap-py     PyO3 bindings (module difmapy._core)
python/difmapy       Python API, UVFITS/MS I/O, pyqtgraph plots
tests/               pytest suite: synthetic data with analytic truth
                     plus real EVN data (3C345, UVFITS and MS)
difmap-master/       reference: original difmap C sources (not built)
```

## Validation

The test suite checks against analytic ground truth and real data:

* the **same EVN observation in UVFITS and MS** loads to identical
  visibilities, weights, flags and uvw, and images identically
* a **point source** inverts to the right pixel with the right flux;
  CLEAN recovers its flux; restore reproduces the peak
* **self-cal** recovers injected antenna gain errors to <2%
* **closure phases** change by <1e-4 degrees under full amplitude+phase
  self-calibration (they are gain-invariant by construction), and a
  full clean/selfcal session ends with a model reproducing them to a
  few degrees on high-SNR triangles
* **modelfit** recovers all six parameters of an elliptical gaussian
  injected into the real uv coverage (reduced chi-squared ~1e-11)
* **shift** translates the map rigidly by exactly the requested pixels
* **MS flag write-back** is bit-exact and provably leaves DATA and
  WEIGHT untouched

## Performance

Release build, 16.6M visibilities (10 antennas, 12 h, 8 IFs x 32
channels x 2 pols, 1024² maps), best of three runs on one desktop:

| operation | time |
|---|---|
| select (average 16.6M vis to 259k) | 8 ms |
| invert 1024² | 40 ms |
| clean 100 iterations | 43 ms |
| selfcal (phase, per integration) | 15 ms |
| flag/unflag a station | 13 ms |
| restore | 39 ms |

Bulk operations are slower but one-shot: loading the data and
`uvaver` are dominated by moving the full cube (~0.9 s each here).

## Testing

```sh
cargo test                 # Rust core tests
maturin develop && pytest  # Python end-to-end tests
```
