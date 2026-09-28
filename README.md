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
  Intensity is also loaded by default if not specified.
* reads **UVFITS and Measurement Sets** (both single-source)
* **flags are stored in an explicit FLAG column**, the MS convention,
  and can be written straight back into the MS with `save_flags()`
  instead of having to write out a new UV file.
* **Calibration tables can be exported for CASA and AIPS** via `savecaltable()`: a CASA "G Jones" table and/or an AIPS SN table in a TASAV FITS file (by default whichever matches the data you loaded). The table compiles all calibration corrections performed to the data in the Difmapy session, so it can be applied inside CASA or AIPS to other data sets, such as the other sources of the observation.
* **Bayesian station amplitude calibration** with `bayes_gscale()`: `gscale` with each station left out of the source model in turn, several source models compared, and the corrections shrunk towards the a-priori calibration, with a JSON report, a diagnostic figure and a calibration table (see below).
* native **multi-IF / multi-channel** handling: all subbands (IFs /
  SPWs) are gridded together (multi-frequency synthesis), and each may
  have a different number of channels
* **loading selects total intensity**, since that is how nearly every
  session starts (`stokes=None` to skip it)
* **Briggs robust weighting** as a single number from -2 to +2, in
  addition to difmap's own `binwid`/`errpow` scheme
* **Modelfitting done inside mapplot() via 'f'**: If you set a component in the image,
  you can directly fit the component to the uv data by pressing "f". No need to do it in the terminal.
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
pip install .              # pyqtgraph + PySide6 for plots
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
difmapy big.uvfits --timeavg 10s          # time-average on load
difmapy big.ms --freqavg all              # one channel per IF on load
difmapy multi.ms --field 3C345 --data-column CORRECTED_DATA
difmapy data.ms -c "clean(200, 0.03)"     # run commands on startup
difmapy data.ms --batch -c "wmap('m.fits')"   # scripted, no prompt
```

`difmapy --help` lists every option; the loading ones are the parameters
of `difmapy.load`, which `help(difmapy.load)` documents one by one.

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

obs = difmapy.load("mysource.uvfits")  # or a .ms directory; difmapy.observe
obs = difmapy.load("big.uvfits", timeavg="10s") # time-average on load (also difmapy.uvaver)
obs = difmapy.load("big.ms", freqavg=4)        # average 4 channels at a time (also chanaver)
obs = difmapy.load("big.ms", channels=[(2, 29)], freqavg="all")  # drop edges, then average
print(obs.header())                    # is the same function
print(obs.pols)                        # ['RR', 'LL', 'RL', 'LR']

# Loading already selects total intensity; select() changes it at will:
# obs.select("I", channels=[(0, 31)])  # channel ranges (global axis)
# obs.select("RR") / "LL" / "Q" / "U" / "V" / "XX" ... also available
# difmapy.load(path, stokes=None)      # load without selecting anything

obs.mapsize(2048, 0.5)                 # pixels (multiple of 4), mas/pixel
# obs.auto_mapsize()                   # 4096 pixels of resolution/10
obs.uvweight(robust=0)                 # Briggs: -2 uniform ... +2 natural
                                       # (default is difmap's uniform 2, 0)
obs.startmod(flux=1.0)                 # phase selfcal to a point source

for _ in range(4):                     # the classic difmap loop
    obs.clean(200, 0.03)               # prints and returns a summary dict
    obs.selfcal(phase=True)
obs.gscale()                           # amplitude scale per station
obs.selfcal(amp=True, phase=True, solint="scan")
obs.clean(400, 0.02)

obs.mapplot()                          # interactive; also spelled maplot()
obs.radplot()                          # amp+phase vs uv-radius, model in red
obs.projplot(30)                       # vs uv distance projected at PA 30; < / > turn it
obs.vplot(3)                           # amp+phase vs time, 3 baselines/page
obs.vplot(3, "EF")                     # only EF's baselines (difmap's order)
obs.fplot()                            # amp+phase vs frequency, time-averaged
obs.cpplot()                           # closure phases vs the model
obs.corplot()                          # self-cal gains: amp+phase per antenna

info = obs.mapinfo()                   # beam, peak, model, residual noise
m = obs.restore()                      # restored map (numpy array)
obs.wmap("clean.fits")                 # FITS output with WCS + beam
obs.save("mysession")                  # .uvf + .mod + .win + parameters
                                       # (+ mysession.ms if loaded from an MS)
# later: obs = difmapy.Observation.get("mysession")

obs.clearmodel()                       # drop every component and start over
```

`clean`, `modelfit`, `selfcal` and `gscale` print a short report and
return it as a dict (and will write it to a file, see `outfile=` below),
so a scripted run can keep the numbers:

```python
res = obs.clean(200, 0.03)
res["ncomp"], res["cleaned_flux"], res["total_flux"], res["residual_rms"]

fit = obs.modelfit()          # niter=-1: iterate until it converges
fit["rchisq"], fit["converged"], fit["components"], fit["errors"]

g = obs.gscale()              # {"gains": {"EF": 1.03, ...}, ...}
g["gains"]["EF"], g["gains_per_if"]["EF"]

sc = obs.selfcal(phase=True)  # difmap's "fit before/after self-cal"
sc["fit_before"]["rms"], sc["fit_after"]["rms"], sc["fit_after"]["sigma"]
sc["map_before"]["max"], sc["map_after"]["rms"]   # when a map exists
```

`gscale` reports the same before/after numbers. Note that with its
default `float_scale=False` (difmap's behaviour) the gains are
renormalised to preserve the *data's* flux scale rather than pull it
onto the model, so the corrections can be large while the map peak
barely moves - the fit statistics are what show that they were applied.

Every one of them also takes `outfile=`, which writes the same result
to a JSON file:

```python
obs.clean(200, 0.03, outfile="clean.json")
obs.selfcal(phase=True, outfile="selfcal1.json")
obs.mapinfo(outfile="image.json")
```

`invert`, `clean`, `modelfit`, `selfcal`, `gscale`, `imstat`,
`noise_stats`, `moddif`, `mapinfo`, `station_gains`,
`baseline_corrections`, `spectrum`, `closure_phases` and `scans` all
accept it. Arrays become lists and NaN - which JSON has no syntax for,
and which means "no solution" here - becomes `null`.

`selfcal` measures the model-data fit on both sides of the solution, as
difmap does (`obs.moddif()` on its own returns the same numbers), and
adds the residual-map statistics whenever a map is already there -
which costs nothing in an imaging loop, because self-cal invalidates the
map and the next `clean` would have to re-invert anyway. Pass
`mapstats=False` to skip them or `quiet=True` to print nothing.

With no model at all, `modelfit` seeds itself with a circular Gaussian of
zero width at the peak of the residual map, so `obs.modelfit()` on a
freshly loaded dataset already does something sensible.

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
sweeps a box to flag, **Ctrl+drag** unflags, `f`/`F` act on the nearest
point, **Ctrl+Z** / **Ctrl+Shift+Z** (or Ctrl+Y) undo and redo the last
edit, and `x` shows or hides the flagged points (hidden by default).
Plain drag/wheel keep pyqtgraph's pan/zoom, and `h` shows the full key
legend of whichever plot is in front. In `vplot` the space bar switches
flagging between the displayed baseline and every baseline of its first
antenna.

Plots against time (`vplot`, `tplot`, `corplot`, `cpplot`) cut out any
gap between integrations longer than 10% of the observation, so scans
hours apart sit side by side instead of leaving most of the axis empty.
The cuts are shaded, and the axis still reads real times. Every plot
opens scaled to show all of its data.

Undo is exact: each edit stores the affected rows of the FLAG column
before and after, so it restores what was there even where the edit
overlapped data that was already flagged.

Three keys mean the same thing in **every** plot, as they do in difmap:

| key | action |
| --- | --- |
| `z` | restore the y axis range (difmap's `Z`); phase panels return to +-180 |
| `u` | restore the x axis range (difmap's `U`) |
| `r` | reload the plot from the data, after editing it from the prompt |

`r` matters because a window does not know when the data behind it
changes: flag, calibrate, `ignore` a station or re-image from the
prompt, then press `r` to catch up.

Open windows are held for you, so `mapplot()` at the prompt stays alive
without having to keep the returned object, and they are destroyed when
the session ends. `difmapy.plots.open_windows()` lists them and
`close_all_windows()` shuts them all.

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

**Briggs robust weighting** is available as a single number,
`obs.uvweight(robust=R)` with R from -2 (uniform, sharpest beam) to +2
(natural, lowest noise); it supersedes `binwid`/`errpow` while set, and
`mapplot(uvweight=R)` - or the "Weighting" box at the top of the
mapplot window, which offers difmap's own scheme and R = -2, -1, 0, +1,
+2 - is the same knob. The two ends reproduce difmap's
own uniform and natural weighting.

**Map dimensions no longer have to be powers of two** - any multiple of
four works, since the FFTs handle arbitrary lengths - but powers of two
(or products of small primes) are still much the fastest.

### Setting a station aside

```python
obs.ignore("ef")                  # case-insensitive; several names allowed
obs.ignored                       # ['EF']
...                               # image, self-cal, fit, plot without it
obs.unignore()                    # or unignore("EF"); no argument = all
```

`ignore` flags every baseline of those stations, so imaging, model
fitting, self-calibration and the plots all skip them - but unlike
`flag` it remembers their exact flag state first, and `unignore` puts
that back rather than unflagging wholesale. That is the difference:
everyone else's data can be flagged and calibrated in the meantime, and
the station still comes back with its own history intact. An `unflag`
does not resurrect an ignored station either, whether it comes from the
prompt or from a rubber-band in a plot.

Two consequences worth knowing. Flag edits made to an ignored station's
own baselines while it is away are discarded by `unignore`, since it
restores the remembered state. And self-calibration solved while a
station is ignored has no solution for it, so it returns uncalibrated
for those intervals - self-calibrate again, or `selfant` to hold it
fixed. `selfant()` with no arguments lists every antenna's constraints;
`selfant("all", weight=...)` (or `"*"`) sets them for the whole array,
and `fix`/`weight` left out keep their current values.

### Units, and solutions per scan

Every argument that carries a time takes a string with its unit as well
as a bare number:

```python
obs.selfcal(phase=True, solint="30s")      # seconds
obs.selfcal(phase=True, solint="1min")     # 1 minute
obs.selfcal(amp=True, solint="1.5 hours")
obs.uvaver("2min")                         # averaging interval
obs.flag(station="EF", tmin="1h", tmax="1.5h")
obs.spectrum(tmax="30min")
```

Units are `s`/`sec`/`second(s)`, `m`/`min`/`minute(s)`,
`h`/`hr`/`hour(s)` and `d`/`day(s)`. A **bare number keeps the unit that
argument has always had** - minutes for `selfcal`'s `solint`, as in
difmap, seconds for time ranges and for `uvaver` - so existing scripts
are unaffected; a string is how you ask for something else.

A solution interval also accepts scans:

```python
obs.selfcal(phase=True, solint="scan")     # exactly one solution per scan
obs.selfcal(phase=True, solint="2scan")    # one per two scans
```

`"scan"` bins by whole scans rather than by a duration, so each scan
gets one and only one solution however long it is - which a fixed
interval cannot guarantee, since its bins are aligned to the clock and
straddle the gaps. Scans are difmap's definition: two integrations more
than a gap apart belong to different scans. The default gap is five
times the median integration spacing (difmap's own default is a flat
hour, which suits breaking a plot axis rather than self-calibrating),
and `scangap=` overrides it:

```python
obs.scans()                                # [{'first':…, 'tmin':…, 'nint':…}, …]
obs.default_scangap                        # the gap scans() used, in seconds
obs.selfcal(phase=True, solint="scan", scangap="4min")
```

With a finite interval the solutions are not applied as steps. As
difmap does, each bin's solution is smoothed and interpolated onto the
integrations with a Gaussian of `sigma = 0.375 * solint` truncated at
2.5 sigma, each bin weighted by the area under that Gaussian within it,
so `corplot` shows the corrections evolving smoothly rather than
jumping from one bin to the next. Per-integration solving (`solint=0`)
has nothing to interpolate, `gscale` is one solution by definition, and
a scan-based interval is a step per scan by construction - blending
across a slew gap would undo the point of solving per scan. Where a
station's data gap is wider than the interpolation can reach, those
integrations get no solution at all and are left out of the plot.

`selfcal` reports the interval it used and how many solution bins came
out, so a per-scan solution can be checked at a glance:

```
selfcal: phase per scan; 8 solution bins, 0 unusable, 0 bad telescope solutions
```

### Exporting calibration to CASA and AIPS

difmapy never rewrites an MS's visibilities: `DATA`/`CORRECTED_DATA` are
only ever read, corrections live in a separate gain table in memory, and
the only column written back is `FLAG` (see `save_flags`). That is the
same separation CASA and AIPS make, so the gains can be handed over as a
calibration table. `savecaltable(path, outformat=...)` writes a CASA
table (`"CASA"`), an AIPS SN table in a TASAV FITS file (`"AIPS"`), or
both (`"both"`; case does not matter). By default it writes the one that
matches the data: CASA for a Measurement Set, AIPS for UVFITS (or, if
you pass only `ms=` or only `uvfits=`, the one that file is for). With
`"both"`, `path` names the CASA table and the AIPS file is
`<path>.TASAV.FITS` (or give a `.fits` path, and the CASA table drops
the extension).

```python
obs.selfcal(phase=True)
obs.selfcal(amp=True, phase=True, solint=30)
obs.savecaltable("mysource.G")            # CASA "G Jones" table
```

```python
# in CASA, on the same or another MS with the same stations:
applycal(vis='other.ms', gaintable=['mysource.G'], interp=['nearest'])
```

The table is a snapshot of everything accumulated so far. To keep one
table per self-cal round, in the usual CASA style, mark a point and
export the increment - applying the chain is equivalent to applying the
cumulative table:

```python
mark = obs.gain_snapshot()
obs.selfcal(amp=True, phase=True)
obs.savecaltable("round2.G", since=mark)
# applycal(..., gaintable=['round1.G', 'round2.G'])
```

difmapy stores the *correction* it applies to the data
(`V_corr = V·c_p·conj(c_q)`) whereas CASA divides by antenna *gains*
(`CORRECTED = DATA/(G_p·conj(G_q))`), so the exported table holds
`G = 1/c`. This is verified against CASA in the test suite: `applycal`
reproduces difmapy's visibilities to float32 precision.

Notes and limits:

* an MS is needed for the antenna/spw/field metadata a caltable refers
  to; it defaults to the one loaded, and `ms=` accepts another. Antennas
  are matched by **name**, so gains derived from an averaged dataset can
  be applied to the full-resolution MS.
* the gains are polarization-independent (they are solved on the
  total-intensity stream) and are written to both parallel hands.
* `resoff` baseline corrections and `shift` cannot be expressed in a
  G table; `savecaltable` reports them rather than dropping them
  silently. Use `wobs()` to write data with everything applied.
* exporting from UVFITS-loaded data works with `ms=`, but timestamps in
  a UVFITS file can differ from the MS's by a fraction of an
  integration, so prefer exporting from the MS-loaded observation.

#### AIPS

```python
obs = difmapy.load("mysource.uvfits")
obs.selfcal(amp=True, phase=True, solint=30)
obs.savecaltable("mysource.TASAV.FITS")   # AIPS SN table (the default here)
```

```
FITLD  the TASAV file                      -> MYSRC.TASAV.1
TACOP  inext 'SN' from it onto the UV data
single-source data: DOCAL 1, GAINUSE <that SN version> (SPLIT, IMAGR...)
multi-source data:  CLCAL first, to turn it into a new CL table
```

The file is laid out as AIPS 31DEC24's own TASAV/FITTP output: one dummy
visibility plus the FQ, AN and SN (revision 11) tables. An SN table
refers to AIPS **station numbers** and counts time in days from the
file's reference date, so both are taken from the UV file the table is
meant for: the one loaded, or another given as `uvfits=` (antennas
matched by name). For data loaded from an MS, antennas are numbered in
ANTENNA-table order from 1, as CASA's `exportuvfits` does. `SOURCE ID`
is 0, so the table applies to every source once copied onto a
multi-source file.

AIPS multiplies the data by `conj(g_p)·g_q` - the same amplitude
convention as difmapy but the opposite phase sign - so the SN table
holds `g = conj(c)`. The test suite checks this in AIPS itself when it
is installed: FITLD, TACOP and SPLIT with `DOCAL 1` reproduce difmapy's
corrected visibilities to float32 precision.

### Bayesian amplitude calibration

`gscale` finds one amplitude correction per station against a model -
but that model was built from the same data, so a station whose
amplitude scale is wrong has already pulled the model towards its error.
`bayes_gscale` breaks that circle and puts error bars on the result:

```python
r = obs.bayes_gscale(prefix="3C345_bayes")  # runs, applies, writes report
print(r)                                    # the summary again
r.plot()                                    # the diagnostic figure
r.factors                                   # [IF, station] corrections applied
```

Like `gscale`, it solves every IF separately (`per_if=True`, the
default): the summary has one column per IF with the correction, its
1-sigma uncertainty and a mark where it is needed (`*`, P >= 0.95) or
probably needed (`?`, P >= 0.75), the findings name the IFs, and the
figure shows each station's IFs side by side. `per_if=False` combines
them into one correction per station.

1. **Leave one station out.** For each station (and once for the full
   array) the source model is rebuilt with that station ignored, phase
   self-calibrating as it goes. The station is then brought back, phased
   up against that fixed model, and the whole array `gscale`d, so its
   correction comes from a model it had no say in. The runs that leave
   out the *other* stations give a jackknife spread. Where leaving a
   station out moves its own estimate a lot, that shift counts as
   systematic uncertainty too: it may mean the full-array model had
   absorbed the station's error, or just that the station's baselines
   reach spatial frequencies nobody else constrains.
2. **Compare source models** (`models=`, default CLEAN and one to three
   Gaussians; also `pointN`, `cleanN` and `current`) by the Bayesian
   information criterion on the full array, with the chi-squared
   rescaled by the best model's reduced chi-squared (VLBI weights are
   rarely absolute). The corrections are averaged over models by their
   posterior probabilities.
3. **Shrink to the prior**: a Gaussian on each log-amplitude correction
   of width `prior_sigma` (default 10%, the a-priori calibration
   accuracy). The Bayes factor between "a correction is needed" and "it
   is not" gives `P(needed)` for every station, and what is applied is
   the posterior average over both, so corrections the data do not
   demand are left out.

Every run works on its own copy of the observation and releases the GIL
in the heavy steps, so the runs go in parallel threads (`workers=`; 36
runs on the 3C345 test data take 2 s on 10 cores). `timeavg=` averages
the working copy first for long tracks. `prefix` writes
`<prefix>.json` (settings, per-model evidence, per-station and per-IF
corrections with uncertainties and probabilities, the influence matrix,
every run, and the findings in words), `<prefix>.png` (the diagnostics:
corrections against the prior, `P(needed)`, model evidence, the
leave-one-out influence matrix, per-station fit before/after) and a
calibration table of the constant corrections. The table goes to
`<prefix>.G` and/or `<prefix>.TASAV.FITS` (`outformat=`, `ms=` and
`uvfits=` as for `savecaltable`), ready to apply to the other sources.
The figure needs matplotlib.

### Notebooks and pipelines

Every plot also works without a window. In a Jupyter notebook - or
anywhere without a display, such as a server or a CI job - plots are
drawn off-screen and shown in the cell output, and every plot command
returns an object that can be saved:

```python
obs.radplot()                              # appears in the notebook
p = obs.vplot(3)
p.savefig("plots/vplot_{page}.png")        # every page, numbered from 1
obs.mapplot().savefig("map.png", width=1400, height=1000)
p.set_page(2); p                           # show page 3 inline
```

`difmapy.plots.set_mode("window")` switches a notebook to interactive Qt
windows instead (the Qt event loop is hooked in, as `%gui qt` does, so
the kernel stays responsive); `"inline"` forces off-screen plots
anywhere, and `"auto"` is the default. `$DIFMAPY_PLOT_MODE` sets the
same from outside, and `difmapy --batch` always draws inline. Paged
plots (`vplot`, `cpplot`, `corplot`, `fplot`) take `set_page(n)` in
place of the `n`/`p` keys, and radplot's antenna highlight and
projplot's angle have `cycle_antenna()` and `rotate_projection()`.

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
res = obs.modelfit()            # niter=-1: run until it converges
print(res["rchisq"], res["converged"], obs.model, res["errors"])
```

`niter=-1` (the default) iterates until successive Levenberg-Marquardt
steps stop improving the reduced chi-squared; pass a positive `niter`
for a fixed number of steps. With no model at all, `seed_model()` is
called first: a circular Gaussian of zero width carrying the peak flux,
placed at the peak of the residual map.

What `addcmp`, `clean`, `rmodel` and `modelfit` produce is part of the
model straight away: `obs.model` is always the whole model, and there is
no `keep` step. Components keep their free parameters, so a fit can
simply be run again to iterate. As in difmap (`obvarmod`), components
without free parameters - CLEAN components above all - are held fixed,
their visibilities subtracted before the fit so that it does not absorb
their flux a second time. `modelfit(free=...)` overrides the masks: a
list gives one spec per model component; a single spec sets which
parameters vary on the components that already have free parameters (or
on every component, if none has).

Like difmap's, this is a local optimizer: it converges from a sensible
starting guess but can settle in a local minimum from a far-off one, so
start from something like the map peak. The reported `errors` are
first-order estimates from the inverse Hessian and ignore parameter
covariances.

### The interactive map display

`obs.mapplot()` (or `maplot()`) takes `mapsize=`, `cellsize=` and
`uvweight=` (a Briggs robustness), which override the current imaging
setup. With none given and `mapsize()` never called, it images 4096
pixels of a tenth of the estimated resolution (lambda / B_max), and it
prints the beam it is showing. The "Weighting" box above the image
switches between difmap's `uvweight` scheme and Briggs robust -2 ... +2
and re-images straight away (the choice stays set, as `uvweight()`
would leave it).

| key | action |
| --- | --- |
| double-click | add a CLEAN window (drag to move/resize) |
| `d` | delete the window under the cursor |
| `c` / `i` | CLEAN / re-invert |
| `1` `2` `3` `4` | residual map, dirty beam, restored map, model |
| `m` | add a component: click the centre, then any two points on it |
| `d` (while adding) | point source, then circular Gaussian |
| `f` | fit the placed components to the UV data (modelfit) |
| `C` | clear every model component |
| `l` | logarithmic or linear colour scale |
| `z` / `u` | restore the y / x axis range |
| `r` | reload the plot from the data |
| `x` | close and report the image properties |
| `h` / `q` | key legend / close |

The same list runs along the bottom of the window as a footnote, and a
heading over the image names what is being displayed (residual map,
dirty beam, restored map or model) with its peak.

A component placed with `m` is only *drawn*: the flux it starts from is
a guess read off the map, so it is held out of the image until `f` fits
it (or `k` establishes it as it stands). Placing one therefore never
changes the map underneath it, and `f` never invents a component of its
own - place one first.

### Amplitude and phase against frequency

`obs.fplot()` is `vplot`'s frequency counterpart: amplitude and phase
per baseline, every channel of every IF averaged over the whole
observation (or over `tmin`..`tmax`), `nplot` baselines to a page with
`n`/`p` to page through them and `reftel=` to keep one station's.
Amplitudes are scalar-averaged so they do not decorrelate as the fringe
turns; phases come from the vector average, which is what makes a slope
across the band visible. `specplot()` shows the same average over all
baselines at once.

Both draw on `obs.spectrum()`, which averages *all* channels - the
unselected ones too, so it can be used to choose them - with the
accumulated calibration applied; `calibrated=False` averages the data
as it was loaded.

### The colour bar

Drag the two handles on the colour bar to set the displayed range.
`l` (or `mapplot(scale="log")`) switches the colours to a logarithmic
scale: the stretch is a redistribution of the colour map, not of the
pixel values, so the levels and the bar's axis stay in Jy/beam and the
negative half of a residual map keeps its place.

Beside the colour bar is a histogram of the displayed pixel values with
a Gaussian fitted to the residual noise, which shows the noise
distribution and any emission standing above it at a glance. Closing
with `x` prints (and leaves in `plot.result`) the `mapinfo()` dict: beam,
peak position and flux, model components, total flux, residual rms.

### Other commands

`shift`/`unshift`, `resoff`/`clroff`, `uvaver`, `uvtaper`, `uvrange`,
`uvzero`, `gscale`, `station_gains`, `uncalib`, `selfant`,
`clrmod`, `clearmodel`, `auto_mapsize`, `estimated_resolution`,
`mapinfo`, `noise_stats`, `wmodel`/`rmodel`, `wwins`/`rwins`, `wobs`,
`wdmap`, `wbeam`, `imstat`, `spectrum`, `closure_phases`, `specplot`,
`fplot`, `tplot`, `scans`, `ignore`/`unignore`.

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
