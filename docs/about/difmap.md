# difmapy and Difmap

difmapy is a reimplementation of
[Difmap](ftp://ftp.astro.caltech.edu/pub/difmap/difmap.html) (Shepherd
1997). Its numerics are faithful ports of Difmap's algorithms; what
changes is how the data are held, and a number of deliberate choices
listed here.

!!! note "Licence"

    difmapy is licensed under AGPL-3.0-only. See `NOTICE.md` in the
    repository for provenance and upstream licensing details. The
    upstream Difmap source code retains its own terms.

## What is ported

| Difmap | difmapy |
|---|---|
| `uvinvert.c`, `uvtrans.c`, `costran.c` | gridding and FFT |
| `mapclean.c`, `mapres.c` | Högbom CLEAN and restore |
| `modvis.c`, `besj.c` | model visibilities |
| `modfit.c`, `lmfit.c` | model fitting |
| `slfcal.c` | self-calibration, including the smoothing of solutions |
| `obutil.c`, `obpol.c` | channel averaging and polarization combination |
| `obedit.c` | editing |
| `obshift.c`, `resoff.c` | shifts and baseline corrections |
| `clphs.c` | closure phases |

Conventions are Difmap's: uvw in light-seconds, weights as 1/variance,
model phase `V = A·exp(+2πi(ux+vy))`, and the weighting defaults
(uniform, 2-pixel bins, no amplitude-error weighting).

## What is different by design

**Everything is in memory.**
:   Difmap paged one IF at a time through scratch files. difmapy holds
    the whole dataset, never modifies the visibilities or weights, and
    composes flags and calibration on the fly. Every operation is fast
    and reversible.

**Loading selects Stokes I.**
:   Difmap's `observe` selects nothing. `stokes=None` restores that.

**Stokes I is permissive.**
:   `select("I")` forms total intensity from whichever parallel hands
    are usable - Difmap's `pi`, which remains as an alias. With both
    hands present and equal weights it is exactly (RR+LL)/2; where only
    one survives the other is assumed identical, which neglects circular
    polarization. Q, U and V stay strict.

**Flags are a FLAG column.**
:   Difmap encoded flags in the sign of the weights. difmapy keeps an
    explicit boolean array, as a Measurement Set does, which is what
    lets flags be written back into the MS or exported as tables.
    Weight zero still means deleted data, which stays flagged.

**No tentative model.**
:   There is no `keep` step: whatever `clean`, `addcmp` or `modelfit`
    produce is part of the model at once.

**Multi-IF and multi-channel data are native.**
:   All IFs are gridded together, and each may have its own number of
    channels - which UVFITS cannot express, but a Measurement Set can.

**Scans for self-calibration.**
:   `solint="scan"` gives exactly one solution per scan. The default gap
    that separates scans is five times the median integration spacing,
    not Difmap's flat hour, which is meant for breaking plot axes.

**Additions.**
:   Briggs robust weighting; Measurement Set input and output;
    calibration and flag tables for CASA and AIPS; Bayesian amplitude
    calibration; model fitting from the map window; map sizes that are
    any multiple of four; results returned as dicts and written as JSON;
    plots that work in notebooks and without a display.

## What is the same on purpose

- **Only the inner quarter of a map is valid**, exactly as Difmap
  defines its map area.
- **A bare `solint` is in minutes.** `selfcal(solint=30)` means half an
  hour, as in every existing Difmap script.
- **`gscale` renormalises** the gains to preserve the data's flux scale
  unless told to float.
- **`modelfit` is a local optimizer** and holds components without free
  parameters fixed, subtracting them first.
- **Shifts are not frozen** into written UV data unless asked.
- The keys ++z++, ++u++ and ++r++ mean the same in every plot as
  Difmap's `Z`, `U` and `L`.

## Layout of the code

```
crates/difmap-core   the Rust engine: data model, selection, gridding and
                     FFT, CLEAN, restore, self-calibration, model fitting,
                     closure quantities, editing, geometry
crates/difmap-py     the Python bindings (module difmapy._core)
python/difmapy       the Python API, UVFITS and MS input/output, the plots
tests/               synthetic data with analytic truth, plus real EVN data
difmap-master/       the original Difmap sources, for reference (not built)
```
