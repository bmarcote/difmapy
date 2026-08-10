# difmapy

A modern reimplementation of [Difmap](ftp://ftp.astro.caltech.edu/pub/difmap/difmap.html)
(Shepherd 1997) for interactive VLBI imaging and self-calibration:
a **Rust** compute engine with a **Python** API and **pyqtgraph**
interactive plots.

Where the original paged one IF at a time through scratch files
(`uvdata.scr`, `ifdata.scr`, `modvis.scr`), difmapy keeps **everything
in RAM**: the raw visibility cube is never modified (flags only toggle
weight signs, calibrations are composed on the fly), so every
operation — selection, gridding, CLEAN, self-cal, flagging — happens
in real time and is reversible.

Differences from the original by design:

* reads **UVFITS and Measurement Sets** (both single-source)
* native **multi-IF / multi-channel** handling: all subbands (IFs /
  SPWs) are gridded together (multi-frequency synthesis), each may
  have a different number of channels
* Python API instead of the sphere command language; interactive
  plots use pyqtgraph instead of PGPLOT

The numerics are faithful ports of the difmap algorithms
(uvinvert/uvtrans gridding+FFT, Högbom mapclean, mapres restore,
modvis models, slfcal self-calibration), validated by synthetic-data
tests against analytic ground truth.

## Install (development)

```sh
pip install maturin
maturin develop --release          # builds the Rust core into your env
pip install ".[plot]"              # pyqtgraph + PySide6 for plots
pip install ".[ms]"                # casatools for Measurement Sets
```

## Usage

```python
import difmapy

obs = difmapy.load("mysource.uvf")     # or a .ms directory
print(obs.header())

obs.select("I")                        # Stokes I, all channels
# obs.select("I", channels=[(0, 31)])  # or channel ranges (global axis)

obs.mapsize(1024, 0.1)                 # pixels (power of 2), mas/pixel
obs.uvweight(binwid=2, errpow=-1)      # uniform weighting
obs.invert()                           # dirty map + beam
obs.mapplot()                          # interactive: double-click to add
                                       # CLEAN windows, c=clean, i=invert

obs.clean(niter=200, gain=0.03)        # Högbom CLEAN in the windows
obs.selfcal(phase=True)                # phase self-cal against the model
obs.clean(niter=200, gain=0.03)
obs.selfcal(amp=True, phase=True, solint=30)

obs.radplot()                          # amp vs uv-radius; Shift+drag to flag
obs.vplot(reftel="EF")                 # amp vs time per baseline
obs.uvplot()                           # uv coverage

m = obs.restore()                      # restored map (numpy array)
obs.wmap("clean.fits")                 # FITS output with WCS + beam
obs.save("mysession")                  # .uvf + .mod + .win + parameters
# later: obs = difmapy.Observation.get("mysession")
```

Flagging from scripts:

```python
obs.flag(station="EF", if_index=2)                # all EF baselines, IF 3
obs.flag(baseline=("EF", "JB"), tmin=0, tmax=3600)
obs.unflag(station="EF", if_index=2)              # fully reversible
```

## Layout

```
crates/difmap-core   pure-Rust engine (no Python): data model, stream
                     selection, gridding/FFT, clean, restore, selfcal,
                     model visibilities, editing
crates/difmap-py     PyO3 bindings (module difmapy._core)
python/difmapy       Python API, UVFITS/MS I/O, pyqtgraph plots
tests/               end-to-end pytest suite on synthetic data
difmap-master/       reference: original difmap C sources (not built)
```

## Testing

```sh
cargo test                 # Rust core tests
maturin develop && pytest  # Python end-to-end tests
```
