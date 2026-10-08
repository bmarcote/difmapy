# Getting started

A first session, from a UV file to a saved image. Everything here works
the same from the `difmapy` prompt and from a Python script.

## Start a session

```sh
difmapy mysource.uvfits            # or a Measurement Set directory
```

This opens an IPython prompt with the observation loaded as `obs` and
Difmap-style commands available as bare functions, so `clean(200, 0.03)`
and `obs.clean(200, 0.03)` are the same thing. Plot windows stay live
while you keep typing.

In a script or notebook:

```python
import difmapy

obs = difmapy.load("mysource.uvfits")
print(obs.header())
```

Loading selects total intensity straight away. To choose something
else, or particular channels:

```python
obs.select("RR")
obs.select("I", channels=[(0, 31)])     # inclusive ranges, all IFs concatenated
```

Long, finely sampled data can be averaged as they are read - see
[Loading data](guide/loading.md).

## Set up the image

```python
obs.mapsize(2048, 0.5)        # pixels (a multiple of 4) and mas per pixel
obs.uvweight(robust=0)        # Briggs: -2 uniform ... +2 natural
```

Left alone, the weighting is Difmap's default (uniform, 2-pixel bins),
and `mapplot()` picks a map size from the data if you have not set one.

## The imaging loop

```python
obs.startmod(flux=1.0)               # phase self-cal against a point source

for _ in range(4):                   # the classic Difmap loop
    obs.clean(200, 0.03)             # iterations, loop gain
    obs.selfcal(phase=True)

obs.gscale()                         # one amplitude correction per station
obs.selfcal(amp=True, phase=True, solint="scan")
obs.clean(400, 0.02)
```

Each command prints a short report and returns it as a dict:

```python
res = obs.clean(200, 0.03)
res["cleaned_flux"], res["residual_rms"]

sc = obs.selfcal(phase=True)
sc["fit_before"]["rms"], sc["fit_after"]["rms"]
```

## Look at it

```python
obs.mapplot()        # the map: CLEAN windows, model components, contours
obs.radplot()        # amplitude and phase against uv radius, model in red
obs.vplot(3)         # against time, three baselines a page
obs.corplot()        # the self-cal corrections per station
```

In the map window, double-click adds a CLEAN window, ++c++ cleans, ++3++
shows the restored map and ++h++ lists every key. See
[The map display](guide/mapplot.md) and [Plots](guide/plots.md).

## Keep the result

```python
obs.wmap("clean.fits")          # the restored map, with WCS and beam
obs.save("mysession")           # the whole session
```

`save` writes the calibrated UV data, the model, the CLEAN windows and
the imaging parameters - and, when there is something to write, the
image, the calibration table, the flags and a Measurement Set. See
[Saving a session](guide/saving.md). To come back to it:

```python
obs = difmapy.Observation.get("mysession")
```

## Scripted runs

```sh
difmapy data.ms -c "clean(200, 0.03)"              # run, then give a prompt
difmapy data.ms --batch -c "clean(200, 0.03)" -c "wmap('m.fits')"
```

`--batch` runs the commands and exits, drawing any plots off-screen. All
options are listed in the [command-line reference](reference/cli.md).
