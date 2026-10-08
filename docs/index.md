# difmapy

A modern reimplementation of [Difmap](ftp://ftp.astro.caltech.edu/pub/difmap/difmap.html)
(Shepherd 1997) for interactive VLBI imaging and self-calibration: a
**Rust** compute engine with a **Python** API and **pyqtgraph**
interactive plots.

Where the original paged one IF at a time through scratch files,
difmapy keeps **everything in RAM**. Visibilities and weights are never
modified: flags live in a separate FLAG array and calibrations in
separate gain and baseline tables, composed on the fly when the working
data are built. Every operation - selection, gridding, CLEAN, self-cal,
flagging - therefore runs in real time and is fully reversible.

![The interactive map display](images/mapplot_color.png){ loading=lazy }

## In a few lines

=== "Command line"

    ```sh
    difmapy mysource.uvfits
    ```

    ```
    In [1]: mapsize(2048, 0.5)
    In [2]: clean(200, 0.03)
    In [3]: selfcal(phase=True)
    In [4]: mapplot()          # opens; the prompt stays usable
    In [5]: save('mysession')
    ```

=== "Python"

    ```python
    import difmapy

    obs = difmapy.load("mysource.uvfits")   # or a Measurement Set
    obs.mapsize(2048, 0.5)                  # pixels, mas per pixel
    obs.startmod()                          # phase self-cal on a point source
    for _ in range(4):
        obs.clean(200, 0.03)
        obs.selfcal(phase=True)
    obs.mapplot()
    obs.save("mysession")
    ```

## What it adds to Difmap

- Reads **UVFITS and Measurement Sets** (single source), and writes
  both back - including a new MS for data that were averaged.
- **Flags are an explicit FLAG column**, as in a Measurement Set: they
  can be written straight back into the MS, or exported as an AIPS FG
  table or a CASA flag-command list.
- **Calibration tables for CASA and AIPS**: everything self-calibration
  has solved, as a CASA "G Jones" table or an AIPS SN table, to apply
  to other data such as the other sources of the observation.
- **Bayesian station amplitude calibration**
  ([`bayes_gscale`](guide/bayes-gscale.md)): `gscale` with every station
  left out of the model in turn, source models compared, and error bars
  on the result.
- Native **multi-IF, multi-channel** handling: all subbands are gridded
  together, each with its own number of channels.
- **Briggs robust weighting** as one number from -2 to +2, next to
  Difmap's own scheme.
- **Model fitting from the map window**: place a component with the
  mouse and press ++f++.
- Loading selects **total intensity** straight away, and forms it from
  whichever parallel hands are usable.
- A Python API in place of the command language, with every command
  returning its numbers as a dict (and writing them as JSON on request).

The numerics are faithful ports of Difmap's algorithms - see
[difmapy and Difmap](about/difmap.md) for what is the same and what is
deliberately different.

## Where to go next

<div class="grid cards" markdown>

- **[Installation](install.md)** - build it into your environment
- **[Getting started](getting-started.md)** - a first imaging session
- **[User guide](guide/loading.md)** - each part of the workflow in turn
- **[Reference](reference/observation.md)** - every command and its
  parameters

</div>
