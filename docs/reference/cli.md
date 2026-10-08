# Command line

```
difmapy [options] [file]
```

With no `--batch`, an IPython session is started: the loaded observation
is `obs`, and its methods are available as bare, Difmap-style commands
(`clean(200, 0.03)`, `selfcal(phase=True)`, `mapplot()` ...).
`load('other.uvfits')` switches dataset and rebinds the commands.

## Loading

These are the parameters of [`difmapy.load`](loading.md).

| option | meaning |
|---|---|
| `file`, `-o FILE` | the UVFITS file or Measurement Set to load |
| `-s`, `--stokes STOKES` | polarization to select (default `I`); `none` to skip the selection |
| `--channels RANGES` | channel ranges, e.g. `0-31` or `0-3,8-11` |
| `--timeavg TIME` | time-average on load, e.g. `10s` or `2min` (`--average` is its older name) |
| `--freqavg N` | average N adjacent channels, or `all` for one channel per IF |
| `--scatter` | with `--timeavg`: weights from the scatter of the averaged samples |
| `--wtscale F` | multiply the data weights by F |
| `--field NAME\|ID` | Measurement Sets: the field to load |
| `--data-column COLUMN` | Measurement Sets: `DATA` (default) or `CORRECTED_DATA` |

## Imaging setup

| option | meaning |
|---|---|
| `--mapsize N` | map size in pixels (a multiple of 4) |
| `--cell MAS` | pixel size in mas |
| `--uvweight BINWID ERRPOW` | gridding weights, e.g. `--uvweight 0 -1` for natural |
| `--robust R` | Briggs robustness, from -2 (uniform) to 2 (natural) |

## Running

| option | meaning |
|---|---|
| `-c`, `--command CODE` | Python to run after loading (repeatable) |
| `--batch` | run the `--command` code and exit, without a prompt; plots are drawn off-screen |
| `--no-gui` | do not enable the Qt event loop |
| `--version` | print the version and exit |

## Examples

```sh
difmapy                                   # an empty interactive session
difmapy mysource.uvfits                   # load and select Stokes I
difmapy data.ms --stokes I --mapsize 2048 --cell 0.5
difmapy big.uvfits --timeavg 10s
difmapy big.ms --freqavg all
difmapy multi.ms --field 3C345 --data-column CORRECTED_DATA
difmapy data.ms -c "clean(200, 0.03)"
difmapy data.ms --batch -c "clean(200, 0.03)" -c "wmap('m.fits')"
```

## Commands at the prompt

Selection and data
:   `select`, `header`, `flag`, `unflag`, `ignore`, `unignore`,
    `save_flags`, `wflags`, `uvaver`, `chanaver`, `savecaltable`,
    `gain_snapshot`

Imaging
:   `mapsize`, `auto_mapsize`, `estimated_resolution`, `uvweight`,
    `uvtaper`, `uvrange`, `uvzero`, `invert`, `clean`, `clrmod`,
    `clearmodel`, `restore`, `imstat`, `peak_offset`, `noise_stats`,
    `mapinfo`, `add_window`, `clear_windows`

Calibration
:   `selfcal`, `gscale`, `bayes_gscale`, `station_gains`, `uncalib`,
    `selfant`, `startmod`, `resoff`, `clroff`

Models
:   `addcmp`, `seed_model`, `modelfit`, `rmodel`, `wmodel`

Geometry
:   `shift`, `unshift`

Plots
:   `mapplot` (`maplot`), `radplot`, `projplot`, `uvplot`, `vplot`,
    `cpplot`, `tplot`, `corplot`, `specplot`, `fplot`

Output
:   `wobs`, `wmap`, `wdmap`, `wbeam`, `wwins`, `rwins`, `save`

Each is a method of [`Observation`](observation.md), where its
parameters are documented; `help(clean)` at the prompt shows the same.
