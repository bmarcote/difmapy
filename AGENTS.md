# difmapy development notes

## Build & test

- If pytest dies at start-up with ``numpy.dtypes has no attribute
  'StringDType'``, it is `zarr`'s auto-loaded pytest plugin against an
  older numpy, nothing to do with difmapy: run with
  ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``.

- Python venv: `/home/marcote/.venv313` on Linux, `/Users/hawky/.venv`
  on the macOS box (both uv-managed, no pip module — use
  `uv pip install --python <venv>/bin/python3 ...`)
- Build the extension: `VIRTUAL_ENV=<venv> maturin develop --uv`
  (add `--release` for anything where speed matters — the debug build
  is ~10x slower, which shows on 4096-pixel maps)
- Rust tests: `cargo test -p difmap-core`
- Python tests: `<venv>/bin/python3 -m pytest tests/ -q`
  (plot tests self-configure offscreen Qt; the user environment sets
  `QT_QPA_PLATFORMTHEME=gtk3` which aborts without a display, so the
  tests override it)
- **Beware a stale non-editable install.** `maturin develop` leaves an
  editable install pointing at `python/`, but a plain
  `pip install .` copies the sources into site-packages, and then
  editing the repo changes nothing at the prompt. Check
  `python -c "import difmapy; print(difmapy.__file__)"` before
  believing a bug report about the Python layer.

## Documentation site

`docs/` + `zensical.toml`, built with Zensical (`pip install -r
docs/requirements.txt`, then `zensical build --clean` -> `site/`, which
is gitignored) and published to GitHub Pages by
`.github/workflows/docs.yml`.

- The reference pages (`docs/reference/*.md`) are `:::` directives:
  mkdocstrings reads the **sources** with griffe, statically. Nothing is
  imported, so the build needs no Rust toolchain - and a docstring that
  is assembled at import time (a decorator filling in a placeholder)
  shows up unassembled. Keep docstrings literal.
- griffe parses numpy-style sections: prose placed after a `Parameters`
  block is read as more parameters ("Parameter 'Returns' does not appear
  in the function signature"). Put it under `Returns`/`Notes` headings.
  A clean build prints only "No issues found".
- Zensical does not check image paths. Pages in `docs/guide/` reach the
  figures as `../images/...`.
- No MathJax is configured; write formulas as code.
- The figures in `docs/images/` are rendered from the 3C345 test data
  (`savefig` on the plot windows, `BayesGainResult.plot`). Regenerate
  them when a plot's look changes.
- To look at the built site without a browser: serve `site/` and grab
  pages with PySide6's `QWebEngineView` under
  `QT_QPA_PLATFORM=offscreen` and
  `QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu --disable-gpu-compositing"`
  (without those flags the capture is blank).
- A user-facing change needs its guide page updated as well as the
  README.

## Test data

`tests/rsm07_3C345.uvfits` and `tests/rsm07_3C345.ms` are the same real
EVN observation of 3C345 (14 antennas, 4 IFs x 1 channel, full pol,
~83% flagged) in both formats. They are gitignored but the tests use
them when present and skip otherwise. Cross-format agreement is a
regression test (`tests/test_realdata.py`) — keep it exact.

Note the MS contains autocorrelations; both loaders drop them.

## Architecture invariants

- The raw visibility cube (`Observation.vis`) and its weights are never
  modified. Flags live in the separate boolean `Observation.flag` array
  (the MS FLAG column convention); weights are always >= 0, and
  weight == 0 means deleted/absent data, which stays permanently
  flagged (difmap FLAG_DEL) — `unflag` must never clear it.
- Calibration lives in `GainTable` (antenna gains) and `bcor`
  (per-baseline resoff corrections), plus the accumulated `geom`
  phase-center shift. All three are composed in
  `Stream::apply_calibration_rows`, which is the single place data
  corrections are applied. Anything that must survive re-averaging of
  edited rows (`Stream::rebuild_rows`) belongs there, not in a
  one-shot mutation of the stream.
- `uncalib` must remain a pure gain-table reset.
- The *derived* stream keeps difmap's signed-weight convention
  internally (wt < 0 = flagged) as a compact per-point display state;
  do not confuse it with the raw store's representation.
- Stream semantics (channel averaging, pol combination, weight signs)
  are exact ports of difmap's ob_select()/obpol.c — see comments in
  `crates/difmap-core/src/stream.rs`. Do not "fix" them without
  checking the difmap sources in `difmap-master/`.
- Briggs robust weighting (`InvertPars.robust`) supersedes
  `binwid`/`errpow` when set: it reuses difmap's own UV bin array, but
  sums the natural weights per bin instead of counting points. R = -2
  reproduces difmap's uniform weighting to ~15% (the binning is
  difmap's, not a true single-cell grid) and R = +2 reproduces its
  natural weighting to <2%; the tests pin both ends.
- Map dimensions must be multiples of 4 (nx/2+1 half-plane grid, nx/4
  bin and inner-quarter arrays); powers of two are no longer required
  because rustfft/realfft take any length, but are still much faster.
- `fit_uvmodel(niter)` takes an i64: negative means "iterate to
  convergence", capped at `modelfit::MAX_ITER`. `FitResult` reports
  `niter` and `converged`.
- difmap numeric conventions: uvw stored in light-seconds (multiply by
  frequency in Hz for wavelengths); phases in radians; weights =
  1/variance with sign encoding flag state (>0 good, <0 flagged,
  ==0 deleted); model phase convention V = A·exp(+2πi(ux+vy)).
- Map pixel (ix, iy) ↔ sky: x = (ix−nx/2)·xinc east, y = (iy−ny/2)·yinc
  north; FITS output flips the x axis (RA decreases with pixel).
- User-facing units in the Python API: mas for positions/sizes,
  degrees for angles, wavelengths for uv radii.

## Reference

`difmap-master/` holds the original C sources; ported algorithms map:
uvinvert.c/uvtrans.c/costran.c → grid.rs; mapclean.c/mapres.c →
clean.rs; modvis.c/besj.c → model.rs; modfit.c/lmfit.c → modelfit.rs;
slfcal.c → selfcal.rs; obutil.c/obpol.c → stream.rs; obedit.c →
edit.rs; obshift.c/resoff.c → geom.rs; clphs.c → closure.rs.

## Python-layer invariants

- `Observation._mapsize_set` records whether the user chose a map size.
  `mapplot()` calls `auto_mapsize()` (4096 pixels of
  `estimated_resolution()/10`) only when they did not; `invert()` keeps
  its 256 x 1 mas default so the test suite stays fast and
  deterministic.
- Interactive flag undo (`plots.base.EditHistory`) stores before/after
  snapshots of the FLAG column rows an edit touched, via the
  `flags_rows`/`set_flags_rows` bindings. Do not replace it with an
  inverse `edit_rows` call: that would wrongly unflag channels inside a
  block that were already flagged before the edit.
- `gscale()` reports the corrections *this call* applied, by diffing the
  gain table before and after (unsolved entries read as 1.0 before, NaN
  after), so a gscale following a selfcal is not confused by the
  calibration already in the table. `station_gains()` instead reports
  the accumulated state, with 1.0 for stations with no solution.
- `FlagPlotBase` subclasses supply `_collect()` returning parallel
  arrays keyed by panel quantity (`amp`, `phase`, ...), plus `wt`,
  `row`, `cif` and, for paged plots, `group` (which panel row a point
  belongs to). Flag edits are index-based and shared across panels.
- Stacked panels share one x axis: `_share_x_axis()` links them all to
  the first and strips the tick labels from every panel but the bottom
  one. pyqtgraph's link matches the *data* across panels of different
  widths, so the linked view ranges are close but not equal - do not
  assert equality on them.
- `mapplot` keeps hand-placed components in `MapPlot._placed`, drawn
  (dashed) but not in `obs.model`, because their flux is only a guess
  and anything in the model is imaged at once. `run_modelfit` adds them
  (`_commit_placed`) and restores the previous model if the fit fails;
  closing the window adds any still unfitted. `run_modelfit` refuses to
  seed a component of its own, unlike `Observation.modelfit`; it fits
  whenever something is placed or `obs.nvariable` is non-zero.
- Point plots draw with `plots.base.FastScatter`, not pyqtgraph's
  `ScatterPlotItem`: the latter builds a per-point record costing
  seconds per refresh at ~2M points (a 28-station, 16-IF VLBI run).
  Use `FastScatter` for anything that can have many points; it keeps
  `getData()` for tests.
- `cpplot` lists triangles up front but calls `closure_phases` one
  triangle at a time for the page shown; computing all triangles of a
  large array takes seconds.
- Plot presentation goes through `run_if_needed` and the mode in
  `plots.base` (`set_mode`/`get_mode`: auto/window/inline). Inline
  renders off-screen (`render`/`to_png`/`savefig`) and is the default in
  Jupyter and without a display; window mode hooks IPython's Qt loop
  instead of blocking. Every `refresh()` bumps `_revision` (wrapped in
  `PlotWindow.__init_subclass__`), which is how a notebook avoids
  showing an unchanged plot twice.
- `vplot` takes difmap's argument order: `vplot(nplot, reftel, ...)`.
- mapplot **updates its CLEAN-window ROIs in place** on refresh
  (`_sync_rois_from_obs`) and detaches an ROI's handles before removing
  it (`_retire_roi`). Rebuilding them on every refresh left cyclic
  garbage that owns child items; when Python's collector ran while
  pyqtgraph was constructing the next ROI, Qt segfaulted
  ("Garbage-collecting" in the faulthandler trace). Whether it fired
  depended on allocation counts, so it appeared out of nowhere when the
  venv's numpy changed. Do not go back to remove-all/add-all for any
  item with children or signal connections.
- mapplot shows maps as difmap does: its `rainbow` table (stops
  copied from color.c, clipped to 0..1) or its grey scale, toggled with
  `g` (difmap's `c`/`g`; `c` is CLEAN here), and levels from the
  **minimum to the maximum of the valid area** (difmap's setcmpar). The
  earlier 2-99.9 percentile cut put the top of the scale inside the
  noise on a big field with a compact source - the display then showed
  noise and hid the source. Do not go back to percentiles.
- The restored map is contoured from 3 sigma (residual `noise_stats`)
  in factors of sqrt(2), negatives dashed, toggled with `k`.
  `plots/contours.py` has its own marching squares because pyqtgraph's
  `isocurve` costs ~0.35 s per level on a 2048-pixel map whatever the
  level; ours works only on crossed cells (16 levels in 0.2 s) and
  matches it point for point, apart from pyqtgraph's built-in half
  pixel. Contours are penned light or dark for contrast against the
  colour scale at their level - a contour in the scale's own colour is
  invisible - and re-penned when the colour-bar levels change.
- Stacked panels get one fixed left-axis width (`LEFT_AXIS_WIDTH`) and
  no right axis, so amplitude and phase plot areas coincide exactly; a
  right axis on one panel only narrows it.
- mapplot's "Weighting" box (difmap uvweight / robust -2..2) has
  `NoFocus`: with keyboard focus a combo box eats the single-key
  shortcuts and changes the weighting on "c", "i", ... It re-syncs on
  every `refresh()`, so a `uvweight()` typed at the prompt shows up.
- The loaders write the same parameter text out in each docstring
  (literally, so the static API reference can read it); tests check the
  copies are identical and that
  every parameter of every loader appears in its docstring, so a new
  one must be added there.
- `ignore(station)` is a flag edit with a memory: it snapshots the FLAG
  rows of that station's baselines (as they would be with *nothing*
  ignored - see `_flags_without_ignores`, which matters for a baseline
  between two ignored stations) and flags them, and `unignore` restores
  the snapshot. Every path that edits flags calls `_reapply_ignores()`
  afterwards - `flag`, `unflag` and `EditHistory.apply` - so an unflag
  cannot resurrect an ignored station. Ignores are session state: they
  are not written by `save()` and do not survive `get()`.
- `z`, `u` and `r` are reserved in **every** plot (difmap's `Z`, `U` and
  `L`): restore the y range, restore the x range, reload. That is why
  flag undo/redo moved to Ctrl+Z/Ctrl+Shift+Z and unflag-nearest to
  `F`. A subclass adding keys must not take z/u/r, and must fall through
  to `super().keyPressEvent`. Plots expose their panels through
  `view_boxes()`, and a panel whose default view is fixed (phase, or
  fplot's floored amplitude range) reports it from `default_y_range`.
- **Never let Python garbage-collect a live plot window.**
  `plots.base` keeps a strong reference to every window it creates and
  destroys them from an `atexit` handler (`close_all_windows`). Left to
  the interpreter, a window still alive at shutdown has its wrappers
  freed first and Qt then walks `~QGraphicsScene` -> `~QGraphicsItem`
  over items whose Python halves are gone, which aborts the process
  with SIGTRAP ("zsh: trace trap" after exiting the CLI). It reproduced
  in about half of the runs of
  `printf 'mapplot(block=False)\nexit\n' | difmapy file.uvf`;
  `tests/test_plots.py` keeps a subprocess regression test, repeated,
  because of that. A *closed* window is still a live QMainWindow, so it
  stays in the registry until its deferred deletion has actually
  happened - entries are pruned by `_alive()` when the next window is
  created, not by `closeEvent`. Our handler must also be registered
  after `pyqtgraph` is imported so that it runs before Qt's own module
  shutdown (atexit is last-registered-first).
- Windows open at `PlotWindow.DEFAULT_SIZE`, clipped by
  `fit_to_screen()`; the offscreen test platform reports an 800x600
  virtual screen, so tests must compare against `fit_to_screen(...)`
  rather than the raw constant.
- `gscale` **does** apply its solutions. With `float_scale=False`
  (difmap's default) the gains are renormalised to preserve the data's
  flux scale, so the map peak barely moves and it can look like a
  no-op; that is why it reports `fit_before`/`fit_after` like `selfcal`.
- Time arguments go through `difmapy.units.parse_time` /
  `parse_interval`, which accept "30s"/"1min"/"2h" strings but leave a
  bare number in that argument's historical unit - **minutes** for
  `selfcal`'s solint (difmap's unit), seconds everywhere else. Do not
  "unify" those base units: `solint=30` means half an hour in every
  existing difmap script and test.
- `solint="scan"`/`"Nscan"` sets `SelfcalPars.nscan`, and the selfcal
  bin loop then ends each bin on a scan boundary instead of at a fixed
  duration, which is the only way to guarantee one solution per scan.
  Scan numbers come from `scans::scan_index` (difmap's rule: a gap
  bigger than the threshold starts a new scan) and are global, so every
  subarray breaks its bins in the same places. The default threshold is
  five times the median integration spacing, *not* difmap's flat hour -
  that default is for breaking plot axes and would merge a whole track
  into one "scan".
- `spectrum()` (and so `specplot`/`fplot`) averages the *raw*
  channel-resolved cube, which the stream's calibration never touches,
  so it applies the gains, baseline corrections and shift itself, per
  channel - the same composition as `Stream::apply_calibration_rows`,
  but with the channel's own frequency for the shift phase. It read the
  cube uncalibrated until 2026-09-10, which made a vector average of
  self-calibrated data meaningless. `tests/test_diagnostics.py` pins it
  against the stream's own average.
- pyqtgraph's `ColorBarItem` rounds every level to a multiple of
  `rounding` (default **1**) and refuses a narrower span, so on a map in
  Jy/beam the first drag of a handle snapped the levels to (0, 1) and
  the bar died. `MapPlot._tune_rounding()` keeps it at a thousandth of
  the displayed span, on every refresh and after every drag. The log
  colour scale is a re-positioning of the colour map's stops
  (`log_stretch`), never a transform of the pixel values: the levels and
  the bar axis must stay in Jy/beam, and residual maps have negatives.
- `selfcal` reports difmap's `moddif` fit before and after the solution,
  and the residual-map statistics too when a map already exists. The
  extra invert that costs is the one the next `clean` would have done,
  since self-cal invalidates the map anyway.

## Deliberate deviations from difmap

- `load()`/`observe()` selects Stokes I on load; difmap's `observe`
  selects nothing. `stokes=None` restores the old behaviour, and the
  CLI's `--stokes none` goes through the same path.
- `select("I")` uses difmap's permissive `pi` combination
  (`PolOp::PseudoI`), not strict (RR+LL)/2: a visibility survives when
  only one parallel hand is usable. `Stokes::PI` is a legacy alias and
  is canonicalised to `I` in `Stream::select`, so headers and FITS
  output report Stokes I. Q/U/V stay strict.
- Weighting defaults are difmap's, but the map-area, flag-storage and
  write-path differences are listed in the README.

## CASA calibration tables

`python/difmapy/io/caltable.py` writes a NewCalTable ("G Jones") from
the accumulated gains. Hard-won details, all of them load-bearing:

- CASA identifies a caltable by the table **info** (`type='Calibration'`,
  `subType='G Jones'`), not by keywords. Without it applycal reports
  `type found = ""`.
- The conformance check also requires `QuantumUnits=['s']` on TIME and
  INTERVAL, and TIME carries `MEASINFO` (UTC epoch).
- `ANTENNA2 = -1` marks an antenna-based table. `CPARAM` etc. are
  `[ncorr, nchan, nrow]`; `WEIGHT` is left unwritten, as in CASA's own
  tables.
- Subtables (ANTENNA/FIELD/OBSERVATION/SPECTRAL_WINDOW/HISTORY) are deep
  copies of the MS's.
- **Convention**: difmapy multiplies data by corrections, CASA divides by
  gains, so `CPARAM = 1/(amp·exp(i·phs))`. The test suite verifies this
  by running CASA's applycal and comparing CORRECTED_DATA with
  difmapy's own corrected cube - do not "simplify" the reciprocal away.
- When comparing against CASA output, exclude flagged samples: this
  file has flagged junk at |V| ~ 2000 where float32 rounding dwarfs
  physical tolerances.

## AIPS SN tables (TASAV)

`python/difmapy/io/aips.py` writes an SN table inside a TASAV FITS file;
`savecaltable(outformat=...)` picks CASA/AIPS/both, defaulting to the
loaded file's native format. Established against AIPS 31DEC24 itself:

- **Convention**: AIPS multiplies the data by `conj(g_p)·g_q`, difmapy
  by `c_p·conj(c_q)`, so `SN = conj(c)` - same amplitude, phase negated.
  Not the reciprocal (that is CASA). `tests/test_aips_sntable.py` runs
  FITLD/TACOP/SPLIT(DOCAL 1)/FITTP in AIPS and compares with difmapy's
  corrected cube; it needs `/opt/aips/LOGIN.SH` (or
  `$DIFMAPY_AIPS_LOGIN`) and empties AIPS user 7301's catalog.
- The layout copies what AIPS's own CALIB -> TASAV -> FITTP writes: one
  all-zero dummy group whose data axes match the target UV file, then
  FQ, AN, SN (REVISION 11, with the DISP/DDISP columns).
- SN antennas are AIPS station numbers and times are days from the
  target file's reference date (AN `RDATE`), so both come from a UVFITS
  file (`uvfits=`, default the one loaded) matched by name. UVFITS
  loading therefore keeps the NOSTA numbers.
- Driving AIPS from a script: `aips notv < cmds`, first line the user
  number, POPS lines short (a long line fails with "LINE SIZE"), file
  names in upper case (AIPS upper-cases what it writes), and it exits
  through signal 11 after `kleenex` - judge success from the log.

Calibration provenance lives in `core._cal_origin` (format, path, spw
ids, field, AIPS antenna numbers, and for an MS the DATA_DESC, ANTENNA
and ARRAY ids). Unlike `_ms_origin`, which maps rows and so is dropped
by averaging, it survives `uvaver`/`chanaver`/`copy`, which is why
averaged data can still export tables.

## Averaged data and Measurement Sets

- `save()` on averaged MS data writes a *new* MS (`save_averaged_ms`):
  the main-table structure is copied with `norows=True` - which empties
  the subtables too, so each is then copied over in full - and the rows
  are rebuilt from the observation. Scan/observation/state/feed ids come
  from the source row nearest in time. Write each column in one
  `putcol` when all IFs have the same channel count: going through
  per-window `query` views took 24 s instead of 1 s on em163 (the
  `by_window` path remains for unequal channel counts, and a test
  checks both give the same MS).
- `DATA` gets the averaged data without the session's calibration (a
  copy with `uncalib` + `clroff`), `CORRECTED_DATA` the calibrated.
  Rows with no data in any IF are not written, so the reloaded MS can
  have fewer rows than the session.
- `uvaver`/`chanaver` leave `_avg_row_map`/`_avg_chan_map` on the new
  core and `Observation._trace_averaging` composes them into
  `_ms_avg_origin`. `save_flags` uses it to OR a flagged averaged
  sample onto every MS sample behind it. It never unflags: an average
  cannot say which inputs were good.
- `save()` also writes the caltable (native format, once any gain is
  `used`), the flag table (once flags were added) and the restored
  image (once there is a model); each of `ms`/`caltable`/`flags`/`image`
  is None (when applicable) / False / True.

## Flag tables (`wflags`)

`io/flags.py::flag_entries` turns the flags *added since load*
(`Observation._flags_at_load` is the reference; it is shared, not
copied, by `copy()`) into selections, written as an AIPS FG table
(`io/aips.py::save_fgtable`, same TASAV frame as the SN table) or a
CASA flag-command list. Both are verified in the packages themselves.

- Samples already flagged in the file, or with no data, are "don't
  care". Without that, each baseline's time run breaks wherever the
  file's own flags do and nothing merges.
- Stations are detected first, per integration and IF ("everything of
  this station that could be flagged, is"), counting against *all* new
  flags - not what earlier stations left - or a baseline between two
  flagged stations stops the second from being recognised.
- Time ranges of averaged data are the whole bin (`_aver_time`), not
  timestamp +- integration/2: the averaged time is a weighted mean and
  the summed integration time need not reach the bin's edge samples.
  Channels map back through `_chan_origin`.
- FG columns/values are those AIPS's own UVFLG writes (SOURCE 0,
  FREQ ID -1, ANTS sorted with 0 = any, CHANS (1,0) = all, TIME RANGE
  float32 days - rounded outwards with nextafter).
- `ignore()`d stations are not exported; un-flagging cannot be.

## Bayesian gain calibration

`bayescal.bayes_gscale` runs one job per (source model, station left
out), each on `Observation.copy()` (a Rust clone), in threads - the
heavy bindings release the GIL, and the result must not depend on the
number of workers (a test pins it). Things that look odd but are meant:

- The per-station estimate is the *leave-own-out* gscale; its variance
  adds the jackknife over the other exclusions, a floor, and
  `(loo - full)^2 / 4`. That last term is what stops a station with
  unique uv coverage (T6 on the 3C345 data: x1.9 when left out) from
  being "corrected" by an extrapolated model.
- Model evidence is BIC on the full-array runs with chi-squared
  rescaled by the best model's reduced chi-squared; CLEAN counts three
  parameters per distinct component position.
- What is applied is `P(needed) * posterior mean` in log amplitude, so
  on noiseless test data the applied value is the gscale value times
  `tau^2/(tau^2 + sigma^2)`, not the gscale value itself.
- Everything is per (IF, station) when `per_if=True`, and the report
  must show it that way: a station-level median hid opposite
  corrections in two IFs (x0.76 and x1.25 read as "x0.98"). Medians over
  IFs exist in `station_table()` only for sorting; the summary, the
  findings and the figure go per IF.
- The figure (`bayesplot.py`) is matplotlib, not pyqtgraph, and is drawn
  inside an `rc_context` that turns `text.usetex` off: with LaTeX on,
  every "%" in a label starts a comment and truncates it.

## Known behaviour worth remembering

- **Only the inner quarter of a map is meaningful.** Outside it the
  gridding correction (dividing by the kernel's transform) grows
  without bound, so a dirty beam can exceed 1.0 there. difmap defines
  `maparea` the same way; `clean` and `map_stats` honour it, and the
  Python API exposes `valid_slice`/`valid()`/`peak_offset()`. Never
  `argmax` a whole map in tests or examples.
- **Weighting defaults are difmap's** (`invdef`: uvbin=2, errpow=0,
  i.e. uniform weighting ignoring the data weights). Tests that assert
  weighting-dependent numbers (beam size, peak flux) must set the
  weighting explicitly.
- Time-averaging real data smears emission far from the phase centre:
  240 s bins on the 3C345 file lose ~30% of the cleaned flux. Use
  coherence/weight-budget invariants in tests, not flux preservation.
- `uvaver` must give every baseline in a bin *exactly* the same
  timestamp; the core identifies integrations by equal times, so
  per-baseline float noise would split each baseline into its own
  integration and quietly break per-integration self-cal.
- `SelfcalResult.nbins` counts solution bins per (subarray, IF), not
  per integration.
- With a finite `solint`, self-cal does **not** apply a bin's solution
  as a step: `apply_solns` (a port of the same function in slfcal.c)
  smooths and interpolates the bins onto the integration grid with a
  Gaussian of `sigma = 0.37478125 * solint` minutes, truncated at
  2.5 sigma, weighting each bin by the area under that Gaussian inside
  its [begut, endut] times the solution's own weight. That is why
  difmap's corplot shows a smooth evolution, and the give-away that it
  is working is that the corrections vary *within* a bin
  (`tests/test_calibration_paths.py` pins that). `solint=0` (per
  integration) and `doone`/`gscale` skip it, as in difmap, and so does
  scan binning (`nscan > 0`): one solution per scan is a step by
  construction, and blending across a slew gap would undo the point of
  it.
- A gap in a station's data wider than the 2.5-sigma reach leaves
  integrations with *no* solution: their gain-table entries read
  1.0 / 0 deg and `gains_used()` is false. Plots must mask on
  `gains_used()`, not just on the `bad` flag - drawing those entries
  made an interpolated, perfectly smooth run of corrections look like
  it jumped to unity and back, which is what `corplot` did until
  2026-09-11.
- An identically zero visibility counts as deleted (difmap behaviour),
  which bites when constructing test data: if RR == LL then V == 0 and
  the sample is reported deleted rather than flagged.
- The CLI (`difmapy.cli`) binds observation methods into the IPython
  namespace, and prints its banner itself because IPython's
  `display_banner=False` also suppresses `banner1`. `--batch` runs
  `-c` code without a prompt, which is how the CLI is tested.
- `modelfit` is a local optimizer (as in difmap): it converges to the
  truth from a reasonable starting guess but can settle in a local
  minimum (often at the `ratio -> 0` limit) from a far-off start. This
  is expected, not a bug.
- There is no user-visible tentative model and no `keep`: every
  binding that adds or fits components (`add_component`, `clean`,
  `modelfit`) merges `ob.newmod` into `ob.model` before it returns, so
  `newmod` is only ever non-empty inside `modelfit`. Do not reintroduce
  a step to establish the model (removed 2026-09-14 at the user's
  request).
- `modelfit` partitions the model first (`partition_variable_model`,
  difmap's obvarmod): components with no free parameter stay in the
  stream model so the fit sees the residuals after them, components
  with one are moved to `newmod` and fitted, then merged back. Do not
  "simplify" it away - dropping the fixed components from the fit (what
  the port did until 2026-09-10) makes the fit absorb their flux a
  second time, and merging the result then doubles it.
  `ncomp` in the result counts the components actually fitted;
  `total_ncomp` the whole model.
- On the real 3C345 data the map peak lies outside the inner quarter at
  1024 x 1 mas, so `imstat` (which only scans the cleanable inner
  quarter) reports a different peak from `np.argmax(obs.dmap)`.
- `tplot` has labelled its y axis with antenna names since b8919f7;
  a report of numbers there means a stale install, not a bug.
- `selfcal`/`gscale` against an empty model produce NaN gains and
  poison the stream.
- CASA writes correlations as RR, RL, LR, LL, which is not a regular
  FITS STOKES axis; `save_uvfits` reorders them (see the test in
  `tests/test_realdata.py`).
