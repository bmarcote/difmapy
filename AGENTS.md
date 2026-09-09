# difmapy development notes

## Build & test

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
- On the real 3C345 data the map peak lies outside the inner quarter at
  1024 x 1 mas, so `imstat` (which only scans the cleanable inner
  quarter) reports a different peak from `np.argmax(obs.dmap)`.
- `tplot` has labelled its y axis with antenna names since b8919f7;
  a report of numbers there means a stale install, not a bug.
- `selfcal`/`gscale` against an empty established model produce NaN
  gains and poison the stream. `modelfit` leaves its components
  *tentative*, so `keep()` is required before self-calibrating against
  a fitted model.
- CASA writes correlations as RR, RL, LR, LL, which is not a regular
  FITS STOKES axis; `save_uvfits` reorders them (see the test in
  `tests/test_realdata.py`).
