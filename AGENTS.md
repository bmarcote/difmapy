# difmapy development notes

## Build & test

- Python venv: `/home/marcote/.venv313` (uv-managed, no pip module —
  use `uv pip install --python /home/marcote/.venv313/bin/python3 ...`)
- Build the extension:
  `VIRTUAL_ENV=/home/marcote/.venv313 /home/marcote/.venv313/bin/maturin develop --uv`
- Rust tests: `cargo test -p difmap-core`
- Python tests: `/home/marcote/.venv313/bin/python3 -m pytest tests/ -q`
  (plot tests self-configure offscreen Qt; the user environment sets
  `QT_QPA_PLATFORMTHEME=gtk3` which aborts without a display, so the
  tests override it)

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

## Known behaviour worth remembering

- `modelfit` is a local optimizer (as in difmap): it converges to the
  truth from a reasonable starting guess but can settle in a local
  minimum (often at the `ratio -> 0` limit) from a far-off start. This
  is expected, not a bug.
- On the real 3C345 data the map peak lies outside the inner quarter at
  1024 x 1 mas, so `imstat` (which only scans the cleanable inner
  quarter) reports a different peak from `np.argmax(obs.dmap)`.
- CASA writes correlations as RR, RL, LR, LL, which is not a regular
  FITS STOKES axis; `save_uvfits` reorders them (see the test in
  `tests/test_realdata.py`).
