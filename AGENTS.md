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

## Architecture invariants

- The raw visibility cube (`Observation.vis` in Rust) is never
  modified except for weight-sign toggles (flagging). Calibration
  lives in `GainTable` and is applied when building/refreshing the
  stream; `uncalib` must remain a pure gain-table reset.
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
clean.rs; modvis.c/besj.c → model.rs; slfcal.c → selfcal.rs;
obutil.c/obpol.c → stream.rs; obedit.c → edit.rs.
