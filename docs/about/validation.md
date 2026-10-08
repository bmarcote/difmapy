# Validation and performance

## Validation

The test suite checks against analytic ground truth, real data, and the
other packages.

- The **same EVN observation in UVFITS and as a Measurement Set** loads
  to identical visibilities, weights, flags and uvw, and images
  identically.
- A **point source** inverts to the right pixel with the right flux;
  CLEAN recovers its flux; restore reproduces the peak.
- **Self-calibration** recovers injected antenna gain errors to better
  than 2%.
- **Closure phases** change by less than 10⁻⁴ degrees under full
  amplitude and phase self-calibration - they are gain-invariant by
  construction.
- **Model fitting** recovers all six parameters of an elliptical
  Gaussian injected into real uv coverage.
- **Shifting** translates the map rigidly by exactly the requested
  pixels.
- **Writing flags back** into a Measurement Set is bit-exact and leaves
  the data and weights untouched.
- **CASA calibration tables**: CASA's `applycal` reproduces difmapy's
  corrected visibilities to float32 precision.
- **AIPS SN and FG tables**: applied inside AIPS (FITLD, TACOP, SPLIT),
  they reproduce difmapy's corrected visibilities and its flags sample
  for sample. These tests run where AIPS is installed.
- **CASA flag commands**: `flagdata` with the exported list flags
  exactly what difmapy has flagged.
- **Averaged Measurement Sets** written by `save()` read back as exactly
  what the session holds, and CASA's `listobs` accepts them.
- **`bayes_gscale`** recovers injected per-station, per-IF gain errors,
  and gives the same answer whatever the number of threads.

```sh
cargo test -p difmap-core      # the Rust core
pytest tests/                  # the Python suite, end to end
```

The real-data tests use an EVN observation of 3C345 and are skipped when
it is not present.

## Performance

Release build, 16.6 million visibilities (10 antennas, 12 h, 8 IFs × 32
channels × 2 polarizations, 1024² maps), best of three runs on one
desktop:

| operation | time |
|---|---|
| select (average 16.6M visibilities to 259k) | 8 ms |
| invert 1024² | 40 ms |
| CLEAN, 100 iterations | 43 ms |
| self-calibration (phase, per integration) | 15 ms |
| flag or unflag a station | 13 ms |
| restore | 39 ms |

Bulk operations are slower but one-shot: loading the data and `uvaver`
are dominated by moving the full cube (about 0.9 s each here).
