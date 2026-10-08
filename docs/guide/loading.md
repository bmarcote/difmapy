# Loading data

`difmapy.load` (also spelled `observe`, Difmap's name for it) reads a
random-groups **UVFITS** file or a CASA **Measurement Set**. Both must
hold a single source - split a multi-source file first, or pick a field
of a Measurement Set with `field=`.

```python
import difmapy

obs = difmapy.load("mysource.uvfits")
obs = difmapy.load("mysource.ms")                       # a directory: an MS
obs = difmapy.load("multi.ms", field="3C345", data_column="CORRECTED_DATA")
```

```python
print(obs.header())      # source, antennas, IFs, selection
obs.pols                 # ['RR', 'LL', 'RL', 'LR']
obs.antennas, obs.nif, obs.nchan
```

## Selection

Loading selects **Stokes I** at once, since that is how nearly every
session starts. `stokes=` chooses something else and `stokes=None` loads
without selecting anything, as Difmap's `observe` does; `select()`
changes it at any time.

```python
obs.select("I")                         # also Q, U, V, RR, LL, RL, LR, XX, ...
obs.select("I", channels=[(0, 31)])     # inclusive, 0-based channel ranges
```

Channel ranges count along the *global* channel axis: all IFs
concatenated.

!!! note "Stokes I is permissive"

    Total intensity is formed from whichever parallel hands are usable,
    so a visibility survives when one hand is missing or flagged (Difmap
    called this `pi`). With both hands present it is exactly (RR+LL)/2.
    Q, U and V stay strict, since a missing hand cannot be guessed for
    polarization.

## Averaging on load

Averaging a long, finely sampled observation as it is read makes every
later step faster.

```python
obs = difmapy.load("big.uvfits", timeavg="10s")      # time (Difmap's uvaver)
obs = difmapy.load("big.ms", freqavg=4)              # 4 channels into one
obs = difmapy.load("big.ms", freqavg="all")          # one channel per IF
obs = difmapy.load("big.ms", channels=[(2, 29), (34, 61)], freqavg="all")
```

`timeavg`
:   Seconds as a number, or a string with its unit (`"10s"`, `"2min"`).
    `scatter=True` derives the weights from the scatter of the averaged
    samples instead of summing the input weights.

`freqavg`
:   The number of adjacent channels to average, which must divide each
    IF's channel count, or `"all"` for one channel per IF. Only the
    `channels` given go into the averages, which is how band edges are
    dropped - the last example above removes two channels from each end
    of two 32-channel IFs before averaging them.

The same averaging is available afterwards as `obs.uvaver("2min")` and
`obs.chanaver(4)`. Both return a *new* observation: the calibration of
the old one is applied first, and the new one starts uncalibrated.

Averaged data still go back to where they came from: `save()` writes a
new Measurement Set with the averaged rows, `save_flags()` flags every
original sample behind a flagged averaged one, and calibration and flag
tables refer to the original data. See [Saving a session](saving.md) and
[Flagging](flagging.md).

!!! warning "Time averaging smears"

    Averaging in time smears emission away from the phase centre: on
    the 3C345 test data, 240 s bins lose about 30% of the cleaned flux.
    Keep the bins short compared with the fringe period at the edge of
    your field.

## Times and units

Every argument that carries a time takes a string with its unit as well
as a bare number: `"30s"`, `"1min"`, `"1.5 hours"`, `"2d"`.

A **bare number keeps the unit that argument has always had** - minutes
for `selfcal`'s `solint`, as in Difmap, and seconds everywhere else - so
existing Difmap habits and scripts carry over; a string is how you ask
for something else.

All the parameters are listed in the
[reference](../reference/loading.md).
