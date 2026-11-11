# Flagging

Flags are held in an explicit FLAG array - the Measurement Set
convention - separate from the data, so every edit is reversible.

```python
obs.flag(station="EF", if_index=2)                 # every EF baseline, IF 3
obs.flag(baseline=("EF", "JB"), tmin="1h", tmax="1.5h")
obs.unflag(station="EF", if_index=2)
obs.flagged_fraction
```

Time ranges are seconds from the start of the reference day, or strings
with a unit. Data that are absent from the file (zero weight) count as
permanently flagged and cannot be unflagged, matching Difmap's
deleted-data flag.

To take a whole station out for a while and bring it back as it was,
use [`ignore`](selfcal.md#setting-a-station-aside) instead.

## In the plots

`radplot`, `uvplot` and `vplot` flag with the mouse:

| action | effect |
|---|---|
| ++shift++ + drag | flag the points in the box |
| ++ctrl++ + drag | unflag them |
| ++f++ / ++shift+f++ | flag / unflag the point nearest the cursor |
| ++ctrl+z++ / ++ctrl+shift+z++ | undo / redo the last edit |
| ++x++ | show or hide the flagged points (hidden by default) |
| ++space++ (`vplot`) | flag the displayed baseline, or every baseline of its first antenna |
| ++shift+s++, then drag | smart flag: the box, and the antenna behind it (see below) |

Edits act only on what is drawn: an IF switched off in `vplot`'s legend is not flagged by a box drawn over it.

`tplot` takes the same ++shift++/++ctrl++ + drag and undo keys: the box flags (unflags) every baseline of each station whose row it covers, in the selected IFs, at the integrations inside it.

A plain drag and the wheel keep their usual pan and zoom. Undo is
exact: each edit stores the affected rows of the FLAG column before and
after, so it restores what was there even where the edit overlapped data
that were already flagged.

## Smart flagging

A fault is nearly always one antenna's - a receiver, a pointing, a
slew, interference at one site - and shows on all of its baselines at
once. The smart flagger (`difmapy.outliers`) compares every baseline
and IF with a smooth model (each integration with the others of its
scan, each scan with the trend of the others), combines the departures
per antenna, and flags the antenna at fault on every baseline, over the
time range of the fault with its rising and decaying edges.

### All the data

```python
obs.smartflag()                        # antennas beyond 5 sigma, single baselines beyond 5 sigma
obs.smartflag(use_phases=False)        # amplitudes only
obs.smartflag(antenna_sigma=4, quiet=False)
# {'samples': 417, 'antennas': {'EF': 133, 'HH': 210, ...}, 'alone': 0}
```

It prints what it flagged: the stations at fault, with the samples
flagged on their baselines, and the samples flagged on a single
baseline. The dashboard's **Smart flag** button runs it.

### A box in a plot

In `radplot`, `projplot`, `uvplot` and `vplot`, ++shift+s++ arms a
smart selection: the next drag (no modifier needed) draws an orange box,
and ++esc++ cancels. Every point in the box is taken as an outlier and
flagged, whatever the model makes of it. The flagger then looks for the
antenna behind them:

- of the two antennas of a selected baseline, the one whose *other*
  baselines depart with it is at fault, so a box over the bad points of
  one baseline in `vplot` flags the right antenna on all its baselines;
- an antenna most of whose baselines are in the box is at fault
  outright;
- a point no antenna explains is flagged on its own baseline.

The antenna's flag then grows over the time range of the fault, inside
or outside the box. The bar for blaming an antenna is lower than in a
search of all the data: from 5 sigma for a box over everything to
3 sigma for a small one, so that a box never blames more antennas
wrongly than a search of all the data would. On a phase panel the
phases are tested too, elsewhere the amplitudes only. One ++ctrl+z++
takes the whole edit back.

## Keeping the flags

There are two ways to take flags out of a session, for two purposes.

### Write them back into the Measurement Set

```python
obs.save_flags()        # FLAG (and FLAG_ROW) of the MS the data came from
```

Only the flag columns are modified; data and weights are untouched. For
data as loaded, the FLAG column becomes exactly the current flags.

For data that were **averaged** in time or frequency, a flagged averaged
sample flags every sample of the MS that went into it. An average cannot
say which of its inputs were good, so unflagged ones are left as the
file has them: from averaged data, flags are only ever added.

### Export them as a flag table

```python
obs.wflags("mysource.FG.TASAV.FITS")     # AIPS, the default for UVFITS data
obs.wflags("mysource.flagcmd")           # CASA, the default for an MS
obs.wflags("mysource.flagcmd", outformat="both")
```

`wflags` writes the flags **added in the session** as a list of
selections - station or baseline, IFs, channels, time range - so they
can be applied to the original data or to other data of the same
observation.

=== "AIPS"

    An FG table in a TASAV FITS file:

    ```
    FITLD  the file                         -> MYSRC.TASAV.1
    TACOP  inext 'FG' from it onto the UV data
    the flags then apply wherever FLAGVER selects that table
    ```

=== "CASA"

    A flag-command list, one command per line:

    ```python
    flagdata(vis="mysource.ms", mode="list", inpfile="mysource.flagcmd")
    ```

    ```
    mode='manual' field='3C345' antenna='T6' spw='0' timerange='2025/09/16/13:00:22.000~2025/09/16/15:50:00.000' reason='difmapy'
    mode='manual' field='3C345' antenna='JB&WB' spw='2' timerange='2025/09/16/15:45:04.000~2025/09/16/15:45:32.000' reason='difmapy'
    ```

The list is kept short: consecutive integrations become one time range,
consecutive IFs one entry, and a station flagged on all its baselines a
single station entry. Data the file already has flagged never break a
run.

Things to know:

- Flags that were already in the file are not repeated.
- After averaging on load, a time range is the whole averaging bin and
  the channels are the original ones, so every sample behind a flagged
  average is covered.
- Samples you **un**flagged cannot be expressed in such a list. Use
  `save_flags()` when the exact state matters.
- Stations set aside with `ignore` are not exported.
- If you average again later in the session (`uvaver`, `chanaver`),
  flags set before that are folded into the averaged data and are not
  exported afterwards.

Both formats are checked against the packages themselves: AIPS (`SPLIT`
with the table) and CASA (`flagdata`) end up flagging exactly the
samples difmapy has.

[`save()`](saving.md) writes the flag table automatically.
