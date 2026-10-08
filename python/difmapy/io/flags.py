"""The flags of a session as a list of flagging commands.

AIPS and CASA both carry flags separately from the data as a list of
selections - "this baseline, these IFs and channels, this time range" -
an FG table in AIPS, a flag-command list for ``flagdata(mode='list')``
in CASA. `flag_entries` turns the flags a difmapy session has *added*
(those set now that were not set when the data were loaded) into such a
list, as compactly as it can:

* per baseline and IF, consecutive integrations with the same flags
  become one time range;
* the same range on consecutive IFs becomes one entry spanning them;
* an integration and IF in which everything of a station that could
  be flagged has been becomes an entry for the station, found first.

Samples already flagged in the file (or holding no data) are "don't
care": a selection may cover them, so they never break a run.

Entries refer to the data *as it was loaded*: after averaging on load,
a time range is the whole averaging bin (so it covers every original
integration that went into the flagged sample) and channels are the
original ones behind each averaged channel.

Flags that were in the file already are not repeated, and samples
unflagged during the session cannot be expressed this way (neither
package's list can undo a flag in the data) - `save_flags` writes the
exact state back into a Measurement Set when that is what is wanted.

`save_flagcmds` writes the list for CASA; the AIPS FG table is
`difmapy.io.aips.save_fgtable`.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time

__all__ = ["flag_entries", "save_flagcmds"]


def _current_flags(obs):
    """The flags now, with stations set aside by `ignore` given back the
    flags they had: ignoring is session state, not a flag to export."""
    flags = np.array(obs.flags, copy=True)
    for rows, block in obs._ignored.values():
        flags[np.asarray(rows, dtype=int)] = block
    return flags


def _time_edges(obs):
    """Start and end of every row's sample, in seconds since ref_mjd.

    For data averaged in time it is the whole averaging bin - the
    averaged timestamp is a weighted mean and the summed integration
    time need not reach the samples at the bin's edges - otherwise the
    integration centred on the timestamp.
    """
    core = obs._core
    t = np.asarray(core.rows()[0], dtype=np.float64)
    bin_width = getattr(obs, "_aver_time", None)
    if bin_width:
        start = np.floor(t / bin_width) * bin_width
        return start, start + bin_width
    half = 0.5 * np.asarray(core.inttimes(), dtype=np.float64)
    times = np.asarray(core.times(), dtype=np.float64)
    step = np.diff(times)
    step = step[step > 0]
    # No integration time recorded: stay well inside the spacing.
    fallback = 0.25 * float(step.min()) if step.size else 0.5
    half = np.where(half > 0, half, fallback)
    return t - half, t + half


def _original_channels(obs):
    """For each IF: (number of original channels, [nchan, 2] array of
    the first and last original local channel behind each current one)."""
    nchan = obs.nchan
    origin = getattr(obs, "_chan_origin", None)
    out, coff = [], 0
    if origin is None:
        for n in nchan:
            idx = np.arange(n)
            out.append((n, np.column_stack([idx, idx])))
        return out
    orig_nchan = obs._orig_if_nchan
    orig_coff = np.concatenate([[0], np.cumsum(orig_nchan)])
    for i, n in enumerate(nchan):
        out.append((int(orig_nchan[i]), origin[coff : coff + n] - orig_coff[i]))
        coff += n
    return out


def _runs(mask_changes):
    """Start and end (inclusive) of runs, given where a new one starts."""
    starts = np.nonzero(mask_changes)[0]
    ends = np.append(starts[1:], len(mask_changes)) - 1
    return starts, ends


def flag_entries(obs) -> dict:
    """The flags added in this session, as a compact list of entries.

    Returns ``{"rows": [...], "nif": ..., "nsamples": ...}``. Each row
    is a dict with `ant1` and `ant2` (antenna entry indices; `ant2` is
    None for "every baseline of `ant1`"), `t0`/`t1` (seconds since
    `ref_mjd`), `if0`/`if1` (inclusive, 0-based), `chans` (inclusive
    0-based original channels within each IF, or None for all) and
    `pols` (one bool per recorded polarization, or None for all).
    `nsamples` counts the visibility samples the entries were made
    from.
    """
    core = obs._core
    now = _current_flags(obs)
    base = getattr(obs, "_flags_at_load", None)
    if base is not None and base.shape == now.shape:
        new = now & ~base
    else:
        # No record of the flags at load: everything flagged that holds
        # data.
        new = now & (np.asarray(core.calibrated_cube()[1]) != 0)
    nif, npol = core.nif, core.npol
    out = {"rows": [], "nif": nif, "nsamples": int(new.sum())}
    if not out["nsamples"]:
        return out

    _, a1, a2, *_ = (np.asarray(x) for x in core.rows())
    t_start, t_end = _time_edges(obs)
    nant = len(core.antenna_names)
    chan_info = _original_channels(obs)
    coffs = np.concatenate([[0], np.cumsum(obs.nchan)])
    bl = a1.astype(np.int64) * nant + a2
    # `new` is whittled down as entries claim samples; the station test
    # counts against all of them, so that a baseline between two flagged
    # stations does not stop the second from being recognised.
    every = new
    new = new.copy()
    # Samples there is nothing to say about: flagged in the file
    # already, or holding no data. A selection may cover them freely,
    # which is what lets a run of flags continue across them.
    care = ~base if (base is not None and base.shape == now.shape) else \
        (np.asarray(core.calibrated_cube()[1]) != 0)

    # ---- whole stations first -----------------------------------------
    # An integration and IF in which every sample of a station that
    # could be flagged has been: one entry for the station, not one per
    # baseline (whose own runs differ wherever the file's flags do).
    tidx = np.asarray(core.time_index(), dtype=np.int64)
    ntime = int(core.ntimes)
    it_start = np.full(ntime, np.inf)
    it_end = np.full(ntime, -np.inf)
    np.minimum.at(it_start, tidx, t_start)
    np.maximum.at(it_end, tidx, t_end)
    station_rows = {}   # (a, t0, t1) -> [IFs]
    load = [int(every[(a1 == a) | (a2 == a)].sum()) for a in range(nant)]
    for a in sorted(range(nant), key=lambda a: -load[a]):
        if not load[a]:
            continue
        rows = np.nonzero((a1 == a) | (a2 == a))[0]
        for i in range(nif):
            sl = slice(coffs[i], coffs[i + 1])
            n_new = np.bincount(tidx[rows], every[rows, sl].sum(axis=(1, 2)),
                                ntime)
            if not n_new.any():
                continue
            n_care = np.bincount(tidx[rows], care[rows, sl].sum(axis=(1, 2)),
                                 ntime)
            on = (n_new > 0) & (n_new == n_care)
            if not on.any():
                continue
            # Runs of "on" integrations, continuing over those where the
            # station has nothing that could be flagged.
            active = np.nonzero(on | (n_care == 0))[0]
            groups = np.split(active, np.nonzero(np.diff(active) != 1)[0] + 1)
            for grp in groups:
                hit = grp[on[grp]]
                if hit.size == 0:
                    continue
                key = (a, float(it_start[hit[0]]), float(it_end[hit[-1]]))
                station_rows.setdefault(key, []).append(i)
                sel = rows[(tidx[rows] >= hit[0]) & (tidx[rows] <= hit[-1])]
                new[sel, sl] = False

    # ---- per baseline and IF: runs in time, then channel/pol blocks ---
    # entries keyed by everything but the IF, to merge IFs afterwards.
    per_if = {}
    touched = np.nonzero(new.any(axis=(1, 2)))[0]
    order = touched[np.argsort(bl[touched], kind="stable")]
    cuts = np.nonzero(np.diff(bl[order]))[0] + 1
    for rows in (np.split(order, cuts) if order.size else []):
        b = int(bl[rows[0]])
        # Every row of the baseline, so that a run is only as long as
        # the flags really are continuous.
        all_rows = np.nonzero(bl == b)[0]
        for i in range(nif):
            sl = slice(coffs[i], coffs[i + 1])
            block = new[all_rows, sl, :]                 # [n, nch, npol]
            if not block.any():
                continue
            # Rows with nothing that could be flagged take the pattern
            # of the row before, so they do not break a run.
            wild = ~care[all_rows, sl, :].any(axis=(1, 2))
            src = np.maximum.accumulate(
                np.where(wild, 0, np.arange(len(all_rows))))
            eff = block[src]
            change = np.ones(len(all_rows), dtype=bool)
            change[1:] = (eff[1:] != eff[:-1]).any(axis=(1, 2))
            for s, e in zip(*_runs(change)):
                pat = eff[s]
                if not pat.any():
                    continue
                real = np.nonzero(~wild[s : e + 1])[0]
                first, last = s + real[0], s + real[-1]
                t0 = float(t_start[all_rows[first]])
                t1 = float(t_end[all_rows[last]])
                norig, cmap = chan_info[i]
                # Channel runs sharing one polarization pattern.
                cchange = np.ones(pat.shape[0], dtype=bool)
                cchange[1:] = (pat[1:] != pat[:-1]).any(axis=1)
                for c0, c1 in zip(*_runs(cchange)):
                    pols = pat[c0]
                    if not pols.any():
                        continue
                    o0, o1 = int(cmap[c0, 0]), int(cmap[c1, 1])
                    chans = None if (o0 <= 0 and o1 >= norig - 1) else (o0, o1)
                    pkey = None if pols.all() else tuple(bool(p) for p in pols)
                    per_if.setdefault((b, chans, pkey, t0, t1), []).append(i)

    def if_groups(ifs):
        ifs = np.sort(np.asarray(ifs))
        return np.split(ifs, np.nonzero(np.diff(ifs) != 1)[0] + 1)

    for (a, t0, t1), ifs in station_rows.items():
        for grp in if_groups(ifs):
            out["rows"].append(dict(ant1=a, ant2=None, t0=t0, t1=t1,
                                    if0=int(grp[0]), if1=int(grp[-1]),
                                    chans=None, pols=None))

    # ---- merge consecutive IFs, and list what is left per baseline ---
    for (b, chans, pkey, t0, t1), ifs in per_if.items():
        for grp in if_groups(ifs):
            out["rows"].append(dict(
                ant1=b // nant, ant2=b % nant, t0=t0, t1=t1,
                if0=int(grp[0]), if1=int(grp[-1]), chans=chans,
                pols=None if pkey is None else list(pkey)))
    out["rows"].sort(key=lambda e: (e["t0"], e["ant1"],
                                    -1 if e["ant2"] is None else e["ant2"],
                                    e["if0"]))
    return out


def _casa_time(mjd_seconds, up):
    """A CASA time string, rounded outwards to the millisecond."""
    ms = int((np.ceil if up else np.floor)(mjd_seconds * 1e3))
    whole, frac = divmod(ms, 1000)
    day, sec = divmod(whole, 86400)
    date = Time(float(day), format="mjd", scale="utc").strftime("%Y/%m/%d")
    return (f"{date}/{sec // 3600:02d}:{sec % 3600 // 60:02d}:"
            f"{sec % 60:02d}.{frac:03d}")


def save_flagcmds(obs, path, entries, reason="difmapy"):
    """Write flags as a CASA flag-command list, one command per line,
    for ``flagdata(vis=..., mode='list', inpfile=path)``.

    `entries` are those of `flag_entries`. Antennas and the field are
    given by name and spectral windows by the ids the data came from,
    so the list applies to the Measurement Set that was loaded - or to
    another one of the same observation, such as the multi-source
    parent.

    Returns a summary dict (`path`, `nrows`).
    """
    core = obs._core
    origin = getattr(core, "_cal_origin", None) or {}
    names = core.antenna_names
    spw_ids = origin.get("spw_ids") or list(range(core.nif))
    pol_names = obs.pols
    ref = core.ref_mjd * 86400.0
    lines = [
        f"# Flags added in a difmapy session on {core.source_name}.",
        "# Apply with: flagdata(vis='<ms>', mode='list', "
        f"inpfile='{path}')",
    ]
    for e in entries["rows"]:
        ant = names[e["ant1"]] if e["ant2"] is None else \
            f"{names[e['ant1']]}&{names[e['ant2']]}"
        spws = [int(spw_ids[i]) for i in range(e["if0"], e["if1"] + 1)]
        if e["chans"] is None:
            spw = ",".join(str(s) for s in spws)
        else:
            c0, c1 = e["chans"]
            spw = ",".join(f"{s}:{c0}~{c1}" if c1 > c0 else f"{s}:{c0}"
                           for s in spws)
        cmd = [
            "mode='manual'",
            f"field='{core.source_name}'",
            f"antenna='{ant}'",
            f"spw='{spw}'",
            f"timerange='{_casa_time(ref + e['t0'], False)}"
            f"~{_casa_time(ref + e['t1'], True)}'",
        ]
        if e["pols"] is not None:
            corr = ",".join(n for n, on in zip(pol_names, e["pols"]) if on)
            cmd.append(f"correlation='{corr}'")
        cmd.append(f"reason='{reason}'")
        lines.append(" ".join(cmd))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"path": path, "nrows": len(entries["rows"])}
