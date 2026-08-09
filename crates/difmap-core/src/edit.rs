//! Visibility editing (flagging/unflagging).
//!
//! The equivalent of difmap's obedit.c, radically simplified by the
//! in-RAM design: edits are applied immediately to the raw visibility
//! cube (weight-sign convention) and the affected stream rows are
//! re-averaged. No deferred-edit lists or scratch-file paging needed.

use crate::obs::Observation;

/// What to edit. All criteria are ANDed; `None` matches everything.
#[derive(Clone, Copy, Debug, Default)]
pub struct EditSelection {
    /// Time range (seconds since ref_mjd, inclusive).
    pub time_range: Option<(f64, f64)>,
    /// Restrict to one baseline (global antenna indices, unordered).
    pub baseline: Option<(u32, u32)>,
    /// Restrict to all baselines of one station (global antenna index).
    pub station: Option<u32>,
    /// Restrict to one subarray.
    pub subarray: Option<u32>,
    /// Restrict to one IF (0-based); None = all IFs.
    pub if_index: Option<usize>,
    /// Restrict to the channels selected in the current stream
    /// (difmap's selchan); otherwise all channels.
    pub sel_chan: bool,
}

/// The global channel spans selected by if_index/sel_chan.
fn edit_spans(ob: &Observation, if_index: Option<usize>, sel_chan: bool) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for (cif, band) in ob.ifs.iter().enumerate() {
        if let Some(want) = if_index {
            if cif != want {
                continue;
            }
        }
        if sel_chan {
            if let Some(stream) = ob.stream.as_ref() {
                for &(ca, cb) in &stream.if_ranges[cif] {
                    spans.push((band.coff + ca, band.coff + cb));
                }
                continue;
            }
        }
        spans.push((band.coff, band.coff + band.nchan - 1));
    }
    spans
}

/// Flag (or unflag) explicit rows (used by interactive plot editing).
pub fn edit_rows(
    ob: &mut Observation,
    rows: &[usize],
    if_index: Option<usize>,
    sel_chan: bool,
    flag: bool,
) {
    let spans = edit_spans(ob, if_index, sel_chan);
    let npol = ob.npol();
    let nctotal = ob.nctotal;
    for &row in rows {
        let base = row * nctotal * npol;
        for &(ca, cb) in &spans {
            for gc in ca..=cb {
                for p in 0..npol {
                    let wt = &mut ob.vis[base + gc * npol + p].wt;
                    *wt = if flag { -wt.abs() } else { wt.abs() };
                }
            }
        }
    }
    if let Some(mut stream) = ob.stream.take() {
        stream.rebuild_rows(ob, rows);
        ob.stream = Some(stream);
    }
}

/// Flag (or unflag) matching visibilities. Returns the number of rows
/// affected. The stream (if any) is re-averaged for those rows.
pub fn edit(ob: &mut Observation, sel: &EditSelection, flag: bool) -> usize {
    let npol = ob.npol();
    let nctotal = ob.nctotal;
    let spans = edit_spans(ob, sel.if_index, sel.sel_chan);

    let mut touched: Vec<usize> = Vec::new();
    for row in 0..ob.nrow {
        if let Some((t0, t1)) = sel.time_range {
            let t = ob.time[row];
            if t < t0 || t > t1 {
                continue;
            }
        }
        let (a1, a2) = (ob.ant1[row], ob.ant2[row]);
        if let Some((ba, bb)) = sel.baseline {
            if !((a1 == ba && a2 == bb) || (a1 == bb && a2 == ba)) {
                continue;
            }
        }
        if let Some(st) = sel.station {
            if a1 != st && a2 != st {
                continue;
            }
        }
        if let Some(isub) = sel.subarray {
            if ob.antennas[a1 as usize].subarray != isub {
                continue;
            }
        }
        for &(ca, cb) in &spans {
            let base = row * nctotal * npol;
            for gc in ca..=cb {
                for p in 0..npol {
                    let wt = &mut ob.vis[base + gc * npol + p].wt;
                    *wt = if flag { -wt.abs() } else { wt.abs() };
                }
            }
        }
        touched.push(row);
    }

    if !touched.is_empty() {
        if let Some(mut stream) = ob.stream.take() {
            stream.rebuild_rows(ob, &touched);
            ob.stream = Some(stream);
        }
    }
    touched.len()
}
