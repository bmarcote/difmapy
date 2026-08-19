//! The processing "stream": selection of polarization + channel
//! ranges, reduced to one visibility per (row, IF).
//!
//! This replaces difmap's ob_select()/getIF() machinery. Where difmap
//! paged a single IF into memory at a time, the stream holds every IF
//! simultaneously:
//!
//! * `raw`: the channel-averaged, polarization-combined visibilities
//!   before calibration (equivalent to difmap state OB_RAWIF /
//!   ifdata.scr contents),
//! * `vis`: the same after applying the antenna gain table
//!   (equivalent to OB_GETIF), kept up to date incrementally,
//! * `model`: the model visibilities per (row, IF).
//!
//! Channel averaging semantics are an exact port of ob_select()
//! (obutil.c): unweighted mean of the channel visibilities, combined
//! weight npts^2 / sum(1/wt); deletion of any selected channel deletes
//! the output; a flagged channel flags the output.

use crate::obs::Observation;
use crate::stokes::{Cvis, PolOp, Stokes};
use rayon::prelude::*;

/// Inclusive channel range over the global channel axis.
pub type ChanRange = (usize, usize);

#[derive(thiserror::Error, Debug)]
pub enum SelectError {
    #[error("polarization {0} is unavailable")]
    NoPol(String),
    #[error("invalid channel range {0}..{1}")]
    BadRange(usize, usize),
    #[error("no channels selected")]
    Empty,
}

pub struct Stream {
    pub stokes: Stokes,
    /// The polarization construction used.
    pub polop: PolOp,
    /// Selected global channel ranges (inclusive, sorted, merged).
    pub chlist: Vec<ChanRange>,
    /// Per-IF selected channel ranges (IF-local indices).
    pub if_ranges: Vec<Vec<ChanRange>>,
    /// Uncalibrated selected visibilities `[nrow * nif]`.
    pub raw: Vec<Cvis>,
    /// Calibrated visibilities `[nrow * nif]`.
    pub vis: Vec<Cvis>,
    /// Model visibilities `[nrow * nif]` (re, im).
    pub model: Vec<(f32, f32)>,
    /// Effective frequency per IF (Hz): channel-weighted mean
    /// frequency of the selected channels (difmap getfreq()).
    /// 0.0 for IFs with no selected channels.
    pub if_freq: Vec<f64>,
    /// True for IFs with at least one selected channel.
    pub if_used: Vec<bool>,
}

/// Normalize a channel list: sort, clip and merge overlapping ranges.
fn normalize_chlist(chlist: &[ChanRange], nctotal: usize) -> Result<Vec<ChanRange>, SelectError> {
    let mut list: Vec<ChanRange> = Vec::new();
    for &(ca, cb) in chlist {
        if ca > cb || cb >= nctotal {
            return Err(SelectError::BadRange(ca, cb));
        }
        list.push((ca, cb));
    }
    if list.is_empty() {
        // Default: all channels.
        list.push((0, nctotal - 1));
    }
    list.sort_unstable();
    let mut merged: Vec<ChanRange> = Vec::with_capacity(list.len());
    for (ca, cb) in list {
        match merged.last_mut() {
            Some(last) if ca <= last.1 + 1 => last.1 = last.1.max(cb),
            _ => merged.push((ca, cb)),
        }
    }
    Ok(merged)
}

impl Stream {
    /// Build a new stream for the given selection. Equivalent to
    /// difmap's `select` command (ob_select()).
    pub fn select(
        ob: &Observation,
        stokes: Stokes,
        chlist: &[ChanRange],
    ) -> Result<Stream, SelectError> {
        let polop =
            PolOp::find(&ob.pols, stokes).ok_or_else(|| SelectError::NoPol(stokes.name().into()))?;
        let chlist = normalize_chlist(chlist, ob.nctotal)?;
        let nif = ob.nif();

        // Intersect the global channel list with each IF's channel
        // span, producing per-IF local channel ranges.
        let mut if_ranges: Vec<Vec<ChanRange>> = vec![Vec::new(); nif];
        for (cif, band) in ob.ifs.iter().enumerate() {
            let (lo, hi) = (band.coff, band.coff + band.nchan - 1);
            for &(ca, cb) in &chlist {
                let (a, b) = (ca.max(lo), cb.min(hi));
                if a <= b {
                    if_ranges[cif].push((a - band.coff, b - band.coff));
                }
            }
        }
        let if_used: Vec<bool> = if_ranges.iter().map(|r| !r.is_empty()).collect();
        if !if_used.iter().any(|&u| u) {
            return Err(SelectError::Empty);
        }

        // Effective mean frequency of the selected channels per IF.
        let if_freq: Vec<f64> = ob
            .ifs
            .iter()
            .zip(&if_ranges)
            .map(|(band, ranges)| {
                let mut fsum = 0.0;
                let mut n = 0usize;
                for &(ca, cb) in ranges {
                    for ch in ca..=cb {
                        fsum += band.chan_freq(ch);
                        n += 1;
                    }
                }
                if n > 0 {
                    fsum / n as f64
                } else {
                    0.0
                }
            })
            .collect();

        // Channel-average and polarization-combine each (row, IF).
        let nrow = ob.nrow;
        let mut raw = vec![Cvis::default(); nrow * nif];
        raw.par_chunks_mut(nif).enumerate().for_each(|(row, out)| {
            average_row(ob, polop, &if_ranges, row, out);
        });

        let mut stream = Stream {
            stokes,
            polop,
            chlist,
            if_ranges,
            vis: raw.clone(),
            raw,
            model: vec![(0.0, 0.0); nrow * nif],
            if_freq,
            if_used,
        };
        stream.apply_calibration(ob);
        Ok(stream)
    }

    /// Re-average and re-calibrate the given rows from the raw cube
    /// (used after editing/flagging raw data).
    pub fn rebuild_rows(&mut self, ob: &Observation, rows: &[usize]) {
        let nif = ob.nif();
        for &row in rows {
            let out = &mut self.raw[row * nif..(row + 1) * nif];
            average_row(ob, self.polop, &self.if_ranges, row, out);
        }
        self.apply_calibration_rows(ob, Some(rows));
    }

    /// (Re-)apply the observation's gain table to produce `vis` from
    /// `raw`. Port of app_Telcor() semantics (telcor.c):
    /// amp *= Aa*Ab; phs += pa-pb; wt /= (Aa*Ab)^2; flagged if either
    /// gain is marked bad.
    pub fn apply_calibration(&mut self, ob: &Observation) {
        self.apply_calibration_rows(ob, None);
    }

    /// Apply calibration to a subset of rows (None = all rows).
    pub fn apply_calibration_rows(&mut self, ob: &Observation, rows: Option<&[usize]>) {
        let nif = ob.nif();
        let gains = &ob.gains;
        // Baseline corrections (resoff) are skipped when they are the
        // identity, which is the common case.
        let bcor = ob
            .bcor
            .as_ref()
            .filter(|b| !b.is_identity());
        // Accumulated phase-center shift (difmap uvshift), applied here
        // so that it survives re-averaging of edited rows.
        let (east, north) = (ob.geom.east, ob.geom.north);
        let doshift = east != 0.0 || north != 0.0;
        let if_freq = self.if_freq.clone();
        let cal_row = |row: usize, vis: &mut [Cvis], raw: &[Cvis]| {
            let it = ob.time_idx[row] as usize;
            let (a1, a2) = (ob.ant1[row] as usize, ob.ant2[row] as usize);
            for cif in 0..nif {
                let mut v = raw[cif];
                if v.wt == 0.0 {
                    vis[cif] = v;
                    continue;
                }
                let ia = gains.idx(it, cif, a1);
                let ib = gains.idx(it, cif, a2);
                let mut ampcor = gains.amp[ia] * gains.amp[ib];
                let mut phscor = gains.phs[ia] - gains.phs[ib];
                if let Some(bc) = bcor {
                    if let Some(k) = bc.index(ob.ant1[row], ob.ant2[row], cif) {
                        ampcor *= bc.amp[k];
                        phscor += bc.phs[k];
                    }
                }
                if doshift {
                    let freq = if_freq[cif];
                    let (us, vs) = (ob.uvw[row * 3], ob.uvw[row * 3 + 1]);
                    phscor += (std::f64::consts::TAU
                        * (us * freq * east + vs * freq * north))
                        as f32;
                }
                if ampcor != 1.0 || phscor != 0.0 {
                    let (s, c) = phscor.sin_cos();
                    let (re, im) = (v.re, v.im);
                    v.re = ampcor * (re * c - im * s);
                    v.im = ampcor * (re * s + im * c);
                    v.wt /= ampcor * ampcor;
                }
                // Bad gain solutions flag the visibility (difmap
                // FLAG_TA/FLAG_TB).
                if (gains.bad[ia] || gains.bad[ib]) && v.wt > 0.0 {
                    v.wt = -v.wt;
                }
                vis[cif] = v;
            }
        };
        match rows {
            None => {
                self.vis
                    .par_chunks_mut(nif)
                    .zip(self.raw.par_chunks(nif))
                    .enumerate()
                    .for_each(|(row, (vis, raw))| cal_row(row, vis, raw));
            }
            Some(rows) => {
                for &row in rows {
                    let raw = &self.raw[row * nif..(row + 1) * nif];
                    // Safe split: vis and raw are distinct fields, but
                    // the closure borrows raw by slice; copy locally.
                    let raw_copy: Vec<Cvis> = raw.to_vec();
                    let vis = &mut self.vis[row * nif..(row + 1) * nif];
                    cal_row(row, vis, &raw_copy);
                }
            }
        }
    }

    #[inline]
    pub fn nif(&self) -> usize {
        self.if_used.len()
    }
}

/// Channel-average and polarization-combine one row into `out[nif]`
/// (the inner loop of ob_select(); see the module docs for semantics).
fn average_row(
    ob: &Observation,
    polop: PolOp,
    if_ranges: &[Vec<ChanRange>],
    row: usize,
    out: &mut [Cvis],
) {
    let mut pvis = [Cvis::default(); 8];
    let pvis = &mut pvis[..ob.npol().min(8)];
    for (cif, band) in ob.ifs.iter().enumerate() {
        if if_ranges[cif].is_empty() {
            continue;
        }
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        let mut var_sum = 0.0f32; // sum of 1/wt
        let mut npts = 0u32;
        let mut flagged = false;
        let mut deleted = false;
        'ranges: for &(ca, cb) in &if_ranges[cif] {
            for ch in ca..=cb {
                ob.pvis(row, band.coff + ch, pvis);
                let mut cur = polop.get(pvis);
                if cur.wt == 0.0 {
                    deleted = true;
                    break 'ranges;
                }
                if cur.wt < 0.0 {
                    flagged = true;
                    cur.wt = -cur.wt;
                }
                npts += 1;
                sum_re += cur.re;
                sum_im += cur.im;
                var_sum += 1.0 / cur.wt;
            }
        }
        out[cif] = if deleted || var_sum == 0.0 || npts == 0 {
            Cvis::default()
        } else {
            let n = npts as f32;
            let re = sum_re / n;
            let im = sum_im / n;
            if re == 0.0 && im == 0.0 {
                // difmap treats identically-zero visibilities as deleted
                Cvis::default()
            } else {
                let wt = n * n / var_sum;
                Cvis {
                    re,
                    im,
                    wt: if flagged { -wt } else { wt },
                }
            }
        };
    }
}
