//! The in-memory observation data model.
//!
//! Unlike the original difmap (which paged one IF at a time from
//! scratch files), everything is held in RAM:
//!
//! * an immutable-by-convention raw visibility cube
//!   `[nrow, nctotal, npol]` (all IFs/subbands concatenated along the
//!   channel axis, which supports a different number of channels per
//!   IF, unlike UVFITS),
//! * row metadata (time, baseline, uvw),
//! * a separate table of antenna gain corrections (self-cal results),
//! * an optional derived "stream" (see [`crate::stream`]) holding the
//!   channel-averaged, polarization-combined, calibrated visibilities
//!   that all interactive operations work from.
//!
//! Unlike the original difmap (which encoded flags in the sign of the
//! weights), the raw store keeps an explicit boolean FLAG array like a
//! Measurement Set FLAG column: weights are always >= 0, flagging
//! never modifies data or weights, and flags can be written back to an
//! MS. Weight == 0 marks deleted/absent data (difmap FLAG_DEL).
//! The *derived* stream still uses difmap's signed-weight convention
//! internally as a compact per-point display state.

use crate::model::ModComp;
use crate::stokes::Cvis;
use crate::stream::Stream;

#[derive(Clone, Debug, Default)]
pub struct Source {
    pub name: String,
    /// J2000/B1950 right ascension (radians).
    pub ra: f64,
    /// Declination (radians).
    pub dec: f64,
    /// Equinox of ra/dec (years, e.g. 2000.0).
    pub epoch: f64,
}

#[derive(Clone, Debug)]
pub struct Antenna {
    pub name: String,
    /// Geocentric station coordinates (metres).
    pub xyz: [f64; 3],
    /// Subarray this antenna entry belongs to (0-based). The same
    /// physical antenna appearing in two subarrays gets two entries,
    /// as in difmap.
    pub subarray: u32,
    /// If true, self-cal must not change this antenna's gain
    /// (difmap `selfant <name>, true`).
    pub fixed: bool,
    /// Extra self-cal weight multiplier (difmap antwt).
    pub weight: f32,
}

/// One IF / subband / spectral window.
#[derive(Clone, Debug)]
pub struct IfBand {
    /// Frequency of the first channel (Hz).
    pub freq: f64,
    /// Signed channel increment (Hz).
    pub df: f64,
    /// Number of channels in this IF.
    pub nchan: usize,
    /// Offset of this IF's first channel in the global channel axis.
    pub coff: usize,
}

impl IfBand {
    /// Center frequency of channel `ch` (local index).
    #[inline]
    pub fn chan_freq(&self, ch: usize) -> f64 {
        self.freq + ch as f64 * self.df
    }
}

/// Antenna gain corrections per (integration-time, IF, antenna).
/// Equivalent to difmap's `Telcor`. Applied to the data as:
/// amp *= amp_cor(a)*amp_cor(b); phs += phs_cor(a)-phs_cor(b);
/// wt /= (amp_cor(a)*amp_cor(b))^2.
#[derive(Clone, Debug)]
pub struct GainTable {
    pub ntime: usize,
    pub nif: usize,
    pub nant: usize,
    pub amp: Vec<f32>,
    pub phs: Vec<f32>,
    pub bad: Vec<bool>,
    /// True where a self-cal solution has been applied (used by the
    /// amplitude normalization, like difmap's negative-amp_cor marker).
    pub used: Vec<bool>,
}

impl GainTable {
    pub fn new(ntime: usize, nif: usize, nant: usize) -> Self {
        let n = ntime * nif * nant;
        GainTable {
            ntime,
            nif,
            nant,
            amp: vec![1.0; n],
            phs: vec![0.0; n],
            bad: vec![false; n],
            used: vec![false; n],
        }
    }

    #[inline]
    pub fn idx(&self, itime: usize, cif: usize, iant: usize) -> usize {
        (itime * self.nif + cif) * self.nant + iant
    }

    /// Reset to identity (difmap `uncalib`).
    pub fn reset(&mut self, do_amp: bool, do_phs: bool, do_flags: bool) {
        if do_amp {
            self.amp.fill(1.0);
        }
        if do_phs {
            self.phs.fill(0.0);
        }
        if do_flags {
            self.bad.fill(false);
        }
        if do_amp && do_phs {
            self.used.fill(false);
        }
    }
}

pub struct Observation {
    pub source: Source,
    pub antennas: Vec<Antenna>,
    pub nsub: usize,
    pub ifs: Vec<IfBand>,
    /// AIPS codes of the recorded polarizations (the pol axis).
    pub pols: Vec<i32>,
    /// Total number of channels over all IFs.
    pub nctotal: usize,

    // ---- row-based visibility table (time-sorted) ----
    pub nrow: usize,
    /// Time of each row: seconds since `ref_mjd` (UTC).
    pub time: Vec<f64>,
    /// Integration time of each row (seconds; 0 if unknown).
    pub inttime: Vec<f32>,
    /// First/second antenna index of each row (into `antennas`).
    pub ant1: Vec<u32>,
    pub ant2: Vec<u32>,
    /// UVW coordinates in light-seconds, `[nrow * 3]`.
    /// Multiply by frequency (Hz) to get wavelengths.
    pub uvw: Vec<f64>,
    /// Raw visibilities `[nrow * nctotal * npol]` as (re, im, wt),
    /// with wt >= 0 (0 = deleted/absent).
    pub vis: Vec<Cvis>,
    /// The FLAG column: true = flagged, `[nrow * nctotal * npol]`.
    pub flag: Vec<bool>,

    // ---- integration (unique time) index ----
    pub ntimes: usize,
    pub times: Vec<f64>,
    /// Map from row to integration index, `[nrow]`.
    pub time_idx: Vec<u32>,

    /// Reference MJD (UTC days) that `time`/`times` are relative to.
    pub ref_mjd: f64,

    /// Antenna gain corrections (self-cal results).
    pub gains: GainTable,

    /// The established model (Fourier-transformed into the stream).
    pub model: Vec<ModComp>,
    /// The tentative model (not yet transformed; see model::merge_model).
    pub newmod: Vec<ModComp>,

    /// The current selection-derived data stream, if one is selected.
    pub stream: Option<Stream>,
}

#[derive(thiserror::Error, Debug)]
pub enum ObsError {
    #[error("inconsistent array lengths: {0}")]
    Shape(String),
    #[error("rows must be sorted by time (row {0} goes backwards)")]
    Unsorted(usize),
    #[error("antenna index out of range at row {0}")]
    BadAntenna(usize),
    #[error("{0}")]
    Invalid(String),
}

impl Observation {
    /// Assemble an observation from loader-provided arrays.
    ///
    /// `vis` must be `[nrow, nctotal, npol]` flattened with wt >= 0
    /// (0 = deleted); `flag` is the matching FLAG column. Rows must be
    /// time-sorted.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source: Source,
        antennas: Vec<Antenna>,
        ifs: Vec<IfBand>,
        pols: Vec<i32>,
        time: Vec<f64>,
        inttime: Vec<f32>,
        ant1: Vec<u32>,
        ant2: Vec<u32>,
        uvw: Vec<f64>,
        mut vis: Vec<Cvis>,
        mut flag: Vec<bool>,
        ref_mjd: f64,
    ) -> Result<Observation, ObsError> {
        let nrow = time.len();
        let npol = pols.len();
        let nctotal: usize = ifs.iter().map(|f| f.nchan).sum();
        if ifs.is_empty() || npol == 0 || nrow == 0 {
            return Err(ObsError::Invalid("empty observation".into()));
        }
        // Validate the IF channel offsets are contiguous.
        let mut coff = 0usize;
        for (i, band) in ifs.iter().enumerate() {
            if band.coff != coff {
                return Err(ObsError::Invalid(format!(
                    "IF {} channel offset {} != expected {}",
                    i, band.coff, coff
                )));
            }
            coff += band.nchan;
        }
        if inttime.len() != nrow || ant1.len() != nrow || ant2.len() != nrow {
            return Err(ObsError::Shape("time/inttime/ant1/ant2".into()));
        }
        if uvw.len() != nrow * 3 {
            return Err(ObsError::Shape(format!(
                "uvw has {} elements, expected {}",
                uvw.len(),
                nrow * 3
            )));
        }
        if vis.len() != nrow * nctotal * npol {
            return Err(ObsError::Shape(format!(
                "vis has {} elements, expected {}*{}*{}",
                vis.len(),
                nrow,
                nctotal,
                npol
            )));
        }
        if flag.is_empty() {
            flag = vec![false; vis.len()];
        } else if flag.len() != vis.len() {
            return Err(ObsError::Shape("flag array length mismatch".into()));
        }
        // Enforce non-negative weights; a deleted point is also flagged.
        for (v, f) in vis.iter_mut().zip(flag.iter_mut()) {
            if v.wt < 0.0 {
                v.wt = -v.wt;
                *f = true;
            } else if v.wt == 0.0 {
                *f = true;
            }
        }
        let nant = antennas.len() as u32;
        for (i, (&a1, &a2)) in ant1.iter().zip(ant2.iter()).enumerate() {
            if a1 >= nant || a2 >= nant {
                return Err(ObsError::BadAntenna(i));
            }
        }
        // Build the integration index from unique consecutive times.
        let mut times = Vec::new();
        let mut time_idx = Vec::with_capacity(nrow);
        for (i, &t) in time.iter().enumerate() {
            match times.last() {
                Some(&last) if t == last => {}
                Some(&last) if t < last => return Err(ObsError::Unsorted(i)),
                _ => times.push(t),
            }
            time_idx.push((times.len() - 1) as u32);
        }
        let ntimes = times.len();
        let nsub = antennas.iter().map(|a| a.subarray).max().unwrap_or(0) as usize + 1;
        let nif = ifs.len();
        Ok(Observation {
            source,
            antennas,
            nsub,
            ifs,
            pols,
            nctotal,
            nrow,
            time,
            inttime,
            ant1,
            ant2,
            uvw,
            vis,
            flag,
            ntimes,
            times,
            time_idx,
            ref_mjd,
            gains: GainTable::new(ntimes, nif, nant as usize),
            model: Vec::new(),
            newmod: Vec::new(),
            stream: None,
        })
    }

    #[inline]
    pub fn nif(&self) -> usize {
        self.ifs.len()
    }

    #[inline]
    pub fn npol(&self) -> usize {
        self.pols.len()
    }

    /// The `npol` recorded visibilities of (row, global channel), in
    /// difmap's signed-weight form (wt < 0 where the FLAG column is
    /// set), as expected by the polarization combiners.
    #[inline]
    pub fn pvis(&self, row: usize, gchan: usize, out: &mut [Cvis]) {
        let npol = self.pols.len();
        let base = (row * self.nctotal + gchan) * npol;
        for p in 0..npol {
            let mut v = self.vis[base + p];
            if self.flag[base + p] {
                v.wt = -v.wt;
            }
            out[p] = v;
        }
    }
}
