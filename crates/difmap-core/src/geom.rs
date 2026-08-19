//! Phase-center shifts and baseline-based corrections.
//!
//! * `shift` / `unshift` (difmap obshift.c): move the phase center by
//!   rotating visibility phases by 2*pi*(u*east + v*north), and move
//!   the model components with it so the model stays attached to the
//!   sky. The accumulated shift is recorded so it can be undone or
//!   frozen into an output file.
//! * `resoff` / `clroff` (difmap resoff.c): per-baseline, per-IF
//!   time-invariant amplitude and phase corrections, used to absorb
//!   non-closing errors. Like the antenna gains, they are stored
//!   separately from the data and applied when the stream is built.

use crate::obs::Observation;

/// Accumulated geometric state of an observation.
#[derive(Clone, Copy, Debug, Default)]
pub struct UVGeom {
    /// Eastward phase-center shift applied so far (radians).
    pub east: f64,
    /// Northward phase-center shift applied so far (radians).
    pub north: f64,
}

/// Per-baseline, per-IF corrections (difmap Bascor).
#[derive(Clone, Debug)]
pub struct BaselineCor {
    pub nbase: usize,
    pub nif: usize,
    /// Baseline key -> index into amp/phs.
    keys: Vec<(u32, u32)>,
    pub amp: Vec<f32>,
    pub phs: Vec<f32>,
}

impl BaselineCor {
    /// Build an identity correction table for the baselines present.
    pub fn new(ob: &Observation) -> BaselineCor {
        let mut seen = std::collections::HashSet::new();
        let mut keys: Vec<(u32, u32)> = Vec::new();
        for row in 0..ob.nrow {
            let (a, b) = (ob.ant1[row], ob.ant2[row]);
            let key = (a.min(b), a.max(b));
            if seen.insert(key) {
                keys.push(key);
            }
        }
        keys.sort_unstable();
        let nif = ob.nif();
        let n = keys.len() * nif;
        BaselineCor {
            nbase: keys.len(),
            nif,
            keys,
            amp: vec![1.0; n],
            phs: vec![0.0; n],
        }
    }

    #[inline]
    pub fn index(&self, a: u32, b: u32, cif: usize) -> Option<usize> {
        let key = (a.min(b), a.max(b));
        self.keys
            .binary_search(&key)
            .ok()
            .map(|i| i * self.nif + cif)
    }

    pub fn baselines(&self) -> &[(u32, u32)] {
        &self.keys
    }

    pub fn reset(&mut self) {
        self.amp.fill(1.0);
        self.phs.fill(0.0);
    }

    pub fn is_identity(&self) -> bool {
        self.amp.iter().all(|&a| a == 1.0) && self.phs.iter().all(|&p| p == 0.0)
    }
}

/// Shift the phase center by (east, north) radians (difmap `shift`):
/// the data phases are rotated by +2*pi*(u*east + v*north) and the
/// model components are moved by the same amount, so the map contents
/// move to (x + east, y + north).
///
/// The accumulated shift is recorded in `ob.geom` and applied when the
/// stream is (re)built, so it survives re-averaging after editing.
pub fn shift(ob: &mut Observation, east: f64, north: f64) {
    ob.geom.east += east;
    ob.geom.north += north;
    // Move the model components with the phase center (difmap shiftmod).
    for c in ob.model.iter_mut().chain(ob.newmod.iter_mut()) {
        c.x += east as f32;
        c.y += north as f32;
    }
    // The model visibilities follow from the moved components, and the
    // data shift is applied by apply_calibration().
    crate::model::recompute_stream_model(ob);
    if let Some(mut stream) = ob.stream.take() {
        stream.apply_calibration(ob);
        ob.stream = Some(stream);
    }
}

/// Undo all accumulated shifts (difmap unshift).
pub fn unshift(ob: &mut Observation) {
    let (e, n) = (ob.geom.east, ob.geom.north);
    if e != 0.0 || n != 0.0 {
        shift(ob, -e, -n);
    }
    ob.geom = UVGeom::default();
}

/// Determine per-baseline amplitude and phase corrections that best
/// align the data with the current model, and record them (difmap
/// `resoff`). Returns the number of (baseline, IF) corrections set.
///
/// `baseline` optionally restricts the operation to one baseline.
pub fn resoff(ob: &mut Observation, baseline: Option<(u32, u32)>) -> usize {
    // The model must exist to compare against.
    crate::model::merge_model(ob);
    if ob.model.is_empty() {
        return 0;
    }
    let stream = match ob.stream.as_ref() {
        Some(s) => s,
        None => return 0,
    };
    let nif = ob.nif();
    let mut bcor = ob.bcor.take().unwrap_or_else(|| BaselineCor::new(ob));
    let nb = bcor.nbase;
    // Weighted sums of the observed/model ratio per (baseline, IF).
    let mut sre = vec![0.0f64; nb * nif];
    let mut sim = vec![0.0f64; nb * nif];
    let mut swt = vec![0.0f64; nb * nif];

    for row in 0..ob.nrow {
        let (a, b) = (ob.ant1[row], ob.ant2[row]);
        if let Some((ba, bb)) = baseline {
            let key = (a.min(b), a.max(b));
            if key != (ba.min(bb), ba.max(bb)) {
                continue;
            }
        }
        for cif in 0..nif {
            if !stream.if_used[cif] {
                continue;
            }
            let k = row * nif + cif;
            let v = stream.vis[k];
            let m = stream.model[k];
            let m2 = (m.0 * m.0 + m.1 * m.1) as f64;
            if v.wt <= 0.0 || m2 == 0.0 {
                continue;
            }
            let Some(bi) = bcor.index(a, b, cif) else {
                continue;
            };
            // ratio = V_model / V_obs (the correction to apply to data)
            let vr = v.re as f64;
            let vi = v.im as f64;
            let v2 = vr * vr + vi * vi;
            if v2 == 0.0 {
                continue;
            }
            let w = v.wt as f64 * m2;
            // (m / v) = m * conj(v) / |v|^2
            let rr = (m.0 as f64 * vr + m.1 as f64 * vi) / v2;
            let ri = (m.1 as f64 * vr - m.0 as f64 * vi) / v2;
            sre[bi] += w * rr;
            sim[bi] += w * ri;
            swt[bi] += w;
        }
    }

    // Convert the sums into corrections.
    let mut nset = 0usize;
    for bi in 0..nb * nif {
        if swt[bi] <= 0.0 {
            continue;
        }
        let (re, im) = (sre[bi] / swt[bi], sim[bi] / swt[bi]);
        let amp = (re * re + im * im).sqrt();
        if amp <= 0.0 {
            continue;
        }
        bcor.amp[bi] *= amp as f32;
        bcor.phs[bi] += im.atan2(re) as f32;
        nset += 1;
    }
    ob.bcor = Some(bcor);
    // Re-apply calibration so the corrections take effect.
    if let Some(mut stream) = ob.stream.take() {
        stream.apply_calibration(ob);
        ob.stream = Some(stream);
    }
    nset
}

/// Undo all baseline corrections (difmap clroff).
pub fn clroff(ob: &mut Observation) {
    if let Some(bcor) = ob.bcor.as_mut() {
        bcor.reset();
    }
    if let Some(mut stream) = ob.stream.take() {
        stream.apply_calibration(ob);
        ob.stream = Some(stream);
    }
}
