//! Closure quantities: closure phases over antenna triangles
//! (difmap `cpplot`) and spectra of the selected stream.
//!
//! The closure phase of a triangle (a, b, c) is
//! arg(V_ab) + arg(V_bc) + arg(V_ca), a quantity unaffected by
//! antenna-based phase errors, so it is the classic diagnostic for
//! judging whether a model fits the data independently of
//! self-calibration.

use crate::obs::Observation;

/// Closure phases of one triangle, sampled in time.
pub struct ClosureSeries {
    /// Antenna indices (a < b < c) of the triangle.
    pub tri: (u32, u32, u32),
    /// Times (seconds since ref_mjd).
    pub time: Vec<f64>,
    /// Observed closure phase (radians).
    pub phase: Vec<f32>,
    /// Model closure phase (radians); NaN where the model is zero.
    pub model: Vec<f32>,
    /// Formal uncertainty of the closure phase (radians).
    pub error: Vec<f32>,
    /// IF index this series belongs to.
    pub cif: usize,
}

/// Compute closure phases for all (or one) triangle of the current
/// stream. Only integrations where all three baselines are unflagged
/// contribute.
///
/// `tri` optionally selects a single triangle; `cif` optionally
/// restricts to one IF.
pub fn closure_phases(
    ob: &Observation,
    tri: Option<(u32, u32, u32)>,
    cif_only: Option<usize>,
) -> Vec<ClosureSeries> {
    let stream = match ob.stream.as_ref() {
        Some(s) => s,
        None => return Vec::new(),
    };
    let nif = ob.nif();
    let nant = ob.antennas.len();

    // Index the rows of each integration by baseline, so that looking
    // up the three baselines of a triangle is O(1) rather than a scan
    // (which would make many-antenna arrays impractically slow).
    // Key: (a_lo, a_hi) -> (row, stored as (a_lo, a_hi)?)
    let mut rows_by_time: Vec<std::collections::HashMap<(u32, u32), (usize, bool)>> =
        vec![std::collections::HashMap::new(); ob.ntimes];
    for row in 0..ob.nrow {
        let (x, y) = (ob.ant1[row], ob.ant2[row]);
        rows_by_time[ob.time_idx[row] as usize]
            .insert((x.min(y), x.max(y)), (row, x < y));
    }

    // Enumerate the requested triangles.
    let mut triangles: Vec<(u32, u32, u32)> = Vec::new();
    match tri {
        Some((a, b, c)) => {
            let mut t = [a, b, c];
            t.sort_unstable();
            triangles.push((t[0], t[1], t[2]));
        }
        None => {
            for a in 0..nant {
                for b in (a + 1)..nant {
                    for c in (b + 1)..nant {
                        // Only triangles within one subarray are closed.
                        if ob.antennas[a].subarray == ob.antennas[b].subarray
                            && ob.antennas[b].subarray == ob.antennas[c].subarray
                        {
                            triangles.push((a as u32, b as u32, c as u32));
                        }
                    }
                }
            }
        }
    }

    let mut out = Vec::new();
    let cifs: Vec<usize> = match cif_only {
        Some(c) => vec![c],
        None => (0..nif).filter(|&c| stream.if_used[c]).collect(),
    };

    // The complete model, including any components held apart from the
    // stream model while they are being fitted.
    let model = crate::model::full_model_vis(ob);

    for &(a, b, c) in &triangles {
        for &cif in &cifs {
            if !stream.if_used[cif] {
                continue;
            }
            let mut series = ClosureSeries {
                tri: (a, b, c),
                time: Vec::new(),
                phase: Vec::new(),
                model: Vec::new(),
                error: Vec::new(),
                cif,
            };
            for it in 0..ob.ntimes {
                // Look up the three baselines of the triangle. `fwd`
                // records whether the stored visibility runs in the
                // direction the closure sum needs (a->b, b->c, c->a);
                // a >= b >= c ordering means the third leg is stored
                // as (a, c) and therefore always needs conjugating.
                let idx = &rows_by_time[it];
                let (Some(&(row_ab, fwd_ab)), Some(&(row_bc, fwd_bc)), Some(&(row_ac, fwd_ac))) =
                    (idx.get(&(a, b)), idx.get(&(b, c)), idx.get(&(a, c)))
                else {
                    continue;
                };
                // V_ca = conj(V_ac), so invert the stored direction.
                let fwd_ca = !fwd_ac;
                let row_ca = row_ac;
                let v_ab = stream.vis[row_ab * nif + cif];
                let v_bc = stream.vis[row_bc * nif + cif];
                let v_ca = stream.vis[row_ca * nif + cif];
                if v_ab.wt <= 0.0 || v_bc.wt <= 0.0 || v_ca.wt <= 0.0 {
                    continue;
                }
                let sgn = |fwd: bool| if fwd { 1.0f32 } else { -1.0f32 };
                let phs = sgn(fwd_ab) * v_ab.im.atan2(v_ab.re)
                    + sgn(fwd_bc) * v_bc.im.atan2(v_bc.re)
                    + sgn(fwd_ca) * v_ca.im.atan2(v_ca.re);
                // Wrap into -pi..pi.
                let tau = std::f32::consts::TAU;
                let phs = phs - tau * ((phs / tau + 0.5).floor());

                // Model closure phase, if a model exists.
                let m_ab = model[row_ab * nif + cif];
                let m_bc = model[row_bc * nif + cif];
                let m_ca = model[row_ca * nif + cif];
                let mphs = if (m_ab.0 != 0.0 || m_ab.1 != 0.0)
                    && (m_bc.0 != 0.0 || m_bc.1 != 0.0)
                    && (m_ca.0 != 0.0 || m_ca.1 != 0.0)
                {
                    let p = sgn(fwd_ab) * m_ab.1.atan2(m_ab.0)
                        + sgn(fwd_bc) * m_bc.1.atan2(m_bc.0)
                        + sgn(fwd_ca) * m_ca.1.atan2(m_ca.0);
                    p - tau * ((p / tau + 0.5).floor())
                } else {
                    f32::NAN
                };

                // Phase error of each visibility is ~1/(amp*sqrt(wt));
                // closure errors add in quadrature.
                let perr = |v: crate::stokes::Cvis| -> f32 {
                    let amp = (v.re * v.re + v.im * v.im).sqrt();
                    if amp > 0.0 && v.wt > 0.0 {
                        1.0 / (amp * v.wt.sqrt())
                    } else {
                        0.0
                    }
                };
                let e = (perr(v_ab).powi(2) + perr(v_bc).powi(2) + perr(v_ca).powi(2)).sqrt();

                series.time.push(ob.times[it]);
                series.phase.push(phs);
                series.model.push(mphs);
                series.error.push(e);
            }
            if !series.time.is_empty() {
                out.push(series);
            }
        }
    }
    out
}

/// A time-averaged spectrum of one baseline (or of all baselines).
pub struct Spectrum {
    /// Global channel indices.
    pub chan: Vec<usize>,
    /// Channel frequencies (Hz).
    pub freq: Vec<f64>,
    /// Vector-averaged visibility per channel.
    pub re: Vec<f32>,
    pub im: Vec<f32>,
    /// Scalar-averaged amplitude per channel.
    pub amp: Vec<f32>,
    /// Sum of weights per channel (0 where nothing was averaged).
    pub wt: Vec<f32>,
}

/// Time-average the raw (channel-resolved) visibilities of the current
/// polarization selection into a spectrum (difmap `specplot` data).
///
/// Note this deliberately covers *all* channels, not just the ones in
/// the current channel selection: the point of a spectrum is to show
/// what the unselected channels look like too. Only the polarization
/// combination of the selection is applied.
///
/// `baseline` optionally restricts to one baseline (global antenna
/// indices, unordered); `time_range` restricts the averaging window.
///
/// With `calibrated`, the antenna gains, baseline corrections and
/// accumulated phase-center shift are applied per channel, exactly as
/// `Stream::apply_calibration_rows` applies them to the selected
/// stream - otherwise the average would ignore self-cal and a
/// vector-averaged spectrum of calibrated data would be meaningless.
/// The channel's own frequency is used for the shift phase, where the
/// stream can only use its IF's effective frequency.
pub fn spectrum(
    ob: &Observation,
    baseline: Option<(u32, u32)>,
    time_range: Option<(f64, f64)>,
    calibrated: bool,
) -> Spectrum {
    let polop = match ob.stream.as_ref() {
        Some(s) => s.polop,
        None => return Spectrum {
            chan: Vec::new(),
            freq: Vec::new(),
            re: Vec::new(),
            im: Vec::new(),
            amp: Vec::new(),
            wt: Vec::new(),
        },
    };
    let nct = ob.nctotal;
    let mut sre = vec![0.0f64; nct];
    let mut sim = vec![0.0f64; nct];
    let mut samp = vec![0.0f64; nct];
    let mut swt = vec![0.0f64; nct];
    let mut pvis = vec![crate::stokes::Cvis::default(); ob.npol()];

    // Which IF each global channel belongs to, and its frequency.
    let mut chan_if = vec![0usize; nct];
    let mut freq = vec![0.0f64; nct];
    for (cif, band) in ob.ifs.iter().enumerate() {
        for ch in 0..band.nchan {
            chan_if[band.coff + ch] = cif;
            freq[band.coff + ch] = band.chan_freq(ch);
        }
    }
    let bcor = ob.bcor.as_ref().filter(|b| !b.is_identity());
    let (east, north) = (ob.geom.east, ob.geom.north);
    let doshift = calibrated && (east != 0.0 || north != 0.0);

    for row in 0..ob.nrow {
        if let Some((ba, bb)) = baseline {
            let (a1, a2) = (ob.ant1[row], ob.ant2[row]);
            if !((a1 == ba && a2 == bb) || (a1 == bb && a2 == ba)) {
                continue;
            }
        }
        if let Some((t0, t1)) = time_range {
            let t = ob.time[row];
            if t < t0 || t > t1 {
                continue;
            }
        }
        let it = ob.time_idx[row] as usize;
        let (a1, a2) = (ob.ant1[row] as usize, ob.ant2[row] as usize);
        for gc in 0..nct {
            ob.pvis(row, gc, &mut pvis);
            let mut v = polop.get(&pvis);
            if v.wt <= 0.0 {
                continue; // flagged or absent
            }
            if calibrated {
                let cif = chan_if[gc];
                let ia = ob.gains.idx(it, cif, a1);
                let ib = ob.gains.idx(it, cif, a2);
                if ob.gains.bad[ia] || ob.gains.bad[ib] {
                    continue; // an unusable solution flags the sample
                }
                let mut ampcor = ob.gains.amp[ia] * ob.gains.amp[ib];
                let mut phscor = ob.gains.phs[ia] - ob.gains.phs[ib];
                if let Some(bc) = bcor {
                    if let Some(k) = bc.index(ob.ant1[row], ob.ant2[row], cif) {
                        ampcor *= bc.amp[k];
                        phscor += bc.phs[k];
                    }
                }
                if doshift {
                    let f = freq[gc];
                    let (us, vs) = (ob.uvw[row * 3], ob.uvw[row * 3 + 1]);
                    phscor +=
                        (std::f64::consts::TAU * (us * f * east + vs * f * north)) as f32;
                }
                if ampcor != 1.0 || phscor != 0.0 {
                    let (s, c) = phscor.sin_cos();
                    let (re, im) = (v.re, v.im);
                    v.re = ampcor * (re * c - im * s);
                    v.im = ampcor * (re * s + im * c);
                    v.wt /= ampcor * ampcor;
                }
            }
            let w = v.wt as f64;
            sre[gc] += w * v.re as f64;
            sim[gc] += w * v.im as f64;
            samp[gc] += w * ((v.re * v.re + v.im * v.im).sqrt() as f64);
            swt[gc] += w;
        }
    }
    Spectrum {
        chan: (0..nct).collect(),
        freq,
        re: (0..nct)
            .map(|c| if swt[c] > 0.0 { (sre[c] / swt[c]) as f32 } else { 0.0 })
            .collect(),
        im: (0..nct)
            .map(|c| if swt[c] > 0.0 { (sim[c] / swt[c]) as f32 } else { 0.0 })
            .collect(),
        amp: (0..nct)
            .map(|c| if swt[c] > 0.0 { (samp[c] / swt[c]) as f32 } else { 0.0 })
            .collect(),
        wt: swt.iter().map(|&w| w as f32).collect(),
    }
}

/// Per-antenna, per-integration time sampling of the current stream
/// (difmap `tplot`): for each (integration, antenna) the number of
/// unflagged baselines it appears on, summed over used IFs.
pub fn sampling(ob: &Observation) -> Vec<u32> {
    let mut out = vec![0u32; ob.ntimes * ob.antennas.len()];
    let nant = ob.antennas.len();
    if let Some(stream) = ob.stream.as_ref() {
        let nif = ob.nif();
        for row in 0..ob.nrow {
            let it = ob.time_idx[row] as usize;
            let good = (0..nif)
                .filter(|&cif| stream.if_used[cif] && stream.vis[row * nif + cif].wt > 0.0)
                .count() as u32;
            if good > 0 {
                out[it * nant + ob.ant1[row] as usize] += good;
                out[it * nant + ob.ant2[row] as usize] += good;
            }
        }
    }
    out
}
