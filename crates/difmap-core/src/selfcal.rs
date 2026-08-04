//! Self-calibration: solve for antenna complex gain corrections that
//! make the calibrated data best fit the current model.
//!
//! Port of difmap's slfcal.c: the iterative solver of eq. 9.5 of
//! "Self-calibration" (Cornwell & Fomalont, in Synthesis Imaging in
//! Radio Astronomy, 1989), with per-solution-interval binning,
//! closure-based usability analysis, correction limits, gaussian
//! smoothing of solutions, and amplitude normalization.
//!
//! Each IF and subarray is solved independently, as in difmap.
//! Corrections are accumulated into the observation's [`GainTable`]
//! and the stream calibration is re-applied by the caller.

use crate::obs::Observation;
use crate::stokes::Cvis;

const NITER: usize = 100; // Max gradient-search iterations
const SLFGAIN: f32 = 0.5; // Loop gain of the gradient search
const EPSILON: f32 = 1.0e-6; // Acceptable relative change in residuals

/// Self-calibration options (difmap selfcal/selftaper/selflims/selfflag).
#[derive(Clone, Copy, Debug)]
pub struct SelfcalPars {
    /// Solve for amplitude corrections.
    pub doamp: bool,
    /// Solve for phase corrections.
    pub dophs: bool,
    /// Allow the overall flux scale to float (skip normalization).
    pub dofloat: bool,
    /// Solution interval (minutes); <= 1/60 means per-integration.
    pub solint: f32,
    /// One single solution for the whole time range (difmap gscale).
    pub doone: bool,
    /// selftaper: weight down short baselines by 1-gaussian.
    pub gauval: f32,
    pub gaurad: f32,
    /// selflims: reject solutions with amp_cor outside
    /// [1/maxamp, maxamp] (if maxamp > 1) or |phs_cor| > maxphs
    /// (radians, if maxphs > 0).
    pub maxamp: f32,
    pub maxphs: f32,
    /// UV radius range (wavelengths); applied if max > 0.
    pub uvmin: f32,
    pub uvmax: f32,
    /// Minimum number of closed telescopes required for a solution
    /// (difmap selfflag: typically 3 for phase, 4 for amplitude).
    pub mintel: usize,
    /// Flag the gains of uncorrectable telescopes.
    pub doflag: bool,
}

impl Default for SelfcalPars {
    fn default() -> Self {
        SelfcalPars {
            doamp: false,
            dophs: true,
            dofloat: false,
            solint: 0.0,
            doone: false,
            gauval: 0.0,
            gaurad: 0.0,
            maxamp: 0.0,
            maxphs: 0.0,
            uvmin: 0.0,
            uvmax: 0.0,
            mintel: 3,
            doflag: false,
        }
    }
}

/// Summary of a self-cal run.
#[derive(Clone, Debug, Default)]
pub struct SelfcalResult {
    /// Number of unusable solution intervals per (subarray, IF).
    pub nbadsol: usize,
    /// Number of telescope corrections flagged/ignored.
    pub nbadtel: usize,
    /// Amplitude normalization factors applied, one per (subarray, IF).
    pub norms: Vec<f64>,
    /// Total number of solution bins processed.
    pub nbins: usize,
}

#[derive(thiserror::Error, Debug)]
pub enum SelfcalError {
    #[error("no stream selected; call select() first")]
    NoStream,
    #[error("selfcal requires a model; run clean or set a model first")]
    NoModel,
}

/// A per-telescope correction from one solution interval.
#[derive(Clone, Copy, Default)]
struct Cor {
    amp_cor: f32,
    phs_cor: f32,
    weight: f32,
}

/// One solution bin.
struct Soln {
    begut: f64,
    endut: f64,
    cors: Vec<Cor>,
}

/// Precomputed per-subarray structures.
struct SubData {
    /// Global antenna indexes of this subarray's stations.
    ants: Vec<usize>,
    /// Integration (global time) indexes sampled by this subarray.
    itimes: Vec<usize>,
    /// Rows of each integration: (row, local_a, local_b).
    rows: Vec<Vec<(usize, usize, usize)>>,
}

fn subarray_data(ob: &Observation, isub: u32) -> SubData {
    let nant = ob.antennas.len();
    let mut local = vec![None; nant];
    let mut ants = Vec::new();
    for (ia, a) in ob.antennas.iter().enumerate() {
        if a.subarray == isub {
            local[ia] = Some(ants.len());
            ants.push(ia);
        }
    }
    let mut itimes: Vec<usize> = Vec::new();
    let mut rows: Vec<Vec<(usize, usize, usize)>> = Vec::new();
    for row in 0..ob.nrow {
        let a1 = ob.ant1[row] as usize;
        if ob.antennas[a1].subarray != isub {
            continue;
        }
        let it = ob.time_idx[row] as usize;
        if itimes.last() != Some(&it) {
            itimes.push(it);
            rows.push(Vec::new());
        }
        let la = local[a1].unwrap();
        let lb = local[ob.ant2[row] as usize].unwrap();
        rows.last_mut().unwrap().push((row, la, lb));
    }
    SubData { ants, itimes, rows }
}

/// Residual of the current gain estimates (port of slfdif()).
fn slfdif(nvis: &[Cvis], gain: &[(f32, f32)], nstat: usize) -> f32 {
    let mut resid = 0.0f32;
    let mut wtsum = 0.0f32;
    for ita in 0..nstat {
        let ga = gain[ita];
        for itb in 0..nstat {
            let gb = gain[itb];
            let c = nvis[ita * nstat + itb];
            let re = ga.0 * gb.0 + ga.1 * gb.1 - c.re;
            let im = ga.1 * gb.0 - ga.0 * gb.1 - c.im;
            resid += c.wt * (re * re + im * im);
            wtsum += c.wt;
        }
    }
    if resid > 0.0 && wtsum > 0.0 {
        resid / wtsum
    } else {
        0.0
    }
}

/// One gain-update step (port of getgain()).
/// `gain` entries are (re, im); `gwt` receives the solution weights.
#[allow(clippy::too_many_arguments)]
fn getgain(
    nvis: &[Cvis],
    gain: &mut [(f32, f32)],
    gwt: &mut [f32],
    fixed: &[bool],
    nstat: usize,
    doamp: bool,
    dophs: bool,
    slfgain: f32,
) {
    let old: Vec<(f32, f32)> = gain.to_vec();
    let mut gnew = old.clone();
    let mut nwt = vec![0.0f32; nstat];
    for ita in 0..nstat {
        let ga = old[ita];
        let mut top = (0.0f32, 0.0f32);
        let mut bot = 0.0f32;
        let mut wt_sum = 0.0f32;
        for itb in 0..nstat {
            let c = nvis[ita * nstat + itb];
            if c.wt > 0.0 {
                let gb = old[itb];
                top.0 += c.wt * (gb.0 * c.re - gb.1 * c.im);
                top.1 += c.wt * (gb.0 * c.im + gb.1 * c.re);
                bot += c.wt * (gb.0 * gb.0 + gb.1 * gb.1);
                wt_sum += c.wt;
            }
        }
        if bot > 0.0 {
            gnew[ita] = (
                (1.0 - slfgain) * ga.0 + slfgain * top.0 / bot,
                (1.0 - slfgain) * ga.1 + slfgain * top.1 / bot,
            );
            nwt[ita] = wt_sum;
        }
        if bot <= 0.0 || (gnew[ita].0 == 0.0 && gnew[ita].1 == 0.0) {
            gnew[ita] = ga;
            nwt[ita] = gwt[ita];
        }
    }
    // Enforce fixed antennas and the doamp/dophs restrictions.
    for ita in 0..nstat {
        let g = &mut gnew[ita];
        if nwt[ita] > 0.0 {
            let amp = (g.0 * g.0 + g.1 * g.1).sqrt();
            if fixed[ita] {
                *g = (1.0, 0.0);
            } else if !dophs {
                *g = (amp, 0.0);
            } else if !doamp {
                *g = (g.0 / amp, g.1 / amp);
            }
        }
        gain[ita] = *g;
        gwt[ita] = nwt[ita];
    }
}

/// Usability/closure analysis of one integration (port of
/// get_usable()/count_tel()). Marks `usable` per row-triplet and
/// returns the number of solvable telescopes.
#[allow(clippy::too_many_arguments)]
fn get_usable(
    ob: &Observation,
    stream_vis: &[Cvis],
    nif: usize,
    cif: usize,
    rows: &[(usize, usize, usize)],
    nstat: usize,
    pars: &SelfcalPars,
    freq: f64,
    usable: &mut Vec<bool>,
    telnum: &mut [usize],
) -> usize {
    usable.clear();
    let docut = pars.uvmin.max(pars.uvmax) > 0.0;
    let (uvmin, uvmax) = if pars.uvmin <= pars.uvmax {
        (pars.uvmin as f64, pars.uvmax as f64)
    } else {
        (pars.uvmax as f64, pars.uvmin as f64)
    };
    for &(row, _, _) in rows {
        let v = stream_vis[row * nif + cif];
        let mut ok = v.wt > 0.0;
        if ok && docut {
            let uu = ob.uvw[row * 3] * freq;
            let vv = ob.uvw[row * 3 + 1] * freq;
            let uvrad = (uu * uu + vv * vv).sqrt();
            ok = uvrad >= uvmin && uvrad <= uvmax;
        }
        usable.push(ok);
    }
    // Count usable baselines per telescope.
    telnum.iter_mut().for_each(|t| *t = 0);
    for (i, &(_, la, lb)) in rows.iter().enumerate() {
        if usable[i] {
            telnum[la] += 1;
            telnum[lb] += 1;
        }
    }
    // Closure pruning: iteratively remove telescopes with only one
    // usable baseline (needed when mintel > 2).
    if pars.mintel > 2 {
        for itel in 0..nstat {
            let mut newtel = itel;
            while telnum[newtel] == 1 {
                for (i, &(_, la, lb)) in rows.iter().enumerate() {
                    if usable[i] && (la == newtel || lb == newtel) {
                        usable[i] = false;
                        telnum[la] -= 1;
                        telnum[lb] -= 1;
                        newtel = if la == newtel { lb } else { la };
                        break;
                    }
                }
            }
        }
    }
    let ntel = telnum.iter().filter(|&&n| n > 0).count();
    if ntel < pars.mintel {
        usable.iter_mut().for_each(|u| *u = false);
        telnum.iter_mut().for_each(|t| *t = 0);
        0
    } else {
        ntel
    }
}

/// Accumulate model-normalized visibility ratios of one integration
/// into the nstat x nstat matrix (port of sum_ratios()).
#[allow(clippy::too_many_arguments)]
fn sum_ratios(
    ob: &Observation,
    stream_vis: &[Cvis],
    stream_model: &[(f32, f32)],
    nif: usize,
    cif: usize,
    rows: &[(usize, usize, usize)],
    usable: &[bool],
    antwt: &[f32],
    gfac: f64,
    freq: f64,
    nvis: &mut [Cvis],
    nstat: usize,
) {
    for (i, &(row, la, lb)) in rows.iter().enumerate() {
        if !usable[i] {
            continue;
        }
        let v = stream_vis[row * nif + cif];
        let m = stream_model[row * nif + cif];
        let modamp2 = m.0 * m.0 + m.1 * m.1;
        if modamp2 == 0.0 {
            continue;
        }
        let mut wt = v.wt * modamp2;
        if gfac < 0.0 {
            let uu = ob.uvw[row * 3] * freq;
            let vv = ob.uvw[row * 3 + 1] * freq;
            wt *= (1.0 - (gfac * (uu * uu + vv * vv)).exp()) as f32;
        }
        wt *= (antwt[la] * antwt[lb]).abs();
        // ratio = wt * V_obs / V_model
        let re = wt * (v.re * m.0 + v.im * m.1) / modamp2;
        let im = wt * (v.im * m.0 - v.re * m.1) / modamp2;
        let c = &mut nvis[la * nstat + lb];
        c.re += re;
        c.im += im;
        c.wt += wt;
        let c = &mut nvis[lb * nstat + la];
        c.re += re;
        c.im -= im;
        c.wt += wt;
    }
}

/// Convert reciprocal gains to corrections, checking limits (port of
/// get_cors()). `isbad` marks the solution unusable on input (e.g.
/// because the fit diverged); returns true if the solution is
/// unusable, in which case unit zero-weight corrections are stored.
fn get_cors(
    mut isbad: bool,
    gain: &[(f32, f32)],
    gwt: &[f32],
    pars: &SelfcalPars,
    cors: &mut [Cor],
) -> bool {
    let doplim = pars.dophs && pars.maxphs > 0.0;
    let doalim = pars.doamp && pars.maxamp > 1.0;
    let minamp = if doalim { 1.0 / pars.maxamp } else { 0.0 };
    for (i, c) in cors.iter_mut().enumerate() {
        if isbad {
            break;
        }
        let g = gain[i];
        if g.0 == 0.0 && g.1 == 0.0 {
            *c = Cor {
                amp_cor: 1.0,
                phs_cor: 0.0,
                weight: 0.0,
            };
        } else {
            c.amp_cor = 1.0 / (g.0 * g.0 + g.1 * g.1).sqrt();
            c.phs_cor = -g.1.atan2(g.0);
            c.weight = gwt[i];
            if (doplim && c.phs_cor.abs() > pars.maxphs)
                || (doalim && (c.amp_cor > pars.maxamp || c.amp_cor < minamp))
            {
                isbad = true;
            }
        }
    }
    if isbad {
        for c in cors.iter_mut() {
            *c = Cor {
                amp_cor: 1.0,
                phs_cor: 0.0,
                weight: 0.0,
            };
        }
    }
    isbad
}

/// Apply per-telescope corrections to the gain table for the given
/// integrations (the equivalent of apply_cors(), but accumulating in
/// the gain table rather than mutating visibilities directly).
fn apply_cors(
    ob: &mut Observation,
    sd: &SubData,
    cif: usize,
    ita: usize,
    itb: usize,
    doamp: bool,
    dophs: bool,
    cors: &[Cor],
) {
    for &it in &sd.itimes[ita..=itb] {
        for (li, &ga) in sd.ants.iter().enumerate() {
            let gi = ob.gains.idx(it, cif, ga);
            if dophs {
                ob.gains.phs[gi] += cors[li].phs_cor;
            }
            if doamp {
                ob.gains.amp[gi] *= cors[li].amp_cor;
            }
            if cors[li].weight > 0.0 {
                ob.gains.used[gi] = true;
            }
        }
    }
}

/// Area under a unit gaussian of std-dev `sigma` between xa and xb
/// (port of get_area(); a direct rational approximation of erf/2 is
/// used instead of difmap's 16-entry interpolation table).
fn get_area(xa: f64, xb: f64, sigma: f64) -> f64 {
    const NSIGMA: f64 = 2.5;
    let s2 = std::f64::consts::SQRT_2;
    let zmax = NSIGMA / s2;
    let erf_half = |z: f64| -> f64 {
        let z = z.min(zmax);
        let t = 1.0 / (1.0 + 0.47047 * z);
        0.5 - (0.1740121 * t * (1.0 + -0.2754975 * t * (1.0 + -7.7999287 * t))) * (-z * z).exp()
    };
    let za = xa / (s2 * sigma);
    let zb = xb / (s2 * sigma);
    let asgn = if za < 0.0 { -1.0 } else { 1.0 };
    let bsgn = if zb < 0.0 { -1.0 } else { 1.0 };
    (asgn * erf_half(za.abs()) - bsgn * erf_half(zb.abs())).abs()
}

/// Self-calibrate all subarrays and all used IFs of the current
/// stream. On success the gain table has been updated and the stream
/// re-calibrated.
pub fn selfcal(ob: &mut Observation, pars: &SelfcalPars) -> Result<SelfcalResult, SelfcalError> {
    if ob.stream.is_none() {
        return Err(SelfcalError::NoStream);
    }
    // Establish the tentative model first (difmap does the same).
    crate::model::merge_model(ob);
    if ob.model.is_empty() {
        return Err(SelfcalError::NoModel);
    }
    let nif = ob.nif();
    let mut result = SelfcalResult::default();

    // Take the stream out of ob to satisfy the borrow checker; it is
    // restored (and recalibrated) before returning.
    let mut stream = ob.stream.take().unwrap();

    let utint = {
        let s = pars.solint as f64 * 60.0;
        if s <= 1.0 {
            0.0
        } else {
            s
        }
    };

    for isub in 0..ob.nsub as u32 {
        let sd = subarray_data(ob, isub);
        let nstat = sd.ants.len();
        if nstat < 2 || sd.itimes.is_empty() {
            continue;
        }
        let fixed: Vec<bool> = sd.ants.iter().map(|&a| ob.antennas[a].fixed).collect();
        let antwt: Vec<f32> = sd.ants.iter().map(|&a| ob.antennas[a].weight).collect();

        for cif in 0..nif {
            if !stream.if_used[cif] {
                continue;
            }
            let freq = stream.if_freq[cif];
            //

            let gfac = if pars.gaurad > 0.0 && pars.gauval > 0.0 && pars.gauval < 1.0 {
                ((1.0 - pars.gauval) as f64).ln() / (pars.gaurad as f64 * pars.gaurad as f64)
            } else {
                0.0
            };

            let mut solns: Vec<Soln> = Vec::new();
            let mut nvis = vec![Cvis::default(); nstat * nstat];
            let mut usable: Vec<bool> = Vec::new();
            let mut telnum = vec![0usize; nstat];
            let mut gain = vec![(1.0f32, 0.0f32); nstat];
            let mut gwt = vec![0.0f32; nstat];

            let ntime_sub = sd.itimes.len();
            let mut uta = 0usize;
            while uta < ntime_sub {
                // Find the end of this solution bin (port of endbin()).
                let utb = if pars.doone {
                    ntime_sub - 1
                } else if utint > 0.0 {
                    let t0 = ob.times[sd.itimes[uta]];
                    let endut = utint * (t0 / utint).floor() + utint;
                    let mut utb = uta;
                    while utb < ntime_sub && ob.times[sd.itimes[utb]] <= endut {
                        utb += 1;
                    }
                    utb.saturating_sub(1).max(uta)
                } else {
                    uta
                };

                let ta = ob.times[sd.itimes[uta]];
                let tb = ob.times[sd.itimes[utb]];
                let utmid = ta + (tb - ta) / 2.0;

                nvis.iter_mut().for_each(|c| *c = Cvis::default());
                let mut n_ut = 0usize;
                for ut in uta..=utb {
                    let rows = &sd.rows[ut];
                    let ntel = get_usable(
                        ob, &stream.vis, nif, cif, rows, nstat, pars, freq, &mut usable,
                        &mut telnum,
                    );
                    if ntel >= pars.mintel {
                        n_ut += 1;
                        sum_ratios(
                            ob,
                            &stream.vis,
                            &stream.model,
                            nif,
                            cif,
                            rows,
                            &usable,
                            &antwt,
                            gfac,
                            freq,
                            &mut nvis,
                            nstat,
                        );
                    } else if pars.doflag {
                        // Flag gains of uncorrectable telescopes.
                        let it = sd.itimes[ut];
                        for (li, &ga) in sd.ants.iter().enumerate() {
                            if telnum[li] == 0 && !fixed[li] {
                                let gi = ob.gains.idx(it, cif, ga);
                                if !ob.gains.bad[gi] {
                                    ob.gains.bad[gi] = true;
                                    result.nbadtel += 1;
                                }
                            }
                        }
                    }
                }

                let mut soln = Soln {
                    begut: utmid - utint / 2.0,
                    endut: utmid + utint / 2.0,
                    cors: vec![Cor::default(); nstat],
                };

                if n_ut > 0 {
                    // Weighted sums -> weighted means.
                    for c in nvis.iter_mut() {
                        if c.wt > 0.0 {
                            c.re /= c.wt;
                            c.im /= c.wt;
                        }
                    }
                    // Iterative solution.
                    gain.iter_mut().for_each(|g| *g = (1.0, 0.0));
                    gwt.iter_mut().for_each(|w| *w = 0.0);
                    let ini_res = slfdif(&nvis, &gain, nstat);
                    getgain(
                        &nvis, &mut gain, &mut gwt, &fixed, nstat, pars.doamp, pars.dophs, 1.0,
                    );
                    let mut old_res = slfdif(&nvis, &gain, nstat);
                    let mut new_res = old_res;
                    for _ in 0..NITER {
                        getgain(
                            &nvis, &mut gain, &mut gwt, &fixed, nstat, pars.doamp, pars.dophs,
                            SLFGAIN,
                        );
                        new_res = slfdif(&nvis, &gain, nstat);
                        if (new_res - old_res).abs() <= EPSILON * ini_res {
                            break;
                        }
                        old_res = new_res;
                    }
                    let isbad = get_cors(ini_res < new_res, &gain, &gwt, pars, &mut soln.cors);
                    if isbad {
                        result.nbadsol += 1;
                    } else if pars.doone || utint <= 0.0 {
                        apply_cors(ob, &sd, cif, uta, utb, pars.doamp, pars.dophs, &soln.cors);
                    }
                }
                result.nbins += 1;
                solns.push(soln);
                uta = utb + 1;
            }

            // Smooth/interpolate binned solutions onto integrations
            // (port of apply_solns()).
            if utint > 0.0 && !pars.doone {
                let sigma = pars.solint as f64 * 0.37478125;
                let maxoff = 2.5 * sigma;
                let mut sa = 0usize;
                for ut in 0..ntime_sub {
                    let utval = ob.times[sd.itimes[ut]];
                    while sa < solns.len() && (utval - solns[sa].endut) / 60.0 >= maxoff {
                        sa += 1;
                    }
                    let mut cors = vec![Cor::default(); nstat];
                    let mut sb = sa;
                    while sb < solns.len() && (solns[sb].begut - utval) / 60.0 < maxoff {
                        let b_start = ((solns[sb].begut - utval) / 60.0).max(-maxoff);
                        let b_end = ((solns[sb].endut - utval) / 60.0).min(maxoff);
                        let area = get_area(b_start, b_end, sigma) as f32;
                        for (li, c) in cors.iter_mut().enumerate() {
                            let icor = &solns[sb].cors[li];
                            if icor.weight > 0.0 {
                                let wt = area * icor.weight;
                                c.amp_cor += wt * icor.amp_cor;
                                c.phs_cor += wt * icor.phs_cor;
                                c.weight += wt;
                            }
                        }
                        sb += 1;
                    }
                    for c in cors.iter_mut() {
                        if c.weight > 0.0 {
                            c.amp_cor /= c.weight;
                            c.phs_cor /= c.weight;
                        } else {
                            c.amp_cor = 1.0;
                            c.phs_cor = 0.0;
                        }
                    }
                    apply_cors(ob, &sd, cif, ut, ut, pars.doamp, pars.dophs, &cors);
                }
            }

            // Amplitude normalization (port of norm_cors()).
            if pars.doamp && !pars.dofloat {
                let mut amp_sum = 0.0f64;
                let mut namp = 0usize;
                for &it in &sd.itimes {
                    for (li, &ga) in sd.ants.iter().enumerate() {
                        let gi = ob.gains.idx(it, cif, ga);
                        if ob.gains.used[gi] && !fixed[li] {
                            amp_sum += ob.gains.amp[gi] as f64;
                            namp += 1;
                        }
                    }
                }
                let norm = if namp > 0 {
                    namp as f64 / amp_sum
                } else {
                    1.0
                };
                if namp > 0 {
                    for &it in &sd.itimes {
                        for (li, &ga) in sd.ants.iter().enumerate() {
                            if !fixed[li] {
                                let gi = ob.gains.idx(it, cif, ga);
                                ob.gains.amp[gi] *= norm as f32;
                            }
                        }
                    }
                }
                result.norms.push(norm);
            }
        }
    }

    // Re-apply the updated calibration to the stream.
    stream.apply_calibration(ob);
    ob.stream = Some(stream);
    Ok(result)
}
