//! Model fitting: least-squares fit of model components to the
//! visibilities of the current stream (difmap `modelfit`).
//!
//! Port of difmap's modfit.c + lmfit.c: Levenberg-Marquardt
//! minimization of chi-squared over the real and imaginary parts of
//! the residual visibilities, with difmap's parameterization:
//!
//! * positions and axes are re-normalized by the maximum UV radius so
//!   that the Hessian stays well conditioned,
//! * the elliptical shape is fitted as (X, Y, Z) with
//!   X = a²/2·(1-r²)·cos2φ, Y = a²/2·(1-r²)·sin2φ, Z = a²/2·(1+r²),
//!   which removes the degeneracy/limits of (major, ratio, phi),
//! * the *established* model is subtracted first; only components
//!   whose `freepar` bits are set are varied.

use crate::model::{besj0, besj1, besj2, CmpType, ModComp};
use crate::obs::Observation;

/// Free-parameter bits (difmap M_FLUX etc.).
pub const M_FLUX: u32 = 1;
pub const M_CENT: u32 = 2;
pub const M_MAJOR: u32 = 4;
pub const M_RATIO: u32 = 8;
pub const M_PHI: u32 = 16;
pub const M_SPCIND: u32 = 32;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const PI: f64 = std::f64::consts::PI;

#[derive(thiserror::Error, Debug)]
pub enum FitError {
    #[error("no stream selected; call select() first")]
    NoStream,
    #[error("no free parameters to fit")]
    NoFreePars,
    #[error("no usable visibilities in the fit range")]
    NoData,
    #[error("fewer measurements ({0}) than free parameters ({1})")]
    Underdetermined(usize, usize),
    #[error("rectangular components are not supported by modelfit")]
    BadType,
    #[error("the fit became singular")]
    Singular,
}

/// Result of a model fit.
#[derive(Clone, Debug)]
pub struct FitResult {
    /// Reduced chi-squared of the best fit.
    pub rchisq: f64,
    /// Chi-squared of the best fit.
    pub chisq: f64,
    /// Degrees of freedom (2*nvis - nfree).
    pub ndfree: i64,
    /// Number of visibilities used.
    pub nvis: usize,
    /// Number of free parameters.
    pub nfree: usize,
    /// Iterations that improved the fit.
    pub nbetter: usize,
    /// 1-sigma uncertainties of the fitted component parameters, in
    /// the same (flux, x, y, major, ratio, phi, spcind) order used by
    /// [`fit_uvmodel`]'s component list; NaN where not fitted.
    pub errors: Vec<CompErrors>,
}

/// Formal 1-sigma uncertainties of one component (NaN if not free).
#[derive(Clone, Copy, Debug)]
pub struct CompErrors {
    pub flux: f64,
    pub x: f64,
    pub y: f64,
    pub major: f64,
    pub ratio: f64,
    pub phi: f64,
    pub spcind: f64,
}

impl Default for CompErrors {
    fn default() -> Self {
        CompErrors {
            flux: f64::NAN,
            x: f64::NAN,
            y: f64::NAN,
            major: f64::NAN,
            ratio: f64::NAN,
            phi: f64::NAN,
            spcind: f64::NAN,
        }
    }
}

/// One usable visibility for the fit.
struct FitVis {
    uu: f64, // wavelengths
    vv: f64,
    re: f64, // residual after the established model
    im: f64,
    wt: f64,
    freq: f64,
}

/// The mapping of a component's free parameters onto the parameter
/// vector, and how many slots each occupies.
struct CompMap {
    /// Index of the component in the fitted list.
    icmp: usize,
    /// Offset of this component's first parameter in the vector.
    off: usize,
    freepar: u32,
}

fn count_free(freepar: u32) -> usize {
    let mut n = 0;
    if freepar & M_FLUX != 0 {
        n += 1;
    }
    if freepar & M_CENT != 0 {
        n += 2;
    }
    if freepar & (M_MAJOR | M_RATIO | M_PHI) != 0 {
        // Z always; X and Y only when the axial ratio is free.
        n += if freepar & M_RATIO != 0 { 3 } else { 1 };
    }
    if freepar & M_SPCIND != 0 {
        n += 1;
    }
    n
}

/// Pack component parameters into the normalized parameter vector
/// (port of getfree()).
fn get_free(comps: &[ModComp], maps: &[CompMap], uvrmax: f64, pars: &mut [f64]) {
    for m in maps {
        let c = &comps[m.icmp];
        let mut k = m.off;
        if m.freepar & M_FLUX != 0 {
            pars[k] = c.flux as f64;
            k += 1;
        }
        if m.freepar & M_CENT != 0 {
            pars[k] = c.x as f64 * uvrmax;
            pars[k + 1] = c.y as f64 * uvrmax;
            k += 2;
        }
        if m.freepar & (M_MAJOR | M_RATIO | M_PHI) != 0 {
            let anorm = c.major as f64 * uvrmax;
            let half_aa = 0.5 * anorm * anorm;
            let gg = (c.ratio as f64) * (c.ratio as f64);
            if m.freepar & M_RATIO != 0 {
                let s = half_aa * (1.0 - gg);
                pars[k] = s * (2.0 * c.phi as f64).cos();
                pars[k + 1] = s * (2.0 * c.phi as f64).sin();
                k += 2;
            }
            pars[k] = half_aa * (1.0 + gg);
            k += 1;
        }
        if m.freepar & M_SPCIND != 0 {
            pars[k] = c.spcind as f64;
        }
    }
}

/// Unpack the parameter vector into component parameters, enforcing
/// physical limits (port of setfree()).
fn set_free(comps: &mut [ModComp], maps: &[CompMap], uvrmax: f64, pars: &[f64]) {
    for m in maps {
        let c = &mut comps[m.icmp];
        let mut k = m.off;
        if m.freepar & M_FLUX != 0 {
            c.flux = pars[k] as f32;
            k += 1;
        }
        if m.freepar & M_CENT != 0 {
            c.x = (pars[k] / uvrmax) as f32;
            c.y = (pars[k + 1] / uvrmax) as f32;
            k += 2;
        }
        if m.freepar & (M_MAJOR | M_RATIO | M_PHI) != 0 {
            let renorm = uvrmax * uvrmax;
            if m.freepar & M_RATIO != 0 {
                let x = pars[k] / renorm;
                let y = pars[k + 1] / renorm;
                let mut z = pars[k + 2] / renorm;
                let xyrad = (x * x + y * y).sqrt();
                // An axial ratio must be real: clamp Z.
                if z < xyrad {
                    z = xyrad;
                }
                let major = (z + xyrad).abs().sqrt();
                c.major = major as f32;
                c.ratio = if major == 0.0 {
                    1.0
                } else {
                    ((z - xyrad).abs().sqrt() / major) as f32
                };
                c.phi = if x == 0.0 && y == 0.0 {
                    0.0
                } else {
                    (0.5 * y.atan2(x)) as f32
                };
                k += 3;
            } else {
                let z = pars[k] / renorm;
                c.major = z.abs().sqrt() as f32;
                k += 1;
            }
        }
        if m.freepar & M_SPCIND != 0 {
            c.spcind = pars[k] as f32;
        }
    }
}

/// Model visibility and its derivatives wrt the free parameters at one
/// UV point (port of getmodvis()). Returns (model re, model im).
fn model_and_grad(
    comps: &[ModComp],
    maps: &[CompMap],
    uvrmax: f64,
    v: &FitVis,
    dre: &mut [f64],
    dim: &mut [f64],
) -> Result<(f64, f64), FitError> {
    let (uu, vv) = (v.uu, v.vv);
    let uun = uu / uvrmax;
    let vvn = vv / uvrmax;
    let mut mre = 0.0;
    let mut mim = 0.0;
    dre.fill(0.0);
    dim.fill(0.0);
    for m in maps {
        let c = &comps[m.icmp];
        let cmpphs = TWO_PI * (uu * c.x as f64 + vv * c.y as f64);
        let (sinphi, cosphi) = (c.phi as f64).sin_cos();
        let tmpa = (uu * cosphi - vv * sinphi) * c.ratio as f64;
        let tmpb = uu * sinphi + vv * cosphi;
        let uvrad = (PI * c.major as f64 * (tmpa * tmpa + tmpb * tmpb).sqrt()).max(1.0e-9);
        let si = if c.spcind == 0.0 {
            1.0
        } else {
            (v.freq / c.freq0 as f64).powf(c.spcind as f64)
        };
        let flux = c.flux as f64 * si;
        let cmpamp = match c.ctype {
            CmpType::Delta => flux,
            CmpType::Gaussian => {
                flux * if uvrad < 12.0 {
                    (-0.3606737602 * uvrad * uvrad).exp()
                } else {
                    0.0
                }
            }
            CmpType::Disk => 2.0 * flux * besj1(uvrad) / uvrad,
            CmpType::Ellipse => {
                3.0 * flux * (uvrad.sin() - uvrad * uvrad.cos()) / (uvrad * uvrad * uvrad)
            }
            CmpType::Ring => flux * besj0(uvrad),
            CmpType::Sz => flux * if uvrad < 50.0 { (-uvrad).exp() } else { 0.0 } / uvrad,
            CmpType::Rectangle => return Err(FitError::BadType),
        };
        let (s, cph) = cmpphs.sin_cos();
        let cmpre = cmpamp * cph;
        let cmpim = cmpamp * s;
        mre += cmpre;
        mim += cmpim;

        let mut k = m.off;
        if m.freepar & M_FLUX != 0 {
            // d/dflux is the component divided by its flux.
            if c.flux != 0.0 {
                dre[k] = cmpre / c.flux as f64;
                dim[k] = cmpim / c.flux as f64;
            } else {
                // Flux is zero: derivative is the unit-flux component.
                let unit = if cmpamp == 0.0 { si } else { cmpamp };
                dre[k] = unit * cph;
                dim[k] = unit * s;
            }
            k += 1;
        }
        if m.freepar & M_CENT != 0 {
            dre[k] = TWO_PI * uun * -cmpim;
            dim[k] = TWO_PI * uun * cmpre;
            dre[k + 1] = TWO_PI * vvn * -cmpim;
            dim[k + 1] = TWO_PI * vvn * cmpre;
            k += 2;
        }
        if m.freepar & (M_MAJOR | M_RATIO | M_PHI) != 0 {
            let comfac = match c.ctype {
                CmpType::Delta => 0.0,
                CmpType::Gaussian => -0.7213475204 * uvrad,
                CmpType::Disk => -2.0 * besj2(uvrad) / uvrad,
                CmpType::Ellipse => {
                    (9.0 * uvrad.cos() / uvrad - 9.0 * uvrad.sin() / (uvrad * uvrad)
                        + 3.0 * uvrad.sin())
                        / uvrad
                        / uvrad
                }
                CmpType::Ring => -besj1(uvrad),
                CmpType::Sz => {
                    -(if uvrad < 50.0 { (-uvrad).exp() } else { 0.0 }) * (uvrad + 1.0)
                        / uvrad
                        / uvrad
                }
                CmpType::Rectangle => return Err(FitError::BadType),
            };
            let newfac = comfac * 0.5 * PI * PI / uvrad;
            if m.freepar & M_RATIO != 0 {
                let t = newfac * (vvn - uun) * (vvn + uun);
                dre[k] = cmpre * t;
                dim[k] = cmpim * t;
                let t = newfac * (2.0 * uun * vvn);
                dre[k + 1] = cmpre * t;
                dim[k + 1] = cmpim * t;
                k += 2;
            }
            let t = newfac * (vvn * vvn + uun * uun);
            dre[k] = cmpre * t;
            dim[k] = cmpim * t;
            k += 1;
        }
        if m.freepar & M_SPCIND != 0 {
            let factor = (v.freq / c.freq0 as f64).ln();
            dre[k] = cmpre * factor;
            dim[k] = cmpim * factor;
        }
    }
    Ok((mre, mim))
}

/// Gauss-Jordan solve of A x = b, in place (difmap gj_solve()).
fn gj_solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Result<(), FitError> {
    let n = b.len();
    let mut indxc = vec![0usize; n];
    let mut indxr = vec![0usize; n];
    let mut ipiv = vec![false; n];
    for _ in 0..n {
        // Find the pivot.
        let mut big = 0.0;
        let (mut irow, mut icol) = (0usize, 0usize);
        for j in 0..n {
            if ipiv[j] {
                continue;
            }
            for k in 0..n {
                if !ipiv[k] && a[j][k].abs() >= big {
                    big = a[j][k].abs();
                    irow = j;
                    icol = k;
                }
            }
        }
        if big == 0.0 {
            return Err(FitError::Singular);
        }
        ipiv[icol] = true;
        if irow != icol {
            a.swap(irow, icol);
            b.swap(irow, icol);
        }
        indxr[icol] = irow;
        indxc[icol] = icol;
        let piv = a[icol][icol];
        if piv == 0.0 {
            return Err(FitError::Singular);
        }
        let pivinv = 1.0 / piv;
        a[icol][icol] = 1.0;
        for v in a[icol].iter_mut() {
            *v *= pivinv;
        }
        b[icol] *= pivinv;
        for r in 0..n {
            if r == icol {
                continue;
            }
            let f = a[r][icol];
            if f == 0.0 {
                continue;
            }
            a[r][icol] = 0.0;
            for c in 0..n {
                a[r][c] -= a[icol][c] * f;
            }
            b[r] -= b[icol] * f;
        }
    }
    Ok(())
}

/// Invert a symmetric positive matrix by Gauss-Jordan elimination
/// (used for the covariance matrix).
fn gj_invert(a: &mut [Vec<f64>]) -> Result<(), FitError> {
    let n = a.len();
    let mut inv: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut r = vec![0.0; n];
            r[i] = 1.0;
            r
        })
        .collect();
    for col in 0..n {
        // Partial pivoting.
        let mut piv = col;
        for r in col..n {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-300 {
            return Err(FitError::Singular);
        }
        a.swap(col, piv);
        inv.swap(col, piv);
        let d = a[col][col];
        for c in 0..n {
            a[col][c] /= d;
            inv[col][c] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = a[r][col];
            if f == 0.0 {
                continue;
            }
            for c in 0..n {
                a[r][c] -= f * a[col][c];
                inv[r][c] -= f * inv[col][c];
            }
        }
    }
    for (r, row) in inv.into_iter().enumerate() {
        a[r] = row;
    }
    Ok(())
}

/// The accumulated normal equations of one trial parameter set.
struct Trial {
    hessian: Vec<Vec<f64>>,
    cgrad: Vec<f64>,
    chisq: f64,
    rchisq: f64,
    ndfree: i64,
    pars: Vec<f64>,
}

impl Trial {
    fn new(nfree: usize) -> Trial {
        Trial {
            hessian: vec![vec![0.0; nfree]; nfree],
            cgrad: vec![0.0; nfree],
            chisq: 0.0,
            rchisq: f64::MAX,
            ndfree: 0,
            pars: vec![0.0; nfree],
        }
    }
}

/// Accumulate chi-squared, the Hessian and the gradient for the
/// current parameters (the body of lm_getfit()).
fn accumulate(
    comps: &[ModComp],
    maps: &[CompMap],
    uvrmax: f64,
    data: &[FitVis],
    out: &mut Trial,
) -> Result<(), FitError> {
    let nfree = out.cgrad.len();
    for row in out.hessian.iter_mut() {
        row.fill(0.0);
    }
    out.cgrad.fill(0.0);
    out.chisq = 0.0;
    out.ndfree = -(nfree as i64);
    let mut dre = vec![0.0; nfree];
    let mut dim = vec![0.0; nfree];
    for v in data {
        let (mre, mim) = model_and_grad(comps, maps, uvrmax, v, &mut dre, &mut dim)?;
        // Real and imaginary parts are two independent measurements.
        for (dy, grad) in [(v.re - mre, &dre), (v.im - mim, &dim)] {
            out.ndfree += 1;
            for r in 0..nfree {
                let tmp = v.wt * grad[r];
                for c in 0..=r {
                    out.hessian[r][c] += tmp * grad[c];
                }
                out.cgrad[r] += dy * tmp;
            }
            out.chisq += v.wt * dy * dy;
        }
    }
    // Symmetrize.
    for r in 0..nfree {
        for c in (r + 1)..nfree {
            out.hessian[r][c] = out.hessian[c][r];
        }
    }
    if out.ndfree < 1 {
        return Err(FitError::Underdetermined(2 * data.len(), nfree));
    }
    out.rchisq = out.chisq / out.ndfree as f64;
    Ok(())
}

/// Fit the free parameters of `comps` to the residual visibilities of
/// the current stream (i.e. after subtraction of the established
/// model). `comps` is updated in place with the best-fit values.
///
/// `niter` is the number of Levenberg-Marquardt iterations; `uvmin`
/// and `uvmax` optionally restrict the UV radius range (wavelengths).
pub fn fit_uvmodel(
    ob: &Observation,
    comps: &mut [ModComp],
    niter: usize,
    uvmin: f32,
    uvmax: f32,
) -> Result<FitResult, FitError> {
    let stream = ob.stream.as_ref().ok_or(FitError::NoStream)?;

    // Map the free parameters.
    let mut maps = Vec::new();
    let mut nfree = 0usize;
    for (icmp, c) in comps.iter().enumerate() {
        let n = count_free(c.freepar);
        if n > 0 {
            maps.push(CompMap {
                icmp,
                off: nfree,
                freepar: c.freepar,
            });
            nfree += n;
        }
    }
    if nfree == 0 {
        return Err(FitError::NoFreePars);
    }

    // Collect the usable visibilities (residual after the established
    // model, which is already subtracted from the stream model).
    let nif = ob.nif();
    let mut docut = uvmin.max(uvmax) > 0.0;
    let (mut lo, mut hi) = (uvmin as f64, uvmax as f64);
    if lo > hi {
        std::mem::swap(&mut lo, &mut hi);
    }
    if hi <= 0.0 {
        docut = false;
    }
    let mut data: Vec<FitVis> = Vec::new();
    let mut uvrmax = 0.0f64;
    for row in 0..ob.nrow {
        let (us, vs) = (ob.uvw[row * 3], ob.uvw[row * 3 + 1]);
        for cif in 0..nif {
            if !stream.if_used[cif] {
                continue;
            }
            let v = stream.vis[row * nif + cif];
            if v.wt <= 0.0 {
                continue;
            }
            let freq = stream.if_freq[cif];
            let (uu, vv) = (us * freq, vs * freq);
            let uvrad = (uu * uu + vv * vv).sqrt();
            if docut && (uvrad < lo || uvrad > hi) {
                continue;
            }
            let m = stream.model[row * nif + cif];
            if uvrad > uvrmax {
                uvrmax = uvrad;
            }
            data.push(FitVis {
                uu,
                vv,
                re: v.re as f64 - m.0 as f64,
                im: v.im as f64 - m.1 as f64,
                wt: v.wt as f64,
                freq,
            });
        }
    }
    if data.is_empty() {
        return Err(FitError::NoData);
    }
    if 2 * data.len() <= nfree {
        return Err(FitError::Underdetermined(2 * data.len(), nfree));
    }
    if uvrmax <= 0.0 {
        uvrmax = 1.0;
    }

    // Levenberg-Marquardt iterations (lm_fit()).
    let mut best = Trial::new(nfree);
    let mut trial = Trial::new(nfree);
    let mut incfac = 0.001f64;
    let mut nbetter = 0usize;
    let mut work: Vec<ModComp> = comps.to_vec();

    // Pre-pass: evaluate the starting model.
    get_free(&work, &maps, uvrmax, &mut best.pars);
    accumulate(&work, &maps, uvrmax, &data, &mut best)?;
    nbetter += 1;

    for _ in 0..niter {
        // Solve (H + incfac*diag(H)) dp = cgrad for the increments.
        let mut h = best.hessian.clone();
        for (i, row) in h.iter_mut().enumerate() {
            row[i] *= 1.0 + incfac;
        }
        let mut dp = best.cgrad.clone();
        if gj_solve(&mut h, &mut dp).is_err() {
            // Singular: shrink the step and try again.
            incfac *= 10.0;
            continue;
        }
        for i in 0..nfree {
            trial.pars[i] = best.pars[i] + dp[i];
        }
        set_free(&mut work, &maps, uvrmax, &trial.pars);
        // setfree may clamp values, so read them back.
        get_free(&work, &maps, uvrmax, &mut trial.pars);
        if accumulate(&work, &maps, uvrmax, &data, &mut trial).is_err() {
            incfac *= 10.0;
            continue;
        }
        if trial.rchisq < best.rchisq {
            std::mem::swap(&mut best, &mut trial);
            incfac *= 0.5;
            nbetter += 1;
        } else {
            incfac *= 10.0;
        }
        // Always leave the model at the best-fit parameters.
        set_free(&mut work, &maps, uvrmax, &best.pars);
    }
    set_free(&mut work, &maps, uvrmax, &best.pars);
    comps.copy_from_slice(&work);

    // Parameter uncertainties from the covariance matrix
    // (inverse Hessian), scaled by reduced chi-squared.
    let mut errors = vec![CompErrors::default(); comps.len()];
    let mut cov = best.hessian.clone();
    if gj_invert(&mut cov).is_ok() {
        let scale = best.rchisq.max(0.0);
        for m in &maps {
            let c = &comps[m.icmp];
            let sd = |k: usize| -> f64 { (cov[k][k].max(0.0) * scale).sqrt() };
            let e = &mut errors[m.icmp];
            let mut k = m.off;
            if m.freepar & M_FLUX != 0 {
                e.flux = sd(k);
                k += 1;
            }
            if m.freepar & M_CENT != 0 {
                e.x = sd(k) / uvrmax;
                e.y = sd(k + 1) / uvrmax;
                k += 2;
            }
            if m.freepar & (M_MAJOR | M_RATIO | M_PHI) != 0 {
                let renorm = uvrmax * uvrmax;
                if m.freepar & M_RATIO != 0 {
                    // Propagate (X, Y, Z) errors to (major, ratio, phi)
                    // ignoring covariances, as difmap does.
                    let (ex, ey, ez) = (sd(k) / renorm, sd(k + 1) / renorm, sd(k + 2) / renorm);
                    let a = c.major as f64;
                    let r = c.ratio as f64;
                    let xyrad = 0.5 * a * a * (1.0 - r * r);
                    // major = sqrt(Z + |XY|): d major/dZ = 1/(2 major)
                    e.major = if a > 0.0 { ez / (2.0 * a) } else { f64::NAN };
                    e.ratio = if a > 0.0 && r > 0.0 {
                        (ex.hypot(ey) + ez) / (2.0 * a * a * r)
                    } else {
                        f64::NAN
                    };
                    e.phi = if xyrad > 0.0 {
                        0.5 * ex.hypot(ey) / xyrad
                    } else {
                        f64::NAN
                    };
                    k += 3;
                } else {
                    let ez = sd(k) / renorm;
                    let a = c.major as f64;
                    e.major = if a > 0.0 { ez / (2.0 * a) } else { f64::NAN };
                    k += 1;
                }
            }
            if m.freepar & M_SPCIND != 0 {
                e.spcind = sd(k);
            }
        }
    }

    Ok(FitResult {
        rchisq: best.rchisq,
        chisq: best.chisq,
        ndfree: best.ndfree,
        nvis: data.len(),
        nfree,
        nbetter,
        errors,
    })
}
