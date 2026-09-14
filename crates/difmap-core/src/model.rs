//! Model components and model visibility computation.
//!
//! Port of difmap's model.h/modvis.c. An observation carries an
//! *established* model (whose visibilities are included in the
//! stream's model arrays) and a *tentative* model (new components from
//! clean/modelfit/addcmp that have not yet been Fourier-transformed).
//! `merge_model` (difmap mergemod/keep) transforms the tentative
//! components and moves them into the established model.

use crate::obs::Observation;
use rayon::prelude::*;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Model component types (difmap Modtyp).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpType {
    Delta,
    Gaussian,
    Disk,
    Ellipse,
    Ring,
    Rectangle,
    Sz,
}

impl CmpType {
    pub fn from_code(code: i32) -> Option<CmpType> {
        Some(match code {
            0 => CmpType::Delta,
            1 => CmpType::Gaussian,
            2 => CmpType::Disk,
            3 => CmpType::Ellipse,
            4 => CmpType::Ring,
            5 => CmpType::Rectangle,
            6 => CmpType::Sz,
            _ => return None,
        })
    }
    pub fn code(self) -> i32 {
        match self {
            CmpType::Delta => 0,
            CmpType::Gaussian => 1,
            CmpType::Disk => 2,
            CmpType::Ellipse => 3,
            CmpType::Ring => 4,
            CmpType::Rectangle => 5,
            CmpType::Sz => 6,
        }
    }
}

/// One model component (difmap Modcmp).
#[derive(Clone, Copy, Debug)]
pub struct ModComp {
    pub ctype: CmpType,
    /// Flux at `freq0` (Jy).
    pub flux: f32,
    /// Position offset east/north of the phase center (radians).
    pub x: f32,
    pub y: f32,
    /// FWHM major axis (radians); unused for delta components.
    pub major: f32,
    /// Axial ratio minor/major (<= 1).
    pub ratio: f32,
    /// Major-axis position angle (radians, north -> east).
    pub phi: f32,
    /// Reference frequency for `spcind` (Hz); 0 = none.
    pub freq0: f32,
    /// Spectral index.
    pub spcind: f32,
    /// Bitmap of free parameters for model fitting (difmap freepar).
    pub freepar: u32,
}

impl ModComp {
    pub fn delta(flux: f32, x: f32, y: f32) -> ModComp {
        ModComp {
            ctype: CmpType::Delta,
            flux,
            x,
            y,
            major: 0.0,
            ratio: 1.0,
            phi: 0.0,
            freq0: 0.0,
            spcind: 0.0,
            freepar: 0,
        }
    }

    /// True if `modelfit` would vary any of this component's
    /// parameters. Components with none are held fixed by being
    /// established (see `partition_variable_model`).
    pub fn is_variable(&self) -> bool {
        crate::modelfit::count_free(self.freepar) > 0
    }

    /// Amplitude and phase of this component at (uu, vv) wavelengths
    /// and frequency `freq` (Hz). Exact port of cmpvis() (modvis.c),
    /// currently without primary-beam attenuation.
    pub fn vis(&self, freq: f64, uu: f64, vv: f64) -> (f64, f64) {
        let spec = if self.spcind == 0.0 {
            1.0
        } else {
            (freq / self.freq0 as f64).powf(self.spcind as f64)
        };
        let flux = self.flux as f64 * spec;
        let phs = TWO_PI * (uu * self.x as f64 + vv * self.y as f64);
        if self.ctype == CmpType::Delta {
            return (flux, phs);
        }
        let (sinphi, cosphi) = (self.phi as f64).sin_cos();
        let tmpa = vv * cosphi + uu * sinphi;
        let tmpb = self.ratio as f64 * (uu * cosphi - vv * sinphi);
        let tmpc = (std::f64::consts::PI * self.major as f64
            * (tmpa * tmpa + tmpb * tmpb).sqrt())
        .max(1.0e-9);
        let amp = match self.ctype {
            CmpType::Delta => unreachable!(),
            CmpType::Gaussian => {
                flux * if tmpc < 12.0 {
                    (-0.3606737602 * tmpc * tmpc).exp()
                } else {
                    0.0
                }
            }
            CmpType::Disk => 2.0 * flux * besj1(tmpc) / tmpc,
            CmpType::Ellipse => 3.0 * flux * (tmpc.sin() - tmpc * tmpc.cos()) / (tmpc * tmpc * tmpc),
            CmpType::Ring => flux * besj0(tmpc),
            CmpType::Rectangle => {
                let ta = std::f64::consts::PI * self.major as f64 * (uu * sinphi + vv * cosphi);
                flux * if ta.abs() > 0.001 { ta.sin() / ta } else { 1.0 }
            }
            CmpType::Sz => flux * if tmpc < 50.0 { (-tmpc).exp() } else { 0.0 } / tmpc,
        };
        (amp, phs)
    }
}

/// Accumulate the visibilities of `comps` into `out` (one (re, im) per
/// row and IF, row-major), scaled by `sign`.
///
/// Split out of [`add_to_stream_model`] so that the same kernel can
/// also fill a scratch buffer - see [`full_model_vis`] - rather than
/// only the stream's own model array.
fn accumulate_model(
    uvw: &[f64],
    if_freq: &[f64],
    if_used: &[bool],
    nif: usize,
    comps: &[ModComp],
    out: &mut [(f32, f32)],
    sign: f64,
) {
    out.par_chunks_mut(nif).enumerate().for_each(|(row, mrow)| {
        let u_sec = uvw[row * 3];
        let v_sec = uvw[row * 3 + 1];
        for (cif, m) in mrow.iter_mut().enumerate() {
            if !if_used[cif] {
                continue;
            }
            let freq = if_freq[cif];
            let (uu, vv) = (u_sec * freq, v_sec * freq);
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for c in comps {
                let (amp, phs) = c.vis(freq, uu, vv);
                let (s, cph) = phs.sin_cos();
                re += amp * cph;
                im += amp * s;
            }
            m.0 += (sign * re) as f32;
            m.1 += (sign * im) as f32;
        }
    });
}

/// Add the visibilities of `comps` to (or subtract from) the stream
/// model arrays of the current selection, at each row/IF's uv
/// coordinates and effective frequency.
pub fn add_to_stream_model(ob: &mut Observation, comps: &[ModComp], subtract: bool) {
    if comps.is_empty() {
        return;
    }
    let nif = ob.ifs.len();
    let sign = if subtract { -1.0f64 } else { 1.0f64 };
    let uvw = &ob.uvw;
    let stream = match ob.stream.as_mut() {
        Some(s) => s,
        None => return,
    };
    accumulate_model(
        uvw,
        &stream.if_freq,
        &stream.if_used,
        nif,
        comps,
        &mut stream.model,
        sign,
    );
}

/// Visibilities of the *complete* model - `ob.model` and `ob.newmod`
/// alike - at every row and IF of the current selection.
///
/// The stream model holds `ob.model` only. The bindings merge `newmod`
/// before returning, so the two agree outside a fit, but
/// [`partition_variable_model`] takes the components being fitted out
/// of the stream model while `modelfit` runs; this stays right then too.
pub fn full_model_vis(ob: &Observation) -> Vec<(f32, f32)> {
    let stream = match ob.stream.as_ref() {
        Some(s) => s,
        None => return Vec::new(),
    };
    let mut out = stream.model.clone();
    if !ob.newmod.is_empty() {
        accumulate_model(
            &ob.uvw,
            &stream.if_freq,
            &stream.if_used,
            ob.ifs.len(),
            &ob.newmod,
            &mut out,
            1.0,
        );
    }
    out
}

/// Establish the tentative model: Fourier transform `ob.newmod` into
/// the stream model and append the components to `ob.model`.
/// Equivalent to difmap's mergemod()/keep.
pub fn merge_model(ob: &mut Observation) {
    if ob.newmod.is_empty() {
        return;
    }
    let comps = std::mem::take(&mut ob.newmod);
    add_to_stream_model(ob, &comps, false);
    ob.model.extend(comps);
}

/// Split the model into its fixed and variable parts, as difmap's
/// obvarmod() does before every fit.
///
/// Afterwards the *established* model holds every component with no
/// free parameter - CLEAN components above all - so its visibilities
/// are in the stream model and hence removed from the residuals the fit
/// works on; the *tentative* model holds every component that has one,
/// established or not, with its contribution taken back out of the
/// stream model. The fit only ever sees the tentative model, so without
/// this a fixed component would be missing from the model being fitted
/// and then merged again afterwards, counting its flux twice.
///
/// As in difmap, components taken out of the established model come
/// first, which preserves their original order.
pub fn partition_variable_model(ob: &mut Observation) {
    let (fixed, variable): (Vec<ModComp>, Vec<ModComp>) = std::mem::take(&mut ob.newmod)
        .into_iter()
        .partition(|c| !c.is_variable());
    add_to_stream_model(ob, &fixed, false);
    ob.model.extend(fixed);

    let (kept, freed): (Vec<ModComp>, Vec<ModComp>) = std::mem::take(&mut ob.model)
        .into_iter()
        .partition(|c| !c.is_variable());
    add_to_stream_model(ob, &freed, true);
    ob.model = kept;
    ob.newmod = freed;
    ob.newmod.extend(variable);
}

/// Clear models (difmap clrmod). Removing the established model also
/// zeroes the corresponding stream model visibilities.
pub fn clear_models(ob: &mut Observation, do_old: bool, do_new: bool) {
    if do_new {
        ob.newmod.clear();
    }
    if do_old && !ob.model.is_empty() {
        let comps = std::mem::take(&mut ob.model);
        add_to_stream_model(ob, &comps, true);
        // Avoid accumulating rounding drift: if nothing remains, zero.
        if let Some(s) = ob.stream.as_mut() {
            if ob.model.is_empty() && ob.newmod.is_empty() {
                for m in s.model.iter_mut() {
                    *m = (0.0, 0.0);
                }
            }
        }
    }
}

/// Recompute the stream model visibilities of the established model
/// from scratch (used after `select` builds a new stream).
pub fn recompute_stream_model(ob: &mut Observation) {
    if let Some(s) = ob.stream.as_mut() {
        for m in s.model.iter_mut() {
            *m = (0.0, 0.0);
        }
    }
    let comps = std::mem::take(&mut ob.model);
    add_to_stream_model(ob, &comps, false);
    ob.model = comps;
}

/// Total flux in a list of components.
pub fn total_flux(comps: &[ModComp]) -> f64 {
    comps.iter().map(|c| c.flux as f64).sum()
}

// ----------------------------------------------------------------------
// Bessel functions J0, J1, J2.
//
// The rational approximations below are the classic ones tabulated in
// Abramowitz & Stegun, "Handbook of Mathematical Functions", eqs.
// 9.4.1-9.4.6 (a US Government work, in the public domain; the same
// coefficients are reproduced in several textbooks). difmap's besj.c
// uses the same approximations.
// ----------------------------------------------------------------------

pub fn besj0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        (57568490574.0
            + y * (-13362590354.0
                + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456)))))
            / (57568490411.0
                + y * (1029532985.0
                    + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y)))))
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        (0.636619772 / ax).sqrt()
            * (xx.cos()
                * (1.0
                    + y * (-1.098628627e-3
                        + y * (2.734510407e-5 + y * (-2.073370639e-6 + y * 2.093887211e-7))))
                - z * xx.sin()
                    * (-1.562499995e-2
                        + y * (1.430488765e-4
                            + y * (-6.911147651e-6 + y * (7.621095161e-7 + y * -9.34945152e-8)))))
    }
}

/// J2(x) via the upward recurrence J2 = 2*J1/x - J0 for x > 2, where it
/// is stable, and Miller's downward recurrence below that.
pub fn besj2(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 {
        return 0.0;
    }
    if x > 2.0 {
        return 2.0 * besj1(x) / x - besj0(x);
    }
    const LARGE: f64 = 1.0e10;
    let recfac = 2.0 / x;
    let mut bjpp = 0.0f64; // J(n+2)
    let mut bjp = 1.0f64; // J(n+1)
    let mut normsum = 0.0f64; // J0 + 2*(J2 + J4 + ...)
    let mut retval = 0.0f64;
    for order in (2..=10).rev() {
        let mut bj = recfac * (order + 1) as f64 * bjp - bjpp;
        bjpp = bjp;
        bjp = bj;
        if bj > LARGE {
            bj /= LARGE;
            bjp /= LARGE;
            bjpp /= LARGE;
            retval /= LARGE;
        }
        if order % 2 == 0 {
            normsum += 2.0 * bj;
        }
        if order == 2 {
            retval = bj;
        }
    }
    if normsum != 0.0 {
        retval /= normsum;
    }
    retval
}

pub fn besj1(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        x * (72362614232.0
            + y * (-7895059235.0
                + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * -30.16036606)))))
            / (144725228442.0
                + y * (2300535178.0
                    + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y)))))
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;
        x.signum()
            * (0.636619772 / ax).sqrt()
            * (xx.cos()
                * (1.0
                    + y * (1.83105e-3
                        + y * (-3.516396496e-5 + y * (2.457520174e-6 + y * -2.40337019e-7))))
                - z * xx.sin()
                    * (0.04687499995
                        + y * (-2.002690873e-4
                            + y * (8.449199096e-6 + y * (-8.8228987e-7 + y * 1.05787412e-7)))))
    }
}
