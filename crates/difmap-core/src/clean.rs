//! Högbom CLEAN, restore and map statistics.
//!
//! Port of difmap's mapclean.c and mapres.c. The cleanable area is the
//! inner quarter of the grid (so that the beam patch always covers it).
//! CLEAN windows are given in map coordinates (radians) and clipped to
//! the clean area.

use crate::grid::MapBeam;
use crate::model::{CmpType, ModComp};
use std::collections::HashMap;

/// A rectangular CLEAN window in map coordinates (radians).
#[derive(Clone, Copy, Debug)]
pub struct Window {
    pub xmin: f64,
    pub xmax: f64,
    pub ymin: f64,
    pub ymax: f64,
}

/// Pixel-range window (inclusive).
#[derive(Clone, Copy, Debug)]
struct PixWin {
    xa: usize,
    xb: usize,
    ya: usize,
    yb: usize,
}

#[derive(thiserror::Error, Debug)]
pub enum CleanError {
    #[error("ridiculous clean gain: {0}")]
    BadGain(f32),
    #[error("invalid dirty beam supplied - run invert first")]
    BadBeam,
    #[error("all CLEAN windows lie outside the CLEAN area")]
    NoWindows,
}

pub struct CleanResult {
    /// Number of components subtracted.
    pub niter: usize,
    /// Total flux cleaned (Jy).
    pub cleaned_flux: f64,
    /// New delta components (also appended to the caller's tentative
    /// model). Components at the same pixel are merged.
    pub comps: Vec<ModComp>,
    /// True if stopped because the cutoff was reached.
    pub hit_cutoff: bool,
    /// True if stopped at the first negative component.
    pub hit_negative: bool,
}

/// Convert world-coordinate windows to pixel windows clipped to the
/// clean area; with no windows, the whole clean area is used.
fn pixel_windows(mb: &MapBeam, windows: &[Window]) -> Vec<PixWin> {
    let (nx, ny) = (mb.geom.nx, mb.geom.ny);
    let (ixmin, iymin) = (nx / 4, ny / 4);
    let (ixmax, iymax) = (nx - nx / 4 - 1, ny - ny / 4 - 1);
    if windows.is_empty() {
        return vec![PixWin {
            xa: ixmin,
            xb: ixmax,
            ya: iymin,
            yb: iymax,
        }];
    }
    let (xinc, yinc) = (mb.geom.xinc, mb.geom.yinc);
    let (xcent, ycent) = ((nx / 2) as f64, (ny / 2) as f64);
    let mut out = Vec::new();
    for w in windows {
        let (wxa, wxb) = if w.xmin <= w.xmax {
            (w.xmin, w.xmax)
        } else {
            (w.xmax, w.xmin)
        };
        let (wya, wyb) = if w.ymin <= w.ymax {
            (w.ymin, w.ymax)
        } else {
            (w.ymax, w.ymin)
        };
        let xa = (xcent + (wxa / xinc).round()).max(ixmin as f64);
        let xb = (xcent + (wxb / xinc).round()).min(ixmax as f64);
        let ya = (ycent + (wya / yinc).round()).max(iymin as f64);
        let yb = (ycent + (wyb / yinc).round()).min(iymax as f64);
        if xa <= xb && ya <= yb {
            out.push(PixWin {
                xa: xa as usize,
                xb: xb as usize,
                ya: ya as usize,
                yb: yb as usize,
            });
        }
    }
    out
}

/// Högbom CLEAN of the residual map (port of mapclean()).
///
/// `niter < 0` stops at the first negative component (|niter| max
/// iterations). The subtracted delta components are returned; the map
/// in `mb` becomes the updated residual map.
pub fn clean(
    mb: &mut MapBeam,
    windows: &[Window],
    niter: i32,
    gain: f32,
    cutoff: f32,
) -> Result<CleanResult, CleanError> {
    if gain <= 0.0 || gain > 1.0 {
        return Err(CleanError::BadGain(gain));
    }
    let (nx, ny) = (mb.geom.nx, mb.geom.ny);
    let (ixmin, iymin) = (nx / 4, ny / 4);
    let (ixmax, iymax) = (nx - nx / 4 - 1, ny - ny / 4 - 1);
    let wins = pixel_windows(mb, windows);
    if wins.is_empty() {
        return Err(CleanError::NoWindows);
    }
    let cutoff = cutoff.abs();
    let noneg = niter < 0;
    let maxcmp = niter.unsigned_abs() as usize;
    let cntr = nx / 2 + nx * (ny / 2);
    let bmax = mb.beam[cntr];
    if bmax == 0.0 {
        return Err(CleanError::BadBeam);
    }

    let mut comps: Vec<ModComp> = Vec::new();
    let mut by_pixel: HashMap<(i64, i64), usize> = HashMap::new();
    let mut ccsum = 0.0f64;
    let mut n = 0usize;
    let mut hit_cutoff = false;
    let mut hit_negative = false;

    while n < maxcmp {
        // Find the peak absolute residual within the windows.
        let mut maxabs = 0.0f32;
        let mut maxidx: Option<usize> = None;
        for w in &wins {
            for iy in w.ya..=w.yb {
                let row = iy * nx;
                for ix in w.xa..=w.xb {
                    let v = mb.map[row + ix];
                    if v < -maxabs || v > maxabs {
                        maxabs = v.abs();
                        maxidx = Some(row + ix);
                    }
                }
            }
        }
        let idx = match maxidx {
            Some(i) => i,
            None => break, // no flux left
        };
        let mut maxval = mb.map[idx] / bmax;
        if maxval.abs() <= cutoff {
            hit_cutoff = true;
            break;
        }
        if noneg && maxval < 0.0 {
            hit_negative = true;
            break;
        }
        maxval *= gain;

        // Subtract the scaled, shifted beam over the clean area.
        let (cx, cy) = ((idx % nx) as i64, (idx / nx) as i64);
        for iy in iymin..=iymax {
            let by = (iy as i64 - cy) + (ny / 2) as i64;
            let brow = by as usize * nx;
            let mrow = iy * nx;
            let bxoff = (nx / 2) as i64 - cx;
            for ix in ixmin..=ixmax {
                let bx = (ix as i64 + bxoff) as usize;
                mb.map[mrow + ix] -= mb.beam[brow + bx] * maxval;
            }
        }

        n += 1;
        ccsum += maxval as f64;
        // Append/merge the delta component.
        let (px, py) = (cx - (nx / 2) as i64, cy - (ny / 2) as i64);
        match by_pixel.get(&(px, py)) {
            Some(&ci) => comps[ci].flux += maxval,
            None => {
                by_pixel.insert((px, py), comps.len());
                comps.push(ModComp::delta(
                    maxval,
                    (px as f64 * mb.geom.xinc) as f32,
                    (py as f64 * mb.geom.yinc) as f32,
                ));
            }
        }
    }

    Ok(CleanResult {
        niter: n,
        cleaned_flux: ccsum,
        comps,
        hit_cutoff,
        hit_negative,
    })
}

/// Convolution of two elliptical gaussians (Wild 1970); port of
/// gauconv(). Axes are FWHMs, angles in radians.
fn gauconv(
    min_a: f64,
    maj_a: f64,
    ang_a: f64,
    min_b: f64,
    maj_b: f64,
    ang_b: f64,
) -> (f64, f64, f64) {
    let (maj_a2, min_a2) = (maj_a * maj_a, min_a * min_a);
    let (maj_b2, min_b2) = (maj_b * maj_b, min_b * min_b);
    let sum7 = (maj_a2 - min_a2) * (2.0 * ang_a).sin() + (maj_b2 - min_b2) * (2.0 * ang_b).sin();
    let sum8 = (maj_a2 + min_a2) + (maj_b2 + min_b2);
    let sum9 = (maj_a2 - min_a2) * (2.0 * ang_a).cos() + (maj_b2 - min_b2) * (2.0 * ang_b).cos();
    let angle = if sum7.abs() == 0.0 && sum9.abs() == 0.0 {
        0.0
    } else {
        0.5 * sum7.atan2(sum9)
    };
    let sumvar = (sum7 * sum7 + sum9 * sum9).sqrt();
    let major = (0.5 * (sum8 + sumvar)).sqrt();
    let minor = (0.5 * (sum8 - sumvar)).abs().sqrt();
    (minor, major, angle)
}

/// Smooth the central nx/2 x ny/2 area with a fixed 3x3 mask (port of
/// res_smooth()).
fn res_smooth(map: &mut [f32], nx: usize, ny: usize) {
    const MASK: [[f32; 3]; 3] = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ];
    let (xa, ya) = (nx / 4, ny / 4);
    let (xb, yb) = (3 * xa - 1, 3 * ya - 1);
    let src = map.to_vec();
    for iy in (ya + 1)..=(yb - 1) {
        for ix in (xa + 1)..=(xb - 1) {
            let mut sum = 0.0f32;
            for (my, mrow) in MASK.iter().enumerate() {
                for (mx, &m) in mrow.iter().enumerate() {
                    sum += src[(iy + my - 1) * nx + (ix + mx - 1)] * m;
                }
            }
            map[iy * nx + ix] = sum;
        }
    }
}

/// Restore the clean map: convolve the model with the elliptical
/// gaussian restoring beam and add it to the residual map (port of
/// mapres()). Only delta and gaussian components are restored; other
/// types are skipped, as in difmap.
///
/// `bmaj`/`bmin`/`bpa` in radians; `freq` (Hz) evaluates spectral
/// indices. Returns the restored map `[ny * nx]`.
#[allow(clippy::too_many_arguments)]
pub fn restore(
    mb: &MapBeam,
    comps: &[ModComp],
    bmaj: f64,
    bmin: f64,
    bpa: f64,
    no_residual: bool,
    do_smooth: bool,
    freq: f64,
) -> Vec<f32> {
    const NSIGMA: f64 = 4.5;
    let (nx, ny) = (mb.geom.nx, mb.geom.ny);
    let (xinc, yinc) = (mb.geom.xinc, mb.geom.yinc);
    let (bmin, bmaj) = if bmin > bmaj { (bmaj, bmin) } else { (bmin, bmaj) };
    let bfac = 1.0 / (256.0f64.ln()).sqrt();

    let mut cln: Vec<f32> = if no_residual {
        vec![0.0; nx * ny]
    } else {
        let mut m = mb.map.clone();
        if do_smooth {
            res_smooth(&mut m, nx, ny);
        }
        m
    };

    for cmp in comps {
        let (cmin, cmaj, cpa) = match cmp.ctype {
            CmpType::Delta => (bmin, bmaj, bpa),
            CmpType::Gaussian => gauconv(
                bmin,
                bmaj,
                bpa,
                (cmp.ratio * cmp.major) as f64,
                cmp.major as f64,
                cmp.phi as f64,
            ),
            _ => continue, // only delta/gaussian supported (as in difmap)
        };
        // Scale factor for Jy/beam.
        let mut flux = cmp.flux as f64 * bmaj * bmin / (cmin * cmaj);
        if cmp.spcind != 0.0 {
            flux *= (freq / cmp.freq0 as f64).powf(cmp.spcind as f64);
        }
        // FWHM -> standard deviations.
        let (cmin, cmaj) = (cmin * bfac, cmaj * bfac);
        let nxpix = (NSIGMA * cmaj / xinc) as i64;
        let nypix = (NSIGMA * cmaj / yinc) as i64;
        let minfac = 0.5 / (cmin * cmin);
        let majfac = 0.5 / (cmaj * cmaj);
        let xminor = xinc * cpa.cos();
        let yminor = -yinc * cpa.sin();
        let xmajor = xinc * cpa.sin();
        let ymajor = yinc * cpa.cos();
        let modx = nx as f64 / 2.0 + cmp.x as f64 / xinc;
        let mody = ny as f64 / 2.0 + cmp.y as f64 / yinc;
        let (imodx, imody) = (modx as i64, mody as i64);
        let xa = (imodx - nxpix).max(0) as usize;
        let xb = (imodx + nxpix).min(nx as i64 - 1) as usize;
        let ya = (imody - nypix).max(0) as usize;
        let yb = (imody + nypix).min(ny as i64 - 1) as usize;
        let argmax = 0.5 * NSIGMA * NSIGMA;
        for iy in ya..=yb {
            let fy = mody - iy as f64;
            let row = iy * nx;
            for ix in xa..=xb {
                let fx = modx - ix as f64;
                let minor = xminor * fx + yminor * fy;
                let major = xmajor * fx + ymajor * fy;
                let arg = minfac * minor * minor + majfac * major * major;
                if arg < argmax {
                    cln[row + ix] += (flux * (-arg).exp()) as f32;
                }
            }
        }
    }
    cln
}

/// Statistics of the inner (cleanable) map area: (min, max, mean, rms)
/// and the pixel indexes of the extrema. Port of mapstats() behavior.
pub struct MapStats {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub rms: f64,
    pub minpos: (usize, usize),
    pub maxpos: (usize, usize),
}

pub fn map_stats(map: &[f32], nx: usize, ny: usize) -> MapStats {
    let (ixmin, iymin) = (nx / 4, ny / 4);
    let (ixmax, iymax) = (nx - nx / 4 - 1, ny - ny / 4 - 1);
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut minpos = (0, 0);
    let mut maxpos = (0, 0);
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut npix = 0usize;
    for iy in iymin..=iymax {
        for ix in ixmin..=ixmax {
            let v = map[iy * nx + ix];
            if v < min {
                min = v;
                minpos = (ix, iy);
            }
            if v > max {
                max = v;
                maxpos = (ix, iy);
            }
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
            npix += 1;
        }
    }
    let mean = sum / npix as f64;
    MapStats {
        min,
        max,
        mean,
        rms: (sumsq / npix as f64 - mean * mean).max(0.0).sqrt(),
        minpos,
        maxpos,
    }
}
