//! UV gridding and Fourier inversion to dirty map and beam.
//!
//! Faithful port of difmap's uvinvert.c, uvtrans.c and costran.c:
//! * 5x5 Gaussian gridding kernel (nmask=2, hwhm=0.7 grid pixels,
//!   sampled on a 301-point lookup table),
//! * optional uniform weighting (bin counts over a nx/4 x ny/2 array),
//!   gaussian taper, radial weighting and amplitude-error weighting,
//! * gridding onto a half conjugate-symmetric complex array,
//! * FFT with center shift, and gridding-convolution correction by the
//!   reciprocal cosine transform of the kernel,
//! * Tim Pearson's beam-size estimate from UV second moments.
//!
//! The map pixel (ix, iy) corresponds to the sky offset
//! x = (ix - nx/2)*xinc (east), y = (iy - ny/2)*yinc (north), radians,
//! consistent with the model phase convention 2*pi*(u*x + v*y).

use crate::obs::Observation;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use realfft::RealFftPlanner;

const NGCF: usize = 301;
const NMASK: i64 = 2;
const HWHM: f32 = 0.7;
const BEAM_FUDGE: f64 = 0.7;

/// Map geometry: grid dimensions (powers of 2) and cell sizes.
#[derive(Clone, Copy, Debug)]
pub struct MapGeom {
    pub nx: usize,
    pub ny: usize,
    /// Cell sizes (radians/pixel).
    pub xinc: f64,
    pub yinc: f64,
}

impl MapGeom {
    /// UV cell sizes in wavelengths/pixel.
    #[inline]
    pub fn uinc(&self) -> f64 {
        1.0 / (self.xinc * self.nx as f64)
    }
    #[inline]
    pub fn vinc(&self) -> f64 {
        1.0 / (self.yinc * self.ny as f64)
    }
    /// Largest |U|,|V| (wavelengths) that can be gridded (uv_limits()).
    pub fn uv_limits(&self) -> (f64, f64) {
        (
            self.uinc() * (self.nx as f64 / 4.0 - NMASK as f64),
            self.vinc() * (self.ny as f64 / 4.0 - NMASK as f64),
        )
    }
}

/// Weighting / data-selection parameters of `invert`.
#[derive(Clone, Copy, Debug)]
pub struct InvertPars {
    /// UV radius range (wavelengths); no cut if max(uvmin,uvmax) <= 0.
    pub uvmin: f32,
    pub uvmax: f32,
    /// Gaussian taper: value `gauval` (0..1) at radius `gaurad`.
    pub gauval: f32,
    pub gaurad: f32,
    /// Radial (|uv|) weighting.
    pub dorad: bool,
    /// If < 0, weights are scaled by wt^(-errpow/2).
    pub errpow: f32,
    /// Uniform-weighting bin width in UV grid pixels; <= 0 = natural.
    pub binwid: f32,
    /// Briggs robustness. When set, it supersedes `binwid`/`errpow`:
    /// weights become `w_i / (1 + W_k f^2)` with `W_k` the summed
    /// natural weight of the point's UV bin and
    /// `f^2 = (5*10^-R)^2 / (sum_k W_k^2 / sum_i w_i)`, so that
    /// R = -2 is (nearly) uniform and R = +2 (nearly) natural.
    pub robust: Option<f32>,
    /// Optional zero-spacing flux (uvzero): amplitude, model amp, weight.
    pub uvzero_amp: f32,
    pub uvzero_modamp: f32,
    pub uvzero_wt: f32,
}

impl Default for InvertPars {
    fn default() -> Self {
        InvertPars {
            uvmin: 0.0,
            uvmax: 0.0,
            gauval: 0.0,
            gaurad: 0.0,
            dorad: false,
            errpow: 0.0,
            binwid: 0.0,
            robust: None,
            uvzero_amp: 0.0,
            uvzero_modamp: 0.0,
            uvzero_wt: 0.0,
        }
    }
}

/// The result of `invert`: residual dirty map and dirty beam
/// (both `[ny * nx]`, row-major), plus beam/noise estimates.
#[derive(Clone)]
pub struct MapBeam {
    pub geom: MapGeom,
    pub map: Vec<f32>,
    pub beam: Vec<f32>,
    /// Estimated elliptical clean beam (radians, radians, radians).
    pub e_bmaj: f64,
    pub e_bmin: f64,
    pub e_bpa: f64,
    /// Estimated map noise (Jy/beam).
    pub noise: f64,
    /// Number of visibilities gridded.
    pub nused: usize,
}

#[derive(thiserror::Error, Debug)]
pub enum GridError {
    #[error("map dimensions must be multiples of 4 and >= 32 (got {0}x{1})")]
    BadDims(usize, usize),
    #[error("invalid cell size")]
    BadCell,
    #[error("no data with the current selection/uv range")]
    NoData,
    #[error("no stream selected; call select() first")]
    NoStream,
}

#[inline]
fn fnint(x: f64) -> i64 {
    (x + 0.5).floor() as i64
}

/// The gridding convolution function and its image-plane correction.
struct Gcf {
    convfn: [f32; NGCF],
    tgtocg: f32,
}

impl Gcf {
    /// Port of uvgcf(): gaussian kernel with HWHM=0.7 target pixels.
    fn new() -> Gcf {
        let tgtocg = (NGCF as f32 - 1.0) / (NMASK as f32 + 0.5);
        let cghwhm = tgtocg * HWHM;
        let recvar = (2.0f32).ln() / (cghwhm * cghwhm);
        let mut convfn = [0.0f32; NGCF];
        for (i, c) in convfn.iter_mut().enumerate() {
            *c = (-recvar * (i * i) as f32).exp();
        }
        Gcf { convfn, tgtocg }
    }

    #[inline]
    fn eval(&self, dist: f32) -> f32 {
        self.convfn[(self.tgtocg * dist.abs() + 0.5) as usize]
    }

    /// costran(): discrete cosine transform of the kernel onto an
    /// n-element grid (center at n/2), then normalized reciprocal.
    fn correction(&self, n: usize) -> Vec<f32> {
        let ninp = NGCF - 1;
        let inwid = NMASK as f64 + 0.5;
        let icent = n / 2;
        let theta = 2.0 * std::f64::consts::PI * inwid / ninp as f64 / n as f64;
        let mut out = vec![0.0f32; n];
        for iout in 0..=icent {
            let ang = theta * (iout as f64 - icent as f64);
            let (sininc, cosinc) = ang.sin_cos();
            let mut newcos = 1.0f64;
            let mut newsin = 0.0f64;
            let mut sum = 0.0f64;
            for inp in 0..ninp {
                sum += self.convfn[inp] as f64 * newcos;
                let w = newcos;
                newcos = w * cosinc - newsin * sininc;
                newsin = w * sininc + newsin * cosinc;
            }
            out[iout] = sum as f32;
        }
        // Mirror the first half into the second half.
        for i in 0..icent.saturating_sub(1) {
            out[icent + 1 + i] = out[icent - 1 - i];
        }
        // Normalized reciprocal.
        let peak = out[icent];
        for v in out.iter_mut() {
            *v = peak / *v;
        }
        out
    }
}

/// Uniform-weighting bin array (port of uvbin()/getuvbin()).
///
/// `bins` holds difmap's point counts (uniform weighting); `sums` holds
/// the summed natural weights of each bin, which is what Briggs robust
/// weighting needs. Only the one in use is filled.
struct UVbin {
    bins: Vec<i32>,
    sums: Vec<f32>,
    nu: i64,
    nbin: i64,
    utopix: f64,
    vtopix: f64,
}

impl UVbin {
    fn index(&self, mut uu: f64, mut vv: f64) -> Option<usize> {
        if uu >= 0.0 {
            uu = -uu;
            vv = -vv;
        }
        let nv = self.nbin / self.nu;
        let binpix = self.nu * (nv / 2 + (vv * self.vtopix + 0.5).floor() as i64)
            + (uu * self.utopix + 0.5).floor() as i64;
        if binpix >= 0 && binpix < self.nbin {
            Some(binpix as usize)
        } else {
            None
        }
    }
    fn cell(&mut self, uu: f64, vv: f64) -> Option<&mut i32> {
        self.index(uu, vv).map(|i| &mut self.bins[i])
    }
    fn count(&mut self, uu: f64, vv: f64) -> f32 {
        match self.cell(uu, vv) {
            Some(&mut c) if c > 0 => c as f32,
            _ => 1.0,
        }
    }
    /// Summed natural weight of a point's bin (Briggs `W_k`).
    fn wsum(&self, uu: f64, vv: f64) -> f32 {
        match self.index(uu, vv) {
            Some(i) => self.sums[i],
            None => 0.0,
        }
    }
}

/// One gridded visibility source point after selection/weighting.
struct UVPoint {
    uu: f64,
    vv: f64,
    re: f32,
    im: f32,
    wt: f32, // stream weight (1/variance), used for errpow & noise
}

/// Fourier invert the current stream to a residual dirty map and
/// dirty beam. Port of uvinvert(); grids every used IF at its own
/// effective frequency (multi-frequency synthesis).
pub fn invert(ob: &Observation, geom: MapGeom, pars: &InvertPars) -> Result<MapBeam, GridError> {
    let stream = ob.stream.as_ref().ok_or(GridError::NoStream)?;
    // The FFTs handle any length; the map only has to split evenly into
    // the half-plane grid (nx/2+1) and the bin/inner-quarter arrays
    // (nx/4, ny/2). Powers of two (or products of small primes) are
    // still much the fastest, but are no longer required.
    if geom.nx % 4 != 0 || geom.ny % 4 != 0 || geom.nx < 32 || geom.ny < 32 {
        return Err(GridError::BadDims(geom.nx, geom.ny));
    }
    if !(geom.xinc > 0.0) || !(geom.yinc > 0.0) {
        return Err(GridError::BadCell);
    }
    let (nx, ny) = (geom.nx, geom.ny);
    let (uinc, vinc) = (geom.uinc(), geom.vinc());
    let (ulimit, vlimit) = geom.uv_limits();

    // Normalize the uv radius cut (uvinvert semantics).
    let mut uvmin = pars.uvmin.max(0.0) as f64;
    let mut uvmax = pars.uvmax.max(0.0) as f64;
    if uvmin > uvmax {
        std::mem::swap(&mut uvmin, &mut uvmax);
    }
    let docut = uvmax > 0.0;

    // Gaussian taper factor.
    let dotaper = pars.gaurad > 0.0 && pars.gauval > 0.0 && pars.gauval < 1.0;
    let gfac = if dotaper {
        (pars.gauval as f64).ln() / (pars.gaurad as f64 * pars.gaurad as f64)
    } else {
        0.0
    };

    // Collect the usable UV points once (u/v in wavelengths).
    let nif = ob.nif();
    let mut pts: Vec<UVPoint> = Vec::new();
    for row in 0..ob.nrow {
        let u_sec = ob.uvw[row * 3];
        let v_sec = ob.uvw[row * 3 + 1];
        for cif in 0..nif {
            if !stream.if_used[cif] {
                continue;
            }
            let v = stream.vis[row * nif + cif];
            if v.wt <= 0.0 {
                continue; // flagged or deleted
            }
            let uvscale = stream.if_freq[cif];
            let uu = u_sec * uvscale;
            let vv = v_sec * uvscale;
            let uvrad = (uu * uu + vv * vv).sqrt();
            if docut && (uvrad < uvmin || uvrad > uvmax) {
                continue;
            }
            if uu.abs() > ulimit || vv.abs() > vlimit {
                continue;
            }
            let m = stream.model[row * nif + cif];
            pts.push(UVPoint {
                uu,
                vv,
                re: v.re - m.0,
                im: v.im - m.1,
                wt: v.wt,
            });
        }
    }
    if pts.is_empty() {
        return Err(GridError::NoData);
    }

    // UV binning: difmap's point counts for uniform weighting, or the
    // summed natural weights for Briggs robust weighting.
    let dorobust = pars.robust.is_some();
    let dounif = !dorobust && pars.binwid > 0.0;
    let dobin = dorobust || dounif;
    let binwid = if dobin { pars.binwid.max(1.0) as f64 } else { 0.0 };
    let nbin = (nx / 4) * (ny / 2);
    let mut bin = UVbin {
        bins: if dounif { vec![0; nbin] } else { Vec::new() },
        sums: if dorobust { vec![0.0; nbin] } else { Vec::new() },
        nu: (nx / 4) as i64,
        nbin: nbin as i64,
        utopix: if dobin { 1.0 / uinc / binwid } else { 0.0 },
        vtopix: if dobin { 1.0 / vinc / binwid } else { 0.0 },
    };
    if dounif {
        for p in &pts {
            if let Some(c) = bin.cell(p.uu, p.vv) {
                *c += 1;
            }
            // Points in the U=0 bin also appear mirrored across V=0.
            if fnint(p.uu.abs() * bin.utopix) == 0 {
                if let Some(c) = bin.cell(p.uu, -p.vv) {
                    *c += 1;
                }
            }
        }
        if let Some(c) = bin.cell(0.0, 0.0) {
            *c += 1; // zero-spacing / natural weighting entry
        }
    }

    // Briggs robust weighting factor f^2 (Briggs 1995, eq. 3.3), with
    // the same binning as uniform weighting so that the two are
    // directly comparable.
    let mut f2 = 0.0f64;
    if dorobust {
        for p in &pts {
            if let Some(i) = bin.index(p.uu, p.vv) {
                bin.sums[i] += p.wt.abs();
            }
            if fnint(p.uu.abs() * bin.utopix) == 0 {
                if let Some(i) = bin.index(p.uu, -p.vv) {
                    bin.sums[i] += p.wt.abs();
                }
            }
        }
        let wsum: f64 = pts.iter().map(|p| p.wt.abs() as f64).sum();
        let w2sum: f64 = bin.sums.iter().map(|&s| (s as f64) * (s as f64)).sum();
        if w2sum > 0.0 && wsum > 0.0 {
            let r = pars.robust.unwrap().clamp(-2.0, 2.0) as f64;
            f2 = (5.0 * 10f64.powf(-r)).powi(2) / (w2sum / wsum);
        }
    }

    // Grid map and beam simultaneously (same weights and kernel).
    let gcf = Gcf::new();
    let nugrid = nx / 2 + 1;
    let mut mgrid = vec![Complex::<f32>::new(0.0, 0.0); nugrid * ny];
    let mut bgrid = vec![Complex::<f32>::new(0.0, 0.0); nugrid * ny];
    let mut wsum = 0.0f64;
    // Beam and noise estimation running means (uvgrid()).
    let mut bm_wsum = 0.0f64;
    let mut bm_muu = 0.0f64;
    let mut bm_mvv = 0.0f64;
    let mut bm_muv = 0.0f64;
    let mut bm_nsum = 0.0f64;

    let grid_point = |mgrid: &mut Vec<Complex<f32>>,
                          bgrid: &mut Vec<Complex<f32>>,
                          wsum: &mut f64,
                          uu: f64,
                          vv: f64,
                          re: f32,
                          im: f32,
                          weight: f32| {
        let ufrc = uu / uinc;
        let vfrc = vv / vinc;
        let upix = fnint(ufrc);
        let vpix = fnint(vfrc);
        for iv in (vpix - NMASK)..=(vpix + NMASK) {
            let fv = weight * gcf.eval((iv as f64 - vfrc) as f32);
            let rnorm = iv.rem_euclid(ny as i64) as usize;
            let rconj = (-iv).rem_euclid(ny as i64) as usize;
            for iu in (upix - NMASK)..=(upix + NMASK) {
                let fuv = fv * gcf.eval((iu as f64 - ufrc) as f32);
                *wsum += fuv as f64;
                let rval = re * fuv;
                let ival = im * fuv;
                if iu <= 0 {
                    let idx = rconj * nugrid + (-iu) as usize;
                    mgrid[idx].re += rval;
                    mgrid[idx].im -= ival;
                    bgrid[idx].re += fuv;
                }
                if iu >= 0 {
                    let idx = rnorm * nugrid + iu as usize;
                    mgrid[idx].re += rval;
                    mgrid[idx].im += ival;
                    bgrid[idx].re += fuv;
                }
            }
        }
    };

    for p in &pts {
        let uvrad = (p.uu * p.uu + p.vv * p.vv).sqrt();
        let mut weight = 1.0f32;
        if dotaper {
            weight *= ((gfac * uvrad * uvrad) as f32).exp();
        }
        if pars.dorad {
            weight *= uvrad as f32;
        }
        if dorobust {
            // Briggs: the data weight damped by the local UV density.
            weight *= p.wt.abs() / (1.0 + (bin.wsum(p.uu, p.vv) as f64 * f2) as f32);
        } else {
            if pars.errpow < -0.001 {
                let power = -pars.errpow / 2.0;
                let wt = p.wt.abs();
                if power == 1.0 {
                    weight *= wt;
                } else if power == 0.5 {
                    weight *= wt.sqrt();
                } else {
                    weight *= wt.powf(power);
                }
            }
            if dounif {
                weight /= bin.count(p.uu, p.vv);
            }
        }
        // Beam-size and noise estimation sums.
        {
            let w = weight as f64;
            bm_wsum += w;
            let runwt = w / bm_wsum;
            bm_muu += runwt * (p.uu * p.uu - bm_muu);
            bm_mvv += runwt * (p.vv * p.vv - bm_mvv);
            bm_muv += runwt * (p.uu * p.vv - bm_muv);
            bm_nsum += w * w / p.wt as f64;
        }
        grid_point(&mut mgrid, &mut bgrid, &mut wsum, p.uu, p.vv, p.re, p.im, weight);
    }

    // Optional zero-spacing flux (ignored with radial weighting).
    if pars.uvzero_wt > 0.0 && !pars.dorad {
        let mut weight = 1.0f32;
        if dorobust {
            weight = pars.uvzero_wt / (1.0 + (bin.wsum(0.0, 0.0) as f64 * f2) as f32);
        } else {
            if pars.errpow < -0.001 {
                weight *= pars.uvzero_wt.powf(-pars.errpow / 2.0);
            }
            if dounif {
                weight /= bin.count(0.0, 0.0);
            }
        }
        let re = pars.uvzero_amp - pars.uvzero_modamp;
        // Note: the zero-spacing flux contributes to the gridding
        // weight sum (inside grid_point) but not to the beam moments.
        grid_point(&mut mgrid, &mut bgrid, &mut wsum, 0.0, 0.0, re, 0.0, weight);
    }

    if wsum <= 0.0 || bm_wsum <= 0.0 {
        return Err(GridError::NoData);
    }

    // Normalize by twice the weight sum (conjugate half counted).
    let scale = (1.0 / (2.0 * wsum)) as f32;
    for v in mgrid.iter_mut() {
        *v = *v * scale;
    }
    for v in bgrid.iter_mut() {
        *v = *v * scale;
    }

    // Beam estimate (Tim Pearson's method) and expected noise.
    let ftmp = ((bm_muu - bm_mvv) * (bm_muu - bm_mvv) + 4.0 * bm_muv * bm_muv).sqrt();
    let e_bpa = -0.5 * (2.0 * bm_muv).atan2(bm_muu - bm_mvv);
    let e_bmin = BEAM_FUDGE / (2.0 * (bm_muu + bm_mvv) + 2.0 * ftmp).sqrt();
    let e_bmaj = BEAM_FUDGE / (2.0 * (bm_muu + bm_mvv) - 2.0 * ftmp).sqrt();
    let noise = (bm_nsum / bm_wsum / bm_wsum).sqrt();

    // Transform both grids to the image plane.
    let map = uvtrans(&mut mgrid, nx, ny, &gcf);
    let beam = uvtrans(&mut bgrid, nx, ny, &gcf);

    Ok(MapBeam {
        geom,
        map,
        beam,
        e_bmaj,
        e_bmin,
        e_bpa,
        noise,
        nused: pts.len(),
    })
}

/// Port of uvtrans(): center-shift, inverse FFT to the image plane,
/// and gridding-convolution correction.
///
/// The image transform is I(x,y) = sum_uv V(u,v) exp(-2i*pi*(ux+vy)),
/// matching difmap's newfft(isign=-1) convention, so that a model
/// visibility with phase +2*pi*(ux+vy) peaks at (x,y).
fn uvtrans(grid: &mut [Complex<f32>], nx: usize, ny: usize, gcf: &Gcf) -> Vec<f32> {
    let nugrid = nx / 2 + 1;
    // Center shift: multiply by (-1)^(iu+iv).
    for iv in 0..ny {
        for iu in 0..nugrid {
            if (iu + iv) % 2 == 1 {
                grid[iv * nugrid + iu] = -grid[iv * nugrid + iu];
            }
        }
    }
    // Our FFT libraries implement exp(+2i*pi) for the complex inverse
    // and complex->real directions, so conjugate once to obtain the
    // exp(-2i*pi) transform of a conjugate-symmetric grid.
    for v in grid.iter_mut() {
        v.im = -v.im;
    }
    // Column FFTs along V (complex, length ny).
    let mut cplanner = FftPlanner::<f32>::new();
    let colfft = cplanner.plan_fft_inverse(ny);
    let mut col = vec![Complex::<f32>::new(0.0, 0.0); ny];
    for iu in 0..nugrid {
        for iv in 0..ny {
            col[iv] = grid[iv * nugrid + iu];
        }
        colfft.process(&mut col);
        for iv in 0..ny {
            grid[iv * nugrid + iu] = col[iv];
        }
    }
    // Row transforms along U (complex -> real, length nx).
    let mut rplanner = RealFftPlanner::<f32>::new();
    let rowfft = rplanner.plan_fft_inverse(nx);
    let mut image = vec![0.0f32; nx * ny];
    let mut spec = rowfft.make_input_vec();
    let mut outrow = rowfft.make_output_vec();
    for iv in 0..ny {
        spec.copy_from_slice(&grid[iv * nugrid..(iv + 1) * nugrid]);
        // Bins 0 and nx/2 must be purely real (they are, up to
        // rounding, thanks to conjugate-symmetric gridding).
        spec[0].im = 0.0;
        spec[nx / 2].im = 0.0;
        rowfft
            .process(&mut spec, &mut outrow)
            .expect("real FFT size mismatch");
        image[iv * nx..(iv + 1) * nx].copy_from_slice(&outrow);
    }
    // Gridding convolution correction (normalized reciprocal cosine
    // transforms of the kernel along each axis).
    let rxft = gcf.correction(nx);
    let ryft = gcf.correction(ny);
    for iy in 0..ny {
        let ry = ryft[iy];
        for ix in 0..nx {
            image[iy * nx + ix] *= rxft[ix] * ry;
        }
    }
    image
}
