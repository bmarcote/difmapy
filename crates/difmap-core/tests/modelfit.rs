//! modelfit tests: fitting must recover the parameters of a known
//! synthetic model (point source and elliptical gaussian).

use difmap_core::model::{recompute_stream_model, CmpType, ModComp};
use difmap_core::modelfit::{fit_uvmodel, M_CENT, M_FLUX, M_MAJOR, M_PHI, M_RATIO};
use difmap_core::obs::{Antenna, IfBand, Observation, Source};
use difmap_core::stokes::{Cvis, Stokes};
use difmap_core::stream::Stream;

const C: f64 = 299792458.0;
const MAS: f64 = std::f64::consts::PI / 180.0 / 3600.0 / 1000.0;

/// Build an observation whose visibilities are exactly those of the
/// given model components (2 IFs, 1 channel, 1 pol, 6 antennas).
fn obs_from_model(comps: &[ModComp]) -> Observation {
    let nant = 6usize;
    let ntime = 40usize;
    let nif = 2usize;
    let antennas: Vec<Antenna> = (0..nant)
        .map(|i| Antenna {
            name: format!("A{i}"),
            xyz: [0.0, 0.0, 0.0],
            subarray: 0,
            fixed: false,
            weight: 1.0,
        })
        .collect();
    let ifs: Vec<IfBand> = (0..nif)
        .map(|i| IfBand {
            freq: 5.0e9 + 1.0e8 * i as f64,
            df: 1.0e6,
            nchan: 1,
            coff: i,
        })
        .collect();

    let mut time = Vec::new();
    let mut ant1 = Vec::new();
    let mut ant2 = Vec::new();
    let mut uvw = Vec::new();
    let mut vis: Vec<Cvis> = Vec::new();

    for it in 0..ntime {
        let t = it as f64 * 60.0;
        for a in 0..nant {
            for b in (a + 1)..nant {
                // Spread baselines over a range of lengths/angles so
                // the fit is well conditioned.
                let blen = 3.0e6 * ((a + b + 1) as f64 / (2 * nant) as f64);
                let theta = std::f64::consts::PI * (it as f64 / ntime as f64)
                    + 0.7 * (a * nant + b) as f64;
                let (u_m, v_m) = (blen * theta.cos(), blen * theta.sin());
                time.push(t);
                ant1.push(a as u32);
                ant2.push(b as u32);
                uvw.extend_from_slice(&[u_m / C, v_m / C, 0.0]);
                for band in &ifs {
                    let f = band.freq;
                    let (uu, vv) = (u_m / C * f, v_m / C * f);
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for c in comps {
                        let (amp, phs) = c.vis(f, uu, vv);
                        let (s, cp) = phs.sin_cos();
                        re += amp * cp;
                        im += amp * s;
                    }
                    vis.push(Cvis {
                        re: re as f32,
                        im: im as f32,
                        wt: 1.0,
                    });
                }
            }
        }
    }
    let n = time.len();
    Observation::new(
        Source {
            name: "FIT".into(),
            ra: 0.0,
            dec: 0.6,
            epoch: 2000.0,
        },
        antennas,
        ifs,
        vec![-1],
        time,
        vec![60.0; n],
        ant1,
        ant2,
        uvw,
        vis,
        Vec::new(),
        60000.0,
    )
    .unwrap()
}

#[test]
fn fit_point_source() {
    let truth = ModComp::delta(1.7, (2.5 * MAS) as f32, (-1.25 * MAS) as f32);
    let mut ob = obs_from_model(&[truth]);
    ob.stream = Some(Stream::select(&ob, Stokes::RR, &[]).unwrap());

    // Start displaced from the truth with flux, x and y free.
    let mut comps = vec![ModComp {
        freepar: M_FLUX | M_CENT,
        flux: 1.0,
        x: (1.0 * MAS) as f32,
        y: 0.0,
        ..ModComp::delta(1.0, 0.0, 0.0)
    }];
    let res = fit_uvmodel(&ob, &mut comps, 40, 0.0, 0.0).expect("fit");

    assert_eq!(res.nfree, 3);
    assert!(res.rchisq < 1e-6, "reduced chi-squared = {}", res.rchisq);
    let c = comps[0];
    assert!((c.flux - truth.flux).abs() < 1e-3, "flux = {}", c.flux);
    assert!(
        (c.x as f64 - truth.x as f64).abs() / MAS < 1e-3,
        "x = {} mas",
        c.x as f64 / MAS
    );
    assert!(
        (c.y as f64 - truth.y as f64).abs() / MAS < 1e-3,
        "y = {} mas",
        c.y as f64 / MAS
    );
    // Formal errors must be finite and small for noiseless data.
    let e = res.errors[0];
    assert!(e.flux.is_finite() && e.flux < 0.01);
    assert!(e.x.is_finite() && e.y.is_finite());
}

#[test]
fn fit_elliptical_gaussian() {
    let truth = ModComp {
        ctype: CmpType::Gaussian,
        flux: 2.2,
        x: (1.5 * MAS) as f32,
        y: (0.8 * MAS) as f32,
        major: (3.0 * MAS) as f32,
        ratio: 0.5,
        phi: 0.6,
        freq0: 0.0,
        spcind: 0.0,
        freepar: 0,
    };
    let mut ob = obs_from_model(&[truth]);
    ob.stream = Some(Stream::select(&ob, Stokes::RR, &[]).unwrap());

    let mut comps = vec![ModComp {
        freepar: M_FLUX | M_CENT | M_MAJOR | M_RATIO | M_PHI,
        flux: 1.5,
        x: 0.0,
        y: 0.0,
        major: (2.0 * MAS) as f32,
        ratio: 0.8,
        phi: 0.2,
        ..truth
    }];
    let res = fit_uvmodel(&ob, &mut comps, 100, 0.0, 0.0).expect("fit");
    assert_eq!(res.nfree, 6); // flux, x, y, X, Y, Z
    let c = comps[0];
    assert!(res.rchisq < 1e-4, "rchisq = {}", res.rchisq);
    assert!((c.flux - truth.flux).abs() / truth.flux < 0.02, "flux={}", c.flux);
    assert!((c.x as f64 - truth.x as f64).abs() / MAS < 0.02);
    assert!((c.y as f64 - truth.y as f64).abs() / MAS < 0.02);
    assert!(
        (c.major as f64 - truth.major as f64).abs() / (truth.major as f64) < 0.03,
        "major = {} mas (truth {})",
        c.major as f64 / MAS,
        truth.major as f64 / MAS
    );
    assert!(
        (c.ratio - truth.ratio).abs() < 0.03,
        "ratio = {} (truth {})",
        c.ratio,
        truth.ratio
    );
    // The position angle is defined modulo pi.
    let dphi = ((c.phi - truth.phi) as f64).rem_euclid(std::f64::consts::PI);
    let dphi = dphi.min(std::f64::consts::PI - dphi);
    assert!(dphi < 0.05, "phi = {} (truth {})", c.phi, truth.phi);
}

#[test]
fn fit_on_top_of_established_model() {
    // Two components: one is already established (fixed), the second
    // must be recovered by the fit from the residuals.
    let established = ModComp::delta(3.0, 0.0, 0.0);
    let extra = ModComp::delta(0.8, (5.0 * MAS) as f32, (2.0 * MAS) as f32);
    let mut ob = obs_from_model(&[established, extra]);
    ob.stream = Some(Stream::select(&ob, Stokes::RR, &[]).unwrap());
    ob.model = vec![established];
    recompute_stream_model(&mut ob);

    let mut comps = vec![ModComp {
        freepar: M_FLUX | M_CENT,
        ..ModComp::delta(0.5, (4.0 * MAS) as f32, (1.0 * MAS) as f32)
    }];
    let res = fit_uvmodel(&ob, &mut comps, 50, 0.0, 0.0).expect("fit");
    assert!(res.rchisq < 1e-6);
    let c = comps[0];
    assert!((c.flux - extra.flux).abs() < 5e-3, "flux = {}", c.flux);
    assert!((c.x as f64 - extra.x as f64).abs() / MAS < 5e-3);
    assert!((c.y as f64 - extra.y as f64).abs() / MAS < 5e-3);
}

#[test]
fn fit_respects_uvrange_and_errors() {
    let truth = ModComp::delta(1.0, 0.0, 0.0);
    let mut ob = obs_from_model(&[truth]);
    ob.stream = Some(Stream::select(&ob, Stokes::RR, &[]).unwrap());
    let mut comps = vec![ModComp {
        freepar: M_FLUX,
        ..ModComp::delta(0.5, 0.0, 0.0)
    }];
    // Restrict to long baselines only: still recovers the flux, but
    // uses fewer visibilities.
    let all = fit_uvmodel(&ob, &mut comps.clone(), 20, 0.0, 0.0).unwrap();
    let cut = fit_uvmodel(&ob, &mut comps, 20, 2.0e7, 1.0e9).unwrap();
    assert!(cut.nvis < all.nvis && cut.nvis > 0);
    assert!((comps[0].flux - 1.0).abs() < 1e-4);

    // No free parameters is an error.
    let mut fixed = vec![ModComp::delta(1.0, 0.0, 0.0)];
    assert!(fit_uvmodel(&ob, &mut fixed, 10, 0.0, 0.0).is_err());
}
