//! End-to-end tests on synthetic data: a point source of known flux
//! and position must invert to a dirty map peaking at the right pixel
//! with the right amplitude, and the dirty beam must peak at 1.0.

use difmap_core::clean::{clean, restore};
use difmap_core::grid::{invert, InvertPars, MapGeom};
use difmap_core::model::{merge_model, recompute_stream_model, ModComp};
use difmap_core::obs::{Antenna, IfBand, Observation, Source};
use difmap_core::selfcal::{selfcal, SelfcalPars};
use difmap_core::stokes::{Cvis, Stokes};
use difmap_core::stream::Stream;

const C: f64 = 299792458.0;

/// Build a synthetic observation of a point source with flux `flux`
/// (Jy) offset by (x0, y0) radians from the phase center.
/// 2 IFs x 4 channels, RR+LL, 5 antennas, 60 integrations.
fn synthetic_obs(flux: f32, x0: f64, y0: f64) -> Observation {
    synthetic_obs_with_gains(flux, x0, y0, None)
}

/// As `synthetic_obs`, but optionally corrupt the visibilities with
/// per-antenna gain errors (amp, phase): V *= Aa*Ab * e^(i(pa-pb)).
fn synthetic_obs_with_gains(
    flux: f32,
    x0: f64,
    y0: f64,
    gerr: Option<&[(f32, f32)]>,
) -> Observation {
    let nant = 5usize;
    let ntime = 60usize;
    let nif = 2usize;
    let nchan = 4usize;
    let pols = vec![-1i32, -2]; // RR, LL

    let antennas: Vec<Antenna> = (0..nant)
        .map(|i| Antenna {
            name: format!("AN{i}"),
            xyz: [6.0e6 + 1.0e5 * i as f64, 1.0e5 * i as f64, 1.0e5],
            subarray: 0,
            fixed: false,
            weight: 1.0,
        })
        .collect();

    let ifs: Vec<IfBand> = (0..nif)
        .map(|i| IfBand {
            freq: 4.9e9 + 1.0e8 * i as f64,
            df: 1.0e6,
            nchan,
            coff: i * nchan,
        })
        .collect();
    let nctotal = nif * nchan;

    let mut time = Vec::new();
    let mut inttime = Vec::new();
    let mut ant1 = Vec::new();
    let mut ant2 = Vec::new();
    let mut uvw = Vec::new();
    let mut vis: Vec<Cvis> = Vec::new();

    // Baseline "lengths" in metres for synthetic circular uv tracks.
    for it in 0..ntime {
        let t = it as f64 * 60.0;
        for a in 0..nant {
            for b in (a + 1)..nant {
                let blen = 1.0e6 * ((a + b) as f64 + 1.0); // metres
                let theta = 0.7 * (it as f64 / ntime as f64) * std::f64::consts::PI
                    + (a * nant + b) as f64;
                let u_m = blen * theta.cos();
                let v_m = blen * theta.sin();
                time.push(t);
                inttime.push(60.0f32);
                ant1.push(a as u32);
                ant2.push(b as u32);
                uvw.extend_from_slice(&[u_m / C, v_m / C, 0.0]);
                // Optional per-antenna gain corruption.
                let (gamp, gphs) = match gerr {
                    Some(g) => (g[a].0 * g[b].0, (g[a].1 - g[b].1) as f64),
                    None => (1.0, 0.0),
                };
                for band in &ifs {
                    for ch in 0..nchan {
                        let f = band.chan_freq(ch);
                        let uu = (u_m / C) * f;
                        let vv = (v_m / C) * f;
                        let phs = 2.0 * std::f64::consts::PI * (uu * x0 + vv * y0) + gphs;
                        let (s, c) = phs.sin_cos();
                        for _pol in 0..2 {
                            vis.push(Cvis {
                                re: gamp * flux * c as f32,
                                im: gamp * flux * s as f32,
                                wt: 1.0,
                            });
                        }
                    }
                }
            }
        }
    }

    Observation::new(
        Source {
            name: "TEST".into(),
            ra: 0.0,
            dec: 0.5,
            epoch: 2000.0,
        },
        antennas,
        ifs,
        pols,
        time,
        inttime,
        ant1,
        ant2,
        uvw,
        vis,
        Vec::new(), // no flags initially
        60000.0,
    )
    .expect("valid synthetic observation")
}

const MAS: f64 = std::f64::consts::PI / 180.0 / 3600.0 / 1000.0;

#[test]
fn point_source_invert() {
    let flux = 2.5f32;
    // Source offset in radians: +8 pixels east, -5 pixels north at
    // 0.5 mas/pixel.
    let cell = 0.5 * MAS;
    let (x0, y0) = (8.0 * cell, -5.0 * cell);
    let mut ob = synthetic_obs(flux, x0, y0);

    let stream = Stream::select(&ob, Stokes::I, &[]).expect("select I");
    // All IFs used, mean freq of 4 channels: freq + 1.5*df.
    assert!(stream.if_used.iter().all(|&u| u));
    assert!((stream.if_freq[0] - (4.9e9 + 1.5e6)).abs() < 1.0);
    ob.stream = Some(stream);

    let (nx, ny) = (256usize, 256usize);
    let geom = MapGeom {
        nx,
        ny,
        xinc: cell,
        yinc: cell,
    };
    let mb = invert(&ob, geom, &InvertPars::default()).expect("invert");

    // The beam must peak at exactly the map center with value ~1.
    let bpeak = mb.beam[(ny / 2) * nx + nx / 2];
    assert!(
        (bpeak - 1.0).abs() < 0.01,
        "beam center = {bpeak}, expected 1.0"
    );
    let (mut bmax, mut bargmax) = (f32::MIN, 0usize);
    for (i, &b) in mb.beam.iter().enumerate() {
        if b > bmax {
            bmax = b;
            bargmax = i;
        }
    }
    assert_eq!(
        (bargmax % nx, bargmax / nx),
        (nx / 2, ny / 2),
        "beam peak position"
    );

    // The dirty map must peak at the source pixel with ~the flux.
    let (mut mmax, mut margmax) = (f32::MIN, 0usize);
    for (i, &m) in mb.map.iter().enumerate() {
        if m > mmax {
            mmax = m;
            margmax = i;
        }
    }
    let (ix, iy) = (margmax % nx, margmax / nx);
    assert_eq!(
        (ix, iy),
        (nx / 2 + 8, ny / 2 - 5),
        "map peak position (map peak = {mmax})"
    );
    assert!(
        (mmax - flux).abs() / flux < 0.02,
        "map peak = {mmax}, expected ~{flux}"
    );
}

#[test]
fn flagged_channels_average() {
    let mut ob = synthetic_obs(1.0, 0.0, 0.0);
    // Flag one channel of the first row (both pols) via the FLAG
    // column: the average of the remaining channels must still be good.
    let npol = ob.npol();
    for p in 0..npol {
        ob.flag[p] = true; // row 0, gchan 0
    }
    let stream = Stream::select(&ob, Stokes::I, &[]).expect("select I");
    // Row 0 IF 0 is flagged (one flagged channel flags the average, as
    // in difmap); IF 1 is unaffected.
    assert!(stream.vis[0].wt < 0.0);
    assert!(stream.vis[1].wt > 0.0);
    // A deleted channel (zero weight) deletes the averaged visibility.
    for p in 0..npol {
        ob.vis[npol + p].wt = 0.0; // row 0, gchan 1
    }
    let stream = Stream::select(&ob, Stokes::I, &[]).expect("select I");
    assert_eq!(stream.vis[0].wt, 0.0);

    // Restricting the channel selection to IF 2 only (gchan 4..7)
    // excludes IF 1 from the stream.
    let stream = Stream::select(&ob, Stokes::I, &[(4, 7)]).expect("select");
    assert!(!stream.if_used[0] && stream.if_used[1]);

    // Stokes combination sanity: I == RR here, weight doubles.
    let s_i = Stream::select(&ob, Stokes::I, &[]).unwrap();
    let s_rr = Stream::select(&ob, Stokes::RR, &[]).unwrap();
    let (a, b) = (s_i.vis[1], s_rr.vis[1]);
    assert!((a.re - b.re).abs() < 1e-6 && (a.im - b.im).abs() < 1e-6);
    assert!((a.wt - 2.0 * b.wt).abs() < 1e-5);
}

#[test]
fn clean_and_restore_point_source() {
    let flux = 2.5f32;
    let cell = 0.5 * MAS;
    let (x0, y0) = (8.0 * cell, -5.0 * cell);
    let mut ob = synthetic_obs(flux, x0, y0);
    ob.stream = Some(Stream::select(&ob, Stokes::I, &[]).unwrap());

    let (nx, ny) = (256usize, 256usize);
    let geom = MapGeom {
        nx,
        ny,
        xinc: cell,
        yinc: cell,
    };
    let mut mb = invert(&ob, geom, &InvertPars::default()).unwrap();

    // CLEAN with a window around the source.
    let win = difmap_core::clean::Window {
        xmin: x0 - 5.0 * cell,
        xmax: x0 + 5.0 * cell,
        ymin: y0 - 5.0 * cell,
        ymax: y0 + 5.0 * cell,
    };
    let res = clean(&mut mb, &[win], 300, 0.1, 0.0).unwrap();
    assert!(
        (res.cleaned_flux - flux as f64).abs() / (flux as f64) < 0.02,
        "cleaned flux = {} expected ~{flux}",
        res.cleaned_flux
    );
    // The dominant merged component must sit at the source position.
    let main = res
        .comps
        .iter()
        .max_by(|a, b| a.flux.partial_cmp(&b.flux).unwrap())
        .unwrap();
    assert!((main.x as f64 - x0).abs() < 0.51 * cell);
    assert!((main.y as f64 - y0).abs() < 0.51 * cell);
    assert!((main.flux as f64) > 0.9 * flux as f64);

    // Residual map must be nearly empty now.
    let stats = difmap_core::clean::map_stats(&mb.map, nx, ny);
    assert!(
        stats.max.abs().max(stats.min.abs()) < 0.05 * flux,
        "residual peak = {} / {}",
        stats.min,
        stats.max
    );

    // Restore with the estimated beam: peak ~ flux at the source pixel.
    let cln = restore(
        &mb,
        &res.comps,
        mb.e_bmaj,
        mb.e_bmin,
        mb.e_bpa,
        false,
        false,
        4.95e9,
    );
    let (mut mmax, mut argmax) = (f32::MIN, 0usize);
    for (i, &v) in cln.iter().enumerate() {
        if v > mmax {
            mmax = v;
            argmax = i;
        }
    }
    assert_eq!((argmax % nx, argmax / nx), (nx / 2 + 8, ny / 2 - 5));
    assert!(
        (mmax - flux).abs() / flux < 0.05,
        "restored peak = {mmax}, expected ~{flux}"
    );

    // Establish the model and re-invert: the residual map should be
    // nearly flat (validates model visibility computation).
    ob.newmod = res.comps.clone();
    merge_model(&mut ob);
    let mb2 = invert(&ob, geom, &InvertPars::default()).unwrap();
    let stats2 = difmap_core::clean::map_stats(&mb2.map, nx, ny);
    assert!(
        stats2.max.abs().max(stats2.min.abs()) < 0.05 * flux,
        "model-subtracted residual = {} / {}",
        stats2.min,
        stats2.max
    );
}

#[test]
fn selfcal_recovers_gains() {
    let flux = 1.5f32;
    // Known per-antenna gain errors (amp, phase in radians).
    let gerr = [
        (1.30f32, 0.40f32),
        (0.75, -0.30),
        (1.10, 0.15),
        (0.90, -0.50),
        (1.05, 0.25),
    ];
    let mut ob = synthetic_obs_with_gains(flux, 0.0, 0.0, Some(&gerr));
    ob.stream = Some(Stream::select(&ob, Stokes::I, &[]).unwrap());

    // The true model: the point source at the phase center.
    ob.newmod = vec![ModComp::delta(flux, 0.0, 0.0)];
    merge_model(&mut ob);
    recompute_stream_model(&mut ob);

    // Amplitude + phase self-cal, floating flux scale, per-integration.
    let pars = SelfcalPars {
        doamp: true,
        dophs: true,
        dofloat: true,
        mintel: 4,
        ..Default::default()
    };
    let res = selfcal(&mut ob, &pars).unwrap();
    assert_eq!(res.nbadsol, 0);

    // The corrected data must now match the model closely.
    let stream = ob.stream.as_ref().unwrap();
    let nif = ob.nif();
    let mut worst = 0.0f32;
    for row in 0..ob.nrow {
        for cif in 0..nif {
            let v = stream.vis[row * nif + cif];
            let m = stream.model[row * nif + cif];
            if v.wt > 0.0 {
                let d = ((v.re - m.0).powi(2) + (v.im - m.1).powi(2)).sqrt();
                worst = worst.max(d);
            }
        }
    }
    assert!(
        worst < 0.01 * flux,
        "worst corrected-data/model mismatch = {worst}"
    );

    // The applied amplitude corrections must approximate 1/amp_err.
    let nant = ob.antennas.len();
    for (ia, &(aerr, _)) in gerr.iter().enumerate() {
        let idx = ob.gains.idx(10, 0, ia); // integration 10, IF 0
        let expect = 1.0 / aerr;
        let got = ob.gains.amp[idx];
        assert!(
            (got - expect).abs() / expect < 0.02,
            "ant {ia}: amp_cor = {got}, expected ~{expect}"
        );
        assert!(ia < nant);
    }
}

#[test]
fn calibration_roundtrip() {
    let flux = 1.0f32;
    let mut ob = synthetic_obs(flux, 0.0, 0.0);
    let stream = Stream::select(&ob, Stokes::I, &[]).unwrap();
    ob.stream = Some(stream);

    // Apply a known gain error to antenna 2 in IF 0 at all times.
    let (nif, nant) = (ob.nif(), ob.antennas.len());
    for it in 0..ob.ntimes {
        let idx = (it * nif) * nant + 2;
        ob.gains.amp[idx] = 2.0;
        ob.gains.phs[idx] = 0.5;
    }
    let mut stream = ob.stream.take().unwrap();
    stream.apply_calibration(&ob);

    // Rows on baselines including antenna 2 in IF 0 are scaled by 2
    // and rotated; others untouched.
    for row in 0..ob.nrow {
        let v = stream.vis[row * nif];
        let r = stream.raw[row * nif];
        let (a1, a2) = (ob.ant1[row], ob.ant2[row]);
        let amp = (v.re * v.re + v.im * v.im).sqrt();
        let ramp = (r.re * r.re + r.im * r.im).sqrt();
        if a1 == 2 || a2 == 2 {
            assert!((amp - 2.0 * ramp).abs() < 1e-4);
            assert!((v.wt - r.wt / 4.0).abs() < 1e-6);
        } else {
            assert!((amp - ramp).abs() < 1e-4);
        }
        // IF 1 untouched everywhere.
        let v1 = stream.vis[row * nif + 1];
        let r1 = stream.raw[row * nif + 1];
        assert_eq!(v1.re, r1.re);
        assert_eq!(v1.wt, r1.wt);
    }

    // Reset (uncalib) and re-apply: vis == raw again.
    ob.gains.reset(true, true, true);
    stream.apply_calibration(&ob);
    for row in 0..ob.nrow {
        assert_eq!(stream.vis[row * nif].re, stream.raw[row * nif].re);
    }
}

/// Briggs robust weighting must interpolate between difmap's own
/// uniform and natural weighting, and must not need a power-of-two map.
#[test]
fn robust_weighting_spans_uniform_to_natural() {
    let cell = 0.5 * MAS;
    let mut ob = synthetic_obs(2.5, 8.0 * cell, -5.0 * cell);
    ob.stream = Some(Stream::select(&ob, Stokes::I, &[]).expect("select I"));
    // Not a power of two: any multiple of four is a valid map size.
    let geom = MapGeom {
        nx: 320,
        ny: 320,
        xinc: cell,
        yinc: cell,
    };

    let beam_of = |pars: &InvertPars| -> (f64, f64) {
        let mb = invert(&ob, geom, pars).expect("invert");
        (mb.e_bmaj, mb.noise)
    };

    // difmap's own two extremes.
    let (uniform, uniform_noise) = beam_of(&InvertPars {
        binwid: 2.0,
        ..Default::default()
    });
    let (natural, natural_noise) = beam_of(&InvertPars {
        errpow: -2.0,
        ..Default::default()
    });
    assert!(uniform < natural, "uniform beam {uniform} !< natural {natural}");
    assert!(natural_noise < uniform_noise);

    let mut prev = 0.0;
    for (i, r) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let (bmaj, _) = beam_of(&InvertPars {
            binwid: 2.0,
            robust: Some(*r),
            ..Default::default()
        });
        assert!(bmaj > prev, "beam must grow with robustness (R = {r})");
        prev = bmaj;
        if i == 0 {
            // R = -2 is uniform to within the binning approximation.
            assert!((bmaj - uniform).abs() / uniform < 0.2, "R=-2: {bmaj} vs {uniform}");
        }
        if i == 4 {
            assert!((bmaj - natural).abs() / natural < 0.02, "R=2: {bmaj} vs {natural}");
        }
    }
}

/// Map dimensions must be multiples of four, but need not be powers of
/// two any more.
#[test]
fn map_dimensions_need_only_be_multiples_of_four() {
    let cell = 0.5 * MAS;
    let mut ob = synthetic_obs(2.5, 0.0, 0.0);
    ob.stream = Some(Stream::select(&ob, Stokes::I, &[]).expect("select I"));
    let ok = MapGeom {
        nx: 132,
        ny: 260,
        xinc: cell,
        yinc: cell,
    };
    let mb = invert(&ob, ok, &InvertPars::default()).expect("132x260 must invert");
    assert_eq!(mb.map.len(), 132 * 260);
    let bad = MapGeom {
        nx: 130,
        ny: 260,
        xinc: cell,
        yinc: cell,
    };
    assert!(invert(&ob, bad, &InvertPars::default()).is_err());
}
