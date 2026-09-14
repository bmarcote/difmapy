//! Scan detection: a gap bigger than the threshold starts a new scan.

use difmap_core::obs::{Antenna, IfBand, Observation, Source};
use difmap_core::scans::{default_gap, scan_index, scans};
use difmap_core::stokes::Cvis;

/// A minimal two-antenna observation sampled at the given times.
fn obs_at(times: &[f64]) -> Observation {
    let antennas: Vec<Antenna> = (0..2)
        .map(|i| Antenna {
            name: format!("A{i}"),
            xyz: [6.0e6, 1.0e5 * i as f64, 1.0e5],
            subarray: 0,
            fixed: false,
            weight: 1.0,
        })
        .collect();
    let ifs = vec![IfBand {
        freq: 5.0e9,
        df: 1.0e6,
        nchan: 1,
        coff: 0,
    }];
    let n = times.len();
    Observation::new(
        Source {
            name: "TEST".into(),
            ra: 0.0,
            dec: 0.5,
            epoch: 2000.0,
        },
        antennas,
        ifs,
        vec![-1i32],
        times.to_vec(),
        vec![10.0f32; n],
        vec![0u32; n],
        vec![1u32; n],
        vec![0.0f64; 3 * n],
        vec![
            Cvis {
                re: 1.0,
                im: 0.0,
                wt: 1.0
            };
            n
        ],
        Vec::new(),
        60000.0,
    )
    .expect("valid observation")
}

/// Integrations every `dt` seconds in `nscan` scans `gap` apart.
fn scan_layout(nscan: usize, per: usize, dt: f64, gap: f64) -> Vec<f64> {
    let mut out = Vec::new();
    let mut t = 0.0;
    for s in 0..nscan {
        if s > 0 {
            t += gap;
        }
        for _ in 0..per {
            out.push(t);
            t += dt;
        }
    }
    out
}

#[test]
fn scans_split_on_the_gaps() {
    let ob = obs_at(&scan_layout(4, 10, 10.0, 300.0));
    // The default threshold is five median spacings: 50 s, well under
    // the 300 s gaps and well over the 10 s spacing.
    assert_eq!(default_gap(&ob), 50.0);
    let found = scans(&ob, 0.0);
    assert_eq!(found, vec![(0, 9), (10, 19), (20, 29), (30, 39)]);
    let idx = scan_index(&ob, 0.0);
    assert_eq!(idx.len(), 40);
    assert_eq!(idx[9], 0);
    assert_eq!(idx[10], 1);
    assert_eq!(*idx.last().unwrap(), 3);
}

#[test]
fn a_gap_longer_than_the_data_gaps_is_one_scan() {
    let ob = obs_at(&scan_layout(4, 10, 10.0, 300.0));
    assert_eq!(scans(&ob, 3600.0), vec![(0, 39)]);
    // ... and a threshold below the integration spacing makes every
    // integration its own scan.
    assert_eq!(scans(&ob, 1.0).len(), 40);
}

#[test]
fn uniform_sampling_is_a_single_scan() {
    let ob = obs_at(&(0..30).map(|i| i as f64 * 10.0).collect::<Vec<_>>());
    assert_eq!(default_gap(&ob), 50.0);
    assert_eq!(scans(&ob, 0.0), vec![(0, 29)]);
}

#[test]
fn one_integration_is_one_scan() {
    let ob = obs_at(&[0.0]);
    assert!(default_gap(&ob).is_infinite());
    assert_eq!(scans(&ob, 0.0), vec![(0, 0)]);
}

#[test]
fn scans_of_unequal_length_are_still_one_scan_each() {
    // Scan lengths 3, 1 and 6 integrations: what a duration-based
    // interval cannot bin one-to-one.
    let mut times = vec![0.0, 10.0, 20.0];
    times.push(400.0);
    times.extend((0..6).map(|i| 800.0 + i as f64 * 10.0));
    let ob = obs_at(&times);
    assert_eq!(scans(&ob, 0.0), vec![(0, 2), (3, 3), (4, 9)]);
}
