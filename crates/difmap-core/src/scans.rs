//! Scan detection.
//!
//! Port of difmap's scans.c, which defines a scan by nothing more than
//! a time gap: two integrations separated by more than the gap belong
//! to different scans.

use crate::obs::Observation;

/// The default scan-delimiting gap (seconds): five times the median
/// spacing between consecutive integrations.
///
/// difmap's own default is a flat hour (`DEFGAP` in scans.h), which
/// suits what it uses scans for - breaking a time axis at obvious
/// discontinuities - but not self-calibrating one solution per scan,
/// where the gaps that matter are the slews between scans, minutes
/// rather than hours. Scaling the threshold to the integration time
/// instead means a few dropped integrations do not split a scan, while
/// a slew does.
pub fn default_gap(ob: &Observation) -> f64 {
    let mut dt: Vec<f64> = ob
        .times
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|d| *d > 0.0)
        .collect();
    if dt.is_empty() {
        return f64::INFINITY; // nothing to separate: one scan
    }
    dt.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    5.0 * dt[dt.len() / 2]
}

/// The scan number of each integration, counting from zero.
///
/// `gap` in seconds; zero or negative selects [`default_gap`].
pub fn scan_index(ob: &Observation, gap: f64) -> Vec<u32> {
    let gap = if gap > 0.0 { gap } else { default_gap(ob) };
    let mut out = Vec::with_capacity(ob.times.len());
    let mut scan = 0u32;
    let mut prev = f64::NAN;
    for &t in &ob.times {
        if prev.is_finite() && t - prev > gap {
            scan += 1;
        }
        out.push(scan);
        prev = t;
    }
    out
}

/// The first and last integration index of every scan.
pub fn scans(ob: &Observation, gap: f64) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for (i, &s) in scan_index(ob, gap).iter().enumerate() {
        match out.get_mut(s as usize) {
            Some(span) => span.1 = i,
            None => out.push((i, i)),
        }
    }
    out
}
