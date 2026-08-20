//! Stokes parameters / polarizations and their combination rules.
//!
//! Ported from difmap obs.h (Stokes enum, AIPS codes) and obpol.c
//! (polarization combination semantics).

/// AIPS-convention Stokes / polarization codes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stokes {
    I,
    Q,
    U,
    V,
    RR,
    LL,
    RL,
    LR,
    XX,
    YY,
    XY,
    YX,
    /// Legacy alias of [`Stokes::I`], kept because difmap spelled the
    /// permissive combination "pi" (polarization intensity).
    PI,
}

impl Stokes {
    /// Convert from the AIPS FITS convention code.
    pub fn from_code(code: i32) -> Option<Stokes> {
        Some(match code {
            1 => Stokes::I,
            2 => Stokes::Q,
            3 => Stokes::U,
            4 => Stokes::V,
            -1 => Stokes::RR,
            -2 => Stokes::LL,
            -3 => Stokes::RL,
            -4 => Stokes::LR,
            -5 => Stokes::XX,
            -6 => Stokes::YY,
            -7 => Stokes::XY,
            -8 => Stokes::YX,
            -9 => Stokes::PI,
            _ => return None,
        })
    }

    pub fn code(self) -> i32 {
        match self {
            Stokes::I => 1,
            Stokes::Q => 2,
            Stokes::U => 3,
            Stokes::V => 4,
            Stokes::RR => -1,
            Stokes::LL => -2,
            Stokes::RL => -3,
            Stokes::LR => -4,
            Stokes::XX => -5,
            Stokes::YY => -6,
            Stokes::XY => -7,
            Stokes::YX => -8,
            Stokes::PI => -9,
        }
    }

    pub fn from_name(name: &str) -> Option<Stokes> {
        Some(match name.to_ascii_uppercase().as_str() {
            "I" => Stokes::I,
            "Q" => Stokes::Q,
            "U" => Stokes::U,
            "V" => Stokes::V,
            "RR" => Stokes::RR,
            "LL" => Stokes::LL,
            "RL" => Stokes::RL,
            "LR" => Stokes::LR,
            "XX" => Stokes::XX,
            "YY" => Stokes::YY,
            "XY" => Stokes::XY,
            "YX" => Stokes::YX,
            "PI" => Stokes::PI,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Stokes::I => "I",
            Stokes::Q => "Q",
            Stokes::U => "U",
            Stokes::V => "V",
            Stokes::RR => "RR",
            Stokes::LL => "LL",
            Stokes::RL => "RL",
            Stokes::LR => "LR",
            Stokes::XX => "XX",
            Stokes::YY => "YY",
            Stokes::XY => "XY",
            Stokes::YX => "YX",
            Stokes::PI => "PI",
        }
    }
}

/// A complex visibility with its weight. The weight sign encodes flag
/// status (difmap convention): >0 good, <0 flagged, ==0 deleted/absent.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cvis {
    pub re: f32,
    pub im: f32,
    pub wt: f32,
}

/// How to derive the requested Stokes parameter from the recorded
/// polarizations (indexes into the observation's pol axis).
///
/// Mirrors difmap's `Obpol`/`get_Obpol()` (obpol.c).
#[derive(Clone, Copy, Debug)]
pub enum PolOp {
    /// Directly recorded: take pol index.
    Direct(usize),
    /// (a + b) / 2, e.g. Q = (RL+LR)/2.
    HalfSum(usize, usize),
    /// (a - b) / 2, e.g. V = (RR-LL)/2.
    HalfDiff(usize, usize),
    /// i(a - b) / 2, e.g. U = i(LR-RL)/2.
    HalfIDiff(usize, usize),
    /// Total intensity from the parallel hands, tolerating the absence
    /// of one of them (difmap's "pi", and what [`Stokes::I`] now uses).
    ///
    /// With both hands usable this is the weight-weighted mean, which
    /// for the usual case of equal weights is exactly (RR+LL)/2 = I,
    /// with the same summed weight as the strict combination - so the
    /// flux scale is unchanged. Where only one hand survives, that hand
    /// is used on its own, i.e. the other is assumed identical. That
    /// assumption is exact for unpolarized emission and neglects
    /// circular polarization (RR = I + V, LL = I - V).
    PseudoI(usize, Option<usize>),
}

impl PolOp {
    /// Find a way to construct `stokes` from the recorded pol codes.
    /// Ported from get_Obpol() in obpol.c, with two deliberate
    /// differences: I/Q/U/V may also be derived from linear feeds, and
    /// `I` uses the permissive parallel-hand combination that difmap
    /// called `pi` (see [`PolOp::PseudoI`]).
    pub fn find(pols: &[i32], stokes: Stokes) -> Option<PolOp> {
        let idx = |s: Stokes| pols.iter().position(|&p| p == s.code());
        if let Some(pa) = idx(stokes) {
            return Some(PolOp::Direct(pa));
        }
        match stokes {
            // I and PI are the same operation; PI is kept as a legacy
            // spelling. Circular feeds are preferred, then linear.
            Stokes::I | Stokes::PI => {
                let (mut a, mut b) = (idx(Stokes::RR), idx(Stokes::LL));
                if a.is_none() && b.is_none() {
                    (a, b) = (idx(Stokes::XX), idx(Stokes::YY));
                }
                match (a, b) {
                    (Some(a), Some(b)) => Some(PolOp::PseudoI(a, Some(b))),
                    // Only one hand recorded: assume the other matches.
                    (Some(a), None) | (None, Some(a)) => Some(PolOp::PseudoI(a, None)),
                    (None, None) => None,
                }
            }
            Stokes::V => match (idx(Stokes::RR), idx(Stokes::LL)) {
                (Some(a), Some(b)) => Some(PolOp::HalfDiff(a, b)),
                _ => None,
            },
            Stokes::Q => match (idx(Stokes::RL), idx(Stokes::LR)) {
                (Some(a), Some(b)) => Some(PolOp::HalfSum(a, b)),
                _ => None,
            },
            Stokes::U => match (idx(Stokes::LR), idx(Stokes::RL)) {
                (Some(a), Some(b)) => Some(PolOp::HalfIDiff(a, b)),
                _ => None,
            },
            _ => None,
        }
    }

    /// Extract/combine one visibility from the `npol` recorded
    /// visibilities of a single (row, channel). Exact port of the
    /// getpol functions in obpol.c, including flag/weight semantics.
    #[inline]
    pub fn get(self, pvis: &[Cvis]) -> Cvis {
        #[inline]
        fn combine(a: Cvis, b: Cvis, re: f32, im: f32) -> Cvis {
            if a.wt == 0.0 || b.wt == 0.0 {
                Cvis::default()
            } else {
                let mut wt = 4.0 / (1.0 / a.wt.abs() + 1.0 / b.wt.abs());
                if a.wt < 0.0 || b.wt < 0.0 {
                    wt = -wt;
                }
                Cvis { re, im, wt }
            }
        }
        match self {
            PolOp::Direct(pa) => pvis[pa],
            PolOp::HalfSum(pa, pb) => {
                let (a, b) = (pvis[pa], pvis[pb]);
                combine(a, b, 0.5 * (a.re + b.re), 0.5 * (a.im + b.im))
            }
            PolOp::HalfDiff(pa, pb) => {
                let (a, b) = (pvis[pa], pvis[pb]);
                combine(a, b, 0.5 * (a.re - b.re), 0.5 * (a.im - b.im))
            }
            PolOp::HalfIDiff(pa, pb) => {
                let (a, b) = (pvis[pa], pvis[pb]);
                combine(a, b, -0.5 * (a.im - b.im), 0.5 * (a.re - b.re))
            }
            PolOp::PseudoI(pa, pb) => {
                let a = pvis[pa];
                match pb {
                    None => a,
                    Some(pb) => {
                        let b = pvis[pb];
                        if (a.wt > 0.0 && b.wt > 0.0) || (a.wt < 0.0 && b.wt < 0.0) {
                            let (aw, bw) = (a.wt.abs(), b.wt.abs());
                            Cvis {
                                re: (a.re * aw + b.re * bw) / (aw + bw),
                                im: (a.im * aw + b.im * bw) / (aw + bw),
                                wt: a.wt + b.wt,
                            }
                        } else if a.wt > 0.0 {
                            a
                        } else if b.wt > 0.0 {
                            b
                        } else {
                            Cvis::default()
                        }
                    }
                }
            }
        }
    }
}
