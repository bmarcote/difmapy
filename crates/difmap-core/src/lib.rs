//! difmap-core: in-memory VLBI visibility engine.
//!
//! A modern re-implementation of the numerical core of Difmap
//! (Shepherd 1997). All visibility data are held in RAM; visibility
//! values and weights are never modified (flags live in a separate
//! FLAG array, so all edits are reversible) and calibrations are
//! stored separately and composed on the fly.

pub mod clean;
pub mod closure;
pub mod edit;
pub mod grid;
pub mod model;
pub mod modelfit;
pub mod obs;
pub mod selfcal;
pub mod stokes;
pub mod stream;

pub use clean::{clean, map_stats, restore, CleanResult, Window};
pub use closure::{closure_phases, sampling, spectrum, ClosureSeries, Spectrum};
pub use edit::{edit, edit_rows, EditSelection};
pub use grid::{invert, InvertPars, MapBeam, MapGeom};
pub use model::{CmpType, ModComp};
pub use modelfit::{fit_uvmodel, FitResult};
pub use obs::{Antenna, GainTable, IfBand, Observation, Source};
pub use selfcal::{selfcal, SelfcalPars, SelfcalResult};
pub use stokes::{Cvis, PolOp, Stokes};
pub use stream::Stream;
