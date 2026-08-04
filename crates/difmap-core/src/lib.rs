//! difmap-core: in-memory VLBI visibility engine.
//!
//! A modern re-implementation of the numerical core of Difmap
//! (Shepherd 1997). All visibility data are held in RAM; raw data are
//! immutable and calibrations/flags are stored separately and composed
//! on the fly.

pub mod clean;
pub mod grid;
pub mod model;
pub mod obs;
pub mod selfcal;
pub mod stokes;
pub mod stream;

pub use clean::{clean, map_stats, restore, CleanResult, Window};
pub use grid::{invert, InvertPars, MapBeam, MapGeom};
pub use model::{CmpType, ModComp};
pub use obs::{Antenna, GainTable, IfBand, Observation, Source};
pub use selfcal::{selfcal, SelfcalPars, SelfcalResult};
pub use stokes::{Cvis, PolOp, Stokes};
pub use stream::Stream;
