//! Python bindings for difmap-core.
//!
//! Exposes a single `CoreObservation` class that owns the in-memory
//! visibility data, the current selection stream, the gain table, the
//! models, and the latest map/beam. The Python package `difmapy`
//! wraps this in a friendlier high-level API.

use difmap_core::clean::{clean, map_stats, restore, Window};
use difmap_core::edit::{edit, edit_rows, EditSelection};
use difmap_core::grid::{invert, InvertPars, MapBeam, MapGeom};
use difmap_core::model::{
    add_to_stream_model, clear_models, merge_model, recompute_stream_model, CmpType, ModComp,
};
use difmap_core::obs::{Antenna, IfBand, Observation, Source};
use difmap_core::selfcal::{selfcal, SelfcalPars};
use difmap_core::stokes::{Cvis, Stokes};
use difmap_core::stream::Stream;
use numpy::{PyArray3, PyArrayMethods, 
    Complex32, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn val_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}
fn run_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pyclass]
struct CoreObservation {
    ob: Observation,
    mb: Option<MapBeam>,
}

fn comp_to_tuple(c: &ModComp) -> (i32, f32, f32, f32, f32, f32, f32, f32, f32) {
    (
        c.ctype.code(),
        c.flux,
        c.x,
        c.y,
        c.major,
        c.ratio,
        c.phi,
        c.freq0,
        c.spcind,
    )
}

#[allow(clippy::too_many_arguments)]
fn comp_from_args(
    ctype: i32,
    flux: f32,
    x: f32,
    y: f32,
    major: f32,
    ratio: f32,
    phi: f32,
    freq0: f32,
    spcind: f32,
) -> PyResult<ModComp> {
    Ok(ModComp {
        ctype: CmpType::from_code(ctype)
            .ok_or_else(|| PyValueError::new_err(format!("bad component type {ctype}")))?,
        flux,
        x,
        y,
        major,
        ratio,
        phi,
        freq0,
        spcind,
        freepar: 0,
    })
}

fn parse_windows(windows: Vec<(f64, f64, f64, f64)>) -> Vec<Window> {
    windows
        .into_iter()
        .map(|(xmin, xmax, ymin, ymax)| Window {
            xmin,
            xmax,
            ymin,
            ymax,
        })
        .collect()
}

#[pymethods]
impl CoreObservation {
    /// Assemble an observation from arrays (see the Python loader).
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (source_name, ra, dec, epoch, ant_names, ant_xyz, ant_subarray,
                        if_freq, if_df, if_nchan, pols, time, inttime, ant1, ant2,
                        uvw, vis, wt, ref_mjd))]
    fn new(
        source_name: &str,
        ra: f64,
        dec: f64,
        epoch: f64,
        ant_names: Vec<String>,
        ant_xyz: PyReadonlyArray2<f64>,
        ant_subarray: Vec<u32>,
        if_freq: Vec<f64>,
        if_df: Vec<f64>,
        if_nchan: Vec<usize>,
        pols: Vec<i32>,
        time: PyReadonlyArray1<f64>,
        inttime: PyReadonlyArray1<f32>,
        ant1: PyReadonlyArray1<u32>,
        ant2: PyReadonlyArray1<u32>,
        uvw: PyReadonlyArray2<f64>,
        vis: PyReadonlyArray3<Complex32>,
        wt: PyReadonlyArray3<f32>,
        ref_mjd: f64,
    ) -> PyResult<Self> {
        let xyz = ant_xyz.as_array();
        if xyz.shape() != [ant_names.len(), 3] {
            return Err(PyValueError::new_err("ant_xyz must be [nant, 3]"));
        }
        if ant_subarray.len() != ant_names.len() {
            return Err(PyValueError::new_err("ant_subarray length mismatch"));
        }
        let antennas: Vec<Antenna> = ant_names
            .into_iter()
            .enumerate()
            .map(|(i, name)| Antenna {
                name,
                xyz: [xyz[[i, 0]], xyz[[i, 1]], xyz[[i, 2]]],
                subarray: ant_subarray[i],
                fixed: false,
                weight: 1.0,
            })
            .collect();
        if if_freq.len() != if_df.len() || if_freq.len() != if_nchan.len() {
            return Err(PyValueError::new_err("IF array length mismatch"));
        }
        let mut coff = 0usize;
        let ifs: Vec<IfBand> = if_freq
            .iter()
            .zip(&if_df)
            .zip(&if_nchan)
            .map(|((&freq, &df), &nchan)| {
                let band = IfBand {
                    freq,
                    df,
                    nchan,
                    coff,
                };
                coff += nchan;
                band
            })
            .collect();
        let visarr = vis.as_array();
        let wtarr = wt.as_array();
        if visarr.shape() != wtarr.shape() {
            return Err(PyValueError::new_err("vis and wt shapes differ"));
        }
        let (nrow, nctotal, npol) = (visarr.shape()[0], visarr.shape()[1], visarr.shape()[2]);
        if nctotal != coff || npol != pols.len() {
            return Err(PyValueError::new_err(format!(
                "vis shape [{nrow},{nctotal},{npol}] does not match IFs/pols"
            )));
        }
        let mut cvis: Vec<Cvis> = Vec::with_capacity(nrow * nctotal * npol);
        for ((_, v), (_, w)) in visarr.indexed_iter().zip(wtarr.indexed_iter()) {
            cvis.push(Cvis {
                re: v.re,
                im: v.im,
                wt: *w,
            });
        }
        let ob = Observation::new(
            Source {
                name: source_name.to_string(),
                ra,
                dec,
                epoch,
            },
            antennas,
            ifs,
            pols,
            time.as_array().to_vec(),
            inttime.as_array().to_vec(),
            ant1.as_array().to_vec(),
            ant2.as_array().to_vec(),
            uvw.as_array().iter().cloned().collect(),
            cvis,
            ref_mjd,
        )
        .map_err(val_err)?;
        Ok(CoreObservation { ob, mb: None })
    }

    // ---------------- header/introspection ----------------

    #[getter]
    fn source_name(&self) -> String {
        self.ob.source.name.clone()
    }
    #[getter]
    fn ra(&self) -> f64 {
        self.ob.source.ra
    }
    #[getter]
    fn dec(&self) -> f64 {
        self.ob.source.dec
    }
    #[getter]
    fn nrow(&self) -> usize {
        self.ob.nrow
    }
    #[getter]
    fn nif(&self) -> usize {
        self.ob.nif()
    }
    #[getter]
    fn npol(&self) -> usize {
        self.ob.npol()
    }
    #[getter]
    fn nsub(&self) -> usize {
        self.ob.nsub
    }
    #[getter]
    fn nctotal(&self) -> usize {
        self.ob.nctotal
    }
    #[getter]
    fn ntimes(&self) -> usize {
        self.ob.ntimes
    }
    #[getter]
    fn ref_mjd(&self) -> f64 {
        self.ob.ref_mjd
    }
    #[getter]
    fn pols(&self) -> Vec<i32> {
        self.ob.pols.clone()
    }
    #[getter]
    fn antenna_names(&self) -> Vec<String> {
        self.ob.antennas.iter().map(|a| a.name.clone()).collect()
    }
    #[getter]
    fn antenna_subarrays(&self) -> Vec<u32> {
        self.ob.antennas.iter().map(|a| a.subarray).collect()
    }
    #[getter]
    fn antenna_xyz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut xyz = Vec::with_capacity(self.ob.antennas.len() * 3);
        for a in &self.ob.antennas {
            xyz.extend_from_slice(&a.xyz);
        }
        Ok(PyArray1::from_vec(py, xyz).reshape([self.ob.antennas.len(), 3])?)
    }

    /// Per-IF (freq of first channel, channel width, nchan).
    #[getter]
    fn ifs(&self) -> Vec<(f64, f64, usize)> {
        self.ob
            .ifs
            .iter()
            .map(|b| (b.freq, b.df, b.nchan))
            .collect()
    }

    fn set_antenna_constraints(&mut self, name: &str, fixed: bool, weight: f32) -> PyResult<()> {
        let mut found = false;
        for a in self.ob.antennas.iter_mut() {
            if a.name.eq_ignore_ascii_case(name) {
                a.fixed = fixed;
                a.weight = weight;
                found = true;
            }
        }
        if found {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("unknown antenna {name}")))
        }
    }

    // ---------------- selection ----------------

    /// Select a polarization and (optional) inclusive channel ranges
    /// over the global channel axis.
    #[pyo3(signature = (stokes, chlist=vec![]))]
    fn select(&mut self, py: Python<'_>, stokes: &str, chlist: Vec<(usize, usize)>) -> PyResult<()> {
        let st = Stokes::from_name(stokes)
            .ok_or_else(|| PyValueError::new_err(format!("unknown polarization {stokes}")))?;
        py.detach(|| -> PyResult<()> {
            let stream = Stream::select(&self.ob, st, &chlist).map_err(val_err)?;
            self.ob.stream = Some(stream);
            recompute_stream_model(&mut self.ob);
            Ok(())
        })?;
        self.mb = None;
        Ok(())
    }

    /// Stokes name and per-IF effective frequency of the selection.
    fn selection<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self
            .ob
            .stream
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no selection"))?;
        let d = PyDict::new(py);
        d.set_item("stokes", s.stokes.name())?;
        d.set_item("chlist", s.chlist.clone())?;
        d.set_item("if_freq", s.if_freq.clone())?;
        d.set_item("if_used", s.if_used.clone())?;
        Ok(d)
    }

    // ---------------- stream data access (for plots) ----------------

    /// Calibrated stream visibilities as (vis[nrow, nif] complex64,
    /// wt[nrow, nif] float32).
    fn stream_vis<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<Complex32>>, Bound<'py, PyArray2<f32>>)> {
        let s = self
            .ob
            .stream
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no selection"))?;
        let nif = self.ob.nif();
        let mut v = Vec::with_capacity(s.vis.len());
        let mut w = Vec::with_capacity(s.vis.len());
        for c in &s.vis {
            v.push(Complex32::new(c.re, c.im));
            w.push(c.wt);
        }
        Ok((
            PyArray1::from_vec(py, v).reshape([self.ob.nrow, nif])?,
            PyArray1::from_vec(py, w).reshape([self.ob.nrow, nif])?,
        ))
    }

    /// Model visibilities as complex64 [nrow, nif].
    fn stream_model<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex32>>> {
        let s = self
            .ob
            .stream
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no selection"))?;
        let m: Vec<Complex32> = s.model.iter().map(|&(re, im)| Complex32::new(re, im)).collect();
        Ok(PyArray1::from_vec(py, m).reshape([self.ob.nrow, self.ob.nif()])?)
    }

    /// Row metadata: (time[s], ant1, ant2, u[sec], v[sec], w[sec]).
    fn rows<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    ) {
        let n = self.ob.nrow;
        let mut u = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        for r in 0..n {
            u.push(self.ob.uvw[r * 3]);
            v.push(self.ob.uvw[r * 3 + 1]);
            w.push(self.ob.uvw[r * 3 + 2]);
        }
        (
            self.ob.time.clone().into_pyarray(py),
            self.ob.ant1.clone().into_pyarray(py),
            self.ob.ant2.clone().into_pyarray(py),
            u.into_pyarray(py),
            v.into_pyarray(py),
            w.into_pyarray(py),
        )
    }

    // ---------------- imaging ----------------

    /// Grid + FFT the current stream into a residual map and beam.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (nx, ny, xinc, yinc, uvmin=0.0, uvmax=0.0, gauval=0.0, gaurad=0.0,
                        dorad=false, errpow=0.0, binwid=0.0,
                        uvzero_amp=0.0, uvzero_modamp=0.0, uvzero_wt=0.0))]
    fn invert<'py>(
        &mut self,
        py: Python<'py>,
        nx: usize,
        ny: usize,
        xinc: f64,
        yinc: f64,
        uvmin: f32,
        uvmax: f32,
        gauval: f32,
        gaurad: f32,
        dorad: bool,
        errpow: f32,
        binwid: f32,
        uvzero_amp: f32,
        uvzero_modamp: f32,
        uvzero_wt: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let geom = MapGeom { nx, ny, xinc, yinc };
        let pars = InvertPars {
            uvmin,
            uvmax,
            gauval,
            gaurad,
            dorad,
            errpow,
            binwid,
            uvzero_amp,
            uvzero_modamp,
            uvzero_wt,
        };
        // difmap establishes the tentative model before inverting.
        let mb = py.detach(|| -> PyResult<MapBeam> {
            merge_model(&mut self.ob);
            invert(&self.ob, geom, &pars).map_err(run_err)
        })?;
        let d = PyDict::new(py);
        d.set_item("e_bmaj", mb.e_bmaj)?;
        d.set_item("e_bmin", mb.e_bmin)?;
        d.set_item("e_bpa", mb.e_bpa)?;
        d.set_item("noise", mb.noise)?;
        d.set_item("nused", mb.nused)?;
        self.mb = Some(mb);
        Ok(d)
    }

    /// The residual dirty map [ny, nx] (after the last invert/clean).
    fn map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mb = self
            .mb
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no map; run invert first"))?;
        Ok(PyArray1::from_slice(py, &mb.map).reshape([mb.geom.ny, mb.geom.nx])?)
    }

    /// The dirty beam [ny, nx].
    fn beam<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let mb = self
            .mb
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no beam; run invert first"))?;
        Ok(PyArray1::from_slice(py, &mb.beam).reshape([mb.geom.ny, mb.geom.nx])?)
    }

    /// Statistics of the residual map inner quarter.
    fn map_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let mb = self
            .mb
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no map; run invert first"))?;
        let s = map_stats(&mb.map, mb.geom.nx, mb.geom.ny);
        let d = PyDict::new(py);
        d.set_item("min", s.min)?;
        d.set_item("max", s.max)?;
        d.set_item("mean", s.mean)?;
        d.set_item("rms", s.rms)?;
        d.set_item("minpos", s.minpos)?;
        d.set_item("maxpos", s.maxpos)?;
        Ok(d)
    }

    /// Högbom CLEAN of the residual map. Windows are
    /// (xmin, xmax, ymin, ymax) tuples in radians; empty = whole area.
    #[pyo3(signature = (niter=100, gain=0.05, cutoff=0.0, windows=vec![]))]
    fn clean<'py>(
        &mut self,
        py: Python<'py>,
        niter: i32,
        gain: f32,
        cutoff: f32,
        windows: Vec<(f64, f64, f64, f64)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mb = self
            .mb
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("no map; run invert first"))?;
        let wins = parse_windows(windows);
        let res =
            py.detach(|| clean(mb, &wins, niter, gain, cutoff).map_err(run_err))?;
        self.ob.newmod.extend(res.comps.iter().cloned());
        let d = PyDict::new(py);
        d.set_item("niter", res.niter)?;
        d.set_item("cleaned_flux", res.cleaned_flux)?;
        d.set_item("ncomp", res.comps.len())?;
        d.set_item("hit_cutoff", res.hit_cutoff)?;
        d.set_item("hit_negative", res.hit_negative)?;
        Ok(d)
    }

    /// Restore: convolve model with the restoring beam and add the
    /// residuals. Returns the restored map [ny, nx].
    #[pyo3(signature = (bmaj, bmin, bpa, no_residual=false, do_smooth=false, freq=0.0))]
    fn restore<'py>(
        &mut self,
        py: Python<'py>,
        bmaj: f64,
        bmin: f64,
        bpa: f64,
        no_residual: bool,
        do_smooth: bool,
        freq: f64,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        // Establish the tentative model first, as difmap does.
        merge_model(&mut self.ob);
        let mb = self
            .mb
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("no map; run invert first"))?;
        let comps = &self.ob.model;
        let cln = py.detach(|| {
            restore(mb, comps, bmaj, bmin, bpa, no_residual, do_smooth, freq)
        });
        Ok(PyArray1::from_vec(py, cln).reshape([mb.geom.ny, mb.geom.nx])?)
    }

    // ---------------- model ----------------

    /// Established + tentative model components as tuples
    /// (type, flux, x, y, major, ratio, phi, freq0, spcind).
    fn get_models<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
        let old = PyList::new(py, self.ob.model.iter().map(comp_to_tuple))?;
        let new = PyList::new(py, self.ob.newmod.iter().map(comp_to_tuple))?;
        Ok((old, new))
    }

    /// Add a component to the tentative model (difmap addcmp).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ctype, flux, x, y, major=0.0, ratio=1.0, phi=0.0, freq0=0.0, spcind=0.0))]
    fn add_component(
        &mut self,
        ctype: i32,
        flux: f32,
        x: f32,
        y: f32,
        major: f32,
        ratio: f32,
        phi: f32,
        freq0: f32,
        spcind: f32,
    ) -> PyResult<()> {
        self.ob
            .newmod
            .push(comp_from_args(ctype, flux, x, y, major, ratio, phi, freq0, spcind)?);
        Ok(())
    }

    /// Establish the tentative model (difmap keep).
    fn keep(&mut self, py: Python<'_>) {
        py.detach(|| merge_model(&mut self.ob));
    }

    /// Clear models (difmap clrmod).
    #[pyo3(signature = (do_old=true, do_new=true))]
    fn clear_models(&mut self, py: Python<'_>, do_old: bool, do_new: bool) {
        py.detach(|| clear_models(&mut self.ob, do_old, do_new));
    }

    /// Replace the established model with the given components and
    /// recompute the stream model.
    fn set_model(&mut self, py: Python<'_>, comps: Vec<(i32, f32, f32, f32, f32, f32, f32, f32, f32)>) -> PyResult<()> {
        let mut model = Vec::with_capacity(comps.len());
        for (t, f, x, y, maj, r, p, f0, si) in comps {
            model.push(comp_from_args(t, f, x, y, maj, r, p, f0, si)?);
        }
        py.detach(|| {
            // Remove old model contribution, install the new one.
            let old = std::mem::take(&mut self.ob.model);
            add_to_stream_model(&mut self.ob, &old, true);
            add_to_stream_model(&mut self.ob, &model, false);
            self.ob.model = model;
        });
        Ok(())
    }

    // ---------------- calibration ----------------

    /// Self-calibration. Returns a summary dict.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (doamp=false, dophs=true, dofloat=false, solint=0.0, doone=false,
                        gauval=0.0, gaurad=0.0, maxamp=0.0, maxphs=0.0,
                        uvmin=0.0, uvmax=0.0, mintel=0, doflag=false))]
    fn selfcal<'py>(
        &mut self,
        py: Python<'py>,
        doamp: bool,
        dophs: bool,
        dofloat: bool,
        solint: f32,
        doone: bool,
        gauval: f32,
        gaurad: f32,
        maxamp: f32,
        maxphs: f32,
        uvmin: f32,
        uvmax: f32,
        mintel: usize,
        doflag: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pars = SelfcalPars {
            doamp,
            dophs,
            dofloat,
            solint,
            doone,
            gauval,
            gaurad,
            maxamp,
            maxphs,
            uvmin,
            uvmax,
            mintel: if mintel == 0 {
                if doamp {
                    4
                } else {
                    3
                }
            } else {
                mintel
            },
            doflag,
        };
        let res = py.detach(|| selfcal(&mut self.ob, &pars).map_err(run_err))?;
        self.mb = None; // corrected data invalidate the current map
        let d = PyDict::new(py);
        d.set_item("nbadsol", res.nbadsol)?;
        d.set_item("nbadtel", res.nbadtel)?;
        d.set_item("norms", res.norms)?;
        d.set_item("nbins", res.nbins)?;
        Ok(d)
    }

    /// Undo telescope calibrations (difmap uncalib).
    #[pyo3(signature = (do_amp=true, do_phs=true, do_flags=false))]
    fn uncalib(&mut self, py: Python<'_>, do_amp: bool, do_phs: bool, do_flags: bool) {
        py.detach(|| {
            self.ob.gains.reset(do_amp, do_phs, do_flags);
            let mut stream = self.ob.stream.take();
            if let Some(s) = stream.as_mut() {
                s.apply_calibration(&self.ob);
            }
            self.ob.stream = stream;
        });
        self.mb = None;
    }

    // ---------------- editing / flagging ----------------

    /// Flag or unflag visibilities matching the given criteria
    /// (difmap flag/unflag). Antenna indices are global; time range in
    /// seconds since ref_mjd. Returns the number of rows affected.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (flag, tmin=None, tmax=None, baseline=None, station=None,
                        subarray=None, if_index=None, sel_chan=false))]
    fn edit(
        &mut self,
        py: Python<'_>,
        flag: bool,
        tmin: Option<f64>,
        tmax: Option<f64>,
        baseline: Option<(u32, u32)>,
        station: Option<u32>,
        subarray: Option<u32>,
        if_index: Option<usize>,
        sel_chan: bool,
    ) -> usize {
        let sel = EditSelection {
            time_range: match (tmin, tmax) {
                (None, None) => None,
                (a, b) => Some((a.unwrap_or(f64::MIN), b.unwrap_or(f64::MAX))),
            },
            baseline,
            station,
            subarray,
            if_index,
            sel_chan,
        };
        let n = py.detach(|| edit(&mut self.ob, &sel, flag));
        if n > 0 {
            self.mb = None;
        }
        n
    }

    /// Flag or unflag explicit row indices (interactive plot editing).
    #[pyo3(signature = (rows, flag, if_index=None, sel_chan=false))]
    fn edit_rows(
        &mut self,
        py: Python<'_>,
        rows: Vec<usize>,
        flag: bool,
        if_index: Option<usize>,
        sel_chan: bool,
    ) -> PyResult<()> {
        if rows.iter().any(|&r| r >= self.ob.nrow) {
            return Err(PyValueError::new_err("row index out of range"));
        }
        py.detach(|| edit_rows(&mut self.ob, &rows, if_index, sel_chan, flag));
        self.mb = None;
        Ok(())
    }

    /// Integration times per row (seconds).
    fn inttimes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        PyArray1::from_slice(py, &self.ob.inttime)
    }

    /// The full raw cube with the current calibrations applied:
    /// (vis[nrow, nctotal, npol] complex64, wt[nrow, nctotal, npol]).
    /// Flag state is preserved in the weight signs. Used by writers.
    fn calibrated_cube<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray3<Complex32>>,
        Bound<'py, PyArray3<f32>>,
    )> {
        let ob = &self.ob;
        let (nrow, nctotal, npol) = (ob.nrow, ob.nctotal, ob.npol());
        let n = nrow * nctotal * npol;
        let mut vis: Vec<Complex32> = Vec::with_capacity(n);
        let mut wt: Vec<f32> = Vec::with_capacity(n);
        // Global channel -> IF index map.
        let mut chan_if = vec![0usize; nctotal];
        for (cif, band) in ob.ifs.iter().enumerate() {
            for c in band.coff..band.coff + band.nchan {
                chan_if[c] = cif;
            }
        }
        py.detach(|| {
            let gains = &ob.gains;
            for row in 0..nrow {
                let it = ob.time_idx[row] as usize;
                let (a1, a2) = (ob.ant1[row] as usize, ob.ant2[row] as usize);
                for gc in 0..nctotal {
                    let cif = chan_if[gc];
                    let ia = gains.idx(it, cif, a1);
                    let ib = gains.idx(it, cif, a2);
                    let ampcor = gains.amp[ia] * gains.amp[ib];
                    let phscor = gains.phs[ia] - gains.phs[ib];
                    let bad = gains.bad[ia] || gains.bad[ib];
                    let (s, c) = phscor.sin_cos();
                    for p in 0..npol {
                        let v = ob.vis[(row * nctotal + gc) * npol + p];
                        let mut w = v.wt;
                        let (re, im) = if w != 0.0 && (ampcor != 1.0 || phscor != 0.0) {
                            w /= ampcor * ampcor;
                            (
                                ampcor * (v.re * c - v.im * s),
                                ampcor * (v.re * s + v.im * c),
                            )
                        } else {
                            (v.re, v.im)
                        };
                        if bad && w > 0.0 {
                            w = -w;
                        }
                        vis.push(Complex32::new(re, im));
                        wt.push(w);
                    }
                }
            }
        });
        Ok((
            PyArray1::from_vec(py, vis).reshape([nrow, nctotal, npol])?,
            PyArray1::from_vec(py, wt).reshape([nrow, nctotal, npol])?,
        ))
    }

    /// Gain table (amp[nt, nif, nant], phs, bad) copies.
    fn gains<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
    )> {
        let g = &self.ob.gains;
        Ok((
            PyArray1::from_slice(py, &g.amp),
            PyArray1::from_slice(py, &g.phs),
            PyArray1::from_vec(py, g.bad.clone()),
        ))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<CoreObservation>()?;
    Ok(())
}
