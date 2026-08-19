"""High-level difmap-like observation API.

Mirrors the classic difmap workflow::

    obs = difmapy.load("data.uvf")     # observe
    obs.select("I")                    # select
    obs.mapsize(512, 0.2)              # mapsize (mas/pixel)
    obs.invert()                       # invert
    obs.add_window(-2, 2, -2, 2)       # clean window (mas)
    obs.clean(200, 0.05)               # clean
    obs.selfcal(phase=True)            # selfcal
    m = obs.restore()                  # restore
    obs.wmap("clean.fits")             # wmap

All user-facing positions/sizes are in milliarcseconds (mas), angles in
degrees and UV radii in wavelengths; everything is converted to
radians internally.
"""

from __future__ import annotations

import numpy as np

from difmapy._core import CoreObservation

MAS = np.pi / (180.0 * 3600.0 * 1000.0)  # mas -> radians
DEG = np.pi / 180.0

CMP_TYPES = {0: "delta", 1: "gauss", 2: "disk", 3: "ellipse", 4: "ring", 5: "rect", 6: "sz"}
CMP_CODES = {v: k for k, v in CMP_TYPES.items()}

# Free-parameter bits for modelfit (difmap M_FLUX etc.).
FREE_BITS = {
    "flux": 1,
    "pos": 2,      # x and y together
    "major": 4,
    "ratio": 8,
    "phi": 16,
    "spcind": 32,
}
# Fitting a non-circular shape requires all three shape parameters,
# because they are fitted through the (X, Y, Z) parameterization.
SHAPE_BITS = FREE_BITS["major"] | FREE_BITS["ratio"] | FREE_BITS["phi"]


def _free_mask(free) -> int:
    """Translate a free-parameter spec into difmap's bitmask."""
    if free is None:
        return 0
    if isinstance(free, int):
        return free
    if isinstance(free, str):
        free = [free]
    mask = 0
    for name in free:
        key = str(name).lower()
        if key in ("xy", "x", "y", "position"):
            key = "pos"
        if key == "shape":
            mask |= SHAPE_BITS
            continue
        if key not in FREE_BITS:
            raise ValueError(
                f"unknown free parameter {name!r}; use "
                f"{sorted(FREE_BITS) + ['shape']}"
            )
        mask |= FREE_BITS[key]
    # ratio/phi cannot be fitted without major (shared X,Y,Z parameters).
    if mask & (FREE_BITS["ratio"] | FREE_BITS["phi"]):
        mask |= SHAPE_BITS
    return mask


__all__ = ["Observation", "load"]


class Observation:
    """An in-memory interferometric observation (difmap-style)."""

    def __init__(self, core: CoreObservation):
        self._core = core
        # Imaging state (difmap mapsize/uvweight/uvtaper/uvrange/uvzero).
        self._nx = self._ny = 256
        self._xinc = self._yinc = 1.0 * MAS
        self._binwid = 0.0  # natural weighting
        self._errpow = -1.0  # amplitude-error weighting (difmap default)
        self._dorad = False
        self._gauval = self._gaurad = 0.0
        self._uvmin = self._uvmax = 0.0
        self._uvzero = (0.0, 0.0)  # (flux, weight)
        self.windows: list[tuple[float, float, float, float]] = []  # mas
        self._invert_result = None
        self._restored = None
        self._restore_beam = None
        # Free-parameter bitmasks of the tentative model components,
        # used by modelfit (parallel to the tentative component list).
        self._freepars: list[int] = []

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_uvfits(cls, path, wtscale=1.0) -> "Observation":
        from difmapy.io.uvfits import load_uvfits

        return cls(load_uvfits(path, wtscale=wtscale))

    @classmethod
    def from_ms(cls, path, **kwargs) -> "Observation":
        from difmapy.io.ms import load_ms

        return cls(load_ms(path, **kwargs))

    # ------------------------------------------------------------------
    # header
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        return self._core.source_name

    @property
    def nif(self) -> int:
        return self._core.nif

    @property
    def nchan(self) -> list[int]:
        return [n for (_, _, n) in self._core.ifs]

    @property
    def npol(self) -> int:
        return self._core.npol

    @property
    def antennas(self) -> list[str]:
        return self._core.antenna_names

    def header(self) -> str:
        c = self._core
        lines = [
            f"Source: {c.source_name}   "
            f"RA={np.rad2deg(c.ra):.6f} deg  Dec={np.rad2deg(c.dec):.6f} deg",
            f"Rows: {c.nrow}  integrations: {c.ntimes}  subarrays: {c.nsub}",
            f"Antennas ({len(c.antenna_names)}): {', '.join(c.antenna_names)}",
            f"Polarizations: {c.pols}",
            f"IFs: {c.nif}",
        ]
        for i, (freq, df, nchan) in enumerate(c.ifs):
            lines.append(
                f"  IF {i + 1}: {freq / 1e9:.6f} GHz, {nchan} x {df / 1e6:.4f} MHz"
            )
        try:
            sel = c.selection()
            lines.append(f"Selection: {sel['stokes']}  channels {sel['chlist']}")
        except RuntimeError:
            lines.append("Selection: none (call select())")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<difmapy.Observation {self.source}: {self._core.nrow} rows, {self.nif} IFs>"

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def select(self, pol="I", channels=None):
        """Select polarization and optional channel ranges.

        `channels` is a list of inclusive (start, end) 0-based ranges
        over the global channel axis (all IFs concatenated), like
        difmap's `select`.
        """
        self._core.select(str(pol), [(int(a), int(b)) for a, b in (channels or [])])
        self._dirty()
        return self

    # ------------------------------------------------------------------
    # imaging setup
    # ------------------------------------------------------------------

    def mapsize(self, nx, cell, ny=None, ycell=None):
        """Set map dimensions (pixels, powers of 2) and cell size (mas)."""
        self._nx = int(nx)
        self._ny = int(ny) if ny else int(nx)
        self._xinc = float(cell) * MAS
        self._yinc = float(ycell) * MAS if ycell else self._xinc
        self._dirty()
        return self

    def uvweight(self, binwid=0.0, errpow=-1.0, radial=False):
        """Set gridding weights: uniform bin width (pixels; 0=natural),
        error power and radial weighting (difmap uvweight)."""
        self._binwid = float(binwid)
        self._errpow = float(errpow)
        self._dorad = bool(radial)
        self._dirty()
        return self

    def uvtaper(self, gauval=0.0, gaurad=0.0):
        """Gaussian taper: weight `gauval` at radius `gaurad` (wavelengths)."""
        self._gauval, self._gaurad = float(gauval), float(gaurad)
        self._dirty()
        return self

    def uvrange(self, uvmin=0.0, uvmax=0.0):
        """Restrict gridding to a UV radius range (wavelengths)."""
        self._uvmin, self._uvmax = float(uvmin), float(uvmax)
        self._dirty()
        return self

    def uvzero(self, flux=0.0, weight=0.0):
        """Set a zero-spacing flux estimate (difmap uvzero)."""
        self._uvzero = (float(flux), float(weight))
        self._dirty()
        return self

    def _dirty(self):
        self._invert_result = None
        self._restored = None

    # ------------------------------------------------------------------
    # invert / clean / restore
    # ------------------------------------------------------------------

    def invert(self):
        """Grid + FFT the current selection into dirty map and beam."""
        res = self._core.invert(
            self._nx,
            self._ny,
            self._xinc,
            self._yinc,
            uvmin=self._uvmin,
            uvmax=self._uvmax,
            gauval=self._gauval,
            gaurad=self._gaurad,
            dorad=self._dorad,
            errpow=self._errpow,
            binwid=self._binwid,
            uvzero_amp=self._uvzero[0],
            uvzero_wt=self._uvzero[1],
        )
        self._invert_result = dict(res)
        self._restored = None
        return self._invert_result

    def _ensure_map(self):
        if self._invert_result is None:
            self.invert()

    @property
    def dmap(self) -> np.ndarray:
        """Residual dirty map [ny, nx] (Jy/beam)."""
        self._ensure_map()
        return self._core.map()

    @property
    def dbeam(self) -> np.ndarray:
        """Dirty beam [ny, nx]."""
        self._ensure_map()
        return self._core.beam()

    @property
    def restored_map(self) -> np.ndarray:
        if self._restored is None:
            raise RuntimeError("no restored map; call restore()")
        return self._restored

    @property
    def extent(self):
        """(east_max, east_min, south, north) map extent in mas, for
        plotting with RA increasing leftward."""
        ex = self._nx / 2 * self._xinc / MAS
        ey = self._ny / 2 * self._yinc / MAS
        return (ex, -ex, -ey, ey)

    @property
    def estimated_beam(self):
        """(bmaj_mas, bmin_mas, bpa_deg) estimated from the UV coverage."""
        self._ensure_map()
        r = self._invert_result
        return (r["e_bmaj"] / MAS, r["e_bmin"] / MAS, r["e_bpa"] / DEG)

    def add_window(self, xmin, xmax, ymin, ymax):
        """Add a CLEAN window (mas, relative to phase center)."""
        self.windows.append((float(xmin), float(xmax), float(ymin), float(ymax)))
        return self

    def clear_windows(self):
        self.windows.clear()
        return self

    def clean(self, niter=100, gain=0.05, cutoff=0.0):
        """Högbom CLEAN within the current windows (difmap clean).
        Negative `niter` stops at the first negative component."""
        self._ensure_map()
        wins = [
            (x0 * MAS, x1 * MAS, y0 * MAS, y1 * MAS)
            for (x0, x1, y0, y1) in self.windows
        ]
        res = self._core.clean(int(niter), float(gain), float(cutoff), wins)
        self._restored = None
        return dict(res)

    def keep(self):
        """Establish the tentative model (difmap keep)."""
        self._core.keep()
        self._freepars.clear()
        return self

    def clrmod(self, old=True, new=True):
        """Discard models (difmap clrmod)."""
        self._core.clear_models(old, new)
        if new:
            self._freepars.clear()
        self._dirty()
        return self

    def restore(self, bmaj=None, bmin=None, bpa=None, no_residual=False, do_smooth=False):
        """Convolve the model with the restoring beam and add residuals.

        Beam axes in mas, position angle in degrees; defaults to the
        beam estimated by invert. Returns the restored map [ny, nx].
        """
        self._ensure_map()
        r = self._invert_result
        bmaj = r["e_bmaj"] if bmaj is None else bmaj * MAS
        bmin = r["e_bmin"] if bmin is None else bmin * MAS
        bpa = r["e_bpa"] if bpa is None else bpa * DEG
        sel = self._core.selection()
        freqs = [f for f, u in zip(sel["if_freq"], sel["if_used"]) if u]
        freq = float(np.mean(freqs)) if freqs else 0.0
        self._restored = self._core.restore(bmaj, bmin, bpa, no_residual, do_smooth, freq)
        self._restore_beam = (bmaj / MAS, bmin / MAS, bpa / DEG)
        return self._restored

    def imstat(self):
        """Statistics of the residual map inner quarter."""
        self._ensure_map()
        return dict(self._core.map_stats())

    # ------------------------------------------------------------------
    # model access
    # ------------------------------------------------------------------

    @property
    def model(self) -> list[dict]:
        """Established + tentative model components (mas/deg units)."""
        old, new = self._core.get_models()
        out = []
        for tentative, comps in ((False, old), (True, new)):
            for (t, flux, x, y, major, ratio, phi, freq0, spcind) in comps:
                out.append(
                    {
                        "type": CMP_TYPES.get(t, "?"),
                        "flux": flux,
                        "x": x / MAS,
                        "y": y / MAS,
                        "major": major / MAS,
                        "ratio": ratio,
                        "phi": phi / DEG,
                        "freq0": freq0,
                        "spcind": spcind,
                        "tentative": tentative,
                    }
                )
        return out

    @property
    def model_flux(self) -> float:
        return float(sum(c["flux"] for c in self.model))

    def addcmp(self, flux, x, y, type="delta", major=0.0, ratio=1.0, phi=0.0,
               freq0=0.0, spcind=0.0, free=None):
        """Add a model component by hand (difmap addcmp); positions and
        sizes in mas, angles in degrees.

        `free` marks parameters as variable for modelfit, e.g.
        ``free=["flux", "pos"]`` or ``free="shape"``.
        """
        self._core.add_component(
            CMP_CODES[type],
            float(flux),
            float(x) * MAS,
            float(y) * MAS,
            float(major) * MAS,
            float(ratio),
            float(phi) * DEG,
            float(freq0),
            float(spcind),
        )
        self._freepars.append(_free_mask(free))
        return self

    def modelfit(self, niter=10, free=None, uvmin=0.0, uvmax=0.0):
        """Fit the tentative model to the visibilities (difmap
        modelfit), by Levenberg-Marquardt least squares on the residual
        real/imaginary parts.

        Components come from the tentative model (see `addcmp`); which
        of their parameters vary is set per component by `addcmp(free=)`
        or overridden here by `free` (a single spec applied to all
        components, or a list, one per component).

        Returns a dict with rchisq/chisq/ndfree/nvis/nfree and a list of
        1-sigma `errors` per component, in mas/degrees.
        """
        _, tentative = self._core.get_models()
        ncmp = len(tentative)
        if ncmp == 0:
            raise RuntimeError("no tentative model to fit; use addcmp() first")
        if free is None:
            masks = list(self._freepars[:ncmp])
            if len(masks) < ncmp:
                masks += [0] * (ncmp - len(masks))
        elif isinstance(free, (list, tuple)) and free and isinstance(
            free[0], (list, tuple, int)
        ):
            if len(free) != ncmp:
                raise ValueError(f"free has {len(free)} entries for {ncmp} components")
            masks = [_free_mask(f) for f in free]
        else:
            masks = [_free_mask(free)] * ncmp
        if not any(masks):
            raise ValueError(
                "no free parameters; pass free=... to modelfit() or addcmp()"
            )
        res = dict(
            self._core.modelfit(
                niter=int(niter),
                freepars=[int(m) for m in masks],
                uvmin=float(uvmin),
                uvmax=float(uvmax),
            )
        )
        # Convert uncertainties to user units (mas, degrees).
        res["errors"] = [
            {
                "flux": e[0],
                "x": e[1] / MAS,
                "y": e[2] / MAS,
                "major": e[3] / MAS,
                "ratio": e[4],
                "phi": e[5] / DEG,
                "spcind": e[6],
            }
            for e in res["errors"]
        ]
        self._freepars = list(masks)
        self._dirty()
        return res

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def selfcal(self, amp=False, phase=True, float_scale=False, solint=0.0,
                gauval=0.0, gaurad=0.0, maxamp=0.0, maxphs=0.0,
                uvmin=0.0, uvmax=0.0, mintel=0, flag=False):
        """Self-calibrate against the current model (difmap selfcal).

        solint in minutes (0 = per integration); maxphs in degrees.
        """
        res = self._core.selfcal(
            doamp=bool(amp),
            dophs=bool(phase),
            dofloat=bool(float_scale),
            solint=float(solint),
            doone=False,
            gauval=float(gauval),
            gaurad=float(gaurad),
            maxamp=float(maxamp),
            maxphs=float(maxphs) * DEG,
            uvmin=float(uvmin),
            uvmax=float(uvmax),
            mintel=int(mintel),
            doflag=bool(flag),
        )
        self._dirty()
        return dict(res)

    def gscale(self, float_scale=False):
        """Overall telescope amplitude corrections (difmap gscale)."""
        res = self._core.selfcal(
            doamp=True, dophs=False, dofloat=bool(float_scale), doone=True, mintel=4
        )
        self._dirty()
        return dict(res)

    def uncalib(self, amp=True, phase=True, flags=False):
        """Undo selfcal corrections (difmap uncalib)."""
        self._core.uncalib(amp, phase, flags)
        self._dirty()
        return self

    def selfant(self, name, fix=False, weight=1.0):
        """Constrain an antenna in selfcal (difmap selfant)."""
        self._core.set_antenna_constraints(str(name), bool(fix), float(weight))
        return self

    def startmod(self, model=None, solint=0.0, flux=1.0):
        """Phase self-calibrate against a starting model, then discard
        it (difmap startmod).

        `model` is a .mod file name; with no model a point source of
        `flux` Jy at the phase center is used, as difmap does.
        """
        self.clrmod(old=True, new=True)
        if model is None:
            self.addcmp(flux, 0.0, 0.0)
        else:
            self.rmodel(model)
        self.keep()
        res = self.selfcal(phase=True, solint=solint)
        self.clrmod(old=True, new=True)
        return res

    def resoff(self, baseline=None):
        """Solve for per-baseline amplitude/phase offsets against the
        current model, to absorb non-closing errors (difmap resoff).
        Returns the number of (baseline, IF) corrections set."""
        bl = None
        if baseline is not None:
            bl = (self._ant_index(baseline[0]), self._ant_index(baseline[1]))
        n = self._core.resoff(baseline=bl)
        self._dirty()
        return n

    def clroff(self):
        """Undo all baseline corrections (difmap clroff)."""
        self._core.clroff()
        self._dirty()
        return self

    def baseline_corrections(self):
        """Baseline corrections as a list of dicts (amp, phase in deg)."""
        bls, amp, phs = self._core.baseline_corrections()
        names = self._core.antenna_names
        amp = np.asarray(amp)
        phs = np.asarray(phs)
        return [
            {
                "baseline": (names[a], names[b]),
                "amp": amp[i].tolist(),
                "phase": (phs[i] / DEG).tolist(),
            }
            for i, (a, b) in enumerate(bls)
        ]

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    def shift(self, east, north):
        """Shift the phase center by (east, north) in mas: the map
        contents move by the same amount and the model follows
        (difmap shift)."""
        self._core.shift(float(east) * MAS, float(north) * MAS)
        self._dirty()
        return self

    def unshift(self):
        """Undo all accumulated shifts (difmap unshift)."""
        self._core.unshift()
        self._dirty()
        return self

    @property
    def total_shift(self):
        """Accumulated (east, north) shift in mas."""
        e, n = self._core.shift_total
        return (e / MAS, n / MAS)

    def uvaver(self, aver_time, doscatter=False):
        """Return a new observation with the calibrated data averaged
        into `aver_time`-second integrations (difmap uvaver)."""
        from difmapy.average import uvaver

        new = Observation(uvaver(self._core, float(aver_time), bool(doscatter)))
        # Carry over the imaging setup and selection.
        new._nx, new._ny = self._nx, self._ny
        new._xinc, new._yinc = self._xinc, self._yinc
        new._binwid, new._errpow, new._dorad = self._binwid, self._errpow, self._dorad
        new._gauval, new._gaurad = self._gauval, self._gaurad
        new._uvmin, new._uvmax = self._uvmin, self._uvmax
        new._uvzero = self._uvzero
        new.windows = list(self.windows)
        try:
            sel = self._core.selection()
            new.select(sel["stokes"], channels=[tuple(r) for r in sel["chlist"]])
        except RuntimeError:
            pass
        return new

    # ------------------------------------------------------------------
    # editing / flagging
    # ------------------------------------------------------------------

    def _ant_index(self, name) -> int:
        names = [n.upper() for n in self._core.antenna_names]
        key = str(name).upper()
        if key not in names:
            raise ValueError(f"unknown antenna {name!r}; have {self._core.antenna_names}")
        return names.index(key)

    def _edit(self, flag, baseline=None, station=None, tmin=None, tmax=None,
              subarray=None, if_index=None, selected_channels_only=False):
        bl = None
        if baseline is not None:
            a, b = baseline
            bl = (self._ant_index(a), self._ant_index(b))
        st = self._ant_index(station) if station is not None else None
        return self._core.edit(
            bool(flag),
            tmin=None if tmin is None else float(tmin),
            tmax=None if tmax is None else float(tmax),
            baseline=bl,
            station=st,
            subarray=subarray,
            if_index=if_index,
            sel_chan=bool(selected_channels_only),
        )

    def flag(self, baseline=None, station=None, tmin=None, tmax=None,
             subarray=None, if_index=None, selected_channels_only=False):
        """Flag visibilities (difmap flag). Restrict by baseline
        (name pair), station name, time range (seconds since the
        reference day), subarray or IF. Returns #rows affected."""
        n = self._edit(True, baseline, station, tmin, tmax, subarray,
                       if_index, selected_channels_only)
        self._dirty()
        return n

    def unflag(self, baseline=None, station=None, tmin=None, tmax=None,
               subarray=None, if_index=None, selected_channels_only=False):
        """Unflag visibilities (difmap unflag)."""
        n = self._edit(False, baseline, station, tmin, tmax, subarray,
                       if_index, selected_channels_only)
        self._dirty()
        return n

    @property
    def flags(self) -> np.ndarray:
        """The FLAG column, [nrow, nchan_total, npol] (True = flagged)."""
        return self._core.flags()

    @flags.setter
    def flags(self, value):
        self._core.set_flags(np.ascontiguousarray(value, dtype=bool))
        self._dirty()

    @property
    def flagged_fraction(self) -> float:
        c = self._core
        return c.nflagged / (c.nrow * c.nctotal * c.npol)

    def save_flags(self, path=None, flag_row=True):
        """Write the FLAG column back to the source Measurement Set.

        This is the standard MS way of persisting flags: only FLAG (and
        FLAG_ROW) are modified, leaving data and weights untouched.
        Requires that the observation was loaded from an MS.
        """
        from difmapy.io.ms import save_flags

        return save_flags(self._core, path=path, flag_row=flag_row)

    # ------------------------------------------------------------------
    # interactive plots (pyqtgraph; require the [plot] extra)
    # ------------------------------------------------------------------

    def radplot(self, quantity="amp", block=None):
        """Amplitude/phase vs UV radius with interactive flagging."""
        from difmapy.plots import radplot

        return radplot(self, quantity=quantity, block=block)

    def projplot(self, angle=0.0, quantity="amp", block=None):
        """Amp/phase vs projected UV distance (difmap projplot)."""
        from difmapy.plots import projplot

        return projplot(self, angle_deg=angle, quantity=quantity, block=block)

    def uvplot(self, block=None):
        """UV coverage with interactive flagging."""
        from difmapy.plots import uvplot

        return uvplot(self, block=block)

    def vplot(self, reftel=None, quantity="amp", block=None):
        """Visibilities vs time for one telescope's baselines."""
        from difmapy.plots import vplot

        return vplot(self, reftel=reftel, quantity=quantity, block=block)

    def mapplot(self, what="map", block=None, **clean_args):
        """Interactive map/beam display with CLEAN window editing."""
        from difmapy.plots import mapplot

        return mapplot(self, what=what, block=block, **clean_args)

    def cpplot(self, triangles=None, if_index=None, nplot=4, block=None):
        """Closure phases vs time, with the model overplotted
        (difmap cpplot). `triangles` is a list of antenna-name triples;
        all closed triangles are shown by default."""
        from difmapy.plots import cpplot

        return cpplot(self, triangles=triangles, if_index=if_index,
                      nplot=nplot, block=block)

    def tplot(self, block=None):
        """Per-antenna time sampling of unflagged data (difmap tplot)."""
        from difmapy.plots import tplot

        return tplot(self, block=block)

    def corplot(self, quantity="phase", nplot=4, block=None):
        """Self-cal gain corrections vs time (difmap corplot)."""
        from difmapy.plots import corplot

        return corplot(self, quantity=quantity, nplot=nplot, block=block)

    def specplot(self, baseline=None, tmin=None, tmax=None, xaxis="freq", block=None):
        """Time-averaged spectrum of the selected polarization
        (difmap specplot)."""
        from difmapy.plots import specplot

        return specplot(self, baseline=baseline, tmin=tmin, tmax=tmax,
                        xaxis=xaxis, block=block)

    # ------------------------------------------------------------------
    # closure / spectral data (without plotting)
    # ------------------------------------------------------------------

    def closure_phases(self, triangle=None, if_index=None):
        """Closure phase time series (radians). Returns a list of dicts
        with triangle/if_index/time/phase/model/error."""
        idx = None
        if triangle is not None:
            idx = tuple(sorted(self._ant_index(t) for t in triangle))
        return self._core.closure_phases(triangle=idx, if_index=if_index)

    def spectrum(self, baseline=None, tmin=None, tmax=None):
        """Time-averaged spectrum of the current polarization selection."""
        bl = None
        if baseline is not None:
            bl = (self._ant_index(baseline[0]), self._ant_index(baseline[1]))
        return dict(self._core.spectrum(baseline=bl, tmin=tmin, tmax=tmax))

    # ------------------------------------------------------------------
    # file output (difmap wmap/wbeam/wmodel/wwins ...)
    # ------------------------------------------------------------------

    def _write_fits_image(self, path, data, bunit, beam=None, overwrite=True):
        from astropy.io import fits

        c = self._core
        nx, ny = self._nx, self._ny
        # Flip x so that RA increases with decreasing pixel index.
        img = np.ascontiguousarray(data[:, ::-1], dtype=np.float32)
        hdr = fits.Header()
        hdr["OBJECT"] = c.source_name
        hdr["BUNIT"] = bunit
        hdr["EQUINOX"] = 2000.0
        hdr["CTYPE1"] = "RA---SIN"
        hdr["CRVAL1"] = np.rad2deg(c.ra)
        hdr["CDELT1"] = -self._xinc / DEG
        hdr["CRPIX1"] = nx / 2.0  # 1-based: pixel nx/2 after flip
        hdr["CTYPE2"] = "DEC--SIN"
        hdr["CRVAL2"] = np.rad2deg(c.dec)
        hdr["CDELT2"] = self._yinc / DEG
        hdr["CRPIX2"] = ny / 2.0 + 1.0
        sel = c.selection()
        freqs = [f for f, u in zip(sel["if_freq"], sel["if_used"]) if u]
        if freqs:
            hdr["CTYPE3"] = "FREQ"
            hdr["CRVAL3"] = float(np.mean(freqs))
            hdr["CRPIX3"] = 1.0
        if beam is not None:
            bmaj, bmin, bpa = beam
            hdr["BMAJ"] = bmaj * MAS / DEG
            hdr["BMIN"] = bmin * MAS / DEG
            hdr["BPA"] = bpa
        hdr["ORIGIN"] = "difmapy"
        fits.PrimaryHDU(data=img[np.newaxis] if freqs else img, header=hdr).writeto(
            path, overwrite=overwrite
        )

    def wmap(self, path, overwrite=True):
        """Write the restored map (restoring first if needed)."""
        if self._restored is None:
            self.restore()
        self._write_fits_image(path, self._restored, "JY/BEAM", beam=self._restore_beam,
                               overwrite=overwrite)

    def wdmap(self, path, overwrite=True):
        """Write the residual dirty map."""
        self._ensure_map()
        self._write_fits_image(path, self._core.map(), "JY/BEAM", overwrite=overwrite)

    def wbeam(self, path, overwrite=True):
        """Write the dirty beam."""
        self._ensure_map()
        self._write_fits_image(path, self._core.beam(), "", overwrite=overwrite)

    def wmodel(self, path):
        """Write the model in difmap/Caltech .mod format."""
        comps = self.model
        with open(path, "w") as f:
            f.write("! Flux (Jy) Radius (mas)  Theta (deg)  "
                    "Major (mas)  Axial ratio   Phi (deg) T  Freq (Hz)  SpecIndex\n")
            for m in comps:
                r = float(np.hypot(m["x"], m["y"]))
                theta = float(np.rad2deg(np.arctan2(m["x"] * MAS, m["y"] * MAS)))
                t = CMP_CODES[m["type"]]
                if t == 0 and m["freq0"] == 0.0:
                    f.write(f"{m['flux']:11.6g}v {r:11.6g}v {theta:11.6g}v\n")
                else:
                    f.write(
                        f"{m['flux']:11.6g}v {r:11.6g}v {theta:11.6g}v "
                        f"{m['major']:11.6g}v {m['ratio']:9.6g}v {m['phi']:9.6g}v "
                        f"{t} {m['freq0']:.5g} {m['spcind']:9.6g}\n"
                    )

    def rmodel(self, path):
        """Read a difmap/Caltech .mod model file as the tentative model."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("!"):
                    continue
                vals = [v.rstrip("v") for v in line.split()]
                flux, r, theta = (float(vals[0]), float(vals[1]), float(vals[2]))
                x = r * np.sin(np.deg2rad(theta))
                y = r * np.cos(np.deg2rad(theta))
                major = float(vals[3]) if len(vals) > 3 else 0.0
                ratio = float(vals[4]) if len(vals) > 4 else 1.0
                phi = float(vals[5]) if len(vals) > 5 else 0.0
                t = int(vals[6]) if len(vals) > 6 else 0
                freq0 = float(vals[7]) if len(vals) > 7 else 0.0
                spcind = float(vals[8]) if len(vals) > 8 else 0.0
                self.addcmp(flux, x, y, type=CMP_TYPES[t], major=major, ratio=ratio,
                            phi=phi, freq0=freq0, spcind=spcind)
        return self

    def wobs(self, path, overwrite=True):
        """Write the (calibrated, edited) UV data to a random-groups
        UVFITS file (difmap wobs)."""
        from difmapy.io.uvfits import save_uvfits

        save_uvfits(self._core, path, overwrite=overwrite)

    def save(self, prefix):
        """Save UV data, model, windows and imaging parameters with a
        common prefix (difmap save)."""
        import json

        self.wobs(f"{prefix}.uvf")
        self.wmodel(f"{prefix}.mod")
        self.wwins(f"{prefix}.win")
        sel = None
        try:
            s = self._core.selection()
            sel = {"stokes": s["stokes"], "chlist": [list(r) for r in s["chlist"]]}
        except RuntimeError:
            pass
        pars = {
            "select": sel,
            "mapsize": [self._nx, self._xinc / MAS, self._ny, self._yinc / MAS],
            "uvweight": [self._binwid, self._errpow, self._dorad],
            "uvtaper": [self._gauval, self._gaurad],
            "uvrange": [self._uvmin, self._uvmax],
            "uvzero": list(self._uvzero),
        }
        with open(f"{prefix}.par.json", "w") as f:
            json.dump(pars, f, indent=1)

    @classmethod
    def get(cls, prefix) -> "Observation":
        """Restore a session saved with save() (difmap get)."""
        import json
        import os

        obs = cls.from_uvfits(f"{prefix}.uvf")
        if os.path.exists(f"{prefix}.par.json"):
            with open(f"{prefix}.par.json") as f:
                pars = json.load(f)
            if pars.get("select"):
                obs.select(pars["select"]["stokes"],
                           channels=[tuple(r) for r in pars["select"]["chlist"]])
            nx, cell, ny, ycell = pars["mapsize"]
            obs.mapsize(nx, cell, ny, ycell)
            obs.uvweight(*pars["uvweight"])
            obs.uvtaper(*pars["uvtaper"])
            obs.uvrange(*pars["uvrange"])
            obs.uvzero(*pars["uvzero"])
        if os.path.exists(f"{prefix}.mod"):
            obs.rmodel(f"{prefix}.mod")
            obs.keep()
        if os.path.exists(f"{prefix}.win"):
            obs.rwins(f"{prefix}.win")
        return obs

    def wwins(self, path):
        """Write CLEAN windows (mas) to a difmap .win file."""
        with open(path, "w") as f:
            for (x0, x1, y0, y1) in self.windows:
                f.write(f"{x0:g} {x1:g} {y0:g} {y1:g}\n")

    def rwins(self, path):
        """Read CLEAN windows from a difmap .win file."""
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("!"):
                    continue
                x0, x1, y0, y1 = (float(v) for v in line.split()[:4])
                self.windows.append((x0, x1, y0, y1))
        return self


def load(path, **kwargs) -> Observation:
    """Load a UVFITS file or Measurement Set (difmap observe)."""
    import os

    if os.path.isdir(path):
        return Observation.from_ms(path, **kwargs)
    return Observation.from_uvfits(path, **kwargs)
