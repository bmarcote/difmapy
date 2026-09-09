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

# Defaults used by `auto_mapsize()` (and hence by `mapplot()` when the
# map size has not been set by hand): a large field sampled with ten
# pixels across the diffraction limit lambda/B_max. The number of pixels
# no longer has to be a power of two - any multiple of four works - but
# powers of two are still the fastest to transform.
DEFAULT_NPIX = 4096
DEFAULT_OVERSAMPLE = 10

# AIPS/FITS polarization codes, for data whose loader predates
# CoreObservation.pol_names.
POL_NAMES = {
    1: "I", 2: "Q", 3: "U", 4: "V",
    -1: "RR", -2: "LL", -3: "RL", -4: "LR",
    -5: "XX", -6: "YY", -7: "XY", -8: "YX", -9: "PI",
}

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


__all__ = ["Observation", "load", "observe"]


class Observation:
    """An in-memory interferometric observation (difmap-style)."""

    def __init__(self, core: CoreObservation):
        self._core = core
        # Imaging state (difmap mapsize/uvweight/uvtaper/uvrange/uvzero).
        self._nx = self._ny = 256
        self._xinc = self._yinc = 1.0 * MAS
        # difmap's own defaults (invdef in difmap.c): uniform weighting
        # with a 2-pixel bin and no amplitude-error weighting, so that
        # a first image matches what difmap would produce.
        self._binwid = 2.0
        self._errpow = 0.0
        self._dorad = False
        self._robust = None      # Briggs robustness; overrides binwid/errpow
        self._mapsize_set = False  # False until mapsize()/auto_mapsize()
        self._gauval = self._gaurad = 0.0
        self._uvmin = self._uvmax = 0.0
        self._uvzero = (0.0, 0.0)  # (flux, weight)
        self.windows: list[tuple[float, float, float, float]] = []  # mas
        self._invert_result = None
        self._restored = None
        self._restore_beam = None

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_uvfits(cls, path, wtscale=1.0, stokes="I", channels=None) -> "Observation":
        from difmapy.io.uvfits import load_uvfits

        obs = cls(load_uvfits(path, wtscale=wtscale))
        obs._initial_select(stokes, channels)
        return obs

    @classmethod
    def from_ms(cls, path, stokes="I", channels=None, **kwargs) -> "Observation":
        from difmapy.io.ms import load_ms

        obs = cls(load_ms(path, **kwargs))
        obs._initial_select(stokes, channels)
        return obs

    def _initial_select(self, stokes, channels=None):
        """Apply the selection asked for at load time (difmap `observe`
        selects nothing; difmapy defaults to total intensity, which is
        what almost every session starts with). `stokes=None` skips it,
        and `select()` can change it at any time."""
        if stokes is None or str(stokes).lower() == "none":
            return
        try:
            self.select(stokes, channels=channels)
        except (ValueError, RuntimeError) as exc:
            print(f"warning: could not select {stokes}: {exc}")

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

    @property
    def pols(self) -> list[str]:
        """The recorded polarizations as names ("RR", "LL", ...), in the
        order they appear on the data's polarization axis."""
        try:
            return list(self._core.pol_names)
        except AttributeError:  # extension older than pol_names
            return [POL_NAMES.get(int(c), f"?({int(c)})") for c in self._core.pols]

    @property
    def pol_codes(self) -> list[int]:
        """The AIPS/FITS codes behind `pols` (-1 = RR, -2 = LL, ...)."""
        return [int(c) for c in self._core.pols]

    def header(self) -> str:
        c = self._core
        lines = [
            f"Source: {c.source_name}   "
            f"RA={np.rad2deg(c.ra):.6f} deg  Dec={np.rad2deg(c.dec):.6f} deg",
            f"Rows: {c.nrow}  integrations: {c.ntimes}  subarrays: {c.nsub}",
            f"Antennas ({len(c.antenna_names)}): {', '.join(c.antenna_names)}",
            f"Polarizations: {', '.join(self.pols)}",
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
        """Set map dimensions (pixels) and cell size (mas).

        The dimensions must be multiples of four; powers of two are no
        longer required, but are still the fastest to transform.
        """
        self._nx = int(nx)
        self._ny = int(ny) if ny else int(nx)
        self._xinc = float(cell) * MAS
        self._yinc = float(ycell) * MAS if ycell else self._xinc
        self._mapsize_set = True
        self._dirty()
        return self

    def estimated_resolution(self) -> float:
        """Diffraction-limited resolution in mas: lambda / B_max over the
        longest unflagged baseline of the current selection."""
        core = self._core
        _, _, _, us, vs, _ = core.rows()
        r_sec = np.hypot(np.asarray(us), np.asarray(vs))  # light-seconds
        bmax = 0.0
        try:
            sel = core.selection()
        except RuntimeError:  # nothing selected yet: use the raw header
            sel = None
        if sel is None:
            freqs = [f for (f, _, _) in core.ifs]
            if freqs and r_sec.size:
                bmax = float(r_sec.max()) * max(freqs)
        else:
            wt = np.asarray(core.stream_vis()[1])
            for cif, (freq, used) in enumerate(zip(sel["if_freq"], sel["if_used"])):
                if not used:
                    continue
                good = wt[:, cif] > 0
                if good.any():
                    bmax = max(bmax, float(r_sec[good].max()) * float(freq))
        if bmax <= 0.0:
            raise RuntimeError("no unflagged data to estimate a resolution from")
        return (1.0 / bmax) / MAS

    def auto_mapsize(self, npix=DEFAULT_NPIX, oversample=DEFAULT_OVERSAMPLE):
        """Set a map size derived from the data: `npix` pixels of
        `estimated_resolution() / oversample` each.

        This is what `mapplot()` uses when `mapsize()` has not been
        called, so a first image needs no guessing.
        """
        return self.mapsize(int(npix), self.estimated_resolution() / float(oversample))

    def _ensure_mapsize(self):
        """Pick a map size from the data if none was set explicitly."""
        if not self._mapsize_set:
            self.auto_mapsize()

    def uvweight(self, binwid=2.0, errpow=0.0, radial=False, robust=None):
        """Set gridding weights (difmap uvweight).

        binwid : uniform-weighting bin width in UV pixels; 0 selects
            natural weighting (i.e. no density correction).
        errpow : if negative, scale weights by wt**(-errpow/2), i.e.
            weight down noisy visibilities; 0 ignores the data weights.
        radial : multiply weights by the UV radius.
        robust : Briggs robustness. A single number from -2 (uniform,
            sharpest beam) to +2 (natural, lowest noise), which
            supersedes `binwid`/`errpow` when given; pass ``None`` to go
            back to difmap's own scheme.

        The defaults are difmap's (uniform, binwid=2, errpow=0). For
        natural weighting using the data weights, use ``uvweight(0, -1)``
        or ``uvweight(robust=2)``.
        """
        self._binwid = float(binwid)
        self._errpow = float(errpow)
        self._dorad = bool(radial)
        self._robust = None if robust is None else float(np.clip(robust, -2.0, 2.0))
        self._dirty()
        return self

    @property
    def robust(self):
        """The Briggs robustness in use, or None for difmap weighting."""
        return self._robust

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
            robust=self._robust,
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
        """Residual dirty map [ny, nx] (Jy/beam).

        Only the inner quarter (see `valid_slice`) is scientifically
        usable: the outer margin is amplified by the gridding
        correction and can even exceed the true peak. Use `imstat()` or
        `valid(...)` rather than scanning the whole array.
        """
        self._ensure_map()
        return self._core.map()

    @property
    def dbeam(self) -> np.ndarray:
        """Dirty beam [ny, nx]; see `dmap` about the outer margin."""
        self._ensure_map()
        return self._core.beam()

    @property
    def restored_map(self) -> np.ndarray:
        if self._restored is None:
            raise RuntimeError("no restored map; call restore()")
        return self._restored

    @property
    def valid_slice(self):
        """Numpy slice of the scientifically valid map area: the inner
        quarter, which is what CLEAN searches and `imstat` measures.

        Outside it the gridding-correction factor grows without bound,
        so pixel values there are meaningless (difmap restricts its
        map area the same way).
        """
        return (
            slice(self._ny // 4, self._ny - self._ny // 4),
            slice(self._nx // 4, self._nx - self._nx // 4),
        )

    def valid(self, image=None) -> np.ndarray:
        """The valid inner quarter of `image` (default: the dirty map)."""
        if image is None:
            image = self.dmap
        return image[self.valid_slice]

    def peak_offset(self, image=None):
        """(east, north) offset in mas of the brightest valid pixel, and
        its value: ``((x, y), value)``."""
        img = self.dmap if image is None else image
        sy, sx = self.valid_slice
        sub = img[sy, sx]
        iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
        ix += sx.start
        iy += sy.start
        return (
            ((ix - self._nx / 2) * self._xinc / MAS,
             (iy - self._ny / 2) * self._yinc / MAS),
            float(img[iy, ix]),
        )

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

    def clean(self, niter=100, gain=0.05, cutoff=0.0, quiet=False):
        """Högbom CLEAN within the current windows (difmap clean).
        Negative `niter` stops at the first negative component.

        Returns (and, unless `quiet`, prints) a dict summarising the
        run: iterations done, components added, flux cleaned in this
        call and in total, and the residual statistics afterwards.
        """
        self._ensure_map()
        wins = [
            (x0 * MAS, x1 * MAS, y0 * MAS, y1 * MAS)
            for (x0, x1, y0, y1) in self.windows
        ]
        res = dict(self._core.clean(int(niter), float(gain), float(cutoff), wins))
        self._restored = None
        stats = self.imstat()
        res.update(
            gain=float(gain),
            cutoff=float(cutoff),
            total_ncomp=len(self.model),
            total_flux=self.model_flux,
            residual_rms=stats["rms"],
            residual_peak=stats["max"],
        )
        if not quiet:
            print(
                f"clean: {res['niter']} iterations, {res['ncomp']} components, "
                f"{res['cleaned_flux']:.5g} Jy cleaned "
                f"(model: {res['total_ncomp']} components, "
                f"{res['total_flux']:.5g} Jy); "
                f"residual rms {res['residual_rms']:.5g}, "
                f"peak {res['residual_peak']:.5g} Jy/beam"
            )
            if res["hit_cutoff"]:
                print("  stopped: reached the cutoff flux")
            if res["hit_negative"]:
                print("  stopped: first negative component")
        return res

    def keep(self):
        """Establish the tentative model (difmap keep)."""
        self._core.keep()
        return self

    def clrmod(self, old=True, new=True):
        """Discard models (difmap clrmod)."""
        self._core.clear_models(old, new)
        self._dirty()
        return self

    def clearmodel(self, windows=False):
        """Remove every model component - CLEAN components as well as
        hand-placed or fitted Gaussians - so that the next `invert()`
        gives the dirty map again and imaging can start from scratch.

        CLEAN windows are kept unless `windows=True`.
        """
        self._core.clear_models(True, True)
        if windows:
            self.windows.clear()
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

    def noise_stats(self, image=None, nsigma=3.0, niter=5):
        """Robust noise estimate of the residual map: the mean and
        standard deviation of a Gaussian fitted to the pixel histogram
        after iterative sigma clipping, which removes real emission.

        Returns a dict with `mean`, `rms` (the clipped, i.e. noise-only,
        standard deviation), `raw_rms` over all valid pixels, the pixel
        count `npix` and how many of them the clipping rejected
        (`nclipped`).
        """
        img = self.valid() if image is None else np.asarray(image)
        flat = np.asarray(img, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            nan = float("nan")
            return {"mean": nan, "rms": nan, "raw_rms": nan,
                    "npix": 0, "nclipped": 0}
        keep = np.ones(flat.shape, dtype=bool)
        mean = float(flat.mean())
        rms = float(flat.std())
        for _ in range(int(niter)):
            if rms <= 0.0:
                break
            new = np.abs(flat - mean) < nsigma * rms
            if new.sum() < 16 or np.array_equal(new, keep):
                keep = new
                break
            keep = new
            mean = float(flat[keep].mean())
            rms = float(flat[keep].std())
        return {
            "mean": mean,
            "rms": rms,
            "raw_rms": float(flat.std()),
            "npix": int(flat.size),
            "nclipped": int((~keep).sum()),
        }

    def mapinfo(self):
        """Everything worth knowing about the current image, as a dict:
        the beam, the peak, the model, and the residual noise.

        This is what `mapplot`'s "x" key prints, and it is a plain dict
        so it can be stored, compared or written out.
        """
        self._ensure_map()
        stats = self.imstat()
        noise = self.noise_stats()
        (px, py), peak = self.peak_offset()
        model = self.model
        beam = self._restore_beam or self.estimated_beam
        types = {}
        for c in model:
            types[c["type"]] = types.get(c["type"], 0) + 1
        return {
            "source": self.source,
            "mapsize": (self._nx, self._ny),
            "cellsize": (self._xinc / MAS, self._yinc / MAS),
            "uvweight": {
                "robust": self._robust,
                "binwid": self._binwid,
                "errpow": self._errpow,
                "radial": self._dorad,
            },
            "beam": {"bmaj": beam[0], "bmin": beam[1], "bpa": beam[2],
                     "restored": self._restore_beam is not None},
            "estimated_beam": self.estimated_beam,
            "peak": {"flux": peak, "x": px, "y": py},
            "flux": self.model_flux,
            "model": {"ncomp": len(model), "types": types, "components": model},
            "residual": {
                "rms": noise["rms"],
                "raw_rms": stats["rms"],
                "mean": noise["mean"],
                "min": stats["min"],
                "max": stats["max"],
            },
            "shift": self.total_shift,
        }

    def print_mapinfo(self, info=None):
        """Print `mapinfo()` as a short human-readable report."""
        d = self.mapinfo() if info is None else info
        b = d["beam"]
        print(f"--- {d['source']}: {d['mapsize'][0]}x{d['mapsize'][1]} pixels of "
              f"{d['cellsize'][0]:.4g} x {d['cellsize'][1]:.4g} mas")
        w = d["uvweight"]
        weighting = (f"robust {w['robust']:+g}" if w["robust"] is not None
                     else f"binwid {w['binwid']:g}, errpow {w['errpow']:g}")
        print(f"    weighting: {weighting}")
        print(f"    {'restored' if b['restored'] else 'estimated'} beam: "
              f"{b['bmaj']:.4g} x {b['bmin']:.4g} mas at {b['bpa']:.4g} deg")
        pk = d["peak"]
        print(f"    peak: {pk['flux']:.5g} Jy/beam at "
              f"({pk['x']:.4g}, {pk['y']:.4g}) mas")
        m = d["model"]
        kinds = ", ".join(f"{n} {t}" for t, n in sorted(m["types"].items()))
        print(f"    model: {m['ncomp']} components ({kinds or 'none'}), "
              f"{d['flux']:.5g} Jy")
        r = d["residual"]
        print(f"    residual: rms {r['rms']:.5g} Jy/beam (noise-only), "
              f"min {r['min']:.5g}, max {r['max']:.5g}")
        return d

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
            freepar=_free_mask(free),
        )
        return self

    def seed_model(self, type="gauss", free=("flux", "pos", "major")):
        """Add one component at the brightest point of the residual map,
        as a starting guess for `modelfit`.

        The default is a circular Gaussian of zero width (i.e. a point
        source that the fit is free to resolve) carrying the peak flux.
        Returns the seeded component as a dict.
        """
        self._ensure_map()
        (x, y), peak = self.peak_offset()
        self.addcmp(float(peak), x, y, type=type, major=0.0, ratio=1.0,
                    free=list(free))
        return self.model[-1]

    def modelfit(self, niter=-1, free=None, uvmin=0.0, uvmax=0.0, quiet=False):
        """Fit the tentative model to the visibilities (difmap
        modelfit), by Levenberg-Marquardt least squares on the residual
        real/imaginary parts.

        Components come from the tentative model (see `addcmp`); which
        of their parameters vary is set per component by `addcmp(free=)`
        or overridden here by `free` (a single spec applied to all
        components, or a list, one per component). With no tentative
        model at all, one is seeded by `seed_model()`: a circular
        Gaussian of zero width at the peak of the residual map.

        `niter` is the number of Levenberg-Marquardt iterations; the
        default of -1 iterates until the fit converges.

        Returns (and, unless `quiet`, prints) a dict with
        rchisq/chisq/ndfree/nvis/nfree, whether it `converged`, the
        fitted `components`, and a list of 1-sigma `errors` per
        component, in mas/degrees.
        """
        _, tentative = self._core.get_models()
        ncmp = len(tentative)
        if ncmp == 0:
            seeded = self.seed_model()
            if not quiet:
                print(
                    f"modelfit: no model given; starting from a circular "
                    f"Gaussian of {seeded['flux']:.5g} Jy at "
                    f"({seeded['x']:.4g}, {seeded['y']:.4g}) mas"
                )
            _, tentative = self._core.get_models()
            ncmp = len(tentative)
        if free is None:
            # Use the per-component masks recorded by addcmp().
            masks = list(self._core.tentative_freepars)
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
                f"none of the {ncmp} tentative components has a free "
                "parameter; pass free=... to modelfit() or addcmp(), or "
                "clearmodel() first to fit from scratch"
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
        self._dirty()
        res["components"] = [c for c in self.model if c["tentative"]]
        res["ncomp"] = len(res["components"])
        res["total_flux"] = self.model_flux
        if not quiet:
            self._print_modelfit(res)
        return res

    @staticmethod
    def _print_modelfit(res):
        end = "converged" if res.get("converged") else "stopped"
        print(
            f"modelfit: {end} after {res.get('niter', '?')} iterations; "
            f"reduced chi-squared {res['rchisq']:.5g} "
            f"({res['nvis']} visibilities, {res['nfree']} free parameters)"
        )
        print(f"  {res['ncomp']} components, {res['total_flux']:.5g} Jy total")
        for i, (c, e) in enumerate(zip(res["components"], res["errors"])):
            size = "" if c["major"] <= 0 else (
                f"  {c['major']:.4g} mas"
                + ("" if c["ratio"] >= 1.0 else
                   f" x {c['major'] * c['ratio']:.4g} mas @ {c['phi']:.4g} deg")
            )
            print(
                f"  [{i}] {c['type']:<6} {c['flux']:9.5g} +- {e['flux']:.2g} Jy  "
                f"at ({c['x']:.4g} +- {e['x']:.2g}, "
                f"{c['y']:.4g} +- {e['y']:.2g}) mas{size}"
            )

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def selfcal(self, amp=False, phase=True, float_scale=False, solint=0.0,
                gauval=0.0, gaurad=0.0, maxamp=0.0, maxphs=0.0,
                uvmin=0.0, uvmax=0.0, mintel=0, flag=False):
        """Self-calibrate against the current model (difmap selfcal).

        solint in minutes (0 = per integration); maxphs in degrees.
        Each IF of each subarray is solved independently, so the
        returned `nbins` counts solution intervals over all of them.
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

    def gscale(self, float_scale=False, quiet=False):
        """Overall telescope amplitude corrections (difmap gscale).

        One amplitude correction per telescope and IF is solved for the
        whole observation. Returns (and, unless `quiet`, prints) a dict
        whose `gains` maps each station name to the correction this call
        applied, with the per-IF values in `gains_per_if`; `norms` holds
        the per-(subarray, IF) renormalisation factors, and is empty
        when the overall scale is left floating.
        """
        # An antenna with no prior solution starts from unity, so that
        # the first gscale reports its own corrections rather than NaN.
        before, _ = self._gain_amps(fill=1.0)
        res = dict(
            self._core.selfcal(
                doamp=True, dophs=False, dofloat=bool(float_scale),
                doone=True, mintel=4,
            )
        )
        self._dirty()
        after, solved = self._gain_amps(fill=np.nan)
        names = self.antennas
        # Report what *this* call did, so that a gscale after a selfcal
        # is not confused by the calibration already in the gain table.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(solved & (before > 0), after / before, np.nan)
        res["gains_per_if"] = {}
        res["gains"] = {}
        for ia, name in enumerate(names):
            per_if = []
            for cif in range(self.nif):
                col = ratio[:, cif, ia]
                col = col[np.isfinite(col)]
                per_if.append(float(np.median(col)) if col.size else float("nan"))
            res["gains_per_if"][name] = per_if
            good = [v for v in per_if if np.isfinite(v)]
            res["gains"][name] = float(np.median(good)) if good else float("nan")
        if not quiet:
            print(f"gscale: amplitude corrections ({res['nbins']} solution bins, "
                  f"{res['nbadtel']} bad telescope solutions)")
            for name in names:
                g = res["gains"][name]
                if np.isfinite(g):
                    per = " ".join(
                        f"{v:.3f}" if np.isfinite(v) else "  -  "
                        for v in res["gains_per_if"][name]
                    )
                    print(f"  {name:<8} {g:7.4f}   per IF: {per}")
                else:
                    print(f"  {name:<8}    -      (no solution)")
        return res

    def _gain_amps(self, fill=np.nan):
        """The gain-table amplitudes and their "solved" mask, both
        [ntimes, nif, nant]; `fill` replaces unsolved entries."""
        amp, _, _ = self._core.gains()
        nt, nif, nant = self._core.ntimes, self.nif, len(self.antennas)
        used = np.asarray(self._core.gains_used()).reshape(nt, nif, nant)
        amp = np.asarray(amp, dtype=np.float64).reshape(nt, nif, nant)
        return np.where(used, amp, fill), used

    def station_gains(self, per_if=False):
        """The accumulated amplitude gain corrections per station.

        With `per_if`, each station maps to a list of per-IF values;
        otherwise to their median. Values are the corrections difmapy
        multiplies the data by (`savecaltable` writes their reciprocal,
        which is CASA's convention), so a station with no solution is
        reported as 1.0 - the data is passed through unchanged.
        """
        amps, _ = self._gain_amps(fill=np.nan)
        out = {}
        for ia, name in enumerate(self.antennas):
            cols = []
            for cif in range(self.nif):
                v = amps[:, cif, ia]
                v = v[np.isfinite(v) & (v > 0)]
                cols.append(float(np.median(v)) if v.size else 1.0)
            out[name] = cols if per_if else float(np.median(cols))
        return out

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
        contents move by the same amount, and the model components and
        CLEAN windows follow (difmap shift)."""
        east, north = float(east), float(north)
        self._core.shift(east * MAS, north * MAS)
        self.windows[:] = [
            (x0 + east, x1 + east, y0 + north, y1 + north)
            for (x0, x1, y0, y1) in self.windows
        ]
        self._dirty()
        return self

    def unshift(self):
        """Undo all accumulated shifts (difmap unshift)."""
        east, north = self.total_shift
        self._core.unshift()
        self.windows[:] = [
            (x0 - east, x1 - east, y0 - north, y1 - north)
            for (x0, x1, y0, y1) in self.windows
        ]
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

    def gain_snapshot(self):
        """A copy of the current gain table, for use as `savecaltable`'s
        `since=` argument so that later calibration can be exported as a
        separate, incremental table."""
        amp, phs, _ = (np.asarray(x) for x in self._core.gains())
        nt, nif, nant = self._core.ntimes, self.nif, len(self.antennas)
        return {
            "amp": amp.reshape(nt, nif, nant).astype(np.float64).copy(),
            "phs": phs.reshape(nt, nif, nant).astype(np.float64).copy(),
            "used": np.asarray(self._core.gains_used()).reshape(
                nt, nif, nant).copy(),
        }

    def savecaltable(self, path, ms=None, spw_ids=None,
                     flag_uncalibrated=False, overwrite=True, quiet=False,
                     since=None):
        """Export the accumulated antenna gains as a CASA calibration
        table, applicable with CASA's `applycal`.

        The table is a snapshot of every correction applied so far (the
        gains accumulate over `selfcal`/`gscale` calls), holding the
        reciprocal of difmapy's corrections so that CASA's
        ``CORRECTED_DATA = DATA / (G_p conj(G_q))`` reproduces what
        difmapy shows.

        `ms` defaults to the Measurement Set the data came from and is
        required for UVFITS input; antennas are matched by name, so a
        different but compatible MS may be given. Baseline corrections
        (`resoff`) and phase-centre `shift`s cannot be expressed in such
        a table and are reported instead of being dropped silently.

        Pass `since=obs.gain_snapshot()` (taken earlier) to write only
        the calibration accumulated since then, giving one table per
        self-cal round in the CASA style; applying the chain together is
        equivalent to applying one cumulative table.
        """
        from difmapy.io.caltable import save_caltable

        info = save_caltable(self, path, ms=ms, spw_ids=spw_ids,
                             flag_uncalibrated=flag_uncalibrated,
                             overwrite=overwrite, since=since)
        if not quiet:
            print(f"Wrote {info['path']}: {info['nrows']} solutions "
                  f"({info['nflagged']} flagged)")
            for w in info["warnings"]:
                print(f"  warning: {w}")
        return info

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

    def radplot(self, quantity="ap", colorby="spw", block=None):
        """Amplitude and/or phase vs UV radius, with interactive
        flagging.

        `quantity` is "amp", "phase", or "ap" (also spelled "anp" or
        "a&p") for stacked amplitude and phase panels, the default.
        `colorby` chooses the point colouring: "spw" (a gradient over
        the IFs, the default), "time", "baseline", "amp", or "none".
        """
        from difmapy.plots import radplot

        return radplot(self, quantity=quantity, colorby=colorby, block=block)

    def projplot(self, angle=0.0, quantity="ap", colorby="spw", block=None):
        """Amp/phase vs projected UV distance (difmap projplot)."""
        from difmapy.plots import projplot

        return projplot(self, angle_deg=angle, quantity=quantity,
                        colorby=colorby, block=block)

    def uvplot(self, colorby="spw", block=None):
        """UV coverage with interactive flagging."""
        from difmapy.plots import uvplot

        return uvplot(self, colorby=colorby, block=block)

    def vplot(self, reftel=None, quantity="ap", nplot=3, block=None):
        """Visibility amplitude and phase vs time, `nplot` baselines to
        a page, with a per-IF legend and interactive flagging."""
        from difmapy.plots import vplot

        return vplot(self, reftel=reftel, quantity=quantity, nplot=nplot,
                     block=block)

    def mapplot(self, what="map", mapsize=None, cellsize=None, uvweight=None,
                block=None, **clean_args):
        """Interactive map display with CLEAN windows, model editing and
        model fitting.

        `mapsize` (pixels), `cellsize` (mas) and `uvweight` (a Briggs
        robustness from -2 to +2) override the current imaging setup for
        this and later images. With none of them given and `mapsize()`
        never called, `auto_mapsize()` picks a sensible default.
        """
        from difmapy.plots import mapplot

        return mapplot(self, what=what, mapsize=mapsize, cellsize=cellsize,
                       uvweight=uvweight, block=block, **clean_args)

    #: `mapplot` under difmap's shorter spelling.
    maplot = mapplot

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
        """Time-averaged spectrum of the current polarization selection.

        Covers all channels (not only the selected ones), so it can be
        used to decide which channels to select.
        """
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
        # FREQ and STOKES axes, as radio images conventionally carry. The
        # stokes code is that of the selection, so a legacy "pi"
        # selection is written as Stokes I - which is what it is.
        sel = c.selection()
        freqs = [f for f, u in zip(sel["if_freq"], sel["if_used"]) if u]
        hdr["CTYPE3"] = "FREQ"
        hdr["CRVAL3"] = float(np.mean(freqs)) if freqs else 0.0
        hdr["CDELT3"] = 1.0
        hdr["CRPIX3"] = 1.0
        hdr["CTYPE4"] = "STOKES"
        hdr["CRVAL4"] = float(sel["stokes_code"])
        hdr["CDELT4"] = 1.0
        hdr["CRPIX4"] = 1.0
        if beam is not None:
            bmaj, bmin, bpa = beam
            hdr["BMAJ"] = bmaj * MAS / DEG
            hdr["BMIN"] = bmin * MAS / DEG
            hdr["BPA"] = bpa
        hdr["ORIGIN"] = "difmapy"
        # Shape [stokes, freq, dec, ra].
        fits.PrimaryHDU(data=img[np.newaxis, np.newaxis], header=hdr).writeto(
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

    def wobs(self, path, overwrite=True, freeze_shift=False):
        """Write the calibrated, edited UV data to a random-groups
        UVFITS file (difmap wobs).

        Antenna gains and baseline corrections are applied. Accumulated
        phase-center shifts are excluded unless `freeze_shift=True`
        (difmap's default too); `save()` records the shift separately.
        """
        from difmapy.io.uvfits import save_uvfits

        save_uvfits(self._core, path, overwrite=overwrite,
                    freeze_shift=freeze_shift)

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
            # The shift is not frozen into the .uvf (difmap behaviour),
            # so record it here and re-apply it in get().
            "shift": list(self.total_shift),
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
            # The saved .uvf holds unshifted data, so re-apply the
            # shift before loading the model and windows, whose
            # coordinates are relative to the shifted phase center.
            east, north = pars.get("shift", (0.0, 0.0))
            if east or north:
                obs.shift(east, north)
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


def load(path, stokes="I", channels=None, **kwargs) -> Observation:
    """Load a UVFITS file or Measurement Set (difmap observe).

    Unlike difmap, the total intensity is selected straight away, since
    that is how nearly every session starts; pass ``stokes=None`` to
    load without a selection, and `select()` can change it at any time.
    """
    import os

    if os.path.isdir(path):
        return Observation.from_ms(path, stokes=stokes, channels=channels, **kwargs)
    return Observation.from_uvfits(path, stokes=stokes, channels=channels, **kwargs)


#: `load` under difmap's own name for it.
observe = load
