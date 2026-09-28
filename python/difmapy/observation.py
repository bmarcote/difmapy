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

import os

import numpy as np

from difmapy._core import CoreObservation
from difmapy.report import write_json
from difmapy.units import format_interval, parse_interval, parse_time

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


def _comp_dict(comp, freepar=0) -> dict:
    """A core component tuple as a dict, in mas/degrees."""
    t, flux, x, y, major, ratio, phi, freq0, spcind = comp
    return {
        "type": CMP_TYPES.get(t, "?"),
        "flux": flux,
        "x": x / MAS,
        "y": y / MAS,
        "major": major / MAS,
        "ratio": ratio,
        "phi": phi / DEG,
        "freq0": freq0,
        "spcind": spcind,
        "freepar": int(freepar),
    }


__all__ = ["Observation", "load", "observe", "uvaver", "chanaver"]

# The parameters every loader takes, documented once: `_load_doc`
# substitutes them into load(), observe() and the from_* constructors,
# so that help() on any of them lists everything that can be passed.
_LOAD_PARAMETERS = """\
stokes : str | None, default "I"
    Polarization to select once loaded: "I", "Q", "U", "V", or a
    recorded one ("RR", "LL", "RL", "LR", "XX", ...). "I" uses difmap's
    permissive combination (a visibility survives when only one
    parallel hand is usable); Q/U/V are strict. ``None`` (or "none")
    loads without a selection, as difmap's `observe` does; `select()`
    changes it at any time.
channels : list of (int, int) | None, default None
    Inclusive, 0-based channel ranges over the global channel axis (all
    IFs concatenated), e.g. ``[(0, 31), (64, 95)]``, as `select` takes
    them. With `freqavg`, only these channels go into the averages - the
    way band edges are dropped - and the selection then covers every
    averaged channel. ``None``: all channels.
timeavg : float | str | None, default None
    Time-average straight after loading (difmap `uvaver`): seconds as a
    number, or a string with its unit ("10s", "2min"). ``None``: keep
    the original integrations.
freqavg : int | "all" | bool | None, default None
    Average this many adjacent channels into one (`chanaver`); it must
    divide every IF's number of channels. ``"all"`` or ``True`` averages
    each IF down to a single channel. ``None``/``False``: no averaging.
scatter : bool, default False
    With `timeavg`: derive the output weights from the scatter of the
    averaged samples instead of summing the input weights.
wtscale : float, default 1.0
    Factor applied to the data weights as they are read.
field : str | int | None, default None
    Measurement Sets only: the field (name or FIELD_ID) to load. Needed
    when the MS has several fields; difmapy, like difmap, is
    single-source.
data_column : str, default "DATA"
    Measurement Sets only: the visibility column to read, "DATA" or
    "CORRECTED_DATA".
average : float | str | None, default None
    The older name of `timeavg`, still accepted."""

_LOAD_NOTES = """\
The work is done in the cheapest order: channels are averaged first,
then the Stokes selection is made, then the time averaging. Averaged
data can no longer be written back into the Measurement Set (see
`save`), but calibration tables for it still can (`savecaltable`)."""


def _load_doc(fn):
    """Put the shared loader parameters (and notes) into `fn`'s
    docstring, at the ``{LOAD_PARAMETERS}``/``{LOAD_NOTES}`` lines and
    at their indentation."""
    import textwrap

    # UVFITS has no fields or data columns: its block leaves them out.
    ms_only = _LOAD_PARAMETERS[_LOAD_PARAMETERS.index("field :"):
                               _LOAD_PARAMETERS.index("average :")]
    doc = fn.__doc__
    for key, text in (("{LOAD_PARAMETERS}", _LOAD_PARAMETERS),
                      ("{LOAD_PARAMETERS_UVFITS}",
                       _LOAD_PARAMETERS.replace(ms_only, "")),
                      ("{LOAD_NOTES}", _LOAD_NOTES)):
        for line in doc.splitlines():
            if line.strip() == key:
                indent = line[: len(line) - len(line.lstrip())]
                doc = doc.replace(line, textwrap.indent(text, indent))
                break
    fn.__doc__ = doc
    return fn


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
        #: station name -> (rows, pre-ignore FLAG snapshot of those rows)
        self._ignored: dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    @_load_doc
    def from_uvfits(cls, path, wtscale=1.0, stokes="I", channels=None,
                    timeavg=None, freqavg=None, scatter=False,
                    average=None) -> "Observation":
        """Load a random-groups UVFITS file (single source).

        Parameters
        ----------
        path : str
            The UVFITS file.
        {LOAD_PARAMETERS_UVFITS}

        Returns
        -------
        Observation

        Notes
        -----
        {LOAD_NOTES}
        """
        from difmapy.io.uvfits import load_uvfits

        return cls(load_uvfits(path, wtscale=wtscale))._on_load(
            stokes, channels, timeavg, freqavg, scatter, average)

    @classmethod
    @_load_doc
    def from_ms(cls, path, stokes="I", channels=None, timeavg=None,
                freqavg=None, scatter=False, average=None, wtscale=1.0,
                field=None, data_column="DATA") -> "Observation":
        """Load a CASA Measurement Set (single field; needs casatools).

        Parameters
        ----------
        path : str
            The Measurement Set directory.
        {LOAD_PARAMETERS}

        Returns
        -------
        Observation

        Notes
        -----
        {LOAD_NOTES}
        Autocorrelations are dropped, as difmap uses cross-correlations
        only.
        """
        from difmapy.io.ms import load_ms

        return cls(load_ms(path, field=field, data_column=data_column,
                           wtscale=wtscale))._on_load(
            stokes, channels, timeavg, freqavg, scatter, average)

    def _on_load(self, stokes, channels, timeavg, freqavg, scatter,
                 average=None):
        """The selection and averaging asked for at load time, in the
        order that costs least: channels are averaged first (only the
        requested `channels` go into the averages, so band edges can be
        dropped), then the Stokes selection, then the time averaging."""
        if average is not None:
            if timeavg is not None and timeavg != average:
                raise ValueError("give the time averaging as timeavg= only "
                                 "(average= is its older name)")
            timeavg = average
        obs = self
        if freqavg is not None and freqavg is not False:
            obs = obs.chanaver(None if freqavg is True else freqavg,
                               channels=channels)
            channels = None
        obs._initial_select(stokes, channels)
        return obs._averaged(timeavg, scatter)

    def _averaged(self, average, scatter=False):
        """This observation time-averaged by `uvaver`, or itself when
        `average` is None/0."""
        if not average:
            return self
        return self.uvaver(average, doscatter=scatter)

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

    def _report(self, data, outfile):
        """Return `data`, writing it to `outfile` as JSON if asked.

        Every command that reports numbers takes `outfile=`, so a run
        can keep its results without re-deriving the structure.
        """
        return data if outfile is None else write_json(outfile, data)

    def _dirty(self):
        self._invert_result = None
        self._restored = None

    # ------------------------------------------------------------------
    # invert / clean / restore
    # ------------------------------------------------------------------

    def invert(self, outfile=None):
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
        return self._report(self._invert_result, outfile)

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

    def clean(self, niter=100, gain=0.05, cutoff=0.0, quiet=False,
              outfile=None):
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
        return self._report(res, outfile)

    def clrmod(self):
        """Discard the model (difmap clrmod); `clearmodel()` does the
        same and can also drop the CLEAN windows."""
        self._core.clear_models(True, True)
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

    def imstat(self, outfile=None):
        """Statistics of the residual map inner quarter."""
        self._ensure_map()
        return self._report(dict(self._core.map_stats()), outfile)

    def noise_stats(self, image=None, nsigma=3.0, niter=5, outfile=None):
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
            return self._report({"mean": nan, "rms": nan, "raw_rms": nan,
                                 "npix": 0, "nclipped": 0}, outfile)
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
        return self._report({
            "mean": mean,
            "rms": rms,
            "raw_rms": float(flat.std()),
            "npix": int(flat.size),
            "nclipped": int((~keep).sum()),
        }, outfile)

    def moddif(self, uvmin=0.0, uvmax=0.0, outfile=None):
        """Goodness of fit between the model and the data
        (a port of difmap's `moddif`, which is what its self-cal reports
        before and after a solution).

        Returns a dict with the `rms` of the complex model-data
        difference in Jy, `sigma` = sqrt(chisq/ndata) (1 for a fit
        consistent with the data weights), the `chisq` itself, the
        number of measurements `ndata` (real and imaginary counted
        separately, as difmap does) and the number of visibilities
        `nvis`. Only unflagged samples inside the UV range count.
        """
        vis, wt = self._core.stream_vis()
        vis = np.asarray(vis)
        wt = np.asarray(wt, dtype=np.float64)
        model = np.asarray(self._core.stream_model())
        sel = self._core.selection()
        good = (wt > 0) & np.asarray(sel["if_used"], dtype=bool)[None, :]
        if uvmin > 0.0 or uvmax > 0.0:
            _, _, _, us, vs, _ = self._core.rows()
            freq = np.asarray(sel["if_freq"], dtype=float)[None, :]
            uvrad = np.hypot(np.asarray(us)[:, None] * freq,
                             np.asarray(vs)[:, None] * freq)
            good &= uvrad >= float(uvmin)
            if uvmax > 0.0:
                good &= uvrad <= float(uvmax)
        nvis = int(good.sum())
        if nvis == 0:
            nan = float("nan")
            return self._report({"rms": nan, "sigma": nan, "chisq": nan,
                                 "ndata": 0, "nvis": 0}, outfile)
        # |V - M|^2, which is difmap's cosine-rule sqrmod.
        sqrmod = np.abs(vis[good].astype(np.complex128) - model[good]) ** 2
        chisq = abs(float(np.sum(wt[good] * sqrmod)))
        ndata = 2 * nvis
        return self._report({
            "rms": float(np.sqrt(np.mean(sqrmod))),
            "sigma": float(np.sqrt(chisq / ndata)),
            "chisq": chisq,
            "ndata": ndata,
            "nvis": nvis,
        }, outfile)

    def mapinfo(self, outfile=None):
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
        return self._report({
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
        }, outfile)

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
        """The model components (mas/deg units), each with the `freepar`
        bitmask of the parameters `modelfit` may vary.

        This is always the whole model: what `addcmp`, `clean`, `rmodel`
        and `modelfit` produce is part of it at once, with no separate
        step to establish it.
        """
        old, new = self._core.get_models()
        masks = list(self._core.freepars)
        return [_comp_dict(c, m) for c, m in zip(list(old) + list(new), masks)]

    @property
    def model_flux(self) -> float:
        return float(sum(c["flux"] for c in self.model))

    @property
    def nvariable(self) -> int:
        """How many components `modelfit` would vary: those with a free
        parameter. They keep it after a fit, so running `modelfit` again
        continues fitting them (as in difmap).
        """
        return int(self._core.nvariable)

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
        self._dirty()
        return self

    def _replace_model(self, comps):
        """Replace the whole model with `comps` (dicts as `model` returns
        them), keeping their free-parameter masks."""
        self._core.clear_models(True, True)
        for c in comps:
            self._core.add_component(
                CMP_CODES[c["type"]], float(c["flux"]),
                float(c["x"]) * MAS, float(c["y"]) * MAS,
                float(c["major"]) * MAS, float(c["ratio"]),
                float(c["phi"]) * DEG, float(c["freq0"]), float(c["spcind"]),
                freepar=int(c.get("freepar", 0)),
            )
        self._dirty()

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

    def modelfit(self, niter=-1, free=None, uvmin=0.0, uvmax=0.0, quiet=False,
                 outfile=None):
        """Fit the model to the visibilities (difmap modelfit), by
        Levenberg-Marquardt least squares on the residual real/imaginary
        parts. The fitted values replace the old ones in the model;
        there is nothing to keep afterwards.

        Which parameters vary is set per component by `addcmp(free=)`,
        and persists, so running modelfit again continues the fit.
        `free` overrides it: a list gives one spec per model component,
        in `model` order; a single spec sets which parameters vary on
        the components that already have a free parameter (or on every
        component, if none has), leaving the fixed ones - CLEAN
        components above all - fixed. As in difmap (`obvarmod`), the
        fixed components' visibilities are subtracted first, so the fit
        works on the residuals after them. With no model at all, one is
        seeded by `seed_model()`: a circular Gaussian of zero width at
        the peak of the residual map.

        `niter` is the number of Levenberg-Marquardt iterations; the
        default of -1 iterates until the fit converges.

        Returns (and, unless `quiet`, prints) a dict with
        rchisq/chisq/ndfree/nvis/nfree, whether it `converged`, the
        fitted `components`, and a list of 1-sigma `errors` per
        component, in mas/degrees.
        """
        if not self.model:
            seeded = self.seed_model()
            if not quiet:
                print(
                    f"modelfit: no model given; starting from a circular "
                    f"Gaussian of {seeded['flux']:.5g} Jy at "
                    f"({seeded['x']:.4g}, {seeded['y']:.4g}) mas"
                )
        current = [c["freepar"] for c in self.model]
        ncmp = len(current)
        if free is None:
            masks = current
        elif isinstance(free, (list, tuple)) and free and isinstance(
            free[0], (list, tuple, int)
        ):
            if len(free) != ncmp:
                raise ValueError(
                    f"free has {len(free)} entries for the {ncmp} model components"
                )
            masks = [_free_mask(f) for f in free]
        else:
            mask = _free_mask(free)
            # Which parameters vary, on the components being fitted; a
            # fixed component (a CLEAN delta) is not freed by it.
            masks = ([mask if m else 0 for m in current] if any(current)
                     else [mask] * ncmp)
        if not any(masks):
            raise ValueError(
                f"none of the {ncmp} model components has a free parameter; "
                "pass free=... to modelfit() or addcmp(), or clearmodel() "
                "first to fit from scratch"
            )
        # The core moves visibilities in and out of the stream model, so
        # the cached map is stale whether or not the fit succeeds.
        try:
            res = dict(
                self._core.modelfit(
                    niter=int(niter),
                    freepars=[int(m) for m in masks],
                    uvmin=float(uvmin),
                    uvmax=float(uvmax),
                )
            )
        finally:
            self._dirty()
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
        # The fitted components are the last ones in the model.
        model = self.model
        nfitted = int(res.pop("nfitted"))
        res["components"] = model[len(model) - nfitted:]
        res["ncomp"] = len(res["components"])
        res["total_ncomp"] = len(self.model)
        res["total_flux"] = self.model_flux
        if not quiet:
            self._print_modelfit(res)
        return self._report(res, outfile)

    @staticmethod
    def _print_modelfit(res):
        end = "converged" if res.get("converged") else "stopped"
        print(
            f"modelfit: {end} after {res.get('niter', '?')} iterations; "
            f"reduced chi-squared {res['rchisq']:.5g} "
            f"({res['nvis']} visibilities, {res['nfree']} free parameters)"
        )
        print(f"  fitted {res['ncomp']} of {res['total_ncomp']} components; "
              f"{res['total_flux']:.5g} Jy in the model")
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

    def scans(self, gap=None, outfile=None) -> list[dict]:
        """The scans of the observation, as a list of dicts with the
        integration index range (`first`, `last`), the time range
        (`tmin`, `tmax`, seconds) and the number of integrations
        (`nint`).

        Two integrations more than `gap` apart are in different scans -
        difmap's definition, with a default of five times the median
        integration spacing rather than difmap's flat hour, which is
        meant for breaking plot axes rather than for self-calibrating
        per scan. `gap` takes a unit string ("5min") or seconds.

        This is what `selfcal(solint="scan")` bins by, so it is the
        thing to check when a per-scan solution looks wrong.
        """
        g = 0.0 if gap is None else parse_time(gap, "s")
        times = np.asarray(self._core.times())
        out = [
            {
                "first": int(a), "last": int(b),
                "tmin": float(times[a]), "tmax": float(times[b]),
                "nint": int(b - a + 1),
            }
            for a, b in self._core.scans(scangap=g)
        ]
        return self._report(out, outfile)

    @property
    def default_scangap(self) -> float:
        """The gap in seconds that `scans()` uses by default."""
        return float(self._core.default_scangap)

    def selfcal(self, amp=False, phase=True, float_scale=False, solint=0.0,
                gauval=0.0, gaurad=0.0, maxamp=0.0, maxphs=0.0,
                uvmin=0.0, uvmax=0.0, mintel=0, flag=False,
                quiet=False, mapstats=None, scangap=None, outfile=None):
        """Self-calibrate against the current model (difmap selfcal).

        `solint` is the solution interval: a bare number in minutes
        (difmap's unit; 0 = one solution per integration), or a string
        carrying its own unit - ``"30s"``, ``"90 sec"``, ``"1min"``,
        ``"2h"``. ``"scan"`` solves each scan as a single interval, so
        every scan gets exactly one solution however long it is, and
        ``"2scan"`` (or ``"3 scans"``) groups that many scans per
        solution. Scans are delimited by a gap of `scangap` (seconds,
        or a string with a unit); the default is five times the median
        integration spacing - see `scans()` to check what that gives.

        maxphs is in degrees. Each IF of each subarray is solved
        independently, so the returned `nbins` counts solution
        intervals over all of them.

        Returns (and, unless `quiet`, prints) how well the model fits
        the data before and after the solution, as difmap does:
        `fit_before` and `fit_after` are `moddif()` dicts. With
        `mapstats` the residual map is measured on both sides too, into
        `map_before`/`map_after`; the default measures it whenever a map
        already exists, which costs nothing in an imaging loop because
        self-cal invalidates the map and the next `clean` would have to
        re-invert anyway.
        """
        secs, nscan = parse_interval(solint, "min")
        gap = 0.0 if scangap is None else parse_time(scangap, "s")
        if mapstats is None:
            mapstats = self._invert_result is not None
        before = self.moddif(uvmin, uvmax)
        map_before = self.imstat() if mapstats else None
        res = self._core.selfcal(
            doamp=bool(amp),
            dophs=bool(phase),
            dofloat=bool(float_scale),
            solint=secs / 60.0,
            nscan=nscan,
            scangap=gap,
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
        res = dict(res)
        res["fit_before"] = before
        res["fit_after"] = self.moddif(uvmin, uvmax)
        if mapstats:
            res["map_before"] = map_before
            res["map_after"] = self.imstat()
        if not quiet:
            self._print_selfcal(res, amp=amp, phase=phase,
                                every=format_interval(secs, nscan))
        return self._report(res, outfile)

    @staticmethod
    def _print_selfcal(res, amp, phase, every):
        what = ("amplitude and phase" if amp and phase else
                "amplitude" if amp else "phase" if phase else "nothing")
        print(f"selfcal: {what} {every}; {res['nbins']} solution bins, "
              f"{res['nbadsol']} unusable, "
              f"{res['nbadtel']} bad telescope solutions")
        Observation._print_fit(res)

    @staticmethod
    def _print_fit(res):
        """The model-data fit, and the residual map, before and after a
        calibration step."""
        for when in ("before", "after"):
            f = res.get(f"fit_{when}")
            if f is None:
                continue
            print(f"  fit {when:<6}: rms {f['rms']:.5g} Jy, "
                  f"sigma {f['sigma']:.5g} ({f['nvis']} visibilities)")
        if "map_after" in res:
            for when in ("before", "after"):
                m = res[f"map_{when}"]
                print(f"  map {when:<6}: peak {m['max']:.5g}, "
                      f"min {m['min']:.5g}, rms {m['rms']:.5g} Jy/beam")

    def gscale(self, float_scale=False, quiet=False, mapstats=None,
               outfile=None):
        """Overall telescope amplitude corrections (difmap gscale).

        One amplitude correction per telescope and IF is solved for the
        whole observation, and applied - like `selfcal`, this changes
        the data the rest of the session sees. Returns (and, unless
        `quiet`, prints) a dict whose `gains` maps each station name to
        the correction this call applied, with the per-IF values in
        `gains_per_if`; `norms` holds the per-(subarray, IF)
        renormalisation factors, and is empty when the overall scale is
        left floating.

        As for `selfcal`, `fit_before`/`fit_after` report how the fit to
        the model changed, and `map_before`/`map_after` the residual map
        when there is one. Note that with `float_scale=False` (the
        default, as in difmap) the gains are renormalised to preserve
        the *data's* flux scale rather than pull it onto the model, so
        the corrections can be large while the map peak barely moves -
        the fit statistics are what show the improvement.
        """
        if mapstats is None:
            mapstats = self._invert_result is not None
        fit_before = self.moddif()
        map_before = self.imstat() if mapstats else None
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
        res["fit_before"] = fit_before
        res["fit_after"] = self.moddif()
        if mapstats:
            res["map_before"] = map_before
            res["map_after"] = self.imstat()
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
            scale = "floating" if float_scale else "renormalised"
            print(f"gscale: amplitude corrections, {scale} scale "
                  f"({res['nbins']} solution bins, "
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
            self._print_fit(res)
        return self._report(res, outfile)

    def bayes_gscale(self, models=("clean", "gauss1", "gauss2", "gauss3"),
                     prior_sigma=0.10, jackknife=True, per_if=True,
                     float_scale=False, nloop=2, solint=0.0, clean_gain=0.05,
                     sigma_floor=0.01, timeavg=None, workers=None, apply=True,
                     prefix=None, outformat=None, ms=None, uvfits=None,
                     plot=None, quiet=False):
        """Bayesian amplitude calibration of every station: `gscale`
        with each station left out of the source model in turn, several
        source models weighed against each other, and the result shrunk
        towards the a-priori calibration (`prior_sigma`). Corrections
        the data do not call for are not applied.

        See `difmapy.bayescal` for the method and every parameter. The
        runs are independent and go in parallel threads (`workers`).
        With `apply` the corrections are applied like `gscale`'s; with
        `prefix` the JSON report, the diagnostic figure and the
        calibration table of the constant corrections (for other sources
        of the observation; `outformat` as for `savecaltable`) are
        written. Returns a `BayesGainResult`: print it for the summary,
        ``.plot()`` for the diagnostics.
        """
        from difmapy.bayescal import bayes_gscale

        return bayes_gscale(
            self, models=models, prior_sigma=prior_sigma, jackknife=jackknife,
            per_if=per_if, float_scale=float_scale, nloop=nloop, solint=solint,
            clean_gain=clean_gain, sigma_floor=sigma_floor, timeavg=timeavg,
            workers=workers, apply=apply, prefix=prefix, outformat=outformat,
            ms=ms, uvfits=uvfits, plot=plot, quiet=quiet)

    def _gain_amps(self, fill=np.nan):
        """The gain-table amplitudes and their "solved" mask, both
        [ntimes, nif, nant]; `fill` replaces unsolved entries."""
        amp, _, _ = self._core.gains()
        nt, nif, nant = self._core.ntimes, self.nif, len(self.antennas)
        used = np.asarray(self._core.gains_used()).reshape(nt, nif, nant)
        amp = np.asarray(amp, dtype=np.float64).reshape(nt, nif, nant)
        return np.where(used, amp, fill), used

    def station_gains(self, per_if=False, outfile=None):
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
        return self._report(out, outfile)

    def uncalib(self, amp=True, phase=True, flags=False):
        """Undo selfcal corrections (difmap uncalib)."""
        self._core.uncalib(amp, phase, flags)
        self._dirty()
        return self

    #: Names that `selfant` takes to mean "every antenna in the array".
    ALL_ANTENNAS = ("all", "*")

    def selfant(self, name=None, fix=None, weight=None, quiet=False,
                outfile=None) -> list[dict]:
        """Constrain antennas in self-calibration (difmap selfant).

        A fixed antenna keeps its gain of 1 while the others solve
        against it, and `weight` scales how much a baseline to that
        antenna counts in the solution.

        `name` is an antenna name, ``"all"`` or ``"*"`` for every
        antenna in the array, or a list of names. With no `name` at all
        nothing is changed and the current constraints are simply
        reported, which is the way to check what a session has
        accumulated.

        `fix` and `weight` left as None keep whatever the named
        antennas already have, so one can be set without disturbing the
        other.

        Returns (and, unless `quiet`, prints) the constraints on every
        antenna as a list of dicts with `antenna`, `subarray`, `fixed`
        and `weight`.
        """
        if name is not None:
            if isinstance(name, str):
                names = ([a for a in self.antennas]
                         if name.strip().lower() in self.ALL_ANTENNAS
                         else [name])
            else:
                names = list(name)
            current = {n.lower(): (f, w) for n, _, f, w
                       in self._core.antenna_constraints}
            for n in names:
                have = current.get(str(n).lower())
                if have is None:
                    raise ValueError(
                        f"unknown antenna {n!r}; the array is "
                        f"{', '.join(self.antennas)}"
                    )
                self._core.set_antenna_constraints(
                    str(n),
                    have[0] if fix is None else bool(fix),
                    have[1] if weight is None else float(weight),
                )
        out = [
            {"antenna": n, "subarray": int(sub),
             "fixed": bool(f), "weight": float(w)}
            for n, sub, f, w in self._core.antenna_constraints
        ]
        if not quiet:
            self._print_selfant(out)
        return self._report(out, outfile)

    @staticmethod
    def _print_selfant(rows):
        nsub = len({r["subarray"] for r in rows})
        width = max([len(r["antenna"]) for r in rows] + [7])
        head = f"{'antenna':<{width}}  fixed  weight"
        if nsub > 1:
            head += "  subarray"
        print(head)
        for r in rows:
            line = (f"{r['antenna']:<{width}}  "
                    f"{'yes' if r['fixed'] else 'no':<5}  "
                    f"{r['weight']:6.3f}")
            if nsub > 1:
                line += f"  {r['subarray']:8d}"
            print(line)
        nfix = sum(r["fixed"] for r in rows)
        nwt = sum(r["weight"] != 1.0 for r in rows)
        print(f"{len(rows)} antennas: {nfix} fixed, "
              f"{nwt} with a weight other than 1")

    def startmod(self, model=None, solint=0.0, flux=1.0):
        """Phase self-calibrate against a starting model, then discard
        it (difmap startmod).

        `model` is a .mod file name; with no model a point source of
        `flux` Jy at the phase center is used, as difmap does. `solint`
        is passed to `selfcal`, so it takes the same units and the
        same "scan"/"Nscan" forms.
        """
        self.clrmod()
        if model is None:
            self.addcmp(flux, 0.0, 0.0)
        else:
            self.rmodel(model)
        res = self.selfcal(phase=True, solint=solint)
        self.clrmod()
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

    def baseline_corrections(self, outfile=None):
        """Baseline corrections as a list of dicts (amp, phase in deg)."""
        bls, amp, phs = self._core.baseline_corrections()
        names = self._core.antenna_names
        amp = np.asarray(amp)
        phs = np.asarray(phs)
        return self._report([
            {
                "baseline": (names[a], names[b]),
                "amp": amp[i].tolist(),
                "phase": (phs[i] / DEG).tolist(),
            }
            for i, (a, b) in enumerate(bls)
        ], outfile)

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
        into `aver_time` integrations (difmap uvaver).

        Parameters
        ----------
        aver_time : float | str
            The new integration time: seconds as a bare number, or a
            string carrying its unit (``"30s"``, ``"2min"``).
        doscatter : bool, default False
            Derive the output weights from the scatter of the averaged
            samples instead of summing the input weights.

        The gains, baseline corrections and any shift are applied before
        averaging, and the new observation starts uncalibrated, with this
        one's selection, imaging setup and windows.
        """
        from difmapy.average import uvaver

        return self._derived(
            uvaver(self._core, parse_time(aver_time, "s"), bool(doscatter))
        )

    def chanaver(self, nchan=None, channels=None):
        """Return a new observation with every `nchan` adjacent channels
        of each IF averaged into one (AIPS AVSPC, CASA split's `width`).

        Parameters
        ----------
        nchan : int | "all" | None, default None
            Channels per output channel; it must divide each IF's number
            of channels. ``None`` (or ``"all"``) averages every IF down
            to a single channel.
        channels : list of (int, int) | None, default None
            Inclusive global channel ranges, as for `select`, to average
            only those - the way band edges are left out. ``None``: all.

        As with `uvaver`, the calibrated data are averaged and the new
        observation starts uncalibrated; the channel selection is reset
        to all (averaged) channels, keeping the Stokes selection.
        """
        from difmapy.average import chanaver

        chlist = [(int(a), int(b)) for a, b in (channels or [])]
        return self._derived(chanaver(self._core, nchan, chlist or None),
                             same_channels=False)

    def _derived(self, core, same_channels=True):
        """A new observation around `core` - made from this one's data by
        averaging - with this one's imaging setup, windows, selection
        and provenance.

        The rows (time averaging) or channels (frequency averaging) no
        longer map onto the source file's, so the data cannot be written
        back to a Measurement Set; `_from_ms` remembers where it came
        from to say so. The calibration provenance (`_cal_origin`) still
        holds, since calibration tables refer to antennas, IFs and times
        rather than rows.
        """
        new = Observation(core)
        origin = getattr(self._core, "_ms_origin", None)
        new._from_ms = origin["path"] if origin else getattr(self, "_from_ms", None)
        cal_origin = getattr(self._core, "_cal_origin", None)
        if cal_origin is not None:
            core._cal_origin = cal_origin
        # Carry over the imaging setup and selection.
        new._nx, new._ny = self._nx, self._ny
        new._xinc, new._yinc = self._xinc, self._yinc
        new._binwid, new._errpow, new._dorad = self._binwid, self._errpow, self._dorad
        new._robust = self._robust
        new._mapsize_set = self._mapsize_set
        new._gauval, new._gaurad = self._gauval, self._gaurad
        new._uvmin, new._uvmax = self._uvmin, self._uvmax
        new._uvzero = self._uvzero
        new.windows = list(self.windows)
        try:
            sel = self._core.selection()
            chans = [tuple(r) for r in sel["chlist"]] if same_channels else None
            new.select(sel["stokes"], channels=chans)
        except RuntimeError:
            pass
        return new

    def copy(self) -> "Observation":
        """An independent copy of this observation: data, flags, gains,
        model, selection, map and every imaging setting. Changing one
        never affects the other (which is what lets calibration
        experiments run side by side, see `bayes_gscale`)."""
        import copy as _copy

        new = Observation.__new__(Observation)
        state = {k: v for k, v in self.__dict__.items() if k != "_core"}
        new.__dict__.update(_copy.deepcopy(state))
        new._core = self._core.copy()
        # Provenance attached from Python (_ms_origin, _cal_origin); it
        # describes the source file and is never modified, so it can be
        # shared.
        new._core.__dict__.update(self._core.__dict__)
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
            tmin=parse_time(tmin, "s"),
            tmax=parse_time(tmax, "s"),
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
        reference day, or a string with a unit such as "1.5h"),
        subarray or IF. Returns #rows affected."""
        n = self._edit(True, baseline, station, tmin, tmax, subarray,
                       if_index, selected_channels_only)
        self._reapply_ignores()
        self._dirty()
        return n

    def unflag(self, baseline=None, station=None, tmin=None, tmax=None,
               subarray=None, if_index=None, selected_channels_only=False):
        """Unflag visibilities (difmap unflag)."""
        n = self._edit(False, baseline, station, tmin, tmax, subarray,
                       if_index, selected_channels_only)
        # An unflag must not resurrect an ignored station.
        self._reapply_ignores()
        self._dirty()
        return n

    # ------------------------------------------------------------------
    # ignoring stations
    # ------------------------------------------------------------------

    @property
    def ignored(self) -> list[str]:
        """The stations currently being ignored (see `ignore`)."""
        return sorted(self._ignored)

    def _station_rows(self, name) -> np.ndarray:
        """Row indices of every baseline that includes this station."""
        ia = self._ant_index(name)
        _, a1, a2, _, _, _ = self._core.rows()
        a1 = np.asarray(a1, dtype=int)
        a2 = np.asarray(a2, dtype=int)
        return np.nonzero((a1 == ia) | (a2 == ia))[0]

    def _flags_without_ignores(self, rows) -> np.ndarray:
        """The flags `rows` would carry if nothing were being ignored.

        A baseline between two ignored stations belongs to both, so its
        current flags are ignore flags, not the user's; the snapshot
        taken when the *other* station was ignored is what that row
        really looked like.
        """
        block = np.array(self._core.flags_rows([int(r) for r in rows]), copy=True)
        where = {int(r): i for i, r in enumerate(rows)}
        for other_rows, other_block in self._ignored.values():
            for j, r in enumerate(other_rows):
                i = where.get(int(r))
                if i is not None:
                    block[i] = other_block[j]
        return block

    def _reapply_ignores(self):
        """Re-flag the ignored stations' data.

        Called after anything that edits flags, so that ignoring
        outlives an `unflag` that would otherwise bring the station
        back without anyone asking.
        """
        for rows, _ in self._ignored.values():
            self._core.edit_rows([int(r) for r in rows], True,
                                 if_index=None, sel_chan=False)

    def ignore(self, *antennas):
        """Set aside every baseline of these stations until `unignore`.

        Names are case-insensitive. The data is flagged, so imaging,
        model fitting, self-calibration and the plots all skip it - but
        unlike `flag`, the exact flag state of those baselines is
        remembered first, so `unignore` puts them back as they were
        rather than wholesale unflagged. That is the point: everyone
        else's data can be flagged and calibrated in the meantime, and
        the station still returns with its own history intact.

        Two things worth knowing. Flag edits made to an ignored
        station's own baselines while it is ignored are discarded by
        `unignore`, since it restores the remembered state. And
        self-calibration solved while a station is ignored has no
        solution for it, so it comes back uncalibrated for those
        intervals - `selfcal` again, or `selfant` to hold it fixed.

        Returns the list of stations ignored after the call.
        """
        if not antennas:
            raise ValueError(
                "ignore() needs at least one antenna name; "
                "obs.ignored lists the ones already set aside"
            )
        names = [self.antennas[self._ant_index(a)] for a in antennas]
        for name in names:
            if name in self._ignored:
                continue
            rows = self._station_rows(name)
            if rows.size == 0:
                continue
            self._ignored[name] = (rows, self._flags_without_ignores(rows))
            self._core.edit_rows([int(r) for r in rows], True,
                                 if_index=None, sel_chan=False)
        self._dirty()
        return self.ignored

    def unignore(self, *antennas):
        """Bring back stations set aside by `ignore` (all, by default).

        Their baselines are restored to the flag state they had when
        they were ignored; baselines shared with a station that is
        still ignored stay out.
        """
        names = ([self.antennas[self._ant_index(a)] for a in antennas]
                 if antennas else list(self._ignored))
        for name in names:
            entry = self._ignored.pop(name, None)
            if entry is None:
                continue
            rows, block = entry
            self._core.set_flags_rows(
                [int(r) for r in rows],
                np.ascontiguousarray(block, dtype=bool),
            )
        self._reapply_ignores()
        self._dirty()
        return self.ignored

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

    #: `savecaltable` output formats, by the names it accepts.
    CALTABLE_FORMATS = {"casa": ("casa",), "aips": ("aips",),
                        "both": ("casa", "aips")}

    def caltable_formats(self, outformat=None, ms=None, uvfits=None) -> tuple:
        """The formats `savecaltable` writes for `outformat`: by default
        the native one of the file the data came from - a CASA table for
        a Measurement Set, an AIPS SN table for UVFITS - unless a
        reference file of the other kind is given (`ms`/`uvfits`), which
        says which table is wanted."""
        if outformat is None or str(outformat).lower() == "auto":
            if ms and not uvfits:
                return ("casa",)
            if uvfits and not ms:
                return ("aips",)
            origin = getattr(self._core, "_cal_origin", None) or {}
            if origin.get("format") == "ms" or getattr(
                    self._core, "_ms_origin", None):
                return ("casa",)
            return ("aips",)
        key = str(outformat).strip().lower()
        if key not in self.CALTABLE_FORMATS:
            raise ValueError(
                f"unknown outformat {outformat!r}; use 'CASA', 'AIPS' or "
                "'both' (case does not matter)"
            )
        return self.CALTABLE_FORMATS[key]

    @staticmethod
    def caltable_paths(path, formats) -> dict:
        """Where each format goes. A single format is written to `path`
        itself. With both, a `path` ending in ``.fits`` is the AIPS file
        and the CASA table is that name without ``.fits`` (and without a
        ``.TASAV`` before it); any other `path` is the CASA table, and
        the AIPS file is ``<path>.TASAV.FITS``."""
        path = os.fspath(path)
        if len(formats) == 1:
            return {formats[0]: path}
        if path.lower().endswith(".fits"):
            base = path[:-5]
            if base.lower().endswith(".tasav"):
                base = base[:-6]
            return {"casa": base, "aips": path}
        return {"casa": path, "aips": f"{path}.TASAV.FITS"}

    def savecaltable(self, path, outformat=None, ms=None, uvfits=None,
                     spw_ids=None, flag_uncalibrated=False, overwrite=True,
                     quiet=False, since=None, solutions=None):
        """Export the accumulated antenna gains as a calibration table
        that CASA or AIPS can apply to the data.

        `outformat` is ``"CASA"`` (a "G Jones" table for `applycal`),
        ``"AIPS"`` (an SN table inside a TASAV FITS file, for FITLD and
        TACOP) or ``"both"``, in any case. By default it follows the
        data: a Measurement Set gets a CASA table, a UVFITS file an AIPS
        one - or, given only `ms` or only `uvfits`, the table that file is
        for. See `caltable_paths` for where `path` puts each of the two.

        The table is a snapshot of every correction applied so far (the
        gains accumulate over `selfcal`/`gscale` calls). CASA divides
        the data by its gains, so a CASA table holds the reciprocal of
        difmapy's corrections; AIPS multiplies by them, so an SN table
        holds them as they are. Either way, applying the table
        reproduces what difmapy shows.

        `ms` is the Measurement Set a CASA table refers to (default: the
        one the data came from; required for UVFITS input), and
        `uvfits` the UVFITS file an SN table is numbered for (default:
        the one the data came from; for MS input, antennas are numbered
        in ANTENNA-table order as exportuvfits does). In both cases
        antennas are matched by name, so a different but compatible
        file - another source of the same observation - may be given.
        Baseline corrections (`resoff`) and phase-centre `shift`s cannot
        be expressed in such a table and are reported instead of being
        dropped silently.

        Pass `since=obs.gain_snapshot()` (taken earlier) to write only
        the calibration accumulated since then, giving one table per
        self-cal round; applying the chain together is equivalent to
        applying one cumulative table. `solutions` writes a given set of
        gains instead of the gain table (`bayes_gscale` uses it).

        Returns the writer's summary (`path`, `nrows`, `nflagged`,
        `warnings`, `format`); with both formats, one per format under
        ``"casa"`` and ``"aips"``, and the warnings once at the top.
        """
        from difmapy.io.aips import save_sntable
        from difmapy.io.caltable import save_caltable

        formats = self.caltable_formats(outformat, ms=ms, uvfits=uvfits)
        paths = self.caltable_paths(path, formats)
        infos = {}
        for fmt in formats:
            if fmt == "casa":
                info = save_caltable(self, paths[fmt], ms=ms, spw_ids=spw_ids,
                                     flag_uncalibrated=flag_uncalibrated,
                                     overwrite=overwrite, since=since,
                                     solutions=solutions)
            else:
                info = save_sntable(self, paths[fmt], uvfits=uvfits,
                                    flag_uncalibrated=flag_uncalibrated,
                                    overwrite=overwrite, since=since,
                                    solutions=solutions)
            info["format"] = fmt
            infos[fmt] = info
        if not quiet:
            label = {"casa": "CASA G table", "aips": "AIPS SN table (TASAV)"}
            for fmt, info in infos.items():
                print(f"Wrote {label[fmt]} {info['path']}: {info['nrows']} "
                      f"rows ({info['nflagged']} flagged solutions)")
            for w in next(iter(infos.values()))["warnings"]:
                print(f"  warning: {w}")
        if len(infos) == 1:
            return next(iter(infos.values()))
        out = dict(infos)
        out["warnings"] = next(iter(infos.values()))["warnings"]
        out["path"] = [info["path"] for info in infos.values()]
        return out

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

    def projplot(self, angle=0.0, quantity="ap", colorby="spw", block=None,
                 step=10.0):
        """Amp/phase vs projected UV distance (difmap projplot).

        `angle` is the position angle of the projection direction in
        degrees, north through east. In the window ``<`` and ``>`` turn
        it by `step` degrees and redraw.
        """
        from difmapy.plots import projplot

        return projplot(self, angle_deg=angle, quantity=quantity,
                        colorby=colorby, block=block, step=step)

    def uvplot(self, colorby="spw", block=None):
        """UV coverage with interactive flagging."""
        from difmapy.plots import uvplot

        return uvplot(self, colorby=colorby, block=block)

    def vplot(self, nplot=3, reftel=None, quantity="ap", block=None):
        """Visibility amplitude and phase vs time, `nplot` baselines to
        a page, with a per-IF legend and interactive flagging.

        The arguments come in difmap's order: ``vplot(3)`` is three
        baselines to a page, ``vplot(3, "EF")`` only EF's baselines, and
        0 puts all of a station's baselines on one page.
        """
        from difmapy.plots import vplot

        return vplot(self, nplot=nplot, reftel=reftel, quantity=quantity,
                     block=block)

    def mapplot(self, what="map", mapsize=None, cellsize=None, uvweight=None,
                block=None, scale="linear", **clean_args):
        """Interactive map display with CLEAN windows, model editing and
        model fitting.

        `mapsize` (pixels), `cellsize` (mas) and `uvweight` (a Briggs
        robustness from -2 to +2) override the current imaging setup for
        this and later images. With none of them given and `mapsize()`
        never called, `auto_mapsize()` picks a sensible default.

        `scale` is "linear" or "log" for the colour scale, which the
        "l" key also toggles.

        The "Weighting" box at the top of the window switches between
        difmap's own `uvweight` scheme and robust -2, -1, 0, +1, +2, and
        re-images at once; like `uvweight()`, the choice stays in effect
        after the window closes.
        """
        from difmapy.plots import mapplot

        return mapplot(self, what=what, mapsize=mapsize, cellsize=cellsize,
                       uvweight=uvweight, block=block, scale=scale,
                       **clean_args)

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

    def corplot(self, quantity="both", nplot=3, block=None):
        """Self-cal gain corrections vs time (difmap corplot).

        One antenna to a pair of panels, amplitude above and phase
        below on a shared time axis, `nplot` antennas to a page.
        `quantity` is "both" (the default), "amp" or "phase".
        """
        from difmapy.plots import corplot

        return corplot(self, quantity=quantity, nplot=nplot, block=block)

    def fplot(self, reftel=None, baselines=None, nplot=3, tmin=None,
              tmax=None, calibrated=True, block=None):
        """Amplitude and phase vs frequency, one baseline to a pair of
        panels, averaged over the whole time range.

        The frequency counterpart of `vplot`: `nplot` baselines to a
        page (n/p to page through them), every channel of every IF
        averaged over the observation, or over `tmin`..`tmax` (seconds,
        or strings with units such as "30min").
        `reftel` keeps only one station's baselines, `baselines` names
        them explicitly. Amplitudes are scalar-averaged and phases
        vector-averaged, and the accumulated calibration is applied
        unless `calibrated=False`.

        Use `specplot()` for the same thing averaged over all baselines
        at once.
        """
        from difmapy.plots import fplot

        return fplot(self, reftel=reftel, baselines=baselines, nplot=nplot,
                     tmin=tmin, tmax=tmax, calibrated=calibrated, block=block)

    def specplot(self, baseline=None, tmin=None, tmax=None, xaxis="freq", block=None):
        """Time-averaged spectrum of the selected polarization
        (difmap specplot)."""
        from difmapy.plots import specplot

        return specplot(self, baseline=baseline, tmin=tmin, tmax=tmax,
                        xaxis=xaxis, block=block)

    # ------------------------------------------------------------------
    # closure / spectral data (without plotting)
    # ------------------------------------------------------------------

    def closure_phases(self, triangle=None, if_index=None, outfile=None):
        """Closure phase time series (radians). Returns a list of dicts
        with triangle/if_index/time/phase/model/error."""
        idx = None
        if triangle is not None:
            idx = tuple(sorted(self._ant_index(t) for t in triangle))
        return self._report(
            [dict(d) for d in self._core.closure_phases(
                triangle=idx, if_index=if_index)],
            outfile,
        )

    def spectrum(self, baseline=None, tmin=None, tmax=None, calibrated=True,
                 outfile=None):
        """Time-averaged spectrum of the current polarization selection.

        Covers all channels (not only the selected ones), so it can be
        used to decide which channels to select. The visibilities are
        vector-averaged over `tmin`..`tmax` (the whole observation by
        default; seconds, or strings with units) with the accumulated
        calibration applied per channel; `calibrated=False` averages
        the data as it was loaded.
        """
        bl = None
        if baseline is not None:
            bl = (self._ant_index(baseline[0]), self._ant_index(baseline[1]))
        return self._report(dict(self._core.spectrum(
            baseline=bl, tmin=parse_time(tmin, "s"),
            tmax=parse_time(tmax, "s"), calibrated=bool(calibrated),
        )), outfile)

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
        """Read a difmap/Caltech .mod model file, adding its components
        to the model."""
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

    def _save_ms(self, prefix, ms):
        """The ``<prefix>.ms`` half of `save`, if this session came from
        a Measurement Set.

        `ms` is None to write one when it is possible and say nothing
        when it is not, False to skip it, True to require it.
        """
        if ms is False:
            return None
        from difmapy.io.ms import save_ms

        if getattr(self._core, "_ms_origin", None) is None:
            if ms:
                raise ValueError(
                    "ms=True, but this observation was not loaded from a "
                    "Measurement Set; only <prefix>.uvf can be written"
                )
            if getattr(self, "_from_ms", None):
                print(f"warning: {prefix}.ms not written: the data were "
                      f"time-averaged after loading {self._from_ms}, so its "
                      "rows no longer match")
            return None
        try:
            return save_ms(self._core, f"{prefix}.ms", overwrite=True)
        except Exception as exc:
            if ms:
                raise
            # The UV data is already safely in the .uvf, so a missing
            # casatools or an unreadable source MS must not lose it.
            print(f"warning: could not write {prefix}.ms: {exc}")
            return None

    def save(self, prefix, ms=None):
        """Save UV data, model, windows and imaging parameters with a
        common prefix (difmap save).

        Writes ``<prefix>.uvf`` (UV data), ``.mod`` (model), ``.win``
        (CLEAN windows) and ``.par.json`` (the imaging parameters), all
        of which `get()` reads back.

        An observation loaded from a Measurement Set also gets a
        ``<prefix>.ms``, so a session that started from CASA can go back
        to it: the originating MS is copied and the calibrated
        visibilities and current flags written into the copy (see
        `difmapy.io.ms.save_ms`). Pass ``ms=False`` to skip it, or
        ``ms=True`` to make its absence an error rather than a warning.
        """
        import json

        self.wobs(f"{prefix}.uvf")
        self._save_ms(prefix, ms)
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


@_load_doc
def load(path, stokes="I", channels=None, timeavg=None, freqavg=None,
         scatter=False, wtscale=1.0, field=None, data_column="DATA",
         average=None) -> Observation:
    """Load a UVFITS file or a Measurement Set (difmap `observe`).

    Unlike difmap, the total intensity is selected straight away, since
    that is how nearly every session starts.

    Parameters
    ----------
    path : str
        A random-groups UVFITS file, or a Measurement Set directory
        (read with casatools); a directory is taken to be an MS. Both
        must hold a single source.
    {LOAD_PARAMETERS}

    Returns
    -------
    Observation

    Notes
    -----
    {LOAD_NOTES}

    Examples
    --------
    >>> obs = difmapy.load("mysource.uvfits")
    >>> obs = difmapy.load("big.ms", timeavg="10s", freqavg="all")
    >>> obs = difmapy.load("big.ms", channels=[(2, 29)], freqavg="all",
    ...                    field="3C345", data_column="CORRECTED_DATA")
    """
    import os

    opts = dict(stokes=stokes, channels=channels, timeavg=timeavg,
                freqavg=freqavg, scatter=scatter, average=average,
                wtscale=wtscale)
    if os.path.isdir(path):
        return Observation.from_ms(path, field=field,
                                   data_column=data_column, **opts)
    if field is not None or str(data_column).upper() != "DATA":
        raise ValueError(
            f"field= and data_column= apply to Measurement Sets only, and "
            f"{path} is a UVFITS file"
        )
    return Observation.from_uvfits(path, **opts)


def chanaver(obs, nchan=None, channels=None) -> Observation:
    """Average adjacent channels of each IF, returning a new observation
    (`obs` is left as it is); the same as ``obs.chanaver(nchan,
    channels)``.

    Parameters
    ----------
    obs : Observation
        The observation to average.
    nchan : int | "all" | None, default None
        Channels per output channel; it must divide every IF's number
        of channels. ``None``/``"all"``: one channel per IF.
    channels : list of (int, int) | None, default None
        Inclusive global channel ranges to use; the others are left out
        of the averages. ``None``: all channels.
    """
    return obs.chanaver(nchan, channels=channels)


def uvaver(obs, aver_time, scatter=False) -> Observation:
    """Time-average an observation into `aver_time` integrations (difmap
    uvaver), returning a new one; `obs` is left as it is. The same as
    ``obs.uvaver(aver_time, doscatter=scatter)``.

    Parameters
    ----------
    obs : Observation
        The observation to average (its calibration is applied first).
    aver_time : float | str
        The new integration time: seconds, or a string with its unit
        ("10s", "2min").
    scatter : bool, default False
        Derive the weights from the scatter of the averaged samples
        instead of summing the input weights.
    """
    return obs.uvaver(aver_time, doscatter=scatter)


#: `load` under difmap's own name for it.
observe = load
