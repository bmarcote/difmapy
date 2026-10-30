"""High-level difmap-like observation API.

Mirrors the classic difmap workflow::

    obs = difmapy.load("data.uvf")     # observe
    obs.select("I")                    # select
    obs.mapsize(512, 0.2)              # mapsize (mas/pixel)
    obs.invert()                       # invert
    obs.add_window(-2, 2, -2, 2)       # clean window (mas)
    obs.clean(200, 0.05)               # clean
    obs.selfcal(phase=True)            # selfcal
    obs.restore()                      # restore (obs.restored_map)
    obs.wmap("clean.fits")             # wmap

All user-facing positions/sizes are in milliarcseconds (mas), angles in degrees and UV radii in wavelengths; everything is converted to radians internally.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from difmapy._core import CoreObservation
from difmapy.report import write_json
from difmapy.units import (format_flux, format_interval, format_uvdist, format_value,
                           parse_interval, parse_time, parse_uvdist)

MAS = np.pi / (180.0 * 3600.0 * 1000.0)  # mas -> radians
DEG = np.pi / 180.0

# Defaults used by `auto_mapsize()` (and hence by `mapplot()` when the
# map size has not been set by hand): a large field sampled with ten
# pixels across the diffraction limit lambda/B_max. The number of pixels
# no longer has to be a power of two - any multiple of four works - but
# powers of two are still the fastest to transform.
DEFAULT_NPIX = 4096
#: The longest UV radius `uvtaper` accepts, in wavelengths (100 G-lambda: beyond any baseline, space VLBI included). A larger one was given in wavelengths where M-lambda was meant.
MAX_UVDIST = 1e11
#: The upper UV limit handed to the gridding when only a lower one is set (it cuts nothing; the core needs one to apply `uvmin` at all).
NO_UVMAX = 1e30
DEFAULT_OVERSAMPLE = 10
#: The Briggs robustness values mapplot offers, and for which `save`
#: writes a restored map each.
ROBUST_CHOICES = (-2.0, -1.0, 0.0, 1.0, 2.0)

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


def _bold(text) -> str:
    """`text` in bold where the output can show it: a terminal or a Jupyter cell, but not a file or a pipe, where the escape codes would be so much noise."""
    out = sys.stdout
    tty = out.isatty() and os.environ.get("TERM") != "dumb"
    if tty or type(out).__module__.startswith("ipykernel"):
        return f"\033[1m{text}\033[0m"
    return text


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

class Observation:
    """An in-memory interferometric observation, with difmap's commands as its methods.

    Create one with `difmapy.load` (difmap's `observe`), `Observation.from_uvfits`, `Observation.from_ms` or `Observation.get`. Positions and sizes are in milliarcseconds (mas), angles in degrees, UV distances in wavelengths and times in seconds unless a method says otherwise; strings with units ("30s", "2min") are accepted wherever a time is.

    Commands that act on the data (``clean``, ``selfcal``, ``select``, ``flag``, ...) follow one convention so that the interactive prompt stays readable:

    - ``quiet=True`` (the default) returns None, so nothing is echoed under the report the command prints. Pass ``quiet=False`` to get the result back - a dict, a count, the observation itself - for use in a script.
    - ``verbose=True`` (the default, on the commands that print a report) prints that report; ``verbose=False`` prints nothing.

    Queries whose whole purpose is their value (``imstat``, ``mapinfo``, ``scans``, ``spectrum``, ...) always return it, and plot commands always return their window.
    """

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
        #: Self-cal solutions flagged by hand (`flag_solutions`, corplot):
        #: a boolean [ntimes, nif, nant] mask, and the gain table as it
        #: was before the first of them, which every edit is re-derived
        #: from. Both None until a solution is flagged, and again after
        #: any new calibration (which supersedes the edits).
        self._sol_flags = None
        self._sol_orig = None
        #: The FLAG column as the data came in (after any averaging), so
        #: that the flags added since can be told apart (`wflags`).
        self._flags_at_load = None
        #: Width in seconds of the time-averaging bins, if averaged.
        self._aver_time = None
        #: After channel averaging: the first and last *original* global
        #: channel behind each channel, and the original channel counts.
        self._chan_origin = None
        self._orig_if_nchan = None

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_uvfits(cls, path, wtscale=1.0, stokes="I", channels=None,
                    timeavg=None, freqavg=None, scatter=False,
                    average=None) -> "Observation":
        """Load a random-groups UVFITS file (single source).

        Parameters
        ----------
        path : str
            The UVFITS file.
        stokes : str | None, default "I"
            Polarization to select once loaded: "I", "Q", "U", "V", or a recorded one ("RR", "LL", "RL", "LR", "XX", ...). "I" uses difmap's permissive combination (a visibility survives when only one parallel hand is usable); Q/U/V are strict. ``None`` (or "none") loads without a selection, as difmap's `observe` does; `select()` changes it at any time.
        channels : list of (int, int) | None, default None
            Inclusive, 0-based channel ranges over the global channel axis (all IFs concatenated), e.g. ``[(0, 31), (64, 95)]``, as `select` takes them. With `freqavg`, only these channels go into the averages - the way band edges are dropped - and the selection then covers every averaged channel. ``None``: all channels.
        timeavg : float | str | None, default None
            Time-average straight after loading (difmap `uvaver`): seconds as a number, or a string with its unit ("10s", "2min"). ``None``: keep the original integrations.
        freqavg : int | "all" | bool | None, default None
            Average this many adjacent channels into one (`chanaver`); it must divide every IF's number of channels. ``"all"`` or ``True`` averages each IF down to a single channel. ``None``/``False``: no averaging.
        scatter : bool, default False
            With `timeavg`: derive the output weights from the scatter of the averaged samples instead of summing the input weights.
        wtscale : float, default 1.0
            Factor applied to the data weights as they are read.
        average : float | str | None, default None
            The older name of `timeavg`, still accepted.

        Returns
        -------
        Observation

        Notes
        -----
        The work is done in the cheapest order: channels are averaged first, then the Stokes selection is made, then the time averaging. Averaged data from a Measurement Set still go back to one: `save` writes a new MS with the averaged rows, `save_flags` flags every original sample behind a flagged averaged one, and `savecaltable` works as usual.
        """
        from difmapy.io.uvfits import load_uvfits

        return cls(load_uvfits(path, wtscale=wtscale))._on_load(
            stokes, channels, timeavg, freqavg, scatter, average)

    @classmethod
    def from_ms(cls, path, stokes="I", channels=None, timeavg=None,
                freqavg=None, scatter=False, average=None, wtscale=1.0,
                field=None, data_column="DATA") -> "Observation":
        """Load a CASA Measurement Set (single field; needs casatools).

        Parameters
        ----------
        path : str
            The Measurement Set directory.
        stokes : str | None, default "I"
            Polarization to select once loaded: "I", "Q", "U", "V", or a recorded one ("RR", "LL", "RL", "LR", "XX", ...). "I" uses difmap's permissive combination (a visibility survives when only one parallel hand is usable); Q/U/V are strict. ``None`` (or "none") loads without a selection, as difmap's `observe` does; `select()` changes it at any time.
        channels : list of (int, int) | None, default None
            Inclusive, 0-based channel ranges over the global channel axis (all IFs concatenated), e.g. ``[(0, 31), (64, 95)]``, as `select` takes them. With `freqavg`, only these channels go into the averages - the way band edges are dropped - and the selection then covers every averaged channel. ``None``: all channels.
        timeavg : float | str | None, default None
            Time-average straight after loading (difmap `uvaver`): seconds as a number, or a string with its unit ("10s", "2min"). ``None``: keep the original integrations.
        freqavg : int | "all" | bool | None, default None
            Average this many adjacent channels into one (`chanaver`); it must divide every IF's number of channels. ``"all"`` or ``True`` averages each IF down to a single channel. ``None``/``False``: no averaging.
        scatter : bool, default False
            With `timeavg`: derive the output weights from the scatter of the averaged samples instead of summing the input weights.
        wtscale : float, default 1.0
            Factor applied to the data weights as they are read.
        field : str | int | None, default None
            Measurement Sets only: the field (name or FIELD_ID) to load. Needed when the MS has several fields; difmapy, like difmap, is single-source.
        data_column : str, default "DATA"
            Measurement Sets only: the visibility column to read, "DATA" or "CORRECTED_DATA".
        average : float | str | None, default None
            The older name of `timeavg`, still accepted.

        Returns
        -------
        Observation

        Notes
        -----
        The work is done in the cheapest order: channels are averaged first, then the Stokes selection is made, then the time averaging. Averaged data from a Measurement Set still go back to one: `save` writes a new MS with the averaged rows, `save_flags` flags every original sample behind a flagged averaged one, and `savecaltable` works as usual. Autocorrelations are dropped, as difmap uses cross-correlations only.
        """
        from difmapy.io.ms import load_ms

        return cls(load_ms(path, field=field, data_column=data_column,
                           wtscale=wtscale))._on_load(
            stokes, channels, timeavg, freqavg, scatter, average)

    def _on_load(self, stokes, channels, timeavg, freqavg, scatter,
                 average=None):
        """The selection and averaging asked for at load time, in the order that costs least: channels are averaged first (only the requested `channels` go into the averages, so band edges can be dropped), then the Stokes selection, then the time averaging."""
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
        obs = obs._averaged(timeavg, scatter)
        obs._flags_at_load = np.array(obs.flags, copy=True)
        return obs

    def _averaged(self, average, scatter=False):
        """This observation time-averaged by `uvaver`, or itself when `average` is None/0."""
        if not average:
            return self
        return self.uvaver(average, doscatter=scatter)

    def _initial_select(self, stokes, channels=None):
        """Apply the selection asked for at load time (difmap `observe` selects nothing; difmapy defaults to total intensity, which is what almost every session starts with). `stokes=None` skips it, and `select()` can change it at any time."""
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
        """The name of the observed source, as recorded in the file."""
        return self._core.source_name

    @property
    def path(self):
        """The file the data were loaded from (UVFITS file or Measurement Set), or None for data built in memory. It survives averaging and `copy()`; for a session restored with `get()` it is the saved ``.uvf``."""
        origin = getattr(self._core, "_cal_origin", None) or {}
        return origin.get("path")

    @property
    def filename(self):
        """The name of the file the data were loaded from, without its directory (see `path`), or None."""
        path = self.path
        if not path:
            return None
        return os.path.basename(os.path.normpath(os.fspath(path)))

    @property
    def nif(self) -> int:
        """The number of IFs (spectral windows)."""
        return self._core.nif

    @property
    def nchan(self) -> list[int]:
        """The number of channels of each IF, in IF order."""
        return [n for (_, _, n) in self._core.ifs]

    @property
    def npol(self) -> int:
        """The number of recorded polarizations (correlations)."""
        return self._core.npol

    @property
    def antennas(self) -> list[str]:
        """The station names, in the order antenna indices refer to."""
        return self._core.antenna_names

    @property
    def pols(self) -> list[str]:
        """The recorded polarizations as names ("RR", "LL", ...), in the order they appear on the data's polarization axis."""
        try:
            return list(self._core.pol_names)
        except AttributeError:  # extension older than pol_names
            return [POL_NAMES.get(int(c), f"?({int(c)})") for c in self._core.pols]

    @property
    def pol_codes(self) -> list[int]:
        """The AIPS/FITS codes behind `pols` (-1 = RR, -2 = LL, ...)."""
        return [int(c) for c in self._core.pols]

    def header(self, quiet=True, verbose=True) -> str:
        """Print a summary of the observation (difmap `header`): source and position, rows, integrations and subarrays, the antennas, polarizations, IFs and the current selection.

        Parameters
        ----------
        quiet : bool, default True
            Return None. ``False`` returns the summary as a string.
        verbose : bool, default True
            Print the summary.

        Returns
        -------
        str | None
            The summary, one item per line, with ``quiet=False``.
        """
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
        text = "\n".join(lines)
        if verbose:
            print(text)
        return None if quiet else text

    def __repr__(self) -> str:  # pragma: no cover
        return f"<difmapy.Observation {self.source}: {self._core.nrow} rows, {self.nif} IFs>"

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def select(self, pol="I", channels=None, quiet=True):
        """Select the polarization and, optionally, the channels to work on (difmap `select`).

        Everything downstream - imaging, model fitting, self-calibration, the plots - sees only the selection; changing it discards the current map.

        Parameters
        ----------
        pol : str, default "I"
            Stokes "I", "Q", "U", "V", or a recorded correlation ("RR", "LL", "RL", "LR", "XX", ...). "I" uses difmap's permissive combination: a visibility survives when only one parallel hand is usable. Q, U and V are strict.
        channels : list of (int, int) | None, default None
            Inclusive, 0-based channel ranges over the global channel axis (all IFs concatenated), e.g. ``[(0, 31), (64, 95)]``. ``None`` selects every channel.
        quiet : bool, default True
            Return None. ``False`` returns the observation, so that calls can be chained.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._core.select(str(pol), [(int(a), int(b)) for a, b in (channels or [])])
        self._dirty()
        return self._done(self, quiet)

    # ------------------------------------------------------------------
    # imaging setup
    # ------------------------------------------------------------------

    def mapsize(self, nx, cell, ny=None, ycell=None, quiet=True):
        """Set the map dimensions and the pixel size (difmap `mapsize`).

        Parameters
        ----------
        nx : int
            Map width in pixels. It must be a multiple of four; powers of two are not required but are still the fastest to transform.
        cell : float
            Pixel width in mas.
        ny : int | None, default None
            Map height in pixels; ``None`` makes the map square (`nx`).
        ycell : float | None, default None
            Pixel height in mas; ``None`` uses `cell`.
        quiet : bool, default True
            Return None. ``False`` returns the observation, so that calls can be chained.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._nx = int(nx)
        self._ny = int(ny) if ny else int(nx)
        self._xinc = float(cell) * MAS
        self._yinc = float(ycell) * MAS if ycell else self._xinc
        self._mapsize_set = True
        self._dirty()
        return self._done(self, quiet)

    def estimated_resolution(self) -> float:
        """The diffraction-limited resolution, lambda / B_max, over the longest unflagged baseline of the current selection.

        Returns
        -------
        float
            The resolution in mas.

        Raises
        ------
        RuntimeError
            If there is no unflagged data.
        """
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

    def auto_mapsize(self, npix=DEFAULT_NPIX, oversample=DEFAULT_OVERSAMPLE, quiet=True):
        """Set a map size derived from the data: `npix` pixels of ``estimated_resolution() / oversample`` each.

        This is what `mapplot()` uses when `mapsize()` has not been called, so a first image needs no guessing.

        Parameters
        ----------
        npix : int, default 4096
            Map width and height in pixels (a multiple of four).
        oversample : float, default 10
            Pixels across the diffraction-limited resolution.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self.mapsize(int(npix), self.estimated_resolution() / float(oversample))
        return self._done(self, quiet)

    def _ensure_mapsize(self):
        """Pick a map size from the data if none was set explicitly."""
        if not self._mapsize_set:
            self.auto_mapsize()

    def uvweight(self, binwid=2.0, errpow=0.0, radial=False, robust=None, quiet=True):
        """Set the weighting of the visibilities on the UV grid (difmap `uvweight`).

        The defaults are difmap's: uniform weighting with a 2-pixel bin, ignoring the data weights. For natural weighting by the data weights use ``uvweight(0, -1)`` or ``uvweight(robust=2)``.

        Parameters
        ----------
        binwid : float, default 2.0
            Width of the uniform-weighting bins in UV pixels; 0 selects natural weighting (no density correction).
        errpow : float, default 0.0
            If negative, the weights are scaled by ``wt ** (-errpow / 2)``, weighting down noisy visibilities; 0 ignores the data weights.
        radial : bool, default False
            Also multiply the weights by the UV radius.
        robust : float | None, default None
            Briggs robustness, from -2 (uniform, sharpest beam) to +2 (natural, lowest noise). When given it supersedes `binwid` and `errpow`; ``None`` goes back to difmap's own scheme.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._binwid = float(binwid)
        self._errpow = float(errpow)
        self._dorad = bool(radial)
        self._robust = None if robust is None else float(np.clip(robust, -2.0, 2.0))
        self._dirty()
        return self._done(self, quiet)

    @property
    def robust(self):
        """The Briggs robustness in use (a float), or None while difmap's own `uvweight` scheme is."""
        return self._robust

    def uvtaper(self, value=0.0, radius=0.0, quiet=True):
        """Taper the gridding weights with a Gaussian in UV radius (difmap `uvtaper`), weighting down the long baselines: a visibility at UV radius ``r`` has its weight multiplied by ``value ** ((r / radius) ** 2)``, i.e. ``exp(ln(value) * r**2 / radius**2)``, which is 1 at the centre of the UV plane and falls to `value` at `radius`. The map gets a broader beam and more sensitivity to extended emission. The taper stays in effect for every later image until it is changed; ``uvtaper()`` (or a `value` of 0 or 1) turns it off.

        Parameters
        ----------
        value : float, default 0.0
            The taper at `radius`, between 0 and 1 (exclusive); 0 or 1 turns the taper off.
        radius : float | str, default 0.0
            The UV radius at which the taper falls to `value`: mega-wavelengths as a bare number, or a string with its unit - ``"50M"``, ``"500k"``, ``"5e7 lambda"``. 0 turns the taper off.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.

        Raises
        ------
        ValueError
            If `radius` is beyond 100 G-lambda - no baseline is that long, so it was almost certainly given in wavelengths, which a bare number no longer means.
        """
        value = float(value)
        rad = parse_uvdist(radius, "M")
        if rad > MAX_UVDIST:
            hint = ("" if isinstance(radius, str) else
                    f": for {float(radius):g} wavelengths write uvtaper({value:g}, {float(radius) / 1e6:g})")
            raise ValueError(f"uvtaper radius {rad:.3g} wavelengths is beyond any baseline; a bare number "
                             f"is in M-lambda{hint}")
        if not 0.0 <= value <= 1.0 or rad < 0.0:
            raise ValueError(f"uvtaper needs 0 <= value <= 1 and radius >= 0, got {value:g}, {radius!r}")
        self._gauval, self._gaurad = value, rad
        self._dirty()
        return self._done(self, quiet)

    def uvrange(self, uvmin=0.0, uvmax=0.0, quiet=True):
        """Restrict imaging to the visibilities between two UV radii (difmap `uvrange`); the rest is left out of the map and beam, but not flagged. It stays in effect for every later image until it is changed; ``uvrange()`` uses every visibility again.

        Parameters
        ----------
        uvmin : float | str, default 0.0
            The shortest UV radius imaged: mega-wavelengths as a bare number, or a string with its unit (``"50M"``, ``"500k"``, ``"5e7 lambda"``).
        uvmax : float | str | None, default 0.0
            The longest UV radius imaged, in the same units; 0 or None means no upper limit.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        lo = parse_uvdist(uvmin or 0.0, "M")
        hi = parse_uvdist(uvmax or 0.0, "M")
        if lo < 0 or hi < 0:
            raise ValueError(f"uvrange needs non-negative radii, got {uvmin!r}, {uvmax!r}")
        if hi and lo > hi:
            lo, hi = hi, lo
        self._uvmin, self._uvmax = lo, hi
        self._dirty()
        return self._done(self, quiet)

    def uvzero(self, flux=0.0, weight=0.0, quiet=True):
        """Add a zero-spacing flux estimate to the gridded data (difmap `uvzero`).

        Parameters
        ----------
        flux : float, default 0.0
            The total flux density of the source in Jy; 0 turns the zero spacing off.
        weight : float, default 0.0
            The weight given to that zero-spacing measurement.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._uvzero = (float(flux), float(weight))
        self._dirty()
        return self._done(self, quiet)

    def _report(self, data, outfile):
        """Return `data`, writing it to `outfile` as JSON if asked.

        Every command that reports numbers takes `outfile=`, so a run can keep its results without re-deriving the structure.
        """
        return data if outfile is None else write_json(outfile, data)

    def _done(self, value, quiet, outfile=None):
        """What a command hands back: `value` when `quiet` is False and None otherwise, so that a command typed at the prompt does not echo its result under the report it has printed. With `outfile` the value is written out as JSON either way."""
        value = self._report(value, outfile)
        return None if quiet else value

    def _dirty(self):
        self._invert_result = None
        self._restored = None

    # ------------------------------------------------------------------
    # invert / clean / restore
    # ------------------------------------------------------------------

    def invert(self, outfile=None, quiet=True, verbose=True):
        """Grid and Fourier-transform the selected visibilities into the dirty map and dirty beam (difmap `invert`).

        The map size, weighting, taper, UV range and zero spacing set so far are used. Any model is subtracted first, so after a `clean` this gives the residual map. Most commands invert by themselves when they need a map; calling it is only needed to see a fresh one.

        Parameters
        ----------
        outfile : str | None, default None
            Also write the result to this file as JSON.
        quiet : bool, default True
            Return None. ``False`` returns the result dict.
        verbose : bool, default True
            Print the number of visibilities gridded, the estimated beam and the expected noise.

        Returns
        -------
        dict | None
            With ``quiet=False``: the estimated beam (`e_bmaj`, `e_bmin` in radians, `e_bpa` in radians), the expected thermal `noise` in Jy/beam and the number of visibilities used (`nused`).
        """
        try:
            res = self._core.invert(
                self._nx,
                self._ny,
                self._xinc,
                self._yinc,
                uvmin=self._uvmin,
                uvmax=self._uvmax or (NO_UVMAX if self._uvmin else 0.0),
                gauval=self._gauval,
                gaurad=self._gaurad,
                dorad=self._dorad,
                errpow=self._errpow,
                binwid=self._binwid,
                robust=self._robust,
                uvzero_amp=self._uvzero[0],
                uvzero_wt=self._uvzero[1],
            )
        except RuntimeError as exc:
            # A taper far inside the shortest baselines leaves every
            # weight at zero (float32 underflow), which the core reports
            # as "no data": say what actually happened.
            if 0.0 < self._gauval < 1.0 and self._gaurad > 0.0 and "no data" in str(exc):
                raise RuntimeError(
                    f"{exc}: the taper ({self._gauval:g} at {format_uvdist(self._gaurad)} M-lambda) leaves "
                    "no weight on the data - use a larger radius (or value), or uvtaper() to turn it off"
                ) from None
            raise
        self._invert_result = dict(res)
        self._restored = None
        if verbose:
            r = self._invert_result
            print(f"invert: {r['nused']} visibilities gridded on {self._nx}x{self._ny} "
                  f"pixels of {self._xinc / MAS:.4g} mas; estimated beam "
                  f"{r['e_bmaj'] / MAS:.4g} x {r['e_bmin'] / MAS:.4g} mas at "
                  f"{r['e_bpa'] / DEG:.4g} deg, expected noise {r['noise']:.4g} Jy/beam")
        return self._done(self._invert_result, quiet, outfile)

    def _ensure_map(self):
        if self._invert_result is None:
            self.invert(verbose=False)

    @property
    def dmap(self) -> np.ndarray:
        """The residual (dirty) map, a float array of shape (ny, nx) in Jy/beam, inverting first if needed.

        Only the inner quarter (see `valid_slice`) is scientifically usable: the outer margin is amplified by the gridding correction and can even exceed the true peak. Use `imstat()` or `valid()` rather than scanning the whole array.
        """
        self._ensure_map()
        return self._core.map()

    @property
    def dbeam(self) -> np.ndarray:
        """The dirty beam, a float array of shape (ny, nx) normalised to a peak of 1, inverting first if needed; see `dmap` about the outer margin."""
        self._ensure_map()
        return self._core.beam()

    @property
    def restored_map(self) -> np.ndarray:
        """The restored map of the last `restore()`, a float array of shape (ny, nx) in Jy/beam.

        Raises
        ------
        RuntimeError
            If nothing has been restored since the map last changed.
        """
        if self._restored is None:
            raise RuntimeError("no restored map; call restore()")
        return self._restored

    @property
    def valid_slice(self):
        """The scientifically valid area of the map as a ``(rows, columns)`` tuple of slices: the inner quarter, which is what CLEAN searches and `imstat` measures.

        Outside it the gridding-correction factor grows without bound, so pixel values there are meaningless (difmap restricts its map area the same way).
        """
        return (
            slice(self._ny // 4, self._ny - self._ny // 4),
            slice(self._nx // 4, self._nx - self._nx // 4),
        )

    def valid(self, image=None) -> np.ndarray:
        """The valid inner quarter of an image (see `valid_slice`).

        Parameters
        ----------
        image : numpy.ndarray | None, default None
            A map of the current map size, e.g. `restored_map`; ``None`` uses the residual map `dmap`.

        Returns
        -------
        numpy.ndarray
            A view of the inner quarter of `image`.
        """
        if image is None:
            image = self.dmap
        return image[self.valid_slice]

    def peak_offset(self, image=None):
        """Where the brightest pixel of the valid map area is, and its value.

        Parameters
        ----------
        image : numpy.ndarray | None, default None
            A map of the current map size; ``None`` uses the residual map `dmap`.

        Returns
        -------
        tuple
            ``((east, north), value)``: the offset from the phase centre in mas and the pixel value in Jy/beam.
        """
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
        """The map extent in mas as ``(east_max, east_min, south, north)``, the order matplotlib's ``imshow(extent=...)`` takes for RA increasing to the left."""
        ex = self._nx / 2 * self._xinc / MAS
        ey = self._ny / 2 * self._yinc / MAS
        return (ex, -ex, -ey, ey)

    @property
    def estimated_beam(self):
        """The beam estimated from the UV coverage by `invert`, as ``(bmaj, bmin, bpa)``: the FWHM axes in mas and the position angle in degrees."""
        self._ensure_map()
        r = self._invert_result
        return (r["e_bmaj"] / MAS, r["e_bmin"] / MAS, r["e_bpa"] / DEG)

    def add_window(self, xmin, xmax, ymin, ymax, quiet=True):
        """Add a rectangular CLEAN window (difmap `addwin`). CLEAN only looks for components inside the windows; with none it searches the whole valid map area.

        Parameters
        ----------
        xmin, xmax : float
            East-west extent of the window in mas relative to the phase centre (east positive).
        ymin, ymax : float
            North-south extent of the window in mas relative to the phase centre (north positive).
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self.windows.append((float(xmin), float(xmax), float(ymin), float(ymax)))
        return self._done(self, quiet)

    def clear_windows(self, quiet=True):
        """Remove every CLEAN window (difmap `delwin`).

        Parameters
        ----------
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self.windows.clear()
        return self._done(self, quiet)

    def clean(self, niter=100, gain=0.05, cutoff=0.0, quiet=True, verbose=True, outfile=None):
        """Run Högbom CLEAN within the current windows (difmap `clean`), adding the components found to the model.

        Parameters
        ----------
        niter : int, default 100
            The number of CLEAN iterations. A negative value runs ``abs(niter)`` iterations but stops at the first negative component.
        gain : float, default 0.05
            The loop gain: the fraction of the peak residual taken out at each iteration.
        cutoff : float, default 0.0
            Stop when the peak residual falls below this flux density, in Jy/beam.
        quiet : bool, default True
            Return None. ``False`` returns the summary dict.
        verbose : bool, default True
            Print the summary.
        outfile : str | None, default None
            Also write the summary to this file as JSON.

        Returns
        -------
        dict | None
            With ``quiet=False``: the iterations done (`niter`), the components added (`ncomp`) and the flux they carry (`cleaned_flux`, Jy), the model afterwards (`total_ncomp`, `total_flux`), the residual statistics (`residual_rms`, `residual_peak`, Jy/beam), the `gain` and `cutoff` used, and whether the run stopped at the cutoff (`hit_cutoff`) or at a negative component (`hit_negative`).
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
        if verbose:
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
        return self._done(res, quiet, outfile)

    def clrmod(self, quiet=True):
        """Discard the model (difmap `clrmod`). `clearmodel()` does the same and can also drop the CLEAN windows.

        Parameters
        ----------
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._core.clear_models(True, True)
        self._dirty()
        return self._done(self, quiet)

    def clearmodel(self, windows=False, quiet=True):
        """Remove every model component - CLEAN components as well as hand-placed or fitted ones - so that the next `invert()` gives the dirty map again and imaging can start from scratch.

        Parameters
        ----------
        windows : bool, default False
            Also remove the CLEAN windows.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._core.clear_models(True, True)
        if windows:
            self.windows.clear()
        self._dirty()
        return self._done(self, quiet)

    def restore(self, bmaj=None, bmin=None, bpa=None, no_residual=False, do_smooth=False, quiet=True, verbose=True):
        """Convolve the model with a restoring beam and add the residual map (difmap `restore`). The result is kept in `restored_map`, which `wmap` writes and `mapplot("clean")` shows.

        Parameters
        ----------
        bmaj : float | None, default None
            FWHM of the restoring beam's major axis in mas; ``None`` uses the beam `invert` estimated.
        bmin : float | None, default None
            FWHM of the minor axis in mas; ``None`` uses the estimated beam.
        bpa : float | None, default None
            Position angle of the major axis in degrees, north through east; ``None`` uses the estimated beam.
        no_residual : bool, default False
            Leave the residuals out: the model convolved with the beam only.
        do_smooth : bool, default False
            Smooth the residual map with a 3x3 mask before adding it (difmap's `restore` smoothing flag).
        quiet : bool, default True
            Return None. ``False`` returns the restored map.
        verbose : bool, default True
            Print the restoring beam and the peak of the restored map.

        Returns
        -------
        numpy.ndarray | None
            With ``quiet=False``: the restored map, shape (ny, nx), in Jy/beam.
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
        if verbose:
            valid = self.valid(self._restored)
            print(f"restore: beam {bmaj / MAS:.4g} x {bmin / MAS:.4g} mas at "
                  f"{bpa / DEG:.4g} deg; peak {float(np.nanmax(valid)):.5g} Jy/beam, "
                  f"model {self.model_flux:.5g} Jy")
        return self._done(self._restored, quiet)

    def imstat(self, outfile=None):
        """Statistics of the valid inner quarter of the residual map (difmap `imstat`), inverting first if needed.

        Parameters
        ----------
        outfile : str | None, default None
            Also write the statistics to this file as JSON.

        Returns
        -------
        dict
            The `min`, `max`, `mean` and `rms` of the pixel values, in Jy/beam.
        """
        self._ensure_map()
        return self._report(dict(self._core.map_stats()), outfile)

    def noise_stats(self, image=None, nsigma=3.0, niter=5, outfile=None):
        """A robust noise estimate of a map: the mean and standard deviation of its pixels after iterative sigma clipping, which removes real emission.

        Parameters
        ----------
        image : numpy.ndarray | None, default None
            The pixels to measure; ``None`` uses the valid area of the residual map (`valid()`).
        nsigma : float, default 3.0
            Pixels further than this many standard deviations from the mean are clipped at each pass.
        niter : int, default 5
            The maximum number of clipping passes.
        outfile : str | None, default None
            Also write the result to this file as JSON.

        Returns
        -------
        dict
            `mean` and `rms` of the clipped (noise-only) pixels, `raw_rms` over all of them, the pixel count `npix` and how many the clipping rejected (`nclipped`). All in the image's units.
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
        """How well the model fits the data (a port of difmap's `moddif`, which is what its self-calibration reports before and after a solution). Only unflagged samples inside the UV range count.

        Parameters
        ----------
        uvmin : float, default 0.0
            The shortest UV radius included, in wavelengths.
        uvmax : float, default 0.0
            The longest UV radius included, in wavelengths; 0 means no limit.
        outfile : str | None, default None
            Also write the result to this file as JSON.

        Returns
        -------
        dict
            `rms`: the rms of the complex model-data difference in Jy; `sigma`: ``sqrt(chisq / ndata)``, which is 1 for a fit consistent with the data weights; the `chisq` itself; the number of measurements `ndata` (real and imaginary parts counted separately, as difmap does) and of visibilities `nvis`.
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
        """Everything worth knowing about the current image, as a dict: the map geometry and weighting, the beam, the peak, the model and the residual noise. `print_mapinfo()` prints it, as `mapplot`'s "x" key does.

        Parameters
        ----------
        outfile : str | None, default None
            Also write the result to this file as JSON.

        Returns
        -------
        dict
            `source`, `mapsize` (pixels), `cellsize` (mas), `uvweight`, `uvtaper` (`value`, `radius` in wavelengths), `uvrange` (`uvmin`, `uvmax` in wavelengths; 0 = no limit), `beam` (mas, degrees; `restored` says whether it is the restoring beam or the estimate), `estimated_beam`, `peak` (Jy/beam and its offset in mas), the model `flux` (Jy), `model` (its components), `residual` (rms, min, max in Jy/beam) and the accumulated `shift` (mas).
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
            "uvtaper": {"value": self._gauval, "radius": self._gaurad},
            "uvrange": {"uvmin": self._uvmin, "uvmax": self._uvmax},
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

    def print_mapinfo(self, info=None, quiet=True):
        """Print `mapinfo()` as a short human-readable report.

        Parameters
        ----------
        info : dict | None, default None
            A dict from an earlier `mapinfo()`; ``None`` measures the current image.
        quiet : bool, default True
            Return None. ``False`` returns the dict printed.

        Returns
        -------
        dict | None
            The `mapinfo()` dict, with ``quiet=False``.
        """
        d = self.mapinfo() if info is None else info
        b = d["beam"]
        print(f"--- {d['source']}: {d['mapsize'][0]}x{d['mapsize'][1]} pixels of "
              f"{d['cellsize'][0]:.4g} x {d['cellsize'][1]:.4g} mas")
        w = d["uvweight"]
        weighting = (f"robust {w['robust']:+g}" if w["robust"] is not None
                     else f"binwid {w['binwid']:g}, errpow {w['errpow']:g}")
        print(f"    weighting: {weighting}")
        t = d.get("uvtaper") or {}
        if 0.0 < t.get("value", 0.0) < 1.0 and t.get("radius", 0.0) > 0.0:
            print(f"    taper: {t['value']:g} at {format_uvdist(t['radius'])} M-lambda")
        r = d.get("uvrange") or {}
        if r.get("uvmin") or r.get("uvmax"):
            hi = f"{format_uvdist(r['uvmax'])} M-lambda" if r.get("uvmax") else "no limit"
            print(f"    uv range: {format_uvdist(r.get('uvmin', 0.0))} to {hi}")
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
        return self._done(d, quiet)

    # ------------------------------------------------------------------
    # model access
    # ------------------------------------------------------------------

    @property
    def model(self) -> list[dict]:
        """The whole model, as a list of dicts, one per component: `type` ("delta", "gauss", "disk", ...), `flux` (Jy), position `x` (east) and `y` (north) in mas, `major` axis (FWHM) in mas, axial `ratio`, position angle `phi` in degrees, reference frequency `freq0` (Hz) and spectral index `spcind`, and the `freepar` bitmask of the parameters `modelfit` may vary.

        What `addcmp`, `clean`, `rmodel` and `modelfit` produce is part of it at once, with no separate step to establish it.
        """
        old, new = self._core.get_models()
        masks = list(self._core.freepars)
        return [_comp_dict(c, m) for c, m in zip(list(old) + list(new), masks)]

    @property
    def model_flux(self) -> float:
        """The total flux density of the model, in Jy."""
        return float(sum(c["flux"] for c in self.model))

    @property
    def nvariable(self) -> int:
        """How many model components `modelfit` would vary: those with a free parameter. They keep it after a fit, so running `modelfit` again continues fitting them (as in difmap)."""
        return int(self._core.nvariable)

    def addcmp(self, flux, x, y, type="delta", major=0.0, ratio=1.0, phi=0.0, freq0=0.0, spcind=0.0, free=None, quiet=True):
        """Add a model component by hand (difmap `addcmp`).

        Parameters
        ----------
        flux : float
            Flux density in Jy.
        x, y : float
            Position east and north of the phase centre, in mas.
        type : str, default "delta"
            "delta" (a point source), "gauss", "disk", "ellipse", "ring", "rect" or "sz".
        major : float, default 0.0
            FWHM of the major axis in mas (0 for a point).
        ratio : float, default 1.0
            Axial ratio, minor over major (1 for a circular component).
        phi : float, default 0.0
            Position angle of the major axis in degrees, north through east.
        freq0 : float, default 0.0
            Reference frequency of the spectral index, in Hz; 0 for none.
        spcind : float, default 0.0
            Spectral index at `freq0`.
        free : str | list of str | int | None, default None
            Parameters `modelfit` may vary: any of "flux", "pos" (x and y), "major", "ratio", "phi", "spcind", or "shape" for the last three, e.g. ``free=["flux", "pos"]``; or difmap's bitmask. ``None`` keeps the component fixed.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
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
        return self._done(self, quiet)

    def _replace_model(self, comps):
        """Replace the whole model with `comps` (dicts as `model` returns them), keeping their free-parameter masks."""
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

    def seed_model(self, type="gauss", free=("flux", "pos", "major"), quiet=True):
        """Add one component at the brightest point of the residual map, as a starting guess for `modelfit`. It carries the peak flux and has zero width: a point source that the fit is free to resolve.

        Parameters
        ----------
        type : str, default "gauss"
            The component type, as for `addcmp`.
        free : sequence of str, default ("flux", "pos", "major")
            The parameters `modelfit` may vary, as for `addcmp`.
        quiet : bool, default True
            Return None. ``False`` returns the component.

        Returns
        -------
        dict | None
            The seeded component, as `model` lists it, with ``quiet=False``.
        """
        self._ensure_map()
        (x, y), peak = self.peak_offset()
        self.addcmp(float(peak), x, y, type=type, major=0.0, ratio=1.0,
                    free=list(free))
        return self._done(self.model[-1], quiet)

    def modelfit(self, niter=-1, free=None, uvmin=0.0, uvmax=0.0, quiet=True, verbose=True, outfile=None):
        """Fit the model to the visibilities (difmap `modelfit`) by Levenberg-Marquardt least squares on the real and imaginary residuals. The fitted values replace the old ones in the model; there is nothing to keep afterwards.

        Which parameters vary is set per component by `addcmp(free=...)`, and persists, so running modelfit again continues the fit. As in difmap (`obvarmod`), the visibilities of the fixed components - CLEAN components above all - are subtracted first, so the fit works on the residuals after them. With no model at all, one is seeded by `seed_model()`: a circular Gaussian of zero width at the peak of the residual map.

        Parameters
        ----------
        niter : int, default -1
            The number of Levenberg-Marquardt iterations; negative iterates until the fit converges (capped at a few hundred).
        free : list | str | None, default None
            Overrides which parameters vary. A list with one spec per component (in `model` order) sets each component's; a single spec ("flux", ["flux", "pos"], "shape", ...) sets it on the components that already have a free parameter - or on every component, if none has - and leaves the fixed ones fixed. ``None`` keeps what the components have.
        uvmin : float, default 0.0
            Fit only visibilities at UV radii of at least this, in wavelengths.
        uvmax : float, default 0.0
            Fit only visibilities at UV radii up to this, in wavelengths; 0 means no limit.
        quiet : bool, default True
            Return None. ``False`` returns the result dict.
        verbose : bool, default True
            Print the fit summary and every fitted component with its uncertainties.
        outfile : str | None, default None
            Also write the result to this file as JSON.

        Returns
        -------
        dict | None
            With ``quiet=False``: `rchisq`, `chisq`, `ndfree`, `nvis` and `nfree` of the fit, whether it `converged` and after how many iterations (`niter`), the fitted `components` (as `model` lists them), their 1-sigma `errors` (same units), the number fitted (`ncomp`) and the whole model's `total_ncomp` and `total_flux` (Jy).

        Raises
        ------
        ValueError
            If no component has a free parameter.
        """
        if not self.model:
            seeded = self.seed_model(quiet=False)
            if verbose:
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
        if verbose:
            self._print_modelfit(res)
        return self._done(res, quiet, outfile)

    @staticmethod
    def _print_modelfit(res):
        end = "converged" if res.get("converged") else "stopped"
        print(
            f"modelfit: {end} after {res.get('niter', '?')} iterations; "
            f"reduced chi-squared {res['rchisq']:.5g} "
            f"({res['nvis']} visibilities, {res['nfree']} free parameters)"
        )
        print(f"  fitted {res['ncomp']} of {res['total_ncomp']} components; "
              f"{format_flux(res['total_flux'])} in the model")
        for i, (c, e) in enumerate(zip(res["components"], res["errors"])):
            # A parameter the fit did not vary has no uncertainty, and
            # says so rather than passing for an exact measurement.
            def fixed(name):
                return "" if c["freepar"] & FREE_BITS[name] else " (fixed)"

            if c["type"] == "delta":
                size = "point source"
            else:
                size = format_value(c["major"], e["major"], "mas")
                if c["type"] == "gauss":
                    size += " FWHM"
                if c["ratio"] < 1.0 or c["freepar"] & FREE_BITS["ratio"]:
                    size += (
                        f", axial ratio {format_value(c['ratio'], e['ratio'])}"
                        f", PA {format_value(c['phi'], e['phi'], 'deg')}"
                    )
                size += fixed("major")
            lines = [
                ("Flux density", format_flux(c["flux"], e["flux"]) + fixed("flux")),
                ("Centroid position",
                 f"{format_value(c['x'], e['x'], 'mas')} East, "
                 f"{format_value(c['y'], e['y'], 'mas')} North" + fixed("pos")),
                ("Size", size),
            ]
            if c["freq0"] or c["freepar"] & FREE_BITS["spcind"]:
                lines.append((
                    "Spectral index",
                    format_value(c["spcind"], e["spcind"])
                    + f" (at {c['freq0'] / 1e9:.6g} GHz)" + fixed("spcind"),
                ))
            # i indexes res["components"] and res["errors"].
            print("\n" + _bold(f"Component {i} ({c['type']}):"))
            for name, text in lines:
                print(f"  {name + ':':<18} {text}")

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def scans(self, gap=None, outfile=None) -> list[dict]:
        """The scans of the observation. Two integrations more than `gap` apart are in different scans - difmap's definition. This is what `selfcal(solint="scan")` bins by, so it is the thing to check when a per-scan solution looks wrong.

        Parameters
        ----------
        gap : float | str | None, default None
            The gap that separates scans: seconds, or a string with a unit ("5min"). ``None`` uses `default_scangap`, five times the median integration spacing (rather than difmap's flat hour, which is meant for breaking plot axes).
        outfile : str | None, default None
            Also write the list to this file as JSON.

        Returns
        -------
        list of dict
            One dict per scan: the integration index range (`first`, `last`), the time range (`tmin`, `tmax`, seconds) and the number of integrations (`nint`).
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
        """The gap in seconds that `scans()` and `selfcal(solint="scan")` use by default to separate scans."""
        return float(self._core.default_scangap)

    def selfcal(self, amp=False, phase=True, float_scale=False, solint=0.0, gauval=0.0, gaurad=0.0, maxamp=0.0, maxphs=0.0, uvmin=0.0, uvmax=0.0, mintel=0, flag=False, quiet=True, verbose=True, mapstats=None, scangap=None, outfile=None, converge=False, converge_threshold=0.01, converge_maxiter=20):
        """Self-calibrate the antenna gains against the current model (difmap `selfcal`). Each IF of each subarray is solved independently, and the solutions are applied to the data the rest of the session sees.

        Parameters
        ----------
        amp : bool, default False
            Solve for amplitudes.
        phase : bool, default True
            Solve for phases.
        float_scale : bool, default False
            With `amp`, let the overall amplitude scale float onto the model's, instead of keeping the data's.
        solint : float | str, default 0.0
            The solution interval: a bare number in minutes (difmap's unit; 0 = one solution per integration), or a string with its own unit - ``"30s"``, ``"90 sec"``, ``"1min"``, ``"2h"``. ``"scan"`` solves each scan as one interval, so every scan gets exactly one solution however long it is; ``"2scan"`` (or ``"3 scans"``) groups that many scans per solution. A finite interval is smoothed onto the integrations as in difmap.
        gauval : float, default 0.0
            Value of a Gaussian taper on the solution weights at radius `gaurad`; 0 for none.
        gaurad : float, default 0.0
            UV radius of that taper, in wavelengths.
        maxamp : float, default 0.0
            With `amp`, reject amplitude corrections outside ``[1/maxamp, maxamp]``; values up to 1 set no limit.
        maxphs : float, default 0.0
            Reject phase corrections larger than this, in degrees; 0 for no limit.
        uvmin : float, default 0.0
            Use only baselines at UV radii of at least this, in wavelengths.
        uvmax : float, default 0.0
            Use only baselines at UV radii up to this, in wavelengths; 0 means no limit.
        mintel : int, default 0
            The fewest telescopes a solution needs; 0 sets no minimum.
        flag : bool, default False
            Flag the data of a solution that fails, instead of leaving it uncorrected.
        quiet : bool, default True
            Return None. ``False`` returns the result dict.
        verbose : bool, default True
            Print the solution summary and the fit before and after.
        mapstats : bool | None, default None
            Also measure the residual map before and after (`map_before`/`map_after`). ``None`` does so whenever a map already exists, which costs nothing in an imaging loop: self-calibration invalidates the map and the next `clean` would re-invert anyway.
        scangap : float | str | None, default None
            The gap that separates scans for ``solint="scan"``: seconds or a string with a unit. ``None``: five times the median integration spacing (see `scans()`).
        outfile : str | None, default None
            Also write the result to this file as JSON.
        converge : bool, default False
            Alternate self-calibration with `modelfit()` (selfcal, modelfit, selfcal, ...; every selfcal with these parameters) until the model-data fit stops improving. The model needs free parameters (`addcmp(free=...)`, or an earlier `modelfit`).
        converge_threshold : float, default 0.01
            With `converge`: stop when the relative change of the fit's `sigma` (`moddif()`) between one selfcal and the next is below this (0.01 is 1%). sigma rather than rms, so that data flagged on the way does not count as a change.
        converge_maxiter : int, default 20
            With `converge`: the most self-calibrations run (at least two always are).

        Returns
        -------
        dict | None
            With ``quiet=False``: the number of solution bins (`nbins`, over every subarray and IF), unusable solutions (`nbadsol`) and bad telescope solutions (`nbadtel`), the fit before and after (`fit_before`, `fit_after`, as `moddif()` gives them) and, with `mapstats`, `map_before`/`map_after` (as `imstat()`). With `converge` also `converged`, `niter`, the final `fit_change`, the last `modelfit` result and a per-iteration `history` that records how much each selfcal moved the gains (`gain_change`, the median of ``|g_new / g_old - 1|``, and its maximum).
        """
        if converge:
            return self._selfcal_converge(
                dict(amp=amp, phase=phase, float_scale=float_scale,
                     solint=solint, gauval=gauval, gaurad=gaurad,
                     maxamp=maxamp, maxphs=maxphs, uvmin=uvmin, uvmax=uvmax,
                     mintel=mintel, flag=flag, scangap=scangap),
                threshold=converge_threshold, maxiter=converge_maxiter,
                quiet=quiet, verbose=verbose, mapstats=mapstats, outfile=outfile,
            )
        secs, nscan = parse_interval(solint, "min")
        gap = 0.0 if scangap is None else parse_time(scangap, "s")
        if mapstats is None:
            mapstats = self._invert_result is not None
        before = self.moddif(uvmin, uvmax)
        map_before = self.imstat() if mapstats else None
        self._forget_solution_flags()
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
        if verbose:
            self._print_selfcal(res, amp=amp, phase=phase,
                                every=format_interval(secs, nscan))
        return self._done(res, quiet, outfile)

    def _selfcal_converge(self, pars, threshold, maxiter, quiet, verbose,
                          mapstats, outfile):
        """selfcal(converge=True): alternate selfcal and modelfit until the gains settle."""
        if not self.nvariable:
            raise ValueError(
                "selfcal(converge=True) alternates with modelfit(), so the "
                "model needs free parameters: addcmp(free=...) or modelfit() "
                "first"
            )
        if not threshold > 0:
            raise ValueError(f"converge_threshold must be > 0, not {threshold}")
        if int(maxiter) < 2:
            raise ValueError(f"converge_maxiter must be >= 2, not {maxiter}")
        secs, nscan = parse_interval(pars["solint"], "min")
        if mapstats is None:
            mapstats = self._invert_result is not None
        map_before = self.imstat() if mapstats else None
        if verbose:
            print(f"selfcal: alternating with modelfit until the fit changes "
                  f"by less than {100 * threshold:.3g}%")
        history, first, fit, prev = [], None, None, None
        change = float("nan")
        for it in range(1, int(maxiter) + 1):
            if it > 1:
                fit = self.modelfit(quiet=False, verbose=False)
            g_old, _ = self._gain_complex()
            res = self.selfcal(**pars, quiet=False, verbose=False, mapstats=False)
            first = first or res
            gain, worst = self._gain_change(g_old)
            sigma = res["fit_after"]["sigma"]
            if prev is not None:
                change = abs(sigma - prev) / prev if prev > 0 else 0.0
            prev = sigma
            history.append({
                "iteration": it, "fit_change": None if it == 1 else change,
                "gain_change": gain, "max_gain_change": worst,
                "rchisq": None if fit is None else fit["rchisq"],
                "fit_after": res["fit_after"],
            })
            if verbose:
                moved = "" if it == 1 else f" ({100 * change:.3g}% change)"
                print(f"  iteration {it}: fit rms {res['fit_after']['rms']:.5g} "
                      f"Jy, sigma {sigma:.5g}{moved}; gains moved "
                      f"{100 * gain:.3g}% (median; max {100 * worst:.3g}%)")
            if it > 1 and change < threshold:
                break
        res["fit_before"] = first["fit_before"]
        if mapstats:
            res["map_before"] = map_before
            res["map_after"] = self.imstat()
        res.update(converged=bool(change < threshold), niter=it,
                   fit_change=change, modelfit=fit, history=history)
        if verbose:
            end = "converged" if res["converged"] else "did not converge"
            print(f"selfcal: {end} after {it} self-calibrations")
            self._print_selfcal(res, amp=pars["amp"], phase=pars["phase"],
                                every=format_interval(secs, nscan))
        return self._done(res, quiet, outfile)

    def _gain_complex(self):
        """The complex gain table [ntimes, nif, nant], 1 where unsolved, and its "solved" mask."""
        amp, phs, _ = self._core.gains()
        shape = (self._core.ntimes, self.nif, len(self.antennas))
        used = np.asarray(self._core.gains_used()).reshape(shape)
        g = (np.asarray(amp, dtype=np.float64).reshape(shape)
             * np.exp(1j * np.asarray(phs, dtype=np.float64).reshape(shape)))
        return np.where(used, g, 1.0), used

    def _gain_change(self, g_old):
        """Median and maximum of ``|g_new / g_old - 1|`` over the solved gains; NaN if nothing is solved."""
        g_new, used = self._gain_complex()
        ok = used & (np.abs(g_old) > 0) & np.isfinite(g_new) & np.isfinite(g_old)
        if not ok.any():
            return float("nan"), float("nan")
        d = np.abs(g_new[ok] / g_old[ok] - 1.0)
        return float(np.median(d)), float(d.max())

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
        """The model-data fit, and the residual map, before and after a calibration step."""
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

    def gscale(self, float_scale=False, quiet=True, verbose=True, mapstats=None, outfile=None):
        """Solve for one overall amplitude correction per telescope and IF over the whole observation, and apply it (difmap `gscale`). Like `selfcal`, this changes the data the rest of the session sees.

        With ``float_scale=False`` (the default, as in difmap) the gains are renormalised to preserve the data's flux scale rather than pull it onto the model, so the corrections can be large while the map peak barely moves; the fit statistics are what show the improvement.

        Parameters
        ----------
        float_scale : bool, default False
            Let the overall flux scale float onto the model's.
        quiet : bool, default True
            Return None. ``False`` returns the result dict.
        verbose : bool, default True
            Print each station's correction and the fit before and after.
        mapstats : bool | None, default None
            Also measure the residual map before and after; ``None`` does so whenever a map already exists.
        outfile : str | None, default None
            Also write the result to this file as JSON.

        Returns
        -------
        dict | None
            With ``quiet=False``: `gains`, station name -> the correction this call applied (median over IFs), `gains_per_if`, station -> one value per IF, the per-(subarray, IF) renormalisation factors `norms` (empty with `float_scale`), `nbins`, `nbadtel`, and `fit_before`/`fit_after` (and `map_before`/`map_after`) as for `selfcal`.
        """
        if mapstats is None:
            mapstats = self._invert_result is not None
        fit_before = self.moddif()
        map_before = self.imstat() if mapstats else None
        # An antenna with no prior solution starts from unity, so that
        # the first gscale reports its own corrections rather than NaN.
        before, _ = self._gain_amps(fill=1.0)
        self._forget_solution_flags()
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
        if verbose:
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
        return self._done(res, quiet, outfile)

    def bayes_gscale(self, models=("gauss1", "gauss2", "gauss3"), prior_sigma=0.10, outlier_sigma=1.0, jackknife=True, per_if=True, float_scale=False, norm="median", nloop=2, solint=0.0, clean_gain=0.05, sigma_floor=0.01, model_sigma=None, timeavg=None, workers=None, apply=True, prefix=None, outformat=None, ms=None, uvfits=None, plot=None, quiet=True, verbose=True):
        """Bayesian amplitude calibration of every station: `gscale` with each station left out of the source model in turn, several source models weighed against each other, and the result shrunk towards the a-priori calibration. A station within its a-priori accuracy (`prior_sigma`) is left alone; one clearly off it gets its correction almost in full (`outlier_sigma`). The flux scale of each IF is kept by its median station, not by the mean gain as `gscale` does, which one station far off its calibration drags. See `difmapy.bayescal` for the method.

        Parameters
        ----------
        models : sequence of str, default ("gauss1", "gauss2", "gauss3")
            The source-model families to compare: ``"gaussN"``/``"pointN"`` (N circular Gaussians or point sources, seeded at the residual peak and fitted), ``"clean"`` (``"clean500"`` for 500 iterations a round) and ``"current"`` (the model in the observation now). CLEAN is not in the default: the comparison often favoured it where it should not have.
        prior_sigma : float, default 0.10
            The fractional accuracy of the a-priori amplitude calibration.
        outlier_sigma : float, default 1.0
            The width of the log-amplitude errors of a station that is off (1.0: a factor of a few).
        jackknife : bool, default True
            Do the leave-one-station-out runs.
        per_if : bool, default True
            One correction per station and IF; ``False`` combines the IFs into one per station.
        float_scale : bool, default False
            Let the flux scale float onto the model's instead of keeping the data's.
        norm : str, default "median"
            How the flux scale is kept without `float_scale`: "median" (the median station of each IF gets no correction) or "mean" (difmap's mean gain of one).
        nloop : int, default 2
            Rounds of model building and phase self-calibration per run.
        solint : float | str, default 0.0
            Phase self-calibration interval, as for `selfcal`.
        clean_gain : float, default 0.05
            CLEAN loop gain for the ``"clean"`` models.
        sigma_floor : float, default 0.01
            The smallest uncertainty assumed for a log-gain.
        model_sigma : dict | None, default None
            Station -> extra uncertainty of its log-gain coming from the source model (one value, or one per IF).
        timeavg : float | str | None, default None
            Average the working data in time first, as `uvaver`.
        workers : int | None, default None
            Threads to run on; ``None`` uses every CPU.
        apply : bool, default True
            Apply the corrections to this observation, as `gscale` would.
        prefix : str | None, default None
            Write ``<prefix>.json``, ``<prefix>.png`` and a calibration table of the corrections.
        outformat : str | None, default None
            Format of that table, as for `savecaltable`.
        ms : str | None, default None
            Measurement Set a CASA table refers to, as for `savecaltable`.
        uvfits : str | None, default None
            UVFITS file an AIPS table is numbered for, as for `savecaltable`.
        plot : bool | None, default None
            Show the diagnostic figure; ``None`` shows it when nothing is written and `verbose` is on.
        quiet : bool, default True
            Return None. ``False`` returns the `BayesGainResult`.
        verbose : bool, default True
            Print the progress and the summary.

        Returns
        -------
        BayesGainResult | None
            With ``quiet=False``: the result; print it for the summary, ``.plot()`` for the diagnostics.
        """
        from difmapy.bayescal import bayes_gscale

        return bayes_gscale(
            self, models=models, prior_sigma=prior_sigma, outlier_sigma=outlier_sigma, jackknife=jackknife,
            per_if=per_if, float_scale=float_scale, norm=norm, nloop=nloop, solint=solint,
            clean_gain=clean_gain, sigma_floor=sigma_floor, model_sigma=model_sigma, timeavg=timeavg,
            workers=workers, apply=apply, prefix=prefix, outformat=outformat,
            ms=ms, uvfits=uvfits, plot=plot, quiet=quiet, verbose=verbose)

    def _gain_amps(self, fill=np.nan):
        """The gain-table amplitudes and their "solved" mask, both [ntimes, nif, nant]; `fill` replaces unsolved entries."""
        amp, _, _ = self._core.gains()
        nt, nif, nant = self._core.ntimes, self.nif, len(self.antennas)
        used = np.asarray(self._core.gains_used()).reshape(nt, nif, nant)
        amp = np.asarray(amp, dtype=np.float64).reshape(nt, nif, nant)
        return np.where(used, amp, fill), used

    def station_gains(self, per_if=False, outfile=None):
        """The accumulated amplitude gain corrections per station: the factors difmapy multiplies the data by (`savecaltable` writes their reciprocal for CASA). A station with no solution reports 1.0, since its data is passed through unchanged.

        Parameters
        ----------
        per_if : bool, default False
            Give one value per IF instead of their median.
        outfile : str | None, default None
            Also write the result to this file as JSON.

        Returns
        -------
        dict
            Station name -> its correction (a float), or a list with one per IF with `per_if`.
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

    # ------------------------------------------------------------------
    # editing self-calibration solutions
    # ------------------------------------------------------------------

    def _forget_solution_flags(self):
        """A new calibration replaces the table the solution edits were made on: keep the corrections as they now are, and forget the edits."""
        self._sol_flags = None
        self._sol_orig = None

    def _gain_shape(self):
        return (self._core.ntimes, self.nif, len(self.antennas))

    @property
    def solution_flags(self) -> np.ndarray:
        """Which self-calibration solutions are flagged by hand, as a boolean array [ntimes, nif, nant] (all False when none are). See `flag_solutions`."""
        if self._sol_flags is None:
            return np.zeros(self._gain_shape(), dtype=bool)
        return self._sol_flags.copy()

    def original_solutions(self):
        """The gain table as it was before any solution was flagged: ``(amp, phs, bad)``, each [ntimes, nif, nant] (phases in radians). This is the table itself while nothing is flagged."""
        if self._sol_orig is not None:
            return tuple(a.copy() for a in self._sol_orig)
        amp, phs, bad = (np.asarray(x) for x in self._core.gains())
        shape = self._gain_shape()
        return (amp.reshape(shape).astype(np.float64), phs.reshape(shape).astype(np.float64),
                bad.reshape(shape).astype(bool))

    def _solution_mask(self, station=None, tmin=None, tmax=None, if_index=None):
        """The [ntimes, nif, nant] entries a station/time/IF selection covers (None: all)."""
        mask = np.ones(self._gain_shape(), dtype=bool)
        if station is not None:
            keep = np.zeros(mask.shape[2], dtype=bool)
            for name in ([station] if isinstance(station, (str, int)) else station):
                keep[name if isinstance(name, int) else self._ant_index(name)] = True
            mask &= keep[None, None, :]
        times = np.asarray(self._core.times())
        lo, hi = parse_time(tmin, "s"), parse_time(tmax, "s")
        if lo is not None:
            mask &= (times >= lo)[:, None, None]
        if hi is not None:
            mask &= (times <= hi)[:, None, None]
        if if_index is not None:
            keep = np.zeros(mask.shape[1], dtype=bool)
            keep[np.atleast_1d(if_index)] = True
            mask &= keep[None, :, None]
        return mask

    def set_solution_flags(self, flags):
        """Make `flags` (boolean [ntimes, nif, nant]) the set of hand-flagged solutions and re-calibrate. Only solved entries can be flagged; the rest of `flags` is ignored. This is what `flag_solutions`, `unflag_solutions` and corplot's editing (and its undo) come down to.

        Every flagged correction is re-derived from the solutions that remain for its antenna and IF: the amplitude interpolated linearly in time, the phase through the unit phasor (so it cannot wrap the long way round), and held at the nearest solution beyond the first or last. An antenna and IF left with no solution at all has its flagged entries marked bad, which flags the data behind them. The raw data are never modified, so the calibration is simply re-applied to them.

        Parameters
        ----------
        flags : numpy.ndarray
            Boolean, of shape [ntimes, nif, nant].

        Returns
        -------
        int
            The number of solutions flagged afterwards.
        """
        shape = self._gain_shape()
        flags = np.asarray(flags, dtype=bool)
        if flags.shape != shape:
            raise ValueError(f"solution flags must have shape {shape} (ntimes, nif, nant), not {flags.shape}")
        used = np.asarray(self._core.gains_used()).reshape(shape)
        flags = flags & used
        amp0, phs0, bad0 = self.original_solutions()
        if self._sol_orig is None:
            self._sol_orig = (amp0, phs0, bad0)
        amp, phs, bad = amp0.copy(), phs0.copy(), bad0.copy()
        t = np.asarray(self._core.times(), dtype=np.float64)
        for cif, ia in zip(*np.nonzero(flags.any(axis=0))):
            f = flags[:, cif, ia]
            good = used[:, cif, ia] & ~f & ~bad0[:, cif, ia]
            if not good.any():
                bad[f, cif, ia] = True
                continue
            amp[f, cif, ia] = np.interp(t[f], t[good], amp0[good, cif, ia])
            re = np.interp(t[f], t[good], np.cos(phs0[good, cif, ia]))
            im = np.interp(t[f], t[good], np.sin(phs0[good, cif, ia]))
            phs[f, cif, ia] = np.arctan2(im, re)
        self._core.set_gains(np.ascontiguousarray(amp.ravel(), np.float32),
                             np.ascontiguousarray(phs.ravel(), np.float32),
                             np.ascontiguousarray(bad.ravel(), bool))
        if flags.any():
            self._sol_flags = flags
        else:
            # Nothing flagged any more: the table is the original again.
            self._forget_solution_flags()
        self._dirty()
        return int(flags.sum())

    def flag_solutions(self, station=None, tmin=None, tmax=None, if_index=None, quiet=True):
        """Flag self-calibration solutions and re-calibrate from the remaining ones (the editing of difmap's `corplot`, which `corplot()` does with the mouse).

        A flagged correction is not used: it is re-derived by interpolating, in time, the solutions that remain for the same antenna and IF (see `set_solution_flags`); where none remain, the data behind it are flagged instead. `unflag_solutions` puts the original corrections back. The raw data are never modified, for Measurement Sets and UVFITS files alike, so the calibration is re-applied to them each time. A new `selfcal`, `gscale`, `bayes_gscale` or `uncalib` works on the corrections as they then are, and the edits can no longer be undone.

        Parameters
        ----------
        station : str | int | list | None, default None
            The antenna(s) whose solutions to flag; ``None``: every antenna.
        tmin : float | str | None, default None
            Start of the time range: seconds since the reference day, or a string with a unit ("1.5h"); ``None``: from the start.
        tmax : float | str | None, default None
            End of the time range; ``None``: to the end.
        if_index : int | list of int | None, default None
            Only these IFs (0-based); ``None``: every IF.
        quiet : bool, default True
            Return None. ``False`` returns the number of solutions newly flagged.

        Returns
        -------
        int | None
            The number of solutions newly flagged, with ``quiet=False``.
        """
        before = self.solution_flags
        used = np.asarray(self._core.gains_used()).reshape(before.shape)
        new = self._solution_mask(station, tmin, tmax, if_index) & used & ~before
        self.set_solution_flags(before | new)
        return self._done(int(new.sum()), quiet)

    def unflag_solutions(self, station=None, tmin=None, tmax=None, if_index=None, quiet=True):
        """Put back self-calibration solutions flagged with `flag_solutions` (or in corplot), and re-calibrate.

        Parameters
        ----------
        station : str | int | list | None, default None
            The antenna(s) whose solutions to restore; ``None``: every antenna.
        tmin : float | str | None, default None
            Start of the time range: seconds, or a string with a unit; ``None``: from the start.
        tmax : float | str | None, default None
            End of the time range; ``None``: to the end.
        if_index : int | list of int | None, default None
            Only these IFs (0-based); ``None``: every IF.
        quiet : bool, default True
            Return None. ``False`` returns the number of solutions restored.

        Returns
        -------
        int | None
            The number of solutions restored, with ``quiet=False``.
        """
        before = self.solution_flags
        back = self._solution_mask(station, tmin, tmax, if_index) & before
        if back.any():
            self.set_solution_flags(before & ~back)
        return self._done(int(back.sum()), quiet)

    def uncalib(self, amp=True, phase=True, flags=False, quiet=True):
        """Undo the self-calibration corrections (difmap `uncalib`): a pure reset of the gain table.

        Parameters
        ----------
        amp : bool, default True
            Reset the amplitude corrections.
        phase : bool, default True
            Reset the phase corrections.
        flags : bool, default False
            Also clear the flags self-calibration set on failed solutions.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._forget_solution_flags()
        self._core.uncalib(amp, phase, flags)
        self._dirty()
        return self._done(self, quiet)

    #: Names that `selfant` takes to mean "every antenna in the array".
    ALL_ANTENNAS = ("all", "*")

    def selfant(self, name=None, fix=None, weight=None, quiet=True, verbose=True, outfile=None) -> list[dict]:
        """Constrain antennas in self-calibration (difmap `selfant`). A fixed antenna keeps its gain of 1 while the others solve against it, and `weight` scales how much its baselines count in the solution. With no `name`, nothing is changed and the current constraints are reported - the way to check what a session has accumulated.

        Parameters
        ----------
        name : str | list of str | None, default None
            An antenna name, ``"all"`` or ``"*"`` for every antenna, or a list of names; ``None`` only reports.
        fix : bool | None, default None
            Hold the antennas' gains fixed (True) or let them solve (False); ``None`` keeps what they have.
        weight : float | None, default None
            Weight of the antennas' baselines in the solution; ``None`` keeps what they have.
        quiet : bool, default True
            Return None. ``False`` returns the constraints.
        verbose : bool, default True
            Print the constraints as a table.
        outfile : str | None, default None
            Also write the constraints to this file as JSON.

        Returns
        -------
        list of dict | None
            With ``quiet=False``: one dict per antenna with `antenna`, `subarray`, `fixed` and `weight`.
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
        if verbose:
            self._print_selfant(out)
        return self._done(out, quiet, outfile)

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

    def startmod(self, model=None, solint=0.0, flux=1.0, quiet=True, verbose=True):
        """Phase self-calibrate against a starting model, then discard it (difmap `startmod`).

        Parameters
        ----------
        model : str | None, default None
            A difmap .mod file to start from; ``None`` uses a point source of `flux` Jy at the phase centre, as difmap does.
        solint : float | str, default 0.0
            The solution interval, as for `selfcal` (minutes, a string with a unit, or "scan").
        flux : float, default 1.0
            Flux density of the point-source model, in Jy.
        quiet : bool, default True
            Return None. ``False`` returns the `selfcal` result dict.
        verbose : bool, default True
            Print the self-calibration summary.

        Returns
        -------
        dict | None
            The `selfcal` result, with ``quiet=False``.
        """
        self.clrmod()
        if model is None:
            self.addcmp(flux, 0.0, 0.0)
        else:
            self.rmodel(model)
        res = self.selfcal(phase=True, solint=solint, quiet=False, verbose=verbose)
        self.clrmod()
        return self._done(res, quiet)

    def resoff(self, baseline=None, quiet=True):
        """Solve for per-baseline amplitude and phase offsets against the current model, to absorb non-closing errors (difmap `resoff`). The corrections are applied until `clroff()`.

        Parameters
        ----------
        baseline : tuple of str | None, default None
            Solve for this baseline only, as a pair of station names; ``None`` solves every baseline.
        quiet : bool, default True
            Return None. ``False`` returns the number of corrections set.

        Returns
        -------
        int | None
            The number of (baseline, IF) corrections set, with ``quiet=False``.
        """
        bl = None
        if baseline is not None:
            bl = (self._ant_index(baseline[0]), self._ant_index(baseline[1]))
        n = self._core.resoff(baseline=bl)
        self._dirty()
        return self._done(n, quiet)

    def clroff(self, quiet=True):
        """Undo every baseline correction set by `resoff` (difmap `clroff`).

        Parameters
        ----------
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        self._core.clroff()
        self._dirty()
        return self._done(self, quiet)

    def baseline_corrections(self, outfile=None):
        """The per-baseline corrections set by `resoff`.

        Parameters
        ----------
        outfile : str | None, default None
            Also write the list to this file as JSON.

        Returns
        -------
        list of dict
            One dict per corrected baseline: `baseline` (a pair of station names), `amp` (the amplitude factor per IF) and `phase` (the phase per IF, in degrees).
        """
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

    def shift(self, east, north, quiet=True):
        """Shift the phase centre (difmap `shift`): the map contents move by (east, north), and the model components and CLEAN windows move with them. Shifts accumulate; `unshift()` undoes them all.

        Parameters
        ----------
        east : float
            Shift towards the east, in mas.
        north : float
            Shift towards the north, in mas.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        east, north = float(east), float(north)
        self._core.shift(east * MAS, north * MAS)
        self.windows[:] = [
            (x0 + east, x1 + east, y0 + north, y1 + north)
            for (x0, x1, y0, y1) in self.windows
        ]
        self._dirty()
        return self._done(self, quiet)

    def unshift(self, quiet=True):
        """Undo every accumulated phase-centre shift (difmap `unshift`).

        Parameters
        ----------
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        east, north = self.total_shift
        self._core.unshift()
        self.windows[:] = [
            (x0 - east, x1 - east, y0 - north, y1 - north)
            for (x0, x1, y0, y1) in self.windows
        ]
        self._dirty()
        return self._done(self, quiet)

    @property
    def total_shift(self):
        """The accumulated phase-centre shift as ``(east, north)`` in mas."""
        e, n = self._core.shift_total
        return (e / MAS, n / MAS)

    def uvaver(self, aver_time, doscatter=False):
        """Return a new observation with the calibrated data averaged into `aver_time` integrations (difmap uvaver).

        Parameters
        ----------
        aver_time : float | str
            The new integration time: seconds as a bare number, or a string carrying its unit (``"30s"``, ``"2min"``).
        doscatter : bool, default False
            Derive the output weights from the scatter of the averaged samples instead of summing the input weights.

        Returns
        -------
        Observation
            A new observation; this one is left as it is.

        Notes
        -----
        The gains, baseline corrections and any shift are applied before averaging, and the new observation starts uncalibrated, with this one's selection, imaging setup and windows.
        """
        from difmapy.average import uvaver

        secs = parse_time(aver_time, "s")
        new = self._derived(uvaver(self._core, secs, bool(doscatter)))
        new._aver_time = float(secs)
        return new

    def chanaver(self, nchan=None, channels=None):
        """Return a new observation with every `nchan` adjacent channels of each IF averaged into one (AIPS AVSPC, CASA split's `width`).

        Parameters
        ----------
        nchan : int | "all" | None, default None
            Channels per output channel; it must divide each IF's number of channels. ``None`` (or ``"all"``) averages every IF down to a single channel.
        channels : list of (int, int) | None, default None
            Inclusive global channel ranges, as for `select`, to average only those - the way band edges are left out. ``None``: all.

        Returns
        -------
        Observation
            A new observation; this one is left as it is.

        Notes
        -----
        As with `uvaver`, the calibrated data are averaged and the new observation starts uncalibrated; the channel selection is reset to all (averaged) channels, keeping the Stokes selection.
        """
        from difmapy.average import chanaver

        chlist = [(int(a), int(b)) for a, b in (channels or [])]
        return self._derived(chanaver(self._core, nchan, chlist or None),
                             same_channels=False)

    def _derived(self, core, same_channels=True):
        """A new observation around `core` - made from this one's data by averaging - with this one's imaging setup, windows, selection and provenance.

        The rows (time averaging) or channels (frequency averaging) no longer map one-to-one onto the source file's, so `_ms_origin` is not carried over. What is: the calibration provenance (`_cal_origin`: antennas, IFs, field - enough for calibration tables and for `save` to build a new MS), and the trace of which MS rows and channels each averaged sample came from (`_ms_avg_origin`, see `_trace_averaging`), which `save_flags` uses.
        """
        new = Observation(core)
        origin = getattr(self._core, "_ms_origin", None)
        new._from_ms = origin["path"] if origin else getattr(self, "_from_ms", None)
        cal_origin = getattr(self._core, "_cal_origin", None)
        if cal_origin is not None:
            core._cal_origin = cal_origin
        # What flag export needs to refer back to the data as loaded:
        # the time bins and the original channels behind each channel.
        new._aver_time = self._aver_time
        new._chan_origin = self._chan_origin
        new._orig_if_nchan = self._orig_if_nchan
        step = getattr(core, "_avg_chan_map", None)
        if step is not None:
            step = np.asarray(step)
            prev = self._chan_origin
            if prev is None:
                idx = np.arange(self._core.nctotal)
                prev = np.column_stack([idx, idx])
                new._orig_if_nchan = list(self.nchan)
            origin = np.zeros((core.nctotal, 2), dtype=np.int64)
            for k in range(core.nctotal):
                src = np.nonzero(step == k)[0]
                if src.size:
                    origin[k] = (prev[src[0], 0], prev[src[-1], 1])
            new._chan_origin = origin
        self._trace_averaging(core)
        # Flags set before this averaging are folded into the averaged
        # data; from here on, "new" flags are counted from this state.
        new._flags_at_load = np.array(core.flags(), copy=True)
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

    def _trace_averaging(self, core):
        """Record, on an averaged `core`, which rows and channels of the Measurement Set each of its samples was made from.

        `uvaver`/`chanaver` leave the map of their own step on the new core; this composes it with the map so far (the identity for data straight from the MS), so that any chain of averagings can be traced back - which is what lets `save_flags` put a flag set on an averaged sample onto every original sample behind it.
        """
        row_step = core.__dict__.pop("_avg_row_map", None)
        chan_step = core.__dict__.pop("_avg_chan_map", None)
        prev = getattr(self._core, "_ms_avg_origin", None)
        if prev is None:
            origin = getattr(self._core, "_ms_origin", None)
            if origin is None:
                return
            prev = {
                "path": origin["path"],
                "ms_row": origin["ms_row"],
                "if_nchan": list(origin["if_nchan"]),
                "row_map": np.arange(self._core.nrow, dtype=np.int64),
                "chan_map": np.arange(self._core.nctotal, dtype=np.int64),
            }
        new = dict(prev)
        if row_step is not None:
            new["row_map"] = np.asarray(row_step)[prev["row_map"]]
        if chan_step is not None:
            cm = prev["chan_map"]
            new["chan_map"] = np.where(
                cm >= 0, np.asarray(chan_step)[np.maximum(cm, 0)], -1)
        core._ms_avg_origin = new

    def copy(self) -> "Observation":
        """An independent copy of this observation: data, flags, gains, model, selection, map and every imaging setting. Changing one never affects the other, which is what lets calibration experiments run side by side (see `bayes_gscale`).

        Returns
        -------
        Observation
            The copy.
        """
        import copy as _copy

        new = Observation.__new__(Observation)
        # The load-time flags are never modified, so they are shared
        # rather than copied (they are as big as the FLAG column).
        shared = ("_core", "_flags_at_load")
        state = {k: v for k, v in self.__dict__.items() if k not in shared}
        new.__dict__.update(_copy.deepcopy(state))
        new._flags_at_load = self._flags_at_load
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

    def flag(self, baseline=None, station=None, tmin=None, tmax=None, subarray=None, if_index=None, selected_channels_only=False, quiet=True):
        """Flag visibilities (difmap `flag`). With no restriction at all, everything is flagged; each argument narrows the selection.

        Parameters
        ----------
        baseline : tuple of str | None, default None
            Only this baseline, as a pair of station names.
        station : str | None, default None
            Only the baselines of this station.
        tmin : float | str | None, default None
            Start of the time range: seconds since the reference day, or a string with a unit ("1.5h"); ``None``: from the start.
        tmax : float | str | None, default None
            End of the time range, as `tmin`; ``None``: to the end.
        subarray : int | None, default None
            Only this subarray (0-based).
        if_index : int | None, default None
            Only this IF (0-based).
        selected_channels_only : bool, default False
            Only the channels of the current selection, instead of every channel of the IFs concerned.
        quiet : bool, default True
            Return None. ``False`` returns the number of rows affected.

        Returns
        -------
        int | None
            The number of rows affected, with ``quiet=False``.
        """
        n = self._edit(True, baseline, station, tmin, tmax, subarray,
                       if_index, selected_channels_only)
        self._reapply_ignores()
        self._dirty()
        return self._done(n, quiet)

    def unflag(self, baseline=None, station=None, tmin=None, tmax=None, subarray=None, if_index=None, selected_channels_only=False, quiet=True):
        """Unflag visibilities (difmap `unflag`). Deleted data (weight 0) stays flagged, and so do the baselines of `ignore`d stations.

        Parameters
        ----------
        baseline : tuple of str | None, default None
            Only this baseline, as a pair of station names.
        station : str | None, default None
            Only the baselines of this station.
        tmin : float | str | None, default None
            Start of the time range: seconds since the reference day, or a string with a unit ("1.5h"); ``None``: from the start.
        tmax : float | str | None, default None
            End of the time range, as `tmin`; ``None``: to the end.
        subarray : int | None, default None
            Only this subarray (0-based).
        if_index : int | None, default None
            Only this IF (0-based).
        selected_channels_only : bool, default False
            Only the channels of the current selection, instead of every channel of the IFs concerned.
        quiet : bool, default True
            Return None. ``False`` returns the number of rows affected.

        Returns
        -------
        int | None
            The number of rows affected, with ``quiet=False``.
        """
        n = self._edit(False, baseline, station, tmin, tmax, subarray,
                       if_index, selected_channels_only)
        # An unflag must not resurrect an ignored station.
        self._reapply_ignores()
        self._dirty()
        return self._done(n, quiet)

    # ------------------------------------------------------------------
    # ignoring stations
    # ------------------------------------------------------------------

    @property
    def ignored(self) -> list[str]:
        """The names of the stations currently set aside by `ignore`, sorted."""
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

        A baseline between two ignored stations belongs to both, so its current flags are ignore flags, not the user's; the snapshot taken when the *other* station was ignored is what that row really looked like.
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

        Called after anything that edits flags, so that ignoring outlives an `unflag` that would otherwise bring the station back without anyone asking.
        """
        for rows, _ in self._ignored.values():
            self._core.edit_rows([int(r) for r in rows], True,
                                 if_index=None, sel_chan=False)

    def ignore(self, *antennas, quiet=True):
        """Set aside every baseline of these stations until `unignore`.

        The data is flagged, so imaging, model fitting, self-calibration and the plots all skip it - but unlike `flag`, the exact flag state of those baselines is remembered first, so `unignore` puts them back as they were rather than wholesale unflagged. Everyone else's data can be flagged and calibrated in the meantime, and the station returns with its own history intact.

        Flag edits made to an ignored station's own baselines while it is ignored are discarded by `unignore`, since it restores the remembered state. And self-calibration solved while a station is ignored has no solution for it, so it comes back uncalibrated for those intervals - `selfcal` again, or `selfant` to hold it fixed. Ignores are session state: `save()` does not write them.

        Parameters
        ----------
        *antennas : str
            Station names (case-insensitive), at least one.
        quiet : bool, default True
            Return None. ``False`` returns the stations ignored after the call.

        Returns
        -------
        list of str | None
            The stations ignored after the call, with ``quiet=False``.
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
        return self._done(self.ignored, quiet)

    def unignore(self, *antennas, quiet=True):
        """Bring back stations set aside by `ignore`. Their baselines are restored to the flag state they had when they were ignored; baselines shared with a station that is still ignored stay out.

        Parameters
        ----------
        *antennas : str
            Station names (case-insensitive); none brings back every ignored station.
        quiet : bool, default True
            Return None. ``False`` returns the stations still ignored.

        Returns
        -------
        list of str | None
            The stations still ignored after the call, with ``quiet=False``.
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
        return self._done(self.ignored, quiet)

    @property
    def flags(self) -> np.ndarray:
        """The FLAG column, a boolean array of shape (nrow, nchan_total, npol), True where flagged. It can be assigned an array of the same shape."""
        return self._core.flags()

    @flags.setter
    def flags(self, value):
        self._core.set_flags(np.ascontiguousarray(value, dtype=bool))
        self._dirty()

    @property
    def flagged_fraction(self) -> float:
        """The fraction of all samples (rows x channels x polarizations) that are flagged, between 0 and 1."""
        c = self._core
        return c.nflagged / (c.nrow * c.nctotal * c.npol)

    def gain_snapshot(self):
        """A copy of the current gain table, for `savecaltable`'s `since=` argument, so that later calibration can be exported as a separate, incremental table.

        Returns
        -------
        dict
            `amp`, `phs` (radians) and the boolean `used`, each of shape (ntimes, nif, nant).
        """
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
        """The formats `savecaltable` writes for `outformat`.

        Parameters
        ----------
        outformat : str | None, default None
            "CASA", "AIPS" or "both" (any case). ``None`` (or "auto") picks the format native to the data - CASA for a Measurement Set, AIPS for UVFITS - unless only one of `ms`/`uvfits` is given, which says which table is wanted.
        ms : str | None, default None
            The Measurement Set a CASA table would refer to.
        uvfits : str | None, default None
            The UVFITS file an AIPS table would be numbered for.

        Returns
        -------
        tuple of str
            ``("casa",)``, ``("aips",)`` or ``("casa", "aips")``.
        """
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
        """Where `savecaltable` puts each format. A single format is written to `path` itself. With both, a `path` ending in ``.fits`` is the AIPS file and the CASA table is that name without ``.fits`` (and without a ``.TASAV`` before it); any other `path` is the CASA table, and the AIPS file is ``<path>.TASAV.FITS``.

        Parameters
        ----------
        path : str
            The path given to `savecaltable`.
        formats : tuple of str
            The formats, as `caltable_formats` returns them.

        Returns
        -------
        dict
            Format -> path.
        """
        path = os.fspath(path)
        if len(formats) == 1:
            return {formats[0]: path}
        if path.lower().endswith(".fits"):
            base = path[:-5]
            if base.lower().endswith(".tasav"):
                base = base[:-6]
            return {"casa": base, "aips": path}
        return {"casa": path, "aips": f"{path}.TASAV.FITS"}

    def savecaltable(self, path, outformat=None, ms=None, uvfits=None, spw_ids=None, flag_uncalibrated=False, overwrite=True, quiet=True, verbose=True, since=None, solutions=None):
        """Export the accumulated antenna gains as a calibration table that CASA or AIPS can apply to the data.

        The table is a snapshot of every correction applied so far (the gains accumulate over `selfcal`/`gscale` calls). CASA divides the data by its gains, so a CASA table holds the reciprocal of difmapy's corrections; AIPS multiplies by them, so an SN table holds their conjugate. Either way, applying the table reproduces what difmapy shows. Baseline corrections (`resoff`) and phase-centre `shift`s cannot be expressed in such a table, and are reported as warnings rather than dropped silently.

        Parameters
        ----------
        path : str
            Where to write; see `caltable_paths` for where each format goes when both are written.
        outformat : str | None, default None
            "CASA" (a "G Jones" table for `applycal`), "AIPS" (an SN table in a TASAV FITS file, for FITLD and TACOP) or "both", in any case. ``None`` follows the data: a Measurement Set gets a CASA table, a UVFITS file an AIPS one - or, given only `ms` or only `uvfits`, the table that file is for.
        ms : str | None, default None
            The Measurement Set a CASA table refers to; ``None`` uses the one the data came from (required for UVFITS input).
        uvfits : str | None, default None
            The UVFITS file an SN table is numbered for; ``None`` uses the one the data came from (for MS input, antennas are numbered in ANTENNA-table order, as exportuvfits does). Antennas are matched by name, so another source of the same observation may be given.
        spw_ids : list of int | None, default None
            CASA only: the spectral-window ids of the IFs; ``None`` uses those recorded on loading.
        flag_uncalibrated : bool, default False
            Write entries with no solution as flagged rather than as unity gains.
        overwrite : bool, default True
            Replace an existing table.
        quiet : bool, default True
            Return None. ``False`` returns the writer's summary.
        verbose : bool, default True
            Print what was written, and any warnings.
        since : dict | None, default None
            A `gain_snapshot()` taken earlier: write only the calibration accumulated since then, giving one table per self-cal round. Applying the chain is equivalent to applying one cumulative table.
        solutions : dict | None, default None
            Write this set of gains instead of the gain table (`bayes_gscale` uses it).

        Returns
        -------
        dict | None
            With ``quiet=False``: `path`, `nrows`, `nflagged`, `warnings` and `format`; with both formats, one such dict under ``"casa"`` and ``"aips"``, and `warnings` and the list of paths at the top.
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
        if verbose:
            label = {"casa": "CASA G table", "aips": "AIPS SN table (TASAV)"}
            for fmt, info in infos.items():
                print(f"Wrote {label[fmt]} {info['path']}: {info['nrows']} "
                      f"rows ({info['nflagged']} flagged solutions)")
            for w in next(iter(infos.values()))["warnings"]:
                print(f"  warning: {w}")
        if len(infos) == 1:
            return self._done(next(iter(infos.values())), quiet)
        out = dict(infos)
        out["warnings"] = next(iter(infos.values()))["warnings"]
        out["path"] = [info["path"] for info in infos.values()]
        return self._done(out, quiet)

    def flag_commands(self) -> dict:
        """The flags added in this session as a compact list of selections (station or baseline, IFs, channels, time range, polarizations); see `difmapy.io.flags.flag_entries`.

        Returns
        -------
        dict
            The selections under `rows`, and the number of flagged samples they cover under `nsamples`.
        """
        from difmapy.io.flags import flag_entries

        return flag_entries(self)

    def wflags(self, path, outformat=None, uvfits=None, quiet=True, verbose=True):
        """Write the flags added in this session as a flag table that AIPS or CASA can apply - to the data that were loaded or to other data of the same observation.

        Only flags set since loading are written (those already in the file are not repeated), as selections of a station or baseline, IFs, channels and a time range. After averaging on load a time range is the whole averaging bin and the channels are the original ones, so every sample behind a flagged average is covered. Samples *un*flagged in the session cannot be expressed in such a list; `save_flags` writes the exact flag state into a Measurement Set instead.

        Parameters
        ----------
        path : str
            Where to write. With both formats `path` is the CASA list and the FG file is ``<path>.FG.TASAV.FITS`` (or the other way round for a `path` ending in ``.fits``).
        outformat : str | None, default None
            "AIPS" - an FG table in a TASAV FITS file (FITLD it, TACOP ``INEXT 'FG'`` onto the UV data) - "CASA" - a flag-command list for ``flagdata(vis, mode='list', inpfile=path)`` - or "both"; ``None`` picks the format native to the data, as for `savecaltable`.
        uvfits : str | None, default None
            The UV file an FG table is numbered for; ``None`` uses the one loaded.
        quiet : bool, default True
            Return None. ``False`` returns the writer's summary.
        verbose : bool, default True
            Print what was written.

        Returns
        -------
        dict | None
            With ``quiet=False``: `path`, `nrows`, `nsamples` and `format`; one such dict under ``"casa"``/``"aips"`` if both were written.
        """
        from difmapy.io.aips import save_fgtable
        from difmapy.io.flags import flag_entries, save_flagcmds

        formats = self.caltable_formats(outformat, uvfits=uvfits)
        path = os.fspath(path)
        if len(formats) == 1:
            paths = {formats[0]: path}
        elif path.lower().endswith(".fits"):
            base = path[:-5]
            for tail in (".tasav", ".fg"):
                if base.lower().endswith(tail):
                    base = base[: -len(tail)]
            paths = {"casa": base, "aips": path}
        else:
            paths = {"casa": path, "aips": f"{path}.FG.TASAV.FITS"}
        entries = flag_entries(self)
        infos = {}
        for fmt in formats:
            if fmt == "casa":
                info = save_flagcmds(self, paths[fmt], entries)
            else:
                info = save_fgtable(self, paths[fmt], entries, uvfits=uvfits)
            info.update(format=fmt, nsamples=entries["nsamples"])
            infos[fmt] = info
            if verbose:
                kind = ("CASA flag commands" if fmt == "casa"
                        else "AIPS FG table (TASAV)")
                print(f"Wrote {kind} {info['path']}: {info['nrows']} "
                      f"entries for {entries['nsamples']} flagged samples")
        return self._done(next(iter(infos.values())) if len(infos) == 1 else infos, quiet)

    def save_flags(self, path=None, flag_row=True, quiet=True):
        """Write the flags back into the Measurement Set the data came from - the standard MS way of persisting them: only FLAG (and FLAG_ROW) are modified, data and weights are untouched.

        For data as loaded, the FLAG column becomes exactly the current flags. For data averaged in time or frequency (on load or later), a flagged averaged sample flags every sample of the MS that went into it, and unflagged ones are left as the file has them - an average cannot say which of its inputs were good - so flags are only added.

        Parameters
        ----------
        path : str | None, default None
            The Measurement Set to write to; it must be the one the data came from, which ``None`` uses.
        flag_row : bool, default True
            Also set FLAG_ROW on rows whose every sample is flagged.
        quiet : bool, default True
            Return None. ``False`` returns the number of rows written.

        Returns
        -------
        int | None
            The number of MS rows written, with ``quiet=False``.
        """
        from difmapy.io.ms import save_flags

        return self._done(save_flags(self._core, path=path, flag_row=flag_row), quiet)

    # ------------------------------------------------------------------
    # interactive plots (pyqtgraph)
    # ------------------------------------------------------------------

    def radplot(self, quantity="ap", colorby="spw", block=None):
        """Amplitude and phase against UV radius (difmap `radplot`), with the model overplotted and interactive flagging. In the window, ``n``/``p`` highlight one antenna's baselines at a time and ``h`` lists every key.

        Parameters
        ----------
        quantity : str, default "ap"
            "amp", "phase", or "ap" (also "anp", "a&p") for an amplitude panel above a phase panel.
        colorby : str, default "spw"
            Point colouring: "spw" (a gradient over the IFs), "time", "baseline", "amp", or "none".
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import radplot

        return radplot(self, quantity=quantity, colorby=colorby, block=block)

    def projplot(self, angle=0.0, quantity="ap", colorby="spw", block=None, step=10.0):
        """Amplitude and phase against the UV distance projected on a direction (difmap `projplot`), with interactive flagging. In the window ``<`` and ``>`` turn the projection by `step` degrees and redraw.

        Parameters
        ----------
        angle : float, default 0.0
            Position angle of the projection direction, in degrees north through east.
        quantity : str, default "ap"
            "amp", "phase", or "ap" (also "anp", "a&p") for an amplitude panel above a phase panel.
        colorby : str, default "spw"
            Point colouring: "spw" (a gradient over the IFs), "time", "baseline", "amp", or "none".
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.
        step : float, default 10.0
            The angle, in degrees, that ``<``/``>`` turn the projection by.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import projplot

        return projplot(self, angle_deg=angle, quantity=quantity,
                        colorby=colorby, block=block, step=step)

    def uvplot(self, colorby="spw", block=None):
        """The UV coverage (difmap `uvplot`), conjugate points included, with interactive flagging.

        Parameters
        ----------
        colorby : str, default "spw"
            Point colouring: "spw" (a gradient over the IFs), "time", "baseline", "amp", or "none".
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import uvplot

        return uvplot(self, colorby=colorby, block=block)

    def vplot(self, nplot=3, reftel=None, quantity="ap", block=None):
        """Visibility amplitude and phase against time (difmap `vplot`), `nplot` baselines to a page, each framed as one, with a per-IF legend and interactive flagging. ``n``/``p`` page through the baselines, the space bar switches between flagging per baseline and per antenna, ``z``/``u`` restore the y/x ranges (the time axis spans the whole observation).

        The arguments come in difmap's order: ``vplot(3)`` is three baselines to a page, ``vplot(3, "EF")`` only EF's baselines, and ``vplot("EF")`` the same with the default page size.

        Parameters
        ----------
        nplot : int | str, default 3
            Baselines per page; 0 puts all of a station's baselines on one page. A station name here is taken as `reftel`.
        reftel : str | int | None, default None
            Show only this station's baselines (a name, case-insensitive, or an antenna index).
        quantity : str, default "ap"
            "amp", "phase", or "ap" (also "anp", "a&p") for an amplitude panel above a phase panel.
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import vplot

        return vplot(self, nplot=nplot, reftel=reftel, quantity=quantity,
                     block=block)

    def mapplot(self, what="map", mapsize=None, cellsize=None, uvweight=None, block=None, scale="linear", cmap="color", uvrange=None, uvtaper=None, **clean_args):
        """The interactive map display (difmap `mapplot`), with CLEAN windows, model editing and model fitting. The title names the file the data came from; ``h`` lists every key.

        The "Weighting" box at the top of the window switches between difmap's own `uvweight` scheme and robust -2, -1, 0, +1, +2, and re-images at once; like `uvweight()`, the choice stays in effect after the window closes. Beside it, the "UV range" (min, max) and "Taper" (value, radius) fields set `uvrange` and `uvtaper` - M-lambda, or a number with a unit such as ``500k`` - and re-image when Enter is pressed. As in difmap, the colours span the displayed map from its minimum to its peak.

        Parameters
        ----------
        what : str, default "map"
            The image shown first: "map" (the residual map), "beam" (the dirty beam), "clean" (the restored map) or "model" (the model restored without residuals).
        mapsize : int | None, default None
            Map size in pixels for this and later images; with none of `mapsize`, `cellsize`, `uvweight` given and `mapsize()` never called, `auto_mapsize()` picks one.
        cellsize : float | None, default None
            Pixel size in mas for this and later images.
        uvweight : float | None, default None
            A Briggs robustness, -2 to +2, for this and later images.
        uvrange : tuple | None, default None
            ``(uvmin, uvmax)`` for this and later images, as `uvrange` takes them: M-lambda, or strings with a unit.
        uvtaper : tuple | None, default None
            ``(value, radius)`` for this and later images, as `uvtaper` takes them.
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.
        scale : str, default "linear"
            The colour scale, "linear" or "log"; the ``l`` key toggles it.
        cmap : str, default "color"
            "color" (difmap's pseudo-colour table), "grey" (the ``g`` key switches between the two) or "viridis".
        **clean_args
            Arguments for the `clean` that the ``c`` key runs (`niter`, `gain`, `cutoff`), and `verbose`.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import mapplot

        return mapplot(self, what=what, mapsize=mapsize, cellsize=cellsize,
                       uvweight=uvweight, block=block, scale=scale,
                       cmap=cmap, uvrange=uvrange, uvtaper=uvtaper, **clean_args)

    #: `mapplot` under difmap's shorter spelling.
    maplot = mapplot

    def cpplot(self, triangles=None, if_index=None, nplot=4, block=None):
        """Closure phases against time, with the model overplotted (difmap `cpplot`), `nplot` triangles to a page.

        Parameters
        ----------
        triangles : list of tuple of str | None, default None
            The triangles to show, as triples of station names; ``None`` shows every closed triangle.
        if_index : int | None, default None
            One IF (0-based); ``None`` shows each.
        nplot : int, default 4
            Triangles per page.
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import cpplot

        return cpplot(self, triangles=triangles, if_index=if_index,
                      nplot=nplot, block=block)

    def tplot(self, block=None):
        """The time sampling of each antenna (difmap `tplot`): one row per station with a mark at every integration where it has unflagged data, in the primary colour when all of its baselines are unflagged and in the secondary colour when some are flagged.

        Parameters
        ----------
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import tplot

        return tplot(self, block=block)

    def corplot(self, quantity="both", nplot=3, block=None):
        """The self-calibration gain corrections against time (difmap `corplot`): one antenna to a pair of panels, amplitude above and phase below on a shared time axis, framed together.

        Solutions can be edited as data are in `vplot`: Shift+drag flags the solutions in the box, Ctrl+drag restores them, ``f``/``F`` act on the one nearest the cursor, Ctrl+Z / Ctrl+Shift+Z undo and redo, ``x`` shows or hides the flagged ones (red crosses at their original values). A flagged solution is replaced by interpolating the remaining ones of its antenna and IF, and the data are re-calibrated from the raw visibilities at once; see `flag_solutions`, which does the same from the prompt.

        Parameters
        ----------
        quantity : str, default "both"
            "both", "amp" or "phase".
        nplot : int, default 3
            Antennas per page.
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import corplot

        return corplot(self, quantity=quantity, nplot=nplot, block=block)

    def fplot(self, reftel=None, baselines=None, nplot=3, tmin=None, tmax=None, calibrated=True, block=None):
        """Amplitude and phase against frequency, one baseline to a pair of panels, averaged over time. Amplitudes are scalar-averaged, so they do not decorrelate, and phases vector-averaged. `possm()` shows the vector average of both, on a grid of baselines; `specplot()` averages every baseline at once.

        Parameters
        ----------
        reftel : str | int | None, default None
            Show only this station's baselines.
        baselines : list of tuple of str | None, default None
            The baselines to show, as pairs of station names; ``None`` shows all.
        nplot : int, default 3
            Baselines per page.
        tmin : float | str | None, default None
            Start of the averaging range: seconds, or a string with a unit ("30min"); ``None``: from the start.
        tmax : float | str | None, default None
            End of the averaging range; ``None``: to the end.
        calibrated : bool, default True
            Apply the accumulated calibration; ``False`` shows the data as loaded.
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import fplot

        return fplot(self, reftel=reftel, baselines=baselines, nplot=nplot,
                     tmin=tmin, tmax=tmax, calibrated=calibrated, block=block)

    def possm(self, nplot=9, reftel=None, baselines=None, ncols=3, tmin=None, tmax=None, calibrated=True, block=None):
        """Vector-averaged amplitude and phase against frequency (AIPS POSSM), a grid of baselines to a page - 3 x 3 by default - each an amplitude panel above a phase panel. Every channel of every IF is vector-averaged over the whole observation (or `tmin`..`tmax`), so phase slopes and ripples across the band show, and decorrelate the amplitudes as POSSM does. ``n``/``p`` page through the baselines in the same order as `vplot`.

        Parameters
        ----------
        nplot : int, default 9
            Baselines per page; 0 puts them all on one.
        reftel : str | int | None, default None
            Show only this station's baselines (a name, case-insensitive, or an antenna index).
        baselines : list of tuple of str | None, default None
            The baselines to show, as pairs of station names; ``None`` shows all.
        ncols : int, default 3
            Baselines per row of the grid.
        tmin : float | str | None, default None
            Start of the averaging range: seconds, or a string with a unit ("30min"); ``None``: from the start.
        tmax : float | str | None, default None
            End of the averaging range; ``None``: to the end.
        calibrated : bool, default True
            Apply the accumulated calibration; ``False`` shows the data as loaded.
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import possm

        return possm(self, nplot=nplot, reftel=reftel, baselines=baselines,
                     ncols=ncols, tmin=tmin, tmax=tmax, calibrated=calibrated,
                     block=block)

    def dashboard(self, block=None):
        """Every view of the session in one window, kept in step: vplot, tplot and possm (top left, in tabs), radplot, projplot and corplot (top right), mapplot (bottom left), and the controls (bottom right).

        The plots are the usual interactive ones, with their keys and flagging; whatever changes the data or the model in one - a flag edit, a CLEAN, a fit - or from the controls reloads the others, as ``r`` would in each. The Main tab loads another file (with optional time and frequency averaging) and runs `selfcal` (phase and/or amplitude, at a solution interval), `gscale` (with or without a floating scale) and `bayes_gscale` (over the source models ticked); the Output tab shows everything the session prints and has buttons to save the maps, the beam, the model, the visibilities, the calibration and flag tables, and the whole session. `difmapy.dashboard()` opens one with no data loaded.

        Parameters
        ----------
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        Dashboard
            The dashboard window; `Dashboard.obs` is the observation it shows, which changes when a file is loaded from it.
        """
        from difmapy.plots import dashboard

        return dashboard(self, block=block)

    def specplot(self, baseline=None, tmin=None, tmax=None, xaxis="freq", block=None):
        """The time-averaged spectrum of the selected polarization (difmap `specplot`), over every baseline or one.

        Parameters
        ----------
        baseline : tuple of str | None, default None
            One baseline, as a pair of station names; ``None`` averages every baseline.
        tmin : float | str | None, default None
            Start of the averaging range: seconds, or a string with a unit; ``None``: from the start.
        tmax : float | str | None, default None
            End of the averaging range; ``None``: to the end.
        xaxis : str, default "freq"
            "freq" for frequency or "chan" for channel number.
        block : bool | None, default None
            Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

        Returns
        -------
        PlotWindow
            The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
        """
        from difmapy.plots import specplot

        return specplot(self, baseline=baseline, tmin=tmin, tmax=tmax,
                        xaxis=xaxis, block=block)

    # ------------------------------------------------------------------
    # closure / spectral data (without plotting)
    # ------------------------------------------------------------------

    def closure_phases(self, triangle=None, if_index=None, outfile=None):
        """Closure-phase time series of the data and of the model.

        Parameters
        ----------
        triangle : tuple of str | None, default None
            One triangle, as three station names; ``None`` gives every closed triangle.
        if_index : int | None, default None
            One IF (0-based); ``None`` gives each IF.
        outfile : str | None, default None
            Also write the list to this file as JSON.

        Returns
        -------
        list of dict
            One dict per triangle and IF with `triangle`, `if_index`, and arrays `time` (seconds), `phase`, `model` and `error` (radians).
        """
        idx = None
        if triangle is not None:
            idx = tuple(sorted(self._ant_index(t) for t in triangle))
        return self._report(
            [dict(d) for d in self._core.closure_phases(
                triangle=idx, if_index=if_index)],
            outfile,
        )

    def spectrum(self, baseline=None, tmin=None, tmax=None, calibrated=True, outfile=None):
        """The time-averaged spectrum of the current polarization selection. It covers every channel, not only the selected ones, so it can be used to decide which channels to select. The visibilities are vector-averaged with the accumulated calibration applied per channel.

        Parameters
        ----------
        baseline : tuple of str | None, default None
            One baseline, as a pair of station names; ``None`` averages every baseline.
        tmin : float | str | None, default None
            Start of the averaging range: seconds, or a string with a unit; ``None``: from the start.
        tmax : float | str | None, default None
            End of the averaging range; ``None``: to the end.
        calibrated : bool, default True
            Apply the accumulated calibration; ``False`` averages the data as loaded.
        outfile : str | None, default None
            Also write the result to this file as JSON.

        Returns
        -------
        dict
            Per-channel arrays: `chan`, `freq` (Hz), the averaged `re` and `im` (Jy), the scalar-averaged `amp` (Jy) and the summed weight `wt` (0 where there is no data).
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
        """Write the restored map to a FITS file (difmap `wmap`), restoring with the default beam first if that has not been done.

        Parameters
        ----------
        path : str
            The FITS file to write.
        overwrite : bool, default True
            Replace an existing file.
        """
        if self._restored is None:
            self.restore(verbose=False)
        self._write_fits_image(path, self._restored, "JY/BEAM", beam=self._restore_beam,
                               overwrite=overwrite)

    def wdmap(self, path, overwrite=True):
        """Write the residual (dirty) map to a FITS file (difmap `wdmap`).

        Parameters
        ----------
        path : str
            The FITS file to write.
        overwrite : bool, default True
            Replace an existing file.
        """
        self._ensure_map()
        self._write_fits_image(path, self._core.map(), "JY/BEAM", overwrite=overwrite)

    def wbeam(self, path, overwrite=True):
        """Write the dirty beam to a FITS file (difmap `wbeam`).

        Parameters
        ----------
        path : str
            The FITS file to write.
        overwrite : bool, default True
            Replace an existing file.
        """
        self._ensure_map()
        self._write_fits_image(path, self._core.beam(), "", overwrite=overwrite)

    def wmodel(self, path):
        """Write the model to a difmap/Caltech ``.mod`` file (difmap `wmodel`).

        Parameters
        ----------
        path : str
            The file to write.
        """
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

    def rmodel(self, path, quiet=True):
        """Read a difmap/Caltech ``.mod`` model file, adding its components to the model (difmap `rmodel`).

        Parameters
        ----------
        path : str
            The file to read.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
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
        return self._done(self, quiet)

    def wobs(self, path, overwrite=True, freeze_shift=False):
        """Write the calibrated, edited UV data to a random-groups UVFITS file (difmap `wobs`). Antenna gains and baseline corrections are applied.

        Parameters
        ----------
        path : str
            The UVFITS file to write.
        overwrite : bool, default True
            Replace an existing file.
        freeze_shift : bool, default False
            Also apply the accumulated phase-centre shift (difmap's default is not to; `save()` records the shift separately).
        """
        from difmapy.io.uvfits import save_uvfits

        save_uvfits(self._core, path, overwrite=overwrite,
                    freeze_shift=freeze_shift)

    def _save_ms(self, prefix, ms):
        """The ``<prefix>.ms`` half of `save`, if this session came from a Measurement Set.

        `ms` is None to write one when it is possible and say nothing when it is not, False to skip it, True to require it. Data still row-for-row with the originating MS are written into a copy of it (`save_ms`); averaged data get a new MS built on its structure (`save_averaged_ms`).
        """
        if ms is False:
            return None
        from difmapy.io.ms import save_averaged_ms, save_ms

        core = self._core
        cal = getattr(core, "_cal_origin", None) or {}
        if getattr(core, "_ms_origin", None) is not None:
            writer = save_ms
        elif cal.get("format") == "ms":
            writer = save_averaged_ms
        else:
            if ms:
                raise ValueError(
                    "ms=True, but this observation was not loaded from a "
                    "Measurement Set; only <prefix>.uvf can be written"
                )
            return None
        try:
            return writer(core, f"{prefix}.ms", overwrite=True)
        except Exception as exc:
            if ms:
                raise
            # The UV data is already safely in the .uvf, so a missing
            # casatools or an unreadable source MS must not lose it.
            print(f"warning: could not write {prefix}.ms: {exc}")
            return None

    def _save_caltable(self, prefix, caltable):
        """The calibration-table part of `save`: the accumulated gains, in the format native to the data (`<prefix>.G` for a Measurement Set, `<prefix>.TASAV.FITS` for UVFITS), when there are any."""
        if caltable is False:
            return None
        if not np.asarray(self._core.gains_used()).any():
            if caltable:
                raise ValueError("caltable=True, but no calibration has "
                                 "been applied in this session")
            return None
        fmt = self.caltable_formats()[0]
        path = f"{prefix}.G" if fmt == "casa" else f"{prefix}.TASAV.FITS"
        try:
            return self.savecaltable(path, outformat=fmt, quiet=False, verbose=False)["path"]
        except Exception as exc:
            if caltable:
                raise
            print(f"warning: could not write {path}: {exc}")
            return None

    def _save_flagtable(self, prefix, flags):
        """The flag part of `save`: the flags added in the session, in the format native to the data (`<prefix>.flagcmd` for a Measurement Set, `<prefix>.FG.TASAV.FITS` for UVFITS), when there are any."""
        if flags is False:
            return None
        from difmapy.io.flags import flag_entries

        fmt = self.caltable_formats()[0]
        path = (f"{prefix}.flagcmd" if fmt == "casa"
                else f"{prefix}.FG.TASAV.FITS")
        try:
            if not flag_entries(self)["rows"]:
                if flags:
                    raise ValueError("flags=True, but no flags have been "
                                     "added in this session")
                return None
            return self.wflags(path, outformat=fmt, quiet=False, verbose=False)["path"]
        except Exception as exc:
            if flags:
                raise
            print(f"warning: could not write {path}: {exc}")
            return None

    def _save_image(self, prefix, image):
        """The image part of `save`: the restored (CLEAN) map as ``<prefix>.fits`` with the weighting in use, and one ``<prefix>.robust<R>.fits`` for each robustness in `ROBUST_CHOICES`, when there is a model to restore.

        Returns ``(main path, [per-weighting paths])``.
        """
        if image is False:
            return None, []
        if not self.model:
            if image:
                raise ValueError("image=True, but there is no model to "
                                 "restore an image from")
            return None, []
        try:
            self.wmap(f"{prefix}.fits")
            main = f"{prefix}.fits"
        except Exception as exc:
            if image:
                raise
            print(f"warning: could not write {prefix}.fits: {exc}")
            return None, []
        return main, self._save_weighted_images(prefix, image)

    def _save_weighted_images(self, prefix, image):
        """Write the restored map once per robustness in `ROBUST_CHOICES`, each with its own estimated beam. The imaging runs on a copy, so the session's weighting, maps and restoring beam are untouched (re-inverting here would replace the residual map CLEAN left with a freshly gridded one)."""
        work = self.copy()
        paths = []
        for r in ROBUST_CHOICES:
            path = f"{prefix}.robust{r:+g}.fits"
            try:
                work.uvweight(self._binwid, self._errpow, radial=self._dorad,
                              robust=r)
                work.restore(verbose=False)
                work._write_fits_image(path, work._restored, "JY/BEAM",
                                       beam=work._restore_beam)
                paths.append(path)
            except Exception as exc:
                if image:
                    raise
                print(f"warning: could not write {path}: {exc}")
        return paths

    def save(self, prefix, ms=None, caltable=None, image=None, flags=None,
             quiet=True, verbose=True):
        """Save the whole session with a common prefix (difmap save).

        Always written, and read back by `get()`: ``<prefix>.uvf`` (the calibrated UV data), ``.mod`` (model), ``.win`` (CLEAN windows) and ``.par.json`` (the imaging parameters).

        Written when there is something to write:

        * ``<prefix>.ms`` - for an observation that came from a Measurement Set, so a session that started from CASA can go back to it. Unaveraged data go into a copy of the originating MS (calibrated visibilities in CORRECTED_DATA, current flags in FLAG). Data averaged in time or frequency, on load or later, get a new MS with the averaged rows and, if channels were averaged, updated spectral windows (see `difmapy.io.ms.save_averaged_ms`).
        * ``<prefix>.G`` or ``<prefix>.TASAV.FITS`` - the calibration table of the gains accumulated in the session, as `savecaltable` writes it for the data's own format (CASA for a Measurement Set, AIPS for UVFITS), once any self-calibration has been applied.
        * ``<prefix>.flagcmd`` or ``<prefix>.FG.TASAV.FITS`` - the flags added in the session, as `wflags` writes them for the data's own format (a CASA flag-command list for a Measurement Set, an AIPS FG table for UVFITS), once any have been.
        * ``<prefix>.fits`` - the restored CLEAN map (`wmap`), once there is a model; it is restored with the current beam if it has not been yet.
        * ``<prefix>.robust-2.fits`` ... ``<prefix>.robust+2.fits`` - the restored map once for each robustness mapplot offers (-2, -1, 0, +1, +2), each with that weighting's own estimated beam, written alongside ``<prefix>.fits``. They are imaged on a copy, so the session's weighting and maps are untouched.

        Parameters
        ----------
        prefix : str
            Common path prefix of every file.
        ms, caltable, image, flags : bool | None, default None
            ``None`` writes that file when it applies, and only warns if it then cannot be written; ``False`` skips it; ``True`` makes its absence an error.
        quiet : bool, default True
            Return None. ``False`` returns the files written.
        verbose : bool, default True
            Print the list of files written.

        Returns
        -------
        dict | None
            With ``quiet=False``, the files written, by kind (``"uvf"``, ``"ms"``, ``"mod"``, ``"win"``, ``"par"``, ``"caltable"``, ``"flags"``, ``"image"``, and ``"images"``, the list of per-weighting maps).
        """
        import json

        files = {"uvf": f"{prefix}.uvf"}
        self.wobs(files["uvf"])
        files["ms"] = self._save_ms(prefix, ms)
        self.wmodel(f"{prefix}.mod")
        self.wwins(f"{prefix}.win")
        files.update(mod=f"{prefix}.mod", win=f"{prefix}.win")
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
        files["par"] = f"{prefix}.par.json"
        files["caltable"] = self._save_caltable(prefix, caltable)
        files["flags"] = self._save_flagtable(prefix, flags)
        files["image"], files["images"] = self._save_image(prefix, image)
        files = {k: v for k, v in files.items() if v}
        if verbose:
            names = [p for v in files.values()
                     for p in (v if isinstance(v, list) else [v])]
            print("Saved " + ", ".join(names))
        return self._done(files, quiet)

    @classmethod
    def get(cls, prefix) -> "Observation":
        """Restore a session written by `save()` (difmap `get`): the UV data, selection, imaging parameters, shift, model and CLEAN windows.

        Parameters
        ----------
        prefix : str
            The prefix given to `save`.

        Returns
        -------
        Observation
            The restored session.
        """
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
            # Saved in wavelengths, which a bare number no longer means.
            value, radius = pars["uvtaper"]
            obs.uvtaper(value, f"{radius!r} lambda")
            obs.uvrange(*(f"{r!r} lambda" for r in pars["uvrange"]))
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
        """Write the CLEAN windows (mas) to a difmap ``.win`` file (difmap `wwins`).

        Parameters
        ----------
        path : str
            The file to write.
        """
        with open(path, "w") as f:
            for (x0, x1, y0, y1) in self.windows:
                f.write(f"{x0:g} {x1:g} {y0:g} {y1:g}\n")

    def rwins(self, path, quiet=True):
        """Read CLEAN windows from a difmap ``.win`` file, adding them to the current ones (difmap `rwins`).

        Parameters
        ----------
        path : str
            The file to read.
        quiet : bool, default True
            Return None. ``False`` returns the observation.

        Returns
        -------
        Observation | None
            This observation, with ``quiet=False``.
        """
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("!"):
                    continue
                x0, x1, y0, y1 = (float(v) for v in line.split()[:4])
                self.windows.append((x0, x1, y0, y1))
        return self._done(self, quiet)


def load(path, stokes="I", channels=None, timeavg=None, freqavg=None,
         scatter=False, wtscale=1.0, field=None, data_column="DATA",
         average=None) -> Observation:
    """Load a UVFITS file or a Measurement Set (difmap `observe`).

    Unlike difmap, the total intensity is selected straight away, since that is how nearly every session starts.

    Parameters
    ----------
    path : str
        A random-groups UVFITS file, or a Measurement Set directory (read with casatools); a directory is taken to be an MS. Both must hold a single source.
    stokes : str | None, default "I"
        Polarization to select once loaded: "I", "Q", "U", "V", or a recorded one ("RR", "LL", "RL", "LR", "XX", ...). "I" uses difmap's permissive combination (a visibility survives when only one parallel hand is usable); Q/U/V are strict. ``None`` (or "none") loads without a selection, as difmap's `observe` does; `select()` changes it at any time.
    channels : list of (int, int) | None, default None
        Inclusive, 0-based channel ranges over the global channel axis (all IFs concatenated), e.g. ``[(0, 31), (64, 95)]``, as `select` takes them. With `freqavg`, only these channels go into the averages - the way band edges are dropped - and the selection then covers every averaged channel. ``None``: all channels.
    timeavg : float | str | None, default None
        Time-average straight after loading (difmap `uvaver`): seconds as a number, or a string with its unit ("10s", "2min"). ``None``: keep the original integrations.
    freqavg : int | "all" | bool | None, default None
        Average this many adjacent channels into one (`chanaver`); it must divide every IF's number of channels. ``"all"`` or ``True`` averages each IF down to a single channel. ``None``/``False``: no averaging.
    scatter : bool, default False
        With `timeavg`: derive the output weights from the scatter of the averaged samples instead of summing the input weights.
    wtscale : float, default 1.0
        Factor applied to the data weights as they are read.
    field : str | int | None, default None
        Measurement Sets only: the field (name or FIELD_ID) to load. Needed when the MS has several fields; difmapy, like difmap, is single-source.
    data_column : str, default "DATA"
        Measurement Sets only: the visibility column to read, "DATA" or "CORRECTED_DATA".
    average : float | str | None, default None
        The older name of `timeavg`, still accepted.

    Returns
    -------
    Observation

    Notes
    -----
    The work is done in the cheapest order: channels are averaged first, then the Stokes selection is made, then the time averaging. Averaged data from a Measurement Set still go back to one: `save` writes a new MS with the averaged rows, `save_flags` flags every original sample behind a flagged averaged one, and `savecaltable` works as usual.

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
    """Average adjacent channels of each IF, returning a new observation (`obs` is left as it is); the same as ``obs.chanaver(nchan, channels)``.

    Parameters
    ----------
    obs : Observation
        The observation to average.
    nchan : int | "all" | None, default None
        Channels per output channel; it must divide every IF's number of channels. ``None``/``"all"``: one channel per IF.
    channels : list of (int, int) | None, default None
        Inclusive global channel ranges to use; the others are left out of the averages. ``None``: all channels.
    """
    return obs.chanaver(nchan, channels=channels)


def uvaver(obs, aver_time, scatter=False) -> Observation:
    """Time-average an observation into `aver_time` integrations (difmap uvaver), returning a new one; `obs` is left as it is. The same as ``obs.uvaver(aver_time, doscatter=scatter)``.

    Parameters
    ----------
    obs : Observation
        The observation to average (its calibration is applied first).
    aver_time : float | str
        The new integration time: seconds, or a string with its unit ("10s", "2min").
    scatter : bool, default False
        Derive the weights from the scatter of the averaged samples instead of summing the input weights.
    """
    return obs.uvaver(aver_time, doscatter=scatter)


#: `load` under difmap's own name for it.
observe = load
