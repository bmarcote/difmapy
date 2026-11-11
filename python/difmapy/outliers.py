"""Antenna-based outlier flagging of visibilities (the "smart outliers" flagger).

Shared by difmapy (``Observation.smartflag``, and Shift+S, then a drag, on a data plot) and vlbipy (``vlbipy flag
smart-outliers`` and the flagging passes of its dask-ms pipeline). Pure numpy: a caller bins its visibilities into
a :class:`VisGrid` (baseline x subband x channel block x correlation x time bin) and turns the masks of
:func:`find_outliers` into flags. Its origin is Cormac Reynolds' AIPS UVFND + ``flagsum.py`` recipe: find the
outlying visibilities, see which antenna is common to most of them, and flag that antenna over the time range
they span.

Model
-----
Every *series* (one baseline, subband, channel block and correlation of one source) should follow one smooth
trend: the source does not change, and what does (elevation, gains) changes on hour timescales. The
statistics are done on the complex visibilities themselves:

* a bin departs from a reference ``M`` made of the *other* bins of its scan within ``level_bins`` of it. Its
  amplitude is that of the component-wise median of those bins (a constant level: an antenna slewing in, a
  dropout, a burst all stand out of it); its phase is that of the median of linear estimators of the bin (the
  mean of every symmetric pair of neighbours and, near a scan edge, the extrapolation of one-sided pairs), so a
  phase drift inside the scan (raw phases wander by tens of degrees) is followed to the very first and last
  bins. ``V - M`` is divided by its own noise, from the ``n`` samples of the bin and those of the estimators;
* a scan's level ``L`` (the component-wise median of its bins) departs from the trend ``T`` of the *other*
  scans of the series: a local linear fit with tricube weights out to ``trend_minutes`` on either side, with
  bisquare weights (first from the median of the other scans) so that a bad scan drags neither its own trend
  nor its neighbours' (a median alone would be biased wherever the trend is steep, the ends of the track first);
* every departure is split along the reference (its amplitude part: the projection, unbiased and Gaussian at
  any signal-to-noise ratio, unlike ``|V| - |M|``) and across it (its phase part), and each part has its own
  robust dispersion (Gaussian MAD): the phases of raw or phase-referenced data scatter more than their
  amplitudes (atmosphere, ionosphere), and one scale for both would let that scatter blunt the amplitude test.
  The bin noise is followed in time over the trend window, so a rising Tsys or a worsening atmosphere is not
  an outlier;
* amplitude departures smaller than ``floor`` (bins) or ``scan_floor`` (scans) times the reference amplitude,
  and phase departures smaller than ``phase_floor`` degrees, are never outliers: on a bright source the thermal
  noise is so small that the residual gain and phase changes any calibration leaves would otherwise exceed it.

Leaving the cell under test out of its own reference matters: with the 5-10 bins of a phase-calibrator scan, a
median that includes the bin pulls the residuals in, the dispersion comes out too small and pure noise is
flagged by the percent.

A cell is an outlier when either departure is beyond ``sigma``. Sigmas are those of a 1-D Gaussian: the
threshold radius of a complex residual is the one with the same tail probability as ``|x| > sigma``
(:func:`sigma_radius`; 3 sigma is a radius of 3.44 per-component sigmas).

Raw data (before any fringe fit) rotate in phase within a scan: with ``align_phases`` every scan of every
series is first rotated by its mean bin-to-bin phase step (a per-baseline fringe rate; a delay is a constant
phase per channel block) and its mean phase, so it follows the same model. What their phases still do -
wander antenna by antenna with the atmosphere - is measured as the phase part's own noise, so raw data are
tested on the complex values too. ``use_phases = False`` tests the amplitude part alone, as ``|V| - |M|``
(one dimension, so the thresholds are the sigmas themselves), for phases that cannot be followed at all.
The checks behind these choices are in vlbipy's ``docs/smart_outliers_appendix.md``.

Antennas
--------
A baseline is formed by two antennas, but a fault is nearly always one antenna's, and shows on all of its
baselines at once - often only at 2 or 3 sigma on each, which no vote over the baselines would catch. Every
departure is therefore split along the reference into an amplitude (radial) and a phase (tangential) part, in
units of its noise. An antenna's fault moves its baselines together: the amplitude parts in the same direction,
the phase parts too once the sign of the baselines it ends is turned (``V = g1 conj(g2)``). The antenna statistic
is the median of each part over its baselines times ``sqrt(2 n / pi)`` (the noise of the median of ``n`` unit
Gaussians): about ``sqrt(n)`` more sensitive than any single baseline, and robust, since one bad partner moves a
median by one place. In every cell the antenna with the largest statistic beyond ``antenna_sigma`` is blamed;
its baselines are then set aside and the next one is looked for, so a bad antenna never gets its partners
blamed. ``antenna_sigma`` is higher than the ``sigma`` of a single visibility on purpose: the test runs on
every antenna, subband, channel block, hand and time bin - a million cells on a phase calibrator - and at
5 sigma noise alone seeds about one of them, so the fits converge instead of eating ever more of the data.
A cell no antenna explains is flagged on its baseline alone only beyond ``baseline_sigma``.

Edges
-----
Interference rises and decays: the blamed cells seed a region that grows along time inside the scan and along
the channel blocks of the subband through every cell where the antenna's statistic, smoothed over
``smooth_bins``, stays beyond ``edge_sigma`` (hysteresis thresholding). An antenna flagged over most channel blocks of a subband
at one time loses the whole subband, and over most subbands all of them (``broadband_fraction``).

Selections
----------
Cells picked out by hand (:attr:`VisGrid.selected`: a box drawn on a plot) are outliers by definition, whatever
their departures: they are kept out of every reference, level and noise estimate, and in an antenna's statistic
each counts as a departure without bound, on the side its other baselines lean to. One selected baseline thus
moves the median of each of its antennas by one place, and the antenna whose other baselines depart with it is
the one at fault; an antenna most of whose baselines are selected is at fault outright (the "antenna common to
most" of the recipe). Only the antennas of a selected baseline, and only where it is selected, may be blamed,
and below ``antenna_sigma``: the test runs on the cells selected, not on every cell of the data, so for the
same number of false blames as a search of everything the threshold drops towards ``sigma`` as the selection
shrinks (:func:`selection_sigma`). A blamed antenna's flag grows from there as above, on every baseline, inside
or outside the selection, its growth judged on its unselected baselines. Whatever selected cell no antenna
explains is flagged on its own baseline; nothing outside the selection is flagged on a baseline alone.

Iterations
----------
Flags change the statistics, so the model is fitted again on what is left. The departures inside scans are
flagged first, until a fit finds none; only then are whole scans judged against the trend, alone (a burst over
half a scan moves its level, and would otherwise condemn the whole scan), and if that flags something the cycle
starts again, until a fit of both flags nothing new (at most ``max_iterations`` fits per subband).
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

__all__ = ["OutlierSettings", "VisGrid", "OutlierResult", "find_outliers", "sigma_radius", "selection_sigma",
           "flag_runs"]

logger = logging.getLogger(__name__)

#: Median radius of a complex Gaussian with unit sigma per component (Rayleigh distribution).
RAYLEIGH_MEDIAN = math.sqrt(2.0 * math.log(2.0))
#: Median absolute value of a unit Gaussian (the MAD of a 1-D residual over this is its sigma).
GAUSSIAN_MAD = 0.6744897501960817
#: Environment variable that turns the compiled (numba) medians off when set to 0.
NUMBA_ENV = "DIFMAPY_NUMBA"


@dataclass
class OutlierSettings:
    """Parameters of :func:`find_outliers` (all sigmas are 1-D Gaussian equivalents, see :func:`sigma_radius`).

    Attributes
    ----------
    sigma : float
        Threshold of an outlying visibility (cell): left out of its neighbours' references and counted as an
        outlier. With a selection (:attr:`VisGrid.selected`), also the lowest antenna threshold
        (:func:`selection_sigma`).
    antenna_sigma : float
        Threshold of the combined statistic of an antenna's baselines that flags the antenna. A selection lowers
        it towards ``sigma`` (:func:`selection_sigma`).
    edge_sigma : float
        Growth threshold of a flagged region: its start and end, where the fault rises and decays.
    baseline_sigma : float
        A departure no antenna explains is flagged on its own baseline beyond this (0 = never).
    level_bins : int
        Bins on either side (inside the scan) a bin's reference is made of.
    floor : float
        Fraction of the reference amplitude added (in quadrature) to the noise of a bin's amplitude part.
    scan_floor : float
        Fraction of the trend amplitude added (in quadrature) to the spread of the scan levels' amplitude part.
    phase_floor : float
        Degrees of phase (times the reference amplitude) added in quadrature to the noise of the phase part, of
        bins and scans alike.
    trend_minutes : float
        Half-width of the window of scans the trend and the bin noise are taken over (elevation timescale).
    min_trend_scans : int
        Other scans the window must hold before a scan is compared with the trend.
    min_scans : int
        Scans a series needs before its scan-to-scan dispersion is measured (fewer: no scan-level test).
    min_baselines : int
        Baselines with data an antenna needs in a cell to be blamed there.
    smooth_bins : int
        Half-width, in time bins, of the running mean of the antenna score used to grow regions.
    margin_bins : int
        Time bins added on both sides of every flagged region (inside its scan).
    broadband_fraction : float
        Fraction of the channel blocks of a subband (of the subbands) flagged at one time above which the whole
        subband (every subband) is flagged for that antenna.
    max_iterations : int
        Fits of the model per subband at most.
    align_phases : bool
        Rotate every scan of every series by its phase rate and mean phase first (raw, unfringe-fitted data).
    use_phases : bool
        Test the complex departures: the amplitude part as the projection on the reference, and the phase
        (tangential) part, each with its own noise. False: the amplitude part only, as ``|V| - |M|``, for phases
        that cannot be followed even within a scan.
    """

    sigma: float = 3.0
    antenna_sigma: float = 5.0
    edge_sigma: float = 1.5
    baseline_sigma: float = 5.0
    level_bins: int = 5
    floor: float = 0.05
    scan_floor: float = 0.2
    phase_floor: float = 10.0
    trend_minutes: float = 60.0
    min_trend_scans: int = 2
    min_scans: int = 5
    min_baselines: int = 2
    smooth_bins: int = 1
    margin_bins: int = 0
    broadband_fraction: float = 0.5
    max_iterations: int = 10
    align_phases: bool = False
    use_phases: bool = True

    def __post_init__(self) -> None:
        if not self.sigma > 0:
            raise ValueError(f"outliers: sigma must be positive, not {self.sigma}")
        if not 0 < self.edge_sigma <= min(self.sigma, self.antenna_sigma):
            raise ValueError(f"outliers: edge_sigma must be in (0, min(sigma, antenna_sigma)], not {self.edge_sigma}")
        if min(self.floor, self.scan_floor, self.phase_floor) < 0:
            raise ValueError(f"outliers: floor, scan_floor and phase_floor must be >= 0, not {self.floor}, "
                             f"{self.scan_floor}, {self.phase_floor}")
        if self.baseline_sigma < 0:
            raise ValueError(f"outliers: baseline_sigma must be >= 0 (0 = off), not {self.baseline_sigma}")
        if not self.trend_minutes > 0:
            raise ValueError(f"outliers: trend_minutes must be positive, not {self.trend_minutes}")
        if not 0 < self.broadband_fraction <= 1:
            raise ValueError(f"outliers: broadband_fraction must be in (0, 1], not {self.broadband_fraction}")
        for name in ("level_bins", "min_trend_scans", "min_scans", "min_baselines", "max_iterations"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"outliers: {name} must be >= 1, not {getattr(self, name)}")
        for name in ("smooth_bins", "margin_bins"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"outliers: {name} must be >= 0, not {getattr(self, name)}")

    @classmethod
    def from_mapping(cls, values: Optional[dict]) -> "OutlierSettings":
        """Settings from a dict (a configuration section); an unknown key is an error."""
        values = dict(values or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"outliers: unknown setting(s) {', '.join(unknown)}; known: {', '.join(sorted(known))}")
        return cls(**values)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class VisGrid:
    """Binned visibilities of one source.

    Attributes
    ----------
    vis : numpy.ndarray
        complex64 ``(nbl, nspw, nblock, ncorr, ntime)``: mean of the unflagged samples of every cell.
    count : numpy.ndarray
        Same shape: the number of unflagged samples averaged (0 = no data in the cell).
    ant1, ant2 : numpy.ndarray
        ``(nbl,)`` antenna indices of every baseline (in ``range(nant)``, different).
    times : numpy.ndarray
        ``(ntime,)`` bin times [s].
    segment : numpy.ndarray
        ``(ntime,)`` scan of every bin; the bins of one scan are contiguous.
    nant : int
        Number of antennas.
    selected : numpy.ndarray, optional
        bool, ``vis`` shape: cells picked out by hand as outliers (a box drawn on a plot). They are outliers by
        definition: kept out of every reference, level and noise estimate, always flagged, and the only cells a
        flag may start from - the antenna behind them is looked for and its flag grows from there (see the module
        notes). None: a search of all the data.
    """

    vis: np.ndarray
    count: np.ndarray
    ant1: np.ndarray
    ant2: np.ndarray
    times: np.ndarray
    segment: np.ndarray
    nant: int
    selected: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.vis = np.asarray(self.vis, dtype=np.complex64)
        self.count = np.asarray(self.count, dtype=np.float32)
        self.ant1 = np.asarray(self.ant1, dtype=np.int64).reshape(-1)
        self.ant2 = np.asarray(self.ant2, dtype=np.int64).reshape(-1)
        self.times = np.asarray(self.times, dtype=np.float64).reshape(-1)
        self.segment = np.asarray(self.segment, dtype=np.int64).reshape(-1)
        self.nant = int(self.nant)
        if self.vis.ndim != 5:
            raise ValueError(f"VisGrid: vis must be (nbl, nspw, nblock, ncorr, ntime), not {self.vis.shape}")
        if self.count.shape != self.vis.shape:
            raise ValueError(f"VisGrid: count {self.count.shape} does not match vis {self.vis.shape}")
        nbl, ntime = self.vis.shape[0], self.vis.shape[-1]
        if self.ant1.size != nbl or self.ant2.size != nbl:
            raise ValueError(f"VisGrid: {nbl} baselines but {self.ant1.size}/{self.ant2.size} antenna indices")
        if np.any(self.ant1 == self.ant2) or np.any(np.minimum(self.ant1, self.ant2) < 0) \
                or np.any(np.maximum(self.ant1, self.ant2) >= self.nant):
            raise ValueError("VisGrid: baselines must join two different antennas in range(nant)")
        if self.times.size != ntime or self.segment.size != ntime:
            raise ValueError(f"VisGrid: {ntime} time bins but {self.times.size} times, {self.segment.size} scans")
        if self.selected is not None:
            self.selected = np.asarray(self.selected, dtype=bool)
            if self.selected.shape != self.vis.shape:
                raise ValueError(f"VisGrid: selected {self.selected.shape} does not match vis {self.vis.shape}")


@dataclass
class OutlierResult:
    """What :func:`find_outliers` flagged.

    Attributes
    ----------
    antenna_flags : numpy.ndarray
        bool ``(nant, nspw, nblock, ncorr, ntime)``: the antenna is flagged in that cell (all its baselines).
    baseline_flags : numpy.ndarray
        bool ``(nbl, nspw, nblock, ncorr, ntime)``: flagged on that baseline only.
    iterations : int
        Largest number of model fits any subband needed.
    history : list of dict
        Per subband and fit: ``spw``, ``iteration``, ``outliers`` (cells beyond sigma), ``antenna_cells`` and
        ``baseline_cells`` newly flagged.
    settings : dict
        The settings used.
    """

    antenna_flags: np.ndarray
    baseline_flags: np.ndarray
    iterations: int
    history: list = field(default_factory=list)
    settings: dict = field(default_factory=dict)

    def cell_flags(self, grid: VisGrid) -> np.ndarray:
        """bool ``(nbl, nspw, nblock, ncorr, ntime)``: every cell flagged, by its antennas or on its own."""
        return self.baseline_flags | self.antenna_flags[grid.ant1] | self.antenna_flags[grid.ant2]

    def summary(self, grid: VisGrid) -> dict:
        """Cells with data and newly flagged ones, in total and per antenna (over the antenna's baselines)."""
        data = grid.count > 0
        flagged = self.cell_flags(grid) & data
        per_baseline_data = data.reshape(data.shape[0], -1).sum(axis=1)
        per_baseline_flagged = flagged.reshape(flagged.shape[0], -1).sum(axis=1)
        per_antenna = {}
        for antenna in range(grid.nant):
            mine = (grid.ant1 == antenna) | (grid.ant2 == antenna)
            total = int(per_baseline_data[mine].sum())
            if total:
                per_antenna[antenna] = {"cells": total, "flagged": int(per_baseline_flagged[mine].sum()),
                                        "fraction": float(per_baseline_flagged[mine].sum()) / total}
        cells = int(data.sum())
        return {"cells": cells, "flagged": int(flagged.sum()),
                "fraction": float(flagged.sum()) / cells if cells else 0.0, "antennas": per_antenna,
                "antenna_cells": int(self.antenna_flags.sum()), "baseline_cells": int(self.baseline_flags.sum()),
                "iterations": self.iterations}


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _limit(sigma: float, use_phases: bool) -> float:
    """Threshold of a departure of ``sigma``: a radius for both parts (:func:`sigma_radius`), sigma for one."""
    if sigma <= 0:
        return 0.0
    return sigma_radius(sigma) if use_phases else float(sigma)


def sigma_radius(sigma: float) -> float:
    """Radius of a complex residual (unit sigma per component) with the tail probability of a 1-D ``sigma``.

    A complex Gaussian residual exceeds radius ``r`` with probability ``exp(-r^2 / 2)``; a 1-D one exceeds
    ``sigma`` with ``erfc(sigma / sqrt 2)``. Equal tails: ``r = sqrt(-2 ln erfc(sigma / sqrt 2))``.
    """
    if sigma <= 0:
        return 0.0
    x = sigma / math.sqrt(2.0)
    tail = math.erfc(x)
    log_tail = math.log(tail) if tail > 0.0 else -x * x - math.log(x * math.sqrt(math.pi))
    return math.sqrt(-2.0 * log_tail)


def _gaussian_tail(sigma: float) -> float:
    """Probability that a 1-D unit Gaussian exceeds ``sigma`` in absolute value."""
    return math.erfc(sigma / math.sqrt(2.0))


def selection_sigma(settings: "OutlierSettings", cells: int, selected: int) -> float:
    """The antenna threshold (1-D sigma) of a selection of ``selected`` antenna cells out of ``cells`` with data.

    A search of all the data blames an antenna at ``antenna_sigma``, which noise alone passes with a tail
    probability ``p`` in each of ``cells`` tests. A selection tests only its own cells, so the same number of false
    blames allows a tail ``p x cells / selected`` in each: the threshold is the sigma of that tail, never below
    ``sigma`` (a small box) nor above ``antenna_sigma`` (a box over everything).
    """
    low, high = min(settings.sigma, settings.antenna_sigma), float(settings.antenna_sigma)
    if selected <= 0:
        return high
    tail = _gaussian_tail(high) * max(int(cells), 1) / int(selected)
    if tail >= _gaussian_tail(low):
        return low
    a, b = low, high                                    # erfc decreases: bisect for the sigma of that tail
    for _ in range(60):
        middle = 0.5 * (a + b)
        a, b = (middle, b) if _gaussian_tail(middle) > tail else (a, middle)
    return 0.5 * (a + b)


def _numpy_median(values: np.ndarray, lower: bool) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_median` with numpy: sort every row (NaN sort last) and read the middle off it."""
    if values.shape[-1] == 0:
        return np.full(values.shape[:-1], np.nan, dtype=values.dtype), np.zeros(values.shape[:-1], dtype=np.int64)
    ordered = np.sort(values, axis=-1)
    count = np.count_nonzero(~np.isnan(values), axis=-1)
    low = np.take_along_axis(ordered, (np.maximum(count - 1, 0) // 2)[..., None], axis=-1)[..., 0]
    if lower:
        result = low
    else:
        high = np.take_along_axis(ordered, np.minimum(count // 2, values.shape[-1] - 1)[..., None], axis=-1)[..., 0]
        result = 0.5 * (low + np.where(count % 2 == 1, low, high))
    return np.where(count > 0, result, np.nan), count


_KERNELS: dict = {}


def _kernel():
    """The compiled gather-median kernel, or None (numba missing, or ``$DIFMAPY_NUMBA=0``). Built once."""
    if os.environ.get(NUMBA_ENV, "1").strip() == "0":
        return None
    if "gather" not in _KERNELS:
        try:
            import numba
        except Exception:  # noqa: BLE001 - numba is optional: the numpy path gives the same numbers
            _KERNELS["gather"] = None
            return None

        @numba.njit(cache=True)
        def order(buffer, n):  # pragma: no cover - compiled
            if n > 32:
                buffer[:n].sort()
                return
            for i in range(1, n):                     # insertion sort: rows are a handful of values
                x = buffer[i]
                j = i - 1
                while j >= 0 and buffer[j] > x:
                    buffer[j + 1] = buffer[j]
                    j -= 1
                buffer[j + 1] = x

        @numba.njit(cache=True)
        def middle(buffer, n, lower=False):  # pragma: no cover - compiled
            order(buffer, n)
            if lower or n % 2 == 1:
                return buffer[(n - 1) // 2]
            return 0.5 * (buffer[n // 2 - 1] + buffer[n // 2])

        @numba.njit(parallel=True, cache=True)
        def gather_median(values, index, lower, out, counts):  # pragma: no cover - compiled
            for row in numba.prange(values.shape[0]):
                buffer = np.empty(index.shape[1], values.dtype)
                for cell in range(index.shape[0]):
                    n = 0
                    for slot in range(index.shape[1]):
                        column = index[cell, slot]
                        if column >= 0:
                            x = values[row, column]
                            if x == x:
                                buffer[n] = x
                                n += 1
                    counts[row, cell] = n
                    out[row, cell] = middle(buffer, n, lower) if n else np.nan

        @numba.njit(parallel=True, cache=True)
        def estimate(real, imag, count, first, second, a, b, out_re, out_im, precision):  # pragma: no cover
            for row in numba.prange(real.shape[0]):
                buffer_re = np.empty(first.shape[1], real.dtype)
                buffer_im = np.empty(first.shape[1], real.dtype)
                for t in range(real.shape[1]):
                    n = 0
                    total = 0.0
                    for m in range(first.shape[1]):
                        i = first[t, m]
                        if i < 0:
                            continue
                        xi, yi, ci = real[row, i], imag[row, i], count[row, i]
                        if not (xi == xi) or ci <= 0:
                            continue
                        j = second[t, m]
                        if j >= 0:
                            xj, yj, cj = real[row, j], imag[row, j], count[row, j]
                            if not (xj == xj) or cj <= 0:
                                continue
                            buffer_re[n] = a[t, m] * xi + b[t, m] * xj
                            buffer_im[n] = a[t, m] * yi + b[t, m] * yj
                            total += 1.0 / (a[t, m] * a[t, m] / ci + b[t, m] * b[t, m] / cj)
                        else:
                            buffer_re[n] = a[t, m] * xi
                            buffer_im[n] = a[t, m] * yi
                            total += ci / (a[t, m] * a[t, m])
                        n += 1
                    precision[row, t] = total
                    if n == 0:
                        out_re[row, t] = np.nan
                        out_im[row, t] = np.nan
                        continue
                    out_re[row, t] = middle(buffer_re, n)
                    out_im[row, t] = middle(buffer_im, n)

        _KERNELS["gather"] = gather_median
        _KERNELS["estimate"] = estimate
    return _KERNELS["gather"]


def _gather_median(values: np.ndarray, index: np.ndarray, *, lower: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Median of ``values[..., index[m]]`` for every row ``m`` of a -1 padded ``index``, NaN ignored.

    ``values`` is ``(..., N)`` and ``index`` ``(M, W)``; returns ``(..., M)`` medians (NaN where nothing is finite)
    and counts. ``lower``: the lower median. Compiled, no ``(..., M, W)`` intermediate is ever built.
    """
    kernel = _kernel()
    if kernel is None or index.size == 0 or values.size == 0:
        return _numpy_median(_gather(values, index), lower)
    flat = np.ascontiguousarray(values.reshape(-1, values.shape[-1]))
    out = np.empty((flat.shape[0], index.shape[0]), dtype=flat.dtype)
    counts = np.empty(out.shape, dtype=np.int64)
    kernel(flat, np.ascontiguousarray(index, dtype=np.int64), bool(lower), out, counts)
    shape = values.shape[:-1] + (index.shape[0],)
    return out.reshape(shape), counts.reshape(shape)


def _median(values: np.ndarray, *, lower: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Median over the last axis ignoring NaN (NaN where nothing is finite), and the count of finite values.

    ``lower``: the lower median (element ``(n - 1) // 2`` of the sorted values). Rows are short (a scan's bins,
    an antenna's baselines): sorting them, compiled when numba is there, is far faster than ``numpy.nanmedian``.
    """
    kernel = _kernel()
    width = values.shape[-1]
    if kernel is None or width == 0 or values.size == 0:
        return _numpy_median(values, lower)
    # Every row is one cell of the kernel, gathering its own span of the flattened array.
    line = np.ascontiguousarray(values).reshape(1, -1)
    index = np.arange(line.size, dtype=np.int64).reshape(-1, width)
    out = np.empty((1, index.shape[0]), dtype=line.dtype)
    counts = np.empty(out.shape, dtype=np.int64)
    kernel(line, index, bool(lower), out, counts)
    return out.reshape(values.shape[:-1]), counts.reshape(values.shape[:-1])


def _reference(real: np.ndarray, imag: np.ndarray, count: np.ndarray,
               estimators: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every bin's reference from its neighbours (:func:`_estimator_tables`), and the precision of it.

    ``real``/``imag`` ``(S, ntime)`` (NaN: no data), ``count`` the samples per bin. Returns the component-wise
    median of the estimators and the sum of their inverse variances in units of one sample's (0: none).
    """
    first, second, a, b = estimators
    if _kernel() is not None and first.size:
        out_re, out_im = np.empty_like(real), np.empty_like(imag)
        precision = np.empty(real.shape, dtype=np.float64)
        _KERNELS["estimate"](np.ascontiguousarray(real), np.ascontiguousarray(imag),
                             np.ascontiguousarray(count, dtype=np.float32), first, second, a, b, out_re, out_im,
                             precision)
        return out_re, out_im, precision
    x_i, y_i = _gather(real, first), _gather(imag, first)
    c_i = _gather(count.astype(np.float32), first)
    has_j = second >= 0
    x_j, y_j = np.nan_to_num(_gather(real, second)), np.nan_to_num(_gather(imag, second))
    c_j = _gather(count.astype(np.float32), second)
    valid = np.isfinite(x_i) & (np.nan_to_num(c_i) > 0) & (~has_j | (np.isfinite(_gather(real, second))
                                                                     & (np.nan_to_num(c_j) > 0)))
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = a * a / c_i + np.where(has_j, b * b / c_j, 0.0)
        precision = np.where(valid, 1.0 / variance, 0.0).sum(axis=-1)
    estimate_re = np.where(valid, a * np.nan_to_num(x_i) + b * x_j, np.nan)
    estimate_im = np.where(valid, a * np.nan_to_num(y_i) + b * y_j, np.nan)
    return _numpy_median(estimate_re, False)[0], _numpy_median(estimate_im, False)[0], precision


def _gather(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """``values[..., index]`` for a -1 padded ``index`` (padding gives NaN); float input only."""
    taken = values[..., np.maximum(index, 0)]
    return np.where(index >= 0, taken, np.nan)


@dataclass
class _Segments:
    """Scan structure of the time axis.

    ``index`` (ntime,) scan position of every bin; ``starts``/``lengths`` (nscan,); ``gather`` (nscan, maxlen)
    bins of every scan, -1 padded; ``neighbours`` (ntime, 2 level_bins) the other bins of the same scan within
    ``level_bins``, -1 padded; ``estimators`` (:func:`_estimator_tables`); ``window`` (nscan, wmax) scans within the trend window, -1 padded; ``others`` the
    same without the scan itself; ``weights``
    (nscan, nscan) tricube weight of every other scan in a scan's trend (0 on the diagonal and outside);
    ``offsets`` (nscan, nscan) time of scan k minus time of scan j [s]; ``first`` (ntime,) True at a scan start.
    """

    index: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray
    gather: np.ndarray
    neighbours: np.ndarray
    estimators: tuple
    window: np.ndarray
    others: np.ndarray
    weights: np.ndarray
    offsets: np.ndarray
    first: np.ndarray


def _segments(times: np.ndarray, segment: np.ndarray, trend_seconds: float, level_bins: int) -> _Segments:
    ntime = segment.size
    if ntime == 0:
        empty, table = np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.int64)
        coefficients = np.zeros((0, 0), dtype=np.float32)
        return _Segments(empty, empty, empty, table, table, (table, table, coefficients, coefficients), table, table,
                         np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=bool))
    starts = np.concatenate([[0], np.flatnonzero(np.diff(segment) != 0) + 1]).astype(np.int64)
    if np.unique(segment[starts]).size != starts.size:
        raise ValueError("outliers: the bins of every scan must be contiguous on the time axis")
    lengths = np.diff(np.concatenate([starts, [ntime]])).astype(np.int64)
    index = np.repeat(np.arange(starts.size), lengths)
    offsets = np.arange(int(lengths.max()))
    gather = starts[:, None] + offsets[None, :]
    gather[offsets[None, :] >= lengths[:, None]] = -1
    steps = np.concatenate([np.arange(-level_bins, 0), np.arange(1, level_bins + 1)])
    neighbours = np.arange(ntime)[:, None] + steps[None, :]
    inside = (neighbours >= 0) & (neighbours < ntime)
    inside[inside] &= index[neighbours[inside]] == np.broadcast_to(index[:, None], neighbours.shape)[inside]
    neighbours = np.where(inside, neighbours, -1)
    centres = np.add.reduceat(times, starts) / lengths
    distance = centres[None, :] - centres[:, None]
    close = np.abs(distance) <= trend_seconds
    weights = np.where(close, (1.0 - np.minimum(np.abs(distance) / trend_seconds, 1.0) ** 3) ** 3, 0.0)
    np.fill_diagonal(weights, 0.0)
    first = np.zeros(ntime, dtype=bool)
    first[starts] = True
    return _Segments(index, starts, lengths, gather, neighbours, _estimator_tables(starts, lengths, level_bins),
                     _padded_rows(close), _padded_rows(close & ~np.eye(starts.size, dtype=bool)), weights, distance,
                     first)


def _estimator_tables(starts: np.ndarray, lengths: np.ndarray, level_bins: int) -> tuple:
    """Linear estimators of every bin from other bins of its scan: ``e = a v[i] + b v[j]``.

    Symmetric pairs ``(t - k, t + k)`` (``a = b = 1/2``, exact for a linear drift) for ``k <= level_bins``; where
    the window is cut by a scan edge, one-sided extrapolations ``(t + k, t + 2k)`` (``a = 2, b = -1``) from the
    other side; in scans of at most three bins also the other bins themselves (``a = 1``, a constant).

    Returns
    -------
    tuple
        ``first`` and ``second`` bin indices ``(ntime, m)`` (-1: unused, or no second term) and the coefficients
        ``a``, ``b`` (float32, same shape).
    """
    rows: list[list[tuple[int, int, float, float]]] = []
    half = max(1, level_bins // 2)
    for start, length in zip(starts.tolist(), lengths.tolist()):
        for p in range(length):
            row = [(start + p - k, start + p + k, 0.5, 0.5) for k in range(1, level_bins + 1)
                   if p - k >= 0 and p + k < length]
            if p < level_bins:
                row += [(start + p + k, start + p + 2 * k, 2.0, -1.0) for k in range(1, half + 1) if p + 2 * k < length]
            if length - 1 - p < level_bins:
                row += [(start + p - k, start + p - 2 * k, 2.0, -1.0) for k in range(1, half + 1) if p - 2 * k >= 0]
            if length <= 3:
                row += [(start + q, -1, 1.0, 0.0) for q in range(length) if q != p]
            rows.append(row)
    width = max(1, max((len(row) for row in rows), default=1))
    first = np.full((len(rows), width), -1, dtype=np.int64)
    second = np.full((len(rows), width), -1, dtype=np.int64)
    a = np.zeros((len(rows), width), dtype=np.float32)
    b = np.zeros((len(rows), width), dtype=np.float32)
    for t, row in enumerate(rows):
        for m, (i, j, ca, cb) in enumerate(row):
            first[t, m], second[t, m], a[t, m], b[t, m] = i, j, ca, cb
    return first, second, a, b


def _padded_rows(table: np.ndarray) -> np.ndarray:
    """Column indices of the True cells of every row of a bool ``table``, -1 padded to the longest row."""
    width = max(int(table.sum(axis=1).max()) if table.size else 0, 1)
    order = np.argsort(~table, axis=1, kind="stable")[:, :width]
    return np.where(np.take_along_axis(table, order, axis=1), order, -1)


def _antenna_tables(ant1: np.ndarray, ant2: np.ndarray, nant: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per antenna, its baselines, their other antenna and the sign of its phase on them (+1 first, -1 second).

    Three ``(nant, nant - 1)`` arrays, -1 padded (sign 0 there).
    """
    width = max(nant - 1, 1)
    baselines = np.full((nant, width), -1, dtype=np.int64)
    partners = np.full((nant, width), -1, dtype=np.int64)
    signs = np.zeros((nant, width), dtype=np.float32)
    filled = np.zeros(nant, dtype=np.int64)
    for index, (a, b) in enumerate(zip(ant1.tolist(), ant2.tolist())):
        for mine, other, sign in ((a, b, 1.0), (b, a, -1.0)):
            if filled[mine] < width:
                baselines[mine, filled[mine]] = index
                partners[mine, filled[mine]] = other
                signs[mine, filled[mine]] = sign
                filled[mine] += 1
    return baselines, partners, signs


def _per_antenna_any(mask: np.ndarray, ant1: np.ndarray, ant2: np.ndarray, nant: int) -> np.ndarray:
    """``(nant, *cell)``: True where any baseline of the antenna is True in ``mask`` ``(nbl, *cell)``."""
    out = np.zeros((nant,) + mask.shape[1:], dtype=bool)
    for antenna in range(nant):
        mine = (ant1 == antenna) | (ant2 == antenna)
        if mine.any():
            out[antenna] = mask[mine].any(axis=0)
    return out


def flag_runs(mask: np.ndarray, segment: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` bin index (inclusive) of every run of True in a 1-D ``mask``, never across scans."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    segment = np.asarray(segment)
    start = mask.copy()
    start[1:] &= ~mask[:-1] | (segment[1:] != segment[:-1])
    stop = mask.copy()
    stop[:-1] &= ~mask[1:] | (segment[1:] != segment[:-1])
    return list(zip(np.flatnonzero(start).tolist(), np.flatnonzero(stop).tolist()))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def _align_phases(vis: np.ndarray, valid: np.ndarray, seg: _Segments) -> np.ndarray:
    """Rotate every scan of every series (rows of ``vis``) by its mean bin-to-bin phase step and mean phase."""
    z = np.where(valid, vis, 0).astype(np.complex64)
    lag = np.zeros_like(z)
    lag[:, :-1] = z[:, 1:] * np.conj(z[:, :-1])
    lag[:, :-1][:, seg.first[1:]] = 0                          # never across two scans
    step = np.angle(np.add.reduceat(lag, seg.starts, axis=1)).astype(np.float32)
    offset = (np.arange(seg.index.size) - seg.starts[seg.index]).astype(np.float32)
    rotated = vis * np.exp(-1j * step[:, seg.index] * offset[None, :]).astype(np.complex64)
    mean = np.add.reduceat(np.where(valid, rotated, 0), seg.starts, axis=1)
    return (rotated * np.exp(-1j * np.angle(mean))[:, seg.index].astype(np.complex64)).astype(np.complex64)


def _local_trend(level_re: np.ndarray, level_im: np.ndarray, seg: _Segments,
                 min_scans: int, *, phases: bool = True) -> tuple:
    """Departure of every scan level from the trend of the *other* scans, and the robust spread of those.

    Parameters
    ----------
    level_re, level_im : numpy.ndarray
        ``(S, nscan)`` scan levels (NaN: no data).
    seg : _Segments
        Its ``weights`` and ``offsets`` define the local linear fit.
    min_scans : int
        Other scans with data and weight a scan needs to be compared with the trend.
    phases : bool
        Departures split along and across the trend (:func:`_parts`), each part with its own spread (True), or
        the differences of amplitudes alone, ``|L| - |T|`` (the phases of the scans are not to be trusted).

    Returns
    -------
    departure_re, departure_im : numpy.ndarray
        ``(S, nscan)`` ``L - T`` (NaN where not compared).
    factor : numpy.ndarray
        ``(S, nscan)`` ``sqrt(1 + sum a_k^2)`` for ``T = sum a_k L_k``: how much the trend's own noise, much larger
        where it is extrapolated (the first and last scans), adds to a departure.
    trend_re, trend_im : numpy.ndarray
        ``(S, nscan)`` the trend.
    spread_radial, spread_tangential : numpy.ndarray
        ``(S,)`` Gaussian MAD of the amplitude and phase parts of ``departure / factor`` (NaN when there is none;
        without ``phases`` both are the amplitude part's).
    """
    def parts(dep_re, dep_im, ref_re, ref_im):
        along, across, _ = _parts(dep_re, dep_im, ref_re, ref_im, phases=phases)
        return along, (across if phases else along)

    def spreads(along, across):
        return tuple(np.where((n > 0) & (m >= 0), m / GAUSSIAN_MAD, np.nan)
                     for m, n in (_median(np.abs(along)), _median(np.abs(across))))

    def size(along, across, spread_radial, spread_tangential):
        # Departure radius in units of each part's spread, so the bisquare weights see both parts alike.
        with np.errstate(invalid="ignore", divide="ignore"):
            along, across = along / spread_radial[:, None], across / spread_tangential[:, None]
        return np.hypot(along, across) if phases else np.abs(along)

    scale = RAYLEIGH_MEDIAN if phases else GAUSSIAN_MAD
    valid = np.isfinite(level_re) & np.isfinite(level_im)
    y_re, y_im = np.where(valid, level_re, 0.0), np.where(valid, level_im, 0.0)
    weights, offsets = seg.weights, seg.offsets
    support = valid.astype(np.float64) @ (weights > 0).T
    median_re, _ = _gather_median(level_re, seg.others)
    median_im, _ = _gather_median(level_im, seg.others)
    along, across = parts(level_re - median_re, level_im - median_im, median_re, median_im)
    robust = _bisquare(size(along, across, *spreads(along, across)), scale)
    spread_radial = spread_tangential = np.full(valid.shape[0], np.nan)
    factor = np.ones(valid.shape)
    for _ in range(2):
        mass = valid * robust
        s0, s1, s2 = mass @ weights.T, mass @ (weights * offsets).T, mass @ (weights * offsets ** 2).T
        squared, weights2 = mass * mass, weights * weights
        q0, q1, q2 = squared @ weights2.T, squared @ (weights2 * offsets).T, squared @ (weights2 * offsets ** 2).T
        det = s0 * s2 - s1 * s1
        linear = det > 1e-9 * np.maximum(s0 * s2, 1e-300)
        with np.errstate(invalid="ignore", divide="ignore"):
            leverage = np.where(linear, (s2 * s2 * q0 - 2.0 * s2 * s1 * q1 + s1 * s1 * q2) / np.where(linear, det, 1.0) ** 2,
                                q0 / np.where(s0 > 0, s0, np.nan) ** 2)
        trend = []
        for y in (y_re, y_im):
            sy, sty = (mass * y) @ weights.T, (mass * y) @ (weights * offsets).T
            with np.errstate(invalid="ignore", divide="ignore"):
                trend.append(np.where(linear, (s2 * sy - s1 * sty) / np.where(linear, det, 1.0),
                                      sy / np.where(s0 > 0, s0, np.nan)))
        compared = valid & (support >= min_scans) & (s0 > 0)
        factor = np.sqrt(1.0 + leverage)
        along, across = parts(level_re - trend[0], level_im - trend[1], trend[0], trend[1])
        along, across = np.where(compared, along, np.nan) / factor, np.where(compared, across, np.nan) / factor
        spread_radial, spread_tangential = spreads(along, across)
        robust = _bisquare(size(along, across, spread_radial, spread_tangential), scale)
    return (np.where(compared, level_re - trend[0], np.nan), np.where(compared, level_im - trend[1], np.nan), factor,
            trend[0], trend[1], spread_radial, spread_tangential)


def _bisquare(departure: np.ndarray, scale: float = RAYLEIGH_MEDIAN) -> np.ndarray:
    """Tukey bisquare weights of ``(S, nscan)`` departure sizes, cut at 4.685 robust sigmas of each row (median
    size over ``scale``; 1 where NaN)."""
    median, n = _median(departure)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = departure / (4.685 * median[:, None] / scale)
    return np.where(np.isfinite(u), np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0), 1.0)


def _parts(deviation_re, deviation_im, reference_re, reference_im, *, phases: bool = True) -> tuple:
    """Amplitude (along the reference) and phase (across it) parts of a complex deviation ``V - M``, unscaled.

    With ``phases`` the amplitude part is the projection of ``V - M`` on the direction of ``M``: unbiased and
    Gaussian at any signal-to-noise ratio. Without, it is the difference of the amplitudes, ``|V| - |M|`` (biased
    upwards at low signal-to-noise, Rician), which a rotation of the phase does not change: the projection would
    count one as a loss of amplitude (``cos(delta) - 1``). A reference of no amplitude has no direction: the real
    axis is used then. Returns ``(along, across, amplitude)``, the last ``|M|``.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        amplitude = np.hypot(reference_re, reference_im)
        cos = np.where(amplitude > 0, reference_re / amplitude, 1.0)
        sin = np.where(amplitude > 0, reference_im / amplitude, 0.0)
        if phases:
            along = deviation_re * cos + deviation_im * sin
        else:
            along = np.hypot(deviation_re + reference_re, deviation_im + reference_im) - amplitude
        across = deviation_im * cos - deviation_re * sin
    return along, across, amplitude


def _components(deviation_re, deviation_im, reference_re, reference_im, noise_radial, noise_tangential,
                amplitude_floor: float, phase_floor: float, *, phases: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude (radial) and phase (tangential) parts of a complex deviation along its reference, in units of noise.

    Each part has its own noise (:func:`_parts` for what the parts are): the phases of uncalibrated or
    phase-referenced data wander (atmosphere, ionosphere) by more than the thermal noise, their amplitudes do
    not, and a scale shared by both would let the phase scatter blunt the amplitude test. The radial part also
    tolerates ``amplitude_floor`` times the reference amplitude, the tangential part ``phase_floor`` degrees of it.
    """
    along, across, amplitude = _parts(deviation_re, deviation_im, reference_re, reference_im, phases=phases)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial = along / np.hypot(noise_radial, amplitude_floor * amplitude)
        tangential = across / np.hypot(noise_tangential, np.radians(phase_floor) * amplitude)
    return radial.astype(np.float32), tangential.astype(np.float32)


def _bin_noise(part: np.ndarray, unit: np.ndarray, seg: _Segments) -> np.ndarray:
    """Noise of one part of every bin's ``V - M`` (``(S, ntime)``): its Gaussian MAD per scan, in units of one
    sample's (``unit``), followed in time as the median over the trend window, times the bin's ``unit``.

    Noise-free data (a model, a simulation) have no noise: 0 then, so the floors alone set the tolerance.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        scan_noise, _ = _gather_median(np.abs(part) / unit, seg.gather)
    noise, _ = _gather_median(scan_noise, seg.window)
    with np.errstate(invalid="ignore"):                       # no noise x no data (unit inf): NaN, as it should be
        return np.where(noise >= 0, noise / GAUSSIAN_MAD, np.nan)[:, seg.index] * unit


def _scores(vis: np.ndarray, count: np.ndarray, flagged: np.ndarray, seg: _Segments,
            settings: OutlierSettings, *, scans: bool, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Departure of every cell from the model, in units of its own noise (NaN where no data).

    ``vis``, ``count``, ``flagged``: ``(..., ntime)`` of one subband. Without ``scans`` the departure of every bin
    from its neighbours, with ``scans`` that of its scan's level from the trend. ``threshold``: the outlier
    limit (:func:`_limit`; bins beyond it in a first pass are left out of the references of the second).

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        The amplitude (radial) and phase (tangential) parts, each ``vis``-shaped; their ``hypot`` is the radius.
    """
    shape = vis.shape
    ntime = shape[-1]
    phases = bool(settings.use_phases)
    v = vis.reshape(-1, ntime)
    c = count.reshape(-1, ntime)
    valid = (c > 0) & ~flagged.reshape(-1, ntime) & np.isfinite(v)
    if settings.align_phases:
        v = _align_phases(v, valid, seg)
    real = np.where(valid, v.real, np.nan).astype(np.float32)
    imag = np.where(valid, v.imag, np.nan).astype(np.float32)
    # Two passes: a strong outlier is one term of its neighbours' estimators (two of them near a scan edge), so
    # the references are built again without the cells the first pass finds beyond the threshold.
    # A cell none of whose estimators survives (a fault over most of a scan) is compared with the median of the
    # scan's surviving bins instead.
    usable = np.where(valid, c, 0)
    for attempt in range(2):
        drift_re, drift_im, precision = _reference(real, imag, usable, seg.estimators)
        kept = usable > 0
        level_re, _ = _gather_median(np.where(kept, real, np.nan), seg.neighbours)
        level_im, _ = _gather_median(np.where(kept, imag, np.nan), seg.neighbours)
        with np.errstate(invalid="ignore", divide="ignore"):
            # Amplitude of the level, phase of the drift; the level alone where the drift has no estimator.
            scale = np.hypot(level_re, level_im) / np.hypot(drift_re, drift_im)
            reference_re = np.where(np.isfinite(scale), drift_re * scale, level_re)
            reference_im = np.where(np.isfinite(scale), drift_im * scale, level_im)
            neighbours = np.nansum(_gather(np.where(kept, c, np.nan), seg.neighbours), axis=-1)
        precision = np.where(precision > 0, precision, neighbours)
        missing = valid & ~np.isfinite(reference_re)
        if attempt == 1 and missing.any():
            scan_re, _ = _gather_median(np.where(kept, real, np.nan), seg.gather)
            scan_im, _ = _gather_median(np.where(kept, imag, np.nan), seg.gather)
            scan_count = np.add.reduceat(usable, seg.starts, axis=1)[:, seg.index] - usable
            reference_re = np.where(missing, scan_re[:, seg.index], reference_re)
            reference_im = np.where(missing, scan_im[:, seg.index], reference_im)
            precision = np.where(missing, scan_count, precision)
        with np.errstate(invalid="ignore", divide="ignore"):
            # Noise of V - M in units of one sample's: the bin's own n samples and the reference's (a median of
            # the estimators: (pi/2) over their summed precision).
            unit = np.sqrt(1.0 / c + 0.5 * np.pi / np.where(precision > 0, precision, np.nan))
        # Each part's noise is measured on that part: the phases of raw or phase-referenced data scatter more than
        # their amplitudes (atmosphere), and must widen the phase tolerance only.
        along, across, _ = _parts(real - reference_re, imag - reference_im, reference_re, reference_im, phases=phases)
        noise_radial = _bin_noise(along, unit, seg)
        noise_tangential = _bin_noise(across, unit, seg) if phases else noise_radial
        radial, tangential = _components(real - reference_re, imag - reference_im, reference_re, reference_im,
                                         noise_radial, noise_tangential, float(settings.floor),
                                         float(settings.phase_floor), phases=phases)
        if attempt == 0:
            size = np.hypot(radial, tangential) if phases else np.abs(radial)
            usable = np.where(size > threshold, 0, usable)
    if scans:
        level_re, _ = _gather_median(real, seg.gather)
        level_im, _ = _gather_median(imag, seg.gather)
        dep_re, dep_im, factor, trend_re, trend_im, spread_radial, spread_tangential = _local_trend(
            level_re, level_im, seg, int(settings.min_trend_scans), phases=phases)
        enough = np.count_nonzero(np.isfinite(dep_re), axis=1) >= settings.min_scans
        spread_radial = np.where(enough, spread_radial, np.nan)
        spread_tangential = np.where(enough, spread_tangential, np.nan)
        radial, tangential = _components(dep_re, dep_im, trend_re, trend_im, spread_radial[:, None] * factor,
                                         spread_tangential[:, None] * factor, float(settings.scan_floor),
                                         float(settings.phase_floor), phases=phases)
        radial, tangential = radial[:, seg.index], tangential[:, seg.index]
    radial = np.where(valid, radial, np.nan).reshape(shape)
    tangential = np.where(valid, tangential, np.nan).reshape(shape)
    return radial, tangential


def _outlying(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``values`` ``(nant, slots, *cell)`` with the ``mask`` slots made infinite, on the side the other slots of the
    antenna lean to (their median's sign; + when there are none)."""
    lean, _ = _median(np.moveaxis(np.where(mask, np.nan, values), 1, -1))
    side = np.where(lean < 0, -np.inf, np.inf).astype(values.dtype)
    return np.where(mask, side[:, None], values)


def _blame(radial: np.ndarray, tangential: np.ndarray, tables: tuple, threshold: float,
           min_baselines: int, allowed: Optional[np.ndarray] = None,
           known: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Which antenna explains the departures of every cell, and every antenna's statistic.

    Parameters
    ----------
    radial, tangential : numpy.ndarray
        ``(nbl, *cell)`` parts of :func:`_scores` (NaN: no data).
    tables : tuple
        :func:`_antenna_tables`.
    threshold : float
        Radius beyond which an antenna statistic is an outlier.
    min_baselines : int
        Baselines with data an antenna needs in a cell to be blamed there.
    allowed : numpy.ndarray, optional
        bool ``(nant, *cell)``: the only antennas that may be blamed in each cell (None: any).
    known : numpy.ndarray, optional
        bool ``(nbl, *cell)``: baselines that are outliers by definition (a selection; their parts are NaN). To
        blame, each counts as a departure without bound, on the side the antenna's other baselines lean to: one
        such baseline moves the median by one place, so its other baselines decide which of its two antennas is
        at fault, and an antenna most of whose baselines are known is blamed outright.

    Returns
    -------
    blamed : numpy.ndarray
        bool ``(nant, *cell)``.
    antenna_score : numpy.ndarray
        ``(nant, *cell)`` radius of the antenna statistic (medians of both parts over its baselines, the phase
        part sign-turned, times ``sqrt(2 n / pi)``), its baselines to blamed antennas left out (NaN where it has
        fewer than ``min_baselines``); ``known`` baselines do not count in it.
    """
    baselines, partners, signs = tables
    nant = baselines.shape[0]
    cell = radial.shape[1:]
    expand = (slice(None), slice(None)) + (None,) * len(cell)

    def per_antenna(values: np.ndarray) -> np.ndarray:                       # (nant, slots, *cell)
        return np.moveaxis(_gather(np.moveaxis(values, 0, -1), baselines), (-2, -1), (0, 1))

    amplitude, phase = per_antenna(radial), per_antenna(tangential) * signs[expand]
    mine = None if known is None else per_antenna(known.astype(np.float32)) == 1.0

    def statistic(gone: np.ndarray, outliers: bool) -> np.ndarray:
        amplitude_part, phase_part = np.where(gone, np.nan, amplitude), np.where(gone, np.nan, phase)
        if outliers:
            amplitude_part, phase_part = _outlying(amplitude_part, mine & ~gone), _outlying(phase_part, mine & ~gone)
        median_amplitude, n = _median(np.moveaxis(amplitude_part, 1, -1))
        median_phase, _ = _median(np.moveaxis(phase_part, 1, -1))
        with np.errstate(invalid="ignore"):
            score = np.hypot(median_amplitude, median_phase) * np.sqrt(2.0 * n / np.pi)
        return np.where(n >= min_baselines, score, np.nan).astype(np.float32)

    blamed = np.zeros((nant,) + cell, dtype=bool)
    padded = np.concatenate([blamed, np.zeros((1,) + cell, dtype=bool)])      # partner -1 -> never blamed
    antenna_score = np.full(blamed.shape, np.nan, dtype=np.float32)
    for _ in range(nant):
        padded[:-1] = blamed
        gone = padded[partners]
        antenna_score = statistic(gone, mine is not None)
        excluded = blamed if allowed is None else blamed | ~allowed
        candidate = np.where(excluded, -np.inf, np.nan_to_num(antenna_score, nan=-np.inf))
        best = np.argmax(candidate, axis=0)
        top = np.take_along_axis(candidate, best[None], axis=0)[0]
        hit = top > threshold
        if not hit.any():
            break
        blamed |= (np.arange(nant).reshape((nant,) + (1,) * hit.ndim) == best[None]) & hit[None]
    if mine is not None:
        padded[:-1] = blamed
        antenna_score = statistic(padded[partners], False)
    return blamed, antenna_score


def _smooth_time(values: np.ndarray, seg: _Segments, half: int) -> np.ndarray:
    """Running mean over ``2 half + 1`` time bins (last axis) ignoring NaN, inside each scan."""
    if half <= 0:
        return values
    finite = np.isfinite(values)
    data = np.where(finite, values, 0.0)
    total, n = data.copy(), finite.astype(np.float32)
    for k in range(1, half + 1):
        if k >= values.shape[-1]:
            break
        same = seg.index[k:] == seg.index[:-k]
        total[..., k:] += np.where(same, data[..., :-k], 0.0)
        n[..., k:] += np.where(same, finite[..., :-k], 0)
        total[..., :-k] += np.where(same, data[..., k:], 0.0)
        n[..., :-k] += np.where(same, finite[..., k:], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, total / n, np.nan).astype(np.float32)


def _runs_touching(candidate: np.ndarray, mask: np.ndarray, breaks: Optional[np.ndarray]) -> np.ndarray:
    """Every run of ``candidate`` along the last axis that holds a ``mask`` cell (``breaks``: forced run starts)."""
    start = candidate.copy()
    start[..., 1:] &= ~candidate[..., :-1] if breaks is None else (~candidate[..., :-1] | breaks[1:])
    if not candidate.size:
        return candidate.copy()
    labels = np.cumsum(start.reshape(-1)).reshape(candidate.shape) * candidate
    hit = np.zeros(int(labels.max()) + 1, dtype=bool)
    hit[labels[mask & candidate]] = True
    hit[0] = False
    return hit[labels]


def _grow(seeds: np.ndarray, candidate: np.ndarray, seg: _Segments) -> np.ndarray:
    """Hysteresis growth of ``seeds`` through ``candidate`` along time (last axis, within scans) and blocks (axis 1).

    Arrays are ``(nant, nblock, ncorr, ntime)``.
    """
    candidate = candidate | seeds
    mask = seeds.copy()
    for _ in range(8):
        grown = _runs_touching(candidate, mask, seg.first)
        grown = np.moveaxis(_runs_touching(np.moveaxis(candidate, 1, -1), np.moveaxis(grown, 1, -1), None), -1, 1)
        if np.array_equal(grown, mask):
            break
        mask = grown
    return mask


def _dilate_time(mask: np.ndarray, seg: _Segments, bins: int) -> np.ndarray:
    """``mask`` widened by ``bins`` along time (last axis), inside each scan."""
    out = mask.copy()
    for k in range(1, bins + 1):
        if k >= mask.shape[-1]:
            break
        same = seg.index[k:] == seg.index[:-k]
        out[..., k:] |= mask[..., :-k] & same
        out[..., :-k] |= mask[..., k:] & same
    return out


def _broadband(flags: np.ndarray, has_data: np.ndarray, fraction: float) -> np.ndarray:
    """Extend antenna flags ``(nant, nspw, nblock, ncorr, ntime)`` to the whole subband / every subband.

    At one time, an antenna flagged in more than ``fraction`` of the channel blocks of a subband where it has
    data loses the subband; flagged so in more than ``fraction`` of its subbands, it loses all of them.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        blocks = (flags & has_data).sum(axis=2) / has_data.sum(axis=2)            # (nant, nspw, ncorr, ntime)
    subband = np.nan_to_num(blocks) > fraction
    if flags.shape[2] > 1:
        flags = flags | (subband[:, :, None] & has_data)
    if flags.shape[1] > 1:
        present = has_data.any(axis=2)                                            # (nant, nspw, ncorr, ntime)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = (subband & present).sum(axis=1) / present.sum(axis=1)         # (nant, ncorr, ntime)
        everywhere = np.nan_to_num(share) > fraction
        flags = flags | (everywhere[:, None, None] & has_data)
    return flags


def find_outliers(grid: VisGrid, settings: Optional[OutlierSettings] = None) -> OutlierResult:
    """Find the faulty antennas (and baselines) of one source and the time/frequency ranges to flag.

    Parameters
    ----------
    grid : VisGrid
        The binned visibilities.
    settings : OutlierSettings, optional
        Thresholds and windows (defaults: :class:`OutlierSettings`).

    Returns
    -------
    OutlierResult
        The antenna and baseline flags (nothing is flagged in cells without data). With a selection
        (``grid.selected``), every selected cell with data is flagged, by an antenna or on its own baseline.
    """
    settings = settings or OutlierSettings()
    nbl, nspw, nblock, ncorr, ntime = grid.vis.shape
    antenna_flags = np.zeros((grid.nant, nspw, nblock, ncorr, ntime), dtype=bool)
    baseline_flags = np.zeros(grid.vis.shape, dtype=bool)
    result = OutlierResult(antenna_flags, baseline_flags, 0, [], settings.to_dict())
    if nbl == 0 or ntime == 0:
        return result
    seg = _segments(grid.times, grid.segment, float(settings.trend_minutes) * 60.0, int(settings.level_bins))
    tables = _antenna_tables(grid.ant1, grid.ant2, grid.nant)
    phases = bool(settings.use_phases)
    threshold = _limit(settings.sigma, phases)
    blame = _limit(settings.antenna_sigma, phases)
    edge = _limit(settings.edge_sigma, phases)
    single = _limit(settings.baseline_sigma, phases) if settings.baseline_sigma > 0 else np.inf
    has_data = _per_antenna_any(grid.count > 0, grid.ant1, grid.ant2, grid.nant)  # (nant, nspw, nblock, ncorr, ntime)
    if grid.selected is not None:
        touched_all = _per_antenna_any(grid.selected & (grid.count > 0), grid.ant1, grid.ant2, grid.nant)
        blame = _limit(selection_sigma(settings, int(has_data.sum()), int(touched_all.sum())), phases)
    for spw in range(nspw):
        vis, count = grid.vis[:, spw], grid.count[:, spw]
        selected = grid.selected[:, spw] & (count > 0) if grid.selected is not None else None
        scans = False
        for iteration in range(1, int(settings.max_iterations) + 1):
            flagged = baseline_flags[:, spw] | antenna_flags[grid.ant1, spw] | antenna_flags[grid.ant2, spw]
            # A selection is outliers already: out of every reference, level and noise, as if flagged, and counted
            # by `_blame` as departures without bound. Only the antennas of a selected baseline not explained yet,
            # and only there, may be blamed - at `selection_sigma`, since the test runs on those cells alone.
            known = None if selected is None else selected & ~flagged
            radial, tangential = _scores(vis, count, flagged if known is None else flagged | known, seg, settings,
                                         scans=scans, threshold=threshold)
            if not phases:
                tangential = np.where(np.isfinite(radial), 0.0, np.nan).astype(np.float32)
            score = np.hypot(radial, tangential)
            allowed = None if known is None else _per_antenna_any(known, grid.ant1, grid.ant2, grid.nant)
            seeds, antenna_score = _blame(radial, tangential, tables, blame, int(settings.min_baselines), allowed,
                                          known)
            smoothed = _smooth_time(antenna_score, seg, int(settings.smooth_bins))
            candidate = (smoothed > edge) | antenna_flags[:, spw]
            grown = _dilate_time(_grow(seeds, candidate, seg), seg, int(settings.margin_bins))
            new_antenna = grown & has_data[:, spw] & ~antenna_flags[:, spw]
            covered = flagged | new_antenna[grid.ant1] | new_antenna[grid.ant2]
            # The selected cells no antenna explains are flagged on their own baselines once the antennas are done:
            # until then they still count for their antennas' statistics.
            new_baseline = (score > single) & ~covered if selected is None else np.zeros_like(covered)
            antenna_flags[:, spw] |= new_antenna
            baseline_flags[:, spw] |= new_baseline
            result.history.append({"spw": spw, "iteration": iteration, "scans": scans,
                                   "outliers": int((score > threshold).sum()),
                                   "antenna_cells": int(new_antenna.sum()), "baseline_cells": int(new_baseline.sum())})
            result.iterations = max(result.iterations, iteration)
            logger.debug("outliers: subband %d fit %d (%s): %d outlier cell(s), %d antenna and %d baseline cell(s) "
                         "newly flagged", spw, iteration, "bins and scans" if scans else "bins",
                         result.history[-1]["outliers"], result.history[-1]["antenna_cells"],
                         result.history[-1]["baseline_cells"])
            if new_antenna.any() or new_baseline.any():
                scans = False                    # something moved: bins first again
            elif scans:
                break                            # neither bins nor scans have anything left
            else:
                scans = True
    result.antenna_flags = _broadband(antenna_flags, has_data, float(settings.broadband_fraction))
    if grid.selected is not None:
        selected = grid.selected & (grid.count > 0)
        result.baseline_flags |= selected & ~result.antenna_flags[grid.ant1] & ~result.antenna_flags[grid.ant2]
    return result
