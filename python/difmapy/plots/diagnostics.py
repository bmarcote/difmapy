"""Diagnostic plots: closure phases (cpplot), time sampling (tplot),
self-cal corrections (corplot) and spectra (specplot, fplot)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from difmapy.plots.base import (
    FLAG_COLOR,
    PlotWindow,
    TimeGaps,
    gradient_colors,
    install_time_axis,
    run_if_needed,
)
from difmapy.units import parse_time

__all__ = ["cpplot", "tplot", "corplot", "specplot", "fplot"]

RAD2DEG = 180.0 / np.pi


def _cor_quantities(spec):
    """Which corplot panels a `quantity=` spec asks for, top to bottom."""
    key = str(spec).lower().replace(" ", "")
    if key in ("both", "ap", "anp", "a&p", "ampphase", "amp&phase"):
        return ("amp", "phase")
    if key in ("amp", "amplitude", "a", "gain"):
        return ("amp",)
    if key in ("phase", "phs", "p"):
        return ("phase",)
    raise ValueError(
        f"unknown quantity {spec!r}; use 'amp', 'phase' or 'both'"
    )


class _MultiPanel(PlotWindow):
    """A page of stacked panels with keyboard paging (n/p)."""

    def __init__(self, obs, title, nplot=4):
        super().__init__(title)
        self.obs = obs
        self.nplot = nplot
        self.page = 0
        #: (plot, default y range) of every panel on the current page,
        #: for the z key and for x-axis sharing.
        self._plots: list = []
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.setCentralWidget(self.glw)
        self.statusBar().showMessage("n: next page | p: previous page | h: help")
        self.refresh()

    @property
    def npages(self) -> int:
        return max(1, int(np.ceil(len(self._items) / self.nplot)))

    @staticmethod
    def _set_amp_range(panel, values):
        """Range an amplitude panel around its data, with a floor.

        A flat band - or a single gain solution - is flat to the last
        bit of float32, and autoscaling to that shows nothing but
        rounding noise under ten-digit tick labels. Never range tighter
        than a thousandth of the level. Returns the range applied, for
        `z` to restore.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        lo, hi = float(values.min()), float(values.max())
        pad = max(0.05 * (hi - lo), 1e-3 * abs(0.5 * (lo + hi)), 1e-12)
        panel.setYRange(lo - pad, hi + pad, padding=0)
        return (lo - pad, hi + pad)

    def _new_plot(self, row, yrange=None):
        """Add a panel, recording the y range ``z`` should restore."""
        p = self.glw.addPlot(row=row, col=0)
        self._plots.append((p, yrange))
        return p

    def _share_x_axis(self):
        """Tie every panel of the page to one x axis, with the tick
        labels only on the bottom one."""
        plots = [p for p, _ in self._plots]
        for p in plots[1:]:
            p.setXLink(plots[0])
        for p in plots[:-1]:
            p.getAxis("bottom").setStyle(showValues=False)
            p.setLabel("bottom", "")

    def view_boxes(self):
        return [p.vb for p, _ in self._plots]

    def default_y_range(self, vb):
        for p, yrange in self._plots:
            if p.vb is vb:
                return yrange
        return None

    def key_help(self):
        return [("n / p", "next / previous page")]

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "n":
            self.page = (self.page + 1) % self.npages
            self.refresh()
        elif key == "p":
            self.page = (self.page - 1) % self.npages
            self.refresh()
        else:
            super().keyPressEvent(ev)


class CpPlot(_MultiPanel):
    """Closure phase vs time for antenna triangles (difmap cpplot)."""

    def __init__(self, obs, triangles=None, if_index=None, nplot=4):
        self._triangles = triangles
        self._if_index = if_index
        self._items = []
        super().__init__(obs, "difmapy cpplot", nplot=nplot)

    def refresh(self):
        core = self.obs._core
        if not self._items:
            names = core.antenna_names
            data = []
            if self._triangles is None:
                data = core.closure_phases(if_index=self._if_index)
            else:
                for tri in self._triangles:
                    idx = tuple(sorted(names.index(str(t)) for t in tri))
                    data += core.closure_phases(triangle=idx, if_index=self._if_index)
            # Group the IFs of each triangle into one panel.
            grouped: dict[tuple, list] = {}
            for d in data:
                grouped.setdefault(tuple(d["triangle"]), []).append(d)
            self._items = sorted(grouped.items())
        self.glw.clear()
        names = core.antenna_names
        colors = gradient_colors(self.obs.nif)
        lo = self.page * self.nplot
        self._plots = []
        gaps = TimeGaps.of(self.obs)
        for row, (tri, series) in enumerate(self._items[lo : lo + self.nplot]):
            p = self._new_plot(row, (-180, 180))
            install_time_axis(p, gaps)
            label = "-".join(names[i] for i in tri)
            p.setLabel("left", f"{label} (deg)")
            p.setYRange(-180, 180)
            p.showGrid(y=True, alpha=0.2)
            for s in series:
                cif = s["if_index"]
                col = colors[cif % len(colors)]
                t = gaps.compress(np.asarray(s["time"]) / 3600.0)
                p.addItem(
                    pg.ScatterPlotItem(
                        t, np.asarray(s["phase"]) * RAD2DEG, size=4,
                        pen=None, brush=pg.mkBrush(*col, 200),
                    )
                )
                m = np.asarray(s["model"]) * RAD2DEG
                if np.isfinite(m).any():
                    p.addItem(
                        pg.ScatterPlotItem(
                            t, m, size=5, symbol="+", pen=pg.mkPen(200, 30, 30), brush=None
                        )
                    )
        self._share_x_axis()
        if self._plots:
            self._plots[-1][0].setLabel("bottom", "Time (hours)")
        self.statusBar().showMessage(
            f"page {self.page + 1}/{self.npages} - {len(self._items)} triangles "
            "| n/p: page | red +: model | h: help"
        )


class TPlot(PlotWindow):
    """Per-antenna time sampling (difmap tplot).

    One row per antenna, labelled by name, showing where each has
    unflagged data.
    """

    def __init__(self, obs):
        super().__init__("difmapy tplot")
        self.obs = obs
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.setCentralWidget(self.glw)
        self.plot = None
        self.refresh()

    def view_boxes(self):
        return [] if self.plot is None else [self.plot.vb]

    def refresh(self):
        self.glw.clear()
        obs = self.obs
        core = obs._core
        samp = np.asarray(core.sampling())  # [ntimes, nant]
        gaps = TimeGaps.of(obs)
        times = gaps.compress(np.asarray(core.times()) / 3600.0)
        names = list(core.antenna_names)
        colors = gradient_colors(len(names), "turbo")
        self.plot = p = self.glw.addPlot()
        install_time_axis(p, gaps)
        p.setLabel("bottom", "Time (hours)")
        p.setLabel("left", "Antenna")
        p.showGrid(x=True, alpha=0.2)
        nsamp = []
        for ia, name in enumerate(names):
            good = samp[:, ia] > 0
            nsamp.append(int(good.sum()))
            if not good.any():
                continue
            p.addItem(
                pg.ScatterPlotItem(
                    times[good], np.full(good.sum(), ia), size=5, symbol="s",
                    pen=None, brush=pg.mkBrush(*colors[ia % len(colors)], 220),
                )
            )
        ax = p.getAxis("left")
        # Label the rows with the station names rather than their index.
        ax.setTicks([[(i, n) for i, n in enumerate(names)], []])
        ax.setWidth(max(48, 9 * max((len(n) for n in names), default=4)))
        p.setYRange(-0.5, len(names) - 0.5)
        dead = [n for n, c in zip(names, nsamp) if c == 0]
        self.statusBar().showMessage(
            "unflagged data per antenna vs time"
            + (f" | no data: {', '.join(dead)}" if dead else "")
            + " | h: help"
        )

    def key_help(self):
        return [("", "one row per antenna; a mark means unflagged data")]


class CorPlot(_MultiPanel):
    """Self-calibration corrections vs time (difmap corplot).

    One antenna to a pair of panels - amplitude above, phase below,
    sharing the time axis, as `vplot` lays out its baselines. `quantity`
    narrows that to "amp" or "phase" alone; the default shows both,
    since an amplitude solution is usually read against its phase.
    """

    #: Tall: a page is several antennas of stacked amp/phase panels.
    DEFAULT_SIZE = (1200, 950)

    def __init__(self, obs, quantity="both", nplot=3):
        self.quantities = _cor_quantities(quantity)
        self._items = []
        super().__init__(obs, "difmapy corplot", nplot=nplot)

    @property
    def quantity(self):
        """What is displayed: "amp", "phase" or "both"."""
        return "both" if len(self.quantities) == 2 else self.quantities[0]

    def refresh(self):
        core = self.obs._core
        names = core.antenna_names
        nant, nif = len(names), self.obs.nif
        amp, phs, bad = core.gains()
        ntimes = core.ntimes
        amp = np.asarray(amp).reshape(ntimes, nif, nant)
        phs = np.asarray(phs).reshape(ntimes, nif, nant)
        bad = np.asarray(bad).reshape(ntimes, nif, nant)
        # An integration a station has no solution for reads 1.0 / 0 deg
        # in the table, which is not a correction and must not be drawn
        # as one: plotting it made a smooth run of corrections look like
        # it jumped to unity and back.
        used = np.asarray(core.gains_used()).reshape(ntimes, nif, nant)
        gaps = TimeGaps.of(self.obs)
        times = gaps.compress(np.asarray(core.times()) / 3600.0)
        if not self._items:
            # Only show antennas that were actually corrected.
            self._items = [
                (ia, names[ia]) for ia in range(nant) if np.any(used[:, :, ia])
            ] or [(ia, names[ia]) for ia in range(nant)]
        self.glw.clear()
        self._plots = []
        colors = gradient_colors(nif)
        lo = self.page * self.nplot
        page = self._items[lo : lo + self.nplot]
        for i, (ia, name) in enumerate(page):
            for j, key in enumerate(self.quantities):
                yrange = (-180, 180) if key == "phase" else None
                p = self._new_plot(i * len(self.quantities) + j, yrange)
                install_time_axis(p, gaps)
                label = "Amplitude" if key == "amp" else "Phase (deg)"
                p.setLabel("left", f"{name}<br>{label}")
                p.getAxis("left").setWidth(82)
                # A gain is dimensionless and sits near 1: pyqtgraph's
                # SI prefixing would render 0.758 as "758.5 (x0.001)".
                p.getAxis("left").enableAutoSIPrefix(False)
                p.showGrid(x=True, y=True, alpha=0.15)
                if yrange is not None:
                    p.setYRange(*yrange)
                shown = []
                for cif in range(nif):
                    col = colors[cif % len(colors)]
                    solved = used[:, cif, ia]
                    good = solved & ~bad[:, cif, ia]
                    flagged = solved & bad[:, cif, ia]
                    y = (phs[:, cif, ia] * RAD2DEG if key == "phase"
                         else amp[:, cif, ia])
                    if good.any():
                        shown.append(y[good])
                        # A line broken wherever there is no solution:
                        # with a solution interval set, self-cal
                        # interpolates its bins onto the integrations
                        # (difmap's apply_solns), so the corrections
                        # really are a smooth curve - draw them as one.
                        p.addItem(
                            pg.PlotDataItem(
                                times, np.where(good, y, np.nan),
                                connect="finite",
                                pen=pg.mkPen(*col, 160, width=1.2),
                                symbol="o", symbolSize=4, symbolPen=None,
                                symbolBrush=pg.mkBrush(*col, 220),
                            )
                        )
                    if flagged.any():
                        p.addItem(
                            pg.ScatterPlotItem(
                                times[flagged], y[flagged], size=6, symbol="x",
                                pen=pg.mkPen(*FLAG_COLOR), brush=None,
                            )
                        )
                if key == "amp" and shown:
                    # A single solution is constant to the last bit of
                    # float32; keep the panel off that scale.
                    self._plots[-1] = (
                        p, self._set_amp_range(p, np.concatenate(shown)),
                    )
        self._share_x_axis()
        if self._plots:
            self._plots[-1][0].setLabel("bottom", "Time (hours)")
        what = {"both": "amplitude and phase", "amp": "amplitude",
                "phase": "phase"}[self.quantity]
        nogap = ""
        if page:
            missing = sum(int((~used[:, :, ia]).all(axis=1).sum())
                          for ia, _ in page)
            if missing:
                nogap = f" | {missing} integrations without a solution (not drawn)"
        self.statusBar().showMessage(
            f"gain {what} - page {self.page + 1}/{self.npages} of "
            f"{len(self._items)} antennas | n/p: page "
            f"| red x: flagged solution{nogap} | h: help"
        )


class SpecPlot(PlotWindow):
    """Time-averaged spectrum (difmap specplot)."""

    def __init__(self, obs, baseline=None, tmin=None, tmax=None, xaxis="freq"):
        super().__init__("difmapy specplot")
        self.obs = obs
        self.baseline = baseline
        self.tmin = parse_time(tmin, "s")
        self.tmax = parse_time(tmax, "s")
        self.xaxis = xaxis
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.setCentralWidget(self.glw)
        self._plots = []
        self.refresh()

    def view_boxes(self):
        return [p.vb for p in self._plots]

    def default_y_range(self, vb):
        return (-180, 180) if self._plots[-1:] and vb is self._plots[-1].vb else None

    def refresh(self):
        self.glw.clear()
        obs, baseline, xaxis = self.obs, self.baseline, self.xaxis
        core = obs._core
        bl = None
        if baseline is not None:
            names = core.antenna_names
            bl = (names.index(str(baseline[0])), names.index(str(baseline[1])))
        s = core.spectrum(baseline=bl, tmin=self.tmin, tmax=self.tmax)
        wt = np.asarray(s["wt"])
        good = wt > 0
        x = np.asarray(s["freq"]) / 1e9 if xaxis == "freq" else np.asarray(s["chan"])
        amp = np.asarray(s["amp"])
        phase = np.rad2deg(np.arctan2(np.asarray(s["im"]), np.asarray(s["re"])))
        pa = self.glw.addPlot(row=0, col=0)
        pa.setLabel("left", "Amplitude (Jy)")
        pa.addItem(pg.PlotDataItem(x[good], amp[good],
                                   pen=pg.mkPen(31, 119, 180, width=2),
                                   symbol="o", symbolSize=4))
        pp = self.glw.addPlot(row=1, col=0)
        pp.setLabel("left", "Phase (deg)")
        pp.setLabel("bottom", "Frequency (GHz)" if xaxis == "freq" else "Channel")
        pp.setYRange(-180, 180)
        pp.addItem(pg.PlotDataItem(x[good], phase[good],
                                   pen=pg.mkPen(255, 127, 14, width=2),
                                   symbol="o", symbolSize=4))
        pp.setXLink(pa)
        pa.getAxis("bottom").setStyle(showValues=False)
        self._plots = [pa, pp]
        label = "all baselines" if baseline is None else "-".join(map(str, baseline))
        self.statusBar().showMessage(
            f"vector-averaged spectrum ({label}) | h: help"
        )


class FPlot(_MultiPanel):
    """Amplitude and phase against frequency, one baseline to a pair of
    panels, averaged over the whole time range (difmapy fplot).

    The frequency counterpart of `vplot`: every channel of every IF is
    averaged over the observation - or over `tmin`..`tmax` - so that a
    slope or a ripple across the band stands out where a single
    integration would only show noise. `n`/`p` page through the
    baselines, `reftel` restricts them to one station's.

    Amplitudes are scalar-averaged, so they do not decorrelate as the
    fringe turns; phases come from the vector average, which is what
    makes a phase slope across the band visible. The accumulated
    calibration is applied (`calibrated=False` shows the data as
    loaded), and channels outside the current selection are shown too,
    so the plot can be used to choose them.

    This is a view rather than an editor: each point averages many
    integrations, while flag editing works on whole rows.
    """

    def __init__(self, obs, reftel=None, baselines=None, nplot=3,
                 tmin=None, tmax=None, calibrated=True):
        self.tmin = parse_time(tmin, "s")
        self.tmax = parse_time(tmax, "s")
        self.calibrated = bool(calibrated)
        self._items = self._select_baselines(obs, reftel, baselines)
        title = "difmapy fplot"
        if reftel is not None:
            title += f" ({obs.antennas[self._ant(obs, reftel)]})"
        super().__init__(obs, title, nplot=nplot)

    @staticmethod
    def _ant(obs, name):
        return int(name) if isinstance(name, int) else obs.antennas.index(str(name))

    @classmethod
    def _select_baselines(cls, obs, reftel, baselines):
        """The (a1, a2) pairs to page through, as global antenna
        indices."""
        if baselines is not None:
            return [tuple(sorted(cls._ant(obs, a) for a in bl)) for bl in baselines]
        _, a1, a2, _, _, _ = obs._core.rows()
        pairs = sorted({(int(a), int(b)) for a, b in zip(a1, a2)})
        if reftel is not None:
            ref = cls._ant(obs, reftel)
            pairs = [p for p in pairs if ref in p]
        return pairs

    def refresh(self):
        self.glw.clear()
        self._plots = []
        names = self.obs.antennas
        colors = gradient_colors(self.obs.nif)
        # Channel -> IF, so each band can be drawn as its own curve.
        nchan = list(self.obs.nchan)
        cif = np.concatenate(
            [np.full(n, i, dtype=int) for i, n in enumerate(nchan)]
        ) if nchan else np.zeros(0, dtype=int)
        lo = self.page * self.nplot
        page = self._items[lo : lo + self.nplot]
        panels = []
        nplotted = 0
        for i, (a, b) in enumerate(page):
            label = f"{names[a]}-{names[b]}"
            s = self.obs._core.spectrum(
                baseline=(a, b), tmin=self.tmin, tmax=self.tmax,
                calibrated=self.calibrated,
            )
            wt = np.asarray(s["wt"])
            freq = np.asarray(s["freq"]) / 1e9
            amp = np.asarray(s["amp"])
            phase = np.rad2deg(np.arctan2(np.asarray(s["im"]), np.asarray(s["re"])))
            for j, (key, y, ylab) in enumerate(
                (("amp", amp, "Amplitude (Jy)"), ("phase", phase, "Phase (deg)"))
            ):
                p = self._new_plot(2 * i + j,
                                   (-180, 180) if key == "phase" else None)
                p.setLabel("left", f"{label}<br>{ylab}")
                # Amplitudes over a narrow range need several digits;
                # pyqtgraph drops the tick labels rather than widen the
                # axis for them, so make room up front.
                p.getAxis("left").setWidth(82)
                p.showGrid(x=True, y=True, alpha=0.15)
                if key == "phase":
                    p.setYRange(-180, 180)
                for band in range(self.obs.nif):
                    m = (cif == band) & (wt > 0)
                    if not m.any():
                        continue
                    col = colors[band % len(colors)]
                    p.addItem(
                        pg.PlotDataItem(
                            freq[m], y[m], pen=pg.mkPen(*col, width=1.5),
                            symbol="o", symbolSize=4, symbolPen=None,
                            symbolBrush=pg.mkBrush(*col, 220),
                        )
                    )
                    nplotted += int(m.sum())
                if key == "amp":
                    # Remember the floored range, so "z" comes back to
                    # it instead of autoscaling onto rounding noise.
                    self._plots[-1] = (p, self._set_amp_range(p, y[wt > 0]))
                panels.append(p)
        self._share_x_axis()
        if panels:
            panels[-1].setLabel("bottom", "Frequency (GHz)")
        self._panels = panels
        window = (
            "the whole observation" if self.tmin is None and self.tmax is None
            else f"{(self.tmin or 0) / 3600.0:.3g}-{(self.tmax or 0) / 3600.0:.3g} h"
        )
        cal = "calibrated" if self.calibrated else "as loaded"
        self.statusBar().showMessage(
            f"page {self.page + 1}/{self.npages} of {len(self._items)} baselines "
            f"| averaged over {window} ({cal}), {nplotted} channels drawn "
            "| n/p: page | h: help"
        )

    def key_help(self):
        return super().key_help() + [
            ("", "amplitudes scalar-averaged, phases vector-averaged"),
            ("", "colours are the IFs, in order"),
        ]


def cpplot(obs, triangles=None, if_index=None, nplot=4, block=None):
    p = CpPlot(obs, triangles=triangles, if_index=if_index, nplot=nplot)
    run_if_needed(p, block)
    return p


def tplot(obs, block=None):
    p = TPlot(obs)
    run_if_needed(p, block)
    return p


def corplot(obs, quantity="both", nplot=3, block=None):
    p = CorPlot(obs, quantity=quantity, nplot=nplot)
    run_if_needed(p, block)
    return p


def specplot(obs, baseline=None, tmin=None, tmax=None, xaxis="freq", block=None):
    p = SpecPlot(obs, baseline=baseline, tmin=tmin, tmax=tmax, xaxis=xaxis)
    run_if_needed(p, block)
    return p


def fplot(obs, reftel=None, baselines=None, nplot=3, tmin=None, tmax=None,
          calibrated=True, block=None):
    p = FPlot(obs, reftel=reftel, baselines=baselines, nplot=nplot,
              tmin=tmin, tmax=tmax, calibrated=calibrated)
    run_if_needed(p, block)
    return p
