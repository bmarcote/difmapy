"""Diagnostic plots: closure phases (cpplot), time sampling (tplot),
self-cal corrections (corplot) and spectra (specplot)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from difmapy.plots.base import (
    FLAG_COLOR,
    PlotWindow,
    gradient_colors,
    run_if_needed,
)

__all__ = ["cpplot", "tplot", "corplot", "specplot"]

RAD2DEG = 180.0 / np.pi


class _MultiPanel(PlotWindow):
    """A page of stacked panels with keyboard paging (n/p)."""

    def __init__(self, obs, title, nplot=4):
        super().__init__(title)
        self.obs = obs
        self.nplot = nplot
        self.page = 0
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.setCentralWidget(self.glw)
        self.statusBar().showMessage("n: next page | p: previous page | h: help")
        self.refresh()

    @property
    def npages(self) -> int:
        return max(1, int(np.ceil(len(self._items) / self.nplot)))

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
        for row, (tri, series) in enumerate(self._items[lo : lo + self.nplot]):
            p = self.glw.addPlot(row=row, col=0)
            label = "-".join(names[i] for i in tri)
            p.setLabel("left", f"{label} (deg)")
            p.setYRange(-180, 180)
            p.showGrid(y=True, alpha=0.2)
            for s in series:
                cif = s["if_index"]
                col = colors[cif % len(colors)]
                t = np.asarray(s["time"]) / 3600.0
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
            if row == min(self.nplot, len(self._items) - lo) - 1:
                p.setLabel("bottom", "Time (hours)")
        self.statusBar().showMessage(
            f"page {self.page + 1}/{self.npages} - {len(self._items)} triangles "
            "| n/p: page | red +: model"
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
        core = obs._core
        samp = np.asarray(core.sampling())  # [ntimes, nant]
        times = np.asarray(core.times()) / 3600.0
        names = list(core.antenna_names)
        colors = gradient_colors(len(names), "turbo")
        self.plot = p = self.glw.addPlot()
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
    """Self-calibration corrections vs time (difmap corplot)."""

    def __init__(self, obs, quantity="both", nplot=4):
        self.quantity = quantity
        self._items = []
        super().__init__(obs, "difmapy corplot", nplot=nplot)

    def refresh(self):
        core = self.obs._core
        names = core.antenna_names
        nant, nif = len(names), self.obs.nif
        amp, phs, bad = core.gains()
        ntimes = core.ntimes
        amp = np.asarray(amp).reshape(ntimes, nif, nant)
        phs = np.asarray(phs).reshape(ntimes, nif, nant)
        bad = np.asarray(bad).reshape(ntimes, nif, nant)
        times = np.asarray(core.times()) / 3600.0
        if not self._items:
            # Only show antennas that were actually corrected.
            self._items = [
                (ia, names[ia])
                for ia in range(nant)
                if np.any(amp[:, :, ia] != 1.0) or np.any(phs[:, :, ia] != 0.0)
            ] or [(ia, names[ia]) for ia in range(nant)]
        self.glw.clear()
        colors = gradient_colors(nif)
        lo = self.page * self.nplot
        for row, (ia, name) in enumerate(self._items[lo : lo + self.nplot]):
            p = self.glw.addPlot(row=row, col=0)
            p.setLabel("left", name)
            p.showGrid(y=True, alpha=0.2)
            for cif in range(nif):
                col = colors[cif % len(colors)]
                good = ~bad[:, cif, ia]
                y = (
                    phs[:, cif, ia] * RAD2DEG
                    if self.quantity == "phase"
                    else amp[:, cif, ia]
                )
                p.addItem(
                    pg.ScatterPlotItem(
                        times[good], y[good], size=4, pen=None,
                        brush=pg.mkBrush(*col, 200),
                    )
                )
                if (~good).any():
                    p.addItem(
                        pg.ScatterPlotItem(
                            times[~good], y[~good], size=6, symbol="x",
                            pen=pg.mkPen(*FLAG_COLOR), brush=None,
                        )
                    )
            if row == min(self.nplot, len(self._items) - lo) - 1:
                p.setLabel("bottom", "Time (hours)")
        what = "phase (deg)" if self.quantity == "phase" else "amplitude"
        self.statusBar().showMessage(
            f"gain {what} - page {self.page + 1}/{self.npages} | n/p: page "
            "| red x: flagged solution"
        )


class SpecPlot(PlotWindow):
    """Time-averaged spectrum (difmap specplot)."""

    def __init__(self, obs, baseline=None, tmin=None, tmax=None, xaxis="freq"):
        super().__init__("difmapy specplot")
        self.obs = obs
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground("w")
        self.setCentralWidget(self.glw)
        core = obs._core
        bl = None
        if baseline is not None:
            names = core.antenna_names
            bl = (names.index(str(baseline[0])), names.index(str(baseline[1])))
        s = core.spectrum(baseline=bl, tmin=tmin, tmax=tmax)
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
        label = "all baselines" if baseline is None else "-".join(map(str, baseline))
        self.statusBar().showMessage(
            f"vector-averaged spectrum ({label}) | h: help"
        )


def cpplot(obs, triangles=None, if_index=None, nplot=4, block=None):
    p = CpPlot(obs, triangles=triangles, if_index=if_index, nplot=nplot)
    run_if_needed(p, block)
    return p


def tplot(obs, block=None):
    p = TPlot(obs)
    run_if_needed(p, block)
    return p


def corplot(obs, quantity="phase", nplot=4, block=None):
    p = CorPlot(obs, quantity=quantity, nplot=nplot)
    run_if_needed(p, block)
    return p


def specplot(obs, baseline=None, tmin=None, tmax=None, xaxis="freq", block=None):
    p = SpecPlot(obs, baseline=baseline, tmin=tmin, tmax=tmax, xaxis=xaxis)
    run_if_needed(p, block)
    return p
