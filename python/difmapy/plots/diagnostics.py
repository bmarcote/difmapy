"""Diagnostic plots: closure phases (cpplot), time sampling (tplot), self-cal corrections (corplot) and spectra (specplot, fplot, possm)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from difmapy.plots.base import (
    FLAG_COLOR,
    FastScatter,
    FluxAxis,
    PlotWindow,
    SelectViewBox,
    TimeGaps,
    frame_panels,
    gradient_colors,
    install_time_axis,
    run_if_needed,
)
from difmapy.units import parse_time

#: Left-axis width of possm's left column, the only one with labels.
LEFT_AXIS_WIDTH = 64
#: Width (and height) of the label-less axes inside possm's grid: just
#: enough to carry the baseline's frame.
INNER_AXIS_WIDTH = 4
#: Height of the baseline title over each possm cell.
TITLE_HEIGHT = 16

__all__ = ["cpplot", "tplot", "corplot", "specplot", "fplot", "possm"]

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

        A flat band - or a single gain solution - is flat to the last bit of float32, and autoscaling to that shows nothing but rounding noise under ten-digit tick labels. Never range tighter than a thousandth of the level. Returns the range applied, for `z` to restore.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return None
        lo, hi = float(values.min()), float(values.max())
        pad = max(0.05 * (hi - lo), 1e-3 * abs(0.5 * (lo + hi)), 1e-12)
        panel.setYRange(lo - pad, hi + pad, padding=0)
        return (lo - pad, hi + pad)

    def _new_plot(self, row, yrange=None, flux=False):
        """Add a panel, recording the y range ``z`` should restore; `flux` gives it a left axis in Jy/mJy/µJy (`FluxAxis`)."""
        axes = {"left": FluxAxis()} if flux else None
        p = self.glw.addPlot(row=row, col=0, axisItems=axes)
        self._plots.append((p, yrange))
        return p

    def _share_x_axis(self):
        """Tie every panel of the page to one x axis, with the tick labels only on the bottom one."""
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
            self.set_page(self.page + 1)
        elif key == "p":
            self.set_page(self.page - 1)
        else:
            super().keyPressEvent(ev)

    def set_page(self, page):
        """Show page `page` (counting from 0, wrapping around)."""
        self.page = int(page) % self.npages
        self.refresh()
        return self.page


class CpPlot(_MultiPanel):
    """Closure phase vs time for antenna triangles (difmap cpplot)."""

    def __init__(self, obs, triangles=None, if_index=None, nplot=4):
        self._triangles = triangles
        self._if_index = if_index
        self._items = []
        super().__init__(obs, "difmapy cpplot", nplot=nplot)

    def _triangle_list(self):
        """The triangles to page through, as sorted antenna indices.

        Worked out from which baselines have unflagged data, without computing any closure phase: those are computed a page at a time, since a 28-station array has 3276 triangles (seconds of work for all of them) and a page shows four.
        """
        core = self.obs._core
        names = core.antenna_names
        if self._triangles is not None:
            return [tuple(sorted(self.obs._ant_index(t) for t in tri))
                    for tri in self._triangles]
        _, wt = core.stream_vis()
        _, a1, a2, *_ = core.rows()
        good = (np.asarray(wt) > 0).any(axis=1)
        a1, a2 = np.asarray(a1)[good], np.asarray(a2)[good]
        pairs = set(zip(np.minimum(a1, a2).tolist(), np.maximum(a1, a2).tolist()))
        ants = sorted({a for pair in pairs for a in pair})
        out = []
        for i, a in enumerate(ants):
            for j in range(i + 1, len(ants)):
                b = ants[j]
                if (a, b) not in pairs:
                    continue
                out.extend((a, b, c) for c in ants[j + 1:]
                           if (a, c) in pairs and (b, c) in pairs)
        return out

    def refresh(self):
        core = self.obs._core
        if not self._items:
            self._items = self._triangle_list()
        self.glw.clear()
        names = core.antenna_names
        colors = gradient_colors(self.obs.nif)
        lo = self.page * self.nplot
        self._plots = []
        gaps = TimeGaps.of(self.obs)
        for row, tri in enumerate(self._items[lo : lo + self.nplot]):
            series = core.closure_phases(triangle=tri, if_index=self._if_index)
            p = self._new_plot(row, (-180, 180))
            install_time_axis(p, gaps)
            label = "-".join(names[i] for i in tri)
            p.setLabel("left", f"{label} (deg)")
            p.setYRange(-180, 180, padding=0)
            for s in series:
                cif = s["if_index"]
                col = colors[cif % len(colors)]
                t = gaps.compress(np.asarray(s["time"]) / 3600.0)
                p.addItem(
                    FastScatter(
                        t, np.asarray(s["phase"]) * RAD2DEG, size=4,
                        pen=None, brush=pg.mkBrush(*col, 200),
                    )
                )
                m = np.asarray(s["model"]) * RAD2DEG
                if np.isfinite(m).any():
                    p.addItem(
                        FastScatter(
                            t, m, size=5, symbol="+", pen=pg.mkPen(200, 30, 30), brush=None
                        )
                    )
        self._share_x_axis()
        for p, _ in self._plots:
            frame_panels([p])
        if self._plots:
            self._plots[-1][0].setLabel("bottom", "Time (hours)")
        self.statusBar().showMessage(
            f"page {self.page + 1}/{self.npages} - {len(self._items)} triangles "
            "| n/p: page | red +: model | h: help"
        )


class TPlot(PlotWindow):
    """Per-antenna time sampling (difmap tplot).

    One row per antenna, labelled by name, with a mark at every integration where the antenna has unflagged data. As in difmap, the colour says whether that is all of it: an integration where every baseline to the antenna is unflagged is drawn in the primary colour (`GOOD_COLOR`, matplotlib's "C0"), one where some of its baselines are flagged and others are not in the secondary colour (`PARTIAL_COLOR`, "C1"). An integration with all of them flagged is not drawn. See `sampling` for what counts as flagged.
    """

    #: Every baseline of the antenna unflagged (matplotlib "C0").
    GOOD_COLOR = (31, 119, 180)
    #: Some flagged, some not (matplotlib "C1").
    PARTIAL_COLOR = (255, 127, 14)

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

    def sampling(self):
        """Per-(integration, antenna) counts of baselines, ``(good, flagged)``, each [ntimes, nant].

        A baseline is good at an integration if any selected IF of it is unflagged, and flagged if it has data but none of it unflagged. A flagged baseline counts against an antenna only when the station at its other end has unflagged data at that integration: a station that is out altogether (flagged or ignored) says nothing about the others, and would otherwise mark every antenna as partly flagged. Deleted data (weight 0) is absent, not flagged.
        """
        core = self.obs._core
        nt, nant = core.ntimes, len(core.antenna_names)
        _, wt = core.stream_vis()
        used = np.asarray(core.selection()["if_used"], dtype=bool)
        wt = np.asarray(wt)[:, used]
        _, a1, a2, *_ = core.rows()
        a1 = np.asarray(a1, dtype=np.int64)
        a2 = np.asarray(a2, dtype=np.int64)
        tidx = np.asarray(core.time_index(), dtype=np.int64)
        row_good = (wt > 0).any(axis=1)
        row_flag = ~row_good & (wt < 0).any(axis=1)
        good = np.zeros(nt * nant, dtype=np.int64)
        for ant in (a1, a2):
            np.add.at(good, tidx * nant + ant, row_good)
        good = good.reshape(nt, nant)
        flagged = np.zeros(nt * nant, dtype=np.int64)
        for ant, other in ((a1, a2), (a2, a1)):
            m = row_flag & (good[tidx, other] > 0)
            np.add.at(flagged, tidx[m] * nant + ant[m], 1)
        return good, flagged.reshape(nt, nant)

    def refresh(self):
        self.glw.clear()
        obs = self.obs
        core = obs._core
        good, flagged = self.sampling()
        gaps = TimeGaps.of(obs)
        times = gaps.compress(np.asarray(core.times()) / 3600.0)
        names = list(core.antenna_names)
        self.plot = p = self.glw.addPlot()
        install_time_axis(p, gaps)
        p.setLabel("bottom", "Time (hours)")
        p.setLabel("left", "Antenna")
        nsamp = []
        npartial = 0
        for ia in range(len(names)):
            some = good[:, ia] > 0
            partial = some & (flagged[:, ia] > 0)
            nsamp.append(int(some.sum()))
            npartial += int(partial.sum())
            for m, col in ((some & ~partial, self.GOOD_COLOR),
                           (partial, self.PARTIAL_COLOR)):
                if m.any():
                    p.addItem(
                        FastScatter(
                            times[m], np.full(m.sum(), ia), size=5, symbol="s",
                            pen=None, brush=pg.mkBrush(*col, 230),
                        )
                    )
        # Framed like every plot, but with no ticks on the y axis: its
        # "ticks" are the station names.
        frame_panels([p], yticks=False)
        ax = p.getAxis("left")
        # Label the rows with the station names rather than their index.
        ax.setTicks([[(i, n) for i, n in enumerate(names)], []])
        ax.setWidth(max(48, 9 * max((len(n) for n in names), default=4)))
        p.setYRange(-0.5, len(names) - 0.5)
        dead = [n for n, c in zip(names, nsamp) if c == 0]
        self.statusBar().showMessage(
            "blue: unflagged | orange: partly flagged"
            + (f" ({npartial} antenna-integrations)" if npartial else "")
            + (f" | no data: {', '.join(dead)}" if dead else "")
            + " | h: help"
        )

    def key_help(self):
        return [
            ("", "one row per antenna, one mark per integration with data"),
            ("", "blue: all its baselines unflagged; orange: some flagged"),
        ]


class CorPlot(_MultiPanel):
    """Self-calibration corrections vs time (difmap corplot), with editing.

    One antenna to a pair of panels - amplitude above, phase below, sharing the time axis and framed together, as `vplot` lays out its baselines. `quantity` narrows that to "amp" or "phase" alone; the default shows both, since an amplitude solution is usually read against its phase.

    Solutions can be flagged as data are in `vplot`: Shift+drag flags the solutions inside the box, Ctrl+drag restores them, ``f``/``F`` act on the nearest one, Ctrl+Z / Ctrl+Shift+Z undo and redo. A flagged solution (one complex gain: its amplitude and phase go together) is replaced by interpolating the remaining solutions of that antenna and IF, and everything is re-calibrated from the raw data at once (`Observation.flag_solutions`). Flagged solutions are drawn as red crosses at their original values (``x`` hides them); a cross on a solution difmap itself rejected marks one that cannot be restored here.
    """

    #: Tall: a page is several antennas of stacked amp/phase panels.
    DEFAULT_SIZE = (1200, 950)

    def __init__(self, obs, quantity="both", nplot=3):
        self.quantities = _cor_quantities(quantity)
        self._items = []
        #: per panel: (antenna index, quantity, PlotItem)
        self._panel_info = []
        #: per panel: the points drawn, for hit-testing (see `_points`)
        self._points = []
        self._show_flagged = True
        self._mouse_pos = None
        self._keep_view = None
        #: undo/redo stacks of (solution flags before, after)
        self._undo, self._redo = [], []
        super().__init__(obs, "difmapy corplot", nplot=nplot)
        self.glw.scene().sigMouseMoved.connect(self._mouse_moved)

    @property
    def quantity(self):
        """What is displayed: "amp", "phase" or "both"."""
        return "both" if len(self.quantities) == 2 else self.quantities[0]

    def refresh(self):
        obs = self.obs
        core = obs._core
        names = core.antenna_names
        nant, nif = len(names), obs.nif
        ntimes = core.ntimes
        shape = (ntimes, nif, nant)
        amp, phs, bad = (np.asarray(x).reshape(shape) for x in core.gains())
        # An integration a station has no solution for reads 1.0 / 0 deg
        # in the table, which is not a correction and must not be drawn
        # as one: plotting it made a smooth run of corrections look like
        # it jumped to unity and back.
        used = np.asarray(core.gains_used()).reshape(shape)
        sflags = obs.solution_flags
        amp0, phs0, bad0 = obs.original_solutions()
        gaps = TimeGaps.of(obs)
        times = gaps.compress(np.asarray(core.times()) / 3600.0)
        if not self._items:
            # Only show antennas that were actually corrected.
            self._items = [
                (ia, names[ia]) for ia in range(nant) if np.any(used[:, :, ia])
            ] or [(ia, names[ia]) for ia in range(nant)]
        # Keep the zoom over an edit: re-drawing after a flag must not
        # throw away the view the user was flagging in.
        keep = self._keep_view
        self._keep_view = None
        self.glw.clear()
        self._plots, self._panel_info, self._points = [], [], []
        colors = gradient_colors(nif)
        lo = self.page * self.nplot
        page = self._items[lo : lo + self.nplot]
        groups = []
        for i, (ia, name) in enumerate(page):
            group = []
            for j, key in enumerate(self.quantities):
                yrange = (-180, 180) if key == "phase" else None
                vb = SelectViewBox()
                p = self.glw.addPlot(row=i * len(self.quantities) + j, col=0, viewBox=vb)
                self._plots.append((p, yrange))
                vb.sigSelected.connect(
                    lambda x0, x1, y0, y1, flag, n=len(self._panel_info):
                        self._on_selected(n, x0, x1, y0, y1, flag))
                self._panel_info.append((ia, key, p))
                install_time_axis(p, gaps)
                # Short: under the antenna name on a short panel, the full
                # words overlapped each other.
                label = "Amp" if key == "amp" else "\u03c6 (deg)"
                p.setLabel("left", f"{name}<br>{label}")
                p.getAxis("left").setWidth(82)
                # A gain is dimensionless and sits near 1: pyqtgraph's
                # SI prefixing would render 0.758 as "758.5 (x0.001)".
                p.getAxis("left").enableAutoSIPrefix(False)
                if yrange is not None:
                    p.setYRange(*yrange, padding=0)
                shown = []
                pts = {"t": [], "cif": [], "x": [], "y": [], "flagged": []}
                for cif in range(nif):
                    col = colors[cif % len(colors)]
                    solved = used[:, cif, ia]
                    removed = solved & sflags[:, cif, ia]
                    good = solved & ~bad[:, cif, ia] & ~removed
                    rejected = solved & bad0[:, cif, ia] & ~removed
                    y = (phs[:, cif, ia] * RAD2DEG if key == "phase"
                         else amp[:, cif, ia])
                    y0 = (phs0[:, cif, ia] * RAD2DEG if key == "phase"
                          else amp0[:, cif, ia])
                    if good.any():
                        shown.append(y[good])
                        # A line broken wherever there is no solution:
                        # with a solution interval set, self-cal
                        # interpolates its bins onto the integrations
                        # (difmap's apply_solns), so the corrections
                        # really are a smooth curve - draw them as one.
                        # It runs through the interpolated values of the
                        # flagged solutions, which are what is applied.
                        line = solved & ~bad[:, cif, ia]
                        p.addItem(
                            pg.PlotDataItem(
                                times, np.where(line, y, np.nan),
                                connect="finite",
                                pen=pg.mkPen(*col, 160, width=1.2),
                            )
                        )
                        # The markers as one fast item: a PlotDataItem's
                        # own symbols cost a per-point record each.
                        p.addItem(FastScatter(
                            times[good], y[good], size=4, pen=None,
                            brush=pg.mkBrush(*col, 220),
                        ))
                    marks = (removed if self._show_flagged else np.zeros_like(removed)) | rejected
                    if marks.any():
                        p.addItem(
                            FastScatter(
                                times[marks], y0[marks], size=7, symbol="x",
                                pen=pg.mkPen(*FLAG_COLOR, width=1.5), brush=None,
                            )
                        )
                    for m, flagged, vals in ((good, False, y), (removed, True, y0)):
                        idx = np.nonzero(m)[0]
                        pts["t"].append(idx)
                        pts["cif"].append(np.full(idx.size, cif))
                        pts["x"].append(times[idx])
                        pts["y"].append(vals[idx])
                        pts["flagged"].append(np.full(idx.size, flagged))
                self._points.append({k: np.concatenate(v) if v else np.zeros(0)
                                     for k, v in pts.items()})
                if key == "amp" and shown:
                    # A single solution is constant to the last bit of
                    # float32; keep the panel off that scale.
                    self._plots[-1] = (
                        p, self._set_amp_range(p, np.concatenate(shown)),
                    )
                group.append(p)
            groups.append(group)
        self._share_x_axis()
        for group in groups:
            frame_panels(group)
        if self._plots:
            self._plots[-1][0].setLabel("bottom", "Time (hours)")
            self.reset_ranges("x")
        if keep is not None and len(keep) == len(self._plots):
            for (p, _), rng in zip(self._plots, keep):
                p.vb.setRange(xRange=rng[0], yRange=rng[1], padding=0)
        what = {"both": "amplitude and phase", "amp": "amplitude",
                "phase": "phase"}[self.quantity]
        nogap = ""
        if page:
            missing = sum(int((~used[:, :, ia]).all(axis=1).sum())
                          for ia, _ in page)
            if missing:
                nogap = f" | {missing} integrations without a solution (not drawn)"
        nflag = int(sflags.sum())
        self.statusBar().showMessage(
            f"gain {what} - page {self.page + 1}/{self.npages} of "
            f"{len(self._items)} antennas | n/p: page "
            f"| shift/ctrl+drag: flag/unflag solutions ({nflag} flagged, x: show/hide){nogap} | h: help"
        )

    # ---- editing -------------------------------------------------------

    def _mouse_moved(self, pos):
        self._mouse_pos = pos

    def _panel_at(self, scene_pos):
        for n, (_, _, p) in enumerate(self._panel_info):
            if p.sceneBoundingRect().contains(scene_pos):
                return n
        return None

    def _edit(self, n, sel, flag):
        """Flag (or restore) the solutions `sel` (indices into panel `n`'s points) of that panel's antenna, and re-calibrate."""
        ia = self._panel_info[n][0]
        pts = self._points[n]
        before = self.obs.solution_flags
        after = before.copy()
        after[pts["t"][sel].astype(int), pts["cif"][sel].astype(int), ia] = flag
        if np.array_equal(before, after):
            return 0
        self._apply_flags(after, before)
        return int(np.abs(after.astype(int) - before.astype(int)).sum())

    def _apply_flags(self, after, before, record=True):
        if record:
            self._undo.append((before, after))
            self._redo.clear()
        self._keep_view = [p.vb.viewRange() for p, _ in self._plots]
        self.obs.set_solution_flags(after)
        self.refresh()
        self._changed()

    def _on_selected(self, n, x0, x1, y0, y1, flag):
        pts = self._points[n]
        inside = ((pts["x"] >= x0) & (pts["x"] <= x1) & (pts["y"] >= y0) & (pts["y"] <= y1)
                  & (pts["flagged"] != flag))
        k = self._edit(n, np.nonzero(inside)[0], flag)
        self.statusBar().showMessage(f"{'flagged' if flag else 'restored'} {k} solutions")
        return k

    def flag_nearest(self, n, x, y, flag=True):
        """Flag (or restore) the solution of panel `n` nearest the view position (x, y)."""
        pts = self._points[n]
        cand = pts["flagged"] != flag
        if not cand.any():
            return 0
        (xr0, xr1), (yr0, yr1) = self._panel_info[n][2].vb.viewRange()
        d = (((pts["x"] - x) / max(xr1 - xr0, 1e-30)) ** 2
             + ((pts["y"] - y) / max(yr1 - yr0, 1e-30)) ** 2)
        return self._edit(n, [int(np.argmin(np.where(cand, d, np.inf)))], flag)

    def undo(self, redo=False):
        """Undo (or redo) the last solution edit. Returns False if there was nothing to do, or the calibration changed since (the edit no longer applies)."""
        stack, other = (self._redo, self._undo) if redo else (self._undo, self._redo)
        if not stack:
            return False
        before, after = stack.pop()
        target = after if redo else before
        current = after if not redo else before
        if not np.array_equal(self.obs.solution_flags, current):
            # A new calibration (or an edit from the prompt) came in
            # between: the stack refers to a table that is gone.
            self._undo.clear()
            self._redo.clear()
            return False
        other.append((before, after))
        self._apply_flags(target, current, record=False)
        return True

    def key_help(self):
        return super().key_help() + [
            ("shift+drag / ctrl+drag", "flag / restore the solutions in the box"),
            ("f / F", "flag / restore the solution nearest the cursor"),
            ("ctrl+z / ctrl+shift+z", "undo / redo the last edit"),
            ("x", "show or hide the flagged solutions (red crosses)"),
        ]

    def keyPressEvent(self, ev):
        from pyqtgraph.Qt import QtCore

        mods = ev.modifiers()
        ctrl = bool(mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
        key = ev.text()
        if ctrl and ev.key() == QtCore.Qt.Key.Key_Z:
            ok = self.undo(redo=shift)
            self.statusBar().showMessage(("redid" if shift else "undid") + " the last edit" if ok
                                         else "nothing to " + ("redo" if shift else "undo"))
        elif ctrl and ev.key() == QtCore.Qt.Key.Key_Y:
            self.undo(redo=True)
        elif key.lower() == "x":
            self._show_flagged = not self._show_flagged
            self._keep_view = [p.vb.viewRange() for p, _ in self._plots]
            self.refresh()
        elif key in ("f", "F") and self._mouse_pos is not None:
            n = self._panel_at(self._mouse_pos)
            if n is not None:
                v = self._panel_info[n][2].vb.mapSceneToView(self._mouse_pos)
                k = self.flag_nearest(n, v.x(), v.y(), flag=(key == "f"))
                self.statusBar().showMessage(f"{'flagged' if key == 'f' else 'restored'} {k} solutions")
        else:
            super().keyPressEvent(ev)


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
            bl = (obs._ant_index(baseline[0]), obs._ant_index(baseline[1]))
        s = core.spectrum(baseline=bl, tmin=self.tmin, tmax=self.tmax)
        wt = np.asarray(s["wt"])
        good = wt > 0
        x = np.asarray(s["freq"]) / 1e9 if xaxis == "freq" else np.asarray(s["chan"])
        amp = np.asarray(s["amp"])
        phase = np.rad2deg(np.arctan2(np.asarray(s["im"]), np.asarray(s["re"])))
        pa = self.glw.addPlot(row=0, col=0, axisItems={"left": FluxAxis()})
        pa.addItem(pg.PlotDataItem(x[good], amp[good],
                                   pen=pg.mkPen(31, 119, 180, width=2),
                                   symbol="o", symbolSize=4))
        pp = self.glw.addPlot(row=1, col=0)
        pp.setLabel("left", "Phase (deg)")
        pp.setLabel("bottom", "Frequency (GHz)" if xaxis == "freq" else "Channel")
        pp.setYRange(-180, 180, padding=0)
        pp.addItem(pg.PlotDataItem(x[good], phase[good],
                                   pen=pg.mkPen(255, 127, 14, width=2),
                                   symbol="o", symbolSize=4))
        pp.setXLink(pa)
        pa.getAxis("bottom").setStyle(showValues=False)
        frame_panels([pa, pp])
        self._plots = [pa, pp]
        label = "all baselines" if baseline is None else "-".join(map(str, baseline))
        self.statusBar().showMessage(
            f"vector-averaged spectrum ({label}) | h: help"
        )


class FPlot(_MultiPanel):
    """Amplitude and phase against frequency, one baseline to a pair of panels, averaged over the whole time range (difmapy fplot).

    The frequency counterpart of `vplot`: every channel of every IF is averaged over the observation - or over `tmin`..`tmax` - so that a slope or a ripple across the band stands out where a single integration would only show noise. `n`/`p` page through the baselines, `reftel` restricts them to one station's.

    Amplitudes are scalar-averaged, so they do not decorrelate as the fringe turns; phases come from the vector average, which is what makes a phase slope across the band visible. The accumulated calibration is applied (`calibrated=False` shows the data as loaded), and channels outside the current selection are shown too, so the plot can be used to choose them.

    This is a view rather than an editor: each point averages many integrations, while flag editing works on whole rows.
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
        return int(name) if isinstance(name, int) else obs._ant_index(name)

    @classmethod
    def _select_baselines(cls, obs, reftel, baselines):
        """The (a1, a2) pairs to page through, as global antenna indices."""
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
                (("amp", amp, "Amplitude"), ("phase", phase, "Phase (deg)"))
            ):
                p = self._new_plot(2 * i + j,
                                   (-180, 180) if key == "phase" else None,
                                   flux=(key == "amp"))
                p.setLabel("left", f"{label}<br>{ylab}")
                # Amplitudes over a narrow range need several digits;
                # pyqtgraph drops the tick labels rather than widen the
                # axis for them, so make room up front.
                p.getAxis("left").setWidth(82)
                if key == "phase":
                    p.setYRange(-180, 180, padding=0)
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
        for k in range(0, len(panels), 2):
            frame_panels(panels[k:k + 2])
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


class PossmPlot(_MultiPanel):
    """Vector-averaged amplitude and phase against frequency, a grid of baselines to a page (AIPS POSSM).

    Every channel of every IF is vector-averaged over the whole observation (or `tmin`..`tmax`): the amplitude is the modulus of the averaged visibility and the phase its argument, so a phase slope or ripple across the band shows, and a decorrelating one lowers the amplitudes too - which is what POSSM shows and how bandpass and delay errors are spotted. Each baseline gets an amplitude panel above a phase panel, framed together; `nplot` baselines go to a page, `ncols` to a row of the grid, and they are paged through with ``n``/``p`` in the same order as `vplot` (`reftel` keeps one station's only).

    The accumulated calibration is applied unless `calibrated=False`. Channels outside the current selection are shown too, so the plot can be used to choose them. The plot is a view, not an editor.
    """

    #: A page is a grid of baselines, each two panels high.
    DEFAULT_SIZE = (1400, 950)

    def __init__(self, obs, nplot=9, reftel=None, baselines=None, ncols=3,
                 tmin=None, tmax=None, calibrated=True):
        self.tmin = parse_time(tmin, "s")
        self.tmax = parse_time(tmax, "s")
        self.calibrated = bool(calibrated)
        self.ncols = max(1, int(ncols))
        self._items = FPlot._select_baselines(obs, reftel, baselines)
        nplot = int(nplot)
        if nplot <= 0:
            # difmap's 0, as for vplot: every baseline on one page.
            nplot = max(len(self._items), 1)
        title = "difmapy possm"
        if reftel is not None:
            title += f" ({obs.antennas[FPlot._ant(obs, reftel)]})"
        super().__init__(obs, title, nplot=nplot)

    def averaged(self, baseline):
        """The vector average of one baseline over the time range, as a dict of per-channel arrays: `freq` (Hz), `if_index`, `amp` (Jy, the modulus of the averaged visibility), `phase` (degrees) and the summed weight `wt` (0 where there is no unflagged data)."""
        a, b = baseline
        s = self.obs._core.spectrum(
            baseline=(a, b), tmin=self.tmin, tmax=self.tmax,
            calibrated=self.calibrated,
        )
        re, im = np.asarray(s["re"]), np.asarray(s["im"])
        nchan = list(self.obs.nchan)
        cif = (np.concatenate([np.full(n, i, dtype=int) for i, n in enumerate(nchan)])
               if nchan else np.zeros(0, dtype=int))
        return {
            "freq": np.asarray(s["freq"]),
            "if_index": cif,
            "amp": np.hypot(re, im),
            "phase": np.rad2deg(np.arctan2(im, re)),
            "wt": np.asarray(s["wt"]),
        }

    def refresh(self):
        self.glw.clear()
        self._plots = []
        names = self.obs.antennas
        colors = gradient_colors(self.obs.nif)
        lo = self.page * self.nplot
        page = self._items[lo: lo + self.nplot]
        nplotted = 0
        groups = []
        amps = []
        n = len(page)
        # Tight: the panels of a page share their axes, so only the left
        # column carries y labels and only the bottom of each column x
        # labels, and the baselines sit close with just their title
        # between them.
        lay = self.glw.ci.layout
        lay.setContentsMargins(4, 4, 8, 4)
        lay.setHorizontalSpacing(6)
        lay.setVerticalSpacing(0)
        for k, (a, b) in enumerate(page):
            r, c = divmod(k, self.ncols)
            left = c == 0
            # The bottom of its column: nothing below it on this page.
            bottom = k + self.ncols >= n
            d = self.averaged((a, b))
            good = d["wt"] > 0
            freq = d["freq"] / 1e9
            pair = []
            for j, key in enumerate(("amp", "phase")):
                p = self.glw.addPlot(row=2 * r + j, col=c,
                                     axisItems={"left": FluxAxis()} if key == "amp" else None)
                self._plots.append((p, (-180, 180) if key == "phase" else None))
                p.layout.setContentsMargins(0, 0, 0, 0)
                p.layout.setSpacing(0)
                ax = p.getAxis("left")
                if left:
                    p.setLabel("left", "Amp" if key == "amp" else "\u03c6 (deg)")
                    ax.setWidth(LEFT_AXIS_WIDTH)
                else:
                    ax.setLabel("")
                    ax.showLabel(False)
                    ax.setStyle(showValues=False)
                    ax.setWidth(INNER_AXIS_WIDTH)
                if key == "amp":
                    p.setTitle(f"{names[a]}-{names[b]}", size="9pt", color="#222")
                    p.layout.setRowFixedHeight(0, TITLE_HEIGHT)
                else:
                    p.setYRange(-180, 180, padding=0)
                y = d[key]
                for band in range(self.obs.nif):
                    m = good & (d["if_index"] == band)
                    if not m.any():
                        continue
                    col = colors[band % len(colors)]
                    p.addItem(pg.PlotDataItem(
                        freq[m], y[m], pen=pg.mkPen(*col, width=1.2),
                        symbol="o", symbolSize=5, symbolPen=None,
                        symbolBrush=pg.mkBrush(*col, 220),
                    ))
                    if key == "amp":
                        nplotted += int(m.sum())
                bax = p.getAxis("bottom")
                if key == "phase" and bottom:
                    p.setLabel("bottom", "Frequency (GHz)")
                else:
                    # The frequency ticks go under the bottom row only.
                    bax.setStyle(showValues=False)
                    bax.setLabel("")
                    bax.showLabel(False)
                    bax.setHeight(INNER_AXIS_WIDTH)
                if key == "amp":
                    amps.append(y[good])
                pair.append(p)
            groups.append(pair)
        # Every baseline has the same channels: one frequency axis. The
        # amplitude panels share one scale too - with the tick labels on
        # the left column only, a scale of its own per panel could not be
        # read - and so do the phase panels (+-180 deg).
        plots = [p for p, _ in self._plots]
        for p in plots[1:]:
            p.setXLink(plots[0])
        amp_plots, phase_plots = plots[0::2], plots[1::2]
        for q in amp_plots[1:]:
            q.setYLink(amp_plots[0])
        for q in phase_plots[1:]:
            q.setYLink(phase_plots[0])
        if amp_plots:
            values = np.concatenate(amps) if amps else np.zeros(0)
            yr = self._set_amp_range(amp_plots[0], values)
            self._plots = [(p, yr if i % 2 == 0 else (-180, 180)) for i, p in enumerate(plots)]
        for pair in groups:
            frame_panels(pair)
        self.reset_ranges("x")
        window = (
            "the whole observation" if self.tmin is None and self.tmax is None
            else f"{(self.tmin or 0) / 3600.0:.3g}-{(self.tmax or 0) / 3600.0:.3g} h"
        )
        cal = "calibrated" if self.calibrated else "as loaded"
        self.statusBar().showMessage(
            f"page {self.page + 1}/{self.npages} of {len(self._items)} baselines "
            f"| vector-averaged over {window} ({cal}), {nplotted} channels drawn "
            "| n/p: page | h: help"
        )

    def key_help(self):
        return super().key_help() + [
            ("", "amplitude and phase of the vector average over time"),
            ("", "colours are the IFs, in order"),
        ]


def cpplot(obs, triangles=None, if_index=None, nplot=4, block=None):
    """Closure phases against time, with the model overplotted (difmap `cpplot`), `nplot` triangles to a page.

    Parameters
    ----------
    obs : Observation
        The observation to plot.
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
    p = CpPlot(obs, triangles=triangles, if_index=if_index, nplot=nplot)
    run_if_needed(p, block)
    return p


def tplot(obs, block=None):
    """The time sampling of each antenna (difmap `tplot`): one row per station with a mark at every integration where it has unflagged data, in the primary colour when all of its baselines are unflagged and in the secondary colour when some are flagged.

    Parameters
    ----------
    obs : Observation
        The observation to plot.
    block : bool | None, default None
        Wait until the window is closed. ``None`` blocks only when nothing else runs the Qt event loop: a plain script blocks, an IPython or Jupyter session does not.

    Returns
    -------
    PlotWindow
        The plot window, for `savefig`, `to_png` or further calls; in a notebook (or with no display) the plot is drawn off-screen and shown inline.
    """
    p = TPlot(obs)
    run_if_needed(p, block)
    return p


def corplot(obs, quantity="both", nplot=3, block=None):
    """The self-calibration gain corrections against time (difmap `corplot`): one antenna to a pair of panels, amplitude above and phase below on a shared time axis, framed together.

    Solutions can be edited as data are in `vplot`: Shift+drag flags the solutions in the box, Ctrl+drag restores them, ``f``/``F`` act on the one nearest the cursor, Ctrl+Z / Ctrl+Shift+Z undo and redo, ``x`` shows or hides the flagged ones (red crosses at their original values). A flagged solution is replaced by interpolating the remaining ones of its antenna and IF, and the data are re-calibrated from the raw visibilities at once; see `flag_solutions`, which does the same from the prompt.

    Parameters
    ----------
    obs : Observation
        The observation to plot.
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
    p = CorPlot(obs, quantity=quantity, nplot=nplot)
    run_if_needed(p, block)
    return p


def specplot(obs, baseline=None, tmin=None, tmax=None, xaxis="freq", block=None):
    """The time-averaged spectrum of the selected polarization (difmap `specplot`), over every baseline or one.

    Parameters
    ----------
    obs : Observation
        The observation to plot.
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
    p = SpecPlot(obs, baseline=baseline, tmin=tmin, tmax=tmax, xaxis=xaxis)
    run_if_needed(p, block)
    return p


def possm(obs, nplot=9, reftel=None, baselines=None, ncols=3, tmin=None,
          tmax=None, calibrated=True, block=None):
    """Vector-averaged amplitude and phase against frequency (AIPS POSSM), a grid of baselines to a page - 3 x 3 by default - each an amplitude panel above a phase panel. Every channel of every IF is vector-averaged over the whole observation (or `tmin`..`tmax`), so phase slopes and ripples across the band show, and decorrelate the amplitudes as POSSM does. ``n``/``p`` page through the baselines in the same order as `vplot`.

    Parameters
    ----------
    obs : Observation
        The observation to plot.
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
    p = PossmPlot(obs, nplot=nplot, reftel=reftel, baselines=baselines,
                  ncols=ncols, tmin=tmin, tmax=tmax, calibrated=calibrated)
    run_if_needed(p, block)
    return p


def fplot(obs, reftel=None, baselines=None, nplot=3, tmin=None, tmax=None,
          calibrated=True, block=None):
    """Amplitude and phase against frequency, one baseline to a pair of panels, averaged over time. Amplitudes are scalar-averaged, so they do not decorrelate, and phases vector-averaged. `possm()` shows the vector average of both, on a grid of baselines; `specplot()` averages every baseline at once.

    Parameters
    ----------
    obs : Observation
        The observation to plot.
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
    p = FPlot(obs, reftel=reftel, baselines=baselines, nplot=nplot,
              tmin=tmin, tmax=tmax, calibrated=calibrated)
    run_if_needed(p, block)
    return p
