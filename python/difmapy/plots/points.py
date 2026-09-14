"""Point-based interactive plots: radplot, projplot, uvplot, vplot."""

from __future__ import annotations

import numpy as np

from difmapy.plots.base import (
    HIGHLIGHT_COLOR,
    FastScatter,
    FlagPlotBase,
    TimeGaps,
    install_time_axis,
    run_if_needed,
)

import pyqtgraph as pg

__all__ = ["RadPlot", "UVPlot", "VPlot", "radplot", "projplot", "uvplot", "vplot"]

RAD2DEG = 180.0 / np.pi

#: Accepted spellings of the two-panel amplitude+phase mode.
AMP_PHASE = {"ap", "anp", "a&p", "amp&phase", "both"}


def _quantities(spec):
    """Which panels a `quantity=` spec asks for, top to bottom."""
    key = str(spec).lower().replace(" ", "")
    if key in AMP_PHASE:
        return ("amp", "phase")
    if key in ("amp", "amplitude", "a"):
        return ("amp",)
    if key in ("phase", "phs", "p"):
        return ("phase",)
    raise ValueError(
        f"unknown quantity {spec!r}; use 'amp', 'phase' or 'ap' (amp+phase)"
    )


def _stream_arrays(obs, with_model=True):
    """Common per-(row, IF) arrays: complex vis, signed wt, u, v in
    wavelengths, row and IF indices, times, antennas and the model."""
    core = obs._core
    vis, wt = core.stream_vis()
    time, a1, a2, us, vs, _ = core.rows()
    sel = core.selection()
    freq = np.asarray(sel["if_freq"])  # [nif]
    nrow, nif = vis.shape
    row = np.repeat(np.arange(nrow), nif)
    cif = np.tile(np.arange(nif), nrow)
    uu = (us[:, None] * freq[None, :]).ravel()
    vv = (vs[:, None] * freq[None, :]).ravel()
    used = np.asarray(sel["if_used"])[cif]
    out = {
        "vis": vis.ravel()[used],
        "wt": wt.ravel()[used],
        "u": uu[used],
        "v": vv[used],
        "row": row[used],
        "cif": cif[used],
        "time": np.repeat(time, nif)[used],
        "a1": np.repeat(a1, nif)[used],
        "a2": np.repeat(a2, nif)[used],
    }
    if with_model:
        # The complete model, whatever part of it the stream model holds
        # (all of it, except while modelfit is working on some).
        model = np.asarray(core.full_model()).ravel()[used]
        out["model"] = model if np.any(model != 0.0) else None
    return out


def _wrap_angle(deg):
    """A projection angle in [-90, 90) degrees: the projected distance
    |u sin(phi) + v cos(phi)| repeats every 180 degrees."""
    return round((float(deg) + 90.0) % 180.0 - 90.0, 9)


def _place_corner(vb, item, pad=6):
    """Pin a TextItem to the top-right corner of a view box."""
    r = vb.boundingRect()
    item.setPos(r.right() - pad, r.top() + pad)


class _VisPlot(FlagPlotBase):
    """Shared behaviour of the visibility scatter plots: amplitude and
    phase panels, and the model overplotted where one is defined."""

    def __init__(self, obs, title, quantity="ap", colorby="spw", legend=False):
        self.quantities = _quantities(quantity)
        super().__init__(obs, title, colorby=colorby, legend=legend)

    def _vis_columns(self, d):
        """amp/phase (and their model counterparts) from complex data."""
        out = {
            "amp": np.abs(d["vis"]),
            "phase": np.rad2deg(np.angle(d["vis"])),
        }
        m = d.get("model")
        if m is not None:
            out["model_amp"] = np.abs(m)
            out["model_phase"] = np.rad2deg(np.angle(m))
        return out

    def _model(self, d, key):
        return d.get(f"model_{key}")

    def has_model(self) -> bool:
        return self._data is not None and self._data.get("model_amp") is not None


class RadPlot(_VisPlot):
    """Amplitude and phase vs UV radius; difmap radplot/projplot.

    Points are coloured by spectral window on a gradient scale by
    default; ``n``/``p`` walk through the antennas, highlighting every
    baseline of one at a time, and the model (CLEAN components or
    Gaussians) is drawn in red where one is defined.

    As projplot (`projection_deg` given), ``<``/``>`` turn the projection
    angle by `angle_step` degrees.
    """

    def __init__(self, obs, quantity="ap", colorby="spw", projection_deg=None,
                 angle_step=10.0):
        self.projection = (None if projection_deg is None
                           else _wrap_angle(projection_deg))
        self.angle_step = float(angle_step)
        self._highlight = None  # antenna index, or None for "all"
        self._hi_items = []
        self._labels = {}
        title = "difmapy projplot" if projection_deg is not None else "difmapy radplot"
        super().__init__(obs, title, quantity=quantity, colorby=colorby)

    def _build_panels(self):
        for i, key in enumerate(self.quantities):
            panel = self._add_panel(i, key)
            # A corner label naming the highlighted antenna. It lives
            # in the view box's pixel frame, so it stays put on zoom.
            lbl = pg.TextItem(anchor=(1, 0), color=HIGHLIGHT_COLOR)
            lbl.setParentItem(panel.vb)
            panel.vb.sigResized.connect(
                lambda *_, vb=panel.vb, t=lbl: _place_corner(vb, t)
            )
            _place_corner(panel.vb, lbl)
            self._labels[key] = lbl
        self._share_x_axis()
        self._set_xlabel()

    def _set_xlabel(self):
        if not self._panels:
            return
        if self.projection is None:
            text = "UV radius (Mλ)"
        else:
            text = f"projected UV distance, PA {self.projection:g}° (Mλ)"
        self._panels[-1].plot.setLabel("bottom", text)

    # ---- projection angle ---------------------------------------------

    def rotate_projection(self, step_deg):
        """Turn the projection angle by `step_deg` degrees and redraw.

        The x axis is rescaled to the new projected distances; the y
        ranges are left alone, since the projection does not change
        amplitudes or phases. Returns the new angle, in [-90, 90).
        """
        if self.projection is None:
            raise ValueError("radplot has no projection angle; use projplot")
        self.projection = _wrap_angle(self.projection + float(step_deg))
        self.refresh()
        self._set_xlabel()
        self.reset_ranges("x")
        return self.projection

    def _collect(self):
        d = _stream_arrays(self.obs)
        if self.projection is None:
            x = np.hypot(d["u"], d["v"]) / 1e6
        else:
            phi = np.deg2rad(self.projection)
            x = np.abs(d["u"] * np.sin(phi) + d["v"] * np.cos(phi)) / 1e6
        d["x"] = x
        d.update(self._vis_columns(d))
        return d

    # ---- antenna highlighting ----------------------------------------

    def _antenna_order(self):
        return list(range(len(self.obs.antennas)))

    def cycle_antenna(self, step):
        """Move the highlight to the next/previous antenna (or off)."""
        order = [None] + self._antenna_order()
        i = order.index(self._highlight) if self._highlight in order else 0
        self._highlight = order[(i + step) % len(order)]
        self.refresh()
        return self._highlight

    def _decorate(self, panel, d, sub, good):
        super()._decorate(panel, d, sub, good)
        lbl = self._labels.get(panel.key)
        if self._highlight is None:
            if lbl is not None:
                lbl.setText("")
            return
        name = self.obs.antennas[self._highlight]
        if lbl is not None:
            lbl.setText(name)
        m = sub & good & ((d["a1"] == self._highlight) | (d["a2"] == self._highlight))
        if m.any():
            self._add_item(
                panel,
                FastScatter(
                    d["x"][m], d[panel.key][m], size=7, symbol="o",
                    pen=pg.mkPen(*HIGHLIGHT_COLOR, width=1),
                    brush=pg.mkBrush(*HIGHLIGHT_COLOR, 120),
                ),
            )

    def key_help(self):
        keys = super().key_help() + [
            ("n / p", "highlight the next / previous antenna"),
        ]
        if self.projection is not None:
            keys.append(("< / >", f"turn the projection angle by "
                                  f"-/+{self.angle_step:g} degrees"))
        return keys

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key in ("n", "p"):
            ant = self.cycle_antenna(1 if key == "n" else -1)
            self._message(
                "no antenna highlighted" if ant is None
                else f"highlighting {self.obs.antennas[ant]}"
            )
        elif key in ("<", ">") and self.projection is not None:
            step = self.angle_step if key == ">" else -self.angle_step
            self._message(f"projection angle {self.rotate_projection(step):g}°")
        else:
            super().keyPressEvent(ev)

    def _status(self):
        base = super()._status()
        who = ("all antennas" if self._highlight is None
               else f"highlighting {self.obs.antennas[self._highlight]}")
        model = " | model in red" if self.has_model() else ""
        proj = ("" if self.projection is None
                else f"PA {self.projection:g}° (</>) | ")
        return f"{proj}{who} (n/p) | {base}{model}"


class UVPlot(FlagPlotBase):
    """UV coverage; difmap uvplot (conjugate points included)."""

    def __init__(self, obs, colorby="spw"):
        super().__init__(obs, "difmapy uvplot", colorby=colorby)

    def _build_panels(self):
        panel = self._add_panel(0, "y", ylabel="V (Mλ)")
        panel.plot.setLabel("bottom", "U (Mλ)")
        panel.vb.setAspectLocked(True)
        panel.vb.invertX(True)

    def _collect(self):
        d = _stream_arrays(self.obs, with_model=False)
        two = lambda a: np.concatenate([a, a])  # noqa: E731
        return {
            "x": np.concatenate([d["u"], -d["u"]]) / 1e6,
            "y": np.concatenate([d["v"], -d["v"]]) / 1e6,
            "wt": two(d["wt"]),
            "row": two(d["row"]),
            "cif": two(d["cif"]),
            "time": two(d["time"]),
            "a1": two(d["a1"]),
            "a2": two(d["a2"]),
        }


class VPlot(_VisPlot):
    """Visibility amplitude and phase vs time, a few baselines to a page
    (difmap vplot).

    Every spectral window is drawn at once and can be switched on and
    off from the legend on the right. Flagging acts either on the
    displayed baseline only or on every baseline of its first antenna;
    the space bar switches between the two.
    """

    #: Tall: a page is several baselines of stacked amp/phase panels.
    DEFAULT_SIZE = (1200, 950)

    def __init__(self, obs, nplot=3, reftel=None, quantity="ap"):
        names = obs.antennas
        if isinstance(nplot, str) and reftel is None:
            # vplot("EF"): a station name where the page size goes.
            nplot, reftel = 3, nplot
        self.reftel = None
        if reftel is not None:
            self.reftel = (names.index(str(reftel)) if not isinstance(reftel, int)
                           else int(reftel))
        # difmap's 0: every baseline of a station on one page.
        self.nplot = int(nplot) if int(nplot) > 0 else max(len(names) - 1, 1)
        self.page = 0
        self.by_antenna = False
        self._baselines = []
        # One cut time axis for the whole observation, so every page
        # (and every other time plot) breaks at the same places.
        self.gaps = TimeGaps.of(obs)
        title = "difmapy vplot"
        if self.reftel is not None:
            title += f" ({names[self.reftel]})"
        super().__init__(obs, title, quantity=quantity, colorby="spw", legend=True)

    # ---- pages of baselines ------------------------------------------

    def _all_baselines(self):
        _, a1, a2, _, _, _ = self.obs._core.rows()
        pairs = sorted({(int(a), int(b)) for a, b in zip(a1, a2)})
        if self.reftel is not None:
            pairs = [p for p in pairs if self.reftel in p]
        return pairs

    @property
    def npages(self) -> int:
        return max(1, int(np.ceil(len(self._baselines) / self.nplot)))

    def _page_baselines(self):
        lo = self.page * self.nplot
        return self._baselines[lo: lo + self.nplot]

    def _build_panels(self):
        names = self.obs.antennas
        page = self._page_baselines()
        for i, bl in enumerate(page):
            label = f"{names[bl[0]]}-{names[bl[1]]}"
            for j, key in enumerate(self.quantities):
                row = i * len(self.quantities) + j
                panel = self._add_panel(
                    row, key, group=i, label=label,
                    ylabel=f"{label}<br>{self.QUANTITIES[key]}",
                )
                if key == "amp":
                    panel.plot.setLabel("right", "")
                install_time_axis(panel.plot, self.gaps)
        self._share_x_axis()
        if self._panels:
            self._panels[-1].plot.setLabel("bottom", "Time (hours)")

    def set_page(self, page):
        """Show page `page` (counting from 0, wrapping around)."""
        self.page = int(page) % self.npages
        self._relayout()
        return self.page

    def _relayout(self):
        self.glw.clear()
        self._panels = []
        self._items = []
        # A new page holds different baselines, so it gets its own
        # auto-scale rather than inheriting the previous page's view.
        self._scaled = False
        self.refresh()

    def reload(self):
        """Re-read the baselines as well as the data: ignoring or
        flagging a station from the prompt can change which baselines
        there are to page through."""
        self._baselines = []
        page = self.page
        self._relayout()
        self.page = min(page, self.npages - 1)
        if self.page != page:
            self._relayout()

    def _collect(self):
        d = _stream_arrays(self.obs)
        if not self._baselines:
            self._baselines = self._all_baselines()
        page = self._page_baselines()
        # group == the panel row a point belongs to; -1 = not displayed.
        group = np.full(d["wt"].shape, -1, dtype=int)
        for i, (a, b) in enumerate(page):
            group[(d["a1"] == a) & (d["a2"] == b)] = i
        keep = group >= 0
        out = {k: (v[keep] if isinstance(v, np.ndarray) else v)
               for k, v in d.items() if v is not None}
        out["group"] = group[keep]
        out["x"] = self.gaps.compress(out["time"] / 3600.0)
        out.update(self._vis_columns(out))
        return out

    # ---- flagging mode ------------------------------------------------

    def _edit_ops(self, idx, flag):
        ops = super()._edit_ops(idx, flag)
        if not self.by_antenna:
            return ops
        # Extend each edit to every baseline of the displayed baseline's
        # first antenna at the same integration.
        core = self.obs._core
        _, a1, a2, _, _, _ = core.rows()
        a1 = np.asarray(a1, dtype=int)
        a2 = np.asarray(a2, dtype=int)
        tidx = np.asarray(core.time_index(), dtype=int)
        d = self._data
        idx = np.asarray(idx, dtype=int)
        out = []
        for rows, cif, fl in ops:
            sel = idx[d["cif"][idx] == cif]
            wanted = np.isin(d["row"][sel], rows)
            ants = np.asarray(d["a1"], dtype=int)[sel][wanted]
            times = tidx[np.asarray(d["row"], dtype=int)[sel][wanted]]
            mask = np.zeros(a1.shape, dtype=bool)
            for ant, t in {(int(x), int(y)) for x, y in zip(ants, times)}:
                mask |= ((a1 == ant) | (a2 == ant)) & (tidx == t)
            out.append((np.nonzero(mask)[0], cif, fl))
        return out

    # ---- interaction --------------------------------------------------

    def key_help(self):
        return super().key_help() + [
            ("n / p", "next / previous page of baselines"),
            ("space", "flag per baseline or per antenna"),
        ]

    def keyPressEvent(self, ev):
        key = ev.text().lower()
        if key == "n":
            self.set_page(self.page + 1)
        elif key == "p":
            self.set_page(self.page - 1)
        elif key == " ":
            self.by_antenna = not self.by_antenna
            self._update_status()
        else:
            super().keyPressEvent(ev)

    def _status(self):
        mode = "antenna-based" if self.by_antenna else "baseline-based"
        return (f"page {self.page + 1}/{self.npages} of "
                f"{len(self._baselines)} baselines (n/p) | "
                f"{mode} flagging (space) | h: help")


def radplot(obs, quantity="ap", colorby="spw", block=None):
    p = RadPlot(obs, quantity=quantity, colorby=colorby)
    run_if_needed(p, block)
    return p


def projplot(obs, angle_deg=0.0, quantity="ap", colorby="spw", block=None,
             step=10.0):
    p = RadPlot(obs, quantity=quantity, colorby=colorby,
                projection_deg=angle_deg, angle_step=step)
    run_if_needed(p, block)
    return p


def uvplot(obs, colorby="spw", block=None):
    p = UVPlot(obs, colorby=colorby)
    run_if_needed(p, block)
    return p


def vplot(obs, nplot=3, reftel=None, quantity="ap", block=None):
    p = VPlot(obs, nplot=nplot, reftel=reftel, quantity=quantity)
    run_if_needed(p, block)
    return p
