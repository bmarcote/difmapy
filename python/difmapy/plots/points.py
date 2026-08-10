"""Point-based interactive plots: radplot, projplot, uvplot, vplot."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg

from difmapy.plots.base import FlagScatterPlot, IF_COLORS, run_if_needed


def _stream_arrays(obs):
    """Common per-(row, IF) arrays: complex vis, signed wt, u, v in
    wavelengths, row and IF indices, times."""
    core = obs._core
    vis, wt = core.stream_vis()
    time, a1, a2, us, vs, ws = core.rows()
    sel = core.selection()
    freq = np.asarray(sel["if_freq"])  # [nif]
    nrow, nif = vis.shape
    row = np.repeat(np.arange(nrow), nif)
    cif = np.tile(np.arange(nif), nrow)
    uu = (us[:, None] * freq[None, :]).ravel()
    vv = (vs[:, None] * freq[None, :]).ravel()
    used = np.asarray(sel["if_used"])[cif]
    return {
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


class RadPlot(FlagScatterPlot):
    """Amplitude (or phase) vs UV radius; difmap radplot."""

    def __init__(self, obs, quantity="amp", projection_deg=None):
        self.quantity = quantity
        self.projection = projection_deg
        title = "difmapy projplot" if projection_deg is not None else "difmapy radplot"
        super().__init__(obs, title)
        xlabel = ("projected UV distance" if projection_deg is not None
                  else "UV radius")
        self.pw.setLabel("bottom", f"{xlabel} (M\u03bb)")
        self.pw.setLabel(
            "left", "Amplitude (Jy)" if quantity == "amp" else "Phase (deg)"
        )

    def _collect(self):
        d = _stream_arrays(self.obs)
        if self.projection is None:
            x = np.hypot(d["u"], d["v"]) / 1e6
        else:
            phi = np.deg2rad(self.projection)
            x = np.abs(d["u"] * np.sin(phi) + d["v"] * np.cos(phi)) / 1e6
        y = (np.abs(d["vis"]) if self.quantity == "amp"
             else np.rad2deg(np.angle(d["vis"])))
        return {"x": x, "y": y, "wt": d["wt"], "row": d["row"], "cif": d["cif"]}


class UVPlot(FlagScatterPlot):
    """UV coverage; difmap uvplot (conjugate points included)."""

    def __init__(self, obs):
        super().__init__(obs, "difmapy uvplot")
        self.pw.setLabel("bottom", "U (M\u03bb)")
        self.pw.setLabel("left", "V (M\u03bb)")
        self.pw.getPlotItem().vb.setAspectLocked(True)
        self.pw.getPlotItem().vb.invertX(True)

    def _collect(self):
        d = _stream_arrays(self.obs)
        return {
            "x": np.concatenate([d["u"], -d["u"]]) / 1e6,
            "y": np.concatenate([d["v"], -d["v"]]) / 1e6,
            "wt": np.concatenate([d["wt"], d["wt"]]),
            "row": np.concatenate([d["row"], d["row"]]),
            "cif": np.concatenate([d["cif"], d["cif"]]),
        }


class VPlot(FlagScatterPlot):
    """Visibility amplitude or phase vs time for the baselines of a
    reference telescope; difmap vplot."""

    def __init__(self, obs, reftel=None, quantity="amp"):
        self.quantity = quantity
        core = obs._core
        names = core.antenna_names
        self.reftel = names.index(str(reftel)) if reftel is not None else 0
        super().__init__(obs, f"difmapy vplot ({names[self.reftel]})")
        self.pw.setLabel("bottom", "Time (hours since reference day)")
        self.pw.setLabel(
            "left", "Amplitude (Jy)" if quantity == "amp" else "Phase (deg)"
        )

    def _collect(self):
        d = _stream_arrays(self.obs)
        m = (d["a1"] == self.reftel) | (d["a2"] == self.reftel)
        y = (np.abs(d["vis"]) if self.quantity == "amp"
             else np.rad2deg(np.angle(d["vis"])))
        return {
            "x": d["time"][m] / 3600.0,
            "y": y[m],
            "wt": d["wt"][m],
            "row": d["row"][m],
            "cif": d["cif"][m],
        }


def radplot(obs, quantity="amp", block=None):
    p = RadPlot(obs, quantity=quantity)
    run_if_needed(p, block)
    return p


def projplot(obs, angle_deg=0.0, quantity="amp", block=None):
    p = RadPlot(obs, quantity=quantity, projection_deg=angle_deg)
    run_if_needed(p, block)
    return p


def uvplot(obs, block=None):
    p = UVPlot(obs)
    run_if_needed(p, block)
    return p


def vplot(obs, reftel=None, quantity="amp", block=None):
    p = VPlot(obs, reftel=reftel, quantity=quantity)
    run_if_needed(p, block)
    return p
