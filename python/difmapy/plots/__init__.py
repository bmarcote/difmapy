"""Interactive pyqtgraph plots."""

from difmapy.plots.base import close_all_windows, open_windows
from difmapy.plots.dashboard import dashboard
from difmapy.plots.diagnostics import corplot, cpplot, fplot, possm, specplot, tplot
from difmapy.plots.mapplot import maplot, mapplot
from difmapy.plots.points import projplot, radplot, uvplot, vplot

__all__ = [
    "radplot",
    "projplot",
    "uvplot",
    "vplot",
    "mapplot",
    "maplot",
    "cpplot",
    "tplot",
    "corplot",
    "specplot",
    "fplot",
    "possm",
    "dashboard",
    "open_windows",
    "close_all_windows",
]

from difmapy.plots.base import get_mode, set_mode  # noqa: E402

__all__ += ["set_mode", "get_mode"]
