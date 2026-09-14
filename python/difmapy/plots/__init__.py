"""Interactive pyqtgraph plots.

Requires the ``plot`` extra: ``pip install difmapy[plot]``.
"""

from difmapy.plots.base import close_all_windows, open_windows
from difmapy.plots.diagnostics import corplot, cpplot, fplot, specplot, tplot
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
    "open_windows",
    "close_all_windows",
]
