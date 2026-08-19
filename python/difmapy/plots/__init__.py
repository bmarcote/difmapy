"""Interactive pyqtgraph plots.

Requires the ``plot`` extra: ``pip install difmapy[plot]``.
"""

from difmapy.plots.diagnostics import corplot, cpplot, specplot, tplot
from difmapy.plots.mapplot import mapplot
from difmapy.plots.points import projplot, radplot, uvplot, vplot

__all__ = [
    "radplot",
    "projplot",
    "uvplot",
    "vplot",
    "mapplot",
    "cpplot",
    "tplot",
    "corplot",
    "specplot",
]
