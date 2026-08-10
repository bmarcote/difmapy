"""Interactive pyqtgraph plots (radplot, uvplot, vplot, mapplot).

Requires the ``plot`` extra: ``pip install difmapy[plot]``.
"""

from difmapy.plots.mapplot import mapplot
from difmapy.plots.points import projplot, radplot, uvplot, vplot

__all__ = ["radplot", "projplot", "uvplot", "vplot", "mapplot"]
