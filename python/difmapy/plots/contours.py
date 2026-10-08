"""Contour levels and contour tracing for the map display.

Pure numpy, so it can be tested without Qt. The tracer is a marching
squares that only does arithmetic on the cells a contour actually
crosses: a level costs one comparison of the map plus work proportional
to the length of the contour, which is what makes twenty-odd levels on
a 2048-pixel map affordable on every redraw (pyqtgraph's own
`isocurve` processes the whole array for each level, about a third of a
second apiece at that size).
"""

from __future__ import annotations

import numpy as np

__all__ = ["contour_levels", "contour_segments"]


def contour_levels(rms, vmax, vmin=0.0, nsigma=3.0, factor=np.sqrt(2.0),
                   max_levels=40):
    """The contour levels of a map with noise `rms`.

    Positive levels start at ``nsigma * rms`` and grow by `factor` up to
    the peak `vmax`; negative ones mirror them, from ``-nsigma * rms``
    down to the minimum `vmin`. Returns ``(positive, negative)`` arrays,
    either of which may be empty (no emission above the first level, no
    negative that deep). `max_levels` bounds each, against a noise
    estimate that is absurdly small.
    """
    base = float(nsigma) * float(rms)
    if not np.isfinite(base) or base <= 0.0:
        return np.array([]), np.array([])

    def ladder(limit):
        if not np.isfinite(limit) or limit < base:
            return np.array([])
        n = int(np.floor(np.log(limit / base) / np.log(factor))) + 1
        return base * float(factor) ** np.arange(min(n, int(max_levels)))

    return ladder(float(vmax)), -ladder(-float(vmin))


def contour_segments(data, level):
    """Trace the `level` contour of the 2-D array `data` (row-major,
    ``data[y, x]``).

    Returns ``(x, y)``: the end points of the line segments, two
    consecutive entries per segment (the form `pyqtgraph.arrayToQPath`
    takes with ``connect="pairs"``), in pixel-index coordinates - a
    point at ``(j, i)`` is the centre of pixel ``data[i, j]``. Both are
    empty when the contour does not cross the array.
    """
    d = np.asarray(data)
    empty = (np.array([]), np.array([]))
    if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 2:
        return empty
    above = d >= level
    # Edges the contour crosses: between horizontal and vertical
    # neighbours.
    h = above[:, :-1] != above[:, 1:]       # [ny, nx-1]
    v = above[:-1, :] != above[1:, :]       # [ny-1, nx]
    top, bottom = h[:-1, :], h[1:, :]       # per cell [ny-1, nx-1]
    left, right = v[:, :-1], v[:, 1:]
    count = (top.view(np.uint8) + bottom.view(np.uint8)
             + left.view(np.uint8) + right.view(np.uint8))
    ci, cj = np.nonzero(count)
    if ci.size == 0:
        return empty

    a = d[ci, cj].astype(np.float64)            # top-left
    b = d[ci, cj + 1].astype(np.float64)        # top-right
    c = d[ci + 1, cj + 1].astype(np.float64)    # bottom-right
    e = d[ci + 1, cj].astype(np.float64)        # bottom-left

    def frac(p, q):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (level - p) / (q - p)
        return np.clip(np.nan_to_num(t, nan=0.5), 0.0, 1.0)

    # Where the contour meets each edge of the cell (meaningful only
    # for the edges it does cross): top, right, bottom, left.
    px = np.stack([cj + frac(a, b), cj + 1.0, cj + frac(e, c),
                   cj.astype(np.float64)])
    py = np.stack([ci.astype(np.float64), ci + frac(b, c), ci + 1.0,
                   ci + frac(a, e)])
    crossed = np.stack([top[ci, cj], right[ci, cj], bottom[ci, cj],
                        left[ci, cj]])
    n = count[ci, cj]
    cols = np.arange(ci.size)

    # Two crossings: one segment between them.
    two = n == 2
    first = np.argmax(crossed, axis=0)
    rest = crossed.copy()
    rest[first, cols] = False
    second = np.argmax(rest, axis=0)
    k = cols[two]
    xs = [np.column_stack([px[first[k], k], px[second[k], k]])]
    ys = [np.column_stack([py[first[k], k], py[second[k], k]])]

    # Four crossings, a saddle: two segments, cutting off either the
    # top-left and bottom-right corners or the other two, by which side
    # of the level the cell's centre is on.
    k = cols[n == 4]
    if k.size:
        centre = 0.25 * (a[k] + b[k] + c[k] + e[k]) >= level
        same = centre == (a[k] >= level)   # centre joins top-left/bottom-right
        # edges: 0 top, 1 right, 2 bottom, 3 left
        e1 = np.where(same, 1, 3)          # top-right corner / top-left corner
        e2 = np.where(same, 3, 1)
        xs += [np.column_stack([px[0, k], px[e1, k]]),
               np.column_stack([px[2, k], px[e2, k]])]
        ys += [np.column_stack([py[0, k], py[e1, k]]),
               np.column_stack([py[2, k], py[e2, k]])]
    return np.concatenate(xs).ravel(), np.concatenate(ys).ravel()
