"""Writing command results out as JSON.

Every difmapy command that returns a dict of numbers takes an
``outfile=`` argument; the result is written there as JSON as well as
returned, so a scripted run can keep its numbers without repeating the
structure by hand.

Numpy scalars and arrays are converted on the way out, and anything
that has no JSON form at all (a complex number, an unexpected object)
is written as a string rather than aborting a run that has already done
the expensive part.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["to_jsonable", "write_json"]


def to_jsonable(obj):
    """`obj` rebuilt out of types `json` can write.

    Arrays become lists, numpy scalars become Python numbers, and
    non-finite floats - which JSON has no syntax for - become null,
    since a NaN in a report means "no solution" and null says that in
    every language that will read the file.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (str, bytes)) or obj is None:
        return obj.decode() if isinstance(obj, bytes) else obj
    if isinstance(obj, (complex, np.complexfloating)):
        return {"re": to_jsonable(obj.real), "im": to_jsonable(obj.imag)}
    return str(obj)


def write_json(path, data, indent=2):
    """Write `data` to `path` as JSON, creating parent directories.

    Returns `data`, so it can wrap a return value.
    """
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(data), fh, indent=indent, sort_keys=False)
        fh.write("\n")
    return data
