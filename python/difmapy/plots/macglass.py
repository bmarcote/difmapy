"""A glass background for difmapy's windows on macOS.

Off by default: the dashboard's window is opaque. ``$DIFMAPY_GLASS=1`` (or ``glass``) puts it on a Liquid Glass background, ``material`` on the translucent window material. The work is done by a few lines of AppKit in ``_glass.dylib`` (source: ``packaging/macos/glass.m``), called through ctypes so that PyObjC is not needed; where the library is missing (GNU/Linux, a build without it) nothing happens.

The window must be made translucent before Qt creates its native window, which is why `enable` is called from the window's constructor, before it is shown.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

from pyqtgraph.Qt import QtCore

__all__ = ["available", "enable"]

_LIB = None


def _library():
    """The helper library, or None."""
    global _LIB
    if _LIB is None:
        path = Path(__file__).resolve().parent.parent / "_glass.dylib"
        try:
            lib = ctypes.CDLL(str(path))
            lib.difmapy_glass.argtypes = (ctypes.c_void_p, ctypes.c_int)
            lib.difmapy_glass.restype = ctypes.c_int
            _LIB = lib
        except OSError:
            _LIB = False
    return _LIB or None


def available() -> bool:
    """True on macOS when the helper library is there and the glass is asked for with ``$DIFMAPY_GLASS`` (``1``, ``glass`` or ``material``)."""
    if sys.platform != "darwin" or os.environ.get("DIFMAPY_GLASS", "0").lower() not in ("1", "glass", "material"):
        return False
    if os.environ.get("QT_QPA_PLATFORM", "").lower().startswith(("offscreen", "minimal")):
        return False
    return _library() is not None


def enable(window, kind=None) -> int:
    """Put `window` (a top-level QWidget not shown yet) on a glass background.

    Parameters
    ----------
    window : QWidget
        The top-level window.
    kind : int | None, default None
        2 for Liquid Glass (NSGlassEffectView, macOS 26+), 1 for the translucent window material; ``None`` reads ``$DIFMAPY_GLASS`` ("glass" or "material"), defaulting to glass where AppKit has it.

    Returns
    -------
    int
        2 if Liquid Glass was used, 1 for the window material, 0 if nothing was done.
    """
    if not available():
        return 0
    if kind is None:
        kind = 1 if os.environ.get("DIFMAPY_GLASS", "").lower() == "material" else 2
    window.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
    view = int(window.winId())  # creates the native window: an NSView*
    return int(_library().difmapy_glass(ctypes.c_void_p(view), int(kind)))
