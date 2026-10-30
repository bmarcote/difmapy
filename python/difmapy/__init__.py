"""difmapy: modern in-memory reimplementation of Difmap for VLBI imaging."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from difmapy._core import CoreObservation
from difmapy.observation import MAS, Observation, chanaver, load, observe, uvaver

# Single source of truth for the release version is pyproject.toml, read back
# here from the installed distribution metadata. Do NOT re-export
# `_core.__core_version__`: that is the Rust crate version, which is versioned
# independently and would silently go stale.
try:
    __version__ = _package_version("difmapy")
except PackageNotFoundError:  # imported from a source tree with no install
    __version__ = "0.0.0+unknown"



def dashboard(obs=None, block=None):
    """Open the dashboard: every view of a session in one window, with controls to load a file and calibrate it (see `Observation.dashboard`).

    Parameters
    ----------
    obs : Observation | None, default None
        The observation to show; ``None`` opens the dashboard empty, to load a file from its Main tab.
    block : bool | None, default None
        Wait until the window is closed; ``None`` blocks only in a plain script.

    Returns
    -------
    Dashboard
        The dashboard window; its `obs` is the observation shown.
    """
    from difmapy.plots import dashboard as _dashboard

    return _dashboard(obs, block=block)


__all__ = ["Observation", "CoreObservation", "load", "observe", "uvaver", "chanaver", "MAS",
           "dashboard", "__version__"]
