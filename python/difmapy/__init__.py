"""difmapy: modern in-memory reimplementation of Difmap for VLBI imaging."""

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

from difmapy._core import CoreObservation
from difmapy.observation import MAS, Observation, chanaver, load, observe, uvaver


def _release_version() -> str:
    """The release version, whose single source of truth is pyproject.toml.

    From a source checkout (an editable install) it is read from the checkout's pyproject.toml itself: the installed metadata of an editable install is only as new as its last `maturin develop`, and `build_native.py` only swaps the extension, so a venv kept up to date that way went on reporting the version of its first install (0.3.0 on a 0.6.0 tree). Anywhere else (a wheel, the macOS app) it is the installed distribution's metadata. Do NOT re-export `_core.__core_version__`: that is the Rust crate version, which is versioned independently and would silently go stale.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with open(pyproject, "rb") as f:
            project = tomllib.load(f).get("project", {})
        if project.get("name") == "difmapy" and "version" in project:
            return str(project["version"])
    except (OSError, tomllib.TOMLDecodeError):
        pass
    try:
        return _package_version("difmapy")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__ = _release_version()



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
