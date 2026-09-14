"""difmapy: modern in-memory reimplementation of Difmap for VLBI imaging."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from difmapy._core import CoreObservation
from difmapy.observation import MAS, Observation, load, observe

# Single source of truth for the release version is pyproject.toml, read back
# here from the installed distribution metadata. Do NOT re-export
# `_core.__core_version__`: that is the Rust crate version, which is versioned
# independently and would silently go stale.
try:
    __version__ = _package_version("difmapy")
except PackageNotFoundError:  # imported from a source tree with no install
    __version__ = "0.0.0+unknown"

__all__ = ["Observation", "CoreObservation", "load", "observe", "MAS", "__version__"]
