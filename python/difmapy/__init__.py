"""difmapy: modern in-memory reimplementation of Difmap for VLBI imaging."""

from difmapy._core import CoreObservation, __version__
from difmapy.observation import MAS, Observation, load

__all__ = ["Observation", "CoreObservation", "load", "MAS", "__version__"]
