"""difmapy: modern in-memory reimplementation of Difmap for VLBI imaging."""

from difmapy._core import CoreObservation, __version__
from difmapy.observation import MAS, Observation, load, observe

__all__ = ["Observation", "CoreObservation", "load", "observe", "MAS", "__version__"]
