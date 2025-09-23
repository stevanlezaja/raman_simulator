"""
Module exposing fiber classes for easy import.

This module re-exports the main Fiber base class and several commonly used
fiber implementations for convenience. It allows users to import the fiber
types directly from this package without accessing the internal fiber module.

Available classes:
    - Fiber: Abstract base class for all fibers.
    - SuperLargeEffectiveArea: Fiber with very large effective area.
    - NonZeroDispersionFiber: Fiber with moderate area and non-zero dispersion.
    - DispersionCompensatingFiber: Small-area fiber for dispersion compensation.
    - StandardSingleModeFiber: Standard single-mode fiber with measured Raman gain.
    - ChristopheExperimentFiber: Experimental fiber from Christophe's measurements.
"""

from .fiber import Fiber, SuperLargeEffectiveArea, NonZeroDispersionFiber, \
    DispersionCompensatingFiber, StandardSingleModeFiber, ChristopheExperimentFiber

__all__ = ["Fiber", "SuperLargeEffectiveArea", "NonZeroDispersionFiber",
           "DispersionCompensatingFiber", "StandardSingleModeFiber", "ChristopheExperimentFiber"]
