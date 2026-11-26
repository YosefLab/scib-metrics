from ._core import (
    BatchCorrection,
    Benchmarker,
    BioConservation,
    CoordinatePreservation,
    DomainBoundary,
    NichePreservation,
    SpatialConservation,
)

__all__ = [
    "Benchmarker",
    "BioConservation",
    "BatchCorrection",
    "CoordinatePreservation",
    "NichePreservation",
    "DomainBoundary",
    "SpatialConservation",  # backward-compat alias for CoordinatePreservation
]
