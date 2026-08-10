"""Internal helpers for the optional pertpy-backed perturbation module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def import_pertpy() -> ModuleType:
    """Import and return the `pertpy` package, raising a clear error if it's missing.

    Returns
    -------
    The imported `pertpy` module.
    """
    try:
        import pertpy
    except ImportError as e:
        raise ImportError(
            "The `scib_metrics.perturbation` module requires `pertpy`. "
            "Install it with `pip install scib-metrics[perturbation]`."
        ) from e
    return pertpy
