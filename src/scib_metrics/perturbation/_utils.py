"""Internal helpers for the optional pertpy-backed perturbation module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def import_pertpy() -> ModuleType:
    """Import and return the `pertpy` package, raising a clear error if it's missing.

    `pertpy.tools._coda._sccoda` unconditionally calls
    `jax.config.update("jax_enable_x64", True)` at import time — a side effect of
    importing `pertpy` at all, not of using any Sccoda functionality — which silently
    flips JAX's global float precision for the rest of the process. scib-metrics' own
    jax-based code (e.g. `scib_metrics.utils._kmeans`) assumes the default (float32)
    precision and breaks under x64, so this restores whatever the setting was
    immediately before importing `pertpy`.

    Returns
    -------
    The imported `pertpy` module.
    """
    import jax

    was_x64_enabled = jax.config.jax_enable_x64
    try:
        import pertpy
    except ImportError as e:
        raise ImportError(
            "The `scib_metrics.perturbation` module requires `pertpy`. "
            "Install it with `pip install scib-metrics[perturbation]`."
        ) from e
    jax.config.update("jax_enable_x64", was_x64_enabled)
    return pertpy
