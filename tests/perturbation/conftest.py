"""Neutralize a pertpy import side effect before any test in this directory collects.

`pertpy.tools._coda._sccoda` unconditionally calls `jax.config.update("jax_enable_x64",
True)` at import time, regardless of whether Sccoda is ever used. Every test file in this
directory triggers that import via `pytest.importorskip("pertpy")` at collection time, which
would otherwise leak jax's global float precision into unrelated, already-collected test
modules in the same pytest process (e.g. `tests/test_metrics.py::test_kmeans`, whose
`jax.lax.while_loop` carry state is not written to tolerate float64). Importing pertpy once
here and restoring the prior setting, before pytest collects any test module in this
directory, neutralizes the mutation for the rest of the session (the import is cached, so
later `import pertpy` / `pytest.importorskip("pertpy")` calls are no-ops that don't
re-trigger it).
"""

try:
    import jax

    _was_x64_enabled = jax.config.jax_enable_x64
    import pertpy  # noqa: F401

    jax.config.update("jax_enable_x64", _was_x64_enabled)
except ImportError:
    pass
