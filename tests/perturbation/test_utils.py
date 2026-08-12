import sys
from unittest.mock import patch

import pytest

from scib_metrics.perturbation._utils import import_pertpy


def test_import_pertpy_returns_module():
    pertpy = pytest.importorskip("pertpy")
    assert import_pertpy() is pertpy


def test_import_pertpy_raises_clear_error_when_missing():
    with patch.dict(sys.modules, {"pertpy": None}):
        with pytest.raises(ImportError, match=r"scib-metrics\[perturbation\]"):
            import_pertpy()
