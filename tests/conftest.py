"""Shared fixtures.

Optional dependencies are guarded here rather than at import time in the
library, so a minimal install still collects the full test suite (with the
dependent tests skipped).
"""

from __future__ import annotations

import importlib.util

import pytest

requires_skimage = pytest.mark.skipif(
    importlib.util.find_spec("skimage") is None,
    reason="scikit-image required (pip install radar-palette[advection])",
)

requires_finufft = pytest.mark.skipif(
    importlib.util.find_spec("finufft") is None,
    reason="finufft required (pip install radar-palette[spectral])",
)
