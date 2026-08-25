"""Synthetic radar objects and analytic fields for tests and examples.

Validating an interpolation operator needs a ground truth the operator did not
see. The generators here build Py-ART ``Radar`` objects on realistic scan
geometries with explicitly controlled acquisition times, so that timing
behaviour can be tested without depending on a particular field or on real data
being present.
"""

from __future__ import annotations

from radar_palette.testing.volumes import assign_scan_times, make_empty_ppi_volume

__all__ = [
    "assign_scan_times",
    "make_empty_ppi_volume",
]
