"""Shared helpers: beam geometry, decibel handling, sweep bookkeeping.

Both :mod:`radar_palette.advection` and :mod:`radar_palette.gridding` need
to move between native antenna coordinates and Cartesian, and both must be
careful about where in a pipeline a field is in dBZ versus linear Z.  Those
helpers live here so the two subpackages do not depend on each other.

Status
------
Scaffolding only: the implementation lands in follow-up pull requests.
"""

from __future__ import annotations

__all__: list[str] = []
