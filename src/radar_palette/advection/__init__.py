"""Advection-aware time interpolation of radar volumes.

Scope
-----
Given two radar volumes bracketing a target time, estimate the echo motion
field between them and reconstruct the volume at the target time.  Two
pieces make that up, and they are deliberately separable:

1. **Motion estimation** on gridded reflectivity (dense optical flow,
   estimated per height level because storm motion is height dependent).
2. **Reconstruction** on the radar's *native* geometry: for each target
   gate, advect its Cartesian position back to the first volume and forward
   to the second, convert both to native (azimuth, slant range) for the same
   sweep, sample, and blend.

Conventions
-----------
Displacement is reported as the *physical* echo displacement in metres from
the first volume to the second, so ``velocity = displacement / dt`` with no
sign flip.  Optical-flow implementations that return a reference-to-moving
warp field have that sign inversion applied internally.

Reflectivity is interpolated in dBZ, not linear Z.

Object flavours
---------------
Entry points accept a :class:`pyart.core.Radar` or an :class:`xarray.DataTree`
and return the same family by default; pass ``output_flavor`` to convert. See
:mod:`radar_palette.io`.

Modules
-------
:mod:`radar_palette.advection.flow`
    Dense, height-resolved motion estimation between two gridded volumes.
:mod:`radar_palette.advection.morph`
    Reconstruction of a volume at an intermediate time, on native geometry.
:mod:`radar_palette.advection.timing`
    Acquisition-time reconstruction, wired into the morph so its output carries
    the instant it represents rather than a bracketing volume's clock.
:mod:`radar_palette.advection.interpolate`
    Flavour-aware entry point for correcting a volume interpolated elsewhere.

Expected benefit
----------------
On a synthetic rigid translation, advecting reduces reconstruction error by 73%
against a naive time average. On real convection --- which grows, decays and changes
shape between volumes in ways no advection scheme captures --- the improvement is a
few percent. The large figure is the easy case, not the representative one.

Optional dependency
-------------------
Motion estimation needs scikit-image: ``pip install radar-palette[advection]``.
The timing functions do not.
"""

from __future__ import annotations

from radar_palette.advection.flow import grid_optical_flow
from radar_palette.advection.interpolate import retime_interpolated_volume
from radar_palette.advection.morph import advection_interpolate
from radar_palette.advection.timing import (
    apply_interpolated_time,
    interpolate_ray_times,
    volume_reference_time,
)

__all__ = [
    "advection_interpolate",
    "apply_interpolated_time",
    "grid_optical_flow",
    "interpolate_ray_times",
    "retime_interpolated_volume",
    "volume_reference_time",
]
