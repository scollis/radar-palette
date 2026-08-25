"""Radar geometry on the 4/3 effective earth, and its exact analytic inverse.

Both transforms here work in **metres in, metres out**, and use the meteorological
azimuth convention (0 degrees is north, increasing clockwise).

Relationship to Py-ART
----------------------
:func:`antenna_to_cartesian_43` reproduces :func:`pyart.core.antenna_to_cartesian`
exactly, with one interface difference: Py-ART takes range in **kilometres** and
returns metres, while this function takes and returns metres. A round trip that
overlooks that is wrong by a factor of 1000.

:func:`cartesian_to_antenna_43` exists because
:func:`pyart.core.cartesian_to_antenna` is **not the inverse of Py-ART's own
forward transform**. The forward transform returns great-circle *arc length* along
the 4/3 earth; Py-ART's inverse computes a flat straight-line ``sqrt(x^2+y^2+z^2)``
with ``arctan(z / hypot(x, y))`` and no curvature term. The result is a silent
systematic bias, not noise. Measured against this module's forward transform, with
the range error maximised over elevation (it peaks near 35 degrees at every range,
not at grazing incidence):

=========  ====================  =======================  =====================
Range      Range error at 1 deg  Worst range error        Elevation error
=========  ====================  =======================  =====================
10 km      -0.1 m                -2.3 m                   +34 mdeg
118 km     -19 m                 -313 m                   +398 mdeg
250 km     -109 m                -1392 m                  +843 mdeg
460 km     -496 m                -4649 m                  +1552 mdeg
=========  ====================  =======================  =====================

The elevation dependence matters when judging severity: at 118 km the range error
is ~19 m for a near-horizontal beam but ~313 m for a beam near 35 degrees, so a
single headline number understates it for high tilts and overstates it for low
ones.

Azimuth is unaffected --- both use the same convention, agreeing to ~1e-14 degrees.
Any product that inverts a Cartesian grid back to antenna coordinates with the
Py-ART function inherits the range and elevation bias, which is why this module
ships its own inverse; it closes the round trip to ~1e-10 m.
``tests/gridding/test_geometry.py`` asserts the discrepancy explicitly, so a future
Py-ART fix will surface as a test failure rather than going unnoticed.

One further interface trap: Py-ART's *forward* transform takes range in
**kilometres**, while its *inverse* returns range in **metres**. The asymmetry is
easy to get backwards in either direction.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EARTH_RADIUS_EFFECTIVE_M",
    "antenna_to_cartesian_43",
    "cartesian_to_antenna_43",
]

# 4/3 effective earth radius in metres --- the same constant Py-ART uses inside
# antenna_to_cartesian, kept explicit here so the analytic inverse is guaranteed
# consistent with the forward transform rather than approximately so.
EARTH_RADIUS_EFFECTIVE_M = 6371.0 * 1000.0 * 4.0 / 3.0


def antenna_to_cartesian_43(ranges_m, azimuths_deg, elevations_deg):
    """Convert antenna coordinates to Cartesian offsets on the 4/3 earth.

    Parameters
    ----------
    ranges_m : array_like
        Slant range from the radar, in metres.
    azimuths_deg : array_like
        Azimuth in degrees, meteorological convention: 0 is north, increasing
        clockwise, so 90 degrees is east.
    elevations_deg : array_like
        Beam elevation above the horizon, in degrees.

    Returns
    -------
    x, y, z : numpy.ndarray
        Offsets from the radar in metres: ``x`` east, ``y`` north, ``z`` height
        above the radar. ``hypot(x, y)`` is great-circle **arc length** along the
        4/3 earth, not straight-line ground distance.

    Notes
    -----
    Arguments broadcast against one another. Unlike
    :func:`pyart.core.antenna_to_cartesian`, range is in metres rather than
    kilometres.
    """
    ranges_m = np.asarray(ranges_m, dtype=np.float64)
    elevation_rad = np.deg2rad(np.asarray(elevations_deg, dtype=np.float64))
    azimuth_rad = np.deg2rad(np.asarray(azimuths_deg, dtype=np.float64))
    earth_radius = EARTH_RADIUS_EFFECTIVE_M

    height = (
        np.sqrt(
            ranges_m**2
            + earth_radius**2
            + 2.0 * ranges_m * earth_radius * np.sin(elevation_rad)
        )
        - earth_radius
    )
    arc_length = earth_radius * np.arcsin(
        ranges_m * np.cos(elevation_rad) / (earth_radius + height)
    )
    return (
        arc_length * np.sin(azimuth_rad),
        arc_length * np.cos(azimuth_rad),
        height,
    )


def cartesian_to_antenna_43(x, y, z):
    """Convert Cartesian offsets back to antenna coordinates (exact inverse).

    This is the exact analytic inverse of :func:`antenna_to_cartesian_43`, closing
    the round trip to floating-point precision. See the module docstring for why
    :func:`pyart.core.cartesian_to_antenna` cannot be used for this.

    Parameters
    ----------
    x, y, z : array_like
        Offsets from the radar in metres: ``x`` east, ``y`` north, ``z`` height
        above the radar. ``hypot(x, y)`` is interpreted as great-circle arc length,
        matching the forward transform.

    Returns
    -------
    ranges_m : numpy.ndarray
        Slant range in metres.
    azimuths_deg : numpy.ndarray
        Azimuth in degrees on ``[0, 360)``, meteorological convention.
    elevations_deg : numpy.ndarray
        Elevation above the horizon, in degrees.

    Notes
    -----
    Derivation. With ``arc = hypot(x, y)`` the arc length and ``rho = R + z``, the
    forward relations give ``r cos(el) = rho sin(arc / R)`` and, eliminating ``r``,
    ``r sin(el) = rho cos(arc / R) - R``. Hence ``r = hypot`` of those two
    components and ``el = atan2`` of them.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    earth_radius = EARTH_RADIUS_EFFECTIVE_M

    arc_length = np.hypot(x, y)
    radial_distance = earth_radius + z
    horizontal_component = radial_distance * np.sin(arc_length / earth_radius)
    vertical_component = (
        radial_distance * np.cos(arc_length / earth_radius) - earth_radius
    )

    ranges_m = np.hypot(horizontal_component, vertical_component)
    elevations_deg = np.rad2deg(np.arctan2(vertical_component, horizontal_component))
    azimuths_deg = np.rad2deg(np.arctan2(x, y)) % 360.0
    return ranges_m, azimuths_deg, elevations_deg
