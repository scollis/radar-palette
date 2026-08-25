"""Synthetic radar volumes with explicitly controlled acquisition times.

Time-handling tests need volumes whose acquisition times are known exactly and
can be set independently of any field data. These generators build bare PPI
geometries and stamp a realistic scan clock onto them, so a failing timing test
points at the time code rather than at a data-dependent side effect.
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pyart
from netCDF4 import date2num

__all__ = [
    "assign_scan_times",
    "make_empty_ppi_volume",
]

_CF_CALENDAR = "gregorian"
_CF_TIME_UNIT_FORMAT = "seconds since %Y-%m-%dT%H:%M:%SZ"
_DEFAULT_ELEVATIONS_DEG = (0.5, 1.5, 2.5, 3.5, 4.5, 6.0)


def make_empty_ppi_volume(
    nsweeps=4,
    ngates=100,
    rays_per_sweep=360,
    gate_spacing_m=300.0,
    elevations_deg=None,
):
    """Build a multi-elevation PPI volume with realistic geometry and no fields.

    Parameters
    ----------
    nsweeps : int, optional
        Number of elevation sweeps in the volume.
    ngates : int, optional
        Range gates per ray.
    rays_per_sweep : int, optional
        Azimuthal rays per sweep.
    gate_spacing_m : float, optional
        Spacing between range-gate centres, in metres.
    elevations_deg : sequence of float, optional
        Fixed elevation angles in degrees, one per sweep. Defaults to a
        surveillance-like ladder truncated to ``nsweeps``.

    Returns
    -------
    pyart.core.Radar
        Volume with populated geometry and gate coordinates, and no fields.
    """
    if elevations_deg is None:
        elevations_deg = _DEFAULT_ELEVATIONS_DEG[:nsweeps]
    if len(elevations_deg) != nsweeps:
        raise ValueError(f"got {len(elevations_deg)} elevations for {nsweeps} sweeps")

    radar = pyart.testing.make_empty_ppi_radar(ngates, rays_per_sweep, nsweeps)
    radar.range["data"] = (
        np.arange(ngates) * gate_spacing_m + gate_spacing_m / 2.0
    ).astype("float32")
    radar.fixed_angle["data"] = np.asarray(elevations_deg, dtype="float32")
    radar.elevation["data"] = np.repeat(
        np.asarray(elevations_deg, dtype="float32"), rays_per_sweep
    )
    radar.init_gate_x_y_z()
    radar.init_gate_longitude_latitude()
    radar.init_gate_altitude()
    return radar


def assign_scan_times(
    radar,
    volume_start,
    seconds_per_ray=0.5,
    epoch=None,
):
    """Stamp a realistic sequential scan clock onto a volume, in place.

    Rays are acquired in sequence, so ray ``i`` is timed ``i * seconds_per_ray``
    after the start of the volume. This mirrors how a real radar populates
    ``time["data"]`` and gives timing tests a known, monotonic pattern.

    Parameters
    ----------
    radar : pyart.core.Radar
        Volume to stamp. Modified in place.
    volume_start : datetime.datetime
        Acquisition time of the volume's first ray. Naive values are read as UTC.
    seconds_per_ray : float, optional
        Dwell time per ray, in seconds.
    epoch : datetime.datetime, optional
        CF reference epoch the offsets are expressed against. Defaults to
        ``volume_start``, so offsets begin at zero. Passing a different epoch
        exercises the case where two volumes do not share a reference time.

    Returns
    -------
    pyart.core.Radar
        ``radar``, to allow use as an expression.
    """
    if volume_start.tzinfo is None:
        volume_start = volume_start.replace(tzinfo=UTC)
    if epoch is None:
        epoch = volume_start
    elif epoch.tzinfo is None:
        epoch = epoch.replace(tzinfo=UTC)

    ray_offsets_from_start = np.arange(radar.nrays, dtype="float64") * seconds_per_ray
    units = epoch.strftime(_CF_TIME_UNIT_FORMAT)
    start_offset = float(date2num(volume_start, units, calendar=_CF_CALENDAR))

    radar.time = {
        "data": ray_offsets_from_start + start_offset,
        "units": units,
        "calendar": _CF_CALENDAR,
        "standard_name": "time",
        "long_name": "time_in_seconds_since_volume_start",
    }
    return radar
