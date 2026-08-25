"""Reconstruction of per-ray acquisition times for interpolated radar volumes.

An advection-interpolated volume represents an instant that was never observed.
Everything else about it is derived --- the reflectivity is advected, the geometry
is inherited --- but its *time* has to be constructed, and getting that wrong is
silent: the volume looks valid, plots without complaint, and misreports when it
happened. Downstream that corrupts any time-ordered use of the data (accumulation,
tracking, matching against another instrument) in a way that is hard to trace back
to its cause.

The reconstruction has two parts, and the distinction matters:

*Volume-level offset.*
    A volume observed at ``t1`` and one at ``t2``, interpolated at fraction
    ``alpha``, represents ``t1 + alpha * (t2 - t1)``. This is the part that a
    naive implementation gets wrong by inheriting ``t1`` unchanged.

*Intra-volume structure.*
    A radar volume is not an instant. Its rays are acquired in sequence over the
    scan, so ``time["data"]`` holds one offset per ray, spanning minutes for a
    full volume coverage pattern. That per-ray structure is a property of the
    *scan strategy*, not of the interpolation, so it is preserved rather than
    collapsed to a single stamp: the ray pattern is carried through and the whole
    volume is shifted.

All arithmetic is done on absolute times decoded from each volume's own CF units,
because two volumes are not guaranteed to share an epoch --- a real hazard when
volumes come from separate files.
"""

from __future__ import annotations

from datetime import UTC, timedelta

import numpy as np
from netCDF4 import date2num, num2date

__all__ = [
    "apply_interpolated_time",
    "interpolate_ray_times",
    "volume_reference_time",
]

_CF_CALENDAR = "gregorian"
_CF_TIME_UNIT_FORMAT = "seconds since %Y-%m-%dT%H:%M:%SZ"
_ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _decode_ray_times(time_dict):
    """Decode a CF time dictionary to timezone-aware datetimes, one per ray.

    Parameters
    ----------
    time_dict : dict
        A Py-ART ``radar.time`` dictionary: ``data`` holds per-ray offsets and
        ``units`` the CF reference-time string they are measured against.

    Returns
    -------
    numpy.ndarray
        Object array of timezone-aware :class:`datetime.datetime`, length nrays.
        Masked offsets are excluded, so no-data rays cannot pull a mean toward a
        fill value.
    """
    offsets = np.ma.masked_invalid(np.ma.asarray(time_dict["data"]).astype("float64"))
    valid_offsets = np.atleast_1d(offsets.compressed())
    if valid_offsets.size == 0:
        raise ValueError("radar.time['data'] contains no valid ray times")

    decoded = num2date(
        valid_offsets,
        time_dict["units"],
        calendar=time_dict.get("calendar", _CF_CALENDAR),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.array(
        [
            moment if moment.tzinfo else moment.replace(tzinfo=UTC)
            for moment in np.atleast_1d(decoded)
        ]
    )


def volume_reference_time(radar):
    """Reduce a radar volume's per-ray times to one representative instant.

    A volume spans the duration of its scan, so reducing it to one instant is a
    choice. The mean ray time is used: it is the centroid of the acquisition
    window, so it does not drift with scan strategy the way the first or last ray
    time would.

    Parameters
    ----------
    radar : pyart.core.Radar
        Volume whose ``time`` dictionary is read. Not modified.

    Returns
    -------
    datetime.datetime
        Timezone-aware (UTC) mean acquisition time of the volume's rays.

    Notes
    -----
    The result is independent of the CF epoch the volume happens to be encoded
    against, because offsets are decoded to absolute times before averaging.
    """
    ray_times = _decode_ray_times(radar.time)
    first_ray_time = ray_times[0]
    mean_offset_seconds = float(
        np.mean([(moment - first_ray_time).total_seconds() for moment in ray_times])
    )
    return first_ray_time + timedelta(seconds=mean_offset_seconds)


def interpolate_ray_times(radar_early, radar_late, alpha):
    """Build the per-ray time dictionary of a volume interpolated between two others.

    The interpolated volume represents the instant a fraction ``alpha`` of the way
    from ``radar_early`` to ``radar_late``. Its rays keep the acquisition pattern
    of ``radar_early`` --- ray-to-ray spacing is set by the scan strategy, which
    interpolation does not change --- shifted so that the volume as a whole lands
    at the target instant.

    Parameters
    ----------
    radar_early, radar_late : pyart.core.Radar
        Volumes bracketing the target instant, ``radar_early`` observed first.
        Neither is modified. Their ray counts need not match, and they need *not*
        share a CF epoch. Only ``radar_early``'s ray pattern and the two volumes'
        reference times are used, so ``radar_late`` contributes one scalar.
    alpha : float
        Fractional position of the target between ``radar_early`` (0.0) and
        ``radar_late`` (1.0). Values outside ``[0, 1]`` extrapolate linearly.

    Returns
    -------
    dict
        A CF-compliant Py-ART time dictionary. ``units`` is expressed against the
        interpolated volume's own first ray, so the offsets read naturally rather
        than as a large jump from an inherited epoch.

    Raises
    ------
    ValueError
        If ``alpha`` is not finite.

    Notes
    -----
    Ray counts are deliberately *not* required to match. Consecutive volumes from
    the same radar running the same strategy routinely differ by a ray or two --
    measured on ARM C-SAPR2, three consecutive 15-sweep volumes gave 13955, 13951
    and 13954 rays -- so an equality requirement rejects essentially every real
    pair. Nothing below pairs rays between the volumes: the output carries
    ``radar_early``'s acquisition pattern shifted by a single scalar derived from
    the two reference times, so there is no correspondence to violate.
    """
    alpha = float(alpha)
    if not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha!r}")

    early_ray_times = _decode_ray_times(radar_early.time)
    seconds_between_volumes = (
        volume_reference_time(radar_late) - volume_reference_time(radar_early)
    ).total_seconds()
    volume_shift = timedelta(seconds=alpha * seconds_between_volumes)

    interpolated_ray_times = [moment + volume_shift for moment in early_ray_times]
    epoch = interpolated_ray_times[0]
    units = epoch.strftime(_CF_TIME_UNIT_FORMAT)

    return {
        "data": np.asarray(
            date2num(interpolated_ray_times, units, calendar=_CF_CALENDAR),
            dtype="float64",
        ),
        "units": units,
        "calendar": _CF_CALENDAR,
        "standard_name": "time",
        "long_name": "time_in_seconds_since_volume_start",
        "comment": (
            "Time reconstructed by advection interpolation; this volume "
            "represents an instant between two observed volumes."
        ),
    }


def _format_iso_timestamp(moment):
    """Render a datetime as a CF/cfradial ISO-8601 UTC timestamp string."""
    return moment.astimezone(UTC).strftime(_ISO_TIMESTAMP_FORMAT)


def apply_interpolated_time(radar_out, radar_early, radar_late, alpha):
    """Stamp reconstructed acquisition times onto an interpolated volume, in place.

    An interpolated volume is typically built by copying the geometry of the
    earlier bracketing volume, which means it arrives carrying that volume's
    acquisition times. This replaces them with the times the volume actually
    represents, and refreshes the derived time metadata so the record stays
    self-consistent.

    Parameters
    ----------
    radar_out : pyart.core.Radar
        Interpolated volume to correct. Modified in place.
    radar_early, radar_late : pyart.core.Radar
        Volumes the interpolation was built from, ``radar_early`` observed first.
        Neither is modified.
    alpha : float
        Fractional position of ``radar_out`` between ``radar_early`` (0.0) and
        ``radar_late`` (1.0).

    Returns
    -------
    pyart.core.Radar
        ``radar_out``, to allow use as an expression.

    Notes
    -----
    ``time_coverage_start`` / ``time_coverage_end`` are updated only when the
    volume already carries them: this refreshes stale metadata but does not
    fabricate fields the source volume never had.
    """
    radar_out.time = interpolate_ray_times(radar_early, radar_late, alpha)

    ray_times = _decode_ray_times(radar_out.time)
    for metadata_key, moment in (
        ("time_coverage_start", ray_times.min()),
        ("time_coverage_end", ray_times.max()),
    ):
        if metadata_key in radar_out.metadata:
            radar_out.metadata[metadata_key] = _format_iso_timestamp(moment)

    provenance = (
        f"radar_palette.advection: time reconstructed at alpha={alpha:g} between "
        f"{_format_iso_timestamp(volume_reference_time(radar_early))} and "
        f"{_format_iso_timestamp(volume_reference_time(radar_late))}"
    )
    existing_history = radar_out.metadata.get("history", "")
    radar_out.metadata["history"] = (
        f"{existing_history}\n{provenance}" if existing_history else provenance
    )
    return radar_out
