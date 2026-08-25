"""Reconstruction of a radar volume at a time between two observed volumes.

Averaging two volumes in time smears a moving storm into a ghost: the echo appears in
both its old and new positions at half strength. This module instead *advects* the
bracketing volumes to the target time along an estimated motion field before blending,
so a feature arrives at one position with its intensity intact.

Native geometry
---------------
The morph is applied **per gate, in the radar's own (azimuth, slant range, elevation)
coordinates**, so the result is an ordinary :class:`~pyart.core.Radar` rather than a
Cartesian grid. Only the motion *estimation* goes through a grid, because optical flow
needs a regular raster; the reconstruction itself never leaves antenna coordinates.

Sign convention, and why it is stated so loudly
-----------------------------------------------
For fractional time ``alpha`` in ``[0, 1]``:

* the **earlier** volume is sampled at ``position - alpha * displacement``
* the **later** volume is sampled at ``position + (1 - alpha) * displacement``

Sampling *behind* the gate is correct: to learn what is at a gate now, look where that
echo *was* in the earlier volume. Getting this backwards produces output that looks
entirely plausible but is worse than a naive time average, which is why the direction
is pinned by an end-to-end test rather than by argument --- see
:mod:`radar_palette.advection.flow`.

How much this actually buys you
-------------------------------
On a synthetic rigid translation the improvement is large: 0.047 dBZ RMSE against an
analytic truth, versus 0.171 dBZ for a naive time average, a 73% reduction. **Do not
expect that on real data.** A clean translating blob is the easiest possible case;
real convection grows, decays and changes shape between volumes, and none of that is
motion an advection scheme can capture. On real storm scenes the improvement over a
time average is a few percent, not seventy.

Timing
------
The reconstructed volume represents an instant that was never observed, so its
acquisition time is corrected by :mod:`radar_palette.advection.timing` before it is
returned. Without that step the output would inherit the earlier volume's clock from
the ``deepcopy`` it is built on, and report having been observed up to a full volume
interval before the instant it depicts.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import numpy as np

from radar_palette.advection.flow import grid_optical_flow
from radar_palette.advection.timing import (
    apply_interpolated_time,
    volume_reference_time,
)
from radar_palette.io import (
    resolve_output_flavor,
    to_pyart_radar,
    to_radar_flavor,
)

__all__ = ["advection_interpolate"]

#: Field value (dBZ) above which a grid cell counts as echo when extending the motion
#: field into no-echo regions. Flow is meaningless where there is nothing to track, so
#: those cells take the level mean rather than their own (spurious) estimate.
DEFAULT_ECHO_THRESHOLD_DBZ = 5.0

DEFAULT_FLOW_GRID_SHAPE = (8, 241, 241)
DEFAULT_FLOW_GRID_LIMITS = (
    (1000.0, 8000.0),
    (-120000.0, 120000.0),
    (-120000.0, 120000.0),
)


def _extend_over_no_echo(displacement, echo_mask):
    """Replace the displacement where there is no echo with the level's echo mean.

    Optical flow returns *something* everywhere, but where neither volume has echo
    that something is noise. Substituting the mean of the level's echo-bearing cells
    keeps the field smooth and stops no-echo noise from dragging gates around.
    """
    extended = displacement.copy()
    for level in range(extended.shape[0]):
        level_values = extended[level]
        level_echo = echo_mask[level]
        level_values[~level_echo] = (
            level_values[level_echo].mean() if level_echo.any() else 0.0
        )
    return extended


def _sweep_sampler(radar, sweep, field):
    """Azimuth-sorted (azimuth, range, values) for one sweep, ready to interpolate.

    Repeated azimuths are collapsed to one ray, averaging the rays that share an
    angle. A real antenna does record the same azimuth twice -- measured on ARM
    C-SAPR2, 3 of 195 sweeps across 13 consecutive volumes contain one duplicate,
    which is what a brief dwell looks like in the data -- and the interpolator built
    in :func:`_sample_sweep` needs a *strictly* ascending axis, so a single repeat
    would otherwise make the whole reconstruction raise. Averaging rather than
    discarding keeps both measurements: they are independent samples of the same
    beam position, so their mean is the better estimate of it.
    """
    start, end = radar.get_start_end(sweep)
    azimuths = radar.azimuth["data"][start : end + 1]
    values = np.ma.filled(
        radar.fields[field]["data"][start : end + 1].astype("float64"), np.nan
    )
    order = np.argsort(azimuths)
    azimuths, values = azimuths[order], values[order]

    unique_azimuths, inverse = np.unique(azimuths, return_inverse=True)
    if unique_azimuths.size != azimuths.size:
        # nanmean per group: a duplicated ray may be masked in one copy and not the
        # other, and an all-nan group must stay nan rather than becoming zero.
        finite = np.isfinite(values)
        totals = np.zeros((unique_azimuths.size, values.shape[1]))
        counts = np.zeros_like(totals)
        np.add.at(totals, inverse, np.where(finite, values, 0.0))
        np.add.at(counts, inverse, finite)
        with np.errstate(invalid="ignore"):
            values = np.where(counts > 0, totals / counts, np.nan)
        azimuths = unique_azimuths

    return azimuths, radar.range["data"], values


def _sample_sweep(azimuths_deg, ranges_m, sampler):
    """Bilinear sample of one sweep at arbitrary (azimuth, range), wrapping azimuth.

    The azimuth axis is tiled at -360, 0 and +360 degrees so a query that lands just
    across the 0/360 seam interpolates between real neighbouring rays instead of
    falling off the end of the array.
    """
    from scipy.interpolate import RegularGridInterpolator

    source_azimuths, source_ranges, values = sampler
    wrapped_azimuths = np.concatenate(
        [source_azimuths - 360.0, source_azimuths, source_azimuths + 360.0]
    )
    wrapped_values = np.concatenate([values, values, values], axis=0)
    interpolator = RegularGridInterpolator(
        (wrapped_azimuths, source_ranges),
        wrapped_values,
        bounds_error=False,
        fill_value=np.nan,
    )
    queries = np.column_stack([azimuths_deg.ravel(), ranges_m.ravel()])
    return interpolator(queries).reshape(azimuths_deg.shape)


def _resolve_alpha(earlier, later, alpha, target_time):
    """Fractional position of the target between the two volumes.

    A *naive* ``target_time`` is read as UTC rather than rejected. CF times are UTC
    by construction and :func:`volume_reference_time` returns an aware instant, but
    ``pyart.util.datetime_from_radar`` returns a naive one -- so building a target
    from a Py-ART time and passing it here used to raise ``TypeError: can't subtract
    offset-naive and offset-aware datetimes``. That is the most natural call a
    Py-ART user can make. This mirrors the rule already applied to decoded ray times
    in :func:`radar_palette.advection.timing._decode_ray_times`.

    The normalisation rebuilds the instant field by field rather than calling
    ``replace(tzinfo=UTC)``, because Py-ART hands back a ``cftime.DatetimeGregorian``
    -- not a :class:`datetime.datetime` subclass, and its ``replace()`` rejects
    ``tzinfo`` outright.
    """
    if alpha is not None:
        return float(alpha)
    if target_time is None:
        return 0.5
    if getattr(target_time, "tzinfo", None) is None:
        target_time = datetime(
            target_time.year,
            target_time.month,
            target_time.day,
            target_time.hour,
            target_time.minute,
            target_time.second,
            target_time.microsecond,
            tzinfo=UTC,
        )
    earlier_time = volume_reference_time(earlier)
    span_s = (volume_reference_time(later) - earlier_time).total_seconds()
    if span_s == 0.0:
        raise ValueError(
            "the two volumes have the same reference time, so target_time cannot be "
            "converted to a fraction; pass alpha instead"
        )
    return (target_time - earlier_time).total_seconds() / span_s


def advection_interpolate(
    earlier,
    later,
    alpha=None,
    target_time=None,
    field=None,
    carry_fields=None,
    output_flavor=None,
    grid_shape=DEFAULT_FLOW_GRID_SHAPE,
    grid_limits=DEFAULT_FLOW_GRID_LIMITS,
    echo_threshold_dbz=DEFAULT_ECHO_THRESHOLD_DBZ,
    interp_field_name=None,
    gridding_kwargs=None,
    **flow_kwargs,
):
    """Reconstruct a volume at a time between two observed volumes.

    Parameters
    ----------
    earlier, later : pyart.core.Radar or xarray.DataTree
        Volumes bracketing the target time, ``earlier`` first. Both object families
        are accepted; see :mod:`radar_palette.io`.
    alpha : float, optional
        Fractional time of the target: 0.0 is ``earlier``, 1.0 is ``later``. Defaults
        to the midpoint, or is derived from ``target_time`` when that is given.
    target_time : datetime.datetime, optional
        Absolute target time, converted to ``alpha`` using the volumes' reference
        times. Ignored when ``alpha`` is given.
    field : str, optional
        Field to interpolate, and the **tracer** the motion is estimated from.
        Defaults to ``"reflectivity"``.
    carry_fields : sequence of str, optional
        Additional fields to warp through the *same* motion, resampling and blend.
        The motion is a property of the scene, not of the variable, so estimating it
        once from the tracer and applying it to every variable is both cheaper and
        more defensible than calling this function once per field --- which would
        derive a different motion field from each.

        This matters most for polarimetric variables, none of which is a tracer:
        differential reflectivity is a shape measure, correlation coefficient is
        closer to a quality flag, and Doppler velocity is signed and folded at the
        Nyquist interval. Tracking any of them would be tracking the wrong thing.

        Each carried field keeps its own metadata, so units are not overwritten with
        the tracer's. ``interp_field_name`` renames only the tracer.

        Caveat for **folded velocity**: the blend is a weighted mean, so averaging
        values either side of the Nyquist interval gives an answer near zero where
        the truth wraps. Measured on ARM C-SAPR2, gates whose velocity changes by
        more than one Nyquist interval between scans are 7.1% within echo above
        15 dBZ but 20.4% across all valid gates, the rest being noise. Dealias
        before carrying velocity if the interpolated values are to be read
        quantitatively.
    output_flavor : {"pyart", "xradar"}, optional
        Object family to return. Defaults to mirroring the input.
    grid_shape, grid_limits : optional
        Shape and extent of the intermediate Cartesian grid used *only* for motion
        estimation. The reconstruction itself stays on native geometry.
    echo_threshold_dbz : float, optional
        Value above which a grid cell counts as echo when extending the motion field.
    interp_field_name : str, optional
        Name for the field in the returned volume. Defaults to ``field``.
    gridding_kwargs : dict, optional
        Extra keyword arguments for :func:`pyart.map.grid_from_radars`.
    **flow_kwargs
        Passed to :func:`radar_palette.advection.flow.grid_optical_flow`.

    Returns
    -------
    pyart.core.Radar or xarray.DataTree
        A volume on ``earlier``'s geometry, carrying the reconstructed tracer field,
        any carried fields, and the acquisition time it represents.

    Raises
    ------
    ImportError
        If scikit-image is not installed.
    KeyError
        If ``field`` or any of ``carry_fields`` is absent from either volume.

    Notes
    -----
    The improvement over a naive time average is large on a rigid translation and
    modest on real convection; see the module docstring for measured figures.
    """
    import pyart
    from scipy.interpolate import RegularGridInterpolator

    # to_pyart_radar reports the flavour it adapted from, so no separate detect call
    # is needed; the earlier volume decides the default output family.
    earlier_radar, input_flavor = to_pyart_radar(earlier)
    later_radar, _ = to_pyart_radar(later)

    if field is None:
        field = "reflectivity"
    carried = list(carry_fields or ())
    for name, radar in (("earlier", earlier_radar), ("later", later_radar)):
        for required in (field, *carried):
            if required not in radar.fields:
                raise KeyError(
                    f"field {required!r} is not present in the {name} volume"
                )

    alpha = _resolve_alpha(earlier_radar, later_radar, alpha, target_time)

    grid_kwargs = {
        "grid_shape": grid_shape,
        "grid_limits": grid_limits,
        "fields": [field],
        "weighting_function": "Barnes2",
        "roi_func": "dist_beam",
        "min_radius": 1000.0,
    }
    grid_kwargs.update(gridding_kwargs or {})
    # Pass as a 1-tuple: grid_from_radars calls len() on this argument, which the
    # Xradar wrapper (returned for xradar input) does not support as a bare object.
    earlier_grid = pyart.map.grid_from_radars((earlier_radar,), **grid_kwargs)
    later_grid = pyart.map.grid_from_radars((later_radar,), **grid_kwargs)

    displacement_north, displacement_east = grid_optical_flow(
        earlier_grid, later_grid, field, **flow_kwargs
    )

    earlier_gridded = np.ma.filled(
        earlier_grid.fields[field]["data"].astype("float64"), np.nan
    )
    later_gridded = np.ma.filled(
        later_grid.fields[field]["data"].astype("float64"), np.nan
    )
    has_echo = (earlier_gridded > echo_threshold_dbz) | (
        later_gridded > echo_threshold_dbz
    )

    grid_axes = (
        earlier_grid.z["data"],
        earlier_grid.y["data"],
        earlier_grid.x["data"],
    )
    interpolate_north = RegularGridInterpolator(
        grid_axes,
        _extend_over_no_echo(displacement_north, has_echo),
        bounds_error=False,
        fill_value=None,
    )
    interpolate_east = RegularGridInterpolator(
        grid_axes,
        _extend_over_no_echo(displacement_east, has_echo),
        bounds_error=False,
        fill_value=None,
    )

    # One array per variable being warped. The motion is a property of the scene,
    # estimated once from the tracer above, so every variable moves through exactly
    # the same displacement, resampling and blend.
    warped_names = [field, *carried]
    reconstructed = {
        name: np.full(
            (earlier_radar.nrays, earlier_radar.ngates), np.nan, dtype="float64"
        )
        for name in warped_names
    }
    gate_east_all = earlier_radar.gate_x["data"]
    gate_north_all = earlier_radar.gate_y["data"]
    gate_up_all = earlier_radar.gate_z["data"]
    lowest_level_m, highest_level_m = grid_axes[0][0], grid_axes[0][-1]

    for sweep in range(earlier_radar.nsweeps):
        start, end = earlier_radar.get_start_end(sweep)
        gate_east = gate_east_all[start : end + 1]
        gate_north = gate_north_all[start : end + 1]
        gate_up = gate_up_all[start : end + 1]

        # Slant range is the horizontal distance divided by cos(elevation); guard
        # against a near-vertical sweep making that blow up.
        elevation_deg = earlier_radar.fixed_angle["data"][sweep]
        cos_elevation = max(np.cos(np.deg2rad(elevation_deg)), 1e-3)

        query = np.column_stack(
            [
                np.clip(gate_up, lowest_level_m, highest_level_m).ravel(),
                gate_north.ravel(),
                gate_east.ravel(),
            ]
        )
        gate_displacement_north = interpolate_north(query).reshape(gate_east.shape)
        gate_displacement_east = interpolate_east(query).reshape(gate_east.shape)

        # Look BACK along the motion for the earlier volume, FORWARD for the later.
        earlier_east = gate_east - alpha * gate_displacement_east
        earlier_north = gate_north - alpha * gate_displacement_north
        later_east = gate_east + (1.0 - alpha) * gate_displacement_east
        later_north = gate_north + (1.0 - alpha) * gate_displacement_north

        earlier_azimuths = np.rad2deg(np.arctan2(earlier_east, earlier_north)) % 360.0
        earlier_ranges = np.hypot(earlier_east, earlier_north) / cos_elevation
        later_azimuths = np.rad2deg(np.arctan2(later_east, later_north)) % 360.0
        later_ranges = np.hypot(later_east, later_north) / cos_elevation

        for name in warped_names:
            from_earlier = _sample_sweep(
                earlier_azimuths,
                earlier_ranges,
                _sweep_sampler(earlier_radar, sweep, name),
            )
            from_later = _sample_sweep(
                later_azimuths,
                later_ranges,
                _sweep_sampler(later_radar, sweep, name),
            )

            both_valid = np.isfinite(from_earlier) & np.isfinite(from_later)
            blended = (1.0 - alpha) * from_earlier + alpha * from_later
            # Where only one volume reached the gate, use it rather than discarding
            # the gate: a one-sided estimate beats a hole at the edge of coverage.
            reconstructed[name][start : end + 1] = np.where(
                both_valid,
                blended,
                np.where(np.isfinite(from_earlier), from_earlier, from_later),
            )

    output = copy.deepcopy(earlier_radar)
    output_fields = {}
    for name in warped_names:
        # Each variable keeps its OWN metadata: units, standard_name, valid range.
        # Copying the tracer's would mislabel a carried variable as dBZ.
        field_dict = dict(earlier_radar.fields[name])
        field_dict["data"] = np.ma.masked_invalid(reconstructed[name])
        # interp_field_name renames only the tracer; carried variables keep theirs,
        # since renaming several outputs from one string is not well defined.
        output_fields[interp_field_name or name if name == field else name] = field_dict
    output.fields = output_fields

    # The volume represents an instant that was never observed; give it that instant
    # rather than the clock inherited from the deepcopy above.
    output = apply_interpolated_time(output, earlier_radar, later_radar, alpha)

    return to_radar_flavor(output, resolve_output_flavor(output_flavor, input_flavor))
