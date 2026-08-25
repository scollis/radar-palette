"""Interpolation between tilt cones onto target heights, one column at a time.

The evaluator turns one sweep into a continuous surface. This module turns a stack of
those surfaces into a 3-D field --- and the interesting part is not the interpolation
but the **honesty about coverage**, because a radar volume does not observe a box.

Three geometric facts this is built around
------------------------------------------
1. A tilt is a curved cone, so the heights ``z_k(x, y)`` and the vertical gaps between
   adjacent tilts differ from column to column. Interpolation is therefore per column
   against locally evaluated heights, never against one nominal height per tilt.
2. Height and beam elevation are the same quantity expressed two ways: a target's own
   elevation is ``cartesian_to_antenna_43(x, y, z)``, and both it and ``z`` increase
   monotonically with tilt index. So interpolating linearly in height and linearly in
   elevation share the same bracketing pair and differ only in the weight --- which
   makes comparing them a clean one-variable change rather than two algorithms.
3. The valid tilt run in a column is **usually** contiguous, because at fixed arc
   length slant range grows with elevation, so the near-radar cut removes low tilts
   and the maximum-range cut removes high ones. **But not always, and assuming so is a
   bug.** Maximum valid range is a property of the *echo*, not the geometry, and it can
   be non-monotonic in elevation --- a higher tilt reaching further than the one below
   it. That leaves an annulus where the lower tilt is absent and the higher present.
   Those targets get :attr:`VerticalFlag.INTERIOR_GAP` rather than being silently
   interpolated across the hole or mislabelled as properly bracketed.

Extrapolation policy: none
--------------------------
A target below the lowest valid cone, above the highest (the cone of silence), in a
column with fewer than two valid tilts, or inside a hole in the tilt run is left NaN
and flagged. Every finite value carries :attr:`VerticalFlag.INTERPOLATED` and every
non-interpolated cell is NaN --- the two never disagree, which is asserted directly in
``tests/gridding/test_vertical.py``. A NaN labelled interpolated would be a silent
hole; a finite value labelled otherwise would be a silent extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from radar_palette.gridding.geometry import cartesian_to_antenna_43

__all__ = [
    "DEFAULT_VERTICAL_SCHEME",
    "VERTICAL_SCHEMES",
    "VerticalFlag",
    "VerticalProfile",
    "intercone_gap",
    "interp_column_stack",
    "target_elevation",
]


class VerticalFlag(IntEnum):
    """Why a target height did or did not receive a value.

    ``INTERPOLATED`` is 0 so that a boolean test on the flag array selects exactly
    the trustworthy cells, and it is the **only** value carrying a finite result.
    """

    INTERPOLATED = 0
    """Bracketed by two adjacent valid cones. The only trustworthy value."""

    BELOW_LOWEST = 1
    """Beneath the lowest valid cone in this column; would need extrapolation."""

    ABOVE_HIGHEST = 2
    """Above the highest cone reaching this column --- the cone of silence."""

    NO_COVERAGE = 3
    """Fewer than two valid tilts in this column."""

    INTERIOR_GAP = 4
    """Between the lowest and highest valid cone, but no *adjacent* valid pair
    brackets it: a hole in the tilt run. See the module docstring."""


#: Vertical interpolation schemes. ``linear_z`` and ``linear_elev`` differ only in the
#: weight, ``pchip_z`` is monotone (no new extrema between cones), and
#: ``nearest_tilt`` returns an observed cone value unblended.
VERTICAL_SCHEMES = ("linear_z", "linear_elev", "pchip_z", "nearest_tilt")

#: Default scheme. ``pchip_z`` was measured best of the four on a sharp bright band
#: and, unlike the linear schemes, cannot introduce an extremum between two cones.
#: See :func:`interp_column_stack` for the numbers and the far more important caveat
#: about what actually limits vertical accuracy.
DEFAULT_VERTICAL_SCHEME = "pchip_z"


def target_elevation(east_m, north_m, height_m):
    """Beam elevation of a Cartesian point, in degrees, on the 4/3 earth.

    Uses this package's exact inverse rather than Py-ART's, which omits the curvature
    term and is wrong by hundreds of millidegrees at long range --- an error comparable
    to the tilt spacing itself, which would put targets in the wrong layer.
    """
    east_m = np.asarray(east_m, dtype=np.float64)
    _, _, elevation_deg = cartesian_to_antenna_43(
        east_m,
        np.asarray(north_m, dtype=np.float64),
        np.full_like(east_m, float(height_m)),
    )
    return elevation_deg


@dataclass
class VerticalProfile:
    """Result of interpolating a cone stack onto target heights.

    ``fields`` maps scheme name to a ``(nz, ny, nx)`` array. ``flag`` is the
    :class:`VerticalFlag` for each cell, ``gap_m`` the thickness of the inter-cone
    layer the target sat inside (NaN where not interpolated), and ``elevation_deg``
    the target's own beam elevation.
    """

    fields: dict
    flag: np.ndarray
    gap_m: np.ndarray
    elevation_deg: np.ndarray
    height_m: np.ndarray


def _monotone_slopes(abscissa, values, usable):
    """Fritsch-Carlson slopes for a stack of columns, vectorised over columns.

    The harmonic mean of the two neighbouring secants where they agree in sign, and
    zero at a local extremum --- that zero is precisely what forbids overshoot, which
    is the property ``pchip_z`` exists to provide. Run ends take the one-sided secant,
    the conservative end condition that cannot exceed the local envelope.
    """
    n_tilts = abscissa.shape[0]
    interval = np.diff(abscissa, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        secant = np.diff(values, axis=0) / interval
    usable_interval = usable[:-1] & usable[1:] & np.isfinite(interval) & (interval > 0)
    secant = np.where(usable_interval, secant, np.nan)

    slopes = np.full_like(values, np.nan)
    for tilt in range(n_tilts):
        left = secant[tilt - 1] if tilt > 0 else None
        right = secant[tilt] if tilt < n_tilts - 1 else None
        if left is None and right is None:
            continue
        if left is None:
            slopes[tilt] = right
            continue
        if right is None:
            slopes[tilt] = left
            continue

        width_left = interval[tilt - 1]
        width_right = interval[tilt]
        both_finite = np.isfinite(left) & np.isfinite(right)
        same_sign = both_finite & (left * right > 0)
        weight_left = 2.0 * width_right + width_left
        weight_right = width_right + 2.0 * width_left
        with np.errstate(invalid="ignore", divide="ignore"):
            harmonic = (weight_left + weight_right) / (
                weight_left / np.where(left == 0, np.nan, left)
                + weight_right / np.where(right == 0, np.nan, right)
            )
        chosen = np.where(same_sign, harmonic, 0.0)
        slopes[tilt] = np.where(
            both_finite, chosen, np.where(np.isfinite(left), left, right)
        )
    return slopes


def _cubic_hermite(fraction, width, lower, upper, slope_lower, slope_upper):
    """Cubic Hermite on the unit interval, scaled by the interval width."""
    squared = fraction * fraction
    cubed = squared * fraction
    slope_lower = np.where(np.isfinite(slope_lower), slope_lower, 0.0)
    slope_upper = np.where(np.isfinite(slope_upper), slope_upper, 0.0)
    return (
        (2 * cubed - 3 * squared + 1) * lower
        + (cubed - 2 * squared + fraction) * width * slope_lower
        + (-2 * cubed + 3 * squared) * upper
        + (cubed - squared) * width * slope_upper
    )


def interp_column_stack(
    cones, east_m, north_m, height_levels_m, schemes=VERTICAL_SCHEMES
):
    """Interpolate a cone stack onto target heights, per column.

    Parameters
    ----------
    cones : ~radar_palette.gridding.cones.ConeStack
        Output of :func:`radar_palette.gridding.build_cones`.
    east_m, north_m : numpy.ndarray
        ``(ny, nx)`` Cartesian coordinates of the lattice, in metres.
    height_levels_m : array_like
        Target heights above the radar, in metres.
    schemes : tuple of str, optional
        Which schemes to evaluate; see :data:`VERTICAL_SCHEMES`.

    Returns
    -------
    VerticalProfile

    Raises
    ------
    ValueError
        If a requested scheme is not recognised.

    Notes
    -----
    **Tilt spacing, not scheme choice, is what limits vertical accuracy.** Measured on
    a synthetic bright band at 2500 m with a 400 m standard deviation --- a feature
    sharper than the sampling --- reconstructed onto five height levels:

    ==================  ==============  ==============
    Scheme              6 tilts         23 tilts
    ==================  ==============  ==============
    ``linear_z``        4.086 dB RMSE   0.295 dB RMSE
    ``linear_elev``     4.085 dB RMSE   ---
    ``pchip_z``         4.024 dB RMSE   0.202 dB RMSE
    ``nearest_tilt``    5.194 dB RMSE   ---
    ==================  ==============  ==============

    Going from 6 tilts to 23 (median inter-cone thickness 680 m to 213 m) improves the
    result by a factor of 14. Choosing the best scheme instead of the worst improves it
    by 22%. Within a scan strategy the error tracks the local layer thickness directly:
    median absolute error 0.42 dB where adjacent cones are 400--700 m apart, 3.81 dB
    where they are 1200--2500 m apart.

    So the useful lever is the scan strategy, and the honest reading of a gridded value
    is alongside its ``gap_m``. A value interpolated across a 2 km layer is a different
    kind of number from one interpolated across 400 m, whatever scheme produced it.

    ``pchip_z`` is the default because it measured best in both configurations and,
    being monotone, cannot invent an extremum between two cones --- but the margin over
    linear is small, and no scheme recovers structure the sampling did not capture.
    """
    unknown = sorted(set(schemes) - set(VERTICAL_SCHEMES))
    if unknown:
        raise ValueError(
            f"unknown vertical scheme(s) {unknown}; "
            f"choose from {list(VERTICAL_SCHEMES)}"
        )

    cone_height_m = cones.height_m.astype(np.float64)
    cone_values = cones.reflectivity.astype(np.float64)
    usable = cones.valid & np.isfinite(cone_height_m) & np.isfinite(cone_values)
    fixed_angles = cones.fixed_angle.astype(np.float64)

    n_tilts, n_north, n_east = cone_height_m.shape
    height_levels_m = np.asarray(height_levels_m, dtype=np.float64)
    n_levels = height_levels_m.size

    too_few_tilts = usable.sum(axis=0) < 2
    tilt_index = np.arange(n_tilts)[:, None, None]
    lowest = np.where(usable, tilt_index, n_tilts).min(axis=0)
    highest = np.where(usable, tilt_index, -1).max(axis=0)
    lowest_height = np.take_along_axis(
        cone_height_m, np.clip(lowest, 0, n_tilts - 1)[None], 0
    )[0]
    highest_height = np.take_along_axis(
        cone_height_m, np.clip(highest, 0, n_tilts - 1)[None], 0
    )[0]

    slopes_height = (
        _monotone_slopes(cone_height_m, cone_values, usable)
        if "pchip_z" in schemes
        else None
    )

    fields = {
        name: np.full((n_levels, n_north, n_east), np.nan, dtype=np.float32)
        for name in schemes
    }
    flag = np.full(
        (n_levels, n_north, n_east), VerticalFlag.NO_COVERAGE, dtype=np.uint8
    )
    gap_m = np.full((n_levels, n_north, n_east), np.nan, dtype=np.float32)
    elevation_deg = np.full((n_levels, n_north, n_east), np.nan, dtype=np.float32)

    for level, target_height_m in enumerate(height_levels_m):
        this_elevation = target_elevation(east_m, north_m, target_height_m)
        elevation_deg[level] = this_elevation

        below = (~too_few_tilts) & (target_height_m < lowest_height)
        above = (~too_few_tilts) & (target_height_m > highest_height)
        flag[level] = np.where(
            too_few_tilts,
            VerticalFlag.NO_COVERAGE,
            np.where(
                below,
                VerticalFlag.BELOW_LOWEST,
                np.where(above, VerticalFlag.ABOVE_HIGHEST, VerticalFlag.INTERPOLATED),
            ),
        )
        inside = flag[level] == VerticalFlag.INTERPOLATED
        if not inside.any():
            continue

        for lower_tilt in range(n_tilts - 1):
            upper_tilt = lower_tilt + 1
            pair_usable = usable[lower_tilt] & usable[upper_tilt] & inside
            if not pair_usable.any():
                continue

            lower_height = cone_height_m[lower_tilt]
            upper_height = cone_height_m[upper_tilt]
            bracketed = (
                pair_usable
                & (target_height_m >= lower_height)
                & (target_height_m <= upper_height)
            )
            if not bracketed.any():
                continue

            thickness = upper_height - lower_height
            gap_m[level][bracketed] = thickness[bracketed]
            lower_value = cone_values[lower_tilt]
            upper_value = cone_values[upper_tilt]

            fraction_height = np.zeros_like(thickness)
            np.divide(
                target_height_m - lower_height,
                thickness,
                out=fraction_height,
                where=thickness > 0,
            )
            fraction_height = np.clip(fraction_height, 0.0, 1.0)

            lower_angle, upper_angle = (
                fixed_angles[lower_tilt],
                fixed_angles[upper_tilt],
            )
            if upper_angle > lower_angle:
                fraction_elevation = np.clip(
                    (this_elevation - lower_angle) / (upper_angle - lower_angle),
                    0.0,
                    1.0,
                )
            else:  # pragma: no cover - dedup_sweeps guarantees strict ordering
                fraction_elevation = np.zeros_like(thickness)

            if "linear_z" in schemes:
                fields["linear_z"][level][bracketed] = (
                    lower_value + fraction_height * (upper_value - lower_value)
                )[bracketed]
            if "linear_elev" in schemes:
                fields["linear_elev"][level][bracketed] = (
                    lower_value + fraction_elevation * (upper_value - lower_value)
                )[bracketed]
            if "nearest_tilt" in schemes:
                fields["nearest_tilt"][level][bracketed] = np.where(
                    fraction_height < 0.5, lower_value, upper_value
                )[bracketed]
            if "pchip_z" in schemes:
                fields["pchip_z"][level][bracketed] = _cubic_hermite(
                    fraction_height,
                    thickness,
                    lower_value,
                    upper_value,
                    slopes_height[lower_tilt],
                    slopes_height[upper_tilt],
                )[bracketed]

        # Anything still marked INTERPOLATED that no adjacent valid pair actually
        # bracketed sits in a hole in the tilt run. Relabel it rather than leave a
        # NaN wearing an "interpolated" flag.
        unbracketed = inside & ~np.isfinite(gap_m[level])
        if unbracketed.any():
            flag[level][unbracketed] = VerticalFlag.INTERIOR_GAP

    return VerticalProfile(
        fields=fields,
        flag=flag,
        gap_m=gap_m,
        elevation_deg=elevation_deg,
        height_m=height_levels_m,
    )


def intercone_gap(cones):
    """Vertical thickness between adjacent tilt surfaces, and its midpoint height.

    The fundamental limit on vertical resolution: whatever the interpolation scheme,
    a target between two cones is constrained only by those two samples, and the
    thickness grows with range as the cones diverge.

    Returns
    -------
    thickness_m, midpoint_height_m : numpy.ndarray
        Both ``(n_tilts - 1, ny, nx)``, NaN where either bounding cone is invalid.
    """
    cone_height_m = cones.height_m.astype(np.float64)
    usable = cones.valid & np.isfinite(cone_height_m)
    pair_usable = usable[:-1] & usable[1:]
    thickness_m = np.where(pair_usable, np.diff(cone_height_m, axis=0), np.nan)
    midpoint_height_m = np.where(
        pair_usable, 0.5 * (cone_height_m[:-1] + cone_height_m[1:]), np.nan
    )
    return thickness_m, midpoint_height_m
