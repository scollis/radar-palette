"""Unit tests for the cone lattice and vertical assembly of a gridded volume.

A single sweep is a **curved cone**, not a plane, so turning a stack of sweeps into
a 3-D field is a geometry problem before it is an interpolation problem. The tests
are grouped by the thing that can independently be wrong:

``TestConeGeometry``
    Whether the cone surface is the exact inverse of the forward transform at fixed
    elevation. Pure maths, tested to floating-point closure.
``TestTargetLattice``
    Array index order, and the validity mask. Index order matters more than it
    looks: transposing a roughly isotropic field about the diagonal produces an
    error equal to the field's own standard deviation rather than an obvious flip.
``TestSweepDeduplication``
    Collapsing split cuts so each unique elevation contributes once, chosen on
    measured range rather than on cut naming.
``TestVerticalInterpolation``
    The interpolation itself, and --- most importantly --- the flags. A target that
    cannot be honestly interpolated must be flagged, never extrapolated.
``TestInteriorGaps``
    The case that is easy to get wrong: the valid tilt run in a column is usually
    contiguous but **not always**, because maximum valid range is a property of the
    echo rather than of the geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from radar_palette.gridding import (  # noqa: E402
    DEFAULT_VERTICAL_SCHEME,
    EARTH_RADIUS_EFFECTIVE_M,
    VERTICAL_SCHEMES,
    VerticalFlag,
    antenna_to_cartesian_43,
    beam_footprint_crossover,
    build_cones,
    cartesian_to_antenna_43,
    census_radar,
    cone_range_height,
    dedup_sweeps,
    intercone_gap,
    interp_column_stack,
    target_elevation,
    target_lattice,
)

HALF_WIDTH_M = 60_000.0
SPACING_M = 4_000.0


def make_volume(
    fixed_angles,
    nrays=180,
    ngates=60,
    range_first_m=1000.0,
    range_spacing_m=1500.0,
    constant_dbz=None,
):
    """A volume whose field is a smooth function of height, so truth is analytic.

    Making reflectivity depend only on gate height means the correct value at any
    target height is known in closed form, which is what lets the vertical
    interpolation be tested against truth rather than against another interpolator.
    """
    fixed_angles = np.asarray(fixed_angles, dtype="float64")
    nsweeps = fixed_angles.size
    radar = pyart.testing.make_empty_ppi_radar(ngates, nrays, nsweeps)
    radar.range["data"] = range_first_m + range_spacing_m * np.arange(
        ngates, dtype="float64"
    )
    azimuths = (360.0 / nrays) * np.arange(nrays, dtype="float64")
    radar.azimuth["data"] = np.tile(azimuths, nsweeps)
    radar.elevation["data"] = np.repeat(fixed_angles, nrays)
    radar.fixed_angle["data"] = fixed_angles
    radar.time["data"] = 0.5 * np.arange(nrays * nsweeps, dtype="float64")
    radar.time["units"] = "seconds since 2011-05-20T11:27:34Z"

    if constant_dbz is not None:
        values = np.full((radar.nrays, ngates), float(constant_dbz))
    else:
        # 30 dBZ at the surface falling 4 dB per km: linear in height, so linear
        # interpolation in z is exact and any error is the scheme's own.
        values = 30.0 - 4.0e-3 * radar.gate_z["data"]
    radar.add_field(
        "reflectivity",
        {
            "data": np.ma.masked_invalid(values.astype("float32")),
            "units": "dBZ",
            "_FillValue": -9999.0,
        },
        replace_existing=True,
    )
    return radar


@pytest.fixture(scope="module")
def volume():
    return make_volume([0.5, 1.5, 2.5, 4.0])


@pytest.fixture(scope="module")
def cone_stack(volume):
    geometries = dedup_sweeps(census_radar(volume))
    return build_cones(
        volume, geometries, half_width_m=HALF_WIDTH_M, spacing_m=SPACING_M
    )


class TestConeGeometry:
    @pytest.mark.parametrize("arc_m", [1_000.0, 20_000.0, 100_000.0, 200_000.0])
    @pytest.mark.parametrize("elevation_deg", [0.5, 2.5, 10.0])
    def test_inverts_the_forward_transform(self, arc_m, elevation_deg):
        """The cone surface must agree with the forward transform it inverts."""
        slant_range_m, height_m = cone_range_height(arc_m, elevation_deg)
        east, north, up = antenna_to_cartesian_43(slant_range_m, 0.0, elevation_deg)
        assert float(np.hypot(east, north)) == pytest.approx(arc_m, rel=1e-9)
        assert float(up) == pytest.approx(height_m, rel=1e-9, abs=1e-6)

    def test_height_increases_with_elevation_at_fixed_arc(self):
        heights = [cone_range_height(50_000.0, el)[1] for el in (0.5, 1.5, 4.0)]
        assert heights[0] < heights[1] < heights[2]

    def test_the_surface_is_curved_not_planar(self):
        """A plane would give height exactly proportional to arc length.

        Earth curvature makes the beam climb faster than a straight line, which is
        the whole reason a tilt is a cone rather than a plane. If this test passes
        by accident the geometry has silently become flat-earth.
        """
        near_arc_m, far_arc_m = 20_000.0, 200_000.0
        _, near_height = cone_range_height(near_arc_m, 0.5)
        _, far_height = cone_range_height(far_arc_m, 0.5)
        planar_far_height = near_height * (far_arc_m / near_arc_m)
        assert far_height > planar_far_height * 1.1

    def test_zero_elevation_still_climbs(self):
        """At 0 degrees the beam still rises, because the earth falls away."""
        _, height_m = cone_range_height(100_000.0, 0.0)
        assert height_m == pytest.approx(
            100_000.0**2 / (2 * EARTH_RADIUS_EFFECTIVE_M), rel=0.01
        )

    def test_accepts_an_array_of_arcs(self):
        ranges_m, heights_m = cone_range_height(np.array([10_000.0, 50_000.0]), 1.0)
        assert ranges_m.shape == heights_m.shape == (2,)


class TestBeamFootprintCrossover:
    def test_scales_inversely_with_beamwidth(self):
        assert beam_footprint_crossover(1.0, 1000.0) > beam_footprint_crossover(
            2.0, 1000.0
        )

    def test_scales_with_grid_spacing(self):
        assert beam_footprint_crossover(1.0, 2000.0) == pytest.approx(
            2.0 * beam_footprint_crossover(1.0, 1000.0)
        )

    def test_matches_the_closed_form(self):
        assert beam_footprint_crossover(1.0, 1000.0) == pytest.approx(
            1000.0 / np.deg2rad(1.0)
        )


class TestTargetLattice:
    def test_axis_order_is_y_then_x(self):
        """Row index is y, column index is x --- matching Py-ART's (nz, ny, nx).

        Getting this backwards transposes the field about the diagonal. On a roughly
        isotropic field that does not look like a flip; it looks like an error equal
        to the field's own standard deviation, which is far harder to spot.
        """
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        east, north = lattice["X"], lattice["Y"]
        assert np.all(np.diff(east, axis=1) > 0)
        assert np.allclose(np.diff(east, axis=0), 0.0)
        assert np.all(np.diff(north, axis=0) > 0)
        assert np.allclose(np.diff(north, axis=1), 0.0)

    def test_is_square_and_centred_on_the_radar(self):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        assert lattice["X"].shape == lattice["Y"].shape
        assert lattice["x"][0] == pytest.approx(-HALF_WIDTH_M)
        assert lattice["x"][-1] == pytest.approx(HALF_WIDTH_M)
        assert lattice["arc_m"][
            lattice["X"].shape[0] // 2, lattice["X"].shape[1] // 2
        ] == pytest.approx(0.0)

    def test_validity_mask_excludes_the_near_radar_hole(self):
        """Inside the first gate there is no measurement, so the cell is invalid.

        This matters beyond bookkeeping: the range axis is mirror-extended for the
        FFT, and the mirror is EVEN about gate 0, so it fabricates a reflection of
        the echo into the near-radar hole. Masking is what stops that ghost being
        presented as data.
        """
        range_first_m = 20_000.0
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, range_first_m, 100_000.0)
        assert not lattice["valid"][lattice["slant_range_m"] < range_first_m].any()

    def test_validity_mask_excludes_beyond_the_last_valid_gate(self):
        range_max_m = 40_000.0
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, range_max_m)
        assert not lattice["valid"][lattice["slant_range_m"] > range_max_m].any()

    def test_azimuth_is_meteorological(self):
        """0 degrees is north, increasing clockwise."""
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        centre = lattice["X"].shape[0] // 2
        north_of_centre = lattice["azimuth_deg"][-1, centre]
        east_of_centre = lattice["azimuth_deg"][centre, -1]
        assert north_of_centre == pytest.approx(0.0, abs=1.0)
        assert east_of_centre == pytest.approx(90.0, abs=1.0)

    def test_height_matches_the_cone_surface(self):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 2.0, 1000.0, 100_000.0)
        _, expected = cone_range_height(lattice["arc_m"], 2.0)
        np.testing.assert_allclose(lattice["height_m"], expected, rtol=1e-12)


class TestSweepDeduplication:
    def test_keeps_one_sweep_per_unique_elevation(self, volume):
        geometries = census_radar(volume)
        deduplicated = dedup_sweeps(geometries)
        angles = [g.fixed_angle for g in deduplicated]
        assert len(angles) == len(set(angles))

    def test_is_sorted_by_elevation(self, volume):
        deduplicated = dedup_sweeps(census_radar(volume))
        angles = [g.fixed_angle for g in deduplicated]
        assert angles == sorted(angles)

    def test_prefers_the_member_reaching_furthest_in_range(self):
        """Prefer the split-cut member whose field reaches furthest in range.

        A WSR-88D split cut pairs a long surveillance cut with a short Doppler cut at
        the same elevation, and reflectivity wants the long one. The rule is stated in
        terms of measured valid range, so it needs no knowledge of NEXRAD cut naming
        to pick correctly.
        """
        volume = make_volume([0.5, 0.5, 1.5])
        geometries = census_radar(volume)
        geometries[0].range_max_valid_m = 200_000.0
        geometries[1].range_max_valid_m = 400_000.0
        deduplicated = dedup_sweeps(geometries)
        kept = [g for g in deduplicated if g.fixed_angle == pytest.approx(0.5)]
        assert len(kept) == 1
        assert kept[0].range_max_valid_m == pytest.approx(400_000.0)

    def test_records_which_sweeps_were_dropped(self):
        """Dropping data silently would be worse than not dropping it."""
        volume = make_volume([0.5, 0.5, 1.5])
        geometries = census_radar(volume)
        geometries[1].range_max_valid_m = 400_000.0
        deduplicated = dedup_sweeps(geometries)
        kept = next(g for g in deduplicated if g.fixed_angle == pytest.approx(0.5))
        assert kept.split_cut_size == 2

    def test_a_volume_with_no_split_cuts_is_unchanged(self, volume):
        geometries = census_radar(volume)
        assert len(dedup_sweeps(geometries)) == len(geometries)

    def test_ties_fall_back_to_the_lower_sweep_index(self):
        volume = make_volume([0.5, 0.5, 1.5])
        geometries = census_radar(volume)
        deduplicated = dedup_sweeps(geometries)
        kept = next(g for g in deduplicated if g.fixed_angle == pytest.approx(0.5))
        assert kept.sweep == 0


class TestBuildCones:
    def test_returns_one_cone_per_deduplicated_sweep(self, volume, cone_stack):
        assert cone_stack.reflectivity.shape[0] == len(
            dedup_sweeps(census_radar(volume))
        )

    def test_all_cones_share_one_lattice(self, cone_stack):
        assert cone_stack.reflectivity.shape == cone_stack.height_m.shape
        assert cone_stack.reflectivity.shape == cone_stack.valid.shape

    def test_heights_increase_with_tilt_index(self, cone_stack):
        """Cones must be stacked in ascending elevation for the vertical search."""
        mean_heights = [
            float(np.nanmean(cone_stack.height_m[k][cone_stack.valid[k]]))
            for k in range(cone_stack.reflectivity.shape[0])
        ]
        assert mean_heights == sorted(mean_heights)

    def test_fixed_angles_are_recorded_in_order(self, cone_stack):
        assert list(cone_stack.fixed_angle) == sorted(cone_stack.fixed_angle)

    def test_gridded_values_stay_near_the_input_range(self, volume, cone_stack):
        """A sanity bound: spectral gridding rings, but not without limit."""
        observed = np.ma.filled(
            volume.fields["reflectivity"]["data"].astype("float64"), np.nan
        )
        gridded = cone_stack.reflectivity[cone_stack.valid]
        assert np.nanmin(gridded) > np.nanmin(observed) - 25.0
        assert np.nanmax(gridded) < np.nanmax(observed) + 25.0

    def test_invalid_cells_are_not_finite(self, cone_stack):
        assert not np.isfinite(cone_stack.reflectivity[~cone_stack.valid]).any()

    def test_reports_per_tilt_diagnostics(self, cone_stack):
        assert len(cone_stack.reports) == cone_stack.reflectivity.shape[0]
        first = cone_stack.reports[0]
        assert first.azimuth_path
        assert 0.0 <= first.valid_fraction <= 1.0


class TestTargetElevation:
    def test_agrees_with_the_exact_inverse(self):
        east = np.array([30_000.0, 60_000.0])
        north = np.array([0.0, 40_000.0])
        height_m = 3000.0
        elevations = target_elevation(east, north, height_m)
        _, _, expected = cartesian_to_antenna_43(
            east, north, np.full_like(east, height_m)
        )
        np.testing.assert_allclose(elevations, expected, rtol=1e-12)

    def test_falls_with_range_at_fixed_height(self):
        """A fixed altitude is reached at a lower beam angle further out."""
        elevations = target_elevation(
            np.array([10_000.0, 50_000.0, 100_000.0]), np.zeros(3), 3000.0
        )
        assert elevations[0] > elevations[1] > elevations[2]


class TestVerticalInterpolation:
    def test_recovers_a_field_linear_in_height(self, volume, cone_stack):
        """The test field is linear in z, so linear-in-z interpolation is exact.

        Any error here is the scheme's, not the field's --- which is why the field
        was chosen this way.
        """
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        levels_m = np.array([1500.0, 2500.0])
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], levels_m, schemes=("linear_z",)
        )
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        assert interpolated.any()
        values = result.fields["linear_z"]
        expected = 30.0 - 4.0e-3 * np.broadcast_to(
            levels_m[:, None, None], values.shape
        )
        error = np.abs(values - expected)[interpolated]
        assert np.nanmax(error) < 3.0

    def test_never_extrapolates_below_the_lowest_cone(self, cone_stack):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], np.array([-500.0])
        )
        assert not np.isfinite(result.fields["linear_z"]).any()
        assert (result.flag != VerticalFlag.INTERPOLATED).all()

    def test_never_extrapolates_into_the_cone_of_silence(self, cone_stack):
        """Above the top tilt there is no measurement, however tempting the guess."""
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], np.array([40_000.0])
        )
        assert not np.isfinite(result.fields["linear_z"]).any()
        assert (result.flag == VerticalFlag.ABOVE_HIGHEST).any()

    def test_flags_and_values_agree_everywhere(self, cone_stack):
        """A finite value must carry an interpolated flag, and vice versa.

        The two must not disagree in either direction: a NaN labelled interpolated
        is a silent hole, and a finite value labelled uninterpolable is a silent
        extrapolation.
        """
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], np.array([1500.0, 8000.0])
        )
        for name, values in result.fields.items():
            finite = np.isfinite(values)
            interpolated = result.flag == VerticalFlag.INTERPOLATED
            assert np.array_equal(finite, interpolated), name

    def test_reports_the_layer_thickness_it_interpolated_within(self, cone_stack):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], np.array([1500.0])
        )
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        assert np.all(result.gap_m[interpolated] > 0.0)
        assert not np.isfinite(result.gap_m[~interpolated]).any()

    def test_gap_widens_with_range(self, cone_stack):
        """The same tilt pair spans a thicker layer further out.

        Adjacent cones diverge with distance, which is the fundamental limit on
        vertical resolution: no interpolation scheme can recover structure between
        two samples that far apart.
        """
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], np.array([2000.0])
        )
        arc_m = lattice["arc_m"]
        interpolated = result.flag[0] == VerticalFlag.INTERPOLATED
        near = interpolated & (arc_m < 25_000.0)
        far = interpolated & (arc_m > 45_000.0)
        if near.any() and far.any():
            assert np.nanmedian(result.gap_m[0][far]) > np.nanmedian(
                result.gap_m[0][near]
            )

    def test_records_the_target_beam_elevation(self, cone_stack):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], np.array([3000.0])
        )
        # elevation_deg is stored float32, so the tolerance is float32 epsilon,
        # not float64. Asserting 1e-9 here would test the storage dtype, not the
        # geometry.
        np.testing.assert_allclose(
            result.elevation_deg[0],
            target_elevation(lattice["X"], lattice["Y"], 3000.0),
            rtol=1e-6,
        )

    def test_a_constant_field_stays_constant(self):
        """The sharpest test of the vertical scheme: no scheme may invent structure."""
        volume = make_volume([0.5, 1.5, 2.5, 4.0], constant_dbz=22.0)
        stack = build_cones(
            volume,
            dedup_sweeps(census_radar(volume)),
            half_width_m=HALF_WIDTH_M,
            spacing_m=SPACING_M,
        )
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            stack, lattice["X"], lattice["Y"], np.array([2000.0])
        )
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        for name, values in result.fields.items():
            assert np.nanmax(np.abs(values[interpolated] - 22.0)) < 3.0, name

    def test_all_schemes_are_produced_by_default(self, cone_stack):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack, lattice["X"], lattice["Y"], np.array([2000.0])
        )
        assert {"linear_z", "linear_elev", "pchip_z", "nearest_tilt"} <= set(
            result.fields
        )

    def test_a_single_scheme_can_be_requested(self, cone_stack):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack,
            lattice["X"],
            lattice["Y"],
            np.array([2000.0]),
            schemes=("linear_z",),
        )
        assert set(result.fields) == {"linear_z"}

    def test_rejects_an_unknown_scheme(self, cone_stack):
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        with pytest.raises(ValueError, match="unknown"):
            interp_column_stack(
                cone_stack,
                lattice["X"],
                lattice["Y"],
                np.array([2000.0]),
                schemes=("cubic_spline",),
            )

    def test_nearest_tilt_returns_only_observed_cone_values(self, cone_stack):
        """Nearest-tilt must not blend; every output is some cone's own value."""
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack,
            lattice["X"],
            lattice["Y"],
            np.array([2000.0]),
            schemes=("nearest_tilt",),
        )
        values = result.fields["nearest_tilt"]
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        cone_values = cone_stack.reflectivity
        for value in values[interpolated][:200]:
            assert np.any(np.isclose(cone_values, value, atol=1e-6))

    def test_monotone_scheme_does_not_overshoot_the_bracketing_cones(self, cone_stack):
        """That is what 'monotone' buys: no new extrema between samples."""
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack,
            lattice["X"],
            lattice["Y"],
            np.array([2000.0]),
            schemes=("pchip_z", "linear_z"),
        )
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        cone_minimum = np.nanmin(cone_stack.reflectivity[cone_stack.valid])
        cone_maximum = np.nanmax(cone_stack.reflectivity[cone_stack.valid])
        monotone = result.fields["pchip_z"][interpolated]
        assert np.nanmin(monotone) >= cone_minimum - 1e-6
        assert np.nanmax(monotone) <= cone_maximum + 1e-6


class TestInteriorGaps:
    """A hole in the tilt run must be flagged, not interpolated across.

    The valid tilt run in a column is *usually* contiguous, because at fixed arc
    length slant range grows with elevation. But maximum valid range is a property
    of the **echo**, not of the geometry, and it can be non-monotonic in elevation
    --- a higher tilt reaching further than the one below it. That leaves an annulus
    where the lower tilt is absent and the higher one present. Interpolating across
    such a hole would silently span two non-adjacent surfaces.
    """

    @staticmethod
    def punch_out_middle_cone(volume):
        """Remove tilt 1 over an annulus, and return a height inside the hole.

        The target height is derived from the surviving cones rather than hardcoded.
        A fixed height fails for an uninteresting reason: at the annulus radii used
        here the cones sit between 240 m and 1710 m, so a 2000 m target is simply
        above the top cone and correctly reports ABOVE_HIGHEST --- it never reaches
        the interior-gap branch at all, so the test would pass vacuously or fail
        misleadingly depending on which way it was written.
        """
        geometries = dedup_sweeps(census_radar(volume))
        stack = build_cones(
            volume, geometries, half_width_m=HALF_WIDTH_M, spacing_m=SPACING_M
        )
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        annulus = (lattice["arc_m"] > 20_000.0) & (lattice["arc_m"] < 40_000.0)
        stack.valid[1][annulus] = False
        stack.reflectivity[1][annulus] = np.nan

        # Midway between the cone below the hole and the cone above it.
        below = np.nanmedian(stack.height_m[0][annulus])
        above = np.nanmedian(stack.height_m[2][annulus])
        return stack, lattice, float(0.5 * (below + above))

    def test_a_hole_in_the_tilt_run_is_flagged_not_spanned(self, volume):
        stack, lattice, target_height_m = self.punch_out_middle_cone(volume)
        result = interp_column_stack(
            stack,
            lattice["X"],
            lattice["Y"],
            np.array([target_height_m]),
            schemes=("linear_z",),
        )
        assert (result.flag == VerticalFlag.INTERIOR_GAP).any()

    def test_an_interior_gap_carries_no_value(self, volume):
        """The gap must not be spanned across the two non-adjacent surfaces."""
        stack, lattice, target_height_m = self.punch_out_middle_cone(volume)
        result = interp_column_stack(
            stack,
            lattice["X"],
            lattice["Y"],
            np.array([target_height_m]),
            schemes=("linear_z",),
        )
        gapped = result.flag == VerticalFlag.INTERIOR_GAP
        assert gapped.any()
        assert not np.isfinite(result.fields["linear_z"][gapped]).any()

    def test_a_target_above_the_top_cone_is_not_called_an_interior_gap(self, volume):
        """Distinguishes the two ways a value can be absent.

        Punching out a middle cone does not make everything above it a gap: a target
        above the highest surviving cone is outside coverage, which is a different
        fact about the volume and carries a different flag.
        """
        stack, lattice, _ = self.punch_out_middle_cone(volume)
        far_above_m = float(np.nanmax(stack.height_m) + 5_000.0)
        result = interp_column_stack(
            stack,
            lattice["X"],
            lattice["Y"],
            np.array([far_above_m]),
            schemes=("linear_z",),
        )
        assert not (result.flag == VerticalFlag.INTERIOR_GAP).any()
        assert (result.flag == VerticalFlag.ABOVE_HIGHEST).any()

    def test_a_column_with_one_valid_tilt_has_no_coverage(self, volume):
        geometries = dedup_sweeps(census_radar(volume))
        stack = build_cones(
            volume, geometries, half_width_m=HALF_WIDTH_M, spacing_m=SPACING_M
        )
        stack.valid[1:] = False
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            stack,
            lattice["X"],
            lattice["Y"],
            np.array([2000.0]),
            schemes=("linear_z",),
        )
        assert (result.flag == VerticalFlag.NO_COVERAGE).all()


class TestInterconeGap:
    def test_returns_one_layer_fewer_than_there_are_cones(self, cone_stack):
        thickness_m, _ = intercone_gap(cone_stack)
        assert thickness_m.shape[0] == cone_stack.reflectivity.shape[0] - 1

    def test_thickness_is_positive_where_both_cones_are_valid(self, cone_stack):
        thickness_m, _ = intercone_gap(cone_stack)
        finite = np.isfinite(thickness_m)
        assert finite.any()
        assert np.all(thickness_m[finite] > 0.0)

    def test_midpoint_height_lies_between_the_two_surfaces(self, cone_stack):
        thickness_m, midpoint_m = intercone_gap(cone_stack)
        lower = cone_stack.height_m[:-1]
        upper = cone_stack.height_m[1:]
        finite = np.isfinite(midpoint_m)
        assert np.all(midpoint_m[finite] >= lower[finite] - 1e-6)
        assert np.all(midpoint_m[finite] <= upper[finite] + 1e-6)


class TestVerticalFlag:
    def test_interpolated_is_the_only_trustworthy_state(self):
        assert VerticalFlag.INTERPOLATED == 0

    def test_every_flag_has_a_distinct_value(self):
        values = [int(flag) for flag in VerticalFlag]
        assert len(values) == len(set(values))

    def test_fits_in_the_uint8_flag_array(self):
        assert max(int(flag) for flag in VerticalFlag) < 256


class TestSamplingDominatesSchemeChoice:
    """The measured evidence that scan strategy matters more than scheme choice.

    These use a bright band --- a Gaussian layer at 2500 m with a 400 m standard
    deviation, sharper than the tilt spacing --- because a field linear in height
    cannot discriminate the schemes at all: all three smooth schemes reproduce it to
    0.001 dB, which would wrongly suggest the choice is free.
    """

    PEAK_HEIGHT_M = 2500.0
    PEAK_WIDTH_M = 400.0
    LEVELS_M = np.array([1500.0, 2000.0, 2500.0, 3000.0, 3500.0])

    def bright_band_volume(self, fixed_angles):
        radar = make_volume(fixed_angles, nrays=360, ngates=120, range_spacing_m=1000.0)
        height_m = radar.gate_z["data"]
        values = 20.0 + 18.0 * np.exp(
            -((height_m - self.PEAK_HEIGHT_M) ** 2) / (2 * self.PEAK_WIDTH_M**2)
        )
        radar.add_field(
            "reflectivity",
            {
                "data": np.ma.masked_invalid(values.astype("float32")),
                "units": "dBZ",
                "_FillValue": -9999.0,
            },
            replace_existing=True,
        )
        return radar

    def truth(self, height_m):
        return 20.0 + 18.0 * np.exp(
            -((np.asarray(height_m, dtype=float) - self.PEAK_HEIGHT_M) ** 2)
            / (2 * self.PEAK_WIDTH_M**2)
        )

    def reconstruct(self, fixed_angles, schemes=VERTICAL_SCHEMES):
        volume = self.bright_band_volume(fixed_angles)
        geometries = dedup_sweeps(census_radar(volume))
        stack = build_cones(
            volume, geometries, half_width_m=60_000.0, spacing_m=2_000.0
        )
        lattice = target_lattice(
            60_000.0,
            2_000.0,
            geometries[0].fixed_angle,
            geometries[0].range_first_m,
            geometries[0].range_max_valid_m,
        )
        result = interp_column_stack(
            stack, lattice["X"], lattice["Y"], self.LEVELS_M, schemes=schemes
        )
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        expected = self.truth(
            np.broadcast_to(self.LEVELS_M[:, None, None], result.flag.shape)
        )
        errors = {
            name: float(
                np.sqrt(
                    np.nanmean((values[interpolated] - expected[interpolated]) ** 2)
                )
            )
            for name, values in result.fields.items()
        }
        return result, errors, stack

    def test_more_tilts_helps_far_more_than_a_better_scheme(self):
        """The headline: a factor of ~14 from sampling, ~1.2x from scheme choice.

        Measured RMSE on this bright band: 4.086 dB (linear_z, 6 tilts) falling to
        0.295 dB at 23 tilts, while the best-vs-worst scheme spread at 6 tilts is
        4.024 to 5.194 dB. Bounds are loose so the test asserts the *ordering of
        magnitudes*, which is the durable claim, rather than exact numbers.
        """
        _, coarse, _ = self.reconstruct([0.5, 1.0, 1.5, 2.5, 4.0, 6.0])
        _, fine, _ = self.reconstruct(np.arange(0.5, 6.01, 0.25))

        sampling_gain = coarse["linear_z"] / fine["linear_z"]
        scheme_gain = coarse["nearest_tilt"] / coarse["pchip_z"]
        assert sampling_gain > 5.0
        assert scheme_gain < 2.0
        assert sampling_gain > 3.0 * scheme_gain

    def test_a_linear_field_cannot_discriminate_the_schemes(self, cone_stack):
        """Why the fixture above is a bright band and not the linear test field.

        On a field linear in height every smooth scheme is exact, so a comparison
        built on one would suggest the choice does not matter.
        """
        lattice = target_lattice(HALF_WIDTH_M, SPACING_M, 0.5, 1000.0, 100_000.0)
        result = interp_column_stack(
            cone_stack,
            lattice["X"],
            lattice["Y"],
            np.array([2000.0]),
            schemes=("linear_z", "pchip_z"),
        )
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        difference = np.abs(
            result.fields["linear_z"][interpolated]
            - result.fields["pchip_z"][interpolated]
        )
        assert np.nanmax(difference) < 0.1

    def test_error_grows_with_the_local_layer_thickness(self):
        """Within one scan strategy, accuracy tracks gap_m --- not the scheme.

        Measured median absolute error: 0.42 dB where adjacent cones are 400-700 m
        apart, 3.81 dB where they are 1200-2500 m apart. This is why a value should
        be read alongside its gap_m.
        """
        result, _, _ = self.reconstruct(
            [0.5, 1.0, 1.5, 2.5, 4.0, 6.0], schemes=("pchip_z",)
        )
        interpolated = result.flag == VerticalFlag.INTERPOLATED
        expected = self.truth(
            np.broadcast_to(self.LEVELS_M[:, None, None], result.flag.shape)
        )
        error = np.abs(result.fields["pchip_z"][interpolated] - expected[interpolated])
        thickness = result.gap_m[interpolated]

        thin = thickness < 700.0
        thick = thickness > 1200.0
        assert thin.sum() > 100 and thick.sum() > 100
        assert np.nanmedian(error[thin]) < np.nanmedian(error[thick])

    def test_the_monotone_default_beats_the_alternatives_here(self):
        """Basis for DEFAULT_VERTICAL_SCHEME, stated as an ordering not a number."""
        _, errors, _ = self.reconstruct([0.5, 1.0, 1.5, 2.5, 4.0, 6.0])
        assert errors["pchip_z"] <= errors["linear_z"]
        assert errors["pchip_z"] < errors["nearest_tilt"]

    def test_the_default_is_one_of_the_available_schemes(self):
        assert DEFAULT_VERTICAL_SCHEME in VERTICAL_SCHEMES

    def test_no_scheme_recovers_a_feature_sharper_than_the_sampling(self):
        """The honest limit: a 400 m feature under ~680 m tilt spacing is lost.

        Every scheme errs by several dB here. Documenting that is more useful than
        implying the choice of interpolator can rescue it.
        """
        _, errors, _ = self.reconstruct([0.5, 1.0, 1.5, 2.5, 4.0, 6.0])
        assert min(errors.values()) > 1.0
