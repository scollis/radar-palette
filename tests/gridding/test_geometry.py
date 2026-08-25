"""Unit tests for 4/3-earth geometry transforms and sweep geometry census.

Two things are under test here, and they have different characters.

The transforms are pure maths, so they are tested against exact analytic
properties: the inverse must invert the forward transform to floating-point
closure, and the forward transform must agree with Py-ART's own where Py-ART is
correct. One test deliberately *documents a Py-ART defect* rather than asserting
Py-ART is right --- see ``TestPyartInverseDiscrepancy``.

The census measures sampling geometry and classifies it, so it is tested against
synthetic sweeps built with geometry chosen to sit on each side of every decision
boundary: exact lattices, deliberately jittered lattices just inside and just
outside tolerance, partial sectors, and duplicated fixed angles.
"""

from __future__ import annotations

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from radar_palette.gridding import (  # noqa: E402
    AZ_DEV_TOL_FRAC,
    DR_DEV_TOL_FRAC,
    EARTH_RADIUS_EFFECTIVE_M,
    SPLIT_CUT_TOL_DEG,
    SweepClass,
    SweepGeometry,
    antenna_to_cartesian_43,
    cartesian_to_antenna_43,
    census_radar,
    census_sweep,
)


def make_ppi_radar(
    azimuths_deg,
    fixed_angles_deg=(0.5,),
    ngates=40,
    range_spacing_m=250.0,
    range_first_m=125.0,
    elevation_jitter_deg=0.0,
):
    """Build a Py-ART volume whose azimuth sampling is exactly as specified.

    ``azimuths_deg`` is the per-sweep azimuth sequence, reused for every sweep, so
    a test can put the geometry precisely on either side of a tolerance boundary
    instead of hoping a canned test radar happens to.
    """
    azimuths_deg = np.asarray(azimuths_deg, dtype="float64")
    nrays_per_sweep = azimuths_deg.size
    nsweeps = len(fixed_angles_deg)
    total_rays = nrays_per_sweep * nsweeps

    radar = pyart.testing.make_empty_ppi_radar(ngates, nrays_per_sweep, nsweeps)
    radar.range["data"] = range_first_m + range_spacing_m * np.arange(
        ngates, dtype="float64"
    )
    radar.azimuth["data"] = np.tile(azimuths_deg, nsweeps)

    elevations = np.empty(total_rays, dtype="float64")
    for sweep_index, fixed_angle in enumerate(fixed_angles_deg):
        start = sweep_index * nrays_per_sweep
        jitter = elevation_jitter_deg * np.sin(
            np.linspace(0.0, 2.0 * np.pi, nrays_per_sweep, endpoint=False)
        )
        elevations[start : start + nrays_per_sweep] = fixed_angle + jitter
    radar.elevation["data"] = elevations
    radar.fixed_angle["data"] = np.asarray(fixed_angles_deg, dtype="float64")

    radar.add_field(
        "reflectivity",
        {
            "data": np.ma.masked_invalid(
                np.zeros((total_rays, ngates), dtype="float32")
            ),
            "units": "dBZ",
            "long_name": "reflectivity",
            "_FillValue": -9999.0,
        },
        replace_existing=True,
    )
    return radar


def uniform_azimuths(nrays, start_deg=0.0, direction=1):
    """An exactly uniform, exactly closing full-circle azimuth sequence."""
    spacing = direction * 360.0 / nrays
    return (start_deg + spacing * np.arange(nrays, dtype="float64")) % 360.0


class TestForwardTransform:
    """``antenna_to_cartesian_43``: metres in, metres out, 4/3 earth."""

    def test_zero_range_is_the_radar_origin(self):
        x, y, z = antenna_to_cartesian_43(0.0, 0.0, 0.0)
        assert (float(x), float(y), float(z)) == (0.0, 0.0, 0.0)

    def test_azimuth_zero_points_north(self):
        """Meteorological convention: 0 degrees is +y, not +x."""
        x, y, _ = antenna_to_cartesian_43(10_000.0, 0.0, 0.0)
        assert abs(float(x)) < 1e-6
        assert float(y) > 0.0

    def test_azimuth_ninety_points_east(self):
        x, y, _ = antenna_to_cartesian_43(10_000.0, 90.0, 0.0)
        assert float(x) > 0.0
        assert abs(float(y)) < 1e-6

    def test_azimuth_increases_clockwise(self):
        """90 degrees must be east, not west --- the sign trap in this convention."""
        x_east, _, _ = antenna_to_cartesian_43(10_000.0, 90.0, 0.0)
        x_west, _, _ = antenna_to_cartesian_43(10_000.0, 270.0, 0.0)
        assert float(x_east) > 0.0 > float(x_west)

    def test_vertical_beam_puts_all_range_in_height(self):
        _, _, z = antenna_to_cartesian_43(10_000.0, 0.0, 90.0)
        assert float(z) == pytest.approx(10_000.0, rel=1e-9)

    def test_horizontal_beam_still_gains_height_from_curvature(self):
        """A 0-degree beam rises above the radar horizon: this is the 4/3 earth."""
        _, _, z = antenna_to_cartesian_43(100_000.0, 0.0, 0.0)
        expected = np.sqrt(100_000.0**2 + EARTH_RADIUS_EFFECTIVE_M**2) - (
            EARTH_RADIUS_EFFECTIVE_M
        )
        assert float(z) == pytest.approx(expected, rel=1e-9)
        assert float(z) > 500.0

    def test_ground_distance_is_shorter_than_slant_range(self):
        x, y, _ = antenna_to_cartesian_43(100_000.0, 45.0, 5.0)
        assert float(np.hypot(x, y)) < 100_000.0

    def test_accepts_arrays_and_preserves_shape(self):
        ranges = np.linspace(1_000.0, 100_000.0, 7)
        x, y, z = antenna_to_cartesian_43(ranges, np.full(7, 30.0), np.full(7, 1.0))
        assert x.shape == y.shape == z.shape == (7,)

    def test_broadcasts_scalar_against_array(self):
        azimuths = np.linspace(0.0, 350.0, 36)
        x, y, z = antenna_to_cartesian_43(50_000.0, azimuths, 1.0)
        assert x.shape == (36,)
        np.testing.assert_allclose(np.hypot(x, y), np.hypot(x, y)[0], rtol=1e-12)

    def test_matches_pyart_forward_transform(self):
        """Py-ART's forward transform is correct; ours must agree with it.

        Note the unit trap this test pins: ``pyart.core.antenna_to_cartesian``
        takes range in KILOMETRES and returns metres, while this function takes
        and returns metres.
        """
        ranges_m = np.array([1_000.0, 25_000.0, 118_000.0, 250_000.0])
        azimuths = np.array([0.0, 73.0, 180.0, 297.0])
        elevations = np.array([0.5, 1.5, 4.0, 12.0])
        ours = antenna_to_cartesian_43(ranges_m, azimuths, elevations)
        theirs = pyart.core.antenna_to_cartesian(
            ranges_m / 1000.0, azimuths, elevations
        )
        for ours_axis, theirs_axis in zip(ours, theirs, strict=True):
            np.testing.assert_allclose(ours_axis, theirs_axis, rtol=1e-9, atol=1e-6)


class TestInverseTransform:
    """``cartesian_to_antenna_43`` is the exact analytic inverse."""

    @pytest.mark.parametrize(
        "range_m", [500.0, 10_000.0, 118_000.0, 250_000.0, 460_000.0]
    )
    @pytest.mark.parametrize("elevation_deg", [0.0, 0.5, 5.0, 20.0, 60.0])
    def test_round_trip_closes_to_floating_point(self, range_m, elevation_deg):
        azimuth_deg = 73.0
        x, y, z = antenna_to_cartesian_43(range_m, azimuth_deg, elevation_deg)
        range_back, azimuth_back, elevation_back = cartesian_to_antenna_43(x, y, z)
        assert float(range_back) == pytest.approx(range_m, abs=1e-6)
        assert float(azimuth_back) == pytest.approx(azimuth_deg, abs=1e-9)
        assert float(elevation_back) == pytest.approx(elevation_deg, abs=1e-9)

    def test_round_trip_closure_over_a_full_grid(self):
        ranges = np.linspace(1_000.0, 300_000.0, 13)
        azimuths = np.linspace(0.0, 350.0, 12)
        elevations = np.array([0.0, 1.0, 6.0, 19.5])
        range_mesh, azimuth_mesh, elevation_mesh = np.meshgrid(
            ranges, azimuths, elevations, indexing="ij"
        )
        x, y, z = antenna_to_cartesian_43(range_mesh, azimuth_mesh, elevation_mesh)
        range_back, azimuth_back, elevation_back = cartesian_to_antenna_43(x, y, z)
        assert np.max(np.abs(range_back - range_mesh)) < 1e-6
        assert np.max(np.abs(elevation_back - elevation_mesh)) < 1e-9
        assert np.max(np.abs(azimuth_back - azimuth_mesh % 360.0)) < 1e-9

    def test_azimuth_is_returned_on_zero_to_360(self):
        x, y, z = antenna_to_cartesian_43(50_000.0, -30.0, 1.0)
        _, azimuth_back, _ = cartesian_to_antenna_43(x, y, z)
        assert float(azimuth_back) == pytest.approx(330.0, abs=1e-9)

    def test_origin_inverts_without_error(self):
        range_back, _, elevation_back = cartesian_to_antenna_43(0.0, 0.0, 0.0)
        assert float(range_back) == pytest.approx(0.0, abs=1e-9)
        assert np.isfinite(float(elevation_back))

    def test_accepts_arrays(self):
        x = np.array([0.0, 1_000.0, -2_000.0])
        ranges, azimuths, elevations = cartesian_to_antenna_43(x, x, x)
        assert ranges.shape == azimuths.shape == elevations.shape == (3,)


class TestPyartInverseDiscrepancy:
    """Document that Py-ART's inverse is NOT the inverse of its own forward.

    These tests assert a *defect in a dependency*, deliberately. The forward
    transform returns great-circle arc length on the 4/3 earth; Py-ART's
    ``cartesian_to_antenna`` computes a flat straight-line distance with no
    curvature term, so the round trip carries a systematic range bias that grows
    with range. If a future Py-ART release fixes this, these tests fail and should
    be updated --- that failure is informative, not a regression here.
    """

    # Elevation at which the curvature discrepancy is worst. Measured: the range
    # error peaks near 35 degrees at every range, not at grazing or vertical.
    WORST_CASE_ELEVATION_DEG = 35.0

    @staticmethod
    def pyart_inverse_range_m(x, y, z):
        """Py-ART's inverse range in metres, given metre offsets.

        Wrapped because ``pyart.core.cartesian_to_antenna`` assigns into its
        azimuth result (``azimuths[azimuths < 0] += 360``), so it raises
        ``TypeError`` on scalar input and must be handed arrays.

        Note it returns range in **metres** --- unlike the forward
        ``antenna_to_cartesian``, which takes kilometres. The asymmetry runs the
        opposite way to the usual trap.
        """
        pyart_range_m, _, _ = pyart.core.cartesian_to_antenna(
            np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
        )
        return float(np.asarray(pyart_range_m).ravel()[0])

    def test_pyart_inverse_requires_array_input(self):
        """Pinning a second Py-ART trap found while writing these tests."""
        x, y, z = antenna_to_cartesian_43(50_000.0, 10.0, 1.0)
        with pytest.raises(TypeError):
            pyart.core.cartesian_to_antenna(float(x), float(y), float(z))

    @pytest.mark.parametrize(
        ("range_m", "minimum_error_m"),
        [
            # Measured worst-case range errors at WORST_CASE_ELEVATION_DEG:
            # -313 m at 118 km, -1392 m at 250 km, -4649 m at 460 km. Bounds are
            # set below the measured values so a Py-ART change in either direction
            # is visible without the test being brittle.
            (118_000.0, 250.0),
            (250_000.0, 1_000.0),
            (460_000.0, 3_500.0),
        ],
    )
    def test_pyart_inverse_range_error_is_large_at_long_range(
        self, range_m, minimum_error_m
    ):
        x, y, z = antenna_to_cartesian_43(range_m, 40.0, self.WORST_CASE_ELEVATION_DEG)
        assert abs(self.pyart_inverse_range_m(x, y, z) - range_m) > minimum_error_m

    def test_error_is_elevation_dependent_not_range_alone(self):
        """The headline figure is a maximum over elevation, not a fixed offset.

        Measured at 118 km: ~19 m at 1 degree of elevation, but ~313 m near 35
        degrees. A test that probed only low elevation would understate the defect
        by more than an order of magnitude.
        """
        error_at_low_elevation = abs(
            self.pyart_inverse_range_m(*antenna_to_cartesian_43(118_000.0, 40.0, 1.0))
            - 118_000.0
        )
        error_at_worst_elevation = abs(
            self.pyart_inverse_range_m(
                *antenna_to_cartesian_43(118_000.0, 40.0, self.WORST_CASE_ELEVATION_DEG)
            )
            - 118_000.0
        )
        assert error_at_worst_elevation > 10.0 * error_at_low_elevation

    def test_our_inverse_is_orders_of_magnitude_closer(self):
        range_m = 118_000.0
        x, y, z = antenna_to_cartesian_43(range_m, 40.0, self.WORST_CASE_ELEVATION_DEG)
        ours_m = float(np.asarray(cartesian_to_antenna_43(x, y, z)[0]).ravel()[0])
        pyart_m = self.pyart_inverse_range_m(x, y, z)
        assert abs(ours_m - range_m) < 1e-6 < abs(pyart_m - range_m)

    def test_discrepancy_grows_with_range(self):
        """A curvature term, not noise: monotone in range at fixed elevation."""
        errors = [
            abs(
                self.pyart_inverse_range_m(
                    *antenna_to_cartesian_43(
                        range_m, 0.0, self.WORST_CASE_ELEVATION_DEG
                    )
                )
                - range_m
            )
            for range_m in (10_000.0, 100_000.0, 300_000.0)
        ]
        assert errors[0] < errors[1] < errors[2]

    def test_pyart_elevation_error_is_also_biased(self):
        """Elevation carries the same curvature bias: ~398 mdeg at 118 km."""
        x, y, z = antenna_to_cartesian_43(118_000.0, 40.0, 1.0)
        _, _, pyart_elevation = pyart.core.cartesian_to_antenna(
            np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
        )
        error_mdeg = (float(np.asarray(pyart_elevation).ravel()[0]) - 1.0) * 1000.0
        assert error_mdeg > 300.0

    def test_azimuth_agrees_with_pyart(self):
        """Only range and elevation are affected; both use the same convention."""
        x, y, z = antenna_to_cartesian_43(80_000.0, 213.0, 2.0)
        _, pyart_azimuth, _ = pyart.core.cartesian_to_antenna(
            np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)
        )
        assert float(np.asarray(pyart_azimuth).ravel()[0]) % 360.0 == pytest.approx(
            213.0, abs=1e-6
        )


class TestCensusRangeAxis:
    def test_reports_range_spacing_and_first_gate(self):
        radar = make_ppi_radar(uniform_azimuths(360), range_spacing_m=250.0)
        geometry = census_sweep(radar, 0)
        assert geometry.range_spacing_m == pytest.approx(250.0)
        assert geometry.range_first_m == pytest.approx(125.0)

    def test_uniform_range_axis_is_flagged_uniform(self):
        radar = make_ppi_radar(uniform_azimuths(360))
        assert census_sweep(radar, 0).range_spacing_uniform is True

    def test_non_uniform_range_axis_is_detected(self):
        radar = make_ppi_radar(uniform_azimuths(360))
        radar.range["data"] = radar.range["data"] ** 1.05
        geometry = census_sweep(radar, 0)
        assert geometry.range_spacing_uniform is False
        assert geometry.sweep_class is SweepClass.NON_UNIFORM

    def test_gate_count_is_reported(self):
        radar = make_ppi_radar(uniform_azimuths(180), ngates=55)
        assert census_sweep(radar, 0).ngates == 55

    def test_valid_gate_count_stops_at_last_unmasked_gate(self):
        """NEXRAD split cuts share a range axis but Doppler cuts stop earlier."""
        radar = make_ppi_radar(uniform_azimuths(180), ngates=40)
        field = radar.fields["reflectivity"]["data"]
        field[:, 25:] = np.ma.masked
        radar.fields["reflectivity"]["data"] = field
        assert census_sweep(radar, 0).ngates_valid == 25

    def test_fully_masked_sweep_reports_zero_valid_gates(self):
        radar = make_ppi_radar(uniform_azimuths(60))
        radar.fields["reflectivity"]["data"] = np.ma.masked_all(
            radar.fields["reflectivity"]["data"].shape
        )
        assert census_sweep(radar, 0).ngates_valid == 0


class TestCensusAzimuthAxis:
    def test_exact_full_circle_is_classified_exact(self):
        radar = make_ppi_radar(uniform_azimuths(360))
        assert census_sweep(radar, 0).sweep_class is SweepClass.EXACT_UNIFORM_PERIODIC

    def test_exact_full_circle_reports_full_360(self):
        assert (
            census_sweep(make_ppi_radar(uniform_azimuths(720)), 0).is_full_360 is True
        )

    def test_spacing_median_matches_the_lattice(self):
        geometry = census_sweep(make_ppi_radar(uniform_azimuths(720)), 0)
        assert geometry.az_spacing_median_deg == pytest.approx(0.5, abs=1e-9)

    def test_coverage_of_a_full_circle_is_360(self):
        geometry = census_sweep(make_ppi_radar(uniform_azimuths(360)), 0)
        assert geometry.az_coverage_deg == pytest.approx(360.0, abs=1e-6)
        assert geometry.n_revolutions == pytest.approx(1.0, abs=1e-6)

    def test_counterclockwise_sweep_reports_negative_direction(self):
        radar = make_ppi_radar(uniform_azimuths(360, direction=-1))
        assert census_sweep(radar, 0).az_direction == -1

    def test_wrap_seam_is_counted_not_treated_as_non_monotonic(self):
        """Crossing the 0/360 branch cut is bookkeeping, not a geometry defect."""
        geometry = census_sweep(
            make_ppi_radar(uniform_azimuths(360, start_deg=350.0)), 0
        )
        assert geometry.n_wrap_seams == 1
        assert geometry.az_monotonic is True
        assert geometry.sweep_class is SweepClass.EXACT_UNIFORM_PERIODIC

    def test_partial_sector_is_classified_as_a_sector(self):
        azimuths = 30.0 + 0.5 * np.arange(120, dtype="float64")
        geometry = census_sweep(make_ppi_radar(azimuths), 0)
        assert geometry.sweep_class is SweepClass.UNIFORM_PARTIAL_SECTOR
        assert geometry.is_full_360 is False

    def test_sector_coverage_is_measured(self):
        azimuths = 30.0 + 0.5 * np.arange(120, dtype="float64")
        assert census_sweep(
            make_ppi_radar(azimuths), 0
        ).az_coverage_deg == pytest.approx(60.0, abs=1e-6)

    def test_jitter_just_inside_tolerance_stays_exact(self):
        """The classifier boundary is a measured tolerance, so pin both sides."""
        nrays = 360
        spacing = 360.0 / nrays
        azimuths = uniform_azimuths(nrays)
        azimuths[nrays // 2] += 0.5 * AZ_DEV_TOL_FRAC * spacing
        assert (
            census_sweep(make_ppi_radar(azimuths), 0).sweep_class
            is SweepClass.EXACT_UNIFORM_PERIODIC
        )

    def test_jitter_beyond_tolerance_becomes_non_uniform(self):
        nrays = 360
        spacing = 360.0 / nrays
        azimuths = uniform_azimuths(nrays)
        azimuths[nrays // 2] += 5.0 * AZ_DEV_TOL_FRAC * spacing
        assert (
            census_sweep(make_ppi_radar(azimuths), 0).sweep_class
            is SweepClass.NON_UNIFORM
        )

    def test_deviation_is_reported_as_a_fraction_of_spacing(self):
        """The classifier must expose deviation as a fraction of spacing.

        A fractional tolerance is the correct thing to specify because the
        worst-case Nyquist error is ``pi * f``, independent of ray count.
        """
        nrays = 360
        spacing = 360.0 / nrays
        azimuths = uniform_azimuths(nrays)
        azimuths[10] += 0.004 * spacing
        geometry = census_sweep(make_ppi_radar(azimuths), 0)
        assert geometry.az_max_dev_frac == pytest.approx(0.004, rel=0.35)

    def test_non_monotonic_azimuth_is_non_uniform(self):
        azimuths = uniform_azimuths(180)
        azimuths[[40, 41]] = azimuths[[41, 40]]
        assert (
            census_sweep(make_ppi_radar(azimuths), 0).sweep_class
            is SweepClass.NON_UNIFORM
        )

    def test_more_than_one_revolution_is_non_uniform(self):
        azimuths = (1.5 * np.arange(320, dtype="float64")) % 360.0
        geometry = census_sweep(make_ppi_radar(azimuths), 0)
        assert geometry.n_revolutions > 1.02
        assert geometry.sweep_class is SweepClass.NON_UNIFORM

    def test_class_reason_is_populated_and_human_readable(self):
        geometry = census_sweep(make_ppi_radar(uniform_azimuths(360)), 0)
        assert isinstance(geometry.class_reason, str)
        assert geometry.class_reason


class TestCensusElevation:
    def test_reports_the_median_elevation(self):
        radar = make_ppi_radar(uniform_azimuths(180), fixed_angles_deg=(3.5,))
        assert census_sweep(radar, 0).el_median_deg == pytest.approx(3.5, abs=1e-9)

    def test_elevation_jitter_is_measured(self):
        radar = make_ppi_radar(
            uniform_azimuths(180), fixed_angles_deg=(1.0,), elevation_jitter_deg=0.2
        )
        geometry = census_sweep(radar, 0)
        assert geometry.el_max_dev_deg == pytest.approx(0.2, rel=0.05)
        assert geometry.el_std_deg > 0.0

    def test_elevation_jitter_alone_does_not_change_the_class(self):
        """Elevation wobble is reported, not used to reject a sweep."""
        radar = make_ppi_radar(
            uniform_azimuths(360), fixed_angles_deg=(1.0,), elevation_jitter_deg=0.3
        )
        assert census_sweep(radar, 0).sweep_class is SweepClass.EXACT_UNIFORM_PERIODIC

    def test_fixed_angle_is_taken_from_the_radar(self):
        radar = make_ppi_radar(uniform_azimuths(90), fixed_angles_deg=(0.5, 4.5))
        assert census_sweep(radar, 1).fixed_angle == pytest.approx(4.5)


class TestCensusRadar:
    def test_returns_one_geometry_per_sweep(self):
        radar = make_ppi_radar(uniform_azimuths(120), fixed_angles_deg=(0.5, 1.5, 2.5))
        assert len(census_radar(radar)) == 3

    def test_returns_a_list_of_dataclasses_not_a_dataframe(self):
        """Deliberate: no pandas dependency. ``as_row()`` feeds a DataFrame."""
        geometries = census_radar(
            make_ppi_radar(uniform_azimuths(120), fixed_angles_deg=(0.5, 1.5))
        )
        assert isinstance(geometries, list)
        assert all(isinstance(g, SweepGeometry) for g in geometries)

    def test_sweep_indices_are_sequential(self):
        radar = make_ppi_radar(uniform_azimuths(90), fixed_angles_deg=(0.5, 1.5, 2.5))
        assert [g.sweep for g in census_radar(radar)] == [0, 1, 2]

    def test_distinct_fixed_angles_get_distinct_split_cut_groups(self):
        radar = make_ppi_radar(uniform_azimuths(90), fixed_angles_deg=(0.5, 1.5, 2.5))
        groups = [g.split_cut_group for g in census_radar(radar)]
        assert len(set(groups)) == 3

    def test_duplicate_fixed_angles_are_grouped_as_a_split_cut(self):
        """NEXRAD VCPs repeat the low tilts; de-duplication depends on this."""
        radar = make_ppi_radar(uniform_azimuths(90), fixed_angles_deg=(0.5, 0.5, 1.5))
        geometries = census_radar(radar)
        assert geometries[0].split_cut_group == geometries[1].split_cut_group
        assert geometries[2].split_cut_group != geometries[0].split_cut_group

    def test_split_cut_size_counts_group_members(self):
        radar = make_ppi_radar(
            uniform_azimuths(90), fixed_angles_deg=(0.5, 0.5, 0.5, 4.0)
        )
        geometries = census_radar(radar)
        assert [g.split_cut_size for g in geometries] == [3, 3, 3, 1]

    def test_angles_within_tolerance_are_the_same_group(self):
        radar = make_ppi_radar(
            uniform_azimuths(90),
            fixed_angles_deg=(0.5, 0.5 + 0.5 * SPLIT_CUT_TOL_DEG),
        )
        geometries = census_radar(radar)
        assert geometries[0].split_cut_group == geometries[1].split_cut_group

    def test_angles_beyond_tolerance_are_different_groups(self):
        radar = make_ppi_radar(
            uniform_azimuths(90),
            fixed_angles_deg=(0.5, 0.5 + 10.0 * SPLIT_CUT_TOL_DEG),
        )
        geometries = census_radar(radar)
        assert geometries[0].split_cut_group != geometries[1].split_cut_group

    def test_accepts_an_xradar_datatree(self, tmp_path):
        """The census goes through the shared flavour layer like everything else."""
        xradar = pytest.importorskip("xradar")
        radar = make_ppi_radar(uniform_azimuths(120), fixed_angles_deg=(0.5, 1.5))
        path = str(tmp_path / "volume.nc")
        pyart.io.write_cfradial(path, radar)
        datatree = xradar.io.open_cfradial1_datatree(path).load()
        geometries = census_radar(datatree)
        assert len(geometries) == 2
        assert all(isinstance(g, SweepGeometry) for g in geometries)

    def test_valid_field_selection_is_honoured(self):
        radar = make_ppi_radar(uniform_azimuths(120))
        second = np.ma.masked_invalid(
            np.zeros(radar.fields["reflectivity"]["data"].shape, dtype="float32")
        )
        second[:, 10:] = np.ma.masked
        radar.add_field(
            "velocity",
            {"data": second, "units": "m/s", "_FillValue": -9999.0},
            replace_existing=True,
        )
        assert census_sweep(radar, 0, valid_field="velocity").ngates_valid == 10


class TestSweepGeometryDataclass:
    def test_as_row_returns_a_flat_mapping(self):
        geometry = census_sweep(make_ppi_radar(uniform_azimuths(90)), 0)
        row = geometry.as_row()
        assert isinstance(row, dict)
        assert all(not isinstance(v, (list, dict)) for v in row.values())

    def test_as_row_renders_the_class_as_a_string(self):
        """So the row is trivially serialisable to CSV without enum handling."""
        row = census_sweep(make_ppi_radar(uniform_azimuths(90)), 0).as_row()
        assert row["sweep_class"] == SweepClass.EXACT_UNIFORM_PERIODIC.value
        assert isinstance(row["sweep_class"], str)

    def test_rows_from_a_volume_share_their_keys(self):
        radar = make_ppi_radar(uniform_azimuths(90), fixed_angles_deg=(0.5, 1.5))
        rows = [g.as_row() for g in census_radar(radar)]
        assert rows[0].keys() == rows[1].keys()


class TestTolerances:
    def test_tolerances_are_the_documented_values(self):
        """These are derived, not tuned; a silent change should break a test."""
        assert AZ_DEV_TOL_FRAC == 0.01
        assert DR_DEV_TOL_FRAC == 1.0e-3
        assert SPLIT_CUT_TOL_DEG == 0.05

    def test_effective_earth_radius_is_four_thirds(self):
        assert EARTH_RADIUS_EFFECTIVE_M == pytest.approx(6371.0 * 1000.0 * 4.0 / 3.0)

    def test_effective_radius_matches_pyart(self):
        """Consistency with Py-ART's forward transform depends on this constant."""
        x, y, z = antenna_to_cartesian_43(200_000.0, 0.0, 0.0)
        pyart_x, pyart_y, pyart_z = pyart.core.antenna_to_cartesian(200.0, 0.0, 0.0)
        assert float(z) == pytest.approx(float(pyart_z), rel=1e-12)


class TestSweepClass:
    def test_is_a_string_enum_for_readable_output(self):
        assert SweepClass.EXACT_UNIFORM_PERIODIC.value == "EXACT_UNIFORM_PERIODIC"

    def test_all_three_classes_are_reachable(self):
        assert {c.value for c in SweepClass} == {
            "EXACT_UNIFORM_PERIODIC",
            "UNIFORM_PARTIAL_SECTOR",
            "NON_UNIFORM",
        }
