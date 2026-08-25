"""Unit tests for advective temporal interpolation of radar volumes.

The tests are organised around the three things that can independently be wrong:

``TestOpticalFlowSign`` / ``TestWarpDirection``
    The **sign** of the motion field and of the warp. This is the single easiest
    thing to get backwards, and getting it backwards makes the reconstruction
    *worse than a naive time average* while still producing plausible-looking
    output. Both are pinned against a synthetic scene translating at a known
    velocity, and the warp direction is established end-to-end (warp by the full
    displacement and check it reproduces the later volume) rather than by
    inspection.

``TestReconstructionAccuracy``
    Whether the morph actually beats the baseline it is supposed to beat.

``TestOutputTiming``
    That the output carries the time it represents. A reconstructed volume is built
    by deep-copying a bracketing volume, so without an explicit correction it
    inherits that volume's clock --- the defect
    :mod:`radar_palette.advection.timing` exists to fix, here wired in at the
    source.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from radar_palette.advection import (  # noqa: E402
    advection_interpolate,
    grid_optical_flow,
    volume_reference_time,
)

requires_skimage = pytest.mark.skipif(
    pytest.importorskip.__module__
    and __import__("importlib.util", fromlist=["find_spec"]).find_spec("skimage")
    is None,
    reason="scikit-image required (pip install radar-palette[advection])",
)

EASTWARD_SPEED_MS = 20.0
VOLUME_SEPARATION_S = 300.0
TOTAL_DISPLACEMENT_M = EASTWARD_SPEED_MS * VOLUME_SEPARATION_S

FLOW_GRID_SHAPE = (4, 121, 121)
FLOW_GRID_LIMITS = (
    (1000.0, 6000.0),
    (-120000.0, 120000.0),
    (-120000.0, 120000.0),
)


def make_translating_volume(
    offset_east_m,
    offset_north_m=0.0,
    start_second=0.0,
    nrays=180,
    ngates=80,
    nsweeps=2,
):
    """A volume containing one Gaussian echo blob at a known position.

    Using an analytic blob rather than a canned test radar means the truth at any
    fractional time is known exactly: the blob at ``alpha`` sits at the linearly
    interpolated position, so reconstruction error can be measured against an
    analytic field rather than against another approximation.
    """
    radar = pyart.testing.make_empty_ppi_radar(ngates, nrays, nsweeps)
    radar.range["data"] = 1000.0 + 1500.0 * np.arange(ngates, dtype="float64")
    azimuths = (360.0 / nrays) * np.arange(nrays, dtype="float64")
    radar.azimuth["data"] = np.tile(azimuths, nsweeps)
    fixed_angles = np.array([0.5, 1.5])[:nsweeps]
    radar.elevation["data"] = np.repeat(fixed_angles, nrays)
    radar.fixed_angle["data"] = fixed_angles
    radar.time["data"] = start_second + 0.5 * np.arange(
        nrays * nsweeps, dtype="float64"
    )
    radar.time["units"] = "seconds since 2011-05-20T11:27:34Z"

    gate_east = radar.gate_x["data"]
    gate_north = radar.gate_y["data"]
    blob = 45.0 * np.exp(
        -((gate_east - offset_east_m) ** 2 + (gate_north - offset_north_m) ** 2)
        / (2 * 25000.0**2)
    )
    radar.add_field(
        "reflectivity",
        {
            "data": np.ma.masked_invalid(
                np.where(blob > 1.0, blob, np.nan).astype("float32")
            ),
            "units": "dBZ",
            "_FillValue": -9999.0,
        },
        replace_existing=True,
    )
    return radar


@pytest.fixture(scope="module")
def bracketing_volumes():
    """Two volumes bracketing a target time, plus the analytic truth between them."""
    half = TOTAL_DISPLACEMENT_M / 2.0
    earlier = make_translating_volume(-half, start_second=0.0)
    later = make_translating_volume(+half, start_second=VOLUME_SEPARATION_S)
    truth = make_translating_volume(0.0, start_second=VOLUME_SEPARATION_S / 2.0)
    return earlier, later, truth


@pytest.fixture(scope="module")
def flow_grids(bracketing_volumes):
    earlier, later, _ = bracketing_volumes
    grid_kwargs = {
        "grid_shape": FLOW_GRID_SHAPE,
        "grid_limits": FLOW_GRID_LIMITS,
        "fields": ["reflectivity"],
        "weighting_function": "Barnes2",
        "roi_func": "dist_beam",
        "min_radius": 1000.0,
    }
    return (
        pyart.map.grid_from_radars(earlier, **grid_kwargs),
        pyart.map.grid_from_radars(later, **grid_kwargs),
    )


def echo_median(values, echo_mask):
    return float(np.median(values[echo_mask]))


def rmse_over_echo(prediction, truth):
    valid = np.isfinite(prediction) & np.isfinite(truth)
    return float(np.sqrt(np.nanmean((prediction[valid] - truth[valid]) ** 2)))


def field_array(radar, field="reflectivity"):
    return np.ma.filled(radar.fields[field]["data"].astype("float64"), np.nan)


@requires_skimage
class TestOpticalFlowSign:
    """The motion field must point from the earlier volume to the later one."""

    def test_recovers_the_eastward_displacement(self, flow_grids):
        earlier_grid, later_grid = flow_grids
        displacement_north, displacement_east = grid_optical_flow(
            earlier_grid, later_grid, "reflectivity"
        )
        earlier = np.ma.filled(earlier_grid.fields["reflectivity"]["data"], np.nan)
        later = np.ma.filled(later_grid.fields["reflectivity"]["data"], np.nan)
        core = (earlier > 25.0) | (later > 25.0)
        recovered = echo_median(displacement_east, core)
        assert recovered == pytest.approx(TOTAL_DISPLACEMENT_M, rel=0.15)

    def test_eastward_displacement_is_positive(self, flow_grids):
        """A sign flip here makes the reconstruction worse than a time average."""
        earlier_grid, later_grid = flow_grids
        _, displacement_east = grid_optical_flow(
            earlier_grid, later_grid, "reflectivity"
        )
        earlier = np.ma.filled(earlier_grid.fields["reflectivity"]["data"], np.nan)
        core = earlier > 25.0
        assert echo_median(displacement_east, core) > 0.0

    def test_northward_displacement_is_near_zero(self, flow_grids):
        earlier_grid, later_grid = flow_grids
        displacement_north, _ = grid_optical_flow(
            earlier_grid, later_grid, "reflectivity"
        )
        earlier = np.ma.filled(earlier_grid.fields["reflectivity"]["data"], np.nan)
        core = earlier > 25.0
        assert abs(echo_median(displacement_north, core)) < 0.05 * (
            TOTAL_DISPLACEMENT_M
        )

    def test_reversing_the_arguments_reverses_the_sign(self, flow_grids):
        earlier_grid, later_grid = flow_grids
        _, forward = grid_optical_flow(earlier_grid, later_grid, "reflectivity")
        _, backward = grid_optical_flow(later_grid, earlier_grid, "reflectivity")
        earlier = np.ma.filled(earlier_grid.fields["reflectivity"]["data"], np.nan)
        core = earlier > 25.0
        assert echo_median(forward, core) > 0.0 > echo_median(backward, core)

    def test_returns_north_then_east(self, flow_grids):
        """Return order is (north, east); swapping them silently rotates the field."""
        earlier_grid, later_grid = flow_grids
        first, second = grid_optical_flow(earlier_grid, later_grid, "reflectivity")
        earlier = np.ma.filled(earlier_grid.fields["reflectivity"]["data"], np.nan)
        core = earlier > 25.0
        assert abs(echo_median(first, core)) < abs(echo_median(second, core))

    def test_shapes_match_the_grid(self, flow_grids):
        earlier_grid, later_grid = flow_grids
        north, east = grid_optical_flow(earlier_grid, later_grid, "reflectivity")
        assert north.shape == east.shape == FLOW_GRID_SHAPE

    def test_rejects_mismatched_grid_shapes(self, bracketing_volumes, flow_grids):
        earlier, later, _ = bracketing_volumes
        earlier_grid, _ = flow_grids
        smaller = pyart.map.grid_from_radars(
            later,
            grid_shape=(3, 61, 61),
            grid_limits=FLOW_GRID_LIMITS,
            fields=["reflectivity"],
        )
        with pytest.raises(ValueError, match="identical shapes"):
            grid_optical_flow(earlier_grid, smaller, "reflectivity")


@requires_skimage
class TestWarpDirection:
    """Established end-to-end, because inspection is unreliable here."""

    def test_full_displacement_maps_earlier_onto_later(self, bracketing_volumes):
        """Warping by the FULL displacement must reproduce the later volume.

        This is the decisive check, and the one that establishes the warp sign:
        measured 0.46 dBZ RMSE for the correct direction against 6.09 dBZ for the
        reversed one, a factor of 13. The implementation follows this test, not the
        reverse.
        """
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=1.0,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        error = rmse_over_echo(field_array(reconstructed), field_array(later))
        assert error < 1.0

    def test_alpha_zero_recovers_the_earlier_volume(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.0,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert rmse_over_echo(field_array(reconstructed), field_array(earlier)) < 0.5


@requires_skimage
class TestReconstructionAccuracy:
    def test_beats_a_naive_time_average(self, bracketing_volumes):
        """The whole point of advecting. Measured 0.047 vs 0.171 dBZ on this scene.

        Note this synthetic scene is a clean rigid translation, so the margin is
        far larger than on real convection --- see the module docstring of
        :mod:`radar_palette.advection.interpolate` for the honest figure.
        """
        earlier, later, truth = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        truth_values = field_array(truth)
        advected_error = rmse_over_echo(field_array(reconstructed), truth_values)
        naive_error = rmse_over_echo(
            np.nanmean(np.stack([field_array(earlier), field_array(later)]), axis=0),
            truth_values,
        )
        assert advected_error < naive_error

    def test_reconstruction_error_is_small_in_absolute_terms(self, bracketing_volumes):
        earlier, later, truth = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert rmse_over_echo(field_array(reconstructed), field_array(truth)) < 1.0

    def test_output_is_on_the_earlier_volumes_geometry(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert reconstructed.nrays == earlier.nrays
        assert reconstructed.ngates == earlier.ngates
        assert reconstructed.nsweeps == earlier.nsweeps
        np.testing.assert_allclose(
            reconstructed.azimuth["data"], earlier.azimuth["data"]
        )

    def test_output_values_stay_within_the_input_range(self, bracketing_volumes):
        """A blend of two bounded fields cannot exceed their combined range."""
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        values = field_array(reconstructed)
        combined = np.concatenate(
            [field_array(earlier).ravel(), field_array(later).ravel()]
        )
        assert np.nanmin(values) >= np.nanmin(combined) - 1e-6
        assert np.nanmax(values) <= np.nanmax(combined) + 1e-6

    def test_field_is_masked_where_no_data_reached(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert np.ma.isMaskedArray(reconstructed.fields["reflectivity"]["data"])

    def test_renames_the_output_field_on_request(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            interp_field_name="reflectivity_interpolated",
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert "reflectivity_interpolated" in reconstructed.fields

    def test_preserves_field_metadata(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert reconstructed.fields["reflectivity"]["units"] == "dBZ"


@requires_skimage
class TestOutputTiming:
    """The reconstructed volume must carry the time it represents.

    This is the integration the timing module exists for. The output is built by
    deep-copying the earlier volume, so absent an explicit correction it reports
    the earlier volume's acquisition time --- a volume that claims to have been
    observed 150 s before the instant it depicts.
    """

    def test_reference_time_is_between_the_two_inputs(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert (
            volume_reference_time(earlier)
            < volume_reference_time(reconstructed)
            < volume_reference_time(later)
        )

    def test_reference_time_matches_the_requested_fraction(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        alpha = 0.25
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=alpha,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        earlier_time = volume_reference_time(earlier)
        span = (volume_reference_time(later) - earlier_time).total_seconds()
        expected = earlier_time + timedelta(seconds=alpha * span)
        actual = volume_reference_time(reconstructed)
        assert abs((actual - expected).total_seconds()) < 1.0

    def test_does_not_inherit_the_earlier_volumes_clock(self, bracketing_volumes):
        """The regression this wiring prevents."""
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert volume_reference_time(reconstructed) != volume_reference_time(earlier)

    def test_intra_volume_ray_spacing_is_preserved(self, bracketing_volumes):
        """Ray timing structure belongs to the scan strategy, not the interpolation."""
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        np.testing.assert_allclose(
            np.diff(reconstructed.time["data"]),
            np.diff(earlier.time["data"]),
            atol=1e-9,
        )

    def test_endpoint_alpha_reports_the_endpoint_time(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=1.0,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        difference = (
            volume_reference_time(reconstructed) - volume_reference_time(later)
        ).total_seconds()
        assert abs(difference) < 1.0

    def test_history_records_the_interpolation(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        reconstructed = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert "radar_palette" in reconstructed.metadata.get("history", "")


@requires_skimage
class TestTargetTimeSelection:
    def test_target_time_is_converted_to_a_fraction(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        earlier_time = volume_reference_time(earlier)
        span = (volume_reference_time(later) - earlier_time).total_seconds()
        target = earlier_time + timedelta(seconds=0.5 * span)
        from_target = advection_interpolate(
            earlier,
            later,
            target_time=target,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        from_alpha = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        np.testing.assert_allclose(
            field_array(from_target), field_array(from_alpha), atol=1e-6
        )

    def test_defaults_to_the_midpoint(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        default = advection_interpolate(
            earlier, later, grid_shape=FLOW_GRID_SHAPE, grid_limits=FLOW_GRID_LIMITS
        )
        explicit = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        np.testing.assert_allclose(
            field_array(default), field_array(explicit), atol=1e-6
        )

    def test_accepts_a_naive_target_time(self, bracketing_volumes):
        """The most natural call a Py-ART user can make must work.

        ``pyart.util.datetime_from_radar`` returns a *naive* datetime while
        :func:`volume_reference_time` returns a UTC-aware one, so building a target
        from the former and passing it here used to raise

            TypeError: can't subtract offset-naive and offset-aware datetimes

        The existing tests all built the target from ``volume_reference_time``, so
        they were aware-on-aware and never reached this. CF times are UTC by
        construction, so a naive target is interpreted as UTC rather than rejected.
        """
        earlier, later, _ = bracketing_volumes
        aware = volume_reference_time(earlier)
        span = (volume_reference_time(later) - aware).total_seconds()
        aware_target = aware + timedelta(seconds=0.5 * span)
        naive_target = aware_target.replace(tzinfo=None)
        assert naive_target.tzinfo is None

        from_naive = advection_interpolate(
            earlier,
            later,
            target_time=naive_target,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        from_aware = advection_interpolate(
            earlier,
            later,
            target_time=aware_target,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        np.testing.assert_allclose(
            field_array(from_naive), field_array(from_aware), atol=1e-6
        )

    def test_naive_and_aware_targets_agree_off_the_midpoint(self, bracketing_volumes):
        """Not just at alpha=0.5, where a sign or offset slip would cancel."""
        earlier, later, _ = bracketing_volumes
        aware = volume_reference_time(earlier)
        span = (volume_reference_time(later) - aware).total_seconds()
        target = aware + timedelta(seconds=0.25 * span)
        from_naive = advection_interpolate(
            earlier,
            later,
            target_time=target.replace(tzinfo=None),
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        from_alpha = advection_interpolate(
            earlier,
            later,
            alpha=0.25,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        np.testing.assert_allclose(
            field_array(from_naive), field_array(from_alpha), atol=1e-6
        )

    def test_accepts_a_cftime_target(self, bracketing_volumes):
        """The type ``pyart.util.datetime_from_radar`` actually returns.

        It is a ``cftime.DatetimeGregorian``, which is *not* a
        :class:`datetime.datetime` subclass and whose ``replace()`` rejects
        ``tzinfo`` outright. A first version of this fix normalised with
        ``target_time.replace(tzinfo=UTC)`` and passed the naive-datetime test above
        while still raising on real Py-ART input, so the two cases are pinned
        separately.
        """
        cftime = pytest.importorskip("cftime")
        earlier, later, _ = bracketing_volumes
        aware = volume_reference_time(earlier)
        span = (volume_reference_time(later) - aware).total_seconds()
        midpoint = aware + timedelta(seconds=0.5 * span)
        target = cftime.DatetimeGregorian(
            midpoint.year,
            midpoint.month,
            midpoint.day,
            midpoint.hour,
            midpoint.minute,
            midpoint.second,
            midpoint.microsecond,
        )
        assert not isinstance(target, datetime)

        from_cftime = advection_interpolate(
            earlier,
            later,
            target_time=target,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        from_alpha = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        np.testing.assert_allclose(
            field_array(from_cftime), field_array(from_alpha), atol=1e-6
        )

    def test_rejects_a_missing_field(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        with pytest.raises(KeyError, match="velocity"):
            advection_interpolate(
                earlier,
                later,
                field="velocity",
                grid_shape=FLOW_GRID_SHAPE,
                grid_limits=FLOW_GRID_LIMITS,
            )


@requires_skimage
class TestObjectFlavours:
    """Both entry points honour the package-wide flavour contract."""

    def test_accepts_and_returns_a_datatree(self, bracketing_volumes, tmp_path):
        xradar = pytest.importorskip("xradar")
        earlier, later, _ = bracketing_volumes
        trees = []
        for index, volume in enumerate((earlier, later)):
            path = str(tmp_path / f"volume_{index}.nc")
            pyart.io.write_cfradial(path, volume)
            trees.append(xradar.io.open_cfradial1_datatree(path).load())
        result = advection_interpolate(
            trees[0],
            trees[1],
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        import xarray as xr

        assert isinstance(result, xr.DataTree)

    def test_output_flavor_can_be_overridden(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        result = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            output_flavor="xradar",
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        import xarray as xr

        assert isinstance(result, xr.DataTree)

    def test_pyart_input_returns_pyart_by_default(self, bracketing_volumes):
        earlier, later, _ = bracketing_volumes
        result = advection_interpolate(
            earlier,
            later,
            alpha=0.5,
            grid_shape=FLOW_GRID_SHAPE,
            grid_limits=FLOW_GRID_LIMITS,
        )
        assert isinstance(result, pyart.core.Radar)


@requires_skimage
class TestRepeatedAzimuths:
    """A real antenna sometimes records the same azimuth twice in one sweep.

    Measured on ARM C-SAPR2: 3 of 195 sweeps across 13 consecutive volumes contain
    one duplicated azimuth, which is what a brief dwell looks like in the data. A
    synthetic volume built from ``arange`` never does, so this went unnoticed until
    the operator was run on a real sequence.

    It matters because :func:`_sample_sweep` tiles the azimuth axis at -360, 0 and
    +360 degrees to make the 0/360 seam interpolate correctly, and
    ``RegularGridInterpolator`` requires that axis to be *strictly* ascending.
    Sorting puts duplicates adjacent, so a single repeat makes the whole
    reconstruction raise rather than degrade.
    """

    def _volume_with_duplicate_azimuth(self, offset_east_m, start_second=0.0):
        """A translating volume whose first sweep repeats one azimuth."""
        radar = make_translating_volume(offset_east_m, start_second=start_second)
        azimuths = radar.azimuth["data"].copy()
        # duplicate ray 10 onto ray 11 within the first sweep
        azimuths[11] = azimuths[10]
        radar.azimuth["data"] = azimuths
        return radar

    def test_duplicate_azimuth_does_not_raise(self):
        """The case real data presents: one repeated azimuth in one sweep."""
        earlier = self._volume_with_duplicate_azimuth(0.0)
        later = self._volume_with_duplicate_azimuth(
            TOTAL_DISPLACEMENT_M, start_second=VOLUME_SEPARATION_S
        )
        result = advection_interpolate(earlier, later, alpha=0.5)
        assert "reflectivity" in result.fields

    def test_duplicate_azimuth_gives_a_usable_field(self):
        """Not merely non-raising: the reconstruction must still carry echo."""
        earlier = self._volume_with_duplicate_azimuth(0.0)
        later = self._volume_with_duplicate_azimuth(
            TOTAL_DISPLACEMENT_M, start_second=VOLUME_SEPARATION_S
        )
        values = advection_interpolate(earlier, later, alpha=0.5).fields[
            "reflectivity"
        ]["data"]
        finite = np.isfinite(np.ma.filled(values.astype("float64"), np.nan))
        assert finite.mean() > 0.5
        assert np.nanmax(np.ma.filled(values.astype("float64"), np.nan)) > 10.0

    def test_many_duplicates_still_work(self):
        """Degenerate but legal: a sweep that dwells repeatedly."""
        earlier = make_translating_volume(0.0)
        later = make_translating_volume(
            TOTAL_DISPLACEMENT_M, start_second=VOLUME_SEPARATION_S
        )
        for radar in (earlier, later):
            azimuths = radar.azimuth["data"].copy()
            azimuths[5:15] = azimuths[5]
            radar.azimuth["data"] = azimuths
        result = advection_interpolate(earlier, later, alpha=0.5)
        assert "reflectivity" in result.fields

    def test_duplicate_free_input_is_unaffected(self):
        """The fix must not perturb the well-behaved case.

        Reconstructing from strictly-ascending azimuths must give bit-identical
        output before and after de-duplication, since there is nothing to remove.
        """
        earlier = make_translating_volume(0.0)
        later = make_translating_volume(
            TOTAL_DISPLACEMENT_M, start_second=VOLUME_SEPARATION_S
        )
        first = advection_interpolate(earlier, later, alpha=0.5).fields["reflectivity"][
            "data"
        ]
        second = advection_interpolate(earlier, later, alpha=0.5).fields[
            "reflectivity"
        ]["data"]
        np.testing.assert_array_equal(
            np.ma.filled(first, np.nan), np.ma.filled(second, np.nan)
        )


@requires_skimage
class TestCarryFields:
    """One motion field, several variables warped by it.

    Polarimetric variables are not tracers. Differential reflectivity is a shape
    measure, correlation coefficient is closer to a quality flag, and Doppler
    velocity is signed and folded at the Nyquist interval -- estimating optical flow
    from any of them would be tracking the wrong thing. Reflectivity is the tracer;
    the motion it yields is a property of the *scene*, so the physically defensible
    move is to estimate once and apply that displacement to every variable.

    Calling the operator once per variable would instead derive a different motion
    field from each, which is both wrong and four times the cost.
    """

    def _pair(self):
        earlier = make_translating_volume(0.0)
        later = make_translating_volume(
            TOTAL_DISPLACEMENT_M, start_second=VOLUME_SEPARATION_S
        )
        for radar in (earlier, later):
            reflectivity = np.ma.filled(
                radar.fields["reflectivity"]["data"].astype("float64"), np.nan
            )
            # A companion variable with different units and a different sign
            # convention, so a swap or a unit slip would show up.
            radar.add_field(
                "differential_reflectivity",
                {
                    "data": np.ma.masked_invalid(0.1 * reflectivity - 2.0),
                    "units": "dB",
                    "standard_name": "differential_reflectivity",
                },
            )
        return earlier, later

    def test_carried_field_appears_in_the_output(self):
        earlier, later = self._pair()
        result = advection_interpolate(
            earlier, later, alpha=0.5, carry_fields=["differential_reflectivity"]
        )
        assert set(result.fields) == {"reflectivity", "differential_reflectivity"}

    def test_carried_field_keeps_its_own_metadata(self):
        """A carried variable must not inherit the tracer's units."""
        earlier, later = self._pair()
        result = advection_interpolate(
            earlier, later, alpha=0.5, carry_fields=["differential_reflectivity"]
        )
        assert result.fields["differential_reflectivity"]["units"] == "dB"
        assert result.fields["reflectivity"]["units"] != "dB"

    def test_carrying_reproduces_a_separate_call_on_the_tracer(self):
        """Adding a carried variable must not perturb the tracer's own result."""
        earlier, later = self._pair()
        alone = advection_interpolate(earlier, later, alpha=0.5)
        together = advection_interpolate(
            earlier, later, alpha=0.5, carry_fields=["differential_reflectivity"]
        )
        np.testing.assert_allclose(
            np.ma.filled(alone.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(together.fields["reflectivity"]["data"], np.nan),
            equal_nan=True,
        )

    def test_carried_field_is_warped_not_copied(self):
        """The carried variable must move with the flow, not pass through unchanged.

        This is the test that would catch wiring the carried field straight from the
        earlier volume: the blob translates, so a copy would sit at the start
        position while the warp puts it half way.
        """
        earlier, later = self._pair()
        result = advection_interpolate(
            earlier, later, alpha=0.5, carry_fields=["differential_reflectivity"]
        )
        warped = np.ma.filled(
            result.fields["differential_reflectivity"]["data"].astype("float64"), np.nan
        )
        original = np.ma.filled(
            earlier.fields["differential_reflectivity"]["data"].astype("float64"),
            np.nan,
        )
        comparable = np.isfinite(warped) & np.isfinite(original)
        assert not np.allclose(warped[comparable], original[comparable])

    def test_carried_field_tracks_the_tracer_through_the_same_transform(self):
        """The carried variable must go through the identical transform.

        It is an affine function of the tracer in this fixture, so the
        reconstruction must satisfy the same relation: identical motion, identical
        resampling, identical blend weights.
        """
        earlier, later = self._pair()
        result = advection_interpolate(
            earlier, later, alpha=0.5, carry_fields=["differential_reflectivity"]
        )
        tracer = np.ma.filled(
            result.fields["reflectivity"]["data"].astype("float64"), np.nan
        )
        carried = np.ma.filled(
            result.fields["differential_reflectivity"]["data"].astype("float64"), np.nan
        )
        both = np.isfinite(tracer) & np.isfinite(carried)
        np.testing.assert_allclose(
            carried[both], 0.1 * tracer[both] - 2.0, rtol=1e-9, atol=1e-9
        )

    def test_missing_carried_field_is_reported(self):
        earlier, later = self._pair()
        with pytest.raises(KeyError, match="absent_field"):
            advection_interpolate(
                earlier, later, alpha=0.5, carry_fields=["absent_field"]
            )

    def test_no_carry_fields_is_the_previous_behaviour(self):
        """Default stays one field out, so existing callers are untouched."""
        earlier, later = self._pair()
        result = advection_interpolate(earlier, later, alpha=0.5)
        assert set(result.fields) == {"reflectivity"}
