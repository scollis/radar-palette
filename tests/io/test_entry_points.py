"""Tests for the flavour-aware public entry points of both subpackages.

These cover the contract a user actually touches: pass either object family in,
get the expected family back, with the default mirroring the input.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")
xradar = pytest.importorskip("xradar")
xr = pytest.importorskip("xarray")

from radar_palette.advection import retime_interpolated_volume  # noqa: E402
from radar_palette.gridding import grid_volume  # noqa: E402
from radar_palette.io import (  # noqa: E402
    RadarFlavor,
    detect_grid_flavor,
    detect_radar_flavor,
)
from radar_palette.testing import assign_scan_times, make_empty_ppi_volume  # noqa: E402

VOLUME_START = datetime(2011, 5, 20, 11, 34, 0, tzinfo=UTC)
SECONDS_BETWEEN_VOLUMES = 420.0
GRID_SHAPE = (3, 11, 11)
GRID_LIMITS = ((0.0, 2000.0), (-5000.0, 5000.0), (-5000.0, 5000.0))


def with_reflectivity(radar, value=10.0):
    radar.add_field(
        "reflectivity",
        {
            "data": np.ma.masked_invalid(
                np.full((radar.nrays, radar.ngates), value, dtype="float32")
            ),
            "units": "dBZ",
            "long_name": "reflectivity",
            "_FillValue": -9999.0,
        },
        replace_existing=True,
    )
    return radar


def as_datatree(radar, tmp_path, name):
    path = str(tmp_path / name)
    pyart.io.write_cfradial(path, radar)
    return xradar.io.open_cfradial1_datatree(path).load()


@pytest.fixture
def early_pyart():
    return with_reflectivity(
        assign_scan_times(make_empty_ppi_volume(nsweeps=2), VOLUME_START)
    )


@pytest.fixture
def late_pyart():
    return with_reflectivity(
        assign_scan_times(
            make_empty_ppi_volume(nsweeps=2),
            VOLUME_START + timedelta(seconds=SECONDS_BETWEEN_VOLUMES),
        )
    )


def reference_time_of(volume):
    """Reference time of a volume in either flavour."""
    from radar_palette.advection import volume_reference_time
    from radar_palette.io import to_pyart_radar

    adapted, _ = to_pyart_radar(volume)
    return volume_reference_time(adapted)


class TestRetimeInterpolatedVolumeFlavors:
    """``retime_interpolated_volume`` mirrors the interpolated volume's flavour."""

    def test_pyart_in_pyart_out_by_default(self, early_pyart, late_pyart):
        import copy

        result = retime_interpolated_volume(
            copy.deepcopy(early_pyart), early_pyart, late_pyart, 0.5
        )
        assert detect_radar_flavor(result) is RadarFlavor.PYART

    def test_datatree_in_datatree_out_by_default(
        self, early_pyart, late_pyart, tmp_path
    ):
        interpolated = as_datatree(early_pyart, tmp_path, "interp.nc")
        result = retime_interpolated_volume(interpolated, early_pyart, late_pyart, 0.5)
        assert isinstance(result, xr.DataTree)

    def test_datatree_output_carries_the_corrected_time(
        self, early_pyart, late_pyart, tmp_path
    ):
        """The regression, verified through the xradar object family."""
        interpolated = as_datatree(early_pyart, tmp_path, "interp.nc")
        result = retime_interpolated_volume(interpolated, early_pyart, late_pyart, 0.5)
        advance = (
            reference_time_of(result) - reference_time_of(early_pyart)
        ).total_seconds()
        assert advance == pytest.approx(SECONDS_BETWEEN_VOLUMES / 2, abs=1e-3)

    def test_datatree_time_coordinate_is_actually_rewritten(
        self, early_pyart, late_pyart, tmp_path
    ):
        """Guard the wrapper write-through trap: the tree itself must change."""
        interpolated = as_datatree(early_pyart, tmp_path, "interp.nc")
        first_ray_before = interpolated["sweep_0"].ds["time"].values[0]
        result = retime_interpolated_volume(interpolated, early_pyart, late_pyart, 0.5)
        first_ray_after = result["sweep_0"].ds["time"].values[0]
        elapsed = (first_ray_after - first_ray_before) / np.timedelta64(1, "s")
        assert elapsed == pytest.approx(SECONDS_BETWEEN_VOLUMES / 2, abs=1e-3)

    def test_mixed_flavor_brackets_are_accepted(
        self, early_pyart, late_pyart, tmp_path
    ):
        """Bracketing volumes need not share the interpolated volume's flavour."""
        import copy

        late_as_tree = as_datatree(late_pyart, tmp_path, "late.nc")
        result = retime_interpolated_volume(
            copy.deepcopy(early_pyart), early_pyart, late_as_tree, 0.5
        )
        advance = (
            reference_time_of(result) - reference_time_of(early_pyart)
        ).total_seconds()
        assert advance == pytest.approx(SECONDS_BETWEEN_VOLUMES / 2, abs=1e-3)

    def test_explicit_pyart_output_from_datatree_input(
        self, early_pyart, late_pyart, tmp_path
    ):
        interpolated = as_datatree(early_pyart, tmp_path, "interp.nc")
        result = retime_interpolated_volume(
            interpolated, early_pyart, late_pyart, 0.5, output_flavor="pyart"
        )
        assert isinstance(result, pyart.core.Radar)

    def test_explicit_xradar_output_from_pyart_input(self, early_pyart, late_pyart):
        import copy

        result = retime_interpolated_volume(
            copy.deepcopy(early_pyart),
            early_pyart,
            late_pyart,
            0.5,
            output_flavor="xradar",
        )
        assert isinstance(result, xr.DataTree)

    def test_explicit_output_preserves_the_corrected_time(
        self, early_pyart, late_pyart
    ):
        """Converting flavour must not lose the correction it was called for."""
        import copy

        result = retime_interpolated_volume(
            copy.deepcopy(early_pyart),
            early_pyart,
            late_pyart,
            0.75,
            output_flavor="xradar",
        )
        advance = (
            reference_time_of(result) - reference_time_of(early_pyart)
        ).total_seconds()
        assert advance == pytest.approx(0.75 * SECONDS_BETWEEN_VOLUMES, abs=1e-3)

    def test_rejects_an_unknown_output_flavor(self, early_pyart, late_pyart):
        import copy

        with pytest.raises(ValueError, match="unknown"):
            retime_interpolated_volume(
                copy.deepcopy(early_pyart),
                early_pyart,
                late_pyart,
                0.5,
                output_flavor="grib",
            )


class TestGridVolumeFlavors:
    """``grid_volume`` maps Py-ART input to a Grid and xradar input to a Dataset."""

    def test_pyart_radar_gives_a_pyart_grid_by_default(self, early_pyart):
        result = grid_volume(
            early_pyart, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS
        )
        assert isinstance(result, pyart.core.Grid)

    def test_datatree_gives_an_xarray_dataset_by_default(self, early_pyart, tmp_path):
        volume = as_datatree(early_pyart, tmp_path, "vol.nc")
        result = grid_volume(volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)
        assert isinstance(result, xr.Dataset)

    def test_dataset_output_carries_the_gridded_field(self, early_pyart, tmp_path):
        volume = as_datatree(early_pyart, tmp_path, "vol.nc")
        result = grid_volume(volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)
        assert "reflectivity" in result.data_vars

    def test_dataset_output_has_the_requested_shape(self, early_pyart, tmp_path):
        volume = as_datatree(early_pyart, tmp_path, "vol.nc")
        result = grid_volume(volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)
        assert (result.sizes["z"], result.sizes["y"], result.sizes["x"]) == GRID_SHAPE

    def test_pyart_grid_output_has_the_requested_shape(self, early_pyart):
        result = grid_volume(
            early_pyart, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS
        )
        assert result.fields["reflectivity"]["data"].shape == GRID_SHAPE

    def test_explicit_xarray_output_from_pyart_input(self, early_pyart):
        result = grid_volume(
            early_pyart,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            output_flavor="xarray",
        )
        assert isinstance(result, xr.Dataset)

    def test_explicit_pyart_output_from_datatree_input(self, early_pyart, tmp_path):
        volume = as_datatree(early_pyart, tmp_path, "vol.nc")
        result = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            output_flavor="pyart",
        )
        assert detect_grid_flavor(result) is not None
        assert hasattr(result, "fields")

    def test_both_flavors_grid_to_equivalent_values(self, early_pyart, tmp_path):
        """Flavour must not change the numbers, only their container."""
        volume = as_datatree(early_pyart, tmp_path, "vol.nc")
        from_pyart = grid_volume(
            early_pyart, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS
        )
        from_xradar = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            output_flavor="pyart",
        )
        np.testing.assert_allclose(
            np.ma.filled(from_pyart.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(from_xradar.fields["reflectivity"]["data"], np.nan),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )

    def test_multiple_volumes_are_accepted(self, early_pyart, late_pyart):
        result = grid_volume(
            (early_pyart, late_pyart),
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
        )
        assert isinstance(result, pyart.core.Grid)

    def test_mixed_flavor_volume_sequence_uses_the_first_for_the_default(
        self, early_pyart, late_pyart, tmp_path
    ):
        late_as_tree = as_datatree(late_pyart, tmp_path, "late.nc")
        result = grid_volume(
            (early_pyart, late_as_tree),
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
        )
        assert isinstance(result, pyart.core.Grid)

    def test_rejects_an_unknown_output_flavor(self, early_pyart):
        with pytest.raises(ValueError, match="unknown"):
            grid_volume(
                early_pyart,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                output_flavor="zarr",
            )

    def test_rejects_an_empty_volume_sequence(self):
        with pytest.raises(ValueError, match="at least one"):
            grid_volume((), grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)
