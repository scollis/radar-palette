"""Unit tests for the radar/grid object-flavour interoperability layer.

Two object families describe the same radar data: Py-ART's ``Radar``/``Grid`` and
xradar's ``DataTree``/xarray ``Dataset``. Operators in this package work on the
Py-ART surface internally, so this layer's job is to accept either family, and to
return the family the caller expects -- by default, the one they passed in.

The tests below fix that policy and the round-trip behaviour. They deliberately
assert on *observed* library behaviour rather than on assumptions: several
conversions have sharp edges (see ``test_writes_time_back_per_sweep`` and
``test_xgrid_requires_undecoded_times``) that a naive implementation gets wrong
silently.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")
xradar = pytest.importorskip("xradar")
xr = pytest.importorskip("xarray")

from radar_palette.io import (  # noqa: E402
    GridFlavor,
    RadarFlavor,
    detect_grid_flavor,
    detect_radar_flavor,
    resolve_output_flavor,
    to_grid_flavor,
    to_pyart_grid,
    to_pyart_radar,
    to_radar_flavor,
    write_ray_times_to_datatree,
)
from radar_palette.testing import assign_scan_times, make_empty_ppi_volume  # noqa: E402

VOLUME_START = datetime(2011, 5, 20, 11, 34, 0, tzinfo=UTC)


def add_constant_field(radar, field_name="reflectivity", value=0.0):
    """Give a volume one uniform field, so it can be written to cfradial."""
    radar.add_field(
        field_name,
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


@pytest.fixture
def pyart_volume():
    """A native Py-ART volume with known acquisition times and one field."""
    return add_constant_field(
        assign_scan_times(make_empty_ppi_volume(nsweeps=3), VOLUME_START)
    )


@pytest.fixture
def datatree_volume(pyart_volume, tmp_path):
    """The same volume as an xradar DataTree, via a cfradial round trip."""
    path = str(tmp_path / "volume.nc")
    pyart.io.write_cfradial(path, pyart_volume)
    return xradar.io.open_cfradial1_datatree(path)


@pytest.fixture
def pyart_grid(pyart_volume):
    """A small Py-ART Grid, cheap enough to build per test."""
    return pyart.map.grid_from_radars(
        (pyart_volume,),
        grid_shape=(3, 11, 11),
        grid_limits=((0.0, 2000.0), (-5000.0, 5000.0), (-5000.0, 5000.0)),
    )


class TestDetectRadarFlavor:
    def test_identifies_a_pyart_radar(self, pyart_volume):
        assert detect_radar_flavor(pyart_volume) is RadarFlavor.PYART

    def test_identifies_an_xradar_datatree(self, datatree_volume):
        assert detect_radar_flavor(datatree_volume) is RadarFlavor.XRADAR

    def test_identifies_the_pyart_xradar_wrapper_as_xradar(self, datatree_volume):
        """``Xradar`` presents a Py-ART surface but its home format is xradar."""
        wrapper = datatree_volume.pyart.to_radar()
        assert detect_radar_flavor(wrapper) is RadarFlavor.XRADAR

    def test_rejects_an_unrelated_object(self):
        with pytest.raises(TypeError, match="Radar"):
            detect_radar_flavor(object())

    def test_rejects_a_grid(self, pyart_grid):
        """A Grid is not a radar volume; the error should say so rather than guess."""
        with pytest.raises(TypeError, match="Radar"):
            detect_radar_flavor(pyart_grid)


class TestDetectGridFlavor:
    def test_identifies_a_pyart_grid(self, pyart_grid):
        assert detect_grid_flavor(pyart_grid) is GridFlavor.PYART

    def test_identifies_an_xarray_dataset(self, pyart_grid):
        assert detect_grid_flavor(pyart_grid.to_xarray()) is GridFlavor.XARRAY

    def test_rejects_an_unrelated_object(self):
        with pytest.raises(TypeError, match="Grid"):
            detect_grid_flavor(object())


class TestResolveOutputFlavor:
    """The default output flavour mirrors the input; an explicit request wins."""

    @pytest.mark.parametrize("flavor", list(RadarFlavor))
    def test_none_mirrors_the_input_flavor(self, flavor):
        assert resolve_output_flavor(None, flavor) is flavor

    @pytest.mark.parametrize("flavor", list(GridFlavor))
    def test_none_mirrors_the_input_grid_flavor(self, flavor):
        assert resolve_output_flavor(None, flavor) is flavor

    def test_explicit_request_overrides_the_input(self):
        assert (
            resolve_output_flavor(RadarFlavor.XRADAR, RadarFlavor.PYART)
            is RadarFlavor.XRADAR
        )

    def test_accepts_a_string_alias(self):
        assert resolve_output_flavor("pyart", RadarFlavor.XRADAR) is RadarFlavor.PYART

    def test_string_alias_is_case_insensitive(self):
        assert resolve_output_flavor("XRadar", RadarFlavor.PYART) is RadarFlavor.XRADAR

    def test_accepts_xarray_as_an_alias_for_the_grid_flavor(self):
        assert resolve_output_flavor("xarray", GridFlavor.PYART) is GridFlavor.XARRAY

    def test_rejects_an_unknown_alias(self):
        with pytest.raises(ValueError, match="unknown"):
            resolve_output_flavor("netcdf", RadarFlavor.PYART)

    def test_rejects_a_flavor_from_the_wrong_family(self):
        """Asking for a grid flavour on a radar conversion is a programming error."""
        with pytest.raises(ValueError, match="expected"):
            resolve_output_flavor(GridFlavor.XARRAY, RadarFlavor.PYART)


class TestToPyartRadar:
    """Every input family must present a Py-ART ``Radar`` surface to operators."""

    def test_passes_a_native_radar_through_unchanged(self, pyart_volume):
        converted, flavor = to_pyart_radar(pyart_volume)
        assert converted is pyart_volume
        assert flavor is RadarFlavor.PYART

    def test_converts_a_datatree_to_a_pyart_surface(self, datatree_volume):
        converted, flavor = to_pyart_radar(datatree_volume)
        assert flavor is RadarFlavor.XRADAR
        for required_attribute in ("nrays", "ngates", "nsweeps", "fields", "time"):
            assert hasattr(converted, required_attribute)

    def test_converted_surface_reports_the_same_geometry(
        self, pyart_volume, datatree_volume
    ):
        converted, _ = to_pyart_radar(datatree_volume)
        assert converted.nrays == pyart_volume.nrays
        assert converted.ngates == pyart_volume.ngates
        assert converted.nsweeps == pyart_volume.nsweeps

    def test_converted_surface_preserves_field_values(
        self, pyart_volume, datatree_volume
    ):
        converted, _ = to_pyart_radar(datatree_volume)
        np.testing.assert_allclose(
            np.ma.filled(converted.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(pyart_volume.fields["reflectivity"]["data"], np.nan),
            atol=1e-5,
        )

    def test_converted_surface_is_accepted_by_pyart_gridding(self, datatree_volume):
        """The wrapper must be usable by Py-ART operators, not merely look like one."""
        converted, _ = to_pyart_radar(datatree_volume)
        grid = pyart.map.grid_from_radars(
            (converted,),
            grid_shape=(3, 11, 11),
            grid_limits=((0.0, 2000.0), (-5000.0, 5000.0), (-5000.0, 5000.0)),
        )
        assert "reflectivity" in grid.fields


class TestToRadarFlavor:
    def test_returns_a_pyart_radar_unchanged_for_the_pyart_flavor(self, pyart_volume):
        assert to_radar_flavor(pyart_volume, RadarFlavor.PYART) is pyart_volume

    def test_converts_a_pyart_radar_to_a_datatree(self, pyart_volume):
        converted = to_radar_flavor(pyart_volume, RadarFlavor.XRADAR)
        assert isinstance(converted, xr.DataTree)

    def test_datatree_output_has_one_group_per_sweep(self, pyart_volume):
        converted = to_radar_flavor(pyart_volume, RadarFlavor.XRADAR)
        sweep_groups = [name for name in converted.groups if name.startswith("/sweep_")]
        assert len(sweep_groups) == pyart_volume.nsweeps

    def test_datatree_output_preserves_field_values(self, pyart_volume):
        converted = to_radar_flavor(pyart_volume, RadarFlavor.XRADAR)
        first_sweep_values = converted["sweep_0"].ds["reflectivity"].values
        expected = np.ma.filled(
            pyart_volume.fields["reflectivity"]["data"][: first_sweep_values.shape[0]],
            np.nan,
        )
        np.testing.assert_allclose(first_sweep_values, expected, atol=1e-5)

    def test_round_trip_through_xradar_preserves_geometry(self, pyart_volume):
        as_datatree = to_radar_flavor(pyart_volume, RadarFlavor.XRADAR)
        restored, _ = to_pyart_radar(as_datatree)
        assert restored.nrays == pyart_volume.nrays
        assert restored.nsweeps == pyart_volume.nsweeps

    def test_round_trip_through_xradar_preserves_reference_time(self, pyart_volume):
        from radar_palette.advection import volume_reference_time

        as_datatree = to_radar_flavor(pyart_volume, RadarFlavor.XRADAR)
        restored, _ = to_pyart_radar(as_datatree)
        drift = (
            volume_reference_time(restored) - volume_reference_time(pyart_volume)
        ).total_seconds()
        assert abs(drift) < 1e-3

    def test_datatree_input_returned_as_datatree_is_not_reconverted(
        self, datatree_volume
    ):
        """A DataTree asked for as a DataTree should come back as one."""
        converted, _ = to_pyart_radar(datatree_volume)
        result = to_radar_flavor(converted, RadarFlavor.XRADAR)
        assert isinstance(result, xr.DataTree)

    def test_rejects_an_unknown_flavor(self, pyart_volume):
        with pytest.raises(ValueError, match="unknown"):
            to_radar_flavor(pyart_volume, "hdf5")


class TestWriteRayTimesToDatatree:
    """Time must be written back per sweep, because ``Xradar.time`` is read-only.

    Assigning ``wrapper.time`` mutates only the wrapper, silently leaving the
    underlying DataTree on its original clock -- the same class of failure as
    inheriting a bracketing volume's time. This is the guard against it.
    """

    def test_writes_time_back_per_sweep(self, pyart_volume, datatree_volume):
        shift_seconds = 210.0
        wrapper, _ = to_pyart_radar(datatree_volume)
        shifted_offsets = (
            np.asarray(wrapper.time["data"], dtype="float64") + shift_seconds
        )
        original_first_ray = datatree_volume["sweep_0"].ds["time"].values[0]

        write_ray_times_to_datatree(
            datatree_volume,
            {
                "data": shifted_offsets,
                "units": wrapper.time["units"],
                "calendar": "gregorian",
            },
        )
        updated_first_ray = datatree_volume["sweep_0"].ds["time"].values[0]
        elapsed = (updated_first_ray - original_first_ray) / np.timedelta64(1, "s")
        assert elapsed == pytest.approx(shift_seconds, abs=1e-3)

    def test_shifts_every_sweep_not_only_the_first(self, datatree_volume):
        shift_seconds = 60.0
        wrapper, _ = to_pyart_radar(datatree_volume)
        original_first_rays = [
            datatree_volume[f"sweep_{index}"].ds["time"].values[0] for index in range(3)
        ]
        write_ray_times_to_datatree(
            datatree_volume,
            {
                "data": np.asarray(wrapper.time["data"], dtype="float64")
                + shift_seconds,
                "units": wrapper.time["units"],
                "calendar": "gregorian",
            },
        )
        for index, original in enumerate(original_first_rays):
            updated = datatree_volume[f"sweep_{index}"].ds["time"].values[0]
            elapsed = (updated - original) / np.timedelta64(1, "s")
            assert elapsed == pytest.approx(shift_seconds, abs=1e-3), f"sweep_{index}"

    def test_preserves_the_intra_sweep_ray_pattern(self, datatree_volume):
        wrapper, _ = to_pyart_radar(datatree_volume)
        original_spacing = np.diff(datatree_volume["sweep_0"].ds["time"].values)
        write_ray_times_to_datatree(
            datatree_volume,
            {
                "data": np.asarray(wrapper.time["data"], dtype="float64") + 30.0,
                "units": wrapper.time["units"],
                "calendar": "gregorian",
            },
        )
        updated_spacing = np.diff(datatree_volume["sweep_0"].ds["time"].values)
        np.testing.assert_array_equal(updated_spacing, original_spacing)

    def test_keeps_time_as_a_datetime_coordinate(self, datatree_volume):
        wrapper, _ = to_pyart_radar(datatree_volume)
        write_ray_times_to_datatree(
            datatree_volume,
            {
                "data": np.asarray(wrapper.time["data"], dtype="float64"),
                "units": wrapper.time["units"],
                "calendar": "gregorian",
            },
        )
        time_variable = datatree_volume["sweep_0"].ds["time"]
        assert np.issubdtype(time_variable.dtype, np.datetime64)
        assert "time" in datatree_volume["sweep_0"].ds.coords

    def test_rejects_a_ray_count_mismatch(self, datatree_volume):
        with pytest.raises(ValueError, match="ray"):
            write_ray_times_to_datatree(
                datatree_volume,
                {
                    "data": np.zeros(7, dtype="float64"),
                    "units": "seconds since 2011-05-20T11:34:00Z",
                    "calendar": "gregorian",
                },
            )


class TestGridConversions:
    def test_pyart_grid_passes_through_for_the_pyart_flavor(self, pyart_grid):
        assert to_grid_flavor(pyart_grid, GridFlavor.PYART) is pyart_grid

    def test_converts_a_pyart_grid_to_a_dataset(self, pyart_grid):
        converted = to_grid_flavor(pyart_grid, GridFlavor.XARRAY)
        assert isinstance(converted, xr.Dataset)

    def test_dataset_output_preserves_field_values(self, pyart_grid):
        converted = to_grid_flavor(pyart_grid, GridFlavor.XARRAY)
        np.testing.assert_allclose(
            np.ma.filled(converted["reflectivity"].values[0], np.nan),
            np.ma.filled(pyart_grid.fields["reflectivity"]["data"], np.nan),
            atol=1e-5,
        )

    def test_dataset_output_carries_the_spatial_dimensions(self, pyart_grid):
        converted = to_grid_flavor(pyart_grid, GridFlavor.XARRAY)
        for expected_dimension in ("z", "y", "x"):
            assert expected_dimension in converted.sizes

    def test_converts_a_dataset_back_to_a_pyart_grid_surface(self, pyart_grid):
        as_dataset = to_grid_flavor(pyart_grid, GridFlavor.XARRAY)
        restored = to_pyart_grid(as_dataset)
        assert restored.nx == pyart_grid.nx
        assert restored.ny == pyart_grid.ny
        assert restored.nz == pyart_grid.nz

    def test_xgrid_requires_undecoded_times(self, pyart_grid):
        """Py-ART's Xgrid raises unless time is left encoded; we must not trip it.

        This documents a real sharp edge: ``Grid.to_xarray`` produces decoded
        datetimes, so a direct ``Xgrid(dataset)`` fails. The conversion has to
        re-encode, and this test fails if that handling is removed.
        """
        as_dataset = to_grid_flavor(pyart_grid, GridFlavor.XARRAY)
        restored = to_pyart_grid(as_dataset)
        assert "reflectivity" in restored.fields

    def test_round_trip_preserves_field_values(self, pyart_grid):
        as_dataset = to_grid_flavor(pyart_grid, GridFlavor.XARRAY)
        restored = to_pyart_grid(as_dataset)
        np.testing.assert_allclose(
            np.ma.filled(restored.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(pyart_grid.fields["reflectivity"]["data"], np.nan),
            atol=1e-5,
        )

    def test_rejects_an_unknown_grid_flavor(self, pyart_grid):
        with pytest.raises(ValueError, match="unknown"):
            to_grid_flavor(pyart_grid, "zarr")


class TestDefaultPolicyEndToEnd:
    """The headline contract: output family mirrors input family by default."""

    def test_pyart_radar_in_pyart_radar_out(self, pyart_volume):
        radar, input_flavor = to_pyart_radar(pyart_volume)
        result = to_radar_flavor(radar, resolve_output_flavor(None, input_flavor))
        assert detect_radar_flavor(result) is RadarFlavor.PYART

    def test_datatree_in_datatree_out(self, datatree_volume):
        radar, input_flavor = to_pyart_radar(datatree_volume)
        result = to_radar_flavor(radar, resolve_output_flavor(None, input_flavor))
        assert isinstance(result, xr.DataTree)

    def test_datatree_in_pyart_out_when_requested(self, datatree_volume):
        radar, input_flavor = to_pyart_radar(datatree_volume)
        result = to_radar_flavor(radar, resolve_output_flavor("pyart", input_flavor))
        assert detect_radar_flavor(result) is RadarFlavor.PYART

    def test_pyart_radar_in_datatree_out_when_requested(self, pyart_volume):
        radar, input_flavor = to_pyart_radar(pyart_volume)
        result = to_radar_flavor(radar, resolve_output_flavor("xradar", input_flavor))
        assert isinstance(result, xr.DataTree)

    def test_pyart_grid_in_pyart_grid_out(self, pyart_grid):
        input_flavor = detect_grid_flavor(pyart_grid)
        result = to_grid_flavor(pyart_grid, resolve_output_flavor(None, input_flavor))
        assert detect_grid_flavor(result) is GridFlavor.PYART

    def test_dataset_in_dataset_out(self, pyart_grid):
        as_dataset = to_grid_flavor(pyart_grid, GridFlavor.XARRAY)
        input_flavor = detect_grid_flavor(as_dataset)
        result = to_grid_flavor(as_dataset, resolve_output_flavor(None, input_flavor))
        assert isinstance(result, xr.Dataset)
