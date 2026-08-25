"""Tests for method dispatch on the public gridding entry point.

:func:`radar_palette.gridding.grid_volume` now has two backings: Py-ART's Cressman
gridder and this package's spectral operator. The tests are grouped by the thing that
can independently be wrong:

``TestDefaultIsUnchanged``
    That adding the option did not change what existing callers get. This is the
    reason the default is ``"pyart"`` rather than ``"spectral"``.
``TestSpectralPath``
    That the spectral backing actually runs end to end and produces a grid of the
    right shape, in either object family.
``TestSpectralRefusals``
    That the spectral path fails loudly where it is undefined, rather than quietly
    producing something plausible.
``TestMethodsDiffer``
    That the two backings are genuinely different operators --- if they agreed
    everywhere, one of them would not be doing what it claims.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from radar_palette.gridding import (  # noqa: E402
    DEFAULT_GRIDDING_METHOD,
    GRIDDING_METHODS,
    VerticalFlag,
    build_cones,
    census_radar,
    dedup_sweeps,
    grid_volume,
    resolve_tilt_workers,
)
from radar_palette.gridding.cones import (  # noqa: E402
    TiltReport,
    UnphysicalGridWarning,
    _check_physical,
)

GRID_SHAPE = (4, 41, 41)
GRID_LIMITS = ((500.0, 6000.0), (-50_000.0, 50_000.0), (-50_000.0, 50_000.0))


def make_volume(fixed_angles=(0.5, 1.0, 1.5, 2.5, 4.0), nrays=240, ngates=100):
    """A volume whose reflectivity is a smooth function of gate height."""
    fixed_angles = np.asarray(fixed_angles, dtype="float64")
    radar = pyart.testing.make_empty_ppi_radar(ngates, nrays, fixed_angles.size)
    radar.range["data"] = 1000.0 + 1000.0 * np.arange(ngates, dtype="float64")
    azimuths = (360.0 / nrays) * np.arange(nrays, dtype="float64")
    radar.azimuth["data"] = np.tile(azimuths, fixed_angles.size)
    radar.elevation["data"] = np.repeat(fixed_angles, nrays)
    radar.fixed_angle["data"] = fixed_angles
    radar.time["data"] = 0.5 * np.arange(radar.nrays, dtype="float64")
    radar.time["units"] = "seconds since 2011-05-20T11:27:34Z"
    values = 30.0 - 3.0e-3 * radar.gate_z["data"]
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
    return make_volume()


class TestMethodRegistry:
    def test_the_default_is_a_known_method(self):
        assert DEFAULT_GRIDDING_METHOD in GRIDDING_METHODS

    def test_both_backings_are_registered(self):
        assert set(GRIDDING_METHODS) == {"pyart", "spectral"}

    def test_the_default_preserves_previous_behaviour(self):
        """Deliberate: adding the spectral operator must not silently change results.

        The spectral path is opt-in because it is a different operator, not a
        strictly better one --- it resolves sharp gradients that Cressman weighting
        smooths away, at the cost of ringing at hard echo edges.
        """
        assert DEFAULT_GRIDDING_METHOD == "pyart"

    def test_rejects_an_unknown_method(self, volume):
        with pytest.raises(ValueError, match="unknown gridding method"):
            grid_volume(
                volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="magic"
            )


class TestDefaultIsUnchanged:
    def test_default_matches_an_explicit_pyart_request(self, volume):
        implicit = grid_volume(volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)
        explicit = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
        )
        np.testing.assert_allclose(
            np.ma.filled(implicit.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(explicit.fields["reflectivity"]["data"], np.nan),
            equal_nan=True,
        )

    def test_default_matches_calling_pyart_directly(self, volume):
        """The entry point must add nothing of its own on the default path."""
        through_entry_point = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS
        )
        direct = pyart.map.grid_from_radars(
            (volume,), grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS
        )
        np.testing.assert_allclose(
            np.ma.filled(through_entry_point.fields["reflectivity"]["data"], np.nan),
            np.ma.filled(direct.fields["reflectivity"]["data"], np.nan),
            equal_nan=True,
        )


class TestSpectralPath:
    def test_returns_a_grid_of_the_requested_shape(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert grid.fields["reflectivity"]["data"].shape == GRID_SHAPE

    def test_axes_span_the_requested_limits(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        for axis, (lower, upper) in zip(
            (grid.z, grid.y, grid.x), GRID_LIMITS, strict=True
        ):
            assert axis["data"][0] == pytest.approx(lower)
            assert axis["data"][-1] == pytest.approx(upper)

    def test_carries_the_coverage_flag_as_a_field(self, volume):
        """The flag is the honest part of the product and must survive to the output.

        A spectral grid cell is NaN wherever the volume did not observe it, and the
        flag says which of the several reasons applies. Dropping it would leave a
        user unable to tell a data gap from a coverage limit.
        """
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert "coverage_flag" in grid.fields
        flags = np.ma.filled(grid.fields["coverage_flag"]["data"], 255).astype(int)
        assert set(np.unique(flags)) <= {int(f) for f in VerticalFlag} | {255}

    def test_finite_values_coincide_with_the_interpolated_flag(self, volume):
        """The same invariant the vertical layer guarantees, preserved end to end."""
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"], np.nan)
        flags = np.ma.filled(grid.fields["coverage_flag"]["data"], 255).astype(int)
        np.testing.assert_array_equal(
            np.isfinite(values), flags == int(VerticalFlag.INTERPOLATED)
        )

    def test_values_track_the_height_dependent_input(self, volume):
        """The test field falls 3 dB per km, so the grid must too."""
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"], np.nan)
        heights_m = grid.z["data"]
        level_means = [np.nanmean(values[k]) for k in range(values.shape[0])]
        finite = [m for m in level_means if np.isfinite(m)]
        assert len(finite) >= 2
        assert finite[0] > finite[-1]
        expected_drop = 3.0e-3 * (heights_m[-1] - heights_m[0])
        assert abs(finite[0] - finite[-1]) < 3.0 * expected_drop

    def test_accepts_a_vertical_scheme(self, volume):
        grid = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
            scheme="linear_z",
        )
        assert grid.fields["reflectivity"]["data"].shape == GRID_SHAPE

    def test_preserves_the_grid_origin_of_the_input(self, volume):
        spectral = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        cressman = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
        )
        assert spectral.origin_latitude["data"] == pytest.approx(
            cressman.origin_latitude["data"]
        )
        assert spectral.origin_longitude["data"] == pytest.approx(
            cressman.origin_longitude["data"]
        )

    def test_records_the_method_in_the_metadata(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert "spectral" in grid.metadata.get("history", "")

    def test_returns_an_xarray_dataset_on_request(self, volume):
        import xarray as xr

        result = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
            output_flavor="xarray",
        )
        assert isinstance(result, xr.Dataset)

    def test_flavour_default_still_mirrors_the_input(self, volume):
        grid = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="spectral"
        )
        assert isinstance(grid, pyart.core.Grid)


class TestSpectralRefusals:
    """Where the spectral path is undefined it must say so, not improvise."""

    def test_refuses_a_single_tilt_volume(self):
        """Vertical interpolation needs two cones to bracket a target.

        A single sweep cannot produce a 3-D field by this method at all, so the
        failure belongs at the entry point rather than as an all-NaN grid.
        """
        single = make_volume(fixed_angles=(0.5,))
        with pytest.raises(ValueError, match="at least two"):
            grid_volume(
                single,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            )

    def test_refuses_multiple_volumes(self, volume):
        """Merging several volumes spectrally is not defined by this operator.

        The Cressman path accepts a sequence because its weighting composes across
        radars. The spectral path works per sweep of one volume, and inventing a
        merge rule here would be a research decision disguised as plumbing.
        """
        with pytest.raises(NotImplementedError, match="single volume"):
            grid_volume(
                (volume, volume),
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            )

    def test_refuses_an_unknown_vertical_scheme(self, volume):
        with pytest.raises(ValueError, match="unknown"):
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
                scheme="quintic",
            )

    def test_still_refuses_an_empty_sequence(self, volume):
        with pytest.raises(ValueError, match="at least one"):
            grid_volume((), grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS)


class TestMethodsDiffer:
    def test_the_two_backings_do_not_agree_everywhere(self, volume):
        """If they matched, one would not be doing what it claims.

        Cressman weighting averages neighbouring gates into each cell; the spectral
        operator evaluates a band-limited interpolant. They answer differently by
        construction, and this test exists so that a mis-wired dispatch --- silently
        running Py-ART for both --- cannot pass.
        """
        spectral = np.ma.filled(
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        cressman = np.ma.filled(
            grid_volume(
                volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        both_finite = np.isfinite(spectral) & np.isfinite(cressman)
        assert both_finite.sum() > 100
        assert not np.allclose(spectral[both_finite], cressman[both_finite])

    def test_but_they_broadly_agree_where_both_have_data(self, volume):
        """Different operators, same physics: the two must not disagree wildly.

        A large disagreement here would mean one of them is wrong, not merely
        different, so this is the sanity bound that makes the previous test
        meaningful rather than a licence for anything.
        """
        spectral = np.ma.filled(
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        cressman = np.ma.filled(
            grid_volume(
                volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method="pyart"
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        both_finite = np.isfinite(spectral) & np.isfinite(cressman)
        assert np.nanmedian(np.abs(spectral[both_finite] - cressman[both_finite])) < 5.0


class TestWhyTheDefaultIsPyart:
    """The measured trade-off behind ``DEFAULT_GRIDDING_METHOD``.

    Neither operator dominates, which is the whole reason the older one stays the
    default. On a hard azimuthal echo edge (a 10 to 50 dBZ wedge) the spectral
    interpolant overshoots to 55.5 dBZ and undershoots to 4.7 --- about +5.5 and -5.3
    dB beyond the data --- while Cressman weighting stays exactly inside [10, 50]
    because averaging cannot exceed its inputs.

    Switching the default would hand every existing caller that ringing in exchange
    for sharpness they did not ask for. These tests make the trade-off a fact in the
    suite rather than a claim in a docstring.
    """

    WEDGE_LOW_DBZ = 10.0
    WEDGE_HIGH_DBZ = 50.0
    EDGE_GRID_SHAPE = (3, 81, 81)
    EDGE_GRID_LIMITS = ((500.0, 4000.0), (-40_000.0, 40_000.0), (-40_000.0, 40_000.0))

    @pytest.fixture(scope="class")
    def hard_edge_volume(self):
        """A wedge of strong echo with hard edges in azimuth."""
        radar = make_volume(nrays=360, ngates=100)
        azimuth_of_gate = (
            np.rad2deg(np.arctan2(radar.gate_x["data"], radar.gate_y["data"])) % 360.0
        )
        in_wedge = (azimuth_of_gate > 60.0) & (azimuth_of_gate < 120.0)
        values = np.where(in_wedge, self.WEDGE_HIGH_DBZ, self.WEDGE_LOW_DBZ)
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

    def gridded(self, radar, method):
        grid = grid_volume(
            radar,
            grid_shape=self.EDGE_GRID_SHAPE,
            grid_limits=self.EDGE_GRID_LIMITS,
            method=method,
        )
        return np.ma.filled(grid.fields["reflectivity"]["data"], np.nan)

    def test_the_spectral_path_rings_at_a_hard_edge(self, hard_edge_volume):
        values = self.gridded(hard_edge_volume, "spectral")
        assert np.nanmax(values) > self.WEDGE_HIGH_DBZ + 1.0
        assert np.nanmin(values) < self.WEDGE_LOW_DBZ - 1.0

    # Fields are stored float32, so an exact bound would test the storage dtype
    # rather than the operator: distance weighting overshoots by ~8e-6 dB on this
    # scene purely from rounding. A 0.01 dB tolerance is far below the ~5 dB ringing
    # being distinguished, so the test still separates the two operators cleanly.
    ROUNDING_TOLERANCE_DB = 0.01

    def test_the_default_path_does_not_ring(self, hard_edge_volume):
        """Distance weighting is an average, so it cannot exceed its inputs."""
        values = self.gridded(hard_edge_volume, "pyart")
        assert np.nanmax(values) <= self.WEDGE_HIGH_DBZ + self.ROUNDING_TOLERANCE_DB
        assert np.nanmin(values) >= self.WEDGE_LOW_DBZ - self.ROUNDING_TOLERANCE_DB

    def test_the_default_is_the_non_ringing_one(self, hard_edge_volume):
        """Ties the constant to the measurement rather than to a comment.

        If someone flips DEFAULT_GRIDDING_METHOD to "spectral", this fails and says
        why: the default would then overshoot the data range on a hard edge.
        """
        values = self.gridded(hard_edge_volume, DEFAULT_GRIDDING_METHOD)
        assert np.nanmax(values) <= self.WEDGE_HIGH_DBZ + self.ROUNDING_TOLERANCE_DB
        assert np.nanmin(values) >= self.WEDGE_LOW_DBZ - self.ROUNDING_TOLERANCE_DB


class TestDistanceWeightingLeavesInterConeGaps:
    """The other half of the trade-off, and the half that is easy to miss.

    :class:`TestWhyTheDefaultIsPyart` measures what the spectral path costs: it rings
    at a discontinuity. This class measures what the *default* path costs, which the
    module docstring described only as degrading "gracefully".

    It does not degrade gracefully. A radius of influence is a fixed length, while the
    vertical separation between adjacent tilts grows with range. Wherever the
    separation exceeds the radius no gate is within reach, and the cell is left empty
    --- an interior hole, with valid data both above and below it in the same column.

    This is not a mis-set parameter: Py-ART's own defaults for ``roi_func="dist_beam"``
    produce it, which is why it earns a test rather than a comment.
    """

    GRID_SHAPE = (24, 41, 41)
    GRID_LIMITS = ((500.0, 12_000.0), (-30_000.0, 30_000.0), (-30_000.0, 30_000.0))

    @staticmethod
    def _interior_holes(section):
        """Count empty cells that have data both above and below in the same column.

        A gap at the top or bottom of a column is a coverage limit and is expected. A
        gap *between* two valid samples is the operator failing to reach.
        """
        total = 0
        for column in range(section.shape[1]):
            valid = np.isfinite(section[:, column])
            if valid.sum() < 2:
                continue
            first = int(np.argmax(valid))
            last = len(valid) - 1 - int(np.argmax(valid[::-1]))
            total += int((~valid[first : last + 1]).sum())
        return total

    @pytest.fixture(scope="class")
    @classmethod
    def wide_gap_volume(cls):
        """Tilts spaced so the cone separation outruns a fixed radius."""
        return make_volume(fixed_angles=(0.5, 4.0, 10.0, 20.0), nrays=180, ngates=120)

    def _grid(self, volume, **roi_kwargs):
        grid = grid_volume(
            volume,
            grid_shape=self.GRID_SHAPE,
            grid_limits=self.GRID_LIMITS,
            method="pyart",
            fields=["reflectivity"],
            weighting_function="Barnes2",
            **roi_kwargs,
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"].astype(float), np.nan)
        return values

    def _mid_section(self, values):
        return values[:, values.shape[1] // 2, :]

    def test_beam_width_roi_leaves_interior_holes(self, wide_gap_volume):
        """The documented default leaves holes between cones."""
        section = self._mid_section(self._grid(wide_gap_volume, roi_func="dist_beam"))
        assert self._interior_holes(section) > 0

    def test_pyart_defaults_leave_them_too(self, wide_gap_volume):
        """Not a mis-set parameter: passing no ROI arguments does the same thing.

        Pinned because the natural reading of a hole-riddled section is that the
        caller chose badly. Py-ART's own defaults produce it, so the honest statement
        is about the operator, not about the arguments.
        """
        explicit = self._interior_holes(
            self._mid_section(self._grid(wide_gap_volume, roi_func="dist_beam"))
        )
        implicit = self._interior_holes(self._mid_section(self._grid(wide_gap_volume)))
        assert implicit > 0
        assert implicit == explicit

    def test_a_wider_radius_closes_them(self, wide_gap_volume):
        """A radius large enough to span the cone separation fills the gaps."""
        narrow = self._interior_holes(
            self._mid_section(self._grid(wide_gap_volume, roi_func="dist_beam"))
        )
        wide = self._interior_holes(
            self._mid_section(
                self._grid(wide_gap_volume, roi_func="constant", constant_roi=4000.0)
            )
        )
        assert wide < narrow

    def test_closing_the_holes_costs_peak_intensity(self, wide_gap_volume):
        """The trade-off itself, asserted so it cannot be quietly forgotten.

        Widening the radius averages over a larger neighbourhood: it fills more cells
        *and* attenuates the peak. Both directions are asserted, because a claim that
        one setting is simply better would be wrong.
        """
        narrow = self._grid(wide_gap_volume, roi_func="dist_beam")
        wide = self._grid(wide_gap_volume, roi_func="constant", constant_roi=4000.0)
        assert np.isfinite(wide).mean() > np.isfinite(narrow).mean()
        assert np.nanmax(wide) < np.nanmax(narrow)

    def test_the_spectral_path_has_no_interior_holes(self, wide_gap_volume):
        """The contrast that makes this a trade-off rather than a defect.

        Vertical assembly interpolates between whichever cones bracket the target
        height, so a separation wider than any fixed radius is not a problem for it.
        It declines to extrapolate *beyond* the outermost cones, which is a different
        thing and is what ``coverage_flag`` reports.
        """
        grid = grid_volume(
            wide_gap_volume,
            grid_shape=self.GRID_SHAPE,
            grid_limits=self.GRID_LIMITS,
            method="spectral",
            field_name="reflectivity",
        )
        values = np.ma.filled(grid.fields["reflectivity"]["data"].astype(float), np.nan)
        assert self._interior_holes(self._mid_section(values)) == 0


class TestFlavourAndMethodCompose:
    """Both axes of the public contract are independent and must stay that way.

    Object family and gridding method are orthogonal choices, so all four
    combinations are exercised here. The xradar-plus-spectral case is the one most
    likely to break silently: it is the only path where the volume is converted
    between object families *and* gridded by this package's own operator.
    """

    @pytest.fixture(scope="class")
    def datatree(self, tmp_path_factory):
        xradar = pytest.importorskip("xradar")
        path = tmp_path_factory.mktemp("flavour") / "volume.nc"
        pyart.io.write_cfradial(str(path), make_volume())
        return xradar.io.open_cfradial1_datatree(str(path))

    @pytest.mark.parametrize("method", GRIDDING_METHODS)
    def test_a_datatree_grids_to_a_dataset(self, datatree, method):
        import xarray as xr

        result = grid_volume(
            datatree, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method=method
        )
        assert isinstance(result, xr.Dataset)
        assert "reflectivity" in result

    @pytest.mark.parametrize("method", GRIDDING_METHODS)
    def test_a_radar_grids_to_a_grid(self, volume, method):
        result = grid_volume(
            volume, grid_shape=GRID_SHAPE, grid_limits=GRID_LIMITS, method=method
        )
        assert isinstance(result, pyart.core.Grid)

    def test_the_coverage_flag_survives_conversion_to_xarray(self, volume):
        """A flag dropped in conversion would be worse than never emitting one."""
        result = grid_volume(
            volume,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
            output_flavor="xarray",
        )
        assert "coverage_flag" in result

    def test_both_families_give_the_same_spectral_answer(self, volume, datatree):
        """Converting the volume must not change the result.

        The same data arriving as a DataTree rather than a Radar is a representation
        difference, not a physical one, so any discrepancy here would be a defect in
        the conversion layer rather than in the operator.
        """
        from_radar = np.ma.filled(
            grid_volume(
                volume,
                grid_shape=GRID_SHAPE,
                grid_limits=GRID_LIMITS,
                method="spectral",
            ).fields["reflectivity"]["data"],
            np.nan,
        )
        from_tree = grid_volume(
            datatree,
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            method="spectral",
        )["reflectivity"].values
        np.testing.assert_allclose(
            from_radar, np.squeeze(from_tree), equal_nan=True, rtol=1e-5
        )


class TestTiltsCanBeGriddedConcurrently:
    """Parallelism must not change the answer, and must not swap.

    Tilts are independent, so this is the one place in the spectral path where
    hardware helps without a change of algorithm. The tests that matter are the two
    that could silently go wrong: identical output, and a worker count that respects
    memory rather than core count.
    """

    def test_parallel_and_serial_produce_identical_grids(self, volume):
        """Bit-identical, not merely close.

        The per-tilt work touches no shared state, so any difference at all would
        mean a real race rather than a tolerable reordering --- and a tolerance here
        would hide exactly that.
        """
        shape, limits = (7, 21, 21), ((500.0, 6500.0), (-20e3, 20e3), (-20e3, 20e3))
        serial = grid_volume(
            volume, grid_shape=shape, grid_limits=limits, method="spectral"
        )
        parallel = grid_volume(
            volume, grid_shape=shape, grid_limits=limits, method="spectral", n_jobs=4
        )
        left = np.ma.filled(serial.fields["reflectivity"]["data"], np.nan)
        right = np.ma.filled(parallel.fields["reflectivity"]["data"], np.nan)
        np.testing.assert_array_equal(left, right)

    def test_the_cone_stack_stays_in_tilt_order(self, volume):
        """Threads complete out of order; the stack must not.

        ``ConeStack`` rows are indexed by tilt and the vertical interpolation reads
        them as ordered by ascending elevation, so an order scrambled by completion
        time would corrupt every column without failing anything loudly. Asserted on
        ``build_cones`` rather than through ``grid_volume`` because that is where the
        invariant lives --- the grid does not carry the per-tilt reports.
        """
        geometries = dedup_sweeps(census_radar(volume))
        serial = build_cones(volume, geometries, half_width_m=20e3, spacing_m=2000.0)
        parallel = build_cones(
            volume, geometries, half_width_m=20e3, spacing_m=2000.0, n_jobs=4
        )

        expected = [float(g.fixed_angle) for g in geometries]
        assert [r.fixed_angle for r in parallel.reports] == expected
        assert [r.tilt_index for r in parallel.reports] == list(range(len(expected)))
        np.testing.assert_array_equal(parallel.fixed_angle, serial.fixed_angle)
        np.testing.assert_array_equal(
            np.nan_to_num(parallel.reflectivity), np.nan_to_num(serial.reflectivity)
        )


class TestTheWorkerCountRespectsMemoryAndThreads:
    """Two resources bind, and ignoring either is worse than staying serial.

    The thread half of this exists because it was got wrong: ``n_jobs=-1`` on a
    14-core machine drove the load average past 500 and had to be killed, since 14
    workers each started their own ~14-thread BLAS pool. A test that only checked
    the worker count would have passed throughout.
    """

    def test_the_default_is_serial_and_leaves_thread_pools_alone(self):
        """Unchanged behaviour unless a caller opts in."""
        for request in (None, 1):
            workers, threads = resolve_tilt_workers(request, 15, 85_500.0, 1000.0)
            assert workers == 1
            assert threads == (os.cpu_count() or 1)

    def test_workers_times_threads_never_exceeds_the_core_count(self):
        """The property that actually prevents the failure.

        Asserted across the whole plausible range rather than at one point, because
        the bug was a *product* being too large --- either factor alone looked fine.
        """
        cores = os.cpu_count() or 1
        for request in (2, 3, 4, 8, 16, -1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                workers, threads = resolve_tilt_workers(request, 15, 20_000.0, 2000.0)
            assert workers >= 1
            assert threads >= 1
            assert workers * threads <= cores

    def test_more_workers_means_fewer_threads_each(self):
        """The budget is split, not handed out twice."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            few = resolve_tilt_workers(2, 15, 20_000.0, 2000.0)
            many = resolve_tilt_workers(8, 15, 20_000.0, 2000.0)
        if many[0] > few[0]:
            assert many[1] <= few[1]

    def test_a_fine_lattice_caps_below_the_core_count(self):
        """The memory cap, which is the other resource.

        A 171 km domain at 100 m holds a large upsampled lattice per tilt, so
        fifteen at once would need more memory than the machine has. The cap must
        bind, and it must warn: silently running fewer workers than asked looks
        identical to the parallelism not working.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            coarse, _ = resolve_tilt_workers(-1, 15, 85_500.0, 2000.0)
        with pytest.warns(ResourceWarning, match="working memory"):
            fine, _ = resolve_tilt_workers(-1, 15, 85_500.0, 100.0)
        assert fine < coarse
        assert fine >= 1

    def test_it_never_asks_for_more_workers_than_tilts(self):
        """Six-tilt volumes do not benefit from sixty workers."""
        workers, _ = resolve_tilt_workers(-1, 3, 20_000.0, 2000.0)
        assert workers <= 3

    def test_a_nonsense_request_is_rejected_rather_than_clamped(self):
        """``n_jobs=0`` is a mistake, not a request for serial execution."""
        for bad in (0, -2, -7):
            with pytest.raises(ValueError, match="n_jobs"):
                resolve_tilt_workers(bad, 15, 20_000.0, 1000.0)


class TestUnphysicalOutputIsFlagged:
    """A gridded reflectivity of 2471 dBZ must not be returned silently.

    Measured on the ARM BNF C-SAPR2 volume this branch was validated against, the
    grid range for settings reachable through the public API:

    ======================  =====================  ==================
    setting                 grid range (dBZ)       unphysical cells
    ======================  =====================  ==================
    default, ``n_cg=12``    -61 .. 66              0 in the volume, 1 cone at 169
    ``n_cg=25``             **-1907 .. 2471**      1294
    ``n_cg=200``            **-1907 .. 2471**      1294
    ``az_ridge=1e-10``      **-1903 .. 2466**      1292
    ``az_ridge=1e-6``       -98 .. 138             26
    ``az_ridge=1e-2``       -61 .. 58              0
    ``az_ridge='auto'``     -61 .. 59              0
    ======================  =====================  ==================

    Two things that table says. ``n_cg`` is a long-standing documented argument and
    raising it *looks* like asking for a better answer, so it is the setting a user
    is most likely to reach for and least likely to suspect --- past ~12 iterations
    the solve amplifies out-of-band content instead of converging. And the guard is
    not hypothetical for the default either: one cone of fifteen reaches 169 dBZ at
    ``n_cg=12``, masked in the final grid only because the vertical interpolation
    happens to drop it.

    These tests exercise the check directly rather than through a synthetic volume.
    Several attempts to provoke divergence synthetically (jittered azimuths, a sharp
    azimuthal wedge, deliberately tightened ray pairs) all stayed bounded: the real
    volume's ill-conditioning comes from ray pairs 11x to 300x closer than nominal
    spacing, and a fixture reproducing that faithfully enough to diverge would be
    fitting the test to one file. Testing the predicate on constructed reports is
    both honest about what is verified and sensitive to the thing that matters.
    """

    @staticmethod
    def _report(tilt_index=0, grid_min=-60.0, grid_max=60.0, overshoot=0.0):
        """A TiltReport with only the fields the check reads."""
        return TiltReport(
            tilt_index=tilt_index,
            sweep=tilt_index,
            fixed_angle=1.5 + tilt_index,
            sweep_class="SweepClass.NON_UNIFORM",
            azimuth_path="nufft_kb",
            nrays=900,
            masked_input_fraction=0.0,
            valid_fraction=1.0,
            height_min_m=500.0,
            height_max_m=8000.0,
            data_min_dbz=-80.0,
            data_max_dbz=62.0,
            grid_min_dbz=grid_min,
            grid_max_dbz=grid_max,
            overshoot_db=overshoot,
            split_cut_size=1,
        )

    def test_a_diverged_solve_warns(self):
        """The measured n_cg=25 case, at the magnitude it actually reached."""
        reports = [self._report(0), self._report(1, -1907.0, 2471.0, 2409.0)]
        with pytest.warns(UnphysicalGridWarning, match="physical range"):
            _check_physical(reports, "dbz")

    def test_a_well_regularised_solve_stays_silent(self):
        """The guard must not cry wolf.

        A warning on a good run is worse than none: it trains users to filter the
        category, which removes the protection entirely. These are the measured
        ranges for ``az_ridge=1e-2`` and ``'auto'``.
        """
        reports = [
            self._report(0, -60.6, 58.4, -5.3),
            self._report(1, -61.2, 59.0, -5.0),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_physical(reports, "dbz")

    def test_gibbs_ringing_alone_does_not_warn(self):
        """A few dB of overshoot at a sharp edge is expected, not a fault.

        The operator's own documentation quotes 5.5 dB on its worked example, and
        the BNF volume shows 13 dB at its steepest tilt with the default settings.
        Warning at that level would fire on every convective case.
        """
        reports = [self._report(0, -93.1, 61.5, 13.1)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_physical(reports, "dbz")

    def test_large_overshoot_inside_the_physical_range_still_warns(self):
        """The intermediate regime: not yet absurd, no longer just ringing.

        ``az_ridge=1e-6`` reached 138 dBZ on the real volume --- caught by the
        physical limit --- but a solve can also overshoot by tens of dB while
        staying inside it, which is already a sign of under-regularisation.
        """
        reports = [self._report(0, -70.0, 95.0, 33.0)]
        with pytest.warns(UnphysicalGridWarning, match="overshoot"):
            _check_physical(reports, "dbz")

    def test_the_warning_names_the_settings_to_change(self):
        """A diagnostic a user cannot act on is only half a diagnostic.

        The failure is remote from its cause: the user changed an iteration count or
        a ridge, and what they see is reflectivity in the thousands. The message
        must close that gap by naming both knobs and the value to use.
        """
        reports = [self._report(0, -1903.0, 2466.0, 2404.0)]
        with pytest.warns(UnphysicalGridWarning) as caught:
            _check_physical(reports, "dbz")
        message = str(caught[0].message)
        assert "n_cg" in message
        assert "az_ridge" in message
        assert "1e-2" in message or "auto" in message

    def test_it_reports_how_many_tilts_and_which_is_worst(self):
        """Fifteen cones, and the user needs to know it was three of them."""
        reports = [
            self._report(0),
            self._report(1, -500.0, 700.0, 640.0),
            self._report(2, -1907.0, 2471.0, 2409.0),
        ]
        with pytest.warns(UnphysicalGridWarning) as caught:
            _check_physical(reports, "dbz")
        message = str(caught[0].message)
        assert "2 of 3" in message
        assert "tilt 2" in message

    def test_it_can_be_escalated_to_an_error(self):
        """Its own category, so a production pipeline can refuse to proceed.

        The right posture differs by context --- interactive work wants to see both
        the field and the warning, an operational run wants to stop --- so the
        category exists to let the caller choose rather than being decided here.
        """
        reports = [self._report(0, -1907.0, 2471.0, 2409.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UnphysicalGridWarning)
            with pytest.raises(UnphysicalGridWarning):
                _check_physical(reports, "dbz")

    @pytest.mark.parametrize("units", ("linear_z", "other"))
    def test_a_non_reflectivity_field_is_not_checked(self, units):
        """DBZ limits are meaningless for velocity, or for linear Z."""
        reports = [self._report(0, -1907.0, 2471.0, 2409.0)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_physical(reports, units)

    def test_the_check_is_wired_into_build_cones(self, volume, monkeypatch):
        """End to end, so the predicate is actually consulted.

        A correct check that ``build_cones`` never calls would pass every unit test
        above --- and did: cutting the call site was the one seeded fault the first
        version of this class failed to catch, because the synthetic fixture never
        trips the guard and so a silent guard looks identical to a happy one.
        Asserting the call happens, with the reports and units it should receive, is
        what closes that hole.
        """
        seen = {}

        def spy(reports, field_units):
            seen["n_reports"] = len(reports)
            seen["field_units"] = field_units

        monkeypatch.setattr("radar_palette.gridding.cones._check_physical", spy)
        shape, limits = (7, 21, 21), ((500.0, 6500.0), (-20e3, 20e3), (-20e3, 20e3))
        grid_volume(volume, grid_shape=shape, grid_limits=limits, method="spectral")
        assert seen["n_reports"] == volume.nsweeps
        assert seen["field_units"] is None

    def test_a_clean_grid_call_raises_no_warning(self, volume):
        """And the wired-in check stays quiet on a well-behaved volume."""
        shape, limits = (7, 21, 21), ((500.0, 6500.0), (-20e3, 20e3), (-20e3, 20e3))
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UnphysicalGridWarning)
            grid_volume(volume, grid_shape=shape, grid_limits=limits, method="spectral")
