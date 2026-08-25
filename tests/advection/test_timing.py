"""Unit tests for per-ray time reconstruction in advection interpolation.

The property under test throughout is that an interpolated volume must carry the
time it *represents*, not the time of either volume it was built from. These
tests are deliberately independent of the reflectivity machinery: timing is
verified on bare radar geometries so a failure localises to the time handling
rather than to optical flow or sampling.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from netCDF4 import num2date

from radar_palette.advection.timing import (
    apply_interpolated_time,
    interpolate_ray_times,
    volume_reference_time,
)
from radar_palette.testing import assign_scan_times, make_empty_ppi_volume

VOLUME_START = datetime(2011, 5, 20, 11, 34, 0, tzinfo=UTC)
SECONDS_BETWEEN_VOLUMES = 420.0


def absolute_ray_times(time_dict):
    """Decode a CF time dictionary to an array of timezone-aware datetimes."""
    decoded = num2date(
        np.asarray(time_dict["data"]),
        time_dict["units"],
        calendar=time_dict.get("calendar", "gregorian"),
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )
    return np.array([moment.replace(tzinfo=UTC) for moment in np.atleast_1d(decoded)])


def elapsed_seconds(later, earlier):
    """Signed difference between two datetimes, in seconds."""
    return (later - earlier).total_seconds()


@pytest.fixture
def early_volume():
    """Volume observed at VOLUME_START, timed in its own epoch."""
    return assign_scan_times(make_empty_ppi_volume(), VOLUME_START)


@pytest.fixture
def late_volume():
    """Volume observed one scan interval after ``early_volume``."""
    return assign_scan_times(
        make_empty_ppi_volume(),
        VOLUME_START + timedelta(seconds=SECONDS_BETWEEN_VOLUMES),
    )


class TestVolumeReferenceTime:
    """``volume_reference_time`` collapses a volume's ray times to one instant."""

    def test_returns_timezone_aware_datetime(self, early_volume):
        assert volume_reference_time(early_volume).tzinfo is not None

    def test_equals_mean_of_ray_times(self, early_volume):
        ray_times = absolute_ray_times(early_volume.time)
        offsets = [elapsed_seconds(moment, ray_times[0]) for moment in ray_times]
        expected = ray_times[0] + timedelta(seconds=float(np.mean(offsets)))
        assert (
            abs(elapsed_seconds(volume_reference_time(early_volume), expected)) < 1e-6
        )

    def test_falls_within_the_scan_window(self, early_volume):
        ray_times = absolute_ray_times(early_volume.time)
        reference = volume_reference_time(early_volume)
        assert ray_times.min() <= reference <= ray_times.max()

    def test_is_independent_of_the_epoch_used_to_encode_it(self):
        """One absolute scan encoded against two epochs has one reference time."""
        native_epoch = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        shifted_epoch = assign_scan_times(
            make_empty_ppi_volume(),
            VOLUME_START,
            epoch=VOLUME_START - timedelta(days=3650),
        )
        difference = elapsed_seconds(
            volume_reference_time(native_epoch), volume_reference_time(shifted_epoch)
        )
        assert abs(difference) < 1e-3

    def test_ignores_masked_ray_times(self, early_volume):
        """A masked ray must not drag the reference time toward its fill value."""
        unmasked_reference = volume_reference_time(early_volume)
        ray_times = np.ma.masked_array(early_volume.time["data"].astype("float64"))
        ray_times[3] = -9999.0
        ray_times.mask = np.zeros(ray_times.shape, dtype=bool)
        ray_times.mask[3] = True
        early_volume.time["data"] = ray_times
        assert (
            abs(
                elapsed_seconds(volume_reference_time(early_volume), unmasked_reference)
            )
            < 1.0
        )

    def test_advances_when_the_volume_is_observed_later(
        self, early_volume, late_volume
    ):
        difference = elapsed_seconds(
            volume_reference_time(late_volume), volume_reference_time(early_volume)
        )
        assert difference == pytest.approx(SECONDS_BETWEEN_VOLUMES, abs=1e-3)


class TestUnequalRayCounts:
    """Real volumes from the same radar do not repeat ray counts exactly.

    Measured on ARM C-SAPR2 at Bankhead: three consecutive volumes of the same
    15-sweep strategy have 13955, 13951 and 13954 rays, because a sweep occasionally
    records one ray more or fewer. An equality requirement on ``nrays`` rejects
    every such pair, which is to say every real pair.

    The arithmetic never needed it: the interpolated volume keeps ``radar_early``'s
    acquisition pattern and is shifted by a single scalar derived from the two
    volumes' reference times, so there is no ray-to-ray correspondence to violate.
    """

    def _volume(self, rays_per_sweep, start):
        return assign_scan_times(
            make_empty_ppi_volume(nsweeps=1, ngates=5, rays_per_sweep=rays_per_sweep),
            start,
        )

    def test_accepts_volumes_differing_by_one_ray(self):
        """The case that real data actually presents."""
        early = self._volume(360, datetime(2026, 8, 18, 23, 29, tzinfo=UTC))
        late = self._volume(359, datetime(2026, 8, 18, 23, 45, tzinfo=UTC))
        times = interpolate_ray_times(early, late, 0.5)
        assert len(times["data"]) == early.nrays

    def test_output_length_follows_the_earlier_volume(self):
        """Output geometry is ``earlier``'s, so its ray count is ``earlier``'s."""
        early = self._volume(360, datetime(2026, 8, 18, 23, 29, tzinfo=UTC))
        for late_rays in (300, 359, 361, 420):
            late = self._volume(late_rays, datetime(2026, 8, 18, 23, 45, tzinfo=UTC))
            assert len(interpolate_ray_times(early, late, 0.5)["data"]) == 360

    def test_unequal_counts_give_the_same_shift_as_equal_ones(self):
        """The shift depends on reference times, not on ray counts.

        Truncating the later volume changes its ray count and its mean ray time,
        so this compares against a later volume whose reference time is held fixed
        by construction: same start, same interval, different length. The resulting
        volume-level shift must differ only through that reference time, never
        through the count itself.
        """
        early = self._volume(360, datetime(2026, 8, 18, 23, 29, tzinfo=UTC))
        late_full = self._volume(360, datetime(2026, 8, 18, 23, 45, tzinfo=UTC))
        late_short = self._volume(359, datetime(2026, 8, 18, 23, 45, tzinfo=UTC))
        full = interpolate_ray_times(early, late_full, 0.5)
        short = interpolate_ray_times(early, late_short, 0.5)
        expected_gap = (
            0.5
            * (
                volume_reference_time(late_full) - volume_reference_time(late_short)
            ).total_seconds()
        )
        actual_gap = (
            num2date(full["data"][0], full["units"], calendar=full["calendar"])
            - num2date(short["data"][0], short["units"], calendar=short["calendar"])
        ).total_seconds()
        assert actual_gap == pytest.approx(expected_gap, abs=1e-6)

    def test_apply_interpolated_time_accepts_unequal_counts(self):
        """The public entry point must not reimpose the restriction."""
        early = self._volume(360, datetime(2026, 8, 18, 23, 29, tzinfo=UTC))
        late = self._volume(359, datetime(2026, 8, 18, 23, 45, tzinfo=UTC))
        out = apply_interpolated_time(early, early, late, 0.5)
        assert out.nrays == 360


class TestInterpolateRayTimes:
    """``interpolate_ray_times`` blends two volumes' ray times at a fraction."""

    def test_returns_a_cf_compliant_time_dictionary(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        for required_key in ("data", "units", "calendar", "standard_name", "long_name"):
            assert required_key in interpolated, f"missing CF key {required_key!r}"

    def test_units_string_is_decodable(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        assert absolute_ray_times(interpolated).size == early_volume.nrays

    def test_offsets_are_float64(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        assert np.asarray(interpolated["data"]).dtype == np.float64

    def test_one_offset_per_ray(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        assert np.asarray(interpolated["data"]).shape == (early_volume.nrays,)

    def test_alpha_zero_reproduces_the_early_volume(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.0)
        expected = absolute_ray_times(early_volume.time)
        actual = absolute_ray_times(interpolated)
        worst_error = max(
            abs(elapsed_seconds(a, e)) for a, e in zip(actual, expected, strict=True)
        )
        assert worst_error < 1e-3

    def test_alpha_one_reproduces_the_late_volume(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 1.0)
        expected = absolute_ray_times(late_volume.time)
        actual = absolute_ray_times(interpolated)
        worst_error = max(
            abs(elapsed_seconds(a, e)) for a, e in zip(actual, expected, strict=True)
        )
        assert worst_error < 1e-3

    def test_alpha_half_lands_midway(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        offset = elapsed_seconds(
            volume_reference_time(early_volume),
            num2date(
                np.mean(np.asarray(interpolated["data"])),
                interpolated["units"],
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            ).replace(tzinfo=UTC),
        )
        assert abs(offset) == pytest.approx(SECONDS_BETWEEN_VOLUMES / 2, abs=1e-3)

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0 / 3.0, 0.5, 0.75, 1.0])
    def test_reference_time_advances_linearly_with_alpha(
        self, early_volume, late_volume, alpha
    ):
        interpolated = interpolate_ray_times(early_volume, late_volume, alpha)
        mean_moment = num2date(
            np.mean(np.asarray(interpolated["data"])),
            interpolated["units"],
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        ).replace(tzinfo=UTC)
        advance = elapsed_seconds(mean_moment, volume_reference_time(early_volume))
        assert advance == pytest.approx(alpha * SECONDS_BETWEEN_VOLUMES, abs=1e-3)

    def test_preserves_the_intra_volume_scan_pattern(self, early_volume, late_volume):
        """Ray-to-ray spacing is a property of the scan strategy, not of alpha."""
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        expected_spacing = np.diff(
            np.asarray(early_volume.time["data"], dtype="float64")
        )
        actual_spacing = np.diff(np.asarray(interpolated["data"]))
        np.testing.assert_allclose(actual_spacing, expected_spacing, atol=1e-6)

    def test_ray_times_stay_monotonic(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        assert np.all(np.diff(np.asarray(interpolated["data"])) >= 0.0)

    def test_handles_volumes_encoded_against_different_epochs(self):
        """Mismatched epochs are a real cfradial hazard; absolute times must win."""
        early = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        late = assign_scan_times(
            make_empty_ppi_volume(),
            VOLUME_START + timedelta(seconds=SECONDS_BETWEEN_VOLUMES),
            epoch=VOLUME_START - timedelta(days=1000),
        )
        interpolated = interpolate_ray_times(early, late, 0.5)
        mean_moment = num2date(
            np.mean(np.asarray(interpolated["data"])),
            interpolated["units"],
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        ).replace(tzinfo=UTC)
        advance = elapsed_seconds(mean_moment, volume_reference_time(early))
        assert advance == pytest.approx(SECONDS_BETWEEN_VOLUMES / 2, abs=1e-3)

    def test_does_not_mutate_either_input_volume(self, early_volume, late_volume):
        early_before = np.array(early_volume.time["data"], dtype="float64", copy=True)
        late_before = np.array(late_volume.time["data"], dtype="float64", copy=True)
        early_units, late_units = early_volume.time["units"], late_volume.time["units"]
        interpolate_ray_times(early_volume, late_volume, 0.5)
        np.testing.assert_array_equal(early_volume.time["data"], early_before)
        np.testing.assert_array_equal(late_volume.time["data"], late_before)
        assert early_volume.time["units"] == early_units
        assert late_volume.time["units"] == late_units

    def test_result_does_not_alias_input_arrays(self, early_volume, late_volume):
        interpolated = interpolate_ray_times(early_volume, late_volume, 0.5)
        assert not np.shares_memory(
            np.asarray(interpolated["data"]), np.asarray(early_volume.time["data"])
        )

    def test_alpha_beyond_the_bracket_extrapolates(self, early_volume, late_volume):
        """Extrapolation is permitted but must stay linear and correctly signed."""
        interpolated = interpolate_ray_times(early_volume, late_volume, 1.5)
        mean_moment = num2date(
            np.mean(np.asarray(interpolated["data"])),
            interpolated["units"],
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        ).replace(tzinfo=UTC)
        advance = elapsed_seconds(mean_moment, volume_reference_time(early_volume))
        assert advance == pytest.approx(1.5 * SECONDS_BETWEEN_VOLUMES, abs=1e-3)

    def test_a_differing_ray_count_is_not_an_error(self, early_volume):
        """Superseded requirement: ray counts were once required to match.

        They are not, and requiring it rejected every real pair (see
        ``TestUnequalRayCounts``). Halving the later volume's rays changes its
        length and its mean ray time but must still produce a full-length result
        on the earlier volume's pattern.
        """
        mismatched = assign_scan_times(
            make_empty_ppi_volume(rays_per_sweep=180),
            VOLUME_START + timedelta(seconds=SECONDS_BETWEEN_VOLUMES),
        )
        times = interpolate_ray_times(early_volume, mismatched, 0.5)
        assert len(times["data"]) == early_volume.nrays

    def test_rejects_a_non_finite_alpha(self, early_volume, late_volume):
        with pytest.raises(ValueError, match="finite"):
            interpolate_ray_times(early_volume, late_volume, float("nan"))


class TestApplyInterpolatedTime:
    """``apply_interpolated_time`` stamps the reconstructed time onto a volume."""

    def test_replaces_the_inherited_ray_times(self, early_volume, late_volume):
        """Output must not keep the earlier volume's clock -- the core regression.

        The comparison is on *absolute* times rather than raw offsets. Offsets are
        only meaningful against their own ``units`` string, and the implementation
        re-bases ``units`` on the interpolated volume's first ray -- so the offsets
        can legitimately be unchanged while the instant they denote has moved.
        """
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        inherited_times = absolute_ray_times(reconstructed.time)
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        corrected_times = absolute_ray_times(reconstructed.time)
        shifts = [
            elapsed_seconds(corrected, inherited)
            for corrected, inherited in zip(
                corrected_times, inherited_times, strict=True
            )
        ]
        assert min(shifts) == pytest.approx(SECONDS_BETWEEN_VOLUMES / 2, abs=1e-3), (
            "every ray must be shifted to the interpolated instant"
        )

    def test_rebases_the_units_epoch_rather_than_only_the_offsets(
        self, early_volume, late_volume
    ):
        """Guard the aliasing hazard the previous test documents.

        A consumer that reads ``time["data"]`` and ignores ``time["units"]`` would
        see unchanged numbers, so the epoch shift is the load-bearing change and is
        asserted directly.
        """
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        inherited_units = reconstructed.time["units"]
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        assert reconstructed.time["units"] != inherited_units

    def test_stamped_reference_time_matches_the_target(self, early_volume, late_volume):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.25)
        advance = elapsed_seconds(
            volume_reference_time(reconstructed), volume_reference_time(early_volume)
        )
        assert advance == pytest.approx(0.25 * SECONDS_BETWEEN_VOLUMES, abs=1e-3)

    def test_returns_the_same_object_it_was_given(self, early_volume, late_volume):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        assert (
            apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
            is reconstructed
        )

    def test_refreshes_time_coverage_metadata_when_present(
        self, early_volume, late_volume
    ):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        reconstructed.metadata["time_coverage_start"] = "1999-01-01T00:00:00Z"
        reconstructed.metadata["time_coverage_end"] = "1999-01-01T00:07:00Z"
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        ray_times = absolute_ray_times(reconstructed.time)
        assert reconstructed.metadata["time_coverage_start"].startswith(
            ray_times.min().strftime("%Y-%m-%dT%H:%M:%S")
        )
        assert reconstructed.metadata["time_coverage_end"].startswith(
            ray_times.max().strftime("%Y-%m-%dT%H:%M:%S")
        )

    def test_does_not_invent_absent_time_coverage_metadata(
        self, early_volume, late_volume
    ):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        reconstructed.metadata.pop("time_coverage_start", None)
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        assert "time_coverage_start" not in reconstructed.metadata

    def test_records_provenance_in_history(self, early_volume, late_volume):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        assert "radar_palette" in reconstructed.metadata.get("history", "")

    def test_preserves_pre_existing_history(self, early_volume, late_volume):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        reconstructed.metadata["history"] = "created by the observation system"
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        assert "created by the observation system" in reconstructed.metadata["history"]


class TestCfradialRoundTrip:
    """Reconstructed times must survive being written to and read from cfradial.

    An in-memory fix is not enough: interpolated volumes are usually written out
    and consumed by other tooling, so the corrected time has to survive
    serialisation. These are the integration-level guards on that path.
    """

    @staticmethod
    def write_and_read_back(radar, tmp_path):
        """Round-trip a volume through a cfradial file on disk."""
        pyart = pytest.importorskip("pyart")
        radar.add_field(
            "reflectivity",
            {
                "data": np.ma.masked_invalid(
                    np.zeros((radar.nrays, radar.ngates), dtype="float32")
                ),
                "units": "dBZ",
                "long_name": "reflectivity",
                "_FillValue": -9999.0,
            },
            replace_existing=True,
        )
        path = str(tmp_path / "interpolated.nc")
        pyart.io.write_cfradial(path, radar)
        return pyart.io.read_cfradial(path)

    def test_reference_time_survives_serialisation(
        self, early_volume, late_volume, tmp_path
    ):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        expected = volume_reference_time(reconstructed)
        restored = self.write_and_read_back(reconstructed, tmp_path)
        assert abs(elapsed_seconds(volume_reference_time(restored), expected)) < 1e-3

    def test_round_tripped_time_is_not_the_early_volume_time(
        self, early_volume, late_volume, tmp_path
    ):
        """The regression, verified through the file format rather than in memory."""
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        restored = self.write_and_read_back(reconstructed, tmp_path)
        advance = elapsed_seconds(
            volume_reference_time(restored), volume_reference_time(early_volume)
        )
        assert advance == pytest.approx(SECONDS_BETWEEN_VOLUMES / 2, abs=1e-3)

    def test_provenance_history_survives_serialisation(
        self, early_volume, late_volume, tmp_path
    ):
        reconstructed = assign_scan_times(make_empty_ppi_volume(), VOLUME_START)
        apply_interpolated_time(reconstructed, early_volume, late_volume, 0.5)
        restored = self.write_and_read_back(reconstructed, tmp_path)
        assert "radar_palette" in restored.metadata.get("history", "")
