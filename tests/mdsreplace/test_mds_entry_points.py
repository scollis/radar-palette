"""Tests for the volume-level entry points.

These pin the contract a caller depends on: the flavour that comes back, which
gates were touched, what value they hold, and the provenance left behind. The
gate that matters most is the last one -- a filled volume looks exactly like a
measured one unless the module says otherwise.
"""

from __future__ import annotations

import numpy as np
import pyart
import pytest

from radar_palette.gateid.single import NAME_TO_CODE
from radar_palette.mdsreplace import MdsProfile, mds_profile, replace_below_mds
from radar_palette.testing import make_empty_ppi_volume

FLOOR_AT_1KM = -40.0
RNG_SEED = 3


def _volume(
    nsweeps=2,
    ngates=120,
    rays_per_sweep=120,
    weather=True,
    censored=False,
    spread_db=1.5,
    censor_margin_db=3.0,
):
    """A synthetic PPI volume with a known noise floor and optional weather.

    ``censored`` masks everything below ``floor + censor_margin_db``, which is
    how a thresholded product such as WSR-88D Level II arrives. The noise
    spread has to be comparable to that margin for anything to survive the
    threshold, exactly as on a real radar -- a threshold several sigma above
    the floor would deliver an empty volume.
    """
    radar = make_empty_ppi_volume(
        nsweeps=nsweeps, ngates=ngates, rays_per_sweep=rays_per_sweep
    )
    rng = np.random.default_rng(RNG_SEED)
    ranges = np.asarray(radar.range["data"], dtype="f8")
    trend = 20.0 * np.log10(ranges / 1000.0)
    values = (
        FLOOR_AT_1KM
        + trend[None, :]
        + rng.normal(0.0, spread_db, (radar.nrays, radar.ngates))
    )
    truth = np.zeros((radar.nrays, radar.ngates), dtype="i2")
    truth[:] = NAME_TO_CODE["no_scatter"]
    if weather:
        values[10:40, 30:70] = 35.0
        truth[10:40, 30:70] = NAME_TO_CODE["moderate_rain"]
    if censored:
        threshold = FLOOR_AT_1KM + censor_margin_db + trend[None, :]
        values = np.where(values >= threshold, values, np.nan)
    radar.add_field(
        "reflectivity",
        {"data": np.ma.masked_invalid(values), "units": "dBZ", "_FillValue": np.nan},
    )
    radar.add_field(
        "gate_id",
        {"data": np.ma.masked_array(truth, mask=False), "units": ""},
    )
    return radar


def _filter_out_non_weather(radar):
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_all()
    gatefilter.include_equal("gate_id", NAME_TO_CODE["moderate_rain"])
    return gatefilter


# --------------------------------------------------------------------------
# measuring
# --------------------------------------------------------------------------


def test_profile_recovers_the_floor_of_a_synthetic_volume():
    profile = mds_profile(_volume(), min_range_m=0.0)
    assert profile.intercept_dbz == pytest.approx(FLOOR_AT_1KM, abs=0.5)
    assert profile.method == "noise"
    assert profile.meta["mask_source"].startswith("gate_id:no_scatter")


def test_profile_uses_the_gate_id_when_present_and_all_gates_when_not():
    radar = _volume()
    with_id = mds_profile(radar, min_range_m=0.0)
    without = mds_profile(radar, gate_id_field=None, snr_max=None, min_range_m=0.0)
    assert without.meta["mask_source"] == "all_gates"
    # The mode survives the weather either way; this is the property that lets
    # the module work on volumes that were never classified.
    assert without.intercept_dbz == pytest.approx(with_id.intercept_dbz, abs=0.5)


def test_profile_can_be_restricted_to_chosen_sweeps():
    radar = _volume(nsweeps=3)
    profile = mds_profile(radar, sweeps=[0, 1], min_range_m=0.0)
    assert profile.meta["sweeps"] == [0, 1]
    assert profile.meta["n_rays_used"] == 240
    with pytest.raises(ValueError, match="selected no rays"):
        mds_profile(radar, sweeps=[], min_range_m=0.0)


def test_missing_reflectivity_is_a_clear_error():
    radar = make_empty_ppi_volume(nsweeps=1, ngates=10, rays_per_sweep=10)
    with pytest.raises(KeyError, match="no reflectivity field"):
        mds_profile(radar)


def test_a_thresholded_volume_is_measured_by_the_censoring_route():
    radar = _volume(censored=True, weather=False, spread_db=4.0, censor_margin_db=3.0)
    profile = mds_profile(radar, min_range_m=0.0)
    assert profile.method == "censor"
    assert profile.meta["missing_fraction"] > 0.5
    # The censoring boundary, not the receiver floor: it sits above it by the
    # margin the threshold used.
    assert profile.intercept_dbz > FLOOR_AT_1KM
    assert profile.intercept_dbz == pytest.approx(FLOOR_AT_1KM + 3.0, abs=3.0)


# --------------------------------------------------------------------------
# replacing
# --------------------------------------------------------------------------


def test_excluded_gates_take_the_mds_at_their_own_range():
    radar = _volume()
    profile = mds_profile(radar, min_range_m=0.0)
    gatefilter = _filter_out_non_weather(radar)
    filled, meta = replace_below_mds(radar, gatefilter, profile=profile)

    values = np.ma.filled(filled.fields["reflectivity"]["data"].astype("f8"), np.nan)
    flag = np.ma.filled(filled.fields["mds_replaced"]["data"], 0).astype(bool)
    expected = profile.evaluate(np.asarray(filled.range["data"], dtype="f8"))
    assert np.allclose(values[flag], np.broadcast_to(expected, values.shape)[flag])
    # Filling is not masking: no gate is left non-finite.
    assert np.isfinite(values).all()
    assert meta["n_replaced"] == int(flag.sum())
    assert meta["n_replaced_by_source"]["gatefilter"] > 0


def test_weather_gates_are_untouched():
    radar = _volume()
    before = np.ma.filled(
        radar.fields["reflectivity"]["data"].astype("f8"), np.nan
    ).copy()
    gatefilter = _filter_out_non_weather(radar)
    filled, _ = replace_below_mds(radar, gatefilter, min_range_m=0.0)
    after = np.ma.filled(filled.fields["reflectivity"]["data"].astype("f8"), np.nan)
    kept = ~np.ma.filled(filled.fields["mds_replaced"]["data"], 0).astype(bool)
    assert np.array_equal(after[kept], before[kept])
    assert kept.sum() == 30 * 40


def test_masked_gates_are_filled_even_with_no_gate_filter_at_all():
    """The "grid down to the floor" case: no classification involved."""
    radar = _volume(censored=True, weather=False, spread_db=4.0)
    filled, meta = replace_below_mds(radar, min_range_m=0.0)
    values = np.ma.filled(filled.fields["reflectivity"]["data"].astype("f8"), np.nan)
    assert np.isfinite(values).all()
    assert meta["n_replaced_by_source"]["gatefilter"] == 0
    assert meta["n_replaced_by_source"]["masked"] > 0


def test_include_masked_can_be_switched_off():
    radar = _volume(censored=True, weather=False, spread_db=4.0)
    filled, meta = replace_below_mds(radar, include_masked=False, min_range_m=0.0)
    values = np.ma.filled(filled.fields["reflectivity"]["data"].astype("f8"), np.nan)
    assert not np.isfinite(values).all()
    assert meta["n_replaced"] == 0


def test_a_fill_sentinel_is_replaced_along_with_everything_else():
    radar = _volume(weather=False)
    data = radar.fields["reflectivity"]["data"]
    data[:, -10:] = -80.0
    radar.fields["reflectivity"]["data"] = data
    filled, meta = replace_below_mds(radar, min_range_m=0.0)
    values = np.ma.filled(filled.fields["reflectivity"]["data"].astype("f8"), np.nan)
    assert meta["profile"].meta["sentinels"] == [-80.0]
    assert meta["n_replaced_by_source"]["sentinel"] > 0
    assert not (values == -80.0).any()


def test_the_provenance_flag_and_comment_say_the_gates_are_synthetic():
    radar = _volume()
    filled, meta = replace_below_mds(
        radar, _filter_out_non_weather(radar), min_range_m=0.0
    )
    flag = filled.fields["mds_replaced"]
    assert flag["flag_meanings"] == "measured replaced"
    assert "not a measurement" in flag["comment"]
    assert "minimum detectable signal" in filled.fields["reflectivity"]["comment"]
    assert meta["flag_field"] == "mds_replaced"


def test_the_flag_field_can_be_declined():
    radar = _volume()
    filled, _ = replace_below_mds(radar, flag_field=None, min_range_m=0.0)
    assert "mds_replaced" not in filled.fields


def test_the_mds_can_be_written_as_a_field_for_inspection():
    radar = _volume()
    filled, _ = replace_below_mds(
        radar, mds_field="minimum_detectable_signal", min_range_m=0.0
    )
    mds = np.ma.filled(
        filled.fields["minimum_detectable_signal"]["data"].astype("f8"), np.nan
    )
    assert filled.fields["minimum_detectable_signal"]["units"] == "dBZ"
    # Constant along rays, rising along range.
    assert np.allclose(mds[0], mds[-1])
    assert (np.diff(mds[0]) > 0).all()


def test_a_cap_keeps_long_range_fill_out_of_light_rain_values():
    radar = _volume()
    filled, meta = replace_below_mds(
        radar, _filter_out_non_weather(radar), cap_dbz=-20.0, min_range_m=0.0
    )
    values = np.ma.filled(filled.fields["reflectivity"]["data"].astype("f8"), np.nan)
    flag = np.ma.filled(filled.fields["mds_replaced"]["data"], 0).astype(bool)
    assert values[flag].max() == pytest.approx(-20.0)
    assert meta["fill_dbz_range"][1] == pytest.approx(-20.0)


def test_one_profile_can_be_reused_across_volumes():
    first, second = _volume(), _volume()
    profile = mds_profile(first, min_range_m=0.0)
    _, meta = replace_below_mds(second, profile=profile)
    assert meta["profile"] is profile


def test_a_profile_and_estimation_keywords_cannot_both_be_given():
    radar = _volume()
    profile = mds_profile(radar, min_range_m=0.0)
    with pytest.raises(TypeError, match="mutually exclusive"):
        replace_below_mds(radar, profile=profile, min_range_m=0.0)


def test_a_non_profile_is_rejected_before_anything_is_written():
    radar = _volume()
    with pytest.raises(TypeError, match="must be an MdsProfile"):
        replace_below_mds(radar, profile={"intercept_dbz": -40.0})
    assert "mds_replaced" not in radar.fields


def test_a_gate_filter_from_a_different_volume_is_refused():
    radar = _volume(ngates=120)
    other = _volume(ngates=60)
    with pytest.raises(ValueError, match="built on a different object"):
        replace_below_mds(radar, _filter_out_non_weather(other), min_range_m=0.0)


def test_secondary_reflectivity_fields_are_filled_alongside_the_primary():
    radar = _volume(weather=False)
    radar.add_field(
        "reflectivity_v",
        {"data": radar.fields["reflectivity"]["data"].copy(), "units": "dBZ"},
    )
    filled, meta = replace_below_mds(radar, min_range_m=0.0)
    assert meta["fields"] == ["reflectivity", "reflectivity_v"]


def test_naming_a_field_that_is_not_there_is_an_error():
    radar = _volume()
    with pytest.raises(KeyError, match="is not on this volume"):
        replace_below_mds(radar, fields=["nonexistent"], min_range_m=0.0)


def test_the_returned_profile_is_the_one_that_was_used():
    radar = _volume()
    _, meta = replace_below_mds(radar, min_range_m=0.0)
    assert isinstance(meta["profile"], MdsProfile)
    assert meta["profile"].meta["field"] == "reflectivity"
