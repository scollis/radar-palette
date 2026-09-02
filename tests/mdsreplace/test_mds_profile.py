"""Tests for the MDS estimator itself.

The estimator's job is to find one number -- the floor at a reference range --
inside a mixture it does not control: noise, weather, clutter, fill sentinels,
and a near-range span where the power law does not hold. Each test here builds
a mixture with a known answer and checks that the answer comes back, because
the failure mode that matters is not an exception, it is a plausible number
that is wrong by several dB and silently becomes the fill value of every
filtered gate in a volume.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.mdsreplace import profile as mds

RNG_SEED = 7
FLOOR_AT_1KM = -40.0


def _ranges(ngates=200, spacing=250.0):
    return np.arange(ngates, dtype="f8") * spacing + spacing


def _noise_volume(nrays=400, ngates=200, floor_dbz=FLOOR_AT_1KM, spread_db=2.0):
    """A volume that is pure noise: floor + r^2 + per-gate fluctuation."""
    rng = np.random.default_rng(RNG_SEED)
    ranges = _ranges(ngates)
    trend = 20.0 * np.log10(ranges / mds.REFERENCE_RANGE_M)
    values = floor_dbz + trend[None, :] + rng.normal(0.0, spread_db, (nrays, ngates))
    return values, ranges


# --------------------------------------------------------------------------
# the range law
# --------------------------------------------------------------------------


def test_detrending_flattens_a_pure_r_squared_floor():
    ranges = np.array([1e3, 1e4, 1e5])
    dbz = np.array([-40.0, -20.0, 0.0])
    residual, trend = mds.detrend_range(dbz, ranges)
    assert residual == pytest.approx([-40.0, -40.0, -40.0])
    assert trend == pytest.approx([0.0, 20.0, 40.0])


def test_detrending_survives_a_zero_range_gate():
    """A zero-range gate is a metadata artefact, not a reason to return NaN."""
    residual, _ = mds.detrend_range(np.array([-40.0, -20.0]), np.array([0.0, 1e4]))
    assert np.isfinite(residual).all()


def test_fit_holds_the_slope_at_twenty_db_per_decade_by_default():
    ranges = _ranges()
    sample = FLOOR_AT_1KM + 20.0 * np.log10(ranges / mds.REFERENCE_RANGE_M)
    fit = mds.fit_power_law(sample, ranges, min_range_m=0.0)
    assert fit["slope_db_per_decade"] == mds.THEORETICAL_SLOPE_DB_PER_DECADE
    assert fit["intercept_dbz"] == pytest.approx(FLOOR_AT_1KM, abs=1e-6)
    assert fit["residual_rms_db"] == pytest.approx(0.0, abs=1e-6)


def test_fit_can_release_the_slope_and_recover_a_steeper_one():
    ranges = _ranges()
    sample = FLOOR_AT_1KM + 24.0 * np.log10(ranges / mds.REFERENCE_RANGE_M)
    fit = mds.fit_power_law(sample, ranges, fit_slope=True, min_range_m=0.0)
    assert fit["slope_db_per_decade"] == pytest.approx(24.0, abs=0.1)
    assert fit["intercept_dbz"] == pytest.approx(FLOOR_AT_1KM, abs=0.2)


def test_fit_refuses_rather_than_returning_nan_when_nothing_qualifies():
    ranges = _ranges()
    with pytest.raises(ValueError, match="no range gate"):
        mds.fit_power_law(np.full(ranges.size, np.nan), ranges)


# --------------------------------------------------------------------------
# the noise estimator
# --------------------------------------------------------------------------


def test_noise_mode_recovers_a_known_floor_from_pure_noise():
    values, ranges = _noise_volume()
    result = mds.noise_floor_by_range(values, ranges)
    fit = mds.fit_power_law(result["sample_dbz"], ranges, min_range_m=0.0)
    assert fit["intercept_dbz"] == pytest.approx(FLOOR_AT_1KM, abs=0.5)


def test_noise_mode_ignores_a_weather_tail_that_would_move_a_mean():
    """Half the rays carry 40 dBZ weather; the floor must not budge."""
    values, ranges = _noise_volume()
    values[: values.shape[0] // 2, 40:120] = 40.0
    mean_based = np.nanmean(values, axis=0)
    result = mds.noise_floor_by_range(values, ranges)
    fit = mds.fit_power_law(result["sample_dbz"], ranges, min_range_m=0.0)
    assert fit["intercept_dbz"] == pytest.approx(FLOOR_AT_1KM, abs=0.5)
    # The estimator the mode replaces would have been ~20 dB out here.
    contaminated = mean_based - 20.0 * np.log10(ranges / mds.REFERENCE_RANGE_M)
    assert np.nanmax(contaminated) > FLOOR_AT_1KM + 10.0


def test_noise_mode_ignores_a_fill_sentinel_that_would_pin_a_minimum():
    values, ranges = _noise_volume()
    values[:, -20:] = -80.0
    assert mds.find_sentinel_values(values) == [-80.0]
    result = mds.noise_floor_by_range(values, ranges)
    fit = mds.fit_power_law(result["sample_dbz"], ranges, min_range_m=0.0)
    assert fit["intercept_dbz"] == pytest.approx(FLOOR_AT_1KM, abs=0.5)
    # The sentinel gates carry no estimate at all rather than a wrong one.
    assert np.isnan(result["sample_dbz"][-1])


def test_sentinel_detection_declines_when_there_are_too_many_candidates():
    """An already-filled volume repeats a different value at every range.

    Treating those as sentinels would erase the data; the guard makes the
    estimator idempotent instead.
    """
    ranges = _ranges()
    filled = np.broadcast_to(
        FLOOR_AT_1KM + 20.0 * np.log10(ranges / mds.REFERENCE_RANGE_M),
        (50, ranges.size),
    )
    assert mds.find_sentinel_values(filled) == []


def test_running_a_second_pass_on_a_filled_volume_returns_the_same_floor():
    values, ranges = _noise_volume()
    first = mds.estimate_profile(values, ranges, method="noise", min_range_m=0.0)
    filled = np.broadcast_to(first.evaluate(ranges), values.shape)
    second = mds.estimate_profile(filled, ranges, method="noise", min_range_m=0.0)
    assert second.intercept_dbz == pytest.approx(first.intercept_dbz, abs=0.05)


def test_mode_fraction_reports_when_no_population_dominates():
    """A broad uniform spread has no floor to find, and must say so."""
    rng = np.random.default_rng(RNG_SEED)
    ranges = _ranges(ngates=60)
    trend = 20.0 * np.log10(ranges / mds.REFERENCE_RANGE_M)
    values = trend[None, :] + rng.uniform(-40.0, 20.0, (600, ranges.size))
    estimate = mds.estimate_profile(values, ranges, method="noise", min_range_m=0.0)
    assert estimate.meta["mode_fraction_median"] < 0.2
    assert any("may not be the noise floor" in w for w in estimate.meta["warnings"])


def test_a_supplied_mask_restricts_the_population_used():
    """Rays flagged as weather must not contribute, even at the same range."""
    values, ranges = _noise_volume()
    values[:100, :] = 35.0
    mask = np.ones(values.shape, dtype=bool)
    mask[:100, :] = False
    fit = mds.fit_power_law(
        mds.noise_floor_by_range(values, ranges, mask=mask)["sample_dbz"],
        ranges,
        min_range_m=0.0,
    )
    assert fit["intercept_dbz"] == pytest.approx(FLOOR_AT_1KM, abs=0.5)


def test_mask_shape_mismatch_is_an_error_not_a_broadcast():
    values, ranges = _noise_volume()
    with pytest.raises(ValueError, match="does not match"):
        mds.noise_floor_by_range(values, ranges, mask=np.ones((3, 3), dtype=bool))


# --------------------------------------------------------------------------
# the censoring estimator
# --------------------------------------------------------------------------


def test_censor_estimator_traces_a_thresholded_detection_boundary():
    """Everything below floor + 6 dB masked, as a thresholded product is."""
    values, ranges = _noise_volume(spread_db=4.0)
    trend = 20.0 * np.log10(ranges / mds.REFERENCE_RANGE_M)
    threshold = FLOOR_AT_1KM + 6.0 + trend[None, :]
    censored = np.where(values >= threshold, values, np.nan)
    result = mds.detection_floor_by_range(censored, ranges, quantile=5.0)
    fit = mds.fit_power_law(result["sample_dbz"], ranges, min_range_m=0.0)
    assert fit["intercept_dbz"] == pytest.approx(FLOOR_AT_1KM + 6.0, abs=1.5)


def test_auto_picks_censor_for_a_mostly_masked_volume_and_noise_otherwise():
    values, ranges = _noise_volume()
    dense = mds.estimate_profile(values, ranges, method="auto", min_range_m=0.0)
    assert dense.method == "noise"

    sparse = values.copy()
    sparse[:350, :] = np.nan
    thin = mds.estimate_profile(sparse, ranges, method="auto", min_range_m=0.0)
    assert thin.method == "censor"
    assert thin.meta["missing_fraction"] > 0.5


# --------------------------------------------------------------------------
# robust trimming, models and evaluation
# --------------------------------------------------------------------------


def test_near_range_contamination_is_trimmed_rather_than_averaged_in():
    """A contaminated near-range span must not tilt the whole fit.

    This reproduces what a real C-SAPR2 volume does: inside a few kilometres
    the no-return population sits well above the power law.
    """
    values, ranges = _noise_volume()
    values[:, :20] += 15.0
    estimate = mds.estimate_profile(values, ranges, method="noise", min_range_m=0.0)
    assert estimate.intercept_dbz == pytest.approx(FLOOR_AT_1KM, abs=0.5)
    assert estimate.meta["fit_n_trimmed"] >= 15
    assert estimate.meta["fit_min_valid_range_m"] > ranges[15]
    assert any("rejected as inconsistent" in w for w in estimate.meta["warnings"])


def test_trimming_can_be_switched_off():
    values, ranges = _noise_volume()
    values[:, :20] += 15.0
    estimate = mds.estimate_profile(
        values, ranges, method="noise", min_range_m=0.0, trim_sigma=None
    )
    assert estimate.meta["fit_n_trimmed"] == 0
    assert estimate.meta["fit_residual_rms_db"] > 1.0


def test_empirical_model_follows_a_measured_drift_the_power_law_cannot():
    """A floor that drifts with range: the empirical model must track it."""
    values, ranges = _noise_volume(spread_db=0.5)
    drift = np.linspace(0.0, 6.0, ranges.size)
    values += drift[None, :]
    law = mds.estimate_profile(values, ranges, model="power_law", min_range_m=0.0)
    empirical = mds.estimate_profile(values, ranges, model="empirical", min_range_m=0.0)
    far = ranges[-5]
    truth = FLOOR_AT_1KM + drift[-5] + 20.0 * np.log10(far / mds.REFERENCE_RANGE_M)
    assert abs(float(empirical.evaluate([far])[0]) - truth) < abs(
        float(law.evaluate([far])[0]) - truth
    )
    assert float(empirical.evaluate([far])[0]) == pytest.approx(truth, abs=1.0)


def test_empirical_model_extrapolates_with_the_power_law_beyond_its_samples():
    values, ranges = _noise_volume()
    empirical = mds.estimate_profile(values, ranges, model="empirical", min_range_m=0.0)
    beyond = float(ranges[-1] * 3.0)
    expected = empirical.intercept_dbz + 20.0 * np.log10(beyond / mds.REFERENCE_RANGE_M)
    assert float(empirical.evaluate([beyond])[0]) == pytest.approx(expected, abs=1e-6)


def test_margin_and_cap_are_applied_on_evaluation():
    values, ranges = _noise_volume()
    base = mds.estimate_profile(values, ranges, min_range_m=0.0)
    lifted = mds.estimate_profile(values, ranges, min_range_m=0.0, margin_db=3.0)
    assert float(lifted.evaluate([5e4])[0]) == pytest.approx(
        float(base.evaluate([5e4])[0]) + 3.0
    )

    capped = mds.estimate_profile(values, ranges, min_range_m=0.0, cap_dbz=-25.0)
    assert float(capped.evaluate([2e5])[0]) == pytest.approx(-25.0)
    assert float(capped.evaluate([1e3])[0]) < -25.0


def test_evaluation_is_the_power_law_at_every_range():
    values, ranges = _noise_volume()
    estimate = mds.estimate_profile(values, ranges, min_range_m=0.0)
    doubled = estimate.evaluate([1e4, 2e4])
    # Doubling the range raises the floor by 6.02 dB, always.
    assert float(doubled[1] - doubled[0]) == pytest.approx(6.0206, abs=1e-3)


def test_unknown_method_or_model_is_rejected():
    values, ranges = _noise_volume()
    with pytest.raises(ValueError, match="unknown method"):
        mds.estimate_profile(values, ranges, method="vibes")
    with pytest.raises(ValueError, match="unknown model"):
        mds.estimate_profile(values, ranges, model="vibes", min_range_m=0.0)


def test_range_and_field_shapes_must_agree():
    values, ranges = _noise_volume()
    with pytest.raises(ValueError, match="gates but range has"):
        mds.estimate_profile(values, ranges[:-1])


def test_repr_and_to_dict_carry_the_numbers_a_log_line_needs():
    values, ranges = _noise_volume()
    estimate = mds.estimate_profile(values, ranges, min_range_m=0.0)
    text = repr(estimate)
    assert "noise/power_law" in text and "dB/decade" in text
    summary = estimate.to_dict()
    assert summary["method"] == "noise"
    assert set(summary) >= {"intercept_dbz", "residual_rms_db", "model", "meta"}
    # No arrays: this has to survive being written into a field comment.
    assert all(not hasattr(v, "shape") for v in summary["meta"].values())
