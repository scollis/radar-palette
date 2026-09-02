"""Tests for the two-sided KDP bounds.

The properties pinned here are the ones that make the box trustworthy: that the
self-consistency relation is genuinely independent of drop concentration, that
it refuses to answer outside its valid ZDR range instead of returning a negative
KDP, that the sign policy is altitude-conditioned rather than global, and that
the box is released rather than crossed where the rain relation does not apply.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.phase import bounds

# --------------------------------------------------------------------------
# self-consistency relation
# --------------------------------------------------------------------------


def test_kdp_over_z_is_independent_of_concentration():
    """KDP and z_H are both exactly linear in Nw, so the ratio must not move.

    Raising Z by 10 dB at fixed ZDR is exactly a factor-of-ten change in
    concentration, and must raise KDP by exactly ten.
    """
    zdr = np.full(5, 1.5)
    low = bounds.self_consistency_kdp(np.full(5, 30.0), zdr)
    high = bounds.self_consistency_kdp(np.full(5, 40.0), zdr)
    assert np.allclose(high / low, 10.0)


def test_relation_rises_with_reflectivity_at_fixed_zdr():
    z = np.array([20.0, 30.0, 40.0, 50.0])
    kdp = bounds.self_consistency_kdp(z, np.full(4, 1.0))
    assert np.all(np.diff(kdp) > 0.0)


def test_c_band_amplitude_is_physical_in_moderate_rain():
    """A 40 dBZ / 1.5 dB gate is ordinary rain and must give ordinary KDP."""
    kdp = float(bounds.self_consistency_kdp(40.0, 1.5, band="C"))
    assert 0.3 < kdp < 3.0


def test_relation_refuses_to_answer_above_its_root():
    """Above the cubic's first positive root the polynomial goes negative.

    Returning that negative value as a KDP bound would invert the box, so the
    relation reports NaN and the caller's release path handles the gate.
    """
    root = bounds.SELF_CONSISTENCY_ZDR_ROOTS["C"]
    assert np.isfinite(bounds.self_consistency_kdp(45.0, root - 0.1, band="C"))
    assert np.isnan(bounds.self_consistency_kdp(45.0, root + 0.1, band="C"))


def test_non_finite_moments_give_nan():
    result = bounds.self_consistency_kdp(
        np.array([np.nan, 40.0, 40.0]), np.array([1.0, np.nan, 1.0])
    )
    assert np.isnan(result[0]) and np.isnan(result[1])
    assert np.isfinite(result[2])


def test_bands_are_ordered_by_wavelength():
    """KDP scales inversely with wavelength, so X > C > S at matched moments."""
    values = [
        float(bounds.self_consistency_kdp(40.0, 1.5, band=band))
        for band in ("S", "C", "X")
    ]
    assert values[0] < values[1] < values[2]


def test_supplied_coefficients_override_the_band():
    gourley = bounds.GOURLEY_2009_TABLE4["C"]
    ours = bounds.self_consistency_kdp(40.0, 3.0, band="C")
    theirs = bounds.self_consistency_kdp(40.0, 3.0, coefficients=gourley)
    # Documented in the module docstring: the two credible fits disagree by
    # about a third at ZDR = 3 dB, which is why the tolerance factors are wide.
    assert float(ours / theirs) == pytest.approx(1.335, abs=0.02)


def test_unknown_band_is_an_error():
    with pytest.raises(ValueError, match="band must be one of"):
        bounds.self_consistency_kdp(40.0, 1.0, band="Ka")


# --------------------------------------------------------------------------
# reflectivity cap
# --------------------------------------------------------------------------


def test_cap_ladder_matches_the_published_rungs():
    cap = bounds.reflectivity_cap(np.array([20.0, 34.9, 35.0, 44.9, 45.0]))
    assert np.allclose(cap[:2], 8.0)
    assert np.allclose(cap[2:4], 10.0)
    assert np.isinf(cap[4])


def test_cap_above_the_top_rung_is_explicit():
    cap = bounds.reflectivity_cap(np.array([50.0]), cap_above=20.0)
    assert cap[0] == pytest.approx(20.0)


def test_unknown_reflectivity_imposes_no_cap():
    assert np.isinf(bounds.reflectivity_cap(np.array([np.nan]))[0])


# --------------------------------------------------------------------------
# phase-only least-squares floor
# --------------------------------------------------------------------------


def test_lsf_recovers_a_constant_slope_exactly():
    dr_km, kdp = 0.1, 1.5
    phidp = 2.0 * kdp * np.arange(300) * dr_km
    result = bounds.lsf_kdp(phidp[None, :], dr_km, window_km=5.0)
    assert np.allclose(result[0], kdp)


def test_lsf_is_unmoved_by_an_additive_constant():
    dr_km = 0.1
    phidp = 2.0 * 1.5 * np.arange(300) * dr_km
    a = bounds.lsf_kdp(phidp[None, :], dr_km, window_km=5.0)
    b = bounds.lsf_kdp((phidp + 137.0)[None, :], dr_km, window_km=5.0)
    assert np.allclose(a, b, equal_nan=True)


def test_lsf_averages_away_a_local_excursion():
    """The property the Eq. (15) floor exists for."""
    dr_km, kdp = 0.1, 1.5
    clean = 2.0 * kdp * np.arange(400) * dr_km
    spiked = clean.copy()
    spiked[200] += 60.0
    result = bounds.lsf_kdp(spiked[None, :], dr_km, window_km=10.0)
    assert float(result[0, 200]) == pytest.approx(kdp, abs=0.05)


def test_lsf_returns_nan_where_too_few_gates_are_valid():
    phidp = np.full((1, 200), np.nan)
    phidp[0, :5] = np.arange(5.0)
    result = bounds.lsf_kdp(phidp, 0.1, window_km=5.0)
    assert np.isnan(result).all()


def test_lsf_rejects_a_nonpositive_window():
    with pytest.raises(ValueError, match="window_km"):
        bounds.lsf_kdp(np.zeros((1, 10)), 0.1, window_km=0.0)


# --------------------------------------------------------------------------
# rain mask
# --------------------------------------------------------------------------


def test_rain_mask_excludes_hail_cold_and_incoherent_gates():
    z = np.array([[30.0, 55.0, 30.0, 30.0]])
    rhohv = np.array([[0.99, 0.99, 0.80, 0.99]])
    height = np.array([[1.0, 1.0, 1.0, 6.0]])
    mask = bounds.rain_mask_from_context(
        z_dbz=z, rhohv=rhohv, height_km=height, iso0_km=3.2
    )
    assert mask.tolist() == [[True, False, False, False]]


def test_rain_mask_with_no_criteria_raises():
    with pytest.raises(ValueError, match="at least one of"):
        bounds.rain_mask_from_context()


def test_height_without_an_isotherm_raises():
    with pytest.raises(ValueError, match="iso0_km"):
        bounds.rain_mask_from_context(height_km=np.zeros((1, 3)))


# --------------------------------------------------------------------------
# sign policy
# --------------------------------------------------------------------------


def test_altitude_conditioned_floor_is_zero_below_and_negative_above():
    # The ramp is centred on the isotherm, so it spans 2.95 to 3.45 km for a
    # 0.5 km blend and sits at exactly half its depth on the isotherm itself.
    height = np.array([[0.5, 2.0, 2.95, 3.2, 3.45, 6.0]])
    floor = bounds.sign_floor(
        height.shape,
        policy="altitude_conditioned",
        height_km=height,
        iso0_km=3.2,
        blend_km=0.5,
        negative_floor=-2.0,
    )
    assert np.allclose(floor[0, :3], 0.0)
    assert floor[0, 3] == pytest.approx(-1.0)
    assert floor[0, 4] == pytest.approx(-2.0)
    assert floor[0, 5] == pytest.approx(-2.0)


def test_rain_nonnegative_reproduces_the_published_behaviour():
    floor = bounds.sign_floor((2, 3), policy="rain_nonnegative")
    assert np.all(floor == 0.0)


def test_altitude_policy_refuses_to_downgrade_silently():
    with pytest.raises(ValueError, match="requires height_km and iso0_km"):
        bounds.sign_floor((2, 3), policy="altitude_conditioned")


def test_unknown_height_is_treated_as_rain():
    height = np.array([[np.nan]])
    floor = bounds.sign_floor(
        height.shape, height_km=height, iso0_km=3.0, negative_floor=-2.0
    )
    assert floor[0, 0] == pytest.approx(0.0)


def test_unknown_policy_is_an_error():
    with pytest.raises(ValueError, match="policy must be one of"):
        bounds.sign_floor((1, 1), policy="positive")


# --------------------------------------------------------------------------
# assembling the box
# --------------------------------------------------------------------------


def _grid(n_rays=2, n_gates=60):
    z = np.full((n_rays, n_gates), 35.0)
    zdr = np.full((n_rays, n_gates), 1.2)
    height = np.tile(np.linspace(0.5, 6.0, n_gates), (n_rays, 1))
    return z, zdr, height


def test_box_is_ordered_everywhere():
    z, zdr, height = _grid()
    lower, upper, meta = bounds.kdp_bounds(z, zdr, height_km=height, iso0_km=3.2)
    assert np.all(lower <= upper)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert meta["n_crossed"] == 0


def test_tolerance_factors_scale_the_box_around_the_relation():
    z, zdr, height = _grid()
    lower, upper, _ = bounds.kdp_bounds(
        z, zdr, height_km=height, iso0_km=3.2, lower_factor=0.5, upper_factor=2.0
    )
    central = bounds.self_consistency_kdp(z, zdr)
    rain = height <= 3.2
    assert np.allclose(lower[rain], 0.5 * central[rain])
    assert np.allclose(upper[rain], np.minimum(2.0 * central[rain], 8.0))


def test_box_is_released_outside_the_rain_mask():
    z, zdr, height = _grid()
    rain = bounds.rain_mask_from_context(height_km=height, iso0_km=3.2)
    lower, upper, meta = bounds.kdp_bounds(
        z, zdr, rain_mask=rain, height_km=height, iso0_km=3.2, non_rain_cap=10.0
    )
    assert np.all(upper[~rain] == 10.0)
    assert np.all(lower[~rain] <= 0.0)
    assert meta["rain_mask_supplied"] is True
    assert meta["n_rain"] == int(rain.sum())


def test_missing_moments_release_the_box_rather_than_bounding_on_nothing():
    z, zdr, height = _grid()
    z[:, :10] = np.nan
    lower, upper, meta = bounds.kdp_bounds(
        z, zdr, height_km=height, iso0_km=3.2, non_rain_cap=10.0
    )
    assert meta["n_missing_moments"] == 20
    assert np.all(upper[:, :10] == 10.0)
    assert np.all(np.isfinite(lower[:, :10]))


def test_hail_core_crossing_is_counted_not_hidden():
    """A cap below the self-consistency floor is Huang's own failure case."""
    z = np.full((1, 10), 44.0)
    zdr = np.full((1, 10), 0.3)  # high Z with low ZDR drives the floor up
    floor = float(0.5 * bounds.self_consistency_kdp(44.0, 0.3))
    lower, upper, meta = bounds.kdp_bounds(
        z,
        zdr,
        sign_policy="rain_nonnegative",
        cap_above=None,
        ladder=((45.0, 0.5 * floor),),
    )
    assert meta["n_crossed"] == 10
    assert np.all(lower == upper)


def test_the_phase_floor_can_only_raise_the_lower_bound():
    z, zdr, height = _grid()
    dr_km = 0.1
    phidp = 2.0 * 4.0 * np.arange(z.shape[1]) * dr_km  # a strong 4 deg/km ramp
    phidp = np.tile(phidp, (z.shape[0], 1))
    without, _, _ = bounds.kdp_bounds(z, zdr, height_km=height, iso0_km=3.2)
    with_floor, _, _ = bounds.kdp_bounds(
        z,
        zdr,
        phidp=phidp,
        dr_km=dr_km,
        lsf_window_km=4.0,
        height_km=height,
        iso0_km=3.2,
    )
    assert np.all(with_floor >= without - 1e-12)
    assert np.any(with_floor > without)


def test_the_phase_floor_needs_the_phase():
    z, zdr, height = _grid()
    with pytest.raises(ValueError, match="requires both phidp and dr_km"):
        bounds.kdp_bounds(z, zdr, lsf_window_km=4.0, height_km=height, iso0_km=3.2)


def test_mismatched_moment_shapes_are_an_error():
    with pytest.raises(ValueError, match="does not match"):
        bounds.kdp_bounds(np.zeros((2, 5)), np.zeros((2, 6)))


def test_metadata_records_the_settings_used():
    z, zdr, height = _grid()
    _, _, meta = bounds.kdp_bounds(z, zdr, height_km=height, iso0_km=3.2, band="X")
    assert meta["band"] == "X"
    assert meta["coefficients"] == bounds.SELF_CONSISTENCY_COEFFICIENTS["X"]
    assert meta["sign_policy"] == "altitude_conditioned"
    assert meta["iso0_km"] == pytest.approx(3.2)
