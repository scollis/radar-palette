"""Tests for system-phase estimation and the ray-to-ray consistency diagnostic.

These pin two things the module exists to get right: that the sweep-level value
is the median of the lowest rays (and NaN when too few rays support it), and
that the consistency diagnostic measures *offsets* rather than correlation --- a
constant per-ray offset is invisible to correlation, which is why an earlier
diagnostic built on correlation cleared a field that was streaked.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.phase import system


def _sweep(n_rays=90, n_gates=200, sysphi=12.0, kdp=0.5, dr_m=100.0, seed=0):
    """A sweep with a known system offset and a linear phase ramp beyond it."""
    rng = np.random.default_rng(seed)
    range_m = np.arange(n_gates) * dr_m
    ramp = 2.0 * kdp * (range_m / 1000.0)
    phidp = sysphi + ramp[None, :] + rng.normal(0.0, 1.8, size=(n_rays, n_gates))
    rhohv = np.full((n_rays, n_gates), 0.99)
    return phidp, rhohv, range_m


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


def test_rhohv_floor_excludes_low_correlation_gates():
    phidp = np.zeros((4, 10))
    rhohv = np.full((4, 10), 0.99)
    rhohv[:, 5:] = 0.5
    mask = system.valid_phase_mask(phidp, rhohv)
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()


def test_mask_without_rhohv_only_requires_finiteness():
    phidp = np.array([[1.0, np.nan, 3.0]])
    assert system.valid_phase_mask(phidp).tolist() == [[True, False, True]]


def test_mismatched_rhohv_shape_is_an_error():
    with pytest.raises(ValueError, match="shape"):
        system.valid_phase_mask(np.zeros((2, 3)), np.zeros((2, 4)))


# --------------------------------------------------------------------------
# the three estimators
# --------------------------------------------------------------------------


def test_block_recovers_a_known_offset():
    phidp, rhohv, range_m = _sweep(sysphi=12.0)
    result = system.system_phidp_block(phidp, range_m, 1000.0, rhohv=rhohv)
    assert result["n_bins"] == 10
    assert result["valid_bins"].min() == 10
    # The sweep value is the median of the lowest 30 of 90 ray medians, so it
    # sits below the true offset by a predictable amount rather than on it.
    assert result["sysphi"] == pytest.approx(12.0, abs=1.5)
    assert np.isfinite(result["sysphi_ray"]).all()


def test_first_recovers_a_known_offset():
    phidp, rhohv, range_m = _sweep(sysphi=-30.0)
    result = system.system_phidp_first(phidp, range_m, rhohv=rhohv)
    assert result["sysphi"] == pytest.approx(-30.0, abs=1.5)


def test_hist_leading_edge_sits_below_the_peak():
    phidp, rhohv, range_m = _sweep(sysphi=5.0, kdp=1.0)
    result = system.system_phidp_hist(phidp, rhohv=rhohv)
    # The ray accumulates propagation phase, so the histogram peak is above the
    # system offset while the leading edge is near it. That ordering is the
    # reason `sysphi` is aliased to the leading-edge estimate.
    assert result["sysphi_first"] < result["sysphi_peak"]
    assert result["sysphi_first"] == pytest.approx(5.0, abs=3.0)


def test_block_requires_a_contiguous_run():
    """A ray of alternating valid gates supports `first` but not `block`."""
    phidp = np.full((40, 40), 10.0)
    rhohv = np.full((40, 40), 0.99)
    rhohv[:, ::2] = 0.1
    range_m = np.arange(40) * 100.0
    block = system.system_phidp_block(phidp, range_m, 500.0, rhohv=rhohv)
    assert np.isnan(block["sysphi_ray"]).all()
    assert np.isnan(block["sysphi"])

    first = system.system_phidp_first(phidp, range_m, rhohv=rhohv, n_valid_bins=5)
    assert np.isfinite(first["sysphi"])


def test_block_length_longer_than_the_ray_is_an_error():
    with pytest.raises(ValueError, match="only"):
        system.system_phidp_block(np.zeros((3, 10)), np.arange(10) * 100.0, 100_000.0)


# --------------------------------------------------------------------------
# sweep-level aggregation
# --------------------------------------------------------------------------


def test_sweep_value_is_the_median_of_the_lowest_rays():
    ray_values = np.arange(100.0)  # 0 .. 99
    assert system._aggregate_sysphi(ray_values, 10) == pytest.approx(4.5)
    assert system._aggregate_sysphi(ray_values, 100) == pytest.approx(49.5)


def test_too_few_valid_rays_gives_nan_not_a_number():
    """The guard that stops an offset from three rays reaching a whole volume."""
    ray_values = np.full(50, np.nan)
    ray_values[:5] = 7.0
    assert np.isnan(system._aggregate_sysphi(ray_values, 30))
    assert system._aggregate_sysphi(ray_values, 5) == pytest.approx(7.0)


def test_applying_a_nan_sweep_value_raises():
    estimate = {"sysphi": np.nan, "sysphi_ray": np.full(4, np.nan)}
    with pytest.raises(ValueError, match="too few rays"):
        system.apply_system_phidp(np.zeros((4, 5)), estimate)


# --------------------------------------------------------------------------
# ray-to-ray consistency
# --------------------------------------------------------------------------


def test_identical_rays_have_zero_spread():
    result = system.ray_consistency(np.full(50, 8.0))
    assert result["robust_sd_deg"] == pytest.approx(0.0)
    assert result["median_adjacent_jump_deg"] == pytest.approx(0.0)
    assert result["n_valid"] == 50


def test_a_constant_offset_between_ray_groups_is_seen_as_spread():
    """Correlation cannot see this; the offset statistics must."""
    ray_values = np.concatenate([np.full(30, 0.0), np.full(30, 20.0)])
    result = system.ray_consistency(ray_values)
    assert result["p5_to_p95_deg"] == pytest.approx(20.0)
    assert result["robust_sd_deg"] > 10.0


def test_per_ray_is_not_justified_when_spread_matches_estimator_noise():
    rng = np.random.default_rng(3)
    sigma_phi, n_eff = 1.8, 10
    expected = np.sqrt(np.pi / 2.0) * sigma_phi / np.sqrt(n_eff)
    ray_values = rng.normal(0.0, expected, size=400)
    result = system.ray_consistency(ray_values, sigma_phi_deg=sigma_phi, n_eff=n_eff)
    assert result["excess_ratio"] == pytest.approx(1.0, abs=0.2)
    assert result["per_ray_justified"] is False


def test_per_ray_is_justified_when_spread_greatly_exceeds_noise():
    rng = np.random.default_rng(4)
    ray_values = rng.normal(0.0, 10.0, size=400)
    result = system.ray_consistency(ray_values, sigma_phi_deg=1.8, n_eff=10)
    assert result["excess_ratio"] > 2.0
    assert result["per_ray_justified"] is True


def test_verdict_is_withheld_without_a_noise_estimate():
    result = system.ray_consistency(np.arange(50.0))
    assert result["expected_noise_deg"] is None
    assert result["per_ray_justified"] is None


def test_consistency_on_too_few_rays_returns_nan_fields():
    result = system.ray_consistency(np.array([1.0, 2.0]))
    assert result["n_valid"] == 2
    assert np.isnan(result["robust_sd_deg"])


# --------------------------------------------------------------------------
# applying the correction
# --------------------------------------------------------------------------


def test_sweep_correction_subtracts_one_constant_from_every_ray():
    phidp = np.arange(12.0).reshape(3, 4)
    corrected, offset = system.apply_system_phidp(phidp, 5.0)
    assert np.allclose(corrected, phidp - 5.0)
    assert offset.shape == (3, 1)
    assert np.allclose(offset, 5.0)


def test_per_ray_correction_uses_each_ray_and_falls_back_to_the_sweep():
    estimate = {"sysphi": 1.0, "sysphi_ray": np.array([10.0, np.nan, -4.0])}
    phidp = np.zeros((3, 2))
    corrected, offset = system.apply_system_phidp(phidp, estimate, per_ray=True)
    assert np.allclose(offset.ravel(), [10.0, 1.0, -4.0])
    assert np.allclose(corrected[1], -1.0)


def test_kdp_is_blind_to_the_sweep_versus_per_ray_choice():
    """The claim made in the module docstring, asserted rather than asserted-to.

    A per-ray additive constant is annihilated by a range derivative, so the two
    corrections must produce gate-to-gate differences that are identical.
    """
    phidp, rhohv, range_m = _sweep(seed=11)
    estimate = system.system_phidp(
        phidp, range_m, method="block", rhohv=rhohv, block_length_m=1000.0
    )
    sweep_corrected, _ = system.apply_system_phidp(phidp, estimate, per_ray=False)
    ray_corrected, _ = system.apply_system_phidp(phidp, estimate, per_ray=True)
    assert np.allclose(np.diff(sweep_corrected, axis=1), np.diff(ray_corrected, axis=1))


# --------------------------------------------------------------------------
# the dispatcher
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", system.SYSTEM_PHIDP_METHODS)
def test_every_method_returns_the_common_keys(method):
    phidp, rhohv, range_m = _sweep()
    result = system.system_phidp(phidp, range_m, method=method, rhohv=rhohv)
    for key in ("sysphi_ray", "sysphi", "consistency", "method"):
        assert key in result
    assert result["consistency"]["n_rays"] == phidp.shape[0]


def test_unknown_method_is_an_error():
    with pytest.raises(ValueError, match="method must be one of"):
        system.system_phidp(np.zeros((4, 5)), np.arange(5.0), method="magic")


def test_block_without_range_is_an_error():
    with pytest.raises(ValueError, match="requires range_m"):
        system.system_phidp(np.zeros((4, 5)), method="block")


def test_hist_withholds_the_noise_verdict():
    """No per-ray n_eff is defined for a distribution-edge estimator."""
    phidp, rhohv, range_m = _sweep()
    result = system.system_phidp(
        phidp, range_m, method="hist", rhohv=rhohv, sigma_phi_deg=1.8
    )
    assert result["consistency"]["per_ray_justified"] is None


def test_all_masked_sweep_yields_no_estimate():
    phidp = np.full((60, 50), np.nan)
    range_m = np.arange(50) * 100.0
    result = system.system_phidp(phidp, range_m, method="block", block_length_m=500.0)
    assert np.isnan(result["sysphi_ray"]).all()
    assert np.isnan(result["sysphi"])
    assert result["consistency"]["n_valid"] == 0
