"""Tests for the bounded-KDP linear program.

Three of these are here because of a specific failure mode rather than for
coverage: offset invariance, which a zero-fill of invalid gates is known to
break and which is measured here at 2.9e-13 with full window support and
20 deg/km without it; the box guarantee on the *reported* field; and recovery of
a synthetic Gaussian of known width and amplitude --- "it ran" is not a result.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from radar_palette.phase import linear_program as lp

DR_KM = 0.1
SMOOTHING_KM = 1.0


def _gaussian_ray(
    n_gates=400, amplitude=3.0, sigma_km=1.0, centre_km=20.0, noise_deg=0.0, seed=0
):
    """A ray whose true KDP is a Gaussian of known amplitude and width.

    Returns the true KDP and the phase it implies. ``KDP = 0.5 dPhi/dr``, so the
    phase is twice the running integral of KDP.
    """
    range_km = np.arange(n_gates) * DR_KM
    kdp_true = amplitude * np.exp(-0.5 * ((range_km - centre_km) / sigma_km) ** 2)
    trapezoid = 0.5 * (kdp_true[1:] + kdp_true[:-1])
    phidp = 2.0 * DR_KM * np.concatenate([[0.0], np.cumsum(trapezoid)])
    if noise_deg:
        phidp = phidp + np.random.default_rng(seed).normal(0.0, noise_deg, n_gates)
    return range_km, kdp_true, phidp


def _wide_box(n_gates, low=-5.0, high=20.0):
    return np.full(n_gates, low), np.full(n_gates, high)


# --------------------------------------------------------------------------
# the operator
# --------------------------------------------------------------------------


def test_derivative_weights_reproduce_the_giangrande_d5_filter():
    assert np.allclose(lp.savgol_derivative_weights(5), [-0.2, -0.1, 0.0, 0.1, 0.2])


def test_derivative_weights_sum_to_zero_at_every_length():
    """The property that makes the retrieval offset-invariant."""
    for length in (3, 5, 11, 21, 51):
        assert lp.savgol_derivative_weights(length).sum() == pytest.approx(0.0)


def test_even_or_short_filter_lengths_are_rejected():
    for bad in (2, 4, 1):
        with pytest.raises(ValueError, match="odd integer"):
            lp.savgol_derivative_weights(bad)


def test_smoothing_length_is_quantised_to_odd_gates_and_reported():
    length, realised = lp.smoothing_length_gates(1.0, 0.1)
    assert length == 11 and realised == pytest.approx(1.1)
    length, realised = lp.smoothing_length_gates(2.0, 0.25)
    assert length == 9 and realised == pytest.approx(2.25)
    # A request finer than three gates cannot be honoured; the floor is reported
    # rather than silently applied.
    length, realised = lp.smoothing_length_gates(0.1, 0.25)
    assert length == 3 and realised == pytest.approx(0.75)


def test_smoothing_length_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="smoothing_km"):
        lp.smoothing_length_gates(0.0, 0.1)
    with pytest.raises(ValueError, match="dr_km"):
        lp.smoothing_length_gates(1.0, -0.1)


def test_operator_differentiates_a_linear_ramp_exactly():
    n_gates, kdp = 60, 1.7
    phase = 2.0 * kdp * np.arange(n_gates) * DR_KM
    operator = lp.derivative_matrix(n_gates, DR_KM, 11)
    assert np.allclose(operator @ phase, kdp)


def test_operator_annihilates_a_constant():
    operator = lp.derivative_matrix(60, DR_KM, 11)
    assert np.allclose(operator @ np.full(60, 42.0), 0.0)


def test_a_filter_longer_than_the_ray_is_an_error():
    with pytest.raises(ValueError, match="does not fit"):
        lp.derivative_matrix(5, DR_KM, 11)


# --------------------------------------------------------------------------
# offset invariance -- tested, not assumed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [0.5, -37.0, 360.0, 1000.0])
def test_retrieval_is_invariant_to_an_additive_phase_constant(shift):
    _, _, phidp = _gaussian_ray(noise_deg=1.0, seed=1)
    low, high = _wide_box(phidp.size)
    base = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    shifted = lp.solve_ray(phidp + shift, low, high, DR_KM, SMOOTHING_KM)
    assert base["status"] == shifted["status"] == "solved"
    finite = np.isfinite(base["kdp"])
    assert np.allclose(base["kdp"][finite], shifted["kdp"][finite], atol=1e-8)


def test_invariance_holds_with_a_binding_lower_bound():
    """The interesting case: the box is active, so the solution is not x = b."""
    _, _, phidp = _gaussian_ray(noise_deg=2.0, seed=2)
    low = np.zeros(phidp.size)
    high = np.full(phidp.size, 4.0)
    base = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    shifted = lp.solve_ray(phidp + 500.0, low, high, DR_KM, SMOOTHING_KM)
    finite = np.isfinite(base["kdp"])
    assert (base["kdp"][finite] < -1e-9).sum() == 0
    assert np.allclose(base["kdp"][finite], shifted["kdp"][finite], atol=1e-8)


@pytest.mark.parametrize("gap", [1, 5, 20, 60])
def test_invariance_holds_across_masked_gaps(gap):
    """Invariance survives a gap only because the gap is not reported.

    A gate with no weight leaves a free phase variable, so every window
    containing one has a degenerate KDP that the solver puts on a box edge. The
    default support requirement excludes exactly those windows.
    """
    _, _, phidp = _gaussian_ray(noise_deg=1.0, seed=3)
    phidp[150 : 150 + gap] = np.nan
    low, high = _wide_box(phidp.size)
    base = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    shifted = lp.solve_ray(phidp + 250.0, low, high, DR_KM, SMOOTHING_KM)
    finite = np.isfinite(base["kdp"]) & np.isfinite(shifted["kdp"])
    assert finite.sum() > 0
    assert np.allclose(base["kdp"][finite], shifted["kdp"][finite], atol=1e-9)

    # Every window overlapping the gap is withheld, not filled.
    half = (base["length"] - 1) // 2
    touched = slice(150 - half, 150 + gap + half)
    assert not np.isfinite(base["kdp"][touched]).any()
    assert base["n_unsupported_windows"] == gap + base["length"] - 1


def test_relaxing_the_support_requirement_reintroduces_the_degeneracy():
    """The measurement that set the default, kept as a regression guard.

    With a data majority in the window but a free variable inside it, the
    reported KDP is arbitrary within the box: adding a constant to the input
    phase moves it by the full width of the bounds.
    """
    _, _, phidp = _gaussian_ray(noise_deg=1.0, seed=3)
    phidp[150:155] = np.nan
    low, high = _wide_box(phidp.size, low=-5.0, high=20.0)
    kwargs = {"min_window_weight_fraction": 0.5}
    base = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM, **kwargs)
    shifted = lp.solve_ray(phidp + 250.0, low, high, DR_KM, SMOOTHING_KM, **kwargs)
    finite = np.isfinite(base["kdp"]) & np.isfinite(shifted["kdp"])
    assert np.max(np.abs(base["kdp"][finite] - shifted["kdp"][finite])) > 10.0


# --------------------------------------------------------------------------
# the box guarantee
# --------------------------------------------------------------------------


def test_the_reported_kdp_obeys_the_box_it_was_given():
    _, _, phidp = _gaussian_ray(noise_deg=3.0, seed=4)
    rng = np.random.default_rng(5)
    low = rng.uniform(-1.0, 0.0, phidp.size)
    high = low + rng.uniform(0.5, 2.0, phidp.size)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    assert result["status"] == "solved"
    finite = np.isfinite(result["kdp"])
    assert np.all(result["kdp"][finite] >= low[finite] - 1e-7)
    assert np.all(result["kdp"][finite] <= high[finite] + 1e-7)


def test_a_tight_box_pins_the_solution_to_it():
    _, _, phidp = _gaussian_ray(noise_deg=2.0, seed=6)
    target = 1.25
    low = np.full(phidp.size, target)
    high = np.full(phidp.size, target)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    finite = np.isfinite(result["kdp"])
    assert np.allclose(result["kdp"][finite], target, atol=1e-6)


def test_a_crossed_box_is_reported_infeasible_rather_than_solved():
    _, _, phidp = _gaussian_ray()
    low = np.full(phidp.size, 2.0)
    high = np.full(phidp.size, 1.0)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    assert result["status"] == "infeasible"
    assert "lower bound exceeds upper" in result["message"]


def test_a_wide_box_is_always_feasible():
    """The operator has full row rank, so any ordered box is attainable."""
    rng = np.random.default_rng(7)
    phidp = rng.uniform(0.0, 360.0, 300)
    low, high = _wide_box(300)
    assert lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)["status"] == "solved"


# --------------------------------------------------------------------------
# recovery of a known Gaussian
# --------------------------------------------------------------------------


def _fit_gaussian(range_km, kdp):
    finite = np.isfinite(kdp)

    def model(x, amplitude, centre, sigma, offset):
        return amplitude * np.exp(-0.5 * ((x - centre) / sigma) ** 2) + offset

    guess = (np.nanmax(kdp), range_km[np.nanargmax(kdp)], 1.0, 0.0)
    popt, _ = curve_fit(model, range_km[finite], kdp[finite], p0=guess, maxfev=20000)
    return {"amplitude": popt[0], "centre": popt[1], "sigma": abs(popt[2])}


def test_a_noiseless_gaussian_is_recovered_at_the_expected_broadening():
    """Width and peak, not just "it ran".

    An 11-gate derivative filter broadens a Gaussian by the variance of its own
    window: the recovered sigma should be about
    ``sqrt(sigma^2 + (L*dr)^2 / 12)``, which for sigma = 1 km and the realised
    1.1 km filter is 1.049 km, and the peak should fall by the same ratio since
    the integral is conserved. The assertion below recomputes this from the
    realised smoothing length rather than using the number quoted here.
    """
    range_km, kdp_true, phidp = _gaussian_ray(amplitude=3.0, sigma_km=1.0)
    low, high = _wide_box(phidp.size)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    assert result["status"] == "solved"
    fit = _fit_gaussian(range_km, result["kdp"])

    expected_sigma = np.sqrt(1.0**2 + (result["realised_smoothing_km"] ** 2) / 12.0)
    assert fit["sigma"] == pytest.approx(expected_sigma, rel=0.05)
    assert fit["centre"] == pytest.approx(20.0, abs=0.1)
    assert fit["amplitude"] == pytest.approx(3.0 * 1.0 / expected_sigma, rel=0.06)

    # The integral is the quantity a derivative filter conserves, so it is the
    # strictest check available on amplitude and width jointly.
    finite = np.isfinite(result["kdp"])
    assert np.trapezoid(result["kdp"][finite], range_km[finite]) == pytest.approx(
        np.trapezoid(kdp_true, range_km), rel=0.02
    )


def test_the_gaussian_survives_realistic_phase_noise():
    range_km, kdp_true, phidp = _gaussian_ray(
        amplitude=3.0, sigma_km=1.0, noise_deg=1.8, seed=8
    )
    low = np.zeros(phidp.size)
    high = np.full(phidp.size, 10.0)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    fit = _fit_gaussian(range_km, result["kdp"])
    assert fit["amplitude"] == pytest.approx(3.0, rel=0.25)
    assert fit["sigma"] == pytest.approx(1.0, rel=0.35)
    assert fit["centre"] == pytest.approx(20.0, abs=0.5)


def test_a_longer_smoothing_length_broadens_the_recovered_core():
    """The measurable cost of an over-long smoothing length, in one assertion."""
    range_km, _, phidp = _gaussian_ray(amplitude=3.0, sigma_km=1.0)
    low, high = _wide_box(phidp.size)
    widths = []
    for smoothing_km in (1.0, 2.0, 4.66):
        result = lp.solve_ray(phidp, low, high, DR_KM, smoothing_km)
        widths.append(_fit_gaussian(range_km, result["kdp"])["sigma"])
    assert widths[0] < widths[1] < widths[2]
    assert widths[2] / widths[0] > 1.3


# --------------------------------------------------------------------------
# degenerate rays
# --------------------------------------------------------------------------


def test_an_all_masked_ray_returns_nan_with_a_named_reason():
    phidp = np.full(300, np.nan)
    low, high = _wide_box(300)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    assert result["status"] == "no_valid_gates"
    assert result["n_weighted"] == 0
    assert np.isnan(result["kdp"]).all()


def test_a_ray_with_fewer_gates_than_the_filter_is_named_not_crashed():
    phidp = np.full(300, np.nan)
    phidp[:4] = np.arange(4.0)
    low, high = _wide_box(300)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    assert result["status"] == "too_few_valid_gates"
    assert "11-gate filter" in result["message"]


def test_an_all_noise_ray_solves_and_stays_inside_the_box():
    """Pure noise must produce a bounded field, not a spike train."""
    rng = np.random.default_rng(9)
    phidp = rng.uniform(0.0, 360.0, 400)
    low = np.zeros(400)
    high = np.full(400, 2.0)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    assert result["status"] == "solved"
    finite = np.isfinite(result["kdp"])
    assert np.all(result["kdp"][finite] >= -1e-7)
    assert np.all(result["kdp"][finite] <= 2.0 + 1e-7)


def test_zero_weights_are_equivalent_to_masking():
    _, _, phidp = _gaussian_ray(noise_deg=1.0, seed=10)
    low, high = _wide_box(phidp.size)
    weights = np.ones(phidp.size)
    weights[150:170] = 0.0
    masked = phidp.copy()
    masked[150:170] = np.nan
    by_weight = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM, weights=weights)
    by_mask = lp.solve_ray(masked, low, high, DR_KM, SMOOTHING_KM)
    assert by_weight["n_weighted"] == by_mask["n_weighted"]
    finite = np.isfinite(by_mask["kdp"])
    assert np.array_equal(finite, np.isfinite(by_weight["kdp"]))
    assert np.allclose(by_weight["kdp"][finite], by_mask["kdp"][finite], atol=1e-7)


def test_edge_gates_are_nan_not_extrapolated():
    _, _, phidp = _gaussian_ray()
    low, high = _wide_box(phidp.size)
    result = lp.solve_ray(phidp, low, high, DR_KM, SMOOTHING_KM)
    half = (result["length"] - 1) // 2
    assert np.isnan(result["kdp"][:half]).all()
    assert np.isnan(result["kdp"][-half:]).all()
    assert np.isfinite(result["kdp"][half:-half]).all()


# --------------------------------------------------------------------------
# the sweep driver
# --------------------------------------------------------------------------


def test_sweep_driver_reports_a_status_for_every_ray():
    _, _, good = _gaussian_ray(noise_deg=1.0, seed=12)
    sweep = np.stack([good, np.full_like(good, np.nan), good])
    low = np.full(sweep.shape, -5.0)
    high = np.full(sweep.shape, 20.0)
    kdp, retrieved, meta = lp.kdp_linear_program(sweep, low, high, DR_KM, SMOOTHING_KM)
    assert len(meta["status"]) == 3
    assert meta["n_solved"] == 2
    assert meta["status"][1] == "no_valid_gates"
    assert 1 in meta["messages"] or meta["status"][1] != "solved"
    assert np.isnan(kdp[1]).all()
    assert kdp.shape == retrieved.shape == sweep.shape


def test_sweep_metadata_records_the_requested_and_realised_smoothing():
    _, _, ray = _gaussian_ray()
    sweep = ray[None, :]
    low = np.full(sweep.shape, -5.0)
    high = np.full(sweep.shape, 20.0)
    _, _, meta = lp.kdp_linear_program(sweep, low, high, 0.25, 2.0)
    assert meta["requested_smoothing_km"] == pytest.approx(2.0)
    assert meta["realised_smoothing_km"] == pytest.approx(2.25)
    assert meta["length"] == 9


def test_sweep_driver_rejects_mismatched_bound_shapes():
    with pytest.raises(ValueError, match="does not match"):
        lp.kdp_linear_program(
            np.zeros((2, 50)), np.zeros((2, 40)), np.zeros((2, 50)), DR_KM, 1.0
        )


def test_smoothing_length_has_no_default():
    """A required argument, by design -- see the module docstring."""
    with pytest.raises(TypeError):
        lp.kdp_linear_program(
            np.zeros((1, 50)), np.zeros((1, 50)), np.ones((1, 50)), DR_KM
        )
