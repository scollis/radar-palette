"""Tests for fold detection and repair against synthetic profiles of known truth.

Every case here is built from a phase profile whose unfolded values are known
exactly, then folded by construction. That is deliberate: unfolding is the one
stage in the chain where a wrong answer is indistinguishable from a right one on
real data, so it must be verified against truth before it is pointed at a radar.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.phase import unfold


def _ramp(n_gates=400, dr_km=0.1, kdp=1.5, start=5.0):
    """Monotonic true phase from a constant KDP, and its folded counterpart."""
    range_km = np.arange(n_gates) * dr_km
    truth = start + 2.0 * kdp * range_km
    folded = np.mod(truth, 360.0)
    return truth, folded


# --------------------------------------------------------------------------
# known-truth recovery
# --------------------------------------------------------------------------


def test_single_fold_is_recovered_exactly():
    truth, folded = _ramp(kdp=6.0, n_gates=400)  # rises to ~485 deg, one fold
    assert np.max(truth) > 360.0
    unfolded, meta = unfold.unfold_phidp(folded[None, :])
    assert meta["n_confirmed"] == 1
    assert np.allclose(unfolded[0], truth)


def test_two_folds_are_recovered_exactly():
    truth, folded = _ramp(kdp=6.0, n_gates=800)  # rises to ~965 deg, two folds
    assert np.max(truth) > 720.0
    unfolded, meta = unfold.unfold_phidp(folded[None, :])
    assert meta["n_confirmed"] == 2
    assert np.allclose(unfolded[0], truth)


def test_an_unfolded_ray_is_left_untouched():
    truth, _ = _ramp(kdp=1.5, n_gates=400)
    assert np.max(truth) < 360.0
    unfolded, meta = unfold.unfold_phidp(truth[None, :])
    assert meta["n_confirmed"] == 0
    assert np.allclose(unfolded[0], truth)


def test_recovery_under_noise_is_exact_except_at_the_boundary_itself():
    """The intrinsic limit of unfolding, measured rather than glossed.

    With per-gate phase noise, a ray whose true phase passes close to the
    ambiguity boundary crosses it several times, so *which* gate is the fold is
    genuinely ambiguous. Measured here: gates 294, 295 and 296 all cross, and
    exactly one gate ends up on the wrong branch --- off by exactly one whole
    ambiguity interval, never by a fraction of one.
    """
    truth, _ = _ramp(kdp=6.0, n_gates=800)
    rng = np.random.default_rng(0)
    noisy_truth = truth + rng.normal(0.0, 1.8, truth.size)
    noisy_folded = np.mod(noisy_truth, 360.0)
    unfolded, meta = unfold.unfold_phidp(noisy_folded[None, :])
    assert meta["n_confirmed"] == 2

    residual = unfolded[0] - noisy_truth
    wrong = np.flatnonzero(np.abs(residual) > 1e-9)
    assert wrong.size <= 3
    # Every error is a whole interval, so no gate carries a partial correction.
    # Compared as a ratio: a modulo comparison fails on floating-point residue.
    intervals = residual[wrong] / 360.0
    assert np.allclose(intervals, np.round(intervals))
    assert np.all(np.round(intervals) != 0)
    # And every error sits at a genuine boundary crossing.
    crossings = np.flatnonzero(np.diff(np.floor(noisy_truth / 360.0)) != 0)
    assert np.all(np.min(np.abs(wrong[:, None] - crossings[None, :]), axis=1) <= 2)


def test_a_180_degree_ambiguity_is_recovered_with_the_matching_wrap():
    truth, _ = _ramp(kdp=3.0, n_gates=400)  # rises to ~245 deg
    folded = np.mod(truth, 180.0)
    unfolded, meta = unfold.unfold_phidp(folded[None, :], wrap_deg=180.0)
    assert meta["n_confirmed"] == 1
    assert np.allclose(unfolded[0], truth)


def test_the_wrong_wrap_value_leaves_a_residual_step():
    """Why `wrap_deg` is a stated parameter and never inferred."""
    truth, _ = _ramp(kdp=3.0, n_gates=400)
    folded = np.mod(truth, 180.0)
    unfolded, _ = unfold.unfold_phidp(folded[None, :], wrap_deg=360.0)
    residual = unfolded[0] - truth
    assert np.ptp(residual) == pytest.approx(180.0, abs=1e-9)


# --------------------------------------------------------------------------
# the discriminator: persistence, not step size
# --------------------------------------------------------------------------


def test_a_single_gate_excursion_is_not_corrected():
    """The numpy.unwrap failure mode, isolated."""
    truth, _ = _ramp(kdp=1.0, n_gates=200)
    spiked = truth.copy()
    spiked[100] += 200.0
    unfolded, meta = unfold.unfold_phidp(spiked[None, :])
    assert meta["n_candidates"] >= 1
    assert meta["n_confirmed"] == 0
    assert meta["n_rejected_transient"] >= 1
    assert np.allclose(unfolded[0], spiked)


def test_numpy_unwrap_corrupts_the_same_profile():
    """Not a test of this package -- a test of the claim justifying it."""
    truth, _ = _ramp(kdp=1.0, n_gates=200)
    spiked = truth.copy()
    spiked[100] += 200.0
    naive = np.rad2deg(np.unwrap(np.deg2rad(spiked)))
    ours, _ = unfold.unfold_phidp(spiked[None, :])
    assert np.max(np.abs(naive - spiked)) > 100.0
    assert np.max(np.abs(ours[0] - spiked)) == 0.0


def test_a_short_run_of_excursion_gates_is_still_rejected():
    truth, _ = _ramp(kdp=1.0, n_gates=200)
    spiked = truth.copy()
    spiked[100:102] += 250.0
    _, meta = unfold.unfold_phidp(spiked[None, :], confirm_gates=4)
    assert meta["n_confirmed"] == 0


def test_confirm_gates_of_one_reintroduces_the_unwrap_failure():
    """The persistence window is load-bearing, not decoration."""
    truth, _ = _ramp(kdp=1.0, n_gates=200)
    spiked = truth.copy()
    spiked[100] += 200.0
    _, meta = unfold.unfold_phidp(spiked[None, :], confirm_gates=1)
    assert meta["n_confirmed"] > 0


# --------------------------------------------------------------------------
# direction policy
# --------------------------------------------------------------------------


def test_increasing_only_rejects_a_downward_correction():
    truth, _ = _ramp(kdp=1.0, n_gates=200)
    stepped = truth.copy()
    stepped[100:] += 300.0  # a sustained upward step, correctable only downward
    _, increasing = unfold.unfold_phidp(stepped[None, :], direction="increasing")
    _, both = unfold.unfold_phidp(stepped[None, :], direction="both")
    assert increasing["n_confirmed"] == 0
    assert increasing["n_rejected_direction"] == 1
    assert both["n_confirmed"] == 1


def test_unknown_direction_is_an_error():
    with pytest.raises(ValueError, match="direction must be one of"):
        unfold.unfold_phidp(np.zeros((1, 10)), direction="sideways")


# --------------------------------------------------------------------------
# statistics, and the decision they support
# --------------------------------------------------------------------------


def test_statistics_separate_folds_from_excursions():
    truth, folded = _ramp(kdp=6.0, n_gates=800)
    spiked = folded.copy()
    for gate in (100, 300, 500):
        spiked[gate] += 200.0
    stats = unfold.fold_statistics(spiked[None, :])
    assert stats["n_large_steps"] > stats["n_persistent"]
    assert stats["n_persistent"] == 2
    assert 0.0 < stats["persistent_fraction_of_large"] < 1.0


def test_statistics_on_a_clean_ray_report_no_large_steps():
    truth, _ = _ramp(kdp=1.0, n_gates=200)
    stats = unfold.fold_statistics(truth[None, :])
    assert stats["n_large_steps"] == 0
    assert stats["frac_large_steps"] == pytest.approx(0.0)
    assert np.isnan(stats["persistent_fraction_of_large"])
    assert stats["n_steps"] == 199


def test_observed_span_reports_the_dynamic_range_of_the_input():
    truth, folded = _ramp(kdp=6.0, n_gates=800)
    stats = unfold.fold_statistics(folded[None, :])
    assert stats["observed_span_deg"] < 360.0


# --------------------------------------------------------------------------
# degenerate rays
# --------------------------------------------------------------------------


def test_an_all_nan_ray_is_left_alone():
    phidp = np.full((3, 50), np.nan)
    unfolded, meta = unfold.unfold_phidp(phidp)
    assert meta["n_candidates"] == 0
    assert meta["n_confirmed"] == 0
    assert np.isnan(unfolded).all()


def test_a_ray_with_one_valid_gate_is_left_alone():
    phidp = np.full((1, 50), np.nan)
    phidp[0, 10] = 42.0
    unfolded, meta = unfold.unfold_phidp(phidp)
    assert meta["n_candidates"] == 0
    assert unfolded[0, 10] == pytest.approx(42.0)


def test_masked_gates_are_shifted_but_never_examined():
    """A gap must not create a spurious step, and must not be left behind."""
    truth, folded = _ramp(kdp=6.0, n_gates=800)
    mask = np.ones((1, 800), dtype=bool)
    mask[0, 200:220] = False
    unfolded, meta = unfold.unfold_phidp(folded[None, :], mask=mask)
    assert meta["n_confirmed"] == 2
    # Every gate, masked or not, sits on the branch the correction implies.
    assert np.allclose(unfolded[0], truth)


def test_mask_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="mask shape"):
        unfold.unfold_phidp(np.zeros((2, 5)), mask=np.ones((2, 6), dtype=bool))


def test_one_dimensional_input_is_an_error():
    with pytest.raises(ValueError, match="2-D"):
        unfold.unfold_phidp(np.zeros(10))


# --------------------------------------------------------------------------
# detection without modification
# --------------------------------------------------------------------------


def test_detect_reports_gate_indices_on_the_new_branch():
    truth, folded = _ramp(kdp=6.0, n_gates=800)
    detected = unfold.detect_folds(folded[None, :])
    gates = detected["fold_gates"][0]
    assert gates.size == 2
    # The recorded gate is the first on the new branch, so the raw step across
    # it is large and negative.
    for gate in gates:
        assert folded[gate] - folded[gate - 1] < -180.0
    assert (detected["fold_signs"][0] == 1).all()


def test_detect_does_not_modify_its_input():
    truth, folded = _ramp(kdp=6.0, n_gates=800)
    original = folded.copy()
    unfold.detect_folds(folded[None, :])
    assert np.array_equal(folded, original)


def test_multiple_rays_are_handled_independently():
    _, folded_two = _ramp(kdp=6.0, n_gates=800)
    truth_none, _ = _ramp(kdp=1.0, n_gates=800)
    sweep = np.stack([folded_two, truth_none])
    _, meta = unfold.unfold_phidp(sweep)
    assert meta["n_rays_with_folds"] == 1
    assert meta["n_confirmed"] == 2
