"""Tests for the temporal gate-ID tooling.

These pin the measurement conventions as much as the arithmetic, because the
easiest way to get a wrong answer here is to measure the right quantity over
the wrong population: an unmasked transition matrix reports near-perfect
persistence for the trivial reason that most of a scan is empty sky.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.gateid import single as core
from radar_palette.gateid import temporal


def _codes(*names, n=1):
    return np.array([core.NAME_TO_CODE[x] for x in names] * n, dtype="i2")


# --------------------------------------------------------------------------
# transition matrix
# --------------------------------------------------------------------------


def test_perfect_persistence_gives_identity():
    labels = _codes("moderate_rain", "ice_snow", "light_rain", n=40)
    m, order, n = temporal.transition_matrix(labels, labels)
    idx = [order.index(k) for k in ("moderate_rain", "ice_snow", "light_rain")]
    for i in idx:
        assert m[i, i] == pytest.approx(1.0)
    assert n == labels.size


def test_no_scatter_is_masked_by_default():
    """The 80%-empty-sky trap: masking must be on unless asked otherwise.

    Without masking, a scan that is almost all no_scatter reports near-perfect
    persistence and tells you nothing about hydrometeors.
    """
    labels = _codes("no_scatter", n=100)
    labels = np.concatenate([labels, _codes("moderate_rain", n=4)])
    _, order, n = temporal.transition_matrix(labels, labels)
    assert "no_scatter" not in order
    assert n == 4

    _, order_all, n_all = temporal.transition_matrix(labels, labels, mask_classes=())
    assert "no_scatter" in order_all
    assert n_all == 104


def test_row_normalisation_is_conditional_probability():
    t0 = _codes("moderate_rain", n=10)
    t1 = np.concatenate([_codes("moderate_rain", n=7), _codes("ice_snow", n=3)])
    m, order, _ = temporal.transition_matrix(t0, t1)
    i = order.index("moderate_rain")
    j = order.index("ice_snow")
    assert m[i, i] == pytest.approx(0.7)
    assert m[i, j] == pytest.approx(0.3)
    assert np.nansum(m[i]) == pytest.approx(1.0)


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape"):
        temporal.transition_matrix(
            _codes("rain" if False else "light_rain", n=4), _codes("light_rain", n=5)
        )


# --------------------------------------------------------------------------
# persistence against the marginal
# --------------------------------------------------------------------------


def test_persistence_is_reported_against_chance():
    """A class filling the scan persists trivially; skill must show that."""
    t0 = _codes("light_rain", n=99)
    t0 = np.concatenate([t0, _codes("ice_snow", n=1)])
    t1 = t0.copy()
    skill = temporal.persistence_skill(t0, t1)
    assert skill["light_rain"]["persistence"] == pytest.approx(1.0)
    assert skill["light_rain"]["marginal"] == pytest.approx(0.99)
    # Perfect persistence is still skill 1, but the marginal is right there
    # next to it so nobody reads 0.99 as impressive.
    assert skill["light_rain"]["skill"] == pytest.approx(1.0)


def test_no_skill_when_transitions_are_random():
    rng = np.random.default_rng(0)
    names = ["light_rain", "moderate_rain", "ice_snow"]
    t0 = np.array(
        [core.NAME_TO_CODE[names[i]] for i in rng.integers(0, 3, 6000)], dtype="i2"
    )
    t1 = np.array(
        [core.NAME_TO_CODE[names[i]] for i in rng.integers(0, 3, 6000)], dtype="i2"
    )
    skill = temporal.persistence_skill(t0, t1)
    for name in names:
        assert abs(skill[name]["skill"]) < 0.1, (name, skill[name])


def test_persistence_reports_sample_size():
    t0 = _codes("ice_snow", n=17)
    skill = temporal.persistence_skill(t0, t0)
    assert skill["ice_snow"]["n"] == 17


# --------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------


def test_composition_drift_is_zero_for_identical_scans():
    labels = _codes("light_rain", "ice_snow", n=30)
    assert temporal.composition_drift(labels, labels) == pytest.approx(0.0)


def test_composition_drift_is_one_for_disjoint_scans():
    a = _codes("light_rain", n=50)
    b = _codes("ice_snow", n=50)
    assert temporal.composition_drift(a, b) == pytest.approx(1.0)


def test_composition_drift_ignores_bulk_translation():
    """The measure must not react to a storm simply moving.

    Gate-wise agreement collapses when a field shifts; composition does not,
    which is the whole point of having both.
    """
    field = np.full(400, core.NAME_TO_CODE["no_scatter"], dtype="i2")
    field[100:200] = core.NAME_TO_CODE["moderate_rain"]
    shifted = np.roll(field, 60)
    assert temporal.composition_drift(field, shifted) == pytest.approx(0.0)
    # ... whereas gate-wise identity is badly broken by the same shift.
    both = np.isin(field, [core.NAME_TO_CODE["moderate_rain"]]) & np.isin(
        shifted, [core.NAME_TO_CODE["moderate_rain"]]
    )
    assert both.mean() < 0.2


# --------------------------------------------------------------------------
# accumulated clutter prior
# --------------------------------------------------------------------------


def test_clutter_prior_finds_stationary_gates():
    """Clutter recurs at fixed gates; weather moves through."""
    shape = (4, 50)
    stack = []
    rng = np.random.default_rng(3)
    for _ in range(12):
        a = np.full(shape, core.NAME_TO_CODE["no_scatter"], dtype="i2")
        a[:, 2:5] = core.NAME_TO_CODE["clutter"]  # always here
        start = rng.integers(10, 40)  # wanders
        a[:, start : start + 3] = core.NAME_TO_CODE["moderate_rain"]
        stack.append(a)
    prior, n = temporal.accumulate_clutter_prior(stack)
    assert n == 12
    assert np.allclose(prior[:, 2:5], 1.0)
    assert prior[:, 20:30].max() < 0.5


def test_clutter_prior_refuses_a_thin_stack():
    """Too few volumes must return NaN, not a confident-looking number."""
    stack = [np.full((2, 5), core.NAME_TO_CODE["clutter"], dtype="i2")] * 3
    prior, n = temporal.accumulate_clutter_prior(stack, min_volumes=5)
    assert n == 3
    assert np.isnan(prior).all()


def test_clutter_prior_requires_one_geometry():
    stack = [np.zeros((2, 5), dtype="i2"), np.zeros((2, 6), dtype="i2")]
    with pytest.raises(ValueError, match="geometry"):
        temporal.accumulate_clutter_prior(stack)


# --------------------------------------------------------------------------
# the one-way prior
# --------------------------------------------------------------------------


def _ambiguous_features(n):
    """Gates where two classes score closely, so a prior can matter."""
    return {
        "z": np.full(n, 22.0),
        "rhohv": np.full(n, 0.965),
        "zdr": np.full(n, 0.55),
        "snr": np.full(n, 25.0),
        "temp": np.full(n, 1.0),
        "hoi0_km": np.full(n, -0.2),
        "rhohv_texture": np.full(n, 0.02),
    }


def test_prior_only_ever_adds_evidence():
    """It may support a class, never argue against one."""
    n = 40
    feats = _ambiguous_features(n)
    base, info = core.classify(feats, min_run=1)
    prev = np.full(n, core.NAME_TO_CODE["melting_wet"], dtype="i2")
    prior = temporal.temporal_prior(prev, "melting_wet", weight=0.2)
    adjusted = prior(info["scores"], info["order"], info["margin"])
    col = info["order"].index("melting_wet")
    others = [k for k in range(adjusted.shape[1]) if k != col]
    assert np.all(adjusted[:, col] >= info["scores"][:, col])
    assert np.allclose(adjusted[:, others], info["scores"][:, others])


def test_prior_is_vetoed_where_the_instant_answer_is_confident():
    n = 30
    feats = {
        "z": np.full(n, 48.0),
        "rhohv": np.full(n, 0.99),
        "zdr": np.full(n, 1.4),
        "snr": np.full(n, 45.0),
        "temp": np.full(n, 18.0),
        "hoi0_km": np.full(n, -3.0),
        "rhohv_texture": np.full(n, 0.004),
    }
    _, info = core.classify(feats, min_run=1)
    prev = np.full(n, core.NAME_TO_CODE["biological"], dtype="i2")
    prior = temporal.temporal_prior(prev, "biological", weight=0.5, veto_margin=0.05)
    adjusted = prior(info["scores"], info["order"], info["margin"])
    assert np.allclose(adjusted, info["scores"])


def test_prior_cannot_resurrect_a_noise_floor_gate():
    """A gate rejected as noise stays rejected, whatever history says."""
    n = 25
    feats = {
        "z": np.full(n, 10.0),
        "rhohv": np.full(n, 0.97),
        "zdr": np.full(n, 0.3),
        "snr": np.full(n, -8.0),
        "temp": np.full(n, 12.0),
    }
    prev = np.full(n, core.NAME_TO_CODE["light_rain"], dtype="i2")
    prior = temporal.temporal_prior(prev, "light_rain", weight=0.9, veto_margin=1.0)
    codes, info = temporal.apply_temporal_prior(feats, [prior], min_run=1)
    assert (codes == core.NAME_TO_CODE["no_scatter"]).all()


def test_prior_reports_how_many_gates_it_moved():
    """If the prior is deciding rather than tie-breaking, that must be visible."""
    n = 40
    feats = _ambiguous_features(n)
    prev = np.full(n, core.NAME_TO_CODE["melting_wet"], dtype="i2")
    prior = temporal.temporal_prior(prev, "melting_wet", weight=0.9, veto_margin=1.0)
    _, info = temporal.apply_temporal_prior(feats, [prior], min_run=1)
    assert "n_prior_changed" in info
    assert info["n_prior_changed"] >= 0


def test_zero_weight_prior_is_a_no_op():
    n = 30
    feats = _ambiguous_features(n)
    base, _ = core.classify(feats, min_run=1)
    prev = np.full(n, core.NAME_TO_CODE["ice_snow"], dtype="i2")
    prior = temporal.temporal_prior(prev, "ice_snow", weight=0.0)
    codes, info = temporal.apply_temporal_prior(feats, [prior], min_run=1)
    assert np.array_equal(codes, base)
    assert info["n_prior_changed"] == 0
