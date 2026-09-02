"""Differential-phase unfolding, with folds distinguished from phase excursions.

A receiver reports differential phase modulo an ambiguity interval, so a ray
whose propagation phase accumulates past the interval reappears at the bottom of
it. Because KDP is a range derivative, an unrepaired fold becomes a single
enormous spike in KDP, and a *wrongly* repaired one becomes a step in phase that
biases every gate beyond it.

Why this is not :func:`numpy.unwrap`
------------------------------------
:func:`numpy.unwrap` corrects every step larger than the threshold. That is the
wrong rule here, because two physically distinct things produce a large
gate-to-gate step in measured phase:

* a **fold**, which is a permanent branch change --- the phase stays on the new
  branch for the rest of the ray, and
* a **backscatter-phase excursion** (the ``delta`` term) or a noise spike, which
  is transient --- the phase returns to the old branch within a few gates.

Applied blind to a stream whose large steps were excursions rather than folds,
unwrapping has been measured to inject phase errors of order thousands of
degrees: it destroys the data it was meant to repair. The discriminator used
here is **persistence**: a candidate is accepted only if the median phase over
the next ``confirm_gates`` gates has genuinely moved to the new branch. A
one-gate spike, however large, is left alone for the estimator's L1 data term to
absorb.

Measure before you correct
--------------------------
:func:`fold_statistics` is the intended first call. It reports how many
gate-to-gate steps exceed the threshold and, separately, how many of those
persist. A stream with many large steps of which almost none persist has
excursions, not folds, and must not be unfolded. Deciding that from the numbers
costs one call; discovering it from a KDP field costs a session.

The ambiguity interval is a parameter
-------------------------------------
``wrap_deg`` is never inferred from the data. :data:`DEFAULT_WRAP_DEG` is 360
degrees, which is the ambiguity of a receiver measuring the H--V phase
difference on simultaneous transmission --- the mode of every radar this package
has been exercised against. Alternating transmission halves it to 180. The
default is stated rather than detected because the failure it guards against is
silent: correcting with the wrong interval leaves a residual step of the
difference at every repaired gate.

What this module is blind to
----------------------------
It cannot see a fold that occurs inside a masked gap, because there is no step
to measure across the gap --- the phase simply resumes on the far side, already
on the new branch, and looks continuous with itself. It also cannot see a fold
in the first gate of a ray, since a fold is defined by a step and the first gate
has nothing before it; such a ray is offset by one whole interval throughout,
which no range-derivative diagnostic can detect.

A third limit is intrinsic rather than a design choice: when the true phase
passes close to the ambiguity boundary, per-gate noise makes it cross several
times, so *which* gate is the fold is genuinely undetermined. Measured on a
synthetic ray of known truth with 1.8 degrees of per-gate noise and two folds in
800 gates, one gate landed on the wrong branch --- off by exactly one whole
interval, adjacent to a real crossing. Errors of this kind are always a whole
interval, never a fraction of one, which is what makes them recognisable.

All three are reasons the sign and magnitude bounds in
:mod:`radar_palette.phase.bounds` matter: they are the only downstream check on
a branch this module got wrong.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_CONFIRM_GATES",
    "DEFAULT_WRAP_DEG",
    "UNFOLD_DIRECTIONS",
    "detect_folds",
    "fold_statistics",
    "unfold_phidp",
]

#: Differential-phase ambiguity interval in degrees, for a receiver measuring
#: the H--V phase difference on simultaneous transmission. Halve it for
#: alternating transmission. See the module docstring: this is a stated default,
#: not a detected one.
DEFAULT_WRAP_DEG = 360.0

#: Number of gates beyond a candidate that must agree the branch has changed.
#: Three is the smallest window whose median is not set by a single gate, which
#: is the whole point of the test --- a one-gate excursion must not be able to
#: carry the vote.
DEFAULT_CONFIRM_GATES = 3

#: Accepted values of the ``direction`` argument. ``'increasing'`` permits only
#: upward corrections, appropriate wherever propagation phase is monotonically
#: non-decreasing (rain). ``'both'`` permits either sign and is what an ice
#: regime with vertically aligned crystals needs, at the cost of admitting twice
#: as many spurious candidates.
UNFOLD_DIRECTIONS = ("increasing", "both")


def _ray_valid_indices(phidp, mask):
    """Per-ray index arrays of the gates that carry usable phase."""
    return [np.flatnonzero(row) for row in mask]


def _resolve_mask(phidp, mask):
    """Boolean validity mask, defaulting to finiteness of the phase."""
    phidp = np.asarray(phidp, dtype=float)
    if phidp.ndim != 2:
        raise ValueError(f"phidp must be 2-D (n_rays, n_gates), got {phidp.shape}")
    if mask is None:
        return phidp, np.isfinite(phidp)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != phidp.shape:
        raise ValueError(
            f"mask shape {mask.shape} does not match phidp shape {phidp.shape}"
        )
    return phidp, mask & np.isfinite(phidp)


def fold_statistics(
    phidp,
    wrap_deg=DEFAULT_WRAP_DEG,
    mask=None,
    jump_threshold_deg=None,
    confirm_gates=DEFAULT_CONFIRM_GATES,
):
    """Count large phase steps and how many of them persist.

    Call this before :func:`unfold_phidp`. The ratio
    ``n_persistent / n_large_steps`` is the diagnostic that matters: near 1 the
    large steps are folds and unfolding is the right repair; near 0 they are
    backscatter-phase excursions or noise, and unfolding would corrupt the ray.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    wrap_deg : float, optional
        Ambiguity interval, default :data:`DEFAULT_WRAP_DEG`.
    mask : array_like of bool, optional
        Gates to consider. Defaults to the finite gates of ``phidp``.
    jump_threshold_deg : float, optional
        Step size counted as large. Defaults to ``wrap_deg / 2``, the point at
        which the folded and unfolded interpretations of a step swap which is
        the smaller.
    confirm_gates : int, optional
        Persistence window, default :data:`DEFAULT_CONFIRM_GATES`.

    Returns
    -------
    dict
        ``n_steps``, ``n_large_steps``, ``frac_large_steps``, ``n_persistent``,
        ``n_transient``, ``persistent_fraction_of_large`` (NaN when there are no
        large steps), ``max_abs_step_deg``, ``observed_span_deg`` and the
        thresholds used.
    """
    phidp, mask = _resolve_mask(phidp, mask)
    threshold = (
        0.5 * wrap_deg if jump_threshold_deg is None else float(jump_threshold_deg)
    )
    n_steps = n_large = n_persistent = 0
    max_abs_step = 0.0
    for ray, idx in enumerate(_ray_valid_indices(phidp, mask)):
        if idx.size < 2:
            continue
        values = phidp[ray, idx]
        steps = np.diff(values)
        n_steps += steps.size
        if steps.size:
            max_abs_step = max(max_abs_step, float(np.max(np.abs(steps))))
        large = np.flatnonzero(np.abs(steps) > threshold)
        n_large += large.size
        for position in large:
            if not _persists(values, position, threshold, confirm_gates):
                continue
            n_persistent += 1

    finite = phidp[mask]
    return {
        "n_steps": int(n_steps),
        "n_large_steps": int(n_large),
        "frac_large_steps": float(n_large / n_steps) if n_steps else np.nan,
        "n_persistent": int(n_persistent),
        "n_transient": int(n_large - n_persistent),
        "persistent_fraction_of_large": (
            float(n_persistent / n_large) if n_large else np.nan
        ),
        "max_abs_step_deg": max_abs_step,
        "observed_span_deg": (
            float(np.max(finite) - np.min(finite)) if finite.size else np.nan
        ),
        "wrap_deg": float(wrap_deg),
        "jump_threshold_deg": threshold,
        "confirm_gates": int(confirm_gates),
    }


def _persists(values, position, threshold, confirm_gates):
    """Whether the step at ``position`` is a sustained branch change.

    ``values`` is one ray's valid-gate phase sequence and the step runs from
    ``position`` to ``position + 1``. The medians either side are compared, and
    the change must both exceed ``threshold`` and carry the same sign as the
    step itself --- a spike produces a large step whose neighbourhood medians
    barely differ.
    """
    n = values.size
    after_stop = position + 1 + confirm_gates
    if after_stop > n:
        # Not enough gates left to establish persistence. Declining to correct
        # is the conservative choice: a fold in the last two gates of a ray
        # costs two gates, a wrong correction costs the rest of the ray.
        return False
    before = np.median(values[max(0, position + 1 - confirm_gates) : position + 1])
    after = np.median(values[position + 1 : after_stop])
    change = after - before
    step = values[position + 1] - values[position]
    return abs(change) > threshold and np.sign(change) == np.sign(step)


def detect_folds(
    phidp,
    wrap_deg=DEFAULT_WRAP_DEG,
    mask=None,
    jump_threshold_deg=None,
    confirm_gates=DEFAULT_CONFIRM_GATES,
    direction="increasing",
):
    """Locate confirmed folds without modifying the phase.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    wrap_deg : float, optional
        Ambiguity interval, default :data:`DEFAULT_WRAP_DEG`.
    mask : array_like of bool, optional
        Gates to consider. Defaults to the finite gates of ``phidp``.
    jump_threshold_deg : float, optional
        Step size counted as a candidate. Defaults to ``wrap_deg / 2``.
    confirm_gates : int, optional
        Persistence window, default :data:`DEFAULT_CONFIRM_GATES`.
    direction : {'increasing', 'both'}, optional
        Sign of correction permitted, see :data:`UNFOLD_DIRECTIONS`.

    Returns
    -------
    dict
        ``fold_gates``: list of arrays, one per ray, of the *gate indices* at
        which a confirmed fold begins (the first gate on the new branch).
        ``fold_signs``: matching list of +1/-1 correction signs.
        Plus the candidate accounting: ``n_candidates``, ``n_confirmed``,
        ``n_rejected_transient``, ``n_rejected_direction``.
    """
    if direction not in UNFOLD_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {UNFOLD_DIRECTIONS}, got {direction!r}"
        )
    phidp, mask = _resolve_mask(phidp, mask)
    threshold = (
        0.5 * wrap_deg if jump_threshold_deg is None else float(jump_threshold_deg)
    )

    fold_gates, fold_signs = [], []
    n_candidates = n_confirmed = n_transient = n_direction = 0

    for ray, idx in enumerate(_ray_valid_indices(phidp, mask)):
        gates, signs = [], []
        if idx.size >= 2:
            values = np.array(phidp[ray, idx], dtype=float, copy=True)
            position = 0
            while position < values.size - 1:
                step = values[position + 1] - values[position]
                if abs(step) > threshold:
                    n_candidates += 1
                    correction = -np.sign(step)
                    if direction == "increasing" and correction < 0:
                        n_direction += 1
                    elif not _persists(values, position, threshold, confirm_gates):
                        n_transient += 1
                    else:
                        # Apply on the working copy so that later candidates are
                        # judged against already-corrected phase.
                        values[position + 1 :] += correction * wrap_deg
                        gates.append(int(idx[position + 1]))
                        signs.append(int(correction))
                        n_confirmed += 1
                position += 1
        fold_gates.append(np.array(gates, dtype=np.int64))
        fold_signs.append(np.array(signs, dtype=np.int64))

    return {
        "fold_gates": fold_gates,
        "fold_signs": fold_signs,
        "n_candidates": int(n_candidates),
        "n_confirmed": int(n_confirmed),
        "n_rejected_transient": int(n_transient),
        "n_rejected_direction": int(n_direction),
        "wrap_deg": float(wrap_deg),
        "jump_threshold_deg": threshold,
        "confirm_gates": int(confirm_gates),
        "direction": direction,
    }


def unfold_phidp(
    phidp,
    wrap_deg=DEFAULT_WRAP_DEG,
    mask=None,
    jump_threshold_deg=None,
    confirm_gates=DEFAULT_CONFIRM_GATES,
    direction="increasing",
):
    """Unfold differential phase, correcting only sustained branch changes.

    Corrections are cumulative from the fold onward, and are applied to every
    gate beyond it including masked ones, so that a later processing step which
    admits more gates does not find a discontinuity the unfolding left behind.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    wrap_deg : float, optional
        Ambiguity interval, default :data:`DEFAULT_WRAP_DEG`.
    mask : array_like of bool, optional
        Gates whose steps may be examined. Defaults to the finite gates of
        ``phidp``. Masked gates are still shifted by whatever correction applies
        at their range.
    jump_threshold_deg : float, optional
        Step size counted as a candidate. Defaults to ``wrap_deg / 2``.
    confirm_gates : int, optional
        Persistence window, default :data:`DEFAULT_CONFIRM_GATES`.
    direction : {'increasing', 'both'}, optional
        Sign of correction permitted, see :data:`UNFOLD_DIRECTIONS`.

    Returns
    -------
    unfolded : numpy.ndarray
        Phase with confirmed folds repaired. Non-finite gates stay non-finite.
    metadata : dict
        The :func:`detect_folds` accounting, plus ``n_gates_shifted`` and
        ``n_rays_with_folds``.
    """
    phidp = np.asarray(phidp, dtype=float)
    detected = detect_folds(
        phidp,
        wrap_deg=wrap_deg,
        mask=mask,
        jump_threshold_deg=jump_threshold_deg,
        confirm_gates=confirm_gates,
        direction=direction,
    )
    unfolded = np.array(phidp, dtype=float, copy=True)
    n_shifted = 0
    n_rays_with_folds = 0
    for ray, (gates, signs) in enumerate(
        zip(detected["fold_gates"], detected["fold_signs"], strict=True)
    ):
        if gates.size == 0:
            continue
        n_rays_with_folds += 1
        for gate, sign in zip(gates, signs, strict=True):
            unfolded[ray, gate:] += sign * wrap_deg
            n_shifted += unfolded.shape[1] - gate

    metadata = dict(detected)
    metadata["n_gates_shifted"] = int(n_shifted)
    metadata["n_rays_with_folds"] = int(n_rays_with_folds)
    return unfolded, metadata
