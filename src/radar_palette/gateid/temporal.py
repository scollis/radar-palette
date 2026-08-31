"""Temporal gate identification: measurement first, model second.

The single-volume classifier in :mod:`radar_palette.gateid.single` is
memoryless. Nothing couples consecutive volumes, so temporal consistency is
emergent rather than enforced, and a gate can change class between volumes for
no physical reason.

This module exploits the time domain. It is deliberately organised so the
*measurements* come first and stand on their own, because a literature survey
found no published transition matrix or persistence probability for hydrometeor
classes, and no Markov, HMM, CRF or MRF hydrometeor classifier of any kind.
Lukach et al. (2021) raised the criticism that existing schemes ignore temporal
dependence and explicitly left it "unexplored"; none of the works citing them
followed up. Being first is as much a risk as an opportunity, so the tooling
here is built to produce a cheap negative result if the hypotheses do not hold.

Three hypotheses
----------------
**H1 persistence.** A gate classified X at time *t* is more likely than chance
to be X at *t+1*. Quantified by :func:`transition_matrix` and
:func:`persistence_skill`.

**H2 clutter is stationary.** Ground clutter does not move, so in a fixed radar
frame it is persistent where weather is not. Addressed by
:func:`accumulate_clutter_prior`, which follows the accumulated-clutter-map
approach used operationally rather than any sequential model.

**H3 differential motion.** Biological scatterers move with the wind, storms
advect coherently, clutter does not move at all. Addressed by comparing
Eulerian with Lagrangian persistence, which tests H2 and H3 together: clutter
should be *more* persistent in the fixed frame, precipitation more persistent
in the advected frame.

Four cautions, each from a real result
--------------------------------------
1. **Mask before measuring.** Roughly 80% of gates in a scan are no-echo. An
   unmasked transition matrix is dominated by no-echo to no-echo and shows
   near-perfect persistence for trivial reasons. :func:`transition_matrix`
   masks by default.

2. **Report against the marginal.** Persistence only means something relative
   to chance, so :func:`persistence_skill` returns the conditional probability
   *and* the unconditional class frequency, and a skill score comparing them.

3. **Expect non-stationarity.** Cluster-based classifications have repeatedly
   proved non-transferable between datasets. Measure the matrix per storm
   morphology rather than pooling.

4. **Prefer a one-way, evidence-gated prior to a symmetric smoother.** Both
   operational precedents are asymmetric: the MetSignal cross-time rule could
   only promote non-meteorological to meteorological, and CMD's clutter-phase
   weight breaks ties in one direction only. A symmetric smoother is the
   version most likely to lock in a wrong label and defend it -- Zhang et al.
   (2020) removed the one temporal rule in an operational QC scheme after
   documenting anomalous propagation propagating forward through it.
   :func:`temporal_prior` is therefore one-way and vetoable by design.

References
----------
Hubbert, J. C., M. Dixon, and S. M. Ellis, 2009: Weather radar ground clutter.
Part II: Real-time identification and filtering. *J. Atmos. Oceanic Technol.*,
26, 1181-1197, doi:10.1175/2009JTECHA1160.1. The clutter-phase-alignment
weight that breaks ties in one direction only, cited for the asymmetry of
:func:`temporal_prior`. Note CPA is an intra-dwell statistic over ~64 ms, not
a scan-to-scan one.

Krause, J. M., 2016: A simple algorithm to discriminate between meteorological
and nonmeteorological radar echoes. *J. Atmos. Oceanic Technol.*, 33,
1875-1885, doi:10.1175/JTECH-D-15-0239.1. MetSignal, and the only published
per-gate cross-time rule found: a previous-volume reflectivity test that can
promote non-meteorological to meteorological but never demote.

Lukach, M., D. Dufton, J. Crosier, J. M. Hampton, L. Bennett, and R. R. Neely
III, 2021: Hydrometeor classification of quasi-vertical profiles of polarimetric
radar measurements using a top-down iterative hierarchical clustering method.
*Atmos. Meas. Tech.*, 14, 1075-1098, doi:10.5194/amt-14-1075-2021. Source of
the criticism that existing schemes ignore temporal dependence, which their
Sect. 6 leaves "unexplored".

Zhang, S., X. Huang, J. Min, Z. Chu, X. Zhuang, and H. Zhang, 2020: Improved
fuzzy logic method to distinguish between meteorological and non-meteorological
echoes using C-band polarimetric radar data. *Atmos. Meas. Tech.*, 13, 537-551,
doi:10.5194/amt-13-537-2020. Reimplemented the MetSignal cross-time rule and
then removed it, having documented anomalous propagation propagating forward
through it -- the cautionary result behind this module's one-way,
evidence-gated design.
"""

from __future__ import annotations

import numpy as np

from radar_palette.gateid import single

__all__ = [
    "accumulate_clutter_prior",
    "apply_temporal_prior",
    "class_composition",
    "composition_drift",
    "persistence_skill",
    "temporal_prior",
    "transition_matrix",
]

#: Classes excluded from transition statistics by default. Without this the
#: matrix is dominated by empty sky and reports spurious persistence.
DEFAULT_MASK_CLASSES = ("unclassified", "no_scatter")


def _codes(names):
    return [single.NAME_TO_CODE[n] for n in names if n in single.NAME_TO_CODE]


def transition_matrix(
    labels_t0, labels_t1, mask_classes=DEFAULT_MASK_CLASSES, normalise="row"
):
    """Empirical class-to-class transition counts between two volumes.

    Both inputs must be on the same grid, so that element *i* refers to the
    same (ray, gate) in each. Whether that is the same *air* is a separate
    question -- see the Eulerian/Lagrangian discussion in the module docstring.

    Parameters
    ----------
    labels_t0, labels_t1 : ndarray of int
        Class codes at consecutive times, same shape.
    mask_classes : sequence of str, optional
        Classes excluded from the statistics at both times. Pass an empty tuple
        to include everything, but read caution 1 first.
    normalise : {'row', 'total', None}, optional
        ``'row'`` gives P(class at t1 | class at t0), the persistence reading.
        ``'total'`` gives the joint distribution. ``None`` returns raw counts.

    Returns
    -------
    matrix : ndarray
        Square array indexed by ``order``.
    order : list of str
        Class names labelling both axes.
    n_pairs : int
        Gate pairs contributing, after masking.
    """
    a = np.asarray(labels_t0).ravel()
    b = np.asarray(labels_t1).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    order = [n for n in single.CLASSES.values() if n not in set(mask_classes)]
    codes = _codes(order)
    index = {c: i for i, c in enumerate(codes)}

    keep = np.isin(a, codes) & np.isin(b, codes)
    a, b = a[keep], b[keep]

    matrix = np.zeros((len(codes), len(codes)), dtype="f8")
    for ca, cb in zip(a, b, strict=True):
        matrix[index[int(ca)], index[int(cb)]] += 1.0

    n_pairs = int(keep.sum())
    if normalise == "row":
        rows = matrix.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.where(rows > 0, matrix / rows, np.nan)
    elif normalise == "total" and n_pairs:
        matrix = matrix / n_pairs
    return matrix, order, n_pairs


def persistence_skill(labels_t0, labels_t1, mask_classes=DEFAULT_MASK_CLASSES):
    """Per-class persistence measured against the marginal.

    A conditional persistence of 0.9 is unremarkable if the class occupies 90%
    of the scan. What matters is the excess over chance, so this returns both
    and a skill score between them.

    Returns
    -------
    dict
        Per class: ``persistence`` P(X at t1 | X at t0), ``marginal`` P(X at
        t1), ``skill`` = (persistence - marginal) / (1 - marginal), and ``n``
        the number of gates that were X at t0. Skill is 0 when persistence is
        no better than chance and 1 when a class never changes.
    """
    matrix, order, n_pairs = transition_matrix(
        labels_t0, labels_t1, mask_classes=mask_classes, normalise=None
    )
    if n_pairs == 0:
        return {}
    totals = matrix.sum(axis=1)
    marginal = matrix.sum(axis=0) / n_pairs
    out = {}
    for i, name in enumerate(order):
        if totals[i] == 0:
            continue
        p = matrix[i, i] / totals[i]
        m = marginal[i]
        skill = (p - m) / (1.0 - m) if m < 1.0 else np.nan
        out[name] = {
            "persistence": float(p),
            "marginal": float(m),
            "skill": float(skill),
            "n": int(totals[i]),
        }
    return out


def class_composition(labels, classes=None, mask_classes=DEFAULT_MASK_CLASSES):
    """Fraction of classified gates in each class."""
    order = classes or [
        n for n in single.CLASSES.values() if n not in set(mask_classes)
    ]
    codes = _codes(order)
    arr = np.asarray(labels).ravel()
    keep = np.isin(arr, codes)
    total = int(keep.sum())
    if total == 0:
        return dict.fromkeys(order, np.nan), 0
    return (
        {n: float((arr == c).sum() / total) for n, c in zip(order, codes, strict=True)},
        total,
    )


def composition_drift(labels_t0, labels_t1, mask_classes=DEFAULT_MASK_CLASSES):
    """Total-variation distance between two volumes' class proportions.

    Gate-wise agreement conflates classifier instability with advection: at a
    ten-minute cadence a storm moves kilometres, so a fixed gate genuinely
    samples different air and *should* change class. The mix of classes present
    in the scan should not swing abruptly. This measure is invariant to bulk
    motion and isolates classifier behaviour.

    Returns 0 for identical composition, 1 for no overlap.
    """
    c0, n0 = class_composition(labels_t0, mask_classes=mask_classes)
    c1, n1 = class_composition(labels_t1, mask_classes=mask_classes)
    if not n0 or not n1:
        return np.nan
    return float(0.5 * sum(abs(c0[k] - c1[k]) for k in c0))


def accumulate_clutter_prior(label_stack, clutter_class="clutter", min_volumes=5):
    """Per-gate probability of being clutter, accumulated over many volumes.

    Ground clutter is stationary, so in a fixed radar frame it recurs at the
    same gates while weather moves through. Accumulating the clutter fraction
    over many volumes therefore separates the two without any sequential model
    -- this is the well-trodden operational approach and carries far less risk
    than a class-transition model.

    Parameters
    ----------
    label_stack : sequence of ndarray
        Class-code arrays for the same radar geometry at different times.
    clutter_class : str, optional
        Class whose recurrence is accumulated.
    min_volumes : int, optional
        Below this many volumes the estimate is too noisy to be useful and
        every gate returns NaN rather than a number that invites overreading.

    Returns
    -------
    prior : ndarray of float
        Fraction of volumes in which each gate was clutter, or NaN.
    n_volumes : int
    """
    stack = [np.asarray(a) for a in label_stack]
    if not stack:
        raise ValueError("label_stack is empty")
    shape = stack[0].shape
    if any(a.shape != shape for a in stack):
        raise ValueError("all volumes must share one geometry")
    if len(stack) < min_volumes:
        return np.full(shape, np.nan), len(stack)
    code = single.NAME_TO_CODE[clutter_class]
    hits = np.zeros(shape, dtype="f8")
    for a in stack:
        hits += a == code
    return hits / len(stack), len(stack)


def temporal_prior(previous_labels, target_class, weight=0.15, veto_margin=0.5):
    """Build a one-way temporal bonus for one class, in score units.

    Deliberately *not* a symmetric smoother. The bonus can only add evidence
    for ``target_class`` where the previous volume already found it; it can
    never argue against a class, and it is vetoed wherever the instantaneous
    classification is confident. That asymmetry is what both operational
    precedents use, and a symmetric alternative is the version documented to
    lock in and propagate errors.

    Parameters
    ----------
    previous_labels : ndarray of int
        Class codes from the preceding volume, same geometry.
    target_class : str
        Class the prior may support.
    weight : float, optional
        Size of the bonus, in units of the normalised class score (which runs
        0 to 1). Keep it small: this is a tie-breaker, not a vote.
    veto_margin : float, optional
        Where the current winner already leads the runner-up by more than
        this, the prior is suppressed entirely.

    Returns
    -------
    callable
        ``apply(scores, order, margin) -> scores`` suitable for
        :func:`apply_temporal_prior`.
    """
    prev = np.asarray(previous_labels).ravel()
    code = single.NAME_TO_CODE[target_class]

    def apply(scores, order, margin):
        if target_class not in order:
            return scores
        col = order.index(target_class)
        out = scores.copy()
        eligible = prev == code
        if margin is not None:
            eligible = eligible & (np.asarray(margin).ravel() <= veto_margin)
        out[eligible, col] = np.minimum(out[eligible, col] + weight, 1.0)
        return out

    return apply


def apply_temporal_prior(features, priors, shape=None, **classify_kwargs):
    """Classify with one or more one-way temporal priors applied to the scores.

    The priors are handed to :func:`radar_palette.gateid.single.classify` as
    score adjusters rather than being applied to its output. That ordering is
    the whole point: an adjuster runs *after* the hard constraints and *before*
    the argmax, so a prior cannot resurrect a class that geometry or
    thermodynamics has forbidden, and every downstream rule -- the
    unclassified rule, the noise floor, despeckling -- applies to the adjusted
    answer exactly as it does to the instantaneous one.

    An earlier version re-derived labels from the returned score matrix, which
    silently bypassed all of that: a prior could label a gate ``multi_trip``
    where the second-trip geometry said no scatterer could exist.

    Parameters
    ----------
    features : dict
        Feature arrays, as for :func:`radar_palette.gateid.single.classify`.
    priors : sequence of callable
        Functions from :func:`temporal_prior`.
    shape : tuple, optional
        Sweep shape, needed for despeckling.
    **classify_kwargs
        Passed through to the single-volume classifier.

    Returns
    -------
    codes : ndarray of int
    info : dict
        As from the single-volume classifier, plus ``n_prior_changed``: how
        many gates the priors moved relative to classifying without them. If
        that number is large the priors are not tie-breaking, they are
        deciding -- which is the failure mode to watch for.
    """
    baseline, _ = single.classify(features, shape=shape, **classify_kwargs)
    codes, info = single.classify(
        features, shape=shape, score_adjusters=tuple(priors), **classify_kwargs
    )
    info = dict(info)
    info["n_prior_changed"] = int((codes != baseline).sum())
    return codes, info
