r"""Bounded-KDP retrieval by linear programming, after Huang et al. (2016).

The retrieval fits a propagation phase to the measured phase in the L1 sense,
subject to a **two-sided box** on the KDP the fitted phase implies:

.. math::

    \min_x \; \sum_i w_i \, |x_i - b_i|
    \qquad \text{s.t.} \qquad
    K_L \;\le\; M x \;\le\; K_U

where :math:`b` is the measured phase, :math:`M` is the KDP operator (a
Savitzky--Golay first-derivative filter, halved, divided by the gate spacing)
and :math:`K_L, K_U` come from
:func:`radar_palette.phase.bounds.kdp_bounds`.

Architecture, and what it is not
--------------------------------
This is the Huang, Zhang, Zhao & Giangrande (2016) architecture
(`doi:10.1109/TGRS.2016.2596295 <https://doi.org/10.1109/TGRS.2016.2596295>`_),
**not** the Giangrande, McGraw & Lei (2013) one. The structural difference is
the constraint: 2013 imposes one-sided non-negative monotonicity on the filtered
derivative, ``M x >= 0``, and lets the L1 data term do everything else. 2016
replaces that with a box whose *both* sides carry information from Z and ZDR ---
measurements the backscatter-phase term does not contaminate. The augmentation
is exactly the ``+M / -M`` block pair appended to the constraint matrix, with the
right-hand side gaining :math:`K_U` and :math:`-K_L`.

Three properties of this implementation are deliberate and worth stating plainly
because each of them is a place a previous KDP retrieval went wrong:

**There is no curvature or total-variation penalty in the objective.** Neither
Giangrande 2013 nor Huang 2016 has one. Giangrande's own argument is that with
the L1 inequalities alone the cost reduces to zero at ``x = b``, so the
constraints are what regularise; adding a smoothness penalty makes the estimator
a different method with a different solution family. A TV penalty in particular
has a *piecewise-constant* solution family: it does not merely tolerate
staircase KDP fields, it prefers them. None is offered here, not even as an
option.

**The reported KDP is exactly the constrained quantity.** ``kdp = M x``, with the
same ``M`` the box constrains, so ``K_L <= kdp <= K_U`` holds by construction and
is asserted in the test suite. Py-ART's ``phase_proc_lp`` instead constrains one
filter and reports KDP from a different one, which means its output carries no
guarantee from its own constraints. The cost of this choice is that Giangrande's
coupled *strong*-monotonicity post-filter is not applied --- applying it would
break the guarantee --- so sub-filter oscillation is controlled here only by the
smoothing length. That is a documented deviation from Giangrande 2013, not an
oversight; measure it with a step-likeness diagnostic on the output.

**The retrieval is invariant to an additive constant on the input phase.** This
is structural rather than incidental. ``M`` annihilates constants (the
Savitzky--Golay derivative weights sum to zero), the phase variables are
unbounded, nothing anchors ``x`` at an absolute value, and gates with zero weight
contribute *no rows at all* rather than being zero-filled --- a zero fill puts a
value inside the data range which then carries no cost but still enters the
constraint. That failure mode is not hypothetical: an earlier KDP retrieval in
this project recorded a constant phase shift moving its negative-KDP fraction
from 39% to 76% through exactly that mechanism. (That figure is inherited from
the earlier work, not measured here; what *is* measured here is this module's
invariance, in ``tests/phase/test_linear_program.py``, which verifies it rather
than assuming it.)

Smoothing length
----------------
``smoothing_km`` is a **required argument with no default**, in kilometres. It
sets the length of the derivative filter, and therefore the scale below which
KDP structure is not represented. It must be chosen against the scale of the
DSD field being retrieved, which is measurable independently from Z and ZDR: at
ARM BNF the along-range decorrelation length is 0.75 km in Z and 0.98 km in ZDR,
so a smoothing length near 1 km is matched to the field and 4-5 km is not. Huang
et al. used 2 km. There is no defensible default because the right value is a
property of the radar and the storm, not of the code; the realised length after
quantisation to an odd number of gates is returned in the metadata.

Sign policy and scope
---------------------
Both live in :mod:`radar_palette.phase.bounds`: the sign policy is
altitude-conditioned by default rather than globally non-negative, and the
self-consistency constraint is released outside a caller-supplied rain mask.
This module simply solves whatever box it is handed.

What this estimator is blind to
-------------------------------
It cannot separate ``delta`` from propagation phase --- nothing that works on one
ray of phase can. It *bounds* the result with independent measurements instead,
which is a different and weaker claim. It has no cross-ray or vertical coupling,
so non-uniform beam filling is invisible to it. And its output inherits every
error in the bounds, which inherit every error in the ZDR offset.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

__all__ = [
    "RAY_STATUSES",
    "derivative_matrix",
    "kdp_linear_program",
    "savgol_derivative_weights",
    "smoothing_length_gates",
    "solve_ray",
]

#: Per-ray outcome codes returned in ``metadata['status']``. Every ray gets
#: exactly one, and a failure carries its reason: a ray silently reported as
#: failed with no recorded cause is how a wrong enum value once cost a session.
RAY_STATUSES = (
    "solved",
    "no_valid_gates",
    "too_few_valid_gates",
    "infeasible",
    "solver_error",
)


def savgol_derivative_weights(length):
    """Savitzky--Golay first-derivative weights for a linear fit of ``length``.

    Parameters
    ----------
    length : int
        Odd filter length in gates, at least 3.

    Returns
    -------
    numpy.ndarray
        Weights summing to zero, giving the derivative per gate index. For
        ``length=5`` these are ``[-0.2, -0.1, 0, 0.1, 0.2]``, the ``d5`` filter
        of Giangrande et al. (2013).
    """
    length = int(length)
    if length < 3 or length % 2 == 0:
        raise ValueError(f"length must be an odd integer >= 3, got {length}")
    offsets = np.arange(length, dtype=float) - (length - 1) / 2.0
    return offsets / float(np.dot(offsets, offsets))


def smoothing_length_gates(smoothing_km, dr_km):
    """Odd filter length in gates realising a smoothing length in kilometres.

    Parameters
    ----------
    smoothing_km : float
        Requested smoothing length in kilometres.
    dr_km : float
        Gate spacing in kilometres.

    Returns
    -------
    length : int
        Odd number of gates, at least 3.
    realised_km : float
        ``length * dr_km`` --- the length actually applied. It can differ
        substantially from the request on coarse gates, which is why it is
        returned rather than left implicit.
    """
    if not np.isfinite(smoothing_km) or smoothing_km <= 0:
        raise ValueError(
            f"smoothing_km must be positive and finite, got {smoothing_km}"
        )
    if not np.isfinite(dr_km) or dr_km <= 0:
        raise ValueError(f"dr_km must be positive and finite, got {dr_km}")
    length = int(round(smoothing_km / dr_km))
    if length % 2 == 0:
        length += 1
    length = max(3, length)
    return length, length * dr_km


def derivative_matrix(n_gates, dr_km, length):
    """Sparse KDP operator: half the range derivative of phase.

    Parameters
    ----------
    n_gates : int
        Number of gates in the ray.
    dr_km : float
        Gate spacing in kilometres.
    length : int
        Odd filter length in gates.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(n_gates - length + 1, n_gates)``. Row ``j`` differentiates the
        window starting at gate ``j``, so its output belongs to the window
        centre, gate ``j + (length - 1) // 2``. The factor of one half is the
        definition ``KDP = 0.5 dPhi/dr``.
    """
    n_rows = int(n_gates) - int(length) + 1
    if n_rows < 1:
        raise ValueError(f"a {length}-gate filter does not fit in {n_gates} gates")
    weights = savgol_derivative_weights(length) * 0.5 / float(dr_km)
    return sparse.diags(
        diagonals=[np.full(n_rows, w) for w in weights],
        offsets=list(range(int(length))),
        shape=(n_rows, int(n_gates)),
        format="csr",
    )


#: Minimum fraction of a derivative window's gates that must carry data before
#: the KDP at that window centre is reported. The default is **1.0** --- every
#: gate in the window --- and that is not conservatism for its own sake.
#:
#: An unweighted gate contributes no row to the linear program, so its phase
#: variable is free. Every window containing a free variable therefore has a
#: *degenerate* KDP: the objective does not see it, and an LP optimum sits on a
#: vertex, so the solver settles it on a **box edge**. The number that comes back
#: is arbitrary within the bounds and it looks exactly like a retrieval.
#:
#: Measured on a 400-gate ray, 11-gate filter, box [-5, 20] deg/km, worst change
#: in reported KDP when a constant was added to the input phase:
#:
#: =============  ==================  ==================
#: gap in gates   fraction 0.5        fraction 1.0
#: =============  ==================  ==================
#: 0              2.9e-13             2.9e-13
#: 1              --                  2.9e-13
#: 5              2.0e+01             2.9e-13
#: 20             1.99e+01            2.9e-13
#: 60             2.0e+01             2.9e-13
#: =============  ==================  ==================
#:
#: Note the 5-gate row: a gap *narrower* than the filter is enough, so requiring
#: a data majority in the window does not help. Only excluding free variables
#: does.
#:
#: The cost is coverage, and it is not small: with 5% of gates randomly dropped,
#: reported gates fell from 390/400 at fraction 0.5 to 254/400 at 1.0. A caller
#: who needs that coverage should fill the gaps *upstream* --- linear
#: interpolation of phase between surrounding valid gates is offset-equivariant,
#: so it preserves invariance --- and pass full weights. That makes the invented
#: phase visible in the caller's code rather than hidden as a solver artefact.
DEFAULT_MIN_WINDOW_WEIGHT_FRACTION = 1.0


def _window_weight_counts(weight, length):
    """Count the weighted gates in each derivative window, by window index."""
    prefix = np.concatenate([[0.0], np.cumsum((weight > 0.0).astype(float))])
    n_rows = weight.size - length + 1
    starts = np.arange(n_rows)
    return prefix[starts + length] - prefix[starts]


def solve_ray(
    phidp,
    lower,
    upper,
    dr_km,
    smoothing_km,
    weights=None,
    method="highs",
    min_window_weight_fraction=DEFAULT_MIN_WINDOW_WEIGHT_FRACTION,
):
    """Solve the bounded L1 phase fit for a single ray.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase for one ray, length ``n_gates``, degrees.
        Non-finite gates are excluded from the data term (and from the weights
        if none are supplied) but still carry phase variables, so the solution
        is continuous through a gap.
    lower, upper : array_like
        Per-gate KDP bounds in deg/km, length ``n_gates``. Only the entries at
        window centres are used; see :func:`derivative_matrix`.
    dr_km : float
        Gate spacing in kilometres.
    smoothing_km : float
        Smoothing length in kilometres. Required, no default; see the module
        docstring.
    weights : array_like, optional
        Per-gate weights on the L1 data term. Defaults to 1 on finite gates and
        0 elsewhere. Gates of zero weight contribute no constraint rows at all.
    method : str, optional
        Passed to :func:`scipy.optimize.linprog`, default ``'highs'``.
    min_window_weight_fraction : float, optional
        Fraction of a window's gates that must carry data for its KDP to be
        reported, default :data:`DEFAULT_MIN_WINDOW_WEIGHT_FRACTION`. Set it to
        0 to report every window, including those with no data at all -- see
        the constant's own docstring for what that returns.

    Returns
    -------
    dict
        ``kdp`` (length ``n_gates``, NaN in the ``(length-1)//2`` edge gates at
        each end and wherever a window lacked data support),
        ``phidp_retrieved`` (length ``n_gates``; its absolute level follows the
        input, since the problem fixes only differences), ``status``,
        ``message``, ``length``, ``realised_smoothing_km``, ``n_weighted``,
        ``n_unsupported_windows``.
    """
    phidp = np.asarray(phidp, dtype=float).ravel()
    lower = np.asarray(lower, dtype=float).ravel()
    upper = np.asarray(upper, dtype=float).ravel()
    n_gates = phidp.size
    if lower.size != n_gates or upper.size != n_gates:
        raise ValueError(
            f"bounds must have {n_gates} entries, got lower={lower.size} "
            f"upper={upper.size}"
        )

    length, realised_km = smoothing_length_gates(smoothing_km, dr_km)
    nan = np.full(n_gates, np.nan)
    out = {
        "kdp": nan,
        "phidp_retrieved": nan.copy(),
        "status": "no_valid_gates",
        "message": "",
        "length": length,
        "realised_smoothing_km": realised_km,
        "n_weighted": 0,
        "n_unsupported_windows": 0,
    }

    finite = np.isfinite(phidp)
    if weights is None:
        weight = finite.astype(float)
    else:
        weight = np.asarray(weights, dtype=float).ravel()
        if weight.size != n_gates:
            raise ValueError(f"weights must have {n_gates} entries, got {weight.size}")
        weight = np.where(finite & np.isfinite(weight), weight, 0.0)
    weighted = np.flatnonzero(weight > 0.0)
    out["n_weighted"] = int(weighted.size)

    if weighted.size == 0:
        return out
    if n_gates < length or weighted.size < length:
        out["status"] = "too_few_valid_gates"
        out["message"] = f"{weighted.size} weighted gates for a {length}-gate filter"
        return out

    operator = derivative_matrix(n_gates, dr_km, length)
    n_rows = operator.shape[0]
    half = (length - 1) // 2
    centres = np.arange(half, half + n_rows)
    box_lower = lower[centres]
    box_upper = upper[centres]
    if not np.all(np.isfinite(box_lower)) or not np.all(np.isfinite(box_upper)):
        out["status"] = "solver_error"
        out["message"] = "non-finite KDP bounds at window centres"
        return out
    if np.any(box_lower > box_upper):
        out["status"] = "infeasible"
        out["message"] = "lower bound exceeds upper bound at one or more gates"
        return out

    # Variables: [phase (n_gates), L1 auxiliaries t (n_weighted)].
    # Objective: sum of w_i t_i over weighted gates only.
    n_aux = weighted.size
    cost = np.concatenate([np.zeros(n_gates), weight[weighted]])

    selector = sparse.csr_matrix(
        (
            np.ones(n_aux),
            (np.arange(n_aux), weighted),
        ),
        shape=(n_aux, n_gates),
    )
    identity = sparse.identity(n_aux, format="csr")
    zeros = sparse.csr_matrix((n_rows, n_aux))

    # x_i - t_i <= b_i  and  -x_i - t_i <= -b_i  give t_i >= |x_i - b_i|.
    # M x <= K_U  and  -M x <= -K_L  give the two-sided box.
    a_ub = sparse.vstack(
        [
            sparse.hstack([selector, -identity]),
            sparse.hstack([-selector, -identity]),
            sparse.hstack([operator, zeros]),
            sparse.hstack([-operator, zeros]),
        ],
        format="csr",
    )
    b_ub = np.concatenate([phidp[weighted], -phidp[weighted], box_upper, -box_lower])

    # Phase is unbounded: any bound on it would break offset invariance.
    bounds = [(None, None)] * n_gates + [(0.0, None)] * n_aux

    try:
        solution = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method=method)
    except Exception as error:  # noqa: BLE001 - reason is recorded, not swallowed
        out["status"] = "solver_error"
        out["message"] = f"{type(error).__name__}: {error}"
        return out

    if not solution.success or solution.x is None:
        out["status"] = "infeasible" if solution.status == 2 else "solver_error"
        out["message"] = str(solution.message)
        return out

    phase = np.asarray(solution.x[:n_gates], dtype=float)
    kdp = np.full(n_gates, np.nan)
    supported = _window_weight_counts(weight, length) >= (
        min_window_weight_fraction * length
    )
    kdp[centres[supported]] = (operator @ phase)[supported]
    out["kdp"] = kdp
    out["phidp_retrieved"] = phase
    out["status"] = "solved"
    out["n_unsupported_windows"] = int((~supported).sum())
    return out


def kdp_linear_program(
    phidp,
    lower,
    upper,
    dr_km,
    smoothing_km,
    weights=None,
    method="highs",
    min_window_weight_fraction=DEFAULT_MIN_WINDOW_WEIGHT_FRACTION,
):
    """Run the bounded L1 retrieval ray by ray over a sweep.

    Rays are independent, and a failure on one never propagates: each is solved
    inside its own exception guard and its reason recorded, so a partial sweep is
    an auditable result rather than a crash or a silently empty field.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees. Must be
        unfolded and system-phase corrected --- see
        :mod:`radar_palette.phase.unfold` and
        :mod:`radar_palette.phase.system`. The retrieval is invariant to an
        additive constant, so the system-phase step matters for the fold branch
        and for absolute phase products, not for KDP.
    lower, upper : array_like
        Per-gate KDP bounds in deg/km on the same grid, from
        :func:`radar_palette.phase.bounds.kdp_bounds`.
    dr_km : float
        Gate spacing in kilometres.
    smoothing_km : float
        Smoothing length in kilometres. Required, no default.
    weights : array_like, optional
        Per-gate weights on the L1 data term, same shape as ``phidp``.
    method : str, optional
        Linear-programming method, default ``'highs'``.
    min_window_weight_fraction : float, optional
        See :func:`solve_ray`.

    Returns
    -------
    kdp : numpy.ndarray
        Retrieved KDP in deg/km, ``(n_rays, n_gates)``, NaN on unsolved rays and
        in the filter's edge gates.
    phidp_retrieved : numpy.ndarray
        The fitted phase, same shape.
    metadata : dict
        ``status`` (list of per-ray codes), ``messages`` (dict keyed by ray
        index, only for rays that did not solve), ``n_solved``,
        ``status_counts``, ``length``, ``realised_smoothing_km``,
        ``requested_smoothing_km``, ``dr_km``, ``n_unsupported_windows``.
    """
    phidp = np.asarray(phidp, dtype=float)
    if phidp.ndim != 2:
        raise ValueError(f"phidp must be 2-D (n_rays, n_gates), got {phidp.shape}")
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    for name, field in (("lower", lower), ("upper", upper)):
        if field.shape != phidp.shape:
            raise ValueError(
                f"{name} shape {field.shape} does not match phidp shape {phidp.shape}"
            )
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != phidp.shape:
            raise ValueError(
                f"weights shape {weights.shape} does not match phidp shape "
                f"{phidp.shape}"
            )

    length, realised_km = smoothing_length_gates(smoothing_km, dr_km)
    n_rays = phidp.shape[0]
    kdp = np.full(phidp.shape, np.nan)
    retrieved = np.full(phidp.shape, np.nan)
    statuses, messages = [], {}
    n_unsupported = 0

    for ray in range(n_rays):
        try:
            result = solve_ray(
                phidp[ray],
                lower[ray],
                upper[ray],
                dr_km,
                smoothing_km,
                weights=None if weights is None else weights[ray],
                method=method,
                min_window_weight_fraction=min_window_weight_fraction,
            )
        except Exception as error:  # noqa: BLE001 - per-ray reason is the point
            statuses.append("solver_error")
            messages[ray] = f"{type(error).__name__}: {error}"
            continue
        kdp[ray] = result["kdp"]
        retrieved[ray] = result["phidp_retrieved"]
        statuses.append(result["status"])
        n_unsupported += result["n_unsupported_windows"]
        if result["status"] != "solved":
            messages[ray] = result["message"]

    counts = {code: statuses.count(code) for code in RAY_STATUSES}
    metadata = {
        "status": statuses,
        "messages": messages,
        "n_solved": counts["solved"],
        "status_counts": counts,
        "length": length,
        "realised_smoothing_km": realised_km,
        "requested_smoothing_km": float(smoothing_km),
        "dr_km": float(dr_km),
        "n_unsupported_windows": int(n_unsupported),
    }
    return kdp, retrieved, metadata
