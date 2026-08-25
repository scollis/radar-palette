"""Interchangeable backends for the azimuth NUFFT, and for the solve on top of it.

:mod:`radar_palette.gridding.nufft` hand-rolls a Kaiser-Bessel spreading kernel,
its Fourier transform by quadrature, and a conjugate-gradient solve on the
density-weighted normal equations. That implementation is correct and is kept as
the reference. This module factors the same operator into two independent choices
so that faster machinery --- some of it already in SciPy, some in specialised
NUFFT libraries --- can be used without changing the mathematics.

Two axes, chosen separately
---------------------------
An **engine** supplies ``forward`` (type-2: lattice to measured rays) and
``adjoint`` (type-1: rays to lattice). A **solver** turns those two products into
a recovered lattice. Every engine is validated against the reference operator and
every engine passes the same adjoint-consistency test, so the two axes compose
freely: any solver may be run on any engine.

Engines
~~~~~~~
``reference``
    :class:`~radar_palette.gridding.nufft._AzimuthNufftOperator`, unchanged. The
    definition of correct; the baseline every other engine is measured against.
``dense`` **(default)**
    No interpolation kernel at all: the DFT matrix, formed explicitly. Sounds
    wasteful and is not, because on the azimuth axis the matrix is
    ``n_rays x n_modes`` and both are ray counts --- 8 MB at 720 rays. Exact to
    round-off, and the fastest engine measured on the 360-720 ray shapes
    (``finufft``
    **Not safe to run concurrently.** Correct under ``n_jobs>1`` but wildly variable:
    measured on a 15-tilt volume at ``n_jobs=4``, three consecutive runs took 88.2,
    6.5 and 16.3 seconds where ``dense`` and ``torch`` held 1.0-1.1x on the identical
    path. Serially it is steady (0.29 s per tilt over six runs), so this is contention
    in its own global planner state. ``build_cones`` raises
    :class:`~radar_palette.gridding.cones.EngineConcurrencyWarning` when the two are
    combined. wins at both ends of the range). **No new dependencies.**
``scipy``
    The reference algorithm with two mechanical substitutions: the spreading
    stencil becomes one CSR sparse matrix (replacing ``numpy.add.at``, which is an
    unbuffered scatter and was 38% of solve time), and the spread-to-spectrum
    transform becomes :func:`scipy.fft.rfft` (the spread field is real, so the
    reference was computing a conjugate-symmetric half it then discarded). Both
    changes are exact: this engine agrees with ``reference`` to ~1e-15, which is
    round-off, not approximation. **No new dependencies**, so this is the default.
``finufft``
    Type-1/type-2 transforms from the Flatiron ``finufft`` library, with a reused
    ``Plan`` and ``n_trans`` batching over range gates. Uses an
    exponential-of-semicircle kernel rather than Kaiser-Bessel, so ``kb_width``
    and ``oversamp`` do not apply; accuracy is requested directly as ``eps``.
``ducc0``
    The same transforms from ``ducc0.nufft``, batched over gates and
    multithreaded internally. Included for accuracy parity with ``finufft``, not
    for speed: measured on these sweep shapes it is the **slowest** engine here,
    taking 1.5-4.1x the reference's time on ``cg`` (worst at 360 x 1832, best at
    the largest shape, where the gap narrows as the transform starts to dominate).
    A radar azimuth axis is short enough that its per-call setup costs more than
    the transform it is setting up for. Reach for it if you want ``finufft``
    accuracy without the ``finufft`` dependency.
``torch``
    ``torchkbnufft``: genuinely Kaiser-Bessel (table-interpolated), batching range
    gates on the coil axis. The fastest engine measured here on CPU, by roughly
    1.5 s on a 15-tilt volume -- a threading difference in its inner loop, not an
    algorithmic one.

    It is also the only engine that *can* run on a GPU (``device='cuda'`` or
    ``'mps'``), and measurement says not to bother: on an NVIDIA A10, per-tilt
    solves took 93.3 ms in float64 and 91.3 ms in float32 against 83.9 ms on the
    same machine's CPU -- slower on 15 of 15 tilts. The azimuth solve is 931 x 1050
    in double precision, a few tens of milliseconds of arithmetic, so transfer and
    launch overhead exceed what the device saves. This is the same size argument
    that makes ``dense`` competitive: the axis is a ray count, not an image
    dimension. Accuracy on the device is sound if anyone wants it anyway (float64
    matches CPU to 1e-15; float32 is 1.5x worse at 3.3e-4, since the Kaiser-Bessel
    kernel error dominates round-off either way).

    The
    CPU timings here are around the reference's; the reason to select it is an
    already-GPU-resident pipeline.

Engine availability is checked at construction, never at import, so a minimal
install keeps working and asking for an absent engine raises a message naming the
extra to install. :func:`available_engines` reports what the environment supports.

Solvers
~~~~~~~
``cg``
    Conjugate gradients on ``(E^H W E) c = E^H W f``, iteration for iteration the
    same as the reference. Preserves the existing ``cg_resid_*`` diagnostics.
``direct``
    Cholesky factorisation of the same normal matrix, which is the limit CG is
    iterating towards --- reached in one factorisation instead of approached.

Why a direct solve suits *this* problem
---------------------------------------
The normal matrix is ``n_lattice x n_lattice``, and for a radar sweep
``n_lattice`` is a **ray count** --- 120 to 1440, not an image dimension. A dense
Cholesky at that size is microseconds, and the per-gate cost drops to a BLAS-3
triangular solve shared across all gates. The iterative machinery CG exists to
avoid is not expensive enough here to be worth avoiding. It replaces 25
transforms (12 CG iterations, each a forward and an adjoint) with one
factorisation, which is where the speed comes from --- and it is the larger
effect of the two axes. On a 720 x 1200 sweep, relative to ``nufft.py``:

===========  ==========  ==============
engine       ``cg``      ``direct``
===========  ==========  ==============
scipy        2.3x        13.5x
dense        1.8x        **32.7x**
finufft      1.2x        29.2x
===========  ==========  ==============

Choosing a faster engine buys about 2x; changing the solver buys 10-30x.

Which engine leads is **not** monotone in ray count, so pick by measurement
rather than by asymptotics. Direct-solver speedups across the tested shapes:

============  ========  =========  =========
sweep         scipy     dense      finufft
============  ========  =========  =========
120 x 34      1.2x      18.8x      **22.5x**
360 x 500     6.7x      **35.3x**  18.5x
360 x 1832    14.9x     **44.0x**  17.4x
720 x 1200    13.8x     **32.2x**  30.7x
1440 x 1832   16.1x     21.5x      **32.7x**
============  ========  =========  =========

``dense`` leads the middle band (360-720 rays) and ``finufft`` leads at both
ends --- at 1440 rays for the ``O(n log n)`` versus ``O(n^2)`` reason, and at
120 rays for the opposite one: the problem is so small that the dense engine's
per-solve BLAS work no longer amortises the matrix it built. Operational
1-degree and 0.5-degree volumes sit in the band where ``dense`` wins.

**The accuracy it buys depends on the engine, and the distinction is the whole
point.** Measured on a field *exactly* band-limited on the lattice, so the
correct answer is known in closed form (120 rays, +/-0.3-spacing jitter,
relative error against truth):

===========  ==============  ==============
engine       ``cg`` (12)     ``direct``
===========  ==============  ==============
scipy        3.0e-4          3.2e-4
torch        3.0e-5          3.0e-5
dense        3.6e-5          **1.1e-10**
finufft      3.6e-5          **2.5e-10**
ducc0        3.6e-5          **3.1e-10**
===========  ==============  ==============

A direct solve is only as accurate as the operator it inverts. On a
Kaiser-Bessel engine it converges to that engine's own kernel error (~3e-4) and
so gains nothing on accuracy at low jitter --- the iteration was never the
binding constraint there. On an engine whose transform is exact (``dense``) or
near-exact (``finufft``, ``ducc0``), the kernel error is gone and the direct
solve reaches 1e-10: six orders of magnitude, from pairing the right solver with
the right engine rather than from either alone.

Where the direct solver helps on *every* engine is degraded sampling, because
CG's convergence rate degrades with the conditioning while a factorisation does
not. At +/-0.45-spacing jitter, 12 iterations reach only 1.7e-2 on the ``scipy``
engine against 4.5e-4 for the direct solve on the same transforms --- a 37x gain
with no change of engine, and 4.3e7 with ``dense``.

.. warning::

   **Every figure in this section is on a synthetic band-limited field, and the
   accuracy gain does not transfer to real reflectivity.** The table above is a
   statement about which operator best represents a field this operator can
   represent exactly. Real reflectivity is not such a field: speckle, clutter
   and echo edges put energy in modes the sweep cannot determine. Measured by
   held-out ray prediction on three storm-filled sweeps, the direct solve at its
   best ridge is *comparable* to 12 CG iterations --- 6.70 dB against 7.80
   (SPOL), 4.74 against 4.92 (SWX), 8.52 against 8.47 (CSAPR2) --- not orders
   better, and ``ridge='auto'`` does not always find that best ridge. On real
   data the reasons to choose this solver are speed (10-30x) and having no
   iteration count to tune. The ridge, not the solver, is what matters for
   accuracy there; see :data:`DEFAULT_RIDGE` for the per-sweep numbers. Run
   ``benchmarks/bench_nufft_engines.py --real`` before quoting any accuracy
   figure from this module to a user.

The ridge term is load-bearing, not cosmetic
--------------------------------------------
For a **sector**, ``n_lattice`` exceeds the ray count --- a 30-120 deg sector at
1 deg spacing fits 91 rays onto a 366-point lattice --- so the normal matrix is
singular by construction (rank-deficient by 275 there, smallest eigenvalue at
round-off). A bare Cholesky fails outright. ``ridge`` is scaled by
``trace(B) / n_lattice`` so it means the same thing at any sweep size, and it
selects the minimum-norm solution among the infinitely many that fit the data
equally well. On that sector the direct solve reproduces the measured rays to
8.9e-11 where 12 CG iterations reach 3.7e-4, a factor of 4e6 --- and unlike the
full-circle case this gain does not need a high-accuracy engine, because what is
being fixed is the rank deficiency rather than the kernel.

Accuracy of the engines themselves
----------------------------------
Against exact trig-polynomial arithmetic, the reference's own Kaiser-Bessel
kernel at its default ``kb_width=4, oversamp=2`` is accurate to 2.5e-4. The
``dense`` engine is exact to round-off, and ``finufft``/``ducc0`` reach ~2e-13.
So at default settings the approximation error of this operator is the *kernel*,
not the transform: substituting a more accurate transform changes the answer by
~5e-4 rather than converging to the reference. That is a move towards exact
arithmetic, but it is still a change, which is why the default engine is the one
that reproduces the reference bit-for-bit.

Every figure quoted above is produced by ``benchmarks/bench_nufft_engines.py``;
the timings are from an arm64 laptop, so re-run it rather than trusting them on
other hardware.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.fft import fft, ifft, next_fast_len, rfft

from radar_palette.gridding.nufft import (
    _AzimuthNufftOperator,
    kaiser_bessel_beta,
    kaiser_bessel_kernel,
)

__all__ = [
    "DEFAULT_ENGINE",
    "DEFAULT_RIDGE",
    "MIN_RIDGE",
    "TARGET_CONDITION",
    "DEFAULT_SOLVER",
    "ENGINES",
    "SOLVERS",
    "EngineUnavailableError",
    "available_engines",
    "make_operator",
]

# The exact operator, and no dependency beyond the ones this package already
# requires. Chosen over ``scipy`` (which reproduces nufft.py to round-off) because
# the pair below is a safety decision, not a performance one -- see DEFAULT_SOLVER.
# On the azimuth axis the DFT matrix is n_rays x n_modes and both are ray counts, so
# forming it explicitly costs 8 MB at 720 rays and removes the Kaiser-Bessel kernel
# error entirely.
DEFAULT_ENGINE = "dense"

# Changed from ``cg`` because the iterative default produces unphysical reflectivity
# on real data. Measured on a 15-tilt ARM C-SAPR2 volume, per cone: at ``n_cg=12``
# one tilt of fifteen overshoots its input range by 107 dB and reaches 169 dBZ,
# visible in a gridded volume only when the vertical interpolation happens not to
# discard those cells; across a resolution ramp the whole-volume range reached
# -299..297 dBZ. ``direct`` with the default ridge holds -99..66 dBZ on the same
# grids, with the other fourteen tilts within 1 dB of what ``cg`` gave them.
#
# This changes numbers for existing callers, which is why it was not done when the
# solver was introduced. It is done now because the numbers it changes were partly
# wrong: ``n_cg`` is acting as regularisation rather than as a convergence budget,
# so a caller raising it to improve the answer makes it catastrophically worse (25
# iterations: -1907..2471 dBZ), and no iteration count is a defensible substitute for
# an explicit regularisation parameter. ``az_solver='cg'`` still reproduces the old
# behaviour exactly, diagnostics included.
DEFAULT_SOLVER = "direct"

# Relative to trace(B) / n_lattice, so it is scale- and size-independent.
#
# Set from held-out prediction error on REAL storm-filled sweeps, not from
# synthetic jitter. The two disagree by orders of magnitude and the synthetic
# answer is the dangerous one: an evenly-jittered synthetic sweep has a
# full-rank normal matrix (condition number ~7) and wants a ridge at the 1e-10
# floor, while every real sweep measured wants 1e-3 or larger.
#
# Measured by holding out every seventh ray of a storm-filled sweep, recovering
# the lattice from the rest, and predicting the held-out rays. Median absolute
# error in dB; ``cg(12)`` is the iterative default on the same transforms, and
# ``grid-best`` is the best of a fixed 1e-10..1e0 decade sweep:
#
#   sweep                                null  cg(12)  r=1e-10  grid-best   'auto'
#   SPOL S-band, convective line            3    7.80     7.80  6.70@1e-1  7.45@1.5e-2
#   SWX C-band, widespread precipitation    0    4.92     5.08  4.74@1.8e-2 4.74@1.8e-2
#   CSAPR2 C-band, convective               0    8.47    10.19  8.52@1e-3  9.89@1.3e-2
#
# Three readings, in order of how much they should change what you do.
#
# 1e-10 is the worst ridge tested on every sweep, so a near-zero ridge is simply
# wrong. But the driver is NOT rank deficiency, which was the obvious hypothesis
# and is not what the data says: SWX and CSAPR2 are FULL-RANK and still want
# ~1e-2 and 1e-3. (Rank deficiency is real and geometric --- operational azimuth
# sampling leaves gaps of ~2x nominal, so the lattice asks for modes the rays
# cannot determine, and ``normal_null_dim`` counts them --- it just is not what
# sets the ridge.) What sets it is that real reflectivity is not band-limited:
# speckle, clutter and echo edges put energy in every mode, a small ridge fits
# that energy at the measured rays, and the interpolant between them is noise. A
# synthetic field built from a few harmonics has nothing in those modes and shows
# none of this, which is why tuning on synthetic data gave an answer eight orders
# of magnitude too small.
#
# ``'auto'`` is a large improvement on the old default and NOT the per-sweep
# optimum. It matches grid-best on SWX, gives up 0.75 dB on SPOL, and on CSAPR2
# lands 1.37 dB worse than grid-best and 1.42 dB worse than CG. It targets a
# condition number, which is a proxy for the quantity that actually matters
# (out-of-band energy) and a decent one only because the two correlate. A sweep
# whose optimum is far from TARGET_CONDITION is not pathological, just not what
# this heuristic was fitted to --- three sweeps is a thin basis. Pass an explicit
# float when a case matters, and use --real to find it.
#
# And the headline: with a well-chosen ridge the direct solve is COMPARABLE to CG
# on real data, not orders better -- 1.10 dB better on SPOL, 0.18 dB better on
# SWX, 0.05 dB worse on CSAPR2. The ~1e-10 recovery on synthetic band-limited
# fields does not transfer, because that accuracy was against a truth this
# operator can represent exactly and real reflectivity is not such a field. The
# direct solve's remaining advantage on real data is speed (10-30x) and the
# absence of an iteration count to tune, not accuracy.
#
# See benchmarks/bench_nufft_engines.py --real for the full curves.
#
# Kept as the documented fixed value for callers who want one; the solver's own
# default is ``ridge='auto'``, which derives it from the spectrum instead,
# because real and synthetic geometries need ridges eight orders apart.
DEFAULT_RIDGE = 1e-2

# 'auto' targets this condition number. Chosen from the held-out error curve on
# real sweeps, which is flat from ~1e2 to ~1e3 and degrades either side: looser
# (1e8) leaves 80 dB of overfitting on the worst-sampled sweep, tighter (1e1)
# starts over-smoothing. At 1e2 the adaptive ridge matches the CG default's
# predictive accuracy (4.08 dB against 4.06 dB) while remaining a direct solve.
TARGET_CONDITION = 1e2

# Floor for 'auto'. Deliberately left at the value that preserves the near-exact
# solve on a band-limited field, which is the only regime where this solver's
# accuracy advantage is real and large (1.1e-10 at this floor, degrading linearly
# with the ridge to 1.1e-2 at 1e-2).
#
# Raising it to 1e-2 was tried, because on the BNF C-SAPR2 volume 'auto' leaves
# well-conditioned tilts under-regularised -- at 4.5-42 deg the normal matrix is
# well behaved (cond 15-32) so the condition-number term never binds, yet the
# held-out optimum on echo gates is 1e-3..1e-2 at every tilt. The floor at 1e-2
# does win there (5.330 dB against 5.446, mean of per-tilt medians on >10 dBZ
# gates, versus 5.272 for a per-sweep oracle and 5.389 for CG). It was reverted
# because it buys 0.12 dB on real data by giving up eight orders of magnitude on
# the band-limited case, which is a bad trade for a DEFAULT: it hard-codes an
# assumption about the data into a geometry-only rule, and a caller gridding a
# smooth or simulated field would silently get a far worse answer.
#
# Generalised cross-validation was also tried as a data-driven replacement. It
# recovers 1e-10 exactly on the synthetic case and then fails on real sweeps: it
# returns the 1e-10 floor on four of the five BNF tilts, whose measured optima are
# 1e-2 (sweeps 3, 7, 11) and 1e-3 (sweep 14). It scores on training residual, and
# overfitting is precisely what minimises that. Any rule
# fitted on the measured rays has the same defect; distinguishing signal from
# out-of-band noise needs held-out rays, which a solver call does not have.
#
# So: no automatic rule tested is reliable enough to be a silent default. 'auto'
# handles the case it can defend (a degenerate geometry, where it selects ~1.7e-2
# on BNF sweep 0 with cond 1e9) and otherwise stays out of the way. For real
# reflectivity, set ridge=1e-2 explicitly -- DEFAULT_RIDGE is that value, and
# benchmarks/bench_nufft_engines.py --real measures it for a given radar.
MIN_RIDGE = 1e-10

ENGINES = ("reference", "scipy", "dense", "finufft", "ducc0", "torch")
SOLVERS = ("cg", "direct")

# Import name -> the extra that provides it, for the error message.
_ENGINE_REQUIREMENTS = {
    "reference": (),
    "scipy": (),
    "dense": (),
    "finufft": (("finufft", "spectral"),),
    "ducc0": (("ducc0", "spectral-ducc"),),
    "torch": (("torch", "spectral-torch"), ("torchkbnufft", "spectral-torch")),
}


class EngineUnavailableError(RuntimeError):
    """An engine was requested whose optional dependency is not installed."""


def available_engines():
    """Engine names usable in this environment, in preference order.

    Returns
    -------
    tuple of str
        Always contains ``'reference'`` and ``'scipy'``, which need nothing
        beyond this package's required dependencies.
    """
    usable = []
    for name in ENGINES:
        if all(
            importlib.util.find_spec(module) is not None
            for module, _ in _ENGINE_REQUIREMENTS[name]
        ):
            usable.append(name)
    return tuple(usable)


def _require(engine):
    """Raise a message naming the extra to install, rather than an ImportError."""
    missing = [
        (module, extra)
        for module, extra in _ENGINE_REQUIREMENTS[engine]
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        modules = ", ".join(module for module, _ in missing)
        extras = sorted({extra for _, extra in missing})
        raise EngineUnavailableError(
            f"engine {engine!r} needs {modules}, which is not installed "
            f"(pip install radar-palette[{','.join(extras)}]). "
            f"Available engines: {', '.join(available_engines())}"
        )


# --------------------------------------------------------------------------- #
# Solvers. Expressed in terms of forward/adjoint only, so every solver works on
# every engine -- including the reference operator, which is how the equivalence
# tests compare a solver against itself across engines.
# --------------------------------------------------------------------------- #


def solve_cg(operator, values, n_cg=0, cg_tol=1e-10):
    """Conjugate gradients on the density-weighted normal equations.

    Iteration for iteration identical to
    :meth:`~radar_palette.gridding.nufft._AzimuthNufftOperator.solve`, including
    the analytic ``n_lattice / 360`` scale on the initial gridded estimate and the
    explicit iteration count (a loop variable would be left at its initial value
    if the body never ran).

    Parameters
    ----------
    operator : object
        Anything supplying ``forward``, ``adjoint``, ``order``, ``n_lattice``.
    values : array_like
        Ray-major sweep values, shape ``(n_rays, n_gates)``.
    n_cg : int
        Iterations. ``<= 0`` returns the unrefined density-compensated gridding.
    cg_tol : float
        Relative residual at which to stop early.

    Returns
    -------
    lattice : numpy.ndarray
    info : dict
    """
    values = np.asarray(values, dtype=np.float64)
    rays = values[operator.order]
    right_hand_side = operator.adjoint(rays)
    lattice = right_hand_side * (operator.n_lattice / 360.0)
    info = {"solver": "cg", "n_cg": int(n_cg)}
    if n_cg <= 0:
        return lattice, info

    residual = right_hand_side - operator.adjoint(operator.forward(lattice))
    reference_norm = max(float(np.linalg.norm(right_hand_side)), 1e-30)
    info["cg_resid_start"] = float(np.linalg.norm(residual) / reference_norm)

    direction = residual.copy()
    residual_sq = float(np.sum(residual * residual))
    iterations_run = 0
    for _ in range(int(n_cg)):
        applied = operator.adjoint(operator.forward(direction))
        denominator = float(np.sum(direction * applied))
        if denominator == 0.0:
            break
        iterations_run += 1
        step = residual_sq / denominator
        lattice = lattice + step * direction
        residual = residual - step * applied
        residual_sq_new = float(np.sum(residual * residual))
        if np.sqrt(residual_sq_new) / reference_norm < cg_tol:
            residual_sq = residual_sq_new
            break
        direction = residual + (residual_sq_new / residual_sq) * direction
        residual_sq = residual_sq_new

    info["cg_iters_run"] = iterations_run
    info["cg_resid_end"] = float(np.linalg.norm(residual) / reference_norm)
    return lattice, info


def solve_direct(operator, values, ridge="auto", **_ignored):
    """Cholesky solve of the same normal equations CG iterates towards.

    The normal matrix ``B = A^T W A`` is built one column at a time through the
    engine's own ``forward``/``adjoint``, so it is exactly the operator CG would
    have applied --- no separate derivation to drift out of step --- and the
    factorisation is cached on the operator, amortised over every field gridded
    with the same geometry.

    Regularisation is **not** optional here, and its size matters more than the
    engine choice. ``ridge='auto'`` (the default) reads the actual eigenvalue
    spectrum and picks the smallest ridge that caps the condition number at
    :data:`TARGET_CONDITION`; pass a float to fix it, relative to
    ``trace(B) / n_lattice``.

    Why adaptive rather than a constant: real sweeps and evenly-jittered
    synthetic ones need ridges eight orders of magnitude apart, so no single
    number serves both. Operational azimuth sampling leaves gaps of ~2x the
    nominal spacing, which makes the normal matrix genuinely rank-deficient
    (measured: 23 null directions of 993 on an X-band sweep), and a small ridge
    then fits the training rays to 1e-7 dB while predicting held-out rays 805 dB
    away. A synthetic sweep with even jitter is full-rank and wants the small
    ridge. ``'auto'`` recovers both: 1e-10 on the synthetic case, ~1.6e-2 on the
    X-band sweep.

    Note that this solver is only as accurate as the engine it inverts, and that
    the ridge caps it too. On a Kaiser-Bessel engine it converges to that
    engine's kernel error (~3e-4); the ~1e-10 the module docstring tabulates
    needs both an exact engine *and* a well-sampled geometry that admits a small
    ridge.

    Returns
    -------
    lattice : numpy.ndarray
    info : dict
        ``normal_cond`` is the condition number after regularisation and
        ``normal_null_dim`` the number of eigenvalues below ``1e-10`` of the
        largest *before* it --- read that one to see whether the geometry, rather
        than the solver, is what limited the answer.
    """
    values = np.asarray(values, dtype=np.float64)
    rays = values[operator.order]
    right_hand_side = operator.adjoint(rays)
    was_flat = right_hand_side.ndim == 1
    rhs_2d = right_hand_side[:, None] if was_flat else right_hand_side

    factor, condition, resolved, null_dim = _normal_cholesky(operator, ridge)
    lattice = sla.cho_solve(factor, rhs_2d)
    info = {
        "solver": "direct",
        "n_cg": 0,
        "ridge_rel": resolved,
        "ridge_mode": "auto" if isinstance(ridge, str) else "fixed",
        "normal_cond": condition,
        "normal_null_dim": null_dim,
    }
    return (lattice.ravel() if was_flat else lattice), info


def evaluate_lattice(lattice_values, azimuths_deg):
    """Evaluate the recovered band-limited interpolant at arbitrary azimuths.

    The interpolant is a property of the **lattice**, not of the engine that
    recovered it: whichever engine ran, the result is the trig polynomial whose
    coefficients are the lattice DFT, with the Nyquist mode split so the
    interpolant is real. So this is one function rather than a per-engine method,
    and two engines that disagree here disagree about their output rather than
    about their evaluation.

    Needed because ``forward`` only evaluates at the *measured* azimuths, which
    cannot distinguish fitting from overfitting. Holding rays out and predicting
    them requires evaluating somewhere else, and on real data that distinction is
    the whole question --- see :data:`DEFAULT_RIDGE`.

    Parameters
    ----------
    lattice_values : array_like
        ``(n_lattice,)`` or ``(n_lattice, n_gates)``, as returned by ``solve``.
    azimuths_deg : array_like
        Azimuths to evaluate at, in degrees. Need not be measured azimuths, need
        not be sorted, and need not lie inside the measured range.

    Returns
    -------
    numpy.ndarray
        ``(len(azimuths_deg),)`` or ``(len(azimuths_deg), n_gates)`` to match the
        input's dimensionality.
    """
    lattice = np.asarray(lattice_values, dtype=np.float64)
    was_flat = lattice.ndim == 1
    lattice_2d = lattice[:, None] if was_flat else lattice
    n_lattice = lattice_2d.shape[0]

    coefficients = np.fft.fft(lattice_2d, axis=0) / n_lattice
    modes = np.fft.fftfreq(n_lattice, d=1.0 / n_lattice).astype(int)
    phase = np.deg2rad(np.asarray(azimuths_deg, dtype=np.float64))

    out = np.zeros((phase.size, lattice_2d.shape[1]), dtype=np.float64)
    for index, mode in enumerate(modes):
        if n_lattice % 2 == 0 and mode == -(n_lattice // 2):
            # Nyquist: half to +N/2 and half to -N/2, which is a cosine. Assigning
            # it wholly to one side gives a complex interpolant.
            basis = np.cos(mode * phase)
        else:
            basis = np.exp(1j * mode * phase)
        out += np.real(basis[:, None] * coefficients[index][None, :])
    return out.ravel() if was_flat else out


def _normal_cholesky(operator, ridge):
    """Factorise ``A^T W A + ridge * scale * I``, caching on the operator.

    ``ridge`` may be the string ``'auto'``, in which case it is derived from the
    eigenvalue spectrum: the smallest value that brings the condition number
    down to :data:`TARGET_CONDITION`, floored at :data:`MIN_RIDGE` so a
    well-conditioned geometry is left essentially alone. Returns the factor, the
    post-regularisation condition number, the ridge actually used, and the
    pre-regularisation null-space dimension.
    """
    cached = getattr(operator, "_normal_cache", None)
    if cached is not None and cached[0] == ridge:
        return cached[1], cached[2], cached[3], cached[4]

    n_lattice = operator.n_lattice
    normal = operator.adjoint(operator.forward(np.eye(n_lattice)))
    # Symmetrise. Defensive rather than load-bearing, and worth being precise
    # about: the measured asymmetry the transforms leave is ~3e-16 relative on
    # every engine, and ``cho_factor(lower=True)`` reads only the lower triangle
    # anyway, so this changes neither the factorisation nor its success. It is
    # kept because the matrix is symmetric in exact arithmetic and this makes the
    # precondition true rather than nearly true, and because ``normal_cond``
    # below *does* read the whole array.
    normal = 0.5 * (normal + normal.T)
    scale = float(np.trace(normal)) / n_lattice

    # One eigendecomposition serves both the ridge choice and the rank
    # diagnostic; it costs the same order as the Cholesky that follows.
    eigenvalues = np.linalg.eigvalsh(normal)
    largest = float(eigenvalues[-1])
    null_dim = int(np.sum(eigenvalues < 1e-10 * largest))

    if isinstance(ridge, str):
        if ridge != "auto":
            raise ValueError(f"ridge must be a float or 'auto', got {ridge!r}")
        # Smallest ridge bringing cond(B + r*scale*I) down to TARGET_CONDITION.
        # Solved directly from the extreme eigenvalues rather than by search.
        smallest = max(float(eigenvalues[0]), 0.0)
        needed = (
            (largest - TARGET_CONDITION * smallest)
            / (TARGET_CONDITION - 1.0)
            / max(scale, 1e-300)
        )
        resolved = float(max(needed, MIN_RIDGE))
    else:
        resolved = float(ridge)

    regularised = normal + resolved * scale * np.eye(n_lattice)
    factor = sla.cho_factor(regularised, lower=True)
    condition = float(np.linalg.cond(regularised))
    operator._normal_cache = (ridge, factor, condition, resolved, null_dim)
    return factor, condition, resolved, null_dim


_SOLVERS = {"cg": solve_cg, "direct": solve_direct}


# --------------------------------------------------------------------------- #
# Engines.
# --------------------------------------------------------------------------- #


class _EngineBase:
    """Shared geometry: lattice sizing, Voronoi weights, mode map, solve dispatch.

    Subclasses supply ``forward``, ``adjoint`` and ``_engine_extras``. Everything
    that decides *what problem is being solved* --- the lattice size, the density
    weights, the split-Nyquist mode map --- lives here, so engines can only differ
    in how they compute the transforms, never in what they compute.
    """

    name = "base"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        azimuths_deg = np.asarray(azimuths_deg, dtype=np.float64) % 360.0
        self.order = np.argsort(azimuths_deg)
        self.azimuths = azimuths_deg[self.order]
        self.n_rays = self.azimuths.size
        self.kb_width = float(kb_width)
        self.oversamp = float(oversamp)

        # Lattice size is measured, never assumed -- identical rule to the
        # reference operator, so every engine grids onto the same lattice.
        self.n_lattice = (
            int(geometry.nrays)
            if geometry.is_full_360
            else int(round(360.0 / geometry.az_spacing_median_deg))
        )

        # Voronoi arc-length weights in degrees, summing to 360.
        self.weights = (
            (np.roll(self.azimuths, -1) - np.roll(self.azimuths, 1)) % 360.0
        ) / 2.0

        self.lattice_modes = np.fft.fftfreq(
            self.n_lattice, d=1.0 / self.n_lattice
        ).astype(int)
        self._build_mode_map()
        self._normal_cache = None

    def _build_mode_map(self):
        """Map the ``N0`` FFT bins onto a contiguous mode list, Nyquist split.

        The reference maps modes onto an oversampled lattice; the library engines
        want a contiguous ``-N0/2 .. +N0/2`` block instead. Both carry the same
        rule: for even ``N0`` the Nyquist coefficient is **split** half to each
        end, which is what keeps the off-lattice interpolant real and symmetric.
        Assigning it wholly to one side produces a complex interpolant.
        """
        n_lattice = self.n_lattice
        self.n_modes = n_lattice + 1 if n_lattice % 2 == 0 else n_lattice
        self.mode_low = -(self.n_modes // 2)
        rows, columns, weights = [], [], []
        for column, mode in enumerate(self.lattice_modes):
            if n_lattice % 2 == 0 and mode == -n_lattice // 2:
                rows += [0, self.n_modes - 1]
                columns += [column, column]
                weights += [0.5, 0.5]
            else:
                rows.append(mode - self.mode_low)
                columns.append(column)
                weights.append(1.0)
        self._pad = sp.csr_matrix(
            (weights, (rows, columns)),
            shape=(self.n_modes, n_lattice),
            dtype=np.float64,
        )

    @staticmethod
    def _as_columns(values):
        """Accept a 1-D vector or a 2-D (axis, gates) block; remember which."""
        return (values[:, None], True) if values.ndim == 1 else (values, False)

    def _modes_from_lattice(self, values_2d):
        """Lattice values to contiguous mode coefficients, transposed for engines."""
        coefficients = fft(values_2d, axis=0) / self.n_lattice
        return np.ascontiguousarray((self._pad @ coefficients).T)

    def _lattice_from_modes(self, modes_by_column):
        """Contiguous mode coefficients (batch-major) back to lattice values."""
        truncated = self._pad.T @ np.ascontiguousarray(modes_by_column.T)
        return np.real(ifft(truncated, axis=0))

    def adjoint_test(self, seed=0):
        """Relative mismatch of ``<A u, f>_W`` against ``<u, A^H f>``.

        Should be at machine-precision level. A value near 1 means a scale factor
        is missing from the transpose, and conjugate gradients on the normal
        equations would be solving the wrong problem. Every engine is held to
        this, because an engine whose adjoint is not the transpose of its forward
        would break both solvers in ways a smoothness check would not catch.
        """
        rng = np.random.default_rng(seed)
        lattice = rng.normal(size=(self.n_lattice, 1))
        rays = rng.normal(size=(self.n_rays, 1))
        forward_inner = float(
            np.sum(self.forward(lattice) * rays * self.weights[:, None])
        )
        adjoint_inner = float(np.sum(lattice * self.adjoint(rays)))
        return float(
            abs(forward_inner - adjoint_inner)
            / max(abs(forward_inner), abs(adjoint_inner), 1e-30)
        )

    def solve(self, values, n_cg=0, cg_tol=1e-10, solver=DEFAULT_SOLVER, ridge=None):
        """Recover the uniform lattice from the measured rays.

        Parameters
        ----------
        values : array_like
            Ray-major sweep values, ``(n_rays, n_gates)``.
        n_cg, cg_tol : int, float
            Passed to the ``cg`` solver; ignored by ``direct``.
        solver : {'cg', 'direct'}
        ridge : float or {'auto'}, optional
            ``direct`` only. Defaults to ``'auto'``, deriving the ridge from the
            eigenvalue spectrum -- see :func:`solve_direct` for why a constant is
            not safe here.

        Returns
        -------
        lattice : numpy.ndarray
        info : dict
        """
        if solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        if solver == "direct":
            lattice, info = solve_direct(
                self, values, ridge="auto" if ridge is None else ridge
            )
        else:
            lattice, info = solve_cg(self, values, n_cg=n_cg, cg_tol=cg_tol)
        info["engine"] = self.name
        return lattice, info

    def describe(self):
        """Engine parameters, for the evaluator's report.

        ``kb_width``/``oversamp`` are reported as *requested* on every engine, but
        only the Kaiser-Bessel engines act on them; ``finufft`` and ``ducc0``
        select their own grid from ``eps``. ``_engine_extras`` says what each
        engine actually used.
        """
        described = {
            "engine": self.name,
            "kb_width": self.kb_width,
            "oversamp": self.oversamp,
            "n_nominal": self.n_lattice,
            "n_modes": self.n_modes,
        }
        described.update(self._engine_extras())
        return described

    def _engine_extras(self):
        return {}


class _ReferenceEngine(_AzimuthNufftOperator):
    """The unmodified reference operator, exposed through the engine interface.

    Subclasses rather than wraps, so ``forward`` and ``adjoint`` are literally the
    reference implementations and cannot drift from them. The added ``solve``
    accepts a ``solver`` argument so the direct solver can be run on the reference
    transforms --- which is how the tests establish that the speedup comes from
    the solver and the engine independently.
    """

    name = "reference"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        super().__init__(azimuths_deg, geometry, kb_width=kb_width, oversamp=oversamp)
        # This engine delegates the transforms to nufft.py, which carries its own
        # mode bookkeeping, so the base class's padded mode map is unused here.
        # ``n_modes``/``mode_low`` are still set --- consistently with every other
        # engine --- because a caller evaluating the recovered interpolant at
        # azimuths that were never measured needs them, and an engine that
        # silently lacked them would fail only on that path.
        self.n_modes = self.n_lattice
        self.mode_low = -(self.n_modes // 2)
        self._normal_cache = None

    def solve(self, values, n_cg=0, cg_tol=1e-10, solver=DEFAULT_SOLVER, ridge=None):
        """As the reference, plus the ``direct`` solver on the same transforms."""
        if solver == "cg":
            lattice, info = super().solve(values, n_cg=n_cg, cg_tol=cg_tol)
            info["solver"] = "cg"
        elif solver == "direct":
            lattice, info = solve_direct(
                self, values, ridge="auto" if ridge is None else ridge
            )
        else:
            raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
        info["engine"] = self.name
        return lattice, info

    def describe(self):
        """Engine parameters, matching the reference operator's reported set."""
        return {
            "engine": self.name,
            "kb_width": self.kb_width,
            "oversamp": self.oversamp,
            "kb_beta": self.beta,
            "n_nominal": self.n_lattice,
            "M_oversampled": self.n_oversampled,
            "kernel": "kaiser_bessel",
        }


class _ScipyEngine(_EngineBase):
    """Reference algorithm, SciPy machinery: CSR spreading and a real FFT.

    Two exact substitutions, both of which the reference left on the table:

    ``numpy.add.at`` to a CSR matrix product
        The spreading stencil is a fixed sparse pattern, so the scatter is a
        sparse matrix-vector product. ``np.add.at`` is an unbuffered ufunc method
        that cannot use the BLAS and was 38% of the reference's solve time; the
        same operation as ``S @ f`` is a single indexed reduction, and the
        interpolation is ``S.T @ g`` on the transpose of that one matrix, which
        makes the exactness of the adjoint structural rather than a property to
        re-derive.

    ``fft`` to ``rfft``
        The spread field is real. The reference computed the full complex spectrum
        and used the conjugate-symmetric half it had just computed for nothing.
    """

    name = "scipy"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        self.n_oversampled = next_fast_len(int(np.ceil(self.oversamp * self.n_lattice)))
        self.beta = kaiser_bessel_beta(self.kb_width, self.oversamp)
        self.cell_deg = 360.0 / self.n_oversampled

        position = self.azimuths / self.cell_deg
        self.n_stencil = int(np.floor(self.kb_width)) + 1
        first_cell = np.ceil(position - self.kb_width / 2.0).astype(int)
        stencil_offsets = np.arange(self.n_stencil)[None, :]
        cells = (first_cell[:, None] + stencil_offsets) % self.n_oversampled
        kernel_values = kaiser_bessel_kernel(
            position[:, None] - (first_cell[:, None] + stencil_offsets),
            self.kb_width,
            self.beta,
        )
        ray_index = np.repeat(np.arange(self.n_rays), self.n_stencil)
        # One matrix serves both directions: ``spread = S @ f`` and
        # ``interpolate = S.T @ g``, so forward and adjoint cannot disagree.
        self._spread = sp.csr_matrix(
            (kernel_values.ravel(), (cells.ravel(), ray_index)),
            shape=(self.n_oversampled, self.n_rays),
        )
        self._interpolate = self._spread.T.tocsr()

        # Deapodisation by quadrature from the same kernel used for spreading, so
        # the two are consistent for any (width, beta) -- the reference's rule.
        self.n_half = self.n_oversampled // 2 + 1
        self._deapodisation_half = self._kernel_fourier_transform(
            np.arange(self.n_half)
        )
        contiguous_modes = np.arange(self.n_modes) + self.mode_low
        self._deapodisation_modes = self._kernel_fourier_transform(contiguous_modes)
        self._mode_bins = contiguous_modes % self.n_oversampled

    def _kernel_fourier_transform(self, mode_index, n_quadrature=4096):
        offsets = np.linspace(-self.kb_width / 2.0, self.kb_width / 2.0, n_quadrature)
        kernel = kaiser_bessel_kernel(offsets, self.kb_width, self.beta)
        cell_step = offsets[1] - offsets[0]
        phase = np.exp(
            -2j * np.pi * np.outer(np.asarray(mode_index) / self.n_oversampled, offsets)
        )
        return np.real(phase @ kernel) * cell_step

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        coefficients = fft(values_2d, axis=0) / self.n_lattice
        modes = self._pad @ coefficients
        padded = np.zeros((self.n_oversampled, values_2d.shape[1]), dtype=complex)
        padded[self._mode_bins] = modes / self._deapodisation_modes[:, None]
        oversampled = np.real(ifft(padded, axis=0)) * self.n_oversampled
        out = self._interpolate @ oversampled
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice.

        The exact transpose of :meth:`forward` under the density weights, by
        construction: it uses the transpose of the same sparse matrix.
        """
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        spread = self._spread @ weighted_values
        # rfft, then read the negative modes off the conjugate-symmetric half.
        half_spectrum = rfft(spread, axis=0) / self._deapodisation_half[:, None]
        bins = self._mode_bins
        negative = bins >= self.n_half
        modes = np.empty((self.n_modes, values_2d.shape[1]), dtype=complex)
        modes[~negative] = half_spectrum[bins[~negative]]
        modes[negative] = np.conj(half_spectrum[self.n_oversampled - bins[negative]])
        truncated = self._pad.T @ modes
        out = np.real(ifft(truncated, axis=0))
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "kb_beta": self.beta,
            "M_oversampled": self.n_oversampled,
            "kernel": "kaiser_bessel",
        }


class _DenseEngine(_EngineBase):
    """The exact transform: a dense DFT matrix, no interpolation kernel at all.

    Every other engine approximates ``exp(i m phi_j)`` by spreading onto an
    oversampled lattice with a kernel, because for a large problem forming the
    matrix is unthinkable. On the azimuth axis of a radar sweep it is not: the
    matrix is ``n_rays x n_modes``, and both are ray counts. At 720 rays that is
    an 8 MB array and a BLAS-3 ``GEMM`` per transform.

    So this engine has no kernel error, no oversampling factor and no accuracy
    parameter --- ``forward`` is the trig polynomial evaluated exactly, to
    round-off. Combined with the ``direct`` solver it recovers a band-limited
    field to ~1e-10 (the ridge, not the arithmetic), against ~3e-4 for the
    Kaiser-Bessel engines, and it is the fastest engine measured on sweeps of
    360-720 rays. It does **not** win everywhere: ``finufft`` is faster at both
    ends of the tested range --- at 1440 rays because ``O(n_rays * n_modes)``
    loses to ``O(n log n)`` (the crossover the NUFFT literature is written
    about), and at 120 rays because the problem is too small for the dense
    per-solve BLAS work to amortise the matrix. See the table in the module
    docstring; measure rather than extrapolate.

    Memory is the reason this is not the default: the matrix is
    ``16 * n_rays * n_modes`` bytes (33 MB at 1440 rays, 133 MB at 2880), where
    the Kaiser-Bessel engines are linear in the ray count.
    """

    name = "dense"

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        modes = np.arange(self.n_modes) + self.mode_low
        self._matrix = np.exp(1j * np.outer(np.deg2rad(self.azimuths), modes))

    def forward(self, lattice_values):
        """Type-2 transform, exactly: evaluate the trig polynomial at the rays."""
        values_2d, was_flat = self._as_columns(lattice_values)
        coefficients = fft(values_2d, axis=0) / self.n_lattice
        out = np.real(self._matrix @ (self._pad @ coefficients))
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: the conjugate transpose of the same matrix."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        modes = self._matrix.conj().T @ weighted_values.astype(complex)
        out = np.real(ifft(self._pad.T @ modes, axis=0))
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "kernel": "none_exact",
            "matrix_bytes": int(self._matrix.nbytes),
        }


class _FinufftEngine(_EngineBase):
    """Flatiron ``finufft`` type-1/type-2, planned once and batched over gates.

    Accuracy is requested as ``eps`` rather than derived from a kernel width; the
    library picks its own oversampling and its exponential-of-semicircle kernel
    accordingly, so ``kb_width`` and ``oversamp`` are recorded but not used.
    """

    name = "finufft"

    def __init__(
        self,
        azimuths_deg,
        geometry,
        kb_width=4.0,
        oversamp=2.0,
        eps=1e-9,
    ):
        _require("finufft")
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        import finufft

        self._finufft = finufft
        self.eps = float(eps)
        self._points = np.ascontiguousarray(np.deg2rad(self.azimuths))
        self._plans = {}

    def _plan(self, nufft_type, n_columns):
        """Plans are cached per (type, batch width): setpts is the expensive part."""
        key = (nufft_type, n_columns)
        if key not in self._plans:
            plan = self._finufft.Plan(
                nufft_type,
                (self.n_modes,),
                n_trans=n_columns,
                eps=self.eps,
                isign=1 if nufft_type == 2 else -1,
            )
            plan.setpts(self._points)
            self._plans[key] = plan
        return self._plans[key]

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        modes = self._modes_from_lattice(values_2d)
        rays = self._plan(2, values_2d.shape[1]).execute(modes)
        out = np.ascontiguousarray(np.real(rays).T)
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        source = np.ascontiguousarray(weighted_values.T.astype(complex))
        modes = self._plan(1, values_2d.shape[1]).execute(source)
        out = self._lattice_from_modes(modes)
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {"eps": self.eps, "kernel": "exponential_semicircle"}


class _Ducc0Engine(_EngineBase):
    """``ducc0.nufft``: batched over gates and internally multithreaded.

    Same transform contract as the ``finufft`` engine and the same
    ``eps``-driven accuracy, with no PyTorch dependency and thread control
    through ``nthreads`` (``0`` uses every hardware thread).
    """

    name = "ducc0"

    def __init__(
        self,
        azimuths_deg,
        geometry,
        kb_width=4.0,
        oversamp=2.0,
        eps=1e-9,
        nthreads=0,
    ):
        _require("ducc0")
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        import ducc0.nufft

        self._nufft = ducc0.nufft
        self.eps = float(eps)
        self.nthreads = int(nthreads)
        self._coordinates = np.ascontiguousarray(
            np.deg2rad(self.azimuths).reshape(-1, 1)
        )

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        modes = self._modes_from_lattice(values_2d)
        rays = self._nufft.u2nu(
            grid=modes,
            coord=self._coordinates,
            forward=False,
            epsilon=self.eps,
            nthreads=self.nthreads,
            fft_order=False,
        )
        out = np.ascontiguousarray(np.real(rays).T)
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        source = np.ascontiguousarray(weighted_values.T.astype(complex))
        modes = self._nufft.nu2u(
            points=source,
            coord=self._coordinates,
            forward=True,
            epsilon=self.eps,
            nthreads=self.nthreads,
            out=np.empty((source.shape[0], self.n_modes), dtype=complex),
            fft_order=False,
        )
        out = self._lattice_from_modes(modes)
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "eps": self.eps,
            "nthreads": self.nthreads,
            "kernel": "exponential_semicircle",
        }


class _TorchKbNufftEngine(_EngineBase):
    """``torchkbnufft`` table interpolation; range gates ride the coil axis.

    Genuinely Kaiser-Bessel, like the reference, and the only engine here that
    runs on a GPU. ``device`` is passed straight to ``torch``, so ``'cuda'``,
    ``'mps'`` and ``'cpu'`` all work wherever the local build supports them. The
    default is ``'cpu'`` and should stay there: on an A10 the CUDA path measured
    0.90x (float64) and 0.92x (float32) of CPU speed, slower on every tilt of a
    15-tilt volume, because a 931 x 1050 solve is too small to amortise transfer
    and launch. See the module docstring. Note that
    ``torchkbnufft`` defaults its tables to single precision, so a GPU run is also a
    precision change unless ``complex128`` is requested --- worth checking against the
    ``dense`` engine on your own data before trusting it. Note the phase convention:
    this library's
    forward transform is ``exp(-i * omega * (k - N // 2))`` for image index ``k``.
    With ``k = m - mode_low`` and ``N = n_modes``, ``k - N // 2`` is exactly the
    signed mode ``m``, so passing ``omega = -phi`` yields the ``exp(+i m phi)``
    this operator is defined by, and no separate fftshift correction is needed.
    Getting that wrong is not a small error --- it produced a 69% discrepancy
    against exact arithmetic, which the adjoint test still passed, because a
    consistently wrong pair of operators is still a consistent pair.
    """

    name = "torch"

    def __init__(
        self,
        azimuths_deg,
        geometry,
        kb_width=4.0,
        oversamp=2.0,
        numpoints=6,
        device="cpu",
        dtype=None,
    ):
        _require("torch")
        super().__init__(azimuths_deg, geometry, kb_width, oversamp)
        import torch
        import torchkbnufft

        self._torch = torch
        self.numpoints = int(numpoints)
        self.device = torch.device(device)
        # Double precision by default, which is NOT what torchkbnufft itself
        # defaults to. Every other engine here works in float64, so matching that
        # keeps an engine swap a performance decision rather than a silent
        # precision change; a caller who wants the single-precision path -- the
        # one a GPU is actually fast at -- asks for complex64 explicitly and can
        # see in the report what they chose.
        self.dtype = torch.complex128 if dtype is None else dtype
        self.real_dtype = (
            torch.float64 if self.dtype == torch.complex128 else torch.float32
        )
        self.grid_size = int(np.ceil(self.oversamp * self.n_modes))
        self._forward_module = torchkbnufft.KbNufft(
            im_size=(self.n_modes,),
            grid_size=(self.grid_size,),
            numpoints=self.numpoints,
            dtype=self.dtype,
            device=self.device,
        )
        self._adjoint_module = torchkbnufft.KbNufftAdjoint(
            im_size=(self.n_modes,),
            grid_size=(self.grid_size,),
            numpoints=self.numpoints,
            dtype=self.dtype,
            device=self.device,
        )
        omega = -np.deg2rad(self.azimuths)
        omega = (omega + np.pi) % (2.0 * np.pi) - np.pi
        self._omega = torch.tensor(
            omega.reshape(1, -1), dtype=self.real_dtype, device=self.device
        )

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d, was_flat = self._as_columns(lattice_values)
        modes = self._modes_from_lattice(values_2d)[None]
        image = self._torch.tensor(modes, dtype=self.dtype, device=self.device)
        rays = self._forward_module(image, self._omega).cpu().numpy()[0]
        out = np.ascontiguousarray(np.real(rays).T)
        return out.ravel() if was_flat else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice."""
        values_2d, was_flat = self._as_columns(ray_values)
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d
        source = np.ascontiguousarray(weighted_values.T.astype(complex))[None]
        k_space = self._torch.tensor(source, dtype=self.dtype, device=self.device)
        modes = self._adjoint_module(k_space, self._omega).cpu().numpy()[0]
        out = self._lattice_from_modes(modes)
        return out.ravel() if was_flat else out

    def _engine_extras(self):
        return {
            "numpoints": self.numpoints,
            "device": str(self.device),
            "M_oversampled": self.grid_size,
            "kernel": "kaiser_bessel",
        }


_ENGINE_CLASSES = {
    "reference": _ReferenceEngine,
    "scipy": _ScipyEngine,
    "dense": _DenseEngine,
    "finufft": _FinufftEngine,
    "ducc0": _Ducc0Engine,
    "torch": _TorchKbNufftEngine,
}


def make_operator(
    azimuths_deg,
    geometry,
    engine=DEFAULT_ENGINE,
    kb_width=4.0,
    oversamp=2.0,
    **engine_options,
):
    """Build an azimuth NUFFT operator on the named engine.

    Parameters
    ----------
    azimuths_deg : array_like
        Measured azimuths in degrees; need not be sorted or in ``[0, 360)``.
    geometry : SweepGeometry
        Supplies ``nrays``, ``is_full_360`` and ``az_spacing_median_deg``, which
        between them fix the lattice size. Every engine uses the same rule.
    engine : str
        One of :data:`ENGINES`, or ``'auto'`` for the default engine. ``'auto'``
        resolves to :data:`DEFAULT_ENGINE` rather than to the fastest installed
        engine on purpose: a result that silently changes with the contents of the
        environment is not reproducible, and the faster library engines are not
        bit-identical to the reference (see the module docstring).
    kb_width, oversamp : float
        Kaiser-Bessel kernel width and lattice oversampling. Used by the
        ``reference``, ``scipy`` and ``torch`` engines; recorded but unused by
        ``finufft`` and ``ducc0``, which take ``eps`` instead.
    **engine_options
        Engine-specific: ``eps`` (``finufft``, ``ducc0``), ``nthreads``
        (``ducc0``), ``numpoints`` and ``device`` (``torch``).

    Returns
    -------
    object
        An operator with ``forward``, ``adjoint``, ``adjoint_test``, ``solve`` and
        ``describe``.

    Raises
    ------
    ValueError
        Unknown engine name.
    EngineUnavailableError
        Known engine whose optional dependency is not installed.
    """
    if engine == "auto":
        engine = DEFAULT_ENGINE
    if engine not in _ENGINE_CLASSES:
        raise ValueError(f"engine must be one of {ENGINES} or 'auto', got {engine!r}")
    return _ENGINE_CLASSES[engine](
        azimuths_deg,
        geometry,
        kb_width=kb_width,
        oversamp=oversamp,
        **engine_options,
    )
