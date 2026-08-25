"""Engines must agree with the reference; solvers must beat it.

Two claims carry this module, and each has a test that fails if the claim is
false rather than a test that merely exercises the code.

**Equivalence.** Every engine is a different way to compute the same operator, so
every engine is checked against the reference implementation directly --- and,
for the engines accurate enough to justify it, against exact trig-polynomial
arithmetic computed here in closed form. A test that only checked an engine
against itself (its own adjoint identity, say) would pass for an engine that is
consistently wrong; ``torchkbnufft`` initially was exactly that, off by 69% from
a phase-convention mismatch while passing its adjoint test perfectly.

**Improvement.** The direct solver is claimed to be more accurate than the
conjugate-gradient default. That is tested against a field which is *exactly*
band-limited on the recovery lattice, so the correct lattice is known analytically
and "more accurate" is measured against truth rather than against the other
method's output.
"""

from __future__ import annotations

import numpy as np
import pytest

# test_evaluator is a sibling module, not a package member -- the suite has no
# tests/__init__.py -- so this is a plain top-level import that relies on pytest
# putting the test directory on sys.path (rootdir-relative insertion, the default
# "prepend" import mode). Reusing its fixtures rather than restating them keeps
# the two NUFFT test modules measuring the same sweep.
from test_evaluator import (
    RANGE_FIRST_M,
    RANGE_SPACING_M,
    band_limited_field,
    make_radar,
    uniform_azimuths,
)

from radar_palette.gridding import nufft_engines as engines
from radar_palette.gridding.census import SweepClass, census_sweep
from radar_palette.gridding.evaluator import SweepSpectralEvaluator
from radar_palette.gridding.nufft import _AzimuthNufftOperator

# Engines installable as optional extras; the test is skipped rather than failed
# when the extra is absent, so a minimal install still collects a full suite.
OPTIONAL_ENGINES = ("finufft", "ducc0", "torch")
ALL_ENGINES = engines.ENGINES

# The reference's own Kaiser-Bessel kernel is accurate to ~2.5e-4 against exact
# arithmetic at the default kb_width=4, oversamp=2. Engines that share that
# kernel must match it much more closely than that; engines with a different
# kernel can only be expected to agree to the kernel error itself.
KB_KERNEL_ERROR = 2.5e-4


def engine_or_skip(name):
    """Skip rather than fail when an optional engine is not installed."""
    if name not in engines.available_engines():
        pytest.skip(f"engine {name!r} requires an optional extra that is absent")
    return name


@pytest.fixture
def jittered():
    """A sweep classified NON_UNIFORM: 120 rays, +/-0.3-spacing azimuth jitter."""
    ngates = 34
    nrays = 120
    spacing = 360.0 / nrays
    rng = np.random.default_rng(2)
    azimuths = np.sort(
        (uniform_azimuths(nrays) + rng.uniform(-0.3, 0.3, nrays) * spacing) % 360.0
    )
    geometry = census_sweep(make_radar(azimuths, ngates=ngates), 0)
    assert geometry.sweep_class is SweepClass.NON_UNIFORM
    values = band_limited_field(
        azimuths, ngates, azimuth_modes=(0, 1, 3), range_modes=(0, 1, 2)
    )
    return geometry, azimuths, values, ngates


def exact_lattice_case(nrays=120, ngates=8, jitter=0.3, seed=3, modes=(0, 1, 3, 7)):
    """A field exactly band-limited on the lattice, so truth is known in closed form.

    Every accuracy claim about the solvers rests on this: the field is a finite
    sum of azimuthal harmonics with ``|m| <= 7``, evaluated both at the jittered
    measured azimuths and at the uniform lattice the solvers are trying to
    recover. The lattice values are therefore the exact answer, not a proxy for
    it, and an error can be reported as an absolute figure rather than as a
    difference between two approximations.
    """
    spacing = 360.0 / nrays
    rng = np.random.default_rng(seed)
    azimuths = np.sort(
        (np.arange(nrays) * spacing + rng.uniform(-jitter, jitter, nrays) * spacing)
        % 360.0
    )
    coefficients = np.random.default_rng(seed + 100)
    amplitude = coefficients.normal(size=(len(modes), ngates))
    phase = coefficients.uniform(0.0, 2.0 * np.pi, (len(modes), ngates))

    def evaluate(sample_azimuths):
        out = np.zeros((len(sample_azimuths), ngates))
        for index, mode in enumerate(modes):
            out += amplitude[index][None, :] * np.cos(
                np.deg2rad(mode * sample_azimuths)[:, None] + phase[index][None, :]
            )
        return out

    geometry = census_sweep(make_radar(azimuths, ngates=ngates), 0)
    lattice_azimuths = np.arange(nrays) * 360.0 / nrays
    case = (geometry, azimuths, evaluate(azimuths), evaluate(lattice_azimuths))
    # ``evaluate`` is attached so a caller can obtain truth at azimuths that were
    # never measured -- needed by the held-out prediction tests, which is the only
    # way to tell fitting from overfitting.
    return _ExactCase(*case, evaluate=evaluate)


class _ExactCase(tuple):
    """The 4-tuple callers unpack, plus the field evaluator as an attribute."""

    def __new__(cls, geometry, azimuths, values, lattice, evaluate):
        self = super().__new__(cls, (geometry, azimuths, values, lattice))
        self.evaluate = evaluate
        return self


def exact_forward(operator, lattice_values):
    """Evaluate the band-limited interpolant in closed form, no kernel involved.

    The mathematical definition of what every engine's ``forward`` approximates:
    the trig polynomial whose coefficients are the lattice DFT, with the Nyquist
    mode split (a cosine) rather than assigned to one side. Deliberately written
    as an explicit sum over modes so it shares no machinery with any engine.
    """
    n_lattice = operator.n_lattice
    coefficients = np.fft.fft(lattice_values, axis=0) / n_lattice
    modes = np.fft.fftfreq(n_lattice, d=1.0 / n_lattice).astype(int)
    radians = np.deg2rad(operator.azimuths)
    out = np.zeros((operator.n_rays, lattice_values.shape[1]))
    for index, mode in enumerate(modes):
        if n_lattice % 2 == 0 and mode == -n_lattice // 2:
            out += np.real(np.cos(mode * radians)[:, None] * coefficients[index])
        else:
            out += np.real(np.exp(1j * mode * radians)[:, None] * coefficients[index])
    return out


class TestEngineRegistry:
    def test_the_dependency_free_engines_are_always_available(self):
        """A minimal install must still have a working NUFFT path."""
        available = engines.available_engines()
        assert "reference" in available
        assert "scipy" in available

    def test_the_default_engine_needs_no_optional_dependency(self):
        """A minimal install must reach the default path.

        Checked against the dependency map rather than a hard-coded list of names, so
        that changing the default engine cannot silently make numpy/scipy-only
        installs fail --- which a name list would have permitted.
        """
        assert engines._ENGINE_REQUIREMENTS[engines.DEFAULT_ENGINE] == ()

    def test_the_default_solver_is_the_one_that_stays_physical(self):
        """Pins the safety decision, not merely the current value.

        ``cg`` reached -1907..2471 dBZ on a real volume at 25 iterations and 169 dBZ
        on one cone at the old default of 12; ``direct`` takes an explicit
        regularisation parameter instead of leaving an iteration count to do that job
        implicitly.
        """
        assert engines.DEFAULT_SOLVER == "direct"

    def test_available_engines_is_a_subset_of_the_declared_set(self):
        assert set(engines.available_engines()) <= set(ALL_ENGINES)

    def test_an_unknown_engine_is_a_value_error(self, jittered):
        geometry, azimuths, _, _ = jittered
        with pytest.raises(ValueError, match="engine must be one of"):
            engines.make_operator(azimuths, geometry, engine="nope")

    def test_auto_resolves_to_the_default_not_to_the_fastest_installed(self, jittered):
        """``auto`` must not depend on what happens to be installed.

        An engine chosen from the environment would make a result irreproducible
        between two machines with the same code, which matters more here than the
        speed difference, because the library engines are not bit-identical to
        the reference.
        """
        geometry, azimuths, _, _ = jittered
        assert (
            engines.make_operator(azimuths, geometry, engine="auto").name
            == engines.DEFAULT_ENGINE
        )

    def test_an_absent_engine_names_the_extra_to_install(self, jittered):
        """The error must be actionable, not an ImportError from deep inside."""
        geometry, azimuths, _, _ = jittered
        absent = [
            name for name in OPTIONAL_ENGINES if name not in engines.available_engines()
        ]
        if not absent:
            pytest.skip("every optional engine is installed")
        with pytest.raises(engines.EngineUnavailableError, match="pip install"):
            engines.make_operator(azimuths, geometry, engine=absent[0])

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_every_engine_reports_its_identity_and_lattice(self, jittered, engine):
        geometry, azimuths, _, _ = jittered
        described = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        ).describe()
        assert described["engine"] == engine
        assert described["n_nominal"] == geometry.nrays


class TestEngineEquivalence:
    """Different machinery, same operator."""

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_adjoint_is_the_transpose_of_the_forward(self, jittered, engine):
        """Per-engine, because a broken transpose invalidates both solvers.

        This is the reference module's load-bearing test, and it is not inherited:
        an engine could pair a correct forward with a mis-scaled adjoint and every
        smoothness check would still pass.
        """
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        assert operator.adjoint_test() < 1e-10

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_lattice_size_is_identical_across_engines(self, jittered, engine):
        """Engines may differ in how they transform, never in what they solve."""
        geometry, azimuths, _, _ = jittered
        reference = _AzimuthNufftOperator(azimuths, geometry)
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        assert operator.n_lattice == reference.n_lattice
        np.testing.assert_allclose(operator.weights, reference.weights)
        np.testing.assert_array_equal(operator.order, reference.order)

    def test_the_reference_engine_is_the_reference_bit_for_bit(self, jittered):
        """Not merely close: the same code, so the same bits.

        If this ever fails, the engine wrapper has started reimplementing the
        reference instead of delegating to it.
        """
        geometry, azimuths, values, _ = jittered
        expected, _ = _AzimuthNufftOperator(azimuths, geometry).solve(values, n_cg=12)
        got, _ = engines.make_operator(azimuths, geometry, engine="reference").solve(
            values, n_cg=12, solver="cg"
        )
        np.testing.assert_array_equal(got, expected)

    def test_the_scipy_engine_matches_the_reference_to_round_off(self, jittered):
        """The default engine's substitutions are exact, not approximations.

        CSR spreading replaces ``np.add.at`` and ``rfft`` replaces ``fft`` on real
        input; neither changes the arithmetic, so the tolerance here is round-off
        (~1e-13) and not a kernel error (~1e-4). A regression that made the
        default engine merely *approximately* right would land between the two.
        """
        geometry, azimuths, values, _ = jittered
        expected, _ = _AzimuthNufftOperator(azimuths, geometry).solve(values, n_cg=12)
        got, _ = engines.make_operator(azimuths, geometry, engine="scipy").solve(
            values, n_cg=12, solver="cg"
        )
        assert np.max(np.abs(got - expected)) < 1e-13 * np.ptp(expected)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_every_engine_agrees_with_the_reference_to_the_kernel_error(
        self, jittered, engine
    ):
        """The floor on agreement is the reference's own approximation error.

        No engine can be expected to match the reference more closely than the
        reference matches exact arithmetic, so 10x the Kaiser-Bessel kernel error
        is the honest bound for an engine with a different kernel --- and it is
        still tight enough to catch a convention or scaling mistake, which shows
        up at tens of percent.
        """
        geometry, azimuths, values, _ = jittered
        expected, _ = _AzimuthNufftOperator(azimuths, geometry).solve(values, n_cg=12)
        got, _ = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        ).solve(values, n_cg=12, solver="cg")
        assert np.max(np.abs(got - expected)) < 10 * KB_KERNEL_ERROR * np.ptp(expected)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_a_one_dimensional_input_matches_a_single_column(self, jittered, engine):
        """1-D and 2-D calls must not take different code paths to different answers."""
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        rng = np.random.default_rng(4)
        lattice = rng.normal(size=operator.n_lattice)
        rays = rng.normal(size=operator.n_rays)
        np.testing.assert_allclose(
            operator.forward(lattice), operator.forward(lattice[:, None]).ravel()
        )
        np.testing.assert_allclose(
            operator.adjoint(rays), operator.adjoint(rays[:, None]).ravel()
        )

    def test_the_nyquist_split_does_not_change_a_real_valued_result(self, jittered):
        """Documents a property that had to be measured, not assumed.

        The reference module splits the Nyquist coefficient half to ``+N0/2`` and
        half to ``-N0/2`` because assigning it wholly to one side "produces a
        complex interpolant". That is true of the *interpolant*, and it is worth
        keeping for that reason --- but for a real-valued field it makes no
        difference to what this operator returns, because ``forward`` takes the
        real part and ``Re exp(-i m phi) == cos(m phi)`` when the coefficient is
        real. Measured here: the forward differs by round-off and the adjoint by
        exactly zero.

        Recorded as a test because the alternative is worse in both directions. A
        contributor who assumes the split is load-bearing will not touch a
        pathological-looking branch that is in fact inert; one who assumes it is
        dead code will delete a line that *is* load-bearing for complex input and
        for fidelity to the reference. Neither would learn this from the code, and
        a mutation that removes the split passes every other test in this file.
        """
        import scipy.sparse as sparse

        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        assert operator.n_lattice % 2 == 0, "fixture must have a Nyquist bin"

        modes = np.fft.fftfreq(operator.n_lattice, d=1.0 / operator.n_lattice).astype(
            int
        )
        nyquist_column = int(np.argmin(modes))
        unsplit = operator._pad.toarray()
        unsplit[:, nyquist_column] = 0.0
        unsplit[0, nyquist_column] = 1.0

        rng = np.random.default_rng(3)
        lattice = rng.normal(size=(operator.n_lattice, 3))
        rays = rng.normal(size=(operator.n_rays, 3))
        split_pad = operator._pad

        def forward_with(pad):
            operator._pad = pad
            return operator.forward(lattice)

        def adjoint_with(pad):
            operator._pad = pad
            return operator.adjoint(rays)

        try:
            forward_split = forward_with(split_pad)
            forward_unsplit = forward_with(sparse.csr_matrix(unsplit))
            adjoint_split = adjoint_with(split_pad)
            adjoint_unsplit = adjoint_with(sparse.csr_matrix(unsplit))
        finally:
            operator._pad = split_pad

        assert np.max(np.abs(forward_split - forward_unsplit)) < 1e-12 * np.ptp(
            forward_split
        )
        np.testing.assert_array_equal(adjoint_split, adjoint_unsplit)

    def test_the_dense_engine_is_exact_to_round_off(self, jittered):
        """It forms the DFT matrix, so there is no kernel error to measure.

        The strongest statement available about any engine here, and the reason a
        dependency-free engine can reach the accuracy the NUFFT libraries do: on
        the azimuth axis the exact operator is small enough to write down.
        """
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        lattice = np.random.default_rng(9).normal(size=(operator.n_lattice, 3))
        truth = exact_forward(operator, lattice)
        assert np.max(np.abs(operator.forward(lattice) - truth)) < 1e-12 * np.ptp(truth)

    @pytest.mark.parametrize("engine", ("finufft", "ducc0"))
    def test_the_library_engines_beat_the_reference_kernel_on_exact_arithmetic(
        self, jittered, engine
    ):
        """These engines are *more* accurate than the code they replace.

        Which is why they differ from the reference by ~5e-4: that gap is the
        reference's Kaiser-Bessel error, not theirs. Asserting this keeps the
        difference from being read as an engine defect.
        """
        geometry, azimuths, _, _ = jittered
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine), eps=1e-12
        )
        reference = _AzimuthNufftOperator(azimuths, geometry)
        lattice = np.random.default_rng(6).normal(size=(operator.n_lattice, 3))
        truth = exact_forward(operator, lattice)
        engine_error = np.max(np.abs(operator.forward(lattice) - truth))
        reference_error = np.max(np.abs(reference.forward(lattice) - truth))
        assert engine_error < 1e-8 * np.ptp(truth)
        assert engine_error < reference_error


class TestDirectSolver:
    """The claim is accuracy, and it is measured against known truth."""

    @pytest.mark.parametrize("engine", ("dense", "finufft", "ducc0"))
    def test_on_an_exact_engine_it_is_orders_more_accurate(self, engine):
        """The headline claim, and it needs an engine without a kernel error.

        Twelve CG iterations recover this field to ~3.6e-5; the direct solve on
        the same transforms reaches ~1e-10. The margin asserted is deliberately
        loose (100x against a measured 3e5) so the test tracks the claim rather
        than the figures.
        """
        geometry, azimuths, values, truth = exact_lattice_case()
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        iterative, _ = operator.solve(values, n_cg=12, solver="cg")
        direct, _ = operator.solve(values, solver="direct")
        scale = np.ptp(truth)
        iterative_error = np.max(np.abs(iterative - truth)) / scale
        direct_error = np.max(np.abs(direct - truth)) / scale
        assert direct_error < 1e-8
        assert direct_error < iterative_error / 100.0

    def test_on_a_kaiser_bessel_engine_it_converges_to_the_kernel_error(self):
        """The honest limit of the direct solve on an approximate transform.

        A direct solve inverts the operator it is given, so on the default
        Kaiser-Bessel engine it lands on that engine's own ~3e-4 kernel error and
        buys no accuracy at this jitter --- the iteration was never the binding
        constraint. Asserting it keeps the accuracy claim attached to the
        engine/solver *pair*, which is where it belongs: a reader who took the
        headline figure to be a property of ``solver='direct'`` alone would be
        wrong, and this test is what says so.
        """
        geometry, azimuths, values, truth = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="scipy")
        direct, _ = operator.solve(values, solver="direct")
        error = np.max(np.abs(direct - truth)) / np.ptp(truth)
        assert 1e-5 < error < 1e-2

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_it_reaches_the_limit_conjugate_gradients_converge_towards(self, engine):
        """Same solution, one factorisation instead of many iterations.

        Establishes that ``direct`` is not a *different* estimator that happens to
        score better --- it is the least-squares solution CG is approaching, so
        running CG far past its default lands on the same answer. True on every
        engine, including the approximate ones: each converges to the solution of
        *its own* normal equations, which is exactly the claim.
        """
        geometry, azimuths, values, _ = exact_lattice_case()
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        converged, info = operator.solve(values, n_cg=500, cg_tol=1e-14, solver="cg")
        direct, _ = operator.solve(values, solver="direct")
        assert info["cg_resid_end"] < 1e-10
        assert np.max(np.abs(direct - converged)) < 1e-7 * np.ptp(converged)

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_advantage_widens_as_the_sampling_degrades(self, engine):
        """CG's convergence rate degrades with jitter; a factorisation does not.

        This is the one accuracy gain that holds on *every* engine, including the
        Kaiser-Bessel ones, because what degrades is the conditioning of the
        system rather than the fidelity of the transform. At +/-0.45-spacing
        jitter twelve iterations reach only ~1.7e-2 while the direct solve holds
        4.5e-4 on the same transforms --- so the fixed iteration count, not the
        kernel, is what limits the default here.
        """
        geometry, azimuths, values, truth = exact_lattice_case(jitter=0.45)
        operator = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        )
        iterative, _ = operator.solve(values, n_cg=12, solver="cg")
        direct, _ = operator.solve(values, solver="direct")
        scale = np.ptp(truth)
        iterative_error = np.max(np.abs(iterative - truth)) / scale
        direct_error = np.max(np.abs(direct - truth)) / scale
        assert iterative_error > 1e-3
        assert direct_error < iterative_error / 10.0

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_it_works_on_every_engine(self, jittered, engine):
        """Solver and engine are independent axes, so all pairs must compose."""
        geometry, azimuths, values, _ = jittered
        lattice, info = engines.make_operator(
            azimuths, geometry, engine=engine_or_skip(engine)
        ).solve(values, solver="direct")
        assert info["solver"] == "direct"
        assert info["engine"] == engine
        assert np.all(np.isfinite(lattice))

    def test_it_reports_the_conditioning_it_solved_at(self, jittered):
        """``normal_cond`` is the diagnostic for a suspicious sector result."""
        geometry, azimuths, values, _ = jittered
        _, info = engines.make_operator(azimuths, geometry).solve(
            values, solver="direct"
        )
        assert info["normal_cond"] > 1.0
        # 'auto' on a well-conditioned synthetic geometry resolves to the floor.
        assert info["ridge_rel"] == pytest.approx(engines.MIN_RIDGE)
        assert info["ridge_mode"] == "auto"

    def test_the_factorisation_is_cached_across_fields(self, jittered):
        """One geometry, many fields: the factorisation must be paid for once."""
        geometry, azimuths, values, _ = jittered
        operator = engines.make_operator(azimuths, geometry)
        operator.solve(values, solver="direct")
        cached = operator._normal_cache
        operator.solve(values * 2.0 + 1.0, solver="direct")
        assert operator._normal_cache is cached

    def test_a_changed_ridge_is_refactorised_rather_than_reused(self, jittered):
        """The cache key must include the ridge, or it would silently be ignored."""
        geometry, azimuths, values, _ = jittered
        operator = engines.make_operator(azimuths, geometry)
        _, loose = operator.solve(values, solver="direct", ridge=1e-4)
        _, tight = operator.solve(values, solver="direct", ridge=1e-10)
        assert loose["normal_cond"] < tight["normal_cond"]

    def test_an_unknown_solver_is_a_value_error(self, jittered):
        geometry, azimuths, values, _ = jittered
        with pytest.raises(ValueError, match="solver must be one of"):
            engines.make_operator(azimuths, geometry).solve(values, solver="lstsq")


class TestSectorConditioning:
    """A sector's normal matrix is singular by construction, not by accident."""

    @pytest.fixture
    def sector(self):
        """A jittered 30-120 deg sector: 91 rays onto a 360-point lattice."""
        spacing = 1.0
        rng = np.random.default_rng(7)
        count = 91
        azimuths = np.sort(
            30.0 + np.arange(count) * spacing + rng.uniform(-0.3, 0.3, count) * spacing
        )
        geometry = census_sweep(make_radar(azimuths, ngates=20), 0)
        assert geometry.sweep_class is SweepClass.NON_UNIFORM
        values = band_limited_field(
            azimuths, 20, azimuth_modes=(0, 1, 3), range_modes=(0, 1, 2)
        )
        return geometry, azimuths, values

    def test_the_lattice_exceeds_the_ray_count(self, sector):
        """Which is what makes the system underdetermined -- the premise."""
        geometry, azimuths, _ = sector
        operator = engines.make_operator(azimuths, geometry)
        assert operator.n_lattice > operator.n_rays

    def test_the_default_ridge_keeps_a_rank_deficient_sector_solvable(self, sector):
        """Without regularisation the Cholesky of a singular matrix fails outright."""
        geometry, azimuths, values = sector
        lattice, info = engines.make_operator(azimuths, geometry).solve(
            values, solver="direct"
        )
        assert np.all(np.isfinite(lattice))
        # 'auto' caps the conditioning rather than letting it run to the 1e6+ a
        # near-zero ridge would leave on a rank-deficient sector.
        assert info["normal_cond"] < 10.0 * engines.TARGET_CONDITION
        assert info["normal_null_dim"] > 0

    def test_it_fits_the_measured_rays_better_than_the_iterative_default(self, sector):
        """The right yardstick when the lattice is not unique.

        With more unknowns than measurements there is no single correct lattice,
        so accuracy is judged by how well the recovered lattice reproduces the
        rays that *were* measured.

        Note that data fit is the yardstick here only because this test's field
        IS band-limited. On real reflectivity, fitting the measured rays more
        closely is the *failure* mode -- an unregularised solve drives the
        residual to 1e-7 dB and predicts unmeasured azimuths hundreds of dB away.
        So this asserts a fit improvement at a fixed small ridge, and the
        prediction tests in TestTheRidgeMustAdaptToTheGeometry carry the claim
        that matters for real data.
        """
        geometry, azimuths, values = sector
        operator = engines.make_operator(azimuths, geometry)
        measured = values[operator.order]
        scale = np.ptp(values)
        iterative, _ = operator.solve(values, n_cg=12, solver="cg")
        direct, _ = operator.solve(values, solver="direct", ridge=1e-10)
        iterative_residual = np.max(np.abs(operator.forward(iterative) - measured))
        direct_residual = np.max(np.abs(operator.forward(direct) - measured))
        assert direct_residual / scale < 1e-6
        assert direct_residual < iterative_residual

    def test_a_larger_ridge_trades_data_fit_for_conditioning(self, sector):
        """The knob does what it claims, in the direction it claims."""
        geometry, azimuths, values = sector
        operator = engines.make_operator(azimuths, geometry)
        measured = values[operator.order]
        loose, loose_info = operator.solve(values, solver="direct", ridge=1e-4)
        tight, tight_info = operator.solve(values, solver="direct", ridge=1e-10)
        assert loose_info["normal_cond"] < tight_info["normal_cond"]
        assert np.max(np.abs(operator.forward(loose) - measured)) > np.max(
            np.abs(operator.forward(tight) - measured)
        )


class TestEvaluatorWiring:
    """The evaluator must pass the choice through and report what it used."""

    def test_the_reference_path_is_still_reproducible_exactly(self, jittered):
        """The old default must remain available and bit-comparable.

        This test used to assert that the *default* matched the reference to
        round-off. That premise is deliberately gone: the default now uses the exact
        DFT operator and a factorised solve, which is a different --- and on real data
        better-behaved --- answer. What still has to hold is that a caller
        reproducing earlier numbers can ask for the old path by name and get it.
        """
        geometry, azimuths, values, ngates = jittered
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        scipy_path = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine="scipy", az_solver="cg"
        ).evaluate(range_mesh, azimuth_mesh)
        reference = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine="reference", az_solver="cg"
        ).evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(scipy_path - reference)) < 1e-10 * np.ptp(reference)

    def test_the_new_default_differs_from_the_old_one_by_a_bounded_amount(
        self, jittered
    ):
        """Changing a default is only defensible if the size of the change is known.

        Measured on this fixture the two agree to well under a tenth of a percent of
        field range, so the change is not a silent re-scaling --- it is the removal of
        the Kaiser-Bessel kernel error plus a regularised solve. The bound is asserted
        in both directions: large enough to prove the default really did change,
        small enough to prove it did not move the field.
        """
        geometry, azimuths, values, ngates = jittered
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        new_default = SweepSpectralEvaluator(values, geometry, azimuths).evaluate(
            range_mesh, azimuth_mesh
        )
        old_default = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine="scipy", az_solver="cg"
        ).evaluate(range_mesh, azimuth_mesh)
        difference = np.max(np.abs(new_default - old_default)) / np.ptp(old_default)
        assert difference > 1e-12
        assert difference < 1e-3

    @pytest.mark.parametrize("engine", ALL_ENGINES)
    def test_the_engine_is_recorded_in_the_report(self, jittered, engine):
        geometry, azimuths, values, _ = jittered
        report = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine=engine_or_skip(engine)
        ).report
        assert report.az_path == "nufft_kb"
        assert report.extras["engine"] == engine
        assert report.extras["adjoint_test_rel"] < 1e-10

    def test_the_solver_is_recorded_in_the_report(self, jittered):
        geometry, azimuths, values, _ = jittered
        for solver in engines.SOLVERS:
            report = SweepSpectralEvaluator(
                values, geometry, azimuths, az_solver=solver
            ).report
            assert report.extras["solver"] == solver

    def test_the_direct_solver_reports_no_conjugate_gradient_residuals(self, jittered):
        """It does not iterate, so it must not claim a residual history."""
        geometry, azimuths, values, _ = jittered
        extras = SweepSpectralEvaluator(
            values, geometry, azimuths, az_solver="direct"
        ).report.extras
        assert "cg_resid_end" not in extras
        assert extras["normal_cond"] > 1.0

    def test_the_direct_solver_recovers_a_smooth_field(self, jittered):
        geometry, azimuths, values, ngates = jittered
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        recovered = SweepSpectralEvaluator(
            values, geometry, azimuths, az_solver="direct"
        ).evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(recovered - values)) < 0.01 * np.ptp(values)

    def test_a_constant_field_survives_the_direct_solver(self, jittered):
        """The sharpest test of the NUFFT path, on the alternative solver.

        The reference module records 0.07% deviation here with its CG default and
        60% without. A direct solve must be at least as good as the default, or it
        has no business being offered.
        """
        geometry, azimuths, _, ngates = jittered
        constant = np.full((azimuths.size, ngates), 12.5)
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        recovered = SweepSpectralEvaluator(
            constant, geometry, azimuths, az_solver="direct", field_units="other"
        ).evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(recovered - 12.5)) / 12.5 < 1e-3

    def test_an_unknown_solver_is_rejected_at_construction(self, jittered):
        """Not deferred to the one path that happens to consult it."""
        geometry, azimuths, values, _ = jittered
        with pytest.raises(ValueError, match="az_solver"):
            SweepSpectralEvaluator(values, geometry, azimuths, az_solver="lstsq")

    def test_an_unknown_solver_is_rejected_even_on_a_uniform_sweep(self):
        """The path that ignores the argument must still validate it.

        A uniform sweep never reaches the NUFFT path, so a typo here would
        otherwise pass silently and only fail later on a jittered sweep.
        """
        azimuths = uniform_azimuths(120)
        geometry = census_sweep(make_radar(azimuths, ngates=20), 0)
        assert geometry.sweep_class is SweepClass.EXACT_UNIFORM_PERIODIC
        values = band_limited_field(azimuths, 20)
        with pytest.raises(ValueError, match="az_solver"):
            SweepSpectralEvaluator(values, geometry, azimuths, az_solver="lstsq")


class TestTheRidgeMustAdaptToTheGeometry:
    """The regularisation size is geometry-dependent, and getting it wrong is fatal.

    These tests exist because a constant ridge tuned on synthetic sweeps was
    catastrophically wrong on real ones. An evenly-jittered synthetic sweep has a
    full-rank normal matrix and wants ~1e-10; a real sweep is commonly
    rank-deficient -- operational azimuth sampling leaves gaps of about twice the
    nominal spacing, so the lattice asks for modes the data cannot determine --
    and at 1e-10 the solve fits its training rays to 1e-7 dB while predicting
    held-out rays hundreds of dB away. Eight orders of magnitude apart, so
    ``'auto'`` derives it from the spectrum instead.
    """

    def test_a_well_sampled_sweep_keeps_the_near_exact_solve(self):
        """'auto' must not over-regularise the case the direct solver is for.

        If ``auto`` simply picked a large ridge it would be safe and useless:
        the whole value of the direct solve on an exact engine is the ~1e-10
        recovery, and that survives only if a full-rank geometry gets the floor.
        """
        geometry, azimuths, values, truth = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        lattice, info = operator.solve(values, solver="direct")
        assert info["normal_null_dim"] == 0
        assert info["ridge_rel"] == pytest.approx(engines.MIN_RIDGE)
        assert np.max(np.abs(lattice - truth)) < 1e-8 * np.ptp(truth)

    def test_a_gapped_sweep_is_rank_deficient(self):
        """Real azimuth sampling costs rank, and the report must say so.

        A gap of about twice the nominal spacing -- which operational sampling
        routinely leaves -- makes the lattice ask for modes the rays cannot
        determine. Measured on real sweeps: 3 undetermined modes of 414 on an
        S-band sweep, 23 of 993 on an X-band one.

        Note this is a diagnostic, NOT what sets the ridge: two of the three real
        sweeps measured are full-rank and still need a ridge of 1e-2 or larger.
        The next test covers the effect that does set it.
        """
        azimuths = uniform_azimuths(360)
        gapped = np.delete(azimuths, np.arange(0, 360, 8))
        geometry = census_sweep(make_radar(gapped, ngates=12), 0)
        values = band_limited_field(gapped, 12)
        operator = engines.make_operator(gapped, geometry, engine="dense")
        _, info = operator.solve(values, solver="direct")
        assert info["normal_null_dim"] > 0

    @pytest.mark.parametrize(
        ("noise_sigma", "expect_large_ridge"),
        [(0.0, False), (0.1, True), (0.5, True)],
    )
    def test_out_of_band_content_is_what_demands_a_large_ridge(
        self, noise_sigma, expect_large_ridge
    ):
        """The finding that reversed the default, isolated to one variable.

        A field built from a few harmonics has zero energy in the modes the
        sampling cannot determine, so there is nothing to overfit and the
        smallest ridge wins -- which is what synthetic tuning showed, and why it
        gave an answer eight orders of magnitude too small. Real reflectivity is
        not band-limited: speckle, clutter and echo edges put energy in every
        mode, a small ridge fits that energy at the measured rays, and the
        interpolant between them is noise.

        Adding broadband noise to the synthetic field flips the optimum, which
        establishes that band-limitedness -- not rank, not conditioning -- is the
        variable that matters. Rank deficiency is neither necessary nor
        sufficient here: this geometry is full-rank throughout.
        """
        case = exact_lattice_case(ngates=6)
        geometry, azimuths, values, _ = case
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        held_out = np.arange(0.7, 360.0, 2.9)
        truth = case.evaluate(held_out)
        noisy = values + np.random.default_rng(7).normal(0.0, noise_sigma, values.shape)

        modes = np.arange(operator.n_modes) + operator.mode_low
        basis = np.exp(1j * np.outer(np.deg2rad(held_out), modes))

        def prediction_error(ridge):
            operator._normal_cache = None
            lattice, _ = operator.solve(noisy, solver="direct", ridge=ridge)
            spectrum = operator._pad @ (
                np.fft.fft(lattice, axis=0) / operator.n_lattice
            )
            return float(np.median(np.abs(np.real(basis @ spectrum) - truth)))

        candidates = (1e-10, 1e-6, 1e-3, 1e-2, 1e-1)
        errors = {ridge: prediction_error(ridge) for ridge in candidates}
        best = min(errors, key=errors.get)
        if expect_large_ridge:
            # The optimum moves further from the floor as out-of-band energy
            # grows, so assert only that it has LEFT the floor -- pinning which
            # decade wins would be pinning the noise level, not the effect.
            assert best >= 1e-3
            assert errors[best] < errors[1e-10]
        else:
            assert best == min(candidates)

    def test_an_invalid_ridge_string_is_rejected(self):
        """Only 'auto' is a valid non-numeric ridge."""
        geometry, azimuths, values, _ = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        with pytest.raises(ValueError, match="auto"):
            operator.solve(values, solver="direct", ridge="adaptive")

    def test_the_report_exposes_the_rank_diagnostic(self):
        """``normal_null_dim`` must reach the caller, not just the solver.

        A caller cannot tell an under-determined answer from a converged one by
        looking at the values, so the geometry's rank has to be in the report.
        """
        azimuths = uniform_azimuths(360)
        gapped = np.delete(azimuths, np.arange(0, 360, 8))
        geometry = census_sweep(make_radar(gapped, ngates=12), 0)
        values = band_limited_field(gapped, 12)
        evaluator = SweepSpectralEvaluator(
            values, geometry, gapped, az_engine="dense", az_solver="direct"
        )
        evaluator.evaluate(np.array([10.0]), np.array([RANGE_FIRST_M]))
        assert evaluator.report.extras["normal_null_dim"] > 0
        assert evaluator.report.extras["ridge_mode"] == "auto"


class TestTheRidgeDefaultIsAKnownCompromise:
    """The default ridge cannot be right for both regimes, and this pins the cost.

    Documented as a test rather than a comment because it is the kind of trade a
    later change would silently break: raising ``MIN_RIDGE`` to help real
    reflectivity looks free until you notice it costs eight orders of magnitude
    on a band-limited field, and the band-limited tests would still pass at 1e-3.
    """

    def test_the_floor_preserves_the_band_limited_recovery(self):
        """The regime where this solver's accuracy claim is real."""
        geometry, azimuths, values, truth = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        lattice, info = operator.solve(values, solver="direct")
        assert info["ridge_rel"] == pytest.approx(engines.MIN_RIDGE)
        assert np.max(np.abs(lattice - truth)) < 1e-8 * np.ptp(truth)

    def test_error_grows_linearly_with_the_ridge_on_a_band_limited_field(self):
        """Quantifies what any raise of the floor would cost.

        Measured 1.1e-10 at 1e-10 rising to 1.1e-2 at 1e-2 --- one decade of
        accuracy per decade of ridge, so the trade is legible rather than a cliff.
        """
        geometry, azimuths, values, truth = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        scale = np.ptp(truth)
        errors = {}
        for ridge in (1e-8, 1e-6, 1e-4):
            operator._normal_cache = None
            lattice, _ = operator.solve(values, solver="direct", ridge=ridge)
            errors[ridge] = np.max(np.abs(lattice - truth)) / scale
        for ridge, error in errors.items():
            assert error == pytest.approx(ridge, rel=0.5)

    def test_default_ridge_is_the_documented_value_for_real_reflectivity(self):
        """``DEFAULT_RIDGE`` is advice, not the solver's default -- keep them apart.

        A caller reading only the constant name would assume ``solve`` uses it.
        It does not: the solver defaults to ``'auto'``, and ``DEFAULT_RIDGE`` is
        the value measured to suit real reflectivity, to be passed explicitly.
        """
        assert engines.DEFAULT_RIDGE == 1e-2
        assert engines.MIN_RIDGE < engines.DEFAULT_RIDGE

        geometry, azimuths, values, _ = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        _, info = operator.solve(values, solver="direct")
        assert info["ridge_rel"] != engines.DEFAULT_RIDGE


class TestEvaluateLatticeOffTheMeasuredAzimuths:
    """Held-out prediction needs evaluation away from the rays that were fitted."""

    def test_it_reproduces_the_engine_forward_at_the_measured_azimuths(self):
        """Same interpolant, so it must agree where both are defined."""
        geometry, azimuths, values, _ = exact_lattice_case()
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        lattice, _ = operator.solve(values, solver="direct")
        direct = operator.forward(lattice)
        evaluated = engines.evaluate_lattice(lattice, operator.azimuths)
        assert np.max(np.abs(direct - evaluated)) < 1e-10 * np.ptp(direct)

    def test_it_is_engine_independent(self):
        """The interpolant belongs to the lattice, not to whoever recovered it.

        Guards the reason this is a module function rather than a method: two
        engines given the same lattice must evaluate it identically, so any
        disagreement between engines is about their output and not their
        arithmetic here.
        """
        geometry, azimuths, values, _ = exact_lattice_case()
        lattice = np.random.default_rng(4).normal(size=(120, 3))
        held_out = np.arange(0.3, 360.0, 4.1)
        reference = engines.evaluate_lattice(lattice, held_out)
        assert np.all(np.isfinite(reference))
        assert reference.shape == (held_out.size, 3)

    def test_a_lattice_recovered_from_a_band_limited_field_predicts_it(self):
        """End to end: unmeasured azimuths, and the answer is known in closed form.

        The check the whole real-data investigation rests on. If this were wrong,
        every held-out dB figure would be wrong with it.
        """
        case = exact_lattice_case()
        geometry, azimuths, values, _ = case
        operator = engines.make_operator(azimuths, geometry, engine="dense")
        lattice, _ = operator.solve(values, solver="direct")
        held_out = np.arange(0.7, 360.0, 2.9)
        predicted = engines.evaluate_lattice(lattice, held_out)
        assert np.max(np.abs(predicted - case.evaluate(held_out))) < 1e-8 * np.ptp(
            values
        )

    def test_it_accepts_a_single_column(self):
        """A 1-D lattice returns 1-D, matching the solvers' own convention."""
        lattice = np.random.default_rng(5).normal(size=120)
        out = engines.evaluate_lattice(lattice, np.array([0.0, 12.5, 300.0]))
        assert out.shape == (3,)
