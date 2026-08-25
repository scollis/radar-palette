"""Unit tests for band-limited spectral evaluation of a single sweep.

Three azimuth paths exist, chosen from the sweep's class, and they have different
guarantees --- so they are tested to different standards:

``EXACT_UNIFORM_PERIODIC``
    The Fourier series is an *exact* band-limited interpolant. Tests demand
    machine precision against analytically constructed band-limited fields, and
    exact recovery at the original sample points.
``UNIFORM_PARTIAL_SECTOR``
    The gap is closed artificially, so only continuity and in-sector fidelity are
    asserted; values in the synthesised gap are explicitly *not* meaningful.
``NON_UNIFORM``
    Convolution gridding plus optional least-squares refinement. The load-bearing
    test is the adjoint identity, because that is what makes conjugate gradients on
    the normal equations legitimate.

Two guards get dedicated classes because they encode measured findings rather than
preferences: interpolation must run in dBZ rather than linear Z
(``TestLinearReflectivityGuard``), and the input must be gap-free
(``TestGapFreeRequirement``).
"""

from __future__ import annotations

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from radar_palette.gridding import (  # noqa: E402
    DBZ_PHYSICAL_CEILING,
    DEFAULT_CG_ITERATIONS,
    EvalReport,
    LinearReflectivityError,
    SweepClass,
    SweepSpectralEvaluator,
    census_sweep,
    fill_sweep,
    to_dbz,
    to_linear,
)

RANGE_FIRST_M = 117.9
RANGE_SPACING_M = 119.92


def make_radar(azimuths_deg, ngates=64, fixed_angle_deg=0.5):
    """A single-sweep volume with exactly the azimuth sampling requested."""
    azimuths_deg = np.asarray(azimuths_deg, dtype="float64")
    radar = pyart.testing.make_empty_ppi_radar(ngates, azimuths_deg.size, 1)
    radar.range["data"] = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(
        ngates, dtype="float64"
    )
    radar.azimuth["data"] = azimuths_deg
    radar.elevation["data"] = np.full(azimuths_deg.size, fixed_angle_deg)
    radar.fixed_angle["data"] = np.array([fixed_angle_deg])
    radar.add_field(
        "reflectivity",
        {
            "data": np.ma.masked_invalid(
                np.zeros((azimuths_deg.size, ngates), dtype="float32")
            ),
            "units": "dBZ",
            "_FillValue": -9999.0,
        },
        replace_existing=True,
    )
    return radar


def uniform_azimuths(nrays, start_deg=0.0):
    return (start_deg + (360.0 / nrays) * np.arange(nrays, dtype="float64")) % 360.0


def band_limited_field(
    azimuths_deg,
    ngates,
    azimuth_modes=(0, 1, 3, 7),
    range_modes=(0, 1, 2, 5),
    seed=1,
):
    """A field strictly band-limited in both axes of the evaluator's basis.

    Azimuth uses the Fourier basis of period 360 degrees; range uses the cosine
    basis of the mirror extension, period ``(2 * ngates - 2)`` gates. A field built
    this way is representable *exactly*, so the exact path must reproduce it to
    floating-point precision --- which is a far stronger test than a smooth-looking
    field that merely has small high-order content.
    """
    rng = np.random.default_rng(seed)
    azimuth_rad = np.deg2rad(np.asarray(azimuths_deg, dtype="float64"))
    gate_index = np.arange(ngates, dtype="float64")
    mirror_period = 2 * ngates - 2

    field = np.zeros((azimuth_rad.size, ngates))
    for azimuth_mode in azimuth_modes:
        for range_mode in range_modes:
            amplitude = rng.normal()
            phase = rng.uniform(0.0, 2.0 * np.pi)
            field += (
                amplitude
                * np.cos(azimuth_mode * azimuth_rad + phase)[:, None]
                * np.cos(2.0 * np.pi * range_mode * gate_index / mirror_period)[None, :]
            )
    return field


def evaluate_truth(azimuth_deg, gate_index, azimuth_modes, range_modes, ngates, seed=1):
    """The same analytic field as :func:`band_limited_field`, at arbitrary points."""
    rng = np.random.default_rng(seed)
    azimuth_rad = np.deg2rad(np.asarray(azimuth_deg, dtype="float64"))
    mirror_period = 2 * ngates - 2
    total = np.zeros(np.shape(azimuth_rad))
    for azimuth_mode in azimuth_modes:
        for range_mode in range_modes:
            amplitude = rng.normal()
            phase = rng.uniform(0.0, 2.0 * np.pi)
            total += (
                amplitude
                * np.cos(azimuth_mode * azimuth_rad + phase)
                * np.cos(2.0 * np.pi * range_mode * gate_index / mirror_period)
            )
    return total


@pytest.fixture
def exact_sweep():
    """An exactly-sampled full-circle sweep and a band-limited field on it."""
    nrays, ngates = 72, 34
    azimuths = uniform_azimuths(nrays)
    radar = make_radar(azimuths, ngates=ngates)
    geometry = census_sweep(radar, 0)
    assert geometry.sweep_class is SweepClass.EXACT_UNIFORM_PERIODIC
    return geometry, azimuths, band_limited_field(azimuths, ngates), ngates


class TestExactPathAccuracy:
    """The exact path is an interpolant, not an approximation. Hold it to that."""

    def test_reproduces_the_original_samples(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(values.shape[1])
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        recovered = evaluator.evaluate(range_mesh, azimuth_mesh)
        np.testing.assert_allclose(recovered, values, atol=1e-9)

    def test_recovers_a_band_limited_field_off_lattice(self, exact_sweep):
        """Off-lattice is where an interpolant earns the name."""
        geometry, azimuths, values, ngates = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        rng = np.random.default_rng(7)
        target_azimuths = rng.uniform(0.0, 360.0, 400)
        target_gates = rng.uniform(0.0, ngates - 1, 400)
        target_ranges = RANGE_FIRST_M + RANGE_SPACING_M * target_gates
        expected = evaluate_truth(
            target_azimuths, target_gates, (0, 1, 3, 7), (0, 1, 2, 5), ngates
        )
        np.testing.assert_allclose(
            evaluator.evaluate(target_ranges, target_azimuths), expected, atol=1e-8
        )

    def test_is_real_valued_everywhere(self, exact_sweep):
        """The Nyquist bin must be split, or the interpolant goes complex."""
        geometry, azimuths, values, ngates = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        rng = np.random.default_rng(3)
        result = evaluator.evaluate(
            RANGE_FIRST_M + RANGE_SPACING_M * rng.uniform(0, ngates - 1, 50),
            rng.uniform(0, 360, 50),
        )
        assert np.isrealobj(result)
        assert np.all(np.isfinite(result))

    def test_azimuth_is_periodic_across_the_branch_cut(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        radius = RANGE_FIRST_M + RANGE_SPACING_M * 5.0
        just_below = evaluator.evaluate(radius, 359.999)
        just_above = evaluator.evaluate(radius, 0.001)
        assert float(just_below) == pytest.approx(float(just_above), abs=1e-3)

    def test_equivalent_azimuths_give_equal_values(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        radius = RANGE_FIRST_M + RANGE_SPACING_M * 9.0
        assert float(evaluator.evaluate(radius, 47.0)) == pytest.approx(
            float(evaluator.evaluate(radius, 407.0)), abs=1e-9
        )

    def test_ray_order_does_not_matter(self, exact_sweep):
        """Rays arrive in scan order; the evaluator sorts internally."""
        geometry, azimuths, values, _ = exact_sweep
        shuffle = np.random.default_rng(11).permutation(azimuths.size)
        ordered = SweepSpectralEvaluator(values, geometry, azimuths)
        shuffled = SweepSpectralEvaluator(values[shuffle], geometry, azimuths[shuffle])
        radius = RANGE_FIRST_M + RANGE_SPACING_M * 4.0
        probes = np.linspace(0.0, 359.0, 61)
        np.testing.assert_allclose(
            ordered.evaluate(np.full_like(probes, radius), probes),
            shuffled.evaluate(np.full_like(probes, radius), probes),
            atol=1e-9,
        )

    def test_preserves_a_constant_field(self, exact_sweep):
        geometry, azimuths, _, ngates = exact_sweep
        constant = np.full((azimuths.size, ngates), -7.25)
        evaluator = SweepSpectralEvaluator(constant, geometry, azimuths)
        probes = np.linspace(0.0, 359.0, 37)
        result = evaluator.evaluate(
            np.full_like(probes, RANGE_FIRST_M + RANGE_SPACING_M * 3.5), probes
        )
        np.testing.assert_allclose(result, -7.25, atol=1e-9)

    def test_scales_linearly(self, exact_sweep):
        """Linearity is a property of the transform; a scaled field scales exactly."""
        geometry, azimuths, values, ngates = exact_sweep
        probes = np.linspace(1.0, 358.0, 29)
        ranges = np.full_like(probes, RANGE_FIRST_M + RANGE_SPACING_M * 6.5)
        single = SweepSpectralEvaluator(values, geometry, azimuths).evaluate(
            ranges, probes
        )
        tripled = SweepSpectralEvaluator(3.0 * values, geometry, azimuths).evaluate(
            ranges, probes
        )
        np.testing.assert_allclose(tripled, 3.0 * single, atol=1e-9)

    def test_reports_the_exact_path(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.az_path == "fft_exact"
        assert evaluator.report.sweep_class == SweepClass.EXACT_UNIFORM_PERIODIC.value

    def test_lattice_residual_is_reported_and_tiny(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.extras["lattice_resid_max_deg"] < 1e-9


class TestRangeIsNotPeriodic:
    """Range must be mirror-extended: treating it as periodic dominates the error."""

    def test_mirror_extension_doubles_the_range_axis(self, exact_sweep):
        geometry, azimuths, values, ngates = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.Nr_ext == 2 * ngates - 2

    def test_a_range_ramp_does_not_ring_at_the_ends(self, exact_sweep):
        """A ramp has a large end-to-end jump; periodic range would ring badly."""
        geometry, azimuths, _, ngates = exact_sweep
        ramp = np.tile(np.linspace(-20.0, 40.0, ngates), (azimuths.size, 1))
        evaluator = SweepSpectralEvaluator(ramp, geometry, azimuths)
        gate_index = np.linspace(0.0, ngates - 1, 400)
        result = evaluator.evaluate(
            RANGE_FIRST_M + RANGE_SPACING_M * gate_index,
            np.full_like(gate_index, 30.0),
        )
        assert result.min() > -20.0 - 0.5
        assert result.max() < 40.0 + 0.5

    def test_interpolant_passes_through_gates_of_a_ramp(self, exact_sweep):
        geometry, azimuths, _, ngates = exact_sweep
        ramp = np.tile(np.linspace(0.0, 10.0, ngates), (azimuths.size, 1))
        evaluator = SweepSpectralEvaluator(ramp, geometry, azimuths)
        gate_index = np.arange(ngates, dtype="float64")
        result = evaluator.evaluate(
            RANGE_FIRST_M + RANGE_SPACING_M * gate_index,
            np.full_like(gate_index, 12.0),
        )
        np.testing.assert_allclose(result, ramp[0], atol=1e-8)

    def test_rejects_too_few_gates(self, exact_sweep):
        geometry, azimuths, _, _ = exact_sweep
        with pytest.raises(ValueError, match="gates"):
            SweepSpectralEvaluator(np.zeros((azimuths.size, 2)), geometry, azimuths)


class TestFastPath:
    def test_agrees_with_the_direct_path(self, exact_sweep):
        geometry, azimuths, values, ngates = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        rng = np.random.default_rng(5)
        probe_azimuths = rng.uniform(0.0, 360.0, 200)
        probe_gates = rng.uniform(1.0, ngates - 2, 200)
        probe_ranges = RANGE_FIRST_M + RANGE_SPACING_M * probe_gates
        direct = evaluator.evaluate(probe_ranges, probe_azimuths)
        fast = evaluator.evaluate_fast(probe_ranges, probe_azimuths, up_az=8, up_r=8)
        assert np.max(np.abs(fast - direct)) < 0.05 * np.ptp(direct)

    def test_higher_upsampling_reduces_the_gap(self, exact_sweep):
        """The fast path is bilinear on an upsampled lattice, so it must converge."""
        geometry, azimuths, values, ngates = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        rng = np.random.default_rng(6)
        probe_azimuths = rng.uniform(0.0, 360.0, 150)
        probe_gates = rng.uniform(1.0, ngates - 2, 150)
        probe_ranges = RANGE_FIRST_M + RANGE_SPACING_M * probe_gates
        direct = evaluator.evaluate(probe_ranges, probe_azimuths)
        coarse = np.max(
            np.abs(
                evaluator.evaluate_fast(probe_ranges, probe_azimuths, up_az=2, up_r=2)
                - direct
            )
        )
        fine = np.max(
            np.abs(
                evaluator.evaluate_fast(probe_ranges, probe_azimuths, up_az=8, up_r=8)
                - direct
            )
        )
        assert fine < coarse

    def test_dense_lattice_is_cached(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.dense(4, 4) is evaluator.dense(4, 4)

    def test_dense_has_the_upsampled_shape(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        dense = evaluator.dense(up_az=3, up_r=2)
        assert dense.shape == (evaluator.report.M_az * 3, evaluator.report.Nr_ext * 2)

    def test_range_is_clamped_inside_the_measured_span(self, exact_sweep):
        """Beyond the last gate there is no data; the fast path must not wrap."""
        geometry, azimuths, values, ngates = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        beyond = RANGE_FIRST_M + RANGE_SPACING_M * (ngates + 20)
        at_edge = RANGE_FIRST_M + RANGE_SPACING_M * (ngates - 1)
        assert float(evaluator.evaluate_fast(beyond, 45.0)) == pytest.approx(
            float(evaluator.evaluate_fast(at_edge, 45.0)), abs=1e-6
        )

    def test_preserves_input_shape(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        ranges = np.full((4, 5), RANGE_FIRST_M + RANGE_SPACING_M * 3.0)
        result = evaluator.evaluate_fast(ranges, np.full((4, 5), 100.0))
        assert result.shape == (4, 5)

    def test_direct_path_preserves_input_shape(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        ranges = np.full((3, 6), RANGE_FIRST_M + RANGE_SPACING_M * 2.0)
        assert evaluator.evaluate(ranges, np.full((3, 6), 77.0)).shape == (3, 6)


class TestSectorPath:
    """Only in-sector values are meaningful; the gap fill exists to stop ringing."""

    @pytest.fixture
    def sector(self):
        ngates = 34
        azimuths = 40.0 + 0.5 * np.arange(160, dtype="float64")
        radar = make_radar(azimuths, ngates=ngates)
        geometry = census_sweep(radar, 0)
        assert geometry.sweep_class is SweepClass.UNIFORM_PARTIAL_SECTOR
        smooth = (
            np.cos(np.deg2rad(azimuths))[:, None]
            * np.cos(np.linspace(0.0, np.pi, ngates))[None, :]
        )
        return geometry, azimuths, smooth, ngates

    def test_reports_the_sector_path(self, sector):
        geometry, azimuths, values, _ = sector
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.az_path.startswith("sector_")

    def test_recovers_in_sector_samples_reasonably(self, sector):
        geometry, azimuths, values, ngates = sector
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        interior = azimuths[10:-10]
        azimuth_mesh, range_mesh = np.meshgrid(interior, gate_ranges, indexing="ij")
        recovered = evaluator.evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(recovered - values[10:-10])) < 0.05 * np.ptp(values)

    def test_gap_cell_count_is_reported(self, sector):
        geometry, azimuths, values, _ = sector
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.extras["n_gap_cells"] > 0

    def test_mirror_mode_is_available(self, sector):
        geometry, azimuths, values, _ = sector
        evaluator = SweepSpectralEvaluator(
            values, geometry, azimuths, sector_mode="mirror"
        )
        assert evaluator.report.az_path == "sector_mirror"

    def test_both_sector_modes_agree_inside_the_sector(self, sector):
        """The gap fill differs; the measured region should not."""
        geometry, azimuths, values, ngates = sector
        interior = azimuths[20:-20]
        ranges = np.full_like(interior, RANGE_FIRST_M + RANGE_SPACING_M * 8.0)
        taper = SweepSpectralEvaluator(
            values, geometry, azimuths, sector_mode="taper"
        ).evaluate(ranges, interior)
        mirror = SweepSpectralEvaluator(
            values, geometry, azimuths, sector_mode="mirror"
        ).evaluate(ranges, interior)
        assert np.max(np.abs(taper - mirror)) < 0.15 * np.ptp(values)

    def test_rejects_an_unknown_sector_mode(self, sector):
        geometry, azimuths, values, _ = sector
        with pytest.raises(ValueError, match="sector_mode"):
            SweepSpectralEvaluator(values, geometry, azimuths, sector_mode="wrap")


class TestNonUniformPath:
    """Convolution gridding; the adjoint identity is the load-bearing test."""

    @pytest.fixture
    def jittered(self):
        ngates = 34
        nrays = 120
        spacing = 360.0 / nrays
        rng = np.random.default_rng(2)
        azimuths = (
            uniform_azimuths(nrays) + rng.uniform(-0.3, 0.3, nrays) * spacing
        ) % 360.0
        azimuths = np.sort(azimuths)
        radar = make_radar(azimuths, ngates=ngates)
        geometry = census_sweep(radar, 0)
        assert geometry.sweep_class is SweepClass.NON_UNIFORM
        return (
            geometry,
            azimuths,
            band_limited_field(
                azimuths, ngates, azimuth_modes=(0, 1, 3), range_modes=(0, 1, 2)
            ),
            ngates,
        )

    def test_reports_the_nufft_path(self, jittered):
        geometry, azimuths, values, _ = jittered
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.az_path == "nufft_kb"

    def test_adjoint_identity_holds_to_machine_precision(self, jittered):
        """``<A u, f>_W == <u, A^H f>``: what makes CG on the normal equations valid.

        An earlier implementation of this operator returned ~1 here instead of
        ~1e-15, from a stray factor of M or N0 in the transpose. That is exactly
        the kind of error a smoke test would miss and this one cannot.
        """
        geometry, azimuths, values, _ = jittered
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.extras["adjoint_test_rel"] < 1e-10

    def test_recovers_a_smooth_field_closely_by_default(self, jittered):
        """With the CG default on, recovery is sub-1% rather than ~30%."""
        geometry, azimuths, values, ngates = jittered
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        recovered = evaluator.evaluate(range_mesh, azimuth_mesh)
        assert np.max(np.abs(recovered - values)) < 0.01 * np.ptp(values)

    def test_unrefined_gridding_is_much_worse(self, jittered):
        """Quantifies what the default buys: ~30% error becomes <1%.

        This is why ``DEFAULT_CG_ITERATIONS`` is not 0 on the iterative path.
        Gridding computes the adjoint, and the adjoint is not the inverse. The direct
        solver reaches the same place by factorisation instead, which is why it needs
        no iteration count --- so ``az_solver='cg'`` is pinned here to keep this test
        about the iteration.
        """
        geometry, azimuths, values, ngates = jittered
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        unrefined = SweepSpectralEvaluator(
            values, geometry, azimuths, n_cg=0, az_solver="cg"
        )
        error = np.max(np.abs(unrefined.evaluate(range_mesh, azimuth_mesh) - values))
        assert error > 0.1 * np.ptp(values)

    def test_conjugate_gradient_refinement_reduces_the_residual(self, jittered):
        """Gridding is the adjoint, not the inverse; CG closes that gap."""
        geometry, azimuths, values, _ = jittered
        refined = SweepSpectralEvaluator(
            values, geometry, azimuths, n_cg=12, az_solver="cg"
        )
        assert (
            refined.report.extras["cg_resid_end"]
            < refined.report.extras["cg_resid_start"]
        )

    def test_more_refinement_improves_sample_fidelity(self, jittered):
        geometry, azimuths, values, ngates = jittered
        gate_ranges = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(ngates)
        azimuth_mesh, range_mesh = np.meshgrid(azimuths, gate_ranges, indexing="ij")
        # az_solver='cg' is pinned because this test is *about* the iteration: the
        # default solver ignores n_cg, so without this both arms would be the
        # direct solve and the comparison would be vacuous.
        plain = SweepSpectralEvaluator(
            values, geometry, azimuths, n_cg=0, az_solver="cg"
        )
        refined = SweepSpectralEvaluator(
            values, geometry, azimuths, n_cg=25, az_solver="cg"
        )
        plain_error = np.max(np.abs(plain.evaluate(range_mesh, azimuth_mesh) - values))
        refined_error = np.max(
            np.abs(refined.evaluate(range_mesh, azimuth_mesh) - values)
        )
        assert refined_error < plain_error

    def test_the_default_solve_is_direct_not_iterative(self, jittered):
        """The default changed, and this pins what it changed to.

        Refinement-by-iteration is no longer the default: ``n_cg=12`` produced
        unphysical reflectivity on real volumes, because the iteration count was
        acting as regularisation rather than as a convergence budget. The default is
        now a factorisation with an explicit ridge, so there is no iteration count to
        report and no ``cg_resid_*`` pair.
        """
        geometry, azimuths, values, _ = jittered
        extras = SweepSpectralEvaluator(values, geometry, azimuths).report.extras
        assert extras["solver"] == "direct"
        assert extras["n_cg"] == 0
        assert "cg_resid_end" not in extras
        assert extras["ridge_mode"] == "auto"
        assert extras["normal_cond"] > 1.0

    def test_the_iterative_solve_is_still_reachable_and_unchanged(self, jittered):
        """Opting back in must reproduce the old default exactly, diagnostics included.

        The point of changing a default rather than removing a code path: anyone
        reproducing earlier numbers needs the old behaviour available and identical.
        """
        geometry, azimuths, values, _ = jittered
        extras = SweepSpectralEvaluator(
            values, geometry, azimuths, az_solver="cg", az_engine="scipy"
        ).report.extras
        assert extras["n_cg"] == DEFAULT_CG_ITERATIONS
        assert extras["cg_resid_end"] < 1e-3

    def test_refinement_can_be_disabled(self, jittered):
        geometry, azimuths, values, _ = jittered
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths, n_cg=0)
        assert evaluator.report.extras["n_cg"] == 0
        assert "cg_resid_end" not in evaluator.report.extras

    def test_density_weights_are_positive(self, jittered):
        """Voronoi arc-length weights; a non-positive weight would be a bug."""
        geometry, azimuths, values, _ = jittered
        evaluator = SweepSpectralEvaluator(values, geometry, azimuths)
        assert evaluator.report.extras["density_w_min"] > 0.0

    def test_kernel_parameters_are_reported(self, jittered):
        geometry, azimuths, values, _ = jittered
        # az_engine='scipy' is pinned deliberately: these keys describe a
        # Kaiser-Bessel kernel and an oversampled work grid, and the default engine
        # has neither -- it forms the exact DFT matrix. Reporting a kb_beta for an
        # engine with no KB kernel would be a fiction, so the key set is
        # engine-dependent by design.
        evaluator = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine="scipy"
        )
        for key in ("kb_width", "oversamp", "kb_beta", "n_nominal", "M_oversampled"):
            assert key in evaluator.report.extras

    def test_oversampled_lattice_is_larger_than_nominal(self, jittered):
        geometry, azimuths, values, _ = jittered
        extras = SweepSpectralEvaluator(
            values, geometry, azimuths, az_engine="scipy"
        ).report.extras
        assert extras["M_oversampled"] >= 2 * extras["n_nominal"]

    def test_constant_field_survives_non_uniform_sampling(self, jittered):
        """The sharpest test of the NUFFT path: a constant must stay constant.

        Measured relative deviation is 0.07% with the CG default and 60% without,
        which is what motivated changing the default.
        """
        geometry, azimuths, _, ngates = jittered
        constant = np.full((azimuths.size, ngates), 12.5)
        evaluator = SweepSpectralEvaluator(constant, geometry, azimuths)
        probes = np.linspace(0.0, 359.0, 41)
        result = evaluator.evaluate(
            np.full_like(probes, RANGE_FIRST_M + RANGE_SPACING_M * 5.0), probes
        )
        np.testing.assert_allclose(result, 12.5, rtol=2e-3)

    def test_unrefined_gridding_distorts_a_constant_field(self, jittered):
        """Documents the failure mode the default exists to avoid."""
        geometry, azimuths, _, ngates = jittered
        constant = np.full((azimuths.size, ngates), 12.5)
        unrefined = SweepSpectralEvaluator(
            constant, geometry, azimuths, n_cg=0, az_solver="cg"
        )
        probes = np.linspace(0.0, 359.0, 41)
        result = unrefined.evaluate(
            np.full_like(probes, RANGE_FIRST_M + RANGE_SPACING_M * 5.0), probes
        )
        assert np.max(np.abs(result / 12.5 - 1.0)) > 0.1


class TestLinearReflectivityGuard:
    """Interpolation must run in dBZ. This is measured, not stylistic.

    On a real C-SAPR tilt, band-limited interpolation in linear Z drove 23.7-24.2%
    of gates to a negative reflectivity factor --- physically impossible --- with the
    worst excursion at -8171 times the field maximum, and fabricated a 45.3 dBZ
    artefact. The evaluator therefore refuses input it can tell is linear Z.
    """

    @pytest.fixture
    def linear_z_values(self, exact_sweep):
        """A realistic storm-core dBZ field, linearised.

        The peak must exceed ~23 dBZ for linear Z to clear the guard's ceiling; 45
        dBZ is an ordinary convective core, so this is a representative case rather
        than a contrived one.
        """
        geometry, azimuths, values, ngates = exact_sweep
        dbz = 25.0 + 20.0 * values / max(np.ptp(values), 1e-9)
        assert dbz.max() > 23.0
        return geometry, azimuths, to_linear(dbz), ngates

    def test_refuses_linear_reflectivity(self, linear_z_values):
        geometry, azimuths, linear_values, _ = linear_z_values
        with pytest.raises(LinearReflectivityError, match="dBZ"):
            SweepSpectralEvaluator(linear_values, geometry, azimuths)

    def test_error_message_names_the_remedy(self, linear_z_values):
        geometry, azimuths, linear_values, _ = linear_z_values
        with pytest.raises(LinearReflectivityError, match="to_dbz"):
            SweepSpectralEvaluator(linear_values, geometry, azimuths)

    def test_explicit_units_override_allows_it(self, linear_z_values):
        """An escape hatch, because the guard is a heuristic and may be wrong."""
        geometry, azimuths, linear_values, _ = linear_z_values
        evaluator = SweepSpectralEvaluator(
            linear_values, geometry, azimuths, field_units="linear_z"
        )
        assert evaluator.report.extras["field_units"] == "linear_z"

    def test_dbz_input_passes_the_guard(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        evaluator = SweepSpectralEvaluator(20.0 + values, geometry, azimuths)
        assert evaluator.report.extras["field_units"] == "dbz"

    def test_a_constant_positive_field_is_not_flagged(self, exact_sweep):
        """The guard must key on dynamic range, not merely on positivity."""
        geometry, azimuths, _, ngates = exact_sweep
        SweepSpectralEvaluator(
            np.full((azimuths.size, ngates), 35.0), geometry, azimuths
        )

    def test_a_field_with_negatives_is_never_flagged(self, exact_sweep):
        """Linear Z cannot be negative, so negatives prove the field is not linear."""
        geometry, azimuths, values, _ = exact_sweep
        SweepSpectralEvaluator(values - values.mean(), geometry, azimuths)

    def test_rejects_an_unknown_field_units_value(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        with pytest.raises(ValueError, match="field_units"):
            SweepSpectralEvaluator(values, geometry, azimuths, field_units="mm6/m3")


class TestGapFreeRequirement:
    def test_rejects_a_masked_array(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        masked = np.ma.masked_array(values, mask=np.zeros_like(values, dtype=bool))
        masked[0, 0] = np.ma.masked
        with pytest.raises(ValueError, match="fill_sweep"):
            SweepSpectralEvaluator(masked, geometry, azimuths)

    def test_rejects_non_finite_values(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        holed = values.copy()
        holed[2, 3] = np.nan
        with pytest.raises(ValueError, match="fill_sweep"):
            SweepSpectralEvaluator(holed, geometry, azimuths)

    def test_accepts_a_fully_unmasked_masked_array(self, exact_sweep):
        """A masked array with nothing actually masked is gap-free."""
        geometry, azimuths, values, _ = exact_sweep
        SweepSpectralEvaluator(np.ma.masked_array(values), geometry, azimuths)


class TestFillSweep:
    @pytest.fixture
    def gappy(self):
        rng = np.random.default_rng(4)
        values = np.ma.masked_array(rng.normal(size=(24, 40)) + 20.0)
        values[3, 10:20] = np.ma.masked
        values[7, :] = np.ma.masked
        return values

    def test_floor_fill_removes_the_mask(self, gappy):
        filled = fill_sweep(gappy, method="floor")
        assert not np.ma.isMaskedArray(filled) or not np.ma.getmaskarray(filled).any()
        assert np.all(np.isfinite(filled))

    def test_floor_fill_uses_the_supplied_floor(self, gappy):
        filled = fill_sweep(gappy, method="floor", floor=-32.0)
        assert filled[7].min() == pytest.approx(-32.0)

    def test_floor_defaults_to_the_data_minimum(self, gappy):
        filled = fill_sweep(gappy, method="floor")
        assert filled.min() == pytest.approx(gappy.min())

    def test_floor_fill_preserves_valid_data(self, gappy):
        filled = fill_sweep(gappy, method="floor")
        valid = ~np.ma.getmaskarray(gappy)
        np.testing.assert_allclose(filled[valid], np.ma.getdata(gappy)[valid])

    def test_edge_fill_interpolates_within_a_ray(self, gappy):
        """An interior gap should be bridged, not stamped with the floor."""
        filled = fill_sweep(gappy, method="edge")
        assert filled[3, 15] > gappy.min()

    def test_edge_fill_preserves_valid_data(self, gappy):
        filled = fill_sweep(gappy, method="edge")
        valid = ~np.ma.getmaskarray(gappy)
        np.testing.assert_allclose(filled[valid], np.ma.getdata(gappy)[valid])

    def test_edge_fill_falls_back_to_the_floor_for_an_empty_ray(self, gappy):
        filled = fill_sweep(gappy, method="edge", floor=-99.0)
        np.testing.assert_allclose(filled[7], -99.0)

    def test_gerchberg_fill_preserves_measured_samples(self, gappy):
        """The band-limited iteration must re-impose the data every pass."""
        filled = fill_sweep(gappy, method="gerchberg", n_iter=10)
        valid = ~np.ma.getmaskarray(gappy)
        np.testing.assert_allclose(
            filled[valid], np.ma.getdata(gappy)[valid], atol=1e-9
        )

    def test_gerchberg_fill_is_finite(self, gappy):
        assert np.all(np.isfinite(fill_sweep(gappy, method="gerchberg", n_iter=10)))

    def test_can_return_the_mask(self, gappy):
        filled, mask = fill_sweep(gappy, method="floor", return_mask=True)
        assert mask.shape == filled.shape
        assert mask[7].all()
        assert mask[3, 15]

    def test_output_feeds_the_evaluator(self, exact_sweep):
        """The whole point of fill_sweep: its output must satisfy the guard."""
        geometry, azimuths, values, _ = exact_sweep
        masked = np.ma.masked_array(values + 20.0)
        masked[1, 5:9] = np.ma.masked
        SweepSpectralEvaluator(fill_sweep(masked, method="edge"), geometry, azimuths)

    def test_rejects_an_unknown_method(self, gappy):
        with pytest.raises(ValueError, match="unknown fill method"):
            fill_sweep(gappy, method="kriging")

    def test_handles_an_array_with_no_gaps(self):
        clean = np.arange(20.0).reshape(4, 5)
        np.testing.assert_allclose(fill_sweep(clean, method="edge"), clean)


class TestReflectivityConversions:
    def test_round_trip_is_exact(self):
        dbz = np.array([-30.0, -5.0, 0.0, 12.5, 45.0, 70.0])
        recovered, n_clipped = to_dbz(to_linear(dbz))
        np.testing.assert_allclose(recovered, dbz, atol=1e-9)
        assert n_clipped == 0

    def test_zero_dbz_is_unit_linear(self):
        assert float(to_linear(0.0)) == pytest.approx(1.0)

    def test_ten_db_is_a_factor_of_ten(self):
        assert float(to_linear(10.0)) == pytest.approx(10.0)

    def test_negative_linear_values_are_clipped_and_counted(self):
        """Count clipped samples: how the linear-Z failure was quantified."""
        dbz, n_clipped = to_dbz(np.array([1.0, -0.5, 2.0, -3.0]))
        assert n_clipped == 2
        assert np.all(np.isfinite(dbz))

    def test_clip_floor_is_configurable(self):
        dbz, _ = to_dbz(np.array([-1.0]), floor_linear=1e-3)
        assert float(dbz[0]) == pytest.approx(-30.0)

    def test_scalar_input_gives_a_zero_dimensional_result(self):
        """Pinned because the two input shapes need different unwrapping.

        ``np.where`` on a 0-d array returns a NumPy scalar, so ``float(result)``
        works for scalar input but ``result[0]`` raises; for length-1 array input it
        is the other way round. Worth knowing before writing either.
        """
        dbz_scalar, _ = to_dbz(1.0)
        assert np.ndim(dbz_scalar) == 0
        assert float(dbz_scalar) == pytest.approx(0.0)
        with pytest.raises((IndexError, TypeError)):
            _ = dbz_scalar[0]

        dbz_array, _ = to_dbz(np.array([1.0]))
        assert dbz_array.shape == (1,)
        assert float(dbz_array[0]) == pytest.approx(0.0)

    def test_conversions_preserve_shape(self):
        values = np.zeros((3, 4))
        assert to_linear(values).shape == (3, 4)
        assert to_dbz(np.ones((3, 4)))[0].shape == (3, 4)


class TestEvalReport:
    def test_carries_the_range_axis_from_the_census(self, exact_sweep):
        geometry, azimuths, values, ngates = exact_sweep
        report = SweepSpectralEvaluator(values, geometry, azimuths).report
        assert report.r0_m == pytest.approx(RANGE_FIRST_M)
        assert report.dr_m == pytest.approx(RANGE_SPACING_M)
        assert report.ngates == ngates

    def test_range_axis_can_be_overridden(self, exact_sweep):
        """Sweeps with differing range axes need no resampling."""
        geometry, azimuths, values, _ = exact_sweep
        report = SweepSpectralEvaluator(
            values, geometry, azimuths, r0_m=250.0, dr_m=500.0
        ).report
        assert report.r0_m == pytest.approx(250.0)
        assert report.dr_m == pytest.approx(500.0)

    def test_records_the_build_time(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        assert SweepSpectralEvaluator(values, geometry, azimuths).report.build_s >= 0.0

    def test_is_a_dataclass_instance(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        assert isinstance(
            SweepSpectralEvaluator(values, geometry, azimuths).report, EvalReport
        )

    def test_reports_the_sweep_index(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        assert SweepSpectralEvaluator(values, geometry, azimuths).report.sweep == 0


class TestBandLimiting:
    def test_band_fraction_drops_high_azimuth_modes(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        full = SweepSpectralEvaluator(values, geometry, azimuths)
        limited = SweepSpectralEvaluator(values, geometry, azimuths, band_frac=0.25)
        assert limited.C.shape[0] < full.C.shape[0]

    def test_band_limited_evaluation_stays_finite(self, exact_sweep):
        geometry, azimuths, values, ngates = exact_sweep
        limited = SweepSpectralEvaluator(values, geometry, azimuths, band_frac=0.3)
        probes = np.linspace(0.0, 359.0, 50)
        result = limited.evaluate(
            np.full_like(probes, RANGE_FIRST_M + RANGE_SPACING_M * 4.0), probes
        )
        assert np.all(np.isfinite(result))

    def test_band_limited_lattice_round_trips(self, exact_sweep):
        """Reconstruct the lattice from a band-limited coefficient set.

        Regression: the retained set is shorter than the transform length, so the
        modes must be scattered to their own bins rather than reshaped.
        """
        geometry, azimuths, values, _ = exact_sweep
        limited = SweepSpectralEvaluator(values, geometry, azimuths, band_frac=0.3)
        dense = limited.dense(2, 2)
        assert np.all(np.isfinite(dense))
        assert dense.shape == (limited.report.M_az * 2, limited.report.Nr_ext * 2)

    def test_retaining_the_full_band_changes_nothing(self, exact_sweep):
        geometry, azimuths, values, _ = exact_sweep
        probes = np.linspace(0.0, 359.0, 33)
        ranges = np.full_like(probes, RANGE_FIRST_M + RANGE_SPACING_M * 7.0)
        default = SweepSpectralEvaluator(values, geometry, azimuths)
        explicit = SweepSpectralEvaluator(values, geometry, azimuths, band_frac=1.0)
        np.testing.assert_allclose(
            default.evaluate(ranges, probes),
            explicit.evaluate(ranges, probes),
            atol=1e-12,
        )


class TestSharpEdgeBehaviour:
    """The measured evidence behind the dBZ requirement, and its limits.

    These use a hard echo edge --- a 55 dBZ core against a -10 dBZ background ---
    because a smooth field does not provoke the failure at all. That distinction
    matters: an earlier version of this check used a smooth field and showed zero
    negative samples, which would have made the dBZ requirement look unnecessary.
    """

    CORE_DBZ = 55.0
    BACKGROUND_DBZ = -10.0

    @pytest.fixture
    def hard_edge(self):
        nrays, ngates = 180, 60
        azimuths = uniform_azimuths(nrays)
        radar = make_radar(azimuths, ngates=ngates)
        values = np.full((nrays, ngates), self.BACKGROUND_DBZ)
        in_core = (azimuths > 100.0) & (azimuths < 160.0)
        values[np.ix_(in_core, np.arange(20, 45))] = self.CORE_DBZ
        geometry = census_sweep(radar, 0)
        return geometry, azimuths, values, ngates

    def probe(self, evaluator, ngates, n=120):
        azimuth_probes = np.linspace(0.0, 359.5, n)
        gate_probes = np.linspace(0.0, ngates - 1, n)
        azimuth_mesh, range_mesh = np.meshgrid(
            azimuth_probes, RANGE_FIRST_M + RANGE_SPACING_M * gate_probes, indexing="ij"
        )
        return evaluator.evaluate(range_mesh, azimuth_mesh)

    def test_linear_interpolation_produces_impossible_values(self, hard_edge):
        """The core finding: linear Z goes negative, which is unphysical."""
        geometry, azimuths, values, ngates = hard_edge
        linear = SweepSpectralEvaluator(
            to_linear(values), geometry, azimuths, field_units="linear_z"
        )
        result = self.probe(linear, ngates)
        assert (result < 0.0).mean() > 0.05

    def test_linear_interpolation_forces_clipping_on_conversion(self, hard_edge):
        """Those negatives become indistinguishable from real weak echo."""
        geometry, azimuths, values, ngates = hard_edge
        linear = SweepSpectralEvaluator(
            to_linear(values), geometry, azimuths, field_units="linear_z"
        )
        _, n_clipped = to_dbz(self.probe(linear, ngates))
        assert n_clipped > 0

    def test_dbz_interpolation_stays_physically_representable(self, hard_edge):
        """Not free of ringing --- but every value is still a dBZ value."""
        geometry, azimuths, values, ngates = hard_edge
        result = self.probe(SweepSpectralEvaluator(values, geometry, azimuths), ngates)
        assert np.all(np.isfinite(result))
        _, n_clipped = to_dbz(to_linear(result))
        assert n_clipped == 0

    def test_dbz_interpolation_does_overshoot(self, hard_edge):
        """Document the limit of the guarantee, so it is not oversold.

        Gibbs ringing is a property of band-limited interpolation at a
        discontinuity, not a defect here. Working in dBZ bounds the *consequence*,
        not the ringing.
        """
        geometry, azimuths, values, ngates = hard_edge
        result = self.probe(SweepSpectralEvaluator(values, geometry, azimuths), ngates)
        assert result.max() > self.CORE_DBZ + 1.0
        assert result.min() < self.BACKGROUND_DBZ - 1.0

    def test_overshoot_is_asymmetric_high_side_is_worse(self, hard_edge):
        """The two directions are NOT equal, and the docs must not imply they are.

        Measured on this fixture (180 rays, 60 gates, 120x120 probe): +17.5 dB above
        the data maximum against -8.7 dB below its minimum. On a larger 360x100
        sweep the figures are +19.8 and -10.6. Either way the high-side excursion is
        close to twice the low-side one, so summarising both as "about 20 dB in each
        direction" overstates the low side by roughly 2x --- which is the documentation
        error this test exists to stop recurring.
        """
        geometry, azimuths, values, ngates = hard_edge
        result = self.probe(SweepSpectralEvaluator(values, geometry, azimuths), ngates)
        overshoot_high_db = float(result.max() - self.CORE_DBZ)
        overshoot_low_db = float(self.BACKGROUND_DBZ - result.min())
        assert overshoot_high_db > 1.5 * overshoot_low_db

    def test_band_limiting_reduces_the_overshoot(self, hard_edge):
        """``band_frac`` is the knob for trading resolution against ringing."""
        geometry, azimuths, values, ngates = hard_edge
        full = self.probe(SweepSpectralEvaluator(values, geometry, azimuths), ngates)
        limited = self.probe(
            SweepSpectralEvaluator(values, geometry, azimuths, band_frac=0.4), ngates
        )
        assert limited.max() < full.max()

    def test_a_smooth_field_does_not_provoke_the_failure(self, exact_sweep):
        """Why the fixture above is hard-edged: smoothness hides the problem.

        On a smooth field, linear-Z interpolation produces no negatives at all, so a
        test built on one would wrongly suggest the dBZ requirement is unnecessary.
        """
        geometry, azimuths, values, ngates = exact_sweep
        smooth_dbz = 20.0 + 5.0 * values / max(np.ptp(values), 1e-9)
        linear = SweepSpectralEvaluator(
            to_linear(smooth_dbz), geometry, azimuths, field_units="linear_z"
        )
        gate_probes = np.linspace(0.0, ngates - 1, 60)
        azimuth_probes = np.linspace(0.0, 359.0, 60)
        result = linear.evaluate(
            RANGE_FIRST_M + RANGE_SPACING_M * gate_probes, azimuth_probes
        )
        assert np.all(result > 0.0)


class TestGuardMessage:
    def test_names_the_measured_peak_and_the_ceiling(self, exact_sweep):
        """The message must be actionable: what tripped it, and what to do."""
        geometry, azimuths, values, _ = exact_sweep
        linear = to_linear(30.0 + 20.0 * values / max(np.ptp(values), 1e-9))
        with pytest.raises(LinearReflectivityError) as raised:
            SweepSpectralEvaluator(linear, geometry, azimuths)
        message = str(raised.value)
        assert "to_dbz" in message
        assert "field_units" in message
        assert str(int(DBZ_PHYSICAL_CEILING)) in message
