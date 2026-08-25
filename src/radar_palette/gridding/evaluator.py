"""Band-limited evaluation of one sweep at arbitrary (range, azimuth) targets.

:class:`SweepSpectralEvaluator` turns a sampled sweep into a continuous function.
Where the sampling supports it, that function is an *exact* band-limited interpolant
--- no window, no taper --- and where it does not, the evaluator says so in its report
rather than degrading silently.

Azimuth: three paths, chosen by sweep class
-------------------------------------------
``EXACT_UNIFORM_PERIODIC``
    Plain FFT. Azimuth genuinely is periodic and the samples genuinely are on the
    lattice, so the Fourier series is exact.
``UNIFORM_PARTIAL_SECTOR``
    The sector is placed on a full-circle lattice at its own spacing and the unmeasured
    gap is closed --- by a raised-cosine blend between the sector's two edge values
    (``sector_mode='taper'``) or by mirror reflection (``sector_mode='mirror'``). Both
    keep the extension continuous so the Fourier series does not ring. **Values inside
    the synthesised gap are not meaningful**; only the measured sector is.
``NON_UNIFORM``
    Kaiser-Bessel NUFFT gridding, refined by conjugate gradients. See
    :mod:`radar_palette.gridding.nufft`.

    The transforms themselves are supplied by an interchangeable *engine* and the
    solve on top of them by an interchangeable *solver*
    (:mod:`radar_palette.gridding.nufft_engines`), selected with ``az_engine`` and
    ``az_solver``. The defaults reproduce the reference implementation to
    round-off while running about twice as fast; ``az_solver='direct'`` replaces
    the iteration with a factorisation of the same system, which is both more
    accurate and faster still.

    **The refinement is on by default**, which departs from the research code this
    was ported from. Convolution gridding computes the *adjoint* of the sampling
    operator, not its inverse, and the difference is not academic: on a constant
    field sampled with +/-0.3-spacing jitter, plain gridding reproduces the constant
    to only 60% relative deviation, and the error scales with jitter (3.6% at
    +/-0.02, 19% at +/-0.1). Twelve CG iterations bring it to 0.07% for about 50%
    more build time. A default that returns 60% error on a constant field is not
    defensible, so the default is ``DEFAULT_CG_ITERATIONS``; pass ``n_cg=0`` for the
    unrefined operator.

Range: never periodic
---------------------
Range is even/mirror extended to length ``2 * ngates - 2`` before the transform. A
plain FFT would treat the first and last gate as adjacent, and the resulting
end-to-end jump dominates the Gibbs error for any field with a range trend. The
mirror extension removes that jump while leaving the interpolant passing exactly
through every original gate.

Units
-----
Reflectivity must be supplied in **dBZ**, and the constructor refuses input it can
tell is linear Z: interpolating linear reflectivity across a hard echo edge drove
42.6% of evaluated samples negative in a measured reproduction. See
:mod:`radar_palette.gridding.reflectivity`, which also states the limit of that
guarantee --- dBZ keeps ringing representable, it does not remove it. Pass
``field_units`` explicitly to override the heuristic, which is what a
non-reflectivity field will normally want.

Ringing at sharp edges
----------------------
A band-limited interpolant overshoots at a discontinuity; that is Gibbs behaviour,
not a defect in this implementation. On a synthetic 55 dBZ core against a -10 dBZ
background (360 rays, 100 gates, probed on a 600x600 grid) the interpolant reaches
74.8 dBZ and -20.6 dBZ --- **+19.8 dB above the data maximum, -10.6 dB below its
minimum**. The two directions are not symmetric, and the magnitudes depend on the
sampling: the same edge at 180 rays by 60 gates gives +17.5 and -8.7 dB.

Callers gridding fields with sharp edges should expect this and clip or flag
accordingly; ``band_frac`` trades resolution against overshoot if that is
preferable.

Input must be gap-free; run :func:`~radar_palette.gridding.fill.fill_sweep` first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft2, ifft2

from radar_palette.gridding.census import SweepClass
from radar_palette.gridding.nufft_engines import (
    DEFAULT_ENGINE,
    DEFAULT_SOLVER,
    SOLVERS,
    make_operator,
)
from radar_palette.gridding.reflectivity import (
    DBZ_PHYSICAL_CEILING,
    LinearReflectivityError,
    looks_like_linear_reflectivity,
)

__all__ = ["DEFAULT_CG_ITERATIONS", "EvalReport", "SweepSpectralEvaluator"]

# Conjugate-gradient iterations used by the NUFFT path by default. Measured, not
# guessed: on a constant field sampled with +/-0.3-spacing azimuth jitter, plain
# gridding (0 iterations) reproduces the constant to only 60% relative deviation,
# because the adjoint is not the inverse. The error falls to 6% by 4 iterations,
# 0.8% by 8 and 0.07% by 12, then plateaus -- so 12 is where the curve flattens.
# The cost is a ~50% longer build (11 ms -> 17 ms on a 120x34 sweep), which is a
# small price for two orders of magnitude of accuracy.
DEFAULT_CG_ITERATIONS = 12

SECTOR_MODES = ("taper", "mirror")
FIELD_UNITS = ("dbz", "linear_z", "other")


@dataclass
class EvalReport:
    """Bookkeeping from building an evaluator.

    ``az_path`` names the azimuth path actually taken (``fft_exact``,
    ``sector_taper``, ``sector_mirror`` or ``nufft_kb``), so a caller can tell which
    accuracy guarantee applies. ``extras`` carries path-specific diagnostics --- the
    lattice residual for the exact path, the gap-cell count for a sector, and the
    kernel parameters, density-weight range, adjoint-test result and CG residuals for
    the NUFFT path.
    """

    sweep: int
    sweep_class: str
    az_path: str
    M_az: int
    Nr_ext: int
    r0_m: float
    dr_m: float
    ngates: int
    build_s: float
    extras: dict = field(default_factory=dict)


def _split_nyquist(coefficients, frequencies, axis):
    """Halve the Nyquist bin and mirror it, so the Fourier sum is real-valued.

    An even-length transform has a single unpaired Nyquist bin. Evaluating the series
    off-lattice with that bin assigned wholly to ``+N/2`` gives a complex result;
    splitting it symmetrically keeps the interpolant real.
    """
    length = coefficients.shape[axis]
    if length % 2:
        return coefficients, frequencies

    nyquist_index = int(np.argmin(frequencies))
    coefficients = coefficients.copy()
    selector = [slice(None)] * coefficients.ndim
    selector[axis] = slice(nyquist_index, nyquist_index + 1)
    half = coefficients[tuple(selector)] * 0.5
    coefficients[tuple(selector)] = half
    return (
        np.concatenate([coefficients, half], axis=axis),
        np.concatenate([frequencies, [-frequencies[nyquist_index]]]),
    )


def _merge_nyquist(coefficients, frequencies, axis, length):
    """Inverse of :func:`_split_nyquist`.

    ``_split_nyquist`` appends exactly one mirrored row, and only when the axis length
    is even, so a merge is well-defined only for an array of length ``length + 1``.
    Any other length means there is nothing to merge --- in particular a band-limited
    coefficient set, where the Nyquist mode was dropped rather than split, is
    *shorter* than ``length`` and must be returned untouched.
    """
    if coefficients.shape[axis] != length + 1:
        return coefficients, frequencies

    keep = [slice(None)] * coefficients.ndim
    keep[axis] = slice(0, length)
    merged = coefficients[tuple(keep)].copy()

    nyquist_index = int(np.argmin(frequencies[:length]))
    target = [slice(None)] * coefficients.ndim
    target[axis] = slice(nyquist_index, nyquist_index + 1)
    mirrored = [slice(None)] * coefficients.ndim
    mirrored[axis] = slice(length, length + 1)
    merged[tuple(target)] = merged[tuple(target)] + coefficients[tuple(mirrored)]
    return merged, frequencies[:length]


class SweepSpectralEvaluator:
    """A sampled sweep, re-expressed as a continuous band-limited function.

    Parameters
    ----------
    values : array_like
        Sweep of shape ``(nrays, ngates)``, gap-free and finite. Run
        :func:`~radar_palette.gridding.fill.fill_sweep` first if it is not.
    geometry : ~radar_palette.gridding.census.SweepGeometry
        Census record for this sweep, from
        :func:`~radar_palette.gridding.census.census_sweep`. Supplies the sweep class
        (which selects the azimuth path) and the range axis.
    azimuths : array_like
        Measured azimuths in degrees, one per ray, in the same order as ``values``.
        Order need not be sorted.
    kb_width, oversamp : float, optional
        Kaiser-Bessel kernel width in cells and azimuth lattice oversampling factor.
        NUFFT path only.
    sector_mode : {'taper', 'mirror'}, optional
        How to close the unmeasured gap. Sector path only.
    band_frac : float, optional
        Retain only azimuth modes with ``|m| <= band_frac * M / 2``. The default, 1.0,
        retains everything.
    n_cg : int, optional
        Conjugate-gradient refinement iterations for the NUFFT path. Defaults to
        ``DEFAULT_CG_ITERATIONS``; pass 0 for classical gridding only. See the module
        docstring for why the default is not 0. Ignored when ``az_solver='direct'``,
        which has no iteration count.

        **Do not raise this to get a better answer.** The iteration count is acting
        as regularisation, not as a convergence budget: on a real convective volume
        the gridded field spans -61 to 66 dBZ at the default 12 iterations and
        **-1907 to 2471 dBZ at 25**, because past roughly a dozen iterations the
        solve amplifies out-of-band content instead of converging. If the answer is
        not good enough, use ``az_solver='direct'``, which reaches the least-squares
        limit in one factorisation with an explicit regularisation term. Unphysical
        output is flagged with
        :class:`~radar_palette.gridding.cones.UnphysicalGridWarning`.
    cg_tol : float, optional
        Relative residual at which the refinement stops early.
    az_engine : str, optional
        Which NUFFT backend computes the transforms on the ``NON_UNIFORM`` path:
        ``'dense'`` (default: the exact DFT operator, no dependency beyond
        numpy/scipy), ``'scipy'``, ``'reference'``, ``'finufft'``, ``'ducc0'`` or
        ``'torch'``. Note that ``'finufft'`` should not be combined with
        ``n_jobs>1``; see :mod:`radar_palette.gridding.nufft_engines`.

        Only ``'reference'`` and ``'scipy'`` reproduce the existing implementation
        to round-off; for those two, choosing an engine is purely a performance
        decision. ``'dense'``, ``'finufft'``, ``'ducc0'`` and ``'torch'`` all
        change the answer by ~5e-4, because each uses a different (in the case of
        ``'dense'``, exact) transform in place of the reference's Kaiser-Bessel
        kernel, whose own error at the default ``kb_width``/``oversamp`` is 2.5e-4.
        The shift is towards exact arithmetic rather than away from it, but it is a
        shift, and it is larger than the difference the ``az_solver`` choice makes
        on a Kaiser-Bessel engine.
    az_solver : {'cg', 'direct'}, optional
        How the NUFFT path solves the density-weighted normal equations.
        ``'direct'`` (default) factorises the system once with an explicit
        regularisation term. ``'cg'`` is the older conjugate-gradient refinement and
        reproduces the pre-change behaviour exactly, ``cg_resid_*`` diagnostics
        included.

        The default is ``'direct'`` for safety rather than for speed. On a real
        convective volume the iterative path produced unphysical reflectivity --- one
        cone of fifteen overshooting its input range by 107 dB at the old
        ``n_cg=12``, and -1907 to 2471 dBZ if a caller raises ``n_cg`` to 25 --- because
        the iteration count was acting as regularisation rather than as a convergence
        budget. See :mod:`radar_palette.gridding.nufft_engines` for the measurements.
    az_ridge : float or {'auto'}, optional
        Regularisation for ``az_solver='direct'``, relative to the mean diagonal
        of the normal matrix. Defaults to ``'auto'``, which derives it from the
        eigenvalue spectrum --- the right default because the size needed varies
        by eight orders of magnitude with the sweep geometry, and too small a
        value overfits catastrophically rather than degrading gracefully. Real
        sweeps are commonly rank-deficient: operational azimuth sampling leaves
        gaps of ~2x the nominal spacing, and the report's ``normal_null_dim``
        counts the resulting undetermined modes (measured: 23 of 993 on an X-band
        sweep). Pass a float to fix it.
    r0_m, dr_m : float, optional
        Override the census range axis. Sweeps with differing range axes therefore
        need no resampling.
    field_units : {'dbz', 'linear_z', 'other'}, optional
        Declared units. Omit to let the linear-Z heuristic decide; pass explicitly to
        bypass it.

    Raises
    ------
    ~radar_palette.gridding.reflectivity.LinearReflectivityError
        If ``values`` look like linear reflectivity and ``field_units`` was not given.
    ValueError
        If ``values`` contain gaps or non-finite entries, if there are fewer than
        three gates, or if ``sector_mode`` / ``field_units`` is unrecognised.
    """

    def __init__(
        self,
        values,
        geometry,
        azimuths,
        kb_width=4.0,
        oversamp=2.0,
        sector_mode="taper",
        band_frac=1.0,
        n_cg=DEFAULT_CG_ITERATIONS,
        cg_tol=1e-10,
        az_engine=DEFAULT_ENGINE,
        az_solver=DEFAULT_SOLVER,
        az_ridge="auto",
        r0_m=None,
        dr_m=None,
        field_units=None,
    ):
        started_at = time.perf_counter()

        if sector_mode not in SECTOR_MODES:
            raise ValueError(
                f"unknown sector_mode {sector_mode!r}; expected one of {SECTOR_MODES}"
            )
        if field_units is not None and field_units not in FIELD_UNITS:
            raise ValueError(
                f"unknown field_units {field_units!r}; expected one of {FIELD_UNITS}"
            )
        # Validated here, with the other constructor arguments, rather than where
        # it is used: only one of the three azimuth paths consults it, so a typo
        # would otherwise pass silently on a uniform sweep and fail on the next
        # jittered one.
        if az_solver not in SOLVERS:
            raise ValueError(
                f"unknown az_solver {az_solver!r}; expected one of {SOLVERS}"
            )

        sweep_values = np.asarray(values, dtype=np.float64)
        if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
            raise ValueError("values must be gap-free; run fill_sweep() first")
        if not np.all(np.isfinite(sweep_values)):
            raise ValueError("values must be finite; run fill_sweep() first")
        if sweep_values.shape[1] < 3:
            raise ValueError("need at least 3 gates to mirror-extend the range axis")

        resolved_units = field_units
        if resolved_units is None:
            if looks_like_linear_reflectivity(sweep_values):
                raise LinearReflectivityError(
                    f"values look like linear reflectivity factor Z: the field is "
                    f"non-negative and peaks at {float(np.max(sweep_values)):.3g}, "
                    f"above the {DBZ_PHYSICAL_CEILING:g} dBZ ceiling no real echo "
                    f"reaches. Interpolate in dBZ instead -- convert with to_dbz() "
                    f"-- or pass field_units= explicitly to override this check."
                )
            resolved_units = "dbz"

        self.geometry = geometry
        self.r0 = float(geometry.range_first_m if r0_m is None else r0_m)
        self.dr = float(geometry.range_spacing_m if dr_m is None else dr_m)
        self.ng = sweep_values.shape[1]
        self.kb_width = float(kb_width)
        self.oversamp = float(oversamp)
        self.sector_mode = sector_mode
        self.band_frac = float(band_frac)
        self.n_cg = int(n_cg)
        self.cg_tol = float(cg_tol)
        self.az_engine = az_engine
        self.az_solver = az_solver
        self.az_ridge = az_ridge
        self._dense = None

        extras = {"field_units": resolved_units}
        azimuths = np.asarray(azimuths, dtype=np.float64) % 360.0

        sweep_class = geometry.sweep_class
        if sweep_class is SweepClass.EXACT_UNIFORM_PERIODIC:
            lattice, path, path_extras = self._exact_lattice(sweep_values, azimuths)
        elif sweep_class is SweepClass.UNIFORM_PARTIAL_SECTOR:
            lattice, path, path_extras = self._sector_lattice(sweep_values, azimuths)
        else:
            lattice, path, path_extras = self._nufft_lattice(sweep_values, azimuths)
        extras.update(path_extras)

        self.az_path = path
        self.dphi = 360.0 / self.M

        # Range is mirror-extended: f[0..ng-1] then f[ng-2..1], period 2*ng-2.
        extended = np.concatenate([lattice, lattice[:, -2:0:-1]], axis=1)
        self.Nr = extended.shape[1]

        coefficients = fft2(extended) / (self.M * self.Nr)
        azimuth_modes = np.fft.fftfreq(self.M, d=1.0 / self.M)
        range_modes = np.fft.fftfreq(self.Nr, d=1.0 / self.Nr)
        coefficients, azimuth_modes = _split_nyquist(
            coefficients, azimuth_modes, axis=0
        )
        coefficients, range_modes = _split_nyquist(coefficients, range_modes, axis=1)

        if self.band_frac < 1.0:
            retained = np.abs(azimuth_modes) <= self.band_frac * self.M / 2.0
            coefficients = coefficients[retained]
            azimuth_modes = azimuth_modes[retained]

        self.C = coefficients
        self.m_az = azimuth_modes
        self.p_r = range_modes
        self.report = EvalReport(
            sweep=geometry.sweep,
            sweep_class=str(sweep_class.value),
            az_path=path,
            M_az=self.M,
            Nr_ext=self.Nr,
            r0_m=self.r0,
            dr_m=self.dr,
            ngates=self.ng,
            build_s=time.perf_counter() - started_at,
            extras=extras,
        )

    def _exact_lattice(self, values, azimuths):
        """Sort the rays onto the closing lattice they already sit on."""
        order = np.argsort(azimuths)
        sorted_azimuths = azimuths[order]
        self.M = int(self.geometry.nrays)
        ray_index = np.arange(self.M, dtype=np.float64)
        # Least-squares lattice offset, which halves the residual compared with
        # simply taking the first azimuth as the origin.
        self.phi0 = float(np.mean(sorted_azimuths - ray_index * (360.0 / self.M)))
        residual = np.max(
            np.abs(sorted_azimuths - (self.phi0 + ray_index * 360.0 / self.M))
        )
        return (
            values[order],
            "fft_exact",
            {"lattice_resid_max_deg": float(residual)},
        )

    def _sector_lattice(self, values, azimuths):
        """Place the sector on a full circle and close the unmeasured gap."""
        order = np.argsort(azimuths)
        sorted_azimuths = azimuths[order]
        sorted_values = values[order]

        n_cells = max(
            int(round(360.0 / self.geometry.az_spacing_median_deg)),
            self.geometry.nrays + 2,
        )
        self.M = n_cells
        self.phi0 = float(sorted_azimuths[0])

        lattice = np.zeros((n_cells, self.ng))
        cell_of_ray = (
            np.rint(((sorted_azimuths - self.phi0) % 360.0) / (360.0 / n_cells)).astype(
                int
            )
            % n_cells
        )
        lattice[cell_of_ray] = sorted_values

        occupied = np.zeros(n_cells, dtype=bool)
        occupied[cell_of_ray] = True
        gap_cells = np.flatnonzero(~occupied)

        if gap_cells.size:
            # Raised-cosine blend across the (cyclically contiguous) gap, so the
            # extension is continuous at both sector edges.
            blend = 0.5 * (
                1.0
                - np.cos(
                    np.pi * np.arange(1, gap_cells.size + 1) / (gap_cells.size + 1)
                )
            )
            if self.sector_mode == "mirror":
                span = max(sorted_values.shape[0] - 1, 1)
                step = np.minimum(
                    np.arange(1, gap_cells.size + 1), sorted_values.shape[0] - 1
                )
                from_end = sorted_values[-1 - step % span]
                from_start = sorted_values[step % span]
                lattice[gap_cells] = (1 - blend)[:, None] * from_end + blend[
                    :, None
                ] * from_start
            else:
                lattice[gap_cells] = (1 - blend)[:, None] * sorted_values[-1][
                    None, :
                ] + blend[:, None] * sorted_values[0][None, :]

        return (
            lattice,
            f"sector_{self.sector_mode}",
            {
                "n_gap_cells": int(gap_cells.size),
                "sector_cells": int(cell_of_ray.size),
            },
        )

    def _nufft_lattice(self, values, azimuths):
        """Recover a uniform lattice from non-uniform rays by KB gridding.

        The engine supplies the transforms and the solver consumes them; both are
        reported, because which one was used determines the accuracy guarantee.
        ``adjoint_test_rel`` is measured on whichever engine ran --- an engine
        whose adjoint were not the transpose of its forward would invalidate the
        solve, so the check is per-engine and not inherited.
        """
        operator = make_operator(
            azimuths,
            self.geometry,
            engine=self.az_engine,
            kb_width=self.kb_width,
            oversamp=self.oversamp,
        )
        lattice, info = operator.solve(
            values,
            n_cg=self.n_cg,
            cg_tol=self.cg_tol,
            solver=self.az_solver,
            ridge=self.az_ridge,
        )
        self.M = operator.n_lattice
        self.phi0 = 0.0
        extras = dict(operator.describe())
        extras.update(
            {
                "density_w_min": float(operator.weights.min()),
                "density_w_max": float(operator.weights.max()),
                "adjoint_test_rel": operator.adjoint_test(),
            }
        )
        extras.update(info)
        return lattice, "nufft_kb", extras

    def evaluate(self, r_m, az_deg, chunk=256):
        """Evaluate the 2-D Fourier series directly at each target.

        Exact for the ``EXACT_UNIFORM_PERIODIC`` path, up to floating point. Cost is
        ``O(n_targets * M * Nr)``, so this is the reference implementation --- use
        :meth:`evaluate_fast` for production grids.

        Parameters
        ----------
        r_m, az_deg : array_like
            Target ranges in metres and azimuths in degrees. Broadcast together; the
            result takes the shape of ``r_m``.
        chunk : int, optional
            Targets evaluated per batch, trading memory against speed.

        Returns
        -------
        numpy.ndarray
        """
        ranges = np.atleast_1d(np.asarray(r_m, dtype=np.float64)).ravel()
        target_azimuths = np.atleast_1d(np.asarray(az_deg, dtype=np.float64)).ravel()
        out = np.empty(ranges.size)

        gate_position = (ranges - self.r0) / self.dr
        azimuth_radians = np.deg2rad((target_azimuths - self.phi0) % 360.0)
        for start in range(0, ranges.size, chunk):
            batch = slice(start, min(start + chunk, ranges.size))
            azimuth_basis = np.exp(1j * np.outer(azimuth_radians[batch], self.m_az))
            range_basis = np.exp(
                2j * np.pi * np.outer(gate_position[batch], self.p_r) / self.Nr
            )
            out[batch] = np.real(
                np.einsum(
                    "cm,mp,cp->c", azimuth_basis, self.C, range_basis, optimize=True
                )
            )
        return out.reshape(np.shape(r_m))

    def _lattice(self):
        """Reconstruct the uniform (azimuth x extended-range) lattice.

        Coefficients are scattered back to their own FFT frequency bins rather than
        assumed to fill the spectrum densely. That distinction matters when the band
        has been limited: the retained set is then shorter than the transform length,
        and reshaping it directly would both misplace every mode and mis-size the
        array.
        """
        coefficients, azimuth_modes = _merge_nyquist(
            self.C.copy(), self.m_az, axis=0, length=self.M
        )
        coefficients, range_modes = _merge_nyquist(
            coefficients, self.p_r, axis=1, length=self.Nr
        )

        if coefficients.shape == (self.M, self.Nr):
            full = coefficients
        else:
            full = np.zeros((self.M, self.Nr), dtype=complex)
            azimuth_bins = (
                np.rint(np.asarray(azimuth_modes, dtype=float)).astype(int) % self.M
            )
            range_bins = (
                np.rint(np.asarray(range_modes, dtype=float)).astype(int) % self.Nr
            )
            full[np.ix_(azimuth_bins, range_bins)] = coefficients
        return np.real(ifft2(full * (self.M * self.Nr)))

    def dense(self, up_az=4, up_r=4):
        """Band-limited zero-padded upsampling of the lattice, cached.

        Parameters
        ----------
        up_az, up_r : int, optional
            Upsampling factors for azimuth and range.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(M_az * up_az, Nr_ext * up_r)``.
        """
        key = (up_az, up_r)
        if self._dense is not None and self._dense[0] == key:
            return self._dense[1]

        n_azimuth, n_range = self.M * up_az, self.Nr * up_r
        spectrum = fft2(self._lattice())
        padded = np.zeros((n_azimuth, n_range), dtype=complex)
        half_az, half_r = self.M // 2, self.Nr // 2
        padded[:half_az, :half_r] = spectrum[:half_az, :half_r]
        padded[:half_az, n_range - (self.Nr - half_r) :] = spectrum[:half_az, half_r:]
        padded[n_azimuth - (self.M - half_az) :, :half_r] = spectrum[half_az:, :half_r]
        padded[n_azimuth - (self.M - half_az) :, n_range - (self.Nr - half_r) :] = (
            spectrum[half_az:, half_r:]
        )

        upsampled = np.real(ifft2(padded)) * (up_az * up_r)
        self._dense = (key, upsampled)
        return upsampled

    def evaluate_fast(self, r_m, az_deg, up_az=4, up_r=4):
        """Evaluate via band-limited upsampling plus bilinear interpolation.

        Much cheaper than :meth:`evaluate` for large target sets, and converges to it
        as the upsampling factors grow. Azimuth wraps periodically; range is clamped
        inside the physical span of the sweep, since beyond the last gate there is no
        data and the mirror extension must not be mistaken for one.

        Parameters
        ----------
        r_m, az_deg : array_like
            Target ranges in metres and azimuths in degrees.
        up_az, up_r : int, optional
            Upsampling factors passed to :meth:`dense`.

        Returns
        -------
        numpy.ndarray
            Same shape as ``r_m``.
        """
        upsampled = self.dense(up_az, up_r)
        n_azimuth, n_range = upsampled.shape

        azimuth_position = (
            (np.asarray(az_deg, dtype=np.float64) - self.phi0) % 360.0
        ) / (360.0 / n_azimuth)
        range_position = (np.asarray(r_m, dtype=np.float64) - self.r0) / (
            self.dr / up_r
        )
        range_position = np.clip(range_position, 0.0, (self.ng - 1) * up_r)

        azimuth_floor = np.floor(azimuth_position).astype(int)
        range_floor = np.floor(range_position).astype(int)
        azimuth_frac = azimuth_position - azimuth_floor
        range_frac = range_position - range_floor

        azimuth_lo = azimuth_floor % n_azimuth
        azimuth_hi = (azimuth_floor + 1) % n_azimuth
        range_lo = np.clip(range_floor, 0, n_range - 1)
        range_hi = np.clip(range_floor + 1, 0, n_range - 1)

        return (
            upsampled[azimuth_lo, range_lo] * (1 - azimuth_frac) * (1 - range_frac)
            + upsampled[azimuth_hi, range_lo] * azimuth_frac * (1 - range_frac)
            + upsampled[azimuth_lo, range_hi] * (1 - azimuth_frac) * range_frac
            + upsampled[azimuth_hi, range_hi] * azimuth_frac * range_frac
        )
