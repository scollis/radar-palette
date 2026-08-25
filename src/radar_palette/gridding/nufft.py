"""Kaiser-Bessel NUFFT operator pair for the azimuth axis of one sweep.

When azimuths are not on a uniform lattice, recovering the lattice is an inverse
problem rather than a transform. This module supplies the two operators that problem
needs, and --- importantly --- makes them exact transposes of one another.

The model is::

    f_j = sum_m c_m exp(i m phi_j),    |m| <= N0 / 2

where ``phi_j`` are the *measured* azimuths (non-uniform, periodic mod 360) and the
unknown is the uniform lattice of ``N0`` points whose DFT is ``c``.

:meth:`_AzimuthNufftOperator.forward` is the type-2 transform (lattice to rays):
evaluate the band-limited interpolant at the measured azimuths.
:meth:`_AzimuthNufftOperator.adjoint` is the type-1 transform (rays to lattice), and
is the exact transpose of the forward under the density weights.

Why the transpose has to be exact
---------------------------------
Convolution gridding --- weight by sampling density, spread with a Kaiser-Bessel
kernel, FFT, deapodise --- computes the **adjoint** of the sampling operator, not its
inverse. For jittered azimuths the adjoint leaks a smooth field's energy all the way
to Nyquist: measured on a field with no true energy above azimuthal mode 7, plain
gridding placed 2% of the DC amplitude into mode 177.

The fix is to solve the density-weighted normal equations::

    (E^H W E) c = E^H W f

by conjugate gradients, using these two operators for the matrix-vector products.
CG is only valid if ``adjoint`` really is the transpose of ``forward``, which is why
:meth:`_AzimuthNufftOperator.adjoint_test` exists and is recorded in the evaluator's
report. Transposing ``A = K F_M^H S diag(1/Khat) F_N0 / N0`` term by term leaves no
stray factor of ``M`` or ``N0``; an earlier version of this code carried one, and its
adjoint test returned ~1 instead of ~1e-15 --- a discrepancy invisible to any test
that merely checked the output looked smooth.

The kernel's Fourier transform (for deapodisation) is computed by quadrature from the
same function used for spreading, rather than from a closed form, so the two are
guaranteed consistent for any width and shape parameter.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.special import i0

__all__ = ["kaiser_bessel_beta", "kaiser_bessel_kernel"]


def kaiser_bessel_beta(width, oversamp):
    """Kaiser-Bessel shape parameter for minimum aliasing.

    The min-aliasing choice of Beatty, Nishimura and Pauly (2005), which ties the
    shape parameter to the kernel width and the lattice oversampling rather than
    leaving it as a free knob.

    Parameters
    ----------
    width : float
        Kernel full width, in oversampled lattice cells.
    oversamp : float
        Lattice oversampling factor.

    Returns
    -------
    float
    """
    scaled_width = width * (1.0 - 0.5 / oversamp)
    return float(np.pi * np.sqrt(max(scaled_width * scaled_width - 0.8, 1e-6)))


def kaiser_bessel_kernel(offsets, width, beta):
    """Kaiser-Bessel kernel, zero outside ``|offsets| <= width / 2``.

    Parameters
    ----------
    offsets : array_like
        Distances from the kernel centre, in units of lattice cells.
    width : float
        Kernel full width in cells.
    beta : float
        Shape parameter, normally from :func:`kaiser_bessel_beta`.

    Returns
    -------
    numpy.ndarray
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    normalised = 2.0 * offsets / width
    kernel = np.zeros_like(normalised, dtype=np.float64)
    inside = np.abs(normalised) <= 1.0
    kernel[inside] = i0(beta * np.sqrt(1.0 - normalised[inside] ** 2))
    return kernel


def _kernel_fourier_transform(mode_index, n_lattice, width, beta, n_quadrature=4096):
    """Fourier transform of the KB kernel at integer Fourier indices.

    Computed by quadrature from :func:`kaiser_bessel_kernel` itself rather than from
    a remembered closed form, so deapodisation is guaranteed consistent with the
    spreading kernel for any ``(width, beta)``.
    """
    offsets = np.linspace(-width / 2.0, width / 2.0, n_quadrature)
    kernel = kaiser_bessel_kernel(offsets, width, beta)
    cell_step = offsets[1] - offsets[0]
    phase = np.exp(-2j * np.pi * np.outer(mode_index / n_lattice, offsets))
    return np.real(phase @ kernel) * cell_step


class _AzimuthNufftOperator:
    """Forward/adjoint operator pair on the azimuth axis of one sweep.

    Private: constructed by :class:`~radar_palette.gridding.evaluator.
    SweepSpectralEvaluator` for the ``NON_UNIFORM`` path. The public surface is the
    evaluator, whose report exposes this operator's diagnostics.
    """

    def __init__(self, azimuths_deg, geometry, kb_width=4.0, oversamp=2.0):
        azimuths_deg = np.asarray(azimuths_deg, dtype=np.float64) % 360.0
        self.order = np.argsort(azimuths_deg)
        self.azimuths = azimuths_deg[self.order]
        self.n_rays = self.azimuths.size
        self.kb_width = float(kb_width)
        self.oversamp = float(oversamp)

        # Lattice size is measured, never assumed. A full-circle sweep's natural
        # band-limited lattice has exactly as many points as it has rays; for a
        # sector, fall back to the measured spacing.
        self.n_lattice = (
            int(geometry.nrays)
            if geometry.is_full_360
            else int(round(360.0 / geometry.az_spacing_median_deg))
        )
        self.n_oversampled = next_fast_len(int(np.ceil(self.oversamp * self.n_lattice)))
        self.beta = kaiser_bessel_beta(self.kb_width, self.oversamp)
        self.cell_deg = 360.0 / self.n_oversampled

        # Voronoi arc-length weights in degrees: each ray owns half the arc to each
        # neighbour, periodic around the circle, so the weights sum to 360.
        self.weights = (
            (np.roll(self.azimuths, -1) - np.roll(self.azimuths, 1)) % 360.0
        ) / 2.0

        # Kaiser-Bessel spreading stencil, precomputed.
        position = self.azimuths / self.cell_deg
        self.n_stencil = int(np.floor(self.kb_width)) + 1
        first_cell = np.ceil(position - self.kb_width / 2.0).astype(int)
        stencil_offsets = np.arange(self.n_stencil)[None, :]
        self.cells = (first_cell[:, None] + stencil_offsets) % self.n_oversampled
        self.kernel_values = kaiser_bessel_kernel(
            position[:, None] - (first_cell[:, None] + stencil_offsets),
            self.kb_width,
            self.beta,
        )

        self.mode_index = np.fft.fftfreq(
            self.n_oversampled, d=1.0 / self.n_oversampled
        ).astype(int)
        self.deapodisation = _kernel_fourier_transform(
            self.mode_index, self.n_oversampled, self.kb_width, self.beta
        )
        self._build_zero_pad_map()

    def _build_zero_pad_map(self):
        """Map each of the N0 spectral bins onto its oversampled-lattice bin.

        For even ``N0`` the Nyquist bin is SPLIT, half to ``+N0/2`` and half to
        ``-N0/2``, so the off-lattice interpolant stays real-valued and symmetric.
        Assigning it wholly to one side produces a complex interpolant.
        """
        lattice_modes = np.fft.fftfreq(self.n_lattice, d=1.0 / self.n_lattice).astype(
            int
        )
        bin_of_mode = {mode: index for index, mode in enumerate(self.mode_index)}
        rows, columns, weights = [], [], []
        for column, mode in enumerate(lattice_modes):
            if self.n_lattice % 2 == 0 and mode == -self.n_lattice // 2:
                rows += [bin_of_mode[mode], bin_of_mode[-mode]]
                columns += [column, column]
                weights += [0.5, 0.5]
            else:
                rows.append(bin_of_mode[mode])
                columns.append(column)
                weights.append(1.0)
        self._pad_rows = np.asarray(rows)
        self._pad_columns = np.asarray(columns)
        self._pad_weights = np.asarray(weights)

    def forward(self, lattice_values):
        """Type-2 transform: uniform lattice to measured azimuths."""
        values_2d = (
            lattice_values[:, None] if lattice_values.ndim == 1 else lattice_values
        )
        coefficients = fft(values_2d, axis=0) / self.n_lattice

        padded = np.zeros((self.n_oversampled, values_2d.shape[1]), dtype=complex)
        np.add.at(
            padded,
            self._pad_rows,
            self._pad_weights[:, None] * coefficients[self._pad_columns],
        )
        oversampled = (
            np.real(ifft(padded / self.deapodisation[:, None], axis=0))
            * self.n_oversampled
        )

        out = np.zeros((self.n_rays, values_2d.shape[1]))
        for stencil in range(self.n_stencil):
            out += (
                self.kernel_values[:, stencil][:, None]
                * oversampled[self.cells[:, stencil]]
            )
        return out.ravel() if lattice_values.ndim == 1 else out

    def adjoint(self, ray_values, weighted=True):
        """Type-1 transform: measured azimuths to uniform lattice.

        The exact transpose of :meth:`forward` under the density weights ``W``.
        Transposing ``A = K F_M^H S diag(1/Khat) F_N0 / N0`` term by term gives
        ``A^H W = ifft_N0(S^H fft_M(K^T W f) / Khat)`` --- with no leftover factor of
        ``M`` or ``N0``. See the module docstring.
        """
        values_2d = ray_values[:, None] if ray_values.ndim == 1 else ray_values
        weighted_values = values_2d * self.weights[:, None] if weighted else values_2d

        spread = np.zeros((self.n_oversampled, values_2d.shape[1]))
        for stencil in range(self.n_stencil):
            np.add.at(
                spread,
                self.cells[:, stencil],
                self.kernel_values[:, stencil][:, None] * weighted_values,
            )
        spectrum = fft(spread, axis=0) / self.deapodisation[:, None]

        truncated = np.zeros((self.n_lattice, values_2d.shape[1]), dtype=complex)
        np.add.at(
            truncated,
            self._pad_columns,
            self._pad_weights[:, None] * spectrum[self._pad_rows],
        )
        out = np.real(ifft(truncated, axis=0))
        return out.ravel() if ray_values.ndim == 1 else out

    def adjoint_test(self, seed=0):
        """Relative mismatch of ``<A u, f>_W`` against ``<u, A^H f>``.

        Should be at machine-precision level. A value near 1 means a scale factor is
        missing from the transpose, and conjugate gradients on the normal equations
        would be solving the wrong problem.
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

    def solve(self, values, n_cg=0, cg_tol=1e-10):
        """Recover the uniform lattice from the measured rays.

        Stage 1 (always): density-compensated Kaiser-Bessel convolution gridding.
        The Voronoi weights are in degrees and sum to 360, so ``adjoint(f)`` returns
        ``ifft(S_m)`` with ``S_m = sum_j w_j f_j exp(-i m phi_j)``. The Fourier-series
        coefficient is ``c_m = S_m / 360`` and the lattice values are
        ``N0 * ifft(c)``, hence the analytic scale ``N0 / 360``. **No empirical
        normalisation.**

        Stage 2 (``n_cg > 0``, off by default): conjugate gradients on the
        density-weighted normal equations, turning the adjoint into a genuine
        least-squares band-limited fit.

        Returns
        -------
        lattice : numpy.ndarray
        info : dict
            Diagnostics, merged into the evaluator's report.
        """
        values = np.asarray(values, dtype=np.float64)
        rays = values[self.order]
        right_hand_side = self.adjoint(rays)
        lattice = right_hand_side * (self.n_lattice / 360.0)
        info = {"n_cg": int(n_cg)}
        if n_cg <= 0:
            return lattice, info

        residual = right_hand_side - self.adjoint(self.forward(lattice))
        reference_norm = max(float(np.linalg.norm(right_hand_side)), 1e-30)
        info["cg_resid_start"] = float(np.linalg.norm(residual) / reference_norm)

        direction = residual.copy()
        residual_sq = float(np.sum(residual * residual))
        # Counted explicitly rather than read from the loop variable, which would be
        # left at its initial value if the loop body never ran.
        iterations_run = 0
        for _ in range(int(n_cg)):
            applied = self.adjoint(self.forward(direction))
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
