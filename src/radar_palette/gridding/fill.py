"""Making a sweep gap-free, because an FFT cannot see a mask.

Radar sweeps arrive with gaps: below the noise floor, blocked by terrain, or beyond
the last valid gate. A discrete Fourier transform has no notion of a missing sample,
so the gaps must be given values before spectral evaluation --- and *which* values
matters, because whatever is put there is treated as data.

Three strategies, in increasing order of effort and of faithfulness:

``floor``
    Constant fill. Honest and predictable, but introduces a step at every gap edge,
    which is exactly the discontinuity a band-limited interpolant rings on.
``edge``
    Linear interpolation along range within each ray, holding the end values beyond
    the last valid gate. Continuous, so it rings far less; the default choice for
    interior gaps.
``gerchberg``
    Gerchberg-Papoulis iteration: alternately project onto a band and re-impose the
    measured samples. The gap fill is then consistent with the band-limit assumption
    the evaluator makes, rather than merely continuous. Costs a few dozen FFTs.

All three leave measured samples untouched --- that is asserted in the tests, and for
``gerchberg`` it is the load-bearing property, since the iteration would otherwise
drift away from the data it is supposed to honour.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, ifft2

__all__ = ["fill_sweep"]

FILL_METHODS = ("floor", "edge", "gerchberg")


def _fill_along_range(data, mask, floor):
    """Linear interpolation along range within each ray, floor for empty rays."""
    filled = data.copy()
    n_rays, n_gates = filled.shape
    gate_index = np.arange(n_gates)
    for ray in range(n_rays):
        valid = ~mask[ray]
        if not valid.any():
            filled[ray] = floor
            continue
        # np.interp holds the end values outside the valid span, which is the
        # behaviour wanted beyond the last good gate.
        filled[ray] = np.interp(gate_index, gate_index[valid], data[ray][valid])
    return filled


def _band_limited_gap_fill(seed_values, known, band_frac, n_iter, tol):
    """Gerchberg-Papoulis iteration on the mirror-extended sweep."""
    n_gates = seed_values.shape[1]
    filled = seed_values.copy()
    measured = seed_values.copy()

    # Build the band mask once, on the mirror-extended geometry the evaluator uses.
    extended = np.concatenate([filled, filled[:, -2:0:-1]], axis=1)
    n_azimuth, n_extended = extended.shape
    azimuth_frequency = np.fft.fftfreq(n_azimuth, d=1.0 / n_azimuth)
    range_frequency = np.fft.fftfreq(n_extended, d=1.0 / n_extended)
    in_band = (np.abs(azimuth_frequency)[:, None] <= band_frac * n_azimuth / 2.0) & (
        np.abs(range_frequency)[None, :] <= band_frac * n_extended / 2.0
    )

    previous = filled.copy()
    for _ in range(n_iter):
        extended = np.concatenate([filled, filled[:, -2:0:-1]], axis=1)
        spectrum = fft2(extended)
        spectrum[~in_band] = 0.0
        candidate = np.real(ifft2(spectrum))[:, :n_gates]
        # Re-impose the measurements: the iteration may move gap values freely, but
        # never the data.
        filled = np.where(known, measured, candidate)
        largest_change = np.nanmax(np.abs(filled - previous))
        previous = filled.copy()
        if largest_change < tol:
            break
    return filled


def fill_sweep(
    values,
    method="floor",
    floor=None,
    band_frac=0.5,
    n_iter=40,
    tol=1e-4,
    return_mask=False,
):
    """Return a gap-free copy of a sweep, suitable for spectral evaluation.

    Parameters
    ----------
    values : array_like or numpy.ma.MaskedArray
        Sweep of shape ``(nrays, ngates)``. Masked entries and non-finite values are
        both treated as gaps.
    method : {'floor', 'edge', 'gerchberg'}, optional
        Fill strategy; see the module docstring for the trade-offs.
    floor : float, optional
        Constant used by ``floor``, and as the fallback for an entirely masked ray
        under the other methods. Defaults to the minimum of the valid data.
    band_frac : float, optional
        Band half-width, as a fraction of Nyquist, for ``gerchberg``.
    n_iter, tol : int, float, optional
        Iteration limit and convergence tolerance for ``gerchberg``.
    return_mask : bool, optional
        If True, also return the boolean gap mask, so a caller can tell which
        outputs were synthesised.

    Returns
    -------
    filled : numpy.ndarray
        Gap-free float array of the same shape. Never a masked array.
    mask : numpy.ndarray, optional
        The gap mask, returned only when ``return_mask`` is True.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``FILL_METHODS``.
    """
    if method not in FILL_METHODS:
        raise ValueError(
            f"unknown fill method {method!r}; expected one of {FILL_METHODS}"
        )

    as_masked = np.ma.asanyarray(values)
    data = np.ma.filled(as_masked, np.nan).astype(np.float64)
    mask = np.ma.getmaskarray(as_masked) | ~np.isfinite(data)

    if floor is None:
        floor = float(np.nanmin(np.where(mask, np.nan, data))) if (~mask).any() else 0.0

    if method == "floor":
        filled = np.where(mask, floor, data)
    else:
        filled = _fill_along_range(data, mask, floor)
        if method == "gerchberg":
            filled = _band_limited_gap_fill(
                filled, ~mask, band_frac=band_frac, n_iter=n_iter, tol=tol
            )

    # Any residual non-finite value (an all-masked input, say) becomes the floor.
    filled = np.where(np.isfinite(filled), filled, floor)
    return (filled, mask) if return_mask else filled
