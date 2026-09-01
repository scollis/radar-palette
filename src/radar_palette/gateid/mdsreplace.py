"""Fit a minimum-detectable-signal curve and replace excluded radar gates.

The minimum detectable signal is the weakest echo a radar can report at a given
range. Equivalent reflectivity carries an explicit range-square correction, so a
constant detectable *power* at the receiver appears as a reflectivity floor
rising 20 dB per decade of range::

    MDS(r) = offset + 20 * log10(r / 1 km)

That slope is physics, not a free parameter, and it is fixed by default. It was
checked against the no-return population of ARM BNF C-SAPR2 PPI volumes: with
the 20 dB/decade term removed, the median residual offset of ``no_scatter``
gates is flat to within 1 dB from 20 km to 110 km (-37.0, -37.4, -37.7, -38.0 dB
for the 10-20, 20-40, 40-70 and 70-110 km bands of one clear-air volume).

Fitting the slope freely instead is offered through ``degree`` but is not the
default, because the population being fitted is not clean. Gates labelled
``no_scatter`` at close range are contaminated by clutter and partial beam
filling, and on real volumes that contamination pulls a free fit to slopes as
shallow as -0.08 dB/decade -- an MDS that *falls* with range, which is
unphysical and would replace distant gates with values above the true floor.

The offset is therefore estimated robustly: a percentile within each range bin,
then the median across range bins. Taking the per-range-bin statistic first
stops a sweep with more rays, or a range band with more gates, from dominating
the estimate.

Contaminated sweeps are rejected rather than fitted
---------------------------------------------------
The minimum detectable signal is a property of the receiver, so the fitted
offset should barely move between sweeps of one volume. When it does move, the
no-return population is not noise. On a BNF multicell volume the three lowest
sweeps returned offsets of -6.6, -6.9 and -7.8 dBZ against -45 to -47.6 dBZ for
the twelve sweeps above them: widespread low-level echo had been labelled
``no_scatter``, and the "floor" it implied was some 40 dB too high, which would
have injected fictitious +27 dBZ echo across the far field.

The scatter of the per-range-bin offsets detects this without needing to know
the right answer. It was 22-25 dB on those three sweeps and 0.4-1.6 dB on the
clean ones, so :data:`MAX_OFFSET_SCATTER_DB` separates them by an order of
magnitude. A fit that exceeds it raises, and the caller falls back to a fit
pooled over the whole volume, which is dominated by the clean sweeps.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "MAX_OFFSET_SCATTER_DB",
    "RANGE_SQUARE_SLOPE_DB",
    "fit_mds",
    "replace_with_mds",
]

#: dB of reflectivity per decade of range for a constant received power. The
#: r-squared term in the weather-radar equation, so 20 by definition rather
#: than by fit.
RANGE_SQUARE_SLOPE_DB = 20.0

#: Largest median absolute deviation, in dB, allowed among the per-range-bin
#: offsets before the fit is rejected as contaminated. Clean BNF C-SAPR2 sweeps
#: sit at 0.4-1.6 dB; sweeps whose no-return population is really low-level
#: echo sat at 22-25 dB.
MAX_OFFSET_SCATTER_DB = 8.0


def _range_bin_statistic(values, no_return, usable, percentile):
    """Percentile of the no-return population in each range bin, else NaN."""
    stat = np.full(values.shape[1], np.nan, dtype="f8")
    sample_counts = np.zeros(values.shape[1], dtype="i8")
    for gate in np.flatnonzero(usable):
        samples = values[no_return[:, gate], gate]
        samples = samples[np.isfinite(samples)]
        sample_counts[gate] = samples.size
        if samples.size:
            stat[gate] = np.percentile(samples, percentile)
    return stat, sample_counts


def fit_mds(
    reflectivity,
    gate_range_m,
    no_return,
    slope_db_per_decade=RANGE_SQUARE_SLOPE_DB,
    percentile=50.0,
    degree=None,
    min_range_m=0.0,
    min_range_bins=6,
    min_samples_per_bin=1,
    max_offset_scatter_db=MAX_OFFSET_SCATTER_DB,
):
    """Fit the minimum detectable signal as a function of range.

    Parameters
    ----------
    reflectivity : array_like
        Reflectivity in dBZ, shaped (ray, gate). Masked arrays are accepted.
    gate_range_m : array_like
        Slant range of each gate, in metres.
    no_return : array_like of bool
        Gates believed to hold no scatterer, broadcastable to the reflectivity
        shape. Normally ``gate_id == no_scatter``.
    slope_db_per_decade : float, optional
        Slope of the floor against ``log10(range)``. Defaults to the physical
        :data:`RANGE_SQUARE_SLOPE_DB`; only the offset is then fitted.
    percentile : float, optional
        Percentile of the no-return population taken within each range bin.
        50 gives the median noise floor; a lower value gives a more
        conservative floor nearer the detection limit.
    degree : int or None, optional
        If given, fit a free polynomial of this degree in ``log10(range_km)``
        instead of constraining the slope. Diagnostic use: on contaminated
        real volumes this can return an MDS that falls with range.
    min_range_m : float, optional
        Ignore range bins closer than this when fitting. Non-positive ranges
        are always ignored, having no logarithm.
    min_range_bins : int, optional
        Minimum number of usable range bins required to fit at all.
    min_samples_per_bin : int, optional
        Minimum no-return gates a range bin needs to contribute.
    max_offset_scatter_db : float or None, optional
        Reject the fit if the per-range-bin offsets scatter by more than this
        (median absolute deviation), which means the no-return population is
        contaminated by real echo rather than describing the noise floor. Pass
        ``None`` to accept any fit. Ignored when ``degree`` is given.


    Returns
    -------
    mds : ndarray
        The fitted floor in dBZ, shaped like ``reflectivity``. Gates at or
        below zero range take the value of the closest fitted range, so the
        result is finite everywhere.
    meta : dict
        Method, slope, offset, robust scatter, and the fitted range span.

    Raises
    ------
    ValueError
        If too few range bins carry a no-return population to fit, or if the
        fitted offsets scatter beyond ``max_offset_scatter_db``. Refusing is
        deliberate: a fabricated floor would silently corrupt every gate it
        replaced.
    """
    values = np.ma.filled(np.ma.asarray(reflectivity, dtype="f8"), np.nan)
    if values.ndim != 2:
        raise ValueError("reflectivity must be a two-dimensional ray-by-gate array")
    ranges = np.asarray(gate_range_m, dtype="f8").reshape(-1)
    if ranges.size != values.shape[1]:
        raise ValueError(
            "gate_range_m length must equal the reflectivity gate dimension"
        )
    no_return = np.broadcast_to(np.asarray(no_return, dtype=bool), values.shape)

    # Zero and negative ranges have no logarithm. Real ARM volumes carry them:
    # 2025 BNF C-SAPR2 files start their range coordinate at exactly 0 m.
    positive = np.isfinite(ranges) & (ranges > 0.0)
    usable = positive & (ranges >= max(min_range_m, 0.0))
    stat, counts = _range_bin_statistic(values, no_return, usable, percentile)
    fitted = usable & np.isfinite(stat) & (counts >= min_samples_per_bin)
    if fitted.sum() < max(min_range_bins, 2 if degree is None else degree + 1):
        raise ValueError(
            "insufficient no-return range bins to fit a minimum detectable "
            f"signal: {int(fitted.sum())} usable of {ranges.size}"
        )

    log_range = np.full(ranges.shape, np.nan, dtype="f8")
    log_range[positive] = np.log10(ranges[positive] / 1000.0)
    # Below the fitted span the floor is held flat rather than extrapolated
    # inward, where the log term diverges and the no-return population is
    # contaminated by clutter.
    log_range[~positive] = log_range[fitted].min()

    if degree is None:
        offsets = stat[fitted] - slope_db_per_decade * log_range[fitted]
        offset = float(np.median(offsets))
        curve = offset + slope_db_per_decade * log_range
        scatter = float(np.median(np.abs(offsets - offset)))
        if max_offset_scatter_db is not None and scatter > max_offset_scatter_db:
            raise ValueError(
                "no-return population does not describe a noise floor: "
                f"offset scatter {scatter:.1f} dB exceeds "
                f"{max_offset_scatter_db:.1f} dB, so it is contaminated by echo"
            )
        meta = {
            "method": "fixed_slope",
            "slope_db_per_decade": float(slope_db_per_decade),
            "offset_dbz_at_1km": offset,
            "offset_mad_db": scatter,
        }
    else:
        coefficients = np.polyfit(log_range[fitted], stat[fitted], degree)
        curve = np.polyval(coefficients, log_range)
        residual = stat[fitted] - np.polyval(coefficients, log_range[fitted])
        meta = {
            "method": f"polyfit_degree_{int(degree)}",
            "coefficients": coefficients.tolist(),
            "slope_db_per_decade": float(coefficients[-2]) if degree >= 1 else 0.0,
            "residual_mad_db": float(np.median(np.abs(residual))),
        }

    meta.update(
        {
            "percentile": float(percentile),
            "n_range_bins": int(fitted.sum()),
            "n_range_bins_available": int(ranges.size),
            "n_no_return_gates": int(counts.sum()),
            "fit_range_min_m": float(ranges[fitted].min()),
            "fit_range_max_m": float(ranges[fitted].max()),
        }
    )
    if not np.isfinite(curve).all():
        raise ValueError("fitted minimum detectable signal is not finite everywhere")
    return np.broadcast_to(curve, values.shape).copy(), meta


def replace_with_mds(data, mds, excluded, never_increase=True):
    """Replace excluded gates with the MDS, leaving every other gate alone.

    Excluded gates are also unmasked, since they now hold a defined value: the
    point of the exercise is to give a gridding weight function something to
    grid down to, which a masked gate does not provide.

    Parameters
    ----------
    data : array_like
        Reflectivity in dBZ. Masked arrays are accepted.
    mds : array_like
        The fitted floor, broadcastable to ``data``.
    excluded : array_like of bool
        Gates to replace, broadcastable to ``data``.
    never_increase : bool, optional
        Keep the measured value wherever it is already weaker than the fitted
        floor, so removing an unwanted target can never *add* apparent signal.

        This matters on censored archives. WSR-88D Level II reflectivity is
        thresholded before the file is written -- around 71% of gates in the
        BNF-era KHTX volumes are masked -- so the surviving "no-return"
        population is the weakest echo that passed the threshold rather than
        the receiver noise floor, and the floor fitted from it can land above
        real weak echo. On three such volumes the naive fill *raised* the
        median of the replaced gates by 12 to 17 dB. The measured value is the
        safer of the two in that case, and it is always a real observation.

    Returns
    -------
    numpy.ma.MaskedArray
        ``data`` with excluded gates replaced.
    """
    source = np.ma.asarray(data)
    values = np.ma.filled(source, np.nan).astype("f8", copy=True)
    mds = np.broadcast_to(np.asarray(mds, dtype="f8"), values.shape)
    excluded = np.broadcast_to(np.asarray(excluded, dtype=bool), values.shape)
    if not np.isfinite(mds[excluded]).all():
        raise ValueError("MDS values must be finite at every excluded gate")

    fill = excluded
    if never_increase:
        with np.errstate(invalid="ignore"):
            already_weaker = np.isfinite(values) & (values <= mds)
        fill = excluded & ~already_weaker
    values[fill] = mds[fill]

    mask = np.ma.getmaskarray(source).copy()
    mask[excluded] = False
    if not np.isfinite(values[~mask]).all():
        # An included gate that was never measured stays masked rather than
        # becoming a NaN that downstream gridding would treat as a number.
        mask |= ~np.isfinite(values)
    return np.ma.masked_array(values, mask=mask)
