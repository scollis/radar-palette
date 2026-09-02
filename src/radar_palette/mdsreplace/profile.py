r"""Estimation of the minimum detectable signal as a function of range.

The physics is one line. A radar's minimum detectable signal is a fixed
received power, and reflectivity is that power multiplied by :math:`r^2`, so
the smallest reflectivity the radar can report grows as

.. math::

    Z_{min}(r)\;[\mathrm{dBZ}] = c + 20 \log_{10}(r / r_{ref})

with the whole instrument -- transmit power, antenna gain, receiver noise
figure, pulse length, number of samples -- collapsed into the single intercept
``c``. Everything in this module exists to measure ``c`` (and, optionally, to
check that the slope really is 20 dB per decade of range) from the volume in
hand, rather than from a data sheet the file does not carry.

Three estimators, because radars deliver three different kinds of file
---------------------------------------------------------------------
``method="snr"``
    Whenever the volume carries a signal-to-noise field on the same scale as
    its reflectivity. Then no population statistics are needed at all: by
    definition :math:`Z - \mathrm{SNR}` *is* the reflectivity a gate would
    report at 0 dB SNR, gate by gate, whether or not that gate holds weather.
    The estimate is a per-range median of a quantity whose within-range spread
    was measured at 0.03-0.25 dB on a BNF C-SAPR2 volume, and exactly 0.00 dB
    on a differently processed one. This is the preferred route, and ``auto``
    takes it when an SNR field exists.

``method="noise"``
    For volumes that report a value at every gate whether or not anything was
    detected -- ARM C-SAPR2 and most research radars. Nine tenths of such a
    volume *is* the noise floor, so the noise floor is the **mode** of the
    reflectivity distribution once the :math:`r^2` trend is removed. On a BNF
    C-SAPR2 PPI volume this recovers a floor constant to within one 0.5 dB
    histogram bin over 0-113 km.

``method="censor"``
    For volumes that mask everything below a detection threshold -- WSR-88D
    Level II, and any product delivered thresholded. There is no noise
    population to find a mode in; what is measurable is the *lower envelope*
    of the surviving data, which traces the censoring decision. A low
    percentile of the valid gates at each range is used instead.

Why the mode and not a mean, median or minimum
---------------------------------------------
The mode is the only one of the four that survives the company it keeps. A
volume's gates are a mixture: a large noise population, a weather tail
reaching 60 dBZ above it, clutter, and -- on CfRadial files with a floor
sentinel -- a spike of fill values tens of dB below it. A mean is dragged by
the weather tail, a minimum is pinned to the sentinel, and a median moves with
the *proportion* of weather in the sweep, which is a property of the storm and
not of the receiver. Measured on a BNF C-SAPR2 PPI volume, the per-range
median of the gates a gate-ID classifier called ``no_scatter`` scattered with
an RMS of 7.4 dB about the fitted power law; the mode of the same population
held to 0.2 dB.

The same argument rules out selecting the noise population with a hard SNR
threshold and then averaging it. At a fixed range, reflectivity and SNR differ
by a constant, so ``snr < -3 dB`` is a hard cut on reflectivity: it keeps the
lower tail of the noise distribution and discards its peak. On the volume
above, that route reported a floor 5 dB below the mode -- an estimate that is
precise, stable, reproducible and wrong. Where an SNR field selects gates for
``method="noise"`` it is therefore used generously, only to exclude obvious
signal, and the mode does the work. Using SNR *arithmetically* rather than as
a threshold -- ``method="snr"`` -- avoids the problem entirely.

Why the mode is not enough on its own
-------------------------------------
The mode assumes the noise floor is the dominant population, and on some
processing levels it is not. A 2025 BNF QLCS volume, delivered with a censor
mask and noise-subtracted power, gave a mode that wandered from -34.6 dBZ at
1 km equivalent near the radar to -45.7 at 70-80 km and back to -40.7 at
110 km: an 11 dB span, reported by ``mode_fraction_median`` as a peak holding
only 11% of the samples. Noise subtraction leaves near-zero-power gates
scattered over tens of dB, so there is no sharp peak to find. On the same
volume the SNR route returned a floor with zero within-range spread. Hence
the ordering: use the SNR arithmetic where it exists, the mode where it does
not, and the censoring envelope where neither is available.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "REFERENCE_RANGE_M",
    "THEORETICAL_SLOPE_DB_PER_DECADE",
    "MdsProfile",
    "detection_floor_by_range",
    "detrend_range",
    "estimate_profile",
    "find_sentinel_values",
    "fit_power_law",
    "noise_floor_by_range",
    "snr_floor_by_range",
]

#: Range at which the fitted intercept is quoted. 1 km is conventional for
#: radar sensitivity ("-40 dBZ at 1 km") and keeps the intercept a number a
#: radar engineer recognises.
REFERENCE_RANGE_M = 1000.0

#: The :math:`r^2` law in decibels: 20 dB per decade of range. Held fixed by
#: default because it is physics, not a free parameter; ``fit_slope=True``
#: releases it and the fitted value becomes a diagnostic of whether the
#: volume's calibration really applies a clean range correction.
THEORETICAL_SLOPE_DB_PER_DECADE = 20.0


def _smallest_positive(range_m):
    """Smallest positive range on a grid, or None if there is none.

    Used as the default inner limit for evaluation: a CfRadial volume may
    report its first gate at exactly 0 m, where the power law is -inf.
    """
    values = np.asarray(range_m, dtype="f8")
    positive = values[np.isfinite(values) & (values > 0)]
    return float(positive.min()) if positive.size else None


@contextlib.contextmanager
def _quiet_nan_warnings():
    """Silence "all-NaN slice" warnings from range gates with no samples.

    Those columns are the expected result -- a range where nothing usable was
    measured -- and they are already reported as NaN estimates with a zero
    sample count, so the warning carries no information the caller cannot see.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        yield


def detrend_range(dbz, range_m, slope_db_per_decade=None, reference_range_m=None):
    """Remove the range-squared trend from reflectivity.

    Parameters
    ----------
    dbz : ndarray
        Reflectivity in dBZ, shaped ``(nrays, ngates)`` or ``(ngates,)``.
    range_m : ndarray
        Gate range in metres, shaped ``(ngates,)``.
    slope_db_per_decade : float, optional
        Trend to remove. Defaults to :data:`THEORETICAL_SLOPE_DB_PER_DECADE`.
    reference_range_m : float, optional
        Range the detrended values refer to. Defaults to
        :data:`REFERENCE_RANGE_M`.

    Returns
    -------
    residual : ndarray
        ``dbz`` minus the trend, same shape as ``dbz``. A gate at the noise
        floor has the same residual at every range, which is what makes the
        floor estimable by pooling ranges.
    trend : ndarray
        The trend that was removed, shaped ``(ngates,)``.

    Examples
    --------
    A floor of -40 dBZ at 1 km, sampled at 1, 10 and 100 km, detrends flat::

        rng = np.array([1e3, 1e4, 1e5])
        dbz = np.array([-40.0, -20.0, 0.0])
        residual, _ = detrend_range(dbz, rng)
        # residual -> [-40, -40, -40]
    """
    slope = (
        THEORETICAL_SLOPE_DB_PER_DECADE
        if slope_db_per_decade is None
        else float(slope_db_per_decade)
    )
    ref = REFERENCE_RANGE_M if reference_range_m is None else float(reference_range_m)
    range_m = np.asarray(range_m, dtype="f8")
    # Gates at or below zero range exist in a few files as a metadata
    # artefact; clipping keeps the logarithm finite instead of poisoning the
    # whole ray with NaN.
    trend = slope * np.log10(np.maximum(range_m, 1.0) / ref)
    dbz = np.asarray(dbz, dtype="f8")
    return dbz - (trend if dbz.ndim == 1 else trend[None, :]), trend


def find_sentinel_values(dbz, min_fraction=0.002, below_dbz=-40.0, max_values=3):
    """Find fill values masquerading as very low reflectivity.

    CfRadial files written by some processing chains store a floor sentinel --
    a single repeated value such as -80 dBZ -- at gates beyond the maximum
    unambiguous range, or where a moment was not computed. It is a flag, not a
    measurement, and it is far enough below the real floor to wreck a
    quantile estimator.

    Parameters
    ----------
    dbz : ndarray
        Reflectivity in dBZ.
    min_fraction : float, optional
        A value must account for at least this fraction of finite gates to be
        called a sentinel.
    below_dbz : float, optional
        Only values at or below this are considered. A repeated value inside
        the plausible reflectivity range is far more likely to be a
        quantisation level of a real moment than a fill flag.
    max_values : int, optional
        If more than this many candidates are found, none are returned. A fill
        flag is one value, occasionally two; a long list of repeated values
        means something else is going on -- most usefully, a volume this
        module has *already* filled, where every range gate holds its own
        exact MDS value repeated across rays. Dropping those as sentinels
        would erase the data instead of measuring it.

    Returns
    -------
    list of float
        Sentinel values found, ascending. Usually empty or one element.

    Notes
    -----
    Detection is by exact value repetition, which is what a fill flag looks
    like and what a physical measurement does not.
    """
    dbz = np.asarray(dbz, dtype="f8")
    finite = dbz[np.isfinite(dbz)]
    if finite.size == 0:
        return []
    low = finite[finite <= float(below_dbz)]
    if low.size == 0:
        return []
    values, counts = np.unique(low, return_counts=True)
    keep = counts >= max(1.0, float(min_fraction) * finite.size)
    found = [float(v) for v in values[keep]]
    if max_values is not None and len(found) > int(max_values):
        return []
    return found


def _column_mode(residual, bin_db, min_samples, refine_bins=1.5):
    """Per-range mode of a detrended distribution, with sub-bin refinement.

    The histogram peak locates the mode to within one bin; the median of the
    samples inside a window around that peak then places it more finely
    without letting the weather tail back in. Returns ``(mode, n, frac)``,
    where ``frac`` is the share of each column's samples inside that window --
    a check that the peak found is a dominant population and not the tallest
    ripple in a broad distribution.
    """
    residual = np.asarray(residual, dtype="f8")
    if residual.ndim == 1:
        residual = residual[None, :]
    ngates = residual.shape[1]
    counts = np.isfinite(residual).sum(axis=0)
    mode = np.full(ngates, np.nan)
    frac = np.full(ngates, np.nan)

    usable = counts >= int(min_samples)
    if not usable.any():
        return mode, counts, frac

    pool = residual[:, usable]
    finite_pool = pool[np.isfinite(pool)]
    lo, hi = np.percentile(finite_pool, [0.05, 99.95])
    edges = (
        np.arange(lo, hi + 2 * bin_db, bin_db)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo
        else np.empty(0)
    )
    if edges.size < 3:
        # A degenerate spread means every usable sample already agrees -- a
        # volume this module has filled, or a synthetic one. There is no mode
        # to search for, and the column median is exact.
        with _quiet_nan_warnings():
            median = np.nanmedian(
                np.where(np.isfinite(residual), residual, np.nan), axis=0
            )
        mode[usable] = median[usable]
        frac[usable] = 1.0
        return mode, counts, frac
    centres = 0.5 * (edges[:-1] + edges[1:])

    # One vectorised pass: a bin index per sample, then a per-column bincount
    # accumulated with np.add.at. Cheaper than a Python loop over ranges, and
    # indifferent to how unevenly the samples are spread between them.
    nbins = centres.size
    col = np.broadcast_to(np.arange(ngates)[None, :], residual.shape)
    idx = np.digitize(residual, edges) - 1
    ok = np.isfinite(residual) & (idx >= 0) & (idx < nbins)
    hist = np.zeros((ngates, nbins), dtype="i8")
    np.add.at(hist, (col[ok], idx[ok]), 1)

    peak = centres[np.argmax(hist, axis=1)]
    window = refine_bins * bin_db
    near = np.abs(residual - peak[None, :]) <= window
    inside = np.where(near & np.isfinite(residual), residual, np.nan)
    with _quiet_nan_warnings():
        refined = np.nanmedian(inside, axis=0)
    n_inside = np.isfinite(inside).sum(axis=0)

    mode[usable] = np.where(n_inside[usable] > 0, refined[usable], peak[usable])
    with np.errstate(invalid="ignore", divide="ignore"):
        frac[usable] = n_inside[usable] / np.maximum(counts[usable], 1)
    return mode, counts, frac


def noise_floor_by_range(
    dbz,
    range_m,
    mask=None,
    bin_db=0.5,
    min_samples=30,
    slope_db_per_decade=None,
    reference_range_m=None,
    sentinels=None,
):
    """Per-range noise-equivalent reflectivity, from the mode of no-return gates.

    Parameters
    ----------
    dbz : ndarray
        Reflectivity in dBZ, shaped ``(nrays, ngates)``.
    range_m : ndarray
        Gate range in metres, shaped ``(ngates,)``.
    mask : ndarray of bool, optional
        Gates to use -- the ones believed to hold no detected scatterer.
        Default uses every finite gate, which is sound on a volume where
        non-detections dominate, and is reported as such by the caller.
    bin_db : float, optional
        Histogram bin width for the mode search.
    min_samples : int, optional
        Ranges with fewer usable gates than this return NaN.
    slope_db_per_decade, reference_range_m : float, optional
        Passed to :func:`detrend_range`.
    sentinels : sequence of float, optional
        Values to drop as fill flags. Default detects them with
        :func:`find_sentinel_values`.

    Returns
    -------
    dict
        ``sample_dbz`` (per-range estimate, dBZ), ``n_samples``,
        ``mode_fraction`` (share of samples in the mode window, per range) and
        ``sentinels`` (values dropped).

    Notes
    -----
    ``mode_fraction`` is the honesty check. On a volume whose noise floor is
    genuinely the dominant population it runs high; if it is small, the
    "mode" is a ripple and the estimate should not be trusted as a floor.
    """
    dbz = np.asarray(dbz, dtype="f8")
    if dbz.ndim == 1:
        dbz = dbz[None, :]
    values = np.array(dbz, dtype="f8", copy=True)

    if sentinels is None:
        sentinels = find_sentinel_values(values)
    for sentinel in sentinels:
        values[values == sentinel] = np.nan

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match reflectivity "
                f"shape {values.shape}"
            )
        values = np.where(mask, values, np.nan)

    residual, trend = detrend_range(
        values,
        range_m,
        slope_db_per_decade=slope_db_per_decade,
        reference_range_m=reference_range_m,
    )
    mode, counts, frac = _column_mode(residual, float(bin_db), int(min_samples))
    return {
        "sample_dbz": mode + trend,
        "n_samples": counts,
        "mode_fraction": frac,
        "sentinels": list(sentinels),
    }


def snr_floor_by_range(
    dbz,
    snr,
    range_m,
    min_samples=30,
    slope_db_per_decade=None,
    reference_range_m=None,
    sentinels=None,
):
    r"""Per-range noise-equivalent reflectivity from ``Z`` minus ``SNR``.

    The exact route, where the volume supports it. Reflectivity and SNR share
    a receiver: for a gate whose signal power is :math:`P` against noise
    :math:`N`,

    .. math::

        Z = C + 10\log_{10} P + 20\log_{10} r, \qquad
        \mathrm{SNR} = 10\log_{10}(P / N)

    so :math:`Z - \mathrm{SNR} = C + 10\log_{10} N + 20\log_{10} r`, which
    is the reflectivity the gate would report at 0 dB SNR -- the noise-
    equivalent reflectivity, with the signal cancelled out. It does not matter
    whether the gate held weather, clutter or nothing, so no population has to
    be identified and no mixture has to be argued about.

    Parameters
    ----------
    dbz : ndarray
        Reflectivity in dBZ, shaped ``(nrays, ngates)``.
    snr : ndarray
        Signal-to-noise ratio in dB, same shape, on the same scale as ``dbz``.
    range_m : ndarray
        Gate range in metres.
    min_samples : int, optional
        Ranges with fewer usable gate pairs than this return NaN.
    slope_db_per_decade, reference_range_m : float, optional
        Passed to :func:`detrend_range`.
    sentinels : sequence of float, optional
        Reflectivity fill values to drop. Default detects them.

    Returns
    -------
    dict
        ``sample_dbz``, ``n_samples``, ``spread_db`` (per-range interquartile
        range of the difference -- the check that the two fields really do
        share a scale) and ``sentinels``.

    Notes
    -----
    The one way this can mislead is a file whose SNR was computed against a
    *different* reflectivity field than the one being measured -- corrected
    versus uncorrected, or H versus V. Then the difference carries that offset
    too. ``spread_db`` will not catch it, since the offset is constant;
    :func:`estimate_profile` cross-checks against the mode instead.
    """
    dbz = np.asarray(dbz, dtype="f8")
    snr = np.asarray(snr, dtype="f8")
    if dbz.ndim == 1:
        dbz = dbz[None, :]
    if snr.ndim == 1:
        snr = snr[None, :]
    if snr.shape != dbz.shape:
        raise ValueError(
            f"snr shape {snr.shape} does not match reflectivity {dbz.shape}"
        )
    values = np.array(dbz, dtype="f8", copy=True)
    if sentinels is None:
        sentinels = find_sentinel_values(values)
    for sentinel in sentinels:
        values[values == sentinel] = np.nan

    difference = values - snr
    residual, trend = detrend_range(
        difference,
        range_m,
        slope_db_per_decade=slope_db_per_decade,
        reference_range_m=reference_range_m,
    )
    counts = np.isfinite(residual).sum(axis=0)
    with _quiet_nan_warnings():
        median = np.nanmedian(residual, axis=0)
        upper = np.nanpercentile(residual, 75, axis=0)
        lower = np.nanpercentile(residual, 25, axis=0)
    usable = counts >= int(min_samples)
    return {
        "sample_dbz": np.where(usable, median + trend, np.nan),
        "n_samples": counts,
        "spread_db": np.where(usable, upper - lower, np.nan),
        "sentinels": list(sentinels),
    }


def detection_floor_by_range(
    dbz,
    range_m,
    quantile=5.0,
    min_samples=30,
    sentinels=None,
):
    """Per-range detection floor, from the lower envelope of the valid gates.

    For thresholded volumes, where everything below the detection threshold
    was masked before delivery. The threshold itself is not in the file, but
    the weakest echoes that survived it are, so a low percentile of the valid
    gates at each range traces it.

    Parameters
    ----------
    dbz : ndarray
        Reflectivity in dBZ, shaped ``(nrays, ngates)``; non-detections must
        be NaN (the callers here fill Py-ART masked arrays with NaN).
    range_m : ndarray
        Gate range in metres.
    quantile : float, optional
        Percentile of the valid distribution to take, 0-100. Higher is more
        conservative -- further above the true threshold, but less sensitive
        to the handful of gates that scraped past it.
    min_samples : int, optional
        Ranges with fewer valid gates than this return NaN.
    sentinels : sequence of float, optional
        Values to drop as fill flags. Default detects them.

    Returns
    -------
    dict
        ``sample_dbz``, ``n_samples``, ``sentinels``.

    Notes
    -----
    This estimates where the *processing* stopped reporting, which sits above
    the receiver's own floor by whatever SNR margin the threshold used. That
    is usually what a gridding floor should be anyway: it is the weakest echo
    the volume could have contained.
    """
    dbz = np.asarray(dbz, dtype="f8")
    if dbz.ndim == 1:
        dbz = dbz[None, :]
    values = np.array(dbz, dtype="f8", copy=True)
    if sentinels is None:
        sentinels = find_sentinel_values(values)
    for sentinel in sentinels:
        values[values == sentinel] = np.nan

    counts = np.isfinite(values).sum(axis=0)
    with _quiet_nan_warnings():
        sample = np.nanpercentile(values, float(quantile), axis=0)
    sample = np.where(counts >= int(min_samples), sample, np.nan)
    return {
        "sample_dbz": sample,
        "n_samples": counts,
        "sentinels": list(sentinels),
    }


def fit_power_law(
    sample_dbz,
    range_m,
    n_samples=None,
    fit_slope=False,
    min_samples=30,
    min_range_m=3000.0,
    reference_range_m=None,
    trim_sigma=5.0,
):
    """Fit ``dBZ = intercept + slope * log10(r / r_ref)`` to a range profile.

    Parameters
    ----------
    sample_dbz : ndarray
        Per-range floor estimate, NaN where unmeasured.
    range_m : ndarray
        Gate range in metres.
    n_samples : ndarray, optional
        Gates behind each estimate; ranges below ``min_samples`` are excluded.
    fit_slope : bool, optional
        Release the slope. Default holds it at
        :data:`THEORETICAL_SLOPE_DB_PER_DECADE` and takes the intercept as the
        median offset: with the slope fixed the fit is a location estimate,
        and the median is the robust one.
    min_samples : int, optional
        Minimum gates behind a range for it to enter the fit.
    min_range_m : float, optional
        Ranges closer than this are excluded. The first gates carry transmit
        leakage, blanking and near-field effects that are not the far-field
        :math:`r^2` law.
    reference_range_m : float, optional
        Range the intercept refers to.
    trim_sigma : float or None, optional
        Reject ranges whose residual exceeds this many robust standard
        deviations (MAD-scaled) and refit once. ``None`` disables it.

        This is not cosmetic. Measured on a BNF C-SAPR2 PPI volume, the
        no-return population inside about 4.5 km sits 10-20 dB above the
        power law -- close-range ground echo and sidelobe clutter that a
        classifier still calls ``no_scatter`` because it carries no coherent
        Doppler signature. Beyond that the per-range estimates track the law
        to 0.5 dB. A fixed near-range cut would be wrong for the next radar,
        whose contaminated span is a different length; rejecting on the
        residual finds it wherever it is, and ``min_valid_range_m`` reports
        where the floor became measurable.

    Returns
    -------
    dict
        ``intercept_dbz``, ``slope_db_per_decade``, ``residual_rms_db``,
        ``residual_mad_db``, ``n_ranges`` (ranges entering the final fit),
        ``n_trimmed``, ``min_valid_range_m``, ``kept`` (boolean over the input
        ranges) and ``fit_slope``.

    Raises
    ------
    ValueError
        If no range qualifies. A silent NaN here would propagate into every
        replaced gate.
    """
    ref = REFERENCE_RANGE_M if reference_range_m is None else float(reference_range_m)
    range_m = np.asarray(range_m, dtype="f8")
    sample_dbz = np.asarray(sample_dbz, dtype="f8")
    keep = np.isfinite(sample_dbz) & (range_m >= float(min_range_m))
    if n_samples is not None:
        keep &= np.asarray(n_samples) >= int(min_samples)
    if not keep.any():
        raise ValueError(
            "no range gate carries enough samples to estimate the minimum "
            "detectable signal; loosen min_samples/min_range_m or supply a "
            "profile explicitly"
        )

    decades_all = np.log10(np.maximum(range_m, 1.0) / ref)

    def _fit(selected):
        decades = decades_all[selected]
        y = sample_dbz[selected]
        if fit_slope and selected.sum() >= 3:
            from scipy import stats

            result = stats.theilslopes(y, decades)
            return float(result[0]), float(result[1])
        slope = THEORETICAL_SLOPE_DB_PER_DECADE
        return slope, float(np.median(y - slope * decades))

    slope, intercept = _fit(keep)
    n_trimmed = 0
    if trim_sigma is not None:
        residual = sample_dbz - (intercept + slope * decades_all)
        scale = 1.4826 * np.median(np.abs(residual[keep]))
        # A degenerate scale means the surviving estimates agree exactly, in
        # which case there is nothing to trim against.
        if scale > 0:
            trimmed = keep & (np.abs(residual) <= float(trim_sigma) * scale)
            if trimmed.any():
                n_trimmed = int(keep.sum() - trimmed.sum())
                keep = trimmed
                slope, intercept = _fit(keep)

    residual = sample_dbz[keep] - (intercept + slope * decades_all[keep])
    return {
        "intercept_dbz": float(intercept),
        "slope_db_per_decade": float(slope),
        "residual_rms_db": float(np.sqrt(np.mean(residual**2))),
        "residual_mad_db": float(1.4826 * np.median(np.abs(residual))),
        "n_ranges": int(keep.sum()),
        "n_trimmed": n_trimmed,
        "min_valid_range_m": float(np.min(range_m[keep])),
        "kept": keep,
        "fit_slope": bool(fit_slope),
    }


@dataclass
class MdsProfile:
    """A minimum-detectable-signal curve, and how it was obtained.

    Attributes
    ----------
    range_m : ndarray
        Ranges the profile was estimated on.
    intercept_dbz : float
        Fitted MDS at ``reference_range_m``, before ``margin_db``.
    slope_db_per_decade : float
        Fitted or assumed range dependence.
    reference_range_m : float
        Range the intercept refers to.
    margin_db : float
        Offset added when the profile is evaluated. Positive is more
        conservative (a higher floor).
    cap_dbz : float or None
        Ceiling applied on evaluation. At long range an honest MDS exceeds
        light-rain reflectivities, and filling non-detections with it writes
        phantom echo into a grid; a cap trades that bias for an
        understatement of the sensitivity.
    model : {'power_law', 'empirical', 'fixed'}
        ``power_law`` evaluates the fit everywhere. ``empirical`` follows the
        measured per-range profile where it exists (smoothed) and falls back
        to the fit elsewhere. ``fixed`` was supplied, not measured.
    method : {'snr', 'noise', 'censor', 'fixed'}
        Which estimator produced ``sample_dbz``.
    sample_dbz : ndarray
        The raw per-range estimate, NaN where unmeasured. Keeping it makes the
        fit auditable, and is what ``model='empirical'`` interpolates.
    n_samples : ndarray
        Gates behind each per-range estimate.
    residual_rms_db : float
        RMS of the per-range estimates about the fitted power law. The single
        most useful number here: a few tenths of a dB means the volume really
        does obey one power law, several dB means it does not.
    floor_range_m : float or None
        Innermost range the curve is extrapolated to; queries inside it are
        evaluated here instead. Defaults to one gate spacing, which removes
        the divergence at a range-0 first gate without altering the law
        anywhere a gate is really measured. ``None`` extrapolates all the way
        in, which at a range-0 gate means -97 dBZ.
    meta : dict
        Provenance: field used, mask source, sentinels dropped, mode fraction,
        and any warnings.
    """

    range_m: np.ndarray
    intercept_dbz: float
    slope_db_per_decade: float = THEORETICAL_SLOPE_DB_PER_DECADE
    reference_range_m: float = REFERENCE_RANGE_M
    margin_db: float = 0.0
    cap_dbz: float | None = None
    model: str = "power_law"
    method: str = "noise"
    sample_dbz: np.ndarray | None = None
    n_samples: np.ndarray | None = None
    residual_rms_db: float = float("nan")
    floor_range_m: float | None = None
    meta: dict = field(default_factory=dict)

    def evaluate(self, range_m):
        r"""MDS in dBZ at arbitrary ranges.

        Parameters
        ----------
        range_m : array_like
            Ranges in metres.

        Returns
        -------
        ndarray
            MDS in dBZ, including ``margin_db`` and any ``cap_dbz``.

        Notes
        -----
        Ranges inside ``floor_range_m`` are evaluated *at* ``floor_range_m``
        rather than extrapolated, because the power law diverges as
        :math:`r \to 0`: a C-SAPR2 volume whose first gate is reported at 0 m
        otherwise takes a fill of -97 dBZ there. The default is one gate
        spacing, which removes the divergence and leaves the law untouched
        everywhere else.

        A caller who wants the innermost gates treated more conservatively can
        set ``floor_range_m = profile.meta["fit_min_valid_range_m"]``, the
        range from which the floor was actually measurable (3-6 km on C-SAPR2,
        inside which ground echo contaminated the estimate). That holds the
        fill flat across the near field instead of following the law down --
        about 10 dB higher at 1 km on a C-SAPR2 volume. Which is right depends
        on what the fill feeds: the law is the better physical estimate, the
        flat hold is the safer one if near-range fill will be integrated.

        Examples
        --------
        ::

            profile.evaluate([1e3, 1e4, 1e5])
        """
        query = np.asarray(range_m, dtype="f8")
        inner = self.floor_range_m
        if inner is not None and np.isfinite(inner):
            query = np.maximum(query, float(inner))
        law = self.intercept_dbz + self.slope_db_per_decade * np.log10(
            np.maximum(query, 1.0) / self.reference_range_m
        )
        if self.model == "empirical" and self.sample_dbz is not None:
            smoothed = self.meta.get("smoothed_dbz")
            if smoothed is None:
                smoothed = self.sample_dbz
            smoothed = np.asarray(smoothed, dtype="f8")
            known = np.isfinite(smoothed)
            if known.sum() >= 2:
                # Interpolation only inside the measured span; outside it the
                # power law extrapolates, which is why one was fitted.
                base = np.asarray(self.range_m, dtype="f8")[known]
                interpolated = np.interp(
                    query, base, smoothed[known], left=np.nan, right=np.nan
                )
                law = np.where(np.isfinite(interpolated), interpolated, law)
        out = law + float(self.margin_db)
        if self.cap_dbz is not None:
            out = np.minimum(out, float(self.cap_dbz))
        return out

    def to_dict(self):
        """Return a JSON-friendly summary, without the per-range arrays.

        Returns
        -------
        dict
            Scalars and metadata, suitable for a log line or a field comment.
        """
        return {
            "intercept_dbz": self.intercept_dbz,
            "slope_db_per_decade": self.slope_db_per_decade,
            "reference_range_m": self.reference_range_m,
            "margin_db": self.margin_db,
            "cap_dbz": self.cap_dbz,
            "model": self.model,
            "method": self.method,
            "residual_rms_db": self.residual_rms_db,
            "meta": {k: v for k, v in self.meta.items() if not hasattr(v, "shape")},
        }

    def __repr__(self):
        return (
            f"MdsProfile({self.method}/{self.model}: "
            f"{self.intercept_dbz:+.2f} dBZ at "
            f"{self.reference_range_m / 1000.0:g} km, "
            f"{self.slope_db_per_decade:.1f} dB/decade, "
            f"resid {self.residual_rms_db:.2f} dB)"
        )


def estimate_profile(
    dbz,
    range_m,
    mask=None,
    snr=None,
    method="auto",
    model="power_law",
    bin_db=0.5,
    quantile=5.0,
    min_samples=30,
    min_range_m=3000.0,
    fit_slope=False,
    margin_db=0.0,
    cap_dbz=None,
    smooth_gates=9,
    censor_fraction=0.5,
    reference_range_m=None,
    trim_sigma=5.0,
    cross_check=True,
    meta=None,
):
    """Estimate an :class:`MdsProfile` from a reflectivity array.

    The array-level entry point: no radar object, no object flavour, no field
    name resolution. :func:`radar_palette.mdsreplace.mds_profile` wraps this
    for volumes.

    Parameters
    ----------
    dbz : ndarray
        Reflectivity in dBZ, ``(nrays, ngates)``, non-detections NaN.
    range_m : ndarray
        Gate range in metres, ``(ngates,)``.
    mask : ndarray of bool, optional
        No-return gates, for ``method='noise'``.
    snr : ndarray, optional
        Signal-to-noise ratio in dB, same shape as ``dbz``, for
        ``method='snr'``.
    method : {'auto', 'snr', 'noise', 'censor'}, optional
        ``auto`` takes ``snr`` when an SNR array is supplied, otherwise
        ``censor`` when at least ``censor_fraction`` of gates are missing --
        the signature of a thresholded product -- and ``noise`` otherwise.
    model : {'power_law', 'empirical'}, optional
        How the estimate is evaluated away from the ranges it was measured at.
    bin_db, quantile, min_samples, min_range_m, fit_slope
        Estimator and fit controls; see :func:`noise_floor_by_range`,
        :func:`detection_floor_by_range` and :func:`fit_power_law`.
    margin_db : float, optional
        Added on evaluation. Positive keeps the fill strictly above the floor,
        negative strictly below. This is also where a detection threshold
        belongs: ``method='snr'`` returns the floor at 0 dB SNR, so
        ``margin_db=3.0`` asks for the reflectivity detectable at 3 dB SNR.
    cap_dbz : float, optional
        Ceiling on the evaluated MDS.
    smooth_gates : int, optional
        Running-median width for ``model='empirical'``, in range gates.
    censor_fraction : float, optional
        Missing-gate fraction above which ``auto`` chooses ``censor``.
    reference_range_m : float, optional
        Range the intercept is quoted at.
    trim_sigma : float or None, optional
        Robust rejection of range gates inconsistent with one power law; see
        :func:`fit_power_law`.
    cross_check : bool, optional
        For ``method='snr'``, also estimate the floor from the noise mode and
        record it as ``meta['cross_check_noise_mode_dbz']``. Cheap, and the
        only way to notice an SNR field computed against a different moment.
    meta : dict, optional
        Provenance to carry through onto the profile.

    Returns
    -------
    MdsProfile

    Examples
    --------
    ::

        profile = estimate_profile(dbz, range_m, method="noise")
        profile.intercept_dbz
    """
    dbz = np.asarray(dbz, dtype="f8")
    if dbz.ndim == 1:
        dbz = dbz[None, :]
    range_m = np.asarray(range_m, dtype="f8")
    if dbz.shape[1] != range_m.size:
        raise ValueError(
            f"reflectivity has {dbz.shape[1]} gates but range has {range_m.size}"
        )
    meta = dict(meta or {})
    warnings_out = list(meta.get("warnings", []))

    missing_fraction = float(np.mean(~np.isfinite(dbz)))
    if method == "auto":
        if snr is not None:
            method = "snr"
        else:
            method = "censor" if missing_fraction >= float(censor_fraction) else "noise"
        meta["method_auto"] = True
    meta["missing_fraction"] = missing_fraction

    if method == "snr":
        if snr is None:
            raise ValueError("method='snr' needs an snr array")
        estimate = snr_floor_by_range(
            dbz,
            snr,
            range_m,
            min_samples=min_samples,
            reference_range_m=reference_range_m,
        )
        spread = estimate["spread_db"]
        finite_spread = spread[np.isfinite(spread)]
        median_spread = (
            float(np.median(finite_spread)) if finite_spread.size else float("nan")
        )
        meta["snr_spread_db_median"] = median_spread
        if np.isfinite(median_spread) and median_spread < 0.01:
            # Z - SNR with no gate-to-gate scatter at all means SNR was
            # *derived* from Z by subtracting a modelled noise power, not
            # measured alongside it. Measured on 2025-era BNF C-SAPR2 volumes:
            # a spread of 4e-4 dB, and the same intercept to 0.001 dB on files
            # months apart. The number returned is then the noise power that
            # processing assumed -- still the right value to fill with, since
            # it is the floor the moments were computed against, but not an
            # independent measurement of the receiver, and not a check on it.
            warnings_out.append(
                f"reflectivity minus SNR is deterministic to {median_spread:.1e} dB: "
                "SNR appears derived from reflectivity, so this floor is the "
                "processing's assumed noise power, not an independent measurement"
            )
        if np.isfinite(median_spread) and median_spread > 3.0:
            warnings_out.append(
                f"reflectivity minus SNR varies by {median_spread:.1f} dB "
                "within a range gate; the two fields may not share a scale"
            )
        if cross_check:
            # An offset between the two fields -- an SNR computed against a
            # different moment -- is invisible in the spread but shows up
            # against an independent estimator. The mode sits a little above
            # the 0 dB SNR floor (the noise distribution peaks above its own
            # zero-SNR point), so only a gross disagreement is reported.
            try:
                comparison = noise_floor_by_range(
                    dbz,
                    range_m,
                    mask=mask,
                    bin_db=bin_db,
                    min_samples=min_samples,
                    reference_range_m=reference_range_m,
                )
                other = fit_power_law(
                    comparison["sample_dbz"],
                    range_m,
                    n_samples=comparison["n_samples"],
                    min_samples=min_samples,
                    min_range_m=min_range_m,
                    reference_range_m=reference_range_m,
                    trim_sigma=trim_sigma,
                )
            except ValueError:
                other = None
            if other is not None:
                meta["cross_check_noise_mode_dbz"] = other["intercept_dbz"]
                meta["cross_check_mode_fraction"] = float(
                    np.nanmedian(comparison["mode_fraction"])
                )
    elif method == "noise":
        estimate = noise_floor_by_range(
            dbz,
            range_m,
            mask=mask,
            bin_db=bin_db,
            min_samples=min_samples,
            reference_range_m=reference_range_m,
        )
        frac = estimate["mode_fraction"]
        finite_frac = frac[np.isfinite(frac)]
        median_frac = (
            float(np.median(finite_frac)) if finite_frac.size else float("nan")
        )
        meta["mode_fraction_median"] = median_frac
        if np.isfinite(median_frac) and median_frac < 0.2:
            warnings_out.append(
                f"mode holds only {median_frac:.0%} of samples: the dominant "
                "population may not be the noise floor"
            )
    elif method == "censor":
        estimate = detection_floor_by_range(
            dbz,
            range_m,
            quantile=quantile,
            min_samples=min_samples,
        )
        meta["quantile"] = float(quantile)
    else:
        raise ValueError(
            f"unknown method {method!r}; expected snr, noise, censor or auto"
        )

    meta["sentinels"] = estimate["sentinels"]
    fit = fit_power_law(
        estimate["sample_dbz"],
        range_m,
        n_samples=estimate["n_samples"],
        fit_slope=fit_slope,
        min_samples=min_samples,
        min_range_m=min_range_m,
        reference_range_m=reference_range_m,
        trim_sigma=trim_sigma,
    )
    meta.update({f"fit_{k}": v for k, v in fit.items() if k != "intercept_dbz"})

    other_intercept = meta.get("cross_check_noise_mode_dbz")
    if other_intercept is not None:
        gap = float(other_intercept - fit["intercept_dbz"])
        meta["cross_check_delta_db"] = gap
        # The mode is expected 1-2.5 dB above the 0 dB SNR floor (a noise
        # distribution peaks above its own zero-SNR point; 1.0-1.3 dB measured
        # on 2026 BNF volumes, 2.3 dB on another). A gap far outside that says
        # one of the two is not measuring the receiver -- on 2025 BNF volumes
        # the mode ran 5-10 dB low against an SNR route whose within-range
        # spread was 0.0004 dB. Which one is wrong is not decidable here, so
        # both numbers are reported rather than one being silently preferred.
        if abs(gap) > 4.0:
            fraction = meta.get("cross_check_mode_fraction")
            note = (
                f"the noise mode puts the floor {gap:+.1f} dB from the SNR "
                f"route (expect about +1 to +2.5 dB)"
            )
            if fraction is not None and np.isfinite(fraction):
                note += f"; that mode held {fraction:.0%} of samples"
            warnings_out.append(note)

    if fit["n_trimmed"]:
        warnings_out.append(
            f"{fit['n_trimmed']} range gates rejected as inconsistent with a "
            f"single power law; the floor is measurable from "
            f"{fit['min_valid_range_m'] / 1000.0:.1f} km outwards"
        )

    if model == "empirical":
        # Only ranges the fit kept may drive the empirical curve: a rejected
        # range is one whose sample is not a floor, and interpolating it would
        # write that contamination straight into the fill values.
        usable = np.where(fit["kept"], estimate["sample_dbz"], np.nan)
        meta["smoothed_dbz"] = _running_median(usable, int(smooth_gates))
    elif model != "power_law":
        raise ValueError(f"unknown model {model!r}; expected power_law or empirical")

    meta["warnings"] = warnings_out
    return MdsProfile(
        range_m=range_m,
        intercept_dbz=fit["intercept_dbz"],
        slope_db_per_decade=fit["slope_db_per_decade"],
        reference_range_m=(
            REFERENCE_RANGE_M if reference_range_m is None else float(reference_range_m)
        ),
        margin_db=float(margin_db),
        cap_dbz=None if cap_dbz is None else float(cap_dbz),
        model=model,
        method=method,
        sample_dbz=estimate["sample_dbz"],
        n_samples=estimate["n_samples"],
        residual_rms_db=fit["residual_rms_db"],
        floor_range_m=_smallest_positive(range_m),
        meta=meta,
    )


def _running_median(values, width):
    """Take a running median along a 1-D profile, ignoring NaN, keeping length."""
    values = np.asarray(values, dtype="f8")
    width = max(1, int(width))
    if width == 1:
        return values.copy()
    half = width // 2
    out = np.full(values.size, np.nan)
    for i in range(values.size):
        lo = max(0, i - half)
        hi = min(values.size, i + half + 1)
        window = values[lo:hi]
        window = window[np.isfinite(window)]
        if window.size:
            out[i] = np.median(window)
    return out
