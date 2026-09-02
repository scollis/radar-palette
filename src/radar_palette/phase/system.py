r"""System differential phase: per-ray estimation and ray-to-ray consistency.

The system offset :math:`\Phi_{DP}^{sys}` is the differential phase the receive
chain reports for a target at zero range --- an instrumental property of the
radar, not of the atmosphere. It must be removed before the measured phase can
be read as a propagation phase, and before any fold branch can be decided.

Relationship to wradlib
-----------------------
The three estimators here follow the algorithms in :mod:`wradlib.dp`
(``system_phidp_block`` / ``_first`` / ``_hist``), read from the wradlib 2.9.5
source, and reproduce their two-level output: a
**ray-wise** estimate ``sysphi_ray`` and a **sweep-level** estimate ``sysphi``
formed as the median of the ``n_lowest_rays`` smallest ray-wise values.

They are reimplemented natively on plain arrays rather than imported, because
wradlib is not a declared dependency of this package and
``tests/test_package.py`` enforces that every third-party import is declared.
The reimplementation is deliberate and the algorithms are the wradlib ones; it
is not an independent method. Two interface differences:

* wradlib takes an :class:`xarray.DataArray` with a ``range`` dimension and
  returns an :class:`xarray.Dataset`. These take ``(n_rays, n_gates)`` arrays
  and return plain dicts.
* wradlib's aggregation returns NaN when a sweep has fewer than
  ``n_lowest_rays`` finite ray-wise values. That behaviour is kept, because
  quietly aggregating three rays into a sweep constant is how an offset derived
  from almost no data ends up applied to a whole volume.

Sweep-level or per-ray?
-----------------------
This module estimates **both** and applies a **single sweep-level offset by
default** (:func:`apply_system_phidp` with ``per_ray=False``). Three pieces of
evidence support that, and the third also says why the choice is less
consequential than it looks:

1. *The ray-wise estimate has quantifiable noise.* It is a median of
   ``n_eff`` phase samples, so its standard error is approximately
   ``1.253 * sigma_phi / sqrt(n_eff)``. :func:`ray_consistency` computes that
   figure and compares it against the observed robust ray-to-ray spread. When
   the spread is within the expected noise, a per-ray offset is subtracting
   estimator noise from the data --- it makes the field *less* consistent ray
   to ray, not more.
2. *There is no fast physical mechanism.* The system offset lives in the
   transmit/receive hardware. Within one sweep there is nothing that can move
   it from one ray to the next; ray-to-ray structure in the estimate is
   therefore evidence about the *estimator* until proven otherwise.
3. *KDP does not see the difference at all.* KDP is a range derivative, so any
   additive constant applied per ray is annihilated. The sweep-versus-per-ray
   choice affects absolute PhiDP products, the attenuation-correction path, and
   which branch :mod:`radar_palette.phase.unfold` assigns --- never the KDP
   retrieval downstream of it.

Because the per-ray estimate is the evidence for that decision, it is returned
as a first-class output alongside the sweep value and the consistency
diagnostic, not held as an intermediate. A caller whose diagnostic shows real
ray-to-ray structure can pass ``per_ray=True`` and get the per-ray correction.

What the data said, which is not what point 1 predicted
-------------------------------------------------------
The diagnostic **disagrees with the default** on both volumes it has been run
against, and that is recorded here rather than smoothed over. Taking
``sigma_phi = 1.8`` degrees:

===========================  =========  ==========  =============  ========
volume                       estimator  robust SD   expected noise excess
===========================  =========  ==========  =============  ========
BNF C-SAPR2 RHI, 100 m       block      5.18 deg    0.71 deg       7.3
BNF C-SAPR2 RHI, 100 m       first      4.48 deg    0.71 deg       6.3
KHTX WSR-88D PPI, 250 m      block      2.74 deg    1.13 deg       2.4
KHTX WSR-88D PPI, 250 m      first      2.35 deg    0.71 deg       3.3
===========================  =========  ==========  =============  ========

Every row returns ``per_ray_justified = True``. Three things could produce that,
and this diagnostic cannot distinguish them:

* **The estimator samples different ranges on different rays.** ``block`` takes
  the *first* fully valid block, which on a ray with clutter or a data gap in
  the near range sits further out --- where real propagation phase has already
  accumulated. That is a range difference misread as a ray difference, and it is
  a property of the estimator, not the radar.
* **Gates within a ray are correlated,** so ``n_eff`` is smaller than the gate
  count and the expected noise is understated. At BNF the lag-1 autocorrelation
  of measured phase is about 0.5, which for a 10-gate block puts the effective
  count nearer 3 and raises the expected noise by ~1.7x --- moving the BNF
  excess from 7.3 to about 4.2, still above the threshold.
* **Genuine ray-to-ray variation**, for which there is no hardware mechanism
  within a single sweep.

The first is the most likely and the third the least, so the sweep-level default
stands --- but it stands on the *mechanism* argument, not on this diagnostic,
which came out the other way. Read ``excess_ratio`` as an upper bound on real
ray-to-ray inconsistency, not an estimate of it. And note point 3 above: for a
KDP retrieval the question is moot.

What the consistency diagnostic is blind to
-------------------------------------------
It measures *offsets*, not shape: ray-median spread and adjacent-ray jump.
Correlation between adjacent rays is deliberately not used, because a constant
per-ray offset still correlates at ~0.99 and so correlation cannot see the
quantity of interest. The diagnostic is also blind to a *smooth* azimuthal
trend --- a slow drift around the sweep raises the spread but not the
adjacent-ray jump, so read both numbers --- and to any offset shared by every
ray, which is the sweep value itself and which no ray-to-ray statistic can
detect.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_RHOHV_MIN",
    "SYSTEM_PHIDP_METHODS",
    "apply_system_phidp",
    "ray_consistency",
    "system_phidp",
    "system_phidp_block",
    "system_phidp_first",
    "system_phidp_hist",
    "valid_phase_mask",
]

#: Copolar correlation floor for gates admitted to a system-phase estimate.
#: 0.9 is the value specified for this implementation; it is not a wradlib
#: default (wradlib's estimators take an already-masked field). It is a
#: quality floor, not a hydrometeor threshold: the estimate needs gates whose
#: phase is meaningful at all, and below rho_hv ~ 0.9 the per-gate phase
#: variance is large enough to bias a short-window median.
DEFAULT_RHOHV_MIN = 0.9

#: Estimator names accepted by :func:`system_phidp`.
SYSTEM_PHIDP_METHODS = ("block", "first", "hist")

# Ratio of the standard error of a sample median to that of a sample mean for
# Gaussian samples, sqrt(pi / 2). Used to turn a per-gate phase noise into the
# expected noise of a ray-wise median estimate.
_MEDIAN_SE_FACTOR = np.sqrt(np.pi / 2.0)

# Interquartile range of a unit Gaussian; dividing an observed IQR by it gives
# a standard deviation that a few outlying rays cannot inflate.
_GAUSSIAN_IQR = 1.3489795003921634


def valid_phase_mask(phidp, rhohv=None, rhohv_min=DEFAULT_RHOHV_MIN):
    """Gates whose differential phase may enter a system-phase estimate.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, shape ``(n_rays, n_gates)``, in degrees.
        Non-finite entries are excluded.
    rhohv : array_like, optional
        Copolar correlation coefficient on the same grid. When omitted, only
        finiteness of ``phidp`` is required and the returned mask is
        correspondingly optimistic.
    rhohv_min : float, optional
        Correlation floor, default :data:`DEFAULT_RHOHV_MIN`.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the gate is admissible.
    """
    phidp = np.asarray(phidp, dtype=float)
    mask = np.isfinite(phidp)
    if rhohv is not None:
        rhohv = np.asarray(rhohv, dtype=float)
        if rhohv.shape != phidp.shape:
            raise ValueError(
                f"rhohv shape {rhohv.shape} does not match phidp shape {phidp.shape}"
            )
        mask &= np.isfinite(rhohv) & (rhohv >= rhohv_min)
    return mask


def rebranch(values, period=360.0):
    """Put circular ``values`` on one contiguous branch before averaging them.

    Ray-wise system-phase estimates are angles, so a stream whose system phase
    sits near a branch cut returns a mixture of (say) ``+8`` and ``+352`` for
    what is physically the same offset. Any linear statistic over that mixture
    --- mean, median, or "the N lowest" --- lands near the middle of the circle
    instead of near the offset.

    This is not hypothetical. A 360-volume survey of the case library found the
    BNF C-SAPR2 ``>=1.10`` stream sitting at ``+9.75`` degrees on a ``0/360``
    branch, and **5 of 44 volumes (11.4%) returned a system phase near +345 to
    +351 instead of ~+10** for exactly this reason. The same stream's
    ``system_phidp_first`` ray-to-ray scatter was 31.19 degrees against 2.00 for
    ``block``, which is the same effect seen from the other side.

    Values are shifted onto the branch centred on their circular mean, so the
    result is contiguous and ordinary linear statistics apply. Returns
    ``(rebranched, moved)``, where ``moved`` counts values shifted by a whole
    period --- a nonzero count is the diagnostic that the input straddled a cut.
    """
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return v.copy(), 0
    ang = np.deg2rad(v[finite] * (360.0 / period))
    centre = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) * (
        period / 360.0
    )
    out = v.copy()
    shifted = centre + (v[finite] - centre + period / 2.0) % period - period / 2.0
    out[finite] = shifted
    return out, int(np.count_nonzero(np.abs(shifted - v[finite]) > period / 4.0))


def _wrap_like(value, reference, period=360.0):
    """Wrap ``value`` back into whichever branch ``reference`` is expressed in."""
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    if not ref.size or not np.isfinite(value):
        return float(value)
    lo = 0.0 if ref.min() >= -1e-9 else -period / 2.0
    return float(lo + (value - lo) % period)


def _aggregate_sysphi(sysphi_ray, n_lowest_rays):
    """Median of the ``n_lowest_rays`` smallest finite ray-wise estimates.

    Returns NaN when fewer than ``n_lowest_rays`` rays are finite, matching
    wradlib rather than silently aggregating a handful of rays.

    The estimates are re-branched with :func:`rebranch` first, because "the N
    lowest" is a linear ordering applied to angles and is meaningless across a
    branch cut. The aggregate is wrapped back into the branch the input used.
    """
    finite = sysphi_ray[np.isfinite(sysphi_ray)]
    if finite.size < n_lowest_rays:
        return np.nan
    contiguous, _moved = rebranch(finite)
    lowest = np.partition(contiguous, n_lowest_rays - 1)[:n_lowest_rays]
    return _wrap_like(float(np.median(lowest)), finite)


def _masked(phidp, mask):
    """``phidp`` as float with masked gates set to NaN."""
    out = np.array(phidp, dtype=float, copy=True)
    out[~mask] = np.nan
    return out


def system_phidp_block(
    phidp,
    range_m,
    block_length_m,
    rhohv=None,
    rhohv_min=DEFAULT_RHOHV_MIN,
    n_lowest_rays=30,
):
    """Estimate the system phase from the first fully valid block along each ray.

    Each ray is searched for the first window of ``n_bins`` consecutive
    admissible gates, where ``n_bins = ceil(block_length_m / gate_spacing)``;
    the ray-wise estimate is the median phase inside that window. Requiring the
    window to be *fully* valid is what distinguishes this estimator from
    :func:`system_phidp_first`: it will not build an estimate out of scattered
    single gates that survived QC by luck.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    range_m : array_like
        Gate range coordinate in metres, length ``n_gates``, assumed uniform.
    block_length_m : float
        Required length of the contiguous valid block, in metres.
    rhohv : array_like, optional
        Copolar correlation on the same grid, for masking.
    rhohv_min : float, optional
        Correlation floor, default :data:`DEFAULT_RHOHV_MIN`.
    n_lowest_rays : int, optional
        Number of smallest ray-wise estimates aggregated into the sweep value.

    Returns
    -------
    dict
        ``sysphi_ray`` (n_rays,), ``sysphi`` (float), ``start_range``,
        ``stop_range``, ``valid_bins`` (all per-ray), ``n_bins`` and
        ``gate_spacing_m``.
    """
    phidp = np.asarray(phidp, dtype=float)
    range_m = np.asarray(range_m, dtype=float)
    if phidp.ndim != 2:
        raise ValueError(f"phidp must be 2-D (n_rays, n_gates), got {phidp.shape}")
    if range_m.size != phidp.shape[1]:
        raise ValueError(
            f"range_m has {range_m.size} entries but phidp has {phidp.shape[1]} gates"
        )
    spacing = float(range_m[1] - range_m[0]) if range_m.size > 1 else np.nan
    n_bins = max(1, int(np.ceil(block_length_m / spacing)))
    if n_bins > phidp.shape[1]:
        raise ValueError(
            f"block_length_m={block_length_m} needs {n_bins} gates but the ray "
            f"has only {phidp.shape[1]}"
        )

    mask = valid_phase_mask(phidp, rhohv, rhohv_min)
    values = _masked(phidp, mask)
    n_rays = phidp.shape[0]

    # Rolling count of valid gates: a cumulative sum differenced by the window
    # width, which is exact for a boolean and avoids an n_gates x n_bins pass.
    counts = np.cumsum(mask.astype(np.int64), axis=1)
    padded = np.concatenate([np.zeros((n_rays, 1), dtype=np.int64), counts], axis=1)
    window_counts = padded[:, n_bins:] - padded[:, :-n_bins]

    sysphi_ray = np.full(n_rays, np.nan)
    start_range = np.full(n_rays, np.nan)
    stop_range = np.full(n_rays, np.nan)
    valid_bins = np.zeros(n_rays, dtype=np.int64)

    full = window_counts == n_bins
    for ray in range(n_rays):
        hits = np.flatnonzero(full[ray])
        if hits.size == 0:
            continue
        start = int(hits[0])
        stop = start + n_bins
        sysphi_ray[ray] = float(np.median(values[ray, start:stop]))
        start_range[ray] = range_m[start]
        stop_range[ray] = range_m[stop - 1]
        valid_bins[ray] = n_bins

    return {
        "sysphi_ray": sysphi_ray,
        "sysphi": _aggregate_sysphi(sysphi_ray, n_lowest_rays),
        "start_range": start_range,
        "stop_range": stop_range,
        "valid_bins": valid_bins,
        "n_bins": n_bins,
        "gate_spacing_m": spacing,
        "method": "block",
    }


def system_phidp_first(
    phidp,
    range_m=None,
    n_valid_bins=10,
    rhohv=None,
    rhohv_min=DEFAULT_RHOHV_MIN,
    n_lowest_rays=30,
):
    """Estimate the system phase from the first N admissible gates of each ray.

    The gates need not be contiguous, which makes this the more permissive of
    the two range-domain estimators: it will produce a number on a ray whose
    near range is patchy, at the cost of pooling gates that may be far apart.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    range_m : array_like, optional
        Gate range coordinate in metres, used only to report the range span of
        the gates that were used.
    n_valid_bins : int, optional
        Number of admissible gates per ray, default 10.
    rhohv : array_like, optional
        Copolar correlation on the same grid, for masking.
    rhohv_min : float, optional
        Correlation floor, default :data:`DEFAULT_RHOHV_MIN`.
    n_lowest_rays : int, optional
        Number of smallest ray-wise estimates aggregated into the sweep value.

    Returns
    -------
    dict
        Same keys as :func:`system_phidp_block`, with ``n_valid_bins`` in place
        of ``n_bins``.
    """
    phidp = np.asarray(phidp, dtype=float)
    if phidp.ndim != 2:
        raise ValueError(f"phidp must be 2-D (n_rays, n_gates), got {phidp.shape}")
    mask = valid_phase_mask(phidp, rhohv, rhohv_min)
    values = _masked(phidp, mask)
    n_rays, n_gates = phidp.shape
    range_m = np.arange(n_gates, dtype=float) if range_m is None else range_m
    range_m = np.asarray(range_m, dtype=float)

    # Keep a gate if it is valid and among the first n_valid_bins valid ones.
    rank = np.cumsum(mask.astype(np.int64), axis=1)
    selected = mask & (rank <= n_valid_bins)

    sysphi_ray = np.full(n_rays, np.nan)
    start_range = np.full(n_rays, np.nan)
    stop_range = np.full(n_rays, np.nan)
    valid_bins = selected.sum(axis=1).astype(np.int64)

    for ray in range(n_rays):
        idx = np.flatnonzero(selected[ray])
        if idx.size == 0:
            continue
        sysphi_ray[ray] = float(np.median(values[ray, idx]))
        start_range[ray] = range_m[idx[0]]
        stop_range[ray] = range_m[idx[-1]]

    return {
        "sysphi_ray": sysphi_ray,
        "sysphi": _aggregate_sysphi(sysphi_ray, n_lowest_rays),
        "start_range": start_range,
        "stop_range": stop_range,
        "valid_bins": valid_bins,
        "n_valid_bins": int(n_valid_bins),
        "method": "first",
    }


def system_phidp_hist(
    phidp,
    bins=(-180.0, 180.0, 0.1),
    window=11,
    threshold=0.5,
    rhohv=None,
    rhohv_min=DEFAULT_RHOHV_MIN,
    n_lowest_rays=30,
):
    """Estimate the system phase from each ray's phase histogram.

    A per-ray histogram is smoothed along the bin axis and read two ways: the
    location of the peak, and the first bin exceeding ``threshold`` of the
    normalised peak. The second is the lower-biased of the pair by construction,
    which is the point of it --- the system offset is a *floor* on measured
    phase, so once any propagation phase has accumulated the leading edge of the
    distribution sits closer to the offset than the mode does.

    Unlike the two range-domain estimators this one uses the whole ray, so it
    does not need a clean near range; it needs instead a ray with enough
    low-phase gates for the leading edge to be defined.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    bins : tuple of float, optional
        ``(start, stop, step)`` passed to :func:`numpy.arange`.
    window : int, optional
        Moving-average width along the histogram bin axis, in bins.
    threshold : float, optional
        Fraction of the normalised peak defining the leading edge.
    rhohv : array_like, optional
        Copolar correlation on the same grid, for masking.
    rhohv_min : float, optional
        Correlation floor, default :data:`DEFAULT_RHOHV_MIN`.
    n_lowest_rays : int, optional
        Number of smallest ray-wise estimates aggregated into the sweep values.

    Returns
    -------
    dict
        ``sysphi_peak_ray``, ``sysphi_first_ray``, ``sysphi_peak``,
        ``sysphi_first``, plus ``sysphi_ray``/``sysphi`` aliased to the
        leading-edge pair so the return is interchangeable with the other two
        estimators, and ``bin_centres``/``histogram``.
    """
    phidp = np.asarray(phidp, dtype=float)
    if phidp.ndim != 2:
        raise ValueError(f"phidp must be 2-D (n_rays, n_gates), got {phidp.shape}")
    mask = valid_phase_mask(phidp, rhohv, rhohv_min)
    edges = np.arange(*bins)
    if edges.size < 2:
        raise ValueError(f"bins={bins} produced fewer than two edges")
    centres = 0.5 * (edges[:-1] + edges[1:])
    n_rays = phidp.shape[0]

    histogram = np.zeros((n_rays, centres.size))
    for ray in range(n_rays):
        selected = phidp[ray][mask[ray]]
        if selected.size:
            histogram[ray], _ = np.histogram(selected, bins=edges)

    # Moving average along the bin axis, normalised by the number of bins that
    # actually contributed so the ends are not damped toward zero.
    kernel = np.ones(int(window))
    denominator = np.convolve(np.ones(centres.size), kernel, mode="same")
    smooth = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, histogram
    )
    smooth = smooth / denominator

    peak = smooth.max(axis=1)
    sysphi_peak_ray = np.where(peak > 0, centres[np.argmax(smooth, axis=1)], np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = smooth / peak[:, None]
    over = np.isfinite(normalised) & (normalised > threshold)
    sysphi_first_ray = np.where(
        over.any(axis=1), centres[np.argmax(over, axis=1)], np.nan
    )

    return {
        "sysphi_peak_ray": sysphi_peak_ray,
        "sysphi_first_ray": sysphi_first_ray,
        "sysphi_peak": _aggregate_sysphi(sysphi_peak_ray, n_lowest_rays),
        "sysphi_first": _aggregate_sysphi(sysphi_first_ray, n_lowest_rays),
        "sysphi_ray": sysphi_first_ray,
        "sysphi": _aggregate_sysphi(sysphi_first_ray, n_lowest_rays),
        "bin_centres": centres,
        "histogram": histogram,
        "method": "hist",
    }


def ray_consistency(sysphi_ray, sigma_phi_deg=None, n_eff=None):
    """Quantify how consistent the ray-wise system-phase estimates are.

    The statistics are all *offset* statistics, because that is what a
    ray-to-ray inconsistency in system phase actually is. See the module
    docstring for what this is blind to.

    Parameters
    ----------
    sysphi_ray : array_like
        Per-ray system-phase estimates in degrees; NaN rays are ignored.
    sigma_phi_deg : float, optional
        Per-gate differential-phase noise in degrees. Supply it together with
        ``n_eff`` to get ``expected_noise_deg`` and the ``per_ray_justified``
        verdict; without both, those are ``None`` and the function reports only
        the observed spread.
    n_eff : int, optional
        Number of gates that entered each ray-wise median: ``n_bins`` for
        :func:`system_phidp_block`, ``n_valid_bins`` for
        :func:`system_phidp_first`.

    Returns
    -------
    dict
        ``n_rays``, ``n_valid``, ``median_deg``, ``robust_sd_deg`` (from the
        interquartile range), ``iqr_deg``, ``p5_to_p95_deg``,
        ``median_adjacent_jump_deg``, ``p95_adjacent_jump_deg``,
        ``expected_noise_deg``, ``excess_ratio`` and ``per_ray_justified``.

    Notes
    -----
    ``per_ray_justified`` is ``True`` when the observed robust standard
    deviation exceeds twice the standard error expected of the ray-wise
    estimator. Below that, applying a per-ray offset transfers estimator noise
    into the data. The factor of two is a design choice, not a measured
    threshold; ``excess_ratio`` is reported so a caller can apply their own.
    """
    sysphi_ray = np.asarray(sysphi_ray, dtype=float)
    values = sysphi_ray[np.isfinite(sysphi_ray)]
    out = {
        "n_rays": int(sysphi_ray.size),
        "n_valid": int(values.size),
        "median_deg": float(np.median(values)) if values.size else np.nan,
        "robust_sd_deg": np.nan,
        "iqr_deg": np.nan,
        "p5_to_p95_deg": np.nan,
        "median_adjacent_jump_deg": np.nan,
        "p95_adjacent_jump_deg": np.nan,
        "expected_noise_deg": None,
        "excess_ratio": None,
        "per_ray_justified": None,
    }
    if values.size < 4:
        return out

    q25, q75 = np.percentile(values, [25.0, 75.0])
    out["iqr_deg"] = float(q75 - q25)
    out["robust_sd_deg"] = float(q75 - q25) / _GAUSSIAN_IQR
    p5, p95 = np.percentile(values, [5.0, 95.0])
    out["p5_to_p95_deg"] = float(p95 - p5)

    # Jumps are taken between consecutive *valid* rays, so a gap of dropped
    # rays does not register as a jump of zero.
    jumps = np.abs(np.diff(values))
    if jumps.size:
        out["median_adjacent_jump_deg"] = float(np.median(jumps))
        out["p95_adjacent_jump_deg"] = float(np.percentile(jumps, 95.0))

    if sigma_phi_deg is not None and n_eff:
        expected = _MEDIAN_SE_FACTOR * float(sigma_phi_deg) / np.sqrt(float(n_eff))
        out["expected_noise_deg"] = float(expected)
        ratio = out["robust_sd_deg"] / expected if expected > 0 else np.inf
        out["excess_ratio"] = float(ratio)
        out["per_ray_justified"] = bool(ratio > 2.0)
    return out


def system_phidp(
    phidp,
    range_m=None,
    method="block",
    rhohv=None,
    rhohv_min=DEFAULT_RHOHV_MIN,
    n_lowest_rays=30,
    sigma_phi_deg=None,
    **kwargs,
):
    """Estimate the system phase and its ray-to-ray consistency in one call.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    range_m : array_like, optional
        Gate range coordinate in metres. Required for ``method='block'``.
    method : {'block', 'first', 'hist'}, optional
        Estimator, see :data:`SYSTEM_PHIDP_METHODS`.
    rhohv : array_like, optional
        Copolar correlation on the same grid, for masking.
    rhohv_min : float, optional
        Correlation floor, default :data:`DEFAULT_RHOHV_MIN`.
    n_lowest_rays : int, optional
        Number of smallest ray-wise estimates aggregated into the sweep value.
    sigma_phi_deg : float, optional
        Per-gate phase noise, passed to :func:`ray_consistency` so the
        ``per_ray_justified`` verdict can be formed.
    **kwargs
        Estimator-specific arguments: ``block_length_m`` for ``'block'``,
        ``n_valid_bins`` for ``'first'``, ``bins``/``window``/``threshold``
        for ``'hist'``.

    Returns
    -------
    dict
        The estimator's own output plus a ``consistency`` sub-dict from
        :func:`ray_consistency`.
    """
    if method not in SYSTEM_PHIDP_METHODS:
        raise ValueError(
            f"method must be one of {SYSTEM_PHIDP_METHODS}, got {method!r}"
        )
    common = {"rhohv": rhohv, "rhohv_min": rhohv_min, "n_lowest_rays": n_lowest_rays}
    if method == "block":
        if range_m is None:
            raise ValueError("method='block' requires range_m")
        block_length_m = kwargs.pop("block_length_m", 1000.0)
        result = system_phidp_block(phidp, range_m, block_length_m, **common, **kwargs)
        n_eff = result["n_bins"]
    elif method == "first":
        result = system_phidp_first(phidp, range_m=range_m, **common, **kwargs)
        n_eff = result["n_valid_bins"]
    else:
        result = system_phidp_hist(phidp, **common, **kwargs)
        # The histogram estimator reads a distribution edge rather than
        # averaging n samples, so no per-ray n_eff is defined and the
        # noise-based verdict is deliberately withheld.
        n_eff = None
    result["consistency"] = ray_consistency(
        result["sysphi_ray"], sigma_phi_deg=sigma_phi_deg, n_eff=n_eff
    )
    return result


def apply_system_phidp(phidp, estimate, per_ray=False):
    """Subtract the system phase from a measured differential-phase field.

    Parameters
    ----------
    phidp : array_like
        Measured differential phase, ``(n_rays, n_gates)``, degrees.
    estimate : dict or float
        Output of :func:`system_phidp` (or one of the estimators), or a bare
        offset in degrees.
    per_ray : bool, optional
        When ``False`` (default) the sweep-level scalar is subtracted from every
        ray; see the module docstring for the evidence behind that default. When
        ``True`` each ray has its own ``sysphi_ray`` subtracted, and rays whose
        estimate is NaN fall back to the sweep value so a ray is never dropped
        by the correction step.

    Returns
    -------
    corrected : numpy.ndarray
        ``phidp`` with the offset removed.
    offset : numpy.ndarray
        The offset actually subtracted, shape ``(n_rays, 1)``, so the caller can
        record what was applied to each ray.
    """
    phidp = np.asarray(phidp, dtype=float)
    if isinstance(estimate, dict):
        sweep_value = estimate["sysphi"]
        ray_values = np.asarray(estimate["sysphi_ray"], dtype=float)
    else:
        sweep_value = float(estimate)
        ray_values = np.full(phidp.shape[0], sweep_value)

    if not np.isfinite(sweep_value):
        raise ValueError(
            "sweep-level system phase is not finite: too few rays carried a "
            "valid estimate. Lower n_lowest_rays, relax rhohv_min, or choose a "
            "different method -- but note that an offset derived from a handful "
            "of rays is what this guard exists to prevent."
        )
    if per_ray:
        offset = np.where(np.isfinite(ray_values), ray_values, sweep_value)
    else:
        offset = np.full(phidp.shape[0], sweep_value)
    offset = offset.reshape(-1, 1)
    return phidp - offset, offset
