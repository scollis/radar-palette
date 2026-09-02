"""KDP bounds from reflectivity and differential reflectivity.

The two-sided box in :mod:`radar_palette.phase.linear_program` needs a lower and
an upper bound on KDP at every gate. This module builds them. The design choice
that makes the whole approach work is that the bounds come from Z and ZDR, which
are *direct measurements* --- power and a power ratio --- and so are not
contaminated by the backscatter-phase term ``delta`` that corrupts the phase the
estimator is fitting. Bounding a quantity with an independent error channel is
what buys robustness in the regimes where phase-only methods fail.

Three ingredients, following Huang, Zhang, Zhao & Giangrande (2016),
`doi:10.1109/TGRS.2016.2596295 <https://doi.org/10.1109/TGRS.2016.2596295>`_:

1. **Self-consistency** (their Eq. 14). ``KDP / z_H`` is a function of ZDR alone.
   Scaled by a tolerance factor either side, it gives the central box.
2. **A phase-only least-squares floor** (their Eq. 15). A long-window
   least-squares slope of the measured phase is robust to ``delta``, because a
   backscatter excursion is a local perturbation that a long fit averages away.
   It raises the lower bound where the phase itself clearly shows a gradient.
3. **A reflectivity cap** (their Eq. 16). A hard ceiling as a function of Z.

Where the coefficients came from
--------------------------------
:data:`SELF_CONSISTENCY_COEFFICIENTS` was **computed for this package** with a
T-matrix forward model rather than adopted from a published fit: rustmatrix
2.2.0, water at 10 degC, Brandes, Zhang & Vivekanandan (2002) Eq. (2) drop
shapes, 64-point scattering table to D_max = 8 mm, normalized gamma PSD over
D0 = 0.50--3.50 mm, mu in (0, 3, 5, 8) and Nw in (1e3, 8e3, 1e5), fitted over
ZDR = 0.2--4.0 dB. See ``docs/phase-design-notes.md``.

Two properties of that computation are worth carrying:

* ``KDP / z_H`` moved by **2e-13 relative** across two decades of Nw at fixed
  (D0, mu) --- numerically exact independence, as it must be since KDP and z_H
  are both exactly linear in Nw. The relation really is a function of ZDR alone.
* Against Gourley et al. (2009) Table 4, the same functional form fitted to a
  different shape relation, temperature and mu, this fit runs **1.05x at
  ZDR = 0.5 dB rising to 1.34x at 3 dB** at C band. Two credible forward models
  disagree by a third at high ZDR. That is the direct justification for the
  generous default tolerance factors: the relation is not accurate enough to
  bound KDP tightly, and pretending otherwise is how a bound becomes a bias.

Why the constraint is restricted to rain
----------------------------------------
Huang's own assessment is that self-consistency methods fail "in critical
situations such as hail cores where these methods must rigidly adhere to
consistency relationship constraints that do not apply". The relation is derived
for oblate liquid drops. In a hail core, in the melting layer, or above it, it is
a rain relation used out of scope. :func:`kdp_bounds` therefore takes a
``rain_mask`` and *releases* the box outside it to a permissive symmetric
interval, rather than applying a relation it knows to be invalid.

Sign policy is altitude-conditioned, not global
-----------------------------------------------
Huang 2016 is rain-only and enforces ``KDP_L >= 0`` everywhere. That is correct
for its scope and wrong as a general prior: above the melting layer in an
electrically active storm, ice crystals align vertically and **negative KDP is
the signal**, not noise. A global positivity bound erases exactly the quantity
such a study is looking for. The default here is ``'altitude_conditioned'``,
which enforces non-negativity below the freezing level and releases the sign
aloft over a stated blend depth. ``'rain_nonnegative'`` reproduces the published
behaviour, and is not the default.

What the bounds are blind to
----------------------------
Everything they inherit from ZDR calibration. The self-consistency polynomial is
**cubic in ZDR**, so a 1 dB ZDR offset moves it by tens of percent, and it goes
negative above roughly 5.1 dB at C band (:data:`SELF_CONSISTENCY_ZDR_ROOTS`).
The bounds are only as good as the ZDR offset, and this module cannot check that
offset --- it must be settled upstream.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "BRANDES_2002_AXIS_RATIO",
    "DEFAULT_LOWER_FACTOR",
    "DEFAULT_REFLECTIVITY_CAP_LADDER",
    "DEFAULT_UPPER_FACTOR",
    "GOURLEY_2009_TABLE4",
    "SELF_CONSISTENCY_COEFFICIENTS",
    "SELF_CONSISTENCY_ZDR_ROOTS",
    "SIGN_POLICIES",
    "kdp_bounds",
    "lsf_kdp",
    "rain_mask_from_context",
    "reflectivity_cap",
    "self_consistency_kdp",
    "sign_floor",
]

#: Brandes, Zhang & Vivekanandan (2002) Eq. (2), ascending powers of D in mm.
#: Returns the axis ratio as **v/h**, so it is below 1 for an oblate drop. A
#: T-matrix code wanting h/v needs the reciprocal; passing this straight through
#: silently models a prolate particle. Recorded here because it is the shape
#: relation :data:`SELF_CONSISTENCY_COEFFICIENTS` was computed with, and a bound
#: whose shape assumption is undocumented cannot be audited.
BRANDES_2002_AXIS_RATIO = (0.9951, 0.02510, -0.03644, 0.005030, -0.0002492)

#: Self-consistency coefficients ``(a1, a2, a3, a4)`` for
#: ``KDP = z_H * 1e-5 * (a1 + a2*Zdr + a3*Zdr^2 + a4*Zdr^3)`` with KDP in
#: deg/km one-way, ``z_H`` linear in mm^6 m^-3 and ``Zdr`` in dB. Computed with
#: rustmatrix 2.2.0 for this package; see the module docstring for the full
#: configuration. Maximum relative residual of the cubic over the fit window:
#: S 4.6%, C 5.7%, X 10.8% --- X is the poorest fit and its cubic has no
#: positive root, so treat the X entry as the least trustworthy of the three.
SELF_CONSISTENCY_COEFFICIENTS = {
    "S": (
        3.3382526763440796,
        -1.7453756700632714,
        0.4638274335876116,
        -0.04742465511741514,
    ),
    "C": (
        6.947102867400848,
        -2.863335191501533,
        0.7744777791035332,
        -0.09336040523179025,
    ),
    "X": (
        11.505763783008849,
        -3.0208376136688977,
        -0.3670781434108895,
        0.12287001958909603,
    ),
}

#: ZDR in dB at which each cubic first crosses zero, beyond which the relation
#: is unusable and :func:`self_consistency_kdp` returns NaN. ``None`` means the
#: cubic has no positive real root, which is the case for the X-band fit.
SELF_CONSISTENCY_ZDR_ROOTS = {
    "S": 5.36656397125609,
    "C": 5.145602470024442,
    "X": None,
}

#: Gourley et al. (2009) Table 4, the same functional form fitted to BZV shapes
#: at mu = 5 and 0 degC. Kept for comparison, not used by default: the ratio
#: between this and :data:`SELF_CONSISTENCY_COEFFICIENTS` is the evidence for the
#: default tolerance factors. Pass it as ``coefficients`` to use it.
GOURLEY_2009_TABLE4 = {
    "S": (3.696, -1.963, 0.504, -0.051),
    "C": (6.746, -2.970, 0.711, -0.079),
    "X": (11.74, -4.020, -0.140, 0.130),
}

#: Multiplier applied to the self-consistency estimate to form the lower bound.
#: 0.5 rather than something tighter because two credible forward models of the
#: same relation disagree by up to 1.34x at high ZDR, and the relation is cubic
#: in a quantity carrying a calibration offset. A bound tighter than the
#: uncertainty in the relation that generated it is a bias.
DEFAULT_LOWER_FACTOR = 0.5

#: Multiplier applied to the self-consistency estimate to form the upper bound.
DEFAULT_UPPER_FACTOR = 2.0

#: Reflectivity ceiling on KDP as ``((z_dbz_below, cap_deg_per_km), ...)``,
#: ascending in Z. The two rungs are Huang et al. (2016) Eq. (16) as reported
#: for their C-band configuration: 8 deg/km below 35 dBZ, 10 deg/km below
#: 45 dBZ. **Eq. (16) has exactly these two rungs**: verified against the
#: accepted manuscript (p. 27), the paper specifies no cap above 45 dBZ
#: anywhere. :func:`reflectivity_cap` therefore takes ``cap_above`` explicitly
#: and defaults to no reflectivity cap there, leaving the self-consistency upper
#: bound to act alone — which reproduces Eq. (16) rather than working around a
#: gap in it.
DEFAULT_REFLECTIVITY_CAP_LADDER = ((35.0, 8.0), (45.0, 10.0))

#: Accepted sign policies. See the module docstring for why
#: ``'altitude_conditioned'`` is the default and ``'rain_nonnegative'`` --- the
#: published Huang 2016 behaviour --- is not.
SIGN_POLICIES = ("altitude_conditioned", "rain_nonnegative", "free")


def self_consistency_kdp(z_dbz, zdr_db, band="C", coefficients=None):
    """KDP implied by reflectivity and differential reflectivity.

    Parameters
    ----------
    z_dbz : array_like
        Horizontal reflectivity factor in dBZ.
    zdr_db : array_like
        Differential reflectivity in dB, **already offset-corrected**. The
        relation is cubic in this quantity; see the module docstring.
    band : {'S', 'C', 'X'}, optional
        Selects an entry of :data:`SELF_CONSISTENCY_COEFFICIENTS`.
    coefficients : sequence of float, optional
        ``(a1, a2, a3, a4)`` overriding the built-in set, for a locally derived
        relation or to reproduce :data:`GOURLEY_2009_TABLE4`.

    Returns
    -------
    numpy.ndarray
        One-way KDP in deg/km. NaN where the inputs are non-finite, and NaN
        above the polynomial's first positive root, where the relation is out of
        its valid range rather than merely imprecise.
    """
    z_dbz = np.asarray(z_dbz, dtype=float)
    zdr_db = np.asarray(zdr_db, dtype=float)
    if coefficients is None:
        if band not in SELF_CONSISTENCY_COEFFICIENTS:
            raise ValueError(
                f"band must be one of {tuple(SELF_CONSISTENCY_COEFFICIENTS)}, "
                f"got {band!r}"
            )
        coefficients = SELF_CONSISTENCY_COEFFICIENTS[band]
        root = SELF_CONSISTENCY_ZDR_ROOTS[band]
    else:
        real_roots = [
            float(r.real)
            for r in np.roots(np.asarray(coefficients, dtype=float)[::-1])
            if abs(r.imag) < 1e-9 and r.real > 0.0
        ]
        root = min(real_roots) if real_roots else None

    a1, a2, a3, a4 = (float(c) for c in coefficients)
    polynomial = a1 + a2 * zdr_db + a3 * zdr_db**2 + a4 * zdr_db**3
    z_linear = 10.0 ** (0.1 * z_dbz)
    kdp = z_linear * 1.0e-5 * polynomial
    invalid = ~np.isfinite(z_dbz) | ~np.isfinite(zdr_db)
    if root is not None:
        invalid = invalid | (zdr_db > root)
    return np.where(invalid, np.nan, kdp)


def reflectivity_cap(
    z_dbz, ladder=DEFAULT_REFLECTIVITY_CAP_LADDER, cap_above=None, fill=np.inf
):
    """Hard KDP ceiling as a step function of reflectivity.

    Parameters
    ----------
    z_dbz : array_like
        Horizontal reflectivity factor in dBZ.
    ladder : sequence of (float, float), optional
        ``(z_dbz_below, cap_deg_per_km)`` rungs ascending in Z, default
        :data:`DEFAULT_REFLECTIVITY_CAP_LADDER`.
    cap_above : float, optional
        Ceiling for gates above the highest rung. ``None`` (default) means no
        reflectivity ceiling there; see
        :data:`DEFAULT_REFLECTIVITY_CAP_LADDER` for why this is explicit.
    fill : float, optional
        Value where ``z_dbz`` is non-finite, default ``inf`` --- an unknown
        reflectivity must not be allowed to impose a bound.

    Returns
    -------
    numpy.ndarray
        Ceiling in deg/km, same shape as ``z_dbz``.
    """
    z_dbz = np.asarray(z_dbz, dtype=float)
    top = np.inf if cap_above is None else float(cap_above)
    cap = np.full(z_dbz.shape, top, dtype=float)
    for z_threshold, value in sorted(ladder, reverse=True):
        cap = np.where(z_dbz < z_threshold, value, cap)
    return np.where(np.isfinite(z_dbz), cap, fill)


def lsf_kdp(phidp, dr_km, window_km, mask=None, min_gates=None):
    """Least-squares KDP from the measured phase alone, over a long window.

    This is the phase-only floor of Huang et al. (2016) Eq. (15). Its purpose is
    not accuracy: a window long enough to average away a backscatter-phase
    excursion is far too long to resolve a convective core. Its purpose is to be
    *robust* --- if the phase rises by 40 degrees over 10 km then KDP averaged
    over that stretch is 2 deg/km whatever ``delta`` is doing locally, and no
    lower bound below that is defensible.

    Parameters
    ----------
    phidp : array_like
        Differential phase, ``(n_rays, n_gates)``, degrees, unfolded and
        system-phase corrected.
    dr_km : float
        Gate spacing in kilometres.
    window_km : float
        Fitting window length in kilometres. This is the *heavily smoothed*
        window of Eq. (15) and is deliberately much longer than the estimator's
        own smoothing length --- it is a different quantity, not a tuning of the
        same one.
    mask : array_like of bool, optional
        Gates to fit. Defaults to the finite gates of ``phidp``.
    min_gates : int, optional
        Minimum number of valid gates in a window for a slope to be returned.
        Defaults to half the window.

    Returns
    -------
    numpy.ndarray
        One-way KDP in deg/km, same shape as ``phidp``, NaN where the window
        held too few valid gates. Windows are centred, so the result is defined
        to the ends of the ray with progressively fewer contributing gates.
    """
    phidp = np.asarray(phidp, dtype=float)
    if phidp.ndim != 2:
        raise ValueError(f"phidp must be 2-D (n_rays, n_gates), got {phidp.shape}")
    if not np.isfinite(dr_km) or dr_km <= 0:
        raise ValueError(f"dr_km must be positive and finite, got {dr_km}")
    if not np.isfinite(window_km) or window_km <= 0:
        raise ValueError(f"window_km must be positive and finite, got {window_km}")

    half = max(1, int(round(0.5 * window_km / dr_km)))
    span = 2 * half + 1
    if min_gates is None:
        min_gates = max(3, span // 2)

    valid = (
        np.isfinite(phidp)
        if mask is None
        else (np.asarray(mask, dtype=bool) & np.isfinite(phidp))
    )
    n_gates = phidp.shape[1]
    positions = np.arange(n_gates, dtype=float) * dr_km

    # Weighted moments accumulated by cumulative sum, so a centred window of any
    # length costs two lookups per gate instead of a slice and a fit. Exact, not
    # an approximation to the per-window least-squares slope.
    weight = valid.astype(float)
    y = np.where(valid, phidp, 0.0)
    x = np.broadcast_to(positions, phidp.shape)

    def prefix(values):
        return np.concatenate(
            [np.zeros((values.shape[0], 1)), np.cumsum(values, axis=1)], axis=1
        )

    cum_w = prefix(weight)
    cum_x = prefix(weight * x)
    cum_y = prefix(y)
    cum_xx = prefix(weight * x * x)
    cum_xy = prefix(x * y)

    gates = np.arange(n_gates)
    lo = np.clip(gates - half, 0, n_gates)
    hi = np.clip(gates + half + 1, 0, n_gates)

    def window(cumulative):
        return cumulative[:, hi] - cumulative[:, lo]

    n = window(cum_w)
    sum_x = window(cum_x)
    sum_y = window(cum_y)
    sum_xx = window(cum_xx)
    sum_xy = window(cum_xy)

    denominator = n * sum_xx - sum_x**2
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = (n * sum_xy - sum_x * sum_y) / denominator
    usable = (n >= min_gates) & (denominator > 0.0)
    return np.where(usable, 0.5 * slope, np.nan)


def rain_mask_from_context(
    z_dbz=None,
    rhohv=None,
    height_km=None,
    iso0_km=None,
    z_hail_dbz=50.0,
    rhohv_min=0.95,
    margin_km=0.0,
):
    """Gates where the rain self-consistency relation is in scope.

    Every criterion is optional; those supplied are combined with AND, and a
    call with no arguments raises rather than silently declaring everything rain.

    Parameters
    ----------
    z_dbz : array_like, optional
        Reflectivity. Gates at or above ``z_hail_dbz`` are excluded as possible
        hail, which is the regime Huang et al. name as the failure case.
    rhohv : array_like, optional
        Copolar correlation; gates below ``rhohv_min`` are excluded as mixed
        phase or non-meteorological.
    height_km : array_like, optional
        Gate height above mean sea level in kilometres.
    iso0_km : float, optional
        Freezing-level height in kilometres. Required with ``height_km``.
    z_hail_dbz : float, optional
        Hail exclusion threshold in dBZ, default 50.
    rhohv_min : float, optional
        Correlation floor, default 0.95 --- stricter than the system-phase
        floor because this mask decides where a *physical relation* is applied,
        not merely where a phase reading is usable.
    margin_km : float, optional
        Distance below the freezing level to also exclude, for the melting
        layer. Default 0, i.e. the isotherm itself is the boundary.

    Returns
    -------
    numpy.ndarray
        Boolean mask, ``True`` where the rain relation applies.
    """
    criteria = []
    shape = None
    for field in (z_dbz, rhohv, height_km):
        if field is not None:
            shape = np.asarray(field).shape
            break
    if shape is None:
        raise ValueError(
            "rain_mask_from_context needs at least one of z_dbz, rhohv or "
            "height_km; a mask that defaults to all-rain would apply the rain "
            "relation in exactly the regimes it is known to fail"
        )
    if z_dbz is not None:
        z_dbz = np.asarray(z_dbz, dtype=float)
        criteria.append(np.isfinite(z_dbz) & (z_dbz < z_hail_dbz))
    if rhohv is not None:
        rhohv = np.asarray(rhohv, dtype=float)
        criteria.append(np.isfinite(rhohv) & (rhohv >= rhohv_min))
    if height_km is not None:
        if iso0_km is None:
            raise ValueError("height_km requires iso0_km")
        height_km = np.asarray(height_km, dtype=float)
        criteria.append(np.isfinite(height_km) & (height_km <= iso0_km - margin_km))

    mask = criteria[0]
    for criterion in criteria[1:]:
        mask = mask & criterion
    return mask


def sign_floor(
    shape,
    policy="altitude_conditioned",
    height_km=None,
    iso0_km=None,
    blend_km=0.5,
    negative_floor=-2.0,
):
    """Lower bound on KDP imposed by the sign policy alone.

    Parameters
    ----------
    shape : tuple of int
        Output shape.
    policy : {'altitude_conditioned', 'rain_nonnegative', 'free'}, optional
        See :data:`SIGN_POLICIES` and the module docstring.
    height_km : array_like, optional
        Gate height above mean sea level in kilometres. Required for
        ``'altitude_conditioned'``.
    iso0_km : float, optional
        Freezing-level height in kilometres. Required for
        ``'altitude_conditioned'``.
    blend_km : float, optional
        Depth over which the floor ramps from 0 to ``negative_floor`` across the
        isotherm, default 0.5 km. The ramp is centred on ``iso0_km``.
    negative_floor : float, optional
        Most negative KDP permitted in the ice regime, deg/km. Default -2.0.
        **This is a design choice, not a measured quantity** --- no aligned-ice
        KDP magnitude was measured for this package. It exists to keep the
        problem bounded, and should be widened rather than trusted if aligned
        ice is the object of study.

    Returns
    -------
    numpy.ndarray
        Lower bound in deg/km, of shape ``shape``.
    """
    if policy not in SIGN_POLICIES:
        raise ValueError(f"policy must be one of {SIGN_POLICIES}, got {policy!r}")
    if policy == "rain_nonnegative":
        return np.zeros(shape, dtype=float)
    if policy == "free":
        return np.full(shape, float(negative_floor), dtype=float)

    if height_km is None or iso0_km is None:
        raise ValueError(
            "policy='altitude_conditioned' requires height_km and iso0_km. It "
            "is not silently downgraded to a global policy: a global "
            "non-negativity prior erases aligned-ice signal, and a global free "
            "sign admits negative KDP in rain. Pass the geometry, or choose "
            "'rain_nonnegative' / 'free' deliberately."
        )
    height_km = np.asarray(height_km, dtype=float)
    if height_km.shape != tuple(shape):
        raise ValueError(
            f"height_km shape {height_km.shape} does not match shape {tuple(shape)}"
        )
    if blend_km <= 0:
        fraction = (height_km > iso0_km).astype(float)
    else:
        fraction = (height_km - (iso0_km - 0.5 * blend_km)) / blend_km
        fraction = np.clip(fraction, 0.0, 1.0)
    # A gate of unknown height is treated as rain: the conservative direction is
    # the one that does not admit negative KDP on no evidence.
    fraction = np.where(np.isfinite(height_km), fraction, 0.0)
    return fraction * float(negative_floor)


def kdp_bounds(
    z_dbz,
    zdr_db,
    phidp=None,
    dr_km=None,
    band="C",
    coefficients=None,
    lower_factor=DEFAULT_LOWER_FACTOR,
    upper_factor=DEFAULT_UPPER_FACTOR,
    ladder=DEFAULT_REFLECTIVITY_CAP_LADDER,
    cap_above=None,
    rain_mask=None,
    non_rain_cap=10.0,
    lsf_window_km=None,
    lsf_factor=0.5,
    sign_policy="altitude_conditioned",
    height_km=None,
    iso0_km=None,
    blend_km=0.5,
    negative_floor=-2.0,
):
    """Assemble the two-sided KDP box for the linear program.

    Parameters
    ----------
    z_dbz, zdr_db : array_like
        Reflectivity in dBZ and offset-corrected differential reflectivity in
        dB, both ``(n_rays, n_gates)``.
    phidp : array_like, optional
        Unfolded, system-phase-corrected differential phase. Supply it with
        ``dr_km`` and ``lsf_window_km`` to enable the Eq. (15) phase-only floor.
    dr_km : float, optional
        Gate spacing in kilometres, required with ``phidp``.
    band : {'S', 'C', 'X'}, optional
        Band for the self-consistency coefficients.
    coefficients : sequence of float, optional
        Override for the self-consistency coefficients.
    lower_factor, upper_factor : float, optional
        Tolerance multipliers on the self-consistency estimate, defaults
        :data:`DEFAULT_LOWER_FACTOR` and :data:`DEFAULT_UPPER_FACTOR`.
    ladder : sequence of (float, float), optional
        Reflectivity cap rungs, see :func:`reflectivity_cap`.
    cap_above : float, optional
        Cap above the top rung, see :func:`reflectivity_cap`.
    rain_mask : array_like of bool, optional
        Where the self-consistency relation applies. Outside it the box is
        released to ``[sign_floor, non_rain_cap]``. Strongly recommended: build
        it with :func:`rain_mask_from_context`. When omitted the relation is
        applied everywhere and ``metadata['rain_mask_supplied']`` records that.
    non_rain_cap : float, optional
        Upper bound in deg/km where the rain relation does not apply, default
        10.0. This is the magnitude the box permits in a hail core or aloft; it
        is a permissive ceiling, not an estimate.
    lsf_window_km : float, optional
        Window for the phase-only floor. ``None`` disables it.
    lsf_factor : float, optional
        Multiplier on the phase-only slope before it is used as a floor,
        default 0.5.
    sign_policy : {'altitude_conditioned', 'rain_nonnegative', 'free'}, optional
        See :func:`sign_floor`.
    height_km : array_like, optional
        Gate height in kilometres, required for ``'altitude_conditioned'``.
    iso0_km : float, optional
        Freezing level in kilometres, required for ``'altitude_conditioned'``.
    blend_km, negative_floor : float, optional
        See :func:`sign_floor`.

    Returns
    -------
    lower, upper : numpy.ndarray
        Finite bounds in deg/km, guaranteed ``lower <= upper`` everywhere.
    metadata : dict
        Provenance and the accounting that matters for interpreting the result:
        ``n_gates``, ``n_missing_moments`` (gates where Z or ZDR was absent so
        the box was released), ``n_crossed`` (gates where the reflectivity cap
        fell below the self-consistency floor and the lower bound was clipped to
        the upper --- these are the hail-core gates Huang warns about, and a
        large count means the box is doing nothing there), ``n_rain``, and the
        settings used.
    """
    z_dbz = np.asarray(z_dbz, dtype=float)
    zdr_db = np.asarray(zdr_db, dtype=float)
    if z_dbz.shape != zdr_db.shape:
        raise ValueError(
            f"z_dbz shape {z_dbz.shape} does not match zdr_db shape {zdr_db.shape}"
        )
    shape = z_dbz.shape

    floor = sign_floor(
        shape,
        policy=sign_policy,
        height_km=height_km,
        iso0_km=iso0_km,
        blend_km=blend_km,
        negative_floor=negative_floor,
    )

    kdp_sc = self_consistency_kdp(z_dbz, zdr_db, band=band, coefficients=coefficients)
    missing = ~np.isfinite(kdp_sc)

    lower = np.where(missing, floor, lower_factor * kdp_sc)
    upper = np.where(missing, float(non_rain_cap), upper_factor * kdp_sc)

    if lsf_window_km is not None:
        if phidp is None or dr_km is None:
            raise ValueError("lsf_window_km requires both phidp and dr_km")
        lsf = lsf_kdp(phidp, dr_km, lsf_window_km)
        lsf_bound = lsf_factor * lsf
        lower = np.where(np.isfinite(lsf_bound), np.maximum(lower, lsf_bound), lower)

    cap = reflectivity_cap(z_dbz, ladder=ladder, cap_above=cap_above)
    upper = np.minimum(upper, cap)

    if rain_mask is not None:
        rain_mask = np.asarray(rain_mask, dtype=bool)
        if rain_mask.shape != shape:
            raise ValueError(
                f"rain_mask shape {rain_mask.shape} does not match {shape}"
            )
        lower = np.where(rain_mask, lower, floor)
        upper = np.where(rain_mask, upper, float(non_rain_cap))
        n_rain = int(rain_mask.sum())
    else:
        n_rain = int(np.prod(shape))

    # The sign policy is a floor on the floor: it may raise the lower bound but
    # never lower it, so an altitude-conditioned policy cannot be undone by a
    # self-consistency value that happens to be negative.
    lower = np.maximum(lower, floor)

    crossed = lower > upper
    lower = np.where(crossed, upper, lower)

    metadata = {
        "n_gates": int(np.prod(shape)),
        "n_missing_moments": int(missing.sum()),
        "n_crossed": int(crossed.sum()),
        "n_rain": n_rain,
        "rain_mask_supplied": rain_mask is not None,
        "band": band,
        "coefficients": tuple(
            float(c)
            for c in (
                SELF_CONSISTENCY_COEFFICIENTS[band]
                if coefficients is None
                else coefficients
            )
        ),
        "lower_factor": float(lower_factor),
        "upper_factor": float(upper_factor),
        "ladder": tuple(tuple(float(v) for v in rung) for rung in ladder),
        "cap_above": None if cap_above is None else float(cap_above),
        "non_rain_cap": float(non_rain_cap),
        "lsf_window_km": None if lsf_window_km is None else float(lsf_window_km),
        "lsf_factor": float(lsf_factor),
        "sign_policy": sign_policy,
        "iso0_km": None if iso0_km is None else float(iso0_km),
        "blend_km": float(blend_km),
        "negative_floor": float(negative_floor),
    }
    return lower, upper, metadata
