"""Differential-phase processing: system offset, unfolding, and bounded KDP.

Four independently testable stages, each usable on its own:

``radar_palette.phase.system``
    System differential phase. Three estimators following the wradlib
    algorithms, reimplemented natively. Both a per-ray and a sweep-level
    estimate are returned as first-class outputs, together with a ray-to-ray
    consistency diagnostic that says which of the two to apply.

``radar_palette.phase.unfold``
    Fold detection and repair. Folds are distinguished from backscatter-phase
    excursions by *persistence*, not by step size, because those two things look
    identical to :func:`numpy.unwrap` and only one of them should be corrected.

``radar_palette.phase.bounds``
    Two-sided KDP bounds from Z and ZDR via a T-matrix self-consistency
    relation, a phase-only least-squares floor, and a reflectivity cap. The
    relation is released outside a caller-supplied rain mask, and the sign
    policy is altitude-conditioned rather than globally non-negative.

``radar_palette.phase.linear_program``
    The retrieval: an L1 phase fit subject to a two-sided box on the KDP the
    fitted phase implies, after Huang et al. (2016).

The governing assumption
------------------------
All KDP is a retrieval. A radar measures power, a power ratio and a phase; KDP
is defined as half the range gradient of the *propagation* differential phase,
and what the receiver reports is a mixture::

    Psi_DP(measured) = Phi_propagation + delta(backscatter)
                       + non-uniform beam filling + noise

Nothing in this subpackage can separate those terms on a single ray, and no
shipped KDP field --- including this one --- is a measurement. What this
subpackage does instead is *bound* the answer using Z and ZDR, which are direct
measurements carrying no ``delta`` term. Structure claims about a retrieved KDP
field should be anchored on Z and ZDR for the same reason.

Everything here works on plain ``(n_rays, n_gates)`` arrays and returns plain
arrays and dicts. Unlike :mod:`radar_palette.gateid` there is no Py-ART /
xradar object-flavour entry point or accessor yet; that layer is deliberately
left for a follow-up so that each numerical stage could be tested against
synthetic truth first.

Examples
--------
Estimate and remove the system offset, checking ray-to-ray consistency::

    from radar_palette.phase import apply_system_phidp, system_phidp

    estimate = system_phidp(
        phidp, range_m, method="block", rhohv=rhohv,
        block_length_m=1000.0, sigma_phi_deg=1.8,
    )
    estimate["consistency"]["per_ray_justified"]  # False -> use the sweep value
    corrected, offset = apply_system_phidp(phidp, estimate)

Measure folds before repairing them::

    from radar_palette.phase import fold_statistics, unfold_phidp

    stats = fold_statistics(corrected, wrap_deg=360.0)
    stats["persistent_fraction_of_large"]  # near 0 -> excursions, do not unfold
    unfolded, fold_meta = unfold_phidp(corrected, wrap_deg=360.0)

Retrieve KDP inside a two-sided box::

    from radar_palette.phase import kdp_bounds, kdp_linear_program, rain_mask

    rain = rain_mask(z_dbz=z, rhohv=rhohv, height_km=height, iso0_km=3.17)
    low, high, bound_meta = kdp_bounds(
        z, zdr, phidp=unfolded, dr_km=0.1, band="C", rain_mask=rain,
        lsf_window_km=8.0, height_km=height, iso0_km=3.17,
    )
    kdp, phi_fit, meta = kdp_linear_program(
        unfolded, low, high, dr_km=0.1, smoothing_km=1.0,
    )
"""

from __future__ import annotations

from radar_palette.phase import bounds, linear_program, system, unfold
from radar_palette.phase.bounds import (
    BRANDES_2002_AXIS_RATIO,
    DEFAULT_LOWER_FACTOR,
    DEFAULT_REFLECTIVITY_CAP_LADDER,
    DEFAULT_UPPER_FACTOR,
    GOURLEY_2009_TABLE4,
    SELF_CONSISTENCY_COEFFICIENTS,
    SELF_CONSISTENCY_ZDR_ROOTS,
    SIGN_POLICIES,
    kdp_bounds,
    lsf_kdp,
    reflectivity_cap,
    self_consistency_kdp,
    sign_floor,
)
from radar_palette.phase.bounds import rain_mask_from_context as rain_mask
from radar_palette.phase.linear_program import (
    RAY_STATUSES,
    derivative_matrix,
    kdp_linear_program,
    savgol_derivative_weights,
    smoothing_length_gates,
    solve_ray,
)
from radar_palette.phase.system import (
    DEFAULT_RHOHV_MIN,
    SYSTEM_PHIDP_METHODS,
    apply_system_phidp,
    ray_consistency,
    system_phidp,
    system_phidp_block,
    system_phidp_first,
    system_phidp_hist,
    valid_phase_mask,
)
from radar_palette.phase.unfold import (
    DEFAULT_CONFIRM_GATES,
    DEFAULT_WRAP_DEG,
    UNFOLD_DIRECTIONS,
    detect_folds,
    fold_statistics,
    unfold_phidp,
)

__all__ = [
    "BRANDES_2002_AXIS_RATIO",
    "DEFAULT_CONFIRM_GATES",
    "DEFAULT_LOWER_FACTOR",
    "DEFAULT_REFLECTIVITY_CAP_LADDER",
    "DEFAULT_RHOHV_MIN",
    "DEFAULT_UPPER_FACTOR",
    "DEFAULT_WRAP_DEG",
    "GOURLEY_2009_TABLE4",
    "RAY_STATUSES",
    "SELF_CONSISTENCY_COEFFICIENTS",
    "SELF_CONSISTENCY_ZDR_ROOTS",
    "SIGN_POLICIES",
    "SYSTEM_PHIDP_METHODS",
    "UNFOLD_DIRECTIONS",
    "apply_system_phidp",
    "bounds",
    "derivative_matrix",
    "detect_folds",
    "fold_statistics",
    "kdp_bounds",
    "kdp_linear_program",
    "linear_program",
    "lsf_kdp",
    "rain_mask",
    "ray_consistency",
    "reflectivity_cap",
    "savgol_derivative_weights",
    "self_consistency_kdp",
    "sign_floor",
    "smoothing_length_gates",
    "solve_ray",
    "system",
    "system_phidp",
    "system_phidp_block",
    "system_phidp_first",
    "system_phidp_hist",
    "unfold",
    "unfold_phidp",
    "valid_phase_mask",
]
