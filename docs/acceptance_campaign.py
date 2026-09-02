"""Acceptance campaign for radar_palette.phase on real case-library volumes.

Scores the new bounded-L1 retrieval AND ARM's shipped specific_differential_phase
on the same gates, with the kdp-retrieval-cautionary-tale tests plus the 1.18
KDP-core / Z-core width-ratio prediction from kdp-physics-anchor.

Run:  PYTHONPATH=rp/src python acceptance_campaign.py
"""
import json
import os
import sys
import warnings

import numpy as np
import pyart

warnings.filterwarnings("ignore")

ZDR_OFFSET_DB = 0.84  # BNF, vertical sounding, Ryzhkov et al. 2005 Eq (29)
SMOOTHING_KM = 1.0  # matched to the 0.75-1.0 km measured DSD scale
RHOHV_MIN = 0.95
ISO0_KM_FALLBACK = 3.2  # sonde 0 degC not available for these dates

CASES = json.load(open("acceptance_cases.json"))


def lowest_sweep(radar):
    return radar.extract_sweeps([int(np.argmin(radar.fixed_angle["data"]))])


def gap_fill(phi, good):
    """Linear interpolation of phase across masked gaps, per ray.

    Offset-equivariant, so it preserves the retrieval's invariance to an
    additive phase constant; done here rather than inside the solver so the
    invented phase is visible in the caller.
    """
    out = phi.copy()
    for i in range(phi.shape[0]):
        g = good[i]
        if g.sum() < 2:
            continue
        idx = np.flatnonzero(g)
        out[i, idx[0] : idx[-1] + 1] = np.interp(
            np.arange(idx[0], idx[-1] + 1), idx, phi[i, idx]
        )
    return out


def run_case(path, mode, phase_mod):
    radar = lowest_sweep(pyart.io.read(path))
    dr_km = float(np.diff(radar.range["data"][:2])[0]) / 1000.0
    z = radar.fields["reflectivity"]["data"].filled(np.nan)
    zdr = radar.fields["differential_reflectivity"]["data"].filled(np.nan) + ZDR_OFFSET_DB
    rho = radar.fields["copol_correlation_coeff"]["data"].filled(0.0)
    psi = radar.fields["uncorrected_differential_phase"]["data"].filled(np.nan)
    arm = radar.fields["specific_differential_phase"]["data"].filled(np.nan)
    height_km = radar.gate_altitude["data"] / 1000.0

    good = np.isfinite(psi) & (rho >= RHOHV_MIN) & np.isfinite(z) & (z > 5.0)
    rain = good & (height_km < ISO0_KM_FALLBACK)

    sysinfo = phase_mod.system_phidp(psi, range_m=radar.range["data"], method="block",
                                     rhohv=rho, rhohv_min=RHOHV_MIN)
    unfolded, foldmeta = phase_mod.unfold_phidp(psi, mask=good)
    phi, _applied = phase_mod.apply_system_phidp(unfolded, sysinfo)

    bounds_out = phase_mod.kdp_bounds(
        z, zdr, band="C", rain_mask=rain, height_km=height_km,
        iso0_km=ISO0_KM_FALLBACK, sign_policy="altitude_conditioned",
    )
    if isinstance(bounds_out, dict):
        lower, upper = bounds_out["lower"], bounds_out["upper"]
        bmeta = bounds_out
    else:
        lower, upper, bmeta = bounds_out

    phi_filled = gap_fill(phi, good)
    weights = good.astype(float)
    kdp, phi_fit, meta = phase_mod.kdp_linear_program(
        phi_filled, lower, upper, dr_km, SMOOTHING_KM, weights=weights,
    )
    return dict(
        path=path, mode=mode, dr_km=dr_km, n_rays=int(z.shape[0]),
        sysphi=float(sysinfo["sysphi"]), n_folds=int(foldmeta.get("n_confirmed", 0)),
        realised_km=float(meta.get("realised_smoothing_km", np.nan)),
        n_solved=int(meta.get("n_solved", 0)),
        n_crossed=int(bmeta.get("n_crossed", 0)),
    ), kdp, arm, z, rain, height_km, dr_km


def score(kdp, z, rain, height_km, dr_km, label):
    amp = kdp_amplitude_vs_z(kdp, z, rain)
    sca = kdp_scale_vs_z(kdp, z, rain, dr_km)
    stp = steplikeness(kdp, np.isfinite(kdp) & rain)
    sgn = sign_by_altitude(kdp, height_km, ISO0_KM_FALLBACK)
    return dict(label=label, monotonic=bool(amp.get("monotonic")),
                frac_light_above=float(amp.get("frac_light_rain_above_thresh", np.nan)),
                corr_z=float(amp.get("corr_kdp_z", np.nan)),
                L_kdp=float(sca.get("kdp_length_km", np.nan)), L_z=float(sca.get("z_length_km", np.nan)),
                L_ratio=float(sca.get("ratio", np.nan)),
                d2_over_d1=float(stp.get("d2_over_d1", np.nan)),
                frac_repeat=float(stp.get("frac_repeat", np.nan)),
                frac_neg_below=float(sgn.get("frac_neg_below", np.nan)),
                frac_neg_aloft=float(sgn.get("frac_neg_aloft", np.nan)))
