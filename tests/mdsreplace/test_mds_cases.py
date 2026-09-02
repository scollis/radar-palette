"""Case-library checks on real PPI volumes.

Synthetic tests pin the arithmetic; these pin the claims the module makes about
real radars, on volumes from ``~/data/radar_cases``. They skip cleanly where
that library is not present, and are marked ``slow`` because the C-SAPR2
volumes are hundreds of megabytes.

Three volumes, deliberately, because each exercises a different estimator and
the differences between them are the point.

C-SAPR2 2026 (ARM BNF, ``a1``-style moments)
    Reports a value at every gate and carries an SNR field. Both the exact
    ``snr`` route and the ``noise`` mode work, so this is where they can be
    checked against each other.

C-SAPR2 2025 (ARM BNF, censor-masked, noise-subtracted)
    Also carries SNR, but noise subtraction has destroyed the sharp noise
    peak. The per-range mode, expressed at 1 km, spans 11 dB across range on
    this volume (-34.6 dBZ inside 10 km, -45.7 dBZ at 70-80 km, -40.7 dBZ at
    110 km) where a floor should be flat, and the peak it finds holds only
    11% of the samples -- two separate diagnostics, both reported. This is
    the volume that made the ``snr`` route the default rather than a variant.

WSR-88D (KLOT, Level II)
    Delivered thresholded, with no SNR field and no noise population at all,
    so the censoring estimator has to carry it alone.
"""

from __future__ import annotations

import glob

import numpy as np
import pytest

pyart = pytest.importorskip("pyart")

from radar_palette.gateid import gate_id, meteorological_gatefilter  # noqa: E402
from radar_palette.mdsreplace import mds_profile, replace_below_mds  # noqa: E402

CASE_ROOT = "/Users/scollis/data/radar_cases"

#: Fields a floor estimate needs, plus what the gate-ID classifier consumes.
#: Reading the whole C-SAPR2 field list would triple the memory for nothing.
BNF_FIELDS = [
    "uncorrected_reflectivity_h",
    "uncorrected_copol_correlation_coeff",
    "uncorrected_normalized_coherent_power",
    "uncorrected_signal_to_noise_ratio_copolar_h",
    "uncorrected_mean_doppler_velocity_h",
    "uncorrected_differential_reflectivity",
    "uncorrected_differential_phase",
    "uncorrected_spectral_width_h",
]

#: BNF is in northern Alabama; a warm-season freezing level near 4.7 km lets
#: the melting-layer constraints engage. The floor estimate does not depend on
#: it -- only the classification does.
BNF_FREEZING_LEVEL_M = 4700.0


def _first_usable(pattern, read, usable):
    """The smallest volume matching ``pattern`` that ``usable`` accepts.

    Smallest first to keep the suite quick, but "smallest" alone is not
    enough: the library holds a 6 MB BNF PPI file with 89 rays across 15
    sweeps, which reads fine and cannot support any range statistic.
    """
    import os

    matches = sorted(glob.glob(pattern), key=os.path.getsize)
    if not matches:
        pytest.skip(f"no case-library volumes matching {pattern}")
    for path in matches:
        radar = read(path)
        if usable(radar):
            return radar
        del radar
    pytest.skip(f"no volume matching {pattern} carries a usable PPI geometry")


def _full_ppi(radar, min_sweeps=3, min_rays=500):
    return radar.nsweeps >= min_sweeps and radar.nrays >= min_rays


@pytest.fixture(scope="module")
def bnf_2026():
    """A 2026 BNF PPI volume: every gate reported, SNR present, no censoring."""
    return _first_usable(
        f"{CASE_ROOT}/bnf/*/bnf_ppi_csapr2_*_2026*.nc",
        lambda path: pyart.io.read(path, include_fields=BNF_FIELDS),
        lambda radar: _full_ppi(radar, min_sweeps=5),
    )


@pytest.fixture(scope="module")
def bnf_2025():
    """A 2025 BNF PPI volume, from the censor-masked processing level."""
    return _first_usable(
        f"{CASE_ROOT}/bnf/*/bnf_ppi_csapr2_*_2025*.nc",
        lambda path: pyart.io.read(
            path, include_fields=BNF_FIELDS + ["signal_to_noise_ratio_copolar_h"]
        ),
        _full_ppi,
    )


@pytest.fixture(scope="module")
def klot_ppi():
    return _first_usable(
        f"{CASE_ROOT}/klot/*/klot_ppi_wsr88d_*", pyart.io.read, _full_ppi
    )


# --------------------------------------------------------------------------
# C-SAPR2 2026: the exact route, and the mode as a check on it
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_csapr2_floor_is_one_power_law_in_range(bnf_2026):
    profile = mds_profile(bnf_2026)
    # An SNR field is present, so the exact route is what auto should choose.
    assert profile.method == "snr"
    assert profile.meta["snr_field"] is not None
    # The whole premise of the module: one intercept, one slope, small residual.
    assert profile.residual_rms_db < 0.5
    assert profile.slope_db_per_decade == pytest.approx(20.0)
    # A C-SAPR2 floor near -40 dBZ at 1 km is about -20 at 10 km and 0 at
    # 100 km. Wide bounds: this is an instrument property, not a constant.
    assert -50.0 < profile.intercept_dbz < -30.0
    # Z - SNR is a near-deterministic quantity within a range gate.
    assert profile.meta["snr_spread_db_median"] < 1.0


@pytest.mark.slow
def test_csapr2_snr_and_mode_routes_agree_on_an_unsubtracted_volume(bnf_2026):
    """Two independent estimators, one receiver.

    The mode sits *above* the 0 dB SNR floor, because the noise distribution
    peaks above its own zero-SNR point -- 1.2 dB and 2.3 dB on the two 2026
    volumes measured. What matters is the sign and the scale: the two
    estimators bracket the floor within a few dB, rather than differing by
    the 5 dB an SNR-thresholded average produced on this same instrument.
    """
    exact = mds_profile(bnf_2026)
    mode = mds_profile(bnf_2026, method="noise")
    assert mode.meta["mode_fraction_median"] > 0.2
    assert mode.intercept_dbz == pytest.approx(exact.intercept_dbz, abs=3.0)
    assert mode.intercept_dbz > exact.intercept_dbz
    assert exact.meta["cross_check_noise_mode_dbz"] == pytest.approx(
        mode.intercept_dbz, abs=0.01
    )


@pytest.mark.slow
def test_csapr2_free_slope_confirms_the_r_squared_law(bnf_2026):
    """Released, the slope should land near the physical 20 dB/decade."""
    profile = mds_profile(bnf_2026, fit_slope=True)
    assert profile.slope_db_per_decade == pytest.approx(20.0, abs=1.5)


@pytest.mark.slow
def test_csapr2_mode_route_finds_and_rejects_near_range_contamination(bnf_2026):
    """Close-range ground echo is not the floor, and the fit must say so."""
    profile = mds_profile(bnf_2026, method="noise")
    assert profile.meta["fit_n_trimmed"] > 0
    assert profile.meta["fit_min_valid_range_m"] > 3000.0
    assert any("rejected as inconsistent" in w for w in profile.meta["warnings"])


@pytest.mark.slow
def test_no_return_gates_can_be_selected_by_gate_id_or_by_snr_threshold(bnf_2026):
    """Both selections feed the mode, and the mode is indifferent to which."""
    classified, _ = gate_id(bnf_2026, freezing_level_m=BNF_FREEZING_LEVEL_M)
    from_gate_id = mds_profile(classified, method="noise")
    from_snr_cut = mds_profile(classified, method="noise", gate_id_field=None)
    from_everything = mds_profile(
        classified, method="noise", gate_id_field=None, snr_max=None
    )
    assert from_gate_id.meta["mask_source"].startswith("gate_id")
    assert "signal_to_noise" in from_snr_cut.meta["mask_source"]
    assert from_everything.meta["mask_source"] == "all_gates"
    assert from_snr_cut.intercept_dbz == pytest.approx(
        from_gate_id.intercept_dbz, abs=0.3
    )
    assert from_everything.intercept_dbz == pytest.approx(
        from_gate_id.intercept_dbz, abs=0.3
    )


@pytest.mark.slow
def test_per_sweep_estimates_agree_with_the_pooled_one(bnf_2026):
    """The floor is a receiver property, so it must not vary sweep to sweep.

    Measured spread on a 15-sweep C-SAPR2 volume was 0.53 dB, the lowest
    sweeps sitting marginally high where ground echo enters. Pooling is the
    default because on the population routes the lowest sweeps can lack the
    no-return gates to fit at all.
    """
    pooled = mds_profile(bnf_2026)
    per_sweep = [
        mds_profile(bnf_2026, sweeps=[sweep]).intercept_dbz
        for sweep in range(bnf_2026.nsweeps)
    ]
    assert np.allclose(per_sweep, pooled.intercept_dbz, atol=1.0)


@pytest.mark.slow
def test_filling_a_real_volume_leaves_no_holes_and_marks_every_gate(bnf_2026):
    classified, _ = gate_id(bnf_2026, freezing_level_m=BNF_FREEZING_LEVEL_M)
    gatefilter = meteorological_gatefilter(classified)
    profile = mds_profile(classified)
    filled, meta = replace_below_mds(
        classified, gatefilter, profile=profile, mds_field="mds"
    )

    values = np.ma.filled(
        filled.fields["uncorrected_reflectivity_h"]["data"].astype("f8"), np.nan
    )
    flag = np.ma.filled(filled.fields["mds_replaced"]["data"], 0).astype(bool)
    mds = np.ma.filled(filled.fields["mds"]["data"].astype("f8"), np.nan)

    assert np.isfinite(values).all()
    assert flag.sum() == meta["n_replaced"]
    assert np.allclose(values[flag], mds[flag], atol=1e-3)
    # The gates that survived the filter are still measurements, and at least
    # some of them are weather well above the floor.
    assert values[~flag].max() > 20.0
    # The volume's own fill sentinel was found and replaced along with the rest.
    assert not (values == -80.0).any()


@pytest.mark.slow
def test_a_second_pass_on_a_filled_volume_recovers_the_same_floor(bnf_2026):
    """Idempotence, on the population route where it is not trivial.

    A filled volume repeats one exact value per range gate, which the sentinel
    guard has to decline to treat as fill flags.
    """
    first = mds_profile(bnf_2026, method="noise")
    filled, _ = replace_below_mds(bnf_2026, profile=first)
    second = mds_profile(filled, method="noise")
    assert second.intercept_dbz == pytest.approx(first.intercept_dbz, abs=0.2)


# --------------------------------------------------------------------------
# C-SAPR2 2025: where the mode fails and says so
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_noise_subtracted_processing_defeats_the_mode_and_reports_it(bnf_2025):
    """The measurement that decided the default estimator.

    On this processing level the mode is not a floor: it wanders with range
    and holds a small share of the samples. The module must not present that
    as an answer without saying so.
    """
    mode = mds_profile(bnf_2025, method="noise")
    assert mode.meta["mode_fraction_median"] < 0.2
    assert any("may not be the noise floor" in w for w in mode.meta["warnings"])
    assert mode.residual_rms_db > 1.0

    exact = mds_profile(bnf_2025)
    assert exact.method == "snr"
    assert exact.residual_rms_db < 1.0
    # The two disagree by several dB here, which is the point: the mode is the
    # one that is wrong, and its own diagnostics are what say so.
    assert abs(exact.intercept_dbz - mode.intercept_dbz) > 2.0


@pytest.mark.slow
def test_the_exact_route_is_stable_across_sweeps_on_the_2025_volume(bnf_2025):
    pooled = mds_profile(bnf_2025)
    per_sweep = [
        mds_profile(bnf_2025, sweeps=[sweep]).intercept_dbz
        for sweep in range(bnf_2025.nsweeps)
    ]
    assert np.allclose(per_sweep, pooled.intercept_dbz, atol=0.5)


# --------------------------------------------------------------------------
# WSR-88D: the censoring route
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_wsr88d_is_recognised_as_thresholded_and_measured_anyway(klot_ppi):
    profile = mds_profile(klot_ppi)
    assert profile.method == "censor"
    assert profile.meta["missing_fraction"] > 0.5
    assert -50.0 < profile.intercept_dbz < -25.0
    # A censoring boundary is a processing decision and wanders more than a
    # receiver floor, but it is still close to one power law.
    assert profile.residual_rms_db < 3.0


@pytest.mark.slow
def test_wsr88d_floor_rises_through_zero_dbz_at_long_range(klot_ppi):
    """The hazard the cap exists for, on a real volume.

    Filling non-detections at 250 km with the honest MDS writes light-rain
    reflectivities into empty space.
    """
    profile = mds_profile(klot_ppi)
    far = float(profile.evaluate([250e3])[0])
    assert far > 0.0
    capped = mds_profile(klot_ppi, cap_dbz=-10.0)
    assert float(capped.evaluate([250e3])[0]) == pytest.approx(-10.0)


@pytest.mark.slow
def test_wsr88d_filling_needs_no_classification_at_all(klot_ppi):
    """Masked gates alone drive the fill on a thresholded volume."""
    filled, meta = replace_below_mds(klot_ppi)
    assert meta["n_replaced_by_source"]["gatefilter"] == 0
    assert meta["n_replaced_by_source"]["masked"] > 0
    assert meta["replaced_fraction"] > 0.5
    values = np.ma.filled(filled.fields["reflectivity"]["data"].astype("f8"), np.nan)
    assert np.isfinite(values).all()
