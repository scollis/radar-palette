"""Tests for the single-volume gate-ID physics core.

These pin the structural defects the classifier exists to avoid (budget
inversion, silent argmax tie-breaks, classes winning on absent evidence) and
the physics rules derived from measurements on real volumes rather than chosen
by hand. Ported from the standalone package that preceded this module.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.gateid import single as core


def test_trapmf_matches_reference():
    x = np.linspace(-5, 15, 501)
    for pts in ([0, 0, 2.0, 2.1], [0.97, 0.98, 1, 1], [-2, 2, 1000, 1000]):
        a, b, c, d = pts
        up = (x - a) / (b - a) if b > a else np.where(x >= a, np.inf, -np.inf)
        dn = (d - x) / (d - c) if d > c else np.where(x <= d, np.inf, -np.inf)
        ref = np.clip(np.minimum(np.minimum(up, dn), 1.0), 0.0, 1.0)
        assert np.allclose(core.trapmf(x, pts), ref)


def test_trapmf_nan_is_zero_evidence():
    y = core.trapmf(np.array([np.nan, 1.0]), [0, 0.5, 1.5, 2])
    assert y[0] == 0.0 and y[1] > 0


def test_trapmf_bounded_unit_interval():
    x = np.linspace(-100, 100, 1001)
    y = core.trapmf(x, [-2, 2, 10, 20])
    assert y.min() >= 0.0 and y.max() <= 1.0


# --------------------------------------------------------------------------
# the structural defects this package exists to avoid
# --------------------------------------------------------------------------


def test_no_evidence_is_unclassified_not_class_zero():
    """A gate with no positive evidence must not be silently tie-broken."""
    n = 50
    feats = {
        "z": np.full(n, np.nan),
        "rhohv": np.full(n, np.nan),
        "snr": np.full(n, np.nan),
        "zdr": np.full(n, np.nan),
    }
    codes, info = core.classify(feats, snr_min=None, min_run=1)
    assert (codes == 0).all()
    assert core.CLASSES[0] == "unclassified"
    assert (info["top_score"] == 0).all()


def test_scores_are_budget_normalised():
    """No class may win merely by having more evidence terms defined.

    This is the operational defect being guarded against: an additive scheme
    where 'snow' can attain 4.0 and 'rain' only 3.0 sends every precipitating
    gate to snow regardless of membership-function tuning.
    """
    avail = {
        "z",
        "zdr",
        "rhohv",
        "snr",
        "temp",
        "hoi0_km",
        "sw",
        "rhohv_texture",
        "phidp_texture",
        "vel_texture",
        "vel_norm",
        "ncp",
        "range_km",
        "height_km",
    }
    for cls, terms in core.MEMBERSHIP.items():
        b = core._budget(terms, avail)
        assert b > 0, cls
    n = 200
    rng = np.random.default_rng(0)
    feats = {
        "z": rng.uniform(-10, 60, n),
        "zdr": rng.uniform(-2, 6, n),
        "rhohv": rng.uniform(0.2, 1.0, n),
        "snr": rng.uniform(-5, 60, n),
        "temp": rng.uniform(-40, 30, n),
        "hoi0_km": rng.uniform(-5, 8, n),
    }
    _, info = core.classify(feats, snr_min=None, min_run=1)
    # Normalised scores are fractions of an attainable budget.
    assert info["scores"].max() <= 1.0 + 1e-9


def test_rain_reachable_without_temperature():
    """Rain must have a temperature-free evidence path.

    Most BNF case dates have no on-site sounding within hours. If rain can
    only score via a temperature term, it loses everywhere the sounding is
    missing -- which is how the operational scheme came to label 84% of
    Z>=40 dBZ gates as snow.
    """
    n = 100
    feats = {
        "z": np.full(n, 35.0),
        "rhohv": np.full(n, 0.99),
        "zdr": np.full(n, 1.0),
        "rhohv_texture": np.full(n, 0.005),
    }
    codes, _ = core.classify(feats, snr_min=None, min_run=1)
    names = {core.CLASSES[c] for c in np.unique(codes)}
    assert names & set(core.LIQUID), f"no liquid class reachable, got {names}"


def test_hard_constraint_forbids_snow_below_melting_layer():
    n = 20
    feats = {
        "z": np.full(n, 30.0),
        "rhohv": np.full(n, 0.99),
        "zdr": np.full(n, 0.3),
        "temp": np.full(n, 12.0),
        "hoi0_km": np.full(n, -3.0),
    }
    codes, _ = core.classify(feats, snr_min=None, min_run=1)
    assert not np.isin([core.CLASSES[c] for c in codes], core.FROZEN).any()


def test_strong_echo_below_melting_layer_is_liquid():
    n = 40
    feats = {
        "z": np.full(n, 48.0),
        "rhohv": np.full(n, 0.985),
        "zdr": np.full(n, 1.6),
        "snr": np.full(n, 45.0),
        "temp": np.full(n, 18.0),
        "hoi0_km": np.full(n, -3.5),
        "rhohv_texture": np.full(n, 0.004),
    }
    codes, _ = core.classify(feats, min_run=1)
    got = {core.CLASSES[c] for c in np.unique(codes)}
    assert got & set(core.LIQUID), got


def test_class_codes_are_stable():
    """Codes are written into files; they may be appended to, never renumbered."""
    expected = {
        0: "unclassified",
        1: "no_scatter",
        2: "clutter",
        3: "biological",
        4: "multi_trip",
        5: "light_rain",
        6: "moderate_rain",
        7: "heavy_rain",
        8: "melting_wet",
        9: "ice_snow",
        10: "graupel_hail",
    }
    assert core.CLASSES == expected


# --------------------------------------------------------------------------
# texture
# --------------------------------------------------------------------------


def test_angular_texture_handles_folding():
    """Velocity alternating across the branch cut is coherent, not turbulent."""
    nyq = 16.0
    a = np.full((40, 40), nyq - 0.05)
    b = np.full((40, 40), -nyq + 0.05)
    folded = np.where((np.arange(40) % 2)[:, None], a, b)
    ang = core.angular_texture(folded, nyq)
    plain = core.texture(folded)
    assert np.nanmedian(ang) < np.nanmedian(plain) / 5


def test_texture_needs_enough_neighbours():
    """Gates with too few valid neighbours must not get a fabricated texture.

    A single valid sample in a field of NaN can only produce a texture where
    edge replication manufactures neighbours; everywhere genuinely isolated
    must be NaN.
    """
    d = np.full((10, 10), np.nan)
    d[0, 0] = 5.0
    t = core.texture(d)
    assert np.isnan(t).mean() > 0.9
    # Whatever survives edge replication carries no spread, so it cannot be
    # mistaken for real variability.
    assert np.nanmax(t) == 0.0

    # A genuinely isolated interior sample yields nothing at all.
    d2 = np.full((10, 10), np.nan)
    d2[5, 5] = 5.0
    assert np.isnan(core.texture(d2)).all()


# --------------------------------------------------------------------------
# Py-ART / xradar parity
# --------------------------------------------------------------------------


def _cold_anvil(n=60, temp=-25.0):
    """Weak, high, cold, high-rhohv echo -- a storm anvil."""
    return {
        "z": np.full(n, 18.0),
        "rhohv": np.full(n, 0.985),
        "zdr": np.full(n, 0.4),
        "snr": np.full(n, 25.0),
        "temp": np.full(n, temp),
        "hoi0_km": np.full(n, 4.5),
        "height_km": np.full(n, 9.0),
        "rhohv_texture": np.full(n, 0.008),
    }


def test_anvil_is_ice_not_melting_or_hail():
    """Above the freezing level, spreading anvil echo must classify as ice.

    Regression test. An inverted hard constraint previously forbade ice ABOVE
    the melting layer, so the anvil came out as melting_wet or graupel_hail.
    Nothing melts at -25 C.
    """
    codes, _ = core.classify(_cold_anvil(), min_run=1)
    names = {core.CLASSES[c] for c in np.unique(codes)}
    assert names == {"ice_snow"}, names


def test_melting_confined_to_the_bright_band():
    """melting_wet may not appear well above or well below the 0 C level."""
    for temp, hoi0 in ((-20.0, 4.0), (-10.0, 2.5), (12.0, -3.0)):
        n = 30
        feats = {
            "z": np.full(n, 35.0),
            "rhohv": np.full(n, 0.90),
            "zdr": np.full(n, 2.0),
            "snr": np.full(n, 30.0),
            "temp": np.full(n, temp),
            "hoi0_km": np.full(n, hoi0),
            "rhohv_texture": np.full(n, 0.02),
        }
        codes, _ = core.classify(feats, min_run=1)
        got = {core.CLASSES[c] for c in np.unique(codes)}
        assert "melting_wet" not in got, f"melting at {temp} C: {got}"


def test_melting_allowed_at_the_bright_band():
    n = 30
    feats = {
        "z": np.full(n, 35.0),
        "rhohv": np.full(n, 0.90),
        "zdr": np.full(n, 2.0),
        "snr": np.full(n, 30.0),
        "temp": np.full(n, 0.5),
        "hoi0_km": np.full(n, -0.2),
        "rhohv_texture": np.full(n, 0.02),
    }
    codes, _ = core.classify(feats, min_run=1)
    assert "melting_wet" in {core.CLASSES[c] for c in np.unique(codes)}


def test_no_biology_in_subfreezing_air():
    """Insects and birds do not fly below 3 C, whatever the polarimetry says."""
    n = 40
    feats = {
        "z": np.full(n, 8.0),
        "zdr": np.full(n, 5.0),
        "rhohv": np.full(n, 0.80),
        "rhohv_texture": np.full(n, 0.15),
        "snr": np.full(n, 20.0),
        "sw": np.full(n, 3.0),
        "phidp_texture": np.full(n, 90.0),
        "zdr_texture": np.full(n, 3.0),
        "z_texture": np.full(n, 8.0),
    }
    for temp in (-30.0, -10.0, 0.0, 2.9):
        f = dict(feats, temp=np.full(n, temp))
        codes, _ = core.classify(f, min_run=1)
        got = {core.CLASSES[c] for c in np.unique(codes)}
        assert "biological" not in got, f"biology at {temp} C: {got}"
    # ... but it is available in warm air.
    f = dict(feats, temp=np.full(n, 22.0), height_km=np.full(n, 0.7))
    codes, _ = core.classify(f, min_run=1)
    assert "biological" in {core.CLASSES[c] for c in np.unique(codes)}


def test_hail_needs_a_strong_core():
    n = 30
    feats = {
        "z": np.full(n, 25.0),
        "zdr": np.full(n, 0.2),
        "rhohv": np.full(n, 0.97),
        "snr": np.full(n, 30.0),
        "temp": np.full(n, -8.0),
        "hoi0_km": np.full(n, 2.0),
    }
    codes, _ = core.classify(feats, min_run=1)
    assert "graupel_hail" not in {core.CLASSES[c] for c in np.unique(codes)}


# --------------------------------------------------------------------------
# speckle suppression
# --------------------------------------------------------------------------


def test_noise_floor_forces_no_scatter():
    n = 50
    feats = {
        "z": np.full(n, 12.0),
        "rhohv": np.full(n, 0.97),
        "zdr": np.full(n, 0.5),
        "snr": np.full(n, -5.0),
        "temp": np.full(n, 15.0),
    }
    codes, info = core.classify(feats, snr_min=3.0, min_run=1)
    assert (codes == core.NAME_TO_CODE["no_scatter"]).all()
    assert info["n_noise_floor"] == n


def test_despeckle_removes_short_runs():
    """A two-gate island inside a uniform background is not resolvable."""
    shape = (4, 20)
    rain = core.NAME_TO_CODE["moderate_rain"]
    ice = core.NAME_TO_CODE["ice_snow"]
    grid = np.full(shape, rain, dtype="i2")
    grid[:, 8:10] = ice  # 2-gate island -> dissolved
    grid[:, 14:19] = ice  # 5-gate run    -> kept
    out = core.despeckle(grid.ravel(), shape, min_run=3).reshape(shape)
    assert (out[:, 8:10] == rain).all()
    assert (out[:, 14:19] == ice).all()


def test_despeckle_preserves_long_runs_exactly():
    shape = (3, 12)
    rain = core.NAME_TO_CODE["light_rain"]
    grid = np.full(shape, rain, dtype="i2")
    out = core.despeckle(grid.ravel(), shape, min_run=3)
    assert (out == grid.ravel()).all()


def test_velocity_texture_gate_rejects_incoherent_gates():
    n = 40
    feats = {
        "z": np.full(n, 15.0),
        "rhohv": np.full(n, 0.97),
        "zdr": np.full(n, 0.3),
        "snr": np.full(n, 30.0),
        "temp": np.full(n, 12.0),
        "vel_texture": np.full(n, 9.0),
    }
    codes, info = core.classify(feats, vel_texture_max=6.0, min_run=1)
    assert (codes == core.NAME_TO_CODE["no_scatter"]).all()
    assert info["n_noise_floor"] == n


def test_despeckle_skipped_without_shape_and_reported():
    """Without geometry, despeckling is skipped and flagged, never faked."""
    feats = {"z": np.full(10, 20.0), "rhohv": np.full(10, 0.98)}
    _, info = core.classify(feats, min_run=3)
    assert info["n_despeckled"] == -1


# --------------------------------------------------------------------------
# phase coherence: second trip and noise vs first-trip weather
# --------------------------------------------------------------------------


def test_uniform_phase_texture_limit():
    """Random-phase velocity must reach the v_nyq/sqrt(3) texture limit.

    This is the physical basis of the second-trip test: a magnetron's starting
    phase is random pulse to pulse, so an echo from beyond the unambiguous
    range has velocity uniformly distributed over +/-Nyquist.
    """
    nyq = 16.3
    rng = np.random.default_rng(0)
    incoherent = rng.uniform(-nyq, nyq, size=(80, 80))
    coherent = np.full((80, 80), 4.0) + rng.normal(0, 0.15, (80, 80))
    t_inc = np.nanmedian(core.angular_texture(incoherent, nyq))
    t_coh = np.nanmedian(core.angular_texture(coherent, nyq))
    limit = nyq / np.sqrt(3.0)
    assert 0.75 * limit < t_inc < 1.35 * limit, (t_inc, limit)
    assert t_coh < 0.1 * limit
    assert t_inc > 10 * t_coh


def test_incoherent_gates_without_power_are_rejected_as_noise():
    """Incoherent AND weak is receiver noise.

    This test previously asserted that incoherence alone was enough
    "whatever their SNR", which was the bug that made multi_trip
    unreachable: second-trip echo is incoherent too, and returned power is
    the only thing separating it from noise.
    """
    n = 60
    feats = {
        "z": np.full(n, -14.0),
        "rhohv": np.full(n, 0.30),
        "zdr": np.full(n, 0.4),
        "snr": np.full(n, -3.0),
        "temp": np.full(n, 10.0),
        "vel_texture_norm": np.full(n, 0.95),
    }
    codes, info = core.classify(feats, min_run=1)
    assert (codes == core.NAME_TO_CODE["no_scatter"]).all()
    assert info["n_noise_floor"] == n


def test_incoherent_gates_with_real_power_are_left_for_the_classifier():
    """The complement: a powered incoherent gate is a candidate second trip.

    It must survive the noise floor so the membership functions and the
    geometric veto can decide, rather than being blanked before either runs.
    """
    n = 40
    feats = {
        "z": np.full(n, 8.0),
        "rhohv": np.full(n, 0.55),
        "ncp": np.full(n, 0.25),
        "snr": np.full(n, 20.0),
        "rhohv_texture": np.full(n, 0.20),
        "phidp_texture": np.full(n, 90.0),
        "vel_texture_norm": np.full(n, 0.95),
        "trip_possible": np.ones(n),
    }
    codes, _ = core.classify(feats, shape=(1, n))
    assert not (codes == core.NAME_TO_CODE["no_scatter"]).any()


def test_coherent_weak_echo_survives_the_coherence_test():
    """Weak but coherent echo must not be thrown away with the noise."""
    n = 60
    feats = {
        "z": np.full(n, 12.0),
        "rhohv": np.full(n, 0.98),
        "zdr": np.full(n, 0.3),
        "snr": np.full(n, 12.0),
        "temp": np.full(n, 10.0),
        "vel_texture_norm": np.full(n, 0.08),
        "rhohv_texture": np.full(n, 0.005),
    }
    codes, _ = core.classify(feats, min_run=1)
    assert not (codes == core.NAME_TO_CODE["no_scatter"]).any()


def test_second_trip_needs_power_as_well_as_incoherence():
    """Second trip is incoherent WITH returned power; noise is incoherent without.

    Both populations sit at the uniform-phase texture limit, so texture alone
    cannot separate them -- SNR is what distinguishes a real distant target
    from receiver noise.
    """
    n = 40
    base = {
        "rhohv": np.full(n, 0.55),
        "ncp": np.full(n, 0.3),
        "rhohv_texture": np.full(n, 0.2),
        "phidp_texture": np.full(n, 90.0),
        "vel_texture_norm": np.full(n, 0.95),
    }
    # Scored without the up-front noise gate, to test the membership functions.
    # Reflectivity is part of the evidence: a second-trip target returns power
    # (positive Z, high SNR), receiver noise does not (Z at the floor,
    # negative SNR). Holding Z fixed while moving only SNR describes a gate
    # that is genuinely ambiguous, so both are varied together here.
    trip, _ = core.classify(
        dict(base, snr=np.full(n, 20.0), z=np.full(n, 8.0)),
        snr_min=None,
        incoherent_frac=None,
        min_run=1,
    )
    noise, _ = core.classify(
        dict(base, snr=np.full(n, -3.0), z=np.full(n, -12.0)),
        snr_min=None,
        incoherent_frac=None,
        min_run=1,
    )
    assert core.CLASSES[trip[0]] == "multi_trip", core.CLASSES[trip[0]]
    assert core.CLASSES[noise[0]] == "no_scatter", core.CLASSES[noise[0]]


def test_strong_echo_survives_the_noise_gate():
    """A bright core must never be blanked by a texture artefact at its edge."""
    n = 30
    feats = {
        "z": np.full(n, 52.0),
        "rhohv": np.full(n, 0.98),
        "zdr": np.full(n, 1.2),
        "snr": np.full(n, -5.0),
        "temp": np.full(n, 15.0),
        "hoi0_km": np.full(n, -3.0),
        "vel_texture_norm": np.full(n, 0.99),
    }
    codes, _ = core.classify(feats, min_run=1, despeckle_keep_dbz=30.0)
    assert not (codes == core.NAME_TO_CODE["no_scatter"]).any()


# --------------------------------------------------------------------------
# depolarization ratio (Ryzhkov et al. 2017)
# --------------------------------------------------------------------------


def test_dr_ordering_matches_scatterer_physics():
    """DR must rise from well-oriented drops to irregular biological targets."""
    dr = core.depolarization_ratio
    light = dr(0.5, 0.995)
    heavy = dr(3.0, 0.98)
    melt = dr(1.5, 0.90)
    bio = dr(5.0, 0.75)
    assert light < heavy < melt < bio
    # Published behaviour: rain sits well below -14 dB, biology well above.
    assert light < -20.0
    assert bio > -10.0


def test_dr_is_finite_and_negative_for_physical_inputs():
    zdr = np.linspace(-2.0, 8.0, 60)
    rho = np.linspace(0.3, 1.0, 60)
    z, r = np.meshgrid(zdr, rho)
    out = core.depolarization_ratio(z, r)
    assert np.isfinite(out).all()
    assert (out <= 0.0 + 1e-9).all()


def test_dr_handles_nan_without_propagating_garbage():
    out = core.depolarization_ratio(np.array([np.nan, 1.0]), np.array([0.9, np.nan]))
    assert np.isnan(out).all()


def test_dr_separates_biology_from_ice_at_equal_reflectivity():
    """Same weak Z, cold vs warm: DR should split them even before temperature.

    Anvil ice and insects can both be weak echo. Their depolarization differs
    by more than 10 dB, which is what makes DR worth carrying.
    """
    ice = core.depolarization_ratio(0.3, 0.99)
    bio = core.depolarization_ratio(5.5, 0.72)
    assert bio - ice > 10.0


def _synthetic_radar():
    pyart = pytest.importorskip("pyart")
    radar = pyart.testing.make_empty_ppi_radar(60, 36, 3)
    radar.range["data"] = np.arange(60, dtype="f8") * 500.0 + 250.0
    rng = np.random.default_rng(7)
    shape = (radar.nrays, radar.ngates)
    for name, lo, hi in [
        ("reflectivity", -20, 55),
        ("differential_reflectivity", -1, 5),
        ("cross_correlation_ratio", 0.5, 1.0),
        ("normalized_coherent_power", 0.0, 1.0),
        ("signal_to_noise_ratio", -10, 50),
        ("spectral_width", 0.1, 6.0),
        ("differential_phase", 0, 180),
        ("velocity", -16, 16),
    ]:
        radar.add_field(
            name,
            {"data": np.ma.masked_array(rng.uniform(lo, hi, shape), mask=False)},
            replace_existing=True,
        )
    radar.instrument_parameters = {
        "nyquist_velocity": {"data": np.full(radar.nrays, 16.0)}
    }
    return radar


def test_pyart_roundtrip_adds_valid_field():
    pytest.importorskip("pyart")
    from radar_palette.gateid import gate_id

    radar = _synthetic_radar()
    out, meta = gate_id(radar, freezing_level_m=4000.0)
    assert "gate_id" in out.fields
    assert "gate_id_margin" in out.fields
    data = out.fields["gate_id"]["data"]
    assert data.shape == (radar.nrays, radar.ngates)
    assert set(np.unique(np.ma.filled(data, 0))) <= set(core.CLASSES)
    assert meta["resolved"]["z"] == "reflectivity"
    assert sum(meta["class_counts"].values()) == radar.nrays * radar.ngates


def test_field_candidate_lists_are_ordered_consistently():
    """First match wins, so candidate order is load-bearing.

    Both object families resolve through one list now, but the ordering
    contract still matters: CfRadial/ARM names must precede ODIM short names
    so a volume carrying both resolves to the documented one.
    """
    from radar_palette.gateid.features import FIELD_CANDIDATES

    for key, names in FIELD_CANDIDATES.items():
        assert names, key
        assert len(names) == len(set(names)), f"{key} has duplicates"
    z = FIELD_CANDIDATES["z"]
    assert z.index("reflectivity") < z.index("DBZH")


def test_rhi_and_wide_ppi_geometry_are_rejected():
    """An RHI must not be processed as a constant-elevation surface."""
    pytest.importorskip("pyart")
    from radar_palette.gateid.entry_points import _sweep_geometry_ok

    radar = _synthetic_radar()
    ok, mode, span = _sweep_geometry_ok(radar, 0)
    assert ok and span <= 5.0

    # Force a wide elevation span into one sweep.
    sl = radar.get_slice(0)
    radar.elevation["data"][sl] = np.linspace(1.0, 40.0, sl.stop - sl.start)
    ok, mode, span = _sweep_geometry_ok(radar, 0)
    assert not ok and span > 5.0


def test_sweep_mode_decoding_survives_masked_char_arrays():
    """Py-ART stores sweep_mode as masked characters; joining it naively fails.

    A regression guard: mishandling this turned 'sector' into byte junk and
    made every scan look like an unknown type.
    """
    pytest.importorskip("pyart")
    from radar_palette.gateid.entry_points import _decode_sweep_modes

    radar = _synthetic_radar()
    modes = _decode_sweep_modes(radar)
    assert modes
    assert all(m.startswith(("ppi", "sector", "azimuth")) for m in modes), modes


def test_backends_agree_on_a_real_volume(tmp_path):
    """Both object families must classify the same volume the same way.

    This is the test that matters for maintenance: two entry paths over one
    physics core are only safe if something asserts they agree. It caught two
    real bugs -- the Xradar wrapper stores ``sweep_mode`` as one whole string
    per sweep rather than a row of characters (so 14 of 15 sweeps were being
    rejected as unknown scan types), and it does not surface Nyquist velocity
    through ``instrument_parameters`` (so the phase-coherence test silently
    vanished).

    Agreement is compared per class as a *fraction*, for two upstream reasons.
    The readers can differ by a ray -- xradar returned 13,953 where Py-ART
    returned 13,954 on one volume -- and they order rays differently, xradar
    sorting by azimuth where Py-ART keeps acquisition order. Neither is a
    classification difference, so an exact or gate-wise comparison would fail
    for the wrong reason.
    """
    pyart = pytest.importorskip("pyart")
    xradar = pytest.importorskip("xradar")
    import glob

    from radar_palette.gateid import gate_id

    matches = sorted(
        glob.glob("/Users/scollis/data/bnf_radar/*/bnfcsapr2cfrS3.a1.*.nc")
    )
    path = None
    for candidate in matches:
        # Require a multi-sweep PPI. The two readers disagree substantially on
        # ray count for RHI files in this archive (89 vs 7 rays on one volume),
        # which is an upstream difference in how each handles a single
        # elevation-scanning sweep and would make this test fail for a reason
        # that has nothing to do with classification.
        try:
            radar = pyart.io.read(candidate)
        except Exception:
            continue
        if radar.nsweeps >= 5 and radar.nrays > 1000:
            path = candidate
            break
    if path is None:
        pytest.skip("no local multi-sweep BNF PPI available")

    _, meta_p = gate_id(pyart.io.read(path), freezing_level_m=4670.0)
    _, meta_x = gate_id(
        xradar.io.open_cfradial1_datatree(path, first_dim="auto"),
        freezing_level_m=4670.0,
    )

    assert meta_p["resolved"] == meta_x["resolved"]
    assert meta_p["nyquist"] == pytest.approx(meta_x["nyquist"], rel=1e-6)
    assert len(meta_p["skipped_sweeps"]) == len(meta_x["skipped_sweeps"])

    total_p = sum(meta_p["class_counts"].values())
    total_x = sum(meta_x["class_counts"].values())
    # Allow the known one-ray reader difference, nothing larger.
    assert abs(total_p - total_x) <= meta_p.get("ngates", 1100)

    for name, count_p in meta_p["class_counts"].items():
        frac_p = count_p / total_p
        frac_x = meta_x["class_counts"][name] / total_x
        assert frac_x == pytest.approx(frac_p, abs=1.5e-3), name


def test_angular_texture_matches_pyart_where_data_is_valid():
    """The folding-aware texture is Py-ART's, not a reimplementation.

    Pinned so a future edit cannot quietly fork from upstream. The two must
    agree wherever the input is finite; the only intended difference is that
    NaN propagates here instead of being smeared by symmetric-boundary
    convolution.
    """
    pyart = pytest.importorskip("pyart")

    nyq = 16.3
    rng = np.random.default_rng(0)
    for field in (
        np.full((60, 60), 4.0) + rng.normal(0, 0.15, (60, 60)),
        rng.uniform(-nyq, nyq, (60, 60)),
    ):
        mine = core.angular_texture(field, nyq, 4)
        theirs = pyart.util.angular_texture_2d(field, (4, 4), nyq)
        interior = np.zeros_like(mine, dtype=bool)
        interior[3:-3, 3:-3] = True  # away from the boundary
        both = interior & np.isfinite(mine) & np.isfinite(theirs)
        assert both.sum() > 1000
        assert np.allclose(mine[both], theirs[both], atol=1e-9)


def test_angular_texture_propagates_nan_where_pyart_does_not():
    """The one deliberate difference, stated as a test rather than a comment."""
    pytest.importorskip("pyart")

    nyq = 16.3
    rng = np.random.default_rng(1)
    field = rng.uniform(-nyq, nyq, (40, 40))
    field[10:14, 10:14] = np.nan
    out = core.angular_texture(field, nyq, 4)
    assert np.isnan(out[10:14, 10:14]).all()
