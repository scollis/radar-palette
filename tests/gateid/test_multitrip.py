"""Tests for the geometric second-trip veto.

Second trip is the class most easily faked by a moment-based test: incoherent
phase, low SQI and noisy rho_hv are equally the signature of receiver noise and
of a sheared updraught. These tests pin the geometry that breaks the tie, and
the reachability of the class itself -- which a previous revision had silently
destroyed.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.gateid import multitrip
from radar_palette.gateid import single as core

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_unambiguous_range_matches_the_textbook_relation():
    """r_max = c * PRT / 2. BNF C-SAPR2 runs 1240 Hz, giving ~121 km."""
    assert multitrip.unambiguous_range(prt=0.0008064516) == pytest.approx(
        120884.0, abs=50.0
    )
    assert multitrip.unambiguous_range(prf=1240.0) == pytest.approx(120884.0, abs=50.0)
    assert multitrip.unambiguous_range(prt=1.0 / 1240.0) == pytest.approx(
        multitrip.unambiguous_range(prf=1240.0)
    )


def test_unambiguous_range_is_nan_without_usable_input():
    assert np.isnan(multitrip.unambiguous_range())
    assert np.isnan(multitrip.unambiguous_range(prt=0.0))
    assert np.isnan(multitrip.unambiguous_range(prt=-1.0))


def test_beam_height_rises_with_range_and_elevation():
    h_near = multitrip.beam_height(10_000.0, 1.5)
    h_far = multitrip.beam_height(100_000.0, 1.5)
    h_steep = multitrip.beam_height(10_000.0, 20.0)
    assert h_far > h_near
    assert h_steep > h_near
    # A horizontal beam still climbs, because the earth curves away.
    assert multitrip.beam_height(100_000.0, 0.0) > 500.0


def test_beam_height_includes_radar_altitude():
    assert multitrip.beam_height(1000.0, 0.0, 180.0) == pytest.approx(
        multitrip.beam_height(1000.0, 0.0, 0.0) + 180.0
    )


def test_echo_top_uses_a_percentile_not_the_maximum():
    """One noisy gate aloft must not raise the ceiling for the whole sweep."""
    z = np.full((10, 10), 30.0)
    h = np.full((10, 10), 5000.0)
    z[0, 0] = 35.0
    h[0, 0] = 18000.0  # a single spike
    top = multitrip.echo_top_height(z, h, threshold_dbz=10.0, percentile=99.0)
    assert top < 18000.0


def test_echo_top_is_nan_without_echo():
    z = np.full((5, 5), -30.0)
    h = np.full((5, 5), 3000.0)
    assert np.isnan(multitrip.echo_top_height(z, h, threshold_dbz=10.0))


# --------------------------------------------------------------------------
# the veto
# --------------------------------------------------------------------------


def test_second_trip_impossible_at_high_elevation():
    """The point of the whole module.

    At 1240 Hz an echo seen at 40 km apparent range is really at 161 km. At
    1.5 deg that is 6 km up and plausible; at 20 deg it is above 55 km, which
    is not weather.
    """
    r = np.full((1, 1), 40_000.0)
    low, _ = multitrip.second_trip_possible(
        r, [[1.5]], prt=1.0 / 1240.0, ceiling_m=20_000.0
    )
    high, _ = multitrip.second_trip_possible(
        r, [[20.0]], prt=1.0 / 1240.0, ceiling_m=20_000.0
    )
    assert low.all()
    assert not high.any()


def test_veto_reports_the_cutoff_elevation():
    r = np.linspace(1000.0, 113_000.0, 50).reshape(1, -1)
    el = np.full((1, 1), 1.5)
    _, info = multitrip.second_trip_possible(
        r, el, prt=1.0 / 1240.0, ceiling_m=20_000.0
    )
    assert info["unambiguous_range_m"] == pytest.approx(120_884.0, abs=100.0)
    # Somewhere between a few degrees and the tilt where 121 km of extra path
    # puts the beam through 20 km.
    assert 3.0 < info["max_elevation_deg"] < 15.0


def test_veto_is_not_applied_without_a_prt():
    """No PRT means no geometry, so nothing is vetoed rather than everything.

    Silently forbidding a class because a metadata field is missing would be
    worse than leaving the moment evidence to stand alone.
    """
    r = np.full((2, 3), 50_000.0)
    possible, info = multitrip.second_trip_possible(r, [[30.0], [40.0]])
    assert possible.all()
    assert np.isnan(info["unambiguous_range_m"])
    assert "note" in info


def test_ceiling_derived_from_the_data_when_not_given():
    r = np.full((4, 4), 30_000.0)
    z = np.full((4, 4), 40.0)
    h = np.full((4, 4), 8_000.0)
    _, info = multitrip.second_trip_possible(
        r,
        np.full((4, 1), 1.0),
        prt=1.0 / 1240.0,
        reflectivity=z,
        gate_height_m=h,
        ceiling_margin_m=3000.0,
    )
    assert info["ceiling_m"] == pytest.approx(11_000.0, abs=100.0)


# --------------------------------------------------------------------------
# reachability of the class
# --------------------------------------------------------------------------


def _second_trip_signature(n=4, **over):
    feats = {
        "z": np.full(n, 5.0),
        "rhohv": np.full(n, 0.55),
        "ncp": np.full(n, 0.25),
        "snr": np.full(n, 20.0),
        "rhohv_texture": np.full(n, 0.20),
        "phidp_texture": np.full(n, 90.0),
        "vel_texture_norm": np.full(n, 0.95),
    }
    feats.update(over)
    return feats


def test_multi_trip_is_reachable_with_default_settings():
    """Regression: the class was unreachable as shipped.

    multi_trip scored 1.0 and was then overwritten with no_scatter by the
    noise gate, because incoherence alone was treated as absence of a
    scatterer. Second trip IS incoherent -- returned power is what separates
    it from noise.
    """
    feats = _second_trip_signature(trip_possible=np.ones(4))
    codes, _ = core.classify(feats, shape=(1, 4))
    assert core.CLASSES[codes[0]] == "multi_trip"


def test_incoherent_gate_without_power_is_still_noise():
    """The other half of the same rule: no power means no target."""
    feats = _second_trip_signature(
        snr=np.full(4, -3.0), z=np.full(4, -15.0), trip_possible=np.ones(4)
    )
    codes, _ = core.classify(feats, shape=(1, 4))
    assert core.CLASSES[codes[0]] == "no_scatter"


def test_geometry_forbids_multi_trip_where_it_is_impossible():
    feats = _second_trip_signature(trip_possible=np.zeros(4))
    codes, info = core.classify(
        feats, shape=(1, 4), snr_min=None, incoherent_frac=None, min_run=1
    )
    assert core.CLASSES[codes[0]] != "multi_trip"
    assert info["scores"][0, info["order"].index("multi_trip")] == 0.0


def test_turbulent_weather_is_not_called_second_trip():
    """High velocity texture in a strong coherent core must not fool it.

    This is the misdiagnosis the geometry exists to prevent: a sheared
    updraught has high velocity variance but also strong, coherent,
    high-rho_hv echo.
    """
    n = 6
    feats = {
        "z": np.full(n, 52.0),
        "rhohv": np.full(n, 0.985),
        "zdr": np.full(n, 1.2),
        "ncp": np.full(n, 0.85),
        "snr": np.full(n, 45.0),
        "rhohv_texture": np.full(n, 0.006),
        "phidp_texture": np.full(n, 8.0),
        "vel_texture_norm": np.full(n, 0.55),
        "temp": np.full(n, 12.0),
        "hoi0_km": np.full(n, -2.0),
        "trip_possible": np.ones(n),
    }
    codes, _ = core.classify(feats, shape=(1, n))
    assert core.CLASSES[codes[0]] != "multi_trip"


# --------------------------------------------------------------------------
# the ceiling must describe the storm, not the beam
# --------------------------------------------------------------------------


def test_volume_echo_top_takes_the_tallest_sweep():
    """A low sweep cannot see the storm top; the volume can.

    Regression for a real defect. The ceiling was derived from the sweep being
    classified, so a 1.5 deg sweep reported an "echo top" of 3.85 km on a
    volume whose storm reached 17.0 km -- that is the height its beam reaches,
    not the height of the weather.
    """
    low = (np.full((4, 20), 30.0), np.full((4, 20), 3_800.0))
    high = (np.full((4, 20), 30.0), np.full((4, 20), 17_000.0))
    assert multitrip.volume_echo_top([low]) == pytest.approx(3_800.0, abs=50.0)
    assert multitrip.volume_echo_top([low, high]) == pytest.approx(17_000.0, abs=50.0)


def test_volume_echo_top_ignores_sweeps_without_echo():
    empty = (np.full((3, 10), -30.0), np.full((3, 10), 5_000.0))
    real = (np.full((3, 10), 35.0), np.full((3, 10), 9_000.0))
    assert multitrip.volume_echo_top([empty, real]) == pytest.approx(9_000.0, abs=50.0)
    assert np.isnan(multitrip.volume_echo_top([empty]))
    assert np.isnan(multitrip.volume_echo_top([]))


def test_sweep_ceiling_would_veto_the_long_ranges_second_trip_lives_at():
    """Why the per-sweep ceiling was self-defeating.

    Second trip is most likely at long apparent range on low sweeps, and that
    is exactly the regime a per-sweep ceiling forbids.
    """
    r = np.array([[40_000.0, 100_000.0]])
    el = np.array([[1.5]])
    prt = 1.0 / 1240.0

    tight, _ = multitrip.second_trip_possible(
        r, el, prt=prt, ceiling_m=3_850.0 + 3_000.0, radar_altitude_m=180.0
    )
    correct, _ = multitrip.second_trip_possible(
        r, el, prt=prt, ceiling_m=16_990.0 + 3_000.0, radar_altitude_m=180.0
    )

    assert not tight[0, 1], "sweep ceiling wrongly vetoes long range"
    assert correct.all(), "volume ceiling should allow the whole sweep"


def test_gate_id_reports_which_ceiling_it_used():
    """Provenance, so a reader can tell the volume path from the fallback."""
    pyart = pytest.importorskip("pyart")
    import glob

    from radar_palette.gateid import gate_id

    matches = sorted(
        glob.glob("/Users/scollis/data/bnf_radar/*/bnfcsapr2cfrS3.a1.*.nc")
    )
    path = None
    for candidate in matches:
        radar = pyart.io.read(candidate)
        if radar.nsweeps >= 5 and radar.nrays > 1000:
            path = candidate
            break
    if path is None:
        pytest.skip("no local multi-sweep BNF PPI available")

    _, meta = gate_id(pyart.io.read(path), freezing_level_m=4670.0)
    assert meta["trip_ceiling_source"] == "volume_echo_top"
    # The storm, not the lowest beam: comfortably above a low sweep's reach.
    assert meta["trip_ceiling_m"] > 10_000.0
