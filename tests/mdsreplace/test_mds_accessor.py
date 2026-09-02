"""Tests for the ``radarpalette`` accessor's MDS methods.

Two contracts, and the second is the one with teeth. First, the accessor must
fill *the caller's own tree* without disturbing its geometry, and must not
disagree with the functional entry point. Second, a Py-ART gate filter is
indexed by Py-ART's ray order, which is not always the tree's: alignment must
follow azimuth rather than position, and must refuse when the two objects do
not describe the same rays. A silent azimuthal roll in a QC mask survives all
the way to a figure.
"""

from __future__ import annotations

import glob

import numpy as np
import pyart
import pytest

xr = pytest.importorskip("xarray")
xd = pytest.importorskip("xradar")

import radar_palette  # noqa: F401,E402  -- registers the accessor
from radar_palette.gateid.single import NAME_TO_CODE  # noqa: E402
from radar_palette.io.flavors import RadarFlavor, to_radar_flavor  # noqa: E402
from radar_palette.mdsreplace import (  # noqa: E402
    align_gatefilter_to_tree,
    mds_profile,
)
from radar_palette.testing import make_empty_ppi_volume  # noqa: E402

FLOOR_AT_1KM = -40.0


def _pyart_volume(nsweeps=2, ngates=100, rays_per_sweep=90):
    """A synthetic PPI volume with a known floor, weather, and a gate ID."""
    radar = make_empty_ppi_volume(
        nsweeps=nsweeps, ngates=ngates, rays_per_sweep=rays_per_sweep
    )
    rng = np.random.default_rng(5)
    ranges = np.asarray(radar.range["data"], dtype="f8")
    trend = 20.0 * np.log10(ranges / 1000.0)
    values = (
        FLOOR_AT_1KM
        + trend[None, :]
        + rng.normal(0.0, 1.5, (radar.nrays, radar.ngates))
    )
    codes = np.full((radar.nrays, radar.ngates), NAME_TO_CODE["no_scatter"], "i2")
    values[5:25, 20:60] = 35.0
    codes[5:25, 20:60] = NAME_TO_CODE["moderate_rain"]
    radar.add_field(
        "reflectivity", {"data": np.ma.masked_invalid(values), "units": "dBZ"}
    )
    radar.add_field("gate_id", {"data": np.ma.masked_array(codes, mask=False)})
    return radar


def _tree(**kwargs):
    return to_radar_flavor(_pyart_volume(**kwargs), RadarFlavor.XRADAR)


def _real_bnf_tree():
    """A local multi-sweep BNF C-SAPR2 PPI volume, or skip."""
    matches = sorted(
        glob.glob("/Users/scollis/data/radar_cases/bnf/*/bnf_ppi_csapr2_*.nc")
    )
    for path in matches:
        tree = xd.io.open_cfradial1_datatree(path, first_dim="auto")
        sweeps = [k for k in tree.children if str(k).startswith("sweep")]
        if len(sweeps) >= 5:
            return tree
        tree.close()
    pytest.skip("no local multi-sweep BNF PPI volume available")


# --------------------------------------------------------------------------
# the accessor surface
# --------------------------------------------------------------------------


def test_both_methods_are_on_the_one_accessor_alongside_gateid():
    tree = _tree()
    for name in ("gateid", "mds_profile", "mdsreplace"):
        assert callable(getattr(tree.radarpalette, name))


def test_accessor_profile_recovers_the_floor():
    profile = tree_profile = _tree().radarpalette.mds_profile(min_range_m=0.0)
    assert tree_profile.intercept_dbz == pytest.approx(FLOOR_AT_1KM, abs=0.5)
    assert profile.meta["mask_source"].startswith("gate_id:no_scatter")
    assert profile.meta["skipped_sweeps"] == []


def test_accessor_agrees_with_the_functional_path_on_the_same_volume():
    radar = _pyart_volume()
    tree = to_radar_flavor(radar, RadarFlavor.XRADAR)
    functional = mds_profile(radar, min_range_m=0.0)
    accessor = tree.radarpalette.mds_profile(min_range_m=0.0)
    assert accessor.intercept_dbz == pytest.approx(functional.intercept_dbz, abs=0.05)


def test_replacement_lands_on_the_callers_own_tree_and_returns_it():
    tree = _tree()
    returned = tree.radarpalette.mdsreplace(exclude_field="gate_id", min_range_m=0.0)
    assert returned is tree
    for key in (k for k in tree.children if str(k).startswith("sweep")):
        dataset = tree[key].to_dataset()
        assert "mds_replaced" in dataset
        assert np.isfinite(dataset["reflectivity"].values).all()
    assert tree.attrs["mdsreplace_meta"]["n_replaced"] > 0


def test_replacement_leaves_the_geometry_alone():
    tree = _tree()
    before = {
        key: (
            np.asarray(tree[key].to_dataset()["azimuth"].values).copy(),
            np.asarray(tree[key].to_dataset()["range"].values).copy(),
        )
        for key in tree.children
        if str(key).startswith("sweep")
    }
    tree.radarpalette.mdsreplace(exclude_field="gate_id", min_range_m=0.0)
    for key, (azimuth, ranges) in before.items():
        dataset = tree[key].to_dataset()
        assert np.array_equal(dataset["azimuth"].values, azimuth)
        assert np.array_equal(dataset["range"].values, ranges)


def test_filled_gates_hold_the_mds_and_the_others_are_untouched():
    tree = _tree()
    original = np.asarray(
        tree["sweep_0"].to_dataset()["reflectivity"].values, dtype="f8"
    ).copy()
    tree.radarpalette.mdsreplace(
        exclude_field="gate_id", mds_field="mds", min_range_m=0.0
    )
    dataset = tree["sweep_0"].to_dataset()
    filled = np.asarray(dataset["reflectivity"].values, dtype="f8")
    flag = np.asarray(dataset["mds_replaced"].values).astype(bool)
    mds = np.asarray(dataset["mds"].values, dtype="f8")
    assert np.allclose(filled[flag], mds[flag], atol=1e-4)
    assert np.allclose(filled[~flag], original[~flag], atol=1e-4, equal_nan=True)


def test_exclusion_sources_are_mutually_exclusive():
    tree = _tree()
    gatefilter = pyart.filters.GateFilter(tree.pyart.to_radar())
    with pytest.raises(ValueError, match="at most one"):
        tree.radarpalette.mdsreplace(gatefilter=gatefilter, exclude_field="gate_id")


def test_output_flavor_is_refused_on_an_accessor():
    tree = _tree()
    with pytest.raises(TypeError, match="output_flavor is not accepted"):
        tree.radarpalette.mdsreplace(output_flavor="pyart")


def test_an_explicit_per_sweep_mask_bypasses_all_alignment():
    tree = _tree()
    keys = [k for k in tree.children if str(k).startswith("sweep")]
    mask = {}
    for key in keys:
        shape = tree[key].to_dataset()["reflectivity"].shape
        block = np.zeros(shape, dtype=bool)
        block[:, -10:] = True
        mask[key] = block
    tree.radarpalette.mdsreplace(mask=mask, min_range_m=0.0)
    meta = tree.attrs["mdsreplace_meta"]
    assert meta["n_replaced_by_source"]["explicit"] == sum(
        int(m.sum()) for m in mask.values()
    )
    assert np.asarray(tree["sweep_0"].to_dataset()["mds_replaced"].values)[
        :, -10:
    ].all()


# --------------------------------------------------------------------------
# gate-filter alignment: the ray-order trap
# --------------------------------------------------------------------------


def test_alignment_follows_azimuth_rather_than_position():
    """Roll the tree's rays; the mask must roll with them, not stay put."""
    radar = _pyart_volume(nsweeps=1, rays_per_sweep=90)
    tree = to_radar_flavor(radar, RadarFlavor.XRADAR)
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_all()
    gatefilter.include_equal("gate_id", NAME_TO_CODE["moderate_rain"])

    straight = align_gatefilter_to_tree(gatefilter, tree)["sweep_0"]

    rolled = tree.copy()
    dataset = rolled["sweep_0"].to_dataset()
    order = np.roll(np.arange(dataset.sizes["azimuth"]), 17)
    rolled["sweep_0"].dataset = dataset.isel(azimuth=order)
    shifted = align_gatefilter_to_tree(gatefilter, rolled)["sweep_0"]

    assert np.array_equal(shifted, straight[order])
    assert not np.array_equal(shifted, straight)


def test_alignment_refuses_a_filter_from_a_volume_with_other_rays():
    tree = _tree(nsweeps=2, rays_per_sweep=90)
    other = _pyart_volume(nsweeps=2, rays_per_sweep=60)
    gatefilter = pyart.filters.GateFilter(other)
    with pytest.raises(ValueError, match="rays"):
        align_gatefilter_to_tree(gatefilter, tree)


def test_alignment_refuses_a_filter_covering_a_different_sweep_count():
    tree = _tree(nsweeps=2)
    gatefilter = pyart.filters.GateFilter(_pyart_volume(nsweeps=3))
    with pytest.raises(ValueError, match="sweeps"):
        align_gatefilter_to_tree(gatefilter, tree)


def test_alignment_refuses_a_filter_with_no_radar_attached():
    class Bare:
        gate_excluded = np.zeros((10, 10), dtype=bool)

    with pytest.raises(AttributeError, match="does not carry the radar"):
        align_gatefilter_to_tree(Bare(), _tree())


def test_a_gate_filter_and_the_tree_route_replace_the_same_gates():
    radar = _pyart_volume(nsweeps=1)
    tree = to_radar_flavor(radar, RadarFlavor.XRADAR)
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_all()
    gatefilter.include_equal("gate_id", NAME_TO_CODE["moderate_rain"])

    by_filter = tree.copy()
    by_filter.radarpalette.mdsreplace(gatefilter=gatefilter, min_range_m=0.0)
    by_field = tree.copy()
    by_field.radarpalette.mdsreplace(exclude_field="gate_id", min_range_m=0.0)

    assert np.array_equal(
        by_filter["sweep_0"].to_dataset()["mds_replaced"].values,
        by_field["sweep_0"].to_dataset()["mds_replaced"].values,
    )


# --------------------------------------------------------------------------
# a real volume
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_both_families_measure_the_same_floor_on_a_real_bnf_volume():
    """The two object families must not disagree about the receiver.

    They unpack CfRadial1 sweeps differently -- ray counts can differ by one --
    so this checks that the difference does not reach the answer.
    """
    tree = _real_bnf_tree()
    radar = tree.pyart.to_radar()
    from_tree = tree.radarpalette.mds_profile()
    from_pyart = mds_profile(radar)
    assert from_tree.intercept_dbz == pytest.approx(from_pyart.intercept_dbz, abs=0.05)
    assert from_tree.slope_db_per_decade == pytest.approx(20.0, abs=0.1)
    # A C-SAPR2 volume carries SNR, so the exact route is what auto picks.
    assert from_tree.method == "snr" == from_pyart.method
    assert from_tree.residual_rms_db < 0.5

    # The population route has to survive the ray-count difference too: it is
    # the one that pools gates across sweeps rather than pairing two fields.
    mode_tree = tree.radarpalette.mds_profile(method="noise")
    mode_pyart = mds_profile(radar, method="noise")
    assert mode_tree.intercept_dbz == pytest.approx(mode_pyart.intercept_dbz, abs=0.05)
