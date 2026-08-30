"""Tests for the ``radarpalette`` DataTree accessor.

The accessor's contract is narrow and easy to break: it must attach the
classification to *the caller's own tree*, without disturbing that tree's
geometry, and it must not disagree with the functional entry point.
"""

from __future__ import annotations

import glob

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
xd = pytest.importorskip("xradar")

import radar_palette  # noqa: F401,E402  -- registers the accessor
from radar_palette.gateid import CLASSES  # noqa: E402


def _volume():
    """A local multi-sweep BNF PPI, or skip."""
    matches = sorted(
        glob.glob("/Users/scollis/data/bnf_radar/*/bnfcsapr2cfrS3.a1.*.nc")
    )
    for path in matches:
        tree = xd.io.open_cfradial1_datatree(path, first_dim="auto")
        sweeps = [k for k in tree.children if str(k).startswith("sweep")]
        if len(sweeps) >= 5:
            return tree
        tree.close()
    pytest.skip("no local multi-sweep BNF volume available")


def _synthetic_tree(n_az=36, n_rng=40):
    """A minimal single-sweep PPI tree, for tests that need no real data."""
    rng = np.random.default_rng(11)
    dims = ("azimuth", "range")
    shape = (n_az, n_rng)
    ranges = np.arange(n_rng, dtype="f8") * 500.0 + 250.0
    azimuth = np.linspace(0.0, 350.0, n_az)
    elevation = np.full(n_az, 1.5)
    height = np.broadcast_to(ranges * np.sin(np.deg2rad(1.5)) + 180.0, shape).copy()

    sweep = xr.Dataset(
        {
            "reflectivity": (dims, rng.uniform(-20, 55, shape)),
            "differential_reflectivity": (dims, rng.uniform(-1, 5, shape)),
            "cross_correlation_ratio": (dims, rng.uniform(0.5, 1.0, shape)),
            "signal_to_noise_ratio": (dims, rng.uniform(-10, 50, shape)),
            "mean_doppler_velocity": (dims, rng.uniform(-16, 16, shape)),
            "nyquist_velocity": ((), 16.0),
            "sweep_mode": ((), "ppi"),
        },
        coords={
            "range": ranges,
            "azimuth": azimuth,
            "elevation": ("azimuth", elevation),
            "z": (dims, height),
        },
    )
    return xr.DataTree.from_dict({"/": xr.Dataset(), "/sweep_0": sweep})


def test_accessor_is_registered():
    tree = _synthetic_tree()
    assert hasattr(tree, "radarpalette")
    assert hasattr(tree.radarpalette, "gateid")


def test_accessor_mutates_the_callers_tree_and_returns_it():
    """The documented promise: it adds the field to *your* object."""
    tree = _synthetic_tree()
    returned = tree.radarpalette.gateid(freezing_level_m=3000.0)

    assert returned is tree
    sweep = tree["sweep_0"].to_dataset()
    assert "gate_id" in sweep.data_vars
    assert "gate_id_margin" in sweep.data_vars


def test_accessor_writes_cf_flag_metadata():
    tree = _synthetic_tree()
    tree.radarpalette.gateid(freezing_level_m=3000.0)
    attrs = tree["sweep_0"].to_dataset()["gate_id"].attrs
    assert attrs["standard_name"] == "gate_id"
    assert len(attrs["flag_values"]) == len(CLASSES)
    assert attrs["flag_meanings"].split() == [CLASSES[c] for c in sorted(CLASSES)]


def test_accessor_stores_metadata_on_the_tree():
    tree = _synthetic_tree()
    tree.radarpalette.gateid(freezing_level_m=3000.0)
    meta = tree.attrs["gateid_meta"]
    assert meta["classified_sweeps"]
    assert sum(meta["class_counts"].values()) > 0
    assert meta["temp_source"] == "lapse_rate_from_freezing_level"


def test_accessor_honours_field_name_and_margin_switch():
    tree = _synthetic_tree()
    tree.radarpalette.gateid(
        freezing_level_m=3000.0, field_name="hydro", add_margin=False
    )
    sweep = tree["sweep_0"].to_dataset()
    assert "hydro" in sweep.data_vars
    assert "hydro_margin" not in sweep.data_vars


def test_accessor_rejects_output_flavor():
    """Asking a DataTree accessor for a Py-ART object is contradictory."""
    tree = _synthetic_tree()
    with pytest.raises(TypeError, match="output_flavor"):
        tree.radarpalette.gateid(freezing_level_m=3000.0, output_flavor="pyart")


def test_accessor_skips_a_sweep_with_unusable_geometry():
    """A sweep spanning many elevations is not a constant-elevation surface."""
    tree = _synthetic_tree()
    sweep = tree["sweep_0"].to_dataset()
    n = sweep.sizes["azimuth"]
    sweep = sweep.assign_coords(elevation=("azimuth", np.linspace(1.0, 40.0, n)))
    tree["sweep_0"].dataset = sweep

    tree.radarpalette.gateid(freezing_level_m=3000.0)
    meta = tree.attrs["gateid_meta"]
    assert not meta["classified_sweeps"]
    assert len(meta["skipped_sweeps"]) == 1
    assert "gate_id" not in tree["sweep_0"].to_dataset().data_vars


def test_accessor_preserves_sweep_geometry_exactly():
    """The reason this accessor does not round-trip through CfRadial.

    Converting a tree to Py-ART and back is not shape-preserving: on a real
    BNF volume a 931-ray sweep came back with 930 rays, so copying results in
    would have misaligned every gate in it. The accessor classifies the
    caller's own sweeps, so sizes must be untouched.
    """
    tree = _volume()
    before = {
        str(k): dict(tree[k].to_dataset().sizes)
        for k in tree.children
        if str(k).startswith("sweep")
    }

    tree.radarpalette.gateid(freezing_level_m=4670.0)

    after = {
        str(k): dict(tree[k].to_dataset().sizes)
        for k in tree.children
        if str(k).startswith("sweep")
    }
    for key, sizes in before.items():
        for dim, size in sizes.items():
            assert after[key][dim] == size, f"{key}/{dim} changed"


def test_accessor_agrees_with_the_functional_entry_point():
    """Two ways in, one physics core: they must not diverge.

    Compared as per-class fractions rather than counts, because the functional
    path converts through CfRadial and that round trip can lose a ray.
    """
    from radar_palette.gateid import gate_id

    tree = _volume()
    tree.radarpalette.gateid(freezing_level_m=4670.0)
    accessor_counts = tree.attrs["gateid_meta"]["class_counts"]

    fresh = _volume()
    _, meta = gate_id(fresh, freezing_level_m=4670.0)

    total_a = sum(accessor_counts.values())
    total_f = sum(meta["class_counts"].values())
    for name, count in accessor_counts.items():
        assert count / total_a == pytest.approx(
            meta["class_counts"][name] / total_f, abs=2e-4
        ), name


def test_register_accessors_is_idempotent():
    from radar_palette.gateid.accessor import register_accessors

    assert register_accessors() is True
    assert register_accessors() is True
