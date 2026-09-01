"""Tests for fitting a minimum detectable signal and replacing filtered gates.

The defects pinned here were all found on real ARM BNF C-SAPR2 PPI volumes:
a range coordinate starting at exactly 0 m, and a freely-fitted slope pulled
negative by close-range clutter inside the no-return population.
"""

from __future__ import annotations

import numpy as np
import pytest

from radar_palette.gateid.mdsreplace import (
    MAX_OFFSET_SCATTER_DB,
    RANGE_SQUARE_SLOPE_DB,
    fit_mds,
    replace_with_mds,
)

OFFSET_DBZ = -40.0


def _floor(ranges):
    """The physical floor: constant power, so 20 dB per decade of range."""
    return OFFSET_DBZ + RANGE_SQUARE_SLOPE_DB * np.log10(ranges / 1000.0)


def _synthetic(nray=8, ngate=40, first_range_m=1000.0):
    ranges = np.arange(ngate, dtype="f8") * 1000.0 + first_range_m
    truth = _floor(np.where(ranges > 0, ranges, np.nan))
    data = np.broadcast_to(truth, (nray, ngate)).copy()
    return data, ranges, np.ones_like(data, dtype=bool), truth


# ---------------------------------------------------------------------------
# the fit
# ---------------------------------------------------------------------------


def test_fit_recovers_the_floor_from_no_return_gates():
    data, ranges, no_return, truth = _synthetic()
    mds, meta = fit_mds(data, ranges, no_return)

    assert np.allclose(mds, truth[None, :], atol=1e-6)
    assert meta["offset_dbz_at_1km"] == pytest.approx(OFFSET_DBZ, abs=1e-6)
    assert meta["slope_db_per_decade"] == RANGE_SQUARE_SLOPE_DB
    assert meta["method"] == "fixed_slope"


def test_fit_rises_twenty_db_per_decade_of_range():
    """The radar equation, not a tuned parameter.

    Equivalent reflectivity carries an explicit r-squared correction, so a
    constant detectable power appears as a floor rising 20 dB per decade.
    """
    data, ranges, no_return, _ = _synthetic(ngate=100)
    mds, _ = fit_mds(data, ranges, no_return)

    near = mds[0, np.argmin(np.abs(ranges - 10_000.0))]
    far = mds[0, np.argmin(np.abs(ranges - 100_000.0))]
    assert far - near == pytest.approx(RANGE_SQUARE_SLOPE_DB, abs=0.2)
    assert np.all(np.diff(mds[0]) > 0)


def test_fit_ignores_only_the_scatterers_not_the_floor():
    """Gates holding echo must not drag the floor upward."""
    data, ranges, no_return, truth = _synthetic()
    data[:, 10:14] = 45.0
    no_return[:, 10:14] = False

    mds, _ = fit_mds(data, ranges, no_return)
    assert np.allclose(mds, truth[None, :], atol=1e-6)


def test_fit_is_robust_to_contaminated_no_return_gates():
    """A median over range bins, so a minority of bad gates cannot win."""
    data, ranges, no_return, truth = _synthetic(nray=20)
    data[:4, :] = 30.0  # four rays of clutter mislabelled as no-return

    mds, _ = fit_mds(data, ranges, no_return)
    assert np.allclose(mds, truth[None, :], atol=1e-6)


def test_fit_survives_a_zero_range_gate():
    """2025 BNF C-SAPR2 files start their range coordinate at exactly 0 m.

    log10(0) is undefined, and an earlier version produced a non-finite MDS
    for every such gate, which then refused to fill anything at all.
    """
    data, ranges, no_return, _ = _synthetic(first_range_m=0.0)
    assert ranges[0] == 0.0

    mds, meta = fit_mds(data, ranges, no_return)
    assert np.isfinite(mds).all()
    assert meta["fit_range_min_m"] > 0.0
    # Held flat inward rather than extrapolated toward the singularity.
    assert mds[0, 0] == pytest.approx(mds[0, 1])


def test_fixed_slope_beats_a_free_fit_on_clutter_contaminated_close_range():
    """Why the slope is fixed by default.

    On real volumes the close-range no-return population is contaminated by
    clutter and partial beam filling. Fitting the slope freely then returns an
    MDS that *falls* with range, which is unphysical, and would replace distant
    gates with values well above the true floor.
    """
    data, ranges, no_return, truth = _synthetic(ngate=60)
    close = ranges < 8000.0
    data[:, close] = 25.0  # bright close-range contamination

    free, free_meta = fit_mds(data, ranges, no_return, degree=1)
    fixed, _ = fit_mds(data, ranges, no_return)

    assert free_meta["slope_db_per_decade"] < 0.0  # unphysical
    assert np.any(np.diff(free[0]) < 0)
    far = ranges > 50_000.0
    assert np.abs(fixed[0, far] - truth[far]).max() < np.abs(
        free[0, far] - truth[far]
    ).max()


def test_percentile_selects_a_more_conservative_floor():
    rng = np.random.default_rng(3)
    data, ranges, no_return, truth = _synthetic(nray=200)
    data += rng.normal(0.0, 3.0, data.shape)

    median, _ = fit_mds(data, ranges, no_return, percentile=50.0)
    low, _ = fit_mds(data, ranges, no_return, percentile=10.0)
    assert (low < median).all()
    assert np.allclose(median, truth[None, :], atol=1.0)


def test_fit_rejects_a_no_return_population_that_is_really_echo():
    """The floor is a receiver property, so a scattered offset means echo.

    On a BNF multicell volume the three lowest sweeps had widespread low-level
    echo labelled ``no_scatter``. Their fitted offsets were near -7 dBZ against
    -45 to -48 dBZ for the sweeps above, which would have injected fictitious
    +27 dBZ echo across the far field. Their offset scatter was 22-25 dB where
    clean sweeps sat at 0.4-1.6 dB.
    """
    data, ranges, no_return, _ = _synthetic(nray=40, ngate=60)
    rng = np.random.default_rng(5)
    # Half the "no-return" gates are actually echo, at random ranges.
    contaminated = rng.random(data.shape) < 0.5
    data[contaminated] = 30.0

    with pytest.raises(ValueError, match="contaminated by echo"):
        fit_mds(data, ranges, no_return)

    # The same data is fitted if the caller explicitly disables the check.
    mds, meta = fit_mds(data, ranges, no_return, max_offset_scatter_db=None)
    assert np.isfinite(mds).all()
    assert meta["offset_mad_db"] > MAX_OFFSET_SCATTER_DB


def test_clean_noise_scatter_passes_the_contamination_check():
    rng = np.random.default_rng(6)
    data, ranges, no_return, truth = _synthetic(nray=60)
    data += rng.normal(0.0, 2.0, data.shape)

    _, meta = fit_mds(data, ranges, no_return)
    assert meta["offset_mad_db"] < MAX_OFFSET_SCATTER_DB


def test_fit_refuses_rather_than_fabricating_a_floor():
    """Too few no-return bins must raise, never invent a floor."""
    data, ranges, no_return, _ = _synthetic()
    no_return[:, 2:] = False

    with pytest.raises(ValueError, match="insufficient"):
        fit_mds(data, ranges, no_return, min_range_bins=6)


def test_fit_rejects_mismatched_range_coordinate():
    data, ranges, no_return, _ = _synthetic()
    with pytest.raises(ValueError, match="gate dimension"):
        fit_mds(data, ranges[:-1], no_return)


# ---------------------------------------------------------------------------
# the replacement
# ---------------------------------------------------------------------------


def test_replacement_touches_only_excluded_gates():
    data = np.ma.masked_array([[10.0, 20.0, 30.0]], mask=[[False, True, False]])
    mds = np.array([[-30.0, -25.0, -20.0]])

    out = replace_with_mds(data, mds, np.array([[False, True, False]]))

    assert np.allclose(out.data, [[10.0, -25.0, 30.0]])
    assert not np.ma.getmaskarray(out).any()  # the filled gate is now valid


def test_replacement_leaves_unmeasured_included_gates_masked():
    """A masked gate the filter kept has no value and must not become NaN."""
    data = np.ma.masked_array([[10.0, np.nan]], mask=[[False, True]])
    out = replace_with_mds(data, np.array([[-30.0, -25.0]]), np.array([[False, False]]))

    assert np.ma.getmaskarray(out)[0, 1]
    assert out[0, 0] == 10.0


def test_replacement_never_adds_apparent_signal():
    """A gate already weaker than the floor keeps its measured value.

    Found on WSR-88D Level II, which is thresholded before the file is written
    (~71% of gates masked). The surviving "no-return" population is the weakest
    echo that passed the threshold, not the receiver noise floor, so the fitted
    floor can land above real weak echo: on three KHTX/KVNX volumes the naive
    fill *raised* the median of the replaced gates by 12 to 17 dB.
    """
    data = np.ma.masked_array([[10.0, -50.0, np.nan]], mask=[[False, False, True]])
    mds = np.full((1, 3), -30.0)
    excluded = np.ones((1, 3), dtype=bool)

    out = replace_with_mds(data, mds, excluded)
    assert out[0, 0] == -30.0  # strong echo pulled down to the floor
    assert out[0, 1] == -50.0  # already weaker: measurement kept
    assert out[0, 2] == -30.0  # never measured: takes the floor
    assert not np.ma.getmaskarray(out).any()

    louder = replace_with_mds(data, mds, excluded, never_increase=False)
    assert louder[0, 1] == -30.0  # opt out and the value is raised


def test_replacement_refuses_a_non_finite_floor():
    data = np.ma.masked_array([[10.0, 20.0]])
    with pytest.raises(ValueError, match="finite"):
        replace_with_mds(data, np.array([[np.nan, -20.0]]), np.array([[True, False]]))


# ---------------------------------------------------------------------------
# Py-ART entry point
# ---------------------------------------------------------------------------


def _radar(first_range_m=1000.0, nsweeps=2):
    pyart = pytest.importorskip("pyart")
    nray_per_sweep, ngate = 20, 40
    radar = pyart.testing.make_empty_ppi_radar(ngate, nray_per_sweep, nsweeps)
    radar.range["data"] = np.arange(ngate, dtype="f8") * 1000.0 + first_range_m
    shape = (radar.nrays, radar.ngates)
    truth = _floor(np.where(radar.range["data"] > 0, radar.range["data"], np.nan))

    data = np.broadcast_to(truth, shape).copy()
    codes = np.ones(shape, dtype="i2")  # 1 == no_scatter
    data[:, 20:24] = 40.0
    codes[:, 20:24] = 6  # moderate_rain
    radar.add_field("reflectivity", {"data": np.ma.masked_array(data), "units": "dBZ"})
    radar.add_field("gate_id", {"data": np.ma.masked_array(codes)})
    return radar, truth


def test_pyart_entry_point_fills_excluded_gates_only():
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, truth = _radar()
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_all()
    gatefilter.include_equal("gate_id", 6)
    excluded = np.asarray(gatefilter.gate_excluded, dtype=bool)

    out, meta = mdsreplace(radar, gatefilter, output_field_name="filled")
    filled = out.fields["filled"]["data"]

    assert meta["n_replaced"] == int(excluded.sum())
    assert np.allclose(filled[:, 20:24], 40.0)  # kept measurements untouched
    assert np.allclose(filled[:, :20], truth[:20], atol=1e-6)
    assert np.isfinite(np.ma.filled(filled, np.nan)).all()
    assert "filled_mds" in out.fields
    assert all(fit["scope"] == "sweep" for fit in meta["fits"])


def test_pyart_entry_point_never_alters_kept_measurements():
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, _ = _radar()
    before = np.ma.filled(radar.fields["reflectivity"]["data"].astype("f8"), np.nan)
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_equal("gate_id", 1)
    kept = ~np.asarray(gatefilter.gate_excluded, dtype=bool)

    out, _ = mdsreplace(radar, gatefilter, output_field_name="filled")
    after = np.ma.filled(out.fields["filled"]["data"].astype("f8"), np.nan)
    assert np.allclose(after[kept], before[kept], equal_nan=True)


def test_pyart_entry_point_handles_a_zero_range_coordinate():
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, _ = _radar(first_range_m=0.0)
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_equal("gate_id", 1)

    out, meta = mdsreplace(radar, gatefilter, output_field_name="filled")
    assert np.isfinite(np.ma.filled(out.fields["filled"]["data"], np.nan)).all()
    assert meta["n_replaced"] > 0


def test_pyart_entry_point_falls_back_to_the_volume_fit_and_says_so():
    """A sweep too sparse to fit alone must be recorded, not silently filled."""
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, _ = _radar()
    sl = radar.get_slice(1)
    codes = radar.fields["gate_id"]["data"]
    codes[sl] = 6  # no no-return population left in sweep 1
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_all()
    gatefilter.include_equal("gate_id", 6)

    out, meta = mdsreplace(radar, gatefilter, output_field_name="filled")
    scopes = [fit["scope"] for fit in meta["fits"]]
    assert scopes == ["sweep", "volume"]
    assert "fallback_reason" in meta["fits"][1]
    assert np.isfinite(np.ma.filled(out.fields["filled"]["data"], np.nan)).all()


def test_a_contaminated_sweep_falls_back_instead_of_raising_the_floor():
    """The BNF low-sweep case, end to end.

    A sweep whose no-return gates are really low-level echo must not set the
    floor for its own gates. It must fall back to the volume fit, which the
    clean sweeps dominate, and the fallback must be visible in the metadata.
    """
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, truth = _radar(nsweeps=4)
    data = radar.fields["reflectivity"]["data"]
    sl = radar.get_slice(0)
    rng = np.random.default_rng(9)
    dirty = rng.random(data[sl].shape) < 0.5
    data[sl] = np.where(dirty, 30.0, data[sl])

    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_equal("gate_id", 1)
    out, meta = mdsreplace(radar, gatefilter, output_field_name="filled")

    assert meta["fits"][0]["scope"] == "volume"
    assert "contaminated by echo" in meta["fits"][0]["fallback_reason"]
    assert [fit["scope"] for fit in meta["fits"][1:]] == ["sweep"] * 3

    mds = np.ma.filled(out.fields["filled_mds"]["data"], np.nan)
    assert np.allclose(mds[sl], truth[None, :], atol=1.0)


def test_pyart_entry_point_never_raises_a_replaced_gate():
    """End to end: no excluded gate may come out louder than it went in."""
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, _ = _radar()
    # Weak echo well below the floor, labelled as a target to be removed.
    data = radar.fields["reflectivity"]["data"]
    data[:, 30:34] = -70.0
    radar.fields["gate_id"]["data"][:, 30:34] = 2  # clutter
    before = np.ma.filled(data.astype("f8"), np.nan).copy()

    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_equal("gate_id", 2)
    excluded = np.asarray(gatefilter.gate_excluded, dtype=bool)

    out, meta = mdsreplace(radar, gatefilter, output_field_name="filled")
    after = np.ma.filled(out.fields["filled"]["data"].astype("f8"), np.nan)

    assert np.all(after[excluded] <= before[excluded] + 1e-9)
    assert np.allclose(after[:, 30:34], -70.0)
    assert meta["n_kept_already_below_mds"] == int(excluded.sum())
    assert meta["never_increase"] is True


def test_pyart_entry_point_requires_a_gate_id_field():
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, _ = _radar()
    gatefilter = pyart.filters.GateFilter(radar)
    del radar.fields["gate_id"]
    with pytest.raises(ValueError, match="gate-ID field"):
        mdsreplace(radar, gatefilter)


# ---------------------------------------------------------------------------
# xradar accessor
# ---------------------------------------------------------------------------


def _tree(first_range_m=1000.0):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("xradar")
    import radar_palette  # noqa: F401  -- registers the accessor

    data, ranges, _, truth = _synthetic(first_range_m=first_range_m)
    codes = np.ones_like(data, dtype="i2")
    data[:, 10:14] = 40.0
    codes[:, 10:14] = 6
    sweep = xr.Dataset(
        {
            "reflectivity": (("azimuth", "range"), data, {"units": "dBZ"}),
            "gate_id": (("azimuth", "range"), codes),
        },
        coords={"azimuth": np.arange(data.shape[0]), "range": ranges},
    )
    return xr.DataTree.from_dict({"/": xr.Dataset(), "/sweep_0": sweep}), truth


def test_accessor_mutates_the_callers_tree_and_returns_it():
    tree, truth = _tree()
    excluded = np.asarray(tree["sweep_0"]["gate_id"].values) == 1

    returned = tree.radarpalette.mdsreplace({"sweep_0": excluded})
    out = returned["sweep_0"].to_dataset()

    assert returned is tree
    assert "reflectivity_mds" in out
    assert np.allclose(out["reflectivity"].values[:, 10:14], 40.0)
    assert np.allclose(out["reflectivity"].values[:, :10], truth[:10], atol=1e-6)
    assert tree.attrs["mdsreplace_meta"]["n_replaced"] == int(excluded.sum())


def test_accessor_leaves_sweeps_absent_from_the_mapping_alone():
    tree, _ = _tree()
    before = tree["sweep_0"].to_dataset()["reflectivity"].values.copy()

    tree.radarpalette.mdsreplace({})

    after = tree["sweep_0"].to_dataset()["reflectivity"].values
    assert np.allclose(after, before)
    assert tree.attrs["mdsreplace_meta"]["n_replaced"] == 0


def test_accessor_rejects_a_gatefilter_rather_than_misaligning_gates():
    """Py-ART and xradar disagree on ray order, so a flat filter is unsafe."""
    tree, _ = _tree()
    with pytest.raises(TypeError, match="GateFilter"):
        tree.radarpalette.mdsreplace(object())


def test_accessor_rejects_a_mask_of_the_wrong_shape():
    tree, _ = _tree()
    with pytest.raises(ValueError, match="shape"):
        tree.radarpalette.mdsreplace({"sweep_0": np.zeros((3, 3), dtype=bool)})


def test_accessor_and_pyart_entry_point_agree():
    """Two surfaces, one fit: they must not diverge."""
    pyart = pytest.importorskip("pyart")
    from radar_palette.gateid import mdsreplace

    radar, _ = _radar(nsweeps=1)
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_equal("gate_id", 1)
    out, _ = mdsreplace(radar, gatefilter, output_field_name="filled")
    from_pyart = np.ma.filled(out.fields["filled_mds"]["data"], np.nan)

    tree, _ = _tree()
    excluded = np.asarray(tree["sweep_0"]["gate_id"].values) == 1
    tree.radarpalette.mdsreplace({"sweep_0": excluded})
    from_tree = tree["sweep_0"].to_dataset()["reflectivity_mds"].values

    assert np.allclose(from_pyart[0], from_tree[0], atol=1e-9)
