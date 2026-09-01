"""Public MDS fitting and gatefilter replacement entry point."""

from __future__ import annotations

import numpy as np

from radar_palette.gateid.features import resolve_fields
from radar_palette.gateid.mdsreplace import (
    RANGE_SQUARE_SLOPE_DB,
    fit_mds,
    replace_with_mds,
)
from radar_palette.gateid.single import NAME_TO_CODE
from radar_palette.io.flavors import (
    detect_radar_flavor,
    resolve_output_flavor,
    to_pyart_radar,
    to_radar_flavor,
)

__all__ = ["mdsreplace"]


def _excluded_from_gatefilter(gatefilter, shape):
    excluded = np.asarray(gatefilter.gate_excluded, dtype=bool)
    if excluded.shape != shape:
        raise ValueError("gatefilter shape does not match the radar fields")
    return excluded


def mdsreplace(
    radar_like,
    gatefilter,
    field_name=None,
    output_field_name=None,
    gate_id_field="gate_id",
    no_return_code=None,
    slope_db_per_decade=RANGE_SQUARE_SLOPE_DB,
    percentile=50.0,
    degree=None,
    min_range_m=0.0,
    min_range_bins=6,
    per_sweep=True,
    output_flavor=None,
    replace_existing=True,
):
    """Replace gatefilter-excluded gates with a fitted minimum detectable signal.

    Gates the filter excludes hold no usable measurement, but leaving them
    masked gives an objective-analysis weight function nothing to grid down to,
    so an isolated echo bleeds outward into empty space. Replacing them with
    the minimum detectable signal states what the radar would have reported had
    there been nothing there, which is the honest floor to interpolate toward.

    The floor is fitted from the ``no_scatter`` population against
    ``log10(range)`` with the slope fixed at the physical 20 dB per decade; see
    :mod:`radar_palette.gateid.mdsreplace`. Only excluded gates are written;
    every included measurement is left exactly as it was.

    Parameters
    ----------
    radar_like : pyart.core.Radar or xarray.DataTree
        Volume carrying reflectivity and a gate-ID field.
    gatefilter : pyart.filters.GateFilter
        Filter whose excluded gates are replaced. Usually from
        :func:`radar_palette.gateid.meteorological_gatefilter`.
    field_name : str, optional
        Field to replace into. Defaults to the resolved reflectivity.
    output_field_name : str, optional
        Where to write the result. Defaults to overwriting ``field_name``.
    gate_id_field : str, optional
        Field holding the gate classification.
    no_return_code : int, optional
        Class code marking gates with no scatterer. Defaults to ``no_scatter``.
    slope_db_per_decade, percentile, degree, min_range_m, min_range_bins
        Passed to :func:`radar_palette.gateid.mdsreplace.fit_mds`.
    per_sweep : bool, optional
        Fit each sweep separately. Sweeps whose own no-return population is too
        sparse fall back to a whole-volume fit, which is recorded per sweep in
        the returned metadata rather than being applied silently.
    output_flavor : RadarFlavor or str or None, optional
        Family to return. Defaults to mirroring the input.
    replace_existing : bool, optional
        Overwrite existing fields of the output names.

    Returns
    -------
    radar : pyart.core.Radar or xarray.DataTree
        Input volume carrying the filled field and ``<output>_mds``.
    meta : dict
        Resolved names, gates replaced, and the fit for every sweep.

    Examples
    --------
    Identify, filter, and fill in one chain::

        radar, _ = gate_id(radar, freezing_level_m=4670.0)
        gatefilter = meteorological_gatefilter(radar)
        radar, meta = mdsreplace(radar, gatefilter)
    """
    input_flavor = detect_radar_flavor(radar_like)
    output_flavor = resolve_output_flavor(output_flavor, input_flavor)
    radar, _ = to_pyart_radar(radar_like)
    resolved, missing = resolve_fields(radar.fields)
    if "z" not in resolved:
        raise ValueError("mdsreplace needs a reflectivity field")
    if gate_id_field not in radar.fields:
        raise ValueError(f"mdsreplace needs gate-ID field {gate_id_field!r}")

    source_name = field_name or resolved["z"]
    if source_name not in radar.fields:
        raise ValueError(f"field {source_name!r} is not present")
    target_name = output_field_name or source_name
    shape = (radar.nrays, radar.ngates)
    excluded = _excluded_from_gatefilter(gatefilter, shape)
    codes = np.ma.filled(radar.fields[gate_id_field]["data"], -1)
    no_return_code = NAME_TO_CODE["no_scatter"] if no_return_code is None else no_return_code
    ranges = radar.range["data"]
    options = dict(
        slope_db_per_decade=slope_db_per_decade,
        percentile=percentile,
        degree=degree,
        min_range_m=min_range_m,
        min_range_bins=min_range_bins,
    )

    # The volume fit is the fallback for a sweep too sparse to fit alone, and
    # the whole answer when per-sweep fitting is off.
    volume_curve, volume_info = fit_mds(
        radar.fields[source_name]["data"], ranges, codes == no_return_code, **options
    )

    mds = np.empty(shape, dtype="f8")
    fits, replaced = [], 0
    for sweep in range(radar.nsweeps):
        sl = radar.get_slice(sweep)
        info, scope = dict(volume_info), "volume"
        if per_sweep:
            try:
                curve, info = fit_mds(
                    radar.fields[source_name]["data"][sl],
                    ranges,
                    codes[sl] == no_return_code,
                    **options,
                )
                mds[sl], scope = curve, "sweep"
            except ValueError as error:
                mds[sl] = volume_curve[sl]
                info["fallback_reason"] = str(error)
        else:
            mds[sl] = volume_curve[sl]
        count = int(excluded[sl].sum())
        fits.append(dict(sweep=sweep, scope=scope, n_replaced=count, **info))
        replaced += count

    data = replace_with_mds(radar.fields[source_name]["data"], mds, excluded)
    source = dict(radar.fields[source_name])
    source.update(
        {
            "data": data,
            "long_name": "Reflectivity with excluded gates replaced by minimum detectable signal",
            "comment": (
                "Gates excluded by the gatefilter replaced by the fitted "
                "minimum detectable signal, from the no_scatter population "
                "against log10(range) at a fixed 20 dB per decade. Included "
                "measurements are unchanged."
            ),
            "ancillary_variables": f"{gate_id_field} {target_name}_mds",
        }
    )
    radar.add_field(target_name, source, replace_existing=replace_existing)
    radar.add_field(
        f"{target_name}_mds",
        {
            "data": np.ma.masked_invalid(mds),
            "long_name": "Fitted minimum detectable signal",
            "units": radar.fields[source_name].get("units", "dBZ"),
            "comment": (
                "Minimum detectable signal fitted from no_scatter gates "
                "versus log10(range), slope fixed by the radar equation."
            ),
        },
        replace_existing=replace_existing,
    )
    meta = {
        "resolved": resolved,
        "missing": missing,
        "source_field": source_name,
        "output_field": target_name,
        "gate_id_field": gate_id_field,
        "no_return_code": int(no_return_code),
        "n_replaced": replaced,
        "fits": fits,
    }
    return to_radar_flavor(radar, output_flavor), meta
