"""Public entry points: measure the MDS, then fill filtered gates with it.

Following the package convention, these accept a Py-ART
:class:`~pyart.core.Radar` or an xradar :class:`~xarray.DataTree` and return
the family the caller passed in, unless ``output_flavor`` asks otherwise.
Conversion is delegated to :mod:`radar_palette.io.flavors`.

The pair is deliberately split. :func:`mds_profile` measures; it is cheap,
returns something inspectable, and is worth looking at before anything is
overwritten. :func:`replace_below_mds` writes, and will call :func:`mds_profile`
itself if handed no profile.

What the fill is *for*
----------------------
A gate filter answers "is this gate weather?", and a gridding weight answers
"how far is this gate from my grid point?". Neither answers "what value does a
grid point take when every gate near it was filtered out?". Removing gates
leaves the interpolator with nothing, and Barnes, Cressman and the spectral
operator in :mod:`radar_palette.gridding` all respond to nothing by either
leaving the point empty or extending the nearest surviving echo into it. The
minimum detectable signal is the physically correct answer instead: the radar
looked there and saw nothing, so the truth is *below* the sensitivity, and the
sensitivity is a number this module can measure from the volume itself.

The hazard worth knowing before you use it
------------------------------------------
Because the MDS grows as :math:`r^2`, it becomes a large number at long range:
on a WSR-88D it passes 0 dBZ around 100 km and reaches roughly +9 dBZ at
250 km. Filling non-detections out there with the MDS is honest about the
sensitivity but writes light-rain reflectivities into empty space, which any
downstream accumulation will happily integrate. ``cap_dbz`` exists for that,
and the counts in the returned metadata tell you how much of the volume was
affected. Filling with the MDS is an upper bound on a non-detection, not a
measurement of one.
"""

from __future__ import annotations

import numpy as np

from radar_palette.gateid.features import FIELD_CANDIDATES
from radar_palette.io.flavors import (
    detect_radar_flavor,
    resolve_output_flavor,
    to_pyart_radar,
    to_radar_flavor,
)
from radar_palette.mdsreplace.profile import MdsProfile, estimate_profile

__all__ = [
    "NO_RETURN_CLASSES",
    "mds_profile",
    "replace_below_mds",
]

#: Gate-ID class names that mean "nothing was detected here". Clutter,
#: biological scatter and second trip are *detections* of something, and their
#: gates carry real power, so they are not part of the noise population even
#: though a meteorological gate filter removes them too.
NO_RETURN_CLASSES = ("no_scatter",)

#: Reflectivity field names, most-preferred first. Shared with the gate-ID
#: feature resolver so the two subpackages cannot disagree about which field
#: is the reflectivity.
REFLECTIVITY_CANDIDATES = tuple(FIELD_CANDIDATES["z"])

#: Additional dBZ-scaled fields worth filling alongside the primary one when
#: they are present, in the order they are checked.
SECONDARY_REFLECTIVITY_CANDIDATES = (
    "reflectivity_v",
    "uncorrected_reflectivity_v",
    "corrected_reflectivity",
    "DBZV",
    "DBZHC",
)


def _resolve_reflectivity(radar, field=None):
    """Name of the reflectivity field to measure the floor from."""
    if field is not None:
        if field not in radar.fields:
            raise KeyError(f"field {field!r} is not on this volume")
        return field
    for name in REFLECTIVITY_CANDIDATES:
        if name in radar.fields:
            return name
    raise KeyError(
        "no reflectivity field found; looked for "
        f"{list(REFLECTIVITY_CANDIDATES)}. Pass field= explicitly."
    )


def _field_values(radar, name):
    """Return a field as float64 with masked and fill entries as NaN."""
    return np.ma.filled(radar.fields[name]["data"].astype("f8"), np.nan)


def _no_return_mask(radar, mask=None, gate_id_field="gate_id", snr_max=5.0):
    """Gates believed to hold no detected scatterer, and where that came from.

    Priority: an explicit mask, then a gate-ID field, then an SNR field, then
    every finite gate. The last is not a failure: on a PPI volume,
    non-detections are the overwhelming majority, and the mode estimator is
    built to find them inside a mixture.
    """
    from radar_palette.gateid.single import NAME_TO_CODE

    shape = (radar.nrays, radar.ngates)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"mask shape {mask.shape} does not match volume {shape}")
        return mask, "explicit"

    if gate_id_field and gate_id_field in radar.fields:
        codes = np.ma.filled(radar.fields[gate_id_field]["data"], -1)
        wanted = [NAME_TO_CODE[name] for name in NO_RETURN_CLASSES]
        return np.isin(codes, wanted), f"{gate_id_field}:{'+'.join(NO_RETURN_CLASSES)}"

    if snr_max is not None:
        for name in FIELD_CANDIDATES["snr"]:
            if name in radar.fields:
                snr = _field_values(radar, name)
                # Generous: this only has to exclude obvious signal. A tight
                # cut here truncates the noise peak the estimator needs.
                return (snr < float(snr_max)) | ~np.isfinite(snr), (
                    f"{name}<{snr_max:g}dB"
                )

    return np.ones(shape, dtype=bool), "all_gates"


def _sweep_row_mask(radar, sweeps):
    """Boolean over rays selecting the requested sweeps."""
    if sweeps is None:
        return np.ones(radar.nrays, dtype=bool)
    rows = np.zeros(radar.nrays, dtype=bool)
    for sweep in np.atleast_1d(sweeps):
        rows[radar.get_slice(int(sweep))] = True
    if not rows.any():
        raise ValueError(f"sweeps={sweeps!r} selected no rays")
    return rows


def _resolve_snr(radar, snr_field="auto"):
    """SNR field name to use arithmetically, or None."""
    if snr_field is None:
        return None
    if snr_field != "auto":
        if snr_field not in radar.fields:
            raise KeyError(f"SNR field {snr_field!r} is not on this volume")
        return snr_field
    for name in FIELD_CANDIDATES["snr"]:
        if name in radar.fields:
            return name
    return None


def mds_profile(
    radar_like,
    field=None,
    mask=None,
    snr_field="auto",
    gate_id_field="gate_id",
    snr_max=5.0,
    sweeps=None,
    method="auto",
    model="power_law",
    margin_db=0.0,
    cap_dbz=None,
    **kwargs,
):
    """Measure the minimum detectable signal against range for a volume.

    Parameters
    ----------
    radar_like : pyart.core.Radar or xarray.DataTree
        Volume to measure. Not modified.
    field : str, optional
        Reflectivity field to use. Default resolves
        :data:`REFLECTIVITY_CANDIDATES`, so an uncorrected field is preferred
        over a corrected one -- the floor is a property of the receiver, and
        attenuation correction moves gates away from it.
    mask : ndarray of bool, optional
        No-return gates, shaped ``(nrays, ngates)``. Only consulted by
        ``method='noise'``; overrides the gate-ID and SNR-threshold routes.
    snr_field : str or None, optional
        SNR field for the exact ``method='snr'`` route. ``"auto"`` resolves the
        usual candidates and is what makes that the default estimator on
        volumes that carry one; pass ``None`` to force a population-based
        estimator instead.
    gate_id_field : str, optional
        Gate-ID field to take :data:`NO_RETURN_CLASSES` from, if present. Set
        to ``None`` to ignore it. Used by ``method='noise'``.
    snr_max : float or None, optional
        Fallback SNR *threshold*, in dB, for selecting no-return gates when no
        gate ID is available. Not the same thing as ``snr_field``, which uses
        SNR arithmetically rather than as a cut.
    sweeps : int or sequence of int, optional
        Restrict the estimate to these sweeps. Default pools the whole volume,
        which is what the measurements support: on a BNF C-SAPR2 PPI the
        per-sweep intercepts agreed to 0.1 dB, while the lowest sweeps on
        their own had too few no-return gates to fit at all.
    method : {'auto', 'snr', 'noise', 'censor'}, optional
        Estimator. ``auto`` prefers the exact ``snr`` route where an SNR field
        exists, then the censoring envelope on thresholded volumes, then the
        noise mode. See :mod:`radar_palette.mdsreplace.profile`.
    model : {'power_law', 'empirical'}, optional
        Evaluation model.
    margin_db : float, optional
        Offset carried on the profile and applied when it is evaluated.
    cap_dbz : float, optional
        Ceiling carried on the profile.
    **kwargs
        Passed to :func:`radar_palette.mdsreplace.profile.estimate_profile`
        (``bin_db``, ``quantile``, ``min_samples``, ``min_range_m``,
        ``fit_slope``, ``smooth_gates``, ``censor_fraction``).

    Returns
    -------
    MdsProfile
        Carrying the fit, the per-range estimate it came from, and provenance
        in ``profile.meta``.

    Examples
    --------
    ::

        profile = mds_profile(radar)
        profile.intercept_dbz          # MDS at 1 km, dBZ
        profile.evaluate([50e3])       # MDS at 50 km
        profile.meta["mask_source"]    # where the no-return gates came from

    Notes
    -----
    The estimate is only as good as the volume's own calibration: it measures
    where *this file's* reflectivity scale puts the noise floor, which is the
    right quantity for filling gates in that same file, and not a substitute
    for a calibrated sensitivity measurement.
    """
    radar, _ = to_pyart_radar(radar_like)
    name = _resolve_reflectivity(radar, field)
    values = _field_values(radar, name)
    no_return, mask_source = _no_return_mask(
        radar, mask=mask, gate_id_field=gate_id_field, snr_max=snr_max
    )
    snr_name = _resolve_snr(radar, snr_field)
    snr_values = None if snr_name is None else _field_values(radar, snr_name)

    rows = _sweep_row_mask(radar, sweeps)
    meta = {
        "field": name,
        "snr_field": snr_name,
        "mask_source": mask_source,
        "sweeps": None if sweeps is None else [int(s) for s in np.atleast_1d(sweeps)],
        "n_rays_used": int(rows.sum()),
        "n_no_return_gates": int(no_return[rows].sum()),
    }
    return estimate_profile(
        values[rows],
        np.asarray(radar.range["data"], dtype="f8"),
        mask=no_return[rows],
        snr=None if snr_values is None else snr_values[rows],
        method=method,
        model=model,
        margin_db=margin_db,
        cap_dbz=cap_dbz,
        meta=meta,
        **kwargs,
    )


def replace_below_mds(
    radar_like,
    gatefilter=None,
    profile=None,
    fields=None,
    include_masked=True,
    include_sentinels=True,
    flag_field="mds_replaced",
    mds_field=None,
    output_flavor=None,
    replace_existing=True,
    **profile_kwargs,
):
    """Replace filtered, masked and fill gates with the MDS at their range.

    Parameters
    ----------
    radar_like : pyart.core.Radar or xarray.DataTree
        Volume to fill. Filled in place on the Py-ART surface; an xradar input
        is converted, filled and converted back.
    gatefilter : pyart.filters.GateFilter, optional
        Gates it *excludes* are replaced. Build one from a gate ID with
        :func:`radar_palette.gateid.meteorological_gatefilter`. Optional: with
        no filter, only masked and fill gates are replaced, which is the
        "grid down to the floor" case with no classification involved.
    profile : MdsProfile, optional
        Measured profile to use. Default measures one with :func:`mds_profile`
        on this volume, passing ``**profile_kwargs``.
    fields : sequence of str, optional
        Fields to fill. Default is the resolved reflectivity plus any of
        :data:`SECONDARY_REFLECTIVITY_CANDIDATES` present. Only ever pass
        dBZ-scaled fields: the fill value is a reflectivity.
    include_masked : bool, optional
        Also replace gates that are masked or non-finite in the field itself.
        On a thresholded volume this is where most of the filling happens.
    include_sentinels : bool, optional
        Also replace gates holding a detected fill sentinel (see
        :func:`radar_palette.mdsreplace.profile.find_sentinel_values`).
    flag_field : str or None, optional
        Name of an ``int8`` provenance field marking replaced gates, or
        ``None`` to skip it. Written by default: a gridded product built from
        filled gates should be able to say which gates were synthetic.
    mds_field : str or None, optional
        If given, also write the evaluated MDS at every gate under this name.
    output_flavor : RadarFlavor or str or None, optional
        Family to return. Defaults to mirroring the input.
    replace_existing : bool, optional
        Overwrite existing ``flag_field`` / ``mds_field``.
    **profile_kwargs
        Passed to :func:`mds_profile` when ``profile`` is not supplied.

    Returns
    -------
    radar : pyart.core.Radar or xarray.DataTree
        The volume with the named fields filled, in the requested family.
    meta : dict
        ``profile`` (the :class:`MdsProfile` used), ``fields``, per-source
        replaced-gate counts, the fill value range, and the fraction of the
        volume that was replaced.

    Raises
    ------
    ValueError
        If the gate filter's shape does not match the volume.

    Examples
    --------
    Gate ID, filter, fill, in the Py-ART flavour::

        from radar_palette.gateid import gate_id, meteorological_gatefilter
        from radar_palette.mdsreplace import replace_below_mds

        radar, _ = gate_id(radar, freezing_level_m=4700.0)
        gatefilter = meteorological_gatefilter(radar)
        radar, meta = replace_below_mds(radar, gatefilter)
        meta["profile"].intercept_dbz
        meta["n_replaced"]

    Reuse one volume's profile on the next volume of a case, so a time series
    of grids shares one floor::

        profile = mds_profile(radar_0)
        radar_1, _ = replace_below_mds(radar_1, gatefilter_1, profile=profile)
    """
    input_flavor = detect_radar_flavor(radar_like)
    output_flavor = resolve_output_flavor(output_flavor, input_flavor)
    radar, _ = to_pyart_radar(radar_like)

    if profile is None:
        profile = mds_profile(radar, **profile_kwargs)
    elif profile_kwargs:
        raise TypeError(
            "profile= and profile-estimation keywords are mutually exclusive; "
            f"got {sorted(profile_kwargs)} alongside a supplied profile"
        )
    if not isinstance(profile, MdsProfile):
        raise TypeError(f"profile must be an MdsProfile, got {type(profile).__name__}")

    primary = profile.meta.get("field") or _resolve_reflectivity(radar)
    if fields is None:
        fields = [primary] + [
            name
            for name in SECONDARY_REFLECTIVITY_CANDIDATES
            if name in radar.fields and name != primary
        ]
    fields = [str(name) for name in fields]
    for name in fields:
        if name not in radar.fields:
            raise KeyError(f"field {name!r} is not on this volume")

    range_m = np.asarray(radar.range["data"], dtype="f8")
    fill_1d = profile.evaluate(range_m)
    fill = np.broadcast_to(fill_1d[None, :], (radar.nrays, radar.ngates))

    excluded = np.zeros((radar.nrays, radar.ngates), dtype=bool)
    n_from_filter = 0
    if gatefilter is not None:
        excluded = np.asarray(gatefilter.gate_excluded, dtype=bool)
        if excluded.shape != (radar.nrays, radar.ngates):
            raise ValueError(
                f"gatefilter covers {excluded.shape} gates but this volume is "
                f"{(radar.nrays, radar.ngates)}; it was probably built on a "
                "different object -- see the accessor for the xradar case"
            )
        excluded = excluded.copy()
        n_from_filter = int(excluded.sum())

    sentinels = list(profile.meta.get("sentinels") or [])
    counts = {"gatefilter": n_from_filter, "masked": 0, "sentinel": 0}
    replaced_any = np.zeros((radar.nrays, radar.ngates), dtype=bool)

    for name in fields:
        data = radar.fields[name]["data"]
        values = np.ma.filled(data.astype("f8"), np.nan)
        target = excluded.copy()
        if include_masked:
            missing = ~np.isfinite(values)
            counts["masked"] += int((missing & ~target).sum())
            target |= missing
        if include_sentinels and sentinels:
            hit = np.zeros_like(target)
            for sentinel in sentinels:
                hit |= values == sentinel
            counts["sentinel"] += int((hit & ~target).sum())
            target |= hit

        values = np.where(target, fill, values)
        radar.fields[name]["data"] = np.ma.masked_invalid(values)
        radar.fields[name]["comment"] = (
            (radar.fields[name].get("comment", "") or "").strip()
            + " Gates removed by the gate filter, masked, or holding a fill "
            "sentinel were replaced by the minimum detectable signal at their "
            f"range ({profile!r}), so downstream gridding has a floor to "
            "interpolate towards. Such gates are not measurements."
        ).strip()
        replaced_any |= target

    if flag_field:
        radar.add_field(
            flag_field,
            {
                "data": np.ma.masked_array(replaced_any.astype("i1"), mask=False),
                "long_name": "Gate replaced by the minimum detectable signal",
                "units": "",
                "flag_values": np.array([0, 1], dtype="i1"),
                "flag_meanings": "measured replaced",
                "comment": (
                    "1 marks a synthetic value at the sensitivity limit for "
                    "that range, not a measurement: the radar reported "
                    "nothing usable there. Carry this flag into any product "
                    "built from the filled volume."
                ),
            },
            replace_existing=replace_existing,
        )

    if mds_field:
        radar.add_field(
            mds_field,
            {
                "data": np.ma.masked_invalid(np.array(fill, dtype="f4")),
                "long_name": "Minimum detectable signal at this range",
                "standard_name": "minimum_detectable_signal",
                "units": "dBZ",
                "comment": (
                    f"{profile!r}; method={profile.method}, "
                    f"model={profile.model}, margin={profile.margin_db} dB."
                ),
            },
            replace_existing=replace_existing,
        )

    total = int(replaced_any.sum())
    meta = {
        "profile": profile,
        "fields": fields,
        "n_replaced": total,
        "n_replaced_by_source": counts,
        "replaced_fraction": total / float(radar.nrays * radar.ngates),
        "fill_dbz_range": (float(np.nanmin(fill_1d)), float(np.nanmax(fill_1d))),
        "flag_field": flag_field,
        "mds_field": mds_field,
    }
    return to_radar_flavor(radar, output_flavor), meta
