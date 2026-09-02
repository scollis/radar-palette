"""MDS methods on the ``radarpalette`` xarray accessor.

Importing :mod:`radar_palette` adds two methods to the ``radarpalette``
accessor that :mod:`radar_palette.gateid.accessor` registers on
:class:`xarray.DataTree`::

    import xradar as xd
    import radar_palette  # noqa: F401  -- registers the accessor

    tree = xd.io.open_cfradial1_datatree("cfrad.20260820_041403.nc")
    tree.radarpalette.gateid(freezing_level_m=4670.0)
    tree.radarpalette.mdsreplace(exclude_field="gate_id")

    tree["sweep_0"]["mds_replaced"]   # provenance, on the caller's own tree

Both methods mutate the tree they are called on and return it, so calls chain
and a caller finds the result on their own object rather than on a copy they
did not keep.

Why the methods are attached rather than registered separately
-------------------------------------------------------------
``xarray`` allows one accessor per name, and ``radarpalette`` is already taken
by the gate-ID accessor. Registering ``radarpalette_mds`` would give the user
two namespaces for one package, so instead these methods are set onto the
existing accessor class by :func:`register_mds_methods`, which the subpackage
calls on import. The seam is explicit and one-directional: this module imports
the gate-ID accessor class, never the reverse.

Why not simply call the functional entry point
----------------------------------------------
The same reason the gate-ID accessor does not:
:func:`radar_palette.mdsreplace.replace_below_mds` computes on the Py-ART
surface and converts back through :mod:`radar_palette.io.flavors`, which
writes CfRadial and reopens it. That hands back a different object, and the
round trip is not shape-preserving -- Py-ART and xradar disagree on how some
CfRadial1 sweeps unpack. So this fills the caller's own sweeps directly, using
the same estimator and the same evaluation as the functional path.

The ray-order trap this module has to defend against
----------------------------------------------------
A Py-ART :class:`~pyart.filters.GateFilter` is indexed by Py-ART's ray order.
On measured BNF volumes ``radar.get_slice(0)`` and ``tree["sweep_0"]`` hold
the *same rays in a different rotational order*, so copying a filter's mask
onto a tree by position misaligns whole azimuths silently -- and a silent
azimuthal roll in a QC mask is exactly the kind of error that survives to a
figure. :func:`align_gatefilter_to_tree` therefore matches rays by azimuth and
refuses to guess: it raises unless every tree ray finds one distinct Py-ART ray
within a tolerance.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "align_gatefilter_to_tree",
    "register_mds_methods",
]

_REGISTERED = False


def _sweep_keys(tree):
    """Sweep child names in Py-ART's sweep order (numeric, not lexical)."""
    keys = [str(k) for k in tree.children if str(k).startswith("sweep")]

    def _index(key):
        digits = "".join(ch for ch in key if ch.isdigit())
        return int(digits) if digits else 0

    return sorted(keys, key=_index)


def align_gatefilter_to_tree(gatefilter, tree, tolerance_deg=0.1):
    """Map a Py-ART gate filter's excluded mask onto a tree's sweeps.

    Parameters
    ----------
    gatefilter : pyart.filters.GateFilter
        Filter built on a Py-ART view of *this* volume.
    tree : xarray.DataTree
        Tree whose sweeps the mask is wanted on.
    tolerance_deg : float, optional
        Largest azimuth difference accepted when pairing a tree ray with a
        Py-ART ray.

    Returns
    -------
    dict
        Sweep name -> boolean array shaped like that sweep's reflectivity,
        ``True`` where the filter excludes the gate.

    Raises
    ------
    AttributeError
        If the filter does not carry the radar it was built on, so no
        alignment is possible.
    ValueError
        If a sweep's ray or gate count disagrees, or if the azimuth match is
        not one-to-one within ``tolerance_deg``. Refusing beats returning a
        rolled mask.

    Notes
    -----
    Matching is on azimuth alone, which is well defined for a PPI sweep. For
    an RHI sweep azimuth is nearly constant and carries no ray identity, so
    alignment falls back to elevation. A volume mixing both is handled sweep
    by sweep.
    """
    radar = getattr(gatefilter, "_radar", None)
    if radar is None:
        raise AttributeError(
            "this gate filter does not carry the radar it was built on, so "
            "its rays cannot be matched to the tree; pass mask= or "
            "exclude_field= instead"
        )
    excluded = np.asarray(gatefilter.gate_excluded, dtype=bool)
    keys = _sweep_keys(tree)
    if radar.nsweeps != len(keys):
        raise ValueError(
            f"gate filter covers {radar.nsweeps} sweeps but the tree has "
            f"{len(keys)}; it was built on a different volume"
        )

    out = {}
    for index, key in enumerate(keys):
        dataset = tree[key].to_dataset()
        sl = radar.get_slice(index)
        sweep_mask = excluded[sl]

        n_tree = int(dataset.sizes.get("azimuth", dataset.sizes.get("time", 0)))
        if sweep_mask.shape[0] != n_tree:
            raise ValueError(
                f"{key}: gate filter has {sweep_mask.shape[0]} rays, tree has "
                f"{n_tree}; the two object families unpacked this sweep "
                "differently and no gate-wise mapping exists"
            )
        n_gates = int(dataset.sizes["range"])
        if sweep_mask.shape[1] < n_gates:
            raise ValueError(
                f"{key}: gate filter has {sweep_mask.shape[1]} gates, tree "
                f"has {n_gates}"
            )
        sweep_mask = sweep_mask[:, :n_gates]

        # Azimuth identifies a PPI ray; on an RHI it is constant and useless,
        # so elevation identifies the ray instead.
        tree_az = np.asarray(dataset["azimuth"].values, dtype="f8")
        radar_az = np.asarray(radar.azimuth["data"][sl], dtype="f8")
        coordinate = "azimuth"
        if np.nanmax(tree_az) - np.nanmin(tree_az) < 1.0:
            tree_az = np.asarray(dataset["elevation"].values, dtype="f8")
            radar_az = np.asarray(radar.elevation["data"][sl], dtype="f8")
            coordinate = "elevation"

        order = np.argsort(radar_az, kind="stable")
        sorted_az = radar_az[order]
        position = np.searchsorted(sorted_az, tree_az)
        position = np.clip(position, 1, sorted_az.size - 1)
        left = sorted_az[position - 1]
        right = sorted_az[np.minimum(position, sorted_az.size - 1)]
        pick = np.where(
            np.abs(tree_az - left) <= np.abs(tree_az - right), position - 1, position
        )
        rows = order[pick]
        residual = np.abs(radar_az[rows] - tree_az)

        if np.nanmax(residual) > float(tolerance_deg):
            raise ValueError(
                f"{key}: worst {coordinate} mismatch when pairing rays is "
                f"{np.nanmax(residual):.3f} deg, above the "
                f"{tolerance_deg:g} deg tolerance; the filter and the tree "
                "do not describe the same rays"
            )
        if np.unique(rows).size != rows.size:
            raise ValueError(
                f"{key}: {coordinate} matching is not one-to-one "
                f"({rows.size - np.unique(rows).size} rays paired twice); "
                "the sweep probably contains repeated angles"
            )
        out[key] = sweep_mask[rows]
    return out


def _sweep_reflectivity_name(dataset, field=None):
    """Reflectivity variable to use on one sweep."""
    from radar_palette.mdsreplace.entry_points import REFLECTIVITY_CANDIDATES

    if field is not None:
        if field not in dataset:
            raise KeyError(f"field {field!r} is not on this sweep")
        return field
    for name in REFLECTIVITY_CANDIDATES:
        if name in dataset:
            return name
    raise KeyError(
        f"no reflectivity variable found; looked for {list(REFLECTIVITY_CANDIDATES)}"
    )


def _sweep_snr_name(dataset, snr_field="auto"):
    """SNR variable to use arithmetically on one sweep, or None."""
    from radar_palette.gateid.features import FIELD_CANDIDATES

    if snr_field is None:
        return None
    if snr_field != "auto":
        if snr_field not in dataset:
            raise KeyError(f"SNR variable {snr_field!r} is not on this sweep")
        return snr_field
    for name in FIELD_CANDIDATES["snr"]:
        if name in dataset:
            return name
    return None


def _accessor_mds_profile(
    self,
    field=None,
    mask=None,
    snr_field="auto",
    exclude_field=None,
    gate_id_field="gate_id",
    snr_max=5.0,
    method="auto",
    model="power_law",
    margin_db=0.0,
    cap_dbz=None,
    **kwargs,
):
    """Measure the minimum detectable signal from the bound tree.

    Sweeps are pooled, because the floor is a receiver property rather than a
    sweep property: on a BNF C-SAPR2 PPI the per-sweep intercepts agreed to
    0.1 dB, and the lowest sweeps alone had too few no-return gates to fit.
    Sweeps whose range coordinate differs from the majority are left out of
    the fit and named in ``profile.meta["skipped_sweeps"]``; the fitted
    profile is a function of range, so it still evaluates on them.

    Parameters
    ----------
    field : str, optional
        Reflectivity variable. Default resolves the usual candidates.
    mask : dict, optional
        Sweep name -> boolean array of no-return gates.
    snr_field : str or None, optional
        SNR variable for the exact ``method='snr'`` route; ``"auto"`` resolves
        the usual candidates, ``None`` forces a population-based estimator.
    exclude_field : str, optional
        Variable holding a classification, e.g. ``"gate_id"``. Gates whose
        code is one of :data:`radar_palette.mdsreplace.NO_RETURN_CLASSES` form
        the no-return population.
    gate_id_field : str, optional
        Kept for symmetry with the functional entry point; ``exclude_field``
        takes precedence.
    snr_max : float or None, optional
        Fallback SNR ceiling in dB when no classification is available.
    method, model, margin_db, cap_dbz, **kwargs
        As :func:`radar_palette.mdsreplace.mds_profile`.

    Returns
    -------
    MdsProfile

    Examples
    --------
    ::

        profile = tree.radarpalette.mds_profile()
        profile.intercept_dbz
    """
    from radar_palette.mdsreplace.profile import estimate_profile

    tree = self.datatree
    keys = _sweep_keys(tree)
    if not keys:
        raise ValueError("this tree has no sweep groups")

    stacks, signature_counts, per_sweep_range = [], {}, {}
    for key in keys:
        dataset = tree[key].to_dataset()
        name = _sweep_reflectivity_name(dataset, field)
        ranges = np.asarray(dataset["range"].values, dtype="f8")
        per_sweep_range[key] = ranges
        signature = (ranges.size, float(ranges[0]), float(ranges[-1]))
        signature_counts[signature] = signature_counts.get(signature, 0) + 1

    dominant = max(signature_counts, key=signature_counts.get)
    skipped = []
    values_used, masks_used, snr_used = [], [], []
    mask_source, snr_name = None, None

    for key in keys:
        ranges = per_sweep_range[key]
        if (ranges.size, float(ranges[0]), float(ranges[-1])) != dominant:
            skipped.append(key)
            continue
        dataset = tree[key].to_dataset()
        name = _sweep_reflectivity_name(dataset, field)
        values = np.asarray(dataset[name].values, dtype="f8")
        sweep_mask, mask_source = _sweep_no_return_mask(
            dataset,
            values,
            mask=None if mask is None else mask.get(key),
            exclude_field=exclude_field or gate_id_field,
            snr_max=snr_max,
        )
        snr_name = _sweep_snr_name(dataset, snr_field)
        if snr_name is not None:
            snr_used.append(np.asarray(dataset[snr_name].values, dtype="f8"))
        values_used.append(values)
        masks_used.append(sweep_mask)
        stacks.append(key)

    meta = {
        "field": field or _sweep_reflectivity_name(tree[stacks[0]].to_dataset(), field),
        "snr_field": snr_name,
        "mask_source": mask_source,
        "sweeps_used": stacks,
        "skipped_sweeps": skipped,
        "n_no_return_gates": int(sum(int(m.sum()) for m in masks_used)),
    }
    return estimate_profile(
        np.concatenate(values_used, axis=0),
        per_sweep_range[stacks[0]],
        mask=np.concatenate(masks_used, axis=0),
        snr=np.concatenate(snr_used, axis=0) if len(snr_used) == len(stacks) else None,
        method=method,
        model=model,
        margin_db=margin_db,
        cap_dbz=cap_dbz,
        meta=meta,
        **kwargs,
    )


def _sweep_no_return_mask(dataset, values, mask=None, exclude_field=None, snr_max=5.0):
    """No-return gates for one sweep, and where they came from."""
    from radar_palette.gateid.features import FIELD_CANDIDATES
    from radar_palette.gateid.single import NAME_TO_CODE
    from radar_palette.mdsreplace.entry_points import NO_RETURN_CLASSES

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match reflectivity {values.shape}"
            )
        return mask, "explicit"

    if exclude_field and exclude_field in dataset:
        codes = np.asarray(dataset[exclude_field].values)
        wanted = [NAME_TO_CODE[name] for name in NO_RETURN_CLASSES]
        return np.isin(codes, wanted), (
            f"{exclude_field}:{'+'.join(NO_RETURN_CLASSES)}"
        )

    if snr_max is not None:
        for name in FIELD_CANDIDATES["snr"]:
            if name in dataset:
                snr = np.asarray(dataset[name].values, dtype="f8")
                return (snr < float(snr_max)) | ~np.isfinite(snr), (
                    f"{name}<{snr_max:g}dB"
                )

    return np.ones(values.shape, dtype=bool), "all_gates"


def _accessor_mdsreplace(
    self,
    gatefilter=None,
    mask=None,
    exclude_field=None,
    exclude_values=None,
    profile=None,
    fields=None,
    include_masked=True,
    include_sentinels=True,
    flag_field="mds_replaced",
    mds_field=None,
    tolerance_deg=0.1,
    **profile_kwargs,
):
    """Fill filtered, masked and fill gates with the MDS, in place.

    Parameters
    ----------
    gatefilter : pyart.filters.GateFilter, optional
        Filter built on a Py-ART view of this same volume. Its rays are
        matched to the tree's by azimuth (see
        :func:`align_gatefilter_to_tree`), which raises rather than guessing
        if the two do not describe the same rays.
    mask : dict, optional
        Sweep name -> boolean array, ``True`` where a gate should be
        replaced. Bypasses all alignment; the safest route when the mask was
        computed on the tree itself.
    exclude_field : str, optional
        Variable holding a classification, e.g. ``"gate_id"`` as written by
        ``tree.radarpalette.gateid(...)``. Gates whose code is in
        ``exclude_values`` are replaced.
    exclude_values : sequence of int or str, optional
        Codes or class names to replace. Defaults to every non-meteorological
        gate-ID class -- ``no_scatter``, ``clutter``, ``biological`` and
        ``multi_trip`` -- plus ``unclassified``, which is what a
        meteorological gate filter removes.
    profile : MdsProfile, optional
        Profile to use; default measures one from this tree.
    fields : sequence of str, optional
        Variables to fill. Default is the resolved reflectivity on each sweep.
    include_masked, include_sentinels, flag_field, mds_field
        As :func:`radar_palette.mdsreplace.replace_below_mds`.
    tolerance_deg : float, optional
        Azimuth tolerance for gate-filter alignment.
    **profile_kwargs
        Passed to the profile measurement when ``profile`` is not supplied.

    Returns
    -------
    xarray.DataTree
        The same tree, with the reflectivity filled and ``flag_field``
        attached to every sweep touched. A summary sits on
        ``tree.attrs["mdsreplace_meta"]``.

    Raises
    ------
    ValueError
        If more than one exclusion source is given, or if a gate filter cannot
        be aligned.

    Examples
    --------
    Classify, then fill everything that is not meteorological echo::

        tree.radarpalette.gateid(freezing_level_m=4700.0)
        tree.radarpalette.mdsreplace(exclude_field="gate_id")
        tree.attrs["mdsreplace_meta"]["n_replaced"]
    """
    import xarray as xr

    from radar_palette.gateid.single import NAME_TO_CODE, NON_MET
    from radar_palette.mdsreplace.profile import MdsProfile

    if "output_flavor" in profile_kwargs:
        raise TypeError(
            "output_flavor is not accepted here: an accessor on a DataTree "
            "returns a DataTree. Call "
            "radar_palette.mdsreplace.replace_below_mds() to choose the "
            "output family."
        )
    sources = [s is not None for s in (gatefilter, mask, exclude_field)]
    if sum(sources) > 1:
        raise ValueError(
            "give at most one of gatefilter=, mask= and exclude_field=; they "
            "are three ways to say the same thing and cannot be merged safely"
        )

    tree = self.datatree
    keys = _sweep_keys(tree)
    if not keys:
        raise ValueError("this tree has no sweep groups")

    if profile is None:
        profile = self.mds_profile(
            exclude_field=exclude_field if exclude_field else None,
            **profile_kwargs,
        )
    elif profile_kwargs:
        raise TypeError(
            "profile= and profile-estimation keywords are mutually exclusive; "
            f"got {sorted(profile_kwargs)} alongside a supplied profile"
        )
    if not isinstance(profile, MdsProfile):
        raise TypeError(f"profile must be an MdsProfile, got {type(profile).__name__}")

    if exclude_values is None:
        codes = [NAME_TO_CODE[name] for name in NON_MET] + [0]
    else:
        codes = [
            NAME_TO_CODE[v] if isinstance(v, str) else int(v) for v in exclude_values
        ]

    aligned = None
    if gatefilter is not None:
        aligned = align_gatefilter_to_tree(
            gatefilter, tree, tolerance_deg=tolerance_deg
        )

    sentinels = list(profile.meta.get("sentinels") or [])
    counts = {
        "gatefilter": 0,
        "classified": 0,
        "explicit": 0,
        "masked": 0,
        "sentinel": 0,
    }
    filled_sweeps, total = {}, 0

    for key in keys:
        dataset = tree[key].to_dataset()
        names = (
            [str(f) for f in fields]
            if fields is not None
            else [_sweep_reflectivity_name(dataset)]
        )
        ranges = np.asarray(dataset["range"].values, dtype="f8")
        fill_1d = profile.evaluate(ranges)

        target = None
        if aligned is not None:
            target = np.asarray(aligned[key], dtype=bool)
            counts["gatefilter"] += int(target.sum())
        elif mask is not None:
            if key not in mask:
                target = None
            else:
                target = np.asarray(mask[key], dtype=bool)
                counts["explicit"] += int(target.sum())
        elif exclude_field is not None:
            if exclude_field not in dataset:
                raise KeyError(f"{key}: no variable {exclude_field!r} to exclude on")
            target = np.isin(np.asarray(dataset[exclude_field].values), codes)
            counts["classified"] += int(target.sum())

        new = {}
        replaced_any = None
        for name in names:
            if name not in dataset:
                continue
            values = np.asarray(dataset[name].values, dtype="f8")
            fill = np.broadcast_to(fill_1d[None, :], values.shape)
            gate_target = (
                np.zeros(values.shape, dtype=bool) if target is None else target.copy()
            )
            if include_masked:
                missing = ~np.isfinite(values)
                counts["masked"] += int((missing & ~gate_target).sum())
                gate_target |= missing
            if include_sentinels and sentinels:
                hit = np.zeros_like(gate_target)
                for sentinel in sentinels:
                    hit |= values == sentinel
                counts["sentinel"] += int((hit & ~gate_target).sum())
                gate_target |= hit

            filled = np.where(gate_target, fill, values)
            attrs = dict(dataset[name].attrs)
            attrs["comment"] = (
                (attrs.get("comment", "") or "").strip()
                + " Filtered, masked and fill gates replaced by the minimum "
                f"detectable signal at their range ({profile!r}). Replaced "
                "gates are not measurements; see mds_replaced."
            ).strip()
            new[name] = xr.DataArray(
                filled.astype(dataset[name].dtype),
                dims=dataset[name].dims,
                attrs=attrs,
            )
            replaced_any = (
                gate_target if replaced_any is None else (replaced_any | gate_target)
            )

        if replaced_any is None:
            continue
        dims = dataset[names[0]].dims
        if flag_field:
            new[flag_field] = xr.DataArray(
                replaced_any.astype("i1"),
                dims=dims,
                attrs={
                    "long_name": "Gate replaced by the minimum detectable signal",
                    "units": "",
                    "flag_values": np.array([0, 1], dtype="i1"),
                    "flag_meanings": "measured replaced",
                    "comment": (
                        "1 marks a synthetic value at the sensitivity limit "
                        "for that range, not a measurement."
                    ),
                },
            )
        if mds_field:
            new[mds_field] = xr.DataArray(
                np.broadcast_to(fill_1d[None, :], replaced_any.shape).astype("f4"),
                dims=dims,
                attrs={
                    "long_name": "Minimum detectable signal at this range",
                    "standard_name": "minimum_detectable_signal",
                    "units": "dBZ",
                    "comment": repr(profile),
                },
            )
        tree[key].dataset = dataset.assign(new)
        filled_sweeps[key] = sorted(new)
        total += int(replaced_any.sum())

    tree.attrs["mdsreplace_meta"] = {
        "profile": profile.to_dict(),
        "filled_sweeps": filled_sweeps,
        "n_replaced": total,
        "n_replaced_by_source": counts,
        "exclude_values": codes if exclude_field is not None else None,
        "flag_field": flag_field,
        "mds_field": mds_field,
    }
    return tree


def register_mds_methods():
    """Attach ``mds_profile`` and ``mdsreplace`` to the ``radarpalette`` accessor.

    Idempotent, and a no-op if the accessor class cannot be imported, so
    importing the package never fails because of it.

    Returns
    -------
    bool
        Whether the methods are attached.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from radar_palette.gateid.accessor import RadarPaletteDataTreeAccessor
    except ImportError:  # pragma: no cover - only if gateid is unavailable
        return False

    RadarPaletteDataTreeAccessor.mds_profile = _accessor_mds_profile
    RadarPaletteDataTreeAccessor.mdsreplace = _accessor_mdsreplace
    _REGISTERED = True
    return True
