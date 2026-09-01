"""An xarray accessor so gate-ID reads as a method on an xradar object.

Importing :mod:`radar_palette` registers a ``radarpalette`` accessor on
:class:`xarray.DataTree`, alongside the ``xradar`` and ``pyart`` accessors that
those packages register::

    import xradar as xd
    import radar_palette  # noqa: F401  -- registers the accessor

    tree = xd.io.open_cfradial1_datatree("cfrad.20260820_041403.nc")
    tree.radarpalette.gateid(freezing_level_m=4670.0)

    tree["sweep_0"]["gate_id"]        # the classification is on the tree

The accessor mutates the tree it is called on and returns it, so a caller who
follows the example above finds ``gate_id`` on their own object rather than on
a copy they did not keep. It also returns the tree so calls can be chained.

Why this does not simply call the functional entry point
--------------------------------------------------------
:func:`radar_palette.gateid.gate_id` computes on the Py-ART surface and
converts back through :mod:`radar_palette.io.flavors`. That conversion writes
CfRadial and reopens it, which is fine for a function returning a new object
but has two consequences an accessor cannot accept.

First, the tree handed back is a different object, so the user's own tree would
stay unclassified. Second, and worse, the round trip is not shape-preserving:
on an ARM BNF C-SAPR2 volume it returned 930 rays for a sweep that had 931,
because Py-ART and xradar disagree on how a CfRadial1 sweep is unpacked.
Copying results back into the caller's tree would then misalign every gate in
that sweep, silently.

So the accessor classifies each sweep of the caller's tree directly, using the
same feature builder and the same physics core as the functional path. Nothing
round-trips, the geometry is exactly the caller's own, and the two paths cannot
disagree on the classification because they share
:func:`radar_palette.gateid.single.classify`.
"""

from __future__ import annotations

__all__ = ["RadarPaletteDataTreeAccessor", "register_accessors"]

_REGISTERED = False


def _sweep_geometry(dataset, datatree):
    """Gate altitude and slant range for one sweep, georeferenced if needed."""
    import xradar

    if "z" not in dataset.coords:
        root = datatree.to_dataset() if hasattr(datatree, "to_dataset") else None
        for coord in ("latitude", "longitude", "altitude"):
            missing_coord = coord not in dataset.coords
            if missing_coord and root is not None and coord in root:
                dataset = dataset.assign_coords({coord: root[coord]})
        dataset = xradar.georeference.get_x_y_z(dataset)
    return dataset


class RadarPaletteDataTreeAccessor:
    """``radarpalette`` accessor on :class:`xarray.DataTree`.

    Attributes
    ----------
    datatree : xarray.DataTree
        The tree this accessor is bound to.
    """

    def __init__(self, datatree):
        self.datatree = datatree

    def gateid(self, *args, **kwargs):
        """Classify the dominant scatterer and attach the result in place.

        Accepts the same arguments as :func:`radar_palette.gateid.gate_id`,
        except ``output_flavor``, which is fixed: an accessor on a
        ``DataTree`` returns a ``DataTree``.

        The classification metadata that the functional entry point returns is
        stored on the tree's attributes as ``gateid_meta`` rather than being
        returned, so the return value can be the tree itself and calls chain.
        Use :func:`radar_palette.gateid.gate_id` directly if the metadata is
        wanted as a value.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to :func:`radar_palette.gateid.gate_id`. The most common
            is ``freezing_level_m``.

        Returns
        -------
        xarray.DataTree
            The same tree, now carrying ``gate_id`` and ``gate_id_margin``.

        Raises
        ------
        TypeError
            If ``output_flavor`` is passed. Requesting a Py-ART object from an
            accessor on a tree is contradictory; call the function instead.

        Examples
        --------
        Classify and keep the metadata::

            tree.radarpalette.gateid(freezing_level_m=4670.0)
            tree.attrs["gateid_meta"]["class_counts"]

        Chain onto the call::

            counts = tree.radarpalette.gateid(
                freezing_level_m=4670.0
            )["sweep_0"]["gate_id"]
        """
        if "output_flavor" in kwargs:
            raise TypeError(
                "output_flavor is not accepted here: an accessor on a "
                "DataTree returns a DataTree. Call "
                "radar_palette.gateid.gate_id() to choose the output family."
            )
        meta = self._classify_sweeps(*args, **kwargs)
        self.datatree.attrs["gateid_meta"] = meta
        return self.datatree

    def mdsreplace(
        self,
        excluded,
        field_name=None,
        output_field_name=None,
        gate_id_field="gate_id",
        no_return_code=None,
        slope_db_per_decade=None,
        percentile=50.0,
        degree=None,
        min_range_m=0.0,
        min_range_bins=6,
        never_increase=True,
    ):
        """Replace excluded gates with the fitted minimum detectable signal.

        Fills gates that carry no usable measurement with the floor the radar
        would have reported, so objective analysis has something to grid down
        to instead of extrapolating an isolated echo into empty space. See
        :func:`radar_palette.gateid.mdsreplace` for the fit.

        Parameters
        ----------
        excluded : mapping of str to ndarray
            Sweep name -> boolean mask of gates to replace, on that sweep's own
            geometry. A mapping is required rather than a
            :class:`pyart.filters.GateFilter` for the same reason this accessor
            does not round-trip: Py-ART and xradar disagree on ray count and
            ordering, so a flat volume-wide filter cannot be trusted to align
            with this tree. Sweeps absent from the mapping are left alone.
        gate_id_field : str, optional
            Variable holding the gate classification.
        no_return_code : int, optional
            Class code marking no scatterer. Defaults to ``no_scatter``.
        slope_db_per_decade, percentile, degree, min_range_m, min_range_bins
            Passed to :func:`radar_palette.gateid.mdsreplace.fit_mds`.
        never_increase : bool, optional
            Keep the measured value at any excluded gate already weaker than
            the fitted floor, so the fill can never add apparent signal.

        Returns
        -------
        xarray.DataTree
            The same tree, carrying the filled field and ``<output>_mds``, with
            the fit recorded in ``tree.attrs['mdsreplace_meta']``.

        Raises
        ------
        TypeError
            If ``excluded`` is not a mapping, which most likely means a
            ``GateFilter`` was passed.
        ValueError
            If a named sweep lacks the required fields, or a mask does not
            match its sweep's shape.

        Examples
        --------
        Fill the gates the gate ID rejected::

            tree.radarpalette.gateid(freezing_level_m=4670.0)
            keep = tree["sweep_0"]["gate_id"].isin([5, 6, 7, 8, 9, 10])
            tree.radarpalette.mdsreplace({"sweep_0": ~keep.values})
        """
        import numpy as np
        import xarray as xr

        from radar_palette.gateid.features import resolve_fields
        from radar_palette.gateid.mdsreplace import (
            RANGE_SQUARE_SLOPE_DB,
            fit_mds,
            replace_with_mds,
        )
        from radar_palette.gateid.single import NAME_TO_CODE

        if not hasattr(excluded, "items"):
            raise TypeError(
                "excluded must map sweep names to boolean gate masks on this "
                "tree's own geometry, not a GateFilter"
            )
        if slope_db_per_decade is None:
            slope_db_per_decade = RANGE_SQUARE_SLOPE_DB
        no_return_code = (
            NAME_TO_CODE["no_scatter"] if no_return_code is None else no_return_code
        )
        fits, replaced, processed = {}, 0, {}
        for key in self.datatree.children:
            if not str(key).startswith("sweep") or str(key) not in excluded:
                continue
            dataset = self.datatree[key].to_dataset()
            resolved, _ = resolve_fields(dataset.data_vars)
            if "z" not in resolved:
                raise ValueError(f"{key} has no reflectivity field")
            source_name = field_name or resolved["z"]
            if source_name not in dataset or gate_id_field not in dataset:
                raise ValueError(f"{key} is missing {source_name!r} or {gate_id_field!r}")
            target_name = output_field_name or source_name
            mask = np.asarray(excluded[str(key)], dtype=bool)
            source = dataset[source_name]
            if mask.shape != source.shape:
                raise ValueError(f"excluded mask shape does not match {key}/{source_name}")
            mds, info = fit_mds(
                source.values,
                dataset["range"].values,
                np.asarray(dataset[gate_id_field].values) == no_return_code,
                slope_db_per_decade=slope_db_per_decade,
                percentile=percentile,
                degree=degree,
                min_range_m=min_range_m,
                min_range_bins=min_range_bins,
            )
            values = replace_with_mds(
                source.values, mds, mask, never_increase=never_increase
            )
            attrs = dict(source.attrs)
            attrs.update(
                {
                    "long_name": (
                        "Reflectivity with excluded gates replaced by "
                        "minimum detectable signal"
                    ),
                    "comment": (
                        "Excluded gates replaced by the sweep's fitted "
                        "no_scatter floor; measurements are unchanged."
                    ),
                    "ancillary_variables": f"{gate_id_field} {target_name}_mds",
                }
            )
            self.datatree[key].dataset = dataset.assign(
                {
                    target_name: xr.DataArray(values, dims=source.dims, attrs=attrs),
                    f"{target_name}_mds": xr.DataArray(
                        mds, dims=source.dims,
                        attrs={
                            "long_name": "Fitted minimum detectable signal",
                            "units": source.attrs.get("units", "dBZ"),
                        },
                    ),
                }
            )
            count = int(mask.sum())
            fits[str(key)] = dict(n_replaced=count, **info)
            processed[str(key)] = [target_name, f"{target_name}_mds"]
            replaced += count
        self.datatree.attrs["mdsreplace_meta"] = {
            "gate_id_field": gate_id_field,
            "no_return_code": int(no_return_code),
            "n_replaced": replaced,
            "never_increase": bool(never_increase),
            "fits": fits,
            "processed_sweeps": processed,
        }
        return self.datatree

    def _classify_sweeps(
        self,
        freezing_level_m=None,
        temperature_field=None,
        field_name="gate_id",
        texture_window=4,
        add_margin=True,
        snr_min=3.0,
        vel_texture_max=None,
        min_run=3,
        despeckle_keep_dbz=30.0,
        incoherent_frac=None,
    ):
        """Classify every usable sweep of the bound tree, in place."""
        import numpy as np
        import xarray as xr

        from radar_palette.gateid import single
        from radar_palette.gateid.entry_points import (
            PPI_MODES,
            RHI_MODES,
        )
        from radar_palette.gateid.features import build_features, resolve_fields

        if incoherent_frac is None:
            incoherent_frac = single.INCOHERENT_TEXTURE_FRAC

        from radar_palette.gateid.entry_points import TRIP_CEILING_MARGIN_M
        from radar_palette.gateid.multitrip import volume_echo_top

        tree = self.datatree

        # The second-trip ceiling describes the storm, so it is derived once
        # over every usable sweep. A low sweep's own echo top is a property of
        # how high its beam reaches, and using it vetoes second trip at
        # exactly the long ranges where second trip is most likely.
        sweep_keys = [k for k in tree.children if str(k).startswith("sweep")]
        trip_ceiling_m, ceiling_source = None, "sweep_echo_top"
        if len(sweep_keys) > 1:
            pairs = []
            for key in sweep_keys:
                data = _sweep_geometry(tree[key].to_dataset(), tree)
                names, _ = resolve_fields(data.data_vars)
                if "z" not in names or "z" not in data.coords:
                    continue
                pairs.append(
                    (
                        np.asarray(data[names["z"]].values, dtype="f8"),
                        np.asarray(data.coords["z"].values, dtype="f8"),
                    )
                )
            top = volume_echo_top(pairs) if pairs else np.nan
            if np.isfinite(top):
                trip_ceiling_m = float(top) + TRIP_CEILING_MARGIN_M
                ceiling_source = "volume_echo_top"

        counts = dict.fromkeys(single.CLASSES.values(), 0)
        classified, skipped = {}, []
        temp_source, resolved_names, missing_names = "none", {}, []
        n_noise = n_speck = 0

        labels = [single.CLASSES[c] for c in sorted(single.CLASSES)]
        notes = ",".join(f"{c}:{single.CLASSES[c]}" for c in sorted(single.CLASSES))

        for key in tree.children:
            if not str(key).startswith("sweep"):
                continue
            dataset = tree[key].to_dataset()

            mode = dataset["sweep_mode"].values if "sweep_mode" in dataset else b""
            mode = mode.item() if hasattr(mode, "item") else mode
            if isinstance(mode, bytes):
                mode = mode.decode(errors="replace")
            mode = str(mode).strip().lower()

            elevation = np.asarray(dataset["elevation"].values, dtype="f8")
            elevation = elevation[np.isfinite(elevation)]
            span = (
                float(elevation.max() - elevation.min()) if elevation.size else np.nan
            )
            usable = (mode.startswith(PPI_MODES) and span <= 5.0) or (
                mode.startswith(RHI_MODES) and span >= 20.0
            )
            if not usable:
                skipped.append(
                    {"sweep": str(key), "sweep_mode": mode, "elevation_span": span}
                )
                continue

            dataset = _sweep_geometry(dataset, tree)
            resolved, missing = resolve_fields(dataset.data_vars)
            if "z" not in resolved or "rhohv" not in resolved:
                skipped.append(
                    {
                        "sweep": str(key),
                        "sweep_mode": mode,
                        "reason": f"missing {sorted(missing)}",
                    }
                )
                continue
            resolved_names, missing_names = resolved, missing

            moments = {
                short: np.asarray(dataset[name].values, dtype="f8")
                for short, name in resolved.items()
            }

            nyquist = np.nan
            if "nyquist_velocity" in dataset:
                values = np.asarray(dataset["nyquist_velocity"].values, dtype="f8")
                values = values[np.isfinite(values)]
                if values.size:
                    nyquist = float(np.median(values))

            temperature = None
            if temperature_field and temperature_field in dataset:
                temperature = np.asarray(dataset[temperature_field].values, dtype="f8")

            prt = None
            if "prt" in dataset:
                values = np.asarray(dataset["prt"].values, dtype="f8")
                values = values[np.isfinite(values)]
                if values.size:
                    prt = float(np.median(values))
            altitude = 0.0
            for source in (dataset.coords, dataset, tree.to_dataset()):
                if "altitude" in source:
                    altitude = float(np.asarray(source["altitude"].values).ravel()[0])
                    break

            features, fmeta = build_features(
                moments,
                gate_altitude=np.asarray(dataset.coords["z"].values, dtype="f8"),
                gate_range=np.asarray(dataset["range"].values, dtype="f8"),
                nyquist=nyquist,
                temperature=temperature,
                freezing_level_m=freezing_level_m,
                texture_window=texture_window,
                elevation=np.asarray(dataset["elevation"].values, dtype="f8"),
                prt=prt,
                radar_altitude_m=altitude,
                trip_ceiling_m=trip_ceiling_m,
            )
            temp_source = fmeta["temp_source"]

            codes, info = single.classify(
                features,
                shape=fmeta["shape"],
                snr_min=snr_min,
                vel_texture_max=vel_texture_max,
                min_run=min_run,
                despeckle_keep_dbz=despeckle_keep_dbz,
                incoherent_frac=incoherent_frac,
            )

            dims = dataset[resolved["z"]].dims
            shape = fmeta["shape"]
            new = {
                field_name: xr.DataArray(
                    codes.reshape(shape).astype("i2"),
                    dims=dims,
                    attrs={
                        "long_name": "Classification of dominant scatterer",
                        "standard_name": "gate_id",
                        "units": "",
                        "flag_values": np.array(sorted(single.CLASSES), dtype="i2"),
                        "flag_meanings": " ".join(labels),
                        "notes": notes,
                        "comment": (
                            "From uncorrected moments; scores "
                            "normalised by attainable weight budget; "
                            "0=unclassified is explicit. temperature: "
                            f"{temp_source}."
                        ),
                    },
                )
            }
            if add_margin:
                new[f"{field_name}_margin"] = xr.DataArray(
                    info["margin"].reshape(shape).astype("f4"),
                    dims=dims,
                    attrs={
                        "long_name": ("Winning minus runner-up class score"),
                        "units": "",
                    },
                )

            tree[key].dataset = dataset.assign(new)
            classified[str(key)] = sorted(new)
            n_noise += info["n_noise_floor"]
            n_speck += max(info["n_despeckled"], 0)
            for code, name in single.CLASSES.items():
                counts[name] += int((codes == code).sum())

        return {
            "resolved": resolved_names,
            "missing": missing_names,
            "temp_source": temp_source,
            "class_counts": counts,
            "classified_sweeps": classified,
            "skipped_sweeps": skipped,
            "n_noise_floor": n_noise,
            "n_despeckled": n_speck,
            "trip_ceiling_m": trip_ceiling_m,
            "trip_ceiling_source": ceiling_source,
        }


def register_accessors():
    """Register the ``radarpalette`` accessor on :class:`xarray.DataTree`.

    Idempotent, and a no-op if xarray is too old to expose the DataTree
    accessor hook, so importing the package never fails because of it.

    Returns
    -------
    bool
        Whether the accessor is registered.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import xarray as xr

        register = xr.register_datatree_accessor
    except (ImportError, AttributeError):  # pragma: no cover - old xarray
        return False

    register("radarpalette")(RadarPaletteDataTreeAccessor)
    _REGISTERED = True
    return True
