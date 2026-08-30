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

        tree = self.datatree
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
