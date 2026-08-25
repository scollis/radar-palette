"""Flavour-aware entry point for gridding radar volumes onto a Cartesian lattice.

Accepts volumes in either supported object family and returns Cartesian output in the
family that matches: a Py-ART ``Radar`` grids to a :class:`pyart.core.Grid`, an xradar
``DataTree`` grids to an :class:`xarray.Dataset`. Either default can be overridden per
call.

Two operators, and why the old one is still the default
-------------------------------------------------------
``method="pyart"`` (the default) grids with :func:`pyart.map.grid_from_radars`, which
distance-weights neighbouring gates into each cell.

``method="spectral"`` uses this package's own operator: each sweep is evaluated as a
band-limited interpolant on its own cone
(:class:`~radar_palette.gridding.SweepSpectralEvaluator`), and the resulting cone stack
is interpolated vertically per column
(:func:`~radar_palette.gridding.interp_column_stack`).

The spectral path is **opt-in**, and that is a deliberate choice rather than caution
about new code. The two operators answer different questions:

*Cressman weighting* smooths. It cannot resolve a gradient sharper than its radius of
influence, but it also cannot overshoot: a weighted average of gates cannot leave the
range of its inputs.

It does not, however, degrade gracefully where sampling is poor. The radius of influence
is a fixed length while the vertical separation between adjacent tilts grows with range,
so beyond the range where the separation exceeds the radius no gate is within reach
and the cell is empty --- an interior hole, with valid data both above and below it
in the same column. On a C-SAPR2 volume (15 tilts, 1.5 to 42 degrees) gridded at
500 m over a 60 km box, ``roi_func="dist_beam"`` fills 70.7% of cells and leaves
570 interior holes in a single vertical section; the widest tilt pair is separated
by eleven times the radius at far range. Py-ART's own defaults produce the same
result, so this is a property of the operator rather than a mis-set argument.

Widening the radius closes the holes and costs peak intensity, monotonically. Measured
on the same volume against an observed maximum of 68.0 dBZ:

.. list-table::
   :header-rows: 1

   * - ``roi_func``
     - filled
     - interior holes
     - peak
   * - ``dist_beam`` (Py-ART default)
     - 70.7%
     - 570
     - 63.1 dBZ
   * - ``dist``, ``z_factor=0.1``
     - 95.1%
     - 6
     - 57.0 dBZ
   * - ``dist``, ``z_factor=0.2``
     - 96.3%
     - 0
     - 53.1 dBZ
   * - ``constant_roi=4000.0``
     - 98.3%
     - 0
     - 43.0 dBZ

Closing the gaps entirely costs 15 dB of peak; the spectral path reaches zero interior
holes at a cost of 4.7 dB, because vertical assembly interpolates between whichever
cones bracket the target height rather than reaching a fixed distance. Callers who
need filled columns from the default path should pass an explicit ``roi_func`` and
choose where on that trade-off to sit.

*The spectral operator* is exact where the sampling supports it --- no window, no taper
--- but a band-limited interpolant rings at a discontinuity. Measured through this
entry point on a hard azimuthal echo edge (a 10 to 50 dBZ wedge), the gridded field
reaches 55.5 and 4.7 dBZ: an overshoot of +5.5 dB above the data maximum and -5.3 dB
below its minimum. The per-sweep evaluator shows a larger excursion still on a range
step (+19.8 / -10.6 dB, see
:class:`~radar_palette.gridding.SweepSpectralEvaluator`); the smaller figure here
reflects the vertical interpolation and horizontal resampling averaging some of it
away, not a gentler operator.

Neither is strictly better, so switching the default would silently change every
existing caller's numbers in exchange for a trade-off they did not ask for. Callers who
want sharp gradients resolved ask for them.

Honest coverage
---------------
The spectral path emits a ``coverage_flag`` field alongside the data. A cell is NaN
wherever the volume did not observe it, and the flag distinguishes the reasons ---
below the lowest tilt, above the highest, a hole in the tilt run, too few tilts. This
is not decoration: a NaN with no explanation leaves a user unable to tell a data gap
from a geometric coverage limit. See :class:`~radar_palette.gridding.VerticalFlag`.
"""

from __future__ import annotations

import numpy as np

from radar_palette.gridding.census import census_radar
from radar_palette.gridding.cones import build_cones, dedup_sweeps
from radar_palette.gridding.vertical import (
    DEFAULT_VERTICAL_SCHEME,
    VerticalFlag,
    interp_column_stack,
)
from radar_palette.io import (
    GridFlavor,
    RadarFlavor,
    detect_radar_flavor,
    resolve_output_flavor,
    to_grid_flavor,
    to_pyart_radar,
)

__all__ = ["DEFAULT_GRIDDING_METHOD", "GRIDDING_METHODS", "grid_volume"]

#: Available gridding backings.
GRIDDING_METHODS = ("pyart", "spectral")

#: Default backing. ``"pyart"`` preserves the behaviour of every caller written before
#: the spectral operator existed; see the module docstring for why that matters more
#: than making the newer operator prominent.
DEFAULT_GRIDDING_METHOD = "pyart"

_RADAR_TO_GRID_FLAVOR = {
    RadarFlavor.PYART: GridFlavor.PYART,
    RadarFlavor.XRADAR: GridFlavor.XARRAY,
}


def _grid_with_pyart(radar_surfaces, grid_shape, grid_limits, **gridding_kwargs):
    """Grid by distance weighting, exactly as before this option existed."""
    import pyart

    return pyart.map.grid_from_radars(
        radar_surfaces,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        **gridding_kwargs,
    )


def _grid_spectrally(
    radar,
    grid_shape,
    grid_limits,
    field_name="reflectivity",
    scheme=DEFAULT_VERTICAL_SCHEME,
    band_frac=1.0,
    n_cg=None,
    cg_tol=None,
    az_engine=None,
    az_solver=None,
    az_ridge=None,
    fill_method="edge",
    up_az=4,
    up_r=4,
    n_jobs=None,
    field_units=None,
):
    """Grid one volume by per-sweep spectral evaluation plus vertical interpolation."""
    import pyart

    geometries = dedup_sweeps(census_radar(radar))
    if len(geometries) < 2:
        raise ValueError(
            "spectral gridding needs at least two distinct elevations: vertical "
            f"interpolation requires two cones to bracket a target, and this volume "
            f"has {len(geometries)} after de-duplicating split cuts. Use "
            'method="pyart" for a single-tilt volume.'
        )

    (z_min, z_max), (y_min, y_max), (x_min, x_max) = grid_limits
    n_levels, n_north, n_east = grid_shape

    # build_cones works on one square, uniformly spaced lattice, so the horizontal
    # request is honoured by taking the enclosing square and evaluating the requested
    # axes from it. Asserting a square request instead would be an artificial limit.
    half_width_m = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    spacing_m = min(
        (x_max - x_min) / max(n_east - 1, 1), (y_max - y_min) / max(n_north - 1, 1)
    )

    cones = build_cones(
        radar,
        geometries,
        half_width_m=half_width_m,
        spacing_m=spacing_m,
        field_name=field_name,
        band_frac=band_frac,
        n_cg=n_cg,
        cg_tol=cg_tol,
        az_engine=az_engine,
        az_solver=az_solver,
        az_ridge=az_ridge,
        fill_method=fill_method,
        up_az=up_az,
        up_r=up_r,
        n_jobs=n_jobs,
        field_units=field_units,
    )

    east_axis_m = np.linspace(x_min, x_max, n_east)
    north_axis_m = np.linspace(y_min, y_max, n_north)
    height_axis_m = np.linspace(z_min, z_max, n_levels)
    east_m, north_m = np.meshgrid(east_axis_m, north_axis_m, indexing="xy")

    # The cone stack lives on its own lattice; resample it onto the requested one
    # before interpolating vertically, so the output honours the caller's axes.
    resampled = _resample_cones(cones, east_m, north_m)
    profile = interp_column_stack(
        resampled, east_m, north_m, height_axis_m, schemes=(scheme,)
    )

    values = np.ma.masked_invalid(profile.fields[scheme].astype("float32"))
    flags = np.ma.masked_where(profile.flag == 255, profile.flag.astype("uint8"))

    template = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(1, 2, 2),
        grid_limits=((z_min, z_min + 1.0), (y_min, y_max), (x_min, x_max)),
        fields=[field_name],
    )
    metadata = dict(template.metadata)
    metadata["history"] = "; ".join(
        part
        for part in (
            metadata.get("history", ""),
            f"radar_palette.gridding: spectral gridding, vertical scheme {scheme}",
        )
        if part
    )

    field_template = dict(radar.fields[field_name])
    field_template.pop("data", None)
    return pyart.core.Grid(
        time=dict(template.time),
        fields={
            field_name: {**field_template, "data": values},
            "coverage_flag": {
                "data": flags,
                "long_name": "vertical coverage flag",
                "units": "1",
                "comment": (
                    "radar_palette.gridding.VerticalFlag: 0 interpolated, "
                    "1 below lowest tilt, 2 above highest tilt, 3 fewer than two "
                    "tilts, 4 hole in the tilt run"
                ),
                "valid_min": int(min(VerticalFlag)),
                "valid_max": int(max(VerticalFlag)),
            },
        },
        metadata=metadata,
        origin_latitude=dict(template.origin_latitude),
        origin_longitude=dict(template.origin_longitude),
        origin_altitude=dict(template.origin_altitude),
        x={**dict(template.x), "data": east_axis_m},
        y={**dict(template.y), "data": north_axis_m},
        z={**dict(template.z), "data": height_axis_m},
        projection=dict(template.projection),
        radar_latitude=dict(template.radar_latitude),
        radar_longitude=dict(template.radar_longitude),
        radar_altitude=dict(template.radar_altitude),
        radar_time=dict(template.radar_time),
        radar_name=dict(template.radar_name),
    )


def _resample_cones(cones, east_m, north_m):
    """Resample a cone stack onto arbitrary horizontal coordinates.

    Bilinear in the horizontal, per cone, which is appropriate because the spectral
    evaluation has already done the interpolation that matters --- this only moves an
    already-continuous surface onto the caller's requested axes. Invalid cells stay
    invalid: any cone cell touching an invalid neighbour is dropped rather than
    partially filled, so coverage is never overstated.
    """
    from scipy.interpolate import RegularGridInterpolator

    from radar_palette.gridding.cones import ConeStack

    axes = (cones.y, cones.x)
    query = np.column_stack([north_m.ravel(), east_m.ravel()])
    shape = (cones.reflectivity.shape[0],) + east_m.shape

    values = np.full(shape, np.nan, dtype="float32")
    heights = np.full(shape, np.nan, dtype="float32")
    valid = np.zeros(shape, dtype=bool)
    for k in range(shape[0]):
        for source, target in (
            (cones.reflectivity[k], values),
            (cones.height_m[k], heights),
        ):
            interpolator = RegularGridInterpolator(
                axes, source, bounds_error=False, fill_value=np.nan
            )
            target[k] = interpolator(query).reshape(east_m.shape)
        coverage = RegularGridInterpolator(
            axes,
            cones.valid[k].astype("float64"),
            bounds_error=False,
            fill_value=0.0,
        )
        # 1.0 only where every contributing cell was valid.
        valid[k] = coverage(query).reshape(east_m.shape) > 1.0 - 1e-9

    values[~valid] = np.nan
    heights[~valid] = np.nan
    return ConeStack(
        reflectivity=values,
        height_m=heights,
        valid=valid,
        fixed_angle=cones.fixed_angle,
        sweep=cones.sweep,
        x=east_m[0],
        y=north_m[:, 0],
        reports=cones.reports,
    )


def grid_volume(
    volumes,
    grid_shape,
    grid_limits,
    output_flavor=None,
    method=DEFAULT_GRIDDING_METHOD,
    **gridding_kwargs,
):
    """Grid one or more radar volumes onto a Cartesian lattice.

    Parameters
    ----------
    volumes : pyart.core.Radar or xarray.DataTree or sequence of these
        Volume, or volumes, to grid. A sequence may mix object families; the first
        entry determines the default output family. ``method="spectral"`` accepts a
        single volume only.
    grid_shape : tuple of int
        Lattice shape as ``(nz, ny, nx)``.
    grid_limits : tuple of tuple of float
        Lattice extent in metres as ``((z_min, z_max), (y_min, y_max),
        (x_min, x_max))``.
    output_flavor : {'pyart', 'xarray'} or GridFlavor, optional
        Object family of the returned grid. Defaults to the family matching the
        input. ``'xradar'`` is accepted as an alias for ``'xarray'``.
    method : {'pyart', 'spectral'}, optional
        Gridding backing; defaults to :data:`DEFAULT_GRIDDING_METHOD`. See the module
        docstring for the trade-off, which is a real one in both directions.
    **gridding_kwargs
        Passed to the chosen backing. For ``"spectral"`` these are ``field_name``,
        ``scheme``, ``band_frac``, ``n_cg``, ``cg_tol``, ``az_engine``, ``az_solver``,
        ``az_ridge``, ``fill_method``, ``up_az``, ``up_r``, ``n_jobs`` and
        ``field_units``. The three ``az_*``
        arguments choose the azimuth NUFFT backend and the solve on top of it
        (:mod:`radar_palette.gridding.nufft_engines`); they matter here because
        the spectral cost is dominated by building one evaluator per sweep, so it
        is flat in output resolution and set almost entirely by that choice.
        ``n_jobs`` grids that many tilts concurrently; tilts are independent, so
        wall time falls close to linearly until memory binds.
        For ``"pyart"`` these reach :func:`pyart.map.grid_from_radars` unchanged, and
        ``roi_func`` is the consequential one: the default beam-width radius leaves
        interior holes between widely separated tilts. See the module docstring for the
        measured trade-off between coverage and peak intensity.

    Returns
    -------
    pyart.core.Grid or xarray.Dataset
        Gridded output in the requested family. The spectral path adds a
        ``coverage_flag`` field; see the module docstring.

    Raises
    ------
    ValueError
        If ``volumes`` is empty, ``output_flavor`` or ``method`` is unknown, or the
        spectral path is asked for a volume with fewer than two distinct elevations.
    NotImplementedError
        If the spectral path is given more than one volume.
    """
    if method not in GRIDDING_METHODS:
        raise ValueError(
            f"unknown gridding method {method!r}; choose from {list(GRIDDING_METHODS)}"
        )

    if not isinstance(volumes, (list, tuple)):
        volumes = (volumes,)
    if len(volumes) == 0:
        raise ValueError("grid_volume requires at least one radar volume")

    input_radar_flavor = detect_radar_flavor(volumes[0])
    output_flavor = resolve_output_flavor(
        output_flavor, _RADAR_TO_GRID_FLAVOR[input_radar_flavor]
    )
    radar_surfaces = tuple(to_pyart_radar(volume)[0] for volume in volumes)

    if method == "pyart":
        grid = _grid_with_pyart(
            radar_surfaces, grid_shape, grid_limits, **gridding_kwargs
        )
    else:
        if len(radar_surfaces) > 1:
            raise NotImplementedError(
                "spectral gridding accepts a single volume; merging several volumes "
                "spectrally would need a cross-radar combination rule this operator "
                'does not define. Use method="pyart" to grid multiple volumes.'
            )
        grid = _grid_spectrally(
            radar_surfaces[0], grid_shape, grid_limits, **gridding_kwargs
        )

    return to_grid_flavor(grid, output_flavor)
