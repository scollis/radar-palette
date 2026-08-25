"""Spectral (FFT-based) gridding of radar sweeps onto Cartesian lattices.

Scope
-----
A radar sweep is sampled on a polar (azimuth, range) lattice that is very
nearly uniform and, in azimuth, exactly periodic.  That structure admits a
band-limited resampling operator rather than a local weighted average:

1. **Geometry census** classifies a sweep as uniform-and-periodic,
   non-uniform, or degenerate, and reports the azimuth/range sampling
   statistics the choice of operator depends on.
2. **Horizontal operator** resamples each tilt with an exact Fourier series
   where the census permits it, and a non-uniform FFT (Kaiser-Bessel
   gridding kernel) otherwise. The non-uniform transforms and the solve on
   top of them are each pluggable --- see
   :mod:`radar_palette.gridding.nufft_engines` and the evaluator's
   ``az_engine`` / ``az_solver`` arguments. The defaults need no optional
   dependency and reproduce :mod:`radar_palette.gridding.nufft` to round-off.
3. **Vertical operator** combines tilts in elevation.  This is *not*
   spectral: tilt spacing is coarse, non-uniform and non-periodic, so a
   spectral operator in elevation is not defensible.

All three are now wired behind :func:`grid_volume` as ``method="spectral"``.

Choosing a method
-----------------
``method="pyart"`` remains the **default**, and deliberately so: the two operators
are different, not ranked. Distance weighting smooths and cannot overshoot; the
spectral operator is exact where the sampling supports it but rings at a
discontinuity --- measured +5.5 / -5.3 dB beyond the data on a hard azimuthal echo
edge. Switching the default would change every existing caller's numbers in exchange
for a trade-off they did not ask for, so sharpness is opt-in.

The spectral path also emits a ``coverage_flag`` field: a radar volume does not
observe a box, and the flag distinguishes *why* a cell is empty (below the lowest
tilt, above the highest, a hole in the tilt run, too few tilts) rather than leaving
an unexplained NaN. Accuracy is limited by tilt spacing far more than by
interpolation scheme --- see :func:`interp_column_stack` for the measurements.

Conventions
-----------
Spectral interpolation of reflectivity is performed in dBZ.  Interpolating
band-limited fields in linear Z drives large negative excursions (Gibbs
ringing) on real sweeps and is not supported.

Object flavours
---------------
:func:`grid_volume` accepts a :class:`pyart.core.Radar` or an
:class:`xarray.DataTree`, and returns Cartesian output in the matching family:
Py-ART input gives a :class:`pyart.core.Grid`, xradar input gives an
:class:`xarray.Dataset`. Pass ``output_flavor`` to override. See
:mod:`radar_palette.io`.

Status
------
The flavour-aware entry point :func:`grid_volume` is implemented, currently
backed by :func:`pyart.map.grid_from_radars`. The spectral operator described
above lands in a follow-up pull request and will replace that backing without
changing the public signature.
"""

from __future__ import annotations

from radar_palette.gridding.census import (
    AZ_DEV_TOL_FRAC,
    DR_DEV_TOL_FRAC,
    SPLIT_CUT_TOL_DEG,
    SweepClass,
    SweepGeometry,
    census_radar,
    census_sweep,
)
from radar_palette.gridding.cones import (
    PHYSICAL_DBZ_LIMITS,
    ConeStack,
    EngineConcurrencyWarning,
    TiltReport,
    UnphysicalGridWarning,
    beam_footprint_crossover,
    build_cones,
    cone_range_height,
    dedup_sweeps,
    resolve_tilt_workers,
    target_lattice,
)
from radar_palette.gridding.evaluator import (
    DEFAULT_CG_ITERATIONS,
    EvalReport,
    SweepSpectralEvaluator,
)
from radar_palette.gridding.fill import fill_sweep
from radar_palette.gridding.geometry import (
    EARTH_RADIUS_EFFECTIVE_M,
    antenna_to_cartesian_43,
    cartesian_to_antenna_43,
)
from radar_palette.gridding.gridder import (
    DEFAULT_GRIDDING_METHOD,
    GRIDDING_METHODS,
    grid_volume,
)
from radar_palette.gridding.nufft_engines import (
    DEFAULT_ENGINE,
    DEFAULT_RIDGE,
    DEFAULT_SOLVER,
    ENGINES,
    SOLVERS,
    EngineUnavailableError,
    available_engines,
    make_operator,
)
from radar_palette.gridding.reflectivity import (
    DBZ_PHYSICAL_CEILING,
    LinearReflectivityError,
    looks_like_linear_reflectivity,
    to_dbz,
    to_linear,
)
from radar_palette.gridding.vertical import (
    DEFAULT_VERTICAL_SCHEME,
    VERTICAL_SCHEMES,
    VerticalFlag,
    VerticalProfile,
    intercone_gap,
    interp_column_stack,
    target_elevation,
)

__all__ = [
    "GRIDDING_METHODS",
    "DEFAULT_GRIDDING_METHOD",
    "DEFAULT_VERTICAL_SCHEME",
    "target_lattice",
    "target_elevation",
    "interp_column_stack",
    "intercone_gap",
    "dedup_sweeps",
    "cone_range_height",
    "build_cones",
    "beam_footprint_crossover",
    "VerticalProfile",
    "VerticalFlag",
    "VERTICAL_SCHEMES",
    "TiltReport",
    "ConeStack",
    "AZ_DEV_TOL_FRAC",
    "DBZ_PHYSICAL_CEILING",
    "DEFAULT_CG_ITERATIONS",
    "DEFAULT_ENGINE",
    "DEFAULT_RIDGE",
    "DEFAULT_SOLVER",
    "DR_DEV_TOL_FRAC",
    "ENGINES",
    "SOLVERS",
    "EngineUnavailableError",
    "available_engines",
    "make_operator",
    "EngineConcurrencyWarning",
    "PHYSICAL_DBZ_LIMITS",
    "UnphysicalGridWarning",
    "resolve_tilt_workers",
    "EARTH_RADIUS_EFFECTIVE_M",
    "EvalReport",
    "LinearReflectivityError",
    "SPLIT_CUT_TOL_DEG",
    "SweepClass",
    "SweepGeometry",
    "SweepSpectralEvaluator",
    "antenna_to_cartesian_43",
    "cartesian_to_antenna_43",
    "census_radar",
    "census_sweep",
    "fill_sweep",
    "grid_volume",
    "looks_like_linear_reflectivity",
    "to_dbz",
    "to_linear",
]
