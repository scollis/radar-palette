"""The cone surface of a fixed-elevation sweep, and the lattice it is gridded onto.

A sweep at a fixed elevation angle does not sample a horizontal plane, nor even a
tilted one: on a curved earth the beam climbs faster than a straight line, so the
locus of samples is a **curved cone**. Every consequence in this module follows from
that one fact.

Why it matters for assembling a volume
--------------------------------------
Because the surface is curved, the height of a tilt varies from column to column, and
so does the vertical distance to the tilt above it. On C-SAPR the 0.75-to-1.20 degree
gap is about 0.16 km at 20 km arc and about 0.9 km at 100 km. Any vertical
interpolation therefore has to work per column against locally-evaluated heights; a
single nominal height per tilt is wrong everywhere except at one range.

Geometry source
---------------
Heights and slant ranges come from :func:`cone_range_height`, which is the exact
inverse of :func:`radar_palette.gridding.antenna_to_cartesian_43` at fixed elevation.
Py-ART's own inverse is not used here: it omits the curvature term, which is the very
thing that makes a tilt a cone rather than a plane.
"""

from __future__ import annotations

import contextlib
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from radar_palette.gridding.evaluator import SweepSpectralEvaluator
from radar_palette.gridding.fill import fill_sweep
from radar_palette.gridding.geometry import EARTH_RADIUS_EFFECTIVE_M

__all__ = [
    "ConeStack",
    "TiltReport",
    "beam_footprint_crossover",
    "build_cones",
    "cone_range_height",
    "dedup_sweeps",
    "target_lattice",
]


def cone_range_height(arc_length_m, elevation_deg):
    """Slant range and height on a fixed-elevation cone at a given arc length.

    Parameters
    ----------
    arc_length_m : array_like
        Great-circle arc length from the radar, in metres.
    elevation_deg : float
        Beam elevation angle of the sweep, in degrees.

    Returns
    -------
    slant_range_m, height_m : numpy.ndarray
        Slant range along the beam and height above the radar, both in metres.

    Notes
    -----
    Exact inverse of :func:`radar_palette.gridding.antenna_to_cartesian_43` at fixed
    elevation, on the 4/3 effective earth. Note that even at 0 degrees elevation the
    height is **not** zero: the earth curves away beneath the beam, so a horizontal
    ray rises as ``arc**2 / (2 * R_effective)`` to first order.
    """
    arc_length_m = np.asarray(arc_length_m, dtype=np.float64)
    earth_radius = EARTH_RADIUS_EFFECTIVE_M
    central_angle = arc_length_m / earth_radius
    tangent_elevation = np.tan(np.deg2rad(elevation_deg))

    radial_distance = earth_radius / (
        np.cos(central_angle) - tangent_elevation * np.sin(central_angle)
    )
    slant_range_m = (
        radial_distance * np.sin(central_angle) / np.cos(np.deg2rad(elevation_deg))
    )
    return slant_range_m, radial_distance - earth_radius


def beam_footprint_crossover(beamwidth_deg, spacing_m):
    """Arc length at which the cross-beam footprint equals the grid spacing.

    Inside this distance the grid is finer than the measurement it represents;
    outside it, adjacent grid cells are not independent samples. Useful for stating
    honestly where a gridded product stops carrying independent information.

    Parameters
    ----------
    beamwidth_deg : float
        Antenna beamwidth in degrees.
    spacing_m : float
        Cartesian grid spacing in metres.

    Returns
    -------
    float
        Arc length in metres at which ``arc * beamwidth == spacing``.
    """
    return float(spacing_m / np.deg2rad(beamwidth_deg))


def target_lattice(half_width_m, spacing_m, elevation_deg, range_first_m, range_max_m):
    """Build the common horizontal lattice for one tilt, with its cone geometry.

    Parameters
    ----------
    half_width_m : float
        Half-width of the square domain, in metres.
    spacing_m : float
        Grid spacing in metres.
    elevation_deg : float
        Elevation of the tilt whose cone geometry is wanted.
    range_first_m, range_max_m : float
        First and last measured slant range of the sweep. Cells outside this span
        are marked invalid.

    Returns
    -------
    dict
        ``x``, ``y`` (1-D axes); ``X``, ``Y``, ``arc_m``, ``slant_range_m``,
        ``height_m``, ``azimuth_deg`` (2-D); ``valid`` (2-D bool); plus
        ``spacing_m``, ``n`` and ``elevation_deg``.

    Notes
    -----
    Axis order is **row is y, column is x**, matching the ``(nz, ny, nx)`` convention
    of :func:`pyart.map.map_gates_to_grid`. Transposing this does not announce itself
    on a roughly isotropic field: it produces an error equal to the field's own
    standard deviation rather than a visible flip.

    The ``valid`` mask excludes the near-radar hole for a reason beyond bookkeeping.
    The range axis is mirror-extended before the FFT, and the mirror is *even* about
    the first gate, so it reflects the echo into the hole and fabricates a plausible
    near-radar return. Masking is what keeps that reflection out of the product.
    """
    cell_count = int(round(2.0 * half_width_m / spacing_m)) + 1
    axis_m = np.linspace(-half_width_m, half_width_m, cell_count)
    east_m, north_m = np.meshgrid(axis_m, axis_m, indexing="xy")

    arc_m = np.hypot(east_m, north_m)
    slant_range_m, height_m = cone_range_height(arc_m, elevation_deg)
    azimuth_deg = np.rad2deg(np.arctan2(east_m, north_m)) % 360.0
    valid = (slant_range_m >= range_first_m) & (slant_range_m <= range_max_m)

    return {
        "x": axis_m,
        "y": axis_m,
        "X": east_m,
        "Y": north_m,
        "arc_m": arc_m,
        "slant_range_m": slant_range_m,
        "height_m": height_m,
        "azimuth_deg": azimuth_deg,
        "valid": valid,
        "spacing_m": float(spacing_m),
        "n": cell_count,
        "elevation_deg": float(elevation_deg),
    }


def dedup_sweeps(geometries, range_attribute="range_max_valid_m"):
    """Collapse split cuts so each unique elevation contributes exactly once.

    A WSR-88D split cut pairs a long surveillance sweep with a short Doppler sweep at
    the same fixed angle. For reflectivity the long one is wanted, so the member whose
    field reaches furthest in range is kept and ties fall back to the lower sweep
    index.

    Stating the rule in terms of *measured* range rather than cut naming means it
    needs no knowledge of any particular radar's scan-strategy vocabulary.

    Parameters
    ----------
    geometries : list of ~radar_palette.gridding.census.SweepGeometry
        Census of every sweep, from :func:`radar_palette.gridding.census_radar`.
    range_attribute : str, optional
        Attribute compared when choosing between members of a split-cut group.

    Returns
    -------
    list of ~radar_palette.gridding.census.SweepGeometry
        One entry per unique elevation, sorted by ascending fixed angle. Each keeps
        its ``split_cut_size``, so a caller can see that sweeps were dropped.
    """
    by_group = {}
    for geometry in geometries:
        by_group.setdefault(geometry.split_cut_group, []).append(geometry)

    kept = []
    for group in sorted(by_group, key=lambda key: (key is None, key)):
        members = sorted(
            by_group[group],
            key=lambda geometry: (
                -float(getattr(geometry, range_attribute)),
                geometry.sweep,
            ),
        )
        kept.append(members[0])
    return sorted(kept, key=lambda geometry: geometry.fixed_angle)


@dataclass
class TiltReport:
    """Diagnostics for one gridded tilt.

    Fields whose meaning is not obvious from the name:

    ``overshoot_db``
        How far the gridded field strays beyond the range of the data it came from,
        as the larger of the two one-sided excursions. A between-sample metric: a
        check on ray positions alone is blind to an artefact that appears *between*
        rays, which is where band-limited ringing lives.
    ``valid_fraction``
        Fraction of lattice cells inside the sweep's measured range span.
    """

    tilt_index: int
    sweep: int
    fixed_angle: float
    sweep_class: str
    azimuth_path: str
    nrays: int
    masked_input_fraction: float
    valid_fraction: float
    height_min_m: float
    height_max_m: float
    data_min_dbz: float
    data_max_dbz: float
    grid_min_dbz: float
    grid_max_dbz: float
    overshoot_db: float
    split_cut_size: int


@dataclass
class ConeStack:
    """A volume's sweeps gridded onto one shared horizontal lattice.

    ``reflectivity``, ``height_m`` and ``valid`` are all ``(n_tilts, ny, nx)`` and
    index-aligned: cone ``k`` has values ``reflectivity[k]`` on the surface
    ``height_m[k]``, meaningful only where ``valid[k]``. Tilts are ordered by
    ascending elevation, which the vertical search relies on.
    """

    reflectivity: np.ndarray
    height_m: np.ndarray
    valid: np.ndarray
    fixed_angle: np.ndarray
    sweep: np.ndarray
    x: np.ndarray
    y: np.ndarray
    reports: list = field(default_factory=list)


# Bytes per lattice cell held live by one in-flight tilt. The evaluator's upsampled
# working arrays dominate, and there are several of them (float64 slant range,
# azimuth, height, X, Y, the gridded output, plus the evaluator's own internals),
# so this is measured-generous rather than derived: at 250 m over a 171 km domain
# one tilt peaks near 1 GB, which is ~1.4 kB per cell of the 684k-cell lattice.
TILT_BYTES_PER_CELL = 1400.0

# Leave this much of the machine's memory alone. Gridding is not the only thing
# running, and a swapping parallel run is slower than a serial one.
MEMORY_HEADROOM_FRAC = 0.5


# Beyond this the gridded field is not a plausible reflectivity, whatever the
# operator did to get there. Hail spikes and three-body scatter reach the high 70s;
# nothing meteorological approaches 100 dBZ, and the receiver noise floor bounds the
# low side well above -100. Used only to decide whether to WARN -- the values are
# left in place, because silently clipping would hide a settings problem behind a
# plausible-looking field.
class EngineConcurrencyWarning(UserWarning):
    """An engine is being run concurrently that does not behave well that way.

    Separate from :class:`UnphysicalGridWarning` because the consequence is
    different in kind: the numbers are right, only the runtime is unpredictable. A
    caller benchmarking engines wants to see this; one gridding a volume may not
    care.
    """


class UnphysicalGridWarning(UserWarning):
    """The gridded field left the range its physical quantity can take.

    Its own category rather than a bare ``UserWarning`` so a caller can act on it
    programmatically --- ``filterwarnings("error", category=UnphysicalGridWarning)``
    turns it into a hard failure for a production pipeline, which is the right
    posture there and the wrong one for interactive work.
    """


PHYSICAL_DBZ_LIMITS = (-100.0, 100.0)

# Overshoot beyond the input range that is worth flagging even when the result is
# still inside PHYSICAL_DBZ_LIMITS. A band-limited interpolant is EXPECTED to
# overshoot at a sharp edge -- that is Gibbs behaviour, not a bug -- so this sits
# well above the few dB such ringing produces (measured: 5.5 dB on this operator's
# documented example, 7.5 dB at n_cg=12 on a convective volume) and catches the
# regime where the solve has started to diverge instead.
OVERSHOOT_WARN_DB = 25.0


# Engines whose *timing* is not reproducible under concurrency. Note what this is
# and is not: finufft's results are exact -- bit-identical to `dense` on every grid
# measured -- and serially it is indistinguishable from `dense` (medians within 2%
# on eight sweep shapes, 1.00-1.03x spread over three repeats each). Under
# `n_jobs=4` it is bimodal: when it wins the race it is the fastest CPU
# configuration measured here (8.3 s against dense's 10.1 s on a 15-tilt volume),
# and when it loses, the same call takes 12-17x longer. Because it varies run to
# run at fixed workload, it is a race inside finufft's own global planner state
# rather than anything about the grid.
#
# Warned rather than serialised or refused: it may well be the fastest choice, the
# numbers are right either way, and quietly forcing n_jobs=1 for one engine would
# make an engine comparison unfair.
THREAD_UNSAFE_ENGINES = ("finufft",)


def _check_engine_concurrency(az_engine, workers):
    """Warn when an engine is being run concurrently that does not like it."""
    if workers > 1 and az_engine in THREAD_UNSAFE_ENGINES:
        warnings.warn(
            f"az_engine={az_engine!r} with n_jobs>1 gives exact results but "
            f"unreproducible timing: it holds global planner state, so the same "
            f"call has varied 12-17x run to run. It is often the fastest option "
            f"when it wins that race. Use n_jobs=1 to time it reproducibly (where "
            f"it matches 'dense'), or az_engine='dense' for predictable "
            f"parallel timing.",
            EngineConcurrencyWarning,
            stacklevel=4,
        )


def _check_physical(reports, field_units):
    """Warn when a gridded tilt leaves the range reflectivity can physically take.

    Called after gridding rather than before, because the failure this catches
    happens *between* the measured rays: every ray position can be fine and the
    interpolant between them still diverge. A check on inputs cannot see it, and
    the per-tilt ``overshoot_db`` already computed is exactly the quantity that
    can.

    Warns rather than raising or clipping, and that choice is deliberate. Raising
    would abort a volume for one bad tilt out of fifteen; clipping would hide a
    settings problem behind a field that looks plausible. A scientist who sees
    2471 dBZ knows something is wrong with the run --- one who sees a silently
    clipped 100 dBZ does not.

    Parameters
    ----------
    reports : list of TiltReport
        Per-tilt diagnostics, already carrying ``overshoot_db`` and the grid range.
    field_units : {'dbz', 'linear_z', 'other'} or None
        Only ``'dbz'`` (or ``None``, which the evaluator resolves to dBZ for
        reflectivity-like input) has physical limits worth asserting. Any other
        field passes through unchecked.
    """
    if field_units not in (None, "dbz"):
        return

    low, high = PHYSICAL_DBZ_LIMITS
    unphysical = [
        report
        for report in reports
        if report.grid_min_dbz < low or report.grid_max_dbz > high
    ]
    if unphysical:
        worst = max(unphysical, key=lambda r: max(abs(r.grid_min_dbz), r.grid_max_dbz))
        warnings.warn(
            f"{len(unphysical)} of {len(reports)} gridded tilts left the physical "
            f"range for reflectivity ({low:.0f} to {high:.0f} dBZ): worst is tilt "
            f"{worst.tilt_index} at {worst.fixed_angle:.1f} deg, spanning "
            f"{worst.grid_min_dbz:.0f} to {worst.grid_max_dbz:.0f} dBZ from input "
            f"data spanning {worst.data_min_dbz:.0f} to {worst.data_max_dbz:.0f}. "
            f"The azimuth solve has diverged rather than merely ringing. If you "
            f"raised n_cg, lower it -- past ~12 iterations the solve amplifies "
            f"out-of-band noise instead of converging. If you set az_ridge, it is "
            f"too small for real reflectivity; use 1e-2 or 'auto'. Values are left "
            f"unclipped so the problem stays visible.",
            UnphysicalGridWarning,
            stacklevel=4,
        )
        return

    ringing = [r for r in reports if r.overshoot_db > OVERSHOOT_WARN_DB]
    if ringing:
        worst = max(ringing, key=lambda r: r.overshoot_db)
        warnings.warn(
            f"{len(ringing)} of {len(reports)} gridded tilts overshoot their input "
            f"range by more than {OVERSHOOT_WARN_DB:.0f} dB: worst is tilt "
            f"{worst.tilt_index} at {worst.fixed_angle:.1f} deg, by "
            f"{worst.overshoot_db:.0f} dB. A band-limited interpolant overshoots at "
            f"a sharp edge by a few dB, which is expected; this is larger than that "
            f"and suggests the azimuth solve is under-regularised.",
            UnphysicalGridWarning,
            stacklevel=4,
        )


@contextlib.contextmanager
def _limited_threads(n_threads):
    """Cap the native BLAS/FFT thread pools for the duration of the block.

    Optional dependency by design: ``threadpoolctl`` is the only portable way to
    change these pools after import, but requiring it would make the serial default
    -- which needs no such cap -- depend on a package it never uses. Without it the
    parallel path still produces correct output, just over-subscribed, so this warns
    loudly rather than failing: a silent fallback here looks exactly like
    "parallelism does not help on this machine".
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        warnings.warn(
            "threadpoolctl is not installed, so the BLAS and FFT thread pools "
            "cannot be capped. Gridding tilts concurrently will over-subscribe the "
            "CPU (each worker starts its own pool sized to the whole machine) and "
            "may run slower than n_jobs=1. Install threadpoolctl, or set "
            "OMP_NUM_THREADS before importing numpy.",
            ResourceWarning,
            stacklevel=3,
        )
        yield
        return
    with threadpool_limits(limits=n_threads):
        yield


def resolve_tilt_workers(n_jobs, n_tilts, half_width_m, spacing_m):
    """Decide how many tilts to grid concurrently, and how many BLAS threads each gets.

    **Two resources bind here, and missing either one is worse than staying
    serial.**

    *Memory.* One in-flight tilt holds an upsampled lattice that reaches the order
    of a gigabyte on a wide domain at fine spacing, so a core-count pool would try
    to hold a volume's worth of those at once and swap.

    *Threads.* The per-tilt work is BLAS and FFT, and those libraries spawn their
    own pool sized to the machine. Nesting one inside the other multiplies rather
    than divides: measured here, ``n_jobs=-1`` on a 14-core machine produced a load
    average above 500 and had to be killed, because 14 workers each started ~14
    BLAS threads. So the worker count and the per-worker thread count must share
    one core budget, which is why this returns both.

    A caller that uses the worker count and ignores the thread count will reproduce
    that failure, so :func:`build_cones` applies the latter with ``threadpoolctl``
    inside each worker.

    Parameters
    ----------
    n_jobs : int or None
        Requested workers. ``None`` or ``1`` grids serially; ``-1`` requests one per
        core; a positive integer requests that many. An explicit request is honoured
        up to the memory cap.
    n_tilts : int
        Tilts to grid --- never worth more workers than this.
    half_width_m, spacing_m : float
        Lattice geometry, which sets the per-tilt memory estimate.

    Returns
    -------
    workers : int
        Tilts to grid concurrently, at least 1.
    blas_threads : int
        Threads each worker may give its BLAS/FFT pool, at least 1. ``workers *
        blas_threads`` never exceeds the core count.
    """
    cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 1:
        # Serial: leave the native pools alone, which is what every release before
        # n_jobs did and what a single-threaded caller expects.
        return 1, cores
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError(f"n_jobs must be -1, or a positive integer, got {n_jobs!r}")

    requested = cores if n_jobs == -1 else int(n_jobs)
    requested = min(requested, n_tilts, cores)

    cells = (2.0 * half_width_m / spacing_m + 1.0) ** 2
    per_tilt_bytes = cells * TILT_BYTES_PER_CELL
    budget = _available_bytes() * MEMORY_HEADROOM_FRAC
    affordable = max(int(budget // per_tilt_bytes), 1)

    if affordable < requested:
        warnings.warn(
            f"gridding {requested} tilts concurrently needs about "
            f"{requested * per_tilt_bytes / 1e9:.1f} GB of working memory for a "
            f"{2 * half_width_m / 1e3:.0f} km domain at {spacing_m:.0f} m; using "
            f"{affordable} worker(s) to stay inside the available "
            f"{budget / 1e9:.1f} GB. Coarsen the grid or pass a smaller n_jobs to "
            f"silence this.",
            ResourceWarning,
            stacklevel=3,
        )
    workers = max(min(requested, affordable), 1)
    # Split the cores between the two levels. Integer division deliberately floors:
    # over-committing threads is what produced the load-500 failure, and a worker
    # with one thread still makes progress.
    #
    # Measured on a 15-tilt volume over a 60 km domain at 1 km, 14 cores, with the
    # dense engine and the direct solver (build_cones only, seconds):
    #
    #   n_jobs=1   1 worker  x 14 threads   17.47   1.00x
    #   n_jobs=2   2 workers x  7 threads   10.70   1.63x
    #   n_jobs=4   4 workers x  3 threads    8.44   2.07x
    #   n_jobs=8   8 workers x  1 thread     5.65   3.09x
    #
    # Task parallelism beats BLAS threading here, and by enough to be worth saying:
    # the fastest setting gives each worker a SINGLE thread. The azimuth solve is a
    # few hundred rows, too small for a 14-thread GEMM to pay for its own
    # synchronisation, so spreading whole tilts across cores wins over splitting
    # each tilt's linear algebra. Speedup stays well short of linear because the
    # tilts share memory bandwidth and the vertical interpolation afterwards is
    # serial.
    return workers, max(cores // workers, 1)


def _available_bytes():
    """Free physical memory, falling back to a conservative guess."""
    try:  # Linux, and anything else exposing the POSIX name
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS: vm_stat is the only portable-enough source without psutil
        import subprocess

        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, timeout=5
        ).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        free = sum(
            int(match.group(1))
            for name in ("Pages free", "Pages inactive", "Pages speculative")
            for match in [re.search(rf"{name}:\s+(\d+)", out)]
            if match
        )
        return free * page
    except Exception:
        return 2 * 1024**3


def build_cones(
    radar,
    geometries,
    half_width_m,
    spacing_m,
    field_name="reflectivity",
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
    """Spectrally grid every sweep onto one shared horizontal lattice.

    Parameters
    ----------
    radar : pyart.core.Radar
        Volume to grid.
    geometries : list of ~radar_palette.gridding.census.SweepGeometry
        Sweeps to grid, normally the output of :func:`dedup_sweeps`.
    half_width_m, spacing_m : float
        Domain half-width and grid spacing, in metres.
    field_name : str, optional
        Field to grid. Must be in dBZ; see
        :mod:`radar_palette.gridding.reflectivity`.
    az_engine, az_solver, az_ridge : optional
        Azimuth NUFFT backend, the solver on top of it, and that solver's
        regularisation, forwarded to each
        :class:`~radar_palette.gridding.evaluator.SweepSpectralEvaluator`. Left
        unset by default so the evaluator's own defaults remain the single source
        of truth. These are the consequential arguments for spectral gridding
        cost: one evaluator is built per sweep and that build dominates, so the
        total is nearly flat in output resolution and is set here.
    band_frac, n_cg, cg_tol : optional
        Passed to :class:`~radar_palette.gridding.SweepSpectralEvaluator`. ``n_cg``
        defaults to the evaluator's own default.
    fill_method : str, optional
        Gap-fill strategy applied before evaluation; see
        :func:`radar_palette.gridding.fill_sweep`.
    up_az, up_r : int, optional
        Upsampling factors for the evaluator's fast path.

    n_jobs : int, optional
        Tilts to grid concurrently. Defaults to ``None`` (serial), which keeps the
        memory profile of previous releases. ``-1`` requests one worker per core.
        Tilts are independent so this is close to linear in wall time, but the cap
        that matters is memory rather than cores --- see
        :func:`resolve_tilt_workers`, which reduces an over-ambitious request and
        says so rather than swapping.

    Returns
    -------
    ConeStack

    Notes
    -----
    Each tilt's evaluator is released as soon as its cone is built. The extended,
    upsampled lattice for one wide sweep can reach the order of a gigabyte, which is
    why ``n_jobs`` is bounded by available memory and why the default stays serial:
    a volume's worth of evaluators cannot coexist.
    """
    evaluator_kwargs = {"band_frac": band_frac}
    if n_cg is not None:
        evaluator_kwargs["n_cg"] = n_cg
    if cg_tol is not None:
        evaluator_kwargs["cg_tol"] = cg_tol
    # Left unset rather than defaulted here, so the evaluator's own defaults stay
    # the single source of truth for what an unconfigured call does.
    if az_engine is not None:
        evaluator_kwargs["az_engine"] = az_engine
    if az_solver is not None:
        evaluator_kwargs["az_solver"] = az_solver
    if az_ridge is not None:
        evaluator_kwargs["az_ridge"] = az_ridge

    def grid_one_tilt(indexed_geometry):
        """Grid one tilt. Reads ``radar``; writes nothing shared."""
        tilt_index, geometry = indexed_geometry
        start, end = radar.get_start_end(geometry.sweep)
        azimuths = np.asarray(radar.azimuth["data"][start : end + 1], dtype=np.float64)
        observed = radar.fields[field_name]["data"][start : end + 1]
        filled, input_mask = fill_sweep(observed, method=fill_method, return_mask=True)

        lattice = target_lattice(
            half_width_m,
            spacing_m,
            geometry.fixed_angle,
            geometry.range_first_m,
            geometry.range_max_valid_m,
        )
        evaluator = SweepSpectralEvaluator(
            filled, geometry, azimuths, **evaluator_kwargs
        )

        in_range = lattice["valid"]
        gridded = np.full(lattice["X"].shape, np.nan)
        gridded[in_range] = evaluator.evaluate_fast(
            lattice["slant_range_m"][in_range],
            lattice["azimuth_deg"][in_range],
            up_az=up_az,
            up_r=up_r,
        )

        measured = filled[~input_mask] if (~input_mask).any() else np.array([np.nan])
        data_min, data_max = float(np.nanmin(measured)), float(np.nanmax(measured))
        grid_min, grid_max = float(np.nanmin(gridded)), float(np.nanmax(gridded))
        report = TiltReport(
            tilt_index=tilt_index,
            sweep=int(geometry.sweep),
            fixed_angle=float(geometry.fixed_angle),
            sweep_class=str(geometry.sweep_class),
            azimuth_path=evaluator.report.az_path,
            nrays=int(geometry.nrays),
            masked_input_fraction=float(input_mask.mean()),
            valid_fraction=float(in_range.mean()),
            height_min_m=float(np.nanmin(lattice["height_m"][in_range])),
            height_max_m=float(np.nanmax(lattice["height_m"][in_range])),
            data_min_dbz=data_min,
            data_max_dbz=data_max,
            grid_min_dbz=grid_min,
            grid_max_dbz=grid_max,
            overshoot_db=float(max(grid_max - data_max, data_min - grid_min)),
            split_cut_size=int(geometry.split_cut_size),
        )
        entry = (
            np.where(in_range, gridded, np.nan).astype(np.float32),
            np.where(in_range, lattice["height_m"], np.nan).astype(np.float32),
            in_range.copy(),
            report,
            (lattice["x"], lattice["y"]),
        )
        del evaluator, gridded, lattice, filled, input_mask
        return entry

    indexed = list(enumerate(geometries))
    workers, blas_threads = resolve_tilt_workers(
        n_jobs, len(indexed), half_width_m, spacing_m
    )
    _check_engine_concurrency(az_engine, workers)
    if workers == 1:
        results = [grid_one_tilt(item) for item in indexed]
    else:
        # Threads, not processes: the per-tilt cost is NumPy/SciPy linear algebra and
        # FFTs, which release the GIL, and a process pool would have to pickle the
        # radar object to every worker -- hundreds of megabytes for a research volume.
        #
        # The thread limit is not optional. Those same libraries each start a pool
        # sized to the whole machine, so without this the two levels multiply: 14
        # workers x 14 BLAS threads drove the load average past 500 on a 14-core
        # machine and had to be killed. threadpoolctl is the only portable way to
        # set this after import, since the environment variables the runtimes read
        # are consulted when they load.
        with _limited_threads(blas_threads):
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(grid_one_tilt, indexed))

    _check_physical([entry[3] for entry in results], field_units)

    cones = [entry[0] for entry in results]
    heights = [entry[1] for entry in results]
    masks = [entry[2] for entry in results]
    reports = [entry[3] for entry in results]
    axes = results[0][4]

    return ConeStack(
        reflectivity=np.stack(cones),
        height_m=np.stack(heights),
        valid=np.stack(masks),
        fixed_angle=np.array([g.fixed_angle for g in geometries], dtype=np.float64),
        sweep=np.array([g.sweep for g in geometries], dtype=int),
        x=axes[0],
        y=axes[1],
        reports=reports,
    )
