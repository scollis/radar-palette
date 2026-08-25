"""Measurement and classification of a volume's sampling geometry.

Spectral gridding is only valid where the sampling supports it, so the first thing
this package does with a volume is *measure* it rather than assume. The census
reduces each sweep to a :class:`SweepGeometry` record and assigns a
:class:`SweepClass`, which tells the gridder which path a sweep can take:

``EXACT_UNIFORM_PERIODIC``
    Full 360-degree coverage on a lattice that closes on itself, within tolerance.
    A plain FFT is valid on the azimuth axis.
``UNIFORM_PARTIAL_SECTOR``
    Uniform spacing but incomplete coverage; periodicity cannot be assumed.
``NON_UNIFORM``
    Everything else --- non-monotonic azimuth, non-uniform gates, multiple
    revolutions, or jitter beyond tolerance. Needs the non-uniform path.

The census is also useful on its own: it answers "is this volume spectrally
griddable, and if not why" without gridding anything.

On the tolerances
-----------------
``AZ_DEV_TOL_FRAC`` is expressed as a **fraction of the median azimuth spacing**
rather than an absolute angle, and that choice is derived rather than tuned. If the
true sample positions are displaced by ``eps = f * dphi`` from the assumed lattice,
treating them as exact turns a Fourier mode into an amplitude and phase error of
order ``m * eps``. With ``n`` rays, ``dphi = 2*pi/n`` and the Nyquist wavenumber is
``m = n/2``, so::

    m * eps = (n/2) * f * (2*pi/n) = pi * f

The worst-case relative error at Nyquist is therefore ``~pi*f``, **independent of
the ray count** --- which is exactly why a fractional tolerance is the right thing to
specify. ``f = 0.01`` buys roughly 3% worst-case error for energy sitting at
Nyquist, and ~0.8% at quarter-Nyquist where real weather echo lives.

Split cuts
----------
NEXRAD volume coverage patterns repeat the low tilts, appearing twice with
different range extent and Nyquist velocity. Sweeps whose fixed angles agree to
within ``SPLIT_CUT_TOL_DEG`` are tagged as one group, because de-duplicating them is
mandatory before vertical interpolation --- otherwise a tilt is interpolated against
itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np

from radar_palette.io import to_pyart_radar

__all__ = [
    "AZ_DEV_TOL_FRAC",
    "DR_DEV_TOL_FRAC",
    "SPLIT_CUT_TOL_DEG",
    "SweepClass",
    "SweepGeometry",
    "census_radar",
    "census_sweep",
]

# Maximum azimuth deviation from a best-fit uniform lattice, as a fraction of the
# median azimuth spacing. See the module docstring for the derivation --- the
# worst-case Nyquist error is ~pi*f, independent of ray count.
AZ_DEV_TOL_FRAC = 0.01

# Gate-spacing uniformity tolerance, as a fraction of the median gate spacing.
DR_DEV_TOL_FRAC = 1.0e-3

# Two sweeps whose fixed angles agree to better than this are one split-cut group.
SPLIT_CUT_TOL_DEG = 0.05


class SweepClass(StrEnum):
    """Sampling classification of a sweep, determining which gridding path applies.

    A :class:`enum.StrEnum` so that :meth:`SweepGeometry.as_row` serialises directly
    to CSV without enum handling at the call site.
    """

    EXACT_UNIFORM_PERIODIC = "EXACT_UNIFORM_PERIODIC"
    UNIFORM_PARTIAL_SECTOR = "UNIFORM_PARTIAL_SECTOR"
    NON_UNIFORM = "NON_UNIFORM"


@dataclass
class SweepGeometry:
    """Measured geometry of one sweep. Angles in degrees, ranges in metres.

    Field meanings that are not obvious from the name:

    ``ngates`` vs ``ngates_valid``
        Gates on the range axis, versus gates that actually carry data in this
        sweep. These differ for NEXRAD split cuts, which share one range axis while
        the Doppler cuts stop much earlier.
    ``az_max_dev_frac``
        Largest deviation from the best-fit lattice as a fraction of the median
        spacing. This, not the absolute deviation, is what the classifier tests ---
        see the module docstring for why the tolerance is fractional.
    ``az_lattice_closure_deg``
        For a full circle, how far the lattice fails to close; NaN for a sector,
        which is not expected to close.
    ``az_coverage_deg``
        Angular coverage counting one spacing of arc per ray, so a full circle
        measures 360 rather than 360 minus one spacing.
    ``n_wrap_seams``
        Crossings of the 0/360 branch cut. Bookkeeping, not a geometry defect: a
        sweep starting at 350 degrees has one seam and is still a valid lattice.
    ``el_max_dev_deg``, ``el_std_deg``
        Elevation jitter within the sweep. Reported but deliberately not used to
        classify --- wobble does not invalidate the azimuth transform.
    ``split_cut_group``, ``split_cut_size``
        Group index shared by sweeps at the same fixed angle, and the group size.
        :func:`census_sweep` cannot know these on its own, so they are ``-1`` and
        ``1`` until :func:`census_radar` fills them in.
    ``class_reason``
        Human-readable justification for ``sweep_class``, including the measured
        value that decided it.
    """

    sweep: int
    fixed_angle: float

    ngates: int
    ngates_valid: int
    range_first_m: float
    range_spacing_m: float
    range_spacing_uniform: bool
    range_spacing_max_dev_m: float
    range_max_m: float
    range_max_valid_m: float

    nrays: int
    az_spacing_median_deg: float
    az_spacing_min_deg: float
    az_spacing_max_deg: float
    az_max_dev_deg: float
    az_max_dev_frac: float
    az_lattice_closure_deg: float
    az_coverage_deg: float
    is_full_360: bool
    az_monotonic: bool
    az_direction: int
    n_wrap_seams: int
    n_revolutions: float
    az_first_deg: float

    el_median_deg: float
    el_max_dev_deg: float
    el_std_deg: float

    split_cut_group: int
    split_cut_size: int

    sweep_class: SweepClass
    class_reason: str

    def as_row(self):
        """Return the record as a flat dict of scalars, ready for a table.

        ``sweep_class`` is rendered as its string value so the row serialises to CSV
        without special handling. Callers wanting a DataFrame can do
        ``pd.DataFrame([g.as_row() for g in census_radar(radar)])`` --- this package
        does not depend on pandas to offer that.
        """
        row = asdict(self)
        row["sweep_class"] = self.sweep_class.value
        return row


def _wrapped_azimuth_differences(azimuths_deg):
    """Consecutive azimuth differences, wrapped to (-180, 180]."""
    differences = np.diff(azimuths_deg)
    return (differences + 180.0) % 360.0 - 180.0


#: Fields preferred, in order, when no ``valid_field`` is named. Every entry reaches
#: the full surveillance range on a WSR-88D split cut, unlike the Doppler moments.
VALID_FIELD_PREFERENCE = (
    "reflectivity",
    "DBZH",
    "corrected_reflectivity",
    "reflectivity_horizontal",
    "total_power",
)


def _pick_valid_field(radar):
    """Choose which field measures a sweep's usable range, deterministically.

    This is not a cosmetic tidy-up. ``next(iter(radar.fields))`` picks whichever key
    the dict happens to yield first, and Py-ART builds that dict from a set, so under
    hash randomisation the winner differs *between processes* for a byte-identical
    file. On a NEXRAD split cut the choice matters enormously: ``reflectivity``
    reaches ~460 km while ``velocity`` stops at ~2 km, so ``range_max_valid_m`` and
    therefore :func:`~radar_palette.gridding.cones.dedup_sweeps` flip between the
    surveillance and Doppler member of the same elevation. Measured on a KLOT VCP-212
    volume with six SAILS repeats at 0.48 deg, that made the gridded peak alternate
    between 67.6 and 86.9 dBZ across identical runs.

    A reflectivity-like field is preferred because it is the one that spans the full
    surveillance sweep; failing that, the alphabetically first field name is used, so
    the answer still does not depend on dict ordering.
    """
    fields = getattr(radar, "fields", None)
    if not fields:
        return None
    for name in VALID_FIELD_PREFERENCE:
        if name in fields:
            return name
    return sorted(fields)[0]


def _count_valid_gates(radar, sweep_slice, valid_field):
    """Count gates up to the last one carrying unmasked data in this sweep."""
    if valid_field is None:
        valid_field = _pick_valid_field(radar)
    if valid_field is None:
        return None

    data = np.ma.asanyarray(radar.fields[valid_field]["data"][sweep_slice])
    unmasked = ~np.ma.getmaskarray(data)
    if np.isscalar(unmasked) or unmasked.ndim == 0:
        return int(data.shape[-1])
    gates_with_data = np.flatnonzero(unmasked.any(axis=0))
    return int(gates_with_data[-1] + 1) if gates_with_data.size else 0


def _classify(
    range_spacing_uniform,
    range_spacing_max_dev_m,
    az_monotonic,
    n_revolutions,
    is_full_360,
    az_max_dev_frac,
    az_lattice_closure_deg,
    az_spacing_median_deg,
    az_coverage_deg,
):
    """Decide the sweep class, returning it with the measurement that decided it."""
    if not range_spacing_uniform:
        return (
            SweepClass.NON_UNIFORM,
            "range spacing non-uniform "
            f"(max deviation {range_spacing_max_dev_m:.3g} m)",
        )
    if not az_monotonic:
        return SweepClass.NON_UNIFORM, "azimuth not monotonic after unwrapping"
    if n_revolutions > 1.02:
        return (
            SweepClass.NON_UNIFORM,
            f"sweep covers {n_revolutions:.2f} revolutions",
        )
    if (
        is_full_360
        and az_max_dev_frac <= AZ_DEV_TOL_FRAC
        and az_lattice_closure_deg <= AZ_DEV_TOL_FRAC * az_spacing_median_deg
    ):
        return (
            SweepClass.EXACT_UNIFORM_PERIODIC,
            f"full 360, deviation {az_max_dev_frac:.2e} of spacing "
            f"<= {AZ_DEV_TOL_FRAC:g}",
        )
    if not is_full_360 and az_max_dev_frac <= AZ_DEV_TOL_FRAC:
        return (
            SweepClass.UNIFORM_PARTIAL_SECTOR,
            f"sector {az_coverage_deg:.1f} deg, deviation "
            f"{az_max_dev_frac:.2e} of spacing",
        )
    return (
        SweepClass.NON_UNIFORM,
        f"azimuth deviation {az_max_dev_frac:.3f} of spacing > {AZ_DEV_TOL_FRAC:g}",
    )


def census_sweep(radar_like, sweep, valid_field=None):
    """Measure and classify the sampling geometry of one sweep.

    Parameters
    ----------
    radar_like : pyart.core.Radar or xarray.DataTree
        Volume to inspect, in either supported object family.
    sweep : int
        Index of the sweep to census.
    valid_field : str, optional
        Field used to establish how many gates carry data. Defaults to the first
        field present. Matters for NEXRAD split cuts, where the Doppler cuts share
        the reflectivity range axis but stop much earlier.

    Returns
    -------
    SweepGeometry
        The measurement. ``split_cut_group`` and ``split_cut_size`` are placeholders
        here --- a single sweep cannot know about its siblings; use
        :func:`census_radar` for those.
    """
    radar, _ = to_pyart_radar(radar_like)
    start, end = radar.get_start_end(sweep)
    sweep_slice = slice(start, end + 1)

    azimuths = np.asarray(radar.azimuth["data"][sweep_slice], dtype=np.float64)
    elevations = np.asarray(radar.elevation["data"][sweep_slice], dtype=np.float64)
    ranges = np.asarray(radar.range["data"], dtype=np.float64)
    nrays = azimuths.size
    ngates = ranges.size

    gate_spacings = np.diff(ranges)
    range_spacing_m = (
        float(np.median(gate_spacings)) if gate_spacings.size else float("nan")
    )
    range_spacing_max_dev_m = (
        float(np.max(np.abs(gate_spacings - range_spacing_m)))
        if gate_spacings.size
        else 0.0
    )
    range_spacing_uniform = bool(
        gate_spacings.size == 0
        or range_spacing_max_dev_m <= DR_DEV_TOL_FRAC * abs(range_spacing_m)
    )

    ngates_valid = _count_valid_gates(radar, sweep_slice, valid_field)
    if ngates_valid is None:
        ngates_valid = ngates

    differences = _wrapped_azimuth_differences(azimuths)
    n_wrap_seams = int(np.sum(np.abs(np.diff(azimuths)) > 180.0))
    unwrapped = azimuths[0] + np.concatenate(([0.0], np.cumsum(differences)))
    span = float(unwrapped[-1] - unwrapped[0])
    az_direction = 1 if span >= 0 else -1

    spacings = np.abs(differences)
    az_spacing_median_deg = (
        float(np.median(spacings)) if spacings.size else float("nan")
    )
    az_monotonic = bool(
        spacings.size == 0 or np.all(differences > 0) or np.all(differences < 0)
    )
    # Each ray owns one spacing of arc, so coverage exceeds the span by one spacing.
    az_coverage_deg = abs(span) + az_spacing_median_deg
    n_revolutions = az_coverage_deg / 360.0
    is_full_360 = bool(az_coverage_deg >= 360.0 - 0.5 * az_spacing_median_deg)

    ray_index = np.arange(nrays, dtype=np.float64)
    if is_full_360:
        # The candidate lattice is forced to close exactly on the circle, so the
        # only free parameter is the starting azimuth.
        ideal_spacing = az_direction * 360.0 / nrays
        start_azimuth = float(np.mean(unwrapped - ray_index * ideal_spacing))
        residuals = unwrapped - (start_azimuth + ray_index * ideal_spacing)
        az_lattice_closure_deg = abs(
            abs(span + az_direction * az_spacing_median_deg) - 360.0
        )
    else:
        # A sector cannot be assumed to close, so fit a free-slope ramp.
        design = np.vstack([np.ones_like(ray_index), ray_index]).T
        coefficients, *_ = np.linalg.lstsq(design, unwrapped, rcond=None)
        residuals = unwrapped - design @ coefficients
        az_lattice_closure_deg = float("nan")

    az_max_dev_deg = float(np.max(np.abs(residuals))) if nrays > 1 else 0.0
    az_max_dev_frac = (
        az_max_dev_deg / az_spacing_median_deg
        if az_spacing_median_deg and np.isfinite(az_spacing_median_deg)
        else float("nan")
    )

    el_median_deg = float(np.median(elevations))
    sweep_class, class_reason = _classify(
        range_spacing_uniform=range_spacing_uniform,
        range_spacing_max_dev_m=range_spacing_max_dev_m,
        az_monotonic=az_monotonic,
        n_revolutions=n_revolutions,
        is_full_360=is_full_360,
        az_max_dev_frac=az_max_dev_frac,
        az_lattice_closure_deg=az_lattice_closure_deg,
        az_spacing_median_deg=az_spacing_median_deg,
        az_coverage_deg=az_coverage_deg,
    )

    return SweepGeometry(
        sweep=int(sweep),
        fixed_angle=float(radar.fixed_angle["data"][sweep]),
        ngates=int(ngates),
        ngates_valid=int(ngates_valid),
        range_first_m=float(ranges[0]),
        range_spacing_m=range_spacing_m,
        range_spacing_uniform=range_spacing_uniform,
        range_spacing_max_dev_m=range_spacing_max_dev_m,
        range_max_m=float(ranges[-1]),
        range_max_valid_m=float(ranges[max(ngates_valid - 1, 0)]),
        nrays=int(nrays),
        az_spacing_median_deg=az_spacing_median_deg,
        az_spacing_min_deg=float(np.min(spacings)) if spacings.size else float("nan"),
        az_spacing_max_deg=float(np.max(spacings)) if spacings.size else float("nan"),
        az_max_dev_deg=az_max_dev_deg,
        az_max_dev_frac=az_max_dev_frac,
        az_lattice_closure_deg=float(az_lattice_closure_deg),
        az_coverage_deg=float(az_coverage_deg),
        is_full_360=is_full_360,
        az_monotonic=az_monotonic,
        az_direction=int(az_direction),
        n_wrap_seams=n_wrap_seams,
        n_revolutions=float(n_revolutions),
        az_first_deg=float(azimuths[0]),
        el_median_deg=el_median_deg,
        el_max_dev_deg=float(np.max(np.abs(elevations - el_median_deg))),
        el_std_deg=float(np.std(elevations)),
        split_cut_group=-1,
        split_cut_size=1,
        sweep_class=sweep_class,
        class_reason=class_reason,
    )


def census_radar(radar_like, valid_field=None):
    """Census every sweep of a volume and tag split-cut groups.

    Parameters
    ----------
    radar_like : pyart.core.Radar or xarray.DataTree
        Volume to inspect, in either supported object family.
    valid_field : str, optional
        Passed through to :func:`census_sweep`.

    Returns
    -------
    list of SweepGeometry
        One record per sweep, in sweep order, with ``split_cut_group`` and
        ``split_cut_size`` filled in.

    Notes
    -----
    Returns dataclasses rather than a DataFrame so that this package does not
    require pandas. For a table::

        pd.DataFrame([geometry.as_row() for geometry in census_radar(radar)])
    """
    radar, _ = to_pyart_radar(radar_like)
    geometries = [
        census_sweep(radar, sweep, valid_field=valid_field)
        for sweep in range(radar.nsweeps)
    ]

    fixed_angles = np.array([geometry.fixed_angle for geometry in geometries])
    group_of_sweep = -np.ones(len(geometries), dtype=int)
    next_group = 0
    for index in range(len(geometries)):
        if group_of_sweep[index] >= 0:
            continue
        members = np.flatnonzero(
            np.abs(fixed_angles - fixed_angles[index]) <= SPLIT_CUT_TOL_DEG
        )
        group_of_sweep[members] = next_group
        next_group += 1

    for index, geometry in enumerate(geometries):
        geometry.split_cut_group = int(group_of_sweep[index])
        geometry.split_cut_size = int(np.sum(group_of_sweep == group_of_sweep[index]))
    return geometries
