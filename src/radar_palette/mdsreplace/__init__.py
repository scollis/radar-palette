"""MDS replacement: give the gridder a floor to interpolate towards.

A gate filter says which gates are not weather. Removing them leaves a hole,
and every interpolator -- Barnes, Cressman, or the spectral operator in
:mod:`radar_palette.gridding` -- fills a hole either with nothing or with the
nearest surviving echo. Neither is right: the radar looked there and saw
nothing, which means the truth is *below the sensitivity*, and the sensitivity
is measurable from the volume in hand.

``radar_palette.mdsreplace.profile``
    Estimation of the minimum detectable signal against range. The noise floor
    is found as the **mode** of the reflectivity distribution once the
    :math:`r^2` trend is removed (research volumes that report every gate), or
    as the lower envelope of the surviving gates (thresholded products such as
    WSR-88D Level II). The fit is a power law whose slope is held at the
    physical 20 dB per decade unless released as a diagnostic.

``radar_palette.mdsreplace.entry_points``
    :func:`mds_profile` measures; :func:`replace_below_mds` fills gates a
    :class:`pyart.filters.GateFilter` excludes, plus masked and fill gates,
    and marks every one of them in a ``mds_replaced`` provenance field.

Both accept Py-ART ``Radar`` and xradar ``DataTree`` volumes and return the
family they were given.

Importing :mod:`radar_palette` also adds ``mds_profile`` and ``mdsreplace`` to
the ``radarpalette`` accessor, so the whole chain is available as methods on an
xradar object::

    tree.radarpalette.gateid(freezing_level_m=4700.0)
    tree.radarpalette.mdsreplace(exclude_field="gate_id")

Examples
--------
The Py-ART flavour, end to end::

    from radar_palette.gateid import gate_id, meteorological_gatefilter
    from radar_palette.mdsreplace import mds_profile, replace_below_mds

    radar, _ = gate_id(radar, freezing_level_m=4700.0)
    print(mds_profile(radar))
    # MdsProfile(noise/power_law: -38.75 dBZ at 1 km, 20.0 dB/decade, ...)

    gatefilter = meteorological_gatefilter(radar)
    radar, meta = replace_below_mds(radar, gatefilter)
    meta["n_replaced"], meta["fill_dbz_range"]

Reuse one profile across a case so every volume in a time series is filled to
the same floor::

    profile = mds_profile(volumes[0])
    for radar in volumes:
        replace_below_mds(radar, gatefilter_for(radar), profile=profile)

Notes
-----
The fill is an *upper bound* on a non-detection, not a measurement of one.
Because the MDS grows as :math:`r^2` it reaches light-rain values at long
range, so ``cap_dbz`` and the ``mds_replaced`` flag exist to keep that
visible downstream.
"""

from __future__ import annotations

from radar_palette.mdsreplace import accessor, entry_points, profile
from radar_palette.mdsreplace.accessor import (
    align_gatefilter_to_tree,
    register_mds_methods,
)
from radar_palette.mdsreplace.entry_points import (
    NO_RETURN_CLASSES,
    REFLECTIVITY_CANDIDATES,
    SECONDARY_REFLECTIVITY_CANDIDATES,
    mds_profile,
    replace_below_mds,
)
from radar_palette.mdsreplace.profile import (
    REFERENCE_RANGE_M,
    THEORETICAL_SLOPE_DB_PER_DECADE,
    MdsProfile,
    detection_floor_by_range,
    detrend_range,
    estimate_profile,
    find_sentinel_values,
    fit_power_law,
    noise_floor_by_range,
    snr_floor_by_range,
)

__all__ = [
    "NO_RETURN_CLASSES",
    "REFERENCE_RANGE_M",
    "REFLECTIVITY_CANDIDATES",
    "SECONDARY_REFLECTIVITY_CANDIDATES",
    "THEORETICAL_SLOPE_DB_PER_DECADE",
    "MdsProfile",
    "accessor",
    "align_gatefilter_to_tree",
    "detection_floor_by_range",
    "detrend_range",
    "entry_points",
    "estimate_profile",
    "find_sentinel_values",
    "fit_power_law",
    "mds_profile",
    "noise_floor_by_range",
    "profile",
    "register_mds_methods",
    "replace_below_mds",
    "snr_floor_by_range",
]

# Registering on import is what makes ``tree.radarpalette.mdsreplace(...)``
# work after a bare ``import radar_palette``. Idempotent, and never raises.
register_mds_methods()
