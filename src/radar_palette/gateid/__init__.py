"""Gate identification: per-gate classification of the dominant scatterer.

Two capabilities, one shared physics core:

``radar_palette.gateid.single``
    Single-volume classification from polarimetric moments. Memoryless: each
    volume is classified with no reference to any other. Fuzzy membership
    functions with budget-normalised scoring, melting-layer hard constraints,
    and speckle suppression by signal-to-noise, phase coherence and run length.

``radar_palette.gateid.temporal``
    Measurement and exploitation of the time domain. Transition matrices,
    persistence skill against the marginal, accumulated clutter priors, and a
    one-way evidence-gated temporal prior. The measurement tools are usable on
    their own and are deliberately built to be able to return a negative
    result.

Both accept Py-ART ``Radar`` and xradar ``DataTree`` volumes through
:func:`radar_palette.gateid.gate_id`, which returns the family it was given.

The classifier is designed for chains that identify gates *before* correction,
where the gate ID is the control flow rather than a diagnostic: it decides which
gates reach phase processing, attenuation correction, dealiasing and rainfall
estimation.

Importing :mod:`radar_palette` also registers a ``radarpalette`` accessor on
:class:`xarray.DataTree`, so the classifier is available as a method on an
xradar object::

    tree = xd.io.open_cfradial1_datatree("cfrad.20260820_041403.nc")
    tree.radarpalette.gateid(freezing_level_m=4670.0)
    tree["sweep_0"]["gate_id"]

Examples
--------
Classify a volume and build a filter for downstream steps::

    from radar_palette.gateid import gate_id, meteorological_gatefilter

    radar, meta = gate_id(radar, freezing_level_m=4670.0)
    gatefilter = meteorological_gatefilter(radar)

Measure whether classes actually persist between volumes::

    from radar_palette.gateid.temporal import persistence_skill

    skill = persistence_skill(labels_t0, labels_t1)
    skill["moderate_rain"]["skill"]  # excess over chance, 0 to 1
"""

from __future__ import annotations

from radar_palette.gateid import accessor, features, single, temporal
from radar_palette.gateid.accessor import (
    RadarPaletteDataTreeAccessor,
    register_accessors,
)
from radar_palette.gateid.entry_points import gate_id, meteorological_gatefilter
from radar_palette.gateid.features import build_features, resolve_fields
from radar_palette.gateid.single import (
    CLASSES,
    FROZEN,
    LIQUID,
    NAME_TO_CODE,
    NON_MET,
    angular_texture,
    classify,
    depolarization_ratio,
    despeckle,
    noise_floor_mask,
    texture,
    trapmf,
)
from radar_palette.gateid.temporal import (
    accumulate_clutter_prior,
    apply_temporal_prior,
    class_composition,
    composition_drift,
    persistence_skill,
    temporal_prior,
    transition_matrix,
)

__all__ = [
    "CLASSES",
    "RadarPaletteDataTreeAccessor",
    "FROZEN",
    "LIQUID",
    "NAME_TO_CODE",
    "NON_MET",
    "accessor",
    "accumulate_clutter_prior",
    "angular_texture",
    "apply_temporal_prior",
    "build_features",
    "class_composition",
    "classify",
    "composition_drift",
    "depolarization_ratio",
    "despeckle",
    "features",
    "gate_id",
    "meteorological_gatefilter",
    "noise_floor_mask",
    "persistence_skill",
    "register_accessors",
    "resolve_fields",
    "single",
    "temporal",
    "temporal_prior",
    "texture",
    "transition_matrix",
    "trapmf",
]

# Registering on import is what makes ``tree.radarpalette.gateid(...)`` work
# after a bare ``import radar_palette``, which is how xradar and Py-ART both
# behave. It is idempotent and never raises.
register_accessors()
