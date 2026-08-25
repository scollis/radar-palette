"""Census tests, focused on the reproducibility of split-cut selection.

The bug these were written for is worth stating, because a reader would not guess it
from the code: ``_count_valid_gates`` used to fall back to ``next(iter(radar.fields))``
when no field was named. Py-ART builds its field dict from a set, so under hash
randomisation the first key differs *between processes*. On a NEXRAD split cut that
choice decides everything --- reflectivity spans the full surveillance sweep while the
Doppler moments stop at a fraction of the range --- so ``range_max_valid_m`` flipped,
``dedup_sweeps`` picked a different member of the same elevation, and the gridded peak
alternated between two values across identical runs of identical input.

A test that merely calls the census twice in one process cannot catch that: within a
process the dict order is fixed. The tests here attack it two ways --- by asserting the
selection is independent of insertion order, and by asserting the chosen field is a
reflectivity-like one rather than whichever key came first.
"""

import numpy as np
import pyart
import pytest

from radar_palette.gridding import census_radar, census_sweep, dedup_sweeps
from radar_palette.gridding.census import (
    VALID_FIELD_PREFERENCE,
    _pick_valid_field,
)

RANGE_FIRST_M = 500.0
RANGE_SPACING_M = 250.0


def split_cut_radar(ngates=64, nrays=72, surveillance_gates=None, doppler_gates=6):
    """A two-sweep volume at one elevation, mimicking a WSR-88D split cut.

    Both sweeps share the range axis. The surveillance sweep carries reflectivity to
    the end of it; the Doppler sweep carries velocity over a short range and masks
    reflectivity beyond a few gates, which is what makes the choice of validity field
    decisive.
    """
    surveillance_gates = surveillance_gates or ngates
    radar = pyart.testing.make_empty_ppi_radar(ngates, nrays, 2)
    radar.range["data"] = RANGE_FIRST_M + RANGE_SPACING_M * np.arange(
        ngates, dtype="float64"
    )
    azimuths = np.tile(np.arange(nrays) * (360.0 / nrays), 2)
    radar.azimuth["data"] = azimuths
    radar.elevation["data"] = np.full(azimuths.size, 0.5)
    radar.fixed_angle["data"] = np.array([0.5, 0.5])

    reflectivity = np.ma.masked_all((2 * nrays, ngates), dtype="float32")
    reflectivity[:nrays, :surveillance_gates] = 10.0
    reflectivity[nrays:, :doppler_gates] = 10.0
    velocity = np.ma.masked_all((2 * nrays, ngates), dtype="float32")
    velocity[nrays:, :doppler_gates] = 1.0
    velocity[:nrays, :doppler_gates] = 1.0

    for name, data, units in (
        ("reflectivity", reflectivity, "dBZ"),
        ("velocity", velocity, "m/s"),
    ):
        radar.add_field(
            name,
            {"data": data, "units": units, "_FillValue": -9999.0},
            replace_existing=True,
        )
    return radar


class TestTheValidityFieldIsChosenDeterministically:
    """The regression guard for the split-cut selection flip."""

    def test_it_prefers_reflectivity_over_whatever_came_first(self):
        """The failure mode was taking the dict's first key, so order must not decide.

        ``velocity`` is inserted first here. The old fallback would have picked it and
        measured the sweep's usable range from a field that stops early.
        """
        radar = split_cut_radar()
        radar.fields = {
            "velocity": radar.fields["velocity"],
            "reflectivity": radar.fields["reflectivity"],
        }
        assert _pick_valid_field(radar) == "reflectivity"

    def test_the_choice_is_invariant_under_field_insertion_order(self):
        """Same fields, every ordering, one answer.

        This is the property that hash randomisation broke. Enumerating orderings in
        one process is the closest a unit test can get to running in many.
        """
        radar = split_cut_radar()
        original = dict(radar.fields)
        chosen = set()
        for first in original:
            radar.fields = {first: original[first]}
            radar.fields.update({k: v for k, v in original.items() if k != first})
            chosen.add(_pick_valid_field(radar))
        assert chosen == {"reflectivity"}

    def test_a_volume_without_a_preferred_field_still_answers_stably(self):
        """No reflectivity-like field: fall back to sorted order, never dict order."""
        radar = split_cut_radar()
        renamed = {
            "spectrum_width": radar.fields["velocity"],
            "differential_phase": radar.fields["reflectivity"],
        }
        radar.fields = renamed
        assert _pick_valid_field(radar) == "differential_phase"
        radar.fields = {k: renamed[k] for k in reversed(list(renamed))}
        assert _pick_valid_field(radar) == "differential_phase"

    def test_a_fieldless_volume_returns_none_rather_than_raising(self):
        radar = split_cut_radar()
        radar.fields = {}
        assert _pick_valid_field(radar) is None

    def test_every_preferred_name_is_reflectivity_like(self):
        """Guards the preference list itself against a Doppler moment creeping in.

        The whole point is that these fields span the surveillance sweep. A velocity
        or width field in this list would reintroduce the bug silently.
        """
        forbidden = ("velocity", "spectrum_width", "differential_phase")
        assert not set(VALID_FIELD_PREFERENCE) & set(forbidden)


class TestSplitCutDeduplication:
    """dedup_sweeps must keep the member that actually reaches further."""

    def test_it_keeps_the_surveillance_sweep_of_a_split_cut(self):
        radar = split_cut_radar(surveillance_gates=64, doppler_gates=6)
        kept = dedup_sweeps(census_radar(radar))
        assert len(kept) == 1
        assert kept[0].sweep == 0

    def test_it_keeps_the_surveillance_sweep_whichever_index_it_has(self):
        """Order of sweeps in the file must not decide which one survives."""
        radar = split_cut_radar(surveillance_gates=6, doppler_gates=64)
        kept = dedup_sweeps(census_radar(radar))
        assert len(kept) == 1
        assert kept[0].sweep == 1

    def test_the_measured_range_comes_from_reflectivity_not_velocity(self):
        """The quantity dedup_sweeps sorts on must be the reflectivity reach.

        With velocity confined to the first few gates, a census that measured validity
        from velocity would report a short range for the surveillance sweep too, and
        the tie would then be decided by sweep index alone --- which is exactly how
        the wrong member got picked.
        """
        radar = split_cut_radar(surveillance_gates=64, doppler_gates=6)
        survey = census_sweep(radar, 0)
        doppler = census_sweep(radar, 1)
        assert survey.range_max_valid_m > doppler.range_max_valid_m
        assert survey.range_max_valid_m == pytest.approx(
            RANGE_FIRST_M + RANGE_SPACING_M * 63
        )
