# mdsreplace on the C-SAPR2 case library

Seventy-one BNF C-SAPR2 PPI volumes, spanning clear air to a squall line and
two processing levels, run through the workflow this subpackage exists for:
classify every gate, remove the non-meteorological echo -- biological,
clutter, second trip, no-return -- with a Py-ART `GateFilter`, and backfill
the removed gates with the minimum detectable signal at their range, so that
Barnes, Cressman or the NUFFT gridder has a floor to interpolate towards
instead of a hole.

```python
from radar_palette.gateid import gate_id, meteorological_gatefilter
from radar_palette.mdsreplace import mds_profile, replace_below_mds

radar, _ = gate_id(radar, freezing_level_m=4700.0)
gatefilter = meteorological_gatefilter(radar)      # drops NON_MET classes
profile = mds_profile(radar)                       # auto-picks the estimator
radar, meta = replace_below_mds(radar, gatefilter, profile=profile,
                                mds_field="minimum_detectable_signal")
```

## The floor is a property of the receiver, not of the weather

`auto` chose the exact `Z - SNR` route on all 71 volumes. The measured floor
at 1 km, by processing level:

| processing level | volumes | floor at 1 km | spread (sd) | fit residual |
|---|---|---|---|---|
| 2026 (`uncorrected_*` moments) | 44 | -40.05 dBZ | 0.17 dB | 0.15 dB |
| 2025 (censor-masked, noise-subtracted) | 27 | -37.410 dBZ | < 0.001 dB | 0.48 dB |

The 2.64 dB offset between the two groups is a statement about the moment
processing, not about the radar: no receiver changes between deployments by
2.6 dB and then holds to a millidecibel. Measure the floor per processing
level, not once for the instrument.

Within a level the floor ignores the weather. Across the 2.7 h of
`bnf_csapr2_c020` the meteorological gate fraction grew 1.6x (2.67 -> 4.28 %)
while the intercept moved 0.30 dB with no trend (-0.008 dB/h, p = 0.85). Over
the 3.5 h squall line the intercept moved 0.0006 dB while the domain went from
7 % to 47 % rain-filled.

**Reuse.** Measuring once and reusing the profile across a case changes the
fill by at most 0.188 dB (`bnf_csapr2_c020`, 17 volumes) or 0.48 dB
(`bnf_csapr2_c017`, 16 volumes over 4 h). Reuse within a case is safe at
0.2-0.5 dB; reuse across processing levels is not, at 2.6 dB.

## What gets removed, and how much is backfilled

Mean fraction of all gates per class, by storm mode (71 volumes):

| storm mode | n | no return | biological | clutter | second trip | kept as met | backfilled |
|---|---|---|---|---|---|---|---|
| clear air | 2 | 88.2 % | 9.9 % | 0.12 % | 0.01 % | 1.8 % | 99.9 % |
| light drizzle | 6 | 53.8 % | 14.0 % | 1.22 % | 2.03 % | 26.5 % | 95.4 % |
| isolated cells | 21 | 89.4 % | 6.4 % | 0.31 % | 0.15 % | 3.7 % | 98.0 % |
| multicell cluster | 20 | 83.7 % | 4.3 % | 0.28 % | 0.39 % | 11.3 % | 90.1 % |
| QLCS | 22 | 74.0 % | 3.2 % | 0.58 % | 0.67 % | 21.6 % | 78.7 % |

Biological echo is the largest *removed* class after receiver noise in every
mode, and at the lowest sweep it dominates: 19.3 % of sweep-0 gates on the
2026-08-04 cells case, a filled insect disc out to 40-50 km that the filter
removes and the backfill turns into a smooth radial ramp. Second trip is
present on every volume of the squall line, peaking at 7.05 % of sweep-0 gates
at 03:10Z on the far NW edge at 90-110 km.

No volume was left with a non-finite gate, and on all 71 the filled values
matched an independently evaluated MDS field to within 4e-6 dB. Peak
reflectivity was unchanged by the fill on every volume: nothing was written
over weather.

## Things to know before using the fill

**The far-field fill is brighter than 0 dBZ.** An honest C-band MDS reaches
+1.3 dBZ at 113 km (2026 level) and +3.4 dBZ at 110 km (2025 level), and
exceeds 0 dBZ beyond about 72 km. A downstream analysis that separates real
from synthetic gates by a reflectivity threshold will get this wrong: read the
`mds_replaced` flag, or set `cap_dbz`.

**Only the measured field is filled by default.** `meta["fields"]` reports
what was touched; on ARM volumes the floor resolves to
`uncorrected_reflectivity_h`, leaving the corrected `reflectivity` -- which is
what most users grid -- with its holes. Pass `fields=` explicitly.

**`replace_below_mds` mutates its input.** Re-measuring on the returned object
is circular: every removed gate now holds exactly the fill, so the noise route
reads back its own answer.

**A restrictive `include_fields` can starve `auto`.** Naming only
`uncorrected_*` moments on a file that carries corrected SNR dropped one
volume onto the censoring estimator, which returned -59.6 dBZ at 1 km against
-37.4 from the exact route.

**The 2025-level floor is not an independent measurement.** `Z - SNR` there is
deterministic to 4e-4 dB and identical on files months apart, so the route
recovers the noise power the processing assumed. That is still the right value
to fill with -- it is the floor the moments were computed against -- but it
cannot check the receiver, and the noise-mode cross-check disagrees with it by
5-11 dB. Both conditions now raise a warning naming the numbers.

## Findings that belong to `gateid`, not here

Running 71 volumes surfaced three classifier behaviours worth a separate look:

1. **A near-field disc is labelled `heavy_rain` and therefore kept.** Inside
   about 2 km, 95 % of gates take that label on the 2026 cells case
   (16,782 connected sweep-0 gates, median raw Z 51.2 dBZ, |v| < 1 m/s), and
   `frac_heavy_rain` sits at 1.74-1.76 % in every volume of the case
   regardless of the weather. In clear air the only echo surviving the
   meteorological filter is stationary 45-61 dBZ ground clutter carrying that
   label.
2. **Data-void gates are classified on temperature alone.** Every moment
   masked gives NaN for every feature except the range/elevation-derived
   temperature, and the class with the heaviest temperature weight wins.
3. **Clutter fraction rises with elevation** (0.31 % at 1.5 deg, 0.45 % at
   14 deg), where the beam is 25 km AGL at 100 km range.

## Artifacts

Per-lane tables, figures, notes and the three case animations are saved
alongside this document; `mdsreplace_csapr2_summary.csv` is the merged
per-volume table behind every number above.
