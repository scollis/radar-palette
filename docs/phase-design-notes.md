# `radar_palette.phase`: design decisions and their justification

Every number in this note was computed while building the module. Where a
decision rests on a design choice rather than a measurement, it says so.

## The governing assumption

All KDP is a retrieval. A radar measures power, a power ratio and a phase. KDP
is *defined* as half the range gradient of the **propagation** differential
phase, and what a receiver reports is a mixture:

```
Psi_DP(measured) = Phi_propagation + delta(backscatter) + NUBF + noise
```

Nothing that works on one ray of phase can separate those terms, so no shipped
KDP field is a measurement — including this one. The design consequence is the
whole architecture: rather than estimating `delta`, the retrieval **bounds** KDP
using Z and ZDR, which are direct measurements carrying no `delta` term. Any
structure claim about a retrieved KDP field should likewise be anchored on Z and
ZDR.

## 1. System phase

Three estimators following the wradlib `dp.system_phidp_block` / `_first` /
`_hist` algorithms, each returning a per-ray `sysphi_ray` and a sweep-level
`sysphi` (the median of the `n_lowest_rays` smallest ray values). Gates are
masked on `rho_hv >= 0.9` first.

**wradlib is reimplemented, not imported.** `tests/test_package.py` enforces
that every third-party import is a declared dependency, and wradlib is not one.
Adding it for three functions was the alternative; the algorithms are short and
the guard exists for a good reason. The reimplementation is deliberate and is
not an independent method.

**Decision: a single sweep-level offset by default.** Three arguments, in
descending order of how much weight they carry:

1. **No hardware mechanism varies the system offset within one sweep.** It lives
   in the transmit/receive chain. Ray-to-ray structure in the *estimate* is
   evidence about the estimator until something else is shown.
2. **KDP cannot see the difference.** A per-ray additive constant is annihilated
   by a range derivative. The choice affects absolute PhiDP products,
   attenuation correction, and which branch the unfolder assigns — never KDP.
   `test_kdp_is_blind_to_the_sweep_versus_per_ray_choice` asserts this.
3. **The ray-wise estimate has quantifiable noise**, approximately
   `1.253 * sigma_phi / sqrt(n_eff)`, so a spread within that is noise and
   applying it as a correction makes the field *less* consistent ray to ray.

**The measured diagnostic came out against argument 3, and the default stands on
arguments 1 and 2 instead.** With `sigma_phi = 1.8` deg:

| volume | estimator | robust SD | expected noise | excess |
|---|---|---|---|---|
| BNF C-SAPR2 RHI, 100 m gates | block | 5.18 deg | 0.71 deg | 7.3 |
| BNF C-SAPR2 RHI, 100 m gates | first | 4.48 deg | 0.71 deg | 6.3 |
| KHTX WSR-88D PPI, 250 m gates | block | 2.74 deg | 1.13 deg | 2.4 |
| KHTX WSR-88D PPI, 250 m gates | first | 2.35 deg | 0.71 deg | 3.3 |

Every row returns `per_ray_justified = True`, i.e. the diagnostic says use a
per-ray offset. Three mechanisms could produce that and the diagnostic cannot
separate them: the `block` estimator taking its window at a **different range**
on each ray (so real propagation phase is misread as a ray difference);
**within-ray gate correlation** making `n_eff` smaller than the gate count (at
BNF the lag-1 phase autocorrelation is about 0.5, which for a 10-gate block
moves the BNF excess from 7.3 to roughly 4.2 — still above threshold); and
genuine ray-to-ray variation, for which there is no mechanism. The first is most
likely. `excess_ratio` should be read as an **upper bound** on real ray-to-ray
inconsistency, not an estimate of it.

**The consistency diagnostic is blind to**: field shape (it measures offsets
only); a smooth azimuthal trend, which raises the spread but not the
adjacent-ray jump, so both numbers must be read; and any offset shared by every
ray, which is the sweep value itself.

Correlation between adjacent rays is deliberately **not** used: a constant
per-ray offset still correlates at ~0.99, so correlation cannot see the quantity
of interest.

## 2. Unfolding

**`wrap_deg` is a parameter with a measured default of 360 degrees**, the
ambiguity of a receiver measuring the H–V phase difference on simultaneous
transmission. Measured on both smoke-test volumes: the BNF RHI spans exactly
360.0 deg over `[-180.0, 180.0]`, the KHTX volume 359.6 deg over `[0.0, 359.6]`
— the same interval under two sign conventions. It remains a *stated* default
rather than a detected one, because correcting with the wrong interval leaves a
silent residual step of the difference at every repaired gate
(`test_the_wrong_wrap_value_leaves_a_residual_step` pins that at exactly 180
deg for a 180-deg ambiguity corrected as 360).

**Folds are distinguished from backscatter-phase excursions by persistence, not
step size.** Two physically different things produce a large gate-to-gate step:
a fold, which is a permanent branch change, and a `delta` excursion or noise
spike, which reverts within a few gates. `numpy.unwrap` corrects both, and on a
stream whose large steps are excursions it destroys the data it was meant to
repair. A candidate here is accepted only if the median phase over the next
`confirm_gates` (default 3) has genuinely moved to the new branch.

Measured on the two volumes:

| volume | large steps | of total | persistent | confirmed folds |
|---|---|---|---|---|
| BNF C-SAPR2 RHI | 4 | 21,642 (0.018%) | 0 | 0 |
| KHTX WSR-88D PPI | 1,334 | 350,861 (0.380%) | 79 (5.9%) | 38 |

On the KHTX volume `numpy.unwrap` would have applied all 1,334 corrections
where 38 were warranted. `fold_statistics` is the intended first call precisely
so that this ratio is seen before anything is corrected.

**Verified against synthetic truth before touching real data**, as required:
one fold, two folds, no folds, a 180-degree ambiguity, and the wrong wrap value
are all recovered or refused exactly (`tests/phase/test_unfold.py`).

**Blind to**: a fold inside a masked gap (there is no step to measure across
it); a fold in the first gate of a ray (a fold is defined by a step); and the
exact fold gate when noise makes the true phase cross the boundary several
times. That last is intrinsic — measured on synthetic truth with 1.8 deg noise
and two folds in 800 gates, exactly one gate landed on the wrong branch, off by
exactly one whole interval. Errors of this class are always a whole interval,
never a fraction of one, which is what makes them recognisable.

## 3. Bounds

Following Huang et al. (2016) Eq. 14–16: a self-consistency box, a phase-only
least-squares floor, and a reflectivity cap.

**Coefficients come from a forward model, not a fitted polynomial.** Computed
with rustmatrix 2.2.0: C band (also S and X), water at 10 degC, Brandes, Zhang
& Vivekanandan (2002) Eq. (2) drop shapes **inverted to the h/v convention the
T-matrix wants**, 64-point table to D_max = 8 mm, normalized gamma PSD over
D0 = 0.50–3.50 mm, mu in (0, 3, 5, 8), Nw in (1e3, 8e3, 1e5), fitted over
ZDR 0.2–4.0 dB:

| band | a1 | a2 | a3 | a4 | max rel. residual | first positive root |
|---|---|---|---|---|---|---|
| S | 3.3383 | -1.7454 | 0.4638 | -0.04742 | 4.6% | 5.37 dB |
| C | 6.9471 | -2.8633 | 0.7745 | -0.09336 | 5.7% | 5.15 dB |
| X | 11.5058 | -3.0208 | -0.3671 | 0.12287 | 10.8% | none |

Two properties of that computation:

- **`KDP / z_H` moved by 2e-13 relative** across two decades of Nw at fixed
  (D0, mu) — numerically exact independence of drop concentration, as it must
  be since KDP and z_H are both exactly linear in Nw. The relation really is a
  function of ZDR alone. `test_kdp_over_z_is_independent_of_concentration`
  asserts the consequence.
- **Against Gourley et al. (2009) Table 4** — the same functional form fitted to
  a different shape relation, temperature and mu — this fit runs **1.05x at
  ZDR = 0.5 dB rising to 1.34x at 3 dB** at C band. Two credible forward models
  of the same relation disagree by a third at high ZDR.

That disagreement is the **direct justification for the wide default tolerance
factors** (`lower_factor = 0.5`, `upper_factor = 2.0`). The relation is not
accurate enough to bound KDP tightly, and a bound tighter than the uncertainty
in the relation that produced it is a bias, not a constraint. The X-band fit is
the poorest and its cubic has no positive root; treat it as the least
trustworthy of the three.

**The reflectivity cap ladder is complete, verified against the source.**
`((35.0, 8.0), (45.0, 10.0))` is Eq. (16) in full: 8 deg/km where the
self-consistency cap exceeds 8 and ZH < 35 dBZ, 10 where it exceeds 10 and
ZH < 45 dBZ. Eq. (16) has **exactly those two rungs and specifies no cap above
45 dBZ anywhere in the paper**, so `cap_above` defaulting to no reflectivity cap
there — leaving the self-consistency upper bound to act alone — *is* the
published behaviour rather than a gap.

This was checked directly against the accepted manuscript, which **is**
retrievable although `fetch_article_fulltext` does not find it: the OSTI record
redirects to the author-institution copy at
`https://www.bnl.gov/isd/documents/93430.pdf`, reachable via
`https://www.osti.gov/servlets/purl/1336090` once `www.bnl.gov` is allowlisted.
Eq. (16) is on p. 27 of that 58-page manuscript.

**The self-consistency constraint is bounded to rain**, per Huang's own
assessment that such methods fail "in critical situations such as hail cores
where these methods must rigidly adhere to consistency relationship constraints
that do not apply". `kdp_bounds` takes a `rain_mask` and *releases* the box
outside it to a permissive symmetric interval rather than applying a relation
known to be out of scope. `rain_mask_from_context` raises rather than defaulting
to all-rain.

**The sign policy is altitude-conditioned, not global.** Huang 2016 enforces
`KDP_L >= 0` everywhere, correct for its rain-only scope and wrong as a general
prior: above the melting layer in an electrically active storm, vertically
aligned ice crystals make negative KDP the signal rather than noise, and a
global positivity bound erases exactly that. The default ramps the floor from 0
to `negative_floor` over `blend_km` centred on the isotherm.
`'rain_nonnegative'` reproduces the published behaviour and **is not the
default**. The `negative_floor = -2.0` deg/km value is a **design choice, not a
measurement** — no aligned-ice KDP magnitude was measured for this module — and
exists only to keep the problem bounded.

**Bounds inherit everything from ZDR calibration.** The polynomial is *cubic*
in ZDR and goes negative above ~5.1 dB at C band. This module cannot check the
ZDR offset; it must be settled upstream.

**Crossings are counted, not hidden.** Where the reflectivity cap falls below
the self-consistency floor the lower bound is clipped to the upper and
`metadata['n_crossed']` records it. Those are the hail-core gates, and a large
count means the box is doing nothing there. Measured: 3,601 gates on the BNF
RHI, 58,035 on the KHTX volume.

## 4. The estimator

`min sum_i w_i |x_i - b_i|` subject to `K_L <= M x <= K_U`, solved per ray with
HiGHS through `scipy.optimize.linprog`. `M` is a Savitzky–Golay first-derivative
filter, halved and divided by the gate spacing; for length 5 its weights are
`[-0.2, -0.1, 0, 0.1, 0.2]`, exactly Giangrande's `d5`.

### No curvature or total-variation penalty, not even as an option

Neither Giangrande 2013 nor Huang 2016 has one. Giangrande's own argument is
that with the L1 inequalities alone the cost reduces to zero at `x = b`, so the
*constraints* are what regularise. A TV penalty is worse than merely
unfaithful: its solution family is piecewise constant, so it **prefers**
staircase KDP fields rather than tolerating them.

### The reported KDP is exactly the constrained quantity

`kdp = M x` with the same `M` the box constrains, so `K_L <= kdp <= K_U` holds
by construction; `test_the_reported_kdp_obeys_the_box_it_was_given` asserts it,
and both smoke-test volumes returned 100.0000% of reported gates inside the box.
Py-ART's `phase_proc_lp` instead constrains one filter and reports KDP from
another, so its output carries no guarantee from its own constraints.

The cost: Giangrande's coupled **strong**-monotonicity post-filter is not
applied, because applying it would break the guarantee. Sub-filter oscillation
is therefore controlled only by the smoothing length. That is a documented
deviation from Giangrande 2013, and it should be measured on the output with a
step-likeness diagnostic.

### Offset invariance is structural — and was not what it first appeared

Required and tested, not assumed. The mechanism: `M` annihilates constants (the
derivative weights sum to zero, asserted at five filter lengths), the phase
variables are unbounded, nothing anchors `x` at an absolute value, and
zero-weight gates contribute **no rows at all** rather than being zero-filled.

On a clean ray this gives invariance to **2.9e-13** under shifts of 0.5, -37,
360 and 1000 degrees.

**But a first implementation failed the test badly, and the reason was not the
tolerance.** With a masked gap, adding a constant to the input moved the
reported KDP by up to **20.2 deg/km** — and the moving values sat exactly on the
box edges. An unweighted gate leaves a *free* phase variable; every derivative
window containing one has a KDP the objective does not see, and an LP optimum
sits on a vertex, so the solver settles it on a bound. The number that came back
was arbitrary within the box and looked exactly like a retrieval.

Requiring a data *majority* in the window does not fix it — a 5-gate gap under
an 11-gate filter still failed at 20 deg/km. Only excluding free variables
entirely does:

| gap (gates) | worst \|ΔKDP\| at fraction 0.5 | at fraction 1.0 |
|---|---|---|
| 0 | 2.9e-13 | 2.9e-13 |
| 1 | — | 2.9e-13 |
| 5 | 2.0e+01 | 2.9e-13 |
| 20 | 1.99e+01 | 2.9e-13 |
| 60 | 2.0e+01 | 2.9e-13 |

So `min_window_weight_fraction` defaults to **1.0**. **The cost is coverage and
it is not small**: with 5% of gates randomly dropped, reported gates fell from
390/400 to 254/400, and on the BNF RHI smoke test (where `rho_hv >= 0.9` admits
only 22.2% of gates) just 10.0% of the subset was reported against 58.3% on the
KHTX volume. The documented remedy is to fill gaps *upstream* — linear
interpolation of phase between surrounding valid gates is offset-equivariant, so
it preserves invariance — and pass full weights, which makes the invented phase
visible in the caller's code instead of hidden as a solver artefact.

### Smoothing length in kilometres, no default

`smoothing_km` is a required argument. It must be matched to the scale of the
DSD field being retrieved, which is measurable independently from Z and ZDR: at
ARM BNF the along-range decorrelation length is 0.75 km in Z and 0.98 km in ZDR,
so ~1 km is matched and 4–5 km is not; an inherited 4.66 km was about 5x too
long. Huang et al. used 2 km. There is no defensible default because the right
value is a property of the radar and the storm.

Quantisation to an odd gate count is reported, not hidden:
`realised_smoothing_km` was 1.10 km (11 gates) on the 100 m BNF gates and 1.25
km (5 gates) on the 250 m KHTX gates, both from a 1.0 km request.
`test_a_longer_smoothing_length_broadens_the_recovered_core` measures the cost
directly: recovered core width rises monotonically across 1.0, 2.0 and 4.66 km,
by more than 1.3x from the first to the last.

### Recovery of a known Gaussian

Not "it ran". For a true KDP Gaussian of amplitude 3.0 deg/km and sigma 1.0 km
with an 11-gate (1.10 km) filter, the recovered width should be
`sqrt(sigma^2 + (L*dr)^2/12)`, which for the realised 1.10 km filter is
**1.049 km**, and the peak should fall by the same ratio because a derivative
filter conserves the integral. The test recomputes this from
`realised_smoothing_km` at runtime rather than using the quoted number. Both are asserted to
5–6%, together with the integral itself to 2%. Under 1.8 deg per-gate phase
noise the amplitude holds to 25% and the width to 35%.

### Per-ray failure accounting

Every ray gets exactly one of `solved`, `no_valid_gates`,
`too_few_valid_gates`, `infeasible`, `solver_error`, and a failure carries its
reason text. A ray reported as failed with no recorded cause has cost a session
before.

## What the module is blind to, in one place

- It cannot separate `delta` from propagation phase. It bounds the result with
  independent measurements instead, which is a weaker claim.
- No cross-ray or vertical coupling, so non-uniform beam filling is invisible.
- Output inherits every error in the bounds, which inherit every error in the
  ZDR offset — a 1 dB ZDR offset moves the cubic self-consistency relation by
  tens of percent.
- The two-sided box is a *constraint*, not an estimate. Where it does not bind,
  the retrieval reduces to a Savitzky–Golay derivative of the measured phase.
  Bound tightness is what separates this from a plain filter.
- No object-flavour entry point yet: `radar_palette.gateid` accepts Py-ART and
  xradar objects, `radar_palette.phase` takes plain arrays. That layer is left
  for a follow-up.

## Acceptance results from the smoke test

Not the full campaign — one sweep from each of two volumes, 24 rays each.

| | BNF C-SAPR2 RHI (C) | KHTX WSR-88D PPI (S) |
|---|---|---|
| rays solved | 24 / 24 | 24 / 24 |
| cost | 11 ms/ray | 40 ms/ray |
| reported gates | 10.0% | 58.3% |
| inside the box | 100.0000% | 100.0000% |
| KDP median / p99 | 0.100 / 1.704 deg/km | 0.015 / 5.303 deg/km |
| median KDP by Z bin, rain | 0.082 (20–30), 0.297 (30–40) | 0.0015 (5–20), 0.0088 (20–30), 0.0898 (30–40), 0.509 (40–50) |
| fraction above 1 deg/km in light rain | 0.0000 | 0.0000 |

The Z-conditional median rises **monotonically** on both volumes and light rain
sits at the noise floor with no tail above 1 deg/km — a pass on the amplitude
test that a previous KDP configuration failed outright (non-monotonic, with 36%
of 5–20 dBZ gates above 1 deg/km).

Not run here, and required before this estimator is trusted on a campaign: the
decorrelation-length ratio against Z, the step-likeness diagnostic, and the
1.18 predicted KDP-core / Z-core width ratio. Those are the acceptance campaign,
which is a separate phase.

## References

- Huang, Zhang, Zhao & Giangrande (2016), *A Hybrid Method to Estimate Specific
  Differential Phase and Rainfall With Linear Programming and Physics
  Constraints*, [10.1109/TGRS.2016.2596295](https://doi.org/10.1109/TGRS.2016.2596295).
  Accepted manuscript retrieved in-session from `https://www.osti.gov/servlets/purl/1336090`
  (redirects to `www.bnl.gov/isd/documents/93430.pdf`); the architecture followed here is as
  specified in the task brief and the project's own verified notes. Eq. (16) is
  on p. 27 and has exactly two rungs, so the absence of a cap above 45 dBZ in
  this module reproduces the paper rather than working around it.
- Giangrande, McGraw & Lei (2013), the one-sided predecessor this deliberately
  departs from.
- Gourley, Illingworth & Tabary (2009),
  [10.1175/2008JTECHA1152.1](https://doi.org/10.1175/2008JTECHA1152.1), Table 4
  — the published self-consistency coefficients used only for comparison.
- Brandes, Zhang & Vivekanandan (2002), drop-shape relation Eq. (2).
- wradlib `dp.system_phidp_block` / `_first` / `_hist`, the system-phase
  algorithms reimplemented here.
