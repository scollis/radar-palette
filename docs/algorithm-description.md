# A bounded-$L_1$ specific differential phase retrieval

**`radar_palette.phase`, branch `phase-processing` at commit `b8515a6`**
Written 2026-09-02. Every number in this document was measured on real volumes and
read back from the saved artifact before being written here.

---

## 0. Status, stated first

**This is not yet a working retrieval, and the KDP fields it produces should not be
used as evidence about atmospheric structure.**

Across four research radars, between 92.3% and 97.3% of rain gates with a reported
KDP sit *exactly* on one of the retrieval's own bounds. Only **2.7% to 7.7%** of the
output is determined by the phase measurement at all. On the worked example ray,
*every single reported gate* sits on the self-consistency ceiling, and the fitted
phase recovers only **30%** of the measured phase accumulation.

The field is therefore, in the large majority of gates, a deterministic function of
$Z_H$ and $Z_{DR}$ wearing a phase retrieval's clothing. It looks good precisely
because it is a reflectivity product, and the acceptance suite it passes is the suite
a reflectivity product passes trivially. Section 7 documents this in full; Section 8
explains why the tests did not catch it.

The architecture below is, I think, sound. The bounds as currently parameterised are
not.

---

## 1. The problem

A radar measures power, a power ratio, and a phase. It does not measure KDP. KDP is
*defined* as half the range derivative of the propagation differential phase,

$$K_{DP}(r) = \tfrac{1}{2}\,\frac{\mathrm{d}\Phi_{DP}}{\mathrm{d}r}
\qquad [\mathrm{deg\ km^{-1}},\ r\ \mathrm{in\ km}]$$

and the propagation phase is only one term of what the radar actually returns:

$$\Psi_{DP} = \Phi_{\mathrm{prop}} + \delta_{hv} + \mathrm{NUBF} + \epsilon$$

where $\delta_{hv}$ is the backscatter differential phase, NUBF is the contribution of
non-uniform beam filling, and $\epsilon$ is estimation noise. Every term but the first
is a contaminant, and every one of them lives at the high-wavenumber end of the range
spectrum — exactly where a derivative operator has the most gain.

This is the whole difficulty. Differentiating a noisy sum is an ill-posed problem, and
the design freedom in *how* one differentiates it is where every KDP estimator differs
from every other. The tension was visible in the
first observational papers: differential phase can be estimated to better than
0.5° standard error with time and range averaging (Sachidananda and Zrnić 1986), but
the averaging needed to get there is itself what destroys the spatial resolution one
wanted. Forty years of estimator development — iterative range filtering to separate
propagation from backscatter phase (Hubbert and Bringi 1995), adaptive window length
(Wang and Chandrasekar 2009), Kalman-filter ensembles (Schneebeli and Berne 2012),
estimation at native resolution by stripping $\delta$ with polarimetric relations
(Reinoso-Rondinel et al. 2018) — is a sequence of attempts to buy back that
resolution. The compensating prize is real: KDP is immune to attenuation,
miscalibration, partial beam blockage and wet radomes (Zrnić and Ryzhkov 1996). It is
available only to an estimator that does not destroy the structure on the way.

$\delta_{hv}$ deserves separate mention because it is not small. At C band it is
routinely of the same order as the propagation phase change across a convective core,
and it is not a nuisance term that averages to zero — it is a systematic function of
drop size (Trömel et al. 2013). NUBF is likewise systematic rather than random
(Ryzhkov and Zrnić 1998; Gorgucci et al. 1999).

---

## 2. What the design has to respect

Three constraints were measured on the target data before any code was written. They
are what set the free parameters; none of them is a preference.

**Structure scale.** The DSD field at BNF decorrelates at **0.75–1.0 km** along range
below the freezing level, measured directly on $Z$ and (noise-corrected) $Z_{DR}$.
The instrumental floor, measured from the lag-1 autocorrelation of noise gates rather
than assumed, is ~0.1 km — the 100 m gates are not oversampled, so any scale above
that is atmosphere. A smoothing length materially longer than ~1 km is measuring the
filter. The value this replaces, inherited from earlier work, was 4.66 km — roughly
five times the field it was smoothing.

**Amplitude by regime.** A convective-tuned noise floor erases the entire ice-phase
signal. Dendritic-growth-layer bands peak at only 0.15–0.4 deg/km (Kennedy and
Rutledge 2011); ice-alignment regions reach $|K_{DP}| \approx 0.8$ deg/km in *both
signs* with $Z_{DR} \approx 0$ (Hubbert et al. 2014); downburst KDP cores are defined
at $\geq 1.0$ deg/km near and below the melting layer (Kuster et al. 2021). No single
threshold serves all three.

**Expected spatial relationship.** From T-matrix scattering at C band with Brandes
et al. (2002) drop shapes, the size-sensitivity exponents at fixed $N_w$ are
$Z \sim D_0^{7.37}$ and $K_{DP} \sim D_0^{6.23}$, and both are exactly linear in $N_w$.
Two fields with identical concentration dependence and similar size exponents must
have near-identical spatial structure: the predicted KDP-core to Z-core width ratio is
**1.18**. A retrieved KDP field materially broader than $Z$ in the same gates has been
smeared by the estimator, not by the storm. (This prediction is corroborated
analytically by Gourley et al. 2009, who note that $Z_H$ and $K_{DP}$ both scale with
$N_w$ so their ratio does not.)

---

## 3. Architecture

{{artifact:art_2c8d6010-d047-4778-83a3-ec087b82d76d}}

The organising idea is the **separation of channels**. The quantity being fitted
($\Psi_{DP}$) is contaminated by $\delta$ and NUBF. The quantities used to *constrain*
the fit ($Z_H$, $Z_{DR}$) are a power and a power ratio — they have an entirely
different error budget and are untouched by backscatter phase. Bounding an
ill-conditioned fit with an independent error channel is what buys robustness in the
regime where phase-only methods fail. This is the architecture of Huang et al. (2017),
and adopting it was the single most consequential decision in the design.

### 3.1 Gate classification

`radar_palette.gateid.gate_id` performs fuzzy classification of the dominant scatterer
into eleven classes (`no_scatter`, `clutter`, `biological`, `multi_trip`, three rain
intensities, `melting_wet`, `ice_snow`, `graupel_hail`). It takes a freezing level and,
preferentially, a gate temperature field. The retrieval solves over all meteorological
classes — phase accumulates through ice too, and continuity along the ray matters —
but applies the rain-conditioned self-consistency bound only in the liquid classes.

This step is load-bearing, not cosmetic. On the SGP X-SAPR volume it identified
**23.9%** of gates as clutter, against 0.4% at BNF. It also raised solved-ray coverage
from 328 to 361 of 361 at CACTI relative to a plain $\rho_{HV}$ threshold.

### 3.2 System differential phase

The system offset is estimated per ray, then aggregated as the median of the $N$
lowest ray-wise estimates ($N = 30$ by default). A survey of 360 case-library volumes
exposed a defect in that aggregation: *"the N lowest" is a linear ordering applied to
angles*. Where the system phase sits near a branch cut, the ray-wise population
straddles it, and selecting the numerically smallest values selects the wrapped
minority — returning an answer a whole period away from the truth. The survey found
this in 5 of 44 BNF volumes on the 0/360 branch.

`system.rebranch` fixes it by putting the ray-wise estimates on a single contiguous
branch before aggregating, and wrapping the result back into the input's branch. It is
provably a no-op away from a cut (asserted to machine precision in the test suite).

The error is circularly small — a few degrees — which is exactly why it survives
inspection. It is harmless for KDP, which differentiates any additive constant away,
and wrong for anything that uses absolute $\Phi_{DP}$.

Measured per-stream reference values, recorded as a QC baseline:

| stream | branch | system phase (deg) | ray-to-ray MAD (deg) |
|---|---|---|---|
| BNF C-SAPR2 1.7/1.8 | ±180 | −137.65 | 1.29 |
| BNF C-SAPR2 ≥1.10 | 0/360 | +9.75 | 2.00 |
| SGP C-SAPR | ±180 | −132.67 | 2.34 |
| KVNX WSR-88D | 0/360 | +26.97 | 2.88 |
| KHTX WSR-88D | 0/360 | +50.82 | 3.14 |
| KGWX WSR-88D | 0/360 | +44.52 | 3.40 |
| KLOT WSR-88D | 0/360 | +47.69 | 4.18 |

### 3.3 Fold detection

A fold is a shift of the phase level by *exactly one wrap period*, sustained. A
$\delta$ excursion or a noise spike is neither. Two conditions must both hold:

- **Persistence.** The shifted level must hold for `confirm_gates = 8` consecutive
  gates. A reverting spike is rejected.
- **Two-sided band.** The sustained change must land within `shift_tolerance = 0.25`
  of a whole wrap — a band around $\pm 360°$, not a one-sided threshold.

The band replaced a one-sided `jump_threshold` after the survey measured **1,099,263**
candidate steps exceeding 180° across the library and found **96.4% of them were
noise, not folds** — the one-sided form overcounted by roughly 28×.

The wrap interval is 360° for every stream measured, but the **branch cut differs by
stream and must be determined per file by measurement**: `valid_min`/`valid_max` are
absent or template-inherited and cannot be trusted. BNF `sapr2cfr` 1.7/1.8, SGP C-SAPR
and CACTI are ±180; BNF `sapr2cfr` ≥1.10, SGP X-SAPR and all WSR-88D are 0/360.

### 3.4 The bounds

Three ingredients, following Huang et al. (2017) Eqs. 14–16:

1. **Self-consistency ceiling and floor.** $K_{DP}/z_H$ is a function of $Z_{DR}$
   alone, expressed as a cubic $P(Z_{DR})$ with $z_H = 10^{0.1 Z_H}$ in mm⁶ m⁻³.
   Scaled by `lower_factor = 0.5` and `upper_factor = 2.0` this gives the central box.
2. **A phase-only least-squares floor.** A long-window least-squares slope of the
   measured phase is robust to $\delta$, because a backscatter excursion is a local
   perturbation a long fit averages away. It raises the *lower* bound only, and it
   uses $\Phi_{DP}$ alone, so it is independent of the self-consistency relation.
3. **A reflectivity cap.** A hard ceiling of 8 deg/km below 35 dBZ and 10 deg/km below
   45 dBZ, with no cap above — read from the accepted manuscript, not inferred.

The coefficients were **computed for this package** rather than adopted from a
published fit: rustmatrix 2.2.0, water at 10 °C, Brandes et al. (2002) Eq. (2) drop
shapes (inverted to $h/v$ — the published relation returns $v/h$), a 64-point
scattering table to $D_{max} = 8$ mm, normalized gamma PSD over $D_0 = 0.50$–3.50 mm,
$\mu \in \{0,3,5,8\}$, $N_w \in \{10^3, 8{\times}10^3, 10^5\}$, fitted over
$Z_{DR} = 0.2$–4.0 dB.

| band | $a_1$ | $a_2$ | $a_3$ | $a_4$ |
|---|---|---|---|---|
| S | 3.3383 | −1.7454 | 0.4638 | −0.04742 |
| C | 6.9471 | −2.8633 | 0.7745 | −0.09336 |
| X | 11.5058 | −3.0208 | −0.3671 | 0.12287 |

Two properties of that computation are worth carrying. $K_{DP}/z_H$ moved by
$2\times10^{-13}$ relative across two decades of $N_w$ — numerically exact
independence, as it must be. And against Gourley et al. (2009) Table 4 — the same
functional form fitted to different shapes, temperature and $\mu$ — this fit runs
1.05× at $Z_{DR} = 0.5$ dB rising to **1.34× at 3 dB** at C band. Two credible forward
models disagree by a third at high $Z_{DR}$. That disagreement is the direct
justification for generous tolerance factors, and, as Section 7 shows, it is not
generous enough.

Two departures from Huang:

- **The box is released outside rain.** The relation is derived for oblate liquid
  drops. In a hail core or above the melting layer it is a rain relation used out of
  scope — Huang's own assessment is that self-consistency methods fail "in critical
  situations such as hail cores". `kdp_bounds` takes a `rain_mask` and relaxes to a
  permissive symmetric interval outside it.
- **The sign policy is altitude-conditioned, not global.** Huang is rain-only and
  enforces $K_{DP}^{L} \geq 0$ everywhere. That is correct for its scope and wrong as
  a general prior: above the melting layer in an electrically active storm, vertically
  aligned ice makes **negative KDP the signal** (Hubbert et al. 2014). A global
  positivity bound erases exactly what such a study is looking for. The default
  enforces non-negativity below the freezing level, blends over 0.5 km, and releases
  to a floor of −2 deg/km aloft.

### 3.5 The linear program

Per ray, independently:

$$\min_x \|x - b\|_1 \qquad \text{subject to} \qquad
K_{DP}^{L} \;\leq\; Mx \;\leq\; K_{DP}^{U}$$

where $b$ is the measured phase, $x$ the fitted phase, and $M$ a Savitzky–Golay
first-derivative filter whose length follows `smoothing_km`. The $L_1$ norm is
linearised by the standard doubling trick (Portnoy and Koenker 1997), giving $2n$
inequality rows $z_i \geq \pm(x_i - b_i)$ and a cost vector summing $z$. Solved with
HiGHS via `scipy.optimize.linprog`.

For the 5-point case the derivative weights are $\{-0.2, -0.1, 0, 0.1, 0.2\}$
(Madden 1978), which is exactly what Giangrande et al. (2013) constrain.

The critical structural point, and the reason this is *not* the 2013 method:
**there is no curvature, total-variation or smoothness penalty in the objective.**
Giangrande et al. state explicitly that with the $L_1$ inequalities as the only
constraints the cost reduces to zero at $x = b$; all regularisation comes from the
constraint. In 2013 that constraint is one-sided monotonicity ($K_{DP} \geq 0$); here,
following Huang, it is a two-sided box. Self-consistency enters as a *bound*, never as
an objective term — it steers the solution rather than being fitted to.

### 3.6 Offset invariance and full window support

A KDP retrieval must be invariant to an additive constant on the input phase. Any
solver that is not is measuring the system offset.

That invariance is fragile in a bounded $L_1$ program, and the failure mode was
measured rather than reasoned about. A gate with **zero weight** leaves a free phase
variable, so every derivative window containing one has a KDP the objective never
sees. An LP optimum sits on a vertex, so the solver returns a *box edge* as if it were
a retrieval. On a 400-gate ray with an 11-gate filter and a box of $[-5, 20]$ deg/km,
a gap of only 5 gates moved reported KDP by **20 deg/km** under a constant input
shift, while a clean ray was invariant to $2.9\times10^{-13}$.

Requiring a data *majority* in the window does not fix it. Only requiring **full**
window support does, and that is the default: `min_window_weight_fraction = 1.0`.

The cost is coverage, and it is not small — 5% random dropout took reported gates from
390/400 to 254/400. This is the dominant practical limitation of the method.

---

## 4. Worked example

{{artifact:art_6ab18c70-cbc5-4ace-83e4-e00eec738e99}}

BNF C-SAPR2, 2025-04-06 06:00 UTC, 1.5° PPI, ray 353 (227.5° azimuth), 11-gate filter,
1.10 km realised smoothing, 833 of 1100 gates reported.

Panel (a) is the bounding channel. Panel (b) is the fitting channel — and it already
shows the defect: the fitted phase rises far more slowly than the measured phase,
recovering 1.003 deg/km of the measured 3.292 deg/km. Panel (c) shows why: the
retrieved KDP traces the upper edge of its own feasible box for the entire ray.

The grey bands are gates with no data support, where KDP is correctly withheld. The
oscillation of the fitted phase inside them is the free-variable degeneracy of
Section 3.6 in plain view — and the fact that no KDP is reported there is the
`min_window_weight_fraction = 1.0` guard working as designed.

---

## 5. What differs from the published methods

| | Giangrande et al. (2013) | Huang et al. (2017) | this implementation |
|---|---|---|---|
| objective | $\|x-b\|_1$ | $\|x-b\|_1$ | $\|x-b\|_1$ |
| constraint | one-sided $Mx \geq 0$ | two-sided box | two-sided box |
| self-consistency | lower bound, $aZ^b$ | Eq. 14 both sides | T-matrix cubic, computed here |
| coefficient source | fitted | Gourley-lineage | rustmatrix forward model |
| out-of-rain behaviour | same constraint | rain-only scope | box released |
| sign policy | global $\geq 0$ | global $\geq 0$ | altitude-conditioned |
| smoothing | coupled $s_L$ post-filter | 2 km derivative filter | 1.0 km, from measured DSD scale |
| gap handling | weights zeroed | not specified | full-window support required |

Two things worth flagging about the widely used Py-ART implementation of the 2013
method, established by reading its source against the paper (Helmus and Collis 2016):
it is faithful on the constraint filter and the self-consistency form, but the paper's
coupled smoothing stage — derive $s_L$ from $d_L$, smooth, *then* differentiate — is
absent, and the constraint filter is hardcoded at 5 points so `window_len` tunes only
the output stage. That matters because Giangrande et al. show non-negativity of an
$L$-point antisymmetric derivative filter does not prevent sub-filter oscillation, and
that *longer* filters permit more of it, not less; the coupled smoother is the fix. A
synthetic-ramp test confirmed Py-ART's derivative scaling is exactly unbiased at
window lengths 5, 11 and 35, so the deviation is structural rather than a scaling bug.

There is **no corrigendum or erratum** to the 2013 paper — checked across all 81
citers. The one other update in that lineage is Mishler et al. (2024), which restricts
LP processing to classified Rayleigh rain segments so $\delta$ is estimated rather
than absorbed.

---

## 6. What verifies

122 unit tests pass on the branch. On six BNF case-library volumes across three storm
modes and four dates, scored against ARM's shipped `specific_differential_phase` on
identical gates:

| test | this retrieval | ARM shipped |
|---|---|---|
| Z-conditional median monotonic | 6 / 6 | 5 / 6 |
| light rain (5–20 dBZ) above 1 deg/km | 0.000 on all 6 | 0.006 – 0.110 |
| corr($K_{DP}$, $Z$) in rain | 0.43 – 0.67 | 0.02 – 0.20 |
| negative KDP below 0 °C | ≤ 0.013% on all 6 | 9.4% – 35.7% |
| staircase fraction | 0.000 on all 6 | 0.006 – 0.016 |

Two tests returned **no verdict** and are reported as such. The 1.18 width-ratio test
was underpowered — 38 of 60 threshold rows floor-limited, because a PPI at 100 m gates
puts most supra-threshold runs at 2–3 gates. The decorrelation-length ratio was
unstable (0.056–1.265 across six cases), was not noise-corrected, and is not relied
on.

Section 7 is the reason none of the passes in that table should be read as validation.

---

## 7. The defect: the retrieval reports its own constraint

{{artifact:art_7f1a9ee6-56b8-4dad-96a1-d288de6bffb1}}

Measured across four research radars at their lowest tilt, counting rain gates with a
reported KDP and testing whether the solution sits within $10^{-6}$ deg/km of either
bound:

| platform | band | rain gates | at ceiling | at floor | **strictly interior** |
|---|---|---|---|---|---|
| SGP C-SAPR | C | 21,383 | 16.2% | 79.2% | **4.6%** |
| SGP X-SAPR | X | 104,535 | 59.0% | 38.2% | **2.8%** |
| CACTI C-SAPR2 | C | 90,955 | 60.9% | 31.4% | **7.7%** |
| BNF C-SAPR2 | C | 245,085 | 82.3% | 15.0% | **2.7%** |

At BNF the retrieved percentiles are indistinguishable from the ceiling's: p50 0.07
against 0.09, p90 0.62 against 0.63, p99 1.69 against 1.70 deg/km. The linear program
is not choosing a phase-consistent solution inside a permissive box; it is being
driven onto the box wall and reporting the wall.

The constraint itself is satisfied exactly — zero violations in 250,638 gates — so
this is not a solver bug. It is a parameterisation failure: the box is far too tight
relative to the phase information, so the phase term never gets to act.

**Proximate cause.** The ceiling is $z_H \times P(Z_{DR})$, and $Z_{DR}$ is low on
every platform: median in rain gates 0.59 dB at BNF (with the +0.84 dB vertical-pointing
offset already applied), 0.62 dB at SGP C-SAPR, 0.17 dB at CACTI, **0.00 dB** at SGP
X-SAPR. Real rain at these reflectivities should give appreciably more. Since $P$ is a
cubic in $Z_{DR}$, an under-calibrated $Z_{DR}$ collapses the ceiling non-linearly.
Only BNF has a measured offset at all; the other three inherit whatever calibration
error their $Z_{DR}$ carries. Note this is *not* a clean monotonic story — SGP C-SAPR
has the highest median $Z_{DR}$ and the *lowest* ceiling-pinning rate, because it is
pinned at the floor instead. What is universal is that ~92–97% of gates sit on *a*
bound.

**Contributing cause.** `upper_factor = 2.0` was chosen against a known 1.34× spread
between two credible forward models. Given a $Z_{DR}$ bias on top of that, it is not
wide enough to keep the box from binding.

---

## 8. Why the acceptance suite did not catch this

This is the part worth carrying forward, because it is a defect in the *tests*, not
just the code.

Every test in the Section 6 table is a test that a well-behaved function of $Z$ and
$Z_{DR}$ passes trivially:

- *monotonic with $Z$* — the ceiling is monotone in $z_H$ by construction
- *correlates with $Z$* — it **is** a function of $Z$; the 0.69 at BNF is not evidence
  of a good retrieval, it is evidence of the circularity
- *no staircase* — the bound is a smooth function of smooth inputs
- *no negatives below the melting level* — the sign floor guarantees it

A suite assembled to catch *oversmoothing* and *physically impossible amplitudes* is
structurally blind to a retrieval that has quietly stopped using its phase input. The
tests were the right tests for the previous failure and the wrong tests for this one.

**The missing test is cheap and is now written down**: report the fraction of gates
strictly interior to the feasible box. If that fraction is small, the bounds are doing
the retrieving and the phase is decorative. A companion check — compare the fitted
phase accumulation to the measured accumulation over the weighted span, which here
gives a ratio of 0.305 — catches the same thing from the other side.

Both belong in the acceptance suite before any validation campaign begins.

---

## 9. Known defects

1. **Bound saturation** (Section 7). Blocking. The retrieval is not currently
   phase-driven.
2. **Range-edge artifact.** Extreme KDP appears in a narrow window at the end of the
   weighted segment on a ray. On SGP C-SAPR, all 122 gates above 5 deg/km sit at gate
   indices 976–978 of 983 — 67% within five gates of the last valid gate — against a
   rain-gate p99 of 0.49 deg/km, a >30× artifact. Present weakly at BNF, absent at
   CACTI and SGP X-SAPR. Suspected cause is the boundary where gap-fill interpolation
   meets the mask edge, not the array edge, so the existing edge-gate NaN does not
   cover it. Needs an explicit trim at each end of every contiguous valid run.
3. **Coverage.** `min_window_weight_fraction = 1.0` is what makes offset invariance
   structural, and it costs gates: two of six acceptance volumes solved under half
   their rays.
4. **$Z_{DR}$ calibration is an unowned input.** The bounds are cubic in $Z_{DR}$ and
   only BNF has a measured offset. The method currently has no way to detect that its
   $Z_{DR}$ is biased, and no warning when the box collapses as a result.
5. **The X-band self-consistency fit is the poorest of the three**, with a sign change
   in $a_3$ relative to S and C and the largest residual.

---

## 10. What to do next

In order:

1. Add the bound-activity and phase-recovery diagnostics to the acceptance suite. No
   further tuning should happen before the suite can see this failure.
2. Settle $Z_{DR}$ calibration per platform. Goddard et al. (1994) and Gourley et al.
   (2009) give the machinery; the vertical-pointing method of Ryzhkov et al. (2005)
   worked at BNF and the curated scans elsewhere may support it.
3. Re-derive the tolerance factors from the *measured* forward-model spread rather
   than from a round number, and make them $Z_{DR}$-dependent — the two models agree
   at low $Z_{DR}$ and diverge above 1 dB, so a constant factor is the wrong shape.
4. Consider whether the ceiling should be a soft penalty rather than a hard
   constraint. A hard box guarantees the failure in Section 7 is silent; a penalty
   would let the phase overrule a wrong bound and leave a residual to inspect.
5. Fix the range-edge trim.
6. Re-run the 1.18 width-ratio test on RHIs, where it has power.

Validation against independent ground truth is deliberately *not* on this list yet.
Validating a field that is 97% its own constraint would measure the constraint.

---

## References

All DOIs below were resolved against the Crossref registry; none is retracted.

Beard, K. V., and C. Chuang, 1987: A new model for the equilibrium shape of raindrops.
*J. Atmos. Sci.*, **44**, 1509–1524. https://doi.org/10.1175/1520-0469(1987)044<1509:ANMFTE>2.0.CO;2

Brandes, E. A., G. Zhang, and J. Vivekanandan, 2002: Experiments in rainfall estimation
with a polarimetric radar in a subtropical environment. *J. Appl. Meteor.*, **41**,
674–685. https://doi.org/10.1175/1520-0450(2002)041<0674:EIREWA>2.0.CO;2

Giangrande, S. E., R. McGraw, and L. Lei, 2013: An application of linear programming to
polarimetric radar differential phase processing. *J. Atmos. Oceanic Technol.*, **30**,
1716–1729. https://doi.org/10.1175/JTECH-D-12-00147.1

Goddard, J. W. F., J. Tan, and M. Thurai, 1994: Technique for calibration of
meteorological radars using differential phase. *Electron. Lett.*, **30**, 166–167.
https://doi.org/10.1049/el:19940119

Gorgucci, E., G. Scarchilli, and V. Chandrasekar, 1999: Specific differential phase
estimation in the presence of nonuniform rainfall medium along the path. *J. Atmos.
Oceanic Technol.*, **16**, 1690–1697. https://doi.org/10.1175/1520-0426(1999)016<1690:SDPEIT>2.0.CO;2

Gourley, J. J., A. J. Illingworth, and P. Tabary, 2009: Absolute calibration of radar
reflectivity using redundancy of the polarization observations and implied constraints
on drop shapes. *J. Atmos. Oceanic Technol.*, **26**, 689–703.
https://doi.org/10.1175/2008JTECHA1152.1

Helmus, J. J., and S. M. Collis, 2016: The Python ARM Radar Toolkit (Py-ART). *J. Open
Res. Software*, **4**, e25. https://doi.org/10.5334/jors.119

Huang, H., G. Zhang, K. Zhao, and S. E. Giangrande, 2017: A hybrid method to estimate
specific differential phase and rainfall with linear programming and physics
constraints. *IEEE Trans. Geosci. Remote Sens.*, **55**, 96–111.
https://doi.org/10.1109/TGRS.2016.2596295 (accepted manuscript 2016; frequently cited
as Huang et al. 2016)

Hubbert, J. C., and V. N. Bringi, 1995: An iterative filtering technique for the
analysis of copolar differential phase and dual-frequency radar measurements. *J.
Atmos. Oceanic Technol.*, **12**, 643–648. https://doi.org/10.1175/1520-0426(1995)012<0643:AIFTFT>2.0.CO;2

Hubbert, J. C., S. M. Ellis, W.-Y. Chang, and co-authors, 2014: Modeling and
interpretation of S-band ice crystal depolarization signatures. *J. Appl. Meteor.
Climatol.*, **53**, 1659–1677. https://doi.org/10.1175/JAMC-D-13-0158.1

Illingworth, A. J., and T. M. Blackman, 2002: The need to represent raindrop size
spectra as normalized gamma distributions. *J. Appl. Meteor.*, **41**, 286–297.
https://doi.org/10.1175/1520-0450(2002)041<0286:TNTRRS>2.0.CO;2

Kennedy, P. C., and S. A. Rutledge, 2011: S-band dual-polarization radar observations
of winter storms. *J. Appl. Meteor. Climatol.*, **50**, 844–858.
https://doi.org/10.1175/2010JAMC2558.1

Kumjian, M. R., 2013: Principles and applications of dual-polarization weather radar,
Part I. *J. Operational Meteor.*, **1**, 226–242.
https://doi.org/10.15191/nwajom.2013.0119

Kuster, C. M., B. R. Bowers, J. C. Carlin, and co-authors, 2021: Using $K_{DP}$ cores
as a downburst precursor signature. *Wea. Forecasting*, **36**, 1183–1198.
https://doi.org/10.1175/WAF-D-21-0005.1

Leinonen, J., D. Moisseev, M. Leskinen, and W. A. Petersen, 2012: A climatology of
disdrometer measurements of rainfall in Finland over five years. *J. Appl. Meteor.
Climatol.*, **51**, 392–404. https://doi.org/10.1175/JAMC-D-11-056.1

Madden, H. H., 1978: Comments on the Savitzky–Golay convolution method for
least-squares fit smoothing and differentiation of digital data. *Anal. Chem.*, **50**,
1383–1386. https://doi.org/10.1021/ac50031a048

Mishler, M., G. Zhang, and V. Mahale, 2024: Improving estimation of specific
differential phase. *J. Appl. Meteor. Climatol.*, **63**, 415–430.
https://doi.org/10.1175/JAMC-D-23-0204.1

Portnoy, S., and R. Koenker, 1997: The Gaussian hare and the Laplacian tortoise. *Stat.
Sci.*, **12**, 279–300. https://doi.org/10.1214/ss/1030037960

Reinoso-Rondinel, R., C. Unal, and H. Russchenberg, 2018: Adaptive and high-resolution
estimation of specific differential phase for polarimetric X-band weather radars. *J.
Atmos. Oceanic Technol.*, **35**, 555–573. https://doi.org/10.1175/JTECH-D-17-0105.1

Ryzhkov, A. V., and D. S. Zrnić, 1998: Beamwidth effects on the differential phase
measurements of rain. *J. Atmos. Oceanic Technol.*, **15**, 624–634.
https://doi.org/10.1175/1520-0426(1998)015<0624:BEOTDP>2.0.CO;2

Ryzhkov, A. V., S. E. Giangrande, V. M. Melnikov, and T. J. Schuur, 2005: Calibration
issues of dual-polarization radar measurements. *J. Atmos. Oceanic Technol.*, **22**,
1138–1155. https://doi.org/10.1175/JTECH1772.1

Sachidananda, M., and D. S. Zrnić, 1986: Differential propagation phase shift and
rainfall rate estimation. *Radio Sci.*, **21**, 235–247.
https://doi.org/10.1029/RS021i002p00235

Schneebeli, M., and A. Berne, 2012: An extended Kalman filter framework for
polarimetric X-band weather radar data processing. *J. Atmos. Oceanic Technol.*, **29**,
711–730. https://doi.org/10.1175/JTECH-D-10-05053.1

Thurai, M., and V. N. Bringi, 2005: Drop axis ratios from a 2D video disdrometer. *J.
Atmos. Oceanic Technol.*, **22**, 966–978. https://doi.org/10.1175/JTECH1767.1

Trömel, S., M. R. Kumjian, A. V. Ryzhkov, and co-authors, 2013: Backscatter
differential phase — estimation and variability. *J. Appl. Meteor. Climatol.*, **52**,
2529–2548. https://doi.org/10.1175/JAMC-D-13-0124.1

Vulpiani, G., M. Montopoli, L. D. Passeri, and co-authors, 2012: On the use of
dual-polarized C-band radar for operational rainfall retrieval in mountainous areas.
*J. Appl. Meteor. Climatol.*, **51**, 405–425. https://doi.org/10.1175/JAMC-D-10-05024.1

Wang, Y., and V. Chandrasekar, 2009: Algorithm for estimation of the specific
differential phase. *J. Atmos. Oceanic Technol.*, **26**, 2565–2578.
https://doi.org/10.1175/2009JTECHA1358.1

Zrnić, D. S., and A. V. Ryzhkov, 1996: Advantages of rain measurements using specific
differential phase. *J. Atmos. Oceanic Technol.*, **13**, 454–464.
https://doi.org/10.1175/1520-0426(1996)013<0454:AORMUS>2.0.CO;2

Aldana, M., S. Pulkkinen, A. von Lerber, and co-authors, 2025: Benchmarking $K_{DP}$ in
rainfall. *Atmos. Meas. Tech.*, **18**, 793–816.
https://doi.org/10.5194/amt-18-793-2025
