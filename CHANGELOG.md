# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions come from
git tags via `setuptools-scm`.

## [Unreleased]

### Fixed

- **Split-cut selection was not reproducible across processes.** `census_sweep` fell
  back to `next(iter(radar.fields))` when no `valid_field` was named, and Py-ART builds
  its field dict from a set, so under hash randomisation the first key differs between
  runs of the same file. On a NEXRAD split cut the choice decides everything:
  `reflectivity` reaches ~460 km while `velocity` stops at ~2 km, so
  `range_max_valid_m` flipped and `dedup_sweeps` kept the Doppler member instead of
  the surveillance one. Measured on a KLOT VCP-212 volume with six SAILS repeats at
  0.48 deg, the gridded peak alternated between 67.6 and 86.9 dBZ across identical
  runs -- 9 of 20 fresh processes took the wrong sweep. Now resolved through
  `VALID_FIELD_PREFERENCE`, a reflectivity-first list with a sorted fallback, so the
  answer never depends on dict ordering. Nine consecutive fresh processes now agree
  bit-for-bit where before they did not.

  This did not affect the C-SAPR2 results in this changelog: that scan strategy has no
  split cuts, so every elevation has exactly one member and the group has nothing to
  choose between.

### Changed

- **The spectral azimuth path now defaults to `az_engine='dense'`, `az_solver='direct'`.**
  This changes numbers for existing callers, and it is a safety decision rather than a
  performance one: the numbers it changes were partly wrong. Measured per cone on a
  15-tilt ARM C-SAPR2 volume, the old `scipy`/`cg` default at `n_cg=12` had one tilt of
  fifteen overshooting its input range by 107 dB and reaching 169 dBZ, and across a
  resolution ramp the whole-volume range reached −299…297 dBZ. The new default holds
  −99…66 dBZ on the same grids, with the other fourteen tilts within 1 dB.
  - `n_cg` was acting as regularisation, not as a convergence budget, so a caller
    raising it to improve the answer made it catastrophically worse (25 iterations:
    −1907…2471 dBZ). The direct solver takes an explicit ridge instead.
  - `dense` needs no dependency beyond numpy/scipy: on the azimuth axis the DFT matrix
    is `n_rays × n_modes` and both are ray counts, so forming it exactly costs 8 MB at
    720 rays and removes the Kaiser-Bessel kernel error.
  - `az_engine='scipy', az_solver='cg'` reproduces the previous behaviour exactly,
    `cg_resid_*` diagnostics included.

### Added

- `EngineConcurrencyWarning` when `az_engine='finufft'` is combined with `n_jobs>1`.
  finufft's results there are exact — bit-identical to `dense` on every grid measured
  — but its timing is not reproducible. Measured across eight sweep shapes, three
  repeats each:
  - **Serially it is indistinguishable from `dense`** (medians 0.97–1.00 of `dense`,
    1.00–1.03x spread). An earlier note in this branch claiming it was ~1.9x slower
    was comparing serial finufft against *parallel* `dense`; that was wrong.
  - Under `n_jobs=4` it is bimodal: on six of eight shapes it is the **fastest CPU
    configuration measured** (~8.3 s against `dense`'s ~10.1 s on a 15-tilt volume),
    and on the rest the same call takes 12–19x longer. Best and worst single runs
    across the study were 7.9 s and 177.8 s at fixed workload, against `dense`'s
    10.0–11.9 s over the same 24 runs.
  - Because it varies run to run at fixed workload, and not only on the first run,
    this is a race in finufft's own global planner state, not warm-up or a workload
    effect. Warned rather than serialised or removed: it may well be the fastest
    choice, the numbers are right either way, and quietly forcing `n_jobs=1` for one
    engine would make an engine comparison unfair.

- Measured the `torch` engine's GPU path, previously documented but never run.
  On an NVIDIA A10 (via Modal), per-tilt azimuth solves took 93.3 ms in float64 and
  91.3 ms in float32 against 83.9 ms on the same machine's CPU — **slower on 15 of 15
  tilts in both precisions**. A 931 x 1050 solve is a few tens of milliseconds of
  arithmetic, too small to amortise host-to-device transfer and kernel launch; this is
  the same size argument that makes the dependency-free `dense` engine competitive on
  this axis. Accuracy on the device is sound if anyone wants it regardless: float64
  matches CPU to 1e-15, and float32 is only 1.5x worse (3.3e-4), because the
  Kaiser-Bessel kernel error dominates round-off either way. `device='cuda'`/`'mps'`
  remain available and are now documented as measured-and-slower rather than untested.

- `az_engine='torch'` accepts a `dtype` argument. It was hardcoded to `complex128`,
  so the single-precision path a GPU is actually fast at could not be expressed.

- `UnphysicalGridWarning` flags gridded fields that leave the range reflectivity can
  physically take. Measured on the ARM BNF C-SAPR2 volume, settings reachable
  through the public API and what they produce:

  | setting | grid range (dBZ) | cells beyond ±80 |
  | --- | --- | --- |
  | default, `n_cg=12` | −61 … 66 | 0 (but one cone reaches 169) |
  | `n_cg=25` | **−1907 … 2471** | 1294 |
  | `n_cg=200` | **−1907 … 2471** | 1294 |
  | `az_ridge=1e-10` | **−1903 … 2466** | 1292 |
  | `az_ridge=1e-6` | −98 … 138 | 26 |
  | `az_ridge=1e-2` | −61 … 58 | 0 |
  | `az_ridge='auto'` | −61 … 59 | 0 |

  - **`n_cg` is the trap.** It is a long-standing documented argument and raising it
    looks like asking for a better answer, but the iteration count is acting as
    regularisation: past roughly a dozen iterations the solve amplifies out-of-band
    content instead of converging. Its docstring now says so.
  - **The default is not exempt.** One cone of fifteen reaches 169 dBZ at `n_cg=12`,
    masked in the final grid only because the vertical interpolation drops it.
    `az_solver='direct'` with `az_ridge=1e-2` removes it and leaves every other tilt
    within 1 dB.
  - Warns rather than raising or clipping: raising would abort a volume for one bad
    tilt of fifteen, and clipping would hide a settings problem behind a
    plausible-looking field. Its own category, so a pipeline can
    `filterwarnings("error", category=UnphysicalGridWarning)`.
  - A separate, looser check catches large overshoot *inside* the physical range.
    The threshold (25 dB) sits well above the few dB of Gibbs ringing a band-limited
    interpolant is expected to produce at a sharp edge — 5.5 dB on this operator's
    documented example, 13 dB at the steepest tilt of the BNF volume — so it does
    not fire on every convective case.
- `grid_volume(..., method="spectral")` now accepts `cg_tol` and `field_units`, which
  were reachable on `SweepSpectralEvaluator` but not through the public entry point.

- `build_cones(..., n_jobs=...)` and `grid_volume(..., method="spectral", n_jobs=...)`
  grid tilts concurrently. Tilts are independent, so this is the one place in the
  spectral path where hardware helps without changing the algorithm, and on a
  15-tilt volume the serial loop left most of a 14-core machine idle.
  - **Threads, not processes.** The per-tilt cost is NumPy/SciPy linear algebra and
    FFTs, which release the GIL; a process pool would pickle the radar object to
    every worker, hundreds of megabytes for a research volume.
  - **Two resources bind, and `resolve_tilt_workers` returns both.** Memory caps
    the worker count (one in-flight tilt reaches ~1 GB on a 171 km domain at
    250 m, so `n_jobs=-1` resolves to 14 workers at 2 km and 8 at 250 m); the core
    budget caps the BLAS/FFT threads each worker may use. Missing the second is
    worse than staying serial: NumPy and SciPy each start a pool sized to the whole
    machine, so 14 workers × ~14 threads drove the load average past 500 on a
    14-core machine and had to be killed. `workers * blas_threads` is now bounded by
    the core count, applied with `threadpoolctl` inside the pool.
  - Measured on a 15-tilt volume, 60 km domain at 1 km, dense engine and direct
    solver: **3.1x at `n_jobs=8`** (17.5 s serial → 5.7 s). Task parallelism beats
    BLAS threading — the fastest setting gives each worker a *single* thread,
    because the azimuth solve is a few hundred rows and too small for a wide GEMM
    to pay for its synchronisation.
  - `threadpoolctl` is a new optional dependency (`pip install radar-palette[parallel]`).
    The serial default never touches the native pools, so it is not required;
    without it the parallel path is still correct but over-subscribes, and warns.
  - Tests assert the parallel and serial grids are *bit-identical* (any difference
    would be a race, not a tolerable reordering) and that the cone stack stays in
    ascending-elevation order regardless of completion order.
- `resolve_tilt_workers` is exported from `radar_palette.gridding`.

- `radar_palette.gridding.nufft_engines`: interchangeable backends for the azimuth
  NUFFT, and interchangeable solvers on top of them. `nufft.py` is unchanged and
  remains the reference every engine is measured against; `make_operator(...,
  engine=...)` selects who computes the transforms and `solve(..., solver=...)`
  how the normal equations are solved. The evaluator exposes both as `az_engine`
  and `az_solver`, and records which pair ran in its report.
  - Engines: `reference` (`nufft.py` itself), `scipy` (default), `dense`,
    `finufft`, `ducc0`, `torch`. Availability is checked at construction, never at
    import, so a minimal install keeps working and asking for an absent engine
    raises a message naming the extra to install rather than an `ImportError`.
  - The default engine needs **no new dependency** and is the reference algorithm
    with two exact substitutions: the Kaiser-Bessel spreading stencil becomes one
    CSR matrix (replacing `np.add.at`, an unbuffered scatter that was 38% of solve
    time) and the spread-to-spectrum transform becomes `rfft` (the spread field is
    real, so the reference computed a conjugate-symmetric half it then discarded).
    Both are arithmetic-preserving: agreement with `nufft.py` is 1.5e-15. Measured
    speedup on the CG path is 1.9-2.3x for sweeps of 360 rays and up.
  - The `dense` engine has no interpolation kernel at all — it forms the DFT
    matrix. That is affordable here, and only here, because on the azimuth axis
    the matrix is `n_rays x n_modes` and both are ray counts (8 MB at 720 rays).
    It is therefore exact to round-off, needs no dependency, and is the fastest
    engine measured on 360-720 ray sweeps (32-44x with the direct solver). Which
    engine leads is *not* monotone in ray count: `finufft` is faster at both ends
    of the tested range — at 1440 rays for the `O(n log n)` versus `O(n^2)`
    reason, and at 120 rays because the problem is too small for the dense
    per-solve BLAS work to amortise the matrix it built. Operational 1° and 0.5°
    volumes sit in the band where `dense` wins; measure rather than extrapolate.
  - `solver='direct'` factorises the normal equations instead of iterating on
    them, reaching the least-squares solution CG approaches (verified against
    CG run to 1e-14). It is 10-30x faster than 12 CG iterations, against about 2x
    for the fastest engine change — the solver is the larger of the two effects.
  - **The accuracy gain belongs to the engine/solver pair, not to the solver.**
    On a field exactly band-limited on the lattice, so the correct answer is known
    in closed form: `direct` on `dense` or `finufft` recovers it to ~1e-10 against
    ~3e-5 for 12 CG iterations, but `direct` on a Kaiser-Bessel engine converges
    to that engine's own ~3e-4 kernel error and gains nothing at low jitter. A
    test asserts each half of that, because a reader who took the headline figure
    for a property of `solver='direct'` alone would be wrong.
  - **That accuracy gain does not transfer to real reflectivity, and the module
    now says so in a warning.** Validated by held-out ray prediction on three
    storm-filled sweeps (SPOL S-band convective line, SWX C-band widespread
    precipitation, CSAPR2 C-band convective): at its best ridge the direct solve
    is *comparable* to 12 CG iterations — 6.70 dB against 7.80, 4.74 against
    4.92, 8.52 against 8.47 — not orders better. The synthetic figure was a
    statement about representing a field this operator can represent exactly;
    real reflectivity is not such a field. On real data the reasons to choose the
    direct solver are speed and having no iteration count to tune.
  - **`ridge='auto'` is an improvement on the old default, not the per-sweep
    optimum, and on a well-conditioned sweep it does nothing at all.** Validated
    further on a 15-tilt ARM BNF C-SAPR2 volume: at 4.5-42° the normal matrix is
    well behaved (condition number 15-32) so the condition-number term never
    binds and `auto` returns the floor — yet the held-out optimum on echo gates
    is 1e-3..1e-2 at *every* tilt. Conditioning is a correlate of the quantity
    that matters (out-of-band energy), not that quantity, and on a clean geometry
    the correlation fails in the unsafe direction. Two fixes were tried and both
    reverted, with the reasoning recorded at `MIN_RIDGE`: raising the floor to
    1e-2 wins 0.12 dB on real data but gives up eight orders of magnitude on a
    band-limited field, and generalised cross-validation recovers the synthetic
    case exactly then fails on real sweeps because it scores on training residual
    — which is what overfitting minimises (it returns the 1e-10 floor on four of
    five BNF tilts whose measured optima are 1e-3..1e-2). **For real reflectivity pass
    `ridge=1e-2` explicitly** (that is what `DEFAULT_RIDGE` is); `--real`
    measures it for a given radar.
  - **Stratify by echo before reading any dB figure.** Over half the gates in a
    real sweep sit below 0 dBZ, so an unstratified median is dominated by noise
    and by the gap-fill floor — and it ranks the solvers differently. On BNF
    sweep 11 a ridge of 1e-1 scores best on all gates (4.14 dB) and worst on
    gates above 10 dBZ (8.09 dB). `--real` now reports both columns.
  - Across eight real sweeps (three single-sweep files plus five BNF tilts),
    graded on echo gates with a per-sweep oracle ridge, the direct solve beats
    12 CG iterations on 6 of 8 by a mean of 0.13 dB — a wash, consistent with the
    single-sweep result and not the orders-of-magnitude the synthetic figures
    suggest.
  - `evaluate_lattice(lattice_values, azimuths_deg)` evaluates the recovered
    interpolant at arbitrary azimuths. Needed because `forward` only evaluates at
    the *measured* rays, which cannot distinguish fitting from overfitting; it is
    a module function rather than an engine method because the interpolant
    belongs to the lattice, not to whichever engine recovered it. The `reference`
    engine also gains the `mode_low` attribute every other engine already had —
    without it that engine alone failed on this path.
  - What `direct` improves on *every* engine is degraded sampling, where what
    fails is the conditioning rather than the kernel: at ±0.45-spacing jitter,
    12 iterations reach 1.7e-2 against 4.5e-4 for the direct solve on the same
    transforms. Its `ridge` is likewise not cosmetic — a sector's normal matrix is
    singular by construction (a 30-120° sector at 1° spacing fits 91 rays onto a
    366-point lattice) and a bare Cholesky fails on it.
  - **`ridge` defaults to `'auto'`, derived from the eigenvalue spectrum**, because
    no constant serves both regimes: a synthetic band-limited sweep wants the
    1e-10 floor and every real sweep measured wants 1e-3 or larger, and 1e-10 is
    the *worst* value tested on all three real sweeps. Note the driver is not rank
    deficiency — two of the three are full-rank and still want a large ridge — but
    that real reflectivity is not band-limited: speckle, clutter and echo edges
    put energy in modes the sweep cannot determine, a small ridge fits that energy
    at the measured rays, and the interpolant between them is noise. A
    parametrised test adds broadband noise to a synthetic field and shows the
    optimum leave the floor, isolating band-limitedness as the variable.
  - The report gains `normal_null_dim`, the number of undetermined modes in the
    geometry, since a caller cannot tell an under-determined answer from a
    converged one by looking at the values. Real sweeps are commonly
    rank-deficient: operational azimuth sampling leaves gaps of ~2× the nominal
    spacing (measured: 3 undetermined modes of 414 on SPOL, 23 of 993 on a GUC
    X-band sweep).
  - `benchmarks/bench_nufft_engines.py --real` reproduces the held-out ray
    validation on real sweeps, downloading them via `open-radar-data`.
  - `benchmarks/bench_nufft_engines.py` regenerates every figure quoted above.
    Timings are hardware-dependent and the quoted ones are from an arm64 laptop.

### Changed

- The evaluator's `NON_UNIFORM` path now runs on the `scipy` engine rather than
  `nufft.py` directly, which is ~2x faster and agrees to round-off (1e-13 measured
  end-to-end on the evaluated field, asserted by a test). `az_engine='reference'`
  restores the original code path exactly. Note that the other engines are *not*
  round-off equivalent: `dense`, `finufft`, `ducc0` and `torch` each shift the
  answer by ~5e-4, since they replace the reference's Kaiser-Bessel kernel (whose
  own error is 2.5e-4) with a different or exact transform. The shift is towards
  exact arithmetic, but it is a shift.
- The NUFFT path's report `extras` now include `engine` and `solver`, and the
  kernel-parameter keys are supplied by the engine. `finufft`/`ducc0` report `eps`
  in place of `kb_beta`/`M_oversampled`, since a kernel width is not what sets
  their accuracy. `adjoint_test_rel` is measured per engine, not inherited.
- `az_solver` is validated in the constructor rather than where it is consumed.
  Only one of the three azimuth paths reads it, so a typo would otherwise pass
  silently on a uniform sweep and fail later on a jittered one.

- The `gridding` module docstring no longer claims the distance-weighted path "degrades
  gracefully where sampling is poor". It does not: the radius of influence is a fixed
  length while the separation between adjacent tilts grows with range, so beyond a certain
  range no gate is in reach and columns develop interior holes. Py-ART's own `dist_beam`
  defaults leave 570 of them in one vertical section of a C-SAPR2 volume. The docstring now
  carries the measured coverage-versus-peak trade-off and points at `roi_func`, and
  `grid_volume` documents that argument as the consequential one for the `"pyart"` backing.
  Five tests in `TestDistanceWeightingLeavesInterConeGaps` pin the behaviour, including that
  the Py-ART defaults reproduce it and that closing the gaps costs peak intensity.

### Added

- `advection_interpolate` gains `carry_fields`, warping several variables through a
  single motion estimate. The flow must come from a tracer, and the polarimetric
  variables are not tracers: differential reflectivity is a shape measure, copolar
  correlation is closer to a quality flag, and Doppler velocity is signed and folded
  at the Nyquist interval. One call now reconstructs reflectivity, velocity, ZDR and
  RhoHV from one flow estimate and one gridding pass, each field keeping its own
  metadata so units are not overwritten with the tracer's. `interp_field_name`
  renames only the tracer. Folded velocity carries a documented caveat: the blend is
  a weighted mean, so it flattens gates that cross a fold — 7.1% of echo gates above
  15 dBZ on the ARM C-SAPR2 pair used to exercise this, against 20.4% of all valid
  gates, the rest being noise. Dealias first if the interpolated velocity is to be
  read quantitatively.

- `grid_volume` gains `method={"pyart", "spectral"}`, wiring the spectral operator
  behind the public entry point. The FFT gridding capability is now complete end to
  end: sweep census, per-sweep spectral evaluation on its cone, and vertical
  assembly, reachable in one call from either object family.
- The spectral path emits a `coverage_flag` field alongside the data, carrying
  `VerticalFlag` per cell. A radar volume does not observe a box, and an unexplained
  NaN leaves a user unable to tell a data gap from a geometric coverage limit.

- `radar_palette.gridding.cones`: `cone_range_height`, `target_lattice`,
  `dedup_sweeps`, `build_cones`, `beam_footprint_crossover`, plus `ConeStack` and
  `TiltReport`. A sweep samples a curved cone, not a plane, so each tilt is gridded
  onto a shared horizontal lattice carrying its own height surface. `dedup_sweeps`
  collapses split cuts by measured range rather than by cut naming, so it needs no
  knowledge of any radar's scan-strategy vocabulary.
- `radar_palette.gridding.vertical`: `interp_column_stack`, `target_elevation`,
  `intercone_gap`, `VerticalFlag`, `VerticalProfile` and `DEFAULT_VERTICAL_SCHEME`.
  Interpolates a cone stack onto target heights per column, with **no
  extrapolation**: every finite value carries `VerticalFlag.INTERPOLATED` and every
  other cell is NaN with a flag saying why. `INTERIOR_GAP` covers the case that is
  easy to miss — maximum valid range is a property of the echo, not the geometry, so
  it can be non-monotonic in elevation and leave a hole in a column's tilt run that
  must not be interpolated across.

- `radar_palette.advection.flow`: `grid_optical_flow` estimates a dense,
  height-resolved echo motion field between two gridded volumes (TV-L1 optical flow
  per vertical level). Returns displacement in metres in the earlier-to-later sense,
  `(north, east)` order.
- `radar_palette.advection.morph`: `advection_interpolate` reconstructs a volume at
  a time between two observed volumes, advecting each bracketing volume to the target
  time along that field and blending. The morph runs per gate in native
  (azimuth, slant range, elevation) coordinates, so the output is an ordinary volume
  rather than a Cartesian grid; only the motion estimation goes through a grid.
  Accepts and returns both object flavours, and applies the acquisition-time
  correction from `radar_palette.advection.timing` at the source, so a reconstructed
  volume never inherits a bracketing volume's clock.

- `radar_palette.gridding.evaluator`: `SweepSpectralEvaluator` turns a sampled sweep
  into a continuous band-limited function, evaluable at arbitrary
  (range, azimuth). Three azimuth paths chosen by sweep class — exact FFT for a
  closing full-circle lattice, gap-closing extension for a sector, Kaiser-Bessel
  NUFFT for non-uniform rays. Range is mirror-extended, never treated as periodic.
- `radar_palette.gridding.nufft`: Kaiser-Bessel forward/adjoint operator pair with
  an adjoint-consistency test recorded in the evaluator's report, plus conjugate-
  gradient solution of the density-weighted normal equations.
- `radar_palette.gridding.reflectivity`: `to_dbz`, `to_linear`,
  `looks_like_linear_reflectivity` and `LinearReflectivityError`. The evaluator
  refuses input that looks like linear Z, because interpolating linear
  reflectivity across a hard echo edge drove 42.6% of evaluated samples negative in
  a measured reproduction.
- `radar_palette.gridding.fill`: `fill_sweep` makes a sweep gap-free by constant,
  edge-interpolating or band-limited (Gerchberg-Papoulis) fill. All three leave
  measured samples untouched.
- `radar_palette.gridding.geometry`: `antenna_to_cartesian_43` and
  `cartesian_to_antenna_43`, forward and exact analytic inverse radar geometry on
  the 4/3 effective earth, in metres. The inverse exists because
  `pyart.core.cartesian_to_antenna` is not the inverse of Py-ART's own forward
  transform; see below.
- `radar_palette.gridding.census`: `census_sweep`, `census_radar`, `SweepGeometry`
  and `SweepClass` measure a volume's sampling geometry and classify each sweep as
  `EXACT_UNIFORM_PERIODIC`, `UNIFORM_PARTIAL_SECTOR` or `NON_UNIFORM`, which
  determines whether a plain FFT is valid on the azimuth axis. Also tags split-cut
  groups, since NEXRAD VCPs repeat the low tilts and de-duplication is mandatory
  before vertical interpolation. Accepts either object flavour.
- Classifier tolerances are public and documented: `AZ_DEV_TOL_FRAC` (0.01),
  `DR_DEV_TOL_FRAC` (1e-3), `SPLIT_CUT_TOL_DEG` (0.05). The azimuth tolerance is
  fractional rather than absolute because the worst-case Nyquist error is `pi * f`,
  independent of ray count.

- `radar_palette.advection.timing`: reconstruction of per-ray acquisition times
  for advection-interpolated volumes (`volume_reference_time`,
  `interpolate_ray_times`, `apply_interpolated_time`). An interpolated volume now
  carries the time it represents rather than inheriting the earlier bracketing
  volume's clock. Intra-volume ray timing structure is preserved, since ray-to-ray
  spacing is set by the scan strategy and not by the interpolation.
- `radar_palette.testing`: `make_empty_ppi_volume` and `assign_scan_times` build
  bare PPI geometries with explicitly controlled acquisition times, so timing
  behaviour can be tested independently of any field data.
- **Python floor raised to 3.11** (`requires-python = ">=3.11"`). `xradar` 0.12.0 — the release whose cfradial readers this
  package depends on — requires 3.11, and Py-ART made the same move (`arm_pyart`
  2.2.1 dropped 3.10). A 3.10 environment could only resolve `arm_pyart` 2.2.0 /
  `xradar` 0.11.1, an older stack than the one tested here. The CI matrix entry for
  3.10 is removed in a companion change to `.github/workflows/ci.yml`, which a
  personal access token cannot push (GitHub gates workflow files on the classic
  `workflow` scope); until it lands the 3.10 job fails at install, correctly, since
  pip refuses a package whose `requires-python` excludes the interpreter.
- `xarray`, `xradar` and `netCDF4` are now declared dependencies. They were
  previously satisfied transitively through `arm_pyart` while being imported
  directly, which works only as long as a resolver happens to choose that way.
- Packaging tests assert that every third-party module the library imports is a
  declared dependency, and that the declared `xarray` floor actually provides
  `xarray.DataTree`.
- `radar_palette.io`: object-flavour interoperability between Py-ART
  (`Radar`/`Grid`) and xradar (`DataTree`/xarray `Dataset`). Conversions delegate
  to Py-ART's own layer (`pyart.xradar`, `Grid.to_xarray`, xradar's cfradial
  readers) rather than reimplementing the data-model mapping.
- `radar_palette.advection.retime_interpolated_volume` and
  `radar_palette.gridding.grid_volume`: flavour-aware entry points. Both accept
  either object family; by default the advection entry point returns the family it
  was given, and the gridding entry point maps Py-ART input to a `Grid` and xradar
  input to an xarray `Dataset`. `output_flavor` overrides either default.

### Changed

- `DEFAULT_GRIDDING_METHOD` is `"pyart"`, so **existing callers of `grid_volume` get
  byte-identical results** — asserted by a test comparing the default against a
  direct `pyart.map.grid_from_radars` call. The spectral operator is opt-in because
  the two are different rather than ranked: on a hard azimuthal echo edge (10 to
  50 dBZ wedge) the spectral interpolant reaches 55.5 and 4.7 dBZ (+5.5 / −5.3 dB
  beyond the data) while distance weighting stays inside the input range, since an
  average cannot exceed its inputs. Where both produce data on a smooth field they
  agree to 0.35 dB median. A test ties the constant to that measurement, so flipping
  the default fails with a reason rather than silently.

- Vertical interpolation defaults to `pchip_z` (monotone), which measured best of the
  four schemes and cannot introduce an extremum between two cones. The margin is
  small and that is the point: **tilt spacing, not scheme choice, is what limits
  vertical accuracy.** On a bright band at 2500 m with a 400 m standard deviation,
  going from 6 tilts to 23 (median inter-cone thickness 680 m to 213 m) improved RMSE
  from 4.086 dB to 0.295 dB — a factor of 14 — while the best-vs-worst scheme spread
  at 6 tilts was only 4.024 to 5.194 dB. Within one scan strategy the error tracks the
  local layer thickness: 0.42 dB median where adjacent cones are 400–700 m apart,
  3.81 dB where they are 1200–2500 m apart. Read a gridded value alongside its
  `gap_m`.

- `test_third_party_imports_are_declared_dependencies` now accepts a package declared
  in *either* `project.dependencies` or an extra. Previously it failed on any
  correctly-gated optional import, which would have pressured a contributor into
  declaring `scikit-image` as required or deleting the guard. A companion test,
  `test_optional_imports_are_not_required_dependencies`, closes the loophole from the
  other side by asserting `scikit-image` and `finufft` stay optional. Both were
  verified non-vacuous by deleting a declaration and confirming each fires.

- **Conjugate-gradient refinement of the NUFFT path is now on by default**
  (`DEFAULT_CG_ITERATIONS = 12`), departing from the research code it was ported
  from. Convolution gridding computes the adjoint of the sampling operator, not its
  inverse: on a constant field with ±0.3-spacing azimuth jitter, unrefined gridding
  reproduces the constant to only 60% relative deviation, scaling with jitter (3.6%
  at ±0.02, 19% at ±0.1). Twelve iterations bring it to 0.07% for roughly 50% more
  build time. Pass `n_cg=0` for the unrefined operator.
- The linear-Z guard tests **magnitude** rather than dynamic range. A max/min ratio
  is unreliable in both directions — a dBZ field spanning 0–50 has a ratio of 1e5
  and would be falsely flagged, while linear Z from a 6.8–31.8 dBZ scene has a ratio
  of only 316 and would be missed. Magnitude separates the cases cleanly. Known
  blind spot: a linear field peaking below ~23 dBZ stays under the ceiling and is
  not caught.

### Fixed

- `interpolate_ray_times` no longer requires the two volumes to have equal ray
  counts. Consecutive volumes running the *same* scan strategy do not repeat their
  ray count — measured 13955 / 13951 / 13954 on one ARM C-SAPR2 triple — so the
  guard rejected essentially every real pair. The arithmetic never needed it: the
  output carries the earlier volume's acquisition pattern shifted by a single scalar
  derived from the two reference times, so `radar_late` contributes no per-ray
  correspondence to violate.
- `target_time` now accepts a naive or `cftime` instant. `volume_reference_time`
  returns a UTC-aware time but `pyart.util.datetime_from_radar` returns a naive
  `cftime.DatetimeGregorian`, so building a target from a Py-ART time -- the most
  natural call a Py-ART user can make -- raised `TypeError: can't subtract
  offset-naive and offset-aware datetimes`. A naive target is now read as UTC, which
  CF times are by construction, matching the rule already applied to decoded ray
  times. The normalisation rebuilds the instant field by field rather than calling
  `replace(tzinfo=UTC)`, because `cftime.DatetimeGregorian` is not a
  `datetime.datetime` subclass and its `replace()` rejects `tzinfo`.
- `_sweep_sampler` now collapses repeated azimuths instead of raising. A real
  antenna records the same azimuth twice when it dwells — 3 of 195 sweeps across 13
  consecutive volumes — and the sorted-but-not-deduplicated axis made the wrap
  tiling in `_sample_sweep` non-strictly-ascending, which `RegularGridInterpolator`
  rejects outright. Duplicates are averaged rather than discarded: they are
  independent samples of the same beam position. Note the azimuths are otherwise
  well behaved, with zero non-monotonic sweeps once the 360° wrap is accounted for,
  so deduplication is the fix rather than resorting.

  Both were found by running the operator on a real volume sequence for the first
  time, and neither is reachable from synthetic volumes built on `arange`.

- Documented, with measured values, that `pyart.core.cartesian_to_antenna` is not
  the inverse of `pyart.core.antenna_to_cartesian`: the forward transform returns
  great-circle arc length on the 4/3 earth while the inverse computes flat
  straight-line distance with no curvature term. Range error reaches -313 m at
  118 km and -4649 m at 460 km, maximised over elevation (it peaks near 35 degrees,
  not at grazing incidence); elevation error reaches +1552 mdeg. Azimuth is
  unaffected. `radar_palette` ships its own inverse, closing to ~1e-10 m, and
  asserts the discrepancy in tests so an upstream fix surfaces as a failure.

- `Xradar.time` assignment does not propagate to the underlying `DataTree`, so
  correcting an interpolated volume's time through the Py-ART wrapper left an
  xradar caller's tree on its original clock. Time is now written back per sweep
  (`radar_palette.io.write_ray_times_to_datatree`). Note the asymmetry with
  `add_field`, which does write through.

- Advection-interpolated volumes previously reported the acquisition time of the
  earlier bracketing volume. The output looked valid and plotted without
  complaint, so any time-ordered downstream use (accumulation, cell tracking,
  matching against another instrument) was silently misaligned by up to the full
  inter-volume interval.

- Packaging scaffold: `src/` layout, `setuptools` + `setuptools-scm` build via
  `pyproject.toml`, optional-dependency extras (`advection`, `spectral`, `all`,
  `test`, `docs`, `dev`).
- Documented but empty subpackages `radar_palette.advection`,
  `radar_palette.gridding`, `radar_palette.util`, `radar_palette.testing`.
- Packaging-level test suite, `ruff` lint/format config, `pre-commit` hooks,
  Sphinx docs stub, `CONTRIBUTING.md`, `MANIFEST.in`, and a pull-request
  template.
- GitHub Actions CI (lint; tests on Python 3.10-3.13 plus a minimal-install
  job; build with `twine check --strict`) is added as a separate commit, since
  writing `.github/workflows/` requires a token permission the packaging push
  did not carry.
