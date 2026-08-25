# Contributing to radar-palette

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check .
ruff format --check .
pytest
```

CI runs the same three on Python 3.11 through 3.13, plus a minimal-install job
that exercises the optional-dependency skip paths and a build job that runs
`twine check --strict`.

## Layout

`src/` layout, so the installed package is what gets tested — never the working
directory. Add a new capability as a module inside the subpackage it belongs to
(`advection`, `gridding`, `util`, `testing`) and export its public names from
that subpackage's `__init__.py` via `__all__`; `tests/test_package.py` asserts
that everything advertised there is importable.

## Conventions

- **Docstrings**: numpydoc, enforced by `ruff` (pydocstyle, `convention = "numpy"`).
- **Optional dependencies**: `scikit-image`, `finufft`, `ducc0` and
  `torch`/`torchkbnufft` are optional. Guard them at the call site with a clear
  error, and gate tests with the `requires_skimage` / `requires_finufft` markers in
  `tests/conftest.py` so a minimal install still collects a green suite. A NUFFT
  engine is guarded by `nufft_engines._require`, which names the extra to install;
  `tests/gridding/test_nufft_engines.py` skips per engine via `engine_or_skip`.
- **A new NUFFT engine owes three tests**: the adjoint identity on itself (a
  correct forward paired with a mis-scaled adjoint breaks both solvers and no
  smoothness check would notice), agreement with `nufft.py` to within the
  reference's own kernel error, and agreement with exact trig-polynomial
  arithmetic if it claims to beat that kernel. The second without the third is not
  enough: `torchkbnufft` initially passed its adjoint test perfectly while sitting
  69% away from the correct answer, because a consistently wrong pair of operators
  is still a consistent pair. Watch the phase convention — each library has its
  own, and the mismatch is invisible to every test that compares an engine only
  against itself.
- **Engine speed and engine accuracy are separate claims**: quote them separately
  and say which engine a figure belongs to. The direct solver's ~1e-10 recovery is
  a property of the *pair* (exact engine plus direct solve); on a Kaiser-Bessel
  engine the same solver converges to the kernel's ~3e-4 and buys nothing at low
  jitter. Regenerate figures with `benchmarks/bench_nufft_engines.py` rather than
  copying them forward.
- **Units in docstrings**: always. Metres, seconds, degrees, dBZ.
- **No extrapolation, ever**: a vertical target outside the observed cone span gets
  NaN and a `VerticalFlag`, never a guessed value. Keep the invariant that finite
  values and `INTERPOLATED` flags are the same set — a NaN wearing an interpolated
  flag is a silent hole, and a finite value without one is a silent extrapolation.
  `tests/gridding/test_vertical.py` asserts the equivalence directly.
- **Report the layer, not just the value**: `gap_m` is what makes a gridded value
  interpretable. Accuracy is set by tilt spacing rather than interpolation scheme (a
  factor of 14 versus 1.2 in measurement), so any new vertical scheme should be
  presented with its layer thickness rather than as an accuracy improvement.
- **Motion-field sign**: the advection displacement is earlier-to-later, and the
  warp samples the earlier volume *behind* the gate (`position - alpha * D`).
  Reversing either makes reconstruction worse than a naive time average while still
  looking plausible, so establish the sign from the end-to-end test in
  `tests/advection/test_interpolate.py` — warp by the full displacement and check it
  reproduces the later volume — never from reading the code.
- **Reflectivity units**: interpolate in **dBZ**, never linear Z. Linear
  reflectivity through a band-limited interpolant goes negative at echo edges
  (42.6% of samples in a measured reproduction), and the negatives become
  indistinguishable from weak echo on conversion. Note the honest limit: dBZ keeps
  ringing *representable*, it does not remove it — the same edge overshoots +19.8 dB
  past the data maximum and -10.6 dB below its minimum.
- **Adjoint is not inverse**: any new gridding operator that computes an adjoint
  must either refine it (conjugate gradients on the normal equations) or document
  the error it leaves. Record an adjoint-consistency test in the report, as
  `radar_palette.gridding.nufft` does; a smooth-looking output is not evidence.
- **Geometry**: use `radar_palette.gridding.antenna_to_cartesian_43` and
  `cartesian_to_antenna_43`, not the Py-ART equivalents, wherever a round trip
  matters — Py-ART's inverse omits the curvature term (see `CHANGELOG.md`). Note
  Py-ART's forward transform takes kilometres while its inverse returns metres.
- **Object flavours**: public entry points accept both Py-ART and xradar objects
  and return the caller's family by default (`output_flavor` overrides). Route
  conversions through `radar_palette.io` and Py-ART's own interoperability layer
  (`pyart.xradar`, `Grid.to_xarray`, xradar's cfradial readers) — do not hand-roll
  a mapping between the two data models.
- **Reflectivity is interpolated in dBZ**, not linear Z.
- **An interpolated volume carries the time it represents**, never the time of a
  volume it was derived from. Reconstruct times via
  `radar_palette.advection.timing`; do not let a `deepcopy` of a bracketing
  volume carry its clock through to the output.
- **Displacement is physical echo displacement** (first volume to second, metres);
  `velocity = displacement / dt`, no sign flip.
- **Validation**: an interpolation operator gets tested against a ground truth it
  did not see — either an analytic field from `radar_palette.testing`, or a
  held-out real volume. Report the baseline it is being compared against.
- **Tests that hit the network** get `@pytest.mark.network`; slow ones get
  `@pytest.mark.slow`. Neither runs in the default CI job.

## Commits and attribution

Work in this repository is collaborative between human and AI contributors, and
commits record that. Every commit and pull request carries trailers for all
contributors:

```
Co-authored-by: Scott Collis <scollis.acrf@gmail.com>
Co-authored-by: Claude <noreply@anthropic.com>
```

## Docs

```bash
make -C docs html          # builds with -W, so warnings are errors
```

`intersphinx` fetches inventories from python.org, numpy.org, scipy.org and the
Py-ART docs, which needs network access. To build offline without tripping `-W`:

```bash
RADAR_PALETTE_DOCS_OFFLINE=1 make -C docs html
```

## Releases

Versions come from git tags via `setuptools-scm`. Tag `vX.Y.Z` on `main`; there
is no version string to edit by hand.

If a local build reports `0.0.0`, `setuptools-scm` could not derive a version and
fell back to `fallback_version` — expected while no release tag exists, and also
seen in sandboxed or unusual checkouts where it declines to inspect the
repository. Check with:

```bash
python -c "import setuptools_scm, os; print(setuptools_scm.get_version(root=os.getcwd()))"
```

CI builds with `fetch-depth: 0` so tags are available there.
