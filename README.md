# radar-palette

Advective interpolation and spectral (FFT) gridding for weather radar volumes.

Both capabilities began as research code developed against ARM C-SAPR and
NEXRAD volumes, and are staged here as a package while the APIs settle.  The
intent is eventual upstreaming into [Py-ART](https://github.com/ARM-DOE/pyart);
until then this package is the place they live and get tested.

## What is in here

| Subpackage | Purpose |
| --- | --- |
| `radar_palette.advection` | Time interpolation between two radar volumes that accounts for echo motion: dense optical flow for the motion field, then a semi-Lagrangian advect-and-blend on the radar's native geometry. |
| `radar_palette.gridding` | Spectral resampling of radar sweeps onto Cartesian lattices — exact Fourier series where the sweep geometry is uniform and periodic, non-uniform FFT otherwise, with a separate non-spectral vertical operator. |
| `radar_palette.util` | Shared beam-geometry and decibel-handling helpers. |
| `radar_palette.testing` | Synthetic radar objects and analytic fields with known ground truth. |

**Status: pre-alpha.** Both capabilities now work end to end, but nothing here is
API-stable.

## Usage

```python
from radar_palette.advection import advection_interpolate
from radar_palette.gridding import grid_volume

# Reconstruct a volume halfway between two observed ones, accounting for echo
# motion. The motion field is estimated internally; the acquisition time of the
# result is corrected to match.
midpoint = advection_interpolate(earlier_volume, later_volume, alpha=0.5)

# Grid a volume. The default reproduces Py-ART exactly; "spectral" opts into this
# package's operator, which also reports why each empty cell is empty.
grid = grid_volume(
    volume,
    grid_shape=(20, 241, 241),
    grid_limits=((0.0, 10_000.0), (-120_000.0, 120_000.0), (-120_000.0, 120_000.0)),
)
sharp = grid_volume(
    volume,
    grid_shape=(20, 241, 241),
    grid_limits=((0.0, 10_000.0), (-120_000.0, 120_000.0), (-120_000.0, 120_000.0)),
    method="spectral",
)
coverage = sharp.fields["coverage_flag"]["data"]  # radar_palette.gridding.VerticalFlag
```

Either function accepts a `pyart.core.Radar` or an `xarray.DataTree` and returns the
matching family; pass `output_flavor` to override.

### Two gridding operators, and why the default is the older one

`method="pyart"` distance-weights neighbouring gates: it smooths, and cannot
overshoot. `method="spectral"` evaluates a band-limited interpolant per sweep: exact
where the sampling supports it, but it rings at a discontinuity — measured +5.5 /
−5.3 dB beyond the data on a hard azimuthal echo edge. Neither dominates, so the
default preserves existing behaviour and sharpness is opt-in.

For vertical accuracy the scan strategy matters far more than the interpolation
scheme: on a bright band sharper than the tilt spacing, going from 6 tilts to 23
improved RMSE by a factor of 14, while the best-versus-worst scheme choice gained
1.2×. Read a gridded value alongside its layer thickness.

## Install

From a clone, for development:

```bash
git clone https://github.com/DIGR-Legacy/radar-palette.git
cd radar-palette
pip install -e ".[dev]"
pre-commit install
```

The core install pulls `numpy`, `scipy` and `arm_pyart`. The heavier numerical
machinery is behind extras so a minimal install stays light:

| Extra | Pulls | Needed for |
| --- | --- | --- |
| `advection` | `scikit-image` | optical-flow motion estimation |
| `spectral` | `finufft` | faster, more accurate non-uniform FFT engine |
| `spectral-ducc` | `ducc0` | alternative NUFFT engine |
| `spectral-torch` | `torch`, `torchkbnufft` | GPU-capable NUFFT engine |
| `all` | `advection` + `spectral` | |
| `engines` | every NUFFT engine | benchmarking, engine-equivalence tests |
| `test`, `docs`, `dev` | tooling | development |

None of the `spectral*` extras is required. The non-uniform gridding path runs on
`numpy`/`scipy` alone and reproduces the original implementation to round-off;
the extras are alternative **engines** for the same operator, selected per call:

```python
from radar_palette.gridding.evaluator import SweepSpectralEvaluator
from radar_palette.gridding.nufft_engines import available_engines

available_engines()  # what this environment supports

# Exact transform, no kernel error, no extra dependency -- and, paired with the
# direct solver, ~1e-10 recovery of a band-limited field against ~3e-4 for the
# iterative default, about 30x faster.
SweepSpectralEvaluator(
    values, geometry, azimuths, az_engine="dense", az_solver="direct"
)
```

`benchmarks/bench_nufft_engines.py` prints the accuracy and timing tables for
whichever engines are installed. See
`radar_palette.gridding.nufft_engines` for which engine to pick and why.

## Development

```bash
pytest              # test suite
ruff check .        # lint (includes import sort and numpydoc style)
ruff format .       # format
python -m build     # build sdist + wheel
```

Versioning is handled by `setuptools-scm` from git tags — there is no version
string to edit by hand. No release has been tagged yet.

When `setuptools-scm` cannot derive a version — no tags, no git metadata (a source
archive), or a checkout it declines to inspect — the build falls back to the
`fallback_version` configured in `pyproject.toml`, currently `0.0.0`. That is what
builds report today, so do not treat a `0.0.0` wheel as a released artifact. Once
a `vX.Y.Z` tag exists on `main`, tagged builds carry that version and untagged
commits after it carry a `setuptools-scm`-derived development version.

## Conventions

Two that are load-bearing and easy to get wrong:

- **Reflectivity is interpolated in dBZ, not linear Z.** Band-limited
  interpolation in linear Z produces large negative excursions on real sweeps.
- **Displacement is the physical echo displacement**, first volume to second,
  in metres — so `velocity = displacement / dt`, with no sign flip. Optical-flow
  backends that return a reference-to-moving warp have that inversion applied
  internally.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests are co-authored; commits
carry `Co-authored-by` trailers for all contributors, human and otherwise.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
>>>>>>> digr-remote/main
