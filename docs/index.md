# radar-palette

Advective interpolation and spectral (FFT) gridding for weather radar volumes.

:::{warning}
Pre-alpha. This build is the packaging scaffold: the subpackages are documented
but not yet implemented, and nothing here is API-stable.
:::

## Subpackages

`radar_palette.advection`
: Time interpolation between two radar volumes that accounts for echo motion —
  dense optical flow for the motion field, then a semi-Lagrangian
  advect-and-blend on the radar's native geometry.

`radar_palette.gridding`
: Spectral resampling of radar sweeps onto Cartesian lattices — an exact
  Fourier series where the sweep geometry is uniform and periodic, a
  non-uniform FFT otherwise, with a separate non-spectral vertical operator.

`radar_palette.util`
: Shared beam-geometry and decibel-handling helpers.

`radar_palette.testing`
: Synthetic radar objects and analytic fields with known ground truth.

```{toctree}
:maxdepth: 2
:hidden:

api
```
