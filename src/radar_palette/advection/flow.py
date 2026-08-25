"""Dense, height-resolved motion estimation between two gridded radar volumes.

A single rigid displacement is not enough for organised convection. In a mesoscale
convective system the broad stratiform region and the embedded convective cells move
at different speeds, and whole-domain cross-correlation is dominated by the larger,
slower area --- so a rigid estimate can pin the displacement near zero while cells are
moving at 20 m/s. Motion also varies with height, strong in the low levels and weaker
aloft, so the flow is estimated **independently on each vertical level** rather than
once for the volume.

Sign convention
---------------
The returned displacement is in the ``earlier -> later`` sense: an echo feature at
``(east, north)`` in the earlier grid is found near
``(east + displacement_east, north + displacement_north)`` in the later one.

This convention is not a matter of taste --- reversing it makes a reconstruction *worse
than a naive time average* while still producing plausible-looking output. It is
therefore pinned by end-to-end tests in ``tests/advection/test_interpolate.py``:
warping the earlier volume by the **full** displacement reproduces the later volume
(measured 0.46 dBZ RMSE on a synthetic translating scene, against 6.09 dBZ for the
reversed sign --- a factor of 13). Establish the sign from that test, not from
inspection of the code.

Relationship to Py-ART
----------------------
:func:`pyart.retrieve.grid_displacement_pc` estimates a single rigid displacement by
phase correlation. This module is the spatially varying counterpart; the two answer
different questions and neither supersedes the other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["grid_optical_flow"]

#: Reflectivity range (dBZ) used to normalise a field to [0, 1] before flow estimation.
DEFAULT_REFLECTIVITY_RANGE_DBZ = (0.0, 60.0)

#: Value assigned to masked or no-echo gates before normalisation. Below the range
#: minimum so that no-echo reads as uniform low background rather than as a hole.
DEFAULT_NO_ECHO_FLOOR_DBZ = -10.0


def _normalise_for_flow(field_values, minimum_dbz, maximum_dbz, no_echo_floor_dbz):
    """Map a possibly-masked reflectivity field onto [0, 1] for flow estimation.

    Optical flow has no notion of a masked sample, so gaps must be given a value.
    Filling with ``no_echo_floor_dbz`` (below ``minimum_dbz``) makes no-echo regions
    read as a uniform low background; filling with NaN would poison the solver, and
    filling with the field mean would fabricate gradients the flow would then track.
    """
    filled = np.ma.filled(
        np.ma.masked_invalid(field_values).astype("float64"), no_echo_floor_dbz
    )
    clipped = np.clip(filled, minimum_dbz, maximum_dbz)
    return (clipped - minimum_dbz) / (maximum_dbz - minimum_dbz)


def grid_optical_flow(
    earlier_grid,
    later_grid,
    field,
    reflectivity_range_dbz=DEFAULT_REFLECTIVITY_RANGE_DBZ,
    no_echo_floor_dbz=DEFAULT_NO_ECHO_FLOOR_DBZ,
    attachment=15.0,
    tightness=0.3,
    num_warp=5,
    num_iter=10,
):
    """Estimate a dense, per-level motion field between two grids.

    A total-variation L1 optical flow is computed independently on each vertical
    level and converted from pixels to metres using the grid spacing.

    Parameters
    ----------
    earlier_grid, later_grid : pyart.core.Grid
        Grids separated in time, sharing shape and axes. ``earlier_grid`` is the
        earlier of the two; the returned displacement points from it to the other.
    field : str
        Name of the field to track. Must be present in both grids.
    reflectivity_range_dbz : tuple of float, optional
        ``(minimum, maximum)`` dBZ used to normalise the field before flow
        estimation. Values outside are clipped.
    no_echo_floor_dbz : float, optional
        Value given to masked gates. Should be at or below the range minimum.
    attachment, tightness, num_warp, num_iter : optional
        Passed through to :func:`skimage.registration.optical_flow_tvl1`.

    Returns
    -------
    displacement_north, displacement_east : numpy.ndarray
        Echo displacement in **metres**, each shaped like the grid ``(nz, ny, nx)``.
        Note the order: **north first**, matching the row-then-column convention of
        the underlying flow routine. Divide by the volumes' time separation for a
        velocity.

    Raises
    ------
    ImportError
        If scikit-image is not installed.
    ValueError
        If the grids differ in shape.
    KeyError
        If ``field`` is absent from either grid.

    See Also
    --------
    radar_palette.advection.advection_interpolate : uses this field to reconstruct a
        volume at an intermediate time.
    """
    try:
        from skimage.registration import optical_flow_tvl1
    except ImportError as exc:  # pragma: no cover - exercised by the minimal CI job
        raise ImportError(
            "grid_optical_flow requires scikit-image. Install it with "
            "'pip install radar-palette[advection]'."
        ) from exc

    for name, grid in (("earlier_grid", earlier_grid), ("later_grid", later_grid)):
        if field not in grid.fields:
            raise KeyError(f"field {field!r} is not present in {name}")

    earlier = np.ma.filled(earlier_grid.fields[field]["data"].astype("float64"), np.nan)
    later = np.ma.filled(later_grid.fields[field]["data"].astype("float64"), np.nan)
    if earlier.shape != later.shape:
        raise ValueError(
            "the two grids must have identical shapes to estimate flow between "
            f"them, got {earlier.shape} and {later.shape}"
        )

    minimum_dbz, maximum_dbz = reflectivity_range_dbz
    spacing_east_m = float(np.diff(earlier_grid.x["data"])[0])
    spacing_north_m = float(np.diff(earlier_grid.y["data"])[0])

    displacement_north = np.zeros_like(earlier)
    displacement_east = np.zeros_like(earlier)
    for level in range(earlier.shape[0]):
        flow_north, flow_east = optical_flow_tvl1(
            _normalise_for_flow(
                earlier[level], minimum_dbz, maximum_dbz, no_echo_floor_dbz
            ),
            _normalise_for_flow(
                later[level], minimum_dbz, maximum_dbz, no_echo_floor_dbz
            ),
            attachment=attachment,
            tightness=tightness,
            num_warp=num_warp,
            num_iter=num_iter,
        )
        displacement_north[level] = flow_north * spacing_north_m
        displacement_east[level] = flow_east * spacing_east_m

    return displacement_north, displacement_east
