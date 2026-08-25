"""Flavour-aware entry point for correcting an interpolated volume's time.

:mod:`radar_palette.advection.timing` does the time arithmetic on the Py-ART
surface. This module is the boundary layer: it accepts volumes in either supported
object family, applies the correction, and returns the family the caller expects.

The xradar path needs care. ``DataTree.pyart.to_radar()`` gives a wrapper whose
``time`` attribute can be assigned without the change ever reaching the tree, so
correcting time through the wrapper alone would leave the caller's ``DataTree`` on
its original clock --- the same silent-staleness failure the timing module exists
to fix. The correction is therefore written back into the tree explicitly.
"""

from __future__ import annotations

from radar_palette.advection.timing import interpolate_ray_times
from radar_palette.io import (
    RadarFlavor,
    detect_radar_flavor,
    resolve_output_flavor,
    to_pyart_radar,
    to_radar_flavor,
    write_ray_times_to_datatree,
)

__all__ = ["retime_interpolated_volume"]


def retime_interpolated_volume(
    interpolated_volume,
    radar_early,
    radar_late,
    alpha,
    output_flavor=None,
):
    """Stamp reconstructed acquisition times onto an interpolated volume.

    Accepts, and returns, either object family. The bracketing volumes may be in a
    different family than the interpolated volume --- they are read for their times
    only.

    Parameters
    ----------
    interpolated_volume : pyart.core.Radar or xarray.DataTree
        Volume produced by advection interpolation, carrying whatever time it
        inherited from its construction. Corrected in place where the object
        family allows it.
    radar_early, radar_late : pyart.core.Radar or xarray.DataTree
        Volumes the interpolation was built from, ``radar_early`` observed first.
        Neither is modified.
    alpha : float
        Fractional position of ``interpolated_volume`` between ``radar_early``
        (0.0) and ``radar_late`` (1.0).
    output_flavor : {'pyart', 'xradar'} or RadarFlavor, optional
        Object family to return. Defaults to the family of
        ``interpolated_volume``, so callers get back what they passed in.

    Returns
    -------
    pyart.core.Radar or xarray.DataTree
        The volume with corrected acquisition times, in the requested family.

    Examples
    --------
    Correct a Py-ART volume and get a ``Radar`` back::

        corrected = retime_interpolated_volume(interpolated, early, late, 0.5)

    Correct an xradar volume and get a ``DataTree`` back::

        corrected = retime_interpolated_volume(tree, early, late, 0.5)
    """
    input_flavor = detect_radar_flavor(interpolated_volume)
    output_flavor = resolve_output_flavor(output_flavor, input_flavor)

    early_surface, _ = to_pyart_radar(radar_early)
    late_surface, _ = to_pyart_radar(radar_late)
    reconstructed_time = interpolate_ray_times(early_surface, late_surface, alpha)

    if input_flavor is RadarFlavor.XRADAR:
        datatree = to_radar_flavor(interpolated_volume, RadarFlavor.XRADAR)
        write_ray_times_to_datatree(datatree, reconstructed_time)
        return to_radar_flavor(datatree, output_flavor)

    interpolated_volume.time = reconstructed_time
    return to_radar_flavor(interpolated_volume, output_flavor)
