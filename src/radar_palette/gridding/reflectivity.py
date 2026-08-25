"""Reflectivity unit conversions, and why the choice of units is not cosmetic.

Band-limited interpolation must be performed in **dBZ**, not in linear
reflectivity factor Z. This is a measured result, not a stylistic preference.

Linear Z spans several orders of magnitude across a storm edge. A band-limited
interpolant through such a field overshoots on both sides of the jump --- ordinary
Gibbs behaviour --- and on the low side the overshoot drives the value **negative**,
which is physically impossible for a reflectivity factor. Converting back to dBZ
then requires clipping, and the clipped values are indistinguishable from real weak
echo.

Reproduced here on a synthetic 55 dBZ core against a -10 dBZ background --- a hard echo
edge, which is the geometry that provokes it --- interpolating in linear Z put
**42.6% of evaluated samples below zero**, worst excursion -0.16 times the field
maximum. Converting back to dBZ clips every one of those samples, and the clipped
values are indistinguishable from real weak echo. An earlier report on a real C-SAPR
tilt found the same failure at 23.7--24.2% of gates.

In dBZ the same interpolation stays physically representable. It is **not** free of
ringing. On that same edge, sampled on a 360-ray by 100-gate sweep and probed on a
600x600 grid, the dBZ interpolant reaches 74.8 dBZ against a true maximum of 55.0 and
-20.6 against a true minimum of -10.0 --- that is **+19.8 dB above the data on the high
side but only -10.6 dB below it on the low side**, so the excursion is roughly twice
as large upward as downward. (The figures depend on the sampling and probe density: at
180 rays by 60 gates the same edge gives +17.5 and -8.7 dB, which is why the
configuration is stated rather than the numbers quoted bare.)

The difference is that an overshooting
dBZ value is still a dBZ value: it can be clipped to a physical range, compared
against neighbours, or flagged. A negative reflectivity factor cannot be interpreted
at all, and silently becomes plausible weak echo the moment it is converted.

So the guarantee here is narrow and worth stating plainly: working in dBZ does not
remove Gibbs ringing at sharp edges, it keeps the ringing inside the representable
domain. :class:`LinearReflectivityError` exists so that a caller who forgets is
stopped rather than handed a plausible-looking, wrong answer.

The conversions themselves are exact inverses of one another;
:func:`to_dbz` additionally reports how many samples it had to clip, which is how
the failure above was quantified in the first place.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DBZ_PHYSICAL_CEILING",
    "LinearReflectivityError",
    "looks_like_linear_reflectivity",
    "to_dbz",
    "to_linear",
]


# No meteorological dBZ value comes close to this; linear Z exceeds it above about
# 23 dBZ. The gap between those two facts is what makes the guard possible.
DBZ_PHYSICAL_CEILING = 200.0


class LinearReflectivityError(ValueError):
    """Raised when a field appears to be linear Z where dBZ is required.

    A :class:`ValueError` subclass, so callers who only care that the input was
    rejected need not know about this type.
    """


def to_linear(dbz):
    """Convert dBZ to linear reflectivity factor Z.

    Parameters
    ----------
    dbz : array_like
        Reflectivity in dBZ.

    Returns
    -------
    numpy.ndarray
        Reflectivity factor in mm^6 m^-3.
    """
    return 10.0 ** (np.asarray(dbz, dtype=np.float64) / 10.0)


def to_dbz(z, floor_linear=1e-6):
    """Convert linear reflectivity factor Z to dBZ, guarding the logarithm.

    Parameters
    ----------
    z : array_like
        Reflectivity factor in mm^6 m^-3.
    floor_linear : float, optional
        Values below this are raised to it before the logarithm. The default,
        1e-6, is -60 dBZ --- well below any meteorological signal.

    Returns
    -------
    dbz : numpy.ndarray
        Reflectivity in dBZ.
    n_clipped : int
        How many samples were below ``floor_linear`` and had to be raised. **A
        non-zero count is a warning sign**: it means the input contained values a
        reflectivity factor cannot take, which is the signature of having
        interpolated in linear Z. See the module docstring.
    """
    z = np.asarray(z, dtype=np.float64)
    below_floor = z < floor_linear
    return (
        10.0 * np.log10(np.where(below_floor, floor_linear, z)),
        int(np.count_nonzero(below_floor)),
    )


def looks_like_linear_reflectivity(values, dbz_ceiling=DBZ_PHYSICAL_CEILING):
    """Heuristically decide whether ``values`` are linear Z rather than dBZ.

    The test is **magnitude on a non-negative field**: no meteorological dBZ value
    reaches 200, while linear Z passes 200 as soon as the echo exceeds about 23 dBZ.
    A field that is everywhere non-negative *and* peaks above the ceiling is
    therefore linear Z with near-certainty.

    Two candidate rules were measured and rejected in favour of this one:

    *Positivity alone* proves nothing --- a 30 dBZ rain field is positive everywhere.

    *Dynamic range* (max/min ratio) is unreliable in both directions. A dBZ field
    spanning 0 to 50 has a ratio of 1e5, which would be falsely flagged; a linear
    field from a 6.8--31.8 dBZ scene has a ratio of only 316, which would be missed.
    Magnitude separates the two cases cleanly where the ratio does not.

    Parameters
    ----------
    values : array_like
        Field to inspect.
    dbz_ceiling : float, optional
        Value above which a non-negative field is called linear. The default, 200,
        is far above the ~80 dBZ ceiling of real weather echo, so false positives
        require a physically impossible dBZ field.

    Returns
    -------
    bool
        True if the field looks like linear Z.

    Notes
    -----
    Deliberately conservative in one direction, with a known blind spot in the
    other. Any negative value proves the field is not a reflectivity factor, so such
    fields are never flagged. But a linear field whose peak is **below** about 23 dBZ
    stays under the ceiling and is *not* caught --- weak-echo-only scenes can slip
    through. The guard reduces the chance of the mistake; it does not eliminate it,
    which is why the dBZ requirement is also documented at every entry point.
    """
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0 or np.any(finite_values < 0.0):
        return False
    return bool(finite_values.max() > dbz_ceiling)
