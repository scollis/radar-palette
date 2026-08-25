"""Benchmark and accuracy table for the azimuth NUFFT engines and solvers.

Run from the repository root::

    python benchmarks/bench_nufft_engines.py
    python benchmarks/bench_nufft_engines.py --csv engines.csv

Reports two things, because either alone would mislead. **Timing** is total cost
--- operator construction plus solve --- since a caller gridding one volume pays
both, and the engines differ in where they put the work (``finufft`` plans in
microseconds and spends it per transform; ``scipy`` precomputes a sparse stencil).
**Accuracy** is measured against a field that is exactly band-limited on the
recovery lattice, so the correct answer is known in closed form and the error is
absolute rather than a difference between two approximations.

Timings are hardware- and thread-dependent. The figures quoted in
:mod:`radar_palette.gridding.nufft_engines` came from this script on an arm64
laptop; re-run it rather than trusting them on other hardware.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

from radar_palette.gridding import nufft_engines as engines
from radar_palette.gridding.nufft import _AzimuthNufftOperator

# (n_rays, n_gates) pairs spanning real scan strategies: a coarse research sweep,
# an operational 1-degree volume, a 0.5-degree sweep with NEXRAD-like gate counts,
# and a fine-azimuth case past the dense engine's crossover.
SWEEP_SHAPES = (
    (120, 34),
    (360, 500),
    (360, 1832),
    (720, 1200),
    (1440, 1832),
)

CG_ITERATIONS = 12


class _Geometry:
    """The three fields the operators read, without constructing a radar object."""

    def __init__(self, nrays):
        self.nrays = nrays
        self.is_full_360 = True
        self.az_spacing_median_deg = 360.0 / nrays


def jittered_sweep(nrays, ngates, jitter=0.3, seed=2):
    """Build a NON_UNIFORM sweep and a smooth field on it, for timing."""
    spacing = 360.0 / nrays
    rng = np.random.default_rng(seed)
    azimuths = np.sort(
        (np.arange(nrays) * spacing + rng.uniform(-jitter, jitter, nrays) * spacing)
        % 360.0
    )
    gate_fraction = np.arange(ngates) / max(ngates, 1)
    values = np.zeros((nrays, ngates))
    for mode in (0, 1, 3):
        values += (
            np.cos(np.deg2rad(mode * azimuths))[:, None]
            * (1.0 + 0.3 * np.cos(2.0 * np.pi * gate_fraction))[None, :]
        )
    return azimuths, values, _Geometry(nrays)


def exact_lattice_sweep(nrays=120, ngates=8, jitter=0.3, seed=3, modes=(0, 1, 3, 7)):
    """Build a lattice-band-limited field, plus the lattice values themselves.

    The lattice values are the exact answer the solvers are trying to recover,
    which is what makes the accuracy column meaningful.
    """
    spacing = 360.0 / nrays
    rng = np.random.default_rng(seed)
    azimuths = np.sort(
        (np.arange(nrays) * spacing + rng.uniform(-jitter, jitter, nrays) * spacing)
        % 360.0
    )
    coefficients = np.random.default_rng(seed + 100)
    amplitude = coefficients.normal(size=(len(modes), ngates))
    phase = coefficients.uniform(0.0, 2.0 * np.pi, (len(modes), ngates))

    def evaluate(sample_azimuths):
        out = np.zeros((len(sample_azimuths), ngates))
        for index, mode in enumerate(modes):
            out += amplitude[index][None, :] * np.cos(
                np.deg2rad(mode * sample_azimuths)[:, None] + phase[index][None, :]
            )
        return out

    lattice_azimuths = np.arange(nrays) * 360.0 / nrays
    return azimuths, evaluate(azimuths), evaluate(lattice_azimuths), _Geometry(nrays)


def time_call(function, min_seconds=0.4, warmup=1):
    """Repeat until ``min_seconds`` has elapsed; return mean seconds per call.

    A fixed repeat count would either waste minutes on the large shapes or
    measure timer noise on the small ones.
    """
    for _ in range(warmup):
        function()
    calls, started = 0, time.perf_counter()
    while time.perf_counter() - started < min_seconds:
        function()
        calls += 1
    return (time.perf_counter() - started) / calls


def benchmark(engine_names):
    """Time every (engine, solver) pair against the reference, per sweep shape."""
    rows = []
    for nrays, ngates in SWEEP_SHAPES:
        azimuths, values, geometry = jittered_sweep(nrays, ngates)

        # Every closure below binds its loop variables as defaults. Deferred
        # lambdas over a loop variable are a real hazard here rather than a lint
        # nuisance: time_call invokes them after the loop body has moved on, so a
        # late-bound name would silently time the wrong sweep shape.
        baseline_build = time_call(
            lambda azimuths=azimuths, geometry=geometry: _AzimuthNufftOperator(
                azimuths, geometry
            ),
            min_seconds=0.25,
        )
        reference = _AzimuthNufftOperator(azimuths, geometry)
        baseline_solve = time_call(
            lambda reference=reference, values=values: reference.solve(
                values, n_cg=CG_ITERATIONS
            )
        )
        baseline = baseline_build + baseline_solve
        rows.append(
            {
                "nrays": nrays,
                "ngates": ngates,
                "engine": "reference(nufft.py)",
                "solver": "cg",
                "build_ms": 1e3 * baseline_build,
                "solve_ms": 1e3 * baseline_solve,
                "total_ms": 1e3 * baseline,
                "speedup": 1.0,
            }
        )

        for name in engine_names:
            build_seconds = time_call(
                lambda name=name, azimuths=azimuths, geometry=geometry: (
                    engines.make_operator(azimuths, geometry, engine=name)
                ),
                min_seconds=0.25,
            )
            operator = engines.make_operator(azimuths, geometry, engine=name)
            for solver in engines.SOLVERS:
                solve_seconds = time_call(
                    lambda solver=solver, operator=operator, values=values: (
                        operator.solve(values, n_cg=CG_ITERATIONS, solver=solver)
                    )
                )
                total = build_seconds + solve_seconds
                rows.append(
                    {
                        "nrays": nrays,
                        "ngates": ngates,
                        "engine": name,
                        "solver": solver,
                        "build_ms": 1e3 * build_seconds,
                        "solve_ms": 1e3 * solve_seconds,
                        "total_ms": 1e3 * total,
                        "speedup": baseline / total,
                    }
                )
    return rows


def accuracy(engine_names, jitters=(0.3, 0.45)):
    """Relative error against the analytically known lattice, per pair."""
    rows = []
    for jitter in jitters:
        azimuths, values, truth, geometry = exact_lattice_sweep(jitter=jitter)
        scale = np.ptp(truth)
        for name in engine_names:
            operator = engines.make_operator(azimuths, geometry, engine=name)
            for solver in engines.SOLVERS:
                recovered, _ = operator.solve(values, n_cg=CG_ITERATIONS, solver=solver)
                rows.append(
                    {
                        "jitter": jitter,
                        "engine": name,
                        "solver": solver,
                        "rel_error": float(np.max(np.abs(recovered - truth)) / scale),
                    }
                )
    return rows


def main(argv=None):
    """Print both tables, optionally writing the timing rows to CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="write the timing table to this path")
    parser.add_argument(
        "--real",
        action="store_true",
        help="held-out prediction error on real storm sweeps (downloads data)",
    )
    parser.add_argument(
        "--engines",
        nargs="*",
        help="engines to test (default: every installed engine)",
    )
    arguments = parser.parse_args(argv)

    engine_names = arguments.engines or list(engines.available_engines())
    missing = set(engine_names) - set(engines.available_engines())
    if missing:
        parser.error(
            f"not installed: {sorted(missing)}; "
            f"available: {list(engines.available_engines())}"
        )

    if arguments.real:
        print("=== held-out prediction error on real reflectivity (dB) ===")
        print("every 7th ray held out; lower is better")
        rows = real_data()
        print(
            f"{'case':52s} {'solver':8s} {'ridge':>9} "
            f"{'null':>5} {'all':>8} {'>echo':>8}"
        )
        for row in rows:
            ridge = "--" if np.isnan(row["ridge"]) else f"{row['ridge']:.1e}"
            null = "--" if np.isnan(row["null_dim"]) else f"{int(row['null_dim'])}"
            print(
                f"{row['case']:52s} {row['solver']:8s} {ridge:>9} {null:>5} "
                f"{row['median_dB']:>8.3f} {row['echo_median_dB']:>8.3f}"
            )
        if arguments.csv:
            with open(arguments.csv, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nwrote {arguments.csv}")
        return 0

    print(f"engines: {engine_names}")
    print(f"default: engine={engines.DEFAULT_ENGINE} solver={engines.DEFAULT_SOLVER}")

    print("\n=== accuracy: relative error vs the analytically known lattice ===")
    accuracy_rows = accuracy(engine_names)
    print(f"{'jitter':>7} {'engine':<12} {'cg(12)':>12} {'direct':>12} {'gain':>10}")
    for jitter in sorted({row["jitter"] for row in accuracy_rows}):
        for name in engine_names:
            picked = {
                row["solver"]: row["rel_error"]
                for row in accuracy_rows
                if row["jitter"] == jitter and row["engine"] == name
            }
            gain = picked["cg"] / picked["direct"] if picked["direct"] else float("inf")
            print(
                f"{jitter:>7} {name:<12} {picked['cg']:>12.3e} "
                f"{picked['direct']:>12.3e} {gain:>9.1f}x"
            )

    print("\n=== timing: total cost (build + solve), speedup vs nufft.py ===")
    timing_rows = benchmark(engine_names)
    print(
        f"{'sweep':>12} {'engine':<20} {'solver':<7} {'build':>9} "
        f"{'solve':>10} {'total':>10} {'speedup':>8}"
    )
    for row in timing_rows:
        print(
            f"{row['nrays']:>5}x{row['ngates']:<6} {row['engine']:<20} "
            f"{row['solver']:<7} {row['build_ms']:>8.2f}m {row['solve_ms']:>9.2f}m "
            f"{row['total_ms']:>9.2f}m {row['speedup']:>7.2f}x"
        )

    if arguments.csv:
        with open(arguments.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
            writer.writeheader()
            writer.writerows(timing_rows)
        print(f"\nwrote {arguments.csv}")
    return 0


# --------------------------------------------------------------------------- #
# Real-data validation (--real).
# --------------------------------------------------------------------------- #

# Storm-filled low-tilt sweeps from open-radar-data. Selected by inspecting the
# PPI: a clear-air or weak-stratiform sweep measures how well the operator
# reproduces noise, which is not a number to tune a default against. Fractions
# are of valid gates above 10 / 30 dBZ.
# A local volume, if the caller points at one with RADAR_PALETTE_BNF_VOLUME.
# The figures in nufft_engines.py come from the ARM BNF C-SAPR2 case used by the
# performance notebook, which is not downloadable; a full multi-tilt volume is
# more informative than the single sweeps above, because the tilts share a scan
# strategy and differ only in what they are looking at.
BNF_VOLUME = os.environ.get("RADAR_PALETTE_BNF_VOLUME")
BNF_SWEEPS = (0, 3, 7, 11, 14)

# Echo threshold for the stratified figures. Over half the gates in a real sweep
# sit below 0 dBZ, so an unstratified median is dominated by noise and by the
# fill floor, and it ranks the solvers differently from the echo it is supposed
# to be measuring.
ECHO_THRESHOLD_DBZ = 10.0

REAL_CASES = (
    (
        "cfrad.20080604_002217_000_SPOL_v36_SUR.nc",
        0,
        "DBZ",
        "SPOL S-band, convective line (26% >10 dBZ, max 58)",
    ),
    (
        "swx_20120520_0641.nc",
        0,
        "corrected_reflectivity_horizontal",
        "SWX C-band, widespread precip (81% >10 dBZ, max 80)",
    ),
    (
        "corcsapr2cmacppiM1.c1.20181111.030003.nc",
        0,
        "corrected_reflectivity",
        "CSAPR2 C-band, convective (33% >10 dBZ, max 68)",
    ),
)

REAL_RIDGES = (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
HOLDOUT_STRIDE = 7


class _FullCircle:
    """Geometry for a held-out subset, which the census would not classify."""

    def __init__(self, nrays, spacing):
        self.nrays = nrays
        self.is_full_360 = True
        self.az_spacing_median_deg = spacing


def _holdout(radar, sweep, field, stride=HOLDOUT_STRIDE):
    """Split a real sweep into training rays and held-out rays.

    Held-out rays are the only honest test of a regularisation choice: an
    unregularised solve fits the rays it was given to round-off *because* it is
    overfitting, so training residual ranks the choices exactly backwards.
    """
    from radar_palette.gridding.fill import fill_sweep

    sl = radar.get_slice(sweep)
    azimuths = radar.azimuth["data"][sl].astype(float) % 360.0
    values = np.ma.asarray(radar.fields[field]["data"][sl]).astype(float)
    order = np.argsort(azimuths)
    azimuths, values = azimuths[order], values[order]

    keep = np.ones(azimuths.size, dtype=bool)
    keep[::stride] = False
    training = np.asarray(fill_sweep(values[keep], method="floor"))
    if not np.all(np.isfinite(training)):
        return None
    held_out = np.asarray(np.ma.filled(values[~keep], np.nan))
    return azimuths[keep], training, azimuths[~keep], held_out


def _real_sources():
    """List downloadable sweeps, plus a local volume if one is configured."""
    from open_radar_data import DATASETS

    sources = [
        (DATASETS.fetch(filename), sweep, field, label)
        for filename, sweep, field, label in REAL_CASES
    ]
    if BNF_VOLUME and os.path.isfile(BNF_VOLUME):
        for sweep in BNF_SWEEPS:
            sources.append(
                (BNF_VOLUME, sweep, "reflectivity", f"BNF C-SAPR2 sweep {sweep}")
            )
    return sources


def real_data(engine="dense", echo_threshold=ECHO_THRESHOLD_DBZ):
    """Held-out prediction error on real reflectivity, per ridge and for CG.

    Reported on gates above ``echo_threshold`` as well as on all gates, because
    the two rank the solvers differently: over half the gates in a real sweep are
    below 0 dBZ, so an unstratified median mostly measures noise.
    """
    import pyart

    rows = []
    for path, sweep, field, label in _real_sources():
        radar = pyart.io.read(path)
        split = _holdout(radar, sweep, field)
        if split is None:
            continue
        train_az, train_values, test_az, test_values = split
        finite = np.isfinite(test_values)
        geometry = _FullCircle(
            train_az.size, float(np.median(np.diff(np.sort(train_az))))
        )
        operator = engines.make_operator(train_az, geometry, engine=engine)

        echo = finite & (test_values > echo_threshold)

        def error(
            lattice,
            test_az=test_az,
            test_values=test_values,
            finite=finite,
            echo=echo,
        ):
            predicted = engines.evaluate_lattice(lattice, test_az)
            deviation = np.abs(predicted - test_values)
            return (
                float(np.median(deviation[finite])),
                float(np.median(deviation[echo])) if echo.sum() > 50 else np.nan,
            )

        lattice, info = operator.solve(train_values, n_cg=12, solver="cg")
        median, echo_median = error(lattice)
        rows.append(
            {
                "case": label,
                "solver": "cg(12)",
                "ridge": np.nan,
                "null_dim": np.nan,
                "median_dB": median,
                "echo_median_dB": echo_median,
            }
        )

        for ridge in REAL_RIDGES + ("auto",):
            operator._normal_cache = None
            lattice, info = operator.solve(train_values, solver="direct", ridge=ridge)
            median, echo_median = error(lattice)
            rows.append(
                {
                    "case": label,
                    "solver": "direct",
                    "ridge": info["ridge_rel"],
                    "null_dim": info["normal_null_dim"],
                    "median_dB": median,
                    "echo_median_dB": echo_median,
                }
            )
    return rows


if __name__ == "__main__":
    sys.exit(main())
