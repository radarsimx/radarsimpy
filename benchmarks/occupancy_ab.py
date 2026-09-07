"""
A/B comparison of the two SBR stage-1 (occupancy) strategies.

The mesh simulator decides which coarse angular cells need dense rays by firing
one probe ray per cell centre. There are two ways to keep that pass from
missing a target that falls between probes:

**Method A (shipped)** builds a second scene of axis-aligned bounding boxes,
grows each by ``margin = 0.5 * tan(grid) * (range + radius)``, and probes that.
Expanding in object space before sampling guarantees nothing slips between
coarse samples, at the cost of an occupied set inflated twice over: by the AABB
silhouette being larger than the mesh silhouette, and by the half-cell margin.

**Method B (proposed)** probes the real mesh and dilates the occupancy grid
afterwards. Tighter, but dilation only grows around cells that were already
hit, so it cannot find a target no cell centre landed on.

Neither count needs a C++ change to measure. Method A's occupied set is dumped
verbatim by ``sim_radar(..., dry_run=True, log_path=...)``. Method B's is what
``sim_lidar`` computes when handed the same (phi, theta) grid: it fires the
same ``Ray`` code at the real, unexpanded mesh with ``max_ref=1`` and the same
direction convention, so it is stage 1 of the proposal exactly.

Stage-2 ray counts are replayed in Python with the shipped sizing formula
(``occupancy.cpp:163-166``). That replay is validated against method A's own
reported total before any of method B's numbers are believed.

Scope: single-target scenes with no environment mesh. The probe traces two
bounces and applies environment / skip-diffusion logic that the LiDAR path does
not; with one ordinary target the two reduce to the same first-hit test.

Usage::

    python benchmarks/occupancy_ab.py
    python benchmarks/occupancy_ab.py --scenes turbine ball_1m_100
    python benchmarks/occupancy_ab.py --grid 0.5 1 2 --density 0.1 1 4 --out ab.json

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# The repository root carries the ``radarsimpy`` package; ``_HERE`` carries
# ``scenes``. Running as ``python benchmarks/occupancy_ab.py`` puts only the
# latter on the path, so add both.
for _path in (os.path.dirname(_HERE), _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import scenes  # noqa: E402  (needs the path insert above)

from radarsimpy import Radar, Receiver, Transmitter  # noqa: E402
from radarsimpy.simulator import sim_lidar, sim_radar  # noqa: E402

try:
    import h5py
except ImportError:  # pragma: no cover - dev-only dependency
    h5py = None

#: Speed of light, matching the C++ physical constant.
C0 = 299792458.0

#: Highest transmit frequency; the probe sizes its sub-grid from the shortest
#: wavelength (``tx->lambda_min_ = c / freq_max_``).
F_MAX = 77e9

#: Sweep points: label -> (model, distance in metres). ``ball_1m`` appears at
#: two ranges deliberately. At 20 m it subtends 2.9 deg, comfortably above the
#: default grid; at 100 m it subtends 0.57 deg, below it, which is the case
#: method B is expected to lose.
SCENES = {
    "plate5x5": ("plate5x5", 40.0),
    "ball_1m_20": ("ball_1m", 20.0),
    "ball_1m_100": ("ball_1m", 100.0),
    "half_ring": ("half_ring", 20.0),
    "turbine": ("turbine", 40.0),
    "cr": ("cr", 20.0),
}


def make_radar(grid_deg):
    """
    A 76-77 GHz FMCW radar with an explicit coarse ``grid``.

    ``scenes.make_radar`` is the shared builder but does not expose ``grid``,
    which is the parameter this comparison sweeps, so the transmitter is built
    here. Every other setting matches it so the numbers stay comparable with
    ``bench_sbr.py``.
    """
    pulse_length = 20e-6
    samples = 20
    tx = Transmitter(
        f=[76e9, F_MAX],
        t=pulse_length,
        tx_power=15,
        prp=100e-6,
        pulses=1,
        channels=[{"location": (0, 0, 0), "grid": grid_deg}],
    )
    rx = Receiver(
        fs=samples / pulse_length,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[{"location": (0, 0, 0)}],
    )
    return Radar(transmitter=tx, receiver=rx, frame_time=0)


def cell_axes(grid_deg, phi_fov=(-90.0, 90.0), theta_fov=(0.0, 180.0)):
    """
    Coarse cell-centre angles in degrees, replicating the C++ grid.

    ``grid_size = int((fov[1] - fov[0]) / grid) + 1`` and cell n sits at
    ``fov[0] + n * grid`` (``transmitter.cpp:125-129``, ``occupancy.cpp:84``).
    The default pattern spans azimuth and elevation -90..90, and elevation maps
    to the polar angle as ``theta = 90 - elevation``
    (``cp_radarsimc_radar.pyx:168``), so theta runs 0..180.
    """
    n_phi = int((phi_fov[1] - phi_fov[0]) / grid_deg) + 1
    n_theta = int((theta_fov[1] - theta_fov[0]) / grid_deg) + 1
    phi = phi_fov[0] + np.arange(n_phi) * grid_deg
    theta = theta_fov[0] + np.arange(n_theta) * grid_deg
    return phi, theta


def fine_count(ranges, density, lambda_min, grid_rad):
    """
    Rays along one axis of a cell's sub-grid, per ``occupancy.cpp:163-166``.

    ``fine_step = atan(lambda / density / range)``, then
    ``count = int(grid / fine_step) + 1``. A cell contributes ``count ** 2``
    rays, which is why the occupied-cell count is the term that matters.
    """
    rng = np.asarray(ranges, dtype=np.float64)
    if rng.size == 0:
        return np.empty(0, dtype=np.int64)
    fine_step = np.arctan(lambda_min / density / rng)
    return (grid_rad / fine_step).astype(np.int64) + 1


def method_a(radar, targets, density, grid_deg, device):
    """
    Occupied cells and stage-2 ray total for the shipped expanded-box pass.

    ``dry_run`` returns right after the occupancy snapshot, and ``log_path``
    makes it dump ``occup*.h5`` on the way out
    (``simulator_mesh.cpp:600-614``).
    """
    if h5py is None:
        raise RuntimeError(
            "h5py is required to read occup*.h5; pip install h5py"
        )

    out_dir = tempfile.mkdtemp(prefix="occab_")
    try:
        start = time.perf_counter()
        sim_radar(
            radar,
            targets,
            density=density,
            log_path=out_dir,
            dry_run=True,
            device=device,
        )
        elapsed = time.perf_counter() - start

        path = os.path.join(out_dir, "occup_snapshot_0.h5")
        if not os.path.exists(path):
            # Dump() returns early when nothing was occupied.
            return {
                "cells": 0,
                "rays": 0,
                "rays_replayed": 0,
                "seconds": elapsed,
                "cell_idx": set(),
            }

        with h5py.File(path, "r") as handle:
            rows = handle["H5OccupancyStruct"][:]
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    phi_count = rows["PhiCount"].astype(np.int64)
    theta_count = rows["ThetaCount"].astype(np.int64)

    # The sub-grid is anchored at the cell's lower edge, which is the centre
    # less half a cell (``occupancy.cpp:168-170``).
    half = math.radians(grid_deg) * 0.5
    phi_c = np.degrees(rows["PhiStart"].astype(np.float64) + half)
    theta_c = np.degrees(rows["ThetaStart"].astype(np.float64) + half)
    p_idx = np.rint((phi_c + 90.0) / grid_deg).astype(int)
    t_idx = np.rint(theta_c / grid_deg).astype(int)

    return {
        "cells": int(len(rows)),
        "rays": int(rows["PoolAccu"][-1]),
        "rays_replayed": int((phi_count * theta_count).sum()),
        "seconds": elapsed,
        "cell_idx": set(zip(p_idx.tolist(), t_idx.tolist())),
    }


def method_b(targets, density, grid_deg, lambda_min):
    """
    Occupied cells and stage-2 ray totals for a real-mesh probe pass.

    ``sim_lidar`` fires one ray per (phi, theta) pair at the unexpanded mesh
    with ``max_ref=1`` and returns only the hits, each carrying its 3-D hit
    point. Every ray left from a cell centre, so the cell is recovered from
    ``positions - sensor``.

    Returns the plain result and the one-cell-dilated result.
    """
    phi_deg, theta_deg = cell_axes(grid_deg)
    position = np.array([0.0, 0.0, 0.0])
    grid_rad = math.radians(grid_deg)

    lidar = {"position": position, "phi": phi_deg, "theta": theta_deg}
    start = time.perf_counter()
    cloud = sim_lidar(lidar, targets)
    elapsed = time.perf_counter() - start

    empty = {"cells": 0, "rays": 0, "ranges": {}}
    if len(cloud) == 0:
        plain = dict(empty, seconds=elapsed, residual_deg=0.0, cell_idx=set())
        return plain, dict(empty)

    delta = cloud["positions"] - position
    norm = np.linalg.norm(delta, axis=1)
    phi_hit = np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))
    theta_hit = np.degrees(np.arccos(np.clip(delta[:, 2] / norm, -1.0, 1.0)))

    # Cell recovery: a hit must lie on the direction it was launched along, so
    # rounding to the nearest cell should leave a residual far below half a
    # cell. A large one means the two calls disagree on convention or origin.
    p_idx = np.rint((phi_hit - phi_deg[0]) / grid_deg).astype(int)
    t_idx = np.rint((theta_hit - theta_deg[0]) / grid_deg).astype(int)
    residual = max(
        float(np.abs(phi_hit - (phi_deg[0] + p_idx * grid_deg)).max()),
        float(np.abs(theta_hit - (theta_deg[0] + t_idx * grid_deg)).max()),
    )

    # ``range`` is the distance to the first hit, which is what sizes the
    # sub-grid. Keep the nearest if a cell somehow repeats.
    ranges = {}
    for pin, tin, rng in zip(
        p_idx.tolist(), t_idx.tolist(), cloud["range"].tolist()
    ):
        key = (pin, tin)
        if key not in ranges or rng < ranges[key]:
            ranges[key] = rng

    counts = fine_count(list(ranges.values()), density, lambda_min, grid_rad)
    plain = {
        "cells": len(ranges),
        "rays": int((counts**2).sum()),
        "seconds": elapsed,
        "residual_deg": residual,
        "ranges": ranges,
        "cell_idx": set(ranges),
    }
    return plain, dilate(ranges, density, lambda_min, grid_rad)


def dilate(ranges, density, lambda_min, grid_rad):
    """
    Grow the occupied set by one cell in the 8-neighbourhood.

    A dilated cell has no probe of its own, so it inherits the *largest* range
    among its occupied neighbours: a larger range gives a smaller ``fine_step``
    and so a denser sub-grid, which is the conservative direction.
    """
    grown = dict(ranges)
    for (pin, tin), rng in ranges.items():
        for d_p in (-1, 0, 1):
            for d_t in (-1, 0, 1):
                key = (pin + d_p, tin + d_t)
                if key in ranges:
                    continue
                grown[key] = max(grown.get(key, 0.0), rng)

    counts = fine_count(list(grown.values()), density, lambda_min, grid_rad)
    return {"cells": len(grown), "rays": int((counts**2).sum()), "ranges": grown}


def make_targets(model, distance, grid_deg, offset_cells):
    """
    A single target, optionally nudged off the probe lattice.

    ``scenes.make_targets`` places the target on boresight, which is itself a
    cell centre, so a probe ray always lands dead on it however small it is.
    That hides the failure this comparison exists to find. Shifting it by
    ``offset_cells`` of a cell laterally (a cell subtends ``distance *
    tan(grid)`` metres at the target) moves it off the lattice; at 0.5 it sits
    exactly between the four surrounding probes, which is the worst case.
    """
    targets = scenes.make_targets(model, distance)
    if offset_cells:
        shift = offset_cells * distance * math.tan(math.radians(grid_deg))
        location = np.asarray(targets[0]["location"], dtype=float)
        targets[0]["location"] = tuple(location + np.array([0.0, shift, shift]))
    return targets


def run_case(scene, grid_deg, density, device, offset_cells=0.0):
    """Run both methods on one (scene, grid, density) point."""
    model, distance = SCENES[scene]
    radar = make_radar(grid_deg)
    targets = make_targets(model, distance, grid_deg, offset_cells)
    lambda_min = C0 / F_MAX

    res_a = method_a(radar, targets, density, grid_deg, device)
    res_b, res_bd = method_b(targets, density, grid_deg, lambda_min)

    return {
        "scene": scene,
        "model": model,
        "distance_m": distance,
        "grid_deg": grid_deg,
        "density": density,
        "offset_cells": offset_cells,
        "a_cells": res_a["cells"],
        "a_rays": res_a["rays"],
        "b_cells": res_b["cells"],
        "b_rays": res_b["rays"],
        "bd_cells": res_bd["cells"],
        "bd_rays": res_bd["rays"],
        "a_seconds": res_a["seconds"],
        "b_seconds": res_b["seconds"],
        "replay_consistent": res_a["rays"] == res_a["rays_replayed"],
        "recovery_residual_deg": res_b["residual_deg"],
        "lost": res_a["cells"] > 0 and res_b["cells"] == 0,
        # Cells method B found that the expanded-box pass did not. Should be
        # empty: expansion is meant to be a superset.
        "b_only_cells": len(res_b["cell_idx"] - res_a["cell_idx"]),
    }


def _format(row):
    """One line per case, with the ratio that drives the decision."""

    def ratio(num, den):
        return f"{den / num:6.2f}x" if num else "     --"

    flags = ""
    if not row["replay_consistent"]:
        flags += "  !REPLAY-MISMATCH"
    if row["recovery_residual_deg"] > row["grid_deg"] * 0.5:
        flags += "  !CELL-RECOVERY"
    if row["b_only_cells"]:
        flags += f"  !B-ONLY={row['b_only_cells']}"
    if row["lost"]:
        flags += "  !TARGET-LOST"

    return (
        f"{row['scene']:<12} g={row['grid_deg']:<4} d={row['density']:<4} "
        f"o={row['offset_cells']:<4} | "
        f"cells A={row['a_cells']:>5} B={row['b_cells']:>5} "
        f"B+dil={row['bd_cells']:>5} | "
        f"rays A={row['a_rays']:>9} B={row['b_rays']:>9} "
        f"B+dil={row['bd_rays']:>9} | "
        f"saving {ratio(row['bd_rays'], row['a_rays'])}{flags}"
    )


def _summary(rows):
    """The checks the numbers are gated on, then the headline."""
    bad_replay = [r for r in rows if not r["replay_consistent"]]
    bad_recovery = [
        r for r in rows
        if r["recovery_residual_deg"] > r["grid_deg"] * 0.5
    ]
    lost = [r for r in rows if r["lost"]]
    total = len(rows)

    lines = [
        "checks:",
        f"  stage-2 replay matches PoolAccu  : "
        f"{total - len(bad_replay)}/{total}",
        f"  cell recovery within half a cell : "
        f"{total - len(bad_recovery)}/{total}",
    ]
    if lost:
        lines.append("  DETECTION LOSS (A sees the target, B does not):")
        for row in lost:
            lines.append(
                f"    {row['scene']} grid={row['grid_deg']} "
                f"density={row['density']}"
            )
    else:
        lines.append("  detection loss                   : none in this sweep")

    usable = [r for r in rows if r["bd_rays"] and r["b_rays"] and not r["lost"]]
    if usable:
        lines += ["", "stage-2 ray saving vs method A (>1 means cheaper):"]
        for label, key in (("B+dilate", "bd_rays"), ("B plain ", "b_rays")):
            def saving(row, field=key):
                return row["a_rays"] / row[field]

            best = max(usable, key=saving)
            worst = min(usable, key=saving)
            lines.append(
                f"  {label}  best {best['scene']:<12} "
                f"{best['a_rays'] / best[key]:5.2f}x   "
                f"worst {worst['scene']:<12} "
                f"{worst['a_rays'] / worst[key]:5.2f}x"
            )
    return "\n".join(lines)


def main(argv=None):
    """Sweep the requested scenes and print the comparison table."""
    parser = argparse.ArgumentParser(
        description="Compare expanded-box vs real-mesh SBR occupancy."
    )
    parser.add_argument(
        "--scenes", nargs="+", default=list(SCENES), choices=list(SCENES)
    )
    parser.add_argument("--grid", nargs="+", type=float, default=[1.0])
    parser.add_argument("--density", nargs="+", type=float, default=[1.0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--offset-cells",
        nargs="+",
        type=float,
        default=[0.0],
        help="shift the target off the probe lattice by this fraction of a "
        "cell; 0.5 puts it between probes, the worst case for method B",
    )
    parser.add_argument("--out", help="write the raw rows to this JSON file")
    args = parser.parse_args(argv)

    rows = []
    for scene in args.scenes:
        for grid_deg in args.grid:
            for density in args.density:
                for offset in args.offset_cells:
                    row = run_case(
                        scene, grid_deg, density, args.device, offset
                    )
                    rows.append(row)
                    print(_format(row), flush=True)

    print()
    print(_summary(rows))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)
        print(f"\nwrote {args.out}")
    return rows


if __name__ == "__main__":
    main()
