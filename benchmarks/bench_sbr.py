"""
Timing harness for the mesh (SBR) radar simulator.

Sweeps the parameters that drive ray-tracing cost and records wall time, process
CPU time, and the derived thread occupancy (``cpu / wall``) for each point. A run
is written as JSON so two commits can be compared directly.

``cpu_s`` is recorded but should not be read as useful work: the OpenMP runtime
spin-waits between parallel regions, so an idle thread pool still accumulates CPU
time while the serial ray tracer runs. Wall time is the metric that matters, and
``--omp-scaling`` is the way to tell real parallel speedup from spin.

Usage::

    python benchmarks/bench_sbr.py --out baseline.json
    python benchmarks/bench_sbr.py --sweep density level --device cpu
    python benchmarks/bench_sbr.py --compare baseline.json --out after.json
    python benchmarks/bench_sbr.py --sweep level --omp-scaling 1 2 4 16

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# The repository root carries the ``radarsimpy`` package; ``_HERE`` carries
# ``scenes``. Running as ``python benchmarks/bench_sbr.py`` puts only the
# latter on the path, so add both.
for _path in (os.path.dirname(_HERE), _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import scenes  # noqa: E402  (needs the path insert above)

from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a dev-only dependency
    psutil = None


ALL_SWEEPS = ["density", "level", "tx", "rx", "model", "pulses"]


def _cpu_seconds():
    """Total user+system CPU time of this process, or ``None`` without psutil."""
    if psutil is None:
        return None
    times = psutil.Process().cpu_times()
    return times.user + times.system


def time_once(radar, targets, **kwargs):
    """Run one simulation and return its timing record."""
    cpu_before = _cpu_seconds()
    wall_before = time.perf_counter()
    sim_radar(radar, targets, **kwargs)
    wall = time.perf_counter() - wall_before
    cpu = None if cpu_before is None else _cpu_seconds() - cpu_before
    record = {"wall_s": wall, "cpu_s": cpu}
    if cpu is not None and wall > 0:
        record["threads_busy"] = cpu / wall
    return record


def _case(name, params, radar_kwargs, sim_kwargs):
    return {
        "name": name,
        "params": params,
        "radar_kwargs": radar_kwargs,
        "sim_kwargs": sim_kwargs,
    }


def build_cases(sweeps, model, device):
    """Expand the requested sweep names into a flat list of cases."""
    cases = []

    if "density" in sweeps:
        for density in [0.25, 0.5, 1.0, 2.0, 4.0]:
            cases.append(
                _case(
                    "density",
                    {"density": density},
                    {"pulses": 1, "samples": 20},
                    {"density": density},
                )
            )

    if "level" in sweeps:
        for level in [None, "frame", "pulse", "sample"]:
            cases.append(
                _case(
                    "level",
                    {"level": level},
                    {"pulses": 4, "samples": 20},
                    {"density": 0.5, "level": level},
                )
            )

    if "tx" in sweeps:
        for tx_channels in [1, 2, 4]:
            cases.append(
                _case(
                    "tx",
                    {"tx_channels": tx_channels},
                    {"pulses": 1, "samples": 20, "tx_channels": tx_channels},
                    {"density": 0.5},
                )
            )

    if "rx" in sweeps:
        for rx_channels in [1, 2, 4, 8]:
            cases.append(
                _case(
                    "rx",
                    {"rx_channels": rx_channels},
                    {"pulses": 1, "samples": 20, "rx_channels": rx_channels},
                    {"density": 0.5},
                )
            )

    if "model" in sweeps:
        for name in scenes.MODELS:
            cases.append(
                _case(
                    "model",
                    {"model": name},
                    {"pulses": 1, "samples": 20},
                    {"density": 1.0},
                )
            )

    if "pulses" in sweeps:
        for pulses in [1, 2, 4, 8]:
            cases.append(
                _case(
                    "pulses",
                    {"pulses": pulses},
                    {"pulses": pulses, "samples": 20},
                    {"density": 0.5},
                )
            )

    for case in cases:
        case["sim_kwargs"]["device"] = device
        case["params"].setdefault("model", model)
    return cases


def _git_describe(path):
    try:
        return subprocess.run(
            ["git", "-C", path, "describe", "--always", "--dirty"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def environment():
    """Machine and revision context, so a JSON file is self-describing."""
    cpp_dir = os.path.join(scenes.REPO_ROOT, "src", "radarsimcpp")
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "python": platform.python_version(),
        "radarsimpy_rev": _git_describe(scenes.REPO_ROOT),
        "radarsimcpp_rev": _git_describe(cpp_dir),
    }


def run(args):
    """Execute every requested case and return the full result document."""
    cases = build_cases(args.sweep, args.model, args.device)

    # One throwaway run so first-touch allocation and any lazy module loading
    # are not charged to the first measured case.
    warm_radar = scenes.make_radar(pulses=1, samples=20)
    sim_radar(
        warm_radar,
        scenes.make_targets(model=args.model),
        density=0.2,
        device=args.device,
    )

    results = []
    for case in cases:
        model = case["params"].get("model", args.model)
        radar = scenes.make_radar(**case["radar_kwargs"])
        targets = scenes.make_targets(model=model, distance=args.distance)
        sim_kwargs = dict(case["sim_kwargs"])
        if args.dry_run:
            sim_kwargs["dry_run"] = True

        best = None
        for _ in range(args.repeat):
            record = time_once(radar, targets, **sim_kwargs)
            if best is None or record["wall_s"] < best["wall_s"]:
                best = record

        entry = {"sweep": case["name"], "params": case["params"], **best}
        results.append(entry)

        busy = entry.get("threads_busy")
        busy_txt = "" if busy is None else f"  threads_busy={busy:5.2f}"
        params = ", ".join(f"{k}={v}" for k, v in case["params"].items())
        print(f"  [{case['name']:8}] {params:44} {entry['wall_s']:8.3f} s{busy_txt}")

    return {
        "environment": environment(),
        "settings": {
            "device": args.device,
            "model": args.model,
            "distance": args.distance,
            "repeat": args.repeat,
            "dry_run": args.dry_run,
        },
        "results": results,
    }


def _key(entry):
    params = ",".join(f"{k}={v}" for k, v in sorted(entry["params"].items()))
    return f"{entry['sweep']}|{params}"


def compare(baseline_path, current):
    """Print a speedup table of ``current`` against a previously saved run."""
    with open(baseline_path, encoding="utf-8") as handle:
        baseline = json.load(handle)
    before = {_key(e): e for e in baseline["results"]}

    print(f"\nvs {baseline_path}:")
    print(f"  {'case':56} {'before':>9} {'after':>9} {'speedup':>9}")
    for entry in current["results"]:
        key = _key(entry)
        if key not in before:
            continue
        was = before[key]["wall_s"]
        now = entry["wall_s"]
        speedup = was / now if now > 0 else float("inf")
        print(f"  {key:56} {was:8.3f}s {now:8.3f}s {speedup:8.2f}x")


def run_omp_scaling(args):
    """
    Re-run the requested sweeps once per ``OMP_NUM_THREADS`` value.

    The variable is read when the OpenMP runtime loads, so it cannot be changed
    from inside a running process; each thread count gets its own subprocess.
    This is the measurement that separates genuine parallel speedup from lock
    contention -- a loop serialized on a mutex gets *slower* as threads are added.
    """
    documents = {}
    for threads in args.omp_scaling:
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        out_path = f"{args.out or 'omp_scaling'}.omp{threads}.json"
        argv = [
            sys.executable,
            os.path.abspath(__file__),
            "--sweep", *args.sweep,
            "--model", args.model,
            "--device", args.device,
            "--distance", str(args.distance),
            "--repeat", str(args.repeat),
            "--out", out_path,
        ]
        if args.dry_run:
            argv.append("--dry-run")

        print(f"\n--- OMP_NUM_THREADS={threads} ---")
        subprocess.run(argv, env=env, check=True)
        with open(out_path, encoding="utf-8") as handle:
            documents[threads] = json.load(handle)
        os.remove(out_path)

    counts = args.omp_scaling
    print("\nwall time by thread count:")
    header = "".join(f"{n:>10}t" for n in counts)
    print(f"  {'case':50}{header}")
    baseline = documents[counts[0]]
    for idx, entry in enumerate(baseline["results"]):
        row = "".join(
            f"{documents[n]['results'][idx]['wall_s']:10.3f}s" for n in counts
        )
        print(f"  {_key(entry):50}{row}")

    return {
        "environment": environment(),
        "settings": {"omp_scaling": counts, "device": args.device},
        "by_threads": {str(n): documents[n]["results"] for n in counts},
    }


def main():
    parser = argparse.ArgumentParser(description="radarsimpy SBR benchmark")
    parser.add_argument(
        "--sweep",
        nargs="+",
        default=ALL_SWEEPS,
        choices=ALL_SWEEPS,
        help="which parameter sweeps to run (default: all)",
    )
    parser.add_argument(
        "--model",
        default="ball_1m",
        choices=sorted(scenes.MODELS),
        help="target model for every sweep except --sweep model",
    )
    parser.add_argument(
        "--distance", type=float, default=20.0, help="target range in metres"
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs per case; the fastest is kept"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pass dry_run=True to isolate the occupancy stage",
    )
    parser.add_argument(
        "--omp-scaling",
        nargs="+",
        type=int,
        metavar="N",
        help="re-run the sweeps in a subprocess per OMP_NUM_THREADS value",
    )
    parser.add_argument("--out", help="write results to this JSON file")
    parser.add_argument(
        "--compare", help="print a speedup table against this JSON file"
    )
    args = parser.parse_args()

    env = environment()
    print(f"radarsimpy SBR benchmark  (device={args.device}, model={args.model})")
    print(
        f"  {env['cpu_count']} logical cores, "
        f"OMP_NUM_THREADS={env['omp_num_threads']}"
    )
    print(
        f"  radarsimpy {env['radarsimpy_rev']}, "
        f"radarsimcpp {env['radarsimcpp_rev']}\n"
    )

    document = run_omp_scaling(args) if args.omp_scaling else run(args)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
        print(f"\nwrote {args.out}")

    if args.compare and not args.omp_scaling:
        compare(args.compare, document)


if __name__ == "__main__":
    main()
