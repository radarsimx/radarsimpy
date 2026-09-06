"""
Capture and diff baseband references for the mesh (SBR) simulator.

Some fixes to the ray tracer change simulation output by design. This tool
records the baseband of a fixed scene set to an ``.npz`` file so a change can be
reviewed as a numerical delta -- how much moved, and where -- instead of a wall
of failing asserts in the test suite.

Usage::

    python benchmarks/capture_reference.py --out before.npz
    # ... make the change, rebuild ...
    python benchmarks/capture_reference.py --out after.npz --compare before.npz

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(_HERE), _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import scenes  # noqa: E402  (needs the path insert above)

from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module


#: Scenes chosen to exercise distinct code paths rather than to be fast:
#: a flat plate (near-normal incidence, the scalar Fresnel branch), a corner
#: reflector (multi-bounce, so back-tracing and the reflection filter matter),
#: a sphere (a wide spread of incidence angles, including grazing), and a
#: multi-channel case (the per-Rx PO lookup table).
CASES = {
    "plate_normal": {
        "radar": {"pulses": 2, "samples": 20},
        "targets": {"model": "plate5x5", "distance": 20.0},
        "sim": {"density": 0.3},
    },
    "corner_multibounce": {
        "radar": {"pulses": 2, "samples": 20},
        "targets": {"model": "cr", "distance": 10.0},
        "sim": {"density": 1.0},
    },
    "sphere_grazing": {
        "radar": {"pulses": 2, "samples": 20},
        "targets": {"model": "ball_1m", "distance": 20.0},
        "sim": {"density": 0.6},
    },
    "sphere_mimo": {
        "radar": {"pulses": 2, "samples": 20, "tx_channels": 2, "rx_channels": 4},
        "targets": {"model": "ball_1m", "distance": 20.0},
        "sim": {"density": 0.4},
    },
    "turbine_moving": {
        "radar": {"pulses": 4, "samples": 20},
        "targets": {"model": "turbine", "distance": 15.0, "speed": (-5, 0, 0)},
        "sim": {"density": 0.5, "level": "pulse"},
    },
    "sphere_backprop": {
        "radar": {"pulses": 2, "samples": 20},
        "targets": {"model": "ball_1m", "distance": 20.0},
        "sim": {"density": 0.5, "back_propagating": True},
    },
}


def capture(device, names):
    """Run each named case and return ``{case_name: baseband array}``."""
    captured = {}
    for name in names:
        case = CASES[name]
        radar = scenes.make_radar(**case["radar"])
        targets = scenes.make_targets(**case["targets"])
        result = sim_radar(radar, targets, device=device, **case["sim"])
        captured[name] = np.asarray(result["baseband"])
        print(f"  {name:22} shape={captured[name].shape}")
    return captured


def _diff_one(name, before, after):
    """Report one case; returns True when the arrays are bit-identical."""
    if before.shape != after.shape:
        print(f"  {name:22} SHAPE CHANGED {before.shape} -> {after.shape}")
        return False

    if np.array_equal(before, after):
        print(f"  {name:22} identical")
        return True

    delta = np.abs(after - before)
    scale = np.abs(before)
    # Relative error is only meaningful where the reference has signal; bins at
    # the noise floor produce enormous ratios from negligible absolute changes.
    floor = scale.max() * 1e-6
    significant = scale > floor
    rel = delta[significant] / scale[significant] if significant.any() else np.zeros(1)

    print(
        f"  {name:22} max_abs={delta.max():.3e}  "
        f"max_rel={rel.max():.3e}  mean_rel={rel.mean():.3e}  "
        f"changed={100.0 * np.count_nonzero(delta) / delta.size:.1f}%"
    )
    return False


def compare(before_path, after):
    """Diff a freshly captured set against a saved one."""
    before = np.load(before_path)
    print(f"\nvs {before_path}:")

    identical = True
    for name in sorted(set(before.files) | set(after)):
        if name not in before.files:
            print(f"  {name:22} NEW (not in reference)")
            identical = False
        elif name not in after:
            print(f"  {name:22} MISSING (in reference, not captured)")
            identical = False
        elif not _diff_one(name, before[name], after[name]):
            identical = False

    print("\nall cases bit-identical" if identical else "\noutput changed")
    return identical


def main():
    parser = argparse.ArgumentParser(
        description="capture or diff SBR baseband references"
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=sorted(CASES),
        choices=sorted(CASES),
        help="which cases to capture (default: all)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--out", help="write the capture to this .npz file")
    parser.add_argument("--compare", help="diff the capture against this .npz file")
    args = parser.parse_args()

    print(f"capturing {len(args.case)} case(s) on {args.device}:")
    captured = capture(args.device, args.case)

    if args.out:
        np.savez(args.out, **captured)
        print(f"\nwrote {args.out}")

    if args.compare:
        identical = compare(args.compare, captured)
        # Non-zero exit lets CI treat an unexpected output change as a failure.
        sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
