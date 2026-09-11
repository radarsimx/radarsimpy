"""
Accuracy and speed A/B of the GPU baseband kernel's precision modes.

The mesh simulator's GPU baseband kernel can be switched at run time with the
``RADARSIMX_BB_PRECISION`` environment variable:

- ``double``      the all-FP64 reference kernel (the default)
- ``mixed``       FP64 only for the absolute range/delay and the waveform phase
                  difference; everything else in float
- ``float_naive`` a control with range, delay and phase forced to float, to
                  show what the split in ``mixed`` protects against

Every simulation runs in its own subprocess with the variable set there, so a
mode can never leak into the next run and all modes are timed from one binary.
Timing rounds interleave the modes and alternate their order, because absolute
GPU wall times on a laptop drift by a few percent across minutes; only paired,
same-round differences are reported.

Usage::

    python benchmarks/precision_ab.py accuracy --out-dir precision_out
    python benchmarks/precision_ab.py speed --rounds 4 --out-dir precision_out

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (os.path.dirname(_HERE), _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import scenes  # noqa: E402  (needs the path insert above)
from capture_reference import CASES as REFERENCE_CASES  # noqa: E402

from radarsimpy import Radar, Receiver, Transmitter  # noqa: E402

MODES = ["double", "mixed", "float_naive"]


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------


def _fmcw(f, t, prp, pulses, samples, rx_channels=1):
    """FMCW radar whose C++ and Python sample counts agree.

    ``fs = (samples + 0.5) / t`` rather than ``samples / t``: the latter lands
    one short for power-of-two counts because ``t`` is not exact in binary.
    """
    tx = Transmitter(
        f=f,
        t=t,
        tx_power=15,
        prp=prp,
        pulses=pulses,
        channels=[{"location": (0, 0, 0)}],
    )
    rx = Receiver(
        fs=(samples + 0.5) / t,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[{"location": (0, idx * 0.002, 0)} for idx in range(rx_channels)],
    )
    return Radar(transmitter=tx, receiver=rx)


def _pulse_doppler(pulses):
    """The radarsimnb ``waveform_pulse_doppler`` radar, with more pulses.

    10 GHz CW gated into a 3.33 us rectangular pulse, PRF 30 kHz, fs 6 MHz.
    """
    c = 299792458.0
    pulse_width = 1 / (c / (2 * 500))
    prf = c / (2 * 5000)
    fs = 6e6
    samples = int((1 / prf) * fs)
    mod_t = np.arange(0, samples) / fs
    amp = np.zeros_like(mod_t)
    amp[mod_t <= pulse_width] = 1
    tx = Transmitter(
        f=10e9,
        t=1 / prf,
        tx_power=67,
        pulses=pulses,
        channels=[{"location": (0, 0, 0), "amp": amp, "mod_t": mod_t}],
    )
    rx = Receiver(
        fs=fs,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[{"location": (0, 0, 0)}],
    )
    return Radar(transmitter=tx, receiver=rx)


def _reference_case(name):
    # These all use 20 samples, which make_radar's fs = samples / t gets right.
    case = REFERENCE_CASES[name]
    return (
        scenes.make_radar(**case["radar"]),
        scenes.make_targets(**case["targets"]),
        dict(case["sim"]),
    )


def build_scene(name):
    """Return ``(radar, targets, sim_kwargs)`` for a scene name."""
    if name.startswith("ref:"):
        return _reference_case(name[4:])

    if name == "pulse_doppler":
        # The notebook scene: corner reflector at 2 km closing at 150 m/s. The
        # per-sample range step (~5e-5 m) is far below a float ulp at 4 km
        # (~2.4e-4 m), which is what makes an all-float kernel staircase.
        return (
            _pulse_doppler(64),
            [
                {
                    "model": scenes.model_path("cr"),
                    "location": (2000, 0, 0),
                    "speed": (-150, 0, 0),
                }
            ],
            {"density": 0.5},
        )

    if name == "fmcw77_200m":
        # Long-range 77 GHz FMCW, 1 GHz over 100 us, sphere at 200 m closing at
        # 20 m/s. Beat ~13 MHz, inside fs ~20 MHz.
        return (
            _fmcw([77e9, 78e9], 100e-6, 150e-6, 32, 2000),
            [
                {
                    "model": scenes.model_path("ball_1m"),
                    "location": (200, 0, 0),
                    "speed": (-20, 0, 0),
                }
            ],
            # Rays per 1-degree grid cell grow as (range / lambda)^2; density 1
            # at 200 m and 77 GHz overflows the per-cell ray budget.
            {"density": 0.2},
        )

    if name == "long_cpi_low":
        # LOW fidelity (one snapshot per frame) over a 128 ms CPI with a fast
        # target: the largest ``move_time`` and so the largest range delta the
        # float delta formula has to carry (~38 m).
        return (
            _fmcw([10e9, 10.1e9], 50e-6, 1e-3, 128, 200),
            [
                {
                    "model": scenes.model_path("ball_1m"),
                    "location": (500, 0, 0),
                    "speed": (-300, 0, 0),
                }
            ],
            {"density": 1.0},
        )

    if name == "phase_noise_2frames":
        # 24 GHz FMCW with phase noise over two frames: exercises the phase-noise
        # LUT indices (and so the frame start time) in the mixed kernel. The seed
        # pins the realisation so every mode sees the same noise.
        tx = Transmitter(
            f=[24.125e9 - 50e6, 24.125e9 + 50e6],
            t=80e-6,
            tx_power=40,
            prp=100e-6,
            pulses=16,
            pn_f=np.array([1000, 10000, 100000, 1000000]),
            pn_power=np.array([-65, -70, -65, -90]),
            channels=[{"location": (0, 0, 0)}],
        )
        rx = Receiver(
            fs=(160 + 0.5) / 80e-6,
            noise_figure=8,
            rf_gain=20,
            load_resistor=500,
            baseband_gain=30,
            channels=[{"location": (0, 0, 0)}],
        )
        radar = Radar(transmitter=tx, receiver=rx, frame_time=[0, 0.05], seed=7)
        return (
            radar,
            scenes.make_targets("ball_1m", 30.0, speed=(-10, 2, 0)),
            {"density": 1.0},
        )

    if name == "range_gate_111km":
        # X-band stretch radar gated at 60 nmi, corner reflector 150 m past the
        # gate and closing at 250 m/s: the absolute delay (~0.74 ms) is huge and
        # only the residual delay matters, which is the worst case for forming
        # the phase difference and the modulation time.
        c = 299792458.0
        gate_range = 111.12e3
        tx = Transmitter(
            f=[9e9 - 150e6, 9e9 + 150e6],
            t=50e-6,
            tx_power=40,
            prp=1e-3,
            pulses=16,
            channels=[{"location": (0, 0, 0)}],
        )
        rx = Receiver(
            fs=40e6,
            noise_figure=8,
            rf_gain=20,
            load_resistor=500,
            baseband_gain=30,
            gate_delay=2 * gate_range / c,
            channels=[{"location": (0, 0, 0)}],
        )
        return (
            Radar(transmitter=tx, receiver=rx),
            [
                {
                    "model": scenes.model_path("cr"),
                    "location": (gate_range + 150.0, 0, 0),
                    "speed": (-250, 0, 0),
                }
            ],
            {"density": 0.005},
        )

    if name == "timing_ball":
        # The profiled workload, cut to 32 pulses so a round takes seconds.
        return (
            _fmcw([76e9, 77e9], 20e-6, 100e-6, 32, 256, rx_channels=8),
            scenes.make_targets("ball_1m", 20.0),
            {"density": 1.0},
        )

    if name == "timing_turbine":
        return (
            _fmcw([76e9, 77e9], 20e-6, 100e-6, 64, 128, rx_channels=4),
            scenes.make_targets("turbine", 15.0, speed=(-5, 0, 0)),
            {"density": 0.5},
        )

    raise KeyError(name)


ACCURACY_SCENES = [f"ref:{name}" for name in sorted(REFERENCE_CASES)] + [
    "pulse_doppler",
    "fmcw77_200m",
    "long_cpi_low",
    "phase_noise_2frames",
    "range_gate_111km",
    "timing_ball",
]
SPEED_SCENES = ["timing_ball", "timing_turbine"]


# --------------------------------------------------------------------------
# Worker: one simulation, in its own process
# --------------------------------------------------------------------------


def worker(scene, out_path):
    """Run one scene on the GPU and report its wall time as a JSON line."""
    # pylint: disable=import-outside-toplevel
    from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module

    # Pay CUDA context creation outside the timed region.
    warm_radar, warm_targets, _ = build_scene("ref:plate_normal")
    sim_radar(warm_radar, warm_targets, density=0.1, device="gpu")

    radar, targets, sim_kw = build_scene(scene)
    start = time.perf_counter()
    result = sim_radar(radar, targets, device="gpu", **sim_kw)
    elapsed = time.perf_counter() - start

    if out_path:
        np.save(out_path, np.asarray(result["baseband"]))
    print(json.dumps({"scene": scene, "time": elapsed}))


def run_worker(mode, scene, out_path=None):
    """Launch a worker with ``RADARSIMX_BB_PRECISION=mode``; returns seconds."""
    env = dict(os.environ, RADARSIMX_BB_PRECISION=mode)
    cmd = [sys.executable, os.path.abspath(__file__), "worker", scene]
    if out_path:
        cmd += ["--npy", out_path]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"{mode}/{scene} failed:\n{proc.stdout}\n{proc.stderr}")
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    return json.loads(line)["time"]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def _rd_map(bb):
    """Range-Doppler magnitude per channel, Hann-windowed in both dimensions."""
    bb = np.asarray(bb)

    def window(n):
        # np.hanning(n) is all zeros for n <= 2, which would blank the map.
        return np.hanning(n) if n > 2 else np.ones(n)

    win_s = window(bb.shape[-1])
    win_p = window(bb.shape[-2])
    cube = bb * win_s[None, None, :] * win_p[None, :, None]
    return np.fft.fft(np.fft.fft(cube, axis=-1), axis=-2)


def metrics(ref, test):
    """Accuracy of ``test`` against ``ref`` (both complex baseband cubes)."""
    ref = np.asarray(ref)
    test = np.asarray(test)
    diff = test - ref
    norm = np.linalg.norm(ref)
    out = {"rel_l2": float(np.linalg.norm(diff) / norm) if norm > 0 else 0.0}

    rd_ref = _rd_map(ref)
    rd_diff = _rd_map(test) - rd_ref
    peak = np.max(np.abs(rd_ref))
    out["rd_diff_dbc"] = (
        float(20 * np.log10(np.max(np.abs(rd_diff)) / peak + 1e-300))
        if peak > 0
        else float("-inf")
    )

    # Phase error where the reference carries signal (within 40 dB of its peak).
    mag = np.abs(ref)
    sig = mag > mag.max() * 1e-2
    if sig.any():
        dphi = np.angle(test[sig] * np.conj(ref[sig]))
        out["phase_err_rms_deg"] = float(np.degrees(np.sqrt(np.mean(dphi**2))))
        out["phase_err_max_deg"] = float(np.degrees(np.max(np.abs(dphi))))
    return out


def phase_smoothness(bb):
    """Residual of a straight-line fit to the unwrapped phase, in degrees RMS.

    Along pulses at the strongest sample of channel 0, and along samples within
    the strongest pulse over the stretch where the echo is present. A constant
    closing speed makes both close to linear; a float-quantised range turns them
    into staircases, which is what the residual picks up.
    """
    cube = np.asarray(bb)[0]
    mag = np.abs(cube)
    p_pk, s_pk = np.unravel_index(np.argmax(mag), mag.shape)
    res = {}

    if cube.shape[0] > 2:
        phs = np.unwrap(np.angle(cube[:, s_pk]))
        x = np.arange(phs.size)
        fit = np.polyval(np.polyfit(x, phs, 1), x)
        res["pulse_resid_rms_deg"] = float(np.degrees(np.std(phs - fit)))

    row = cube[p_pk]
    idx = np.nonzero(np.abs(row) > np.abs(row).max() * 0.5)[0]
    if idx.size > 3:
        # The longest contiguous run around the peak.
        runs = np.split(idx, np.nonzero(np.diff(idx) != 1)[0] + 1)
        run = max(runs, key=len)
        if run.size > 3:
            phs = np.unwrap(np.angle(row[run]))
            fit = np.polyval(np.polyfit(run, phs, 2), run)
            res["sample_resid_rms_deg"] = float(np.degrees(np.std(phs - fit)))
    return res


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def accuracy(out_dir, scenes_to_run, modes):
    os.makedirs(out_dir, exist_ok=True)
    table = {}
    for scene in scenes_to_run:
        safe = scene.replace(":", "_")
        arrays = {}
        for mode in modes:
            path = os.path.join(out_dir, f"{safe}__{mode}.npy")
            run_worker(mode, scene, path)
            arrays[mode] = np.load(path)
        row = {"shape": list(arrays["double"].shape)}
        row["double"] = phase_smoothness(arrays["double"])
        for mode in modes:
            if mode == "double":
                continue
            row[mode] = metrics(arrays["double"], arrays[mode])
            row[mode].update(phase_smoothness(arrays[mode]))
        table[scene] = row
        print(f"{scene}: {json.dumps(row)}", flush=True)

    with open(os.path.join(out_dir, "accuracy.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)


def speed(out_dir, scenes_to_run, modes, rounds):
    os.makedirs(out_dir, exist_ok=True)
    results = {scene: {mode: [] for mode in modes} for scene in scenes_to_run}
    for rnd in range(rounds):
        order = modes if rnd % 2 == 0 else list(reversed(modes))
        for scene in scenes_to_run:
            for mode in order:
                elapsed = run_worker(mode, scene)
                results[scene][mode].append(elapsed)
                print(f"round {rnd} {scene:16} {mode:12} {elapsed:8.3f} s", flush=True)

    summary = {}
    for scene, per_mode in results.items():
        base = np.array(per_mode["double"])
        summary[scene] = {"times": per_mode}
        for mode in modes:
            t = np.array(per_mode[mode])
            summary[scene][mode] = {
                "median_s": float(np.median(t)),
                "speedup_paired": [float(b / x) for b, x in zip(base, t)],
            }
        print(f"{scene}: " + ", ".join(
            f"{m} median {summary[scene][m]['median_s']:.3f} s "
            f"(x{np.median(summary[scene][m]['speedup_paired']):.2f})"
            for m in modes
        ))

    with open(os.path.join(out_dir, "speed.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker")
    w.add_argument("scene")
    w.add_argument("--npy")

    a = sub.add_parser("accuracy")
    a.add_argument("--out-dir", default="precision_out")
    a.add_argument("--scene", nargs="+", default=ACCURACY_SCENES)
    a.add_argument("--mode", nargs="+", default=MODES)

    s = sub.add_parser("speed")
    s.add_argument("--out-dir", default="precision_out")
    s.add_argument("--scene", nargs="+", default=SPEED_SCENES)
    s.add_argument("--mode", nargs="+", default=MODES)
    s.add_argument("--rounds", type=int, default=4)

    args = parser.parse_args()
    if args.cmd == "worker":
        worker(args.scene, args.npy)
    elif args.cmd == "accuracy":
        if "double" not in args.mode:
            parser.error("accuracy needs the double mode as its reference")
        accuracy(args.out_dir, args.scene, args.mode)
    else:
        speed(args.out_dir, args.scene, args.mode, args.rounds)


if __name__ == "__main__":
    main()
