"""
Shared scene definitions for the SBR benchmark and reference-capture tools.

Both ``bench_sbr.py`` and ``capture_reference.py`` build their radars and target
lists from here, so a timing run and a numerical-reference run always describe
the same scene.

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

"""

import os

from radarsimpy import Radar, Receiver, Transmitter

#: Repository root, so scenes can be built from any working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Models shipped in ``models/``, with their triangle counts (binary STL:
#: ``(size - 84) / 50``). Note that ray count follows the solid angle the target
#: subtends, not the triangle count, so ``plate5x5`` is far more expensive than
#: ``ball_1m`` despite having two triangles.
MODELS = {
    "cr": "models/cr.stl",
    "plate5x5": "models/plate5x5.stl",
    "ball_1m": "models/ball_1m.stl",
    "half_ring": "models/half_ring.stl",
    "surface_60x60": "models/surface_60x60.stl",
    "turbine": "models/turbine.stl",
}


def model_path(name):
    """Resolve a key of :data:`MODELS` to an absolute path."""
    try:
        rel = MODELS[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown model {name!r}; choose from {sorted(MODELS)}"
        ) from exc
    return os.path.join(REPO_ROOT, rel)


def make_radar(pulses=1, samples=20, tx_channels=1, rx_channels=1, frames=1):
    """
    Build a 76-77 GHz FMCW radar with the requested array and waveform size.

    ``samples`` is converted to a sampling rate rather than passed directly, and
    the pulse length is held at 20 us, so the C++ and Python sample counts agree
    (``sim_radar`` rejects a mismatch).
    """
    pulse_length = 20e-6
    tx = Transmitter(
        f=[76e9, 77e9],
        t=pulse_length,
        tx_power=15,
        prp=100e-6,
        pulses=pulses,
        channels=[
            {"location": (0, 0, idx * 0.002)} for idx in range(tx_channels)
        ],
    )
    rx = Receiver(
        fs=samples / pulse_length,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {"location": (0, idx * 0.002, 0)} for idx in range(rx_channels)
        ],
    )
    # ``Radar`` takes frame start times, not a frame count; one frame is a
    # scalar 0, several are a list of start instants spaced by the frame length.
    frame_length = pulses * 100e-6
    frame_time = 0 if frames == 1 else [idx * frame_length for idx in range(frames)]
    return Radar(transmitter=tx, receiver=rx, frame_time=frame_time)


def make_targets(model="ball_1m", distance=20.0, speed=(0, 0, 0), extra=None):
    """One mesh target on boresight at ``distance`` metres, plus any extras.

    ``extra`` is a list of dicts merged over the same defaults, so a scene can
    add a ground plane or a second body without repeating the boilerplate. Each
    entry may set ``model`` (a key of :data:`MODELS`), ``location``, ``speed``
    and any target flag such as ``skip_diffusion``. Multi-target scenes are what
    make multi-bounce and back-propagation paths reachable at all -- a single
    convex body rarely produces a second bounce.
    """
    targets = [
        {
            "model": model_path(model),
            "location": (distance, 0, 0),
            "speed": speed,
        }
    ]
    for spec in extra or []:
        spec = dict(spec)
        target = {
            "model": model_path(spec.pop("model")),
            "location": spec.pop("location", (distance, 0, 0)),
            "speed": spec.pop("speed", (0, 0, 0)),
        }
        target.update(spec)
        targets.append(target)
    return targets
