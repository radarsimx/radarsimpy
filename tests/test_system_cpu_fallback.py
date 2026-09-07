"""
Runtime CPU fallback when the machine has no usable CUDA device

The execution policies are compile-time tags, so a GPU-enabled build has to
decide at runtime whether CUDA is actually usable. These tests hide every CUDA
device from a child process (``CUDA_VISIBLE_DEVICES=-1``) and check that a
``device="gpu"`` request still completes, and that it produces exactly what
``device="cpu"`` produces.

The child process is required: the device probe is cached for the lifetime of
the process, so the environment has to be set before ``radarsimpy`` is imported.

These tests pass on a CPU-only build too, where the request is served on the CPU
for a different reason: no GPU code was compiled in. Both reasons are reported,
because a silent fallback lets a caller benchmark ``device="gpu"`` and record
CPU timings with nothing on screen to say so.

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

::

    ██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝
    ██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝
    ██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗
    ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import radarsimpy
from radarsimpy.simulator import gpu_available, sim_radar

#: Directory holding the ``radarsimpy`` package, so the child imports the same one.
PACKAGE_PARENT = str(Path(radarsimpy.__file__).resolve().parent.parent)

#: Simulates one point target twice, once asking for the GPU and once for the CPU,
#: and reports the comparison plus any warnings each request raised.
CHILD_SCRIPT = """
import json
import sys
import warnings

import numpy as np

from radarsimpy import Radar, Receiver, Transmitter
from radarsimpy.simulator import gpu_available, sim_radar

tx = Transmitter(
    f=[24.075e9, 24.175e9], t=80e-6, tx_power=10, prp=100e-6, pulses=2,
    channels=[{"location": (0, 0, 0)}],
)
rx = Receiver(
    fs=2e6, noise_figure=12, rf_gain=20, load_resistor=500, baseband_gain=30,
    channels=[{"location": (0, 0, 0)}],
)
radar = Radar(transmitter=tx, receiver=rx)
targets = [{"location": (150, 0, 0), "speed": (-5, 0, 0), "rcs": 20, "phase": 0}]


def run(device=None):
    # Omitting the argument entirely is how the default gets exercised.
    kwargs = {} if device is None else {"device": device}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim_radar(radar, targets, **kwargs)
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    return result, messages


gpu_request, gpu_warnings = run("gpu")
cpu_request, cpu_warnings = run("cpu")
auto_request, auto_warnings = run("auto")
default_request, default_warnings = run()

json.dump(
    {
        "match": bool(np.allclose(gpu_request["baseband"], cpu_request["baseband"])),
        "shape": list(gpu_request["baseband"].shape),
        "gpu_warnings": gpu_warnings,
        "cpu_warnings": cpu_warnings,
        "auto_match": bool(
            np.allclose(auto_request["baseband"], cpu_request["baseband"])
        ),
        "auto_warnings": auto_warnings,
        "default_match": bool(
            np.allclose(default_request["baseband"], cpu_request["baseband"])
        ),
        "default_warnings": default_warnings,
        "gpu_available": bool(gpu_available()),
    },
    sys.stdout,
)
"""


@pytest.fixture(scope="module")
def fallback_run(repo_root):
    """Run ``CHILD_SCRIPT`` in a process where no CUDA device is visible."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = PACKAGE_PARENT

    completed = subprocess.run(
        [sys.executable, "-c", CHILD_SCRIPT],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "simulation failed with no CUDA device available:\n" f"{completed.stderr}"
    )
    return json.loads(completed.stdout)


def test_gpu_request_runs_without_a_cuda_device(fallback_run):
    """A ``device="gpu"`` request completes on a machine with no CUDA device."""
    assert fallback_run["shape"] == [1, 2, 160]


def test_fallback_matches_explicit_cpu_request(fallback_run):
    """The fallback result is the CPU result, not a degraded one."""
    assert fallback_run["match"]


def test_fallback_is_reported(fallback_run):
    """A ``device="gpu"`` request served on the CPU always says so, and why."""
    # Two distinct reasons reach this point and both have to be reported: a
    # CUDA build with no visible device fell back at runtime, and a CPU-only
    # build never had GPU code to fall back from. The second used to be silent,
    # which meant a CPU-only wheel would run device="gpu" at CPU speed with no
    # indication -- easy to mistake for a slow GPU.
    messages = fallback_run["gpu_warnings"]
    assert messages, "the fallback to the CPU was not reported at all"
    assert any(
        "No CUDA device" in message or "without CUDA support" in message
        for message in messages
    ), f"no warning explained why the GPU was not used: {messages}"


def test_explicit_cpu_request_does_not_warn(fallback_run):
    """Asking for the CPU is not a fallback, so it stays silent."""
    assert fallback_run["cpu_warnings"] == []


def test_auto_runs_on_the_cpu_when_no_device_is_visible(fallback_run):
    """``device="auto"`` resolves to the CPU and produces the CPU result."""
    assert fallback_run["auto_match"]


def test_auto_does_not_warn(fallback_run):
    """``"auto"`` choosing the CPU is the documented behaviour, not a fallback.

    Warning here would fire on every call for every CPU-only user, which is
    what makes the ``device="gpu"`` warning worth reading when it does appear.
    """
    assert fallback_run["auto_warnings"] == []


def test_default_device_is_auto(fallback_run):
    """Omitting ``device`` behaves exactly like ``device="auto"``."""
    assert fallback_run["default_match"]
    assert fallback_run["default_warnings"] == []


def test_gpu_available_agrees_with_the_device_actually_used(fallback_run):
    """``gpu_available()`` is what ``"auto"`` decides on, so they must agree.

    With every CUDA device hidden it has to report False, whatever the build.
    Tying the two together here is the point: a helper that disagreed with the
    selection it drives would make the skip guards that depend on it silently
    wrong.
    """
    assert fallback_run["gpu_available"] is False
    assert fallback_run["auto_match"], "auto did not run on the CPU"


def test_gpu_available_is_a_bool_on_this_machine():
    """The public helper answers without needing a simulation to be run."""
    assert isinstance(gpu_available(), bool)


def test_unknown_device_is_rejected():
    """An unrecognised device name fails loudly rather than picking one."""
    # No child process needed: this is rejected during validation, before any
    # device is touched, so the outcome does not depend on the machine.
    transmitter = radarsimpy.Transmitter(
        f=[24.075e9, 24.175e9], t=80e-6, tx_power=10, prp=100e-6, pulses=2,
        channels=[{"location": (0, 0, 0)}],
    )
    receiver = radarsimpy.Receiver(
        fs=2e6, noise_figure=12, rf_gain=20, load_resistor=500,
        baseband_gain=30, channels=[{"location": (0, 0, 0)}],
    )
    radar = radarsimpy.Radar(transmitter=transmitter, receiver=receiver)
    targets = [{"location": (150, 0, 0), "rcs": 20}]

    with pytest.raises(ValueError, match="not recognized"):
        sim_radar(radar, targets, device="cuda")
