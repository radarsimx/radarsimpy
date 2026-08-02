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
for a different reason (no GPU code was compiled in) and no warning is raised.

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
from radarsimpy.simulator import sim_radar

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


def run(device):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sim_radar(radar, targets, device=device)
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    return result, messages


gpu_request, gpu_warnings = run("gpu")
cpu_request, cpu_warnings = run("cpu")

json.dump(
    {
        "match": bool(np.allclose(gpu_request["baseband"], cpu_request["baseband"])),
        "shape": list(gpu_request["baseband"].shape),
        "gpu_warnings": gpu_warnings,
        "cpu_warnings": cpu_warnings,
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
    """Any warning raised by the fallback names the missing CUDA device."""
    # A CPU-only build warns about nothing at all: gpu_policy is already CPU
    # there, so no fallback took place.
    for message in fallback_run["gpu_warnings"]:
        assert "No CUDA device" in message


def test_explicit_cpu_request_does_not_warn(fallback_run):
    """Asking for the CPU is not a fallback, so it stays silent."""
    assert fallback_run["cpu_warnings"] == []
