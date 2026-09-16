"""
Shared pytest fixtures and helpers for the RadarSimPy test suite

Provides:

* A stable working directory, so the ``./models/*.stl`` paths used throughout
  the suite resolve no matter where ``pytest`` is invoked from.
* ``models_dir`` / ``model_path`` for referring to bundled 3D models.
* ``mesh_module`` plus the ``mesh`` marker, which skip ray-tracing tests when
  no optional mesh-processing library is installed.
* ``gltf_module`` plus the ``gltf`` marker, which do the same for the
  optional glTF library used by ``radarsimpy.animation_kit``.
* ``make_transmitter`` / ``make_receiver`` / ``make_radar`` factories for the
  small radar configurations that many tests need.
* ``assert_baseband_close`` plus the two peak-referenced tolerances the
  baseband suites compare against.

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

import os
from pathlib import Path

import numpy as np
import pytest

from radarsimpy import Radar, Receiver, Transmitter
from radarsimpy.mesh_kit import check_module_installed

#: Mesh libraries ``radarsimpy.mesh_kit.import_mesh_module`` knows how to use.
MESH_MODULES = ("trimesh", "pyvista", "pymeshlab", "meshio")

#: Library ``radarsimpy.animation_kit.import_gltf_module`` needs.
GLTF_MODULE = "pygltflib"

#: Repository root, i.e. the directory holding ``models/``.
REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Baseband tolerances
# =============================================================================

#: How far a mesh (SBR) baseband sample may sit from a recorded value, as a
#: fraction of the strongest sample (3e-5 is -90 dBc).
#:
#: The arrays in ``test_module_sim_radar_mesh.py`` are recorded at full
#: round-trip precision from a **CPU** run of the shipped build, and reproduce
#: exactly there -- all 48 comparisons return a residual of 0. The CPU path is
#: the deterministic one: it keeps the lookup-table summation innermost and
#: serial, so a bin's contributions are always added in the same order. On the
#: GPU the ray contributions arrive through ``atomicAdd`` in whatever order the
#: scheduler produces, which is reproducible only to ~6e-17 of the peak.
#:
#: What sets this bound is the gap between the two devices: forcing the suite
#: onto the GPU moves samples by up to **8.6e-6** of the peak, the worst case
#: being ``test_scene_rx_offset``, whose density sits near a partial null. The
#: two devices do not share a math library, and the coherent sum over rays
#: amplifies an ulp of difference by ~150x wherever contributions cancel. 3e-5
#: leaves 3.5x over that while staying far tighter than any real defect: for
#: reference, a gate-delay narrowing caught during this work showed up as 56
#: degrees of phase. Building with ``RADARSIMX_DEVICE_PARITY`` closes the gap.
#:
#: Note that this is NOT reproducible from an all-FP64 build, and do not treat
#: it as if it were. ``L`` is also the geometry and BVH type, so ``float_t =
#: double`` changes which rays are launched and what they hit. With these
#: scenes at their default density -- which is not converged -- that moves the
#: answer by up to **159%** (``test_scene_multiple_targets``), and 17-21% on
#: the four antenna-pattern scenes. Measured 2026-09-16, replacing an earlier
#: claim here that an FP64 rebuild reproduced these values to ~5e-7.
#:
#: Re-record with ``plans/rerecord_suite.py`` after any deliberate change:
#: ``record cpu`` runs the suite and saves what it produced, ``rewrite``
#: patches the literals, ``verify [cpu|gpu]`` reports the residual per call,
#: ``repeat [cpu|gpu]`` checks run-to-run reproducibility first.
BASEBAND_PEAK_ATOL = 3e-5

#: The same bound for the point-target and interference paths (1e-6 is
#: -120 dBc).
#:
#: Those paths are far better conditioned than the mesh one: a bin sums a
#: handful of contributions rather than tens of thousands of rays, so nothing
#: amplifies the rounding. Moving them to mixed precision was measured at
#: 5.5e-8 to 6.8e-8 of the peak -- one float ulp -- across the point and
#: interference scenes in ``benchmarks/precision_ab.py``, or -150 to -162 dBc
#: on the range-Doppler map.
#:
#: What sets this bound is neither of those, but the gap between the two
#: devices. The arrays in ``test_module_sim_radar_ideal.py`` were recorded from
#: a GPU run at full round-trip precision, and reproduce **exactly** there --
#: every one of the 25 comparisons returns a residual of 0. Forcing the same
#: tests onto the CPU moves them by at most 2.1e-7 of the peak, because the two
#: devices do not share a math library. 1e-6 leaves 4.7x over that. Building
#: with ``RADARSIMX_DEVICE_PARITY`` closes the gap and would allow much tighter.
#:
#: Re-record with ``<scratchpad>/rerecord_ideal.py`` after any deliberate
#: change: ``record`` runs the suite and saves what it produced, ``rewrite``
#: patches the literals, ``verify [cpu|gpu]`` reports the residual per call.
IDEAL_BASEBAND_PEAK_ATOL = 1e-6


def assert_baseband_close(actual, expected, atol_frac=BASEBAND_PEAK_ATOL):
    """Assert a baseband array matches ``expected`` relative to its peak.

    Referenced to the peak rather than element-wise, because the rounding this
    absorbs is a property of the coherent sum as a whole, not of each sample:
    an element-wise tolerance is simultaneously too tight on samples near a
    null and too loose on the ones carrying the signal.
    """
    expected = np.asarray(expected)
    peak = np.max(np.abs(expected))
    np.testing.assert_allclose(
        np.asarray(actual), expected, rtol=0, atol=atol_frac * peak
    )
    return True


def _any_mesh_module_installed():
    """Return the name of the first installed mesh library, or None."""
    for name in MESH_MODULES:
        if check_module_installed(name):
            return name
    return None


# =============================================================================
# Collection hooks
# =============================================================================


def pytest_collection_modifyitems(config, items):  # pylint: disable=unused-argument
    """Skip marked tests whose optional library is not installed."""
    skip_mesh = None
    if _any_mesh_module_installed() is None:
        skip_mesh = pytest.mark.skip(
            reason="no mesh-processing library installed "
            f"(install one of: {', '.join(MESH_MODULES)})"
        )

    skip_gltf = None
    if not check_module_installed(GLTF_MODULE):
        skip_gltf = pytest.mark.skip(
            reason=f"{GLTF_MODULE} is not installed (pip install {GLTF_MODULE})"
        )

    for item in items:
        if skip_mesh is not None and "mesh" in item.keywords:
            item.add_marker(skip_mesh)
        if skip_gltf is not None and "gltf" in item.keywords:
            item.add_marker(skip_gltf)


# =============================================================================
# Environment fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def _run_from_repo_root():
    """
    Run the whole session from the repository root.

    Many tests reference bundled models with paths such as
    ``"./models/ball_1m.stl"``. Pinning the working directory keeps those
    tests working when pytest is invoked from another directory.
    """
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        yield REPO_ROOT
    finally:
        os.chdir(previous)


@pytest.fixture(scope="session")
def repo_root():
    """Path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def models_dir(repo_root):
    """Path to the bundled 3D model directory."""
    return repo_root / "models"


@pytest.fixture(scope="session")
def model_path(models_dir):
    """
    Factory returning an absolute path (as ``str``) to a bundled 3D model.

    >>> def test_something(model_path):
    ...     target = {"model": model_path("ball_1m.stl")}
    """

    def _model_path(name):
        path = models_dir / name
        if not path.is_file():
            pytest.fail(f"missing test model: {path}")
        return str(path)

    return _model_path


@pytest.fixture(scope="session")
def mesh_module():
    """The mesh-processing module in use, skipping the test if none is installed."""
    name = _any_mesh_module_installed()
    if name is None:
        pytest.skip(f"no mesh-processing library installed ({', '.join(MESH_MODULES)})")
    return pytest.importorskip(name)


@pytest.fixture(scope="session")
def gltf_module():
    """The glTF module used by ``animation_kit``, skipping the test if missing."""
    return pytest.importorskip(GLTF_MODULE)


# =============================================================================
# Radar configuration factories
# =============================================================================


@pytest.fixture
def make_transmitter():
    """
    Factory for a small FMCW :class:`radarsimpy.Transmitter`.

    Defaults produce a single-channel 24 GHz / 100 MHz ramp; any keyword is
    forwarded to the constructor and overrides the default.
    """

    def _make(**kwargs):
        params = {
            "f": [24.075e9, 24.175e9],
            "t": 80e-6,
            "tx_power": 10,
            "prp": 100e-6,
            "pulses": 1,
            "channels": [{"location": (0, 0, 0)}],
        }
        params.update(kwargs)
        return Transmitter(**params)

    return _make


@pytest.fixture
def make_receiver():
    """Factory for a small single-channel :class:`radarsimpy.Receiver`."""

    def _make(**kwargs):
        params = {
            "fs": 2e6,
            "noise_figure": 12,
            "rf_gain": 20,
            "load_resistor": 500,
            "baseband_gain": 30,
            "channels": [{"location": (0, 0, 0)}],
        }
        params.update(kwargs)
        return Receiver(**params)

    return _make


@pytest.fixture
def make_radar(make_transmitter, make_receiver):
    """
    Factory for a :class:`radarsimpy.Radar` built from the default Tx/Rx.

    ``tx`` and ``rx`` keywords accept ready-made objects; ``tx_kwargs`` and
    ``rx_kwargs`` tweak the defaults instead. Remaining keywords go to
    :class:`radarsimpy.Radar`.
    """

    def _make(tx=None, rx=None, tx_kwargs=None, rx_kwargs=None, **kwargs):
        if tx is None:
            tx = make_transmitter(**(tx_kwargs or {}))
        if rx is None:
            rx = make_receiver(**(rx_kwargs or {}))
        return Radar(transmitter=tx, receiver=rx, **kwargs)

    return _make


@pytest.fixture
def rng():
    """Seeded ``numpy`` generator so randomised tests stay reproducible."""
    return np.random.default_rng(12345)
