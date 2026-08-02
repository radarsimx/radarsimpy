"""
Shared pytest fixtures and helpers for the RadarSimPy test suite

Provides:

* A stable working directory, so the ``./models/*.stl`` paths used throughout
  the suite resolve no matter where ``pytest`` is invoked from.
* ``models_dir`` / ``model_path`` for referring to bundled 3D models.
* ``mesh_module`` plus the ``mesh`` marker, which skip ray-tracing tests when
  no optional mesh-processing library is installed.
* ``make_transmitter`` / ``make_receiver`` / ``make_radar`` factories for the
  small radar configurations that many tests need.

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

#: Repository root, i.e. the directory holding ``models/``.
REPO_ROOT = Path(__file__).resolve().parent.parent


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
    """Skip ``mesh``-marked tests when no mesh-processing library is available."""
    if _any_mesh_module_installed() is not None:
        return

    skip_mesh = pytest.mark.skip(
        reason="no mesh-processing library installed "
        f"(install one of: {', '.join(MESH_MODULES)})"
    )
    for item in items:
        if "mesh" in item.keywords:
            item.add_marker(skip_mesh)


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
