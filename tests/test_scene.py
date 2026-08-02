"""
Tests for ``radarsimpy.scene.get_scene_state``

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

import numpy as np
import numpy.testing as npt
import pytest

from radarsimpy import get_scene_state

# ``get_scene_state`` always loads the target mesh.
pytestmark = pytest.mark.mesh


@pytest.fixture
def scene_radar(make_radar):
    """
    Factory for the small 2-Tx / 1-Rx radar used by the scene tests.

    Tx channels sit at ``(0, ±0.1, 0)`` and the Rx channel at ``(0.2, 0, 0)``
    so that platform translation/rotation is easy to verify by hand.
    """

    def _make(**kwargs):
        return make_radar(
            tx_kwargs={
                "f": [24e9, 24.1e9],
                "t": 1e-4,
                "prp": 2e-4,
                "pulses": 3,
                "channels": [{"location": (0, 0.1, 0)}, {"location": (0, -0.1, 0)}],
            },
            rx_kwargs={"fs": 1e5, "channels": [{"location": (0.2, 0, 0)}]},
            **kwargs,
        )

    return _make


@pytest.fixture
def plate_target(model_path):
    """A 5x5 m plate 10 m down-range."""
    return {"model": model_path("plate5x5.stl"), "location": (10, 0, 0)}


def test_static_scene_state(scene_radar, plate_target):
    """
    Test get_scene_state with a static radar.
    """
    radar = scene_radar(location=(5, 2, 1), rotation=(0, 0, 0))

    state = get_scene_state(plate_target, radar, timestamp=0.0)

    # Check structure
    assert set(state) == {
        "targets",
        "tx_locations",
        "rx_locations",
        "radar_boresight",
    }
    assert "points" in state["targets"]
    assert "cells" in state["targets"]

    # Mesh should be non-empty and shaped [N, 3] / [M, 3]
    assert state["targets"]["points"].shape[-1] == 3
    assert state["targets"]["points"].shape[0] > 0
    assert state["targets"]["cells"].shape[-1] == 3

    # Tx channels locations: platform (5,2,1) + local (0, 0.1, 0) and (0, -0.1, 0)
    npt.assert_allclose(state["tx_locations"][0], [5, 2.1, 1], atol=1e-5)
    npt.assert_allclose(state["tx_locations"][1], [5, 1.9, 1], atol=1e-5)

    # Rx channel location: platform (5,2,1) + local (0.2, 0, 0)
    npt.assert_allclose(state["rx_locations"][0], [5.2, 2, 1], atol=1e-5)

    # Radar boresight should be default along +X direction [1, 0, 0]
    npt.assert_allclose(state["radar_boresight"], [1, 0, 0], atol=1e-5)


def test_rotating_scene_state(scene_radar, plate_target):
    """
    Test get_scene_state with a rotated radar platform.
    """
    # Rotate yaw = 90 deg (around Z axis)
    radar = scene_radar(location=(0, 0, 0), rotation=(90, 0, 0))

    state = get_scene_state(plate_target, radar, timestamp=0.0)

    # Radar boresight should rotate 90 degrees around Z, becoming [0, 1, 0] (+Y)
    npt.assert_allclose(state["radar_boresight"], [0, 1, 0], atol=1e-5)

    # Tx local offset (0, 0.1, 0) rotated by 90 deg yaw:
    # x' = x*cos - y*sin = -0.1
    # y' = x*sin + y*cos = 0
    # So Tx location should become [-0.1, 0, 0]
    npt.assert_allclose(state["tx_locations"][0], [-0.1, 0, 0], atol=1e-5)


def test_vectorized_query_scene_state(scene_radar, plate_target):
    """
    Test get_scene_state with multiple query timestamps.
    """
    # Radar moving at 10 m/s along X
    radar = scene_radar(location=(0, 0, 0), speed=(10, 0, 0))

    query_times = np.array([0.0, 1.0, 2.0])
    state = get_scene_state(plate_target, radar, timestamp=query_times)

    # Shape of locations and boresight should be (3, num_channels, 3) and (3, 3)
    assert state["tx_locations"].shape == (3, 2, 3)
    assert state["rx_locations"].shape == (3, 1, 3)
    assert state["radar_boresight"].shape == (3, 3)

    # Tx channel 0 locations:
    # t=0: [0, 0.1, 0]
    # t=1: [10, 0.1, 0]
    # t=2: [20, 0.1, 0]
    npt.assert_allclose(state["tx_locations"][:, 0, 0], [0, 10, 20], atol=1e-5)

    # Radar boresight should remain default [1, 0, 0] since there is no rotation
    for k in range(3):
        npt.assert_allclose(state["radar_boresight"][k], [1, 0, 0], atol=1e-5)


def test_scene_state_2d_timestamp_shape(scene_radar, plate_target):
    """
    A 2-D query timestamp array is broadcast into every output shape,
    the target mesh included.
    """
    radar = scene_radar(location=(0, 0, 0), speed=(10, 0, 0))

    query_times = np.array([[0.0, 1.0], [2.0, 3.0]])
    state = get_scene_state(plate_target, radar, timestamp=query_times)

    assert state["tx_locations"].shape == (2, 2, 2, 3)
    assert state["rx_locations"].shape == (2, 2, 1, 3)
    assert state["radar_boresight"].shape == (2, 2, 3)
    assert state["targets"]["points"].shape[:2] == (2, 2)
    assert state["targets"]["points"].shape[-1] == 3
    npt.assert_allclose(
        state["tx_locations"][..., 0, 0], [[0, 10], [20, 30]], atol=1e-5
    )


def test_scene_state_3d_timestamp_shape(scene_radar, plate_target):
    """
    The simulation timeline is itself 3-D ``[frames, pulses, samples]``, so
    feeding it straight back in has to work.
    """
    radar = scene_radar(location=(0, 0, 0), speed=(10, 0, 0))
    sim_timestamp = radar.time_prop["timestamp"]
    assert sim_timestamp.ndim == 3

    state = get_scene_state(plate_target, radar, timestamp=sim_timestamp)

    shape = sim_timestamp.shape
    assert state["tx_locations"].shape == shape + (2, 3)
    assert state["rx_locations"].shape == shape + (1, 3)
    assert state["radar_boresight"].shape == shape + (3,)
    assert state["targets"]["points"].shape[:3] == shape

    # The platform moves at 10 m/s along +X, so Tx x tracks the query time.
    npt.assert_allclose(
        state["tx_locations"][..., 0, 0], 10.0 * sim_timestamp, atol=1e-3
    )


def test_scene_state_values_are_independent_of_timestamp_rank(
    scene_radar, plate_target
):
    """
    Querying the same instants one-at-a-time, as a 1-D array, and as a 2-D
    array gives identical values -- only the leading shape differs.
    """
    radar = scene_radar(location=(0, 0, 0), speed=(10, 0, 0))
    times = [0.0, 1.0, 2.0, 3.0]

    scalars = [get_scene_state(plate_target, radar, timestamp=t) for t in times]
    flat = get_scene_state(plate_target, radar, timestamp=np.array(times))
    square = get_scene_state(
        plate_target, radar, timestamp=np.array(times).reshape(2, 2)
    )

    for key in ("tx_locations", "rx_locations", "radar_boresight"):
        stacked = np.stack([state[key] for state in scalars])
        npt.assert_allclose(flat[key], stacked, atol=1e-5)
        npt.assert_allclose(
            square[key], stacked.reshape((2, 2) + stacked.shape[1:]), atol=1e-5
        )

    stacked_points = np.stack([state["targets"]["points"] for state in scalars])
    npt.assert_allclose(flat["targets"]["points"], stacked_points, atol=1e-3)
    npt.assert_allclose(
        square["targets"]["points"],
        stacked_points.reshape((2, 2) + stacked_points.shape[1:]),
        atol=1e-3,
    )


def test_scene_state_time_varying_motion_interpolates(scene_radar, plate_target):
    """
    A radar with a time-varying location array is interpolated onto the query
    timestamps rather than extrapolated from a constant speed.
    """
    # Build a radar whose location is sampled on the simulation timeline.
    reference = scene_radar()
    sim_timestamp = reference.time_prop["timestamp"]

    # Move along +X at 100 m/s, expressed directly as a location array.
    radar = scene_radar(location=(100.0 * sim_timestamp, 0, 0))
    assert radar.radar_prop["location"].ndim > 1

    t_query = np.array([sim_timestamp.min(), sim_timestamp.max()])
    state = get_scene_state(plate_target, radar, timestamp=t_query)

    expected_x = 100.0 * t_query
    # Tx channel 0 sits at local (0, 0.1, 0), so its x matches the platform x.
    npt.assert_allclose(state["tx_locations"][:, 0, 0], expected_x, atol=1e-3)
    npt.assert_allclose(state["tx_locations"][:, 0, 1], [0.1, 0.1], atol=1e-3)


def test_scene_state_multiple_targets(scene_radar, model_path):
    """
    Passing a list of targets merges their meshes, and the merged cell indices
    stay within the merged point array.
    """
    single = {"model": model_path("plate5x5.stl"), "location": (10, 0, 0)}
    pair = [
        single,
        {"model": model_path("plate5x5.stl"), "location": (20, 0, 0)},
    ]

    radar = scene_radar()

    one = get_scene_state(single, radar, timestamp=0.0)
    two = get_scene_state(pair, radar, timestamp=0.0)

    n_points = one["targets"]["points"].shape[0]
    n_cells = one["targets"]["cells"].shape[0]

    assert two["targets"]["points"].shape[0] == 2 * n_points
    assert two["targets"]["cells"].shape[0] == 2 * n_cells
    assert two["targets"]["cells"].max() < two["targets"]["points"].shape[0]

    # The second copy is offset 10 m further down-range.
    npt.assert_allclose(
        two["targets"]["points"][n_points:, 0],
        two["targets"]["points"][:n_points, 0] + 10,
        atol=1e-3,
    )
