import numpy as np
import numpy.testing as npt

from radarsimpy import Radar, Transmitter, Receiver, get_scene_state


def create_dummy_radar(
    location=(0, 0, 0), speed=(0, 0, 0), rotation=(0, 0, 0), rotation_rate=(0, 0, 0)
):
    tx = Transmitter(
        f=[24e9, 24.1e9],
        t=1e-4,
        tx_power=10,
        prp=2e-4,
        pulses=3,
        channels=[{"location": (0, 0.1, 0)}, {"location": (0, -0.1, 0)}],
    )
    rx = Receiver(
        fs=1e5,
        channels=[{"location": (0.2, 0, 0)}],
    )
    return Radar(
        transmitter=tx,
        receiver=rx,
        location=location,
        speed=speed,
        rotation=rotation,
        rotation_rate=rotation_rate,
    )


def test_static_scene_state():
    """
    Test get_scene_state with a static radar.
    """
    target = {
        "model": "./models/plate5x5.stl",
        "location": (10, 0, 0),
    }

    radar = create_dummy_radar(location=(5, 2, 1), rotation=(0, 0, 0))

    state = get_scene_state(target, radar, timestamp=0.0)

    # Check structure
    assert "targets" in state
    assert "tx_locations" in state
    assert "rx_locations" in state
    assert "radar_boresight" in state

    assert "points" in state["targets"]
    assert "cells" in state["targets"]

    # Tx channels locations: platform (5,2,1) + local (0, 0.1, 0) and (0, -0.1, 0)
    npt.assert_allclose(state["tx_locations"][0], [5, 2.1, 1], atol=1e-5)
    npt.assert_allclose(state["tx_locations"][1], [5, 1.9, 1], atol=1e-5)

    # Rx channel location: platform (5,2,1) + local (0.2, 0, 0)
    npt.assert_allclose(state["rx_locations"][0], [5.2, 2, 1], atol=1e-5)

    # Radar boresight should be default along +X direction [1, 0, 0]
    npt.assert_allclose(state["radar_boresight"], [1, 0, 0], atol=1e-5)


def test_rotating_scene_state():
    """
    Test get_scene_state with a rotated radar platform.
    """
    target = {
        "model": "./models/plate5x5.stl",
        "location": (10, 0, 0),
    }

    # Rotate yaw = 90 deg (around Z axis)
    radar = create_dummy_radar(location=(0, 0, 0), rotation=(90, 0, 0))

    state = get_scene_state(target, radar, timestamp=0.0)

    # Radar boresight should rotate 90 degrees around Z, becoming [0, 1, 0] (+Y)
    npt.assert_allclose(state["radar_boresight"], [0, 1, 0], atol=1e-5)

    # Tx local offset (0, 0.1, 0) rotated by 90 deg yaw:
    # x' = x*cos - y*sin = -0.1
    # y' = x*sin + y*cos = 0
    # So Tx location should become [-0.1, 0, 0]
    npt.assert_allclose(state["tx_locations"][0], [-0.1, 0, 0], atol=1e-5)


def test_vectorized_query_scene_state():
    """
    Test get_scene_state with multiple query timestamps.
    """
    target = {
        "model": "./models/plate5x5.stl",
        "location": (10, 0, 0),
    }

    # Radar moving at 10 m/s along X
    radar = create_dummy_radar(location=(0, 0, 0), speed=(10, 0, 0))

    query_times = np.array([0.0, 1.0, 2.0])
    state = get_scene_state(target, radar, timestamp=query_times)

    # Shape of locations and boresight should be (3, num_channels, 3) and (3, 3)
    assert state["tx_locations"].shape == (3, 2, 3)
    assert state["radar_boresight"].shape == (3, 3)

    # Tx channel 0 locations:
    # t=0: [0, 0.1, 0]
    # t=1: [10, 0.1, 0]
    # t=2: [20, 0.1, 0]
    npt.assert_allclose(state["tx_locations"][:, 0, 0], [0, 10, 20], atol=1e-5)

    # Radar boresight should remain default [1, 0, 0] since there is no rotation
    for k in range(3):
        npt.assert_allclose(state["radar_boresight"][k], [1, 0, 0], atol=1e-5)
