"""
A Python module for radar simulation

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

from radarsimpy.simulator import sim_lidar  # pylint: disable=no-name-in-module


def test_lidar_scene_basic():
    """
    Basic test case with a single target and a simple lidar setup.
    """
    lidar = {
        "position": np.array([0, 0, 0]),
        "phi": np.array([0]),
        "theta": np.array([90]),
    }
    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    rays = sim_lidar(lidar, targets)
    assert rays.shape[0] == 1  # Check if rays are generated
    assert np.allclose(
        rays[0]["positions"], [10, 0, 0], atol=1e-05
    )  # Check lidar position
    assert np.allclose(
        rays[0]["directions"], [-1, 0, 0], atol=1e-05
    )  # Check ray direction
    assert np.allclose(
        rays[0]["normals"], [-1, 0, 0], atol=1e-05
    )  # Check surface normal
    assert rays[0]["range"] > 0  # Check range is positive
    assert np.isclose(rays[0]["range"], 10.0, atol=1e-04)  # Check range value
    assert rays[0]["intensity"] > 0  # Check intensity is positive


def test_lidar_scene_multiple_targets():
    """
    Test with multiple targets.
    """
    lidar = {
        "position": np.array([0, 0, 0]),
        "phi": np.array([0, 90]),
        "theta": np.array([90]),
    }
    targets = [
        {
            "model": "./models/ball_1m.stl",
            "location": np.array([10, 0, 0]),
        },
        {
            "model": "./models/ball_1m.stl",
            "location": np.array([0, 10, 0]),
        },
    ]
    rays = sim_lidar(lidar, targets)
    assert rays.shape[0] == 2  # Check if rays are generated for all targets
    assert np.allclose(
        rays[0]["positions"], [9.5000010e00, 0.0000000e00, -4.1525823e-07], atol=1e-05
    )
    assert np.allclose(
        rays[0]["directions"], [-0.9956786, 0.06597178, -0.06535852], atol=1e-05
    )
    assert np.allclose(
        rays[1]["positions"], [-4.1526718e-07, 9.5002060e00, -4.1526718e-07], atol=1e-05
    )
    assert np.allclose(
        rays[1]["directions"], [-0.03301523, -0.99731606, -0.06534925], atol=1e-05
    )
    # Verify range and intensity are populated for both hits
    for i in range(2):
        assert rays[i]["range"] > 0
        assert rays[i]["intensity"] >= 0


def test_lidar_scene_target_movement():
    """
    Test with a moving target.
    """
    lidar = {
        "position": np.array([0, 0, 0]),
        "phi": np.array([0]),
        "theta": np.array([90]),
    }
    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([1, 0, 0]),
        }
    ]
    rays_t0 = sim_lidar(lidar, targets, frame_time=0)
    rays_t1 = sim_lidar(lidar, targets, frame_time=1)
    assert np.allclose(rays_t0[0]["positions"], [10, 0, 0], atol=1e-05)
    assert np.allclose(rays_t0[0]["directions"], [-1, 0, 0], atol=1e-05)
    assert np.allclose(rays_t1[0]["positions"], [11, 0, 0], atol=1e-05)
    assert np.allclose(rays_t1[0]["directions"], [-1, 0, 0], atol=1e-05)
    # Range should increase as target moves away
    assert rays_t1[0]["range"] > rays_t0[0]["range"]


def test_lidar_scene_target_rotation():
    """
    Test with a rotating target.
    """
    lidar = {
        "position": np.array([0, 0, 0]),
        "phi": np.array([0]),
        "theta": np.array([90]),
    }
    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "rotation_rate": np.array([45, 0, 0]),
        }
    ]
    rays_t0 = sim_lidar(lidar, targets, frame_time=0)
    rays_t1 = sim_lidar(lidar, targets, frame_time=1)
    assert np.allclose(rays_t0[0]["positions"], [10, 0, 0], atol=1e-05)
    assert np.allclose(rays_t0[0]["directions"], [-1, 0, 0], atol=1e-05)
    assert np.allclose(rays_t1[0]["positions"], [10, 0, 0], atol=1e-05)
    assert np.allclose(rays_t1[0]["directions"], [0, -1, 0], atol=1e-05)
    # Normal should rotate with the target
    assert np.allclose(rays_t0[0]["normals"], [-1, 0, 0], atol=1e-05)
    assert np.allclose(rays_t1[0]["normals"], [0, -1, 0], atol=1e-05)
