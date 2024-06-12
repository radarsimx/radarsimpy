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

import pytest
import numpy as np
import numpy.testing as npt
from radarsimpy.rt import lidar_scene  # pylint: disable=no-name-in-module


# def test_lidar_scene_basic():
#     """
#     Basic test case with a single target and a simple lidar setup.
#     """
#     lidar = {
#         "position": np.array([0, 0, 0]),
#         "phi": np.array([0]),
#         "theta": np.array([0]),
#     }
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     rays = lidar_scene(lidar, targets)
#     assert rays.shape[0] > 0  # Check if rays are generated
#     assert np.allclose(rays[0]["positions"], [0, 0, 0])  # Check lidar position
#     assert np.allclose(rays[0]["directions"], [1, 0, 0])  # Check ray direction


# def test_lidar_scene_multiple_targets():
#     """
#     Test with multiple targets.
#     """
#     lidar = {
#         "position": np.array([0, 0, 0]),
#         "phi": np.array([0, 45]),
#         "theta": np.array([0, 90]),
#     }
#     targets = [
#         {
#             "model": "path/to/model1.obj",
#             "location": np.array([10, 0, 0]),
#         },
#         {
#             "model": "path/to/model2.obj",
#             "location": np.array([0, 10, 0]),
#         },
#     ]
#     rays = lidar_scene(lidar, targets)
#     assert rays.shape[0] > 4  # Check if rays are generated for all targets
#     # Add more specific assertions based on expected ray directions


# def test_lidar_scene_target_movement():
#     """
#     Test with a moving target.
#     """
#     lidar = {
#         "position": np.array([0, 0, 0]),
#         "phi": np.array([0]),
#         "theta": np.array([0]),
#     }
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#             "speed": np.array([1, 0, 0]),
#         }
#     ]
#     rays_t0 = lidar_scene(lidar, targets, t=0)
#     rays_t1 = lidar_scene(lidar, targets, t=1)
#     assert not np.allclose(rays_t0, rays_t1)  # Check if rays change due to movement


# def test_lidar_scene_target_rotation():
#     """
#     Test with a rotating target.
#     """
#     lidar = {
#         "position": np.array([0, 0, 0]),
#         "phi": np.array([0]),
#         "theta": np.array([0]),
#     }
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#             "rotation_rate": np.array([10, 0, 0]),
#         }
#     ]
#     rays_t0 = lidar_scene(lidar, targets, t=0)
#     rays_t1 = lidar_scene(lidar, targets, t=1)
#     assert not np.allclose(rays_t0, rays_t1)  # Check if rays change due to rotation


# def test_lidar_scene_invalid_unit():
#     """
#     Test with an invalid unit for the target model.
#     """
#     lidar = {
#         "position": np.array([0, 0, 0]),
#         "phi": np.array([0]),
#         "theta": np.array([0]),
#     }
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#             "unit": "invalid",
#         }
#     ]
#     with pytest.raises(Exception):
#         lidar_scene(lidar, targets)


# def test_lidar_scene_missing_model():
#     """
#     Test with a missing model file.
#     """
#     lidar = {
#         "position": np.array([0, 0, 0]),
#         "phi": np.array([0]),
#         "theta": np.array([0]),
#     }
#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     with pytest.raises(Exception):
#         lidar_scene(lidar, targets)


def test_lidar():
    """
    This function tests the Lidar point cloud simulator
    """
    ground = {
        "model": "./models/surface_60x60.stl",
        "location": (0, 0, 0),
        "rotation_axis": (0, 0, 0),
        "rotation_angle": 0,
        "speed": (0, 0, 0),
    }

    targets = [ground]

    lidar = {
        "position": [0, 0, 1.5],
        "phi": np.arange(0, 360, 60),
        "theta": np.array([110, 120]),
    }

    points = lidar_scene(lidar, targets)

    npt.assert_almost_equal(
        points["positions"],
        np.array(
            [
                [4.1212144e00, 0.0000000e00, 1.1920929e-07],
                [2.5980759e00, 0.0000000e00, 1.1920929e-07],
                [2.0606072e00, 3.5690768e00, 1.1920929e-07],
                [1.2990378e00, 2.2499995e00, 1.1920929e-07],
                [-2.0606077e00, 3.5690765e00, 1.1920929e-07],
                [-1.2990381e00, 2.2499995e00, 1.1920929e-07],
                [-4.1212144e00, -3.6028803e-07, 1.1920929e-07],
                [-2.5980759e00, -2.2713100e-07, 1.1920929e-07],
                [-2.0606070e00, -3.5690768e00, 1.1920929e-07],
                [-1.2990375e00, -2.2499995e00, 1.1920929e-07],
                [2.0606070e00, -3.5690768e00, 1.1920929e-07],
                [1.2990375e00, -2.2499995e00, 1.1920929e-07],
            ]
        ),
        decimal=3,
    )

    npt.assert_almost_equal(
        points["directions"],
        np.array(
            [
                [9.3969262e-01, 0.0000000e00, 3.4202024e-01],
                [8.6602539e-01, 0.0000000e00, 5.0000006e-01],
                [4.6984628e-01, 8.1379771e-01, 3.4202024e-01],
                [4.3301266e-01, 7.5000000e-01, 5.0000006e-01],
                [-4.6984637e-01, 8.1379765e-01, 3.4202024e-01],
                [-4.3301275e-01, 7.5000000e-01, 5.0000006e-01],
                [-9.3969262e-01, -8.2150535e-08, 3.4202024e-01],
                [-8.6602539e-01, -7.5710346e-08, 5.0000006e-01],
                [-4.6984622e-01, -8.1379771e-01, 3.4202024e-01],
                [-4.3301257e-01, -7.5000000e-01, 5.0000006e-01],
                [4.6984622e-01, -8.1379771e-01, 3.4202024e-01],
                [4.3301257e-01, -7.5000000e-01, 5.0000006e-01],
            ]
        ),
        decimal=3,
    )
    print(points)
