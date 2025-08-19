# distutils: language = c++
"""
The Python Module for Lidar Simulations

This module provides tools for simulating a lidar system in complex 3D environments. It leverages a high-performance C++ backend integrated with Python to support large-scale simulations with high accuracy and computational efficiency. The module includes features for simulating ray interactions with targets, generating point clouds, and modeling the dynamics of targets in the scene.

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

# Core imports
from libcpp.memory cimport shared_ptr, make_shared
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport cython

# RadarSimX imports
from radarsimpy.includes.radarsimc cimport LidarSimulator, TargetsManager
from radarsimpy.includes.radarsimc cimport Mem_Copy
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport float_t, int_t, vector

from radarsimpy.mesh_kit import import_mesh_module, load_mesh

np.import_array()
np_float = np.float32


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sim_lidar(lidar, targets, frame_time=0):
    """
    sim_lidar(lidar, targets, frame_time=0)

    Simulate a Lidar scene and compute ray interactions with targets.

    This function simulates a Lidar scanning scene in a 3D environment, calculating the interaction of Lidar rays with the provided targets. It handles both static and dynamic targets, allowing for customizable positions, velocities, and orientations of objects in the scene. The simulation produces a set of rays representing the Lidar's perception of the environment.

    :param dict lidar:
        Lidar configuration parameters. The following keys are required:

        - **position** (*numpy.ndarray*):  
          The 3D position of the Lidar in meters (m), specified as [x, y, z].
        - **phi** (*numpy.ndarray*):  
          Array of phi scanning angles in degrees (°). Phi represents the horizontal scanning angles in the Lidar's field of view. The total number of scanning directions is determined by the combination of phi and theta angles.
        - **theta** (*numpy.ndarray*):  
          Array of theta scanning angles in degrees (°). Theta represents the vertical scanning angles in the Lidar's field of view. The total number of scanning directions is computed as:  
          ``len(phi) * len(theta)``.

    :param list[dict] targets:
        A list of target objects in the scene. Each target is represented as a dictionary containing the following keys:

        - **model** (*str*):  
          File path to the target model (3D object) in the scene.
        - **origin** (*numpy.ndarray*):  
          The origin position (rotation and translation center) of the target model in meters (m), specified as [x, y, z].  
          Default: ``[0, 0, 0]``.
        - **location** (*numpy.ndarray*):  
          The 3D location of the target in meters (m), specified as [x, y, z].  
          Default: ``[0, 0, 0]``.
        - **speed** (*numpy.ndarray*):  
          Speed vector of the target in meters per second (m/s), specified as [vx, vy, vz].  
          Default: ``[0, 0, 0]``.
        - **rotation** (*numpy.ndarray*):  
          The angular orientation of the target in degrees (°), specified as [yaw, pitch, roll].  
          Default: ``[0, 0, 0]``.
        - **rotation_rate** (*numpy.ndarray*):  
          The angular rotation rate of the target in degrees per second (°/s), specified as [yaw rate, pitch rate, roll rate].  
          Default: ``[0, 0, 0]``.
        - **unit** (*str*):  
          The unit system for the target model's geometry.  
          Supported values: ``mm``, ``cm``, ``m``.  
          Default: ``m``.

    :param float frame_time:
        Simulation timestamp in seconds (s). This parameter determines the time reference for the Lidar's scanning operation and target positions.  
        Default: ``0``.

    :return:  
        Simulated Lidar rays based on the provided configuration and targets.

    :rtype:  
        numpy.ndarray - A structured array representing the Lidar ray interactions with the scene, including details such as ray origins, directions, and intersections.
    """
    cdef LidarSimulator[float_t] lidar_sim_c

    cdef shared_ptr[TargetsManager[float_t]] targets_manager = make_shared[TargetsManager[float_t]]()

    # Memory view declarations
    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv
    cdef float_t[:] origin_mv
    cdef float_t[:] speed_mv
    cdef float_t[:] location_mv
    cdef float_t[:] rotation_mv
    cdef float_t[:] rotation_rate_mv
    cdef float_t scale
    cdef int_t idx_c

    # Process targets
    mesh_module = import_mesh_module()
    for idx_c in range(0, len(targets)):
        # Unit conversion
        unit = targets[idx_c].get("unit", "m")
        scale = 1000 if unit == "mm" else 100 if unit == "cm" else 1

        mesh_data = load_mesh(targets[idx_c]["model"], scale, mesh_module)
        points_mv = mesh_data["points"].astype(np_float)
        cells_mv = mesh_data["cells"].astype(np.int32)

        # Target parameters
        origin_mv = np.array(targets[idx_c].get("origin", (0, 0, 0)), dtype=np_float)
        temp_location = np.array(targets[idx_c].get("location", (0, 0, 0))) + frame_time*np.array(targets[idx_c].get("speed", (0, 0, 0)))
        location_mv = temp_location.astype(np_float)
        speed_mv = np.array(targets[idx_c].get("speed", (0, 0, 0)), dtype=np_float)
        
        temp_rotation = np.array(targets[idx_c].get("rotation", (0, 0, 0))) + frame_time*np.array(targets[idx_c].get("rotation_rate", (0, 0, 0)))
        rotation_mv = np.radians(temp_rotation.astype(np_float))
        rotation_rate_mv = np.radians(
            np.array(targets[idx_c].get("rotation_rate", (0, 0, 0)), dtype=np_float)
        )

        targets_manager.get()[0].AddTargetSimple(&points_mv[0, 0],
                                    &cells_mv[0, 0],
                                    <int_t> cells_mv.shape[0],
                                    Vec3[float_t](&origin_mv[0]),
                                    Vec3[float_t](&location_mv[0]),
                                    Vec3[float_t](&speed_mv[0]),
                                    Vec3[float_t](&rotation_mv[0]),
                                    Vec3[float_t](&rotation_rate_mv[0]),
                                    <bool> targets[idx_c].get("skip_diffusion", False))

    # Lidar parameters
    cdef float_t[:] phi_mv = np.radians(np.array(lidar["phi"], dtype=np_float))
    cdef float_t[:] theta_mv = np.radians(np.array(lidar["theta"], dtype=np_float))
    cdef float_t[:] position_mv = np.array(lidar["position"], dtype=np_float)

    cdef vector[float_t] phi_vt
    Mem_Copy(&phi_mv[0], <int_t>(phi_mv.shape[0]), phi_vt)

    cdef vector[float_t] theta_vt
    Mem_Copy(&theta_mv[0], <int_t>(theta_mv.shape[0]), theta_vt)

    # Perform ray tracing
    lidar_sim_c.Run(targets_manager,
                    phi_vt,
                    theta_vt,
                    Vec3[float_t](&position_mv[0]))

    # Prepare output
    ray_type = np.dtype([("positions", np_float, (3,)),
                        ("directions", np_float, (3,))])
    rays = np.zeros(lidar_sim_c.cloud_.size(), dtype=ray_type)

    for idx_c in range(0, <int_t> lidar_sim_c.cloud_.size()):
        rays[idx_c]["positions"][0] = lidar_sim_c.cloud_[idx_c].location_[1][0]
        rays[idx_c]["positions"][1] = lidar_sim_c.cloud_[idx_c].location_[1][1]
        rays[idx_c]["positions"][2] = lidar_sim_c.cloud_[idx_c].location_[1][2]
        rays[idx_c]["directions"][0] = lidar_sim_c.cloud_[idx_c].direction_[1][0]
        rays[idx_c]["directions"][1] = lidar_sim_c.cloud_[idx_c].direction_[1][1]
        rays[idx_c]["directions"][2] = lidar_sim_c.cloud_[idx_c].direction_[1][2]

    return rays
