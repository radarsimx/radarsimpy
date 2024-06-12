# distutils: language = c++
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


from radarsimpy.includes.radarsimc cimport Target, PointCloud
from radarsimpy.includes.radarsimc cimport Mem_Copy
from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport float_t, int_t, vector

# import meshio
import numpy as np
from libcpp cimport bool

cimport cython
cimport numpy as np

np_float = np.float32


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lidar_scene(lidar, targets, t=0):
    """
    lidar_scene(lidar, targets, t=0)

    Lidar scene simulator

    :param dict lidar:
        Lidar configuration

        {

        - **position** (*numpy.1darray*) --
            Lidar's position (m). [x, y, z]
        - **phi** (*numpy.1darray*) --
            Array of phi scanning angles (deg). The total sweep angles are `the number of phi angles x the number of theta angles`
        - **theta** (*numpy.1darray*) --
            Array of theta scanning angles (deg). The total sweep angles are `the number of phi angles x the number of theta angles`

        }

    :param list[dict] targets:
        Target list

        [{

        - **model** (*str*) --
            Path to the target model
        - **origin** (*numpy.1darray*) --
            Origin position of the target model (m), [x, y, z].
            ``default [0, 0, 0]``
        - **location** (*numpy.1darray*) --
            Location of the target (m), [x, y, z].
            ``default [0, 0, 0]``
        - **speed** (*numpy.1darray*) --
            Speed of the target (m/s), [vx, vy, vz].
            ``default [0, 0, 0]``
        - **rotation** (*numpy.1darray*) --
            Target's angle (deg), [yaw, pitch, roll].
            ``default [0, 0, 0]``
        - **rotation_rate** (*numpy.1darray*) --
            Target's rotation rate (deg/s),
            [yaw rate, pitch rate, roll rate]
            ``default [0, 0, 0]``
        - **unit** (*str*) --
            Unit of target model. Supports `mm`, `cm`, and `m`. Default is `m`.

        }]

    :param float t:
        Simulation timestamp. ``default 0``

    :return: rays
    :rtype: numpy.array
    """
    cdef PointCloud[float_t] pointcloud_c

    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv
    cdef float_t[:] origin_mv
    cdef float_t[:] speed_mv
    cdef float_t[:] location_mv
    cdef float_t[:] rotation_mv
    cdef float_t[:] rotation_rate_mv
    cdef float_t scale

    cdef int_t idx_c

    for idx_c in range(0, len(targets)):
        unit = targets[idx_c].get("unit", "m")
        if unit == "m":
            scale = 1
        elif unit == "cm":
            scale = 100
        elif unit == "mm":
            scale = 1000
        else:
            scale = 1

        try:
            import pymeshlab
        except:
            try:
                import meshio
            except:
                raise("PyMeshLab is requied to process the 3D model.")
            else:
                t_mesh = meshio.read(targets[idx_c]["model"])
                points_mv = t_mesh.points.astype(np_float)/scale
                cells_mv = t_mesh.cells[0].data.astype(np.int32)
        else:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(targets[idx_c]["model"])
            t_mesh = ms.current_mesh()
            v_matrix = np.array(t_mesh.vertex_matrix())
            f_matrix = np.array(t_mesh.face_matrix())
            if np.isfortran(v_matrix):
                points_mv = np.ascontiguousarray(v_matrix).astype(np_float)/scale
                cells_mv = np.ascontiguousarray(f_matrix).astype(np.int32)
            ms.clear()

        origin_mv = np.array(targets[idx_c].get("origin", (0, 0, 0)), dtype=np_float)

        location_mv = np.array(targets[idx_c].get("location", (0, 0, 0)), dtype=np_float) + \
            t*np.array(targets[idx_c].get("speed", (0, 0, 0)), dtype=np_float)
        speed_mv = np.array(targets[idx_c].get("speed", (0, 0, 0)), dtype=np_float)

        rotation_mv = np.radians(
            np.array(targets[idx_c].get("rotation", (0, 0, 0)), dtype=np_float) + \
            t*np.array(targets[idx_c].get("rotation_rate", (0, 0, 0)), dtype=np_float)
            )
        rotation_rate_mv = np.radians(
            np.array(targets[idx_c].get("rotation_rate", (0, 0, 0)), dtype=np_float)
            )

        pointcloud_c.AddTarget(Target[float_t](&points_mv[0, 0],
                                               &cells_mv[0, 0],
                                               <int_t> cells_mv.shape[0],
                                               Vec3[float_t](&origin_mv[0]),
                                               Vec3[float_t](&location_mv[0]),
                                               Vec3[float_t](&speed_mv[0]),
                                               Vec3[float_t](&rotation_mv[0]),
                                               Vec3[float_t](&rotation_rate_mv[0]),
                                               <bool> targets[idx_c].get("is_ground", False)))

    cdef float_t[:] phi_mv = np.radians(np.array(lidar["phi"], dtype=np_float))
    cdef float_t[:] theta_mv = np.radians(np.array(lidar["theta"], dtype=np_float))
    cdef float_t[:] position_mv = np.array(lidar["position"], dtype=np_float)

    cdef vector[float_t] phi_vt
    Mem_Copy(&phi_mv[0], <int_t>(phi_mv.shape[0]), phi_vt)

    cdef vector[float_t] theta_vt
    Mem_Copy(&theta_mv[0], <int_t>(theta_mv.shape[0]), theta_vt)

    pointcloud_c.Sbr(phi_vt,
                     theta_vt,
                     Vec3[float_t](&position_mv[0]))

    ray_type = np.dtype([("positions", np_float, (3,)),
                         ("directions", np_float, (3,))])

    rays = np.zeros(pointcloud_c.cloud_.size(), dtype=ray_type)

    for idx_c in range(0, <int_t> pointcloud_c.cloud_.size()):
        rays[idx_c]["positions"][0] = pointcloud_c.cloud_[idx_c].location_[1][0]
        rays[idx_c]["positions"][1] = pointcloud_c.cloud_[idx_c].location_[1][1]
        rays[idx_c]["positions"][2] = pointcloud_c.cloud_[idx_c].location_[1][2]
        rays[idx_c]["directions"][0] = pointcloud_c.cloud_[idx_c].direction_[1][0]
        rays[idx_c]["directions"][1] = pointcloud_c.cloud_[idx_c].direction_[1][1]
        rays[idx_c]["directions"][2] = pointcloud_c.cloud_[idx_c].direction_[1][2]

    return rays
