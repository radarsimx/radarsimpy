# distutils: language = c++

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - PRESENT  Zhengyu Peng
# E-mail: zpeng.me@gmail.com
# Website: https://zpeng.me

# `                      `
# -:.                  -#:
# -//:.              -###:
# -////:.          -#####:
# -/:.://:.      -###++##:
# ..   `://:-  -###+. :##:
#        `:/+####+.   :##:
# .::::::::/+###.     :##:
# .////-----+##:    `:###:
#  `-//:.   :##:  `:###/.
#    `-//:. :##:`:###/.
#      `-//:+######/.
#        `-/+####/.
#          `+##+.
#           :##:
#           :##:
#           :##:
#           :##:
#           :##:
#            .+:


from radarsimpy.includes.radarsimc cimport Target, PointCloud
from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport float_t, int_t, vector

import meshio
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
            Array of phi scanning angles (deg) 
        - **theta** (*numpy.1darray*) --
            Array of theta scanning angles (deg)

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

    cdef int_t idx

    for idx in range(0, len(targets)):
        t_mesh = meshio.read(targets[idx]['model'])

        points_mv = t_mesh.points.astype(np_float)
        cells_mv = t_mesh.cells[0].data.astype(np.int32)

        origin_mv = np.array(targets[idx].get('origin', (0, 0, 0)), dtype=np_float)

        location_mv = np.array(targets[idx].get('location', (0, 0, 0)), dtype=np_float) + \
            t*np.array(targets[idx].get('speed', (0, 0, 0)), dtype=np_float)
        speed_mv = np.array(targets[idx].get('speed', (0, 0, 0)), dtype=np_float)

        rotation_mv = np.radians(
            np.array(targets[idx].get('rotation', (0, 0, 0)), dtype=np_float) + \
            t*np.array(targets[idx].get('rotation_rate', (0, 0, 0)), dtype=np_float)
            )
        rotation_rate_mv = np.radians(
            np.array(targets[idx].get('rotation_rate', (0, 0, 0)), dtype=np_float)
            )

        pointcloud_c.AddTarget(Target[float_t](&points_mv[0, 0],
                                             &cells_mv[0, 0],
                                             <int_t> cells_mv.shape[0],
                                             Vec3[float_t](& origin_mv[0]),
                                             Vec3[float_t](& location_mv[0]),
                                             Vec3[float_t](& speed_mv[0]),
                                             Vec3[float_t](& rotation_mv[0]),
                                             Vec3[float_t](& rotation_rate_mv[0]),
                                             <bool> targets[idx].get('is_ground', False)))

    cdef float_t[:] phi = np.radians(np.array(lidar['phi'], dtype=np_float))
    cdef float_t[:] theta = np.radians(np.array(lidar['theta'], dtype=np_float))

    cdef vector[float_t] phi_vector
    phi_vector.reserve(phi.shape[0])

    cdef vector[float_t] theta_vector
    theta_vector.reserve(theta.shape[0])

    for idx in range(0, phi.shape[0]):
        phi_vector.push_back(phi[idx])

    for idx in range(0, theta.shape[0]):
        theta_vector.push_back(theta[idx])

    pointcloud_c.Sbr(phi_vector,
                   theta_vector,
                   Vec3[float_t](<float_t> lidar['position'][0],
                                 <float_t> lidar['position'][1],
                                 <float_t> lidar['position'][2])
    )

    ray_type = np.dtype([('positions', np_float, (3,)),
                         ('directions', np_float, (3,))])

    rays = np.zeros(pointcloud_c.cloud_.size(), dtype=ray_type)

    for idx in range(0, <int_t> pointcloud_c.cloud_.size()):
        rays[idx]['positions'][0] = pointcloud_c.cloud_[idx].location_[1][0]
        rays[idx]['positions'][1] = pointcloud_c.cloud_[idx].location_[1][1]
        rays[idx]['positions'][2] = pointcloud_c.cloud_[idx].location_[1][2]
        rays[idx]['directions'][0] = pointcloud_c.cloud_[idx].direction_[1][0]
        rays[idx]['directions'][1] = pointcloud_c.cloud_[idx].direction_[1][1]
        rays[idx]['directions'][2] = pointcloud_c.cloud_[idx].direction_[1][2]

    return rays
