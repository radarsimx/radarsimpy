# distutils: language = c++
# cython: language_level=3

# This script contains classes that define all the parameters for
# a radar system

# This script requires that `numpy` be installed within the Python
# environment you are running this script in.

# This file can be imported as a module and contains the following
# class:

# * Transmitter - A class defines parameters of a radar transmitter
# * Receiver - A class defines parameters of a radar receiver
# * Radar - A class defines basic parameters of a radar system

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - 2020  Zhengyu Peng
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


cimport cython

from libc.math cimport sin, cos, sqrt, atan, atan2, acos, pow, fmax, M_PI
from libcpp cimport bool

from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t, vector
from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.radarsimc cimport Target, PointCloud

import numpy as np
cimport numpy as np
from stl import mesh

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lidar_scene(lidar, targets, t=0):
    """
    Alias: ``radarsimpy.rt.lidar_scene()``
    
    Lidar scene simulator

    :param dict lidar:
        Lidar configuration
    :param list[dict] targets:
        Target list
    :param time t:
        Amplitude correction

    :return: rays
    :rtype: numpy.array
    """
    cdef PointCloud[float_t] pointcloud
    
    cdef float_t[:,:,:] mesh_memview
    cdef float_t[:] origin
    cdef float_t[:] speed
    cdef float_t[:] location
    cdef float_t[:] rotation
    cdef float_t[:] rotation_rate
    
    cdef int_t target_count = len(targets)
    cdef int_t idx
    
    for idx in range(0, target_count):
        t_mesh = mesh.Mesh.from_file(targets[idx]['model'])

        mesh_memview = t_mesh.vectors.astype(np.float32)

        origin = np.array(targets[idx].get('origin', (0,0,0)), dtype=np.float32)

        location = np.array(targets[idx].get('location', (0,0,0)), dtype=np.float32)+t*np.array(targets[idx].get('speed', (0,0,0)), dtype=np.float32)
        speed = np.array(targets[idx].get('speed', (0,0,0)), dtype=np.float32)

        rotation = np.array(targets[idx].get('rotation', (0,0,0)), dtype=np.float32)/180*np.pi+t*np.array(targets[idx].get('rotation_rate', (0,0,0)), dtype=np.float32)/180*np.pi
        rotation_rate = np.array(targets[idx].get('rotation_rate', (0,0,0)), dtype=np.float32)/180*np.pi

        pointcloud.AddTarget(Target[float_t](&mesh_memview[0,0,0],
            <int_t> mesh_memview.shape[0],
            Vec3[float_t](&origin[0]),
            Vec3[float_t](&location[0]),
            Vec3[float_t](&speed[0]),
            Vec3[float_t](&rotation[0]),
            Vec3[float_t](&rotation_rate[0]),
            <bool> targets[idx].get('is_ground', False)))

    cdef float_t[:] phi = np.array(lidar['phi'], dtype=np.float32)/180*np.pi
    cdef float_t[:] theta = np.array(lidar['theta'], dtype=np.float32)/180*np.pi

    cdef vector[float_t] phi_vector
    phi_vector.reserve(phi.shape[0])
    cdef vector[float_t] theta_vector
    theta_vector.reserve(theta.shape[0])

    for idx in range(0, phi.shape[0]):
        phi_vector.push_back(phi[idx])

    for idx in range(0, theta.shape[0]):
        theta_vector.push_back(theta[idx])
    
    pointcloud.Sbr(
        phi_vector,
        theta_vector,
        Vec3[float_t](<float_t> lidar['position'][0], <float_t> lidar['position'][1], <float_t> lidar['position'][2])
    )

    ray_type = np.dtype([('positions', np.float32, (3,)), ('directions', np.float32, (3,))])

    rays = np.zeros(pointcloud.cloud_.size(), dtype=ray_type)

    for idx in range(0, pointcloud.cloud_.size()):
        rays[idx]['positions'][0] = pointcloud.cloud_[idx].loc_[0]
        rays[idx]['positions'][1] = pointcloud.cloud_[idx].loc_[1]
        rays[idx]['positions'][2] = pointcloud.cloud_[idx].loc_[2]
        rays[idx]['directions'][0] = pointcloud.cloud_[idx].dir_[0]
        rays[idx]['directions'][1] = pointcloud.cloud_[idx].dir_[1]
        rays[idx]['directions'][2] = pointcloud.cloud_[idx].dir_[2]
    
    return rays
