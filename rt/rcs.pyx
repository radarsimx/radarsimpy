# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3

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


cimport cython

import numpy as np
cimport numpy as np
# from stl import mesh
import meshio

from radarsimpy.includes.radarsimc cimport Target, Rcs
from radarsimpy.includes.type_def cimport uint64_t, int_t
from radarsimpy.includes.zpvector cimport Vec3


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rcs_sbr(model, f, obs_phi, obs_theta, inc_phi=None, inc_theta=None, pol=[0, 0, 1], density=1):
    """
    rcs_sbr(model, f, obs_phi, obs_theta, inc_phi=None, inc_theta=None, pol=[0, 0, 1], density=1)

    Calculate target's RCS by using raytracing

    :param str model:
        Path of the model
    :param float phi:
        Observation angle phi (deg)
    :param float theta:
        Observation angle theta (deg)
    :param float f:
        Center frequency (Hz)
    :param list pol:
        Polarization [x, y, z]. ``default [0, 0, 1]``
    :param float density:
        Ray density (number of rays per wavelength). ``default 10``
    
    :return: Target's RCS (m^2), use 10*log10(RCS) to convert to dBsm
    :rtype: float
    """
    trig_mesh = meshio.read(model)
    cdef double[:, :] points = trig_mesh.points.astype(np.float64)
    cdef uint64_t[:, :] cells = trig_mesh.cells[0].data.astype(np.uint64)

    if inc_phi is None:
        inc_phi = obs_phi

    if inc_theta is None:
        inc_theta = obs_theta
    
    inc_phi_rad = np.radians(inc_phi)
    inc_theta_rad = np.radians(inc_theta)
    obs_phi_rad = np.radians(obs_phi)
    obs_theta_rad = np.radians(obs_theta)

    cdef Vec3[double] inc_dir = Vec3[double](
        <double> (np.sin(inc_theta_rad)*np.cos(inc_phi_rad)),
        <double> (np.sin(inc_theta_rad)*np.sin(inc_phi_rad)),
        <double> (np.cos(inc_theta_rad)))

    cdef Vec3[double] obs_dir = Vec3[double](
        <double> (np.sin(obs_theta_rad)*np.cos(obs_phi_rad)),
        <double> (np.sin(obs_theta_rad)*np.sin(obs_phi_rad)),
        <double> (np.cos(obs_theta_rad)))

    cdef Rcs[double] rcs

    rcs = Rcs[double](Target[double](&points[0, 0], &cells[0, 0], <int_t> cells.shape[0]),
                      inc_dir,
                      obs_dir,
                      Vec3[double](<double> pol[0], <double> pol[1], <double> pol[2]),
                      <double> f,
                      <double> density)

    return rcs.CalculateRcs()
