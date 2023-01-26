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


from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t
from radarsimpy.includes.radarsimc cimport Target, Rcs

import meshio
import numpy as np

cimport cython
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rcs_sbr(model,
              f,
              obs_phi,
              obs_theta,
              inc_phi=None,
              inc_theta=None,
              pol=[0, 0, 1],
              density=1):
    """
    rcs_sbr(model, f, obs_phi, obs_theta, inc_phi=None, inc_theta=None, pol=[0, 0, 1], density=1)

    Calculate target's RCS by using shooting and bouncing rays (SBR)

    :param str model:
        Path of the model file
    :param float f:
        Center frequency (Hz)
    :param float obs_phi:
        Observation angle phi (deg)
    :param float obs_theta:
        Observation angle theta (deg)
    :param float inc_phi:
        Incidence angle phi (deg).
        ``default None`` means ``inc_phi = obs_phi``
    :param float inc_theta:
        Incidence angle theta (deg).
        ``default None`` means ``inc_theta = obs_theta``
    :param list pol:
        Polarization [x, y, z].
        ``default [0, 0, 1]``
    :param float density:
        Ray density (number of rays per wavelength).
        ``default 1``

    :return: Target's RCS (m^2), use 10*log10(RCS) to convert to dBsm
    :rtype: float
    """
    t_mesh = meshio.read(model)
    cdef float_t[:, :] points_mv = t_mesh.points.astype(np.float32)
    cdef int_t[:, :] cells_mv = t_mesh.cells[0].data.astype(np.int32)

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

    rcs = Rcs[double](Target[float](&points_mv[0, 0], &cells_mv[0, 0], <int_t> cells_mv.shape[0]),
                      inc_dir,
                      obs_dir,
                      Vec3[double](<double> pol[0], <double> pol[1], <double> pol[2]),
                      <double> f,
                      <double> density)

    return rcs.CalculateRcs()
