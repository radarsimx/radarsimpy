# distutils: language = c++

#  ____           _            ____  _          __  __
# |  _ \ __ _  __| | __ _ _ __/ ___|(_)_ __ ___ \ \/ /
# | |_) / _` |/ _` |/ _` | '__\___ \| | '_ ` _ \ \  /
# |  _ < (_| | (_| | (_| | |   ___) | | | | | | |/  \
# |_| \_\__,_|\__,_|\__,_|_|  |____/|_|_| |_| |_/_/\_\

"""
A Python module for radar simulation

----------
RadarSimPy - A Radar Simulator Built with Python
Copyright (C) 2018 - PRESENT  radarsimx.com
E-mail: info@radarsimx.com
Website: https://radarsimx.com

"""


from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport vector
from radarsimpy.lib.cp_radarsimc cimport cp_RCS_Target
from radarsimpy.includes.radarsimc cimport Target, Rcs

import numpy as np

cimport cython
cimport numpy as np

np_float = np.float32


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rcs_sbr(targets,
              f,
              obs_phi,
              obs_theta,
              inc_phi=None,
              inc_theta=None,
              pol=[0, 0, 1],
              density=1):
    """
    rcs_sbr(targets, f, obs_phi, obs_theta, inc_phi=None, inc_theta=None, pol=[0, 0, 1], density=1)

    Calculate target's RCS by using shooting and bouncing rays (SBR)

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
        - **rotation** (*numpy.1darray*) --
            Target's angle (deg), [yaw, pitch, roll].
            ``default [0, 0, 0]``
        - **permittivity** (*complex*) --
            Target's permittivity. Perfect electric conductor (PEC) if not specified.

        }]
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
    cdef vector[Target[float]] targets_vt

    for idx_c in range(0, len(targets)):
        targets_vt.push_back(cp_RCS_Target(targets[idx_c]))

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

    rcs = Rcs[double](targets_vt,
                      inc_dir,
                      obs_dir,
                      Vec3[double](<double> pol[0], <double> pol[1], <double> pol[2]),
                      <double> f,
                      <double> density)

    return rcs.CalculateRcs()
