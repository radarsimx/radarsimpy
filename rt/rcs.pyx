# distutils: language = c++
# cython: language_level=3
"""
    This script contains classes that define all the parameters for
    a radar system

    This script requires that `numpy` be installed within the Python
    environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * Transmitter - A class defines parameters of a radar transmitter
    * Receiver - A class defines parameters of a radar receiver
    * Radar - A class defines basic parameters of a radar system

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - 2020  Zhengyu Peng
    E-mail: zpeng.me@gmail.com
    Website: https://zpeng.me

    `                      `
    -:.                  -#:
    -//:.              -###:
    -////:.          -#####:
    -/:.://:.      -###++##:
    ..   `://:-  -###+. :##:
           `:/+####+.   :##:
    .::::::::/+###.     :##:
    .////-----+##:    `:###:
     `-//:.   :##:  `:###/.
       `-//:. :##:`:###/.
         `-//:+######/.
           `-/+####/.
             `+##+.
              :##:
              :##:
              :##:
              :##:
              :##:
               .+:

"""

cimport cython

from radarsimpy.includes.radarsimc cimport Radarsimc

import numpy as np
cimport numpy as np
from stl import mesh

from radarsimpy.includes.radarsimc cimport Target, Rcs
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t
from radarsimpy.includes.zpvector cimport Vec3


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef rcs_sbr(model, phi, theta, f, pol=[0, 0, 1], density=10):
    trig_mesh = mesh.Mesh.from_file(model)
    cdef float_t[:, :, :] vectors = trig_mesh.vectors.astype(np.float32)

    cdef Rcs[float_t] rcs

    rcs = Rcs[float_t](Target[float_t](&vectors[0, 0, 0], <int_t> vectors.shape[0]),
                      <float_t> phi/180*np.pi,
                      <float_t> theta/180*np.pi,
                      Vec3[float_t](<float_t> pol[0], <float_t> pol[1], <float_t> pol[2]),
                      <float_t> f,
                      <float_t> density)

    return rcs.CalculateRcs()
