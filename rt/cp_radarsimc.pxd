"""
    A Python module for radar simulation

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - 2021  Zhengyu Peng
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
import numpy as np
cimport numpy as np

from radarsimpy.includes.type_def cimport vector
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t
from radarsimpy.includes.zpvector cimport Vec3
from libcpp cimport bool
from libcpp.complex cimport complex as cpp_complex

from radarsimpy.includes.radarsimc cimport Point
from radarsimpy.includes.radarsimc cimport Target
from radarsimpy.includes.radarsimc cimport Transmitter
from radarsimpy.includes.radarsimc cimport TxChannel
from radarsimpy.includes.radarsimc cimport RxChannel

cdef Point[float_t] cp_Point(location, speed, rcs, phase, shape)
cdef Transmitter[float_t] cp_Transmitter(radar, density)
cdef TxChannel[float_t] cp_TxChannel(tx, tx_idx)
cdef RxChannel[float_t] cp_RxChannel(rx, rx_idx)
cdef Target[float_t] cp_Target(radar, target, shape)
