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
from radarsimpy.includes.radarsimc cimport Radar
from radarsimpy.includes.radarsimc cimport RxChannel
from radarsimpy.includes.radarsimc cimport TxChannel
from radarsimpy.includes.radarsimc cimport Transmitter
from radarsimpy.includes.radarsimc cimport Target
from radarsimpy.includes.radarsimc cimport Point
from libcpp.complex cimport complex as cpp_complex
from libcpp cimport bool
from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport float_t, int_t
from radarsimpy.includes.type_def cimport vector
import numpy as np
cimport numpy as np


cdef Point[float_t] cp_Point(location, speed, rcs, phase, shape)
cdef Transmitter[float_t] cp_Transmitter(radar)
cdef TxChannel[float_t] cp_TxChannel(tx, tx_idx)
cdef RxChannel[float_t] cp_RxChannel(rx, rx_idx)
cdef Target[float_t] cp_Target(radar, target, shape)
