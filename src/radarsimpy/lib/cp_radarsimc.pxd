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

from radarsimpy.includes.radarsimc cimport RxChannel
from radarsimpy.includes.radarsimc cimport TxChannel
from radarsimpy.includes.radarsimc cimport Transmitter
from radarsimpy.includes.radarsimc cimport Radar
from radarsimpy.includes.radarsimc cimport Target
from radarsimpy.includes.radarsimc cimport Point
from radarsimpy.includes.type_def cimport float_t


cdef Point[float_t] cp_Point(location, speed, rcs, phase, shape)
cdef Transmitter[float_t] cp_Transmitter(radar)
cdef TxChannel[float_t] cp_TxChannel(tx, tx_idx)
cdef RxChannel[float_t] cp_RxChannel(rx, rx_idx)
cdef Radar[float_t] cp_Radar(radar)
cdef Target[float_t] cp_Target(radar, target, shape)
cdef Target[float_t] cp_RCS_Target(target)
