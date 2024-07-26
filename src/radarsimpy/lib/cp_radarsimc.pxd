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

from radarsimpy.includes.radarsimc cimport Radar
from radarsimpy.includes.radarsimc cimport Target
from radarsimpy.includes.radarsimc cimport Point
from radarsimpy.includes.type_def cimport float_t


cdef Point[float_t] cp_Point(location, speed, rcs, phase, shape)
cdef Radar[float_t] cp_Radar(radar, frame_size, frame_start_time)
cdef Target[float_t] cp_Target(radar, target, timestamp, shape)
cdef Target[float_t] cp_RCS_Target(target)
