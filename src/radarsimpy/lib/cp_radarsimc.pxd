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


# Core conversion functions with exception handling
cdef Point[float_t] cp_Point(location, speed, rcs, phase, shape) except *
cdef Radar[double, float_t] cp_Radar(radar, frame_start_time) except *
cdef Target[float_t] cp_Target(radar, target, timestamp, mesh_module) except *
cdef Target[float_t] cp_RCS_Target(target, mesh_module) except *

# Helper functions for internal use
cdef inline void _validate_vector_3d(object vector, str name) except *
cdef inline float_t _safe_unit_conversion(str unit) except *
