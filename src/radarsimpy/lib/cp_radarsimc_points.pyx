# distutils: language = c++
"""
Point-target wrapper for cp_radarsimc.

This file is textually merged into the ``cp_radarsimc`` extension module via
``include`` (see cp_radarsimc.pyx). It converts Python point-scatterer
parameters into the C++ points manager representation.

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

# Standard imports
import numpy as np

# Cython imports
cimport cython
cimport numpy as np
from radarsimpy.includes.radarsimc cimport (
    PointsManager,
    Mem_Copy, Mem_Copy_Vec3,
)
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t, float_t, vector


# ============================================================================
# Point Target
# ============================================================================

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddPoint(location, speed, rcs, phase, shape, PointsManager[float_t] * points_manager):
    """
    Add a point scatterer to the radar simulation.

    :param list location:
        Target's location (m), [x, y, z]
    :param list speed:
        Target's velocity (m/s), [x, y, z]
    :param float rcs:
        Target's RCS (dBsm)
    :param float phase:
        Target's phase (deg)
    :param tuple shape:
        Shape of the time matrix (channels, frames, pulses)
    :param PointsManager[float_t] * points_manager:
        Pointer to C++ points manager
    :raises: ValueError for invalid input dimensions or types
    """
    # Input validation - check basic requirements
    if not hasattr(location, '__len__') or len(location) != 3:
        raise ValueError("location must be a 3-element array [x, y, z]")
    if not hasattr(speed, '__len__') or len(speed) != 3:
        raise ValueError("speed must be a 3-element array [x, y, z]")
    if not hasattr(shape, '__len__') or len(shape) != 3:
        raise ValueError("shape must be a 3-element tuple (channels, frames, pulses)")

    # Variable declarations for C++ vector storage
    cdef vector[Vec3[float_t]] loc_vt
    cdef vector[float_t] rcs_vt, phs_vt

    # Memory view declarations for time-varying parameters
    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] rcs_mv, phs_mv

    # Calculate total buffer size for time-varying arrays
    cdef int_t bbsize_c = <int_t>(shape[0]*shape[1]*shape[2])

    # Convert speed to memory view (constant for all time steps)
    cdef float_t[:] speed_mv = np.array(speed, dtype=np_float)
    cdef float_t[:] location_mv

    # Check if there are any time varying parameters
    if any(np.size(var) > 1 for var in list(location) + [rcs, phase]):

        locx_mv = _broadcast_time_varying(location[0], shape)
        locy_mv = _broadcast_time_varying(location[1], shape)
        locz_mv = _broadcast_time_varying(location[2], shape)
        rcs_mv = _broadcast_time_varying(rcs, shape)
        phs_mv = _broadcast_time_varying(np.radians(phase), shape)

        Mem_Copy(&rcs_mv[0,0,0], bbsize_c, rcs_vt)
        Mem_Copy(&phs_mv[0,0,0], bbsize_c, phs_vt)
        Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)

    else:
        location_mv = np.array(location, dtype=np_float)

        loc_vt.push_back(Vec3[float_t](&location_mv[0]))
        rcs_vt.push_back(<float_t> rcs)
        phs_vt.push_back(<float_t> np.radians(phase))

    points_manager[0].AddPoint(
        loc_vt,
        Vec3[float_t](&speed_mv[0]),
        rcs_vt,
        phs_vt
    )
