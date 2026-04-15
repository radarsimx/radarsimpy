# distutils: language = c++
"""
A Python module for radar simulation

This module provides Cython wrapper functions for interfacing with the C++ 
radar simulation engine.

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
from radarsimpy.includes.radarsimc cimport TargetsManager
from radarsimpy.includes.radarsimc cimport PointsManager
from radarsimpy.includes.type_def cimport float_t
from libcpp.memory cimport shared_ptr


# ============================================================================
# Point Target
# ============================================================================

# Add a point scatterer to the points manager for simulation
# Converts Python point parameters (location, speed, RCS, phase, shape) to C++
# Validates input parameters and updates the internal points collection
# Raises TypeError for invalid parameter types, ValueError for invalid values
cdef void cp_AddPoint(location, speed, rcs, phase, shape, PointsManager[float_t] * points_manager) except *


# ============================================================================
# Radar System (Transmitter, Receiver, Radar)
# ============================================================================

# Create a Radar system object for simulation 
# Converts Python radar config to C++ with complete transmitter/receiver setup
# Raises ValueError for invalid config, RuntimeError for setup failures
cdef shared_ptr[Radar[double, float_t]] cp_Radar(radar, frame_start_time) except *


# ============================================================================
# Mesh Targets (Radar Simulation and RCS)
# ============================================================================

# Add a complex target to the targets manager for radar simulation
# Converts Python target config to C++ with mesh processing and validation
# Handles time-varying parameters and applies transformations at given timestamp
# Raises ValueError for invalid target params, RuntimeError for mesh/FreeTier issues
cdef void cp_AddTarget(radar, target, timestamp, mesh_module, TargetsManager[float_t] * targets_manager) except *

# Create a Target object specifically optimized for RCS calculations
# Simplified target object without full dynamic simulation requirements
# Raises ValueError for invalid params, RuntimeError for mesh/FreeTier issues
cdef void cp_RCS_Target(target, mesh_module, TargetsManager[float_t] * targets_manager) except *
