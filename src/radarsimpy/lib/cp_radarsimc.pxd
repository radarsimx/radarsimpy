# distutils: language = c++
"""
A Python module for radar simulation

This module provides Cython wrapper functions for interfacing with the C++ 
radar simulation engine. It contains optimized functions for:

- Point target creation and management
- Radar system configuration and setup  
- Complex target modeling with mesh support
- Radar cross-section (RCS) calculations
- Time-varying parameter handling

The module is designed for high-performance radar simulation applications
with support for both trial and full versions with appropriate limitations.

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
from radarsimpy.includes.type_def cimport float_t, int_t
from libcpp.complex cimport complex as cpp_complex
from libcpp.memory cimport shared_ptr


# ============================================================================
# Core Conversion Functions
# ============================================================================


cdef void cp_AddPoint(location, speed, rcs, phase, shape, PointsManager[float_t] * points_manager) except *

# Create a Radar system object for simulation 
# Converts Python radar config to C++ with complete transmitter/receiver setup
# Raises ValueError for invalid config, RuntimeError for setup failures
cdef shared_ptr[Radar[double, float_t]] cp_Radar(radar, frame_start_time) except *

# Create a Target object specifically optimized for RCS calculations
# Simplified target object without full dynamic simulation requirements
# Raises ValueError for invalid params, RuntimeError for mesh/FreeTier issues
cdef void cp_RCS_Target(target, mesh_module, TargetsManager[float_t] * targets_manager) except *

cdef void cp_AddTarget(radar, target, timestamp, mesh_module, TargetsManager[float_t] * targets_manager) except *

# ============================================================================
# Helper Functions for Internal Use
# ============================================================================

# Validate that an input vector has exactly 3 elements
# Raises TypeError if not array-like, ValueError if wrong size
cdef void _validate_vector_3d(object vector, str name) except *

# Safely convert unit string to scale factor
# Supports 'm', 'cm', 'mm' units, raises ValueError for unsupported units  
cdef float_t _safe_unit_conversion(str unit) except *

# Check mesh size limits for free tier with detailed error messages
# Raises RuntimeError if FreeTier limitations are exceeded
cdef void _validate_mesh_for_free_tier(int_t num_faces) except *

# Convert permittivity value to complex float with validation
# Handles both 'PEC' string and numeric values
# Raises ValueError for invalid permittivity values
cdef cpp_complex[float_t] _convert_permittivity(object permittivity) except *

# Convert permeability value to complex float with validation  
# Raises ValueError for invalid permeability values
cdef cpp_complex[float_t] _convert_permeability(object permeability) except *

# Issue standardized deprecation warnings for renamed parameters
# Provides clear migration guidance and consistent warning format
cdef void _warn_deprecated_parameter(str old_param, str new_param) except *
