# distutils: language = c++
"""
RadarSimPy Vector Library - Geometric Vector Operations

This module provides efficient 2D and 3D vector classes optimized for radar simulation
calculations including coordinate transformations, distance computations, and geometric
operations in 3D space.

Key Features:
- **Template-based design**: Support for different numeric types (float, double, int)
- **Memory efficient**: Direct memory access and pointer-based initialization
- **High performance**: Optimized for real-time radar calculations
- **Coordinate systems**: Support for Cartesian, spherical, and radar-specific coordinates

Common Usage:
- Position vectors for targets, radar platforms, and antennas
- Direction vectors for radar beams and ray tracing
- Velocity vectors for moving targets and platforms
- Angular measurements (azimuth, elevation, rotation angles)

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

#------------------------------------------------------------------------------
# RadarSimPy Vector Library Interface
#------------------------------------------------------------------------------
cdef extern from "rsvector.hpp" namespace "rsv" nogil:

    # 3D vector for positions, directions, velocities, and angular measurements.
    # Template T: Numeric type (float, double, int, etc.)
    # Components: [0] X (East/Right), [1] Y (North/Forward), [2] Z (Up/Elevation)
    cdef cppclass Vec3[T]:
        Vec3() except +                                          # Zero vector constructor
        Vec3(const T&, const T&, const T&) except +             # Component constructor (x, y, z)
        Vec3(T*) except +                                        # Array/pointer constructor
        Vec3& operator=(const Vec3&)                             # Assignment operator
        T& operator[](const unsigned int&)                       # Component index access

    # 2D vector for planar calculations, range-angle pairs, and filter bounds.
    # Template T: Numeric type (float, double, int, etc.)
    # Components: [0] X/range/min, [1] Y/angle/max
    cdef cppclass Vec2[T]:
        Vec2() except +                                          # Zero vector constructor
        Vec2(const T&, const T&) except +                       # Component constructor (x, y)
        Vec2(T*) except +                                        # Array/pointer constructor
        Vec2& operator=(const Vec2&)                             # Assignment operator
        T& operator[](const unsigned int&)                       # Component index access

#------------------------------------------------------------------------------
# End of RadarSimPy Vector Library Interface
#------------------------------------------------------------------------------
