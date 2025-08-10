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
# High-performance vector classes for 3D geometric calculations in radar simulation
#------------------------------------------------------------------------------
cdef extern from "rsvector.hpp" namespace "rsv" nogil:
    
    # 3D Vector Class - Primary geometric primitive for radar calculations
    cdef cppclass Vec3[T]:
        """
        3D vector class for positions, directions, and transformations
        
        Template parameter T: Numeric type (float, double, int, etc.)
        
        Components:
        - [0]: X-coordinate (typically East/Right)
        - [1]: Y-coordinate (typically North/Forward) 
        - [2]: Z-coordinate (typically Up/Elevation)
        
        Coordinate Systems:
        - Cartesian: (x, y, z) in meters
        - Spherical: (range, azimuth, elevation) - use conversion functions
        - Platform: Body-fixed coordinates relative to radar platform
        """
        
        # Constructors
        Vec3() except +                                          # Zero vector constructor
        Vec3(const T&, const T&, const T&) except +             # Component constructor (x, y, z)
        Vec3(T*) except +                                        # Array/pointer constructor
        
        # Operators
        Vec3& operator=(const Vec3&)                             # Assignment operator
        T& operator[](const unsigned int&)                       # Index operator for component access

    # 2D Vector Class - For planar calculations and projections
    cdef cppclass Vec2[T]:
        """
        2D vector class for planar calculations and range-angle representations
        
        Template parameter T: Numeric type (float, double, int, etc.)
        
        Components:
        - [0]: X-coordinate or Range
        - [1]: Y-coordinate or Angle
        
        Common Uses:
        - Range-Doppler maps: (range_bin, doppler_bin)
        - Angular coordinates: (azimuth, elevation)
        - Image coordinates: (pixel_x, pixel_y)
        - Filter parameters: (min_value, max_value)
        """
        
        # Constructors
        Vec2() except +                                          # Zero vector constructor
        Vec2(const T&, const T&) except +                       # Component constructor (x, y)
        Vec2(T*) except +                                        # Array/pointer constructor
        
        # Operators
        Vec2& operator=(const Vec2&)                             # Assignment operator
        T& operator[](const unsigned int&)                       # Index operator for component access

#------------------------------------------------------------------------------
# End of RadarSimPy Vector Library Interface
#------------------------------------------------------------------------------
