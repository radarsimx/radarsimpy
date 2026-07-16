# distutils: language = c++
"""
A Python module for radar simulation

This module provides optimized Cython wrapper functions for interfacing with
the high-performance C++ radar simulation engine. Contains core conversion
and management functions for point targets, radar systems, mesh processing,
and RCS calculations.

The implementation is split across several ``include``-d source files that are
textually merged into this single extension module at compile time:

- ``cp_radarsimc_helpers.pyx``: shared constants and private helper functions
- ``cp_radarsimc_points.pyx``:  point-target wrapper (``cp_AddPoint``)
- ``cp_radarsimc_radar.pyx``:   transmitter/receiver/radar wrappers
- ``cp_radarsimc_mesh.pyx``:    mesh-target and scene-state wrappers

Helpers are included first so their module-level constants and ``cdef`` helpers
are available to the sections that follow.

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

include "cp_radarsimc_helpers.pyx"
include "cp_radarsimc_points.pyx"
include "cp_radarsimc_radar.pyx"
include "cp_radarsimc_mesh.pyx"
