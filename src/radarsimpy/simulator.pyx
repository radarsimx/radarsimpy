# distutils: language = c++
"""
The Python Module for Advanced Radar and Lidar Simulations

This module provides tools for simulating and analyzing radar and lidar systems in complex 3D environments.
The module supports a wide range of functionalities, including:

1. **Lidar Simulations**:

   - Simulate Lidar systems in dynamic or static 3D environments.
   - Generate point clouds and compute ray interactions with targets.
   - Model the dynamics of targets, such as motion and rotation.

2. **Radar Simulations**:

   - Simulate radar baseband responses for complex scenes.
   - Handle point targets, 3D mesh objects, interference modeling, and noise simulation.
   - Perform advanced ray-tracing for high-fidelity radar analysis.

3. **Radar Cross Section (RCS) Calculations**:

   - Calculate the RCS of targets using the Shooting and Bouncing Rays (SBR) method.
   - Model electromagnetic wave scattering from complex 3D geometries.
   - Support for defining target materials and permittivity properties.

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

include "simulator_radar.pyx"
include "simulator_lidar.pyx"
include "simulator_rcs.pyx"


def gpu_available():
    """
    gpu_available()

    Whether this build can actually run a simulation on the GPU.

    Two separate things have to be true, and this reports their conjunction:
    the module has to have been compiled with CUDA support, and the machine has
    to expose at least one usable CUDA device. Either one missing means the
    simulators run on the CPU.

    This is what ``sim_radar(..., device="auto")`` calls to choose a device, so
    it also answers "what will auto pick?". ``sim_lidar`` and ``sim_rcs`` have
    no device argument and always follow it.

    The underlying device probe runs once and is cached for the lifetime of the
    process, so this is cheap to call repeatedly. It follows
    ``CUDA_VISIBLE_DEVICES``, which means a child process started with that set
    to ``-1`` will report ``False`` here even on a CUDA build.

    :return:
        ``True`` when a CUDA device is usable, ``False`` otherwise.
    :rtype: bool

    :example:
        >>> from radarsimpy.simulator import gpu_available
        >>> gpu_available()
        True
    """
    # Not `bool(...)`: the included simulator files cimport the C++ `bool`,
    # which shadows the Python builtin in this shared namespace. A `bint`
    # converts to a genuine Python bool on return.
    cdef bint available = _gpu_available_c()
    return available
