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
