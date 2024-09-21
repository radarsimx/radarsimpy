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

import warnings
from radarsimpy.simulator import sim_radar

cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef scene(radar, targets, density=1, level=None, log_path=None, debug=False, interf=None):
    """
    scene(radar, targets, density=1, level=None, log_path=None, debug=False, interf=None)

    **deprecated** Please use `simulator.sim_radar(radar, targets, density=1, level=None, log_path=None, debug=False, interf=None)`
    """

    warnings.warn("The `scene()` function has been deprecated, please use `sim_radar()`.", DeprecationWarning)

    return sim_radar(radar, targets, density=density, level=level, log_path=log_path, debug=debug, interf=interf)
