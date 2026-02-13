# distutils: language = c++

"""
The Python Module for Radar Cross Section (RCS) Simulation

This module provides tools for simulating and calculating the Radar Cross Section (RCS) of targets using the Shooting and Bouncing Rays (SBR) method. By leveraging a high-performance C++ backend integrated with Python, it enables accurate and efficient modeling of electromagnetic wave scattering from complex 3D target geometries. The module is designed for radar researchers, engineers, and educators.

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

from libcpp.memory cimport shared_ptr, make_shared
# Standard library imports
import numpy as np

# Cython imports
cimport cython
cimport numpy as np

# Local imports
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport vector
from radarsimpy.includes.radarsimc cimport RcsSimulator, TargetsManager
from radarsimpy.lib.cp_radarsimc cimport cp_RCS_Target
from libcpp.complex cimport complex as cpp_complex

from radarsimpy.license import is_licensed

from radarsimpy.mesh_kit import import_mesh_module

# Initialize NumPy
np.import_array()


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sim_rcs(
    targets,
    f,
    inc_phi,
    inc_theta,
    inc_pol=[0, 0, 1],
    obs_phi=None,
    obs_theta=None,
    obs_pol=None,
    density=1.0
):
    """Calculate the Radar Cross Section (RCS) of targets using the Shooting and Bouncing Rays (SBR) method.

    This function computes the RCS of one or more targets by simulating how electromagnetic waves interact with the target models. The simulation uses the SBR technique, which accurately models wave scattering by tracing rays that shoot at the target and bounce off its surfaces.

    :param list[dict] targets:
        A list of target dictionaries specifying the properties of each target. Each dictionary contains the following keys:

        - **model** (*str*):  
          File path to the 3D target model.
        - **origin** (*numpy.ndarray*):  
          The origin position (rotation and translation center) of the target model in meters (m), specified as [x, y, z].  
          Default: ``[0, 0, 0]``.
        - **location** (*numpy.ndarray*):  
          The 3D location of the target in meters (m), specified as [x, y, z].  
          Default: ``[0, 0, 0]``.
        - **rotation** (*numpy.ndarray*):  
          The target's orientation in degrees (°), specified as [yaw, pitch, roll].  
          Default: ``[0, 0, 0]``.
        - **permittivity** (*complex*):  
          The target's permittivity, which represents its electromagnetic material properties.  
          Default: Perfect Electric Conductor (PEC).
        - **unit** (*str*):  
          Unit of measurement for the target model's geometry.  
          Supported values: ``mm``, ``cm``, ``m``.  
          Default: ``m``.

    :param float f:
        The center frequency of the incident electromagnetic wave in Hertz (Hz).
    :param float or array_like inc_phi:
        The horizontal incidence angle (phi) of the incoming wave in degrees (°).  
        This angle is measured relative to the target at the transmitter's point of view.
        Can be a single value or an array of values.
    :param float or array_like inc_theta:
        The vertical incidence angle (theta) of the incoming wave in degrees (°).  
        This angle is measured relative to the target at the transmitter's point of view.
        Can be a single value or an array of values.
    :param list[float] inc_pol:
        The polarization of the incident wave, specified as a 3D vector [x, y, z].  
        Default: ``[0, 0, 1]`` (vertical polarization).
    :param float or array_like obs_phi:
        The horizontal observation angle (phi) in degrees (°).  
        This is the angle at which the RCS is observed from the observer's point of view.  
        Can be a single value or an array of values.
        Default: ``None`` (if not specified, it is set to the same value as `inc_phi`).
    :param float or array_like obs_theta:
        The vertical observation angle (theta) in degrees (°).  
        This is the angle at which the RCS is observed from the observer's point of view.  
        Can be a single value or an array of values.
        Default: ``None`` (if not specified, it is set to the same value as `inc_theta`).
    :param list[float] obs_pol:
        The polarization of the observer, specified as a 3D vector [x, y, z].  
        Default: ``None`` (if not specified, it is set to the same value as `inc_pol`).
    :param float density:
        The ray density, defined as the number of rays per wavelength.  
        Higher ray density improves accuracy but increases computational cost.  
        Default: ``1.0``.

    :return:  
        The Radar Cross Section (RCS) of the target(s) in square meters (m²).  
        Returns a single float value if input angles are scalars, or a numpy array if input angles are arrays.
        To convert the result to decibels relative to one square meter (dBsm), use:  
        ``10 * log10(RCS)``.
    :rtype: float or numpy.ndarray
    """
    if not is_licensed():
        if len(targets) > 3:
            raise RuntimeError(
                "\nTrial Version Limitation - Target Count\n"
                "----------------------------------------\n"
                "Current limitation: Maximum 3 targets\n"
                "Your scene: {} targets\n\n"
                "This limitation helps maintain reasonable simulation times in the trial version.\n"
                "To simulate more targets, please purchase a license:\n"
                "→ https://radarsimx.com/product/radarsimpy/\n"
                .format(len(targets))
            )

    cdef shared_ptr[TargetsManager[float]] targets_manager = make_shared[TargetsManager[float]]()
    cdef Vec3[cpp_complex[double]] inc_pol_cpp
    cdef Vec3[cpp_complex[double]] obs_pol_cpp

    # Set default observation parameters if not provided
    if obs_pol is None:
        obs_pol = inc_pol
    if obs_phi is None:
        obs_phi = inc_phi
    if obs_theta is None:
        obs_theta = inc_theta

    if isinstance(inc_phi, (list, tuple, np.ndarray)):
        inc_phi = np.array(inc_phi)
    else:
        inc_phi = np.array([inc_phi])

    if isinstance(inc_theta, (list, tuple, np.ndarray)):
        inc_theta = np.array(inc_theta)
    else:
        inc_theta = np.array([inc_theta])
    
    if np.shape(inc_phi)!=np.shape(inc_theta):
        raise ValueError('The lengths of `inc_phi` and `inc_theta` must be the same')

    if isinstance(obs_phi, (list, tuple, np.ndarray)):
        obs_phi = np.array(obs_phi)
    else:
        obs_phi = np.array([obs_phi])

    if isinstance(obs_theta, (list, tuple, np.ndarray)):
        obs_theta = np.array(obs_theta)
    else:
        obs_theta = np.array([obs_theta])
    
    if np.shape(obs_phi)!=np.shape(obs_theta):
        raise ValueError('The lengths of `obs_phi` and `obs_theta` must be the same')
    
    if np.shape(inc_phi)!=np.shape(obs_phi):
        raise ValueError('The lengths of `inc_phi` and `obs_phi` must be the same')
    
    cdef int array_size = np.size(obs_phi)
    cdef int idx

    # Convert polarization vectors to C++ types
    inc_pol_cpp = Vec3[cpp_complex[double]](
        cpp_complex[double](np.real(inc_pol[0]), np.imag(inc_pol[0])),
        cpp_complex[double](np.real(inc_pol[1]), np.imag(inc_pol[1])),
        cpp_complex[double](np.real(inc_pol[2]), np.imag(inc_pol[2]))
    )
    
    obs_pol_cpp = Vec3[cpp_complex[double]](
        cpp_complex[double](np.real(obs_pol[0]), np.imag(obs_pol[0])),
        cpp_complex[double](np.real(obs_pol[1]), np.imag(obs_pol[1])),
        cpp_complex[double](np.real(obs_pol[2]), np.imag(obs_pol[2]))
    )

    # Process targets
    mesh_module = import_mesh_module()
    for idx_c in range(0, len(targets)):
       cp_RCS_Target(targets[idx_c], mesh_module, targets_manager.get())

    # Convert angles to radians
    inc_phi_rad = np.radians(inc_phi)
    inc_theta_rad = np.radians(inc_theta)
    obs_phi_rad = np.radians(obs_phi)
    obs_theta_rad = np.radians(obs_theta)

    inc_dir_x=np.sin(inc_theta_rad) * np.cos(inc_phi_rad)
    inc_dir_y=np.sin(inc_theta_rad) * np.sin(inc_phi_rad)
    inc_dir_z=np.cos(inc_theta_rad)

    obs_dir_x=np.sin(obs_theta_rad) * np.cos(obs_phi_rad)
    obs_dir_y=np.sin(obs_theta_rad) * np.sin(obs_phi_rad)
    obs_dir_z=np.cos(obs_theta_rad)

    cdef vector[Vec3[double]] inc_dir
    cdef vector[Vec3[double]] obs_dir

    for idx in range(0, array_size):

        # Calculate direction vectors
        inc_dir.push_back(Vec3[double](
            <double>(inc_dir_x[idx]),
            <double>(inc_dir_y[idx]),
            <double>(inc_dir_z[idx])
        ))

        obs_dir.push_back(Vec3[double](
            <double>(obs_dir_x[idx]),
            <double>(obs_dir_y[idx]),
            <double>(obs_dir_z[idx])
        ))

    # Calculate RCS
    cdef RcsSimulator[double] rcs_sim_c

    cdef vector[double] rcs_vect = rcs_sim_c.Run(
        targets_manager,
        inc_dir,
        obs_dir,
        inc_pol_cpp,
        obs_pol_cpp,
        <double>f,
        <double>density)

    rcs=np.zeros(array_size)

    for idx in range(0, array_size):
        rcs[idx] = rcs_vect[idx]

    if array_size == 1:
        return rcs[0]
    else:
        return rcs
