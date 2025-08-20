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

# Standard imports
import numpy as np
import warnings

# Cython imports
cimport cython
cimport numpy as np
from libcpp.complex cimport complex as cpp_complex
from libcpp cimport bool

# Local imports
from radarsimpy.includes.radarsimc cimport (
    Transmitter, Receiver, TxChannel, RxChannel, 
    Radar, TargetsManager, PointsManager, Mem_Copy, Mem_Copy_Vec3,
    Mem_Copy_Complex, IsFreeTier
)
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t, float_t, vector

from radarsimpy.mesh_kit import load_mesh

np.import_array()
np_float = np.float32

# Constants for unit conversion and validation
cdef dict UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}
cdef int_t MAX_FREE_TIER_FACES = 8

# Helper functions for validation and error handling
cdef inline void _validate_vector_3d(object vector, str name) except *:
    """
    Validate that a vector has exactly 3 elements for 3D operations.
    
    Parameters
    ----------
    vector : array_like
        Input vector to validate (should have 3 elements for [x, y, z]).
    name : str
        Parameter name for clear error messages.
        
    Raises
    ------
    TypeError
        If vector is not array-like (missing __len__ method).
    ValueError
        If vector doesn't have exactly 3 elements.
        
    Examples
    --------
    Valid usage:
        _validate_vector_3d([1, 2, 3], "location")
        _validate_vector_3d(np.array([0, 0, 1]), "direction")
        
    Invalid usage (will raise):
        _validate_vector_3d([1, 2], "location")  # ValueError: wrong size
        _validate_vector_3d(5, "location")       # TypeError: not array-like
    """
    if not hasattr(vector, '__len__'):
        raise TypeError(f"{name} must be array-like")
    if len(vector) != 3:
        raise ValueError(f"{name} must have 3 elements [x, y, z], got {len(vector)}")

cdef inline float_t _safe_unit_conversion(str unit) except *:
    """
    Safely convert unit string to scale factor with validation.
    
    Parameters
    ----------
    unit : str
        Unit string specifying distance units. Supported values:
        - 'm': meters (scale factor 1.0)
        - 'cm': centimeters (scale factor 100.0)  
        - 'mm': millimeters (scale factor 1000.0)
        
    Returns
    -------
    float_t
        Scale factor for converting from specified unit to meters.
        
    Raises
    ------
    ValueError
        If unit is not one of the supported values.
        
    Examples
    --------
    >>> _safe_unit_conversion('m')   # Returns 1.0
    >>> _safe_unit_conversion('cm')  # Returns 100.0
    >>> _safe_unit_conversion('mm')  # Returns 1000.0
    >>> _safe_unit_conversion('ft')  # Raises ValueError
    """
    if unit not in UNIT_SCALE:
        raise ValueError(f"Invalid unit '{unit}'. Supported units: {list(UNIT_SCALE.keys())}")
    return <float_t>UNIT_SCALE[unit]

cdef inline void _validate_mesh_for_free_tier(int_t num_faces) except *:
    """
    Check mesh size limits for free tier with detailed error message.
    
    Validates that the number of mesh faces doesn't exceed the FreeTier
    limitation. If the limit is exceeded, provides a detailed error message
    with guidance on upgrading.
    
    Parameters
    ----------
    num_faces : int_t
        Number of mesh faces in the target model.
        
    Raises
    ------
    RuntimeError
        If FreeTier is active and num_faces exceeds MAX_FREE_TIER_FACES (8).
        The error message includes:
        - Current limitation value
        - Actual number of faces in the model
        - Number of faces that need to be reduced
        - Link to upgrade information
        
    Notes
    -----
    This function only performs validation when IsFreeTier() returns True.
    For full/standard versions, this function does nothing and returns
    immediately without any checks.
    
    The limitation helps maintain reasonable simulation times in the trial
    version while encouraging users to upgrade for larger simulations.
    """
    if IsFreeTier() and num_faces > MAX_FREE_TIER_FACES:
        raise RuntimeError(
            f"\n{'='*60}\n"
            f"TRIAL VERSION LIMITATION - Mesh Size\n"
            f"{'='*60}\n"
            f"Current limitation: Maximum {MAX_FREE_TIER_FACES} mesh faces\n"
            f"Your model: {num_faces} faces\n"
            f"Reduction needed: {num_faces - MAX_FREE_TIER_FACES} faces\n\n"
            f"This limitation helps maintain reasonable simulation times in the trial version.\n"
            f"To simulate larger meshes, please upgrade to the Standard Version:\n"
            f"→ https://radarsimx.com/product/radarsimpy/\n"
            f"{'='*60}\n"
        )

cdef inline cpp_complex[float_t] _convert_permittivity(object permittivity) except *:
    """Convert permittivity value to complex float with validation."""
    if permittivity == "PEC":
        return cpp_complex[float_t](<float_t>1e38, <float_t>0.0)
    else:
        try:
            return cpp_complex[float_t](<float_t>np.real(permittivity), <float_t>np.imag(permittivity))
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid permittivity value: {e}")

cdef inline cpp_complex[float_t] _convert_permeability(object permeability) except *:
    """Convert permeability value to complex float with validation."""
    try:
        return cpp_complex[float_t](<float_t>np.real(permeability), <float_t>np.imag(permeability))
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid permeability value: {e}")

cdef inline void _warn_deprecated_parameter(str old_param, str new_param) except *:
    """Issue a deprecation warning for renamed parameters."""
    warnings.warn(
        f"Deprecated: '{old_param}' parameter has been replaced with '{new_param}'. "
        f"Please update your code to use '{new_param}' instead. "
        f"Support for '{old_param}' will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddPoint(location, speed, rcs, phase, shape, PointsManager[float_t] * points_manager):
    """
    cp_Point(location, speed, rcs, phase, shape)

    Create Point object in Cython with enhanced input validation and error handling

    :param list location:
        Target's location (m), [x, y, z]
        *Note*: Target's parameters can be specified with
        ``Radar.timestamp`` to customize the time varying property.
        Example: ``location=(1e-3*np.sin(2*np.pi*1*radar.timestamp), 0, 0)``
    :param list speed:
        Target's velocity (m/s), [x, y, z]
    :param float rcs:
        Target's RCS (dBsm)
        *Note*: Target's RCS can be specified with
        ``Radar.timestamp`` to customize the time varying property.
    :param float phase:
        Target's phase (deg)
        *Note*: Target's phase can be specified with
        ``Radar.timestamp`` to customize the time varying property.
    :param tuple shape:
        Shape of the time matrix (channels, frames, pulses)

    :return: C++ object of a point target
    :rtype: Point
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

        if np.size(location[0]) > 1:
            locx_mv = location[0].astype(np_float)
        else:
            locx_mv = np.full(shape, location[0], dtype=np_float)

        if np.size(location[1]) > 1:
            locy_mv = location[1].astype(np_float)
        else:
            locy_mv = np.full(shape, location[1], dtype=np_float)

        if np.size(location[2]) > 1:
            locz_mv = location[2].astype(np_float)
        else:
            locz_mv = np.full(shape, location[2], dtype=np_float)

        if np.size(rcs) > 1:
            rcs_mv = rcs.astype(np_float)
        else:
            rcs_mv = np.full(shape, rcs, dtype=np_float)

        if np.size(phase) > 1:
            phs_mv = np.radians(phase).astype(np_float)
        else:
            phs_mv = np.full(shape, np.radians(phase), dtype=np_float)

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


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Transmitter[double, float_t] cp_Transmitter(radar):
    """
    cp_Transmitter(radar)

    Create Transmitter object in Cython with comprehensive parameter processing

    :param Radar radar:
        Radar object containing transmitter configuration

    :return: C++ object of a radar transmitter
    :rtype: Transmitter
    :raises: ValueError for invalid radar configuration
    """
    # Extract key dimensions from radar configuration
    cdef int_t channles_c = radar.array_prop["size"]
    cdef int_t pulses_c = radar.radar_prop["transmitter"].waveform_prop["pulses"]
    cdef int_t samples_c = radar.sample_prop["samples_per_pulse"]

    # Vector declarations for transmitter parameters
    cdef vector[double] f_vt, t_vt
    cdef vector[double] f_offset_vt
    cdef vector[double] t_pstart_vt
    cdef vector[cpp_complex[double]] pn_vt

    # frequency
    cdef double[:] f_mv = radar.radar_prop["transmitter"].waveform_prop["f"].astype(np.float64)
    Mem_Copy(&f_mv[0], <int_t>(len(radar.radar_prop["transmitter"].waveform_prop["f"])), f_vt)

    # time
    cdef double[:] t_mv = radar.radar_prop["transmitter"].waveform_prop["t"].astype(np.float64)
    Mem_Copy(&t_mv[0], <int_t>(len(radar.radar_prop["transmitter"].waveform_prop["t"])), t_vt)

    # frequency offset per pulse
    cdef double[:] f_offset_mv = radar.radar_prop["transmitter"].waveform_prop["f_offset"].astype(np.float64)
    Mem_Copy(&f_offset_mv[0], <int_t>(len(radar.radar_prop["transmitter"].waveform_prop["f_offset"])), f_offset_vt)

    # pulse start time
    cdef double[:] t_pstart_mv = radar.radar_prop["transmitter"].waveform_prop["pulse_start_time"].astype(np.float64)
    Mem_Copy(&t_pstart_mv[0], <int_t>(len(radar.radar_prop["transmitter"].waveform_prop["pulse_start_time"])), t_pstart_vt)

    # phase noise
    cdef double[:] pn_real_mv
    cdef double[:] pn_imag_mv
    if radar.sample_prop["phase_noise"] is not None:
        pn_real_mv = np.real(radar.sample_prop["phase_noise"]).astype(np.float64)
        pn_imag_mv = np.imag(radar.sample_prop["phase_noise"]).astype(np.float64)
        Mem_Copy_Complex(&pn_real_mv[0], &pn_imag_mv[0], <int_t>(np.size(radar.sample_prop["phase_noise"])), pn_vt)

    return Transmitter[double, float_t](
        <float_t> radar.radar_prop["transmitter"].rf_prop["tx_power"],
        f_vt,
        t_vt,
        f_offset_vt,
        t_pstart_vt,
        pn_vt
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef TxChannel[float_t] cp_TxChannel(tx,
                                     tx_idx):
    """
    cp_TxChannel(tx, tx_idx)

    Create TxChannel object in Cython with complete antenna pattern processing

    Converts Python transmitter channel configuration into a C++ TxChannel object.
    Handles antenna patterns, polarization, pulse modulation, and waveform modulation
    for a single transmitter channel.

    :param Transmitter tx:
        Python transmitter object containing channel configurations including:
        - Antenna patterns (azimuth and elevation)
        - Polarization vector [x, y, z] (complex)
        - Pulse modulation coefficients (complex array)
        - Waveform modulation parameters
        - Channel locations and gains
        - Timing delays and grid orientation
    :param int tx_idx:
        Transmitter channel index (0-based) for multi-channel systems

    :return: C++ object of a transmitter channel
    :rtype: TxChannel[float_t]
    
    :raises ValueError: If channel index is out of range or pattern data is invalid
    :raises RuntimeError: If antenna pattern conversion fails
    
    Notes
    -----
    - Azimuth patterns are converted from degrees to radians
    - Elevation patterns are flipped (90° - elevation) and reversed for C++ convention
    - Complex polarization vectors are properly converted to C++ complex types
    - Modulation data is validated for correct dimensions
    """
    cdef int_t pulses_c = tx.waveform_prop["pulses"]

    cdef vector[float_t] az_ang_vt, az_ptn_vt
    cdef vector[float_t] el_ang_vt, el_ptn_vt

    cdef float_t[:] az_ang_mv, az_ptn_mv
    cdef float_t[:] el_ang_mv, el_ptn_mv

    cdef vector[cpp_complex[float_t]] pulse_mod_vt

    cdef bool mod_enabled
    cdef vector[cpp_complex[float_t]] mod_var_vt
    cdef vector[float_t] mod_t_vt

    # azimuth pattern
    az_ang_mv = np.radians(np.array(tx.txchannel_prop["az_angles"][tx_idx])).astype(np_float)
    az_ptn_mv = np.array(tx.txchannel_prop["az_patterns"][tx_idx]).astype(np_float)

    Mem_Copy(&az_ang_mv[0], <int_t>(len(tx.txchannel_prop["az_angles"][tx_idx])), az_ang_vt)
    Mem_Copy(&az_ptn_mv[0], <int_t>(len(tx.txchannel_prop["az_patterns"][tx_idx])), az_ptn_vt)

    # elevation pattern
    el_ang_mv = np.radians(np.flip(90-tx.txchannel_prop["el_angles"][tx_idx])).astype(np_float)
    el_ptn_mv = np.flip(tx.txchannel_prop["el_patterns"][tx_idx]).astype(np_float)

    Mem_Copy(&el_ang_mv[0], <int_t>(len(tx.txchannel_prop["el_angles"][tx_idx])), el_ang_vt)
    Mem_Copy(&el_ptn_mv[0], <int_t>(len(tx.txchannel_prop["el_patterns"][tx_idx])), el_ptn_vt)

    # pulse modulation
    cdef float_t[:] pulse_real_mv = np.real(tx.txchannel_prop["pulse_mod"][tx_idx]).astype(np_float)
    cdef float_t[:] pulse_imag_mv = np.imag(tx.txchannel_prop["pulse_mod"][tx_idx]).astype(np_float)
    Mem_Copy_Complex(&pulse_real_mv[0], &pulse_imag_mv[0], <int_t>(pulses_c), pulse_mod_vt)

    # waveform modulation
    mod_enabled = tx.txchannel_prop["waveform_mod"][tx_idx]["enabled"]

    cdef float_t[:] mod_real_mv, mod_imag_mv
    cdef float_t[:] mod_t_mv
    if mod_enabled:
        mod_real_mv = np.real(tx.txchannel_prop["waveform_mod"][tx_idx]["var"]).astype(np_float)
        mod_imag_mv = np.imag(tx.txchannel_prop["waveform_mod"][tx_idx]["var"]).astype(np_float)
        mod_t_mv = tx.txchannel_prop["waveform_mod"][tx_idx]["t"].astype(np_float)

        Mem_Copy_Complex(&mod_real_mv[0], &mod_imag_mv[0], <int_t>(len(tx.txchannel_prop["waveform_mod"][tx_idx]["var"])), mod_var_vt)
        Mem_Copy(&mod_t_mv[0], <int_t>(len(tx.txchannel_prop["waveform_mod"][tx_idx]["t"])), mod_t_vt)

    cdef float_t[:] location_mv = tx.txchannel_prop["locations"][tx_idx].astype(np_float)

    polar = tx.txchannel_prop["polarization"][tx_idx]
    cdef Vec3[cpp_complex[float_t]] polarization_vt = Vec3[cpp_complex[float_t]](cpp_complex[float_t](np.real(polar[0]), np.imag(polar[0])), cpp_complex[float_t](np.real(polar[1]), np.imag(polar[1])), cpp_complex[float_t](np.real(polar[2]), np.imag(polar[2])))

    return TxChannel[float_t](
        Vec3[float_t](&location_mv[0]),
        polarization_vt,
        az_ang_vt,
        az_ptn_vt,
        el_ang_vt,
        el_ptn_vt,
        <float_t> tx.txchannel_prop["antenna_gains"][tx_idx],
        mod_t_vt,
        mod_var_vt,
        pulse_mod_vt,
        <float_t> tx.txchannel_prop["delay"][tx_idx],
        <float_t> np.radians(tx.txchannel_prop["grid"][tx_idx])
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef RxChannel[float_t] cp_RxChannel(rx,
                                     rx_idx):
    """
    cp_RxChannel(rx, rx_idx)

    Create RxChannel object in Cython with antenna pattern processing

    Converts Python receiver channel configuration into a C++ RxChannel object.
    Handles antenna patterns (azimuth and elevation), polarization, and channel
    properties for a single receiver channel.

    :param Receiver rx:
        Python receiver object containing channel configurations including:
        - Antenna patterns (azimuth and elevation angles and gains)
        - Polarization vector [x, y, z] (complex)
        - Channel locations and antenna gains
        - Receiver-specific properties
    :param int rx_idx:
        Receiver channel index (0-based) for multi-channel receiver systems

    :return: C++ object of a receiver channel for simulation
    :rtype: RxChannel[float_t]
    
    :raises ValueError: If channel index is out of range or pattern data is invalid
    :raises RuntimeError: If antenna pattern conversion fails
    
    Notes
    -----
    - Similar to TxChannel but for receiver-specific processing
    - Azimuth patterns are converted from degrees to radians
    - Elevation patterns are flipped (90° - elevation) and reversed for C++ convention
    - Complex polarization vectors are properly converted to C++ complex types
    - No modulation processing needed for receiver channels
    """
    cdef vector[float_t] az_ang_vt, az_ptn_vt
    cdef vector[float_t] el_ang_vt, el_ptn_vt

    cdef float_t[:] az_ang_mv, az_ptn_mv
    cdef float_t[:] el_ang_mv, el_ptn_mv

    # azimuth pattern
    az_ang_mv = np.radians(rx.rxchannel_prop["az_angles"][rx_idx]).astype(np_float)
    az_ptn_mv = rx.rxchannel_prop["az_patterns"][rx_idx].astype(np_float)

    Mem_Copy(&az_ang_mv[0], <int_t>(len(rx.rxchannel_prop["az_angles"][rx_idx])), az_ang_vt)
    Mem_Copy(&az_ptn_mv[0], <int_t>(len(rx.rxchannel_prop["az_patterns"][rx_idx])), az_ptn_vt)

    # elevation pattern
    el_ang_mv = np.radians(np.flip(90-rx.rxchannel_prop["el_angles"][rx_idx])).astype(np_float)
    el_ptn_mv = np.flip(rx.rxchannel_prop["el_patterns"][rx_idx]).astype(np_float)

    Mem_Copy(&el_ang_mv[0], <int_t>(len(rx.rxchannel_prop["el_angles"][rx_idx])), el_ang_vt)
    Mem_Copy(&el_ptn_mv[0], <int_t>(len(rx.rxchannel_prop["el_patterns"][rx_idx])), el_ptn_vt)

    cdef float_t[:] location_mv = rx.rxchannel_prop["locations"][rx_idx].astype(np_float)

    polar = rx.rxchannel_prop["polarization"][rx_idx]
    cdef Vec3[cpp_complex[float_t]] polarization_vt = Vec3[cpp_complex[float_t]](cpp_complex[float_t](np.real(polar[0]), np.imag(polar[0])), cpp_complex[float_t](np.real(polar[1]), np.imag(polar[1])), cpp_complex[float_t](np.real(polar[2]), np.imag(polar[2])))
    # cdef float_t[:] polarization_mv = rx.rxchannel_prop["polarization"][rx_idx].astype(np_float)
    return RxChannel[float_t](
        Vec3[float_t](&location_mv[0]),
        polarization_vt,
        az_ang_vt,
        az_ptn_vt,
        el_ang_vt,
        el_ptn_vt,
        <float_t> rx.rxchannel_prop["antenna_gains"][rx_idx]
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_ConfigureRadar(radar, frame_start_time, Radar[double, float_t] * cptr_radar):
    """
    cp_Radar(radar, frame_start_time)

    Create a complete Radar system object for simulation with enhanced configuration

    Converts Python radar configuration into a comprehensive C++ Radar object.
    This function orchestrates the creation of transmitter and receiver systems,
    handles multi-frame timing, and manages complex radar positioning and motion.

    :param Radar radar:
        Python radar object containing complete system configuration:
        
        Array Properties:
        - size: Number of channels in the radar array
        
        Transmitter Properties:
        - waveform: Signal parameters (frequency, timing, modulation)
        - channels: Individual transmitter channel configurations
        - power levels and antenna patterns
        
        Receiver Properties: 
        - gain settings and noise characteristics
        - channels: Individual receiver channel configurations
        - sampling parameters
        
        Radar Motion:
        - location: Radar position [x, y, z] (can be time-varying)
        - speed: Radar velocity [x, y, z] in m/s
        - rotation: Radar orientation [roll, pitch, yaw] in degrees
        - rotation_rate: Angular velocity [x, y, z] in deg/s

    :param frame_start_time:
        Timing information for multi-frame simulations:
        - Single value for single-frame simulation
        - Array for multi-frame with timing offsets

    :return: Complete C++ Radar object ready for simulation
    :rtype: Radar[double, float_t]
    
    :raises ValueError: If radar configuration is incomplete or invalid
    :raises RuntimeError: If transmitter/receiver setup fails
    
    Notes
    -----
    - Automatically handles single vs. multi-frame configurations
    - Processes both static and time-varying radar parameters
    - Creates and links transmitter and receiver channel objects
    - Handles coordinate transformations and unit conversions
    - Optimized for high-performance simulation engine interface
    """
    cdef Transmitter[double, float_t] tx_c
    cdef Receiver[float_t] rx_c

    # Extract key system dimensions from radar configuration
    cdef int_t txsize_c = radar.radar_prop["transmitter"].txchannel_prop["size"]
    cdef int_t rxsize_c = radar.radar_prop["receiver"].rxchannel_prop["size"]
    cdef int_t frames_c = np.size(frame_start_time)
    cdef int_t channles_c = radar.array_prop["size"]
    cdef int_t pulses_c = radar.radar_prop["transmitter"].waveform_prop["pulses"]
    cdef int_t samples_c = radar.sample_prop["samples_per_pulse"]

    # Calculate total buffer size for simulation data
    cdef int_t bbsize_c = channles_c*frames_c*pulses_c*samples_c

    # Memory views for time-varying position and orientation
    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] rotx_mv, roty_mv, rotz_mv

    # Vector storage for C++ radar object construction
    cdef vector[double] t_frame_vt
    cdef vector[Vec3[float_t]] loc_vt
    cdef Vec3[float_t] spd_vt
    cdef vector[Vec3[float_t]] rot_vt
    cdef Vec3[float_t] rrt_vt

    cdef int_t idx_c

    # frame time offset
    cdef double[:] t_frame_mv
    if frames_c > 1:
        t_frame_mv = frame_start_time.astype(np.float64)
        Mem_Copy(&t_frame_mv[0], frames_c, t_frame_vt)
    else:
        t_frame_vt.push_back(<double> frame_start_time)

    """
    Transmitter
    """
    tx_c = cp_Transmitter(radar)
    for idx_c in range(0, txsize_c):
        tx_c.AddChannel(
            cp_TxChannel(radar.radar_prop["transmitter"], idx_c)
        )

    """
    Receiver
    """
    rx_c = Receiver[float_t](
        <float_t> radar.radar_prop["receiver"].bb_prop["fs"],
        <float_t> radar.radar_prop["receiver"].rf_prop["rf_gain"],
        <float_t> radar.radar_prop["receiver"].bb_prop["load_resistor"],
        <float_t> radar.radar_prop["receiver"].bb_prop["baseband_gain"],
        <float_t> radar.radar_prop["receiver"].bb_prop["noise_bandwidth"]
    )
    for idx_c in range(0, rxsize_c):
        rx_c.AddChannel(
            cp_RxChannel(radar.radar_prop["receiver"], idx_c)
        )

    """
    Radar
    """
    cdef float_t[:] loc_mv, rot_mv

    if len(np.shape(radar.radar_prop["location"])) == 4:
        locx_mv = radar.radar_prop["location"][:,:,:,0].astype(np_float)
        locy_mv = radar.radar_prop["location"][:,:,:,1].astype(np_float)
        locz_mv = radar.radar_prop["location"][:,:,:,2].astype(np_float)
        rotx_mv = radar.radar_prop["rotation"][:,:,:,0].astype(np_float)
        roty_mv = radar.radar_prop["rotation"][:,:,:,1].astype(np_float)
        rotz_mv = radar.radar_prop["rotation"][:,:,:,2].astype(np_float)

        Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)
        Mem_Copy_Vec3(&rotx_mv[0,0,0], &roty_mv[0,0,0], &rotz_mv[0,0,0], bbsize_c, rot_vt)

    else:
        loc_mv = radar.radar_prop["location"].astype(np_float)
        loc_vt.push_back(Vec3[float_t](&loc_mv[0]))

        rot_mv = radar.radar_prop["rotation"].astype(np_float)
        rot_vt.push_back(Vec3[float_t](&rot_mv[0]))
    
    spd_vt = Vec3[float_t](<float_t>radar.radar_prop["speed"][0], <float_t>radar.radar_prop["speed"][1], <float_t>radar.radar_prop["speed"][2])
    rrt_vt = Vec3[float_t](<float_t>radar.radar_prop["rotation_rate"][0], <float_t>radar.radar_prop["rotation_rate"][1], <float_t>radar.radar_prop["rotation_rate"][2])

    cptr_radar[0].Configure(tx_c,
                            rx_c,
                            t_frame_vt,
                            loc_vt,
                            spd_vt,
                            rot_vt,
                            rrt_vt)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddTarget(radar,
                  target,
                  timestamp,
                  mesh_module,
                  TargetsManager[float_t] * targets_manager):
    """
    cp_Target((radar, target, ts_shape)

    Creat Target object in Cython

    :param Radar radar:
        Radar object
    :param dict target:
        Target properties
    :param tuple ts_shape:
        Shape of the time matrix

    :return: C++ object of a target
    :rtype: Target
    """
    # vector of location, speed, rotation, rotation rate
    cdef vector[Vec3[float_t]] loc_vt
    cdef vector[Vec3[float_t]] spd_vt
    cdef vector[Vec3[float_t]] rot_vt
    cdef vector[Vec3[float_t]] rrt_vt

    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] spdx_mv, spdy_mv, spdz_mv
    cdef float_t[:, :, :] rotx_mv, roty_mv, rotz_mv
    cdef float_t[:, :, :] rrtx_mv, rrty_mv, rrtz_mv

    cdef cpp_complex[float_t] ep_c, mu_c

    cdef int_t ch_idx, ps_idx, sp_idx
    ts_shape = np.shape(timestamp)
    cdef int_t bbsize_c = <int_t>(ts_shape[0]*ts_shape[1]*ts_shape[2])

    cdef float_t scale
    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv

    # Enhanced mesh validation and loading with improved error messages
    unit = target.get("unit", "m")
    try:
        scale = _safe_unit_conversion(unit)
    except ValueError as e:
        raise ValueError(f"Invalid unit in target configuration: {e}")

    try:
        mesh_data = load_mesh(target["model"], scale, mesh_module)
        points_mv = mesh_data["points"].astype(np_float)
        cells_mv = mesh_data["cells"].astype(np.int32)
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh model '{target.get('model', 'unknown')}': {e}")
    
    # Enhanced FreeTier validation using helper function
    _validate_mesh_for_free_tier(cells_mv.shape[0])

    cdef float_t[:] origin_mv = np.array(target.get("origin", (0, 0, 0)), dtype=np_float)

    location = list(target.get("location", [0, 0, 0]))
    speed = list(target.get("speed", [0, 0, 0]))
    rotation = list(target.get("rotation", [0, 0, 0]))
    rotation_rate = list(target.get( "rotation_rate", [0, 0, 0]))

    cdef float_t[:] location_mv, speed_mv, rotation_mv, rotation_rate_mv

    permittivity = target.get("permittivity", 1e38)
    permeability = target.get("permeability", 1)
    if permittivity == "PEC":
        ep_c = cpp_complex[float_t](<float_t>1e38, <float_t>0.0)
        mu_c = cpp_complex[float_t](<float_t>1.0, <float_t>0.0)
    else:
        ep_c = cpp_complex[float_t](<float_t>np.real(permittivity), <float_t>np.imag(permittivity))
        mu_c = cpp_complex[float_t](<float_t>np.real(permeability), <float_t>np.imag(permeability))

    if any(np.size(var) > 1 for var in location + speed + rotation + rotation_rate):
        if np.size(location[0]) > 1:
            locx_mv = location[0].astype(np_float)
        else:
            locx_mv = (location[0] + speed[0]*timestamp).astype(np_float)

        if np.size(location[1]) > 1:
            locy_mv = location[1].astype(np_float)
        else:
            locy_mv = (location[1] + speed[1]*timestamp).astype(np_float)

        if np.size(location[2]) > 1:
            locz_mv = location[2].astype(np_float)
        else:
            locz_mv = (location[2] + speed[2]*timestamp).astype(np_float)

        if np.size(speed[0]) > 1:
            spdx_mv = speed[0].astype(np_float)
        else:
            spdx_mv = np.full(ts_shape, speed[0], dtype=np_float)

        if np.size(speed[1]) > 1:
            spdy_mv = speed[1].astype(np_float)
        else:
            spdy_mv = np.full(ts_shape, speed[1], dtype=np_float)

        if np.size(speed[2]) > 1:
            spdz_mv = speed[2].astype(np_float)
        else:
            spdz_mv = np.full(ts_shape, speed[2], dtype=np_float)

        if np.size(rotation[0]) > 1:
            rotx_mv = np.radians(rotation[0]).astype(np_float)
        else:
            rotx_mv = np.radians(
                rotation[0] + rotation_rate[0]*timestamp).astype(np_float)

        if np.size(rotation[1]) > 1:
            roty_mv = np.radians(rotation[1]).astype(np_float)
        else:
            roty_mv = np.radians(
                rotation[1] + rotation_rate[1]*timestamp).astype(np_float)

        if np.size(rotation[2]) > 1:
            rotz_mv = np.radians(rotation[2]).astype(np_float)
        else:
            rotz_mv = np.radians(
                rotation[2] + rotation_rate[2]*timestamp).astype(np_float)

        if np.size(rotation_rate[0]) > 1:
            rrtx_mv = np.radians(rotation_rate[0]).astype(np_float)
        else:
            rrtx_mv = np.full(ts_shape, np.radians(rotation_rate[0]), dtype=np_float)

        if np.size(rotation_rate[1]) > 1:
            rrty_mv = np.radians(rotation_rate[1]).astype(np_float)
        else:
            rrty_mv = np.full(ts_shape, np.radians(rotation_rate[1]), dtype=np_float)

        if np.size(rotation_rate[2]) > 1:
            rrtz_mv = np.radians(rotation_rate[2]).astype(np_float)
        else:
            rrtz_mv = np.full(ts_shape, np.radians(rotation_rate[2]), dtype=np_float)

        Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)
        Mem_Copy_Vec3(&spdx_mv[0,0,0], &spdy_mv[0,0,0], &spdz_mv[0,0,0], bbsize_c, spd_vt)
        Mem_Copy_Vec3(&rotx_mv[0,0,0], &roty_mv[0,0,0], &rotz_mv[0,0,0], bbsize_c, rot_vt)
        Mem_Copy_Vec3(&rrtx_mv[0,0,0], &rrty_mv[0,0,0], &rrtz_mv[0,0,0], bbsize_c, rrt_vt)

    else:
        location_mv = np.array(location, dtype=np_float)
        loc_vt.push_back(Vec3[float_t](&location_mv[0]))

        speed_mv = np.array(speed, dtype=np_float)
        spd_vt.push_back(Vec3[float_t](&speed_mv[0]))

        rotation_mv = np.radians(np.array(rotation, dtype=np_float)).astype(np_float)
        rot_vt.push_back(Vec3[float_t](&rotation_mv[0]))

        rotation_rate_mv = np.radians(np.array(rotation_rate, dtype=np_float)).astype(np_float)
        rrt_vt.push_back(Vec3[float_t](&rotation_rate_mv[0]))
    
    # Handle deprecated parameter with enhanced warning
    if "is_ground" in target:
        target["skip_diffusion"] = target["is_ground"]
        _warn_deprecated_parameter("is_ground", "skip_diffusion")

    targets_manager[0].AddTarget(&points_mv[0, 0],
                           &cells_mv[0, 0],
                           <int_t> cells_mv.shape[0],
                           Vec3[float_t](&origin_mv[0]),
                           loc_vt,
                           spd_vt,
                           rot_vt,
                           rrt_vt,
                           ep_c,
                           mu_c,
                           <bool> target.get("skip_diffusion", False))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_RCS_Target(target, mesh_module, TargetsManager[float_t] * targets_manager):
    """
    cp_RCS_Target(target, mesh_module)

    Create Target object in Cython for RCS calculation with enhanced validation

    :param dict target:
        Target properties dictionary
    :param mesh_module:
        Mesh loading module
        
    :return: C++ object of a target for RCS computation
    :rtype: Target
    :raises: RuntimeError on mesh limitations, ValueError on invalid target
    """
    # Vector declarations
    cdef vector[Vec3[float_t]] loc_vt, spd_vt, rot_vt, rrt_vt
    cdef cpp_complex[float_t] ep_c, mu_c
    cdef float_t scale
    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv

    # Enhanced mesh validation and loading with improved error messages  
    unit = target.get("unit", "m")
    try:
        scale = _safe_unit_conversion(unit)
    except ValueError as e:
        raise ValueError(f"Invalid unit in target configuration: {e}")

    try:
        mesh_data = load_mesh(target["model"], scale, mesh_module)
        points_mv = mesh_data["points"].astype(np_float)
        cells_mv = mesh_data["cells"].astype(np.int32)
    except Exception as e:
        raise RuntimeError(f"Failed to load mesh model '{target.get('model', 'unknown')}': {e}")

    # Enhanced FreeTier validation using helper function
    _validate_mesh_for_free_tier(cells_mv.shape[0])

    cdef float_t[:] origin_mv = np.array(target.get("origin", (0, 0, 0)), dtype=np_float)

    location = np.array(target.get("location", (0, 0, 0)), dtype=object)
    speed = np.array(target.get("speed", (0, 0, 0)), dtype=object)
    rotation = np.array(target.get("rotation", (0, 0, 0)), dtype=object)
    rotation_rate = np.array(target.get( "rotation_rate", (0, 0, 0)), dtype=object)

    cdef float_t[:] location_mv, speed_mv, rotation_mv, rotation_rate_mv

    permittivity = target.get("permittivity", 1e38)
    permeability = target.get("permeability", 1)
    if permittivity == "PEC":
        ep_c = cpp_complex[float_t](<float_t>1e38, <float_t>0.0)
        mu_c = cpp_complex[float_t](<float_t>1.0, <float_t>0.0)
    else:
        ep_c = cpp_complex[float_t](<float_t>np.real(permittivity), <float_t>np.imag(permittivity))
        mu_c = cpp_complex[float_t](<float_t>np.real(permeability), <float_t>np.imag(permeability))

    location_mv = location.astype(np_float)
    loc_vt.push_back(Vec3[float_t](&location_mv[0]))

    speed_mv = speed.astype(np_float)
    spd_vt.push_back(Vec3[float_t](&speed_mv[0]))

    rotation_mv = np.radians(rotation.astype(np_float)).astype(np_float)
    rot_vt.push_back(Vec3[float_t](&rotation_mv[0]))

    rotation_rate_mv = np.radians(rotation_rate.astype(np_float)).astype(np_float)
    rrt_vt.push_back(Vec3[float_t](&rotation_rate_mv[0]))

    # Handle deprecated parameter with enhanced warning
    if "is_ground" in target:
        target["skip_diffusion"] = target["is_ground"]
        _warn_deprecated_parameter("is_ground", "skip_diffusion")

    targets_manager[0].AddTarget(&points_mv[0, 0],
                           &cells_mv[0, 0],
                           <int_t> cells_mv.shape[0],
                           Vec3[float_t](&origin_mv[0]),
                           loc_vt,
                           spd_vt,
                           rot_vt,
                           rrt_vt,
                           ep_c,
                           mu_c,
                           <bool> target.get("skip_diffusion", False))
