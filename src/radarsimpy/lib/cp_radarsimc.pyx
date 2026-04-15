# distutils: language = c++
"""
A Python module for radar simulation

This module provides optimized Cython wrapper functions for interfacing with 
the high-performance C++ radar simulation engine. Contains core conversion
and management functions for point targets, radar systems, mesh processing,
and RCS calculations.

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

# ============================================================================
# Imports
# ============================================================================

# Standard imports
import numpy as np
import warnings

# Cython imports
cimport cython
cimport numpy as np
from libcpp.complex cimport complex as cpp_complex
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared

# Local imports
from radarsimpy.includes.radarsimc cimport (
    Transmitter, Receiver, 
    Radar, TargetsManager, PointsManager,
    Mem_Copy, Mem_Copy_Vec3,
    Mem_Copy_Complex
)
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t, float_t, vector

from radarsimpy.license import is_licensed

from radarsimpy.mesh_kit import load_mesh

np.import_array()
np_float = np.float32


# ============================================================================
# Constants
# ============================================================================

cdef dict UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}
cdef int_t MAX_FREE_TIER_FACES = 8
cdef frozenset _VALID_TARGET_KEYS = frozenset({
    "model", "unit", "origin",
    "location", "speed", "rotation", "rotation_rate",
    "permittivity", "permeability",
    "skip_diffusion", "is_ground",  # is_ground is a deprecated alias
    "density", "environment",
})

# ============================================================================
# Private Helpers (validation, deprecation, unit conversion)
# ============================================================================

cdef inline float_t _safe_unit_conversion(str unit) except *:
    """
    Safely convert unit string to scale factor.

    :param str unit:
        Unit string ('m', 'cm', 'mm')
    :return: Scale factor for converting to meters
    :rtype: float_t
    :raises: ValueError for unsupported units
    """
    if unit not in UNIT_SCALE:
        raise ValueError(f"Invalid unit '{unit}'. Supported units: {list(UNIT_SCALE.keys())}")
    return <float_t>UNIT_SCALE[unit]

cdef inline void _validate_mesh_for_free_tier(int_t num_faces) except *:
    """
    Check mesh size limits for free tier.

    :param int_t num_faces:
        Number of mesh faces in the target model
    :raises: RuntimeError if FreeTier limitations are exceeded
    """
    if not is_licensed() and num_faces > MAX_FREE_TIER_FACES:
        raise RuntimeError(
            f"\nTrial Version Limitation - Mesh Size\n"
            f"----------------------------------------\n"
            f"Current limitation: Maximum {MAX_FREE_TIER_FACES} mesh faces\n"
            f"Your model: {num_faces} faces\n\n"
            f"To simulate larger meshes, please purchase a license:\n"
            f"→ https://radarsimx.com/product/radarsimpy/\n"
        )

cdef inline void _warn_deprecated_parameter(str old_param, str new_param) except *:
    """
    Issue standardized deprecation warnings for renamed parameters.

    :param str old_param:
        Name of the deprecated parameter
    :param str new_param:
        Name of the new parameter to use instead
    :raises: DeprecationWarning with migration guidance
    """
    warnings.warn(
        f"Deprecated: '{old_param}' parameter has been replaced with '{new_param}'. "
        f"Please update your code to use '{new_param}' instead. "
        f"Support for '{old_param}' will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

cdef inline void _validate_target_keys(target) except *:
    """
    Warn about unrecognized keys in a target dict to catch typos.

    :param dict target:
        Target properties dictionary to validate
    :raises: UserWarning for any keys not in the known set
    """
    unknown = set(target.keys()) - _VALID_TARGET_KEYS
    if unknown:
        warnings.warn(
            f"Unrecognized key(s) in target dict: {sorted(unknown)}. "
            f"These will be ignored. Valid keys: {sorted(_VALID_TARGET_KEYS)}",
            UserWarning,
            stacklevel=3
        )

cdef inline tuple _load_and_validate_mesh(target, mesh_module) except *:
    """
    Load mesh model and validate against free tier limits.

    :param dict target:
        Target properties containing 'model' and optional 'unit'
    :param mesh_module:
        Mesh loading module
    :return: Tuple of (points_array, cells_array) as numpy arrays
    :rtype: tuple
    """
    cdef float_t scale = _safe_unit_conversion(target.get("unit", "m"))
    try:
        mesh_data = load_mesh(target["model"], scale, mesh_module)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load mesh model '{target.get('model', 'unknown')}': {e}"
        )
    points = mesh_data["points"].astype(np_float)
    cells = mesh_data["cells"].astype(np.int32)
    _validate_mesh_for_free_tier(<int_t>cells.shape[0])
    return points, cells

cdef inline void _parse_material_properties(
    target,
    cpp_complex[float_t] * ep_out,
    cpp_complex[float_t] * mu_out,
) except *:
    """
    Parse permittivity and permeability from target dict.

    :param dict target:
        Target properties containing optional 'permittivity' and 'permeability'
    :param cpp_complex[float_t] * ep_out:
        Output pointer for permittivity
    :param cpp_complex[float_t] * mu_out:
        Output pointer for permeability
    """
    permittivity = target.get("permittivity", 1e38)
    permeability = target.get("permeability", 1)
    if permittivity == "PEC":
        ep_out[0] = cpp_complex[float_t](<float_t>1e38, <float_t>0.0)
        mu_out[0] = cpp_complex[float_t](<float_t>1.0, <float_t>0.0)
    else:
        ep_out[0] = cpp_complex[float_t](<float_t>np.real(permittivity), <float_t>np.imag(permittivity))
        mu_out[0] = cpp_complex[float_t](<float_t>np.real(permeability), <float_t>np.imag(permeability))

cdef inline void _handle_deprecated_target_params(target) except *:
    """
    Handle deprecated parameter names in target dict.

    :param dict target:
        Target properties dictionary (modified in-place)
    """
    if "is_ground" in target:
        target["skip_diffusion"] = target["is_ground"]
        _warn_deprecated_parameter("is_ground", "skip_diffusion")

# ============================================================================
# Point Target
# ============================================================================

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddPoint(location, speed, rcs, phase, shape, PointsManager[float_t] * points_manager):
    """
    Add a point scatterer to the radar simulation.

    :param list location:
        Target's location (m), [x, y, z]
    :param list speed:
        Target's velocity (m/s), [x, y, z]
    :param float rcs:
        Target's RCS (dBsm)
    :param float phase:
        Target's phase (deg)
    :param tuple shape:
        Shape of the time matrix (channels, frames, pulses)
    :param PointsManager[float_t] * points_manager:
        Pointer to C++ points manager
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


# ============================================================================
# Radar System (Transmitter, Receiver, Radar)
# ============================================================================

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef shared_ptr[Transmitter[double, float_t]] cp_Transmitter(radar):
    """
    Create Transmitter object in Cython with comprehensive parameter processing.

    :param Radar radar:
        Radar object containing transmitter configuration including waveform
        properties (f, t, f_offset, pulse_start_time), RF properties (tx_power),
        and optional phase noise
    :return: C++ object of a radar transmitter
    :rtype: shared_ptr[Transmitter[double, float_t]]
    :raises: ValueError for invalid radar configuration
    """
    # Cache nested property lookups
    tx_prop = radar.radar_prop["transmitter"]
    wf_prop = tx_prop.waveform_prop
    sample_prop = radar.sample_prop

    # Vector declarations for transmitter parameters
    cdef vector[double] f_vt, t_vt
    cdef vector[double] f_offset_vt
    cdef vector[double] t_pstart_vt

    # frequency
    cdef double[:] f_mv = wf_prop["f"].astype(np.float64)
    Mem_Copy(&f_mv[0], <int_t>len(f_mv), f_vt)

    # time
    cdef double[:] t_mv = wf_prop["t"].astype(np.float64)
    Mem_Copy(&t_mv[0], <int_t>len(t_mv), t_vt)

    # frequency offset per pulse
    cdef double[:] f_offset_mv = wf_prop["f_offset"].astype(np.float64)
    Mem_Copy(&f_offset_mv[0], <int_t>len(f_offset_mv), f_offset_vt)

    # pulse start time
    cdef double[:] t_pstart_mv = wf_prop["pulse_start_time"].astype(np.float64)
    Mem_Copy(&t_pstart_mv[0], <int_t>len(t_pstart_mv), t_pstart_vt)

    # phase noise
    cdef vector[double] pn_freq_vt
    cdef vector[double] pn_power_vt
    cdef double[:] pn_freq_mv
    cdef double[:] pn_power_mv

    if sample_prop.get("pn_f") is not None:
        pn_freq_mv = np.asarray(sample_prop["pn_f"]).astype(np.float64)
        pn_power_mv = np.asarray(sample_prop["pn_power"]).astype(np.float64)
        Mem_Copy(&pn_freq_mv[0], <int_t>len(pn_freq_mv), pn_freq_vt)
        Mem_Copy(&pn_power_mv[0], <int_t>len(pn_power_mv), pn_power_vt)

        return make_shared[Transmitter[double, float_t]](
            <float_t> tx_prop.rf_prop["tx_power"],
            f_vt,
            t_vt,
            f_offset_vt,
            t_pstart_vt,
            pn_freq_vt,
            pn_power_vt,
            <double> sample_prop["pn_fs"],
            <int> 0,
            <unsigned long long> sample_prop["pn_seed"],
            <bool> sample_prop["pn_validation"]
        )

    return make_shared[Transmitter[double, float_t]](
        <float_t> tx_prop.rf_prop["tx_power"],
        f_vt,
        t_vt,
        f_offset_vt,
        t_pstart_vt
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddTxChannel(tx, tx_idx, Transmitter[double, float_t] * tx_c):
    """
    Create TxChannel object in Cython with antenna pattern processing.

    :param Transmitter tx:
        Python transmitter object containing channel configurations
        (antenna patterns, polarization, modulation, locations, gains)
    :param int tx_idx:
        Transmitter channel index (0-based) for multi-channel systems
    :param Transmitter[double, float_t] * tx_c:
        Pointer to C++ transmitter object
    :raises: ValueError for invalid channel index or pattern data
    """
    # Cache channel properties
    txch = tx.txchannel_prop
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
    az_angles = txch["az_angles"][tx_idx]
    az_patterns = txch["az_patterns"][tx_idx]
    az_ang_mv = np.radians(np.asarray(az_angles)).astype(np_float)
    az_ptn_mv = np.asarray(az_patterns).astype(np_float)
    Mem_Copy(&az_ang_mv[0], <int_t>len(az_angles), az_ang_vt)
    Mem_Copy(&az_ptn_mv[0], <int_t>len(az_patterns), az_ptn_vt)

    # elevation pattern
    el_angles = txch["el_angles"][tx_idx]
    el_patterns = txch["el_patterns"][tx_idx]
    el_ang_mv = np.radians(np.flip(90 - np.asarray(el_angles))).astype(np_float)
    el_ptn_mv = np.flip(np.asarray(el_patterns)).astype(np_float)
    Mem_Copy(&el_ang_mv[0], <int_t>len(el_angles), el_ang_vt)
    Mem_Copy(&el_ptn_mv[0], <int_t>len(el_patterns), el_ptn_vt)

    # pulse modulation
    pulse_mod = txch["pulse_mod"][tx_idx]
    cdef float_t[:] pulse_real_mv = np.real(pulse_mod).astype(np_float)
    cdef float_t[:] pulse_imag_mv = np.imag(pulse_mod).astype(np_float)
    Mem_Copy_Complex(&pulse_real_mv[0], &pulse_imag_mv[0], pulses_c, pulse_mod_vt)

    # waveform modulation
    wf_mod = txch["waveform_mod"][tx_idx]
    mod_enabled = wf_mod["enabled"]

    cdef float_t[:] mod_real_mv, mod_imag_mv
    cdef float_t[:] mod_t_mv
    if mod_enabled:
        mod_var = wf_mod["var"]
        mod_t = wf_mod["t"]
        mod_real_mv = np.real(mod_var).astype(np_float)
        mod_imag_mv = np.imag(mod_var).astype(np_float)
        mod_t_mv = mod_t.astype(np_float)
        Mem_Copy_Complex(&mod_real_mv[0], &mod_imag_mv[0], <int_t>len(mod_var), mod_var_vt)
        Mem_Copy(&mod_t_mv[0], <int_t>len(mod_t), mod_t_vt)

    cdef float_t[:] location_mv = txch["locations"][tx_idx].astype(np_float)

    polar = txch["polarization"][tx_idx]
    cdef Vec3[cpp_complex[float_t]] polarization_vt = Vec3[cpp_complex[float_t]](
        cpp_complex[float_t](<float_t>np.real(polar[0]), <float_t>np.imag(polar[0])),
        cpp_complex[float_t](<float_t>np.real(polar[1]), <float_t>np.imag(polar[1])),
        cpp_complex[float_t](<float_t>np.real(polar[2]), <float_t>np.imag(polar[2]))
    )

    tx_c[0].AddChannel(
        Vec3[float_t](&location_mv[0]),
        polarization_vt,
        az_ang_vt,
        az_ptn_vt,
        el_ang_vt,
        el_ptn_vt,
        <float_t> txch["antenna_gains"][tx_idx],
        mod_t_vt,
        mod_var_vt,
        pulse_mod_vt,
        <float_t> txch["delay"][tx_idx],
        <float_t> np.radians(txch["grid"][tx_idx])
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddRxChannel(rx, rx_idx, Receiver[float_t] * rx_c):
    """
    Create RxChannel object in Cython with antenna pattern processing.

    :param Receiver rx:
        Python receiver object containing channel configurations
        (antenna patterns, polarization, locations, gains)
    :param int rx_idx:
        Receiver channel index (0-based) for multi-channel systems
    :param Receiver[float_t] * rx_c:
        Pointer to C++ receiver object
    :raises: ValueError for invalid channel index or pattern data
    """
    # Cache channel properties
    rxch = rx.rxchannel_prop

    cdef vector[float_t] az_ang_vt, az_ptn_vt
    cdef vector[float_t] el_ang_vt, el_ptn_vt

    cdef float_t[:] az_ang_mv, az_ptn_mv
    cdef float_t[:] el_ang_mv, el_ptn_mv

    # azimuth pattern
    az_angles = rxch["az_angles"][rx_idx]
    az_patterns = rxch["az_patterns"][rx_idx]
    az_ang_mv = np.radians(np.asarray(az_angles)).astype(np_float)
    az_ptn_mv = np.asarray(az_patterns).astype(np_float)
    Mem_Copy(&az_ang_mv[0], <int_t>len(az_angles), az_ang_vt)
    Mem_Copy(&az_ptn_mv[0], <int_t>len(az_patterns), az_ptn_vt)

    # elevation pattern
    el_angles = rxch["el_angles"][rx_idx]
    el_patterns = rxch["el_patterns"][rx_idx]
    el_ang_mv = np.radians(np.flip(90 - np.asarray(el_angles))).astype(np_float)
    el_ptn_mv = np.flip(np.asarray(el_patterns)).astype(np_float)
    Mem_Copy(&el_ang_mv[0], <int_t>len(el_angles), el_ang_vt)
    Mem_Copy(&el_ptn_mv[0], <int_t>len(el_patterns), el_ptn_vt)

    cdef float_t[:] location_mv = rxch["locations"][rx_idx].astype(np_float)

    polar = rxch["polarization"][rx_idx]
    cdef Vec3[cpp_complex[float_t]] polarization_vt = Vec3[cpp_complex[float_t]](
        cpp_complex[float_t](<float_t>np.real(polar[0]), <float_t>np.imag(polar[0])),
        cpp_complex[float_t](<float_t>np.real(polar[1]), <float_t>np.imag(polar[1])),
        cpp_complex[float_t](<float_t>np.real(polar[2]), <float_t>np.imag(polar[2]))
    )

    rx_c[0].AddChannel(
        Vec3[float_t](&location_mv[0]),
        polarization_vt,
        az_ang_vt,
        az_ptn_vt,
        el_ang_vt,
        el_ptn_vt,
        <float_t> rxch["antenna_gains"][rx_idx]
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef shared_ptr[Radar[double, float_t]] cp_Radar(radar, frame_start_time):
    """
    Create complete Radar system object for simulation.

    :param Radar radar:
        Python radar object containing system configuration (array properties,
        transmitter/receiver settings, radar motion parameters)
    :param frame_start_time:
        Timing information for multi-frame simulations (single value or array)
    :return: C++ Radar object ready for simulation
    :rtype: shared_ptr[Radar[double, float_t]]
    :raises: ValueError for invalid configuration, RuntimeError for setup failures
    """
    cdef shared_ptr[Transmitter[double, float_t]] tx_c
    cdef shared_ptr[Receiver[float_t]] rx_c

    # Cache nested property lookups
    radar_prop = radar.radar_prop
    tx_obj = radar_prop["transmitter"]
    rx_obj = radar_prop["receiver"]

    cdef int_t txsize_c = tx_obj.txchannel_prop["size"]
    cdef int_t rxsize_c = rx_obj.rxchannel_prop["size"]
    cdef int_t frames_c = np.size(frame_start_time)
    cdef int_t channles_c = radar.array_prop["size"]
    cdef int_t pulses_c = tx_obj.waveform_prop["pulses"]
    cdef int_t samples_c = radar.sample_prop["samples_per_pulse"]

    cdef int_t bbsize_c = channles_c * frames_c * pulses_c * samples_c

    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] rotx_mv, roty_mv, rotz_mv

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

    # Transmitter
    tx_c = cp_Transmitter(radar)
    for idx_c in range(0, txsize_c):
        cp_AddTxChannel(tx_obj, idx_c, tx_c.get())

    # Receiver
    rx_bb = rx_obj.bb_prop
    rx_c = make_shared[Receiver[float_t]](
        <float_t> rx_bb["fs"],
        <float_t> rx_obj.rf_prop["rf_gain"],
        <float_t> rx_bb["load_resistor"],
        <float_t> rx_bb["baseband_gain"],
        <float_t> rx_bb["noise_bandwidth"]
    )
    for idx_c in range(0, rxsize_c):
        cp_AddRxChannel(rx_obj, idx_c, rx_c.get())

    # Radar location and rotation
    cdef float_t[:] loc_mv, rot_mv
    radar_location = radar_prop["location"]
    radar_rotation = radar_prop["rotation"]

    if len(np.shape(radar_location)) == 4:
        locx_mv = radar_location[:,:,:,0].astype(np_float)
        locy_mv = radar_location[:,:,:,1].astype(np_float)
        locz_mv = radar_location[:,:,:,2].astype(np_float)
        rotx_mv = radar_rotation[:,:,:,0].astype(np_float)
        roty_mv = radar_rotation[:,:,:,1].astype(np_float)
        rotz_mv = radar_rotation[:,:,:,2].astype(np_float)

        Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)
        Mem_Copy_Vec3(&rotx_mv[0,0,0], &roty_mv[0,0,0], &rotz_mv[0,0,0], bbsize_c, rot_vt)

    else:
        loc_mv = radar_location.astype(np_float)
        loc_vt.push_back(Vec3[float_t](&loc_mv[0]))

        rot_mv = radar_rotation.astype(np_float)
        rot_vt.push_back(Vec3[float_t](&rot_mv[0]))
    
    radar_speed = radar_prop["speed"]
    radar_rot_rate = radar_prop["rotation_rate"]
    spd_vt = Vec3[float_t](<float_t>radar_speed[0], <float_t>radar_speed[1], <float_t>radar_speed[2])
    rrt_vt = Vec3[float_t](<float_t>radar_rot_rate[0], <float_t>radar_rot_rate[1], <float_t>radar_rot_rate[2])

    return make_shared[Radar[double, float_t]](tx_c,
                                               rx_c,
                                               t_frame_vt,
                                               loc_vt,
                                               spd_vt,
                                               rot_vt,
                                               rrt_vt)

# ============================================================================
# Mesh Targets (Radar Simulation and RCS)
# ============================================================================

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddTarget(radar,
                  target,
                  timestamp,
                  mesh_module,
                  TargetsManager[float_t] * targets_manager):
    """
    Add a complex target to the radar simulation.

    :param Radar radar:
        Radar object containing system configuration
    :param dict target:
        Target properties (model, location, speed, rotation, materials, etc.)
    :param timestamp:
        Time array for simulation frames
    :param mesh_module:
        Mesh loading module for processing 3D models
    :param TargetsManager[float_t] * targets_manager:
        Pointer to C++ targets manager
    :raises: ValueError for invalid target config, RuntimeError for mesh issues
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

    ts_shape = np.shape(timestamp)
    cdef int_t bbsize_c = <int_t>(ts_shape[0] * ts_shape[1] * ts_shape[2])

    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv

    _validate_target_keys(target)

    # Load and validate mesh
    points_arr, cells_arr = _load_and_validate_mesh(target, mesh_module)
    points_mv = points_arr
    cells_mv = cells_arr

    cdef float_t[:] origin_mv = np.asarray(target.get("origin", (0, 0, 0)), dtype=np_float)

    location = list(target.get("location", [0, 0, 0]))
    speed = list(target.get("speed", [0, 0, 0]))
    rotation = list(target.get("rotation", [0, 0, 0]))
    rotation_rate = list(target.get("rotation_rate", [0, 0, 0]))

    cdef float_t[:] location_mv, speed_mv, rotation_mv, rotation_rate_mv

    _parse_material_properties(target, &ep_c, &mu_c)

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
    
    _handle_deprecated_target_params(target)

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
                           <bool> target.get("skip_diffusion", False),
                           <float_t> target.get("density", 0.0),
                           <bool> target.get("environment", False))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_RCS_Target(target, mesh_module, TargetsManager[float_t] * targets_manager):
    """
    Create Target object in Cython for RCS calculation.

    :param dict target:
        Target properties dictionary containing model, location, materials
    :param mesh_module:
        Mesh loading module for processing 3D models
    :param TargetsManager[float_t] * targets_manager:
        Pointer to C++ targets manager
    :raises: RuntimeError on mesh limitations, ValueError on invalid target
    """
    # Vector declarations
    cdef vector[Vec3[float_t]] loc_vt, spd_vt, rot_vt, rrt_vt
    cdef cpp_complex[float_t] ep_c, mu_c
    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv

    _validate_target_keys(target)

    # Load and validate mesh
    points_arr, cells_arr = _load_and_validate_mesh(target, mesh_module)
    points_mv = points_arr
    cells_mv = cells_arr

    cdef float_t[:] origin_mv = np.asarray(target.get("origin", (0, 0, 0)), dtype=np_float)

    location = np.asarray(target.get("location", (0, 0, 0)), dtype=object)
    speed = np.asarray(target.get("speed", (0, 0, 0)), dtype=object)
    rotation = np.asarray(target.get("rotation", (0, 0, 0)), dtype=object)
    rotation_rate = np.asarray(target.get("rotation_rate", (0, 0, 0)), dtype=object)

    cdef float_t[:] location_mv, speed_mv, rotation_mv, rotation_rate_mv

    _parse_material_properties(target, &ep_c, &mu_c)

    location_mv = location.astype(np_float)
    loc_vt.push_back(Vec3[float_t](&location_mv[0]))

    speed_mv = speed.astype(np_float)
    spd_vt.push_back(Vec3[float_t](&speed_mv[0]))

    rotation_mv = np.radians(rotation.astype(np_float)).astype(np_float)
    rot_vt.push_back(Vec3[float_t](&rotation_mv[0]))

    rotation_rate_mv = np.radians(rotation_rate.astype(np_float)).astype(np_float)
    rrt_vt.push_back(Vec3[float_t](&rotation_rate_mv[0]))

    _handle_deprecated_target_params(target)

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
                           <bool> target.get("skip_diffusion", False),
                           <float_t> target.get("density", 0.0),
                           <bool> target.get("environment", False))
