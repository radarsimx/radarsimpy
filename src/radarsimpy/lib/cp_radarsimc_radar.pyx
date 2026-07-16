# distutils: language = c++
"""
Radar-system wrappers for cp_radarsimc.

This file is textually merged into the ``cp_radarsimc`` extension module via
``include`` (see cp_radarsimc.pyx). It builds the C++ Transmitter, Receiver,
and Radar objects (including antenna-pattern and modulation processing) from
the Python radar configuration.

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

# Cython imports
cimport cython
cimport numpy as np
from libcpp.complex cimport complex as cpp_complex
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared
from radarsimpy.includes.radarsimc cimport (
    Transmitter, Receiver,
    Radar,
    Mem_Copy, Mem_Copy_Complex,
)
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t, float_t, vector


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
