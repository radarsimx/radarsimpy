# distutils: language = c++

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - PRESENT  Zhengyu Peng
# E-mail: zpeng.me@gmail.com
# Website: https://zpeng.me

# `                      `
# -:.                  -#:
# -//:.              -###:
# -////:.          -#####:
# -/:.://:.      -###++##:
# ..   `://:-  -###+. :##:
#        `:/+####+.   :##:
# .::::::::/+###.     :##:
# .////-----+##:    `:###:
#  `-//:.   :##:  `:###/.
#    `-//:. :##:`:###/.
#      `-//:+######/.
#        `-/+####/.
#          `+##+.
#           :##:
#           :##:
#           :##:
#           :##:
#           :##:
#            .+:


import meshio

from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t
from radarsimpy.includes.type_def cimport vector

from libcpp.complex cimport complex as cpp_complex
from libcpp cimport bool

import numpy as np

cimport cython
cimport numpy as np

np_float = np.float32

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point[float_t] cp_Point(location,
                             speed,
                             rcs,
                             phase,
                             shape):
    """
    cp_Point(location, speed, rcs, phase, shape)

    Creat Point object in Cython

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
        Shape of the time matrix

    :return: C++ object of a point target
    :rtype: Point
    """
    cdef vector[Vec3[float_t]] loc_vt
    cdef vector[float_t] rcs_vt, phs_vt

    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] rcs_mv, phs_mv

    cdef int_t bbsize_c = <int_t>(shape[0]*shape[1]*shape[2])

    cdef float_t[:] speed_mv = np.array(speed, dtype=np_float)
    cdef float_t[:] location_mv

    # check if there are any time varying parameters
    if np.size(location[0]) > 1 or \
            np.size(location[1]) > 1 or \
            np.size(location[2]) > 1 or \
            np.size(rcs) > 1 or \
            np.size(phase) > 1:

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
    
    return Point[float_t](
        loc_vt,
        Vec3[float_t](&speed_mv[0]),
        rcs_vt,
        phs_vt
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Transmitter[float_t] cp_Transmitter(radar):
    """
    cp_Transmitter(radar)

    Creat Transmitter object in Cython

    :param Radar radar:
        Radar object

    :return: C++ object of a radar transmitter
    :rtype: Transmitter
    """
    cdef int_t frames_c = radar.frames
    cdef int_t channles_c = radar.channel_size
    cdef int_t pulses_c = radar.transmitter.pulses
    cdef int_t samples_c = radar.samples_per_pulse

    cdef vector[double] t_frame_vt
    cdef vector[double] f_vt, t_vt
    cdef vector[double] f_offset_vt
    cdef vector[double] t_pstart_vt
    cdef vector[cpp_complex[double]] pn_vt
    
    # frame time offset
    cdef double[:] t_frame_mv
    if frames_c > 1:
        t_frame_mv = radar.t_offset.astype(np.float64)
        Mem_Copy(&t_frame_mv[0], frames_c, t_frame_vt)
    else:
        t_frame_vt.push_back(<double> (radar.t_offset))

    # frequency
    cdef double[:] f_mv = radar.f.astype(np.float64)
    Mem_Copy(&f_mv[0], <int_t>(len(radar.f)), f_vt)

    # time
    cdef double[:] t_mv = radar.t.astype(np.float64)
    Mem_Copy(&t_mv[0], <int_t>(len(radar.t)), t_vt)

    # frequency offset per pulse
    cdef double[:] f_offset_mv = radar.transmitter.f_offset.astype(np.float64)
    Mem_Copy(&f_offset_mv[0], <int_t>(len(radar.transmitter.f_offset)), f_offset_vt)

    # pulse start time
    cdef double[:] t_pstart_mv = radar.transmitter.pulse_start_time.astype(np.float64)
    Mem_Copy(&t_pstart_mv[0], <int_t>(len(radar.transmitter.pulse_start_time)), t_pstart_vt)

    # phase noise
    cdef double[:, :, :] pn_real_mv
    cdef double[:, :, :] pn_imag_mv
    if radar.phase_noise is not None:
        pn_real_mv = np.real(radar.phase_noise).astype(np.float64)
        pn_imag_mv = np.imag(radar.phase_noise).astype(np.float64)
        Mem_Copy_Complex(&pn_real_mv[0,0,0], &pn_imag_mv[0,0,0], <int_t>(frames_c*channles_c*pulses_c*samples_c), pn_vt)

    return Transmitter[float_t](
        f_vt,
        f_offset_vt,
        t_vt,
        <float_t> radar.transmitter.tx_power,
        t_pstart_vt,
        t_frame_vt,
        pn_vt
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef TxChannel[float_t] cp_TxChannel(tx,
                                     tx_idx):
    """
    TxChannel(tx, tx_idx)

    Creat TxChannel object in Cython

    :param Transmitter tx:
        Radar transmitter
    :param int tx_idx:
        Tx channel index

    :return: C++ object of a transmitter channel
    :rtype: TxChannel
    """
    cdef int_t pulses_c = tx.pulses

    cdef vector[float_t] az_ang_vt, az_ptn_vt
    cdef vector[float_t] el_ang_vt, el_ptn_vt

    cdef float_t[:] az_ang_mv, az_ptn_mv
    cdef float_t[:] el_ang_mv, el_ptn_mv

    cdef vector[cpp_complex[float_t]] pulse_mod_vt

    cdef bool mod_enabled
    cdef vector[cpp_complex[float_t]] mod_var_vt
    cdef vector[float_t] mod_t_vt

    # azimuth pattern
    az_ang_mv = np.radians(np.array(tx.az_angles[tx_idx])).astype(np_float)
    az_ptn_mv = np.array(tx.az_patterns[tx_idx]).astype(np_float)

    Mem_Copy(&az_ang_mv[0], <int_t>(len(tx.az_angles[tx_idx])), az_ang_vt)
    Mem_Copy(&az_ptn_mv[0], <int_t>(len(tx.az_patterns[tx_idx])), az_ptn_vt)

    # elevation pattern
    el_ang_mv = np.radians(np.flip(90-tx.el_angles[tx_idx])).astype(np_float)
    el_ptn_mv = np.flip(tx.el_patterns[tx_idx]).astype(np_float)

    Mem_Copy(&el_ang_mv[0], <int_t>(len(tx.el_angles[tx_idx])), el_ang_vt)
    Mem_Copy(&el_ptn_mv[0], <int_t>(len(tx.el_patterns[tx_idx])), el_ptn_vt)

    # pulse modulation
    cdef float_t[:] pulse_real_mv = np.real(tx.pulse_mod[tx_idx]).astype(np_float)
    cdef float_t[:] pulse_imag_mv = np.imag(tx.pulse_mod[tx_idx]).astype(np_float)
    Mem_Copy_Complex(&pulse_real_mv[0], &pulse_imag_mv[0], <int_t>(pulses_c), pulse_mod_vt)

    # waveform modulation
    mod_enabled = tx.waveform_mod[tx_idx]['enabled']

    cdef float_t[:] mod_real_mv, mod_imag_mv
    cdef float_t[:] mod_t_mv
    if mod_enabled:
        mod_real_mv = np.real(tx.waveform_mod[tx_idx]['var']).astype(np_float)
        mod_imag_mv = np.imag(tx.waveform_mod[tx_idx]['var']).astype(np_float)
        mod_t_mv = tx.waveform_mod[tx_idx]['t'].astype(np_float)

        Mem_Copy_Complex(&mod_real_mv[0], &mod_imag_mv[0], <int_t>(len(tx.waveform_mod[tx_idx]['var'])), mod_var_vt)
        Mem_Copy(&mod_t_mv[0], <int_t>(len(tx.waveform_mod[tx_idx]['t'])), mod_t_vt)

    cdef float_t[:] location_mv = tx.locations[tx_idx].astype(np_float)
    cdef float_t[:] polarization_mv = tx.polarization[tx_idx].astype(np_float)
    return TxChannel[float_t](
        Vec3[float_t](&location_mv[0]),
        Vec3[float_t](&polarization_mv[0]),
        az_ang_vt,
        az_ptn_vt,
        el_ang_vt,
        el_ptn_vt,
        <float_t> tx.antenna_gains[tx_idx],
        mod_t_vt,
        mod_var_vt,
        pulse_mod_vt,
        <float_t> tx.delay[tx_idx],
        <float_t> np.radians(tx.grid[tx_idx])
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef RxChannel[float_t] cp_RxChannel(rx,
                                     rx_idx):
    """
    cp_RxChannel(tx, tx_idx)

    Creat RxChannel object in Cython

    :param Receiver rx:
        Radar receiver
    :param int rx_idx:
        Rx channel index

    :return: C++ object of a receiver channel
    :rtype: RxChannel
    """
    cdef vector[float_t] az_ang_vt, az_ptn_vt
    cdef vector[float_t] el_ang_vt, el_ptn_vt

    cdef float_t[:] az_ang_mv, az_ptn_mv
    cdef float_t[:] el_ang_mv, el_ptn_mv

    # azimuth pattern
    az_ang_mv = np.radians(rx.az_angles[rx_idx]).astype(np_float)
    az_ptn_mv = rx.az_patterns[rx_idx].astype(np_float)

    Mem_Copy(&az_ang_mv[0], <int_t>(len(rx.az_angles[rx_idx])), az_ang_vt)
    Mem_Copy(&az_ptn_mv[0], <int_t>(len(rx.az_patterns[rx_idx])), az_ptn_vt)

    # elevation pattern
    el_ang_mv = np.radians(np.flip(90-rx.el_angles[rx_idx])).astype(np_float)
    el_ptn_mv = np.flip(rx.el_patterns[rx_idx]).astype(np_float)

    Mem_Copy(&el_ang_mv[0], <int_t>(len(rx.el_angles[rx_idx])), el_ang_vt)
    Mem_Copy(&el_ptn_mv[0], <int_t>(len(rx.el_patterns[rx_idx])), el_ptn_vt)

    cdef float_t[:] location_mv = rx.locations[rx_idx].astype(np_float)
    return RxChannel[float_t](
        Vec3[float_t](&location_mv[0]),
        Vec3[float_t](0, 0, 1),
        az_ang_vt,
        az_ptn_vt,
        el_ang_vt,
        el_ptn_vt,
        <float_t> rx.antenna_gains[rx_idx]
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Target[float_t] cp_Target(radar,
                               target,
                               shape):
    """
    cp_Target(tx, tx_idx)

    Creat Target object in Cython

    :param Radar radar:
        Radar object
    :param dict target:
        Target properties
    :param tuple shape:
        Shape of the time matrix

    :return: C++ object of a target
    :rtype: Target
    """
    timestamp = radar.timestamp

    # vector of location, speed, rotation, rotation rate
    cdef vector[Vec3[float_t]] loc_vt
    cdef vector[Vec3[float_t]] spd_vt
    cdef vector[Vec3[float_t]] rot_vt
    cdef vector[Vec3[float_t]] rrt_vt

    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] spd_x, spd_y, spd_z
    cdef float_t[:, :, :] rot_x, rot_y, rot_z
    cdef float_t[:, :, :] rrt_x, rrt_y, rrt_z

    cdef cpp_complex[float_t] ep_c, mu_c

    cdef int_t ch_idx, ps_idx, sp_idx
    cdef int_t bbsize_c = <int_t>(radar.channel_size*radar.frames*radar.transmitter.pulses*radar.samples_per_pulse)

    t_mesh = meshio.read(target['model'])
    cdef float_t[:, :] points_mv = t_mesh.points.astype(np_float)
    cdef int_t[:, :] cells_mv = t_mesh.cells[0].data.astype(np.int32)
    cdef float_t[:] origin_mv = np.array(target.get('origin', (0, 0, 0)), dtype=np_float)

    location = np.array(target.get('location', (0, 0, 0)), dtype=object)
    speed = np.array(target.get('speed', (0, 0, 0)), dtype=object)
    rotation = np.array(target.get('rotation', (0, 0, 0)), dtype=object)
    rotation_rate = np.array(target.get( 'rotation_rate', (0, 0, 0)), dtype=object)

    permittivity = target.get('permittivity', 'PEC')
    if permittivity == "PEC":
        ep_c = cpp_complex[float_t](-1, 0)
        mu_c = cpp_complex[float_t](1, 0)
    else:
        ep_c = cpp_complex[float_t](np.real(permittivity), np.imag(permittivity))
        mu_c = cpp_complex[float_t](1, 0)

    if np.size(location[0]) > 1 or \
        np.size(location[1]) > 1 or \
        np.size(location[2]) > 1 or \
        np.size(speed[0]) > 1 or \
        np.size(speed[1]) > 1 or \
        np.size(speed[2]) > 1 or \
        np.size(rotation[0]) > 1 or \
        np.size(rotation[1]) > 1 or \
        np.size(rotation[2]) > 1 or \
        np.size(rotation_rate[0]) > 1 or \
        np.size(rotation_rate[1]) > 1 or \
        np.size(rotation_rate[2]) > 1:

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
            spd_x = speed[0].astype(np_float)
        else:
            spd_x = np.full(shape, speed[0], dtype=np_float)

        if np.size(speed[1]) > 1:
            spd_y = speed[1].astype(np_float)
        else:
            spd_y = np.full(shape, speed[1], dtype=np_float)

        if np.size(speed[2]) > 1:
            spd_z = speed[2].astype(np_float)
        else:
            spd_z = np.full(shape, speed[2], dtype=np_float)

        if np.size(rotation[0]) > 1:
            rot_x = np.radians(rotation[0]).astype(np_float)
        else:
            rot_x = np.radians(
                rotation[0] + rotation_rate[0]*timestamp).astype(np_float)

        if np.size(rotation[1]) > 1:
            rot_y = np.radians(rotation[1]).astype(np_float)
        else:
            rot_y = np.radians(
                rotation[1] + rotation_rate[1]*timestamp).astype(np_float)

        if np.size(rotation[2]) > 1:
            rot_z = np.radians(rotation[2]).astype(np_float)
        else:
            rot_z = np.radians(
                rotation[2] + rotation_rate[2]*timestamp).astype(np_float)

        if np.size(rotation_rate[0]) > 1:
            rrt_x = np.radians(rotation_rate[0]).astype(np_float)
        else:
            rrt_x = np.full(shape, np.radians(rotation_rate[0]), dtype=np_float)

        if np.size(rotation_rate[1]) > 1:
            rrt_y = np.radians(rotation_rate[1]).astype(np_float)
        else:
            rrt_y = np.full(shape, np.radians(rotation_rate[1]), dtype=np_float)

        if np.size(rotation_rate[2]) > 1:
            rrt_z = np.radians(rotation_rate[2]).astype(np_float)
        else:
            rrt_z = np.full(shape, np.radians(rotation_rate[2]), dtype=np_float)

        Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)
        Mem_Copy_Vec3(&spd_x[0,0,0], &spd_y[0,0,0], &spd_z[0,0,0], bbsize_c, spd_vt)
        Mem_Copy_Vec3(&rot_x[0,0,0], &rot_y[0,0,0], &rot_z[0,0,0], bbsize_c, rot_vt)
        Mem_Copy_Vec3(&rrt_x[0,0,0], &rrt_y[0,0,0], &rrt_z[0,0,0], bbsize_c, rrt_vt)

    else:
        loc_vt.push_back(
            Vec3[float_t](
                <float_t> location[0],
                <float_t> location[1],
                <float_t> location[2]
            )
        )
        spd_vt.push_back(
            Vec3[float_t](
                <float_t> speed[0],
                <float_t> speed[1],
                <float_t> speed[2]
            )
        )
        rot_vt.push_back(
            Vec3[float_t](
                <float_t> np.radians(rotation[0]),
                <float_t> np.radians(rotation[1]),
                <float_t> np.radians(rotation[2])
            )
        )
        rrt_vt.push_back(
            Vec3[float_t](
                <float_t> np.radians(rotation_rate[0]),
                <float_t> np.radians(rotation_rate[1]),
                <float_t> np.radians(rotation_rate[2])
            )
        )

    return Target[float_t](&points_mv[0, 0],
                           &cells_mv[0, 0],
                           <int_t> cells_mv.shape[0],
                           Vec3[float_t](&origin_mv[0]),
                           loc_vt,
                           spd_vt,
                           rot_vt,
                           rrt_vt,
                           ep_c,
                           mu_c,
                           <bool> target.get('is_ground', False))

