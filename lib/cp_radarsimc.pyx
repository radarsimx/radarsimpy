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
    cdef vector[Vec3[float_t]] loc_vect
    cdef vector[float_t] rcs_vect
    cdef vector[float_t] phs_vect

    cdef float_t[:, :, :] tgt_loc_x
    cdef float_t[:, :, :] tgt_loc_y
    cdef float_t[:, :, :] tgt_loc_z

    cdef float_t[:, :, :] rcs_mem
    cdef float_t[:, :, :] phs_mem

    # check if there are any time varying parameters
    if np.size(location[0]) > 1 or \
            np.size(location[1]) > 1 or \
            np.size(location[2]) > 1 or \
            np.size(rcs) > 1 or \
            np.size(phase) > 1:

        if np.size(location[0]) > 1:
            tgt_loc_x = location[0].astype(np.float32)
        else:
            tgt_loc_x = np.full(shape, location[0], dtype=np.float32)

        if np.size(location[1]) > 1:
            tgt_loc_y = location[1].astype(np.float32)
        else:
            tgt_loc_y = np.full(shape, location[1], dtype=np.float32)

        if np.size(location[2]) > 1:
            tgt_loc_z = location[2].astype(np.float32)
        else:
            tgt_loc_z = np.full(shape, location[2], dtype=np.float32)

        if np.size(rcs) > 1:
            rcs_mem = rcs.astype(np.float32)
        else:
            rcs_mem = np.full(shape, rcs, dtype=np.float32)

        if np.size(phase) > 1:
            phs_mem = np.radians(phase).astype(np.float32)
        else:
            phs_mem = np.full(shape, np.radians(phase), dtype=np.float32)

        for ch_idx in range(0, shape[0]):
            for ps_idx in range(0, shape[1]):
                for sp_idx in range(0, shape[2]):
                    loc_vect.push_back(Vec3[float_t](
                        <float_t> tgt_loc_x[ch_idx, ps_idx, sp_idx],
                        <float_t> tgt_loc_y[ch_idx, ps_idx, sp_idx],
                        <float_t> tgt_loc_z[ch_idx, ps_idx, sp_idx]
                    ))
                    rcs_vect.push_back(<float_t> rcs_mem[ch_idx, ps_idx, sp_idx])
                    phs_vect.push_back(<float_t> phs_mem[ch_idx, ps_idx, sp_idx])
    else:
        loc_vect.push_back(Vec3[float_t](
            <float_t> location[0],
            <float_t> location[1],
            <float_t> location[2]
        ))
        rcs_vect.push_back(<float_t> rcs)
        phs_vect.push_back(<float_t> np.radians(phase))
    
    return Point[float_t](
        loc_vect,
        Vec3[float_t](
            <float_t> speed[0],
            <float_t> speed[1],
            <float_t> speed[2]
        ),
        rcs_vect,
        phs_vect
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
    cdef int_t frames = radar.frames
    cdef int_t channles = radar.channel_size
    cdef int_t pulses = radar.transmitter.pulses
    cdef int_t samples = radar.samples_per_pulse

    cdef vector[double] t_frame_vect
    cdef vector[double] f_vect
    cdef vector[double] t_vect
    cdef vector[double] f_offset_vect
    cdef vector[double] t_pstart_vect
    cdef vector[cpp_complex[double]] pn_vect

    if frames > 1:
        t_frame_mem = radar.t_offset.astype(np.float64)
        t_frame_vect.reserve(frames)
        for idx in range(0, frames):
            t_frame_vect.push_back(t_frame_mem[idx])
    else:
        t_frame_vect.push_back(<double> (radar.t_offset))

    f_mem = radar.f.astype(np.float64)
    f_vect.reserve(len(radar.f))
    for idx in range(0, len(radar.f)):
        f_vect.push_back(f_mem[idx])

    t_mem = radar.t.astype(np.float64)
    t_vect.reserve(len(radar.t))
    for idx in range(0, len(radar.t)):
        t_vect.push_back(t_mem[idx])

    f_offset_mem = radar.transmitter.f_offset.astype(np.float64)
    f_offset_vect.reserve(len(radar.transmitter.f_offset))
    for idx in range(0, len(radar.transmitter.f_offset)):
        f_offset_vect.push_back(f_offset_mem[idx])

    t_pstart_mem = radar.transmitter.pulse_start_time.astype(np.float64)
    t_pstart_vect.reserve(len(radar.transmitter.pulse_start_time))
    for idx in range(0, len(radar.transmitter.pulse_start_time)):
        t_pstart_vect.push_back(t_pstart_mem[idx])

    if radar.phase_noise is not None:
        pn_vect.reserve(frames*channles*pulses*samples)
        for idx0 in range(0, frames*channles):
            for idx1 in range(0, pulses):
                for idx2 in range(0, samples):
                    pn_vect.push_back(cpp_complex[double](
                        np.real(radar.phase_noise[idx0, idx1, idx2]),
                        np.imag(radar.phase_noise[idx0, idx1, idx2])
                    ))

    return Transmitter[float_t](
        f_vect,
        f_offset_vect,
        t_vect,
        <float_t> radar.transmitter.tx_power,
        t_pstart_vect,
        t_frame_vect,
        pn_vect
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef TxChannel[float_t] cp_TxChannel(tx,
                                     tx_idx):
    cdef int_t pulses = tx.pulses

    cdef vector[float_t] az_ang_vect, az_ptn_vect
    cdef vector[float_t] el_ang_vect, el_ptn_vect

    cdef vector[cpp_complex[float_t]] pulse_mod_vect

    cdef bool mod_enabled
    cdef vector[cpp_complex[float_t]] mod_var_vect
    cdef vector[float_t] mod_t_vect

    az_ang_vect.reserve(len(tx.az_angles[tx_idx]))
    for idx in range(0, len(tx.az_angles[tx_idx])):
        az_ang_vect.push_back(<float_t> np.radians(tx.az_angles[tx_idx][idx]))
    az_ptn_vect.reserve(len(tx.az_patterns[tx_idx]))
    for idx in range(0, len(tx.az_patterns[tx_idx])):
        az_ptn_vect.push_back(<float_t> (tx.az_patterns[tx_idx][idx]))

    el_ang_mem = np.radians(
        np.flip(90-tx.el_angles[tx_idx].astype(np.float32)))
    el_ptn_mem = np.flip(tx.el_patterns[tx_idx].astype(np.float32))
    el_ang_vect.reserve(len(tx.el_angles[tx_idx]))
    for idx in range(0, len(tx.el_angles[tx_idx])):
        el_ang_vect.push_back(el_ang_mem[idx])
    el_ptn_vect.reserve(len(tx.el_patterns[tx_idx]))
    for idx in range(0, len(tx.el_patterns[tx_idx])):
        el_ptn_vect.push_back(el_ptn_mem[idx])

    for idx in range(0, pulses):
        pulse_mod_vect.push_back(cpp_complex[float_t](
            np.real(tx.pulse_mod[tx_idx, idx]), np.imag(tx.pulse_mod[tx_idx, idx])))

    mod_enabled = tx.waveform_mod[tx_idx]['enabled']
    if mod_enabled:
        for idx in range(0, len(tx.waveform_mod[tx_idx]['var'])):
            mod_var_vect.push_back(cpp_complex[float_t](
                np.real(tx.waveform_mod[tx_idx]['var'][idx]), np.imag(tx.waveform_mod[tx_idx]['var'][idx])))

        mod_t_vect.reserve(len(tx.waveform_mod[tx_idx]['t']))
        for idx in range(0, len(tx.waveform_mod[tx_idx]['t'])):
            mod_t_vect.push_back(<float_t> tx.waveform_mod[tx_idx]['t'][idx])

    return TxChannel[float_t](
        Vec3[float_t](
            <float_t> tx.locations[tx_idx, 0],
            <float_t> tx.locations[tx_idx, 1],
            <float_t> tx.locations[tx_idx, 2]
        ),
        Vec3[float_t](
            <float_t> tx.polarization[tx_idx, 0],
            <float_t> tx.polarization[tx_idx, 1],
            <float_t> tx.polarization[tx_idx, 2]
        ),
        az_ang_vect,
        az_ptn_vect,
        el_ang_vect,
        el_ptn_vect,
        <float_t> tx.antenna_gains[tx_idx],
        mod_t_vect,
        mod_var_vect,
        pulse_mod_vect,
        <float_t> tx.delay[tx_idx],
        <float_t> np.radians(tx.grid[tx_idx])
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef RxChannel[float_t] cp_RxChannel(rx,
                                     rx_idx):
    cdef vector[float_t] az_ang_vect, az_ptn_vect
    cdef float_t[:] az_ang_mem, az_ptn_mem
    cdef vector[float_t] el_ang_vect, el_ptn_vect
    cdef float_t[:] el_ang_mem, el_ptn_mem

    az_ang_mem = np.radians(rx.az_angles[rx_idx].astype(np.float32))
    az_ptn_mem = rx.az_patterns[rx_idx].astype(np.float32)
    az_ang_vect.reserve(len(rx.az_angles[rx_idx]))
    for idx in range(0, len(rx.az_angles[rx_idx])):
        az_ang_vect.push_back(az_ang_mem[idx])
    az_ptn_vect.reserve(len(rx.az_patterns[rx_idx]))
    for idx in range(0, len(rx.az_patterns[rx_idx])):
        az_ptn_vect.push_back(az_ptn_mem[idx])

    el_ang_mem = np.radians(
        np.flip(90-rx.el_angles[rx_idx].astype(np.float32)))
    el_ptn_mem = np.flip(rx.el_patterns[rx_idx].astype(np.float32))
    el_ang_vect.reserve(len(rx.el_angles[rx_idx]))
    for idx in range(0, len(rx.el_angles[rx_idx])):
        el_ang_vect.push_back(el_ang_mem[idx])
    el_ptn_vect.reserve(len(rx.el_patterns[rx_idx]))
    for idx in range(0, len(rx.el_patterns[rx_idx])):
        el_ptn_vect.push_back(el_ptn_mem[idx])

    return RxChannel[float_t](
        Vec3[float_t](
            <float_t> rx.locations[rx_idx, 0],
            <float_t> rx.locations[rx_idx, 1],
            <float_t> rx.locations[rx_idx, 2]
        ),
        Vec3[float_t](0, 0, 1),
        az_ang_vect,
        az_ptn_vect,
        el_ang_vect,
        el_ptn_vect,
        <float_t> rx.antenna_gains[rx_idx]
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Target[float_t] cp_Target(radar,
                               target,
                               shape):
    timestamp = radar.timestamp.astype(np.float32)
    cdef float_t[:, :] points_memview
    cdef int_t[:, :] cells_memview
    cdef float_t[:] origin

    cdef vector[Vec3[float_t]] c_loc_array
    cdef vector[Vec3[float_t]] c_speed_array
    cdef vector[Vec3[float_t]] c_rotation_array
    cdef vector[Vec3[float_t]] c_rotation_rate_array

    cdef float_t[:, :, :] tgt_loc_x, tgt_loc_y, tgt_loc_z
    cdef float_t[:, :, :] sptx_t, spty_t, sptz_t
    cdef float_t[:, :, :] rotx_t, roty_t, rotz_t
    cdef float_t[:, :, :] rotratx_t, rotraty_t, rotratz_t

    cdef cpp_complex[float_t] ep, mu

    cdef int_t ch_idx, ps_idx, sp_idx

    t_mesh = meshio.read(target['model'])
    points_memview = t_mesh.points.astype(np.float32)
    cells_memview = t_mesh.cells[0].data.astype(np.int32)

    origin = np.array(target.get('origin', (0, 0, 0)), dtype=np.float32)

    location = np.array(target.get('location', (0, 0, 0)),
                        dtype=object)
    speed = np.array(target.get('speed', (0, 0, 0)),
                        dtype=object)
    rotation = np.array(target.get('rotation', (0, 0, 0)),
                        dtype=object)
    rotation_rate = np.array(target.get(
        'rotation_rate', (0, 0, 0)), dtype=object)

    permittivity = target.get('permittivity', 'PEC')
    if permittivity == "PEC":
        ep = cpp_complex[float_t](-1, 0)
        mu = cpp_complex[float_t](1, 0)
    else:
        ep = cpp_complex[float_t](np.real(permittivity), np.imag(permittivity))
        mu = cpp_complex[float_t](1, 0)

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
            tgt_loc_x = location[0].astype(np.float32)
        else:
            tgt_loc_x = <float_t > location[0] + <float_t > speed[0]*timestamp

        if np.size(location[1]) > 1:
            tgt_loc_y = location[1].astype(np.float32)
        else:
            tgt_loc_y = <float_t > location[1] + <float_t > speed[1]*timestamp

        if np.size(location[2]) > 1:
            tgt_loc_z = location[2].astype(np.float32)
        else:
            tgt_loc_z = <float_t > location[2] + <float_t > speed[2]*timestamp

        if np.size(speed[0]) > 1:
            sptx_t = speed[0].astype(np.float32)
        else:
            sptx_t = np.full(shape, speed[0], dtype=np.float32)

        if np.size(speed[1]) > 1:
            spty_t = speed[1].astype(np.float32)
        else:
            spty_t = np.full(shape, speed[1], dtype=np.float32)

        if np.size(speed[2]) > 1:
            sptz_t = speed[2].astype(np.float32)
        else:
            sptz_t = np.full(shape, speed[2], dtype=np.float32)

        if np.size(rotation[0]) > 1:
            rotx_t = np.radians(rotation[0]).astype(np.float32)
        else:
            rotx_t = np.radians(
                rotation[0] + rotation_rate[0]*timestamp).astype(np.float32)

        if np.size(rotation[1]) > 1:
            roty_t = np.radians(rotation[1]).astype(np.float32)
        else:
            roty_t = np.radians(
                rotation[1] + rotation_rate[1]*timestamp).astype(np.float32)

        if np.size(rotation[2]) > 1:
            rotz_t = np.radians(rotation[2]).astype(np.float32)
        else:
            rotz_t = np.radians(
                rotation[2] + rotation_rate[2]*timestamp).astype(np.float32)

        if np.size(rotation_rate[0]) > 1:
            rotratx_t = np.radians(rotation_rate[0]).astype(np.float32)
        else:
            rotratx_t = np.full(shape, np.radians(
                rotation_rate[0]), dtype=np.float32)

        if np.size(rotation_rate[1]) > 1:
            rotraty_t = np.radians(rotation_rate[1]).astype(np.float32)
        else:
            rotraty_t = np.full(shape, np.radians(
                rotation_rate[1]), dtype=np.float32)

        if np.size(rotation_rate[2]) > 1:
            rotratz_t = np.radians(rotation_rate[2]).astype(np.float32)
        else:
            rotratz_t = np.full(shape, np.radians(
                rotation_rate[2]), dtype=np.float32)

        for ch_idx in range(0, radar.channel_size*radar.frames):
            for ps_idx in range(0, radar.transmitter.pulses):
                for sp_idx in range(0, radar.samples_per_pulse):
                    c_loc_array.push_back(
                        Vec3[float_t](
                            tgt_loc_x[ch_idx, ps_idx, sp_idx],
                            tgt_loc_y[ch_idx, ps_idx, sp_idx],
                            tgt_loc_z[ch_idx, ps_idx, sp_idx]
                        )
                    )
                    c_speed_array.push_back(
                        Vec3[float_t](
                            sptx_t[ch_idx, ps_idx, sp_idx],
                            spty_t[ch_idx, ps_idx, sp_idx],
                            sptz_t[ch_idx, ps_idx, sp_idx]
                        )
                    )
                    c_rotation_array.push_back(
                        Vec3[float_t](
                            rotx_t[ch_idx, ps_idx, sp_idx],
                            roty_t[ch_idx, ps_idx, sp_idx],
                            rotz_t[ch_idx, ps_idx, sp_idx]
                        )
                    )
                    c_rotation_rate_array.push_back(
                        Vec3[float_t](
                            rotratx_t[ch_idx, ps_idx, sp_idx],
                            rotraty_t[ch_idx, ps_idx, sp_idx],
                            rotratz_t[ch_idx, ps_idx, sp_idx]
                        )
                    )
    else:
        c_loc_array.push_back(
            Vec3[float_t](
                <float_t>location[0],
                <float_t>location[1],
                <float_t>location[2]
            )
        )
        c_speed_array.push_back(
            Vec3[float_t](
                <float_t>speed[0],
                <float_t>speed[1],
                <float_t>speed[2]
            )
        )
        c_rotation_array.push_back(
            Vec3[float_t](
                <float_t>np.radians(rotation[0]),
                <float_t>np.radians(rotation[1]),
                <float_t>np.radians(rotation[2])
            )
        )
        c_rotation_rate_array.push_back(
            Vec3[float_t](
                <float_t>np.radians(rotation_rate[0]),
                <float_t>np.radians(rotation_rate[1]),
                <float_t>np.radians(rotation_rate[2])
            )
        )

    return Target[float_t](&points_memview[0, 0],
                           &cells_memview[0, 0],
                           <int_t> cells_memview.shape[0],
                           Vec3[float_t]( &origin[0]),
                           c_loc_array,
                           c_speed_array,
                           c_rotation_array,
                           c_rotation_rate_array,
                           ep,
                           mu,
                           <bool> target.get('is_ground', False))

