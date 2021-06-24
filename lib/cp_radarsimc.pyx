# distutils: language = c++
# cython: language_level=3


import meshio
from radarsimpy.includes.radarsimc cimport Target
from radarsimpy.includes.radarsimc cimport Point
from radarsimpy.includes.radarsimc cimport RxChannel
from radarsimpy.includes.radarsimc cimport TxChannel
from radarsimpy.includes.radarsimc cimport Transmitter
from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t
from radarsimpy.includes.type_def cimport vector, complex_t
from libcpp.complex cimport complex as cpp_complex
from libcpp cimport bool
from libc.math cimport M_PI
from libc.math cimport pow, fmax
from libc.math cimport sin, cos, sqrt, atan, atan2, acos
import numpy as np

cimport cython
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Point[float_t] cp_Point(location, speed, rcs, phase, shape):
    cdef vector[Vec3[float_t]] loc_vect
    cdef vector[float_t] rcs_vect
    cdef vector[float_t] phs_vect

    if np.size(location[0]) > 1 or \
            np.size(location[1]) > 1 or \
            np.size(location[2]) > 1 or \
            np.size(rcs) > 1 or \
            np.size(phase) > 1:

        if np.size(location[0]) > 1:
            tgx_t = location[0]
        else:
            tgx_t = np.full(shape, location[0])

        if np.size(location[1]) > 1:
            tgy_t = location[1]
        else:
            tgy_t = np.full(shape, location[1])

        if np.size(location[2]) > 1:
            tgz_t = location[2]
        else:
            tgz_t = np.full(shape, location[2])

        if np.size(rcs) > 1:
            rcs_t = rcs
        else:
            rcs_t = np.full(shape, rcs)

        if np.size(phase) > 1:
            phs_t = phase
        else:
            phs_t = np.full(shape, phase)

        for ch_idx in range(0, shape[0]):
            for ps_idx in range(0, shape[1]):
                for sp_idx in range(0, shape[2]):
                    loc_vect.push_back(Vec3[float_t](
                        < float_t > tgx_t[ch_idx, ps_idx, sp_idx],
                        < float_t > tgy_t[ch_idx, ps_idx, sp_idx],
                        < float_t > tgz_t[ch_idx, ps_idx, sp_idx]
                    ))
                    rcs_vect.push_back(< float_t > rcs_t[ch_idx, ps_idx, sp_idx])
                    phs_vect.push_back(< float_t > (phs_t[ch_idx, ps_idx, sp_idx]/180*np.pi))
    else:
        loc_vect.push_back(Vec3[float_t](
            < float_t > location[0],
            < float_t > location[1],
            < float_t > location[2]
        ))
        rcs_vect.push_back(< float_t > rcs)
        phs_vect.push_back(< float_t > (phase/180*np.pi))
    return Point[float_t](
        loc_vect,
        Vec3[float_t](
            < float_t > speed[0],
            < float_t > speed[1],
            < float_t > speed[2]
        ),
        rcs_vect,
        phs_vect
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Transmitter[float_t] cp_Transmitter(radar, density):
    cdef int_t frames = radar.frames
    cdef int_t channles = radar.channel_size
    cdef int_t pulses = radar.transmitter.pulses
    cdef int_t samples = radar.samples_per_pulse

    cdef vector[float_t] t_frame_vect
    cdef vector[float_t] f_vect
    cdef vector[float_t] t_vect
    cdef vector[float_t] f_offset_vect
    cdef vector[float_t] t_pstart_vect
    cdef vector[cpp_complex[float_t]] pn_vect = vector[cpp_complex[float_t]]()

    if frames > 1:
        t_frame_mem = radar.t_offset.astype(np.float64)
        t_frame_vect.reserve(frames)
        for idx in range(0, frames):
            t_frame_vect.push_back(t_frame_mem[idx])
    else:
        t_frame_vect.push_back(< float_t > (radar.t_offset))

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

    t_pstart_mem = radar.transmitter.chirp_start_time.astype(np.float64)
    t_pstart_vect.reserve(len(radar.transmitter.chirp_start_time))
    for idx in range(0, len(radar.transmitter.chirp_start_time)):
        t_pstart_vect.push_back(t_pstart_mem[idx])

    if radar.phase_noise is not None:
        pn_vect.reserve(frames*channles*pulses*samples)
        for idx0 in range(0, frames*channles):
            for idx1 in range(0, pulses):
                for idx2 in range(0, samples):
                    pn_vect.push_back(cpp_complex[float_t](
                        np.real(radar.phase_noise[idx0, idx1, idx2]),
                        np.imag(radar.phase_noise[idx0, idx1, idx2])
                    ))

    return Transmitter[float_t](
        < float_t > radar.transmitter.fc_0,
        f_vect,
        f_offset_vect,
        t_vect,
        < float_t > radar.transmitter.tx_power,
        t_pstart_vect,
        t_frame_vect,
        frames,
        pulses,
        < float_t > density,
        pn_vect
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef TxChannel[float_t] cp_TxChannel(tx, tx_idx):
    cdef int_t pulses = tx.pulses

    cdef vector[float_t] az_ang_vect, az_ptn_vect
    cdef vector[float_t] el_ang_vect, el_ptn_vect

    cdef vector[cpp_complex[float_t]] pulse_mod_vect

    cdef bool mod_enabled
    cdef vector[cpp_complex[float_t]] mod_var_vect
    cdef vector[float_t] mod_t_vect

    az_ang_vect.reserve(len(tx.az_angles[tx_idx]))
    for idx in range(0, len(tx.az_angles[tx_idx])):
        az_ang_vect.push_back(< float_t > (tx.az_angles[tx_idx][idx]/180*np.pi))
    az_ptn_vect.reserve(len(tx.az_patterns[tx_idx]))
    for idx in range(0, len(tx.az_patterns[tx_idx])):
        az_ptn_vect.push_back(< float_t > (tx.az_patterns[tx_idx][idx]))

    el_ang_mem = np.flip(90-tx.el_angles[tx_idx].astype(np.float64))/180*np.pi
    el_ptn_mem = np.flip(tx.el_patterns[tx_idx].astype(np.float64))
    el_ang_vect.reserve(len(tx.el_angles[tx_idx]))
    for idx in range(0, len(tx.el_angles[tx_idx])):
        el_ang_vect.push_back(el_ang_mem[idx])
    el_ptn_vect.reserve(len(tx.el_patterns[tx_idx]))
    for idx in range(0, len(tx.el_patterns[tx_idx])):
        el_ptn_vect.push_back(el_ptn_mem[idx])

    for idx in range(0, pulses):
        pulse_mod_vect.push_back(cpp_complex[float_t](
            np.real(tx.pulse_mod[tx_idx, idx]), np.imag(tx.pulse_mod[tx_idx, idx])))

    mod_enabled = tx.mod[tx_idx]['enabled']
    if mod_enabled:
        for idx in range(0, len(tx.mod[tx_idx]['var'])):
            mod_var_vect.push_back(cpp_complex[float_t](
                np.real(tx.mod[tx_idx]['var'][idx]), np.imag(tx.mod[tx_idx]['var'][idx])))

        mod_t_vect.reserve(len(tx.mod[tx_idx]['t']))
        for idx in range(0, len(tx.mod[tx_idx]['t'])):
            mod_t_vect.push_back(< float_t > tx.mod[tx_idx]['t'][idx])

    return TxChannel[float_t](
        Vec3[float_t](
            < float_t > tx.locations[tx_idx, 0],
            < float_t > tx.locations[tx_idx, 1],
            < float_t > tx.locations[tx_idx, 2]
        ),
        Vec3[float_t](
            < float_t > tx.polarization[tx_idx, 0],
            < float_t > tx.polarization[tx_idx, 1],
            < float_t > tx.polarization[tx_idx, 2]
        ),
        pulse_mod_vect,
        mod_enabled,
        mod_t_vect,
        mod_var_vect,
        az_ang_vect,
        az_ptn_vect,
        el_ang_vect,
        el_ptn_vect,
        < float_t > tx.antenna_gains[tx_idx],
        < float_t > tx.delay[tx_idx],
        < float_t > (tx.grid[tx_idx]/180*np.pi)
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef RxChannel[float_t] cp_RxChannel(rx, rx_idx):
    cdef vector[float_t] az_ang_vect, az_ptn_vect
    cdef float_t[:] az_ang_mem, az_ptn_mem
    cdef vector[float_t] el_ang_vect, el_ptn_vect
    cdef float_t[:] el_ang_mem, el_ptn_mem

    az_ang_mem = rx.az_angles[rx_idx].astype(np.float64)/180*np.pi
    az_ptn_mem = rx.az_patterns[rx_idx].astype(np.float64)
    az_ang_vect.reserve(len(rx.az_angles[rx_idx]))
    for idx in range(0, len(rx.az_angles[rx_idx])):
        az_ang_vect.push_back(az_ang_mem[idx])
    az_ptn_vect.reserve(len(rx.az_patterns[rx_idx]))
    for idx in range(0, len(rx.az_patterns[rx_idx])):
        az_ptn_vect.push_back(az_ptn_mem[idx])

    el_ang_mem = np.flip(90-rx.el_angles[rx_idx].astype(np.float64))/180*np.pi
    el_ptn_mem = np.flip(rx.el_patterns[rx_idx].astype(np.float64))
    el_ang_vect.reserve(len(rx.el_angles[rx_idx]))
    for idx in range(0, len(rx.el_angles[rx_idx])):
        el_ang_vect.push_back(el_ang_mem[idx])
    el_ptn_vect.reserve(len(rx.el_patterns[rx_idx]))
    for idx in range(0, len(rx.el_patterns[rx_idx])):
        el_ptn_vect.push_back(el_ptn_mem[idx])

    return RxChannel[float_t](
        Vec3[float_t](
            < float_t > rx.locations[rx_idx, 0],
            < float_t > rx.locations[rx_idx, 1],
            < float_t > rx.locations[rx_idx, 2]
        ),
        Vec3[float_t](0, 0, 1),
        az_ang_vect,
        az_ptn_vect,
        el_ang_vect,
        el_ptn_vect,
        < float_t > rx.antenna_gains[rx_idx]
    )


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Target[float_t] cp_Target(radar, target, shape):
    cdef float_t[:, :] points_memview
    cdef uint64_t[:, :] cells_memview
    cdef float_t[:] origin

    cdef vector[Vec3[float_t]] c_loc_array
    cdef vector[Vec3[float_t]] c_speed_array
    cdef vector[Vec3[float_t]] c_rotation_array
    cdef vector[Vec3[float_t]] c_rotation_rate_array

    cdef float_t[:, :, :] tgx_t, tgy_t, tgz_t
    cdef float_t[:, :, :] sptx_t, spty_t, sptz_t
    cdef float_t[:, :, :] rotx_t, roty_t, rotz_t
    cdef float_t[:, :, :] rotratx_t, rotraty_t, rotratz_t

    cdef cpp_complex[float_t] ep, mu

    cdef int_t ch_idx, ps_idx, sp_idx

    t_mesh = meshio.read(target['model'])
    points_memview = t_mesh.points.astype(np.float64)
    cells_memview = t_mesh.cells[0].data.astype(np.uint64)

    origin = np.array(target.get('origin', (0, 0, 0)), dtype=np.float64)

    location = target.get('location', (0, 0, 0))
    speed = target.get('speed', (0, 0, 0))
    rotation = np.array(target.get('rotation', (0, 0, 0)),
                        dtype=object)/180*np.pi
    rotation_rate = np.array(target.get(
        'rotation_rate', (0, 0, 0)), dtype=object)/180*np.pi

    permittivity = target.get('permittivity', 'PEC')
    if permittivity=="PEC":
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
            tgx_t = location[0].astype(np.float64)
        else:
            tgx_t = np.full(shape, location[0], dtype=np.float64)

        if np.size(location[1]) > 1:
            tgy_t = location[1].astype(np.float64)
        else:
            tgy_t = np.full(shape, location[1], dtype=np.float64)

        if np.size(location[2]) > 1:
            tgz_t = location[2].astype(np.float64)
        else:
            tgz_t = np.full(shape, location[2], dtype=np.float64)

        if np.size(speed[0]) > 1:
            sptx_t = speed[0].astype(np.float64)
        else:
            sptx_t = np.full(shape, speed[0], dtype=np.float64)

        if np.size(speed[1]) > 1:
            spty_t = speed[1].astype(np.float64)
        else:
            spty_t = np.full(shape, speed[1], dtype=np.float64)

        if np.size(speed[2]) > 1:
            sptz_t = speed[2].astype(np.float64)
        else:
            sptz_t = np.full(shape, speed[2], dtype=np.float64)

        if np.size(rotation[0]) > 1:
            rotx_t = rotation[0].astype(np.float64)
        else:
            rotx_t = np.full(shape, rotation[0], dtype=np.float64)

        if np.size(rotation[1]) > 1:
            roty_t = rotation[1].astype(np.float64)
        else:
            roty_t = np.full(shape, rotation[1], dtype=np.float64)

        if np.size(rotation[2]) > 1:
            rotz_t = rotation[2].astype(np.float64)
        else:
            rotz_t = np.full(shape, rotation[2], dtype=np.float64)

        if np.size(rotation_rate[0]) > 1:
            rotratx_t = rotation_rate[0].astype(np.float64)
        else:
            rotratx_t = np.full(shape, rotation_rate[0], dtype=np.float64)

        if np.size(rotation_rate[1]) > 1:
            rotraty_t = rotation_rate[1].astype(np.float64)
        else:
            rotraty_t = np.full(shape, rotation_rate[1], dtype=np.float64)

        if np.size(rotation_rate[2]) > 1:
            rotratz_t = rotation_rate[2].astype(np.float64)
        else:
            rotratz_t = np.full(shape, rotation_rate[2], dtype=np.float64)

        for ch_idx in range(0, radar.channel_size*radar.frames):
            for ps_idx in range(0, radar.transmitter.pulses):
                for sp_idx in range(0, radar.samples_per_pulse):
                    c_loc_array.push_back(
                        Vec3[float_t](
                            tgx_t[ch_idx, ps_idx, sp_idx],
                            tgy_t[ch_idx, ps_idx, sp_idx],
                            tgz_t[ch_idx, ps_idx, sp_idx]
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
                < float_t > location[0],
                < float_t > location[1],
                < float_t > location[2]
            )
        )
        c_speed_array.push_back(
            Vec3[float_t](
                < float_t > speed[0],
                < float_t > speed[1],
                < float_t > speed[2]
            )
        )
        c_rotation_array.push_back(
            Vec3[float_t](
                < float_t > rotation[0],
                < float_t > rotation[1],
                < float_t > rotation[2]
            )
        )
        c_rotation_rate_array.push_back(
            Vec3[float_t](
                < float_t > rotation_rate[0],
                < float_t > rotation_rate[1],
                < float_t > rotation_rate[2]
            )
        )

    return Target[float_t](
        & points_memview[0, 0],
        & cells_memview[0, 0],
        < int_t > cells_memview.shape[0],
        Vec3[float_t](& origin[0]),
        c_loc_array,
        c_speed_array,
        c_rotation_array,
        c_rotation_rate_array,
        ep,
        mu,
        < bool > target.get('is_ground', False)
    )
