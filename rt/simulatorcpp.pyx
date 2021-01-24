# distutils: language = c++
# cython: language_level=3

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - 2020  Zhengyu Peng
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


cimport cython

from libc.math cimport sin, cos, sqrt, atan, atan2, acos
from libc.math cimport pow, fmax
from libc.math cimport M_PI
from libcpp cimport bool
from libcpp.complex cimport complex as cpp_complex

from radarsimpy.includes.radarsimc cimport Point
from radarsimpy.includes.radarsimc cimport TxChannel, Transmitter
from radarsimpy.includes.radarsimc cimport RxChannel, Receiver
from radarsimpy.includes.radarsimc cimport Simulator
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t
from radarsimpy.includes.type_def cimport complex_t
from radarsimpy.includes.type_def cimport vector
from radarsimpy.includes.zpvector cimport Vec3

import numpy as np
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run_simulator(radar, targets, noise=True):
    """
    Alias: ``radarsimpy.simulatorcpp()``
    
    Radar simulator with C++ engine

    :param Radar radar:
        Radar model
    :param list[dict] targets:
        Ideal point target list

        [{

        - **location** (*numpy.1darray*) --
            Location of the target (m), [x, y, z]
        - **rcs** (*float*) --
            Target RCS (dBsm)
        - **speed** (*numpy.1darray*) --
            Speed of the target (m/s), [vx, vy, vz]. ``default
            [0, 0, 0]``
        - **phase** (*float*) --
            Target phase (deg). ``default 0``

        }]

        *Note*: Target's parameters can be specified with
        ``Radar.timestamp`` to customize the time varying property.
        Example: ``location=(1e-3*np.sin(2*np.pi*1*radar.timestamp), 0, 0)``
    :param bool noise:
        Flag to enable noise calculation. ``default True``

    :return:
        {

        - **baseband** (*numpy.3darray*) --
            Time domain complex (I/Q) baseband data.
            ``[channes/frames, pulses, samples]``

            *Channel/frame order in baseband*

            *[0]* ``Frame[0] -- Tx[0] -- Rx[0]``

            *[1]* ``Frame[0] -- Tx[0] -- Rx[1]``

            ...

            *[N]* ``Frame[0] -- Tx[1] -- Rx[0]``

            *[N+1]* ``Frame[0] -- Tx[1] -- Rx[1]``

            ...

            *[M]* ``Frame[1] -- Tx[0] -- Rx[0]``

            *[M+1]* ``Frame[1] -- Tx[0] -- Rx[1]``

        - **timestamp** (*numpy.3darray*) --
            Refer to Radar.timestamp

        }
    :rtype: dict
    """
    cdef Simulator[float_t] sim

    cdef vector[Point[float_t]] points
    cdef Transmitter[float_t] tx
    cdef Receiver[float_t] rx

    cdef int_t frames = radar.frames
    cdef int_t channles = radar.channel_size
    cdef int_t pulses = radar.transmitter.pulses
    cdef int_t samples = radar.samples_per_pulse

    cdef int_t ch_stride = pulses * samples
    cdef int_t pulse_stride = samples
    cdef int_t idx_stride

    """
    Targets
    """
    cdef int_t target_count = len(targets)
    cdef vector[Vec3[float_t]] c_loc
    cdef vector[float_t] c_rcs
    cdef vector[float_t] c_phs

    timestamp = radar.timestamp
    
    for idx in range(0, target_count):
        c_loc.clear()
        c_rcs.clear()
        c_phs.clear()

        location = targets[idx]['location']
        speed = targets[idx].get('speed', (0, 0, 0))
        rcs = targets[idx]['rcs']
        phase = targets[idx].get('phase', 0)

        if np.size(location[0]) > 1 or np.size(location[1])  > 1 or np.size(location[2]) > 1 or np.size(rcs) > 1 or np.size(phase) > 1:

            if np.size(location[0]) > 1:
                tgx_t = location[0]
            else:
                tgx_t = np.full_like(timestamp, location[0])

            if np.size(location[1]) > 1:
                tgy_t = location[1]
            else:
                tgy_t = np.full_like(timestamp, location[1])
            
            if np.size(location[2]) > 1:
                tgz_t = location[2]
            else:
                tgz_t = np.full_like(timestamp, location[2])

            if np.size(rcs) > 1:
                rcs_t = rcs
            else:
                rcs_t = np.full_like(timestamp, rcs)
            
            if np.size(phase) > 1:
                phs_t = phase
            else:
                phs_t = np.full_like(timestamp, phase)

            for ch_idx in range(0, channles*frames):
                for ps_idx in range(0, pulses):
                    for sp_idx in range(0, samples):
                        c_loc.push_back(Vec3[float_t](
                            <float_t> tgx_t[ch_idx, ps_idx, sp_idx],
                            <float_t> tgy_t[ch_idx, ps_idx, sp_idx],
                            <float_t> tgz_t[ch_idx, ps_idx, sp_idx]
                        ))
                        c_rcs.push_back(<float_t> rcs_t[ch_idx, ps_idx, sp_idx])
                        c_phs.push_back(<float_t> (phs_t[ch_idx, ps_idx, sp_idx]/180*np.pi))
        else:
            c_loc.push_back(Vec3[float_t](
                <float_t> location[0],
                <float_t> location[1],
                <float_t> location[2]
            ))
            c_rcs.push_back(<float_t> rcs)
            c_phs.push_back(<float_t> (phase/180*np.pi))

        points.push_back(
            Point[float_t](
                c_loc,
                Vec3[float_t](
                    <float_t> speed[0],
                    <float_t> speed[1],
                    <float_t> speed[2]
                ),
                c_rcs,
                c_phs
            )
        )

    """
    Transmitter
    """
    cdef vector[float_t] t_frame_vect
    cdef float_t[:] t_frame_mem

    cdef vector[float_t] f_vect
    cdef float_t[:] f_mem

    cdef vector[float_t] t_vect
    cdef float_t[:] t_mem

    cdef vector[float_t] f_offset_vect
    cdef float_t[:] f_offset_mem

    cdef vector[float_t] t_pstart_vect
    cdef float_t[:] t_pstart_mem

    cdef vector[cpp_complex[float_t]] pn_vect
    cdef complex_t[:,:,:] pn_mem

    if frames > 1:
        t_frame_mem=radar.t_offset.astype(np.float64)
        t_frame_vect.reserve(frames)
        t_frame_vect.assign(&t_frame_mem[0], &t_frame_mem[0]+frames)
    else:
        t_frame_vect.push_back(<float_t> (radar.t_offset))

    f_mem = radar.f.astype(np.float64)
    f_vect.reserve(len(radar.f))
    f_vect.assign(
        &f_mem[0],
        &f_mem[0]+len(radar.f))

    t_mem = radar.t.astype(np.float64)
    t_vect.reserve(len(radar.t))
    t_vect.assign(
        &t_mem[0],
        &t_mem[0]+len(radar.t))

    f_offset_mem = radar.transmitter.f_offset.astype(np.float64)
    f_offset_vect.reserve(len(radar.transmitter.f_offset))
    f_offset_vect.assign(
        &f_offset_mem[0],
        &f_offset_mem[0]+len(radar.transmitter.f_offset)
        )
    
    t_pstart_mem = radar.transmitter.chirp_start_time.astype(np.float64)
    t_pstart_vect.reserve(len(radar.transmitter.chirp_start_time))
    t_pstart_vect.assign(
        &t_pstart_mem[0],
        &t_pstart_mem[0]+len(radar.transmitter.chirp_start_time)
        )

    if radar.phase_noise is None:
        tx = Transmitter[float_t](
            <float_t> radar.transmitter.fc_0,
            f_vect,
            f_offset_vect,
            t_vect,
            <float_t> radar.transmitter.tx_power,
            t_pstart_vect,
            t_frame_vect,
            frames,
            pulses,
            0.0
        )
    else:
        pn_mem = radar.phase_noise.astype(np.complex128)
        pn_vect.reserve(frames*channles*pulses*samples)
        pn_vect.assign(
            &pn_mem[0,0,0],
            &pn_mem[0,0,0]+frames*channles*pulses*samples
            )

        tx = Transmitter[float_t](
            <float_t> radar.transmitter.fc_0,
            f_vect,
            f_offset_vect,
            t_vect,
            <float_t> radar.transmitter.tx_power,
            t_pstart_vect,
            t_frame_vect,
            frames,
            pulses,
            0.0,
            pn_vect
        )

    """
    Transmitter Channels
    """
    cdef int_t ptn_length
    cdef vector[float_t] az_ang_vect, az_ptn_vect
    cdef float_t[:] az_ang_mem, az_ptn_mem
    cdef vector[float_t] el_ang_vect, el_ptn_vect
    cdef float_t[:] el_ang_mem, el_ptn_mem

    cdef vector[cpp_complex[float_t]] pulse_mod_vect
    cdef complex_t[:] pulse_mod_mem

    cdef bool mod_enabled
    cdef vector[cpp_complex[float_t]] mod_var_vect
    cdef complex_t[:] mod_var_mem
    cdef vector[float_t] mod_t_vect
    cdef float_t[:] mod_t_mem

    for tx_idx in range(0, radar.transmitter.channel_size):
        az_ang_vect.clear()
        az_ptn_vect.clear()
        el_ang_vect.clear()
        el_ptn_vect.clear()
        pulse_mod_vect.clear()
        mod_var_vect.clear()
        mod_t_vect.clear()

        az_ang_mem = radar.transmitter.az_angles[tx_idx].astype(np.float64)/180*np.pi
        az_ptn_mem = radar.transmitter.az_patterns[tx_idx].astype(np.float64)
        az_ang_vect.reserve(len(radar.transmitter.az_angles[tx_idx]))
        az_ang_vect.assign(
            &az_ang_mem[0],
            &az_ang_mem[0]+len(radar.transmitter.az_angles[tx_idx])
        )
        az_ptn_vect.reserve(len(radar.transmitter.az_patterns[tx_idx]))
        az_ptn_vect.assign(
            &az_ptn_mem[0],
            &az_ptn_mem[0]+len(radar.transmitter.az_patterns[tx_idx])
        )

        el_ang_mem = np.flip(90-radar.transmitter.el_angles[tx_idx].astype(np.float64))/180*np.pi
        el_ptn_mem = np.flip(radar.transmitter.el_patterns[tx_idx].astype(np.float64))
        el_ang_vect.reserve(len(radar.transmitter.el_angles[tx_idx]))
        el_ang_vect.assign(
            &el_ang_mem[0],
            &el_ang_mem[0]+len(radar.transmitter.el_angles[tx_idx])
        )
        el_ptn_vect.reserve(len(radar.transmitter.el_patterns[tx_idx]))
        el_ptn_vect.assign(
            &el_ptn_mem[0],
            &el_ptn_mem[0]+len(radar.transmitter.el_patterns[tx_idx])
        )

        pulse_mod_mem = radar.transmitter.pulse_mod[tx_idx, :].astype(np.complex128)
        pulse_mod_vect.reserve(pulses)
        pulse_mod_vect.assign(
            &pulse_mod_mem[0],
            &pulse_mod_mem[0]+pulses
        )

        mod_enabled = radar.transmitter.mod[tx_idx]['enabled']
        if mod_enabled:
            mod_var_mem = radar.transmitter.mod[tx_idx]['var'].astype(np.complex128)
            mod_var_vect.reserve(len(radar.transmitter.mod[tx_idx]['var']))
            mod_var_vect.assign(
                &mod_var_mem[0],
                &mod_var_mem[0]+len(radar.transmitter.mod[tx_idx]['var'])
            )

            mod_t_mem = radar.transmitter.mod[tx_idx]['t'].astype(np.float64)
            mod_t_vect.reserve(len(radar.transmitter.mod[tx_idx]['t']))
            mod_t_vect.assign(
                &mod_t_mem[0],
                &mod_t_mem[0]+len(radar.transmitter.mod[tx_idx]['t']))
        
        tx.AddChannel(
            TxChannel[float_t](
                Vec3[float_t](
                    <float_t> radar.transmitter.locations[tx_idx, 0],
                    <float_t> radar.transmitter.locations[tx_idx, 1],
                    <float_t> radar.transmitter.locations[tx_idx, 2]
                ),
                Vec3[float_t](
                    <float_t> radar.transmitter.polarization[tx_idx, 0],
                    <float_t> radar.transmitter.polarization[tx_idx, 1],
                    <float_t> radar.transmitter.polarization[tx_idx, 2]
                ),
                pulse_mod_vect,
                mod_enabled,
                mod_t_vect,
                mod_var_vect,
                az_ang_vect,
                az_ptn_vect,
                el_ang_vect,
                el_ptn_vect,
                <float_t> radar.transmitter.antenna_gains[tx_idx],
                <float_t> radar.transmitter.delay[tx_idx],
                0.0
            )
        )

    """
    Receiver
    """
    rx = Receiver[float_t](
        <float_t> radar.receiver.fs,
        <float_t> radar.receiver.rf_gain,
        <float_t> radar.receiver.load_resistor,
        <float_t> radar.receiver.baseband_gain,
        samples
    )
    
    for rx_idx in range(0, radar.receiver.channel_size):
        az_ang_vect.clear()
        az_ptn_vect.clear()
        el_ang_vect.clear()
        el_ptn_vect.clear()

        az_ang_mem = radar.receiver.az_angles[rx_idx].astype(np.float64)/180*np.pi
        az_ptn_mem = radar.receiver.az_patterns[rx_idx].astype(np.float64)
        az_ang_vect.reserve(len(radar.receiver.az_angles[rx_idx]))
        az_ang_vect.assign(
            &az_ang_mem[0],
            &az_ang_mem[0]+len(radar.receiver.az_angles[rx_idx])
        )
        az_ptn_vect.reserve(len(radar.receiver.az_patterns[rx_idx]))
        az_ptn_vect.assign(
            &az_ptn_mem[0],
            &az_ptn_mem[0]+len(radar.receiver.az_patterns[rx_idx])
        )

        el_ang_mem = np.flip(90-radar.receiver.el_angles[rx_idx].astype(np.float64))/180*np.pi
        el_ptn_mem = np.flip(radar.receiver.el_patterns[rx_idx].astype(np.float64))
        el_ang_vect.reserve(len(radar.receiver.el_angles[rx_idx]))
        el_ang_vect.assign(
            &el_ang_mem[0],
            &el_ang_mem[0]+len(radar.receiver.el_angles[rx_idx])
        )
        el_ptn_vect.reserve(len(radar.receiver.el_patterns[rx_idx]))
        el_ptn_vect.assign(
            &el_ptn_mem[0],
            &el_ptn_mem[0]+len(radar.receiver.el_patterns[rx_idx])
        )

        rx.AddChannel(
            RxChannel[float_t](
                Vec3[float_t](
                    <float_t> radar.receiver.locations[rx_idx, 0],
                    <float_t> radar.receiver.locations[rx_idx, 1],
                    <float_t> radar.receiver.locations[rx_idx, 2]
                ),
                Vec3[float_t](0,0,1),
                az_ang_vect,
                az_ptn_vect,
                el_ang_vect,
                el_ptn_vect,
                <float_t> radar.receiver.antenna_gains[rx_idx]
            )
        )

    cdef vector[cpp_complex[float_t]] *bb_vect = new vector[cpp_complex[float_t]](
        frames*channles*pulses*samples,
        cpp_complex[float_t](0.0,0.0))

    sim.Run(tx, rx, points, bb_vect[0])

    baseband = np.zeros((frames*channles, pulses, samples), dtype=complex)

    for ch_idx in range(0, frames*channles):
        for p_idx in range(0, pulses):
            for s_idx in range(0, samples):
                idx_stride = ch_idx * ch_stride + p_idx * pulse_stride + s_idx
                baseband[ch_idx, p_idx, s_idx] = bb_vect[0][idx_stride].real()+1j*bb_vect[0][idx_stride].imag()

    if noise:
        baseband = baseband+\
            radar.noise*(np.random.randn(
                    frames*channles,
                    pulses,
                    samples,
                ) + 1j * np.random.randn(
                    frames*channles,
                    pulses,
                    samples,
                ))

    del bb_vect
    
    return {'baseband':baseband,
            'timestamp':radar.timestamp}