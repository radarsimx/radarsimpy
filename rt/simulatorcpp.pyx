# distutils: language = c++
# cython: language_level=3
"""
    This script contains classes that define all the parameters for
    a radar system

    This script requires that `numpy` be installed within the Python
    environment you are running this script in.

    This file can be imported as a module and contains the following
    class:

    * Transmitter - A class defines parameters of a radar transmitter
    * Receiver - A class defines parameters of a radar receiver
    * Radar - A class defines basic parameters of a radar system

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - 2020  Zhengyu Peng
    E-mail: zpeng.me@gmail.com
    Website: https://zpeng.me

    `                      `
    -:.                  -#:
    -//:.              -###:
    -////:.          -#####:
    -/:.://:.      -###++##:
    ..   `://:-  -###+. :##:
           `:/+####+.   :##:
    .::::::::/+###.     :##:
    .////-----+##:    `:###:
     `-//:.   :##:  `:###/.
       `-//:. :##:`:###/.
         `-//:+######/.
           `-/+####/.
             `+##+.
              :##:
              :##:
              :##:
              :##:
              :##:
               .+:

"""

cimport cython

from libc.math cimport sin, cos, sqrt, atan, atan2, acos, pow, fmax, M_PI
from libcpp cimport bool

from radarsimpy.includes.radarsimc cimport Point, TxChannel, Transmitter, RxChannel, Receiver, Simulator
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t, vector
from radarsimpy.includes.zpvector cimport Vec3, Vec2
from libcpp cimport complex

import numpy as np
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef run_simulator(radar, targets, noise=True):
    # cdef Radarsimc[float_t] *rec_ptr
    # rec_ptr = new Radarsimc[float_t]()
    cdef Simulator[float_t] sim

    cdef vector[Point[float_t]] points_
    cdef Transmitter[float_t] tx
    cdef Receiver[float_t] rx

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

            for ch_idx in range(0, radar.channel_size*radar.frames):
                for ps_idx in range(0, radar.transmitter.pulses):
                    for sp_idx in range(0, radar.samples_per_pulse):
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

        points_.push_back(
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
    cdef vector[float_t] frame_time

    if radar.frames > 1:
        for t_idx in range(0, radar.frames):
            frame_time.push_back(<float_t> (radar.t_offset[t_idx]))
    else:
        frame_time.push_back(<float_t> (radar.t_offset))
    
    cdef vector[float_t] fc_vector
    for fc_idx in range(0, len(radar.transmitter.fc)):
        fc_vector.push_back(<float_t> radar.transmitter.fc[fc_idx])
    
    cdef vector[float_t] chirp_start_time
    for ct_idx in range(0, len(radar.transmitter.chirp_start_time)):
        chirp_start_time.push_back(<float_t> radar.transmitter.chirp_start_time[ct_idx])

    tx = Transmitter[float_t](
        fc_vector,
        <float_t> radar.transmitter.slope,
        <float_t> radar.transmitter.tx_power,
        chirp_start_time,
        frame_time,
        <int> radar.frames,
        <int> radar.transmitter.pulses,
        <float_t> 0
    )

    cdef int ptn_length
    cdef vector[float_t] az_ang
    cdef vector[float_t] az
    cdef vector[float_t] el_ang
    cdef vector[float_t] el

    cdef vector[float_t] mod_amp
    cdef vector[float_t] mod_phs
    for tx_idx in range(0, radar.transmitter.channel_size):
        az_ang.clear()
        az.clear()
        el_ang.clear()
        el.clear()

        mod_amp.clear()
        mod_phs.clear()

        ptn_length = len(radar.transmitter.az_angles[tx_idx])
        for ang_idx in range(0, ptn_length):
            az_ang.push_back(<float_t>(radar.transmitter.az_angles[tx_idx][ang_idx]/180*np.pi))
            az.push_back(<float_t>radar.transmitter.az_patterns[tx_idx][ang_idx])

        ptn_length = len(radar.transmitter.el_angles[tx_idx])

        el_angles = np.flip(90-radar.transmitter.el_angles[tx_idx])/180*np.pi
        el_pattern = np.flip(radar.transmitter.el_patterns[tx_idx])
        for ang_idx in range(0, ptn_length):
            el_ang.push_back(<float_t>el_angles[ang_idx])
            el.push_back(<float_t>el_pattern[ang_idx])

        for code_idx in range(0, len(radar.transmitter.phase_code[tx_idx])):
            mod_amp.push_back(<float_t> (np.abs(radar.transmitter.phase_code[tx_idx][code_idx])))
            mod_phs.push_back(<float_t> (np.angle(radar.transmitter.phase_code[tx_idx][code_idx])))
        
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
                mod_amp,
                mod_phs,
                <float_t> radar.transmitter.chip_length[tx_idx],
                az_ang,
                az,
                el_ang,
                el,
                <float_t> radar.transmitter.antenna_gains[tx_idx],
                <float_t> radar.transmitter.delay[tx_idx],
                <float_t> 0
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
        <int> radar.samples_per_pulse
    )
    
    for rx_idx in range(0, radar.receiver.channel_size):
        az_ang.clear()
        az.clear()
        el_ang.clear()
        el.clear()

        ptn_length = len(radar.receiver.az_angles[rx_idx])
        for ang_idx in range(0, ptn_length):
            az_ang.push_back(<float_t>(radar.receiver.az_angles[rx_idx][ang_idx]/180*np.pi))
            az.push_back(<float_t>radar.receiver.az_patterns[rx_idx][ang_idx])

        ptn_length = len(radar.receiver.el_angles[rx_idx])
        el_angles = np.flip(90-radar.receiver.el_angles[rx_idx])/180*np.pi
        el_pattern = np.flip(radar.receiver.el_patterns[rx_idx])
        for ang_idx in range(0, ptn_length):
            el_ang.push_back(<float_t>el_angles[ang_idx])
            el.push_back(<float_t>el_pattern[ang_idx])

        rx.AddChannel(
            RxChannel[float_t](
                Vec3[float_t](
                    <float_t> radar.receiver.locations[rx_idx, 0],
                    <float_t> radar.receiver.locations[rx_idx, 1],
                    <float_t> radar.receiver.locations[rx_idx, 2]
                ),
                Vec3[float_t](0,0,1),
                az_ang,
                az,
                el_ang,
                el,
                <float_t> radar.receiver.antenna_gains[rx_idx]
            )
        )

    cdef float_t[:,:,:] baseband_re = np.zeros((radar.frames*radar.channel_size, radar.transmitter.pulses, radar.samples_per_pulse), dtype=np.float32)
    cdef float_t[:,:,:] baseband_im = np.zeros((radar.frames*radar.channel_size, radar.transmitter.pulses, radar.samples_per_pulse), dtype=np.float32)

    sim.Run(tx, rx, points_, &baseband_re[0,0,0], &baseband_im[0,0,0])

    if noise:
        baseband = np.array(baseband_re)+1j*np.array(baseband_im)+\
            radar.noise*(np.random.randn(
                    radar.frames*radar.channel_size,
                    radar.transmitter.pulses,
                    radar.samples_per_pulse,
                ) + 1j * np.random.randn(
                    radar.frames*radar.channel_size,
                    radar.transmitter.pulses,
                    radar.samples_per_pulse,
                ))
    else:
        baseband = np.array(baseband_re)+1j*np.array(baseband_im)
    
    return {'baseband':baseband,
            'timestamp':radar.timestamp}