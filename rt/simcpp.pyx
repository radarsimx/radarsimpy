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
from radarsimpy.rt.cp_radarsimc cimport cp_Point
from radarsimpy.includes.radarsimc cimport TxChannel, Transmitter
from radarsimpy.rt.cp_radarsimc cimport cp_TxChannel, cp_Transmitter
from radarsimpy.includes.radarsimc cimport RxChannel, Receiver
from radarsimpy.rt.cp_radarsimc cimport cp_RxChannel
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

    timestamp = radar.timestamp

    """
    Targets
    """
    for idx in range(0, len(targets)):
        location = targets[idx]['location']
        speed = targets[idx].get('speed', (0, 0, 0))
        rcs = targets[idx]['rcs']
        phase = targets[idx].get('phase', 0)

        points.push_back(
            cp_Point(location, speed, rcs, phase, np.shape(timestamp))
        )

    """
    Transmitter
    """
    tx = cp_Transmitter(radar)

    """
    Transmitter Channels
    """
    for tx_idx in range(0, radar.transmitter.channel_size):
        tx.AddChannel(cp_TxChannel(radar.transmitter, tx_idx))

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
        rx.AddChannel(cp_RxChannel(radar.receiver, rx_idx))

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