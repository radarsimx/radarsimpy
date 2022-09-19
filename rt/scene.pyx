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


import numpy as np
from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport float_t, int_t, vector
from radarsimpy.lib.cp_radarsimc cimport cp_RxChannel, cp_Target
from radarsimpy.lib.cp_radarsimc cimport cp_TxChannel, cp_Transmitter
from radarsimpy.includes.radarsimc cimport Radar
from radarsimpy.includes.radarsimc cimport TxChannel, Transmitter
from radarsimpy.includes.radarsimc cimport RxChannel, Receiver
from radarsimpy.includes.radarsimc cimport Snapshot, Target, Scene
from libc.stdlib cimport malloc, free

cimport cython
cimport numpy as np


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef scene(radar, targets, density=1, level=None, noise=True, debug=False):
    """
    scene(radar, targets, density=1, level=None, noise=True)

    Radar scene simulator

    :param Radar radar:
        Radar model
    :param list[dict] targets:
        Target list

        [{

        - **model** (*str*) --
            Path to the target model
        - **origin** (*numpy.1darray*) --
            Origin position of the target model (m), [x, y, z].
            ``default [0, 0, 0]``
        - **location** (*numpy.1darray*) --
            Location of the target (m), [x, y, z].
            ``default [0, 0, 0]``
        - **speed** (*numpy.1darray*) --
            Speed of the target (m/s), [vx, vy, vz].
            ``default [0, 0, 0]``
        - **rotation** (*numpy.1darray*) --
            Target's angle (deg), [yaw, pitch, roll].
            ``default [0, 0, 0]``
        - **rotation_rate** (*numpy.1darray*) --
            Target's rotation rate (deg/s),
            [yaw rate, pitch rate, roll rate]
            ``default [0, 0, 0]``

        }]

        *Note*: Target's parameters can be specified with
        ``Radar.timestamp`` to customize the time varying property.
        Example: ``location=(1e-3*np.sin(2*np.pi*1*radar.timestamp), 0, 0)``
    :param float density:
        Ray density (number of rays per wavelength). ``default 1``
    :param str level:
        Fidelity level of the simulation, ``default None``

        - ``None``: Perform one ray tracing simulation for the whole frame
        - ``pulse``: Perform ray tracing for each pulse
        - ``sample``: Perform ray tracing for each sample

    :param bool noise:
        Flag to enable noise calculation, ``default True``
    :param bool debug:
        Flag to enable debug output, ``default False``

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

        - **rays** (*numpy.array*) --
            Received rays

        }
    :rtype: dict
    """
    cdef Transmitter[float_t] c_tx
    cdef Receiver[float_t] c_rx
    cdef Radar[float_t] c_radar
    cdef Scene[double, float_t] c_scene

    cdef float_t[:, :, :, :] radar_loc
    cdef float_t[:, :, :, :] radar_spd
    cdef float_t[:, :, :, :] radar_rot
    cdef float_t[:, :, :, :] radar_rrt

    cdef vector[Vec3[float_t]] c_loc_vect
    cdef vector[Vec3[float_t]] c_spd_vect
    cdef vector[Vec3[float_t]] c_rot_vect
    cdef vector[Vec3[float_t]] c_rrt_vect

    cdef int_t frames = radar.frames
    cdef int_t total_ch = radar.channel_size
    cdef int_t rx_ch = radar.receiver.channel_size
    cdef int_t tx_ch = radar.transmitter.channel_size
    cdef int_t pulses = radar.transmitter.pulses
    cdef int_t samples = radar.samples_per_pulse

    cdef int_t ch_stride = pulses * samples
    cdef int_t pulse_stride = samples
    cdef int_t bb_idx

    cdef int_t fm_idx, tx_idx, ps_idx, sp_idx
    cdef int_t ch_idx, p_idx, s_idx

    """
    Targets
    """
    cdef double[:, :, :] timestamp = radar.timestamp.astype(np.float64)

    cdef int_t target_count = len(targets)

    for idx in range(0, target_count):
        c_scene.AddTarget(
            cp_Target(radar, targets[idx], np.shape(timestamp))
        )

    """
    Transmitter
    """
    c_tx = cp_Transmitter(radar)
    for tx_idx in range(0, tx_ch):
        c_tx.AddChannel(
            cp_TxChannel(radar.transmitter, tx_idx)
        )

    """
    Receiver
    """
    c_rx = Receiver[float_t](
        <float_t> radar.receiver.fs,
        <float_t> radar.receiver.rf_gain,
        <float_t> radar.receiver.load_resistor,
        <float_t> radar.receiver.baseband_gain,
        samples
    )
    for rx_idx in range(0, rx_ch):
        c_rx.AddChannel(
            cp_RxChannel(radar.receiver, rx_idx)
        )

    """
    Radar
    """
    c_radar = Radar[float_t](c_tx, c_rx)

    if len(np.shape(radar.location)) == 4:
        radar_loc = radar.location.astype(np.float32)
        radar_spd = radar.speed.astype(np.float32)
        radar_rot = radar.rotation.astype(np.float32)
        radar_rrt = radar.rotation_rate.astype(np.float32)

        for ch_idx in range(0, radar.channel_size*radar.frames):
            for ps_idx in range(0, radar.transmitter.pulses):
                for sp_idx in range(0, radar.samples_per_pulse):
                    c_loc_vect.push_back(
                        Vec3[float_t](
                            radar_loc[ch_idx, ps_idx, sp_idx,0],
                            radar_loc[ch_idx, ps_idx, sp_idx,1],
                            radar_loc[ch_idx, ps_idx, sp_idx,2]
                        )
                    )
                    c_spd_vect.push_back(
                        Vec3[float_t](
                            radar_spd[ch_idx, ps_idx, sp_idx,0],
                            radar_spd[ch_idx, ps_idx, sp_idx,1],
                            radar_spd[ch_idx, ps_idx, sp_idx,2]
                        )
                    )
                    c_rot_vect.push_back(
                        Vec3[float_t](
                            radar_rot[ch_idx, ps_idx, sp_idx,0],
                            radar_rot[ch_idx, ps_idx, sp_idx,1],
                            radar_rot[ch_idx, ps_idx, sp_idx,2]
                        )
                    )
                    c_rrt_vect.push_back(
                        Vec3[float_t](
                            radar_rrt[ch_idx, ps_idx, sp_idx,0],
                            radar_rrt[ch_idx, ps_idx, sp_idx,1],
                            radar_rrt[ch_idx, ps_idx, sp_idx,2]
                        )
                    )
    else:
        c_loc_vect.push_back(
            Vec3[float_t](
                <float_t>radar.location[0],
                <float_t>radar.location[1],
                <float_t>radar.location[2]
            )
        )
        c_spd_vect.push_back(
            Vec3[float_t](
                <float_t>radar.speed[0],
                <float_t>radar.speed[1],
                <float_t>radar.speed[2]
            )
        )
        c_rot_vect.push_back(
            Vec3[float_t](
                <float_t>radar.rotation[0],
                <float_t>radar.rotation[1],
                <float_t>radar.rotation[2]
            )
        )
        c_rrt_vect.push_back(
            Vec3[float_t](
                <float_t>radar.rotation_rate[0],
                <float_t>radar.rotation_rate[1],
                <float_t>radar.rotation_rate[2]
            )
        )

    c_radar.SetMotion(c_loc_vect,
                      c_spd_vect,
                      c_rot_vect,
                      c_rrt_vect)

    c_scene.SetRadar(c_radar)

    """
    Snapshot
    """
    cdef vector[Snapshot[float_t]] snaps
    cdef int_t level_id = 0

    if level is None:
        level_id = 0
        for fm_idx in range(0, frames):
            for tx_idx in range(0, tx_ch):
                snaps.push_back(
                    Snapshot[float_t](
                        <double> timestamp[fm_idx*total_ch+tx_idx*rx_ch, 0, 0],
                        fm_idx,
                        tx_idx,
                        0,
                        0)
                )
    elif level == 'pulse':
        level_id = 1
        for fm_idx in range(0, frames):
            for tx_idx in range(0, tx_ch):
                for ps_idx in range(0, pulses):
                    snaps.push_back(
                        Snapshot[float_t](
                            timestamp[fm_idx*total_ch+tx_idx*rx_ch, ps_idx, 0],
                            fm_idx,
                            tx_idx,
                            ps_idx,
                            0)
                    )
    elif level == 'sample':
        level_id = 2
        for fm_idx in range(0, frames):
            for tx_idx in range(0, tx_ch):
                for ps_idx in range(0, pulses):
                    for sp_idx in range(0, samples):
                        snaps.push_back(
                            Snapshot[float_t](
                                timestamp[fm_idx*total_ch + tx_idx*rx_ch, ps_idx, sp_idx],
                                fm_idx,
                                tx_idx,
                                ps_idx,
                                sp_idx)
                        )

    cdef double * bb_real = <double *> malloc(frames*total_ch*pulses*samples*sizeof(double))
    cdef double * bb_imag = <double *> malloc(frames*total_ch*pulses*samples*sizeof(double))

    c_scene.RunSimulator(
        level_id,
        debug,
        snaps,
        <float_t>density,
        bb_real,
        bb_imag)

    baseband = np.zeros((frames*total_ch, pulses, samples), dtype=complex)

    for ch_idx in range(0, frames*total_ch):
        for p_idx in range(0, pulses):
            for s_idx in range(0, samples):
                bb_idx = ch_idx * ch_stride + p_idx * pulse_stride + s_idx
                baseband[ch_idx, p_idx, s_idx] = bb_real[bb_idx] +  1j*bb_imag[bb_idx]

    if noise:
        baseband = baseband +\
            radar.noise*(
                np.random.randn(
                    frames*total_ch,
                    pulses,
                    samples,
                ) + \
                1j * np.random.randn(
                    frames*total_ch,
                    pulses,
                    samples,
                )
            )

    free(bb_real)
    free(bb_imag)

    return {'baseband': baseband,
            'timestamp': radar.timestamp}
