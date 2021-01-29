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

from libc.math cimport sin, cos, sqrt, atan, atan2, acos, pow, fmax, M_PI
from libcpp cimport bool

from radarsimpy.includes.radarsimc cimport TxChannel, Transmitter
from radarsimpy.rt.cp_radarsimc cimport cp_TxChannel, cp_Transmitter
from radarsimpy.includes.radarsimc cimport Snapshot, Target, Aperture, Receiver, RxChannel, Scene
from radarsimpy.rt.cp_radarsimc cimport cp_RxChannel, cp_Target
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t, vector
from radarsimpy.includes.zpvector cimport Vec3
from libcpp.complex cimport complex as cpp_complex


import numpy as np
cimport numpy as np
from stl import mesh

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef scene(radar, targets, correction=0, density=10, level=None, noise=True):
    """
    Alias: ``radarsimpy.rt.scene()``
    
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
    :param float correction:
        Amplitude correction (dB). ``default 0``
    :param float density:
        Ray density (number of rays per wavelength). ``default 10``
    :param str level:
        Fidelity level of the simulation, ``default None``

        - ``None``: Perform one ray tracing simulation for the whole frame
        - ``pulse``: Perform ray tracing for each pulse
        - ``sample``: Perform ray tracing for each sample

    :param bool noise:
        Flag to enable noise calculation, ``default True``

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
    cdef Scene[float_t] radar_scene

    """
    Targets
    """
    timestamp = radar.timestamp

    cdef int_t target_count = len(targets)
    for idx in range(0, target_count):
        radar_scene.AddTarget(cp_Target(radar, targets[idx], np.shape(timestamp)))
    
    """
    Aperture
    """
    cdef float_t[:,:,:] aperture

    cdef float_t[:] aperture_location
    cdef float_t[:] aperture_extension
    if radar.aperture_mesh:
        aperture = radar.aperture_mesh.astype(np.float64)

        radar_scene.SetAperture(
            Aperture[float_t](
                &aperture[0,0,0],
                <int_t> aperture.shape[0]
            )
        )

    else:
        aperture_location = radar.aperture_location.astype(np.float64)
        aperture_extension = radar.aperture_extension.astype(np.float64)
        radar_scene.SetAperture(
            Aperture[float_t](
                <float_t> (radar.aperture_phi/180*np.pi),
                <float_t> (radar.aperture_theta/180*np.pi),
                Vec3[float_t](&aperture_location[0]),
                &aperture_extension[0]
            )
        )

    """
    Transmitter
    """
    radar_scene.SetTransmitter(cp_Transmitter(radar, density))

    for tx_idx in range(0, radar.transmitter.channel_size):
        radar_scene.AddTxChannel(cp_TxChannel(radar.transmitter, tx_idx))

    """
    Receiver
    """ 
    radar_scene.SetReceiver(
        Receiver[float_t](
            <float_t> radar.receiver.fs,
            <float_t> radar.receiver.rf_gain,
            <float_t> radar.receiver.load_resistor,
            <float_t> radar.receiver.baseband_gain,
            <int> radar.samples_per_pulse
        )
    )

    for rx_idx in range(0, radar.receiver.channel_size):
        radar_scene.AddRxChannel(cp_RxChannel(radar.receiver, rx_idx))

    """
    Snapshot
    """
    cdef vector[Snapshot[float_t]] snaps
    cdef int level_id

    if level is None:
        level_id = 0
        for frame_idx in range(0, radar.frames):
            for tx_idx in range(0, radar.transmitter.channel_size):
                snaps.push_back(
                    Snapshot[float_t](
                        <float_t> radar.timestamp[frame_idx*radar.channel_size+tx_idx*radar.receiver.channel_size, 0, 0],
                        frame_idx,
                        tx_idx,
                        0,
                        0
                    )
                )
    elif level == 'pulse':
        level_id = 1
        for frame_idx in range(0, radar.frames):
            for tx_idx in range(0, radar.transmitter.channel_size):
                for pulse_idx in range(0, radar.transmitter.pulses):
                    snaps.push_back(
                        Snapshot[float_t](
                        <float_t> radar.timestamp[frame_idx*radar.channel_size+tx_idx*radar.receiver.channel_size, pulse_idx, 0], frame_idx, tx_idx, pulse_idx, 0)
                    )
    elif level == 'sample':
        level_id = 2
        for frame_idx in range(0, radar.frames):
            for tx_idx in range(0, radar.transmitter.channel_size):
                for pulse_idx in range(0, radar.transmitter.pulses):
                    for sample_idx in range(0, radar.samples_per_pulse):
                        snaps.push_back(
                            Snapshot[float_t](
                            <float_t> radar.timestamp[frame_idx*radar.channel_size+tx_idx*radar.receiver.channel_size, pulse_idx, sample_idx], frame_idx, tx_idx, pulse_idx, sample_idx)
                        )

    cdef vector[cpp_complex[float_t]] *bb_vect = new vector[cpp_complex[float_t]](
        radar.frames*radar.channel_size*radar.transmitter.pulses*radar.samples_per_pulse,
        cpp_complex[float_t](0.0,0.0))

    radar_scene.RunSimulator(
        level_id, <float_t> correction, snaps, bb_vect[0]
    )

    cdef complex[:,:,:] baseband = np.zeros((radar.frames*radar.channel_size, radar.transmitter.pulses, radar.samples_per_pulse), dtype=complex)

    cdef int ch_stride = radar.transmitter.pulses * radar.samples_per_pulse
    cdef int pulse_stride = radar.samples_per_pulse
    cdef int idx_stride

    for ch_idx in range(0, radar.frames*radar.channel_size):
        for p_idx in range(0, radar.transmitter.pulses):
            for s_idx in range(0, radar.samples_per_pulse):
                idx_stride = ch_idx * ch_stride + p_idx * pulse_stride + s_idx
                baseband[ch_idx, p_idx, s_idx] = bb_vect[0][idx_stride].real()+1j*bb_vect[0][idx_stride].imag()

    ray_type = np.dtype([
        ('area', np.float64, (1,)),
        ('distance', np.float64, (1,)),
        ('range_rate', np.float64, (1,)),
        ('refCount', int, (1,)),
        ('channel_id', int, (1,)),
        ('pulse_idx', int, (1,)),
        ('sample_idx', int, (1,)),
        ('level', int, (1,)),
        ('positions', np.float64, (3,)),
        ('directions', np.float64, (3,)),
        ('polarization', np.float64, (3,)),
        ('path_pos', np.float64, (20,3))
        ])


    cdef int total_size = 0
    for snapshot_idx in range(0, snaps.size()):
        total_size = total_size+snaps[snapshot_idx].ray_received.size()

    rays = np.zeros(total_size, dtype=ray_type)

    cdef int count = 0
    for snapshot_idx in range(0, snaps.size()):
        for idx in range(0, snaps[snapshot_idx].ray_received.size()):
            rays[count]['area'] = snaps[snapshot_idx].ray_received[idx].area_
            rays[count]['distance'] = snaps[snapshot_idx].ray_received[idx].range_
            rays[count]['range_rate'] = snaps[snapshot_idx].ray_received[idx].range_rate_
            rays[count]['refCount'] = snaps[snapshot_idx].ray_received[idx].ref_count_
            rays[count]['channel_id'] = snaps[snapshot_idx].ch_idx_
            rays[count]['pulse_idx'] = snaps[snapshot_idx].pulse_idx_
            rays[count]['sample_idx'] = snaps[snapshot_idx].sample_idx_
            rays[count]['level'] = level_id
            rays[count]['positions'][0] = snaps[snapshot_idx].ray_received[idx].loc_[0]
            rays[count]['positions'][1] = snaps[snapshot_idx].ray_received[idx].loc_[1]
            rays[count]['positions'][2] = snaps[snapshot_idx].ray_received[idx].loc_[2]
            rays[count]['directions'][0] = snaps[snapshot_idx].ray_received[idx].dir_[0]
            rays[count]['directions'][1] = snaps[snapshot_idx].ray_received[idx].dir_[1]
            rays[count]['directions'][2] = snaps[snapshot_idx].ray_received[idx].dir_[2]
            rays[count]['polarization'][0] = snaps[snapshot_idx].ray_received[idx].pol_[0]
            rays[count]['polarization'][1] = snaps[snapshot_idx].ray_received[idx].pol_[1]
            rays[count]['polarization'][2] = snaps[snapshot_idx].ray_received[idx].pol_[2]
            rays[count]['path_pos'] = np.zeros((20,3))
            for path_idx in range(0, int(rays[count]['refCount']+2)):
                rays[count]['path_pos'][path_idx, 0] = snaps[snapshot_idx].ray_received[idx].path_[path_idx].loc_[0]
                rays[count]['path_pos'][path_idx, 1] = snaps[snapshot_idx].ray_received[idx].path_[path_idx].loc_[1]
                rays[count]['path_pos'][path_idx, 2] = snaps[snapshot_idx].ray_received[idx].path_[path_idx].loc_[2]
            count=count+1


    if noise:
        baseband = baseband+\
            radar.noise*(np.random.randn(
                    radar.frames*radar.channel_size,
                    radar.transmitter.pulses,
                    radar.samples_per_pulse,
                ) + 1j * np.random.randn(
                    radar.frames*radar.channel_size,
                    radar.transmitter.pulses,
                    radar.samples_per_pulse,
                ))

    del bb_vect

    return {'baseband':baseband,
            'timestamp':radar.timestamp,
            'rays':rays}