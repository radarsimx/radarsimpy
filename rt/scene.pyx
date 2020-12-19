# distutils: language = c++
# cython: language_level=3

# This script contains classes that define all the parameters for
# a radar system

# This script requires that `numpy` be installed within the Python
# environment you are running this script in.

# This file can be imported as a module and contains the following
# class:

# * Transmitter - A class defines parameters of a radar transmitter
# * Receiver - A class defines parameters of a radar receiver
# * Radar - A class defines basic parameters of a radar system

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

from radarsimpy.includes.radarsimc cimport Snapshot, Target, Aperture, Transmitter, TxChannel, Receiver, RxChannel, Scene
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t, vector
from radarsimpy.includes.zpvector cimport Vec3
from libcpp cimport complex


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

        [
            {

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

            }

        ]

        *Note*: Target's parameters can be specified with
            ``Radar.timestamp`` to customize the time varying property.
            Example: ``location=(1e-3*np.sin(2*np.pi*1*radar.timestamp), 0, 0)``
    :param float correction:
        Amplitude correction (dB). ``default 0``
    :param float density:
        Ray density (number of rays per wavelength). ``default 10``
    :param int level:
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

    cdef float_t[:,:,:] mesh_memview
    cdef float_t[:] origin

    cdef int_t target_count = len(targets)
    cdef vector[Vec3[float_t]] c_loc_array
    cdef vector[Vec3[float_t]] c_speed_array
    cdef vector[Vec3[float_t]] c_rotation_array
    cdef vector[Vec3[float_t]] c_rotation_rate_array
    
    for idx in range(0, target_count):
        c_loc_array.clear()
        c_speed_array.clear()
        c_rotation_array.clear()
        c_rotation_rate_array.clear()

        t_mesh = mesh.Mesh.from_file(targets[idx]['model'])
        mesh_memview = t_mesh.vectors.astype(np.float32)

        origin = np.array(targets[idx].get('origin', (0,0,0)), dtype=np.float32)

        location = targets[idx].get('location', (0,0,0))
        speed = targets[idx].get('speed', (0,0,0))
        rotation = targets[idx].get('rotation', (0,0,0))
        rotation_rate = targets[idx].get('rotation_rate', (0,0,0))

        if np.size(location[0]) > 1 or np.size(location[1])  > 1 or np.size(location[2]) > 1 or np.size(speed[0]) > 1 or np.size(speed[1]) > 1 or np.size(speed[2]) >1 or np.size(rotation[0]) > 1 or np.size(rotation[1]) > 1 or np.size(rotation[2]) >1 or np.size(rotation_rate[0]) > 1 or np.size(rotation_rate[1]) > 1 or np.size(rotation_rate[2])>1:
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

            if np.size(speed[0]) > 1:
                sptx_t = speed[0]
            else:
                sptx_t = np.full_like(timestamp, speed[0])

            if np.size(speed[1]) > 1:
                spty_t = speed[1]
            else:
                spty_t = np.full_like(timestamp, speed[1])
            
            if np.size(speed[2]) > 1:
                sptz_t = speed[2]
            else:
                sptz_t = np.full_like(timestamp, speed[2])

            if np.size(rotation[0]) > 1:
                rotx_t = rotation[0]
            else:
                rotx_t = np.full_like(timestamp, rotation[0])

            if np.size(rotation[1]) > 1:
                roty_t = rotation[1]
            else:
                roty_t = np.full_like(timestamp, rotation[1])
            
            if np.size(rotation[2]) > 1:
                rotz_t = rotation[2]
            else:
                rotz_t = np.full_like(timestamp, rotation[2])

            if np.size(rotation_rate[0]) > 1:
                rotratx_t = rotation_rate[0]
            else:
                rotratx_t = np.full_like(timestamp, rotation_rate[0])

            if np.size(rotation_rate[1]) > 1:
                rotraty_t = rotation_rate[1]
            else:
                rotraty_t = np.full_like(timestamp, rotation_rate[1])
            
            if np.size(rotation_rate[2]) > 1:
                rotratz_t = rotation_rate[2]
            else:
                rotratz_t = np.full_like(timestamp, rotation_rate[2])

            for ch_idx in range(0, radar.channel_size*radar.frames):
                for ps_idx in range(0, radar.transmitter.pulses):
                    for sp_idx in range(0, radar.samples_per_pulse):
                        c_loc_array.push_back(Vec3[float_t](
                            <float_t> tgx_t[ch_idx, ps_idx, sp_idx],
                            <float_t> tgy_t[ch_idx, ps_idx, sp_idx],
                            <float_t> tgz_t[ch_idx, ps_idx, sp_idx]
                        ))
                        c_speed_array.push_back(Vec3[float_t](
                            <float_t> sptx_t[ch_idx, ps_idx, sp_idx],
                            <float_t> spty_t[ch_idx, ps_idx, sp_idx],
                            <float_t> sptz_t[ch_idx, ps_idx, sp_idx])
                        )
                        c_rotation_array.push_back(Vec3[float_t](
                            <float_t> (rotx_t[ch_idx, ps_idx, sp_idx]/180*np.pi),
                            <float_t> (roty_t[ch_idx, ps_idx, sp_idx]/180*np.pi),
                            <float_t> (rotz_t[ch_idx, ps_idx, sp_idx]/180*np.pi))
                        )
                        c_rotation_rate_array.push_back(Vec3[float_t](
                            <float_t> (rotratx_t[ch_idx, ps_idx, sp_idx]/180*np.pi),
                            <float_t> (rotraty_t[ch_idx, ps_idx, sp_idx]/180*np.pi),
                            <float_t> (rotratz_t[ch_idx, ps_idx, sp_idx]/180*np.pi))
                        )
                        
        else:
            c_loc_array.push_back(Vec3[float_t](
                <float_t> location[0],
                <float_t> location[1],
                <float_t> location[2]
            ))
            c_speed_array.push_back(Vec3[float_t](
                <float_t> speed[0],
                <float_t> speed[1],
                <float_t> speed[2])
            )
            c_rotation_array.push_back(Vec3[float_t](
                <float_t> (rotation[0]/180*np.pi),
                <float_t> (rotation[1]/180*np.pi),
                <float_t> (rotation[2]/180*np.pi))
            )
            c_rotation_rate_array.push_back(Vec3[float_t](
                <float_t> (rotation_rate[0]/180*np.pi),
                <float_t> (rotation_rate[1]/180*np.pi),
                <float_t> (rotation_rate[2]/180*np.pi))
            )

        radar_scene.AddTarget(Target[float_t](&mesh_memview[0,0,0],
            <int_t> mesh_memview.shape[0],
            Vec3[float_t](&origin[0]),
            c_loc_array,
            c_speed_array,
            c_rotation_array,
            c_rotation_rate_array,
            <bool> targets[idx].get('is_ground', False)))
    
    """
    Aperture
    """
    cdef float_t[:,:,:] aperture

    cdef float_t[:] aperture_location
    cdef float_t[:] aperture_extension
    if radar.aperture_mesh:
        aperture = radar.aperture_mesh.astype(np.float32)

        radar_scene.SetAperture(
            Aperture[float_t](
                &aperture[0,0,0],
                <int_t> aperture.shape[0]
            )
        )

    else:
        aperture_location = radar.aperture_location.astype(np.float32)
        aperture_extension = radar.aperture_extension.astype(np.float32)
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

    radar_scene.SetTransmitter(
        Transmitter[float_t](
            fc_vector,
            <float_t> radar.transmitter.slope,
            <float_t> radar.transmitter.tx_power,
            chirp_start_time,
            frame_time,
            <int> radar.frames,
            <int> radar.transmitter.pulses,
            <float_t> density
        )
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

        radar_scene.AddTxChannel(
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
                <float_t> (radar.transmitter.grid[tx_idx]/180*np.pi)
            )
        )

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
        az_ang.clear()
        az.clear()
        el_ang.clear()
        el.clear()

        ptn_length = len(radar.receiver.az_angles[rx_idx])
        for ang_idx in range(0, ptn_length):
            az_ang.push_back(<float_t>(radar.receiver.az_angles[rx_idx][ang_idx]/180*np.pi))
            az.push_back(<float_t>radar.receiver.az_patterns[rx_idx][ang_idx])

        ptn_length = len(radar.receiver.el_angles[rx_idx])
        el_angles = np.flip(90-radar.receiver.el_angles[tx_idx])/180*np.pi
        el_pattern = np.flip(radar.receiver.el_patterns[tx_idx])
        for ang_idx in range(0, ptn_length):
            el_ang.push_back(<float_t>el_angles[ang_idx])
            el.push_back(<float_t>el_pattern[ang_idx])

        radar_scene.AddRxChannel(
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
                # radar_scene.AddSnapshot(
                #     Snapshot[float_t](
                #         <float_t> radar.timestamp[frame_idx*radar.channel_size+tx_idx*radar.receiver.channel_size, 0, 0],
                #         frame_idx,
                #         tx_idx,
                #         0,
                #         0
                #     )
                # )
    elif level == 'pulse':
        level_id = 1
        for frame_idx in range(0, radar.frames):
            for tx_idx in range(0, radar.transmitter.channel_size):
                for pulse_idx in range(0, radar.transmitter.pulses):
                    snaps.push_back(
                        Snapshot[float_t](
                        <float_t> radar.timestamp[frame_idx*radar.channel_size+tx_idx*radar.receiver.channel_size, pulse_idx, 0], frame_idx, tx_idx, pulse_idx, 0)
                    )
                    # radar_scene.AddSnapshot(
                    #     Snapshot[float_t](
                    #     <float_t> radar.timestamp[frame_idx*radar.channel_size+tx_idx*radar.receiver.channel_size, pulse_idx, 0], frame_idx, tx_idx, pulse_idx, 0))
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
                        # radar_scene.AddSnapshot(
                        #     Snapshot[float_t](
                        #     <float_t> radar.timestamp[frame_idx*radar.channel_size+tx_idx*radar.receiver.channel_size, pulse_idx, sample_idx], frame_idx, tx_idx, pulse_idx, sample_idx))

    cdef float_t[:,:,:] baseband_re = np.zeros((radar.frames*radar.channel_size, radar.transmitter.pulses, radar.samples_per_pulse), dtype=np.float32)
    cdef float_t[:,:,:] baseband_im = np.zeros((radar.frames*radar.channel_size, radar.transmitter.pulses, radar.samples_per_pulse), dtype=np.float32)

    # cdef vector[RayPy[float_t]] ray_received
    radar_scene.RunSimulator(
        level_id, <float_t> correction, snaps, &baseband_re[0,0,0], &baseband_im[0,0,0]
    )

    ray_type = np.dtype([
        ('area', np.float32, (1,)),
        ('distance', np.float32, (1,)),
        ('range_rate', np.float32, (1,)),
        ('refCount', int, (1,)),
        ('channel_id', int, (1,)),
        ('pulse_idx', int, (1,)),
        ('sample_idx', int, (1,)),
        ('level', int, (1,)),
        ('positions', np.float32, (3,)),
        ('directions', np.float32, (3,)),
        ('polarization', np.float32, (3,)),
        ('path_pos', np.float32, (20,3))
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
            'timestamp':radar.timestamp,
            'rays':rays}