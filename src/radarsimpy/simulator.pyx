# distutils: language = c++
"""
A Python module for radar simulation

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


import numpy as np

from libcpp.string cimport string

from radarsimpy.includes.type_def cimport vector
from radarsimpy.includes.type_def cimport float_t
from radarsimpy.includes.type_def cimport int_t

from radarsimpy.includes.radarsimc cimport Radar
from radarsimpy.includes.radarsimc cimport Snapshot
from radarsimpy.includes.radarsimc cimport Point
from radarsimpy.includes.radarsimc cimport Scene
from radarsimpy.includes.radarsimc cimport Simulator

from radarsimpy.includes.radarsimc cimport IsFreeTier

from radarsimpy.lib.cp_radarsimc cimport cp_Radar
from radarsimpy.lib.cp_radarsimc cimport cp_Target
from radarsimpy.lib.cp_radarsimc cimport cp_Point

cimport cython
cimport numpy as np
np.import_array()

np_float = np.float32


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sim_radar(radar, targets, density=1, level=None, log_path=None, debug=False, interf=None):
    """
    sim_radar(radar, targets, density=1, level=None, log_path=None, debug=False, interf=None)

    This function generates radar's baseband response of a scene using the given radar and targets.

    :param radar: The radar object used for the scene.
    :type radar: Radar
    :param targets: The targets in the scene. It could be either an ideal point target or a 3D mesh object.
    
        3D mesh target:

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
        - **permittivity** (*complex*) --
            Target's permittivity. Perfect electric conductor (PEC) if not specified.
        - **unit** (*str*) --
            Unit of target model. Supports `mm`, `cm`, and `m`. Default is `m`.

        }]

        Ideal point target:

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
    :type targets: list
    :param density: Ray density. Number of rays per wavelength (default=1).
    :type density: float
    :param level: Fidelity level of the simulation (default=None).
    
        - ``None``: Perform one ray tracing simulation for the whole frame
        - ``pulse``: Perform ray tracing for each pulse
        - ``sample``: Perform ray tracing for each sample
    :type level: str or None
    :param log_path: Provide the path to save ray data (default=None, no data will be saved).
    :type log_path: str
    :param debug: Whether to enable debug mode (default=False).
    :type debug: bool
    :param interf: Interference radar (default=None).
    :type interf: Radar

    :return: A dictionary containing the baseband data, noise, timestamp, and interference (if available).
        {

        - **baseband** (*numpy.3darray*) --
            Time domain baseband data.
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
        
        - **noise** (*numpy.3darray*) --
            Time domain noise data.
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

        - **interference** (*numpy.3darray*) --
            Time domain interference data.
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
    
    # radar
    cdef Radar[float_t] radar_c

    # interference radar
    cdef Radar[float_t] interf_radar_c

    # point targets
    cdef vector[Point[float_t]] point_vt

    # simulator
    cdef Scene[double, float_t] scene_c
    cdef Simulator[float_t] sim_c

    cdef vector[Snapshot[float_t]] snaps

    cdef int_t level_id = 0
    cdef int_t fm_idx, tx_idx, ps_idx, sp_idx

    cdef int_t frames_c = radar.time_prop["frame_size"]
    cdef int_t channles_c = radar.array_prop["size"]
    cdef int_t rxsize_c = radar.radar_prop["receiver"].rxchannel_prop["size"]
    cdef int_t txsize_c = radar.radar_prop["transmitter"].txchannel_prop["size"]
    cdef int_t pulses_c, samples_c

    cdef string log_path_c

    flag_run_scene = False

    if log_path is not None:
        log_path_c = str.encode(log_path)
    else:
        log_path_c = str.encode("")

    if IsFreeTier():
        if len(targets) > 3:
            raise Exception("You're currently using RadarSimPy's FreeTier, which limits RCS simulation to 3 maximum targets. Please consider supporting my work by upgrading to the standard version. Just choose any amount greater than zero on https://radarsimx.com/product/radarsimpy/ to access the standard version download links. Your support will help improve the software. Thank you for considering it.")

        if radar.radar_prop["transmitter"].txchannel_prop["size"] > 2:
            raise Exception("You're currently using RadarSimPy's FreeTier, which imposes a restriction on the maximum number of transmitter channels to 2. Please consider supporting my work by upgrading to the standard version. Just choose any amount greater than zero on https://radarsimx.com/product/radarsimpy/ to access the standard version download links. Your support will help improve the software. Thank you for considering it.")
        
        if radar.radar_prop["receiver"].rxchannel_prop["size"] > 2:
            raise Exception("You're currently using RadarSimPy's FreeTier, which imposes a restriction on the maximum number of receiver channels to 2. Please consider supporting my work by upgrading to the standard version. Just choose any amount greater than zero on https://radarsimx.com/product/radarsimpy/ to access the standard version download links. Your support will help improve the software. Thank you for considering it.")
    
    ts_shape = np.shape(radar.time_prop["timestamp"])

    """
    Targets
    """
    cdef double[:, :, :] timestamp_mv = radar.time_prop["timestamp"].astype(np.float64)

    for _, tgt in enumerate(targets):
        if "model" in tgt:
            flag_run_scene = True
            scene_c.AddTarget(
                cp_Target(radar, tgt, ts_shape)
            )
        else:
            loc = tgt["location"]
            spd = tgt.get("speed", (0, 0, 0))
            rcs = tgt["rcs"]
            phs = tgt.get("phase", 0)

            point_vt.push_back(
                cp_Point(loc, spd, rcs, phs, ts_shape)
            )

    radar_c = cp_Radar(radar)
    
    cdef double[:,:,::1] bb_real = np.empty(ts_shape, order='C', dtype=np.float64)
    cdef double[:,:,::1] bb_imag = np.empty(ts_shape, order='C', dtype=np.float64)

    if point_vt.size() > 0:
        sim_c.Run(radar_c, point_vt, &bb_real[0][0][0], &bb_imag[0][0][0])
        baseband = np.asarray(bb_real)+1j*np.asarray(bb_imag)
    else:
        baseband = 0

    if flag_run_scene:
        scene_c.SetRadar(radar_c)
        """
        Snapshot
        """
        if level is None:
            level_id = 0
            pulses_c  = 1
            samples_c = 1
        elif level == "pulse":
            level_id = 1
            pulses_c  = radar.radar_prop["transmitter"].waveform_prop["pulses"]
            samples_c = 1
        elif level == "sample":
            level_id = 2
            pulses_c = radar.radar_prop["transmitter"].waveform_prop["pulses"]
            samples_c = radar.sample_prop["samples_per_pulse"]
        else:
            raise Exception("Unknown fidelity level. `None`: Perform one ray tracing simulation for the whole frame; `pulse`: Perform ray tracing for each pulse; `sample`: Perform ray tracing for each sample.")

        for fm_idx in range(0, frames_c):
            for tx_idx in range(0, txsize_c):
                for ps_idx in range(0, pulses_c):
                    for sp_idx in range(0, samples_c):
                        snaps.push_back(
                            Snapshot[float_t](
                                timestamp_mv[fm_idx*channles_c + tx_idx*rxsize_c, ps_idx, sp_idx],
                                fm_idx,
                                tx_idx,
                                ps_idx,
                                sp_idx)
                        )

        scene_c.RunSimulator(
            level_id,
            debug,
            snaps,
            <float_t> density,
            log_path_c,
            &bb_real[0][0][0],
            &bb_imag[0][0][0])

        baseband = baseband+np.asarray(bb_real)+1j*np.asarray(bb_imag)

    num_noise_samples = np.ceil((np.max(timestamp_mv)-np.min(timestamp_mv))* radar.radar_prop["receiver"].bb_prop["fs"]).astype(int)+1
   
    if radar.radar_prop["receiver"].bb_prop["bb_type"] == "real":
        noise = np.zeros(ts_shape, dtype=np.float64)
        noise_per_rx = radar.sample_prop["noise"] * np.random.randn(rxsize_c, num_noise_samples)
    elif radar.radar_prop["receiver"].bb_prop["bb_type"] == "complex":
        noise = np.zeros(ts_shape, dtype=complex)
        noise_per_rx = radar.sample_prop["noise"]/ np.sqrt(2) * (np.random.randn(rxsize_c, num_noise_samples) + 1j*np.random.randn(rxsize_c, num_noise_samples))

    for ch_idx in range(0, ts_shape[0]):
        for ps_idx in range(0, ts_shape[1]):
            t0 = (timestamp_mv[ch_idx, ps_idx, 0] - np.min(timestamp_mv))*radar.radar_prop["receiver"].bb_prop["fs"]
            rx_ch = ch_idx%rxsize_c
            noise[ch_idx, ps_idx, :] = noise_per_rx[rx_ch, int(t0):(int(t0)+ts_shape[2])]


    if interf is not None:
        interf_radar_c = cp_Radar(interf)

        sim_c.Interference(radar_c, interf_radar_c, &bb_real[0][0][0], &bb_imag[0][0][0])
        interference = np.asarray(bb_real)+1j*np.asarray(bb_imag)

    else:
        interference = None

    return {"baseband": baseband,
            "noise": noise,
            "timestamp": radar.time_prop["timestamp"],
            "interference": interference}


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef simc(radar, targets, interf=None):
    """
    simc(radar, targets, interf=None)

    **deprecated** Please use `simulator.sim_radar(radar, targets, interf=None)`

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

    :return:
        {

        - **baseband** (*numpy.3darray*) --
            Time domain baseband data.
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
        
        - **noise** (*numpy.3darray*) --
            Time domain noise data.
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

        - **interference** (*numpy.3darray*) --
            Time domain interference data.
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

    return sim_radar(radar, targets, interf=interf)