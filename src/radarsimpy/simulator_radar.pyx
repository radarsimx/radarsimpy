# distutils: language = c++

"""
The Python Module for Radar Simulation

This module provides advanced radar simulation capabilities,
enabling users to simulate radar baseband responses for complex scenes.
It supports point targets, 3D mesh objects, and offers features such as
ray-tracing, interference modeling, and noise simulation.

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

# NumPy imports
import numpy as np
cimport numpy as np
np.import_array()

# Cython decorators
cimport cython

# C++ type definitions
from libcpp.string cimport string
from radarsimpy.includes.type_def cimport (
    vector,
    float_t,
    int_t
)

# RadarSimX core components
from radarsimpy.includes.rsvector cimport Vec2
from radarsimpy.includes.radarsimc cimport (
    Radar,
    Point,
    Target,
    MeshSimulator,
    PointSimulator,
    InterferenceSimulator,
    IsFreeTier,
    RadarSimErrorCode
)

# RadarSimX library components
from radarsimpy.lib.cp_radarsimc cimport (
    cp_Radar,
    cp_Target,
    cp_Point
)

from radarsimpy.mesh_kit import import_mesh_module

def raise_err(err):
    """
    Raises appropriate runtime errors based on simulation error types.

    This function handles error reporting for various simulation scenarios,
    providing detailed error messages and potential solutions.

    :param RadarSimErrorCode err: The error type encountered during simulation
    :raises RuntimeError: When a simulation error is encountered, with detailed message
    """
    if err == RadarSimErrorCode.ERROR_TOO_MANY_RAYS_PER_GRID:
        raise RuntimeError(f"[ERROR_TOO_MANY_RAYS_PER_GRID] The simulation is attempting to launch an excessive number of rays in a grid, which exceeds system's memory limitations. To resolve this issue, please try one or both of the following solutions:\n\
    1. Reduce the `grid` dimensions for the Transmitter (Tx) Channel.\n\
    2. Decrease the `density` parameter value in your `sim_radar()` function call.")


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sim_radar(radar, targets, frame_time=0, density=1, level=None, interf=None, interf_frame_time=None,
                ray_filter=None, back_propagating=False, log_path=None, debug=False):
    """
    sim_radar(radar, targets, frame_time=0, density=1, level=None, interf=None, ray_filter=None, back_propagating=False, log_path=None, debug=False)

    Simulates the radar's baseband response for a given scene.

    This function generates the radar's baseband response using the provided radar configuration and target data.
    It supports both ideal point targets and 3D mesh objects, and allows for varying levels of fidelity in the simulation.
    Additional options include interference modeling, ray density specification, and logging of simulation data.

    :param Radar radar:
        The radar object to be used for the simulation.
    :param list targets:
        The list of targets in the scene. Targets can be either ideal point targets or 3D mesh objects.

        - **3D Mesh Target**:
          A target represented as a 3D model. Each target is defined as a dictionary with the following keys:

            - **model** (*str*): Path to the target model file.
            - **origin** (*numpy.ndarray*): Origin position (rotation and translation center) of the target model [x, y, z] in meters. Default: ``[0, 0, 0]``.
            - **location** (*numpy.ndarray*): Location of the target in meters [x, y, z]. Default: ``[0, 0, 0]``.
            - **speed** (*numpy.ndarray*): Target velocity in meters per second [vx, vy, vz]. Default: ``[0, 0, 0]``.
            - **rotation** (*numpy.ndarray*): Target orientation in degrees [yaw, pitch, roll]. Default: ``[0, 0, 0]``.
            - **rotation_rate** (*numpy.ndarray*): Target's angular velocity in degrees per second [yaw rate, pitch rate, roll rate]. Default: ``[0, 0, 0]``.
            - **permittivity** (*complex*): Target's permittivity. Defaults to a perfect electric conductor (PEC).
            - **unit** (*str*): Unit of the target model. Supported values: ``mm``, ``cm``, ``m``. Default: ``m``.
            - **skip_diffusion** (*boolean*): Flag to skip the calcuation of diffusion reflections of this object. Enable this for large reflectors, such as the ground plane, to reduce processing load. Default: ``False``.

        - **Ideal Point Target**:
          A simplified target defined as a point in space. Each target is represented as a dictionary with the following keys:

            - **location** (*numpy.ndarray*): Target location in meters [x, y, z].
            - **rcs** (*float*): Target's radar cross-section (RCS) in dBsm.
            - **speed** (*numpy.ndarray*): Target velocity in meters per second [vx, vy, vz]. Default: ``[0, 0, 0]``.
            - **phase** (*float*): Target phase in degrees. Default: ``0``.

        *Note*: Target parameters can be time-varying by using ``Radar.timestamp``. For example:
        ``location = (1e-3 * np.sin(2 * np.pi * 1 * radar.timestamp), 0, 0)``

    :param float or list frame_time:
        Radar firing times or frame instances, specified as a float or a list of time values. Default: ``0``.
    :param float density:
        Ray density, defined as the number of rays per wavelength. Default: ``1.0``.
    :param str or None level:
        Fidelity level of the simulation. Default: ``None``.

        - ``None``: Perform one ray-tracing simulation for the entire frame.
        - ``pulse``: Perform ray-tracing for each pulse.
        - ``sample``: Perform ray-tracing for each sample.
    :param Radar or None interf:
        Interference radar object. Default: ``None``.
    :param float or list or None interf_frame_time:
        Frame times for interference radar simulation. If None, uses the same frame_time as the main radar.
        Default: ``None``.
    :param list or None ray_filter:
        Filters rays based on the number of reflections.
        Only rays with the number of reflections between ``ray_filter[0]``
        and ``ray_filter[1]`` are included in the calculations.
        Default: ``None`` (no filtering).
    :param bool back_propagating:
        Whether to enable back propagation in the simulation. When enabled, the simulation will consider
        rays that propagate back towards the radar after reflecting off targets. Default: ``False``.
    :param str or None log_path:
        Path to save ray-tracing data. Default: ``None`` (does not save data).
    :param bool debug:
        Whether to enable debug mode. When enabled, additional debug information will be printed
        during the simulation process. Default: ``False``.

    :return:
        A dictionary containing the simulated baseband response and related data:

        - **baseband** (*numpy.ndarray*): Time-domain baseband data with shape ``[channels/frames, pulses, samples]``. 
          The channel/frame order is as follows (with ``K`` frames, ``M`` Tx channels and ``N`` Rx channels):

            - [0, :, :] ``Frame[0] → Tx[0] → Rx[0]``
            - [1, :, :] ``Frame[0] → Tx[0] → Rx[1]``
            - ...
            - [N-1, :, :] ``Frame[0] → Tx[0] → Rx[N-1]``
            - [N, :, :] ``Frame[0] → Tx[1] → Rx[0]``
            - ...
            - [M·N-1, :, :] ``Frame[0] → Tx[M-1] → Rx[N-1]``
            - [M·N, :, :] ``Frame[1] → Tx[0] → Rx[0]``
            - ...
            - [K·M·N-1, :, :] ``Frame[K-1] → Tx[M-1] → Rx[N-1]``

        - **noise** (*numpy.ndarray*): Time-domain noise data with the same shape and order as `baseband`.
        - **interference** (*numpy.ndarray*): Time-domain interference data (if applicable), with the same shape and order as `baseband`.
        - **timestamp** (*numpy.ndarray*): Timestamp array, directly derived from ``Radar.timestamp``.

    :rtype: dict
    """

    err = RadarSimErrorCode.SUCCESS

    #----------------------
    # C++ Object Declarations
    #----------------------
    # Core simulation objects
    cdef:
        Radar[double, float_t] radar_c
        Radar[double, float_t] interf_radar_c
        vector[Point[float_t]] point_vt
        vector[Target[float_t]] target_vt
        Vec2[int_t] ray_filter_c
        
    # Simulator instances
    cdef:
        MeshSimulator[double, float_t] mesh_sim_c
        PointSimulator[double, float_t] point_sim_c
        InterferenceSimulator[double, float_t] interf_sim_c

    # Size and index variables
    cdef:
        int_t level_id = 0
        int_t ps_idx
        int_t frames_c = np.size(frame_time)
        int_t channles_c = radar.array_prop["size"]
        int_t rxsize_c = radar.radar_prop["receiver"].rxchannel_prop["size"]
        int_t txsize_c = radar.radar_prop["transmitter"].txchannel_prop["size"]
        string log_path_c

    #----------------------
    # Initialization
    #----------------------
    # FreeTier validation
    if IsFreeTier():
        if len(targets) > 2:
            raise RuntimeError(
                "\nTrial Version Limitation - Target Count\n"
                "----------------------------------------\n"
                "Current limitation: Maximum 2 targets\n"
                "Your scene: {} targets\n\n"
                "To simulate more targets, please upgrade to the Standard Version:\n"
                "→ https://radarsimx.com/product/radarsimpy/\n"
                .format(len(targets))
            )

        if radar.radar_prop["transmitter"].txchannel_prop["size"] > 1:
            raise RuntimeError(
                "\nTrial Version Limitation - Transmitter Channels\n"
                "----------------------------------------------\n"
                "Current limitation: 1 transmitter channel\n"
                "Your configuration: {} channels\n\n"
                "To use multiple transmitter channels, please upgrade to the Standard Version:\n"
                "→ https://radarsimx.com/product/radarsimpy/\n"
                .format(radar.radar_prop["transmitter"].txchannel_prop["size"])
            )

        if radar.radar_prop["receiver"].rxchannel_prop["size"] > 1:
            raise RuntimeError(
                "\nTrial Version Limitation - Receiver Channels\n"
                "-------------------------------------------\n"
                "Current limitation: 1 receiver channel\n"
                "Your configuration: {} channels\n\n"
                "To use multiple receiver channels, please upgrade to the Standard Version:\n"
                "→ https://radarsimx.com/product/radarsimpy/\n"
                .format(radar.radar_prop["receiver"].rxchannel_prop["size"])
            )

    # Basic setup
    frame_start_time = np.array(frame_time, dtype=np.float64)
    log_path_c = str.encode(log_path) if log_path is not None else str.encode("")

    #----------------------
    # Timestamp Processing
    #----------------------
    radar_ts = radar.time_prop["timestamp"]
    radar_ts_shape = np.shape(radar.time_prop["timestamp"])

    if frames_c > 1:
        toffset = np.repeat(
            np.tile(
                np.expand_dims(
                    np.expand_dims(frame_start_time, axis=1),
                    axis=2,
                ),
                (
                    1,
                    radar_ts_shape[1],
                    radar_ts_shape[2],
                ),
            ),
            channles_c,
            axis=0,
        )

        timestamp = (
            np.tile(radar_ts, (frames_c, 1, 1)) + toffset
        )
    elif frames_c == 1:
        timestamp = radar_ts + frame_start_time

    ts_shape = np.shape(timestamp)

    # Set ray filter
    if ray_filter is None:
        ray_filter_c = Vec2[int_t](0, 10)
    else:
        ray_filter_c = Vec2[int_t](<int_t>ray_filter[0], <int_t>ray_filter[1])

    #----------------------
    # Target Processing
    #----------------------
    cdef double[:, :, :] timestamp_mv = timestamp.astype(np.float64)

    # Process each target
    mesh_module = None
    for _, tgt in enumerate(targets):
        if "model" in tgt:
            if mesh_module is None:
                mesh_module = import_mesh_module()
            target_vt.push_back(cp_Target(radar, tgt, timestamp, mesh_module))
        else:
            loc = tgt["location"]
            spd = tgt.get("speed", (0, 0, 0))
            rcs = tgt["rcs"]
            phs = tgt.get("phase", 0)

            point_vt.push_back(
                cp_Point(loc, spd, rcs, phs, ts_shape)
            )

    radar_c = cp_Radar(radar, frame_start_time)

    cdef double[:,:,::1] bb_real = np.empty(ts_shape, order='C', dtype=np.float64)
    cdef double[:,:,::1] bb_imag = np.empty(ts_shape, order='C', dtype=np.float64)

    radar_c.InitBaseband(&bb_real[0][0][0],
                         &bb_imag[0][0][0])

    #----------------------
    # Simulation Execution
    #----------------------
    # Run ideal point target simulation
    if point_vt.size() > 0:
        point_sim_c.Run(radar_c, point_vt)

    # Run scene simulation if there are 3D mesh targets
    if target_vt.size() > 0:
        level_map = {None: 0, "frame": 0, "pulse": 1, "sample": 2}
        try:
            level_id = level_map[level]
        except KeyError:
            raise ValueError(
                "\nInvalid Simulation Fidelity Level\n"
                "------------------------------\n"
                "The specified simulation fidelity level is not recognized.\n\n"
                "Available levels:\n"
                "- None or 'frame': One ray tracing simulation per frame\n"
                "    • Assumes linear motion during the frame\n"
                "    • Best performance, suitable for most scenarios\n"
                "- 'pulse': Ray tracing for each pulse\n"
                "    • Assumes linear motion during the pulse\n"
                "    • Increased computation time\n"
                "- 'sample': Ray tracing for each sample\n"
                "    • Highest accuracy for complex motion\n"
                "    • Significantly longer computation time\n\n"
                "Your input: '{}'\n\n"
                "Choose the appropriate level based on:\n"
                "1. Target motion complexity\n"
                "2. Required accuracy\n"
                "3. Available computation time\n"
                .format(level)
            )

        # Run scene simulation
        err = mesh_sim_c.Run(
            radar_c,
            target_vt,
            level_id,
            <float_t> density,
            ray_filter_c,
            back_propagating,
            log_path_c,
            debug)

        if err:
            radar_c.FreeDeviceMemory()
            raise_err(err)

    radar_c.SyncBaseband()

    if radar.radar_prop["receiver"].bb_prop["bb_type"] == "real":
        baseband = np.asarray(bb_real)
    else:
        baseband = np.asarray(bb_real)+1j*np.asarray(bb_imag)

    #----------------------
    # Noise Generation
    #----------------------
    # Generate noise matrix
    max_ts = np.max(radar_ts)
    min_ts = np.min(radar_ts)
    num_noise_samples = int(np.ceil((max_ts-min_ts)* radar.radar_prop["receiver"].bb_prop["fs"]))+1

    if radar.radar_prop["receiver"].bb_prop["bb_type"] == "real":
        noise_mat = np.zeros(ts_shape, dtype=np.float64)
    elif radar.radar_prop["receiver"].bb_prop["bb_type"] == "complex":
        noise_mat = np.zeros(ts_shape, dtype=complex)

    # Add noise to each frame
    for frame_idx in range(0, frames_c):
        if radar.radar_prop["receiver"].bb_prop["bb_type"] == "real":
            noise_per_frame_rx = radar.sample_prop["noise"] * np.random.randn(rxsize_c, num_noise_samples)
        elif radar.radar_prop["receiver"].bb_prop["bb_type"] == "complex":
            noise_per_frame_rx = radar.sample_prop["noise"]/ np.sqrt(2) * (np.random.randn(rxsize_c, num_noise_samples) + 1j*np.random.randn(rxsize_c, num_noise_samples))

        for ch_idx in range(0, radar_ts_shape[0]):
            for ps_idx in range(0, radar_ts_shape[1]):
                f_ch_idx = ch_idx+frame_idx*radar_ts_shape[0]
                t0 = (radar_ts[ch_idx, ps_idx, 0] - min_ts)*radar.radar_prop["receiver"].bb_prop["fs"]
                rx_ch = ch_idx%rxsize_c
                noise_mat[f_ch_idx, ps_idx, :] = noise_per_frame_rx[rx_ch, int(t0):(int(t0)+radar_ts_shape[2])]

    #----------------------
    # Interference Processing
    #----------------------
    # Run interference simulation if interference radar is provided
    if interf is not None:
        if interf_frame_time is None:
            interf_frame_time = frame_time
        
        interf_frame_start_time= np.array(interf_frame_time, dtype=np.float64)
        interf_radar_c = cp_Radar(interf, interf_frame_start_time)
        radar_c.InitBaseband(&bb_real[0][0][0],
                             &bb_imag[0][0][0])

        interf_sim_c.Run(radar_c, interf_radar_c)
        radar_c.SyncBaseband()

        if radar.radar_prop["receiver"].bb_prop["bb_type"] == "real":
            interference = np.asarray(bb_real)
        else:
            interference = np.asarray(bb_real)+1j*np.asarray(bb_imag)

        interf_radar_c.FreeDeviceMemory()
    else:
        interference = None

    radar_c.FreeDeviceMemory()

    # Return the simulation results
    return {"baseband": baseband,
            "noise": noise_mat,
            "timestamp": timestamp,
            "interference": interference}
