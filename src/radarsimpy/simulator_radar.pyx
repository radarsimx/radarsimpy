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
from libcpp.memory cimport shared_ptr, make_shared
from radarsimpy.includes.type_def cimport (
    float_t,
    int_t
)

# RadarSimX core components
from radarsimpy.includes.rsvector cimport Vec2
from radarsimpy.includes.radarsimc cimport (
    Radar,
    TargetsManager,
    PointsManager,
    MeshSimulator,
    PointSimulator,
    InterferenceSimulator,
    IsFreeTier,
    RadarSimErrorCode,
    cpu_policy,
    gpu_policy
)

# RadarSimX library components
from radarsimpy.lib.cp_radarsimc cimport (
    cp_Radar,
    cp_AddTarget,
    cp_AddPoint
)

from radarsimpy.mesh_kit import import_mesh_module

# Constants for better maintainability
cdef:
    int_t MAX_RAY_REFLECTIONS = 10
    int_t DEFAULT_MIN_REFLECTIONS = 0
    float_t SQRT_2 = 1.4142135623730951

cdef inline void validate_free_tier_limits(radar, list targets):
    """
    Validates limitations for the free tier version.
    
    :param radar: The radar configuration object
    :param targets: List of targets to simulate
    :raises RuntimeError: If free tier limitations are exceeded
    """
    if not IsFreeTier():
        return
        
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

cdef inline raise_err(RadarSimErrorCode err):
    """
    Raises appropriate runtime errors based on simulation error types.

    This function handles error reporting for various simulation scenarios,
    providing detailed error messages and potential solutions.

    :param RadarSimErrorCode err: The error type encountered during simulation
    :raises RuntimeError: When a simulation error is encountered, with detailed message
    """
    if err == RadarSimErrorCode.RADARSIMCPP_ERROR_TOO_MANY_RAYS_PER_GRID:
        raise RuntimeError(
            "[ERROR_TOO_MANY_RAYS_PER_GRID] The simulation is attempting to launch an "
            "excessive number of rays in a grid, which exceeds system's memory limitations. "
            "To resolve this issue, please try one or both of the following solutions:\n"
            "1. Reduce the `grid` dimensions for the Transmitter (Tx) Channel.\n"
            "2. Decrease the `density` parameter value in your `sim_radar()` function call."
        )
    else:
        raise RuntimeError(f"Simulation error occurred with code: {err}")


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sim_radar(radar, targets, frame_time=None, density=1, level=None, interf=None, interf_frame_time=None,
                ray_filter=None, back_propagating=False, device="gpu", log_path=None, debug=False):
    """
    sim_radar(radar, targets, density=1, level=None, interf=None, ray_filter=None,
              back_propagating=False, device="gpu", log_path=None, debug=False)

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
            - **skip_diffusion** (*boolean*): Flag to skip the calculation of diffusion reflections of this object. Enable this for large reflectors, such as the ground plane, to reduce processing load. Default: ``False``.

        - **Ideal Point Target**:
          A simplified target defined as a point in space. Each target is represented as a dictionary with the following keys:

            - **location** (*numpy.ndarray*): Target location in meters [x, y, z].
            - **rcs** (*float*): Target's radar cross-section (RCS) in dBsm.
            - **speed** (*numpy.ndarray*): Target velocity in meters per second [vx, vy, vz]. Default: ``[0, 0, 0]``.
            - **phase** (*float*): Target phase in degrees. Default: ``0``.

        *Note*: Target parameters can be time-varying by using ``Radar.timestamp``. For example:
        ``location = (1e-3 * np.sin(2 * np.pi * 1 * radar.timestamp), 0, 0)``

    :param float density:
        Ray density, defined as the number of rays per wavelength. Default: ``1.0``.
    :param str or None level:
        Fidelity level of the simulation. Default: ``None``.

        - ``None`` or ``"frame"``: Perform one ray-tracing simulation for the entire frame.
        - ``"pulse"``: Perform ray-tracing for each pulse.
        - ``"sample"``: Perform ray-tracing for each sample.
    :param Radar or None interf:
        Interference radar object. Default: ``None``.
    :param list or None ray_filter:
        Filters rays based on the number of reflections.
        Only rays with the number of reflections between ``ray_filter[0]``
        and ``ray_filter[1]`` are included in the calculations.
        Default: ``None`` (no filtering).
    :param bool back_propagating:
        Whether to enable back propagation in the simulation. When enabled, the simulation will consider
        rays that propagate back towards the radar after reflecting off targets. Default: ``False``.
    :param str device:
        Execution device for the simulation. Default: ``"gpu"``.

        - ``"gpu"``: Execute simulation on GPU using CUDA (if available, falls back to CPU).
        - ``"cpu"``: Execute simulation on CPU only.
        
        .. note::
            **Performance Consideration**: When using a GPU-compiled module with ``device="cpu"``, 
            OpenMP parallelization is not available for CPU execution, resulting in slower 
            performance compared to a CPU-only compiled module. For optimal CPU performance, 
            use a module compiled without GPU support.
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

    :raises RuntimeError: When simulation limitations are exceeded or errors occur
    :raises ValueError: When invalid simulation parameters are provided
    """

    # Initialize error tracking
    err = RadarSimErrorCode.SUCCESS

    # Validate device parameter
    device_lower = device.lower()
    if device_lower not in ["gpu", "cpu"]:
        raise ValueError(
            f"\nInvalid Device Selection\n"
            f"------------------------\n"
            f"The specified device '{device}' is not recognized.\n\n"
            f"Available devices:\n"
            f"- 'gpu': Execute simulation on GPU (CUDA)\n"
            f"- 'cpu': Execute simulation on CPU\n\n"
            f"Please choose 'gpu' or 'cpu'."
        )

    #----------------------
    # C++ Object Declarations
    #----------------------
    # Core simulation objects
    cdef:
        shared_ptr[Radar[double, float_t]] radar_c = make_shared[Radar[double, float_t]]()
        shared_ptr[Radar[double, float_t]] interf_radar_c = make_shared[Radar[double, float_t]]()
        Vec2[int_t] ray_filter_c
        
    cdef shared_ptr[TargetsManager[float_t]] targets_manager = make_shared[TargetsManager[float_t]]()
    cdef shared_ptr[PointsManager[float_t]] points_manager = make_shared[PointsManager[float_t]]()

    # Simulator instances - declare both CPU and GPU versions
    cdef:
        PointSimulator[double, float_t, cpu_policy] point_sim_cpu
        PointSimulator[double, float_t, gpu_policy] point_sim_gpu
        MeshSimulator[double, float_t, cpu_policy] mesh_sim_cpu
        MeshSimulator[double, float_t, gpu_policy] mesh_sim_gpu
        InterferenceSimulator[double, float_t, cpu_policy] interf_sim_cpu
        InterferenceSimulator[double, float_t, gpu_policy] interf_sim_gpu

    # Size and index variables
    cdef:
        int_t level_id = 0
        int_t ps_idx
        int_t frames_c = np.size(radar.time_prop["frame_start_time"])
        int_t channels_c = radar.array_prop["size"]
        int_t rxsize_c = radar.radar_prop["receiver"].rxchannel_prop["size"]
        int_t txsize_c = radar.radar_prop["transmitter"].txchannel_prop["size"]
        string log_path_c

        # Pre-declare variables for better performance
        int_t frame_idx, ch_idx, rx_ch
        float_t t0
        int_t f_ch_idx
        int_t num_noise_samples

    #----------------------
    # Initialization
    #----------------------
    # Check for deprecated frame_time parameter
    if frame_time is not None or interf_frame_time is not None:
        import warnings
        warnings.warn(
            "The 'frame_time' and 'interf_frame_time' parameters in sim_radar() have been moved to the Radar constructor and are no longer used here. "
            "These parameters will be ignored. Please set frame_time and interf_frame_time when creating the Radar object: "
            "Radar(transmitter, receiver, frame_time=your_value). "
            "These parameters will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )

    # Validate free tier limitations
    validate_free_tier_limits(radar, targets)

    # Basic setup with type safety
    frame_start_time = radar.time_prop["frame_start_time"]
    log_path_c = str.encode(log_path) if log_path is not None else str.encode("")

    #----------------------
    # Timestamp Processing
    #----------------------
    radar_ts = radar.time_prop["origin_timestamp"]
    radar_ts_shape = radar.time_prop["origin_timestamp_shape"]

    timestamp = radar.time_prop["timestamp"]
    ts_shape = radar.time_prop["timestamp_shape"]

    # Set ray filter with constants
    if ray_filter is None:
        ray_filter_c = Vec2[int_t](DEFAULT_MIN_REFLECTIONS, MAX_RAY_REFLECTIONS)
    else:
        ray_filter_c = Vec2[int_t](<int_t>ray_filter[0], <int_t>ray_filter[1])

    #----------------------
    # Target Processing
    #----------------------
    cdef double[:, :, :] timestamp_mv = timestamp.astype(np.float64)

    # Process each target with optimized loop
    mesh_module = None
    cdef int_t target_count = len(targets)

    for target_idx in range(target_count):
        tgt = targets[target_idx]
        if "model" in tgt:
            if mesh_module is None:
                mesh_module = import_mesh_module()

            cp_AddTarget(radar, tgt, timestamp, mesh_module, targets_manager.get())
        else:
            # Extract point target parameters with defaults
            loc = tgt["location"]
            spd = tgt.get("speed", (0, 0, 0))
            rcs = tgt["rcs"]
            phs = tgt.get("phase", 0)

            cp_AddPoint(loc, spd, rcs, phs, ts_shape, points_manager.get())

    radar_c = cp_Radar(radar, frame_start_time)

    cdef double[:,:,::1] bb_real = np.empty(ts_shape, order='C', dtype=np.float64)
    cdef double[:,:,::1] bb_imag = np.empty(ts_shape, order='C', dtype=np.float64)

    radar_c.get()[0].InitBaseband(&bb_real[0][0][0], &bb_imag[0][0][0])

    #----------------------
    # Simulation Execution
    #----------------------
    # Validate simulation fidelity level
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

    # Run ideal point target simulation and mesh simulation based on device selection
    if device_lower == "cpu":
        # CPU execution
        point_sim_cpu.Run(radar_c, points_manager)

        # Run scene simulation
        err = mesh_sim_cpu.Run(
            radar_c,
            targets_manager,
            level_id,
            <float_t> density,
            ray_filter_c,
            back_propagating,
            log_path_c,
            debug)
    else:
        # GPU execution (default)
        point_sim_gpu.Run(radar_c, points_manager)

        # Run scene simulation
        err = mesh_sim_gpu.Run(
            radar_c,
            targets_manager,
            level_id,
            <float_t> density,
            ray_filter_c,
            back_propagating,
            log_path_c,
            debug)

        radar_c.get()[0].SyncBaseband()

    if err:
        raise_err(err)

    if radar.radar_prop["receiver"].bb_prop["bb_type"] == "real":
        baseband = np.asarray(bb_real)
    else:
        baseband = np.asarray(bb_real)+1j*np.asarray(bb_imag)

    #----------------------
    # Noise Generation
    #----------------------
    # Calculate noise parameters
    max_ts = np.max(radar_ts)
    min_ts = np.min(radar_ts)
    num_noise_samples = int(np.ceil((max_ts - min_ts) * radar.radar_prop["receiver"].bb_prop["fs"])) + 1

    # Initialize noise matrix based on baseband type
    cdef str bb_type = radar.radar_prop["receiver"].bb_prop["bb_type"]
    if bb_type == "real":
        noise_mat = np.zeros(ts_shape, dtype=np.float64)
    elif bb_type == "complex":
        noise_mat = np.zeros(ts_shape, dtype=complex)
    else:
        raise ValueError(f"Unsupported baseband type: {bb_type}")

    # Generate noise for each frame
    cdef float_t noise_level = radar.sample_prop["noise"]
    cdef float_t sqrt_2_inv = 1.0 / SQRT_2

    for frame_idx in range(frames_c):
        if bb_type == "real":
            noise_per_frame_rx = noise_level * np.random.randn(rxsize_c, num_noise_samples)
        elif bb_type == "complex":
            noise_per_frame_rx = (noise_level * sqrt_2_inv * 
                                 (np.random.randn(rxsize_c, num_noise_samples) + 
                                  1j * np.random.randn(rxsize_c, num_noise_samples)))

        for ch_idx in range(radar_ts_shape[0]):
            for ps_idx in range(radar_ts_shape[1]):
                f_ch_idx = ch_idx + frame_idx * radar_ts_shape[0]
                t0 = (radar_ts[ch_idx, ps_idx, 0] - min_ts) * radar.radar_prop["receiver"].bb_prop["fs"]
                rx_ch = ch_idx % rxsize_c
                noise_mat[f_ch_idx, ps_idx, :] = noise_per_frame_rx[rx_ch, int(t0):(int(t0) + radar_ts_shape[2])]

    #----------------------
    # Interference Processing
    #----------------------
    cdef object interference = None
    
    # Run interference simulation if interference radar is provided
    if interf is not None:
        # Use main radar frame time if interference frame time not specified
        interf_frame_start_time = np.array(interf.time_prop["frame_start_time"], dtype=np.float64)
        interf_radar_c = cp_Radar(interf, interf_frame_start_time)

        # Initialize baseband for interference calculation
        radar_c.get()[0].InitBaseband(&bb_real[0][0][0], &bb_imag[0][0][0])

        # Run interference simulation based on device selection
        if device_lower == "cpu":
            interf_sim_cpu.Run(radar_c, interf_radar_c)
        else:
            interf_sim_gpu.Run(radar_c, interf_radar_c)
            radar_c.get()[0].SyncBaseband()

        # Extract interference data based on baseband type
        if bb_type == "real":
            interference = np.asarray(bb_real)
        else:
            interference = np.asarray(bb_real) + 1j * np.asarray(bb_imag)

    # Return the simulation results as a structured dictionary
    return {
        "baseband": baseband,
        "noise": noise_mat,
        "timestamp": timestamp,
        "interference": interference
    }
