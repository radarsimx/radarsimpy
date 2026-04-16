# distutils: language = c++
"""
RadarSimPy - C++ Interface Declarations

This module provides Cython interface declarations for the C++ radar simulation backend.
It defines the Python-accessible interface to the core radar simulation engine including:

- **Radar System Components**: Transmitters, receivers, antennas with complex patterns
- **Target Models**: Point targets and 3D mesh targets with material properties  
- **Simulation Engines**: Point cloud, mesh-based ray tracing, RCS calculation
- **Memory Management**: Efficient data transfer utilities and GPU memory handling
- **Error Handling**: Comprehensive error codes and exception handling

Template Parameters:
    T: Precision type for geometric calculations (typically float)
    H: High precision type for time/frequency (typically double) 
    L: Low precision type for spatial data (typically float)

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

#------------------------------------------------------------------------------
# Standard Library Imports
#------------------------------------------------------------------------------
from libcpp cimport bool
from libcpp.complex cimport complex as cpp_complex
from libcpp.string cimport string
from libcpp.memory cimport shared_ptr

#------------------------------------------------------------------------------
# RadarSimPy Type Definitions
#------------------------------------------------------------------------------
from radarsimpy.includes.rsvector cimport Vec3, Vec2
from radarsimpy.includes.type_def cimport int_t, vector

#------------------------------------------------------------------------------
# Execution Policy
# Template-based execution policy for CPU/GPU selection
#------------------------------------------------------------------------------
cdef extern from "core/execution_policy.hpp" namespace "radarsimx":
    # Base execution policy tag
    cdef cppclass execution_policy_base:
        pass

    # CPU execution policy
    cdef cppclass cpu_policy(execution_policy_base):
        @staticmethod
        bool is_gpu
        @staticmethod
        bool is_cpu
        @staticmethod
        const char* name()
        @staticmethod
        int device_id()

    # GPU execution policy
    cdef cppclass gpu_policy(execution_policy_base):
        @staticmethod
        bool is_gpu
        @staticmethod
        bool is_cpu
        @staticmethod
        const char* name()
        @staticmethod
        int device_id()

    # Global policy instances
    cdef cpu_policy cpu
    cdef gpu_policy gpu

#------------------------------------------------------------------------------
# Error Handling
# Error codes for radar simulation operations
#------------------------------------------------------------------------------
cdef extern from "core/enums.hpp":
    cdef enum RadarSimErrorCode:
        SUCCESS                               # Operation completed successfully
        RADARSIMCPP_ERROR_NULL_POINTER        # Null pointer error
        RADARSIMCPP_ERROR_INVALID_PARAMETER   # Invalid parameter provided
        RADARSIMCPP_ERROR_MEMORY_ALLOCATION   # Memory allocation failed
        RADARSIMCPP_ERROR_FREE_TIER_LIMIT     # Free tier usage limit exceeded
        RADARSIMCPP_ERROR_EXCEPTION           # General exception occurred
        RADARSIMCPP_ERROR_TOO_MANY_RAYS_PER_GRID  # Ray density exceeds grid capacity
        RADARSIMCPP_ERROR_CUDA_DEVICE_QUERY   # Failed to query CUDA device memory info

#------------------------------------------------------------------------------
# Memory Management Utilities
# Efficient data transfer between Python and C++
#------------------------------------------------------------------------------
cdef extern from "libs/mem_lib.hpp":
    cdef void Mem_Copy[T](T * ptr, int_t size, vector[T] &vect) except +
    cdef void Mem_Copy_Complex[T](T * ptr_real, T * ptr_imag,
                                  int_t size, vector[cpp_complex[T]] &vect) except +
    cdef void Mem_Copy_Vec3[T](T *ptr_x, T *ptr_y, T *ptr_z,
                               int_t size, vector[Vec3[T]] &vect) except +

#------------------------------------------------------------------------------
# License Management
# License validation functionality via LicenseManager singleton
#------------------------------------------------------------------------------
cdef extern from "libs/license_manager.hpp":
    cdef cppclass LicenseManager:
        @staticmethod
        LicenseManager& GetInstance()
        void SetLicense(const string& license_file_path, const string& product)
        void SetLicense(const vector[string]& license_file_paths, const string& product)
        bint IsLicensed() const
        bint IsFreeTier() const
        string GetLicenseInfo() const

#------------------------------------------------------------------------------
# Transmitter
# Radar transmitter with antenna patterns and waveform modulation
#------------------------------------------------------------------------------
cdef extern from "transmitter.hpp":
    cdef cppclass Transmitter[H, L]:
        Transmitter() except +

        # Basic transmitter constructor
        Transmitter(const L & tx_power,                          # Transmit power (dBm)
                    const vector[H] & freq,                      # Frequency array (Hz)
                    const vector[H] & freq_time,                 # Frequency time array (s)
                    const vector[H] & freq_offset,               # Frequency offset array (Hz)
                    const vector[H] & pulse_start_time) except + # Pulse start times (s)

        # Constructor with SSB phase noise specification (deferred per-frame generation)
        Transmitter(const L & tx_power,
                    const vector[H] & freq,
                    const vector[H] & freq_time,
                    const vector[H] & freq_offset,
                    const vector[H] & pulse_start_time,
                    const vector[H] & pn_freq,                   # Phase noise freq offsets (Hz)
                    const vector[H] & pn_power,                  # Phase noise power (dBc/Hz)
                    const H & pn_fs,                             # Sampling frequency (Hz)
                    const int & pn_num_samples,                  # Number of samples
                    const unsigned long long & pn_seed,          # Random seed
                    const bool & pn_validation) except +         # Validation mode

        void AddChannel(const Vec3[L] & location,                      # Antenna location
                        const Vec3[cpp_complex[L]] & polar,            # Polarization vector
                        const vector[L] & phi,                         # Azimuth angle array
                        const vector[L] & phi_ptn,                     # Azimuth pattern
                        const vector[L] & theta,                       # Elevation angle array
                        const vector[L] & theta_ptn,                   # Elevation pattern
                        const L & antenna_gain,                        # Antenna gain (dB)
                        const vector[L] & mod_t,                       # Modulation time array
                        const vector[cpp_complex[L]] & mod_var,        # Modulation variables
                        const vector[cpp_complex[L]] & pulse_mod,      # Pulse modulation
                        const L & delay,                               # Channel delay (s)
                        const L & grid) except +                       # Time grid resolution

#------------------------------------------------------------------------------
# Receiver
# Radar receiver with antenna patterns and RF/baseband processing
#------------------------------------------------------------------------------
cdef extern from "receiver.hpp":
    cdef cppclass Receiver[T]:
        Receiver() except +

        # Receiver constructor with RF and baseband parameters
        Receiver(const T & fs,                                   # Sampling frequency (Hz)
                 const T & rf_gain,                              # RF gain (dB)
                 const T & resistor,                             # Load resistor (Ohms)
                 const T & baseband_gain,                        # Baseband gain (dB)
                 const T & baseband_bw) except +                 # Baseband bandwidth (Hz)

        void AddChannel(const Vec3[T] & location,                # Antenna location
                        const Vec3[cpp_complex[T]] & polar,      # Polarization vector
                        const vector[T] & phi,                   # Azimuth angle array
                        const vector[T] & phi_ptn,               # Azimuth pattern
                        const vector[T] & theta,                 # Elevation angle array
                        const vector[T] & theta_ptn,             # Elevation pattern
                        const T & antenna_gain)                  # Antenna gain (dB)

#------------------------------------------------------------------------------
# Radar System Configuration
# Complete radar system combining transmitter, receiver, and platform dynamics
#------------------------------------------------------------------------------
cdef extern from "radar.hpp":
    cdef cppclass Radar[H, L]:
        Radar() except +

        # Radar system constructor
        Radar(const shared_ptr[Transmitter[H, L]] & tx,          # Transmitter configuration
              const shared_ptr[Receiver[L]] & rx,                # Receiver configuration
              vector[H] & frame_start_time,                      # Frame timing array (s)
              vector[Vec3[L]] & location_array,                  # Platform locations
              Vec3[L] speed_array,                               # Platform velocity
              vector[Vec3[L]] & rotation_array,                  # Platform orientations
              Vec3[L] rotrate_array) except +                    # Platform rotation rates

        # Memory management for baseband data
        void InitBaseband(H *bb_real,                            # Real baseband buffer
                          H *bb_imag) except +                   # Imaginary baseband buffer
        void SyncBaseband() except +                             # Synchronize device memory

        # Radar properties
        int sample_size_                                         # Number of samples per pulse

#------------------------------------------------------------------------------
# Points Manager
# Container for managing collections of radar point targets
#------------------------------------------------------------------------------
cdef extern from "points_manager.hpp":
    cdef cppclass PointsManager[T]:
        PointsManager() except +

        # Add a point with time-varying properties
        void AddPoint(const vector[Vec3[T]] & location_array,    # Spatial positions over time
                      const Vec3[T] & speed_array,               # Velocity vectors over time
                      const vector[T] & rcs_array,               # RCS values over time (dBsm)
                      const vector[T] & phase_array) except +    # Phase values over time (rad)

        # Add a static point with constant properties
        void AddPointSimple(const Vec3[T] & location,            # Point location
                            const Vec3[T] & speed,               # Point velocity
                            const T & rcs,                       # Point RCS value (dBsm)
                            const T & phase) except +            # Point phase value (rad)

#------------------------------------------------------------------------------
# Targets Manager
# Container for managing multiple 3D mesh targets in radar simulation
#------------------------------------------------------------------------------
cdef extern from "targets_manager.hpp":
    cdef cppclass TargetsManager[T]:
        TargetsManager() except +

        # Add a target with time-varying motion
        void AddTarget(const T * points,                            # Vertex coordinates array
                       const int_t * cells,                         # Cell connectivity array
                       const int_t & cell_size,                     # Number of cells in mesh
                       const Vec3[T] & origin,                      # Target reference origin
                       const vector[Vec3[T]] & location_array,      # Time-varying locations
                       const vector[Vec3[T]] & speed_array,         # Time-varying velocities
                       const vector[Vec3[T]] & rotation_array,      # Time-varying rotations
                       const vector[Vec3[T]] & rotation_rate_array, # Time-varying rotation rates
                       const cpp_complex[T] & ep,                   # Relative permittivity
                       const cpp_complex[T] & mu,                   # Relative permeability
                       const bool & skip_diffusion,                 # Skip diffusion flag
                       const T & density,                           # Ray density (0.0 = global)
                       const bool & environment) except +           # Environment target flag

        # Add a static target with constant motion parameters
        void AddTargetSimple(const T * points,
                             const int_t * cells,
                             const int_t & cell_size,
                             const Vec3[T] & origin,
                             const Vec3[T] & location,
                             const Vec3[T] & speed,
                             const Vec3[T] & rotation,
                             const Vec3[T] & rotation_rate,
                             const bool & skip_diffusion,           # Skip diffusion flag
                             const T & density,                     # Ray density (0.0 = global)
                             const bool & environment) except +     # Environment target flag

#------------------------------------------------------------------------------
# Ray Tracing Primitives
# Ray representation for LiDAR and ray tracing operations
#------------------------------------------------------------------------------
cdef extern from "ray.hpp":
    cdef cppclass Ray[T]:
        Ray() except +
        Vec3[T] * direction_      # Ray direction vector
        Vec3[T] * location_       # Ray origin/intersection point
        Vec3[T] * normal_         # Surface normal at hit point
        T * range_                # Cumulative range at each bounce
        int reflections_          # Number of reflections encountered

#------------------------------------------------------------------------------
# Simulation Engines
#------------------------------------------------------------------------------

# --- Point Target Simulation ---
# High-performance ideal point target simulation engine.
# For fast simulation of simple point scatterers with known RCS values.
cdef extern from "simulator_point.hpp":
    cdef cppclass PointSimulator[H, L, ExecutionPolicy]:
        PointSimulator() except +

        RadarSimErrorCode Run(const shared_ptr[Radar[H, L]] & radar,               # Radar configuration
                              const shared_ptr[PointsManager[L]] & points_manager) # Point targets

# --- Mesh-based Ray Tracing Simulation ---
# Physics-based 3D mesh target simulation using ray tracing and physical optics.
# For realistic simulation of complex targets with detailed geometry.
cdef extern from "simulator_mesh.hpp":
    cdef cppclass MeshSimulator[H, L, ExecutionPolicy]:
        MeshSimulator() except +

        RadarSimErrorCode Run(const shared_ptr[Radar[H, L]] & radar,                  # Radar configuration
                              const shared_ptr[TargetsManager[L]] & targets_manager,  # Targets manager
                              int level,                         # Fidelity (0=LOW, 1=MED, 2=HIGH)
                              L density,                         # Ray density for physical optics
                              Vec2[int_t] ray_filter,            # Ray index filter [min, max]
                              bool back_propagating,             # Enable back-propagation
                              string log_path,                   # Debug log file path
                              bool dry_run)                      # Dry run mode flag

# --- Radar Cross Section (RCS) Calculation ---
# RCS calculation using physical optics and scattering theory.
cdef extern from "simulator_rcs.hpp":
    cdef cppclass RcsSimulator[T]:
        RcsSimulator() except +

        RadarSimErrorCode Run(const shared_ptr[TargetsManager[float]] & targets_manager,  # Targets manager
                              vector[Vec3[T]] inc_dir_array,         # Incident wave directions
                              vector[Vec3[T]] obs_dir_array,         # Observation directions
                              Vec3[cpp_complex[T]] inc_polarization,  # Incident polarization
                              Vec3[cpp_complex[T]] obs_polarization,  # Observation polarization
                              T frequency,                           # Operating frequency (Hz)
                              T density) except +                    # Ray density

        const vector[T]& GetRcs()                                    # Get computed RCS values

# --- LiDAR Point Cloud Generation ---
# LiDAR simulation for generating 3D point clouds from mesh targets.
cdef extern from "simulator_lidar.hpp":
    cdef cppclass LidarSimulator[T]:
        LidarSimulator() except +

        RadarSimErrorCode Run(const shared_ptr[TargetsManager[T]] & targets_manager,  # Targets manager
                              const vector[T] & phi,               # Azimuth angles (radians)
                              const vector[T] & theta,             # Elevation angles (radians)
                              const Vec3[T] & position)            # LiDAR sensor position

        vector[Ray[T]] cloud_                                       # Generated point cloud rays

# --- Radar Interference Simulation ---
# Radar-to-radar interference simulation for EMC analysis.
cdef extern from "simulator_interference.hpp":
    cdef cppclass InterferenceSimulator[H, L, ExecutionPolicy]:
        InterferenceSimulator() except +

        RadarSimErrorCode Run(const shared_ptr[Radar[H, L]] & radar,         # Victim radar
                              const shared_ptr[Radar[H, L]] & interf_radar)  # Interfering radar

# --- Noise Simulation ---
# Receiver thermal noise generation with proper channel/pulse correlation.
cdef extern from "simulator_noise.hpp":
    cdef cppclass NoiseSimulator[H, L, ExecutionPolicy]:
        NoiseSimulator() except +

        RadarSimErrorCode Run(const shared_ptr[Radar[H, L]] & radar,  # Radar configuration
                              H noise_level,                          # RMS noise amplitude
                              bint is_complex,                        # Complex baseband flag
                              const H* timestamps,                    # Origin timestamp array
                              int ts_channel_size,                    # Timestamp channels
                              int ts_pulse_size,                      # Timestamp pulses
                              int ts_sample_size,                     # Timestamp samples
                              H* noise_real,                          # Output noise real
                              H* noise_imag,                          # Output noise imag
                              unsigned long long seed)                # Random seed

#------------------------------------------------------------------------------
# End of RadarSimPy C++ Interface Declarations
#------------------------------------------------------------------------------
