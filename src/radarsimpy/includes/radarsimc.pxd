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

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------
from libcpp cimport bool
from libcpp.complex cimport complex as cpp_complex
from libcpp.string cimport string
from radarsimpy.includes.rsvector cimport Vec3, Vec2
from radarsimpy.includes.type_def cimport int_t, vector

#------------------------------------------------------------------------------
# Memory Management
#------------------------------------------------------------------------------
cdef extern from "libs/mem_lib.hpp":
    cdef void Mem_Copy[T](T * ptr, int_t size, vector[T] &vect) except +
    cdef void Mem_Copy_Complex[T](T * ptr_real, T * ptr_imag, 
                                 int_t size, vector[cpp_complex[T]] &vect) except +
    cdef void Mem_Copy_Vec3[T](T *ptr_x, T *ptr_y, T *ptr_z,
                              int_t size, vector[Vec3[T]] &vect) except +

cdef extern from "libs/free_tier.hpp":
    cdef int IsFreeTier() except +

cdef extern from "type_def.hpp":
    cdef enum ErrorType:
        NO_ERROR
        ERROR_TOO_MANY_RAYS_PER_GRID

#------------------------------------------------------------------------------
# Target
#------------------------------------------------------------------------------
cdef extern from "target.hpp":
    cdef cppclass Target[T]:
        Target() except +
        Target(const T * points,
               const int_t * cells,
               const int_t & cell_size,
               const Vec3[T] & origin,
               const vector[Vec3[T]] & location_array,
               const vector[Vec3[T]] & speed_array,
               const vector[Vec3[T]] & rotation_array,
               const vector[Vec3[T]] & rotation_rate_array,
               const cpp_complex[T] & ep,
               const cpp_complex[T] & mu,
               const bool & skip_diffusion) except +
        Target(const T * points,
               const int_t * cells,
               const int_t & cell_size) except +
        Target(const T * points,
               const int_t * cells,
               const int_t & cell_size,
               const Vec3[T] & origin,
               const Vec3[T] & location,
               const Vec3[T] & speed,
               const Vec3[T] & rotation,
               const Vec3[T] & rotation_rate,
               const bool & skip_diffusion) except +

#------------------------------------------------------------------------------
# Ray
#------------------------------------------------------------------------------
cdef extern from "ray.hpp":
    cdef cppclass Ray[T]:
        Ray() except +
        Vec3[T] * direction_
        Vec3[T] * location_
        int reflections_

#------------------------------------------------------------------------------
# RCS
#------------------------------------------------------------------------------
cdef extern from "simulator_rcs.hpp":
    cdef cppclass RcsSimulator[T]:
        RcsSimulator() except +
        T Run(vector[Target[float]] & targets,
              Vec3[T] inc_dir,
              Vec3[T] obs_dir,
              Vec3[cpp_complex[T]] inc_polarization,
              Vec3[cpp_complex[T]] obs_polarization,
              T frequency,
              T density) except +

#------------------------------------------------------------------------------
# Point Cloud
#------------------------------------------------------------------------------
cdef extern from "simulator_lidar.hpp":
    cdef cppclass LidarSimulator[T]:
        LidarSimulator() except +
        void AddTarget(const Target[T] & target)
        void Run(const vector[T] & phi,
                 const vector[T] & theta,
                 const Vec3[T] & position)

        vector[Ray[T]] cloud_

#------------------------------------------------------------------------------
# Point Target
#------------------------------------------------------------------------------
cdef extern from "point.hpp":
    cdef cppclass Point[T]:
        Point() except +
        Point(const vector[Vec3[T]] & loc,
              const Vec3[T] & speed,
              const vector[T] & rcs,
              const vector[T] & phs) except +

#------------------------------------------------------------------------------
# Transmitter Components
#------------------------------------------------------------------------------
cdef extern from "transmitter.hpp":
    cdef cppclass TxChannel[T]:
        TxChannel() except +
        TxChannel(const Vec3[T] & location,
                  const Vec3[cpp_complex[T]] & polar,
                  const vector[T] & phi,
                  const vector[T] & phi_ptn,
                  const vector[T] & theta,
                  const vector[T] & theta_ptn,
                  const T & antenna_gain,
                  const vector[T] & mod_t,
                  const vector[cpp_complex[T]] & mod_var,
                  const vector[cpp_complex[T]] & pulse_mod,
                  const T & delay,
                  const T & grid) except +

    cdef cppclass Transmitter[H, L]:
        Transmitter() except +
        Transmitter(const L & tx_power,
                    const vector[H] & freq,
                    const vector[H] & freq_time,
                    const vector[H] & freq_offset,
                    const vector[H] & pulse_start_time) except +
        Transmitter(const L & tx_power,
                    const vector[H] & freq,
                    const vector[H] & freq_time,
                    const vector[H] & freq_offset,
                    const vector[H] & pulse_start_time,
                    const vector[cpp_complex[H]] & phase_noise) except +
        void AddChannel(const TxChannel[L] & channel)

#------------------------------------------------------------------------------
# Receiver Components
#------------------------------------------------------------------------------
cdef extern from "receiver.hpp":
    cdef cppclass RxChannel[T]:
        RxChannel() except +
        RxChannel(const Vec3[T] & location,
                  const Vec3[cpp_complex[T]] & polar,
                  const vector[T] & phi,
                  const vector[T] & phi_ptn,
                  const vector[T] & theta,
                  const vector[T] & theta_ptn,
                  const T & antenna_gain) except +

    cdef cppclass Receiver[T]:
        Receiver() except +
        Receiver(const T & fs,
                 const T & rf_gain,
                 const T & resistor,
                 const T & baseband_gain,
                 const T & baseband_bw) except +
        void AddChannel(const RxChannel[T] & channel)

#------------------------------------------------------------------------------
# Radar System
#------------------------------------------------------------------------------
cdef extern from "radar.hpp":
    cdef cppclass Radar[H, L]:
        Radar() except +
        Radar(Transmitter[H, L] & tx,
              Receiver[L] & rx,
              vector[H] & frame_start_time,
              vector[Vec3[L]] & location_array,
              Vec3[L] speed_array,
              vector[Vec3[L]] & rotation_array,
              Vec3[L] rotrate_array) except +
        void InitBaseband(H *bb_real,
                          H *bb_imag) except +
        void SyncBaseband() except +
        void FreeDeviceMemory() except +

#------------------------------------------------------------------------------
# Simulators
#------------------------------------------------------------------------------
cdef extern from "simulator_point.hpp":
    cdef cppclass PointSimulator[H, L]:
        PointSimulator() except +
        void Run(Radar[H, L] & radar,
                 vector[Point[L]] & points)

cdef extern from "simulator_mesh.hpp":
    cdef cppclass MeshSimulator[H, L]:
        MeshSimulator() except +
        ErrorType Run(Radar[H, L] & radar,
                      vector[Target[L]] & targets,
                      int level,
                      L density,
                      Vec2[int_t] ray_filter,
                      bool back_propagating,
                      string log_path,
                      bool debug)

cdef extern from "simulator_interference.hpp":
    cdef cppclass InterferenceSimulator[H, L]:
        InterferenceSimulator() except +
        void Run(Radar[H, L] & radar,
                 Radar[H, L] & interf_radar)
