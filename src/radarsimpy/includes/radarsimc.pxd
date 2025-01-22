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

#------------------------------------------------------------------------------
# Target and Ray Classes
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
               const bool & is_ground) except +
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
               const bool & is_ground) except +

cdef extern from "ray.hpp":
    cdef cppclass Ray[T]:
        Ray() except +
        Vec3[T] * direction_
        Vec3[T] * location_
        int reflections_

#------------------------------------------------------------------------------
# RCS and Point Cloud
#------------------------------------------------------------------------------
cdef extern from "rcs.hpp":
    cdef cppclass Rcs[T]:
        Rcs() except +
        Rcs(vector[Target[float]] & targets,
            const Vec3[T] & inc_dir,
            const Vec3[T] & obs_dir,
            const Vec3[cpp_complex[T]] & inc_polarization,
            const Vec3[cpp_complex[T]] & obs_polarization,
            const T & frequency,
            const T & density) except +

        T CalculateRcs()

cdef extern from "pointcloud.hpp":
    cdef cppclass PointCloud[T]:
        PointCloud() except +
        void AddTarget(const Target[T] & target)
        void Sbr(const vector[T] & phi,
                 const vector[T] & theta,
                 const Vec3[T] & position)

        vector[Ray[T]] cloud_

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
                    const vector[H] & pulse_start_time,
                    const vector[H] & frame_start_time) except +
        Transmitter(const L & tx_power,
                    const vector[H] & freq,
                    const vector[H] & freq_time,
                    const vector[H] & freq_offset,
                    const vector[H] & pulse_start_time,
                    const vector[H] & frame_start_time,
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
              vector[Vec3[L]] & location_array,
              Vec3[L] speed_array,
              vector[Vec3[L]] & rotation_array,
              Vec3[L] rotrate_array) except +

cdef extern from "snapshot.hpp":
    cdef cppclass Snapshot[T]:
        Snapshot() except +
        Snapshot(const double & time,
                 const int & frame_idx,
                 const int & ch_idx,
                 const int & pulse_idx,
                 const int & sample_idx) except +

#------------------------------------------------------------------------------
# Simulators
#------------------------------------------------------------------------------
cdef extern from "simulator_ideal.hpp":
    cdef cppclass IdealSimulator[H, L]:
        IdealSimulator() except +
        void Run(Radar[H, L] radar,
                 vector[Point[L]] points,
                 double * bb_real,
                 double * bb_imag)

cdef extern from "simulator_scene.hpp":
    cdef cppclass SceneSimulator[H, L]:
        SceneSimulator() except +
        void Run(Radar[H, L] & radar,
                 vector[Target[L]] & targets,
                 int level,
                 bool debug,
                 vector[Snapshot[L]] & snapshots,
                 L density,
                 Vec2[int_t] ray_filter,
                 string log_path,
                 double * bb_real,
                 double * bb_imag)

cdef extern from "simulator_interference.hpp":
    cdef cppclass InterferenceSimulator[H, L]:
        InterferenceSimulator() except +
        void Run(Radar[H, L] radar,
                 Radar[H, L] interf_radar,
                 double *interf_bb_real,
                 double *interf_bb_imag)
