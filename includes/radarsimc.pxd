# distutils: language = c++
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
# cython: language_level=3

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

from radarsimpy.includes.zpvector cimport Vec3
from radarsimpy.includes.type_def cimport float_t, int_t
from radarsimpy.includes.type_def cimport vector
from libcpp cimport bool
from libcpp.complex cimport complex as cpp_complex

import numpy as np
cimport numpy as np


"""
target interface
"""
cdef extern from "target.hpp":
    cdef cppclass Target[T]:
        Target() except +
        Target(T * points,
               int_t * cells,
               int cell_size,
               Vec3[T] origin,
               vector[Vec3[T]] location_array,
               vector[Vec3[T]] speed_array,
               vector[Vec3[T]] rotation_array,
               vector[Vec3[T]] rotation_rate_array,
               cpp_complex[T] ep,
               cpp_complex[T] mu,
               bool is_ground) except +
        Target(T * points,
               int_t * cells,
               int cell_size) except +
        Target(T * points,
               int_t * cells,
               int cell_size,
               Vec3[T] origin,
               Vec3[T] location,
               Vec3[T] speed,
               Vec3[T] rotation,
               Vec3[T] rotation_rate,
               bool is_ground) except +


cdef extern from "ray.hpp":
    cdef cppclass Ray[T, Tg = *]:
        Ray() except +
        Vec3[T] * direction_
        Vec3[T] * location_
        int reflections_


cdef extern from "raypool.hpp":
    cdef cppclass RayPool[T, Tg = *]:
        RayPool() except +
        vector[Ray[T]] pool_


"""
rcs
"""
cdef extern from "rcs.hpp":
    cdef cppclass Rcs[T]:
        Rcs() except +
        Rcs(const Target[float] & mesh,
            const Vec3[T] & inc_dir,
            const Vec3[T] & obs_dir,
            const Vec3[T] & polarization,
            const T & frequency,
            const T & density) except +

        T CalculateRcs()


"""
pointcloud
"""
cdef extern from "pointcloud.hpp":
    cdef cppclass PointCloud[T]:
        PointCloud() except +
        void AddTarget(const Target[T] & target)
        void Sbr(const vector[T] & phi,
                 const vector[T] & theta,
                 const Vec3[T] & position)

        vector[Ray[T]] cloud_


"""
point
"""
cdef extern from "point.hpp":
    cdef cppclass Point[T]:
        Point() except +
        Point(const vector[Vec3[T]] & loc,
              const Vec3[T] & speed,
              const vector[T] & rcs,
              const vector[T] & phs) except +

"""
transmitter
"""
cdef extern from "transmitter.hpp":
    cdef cppclass TxChannel[T]:
        TxChannel() except +

        """
        * Constructor of TxChannel
        *
        * @param[in] location location of the channel [x, y, z] (m)
        * @param[in] polar polarization of the channel [x, y, z]
        * @param[in] phi phi angles of the channel's radiation pattern (rad)
        *                angles must be equal-spaced incremental array
        * @param[in] phi_ptn normalized radiation pattern along phi (dB)
        * @param[in] theta theta angles of the channel's radiation pattern (rad)
        *                  angles must be equal-spaced incremental array
        * @param[in] theta_ptn normalized radiation pattern along theta (dB)
        * @param[in] antenna_gain antenna gain (dB)
        * @param[in] mod_t timestamp of the modulation data (s)
        *                  mod_t must be equal-spaced incremental array
        * @param[in] mod_var modulation value vector
        * @param[in] pulse_mod pulse modulation vector
        *                      the length should be the same
        *                      as the number of pulses defined in Transmitter
        * @param[in] delay transmitting delay (s)
        * @param[in] grid ray occupancy checking grid (rad)
        """
        TxChannel(Vec3[T] location,
                  Vec3[T] polar,
                  vector[T] phi,
                  vector[T] phi_ptn,
                  vector[T] theta,
                  vector[T] theta_ptn,
                  T antenna_gain,
                  vector[T] mod_t,
                  vector[cpp_complex[T]] mod_var,
                  vector[cpp_complex[T]] pulse_mod,
                  T delay,
                  T grid) except +

    cdef cppclass Transmitter[T]:
        Transmitter() except +

        """
        * Constructor of Transmitter without phase noise
        *
        * @param[in] freq frequency vector (Hz)
        * @param[in] freq_offset frequency offset per pulse (Hz)
        *                        length should equal to the number of pulses
        * @param[in] freq_time timestamp vector for the frequency vector (s)
        *                      freq_time should be equal-spaced incremental array
        * @param[in] tx_power transmitter std::power (dBm)
        * @param[in] pulse_start_time pulse start time vector (s)
        *                             length should equal to the number of pulses
        * @param[in] frame_start_time frame start time vector (s)
        *                             length should equal to the number of frames
        * @param[in] density ray density, number of rays per wavelength
        """
        Transmitter(vector[double] freq,
                    vector[double] freq_offset,
                    vector[double] freq_time,
                    T tx_power,
                    vector[double] pulse_start_time,
                    vector[double] frame_start_time,
                    T density) except +

        """
        * Constructor of Transmitter with phase noise
        *
        * @param[in] freq frequency vector (Hz)
        * @param[in] freq_offset frequency offset per pulse (Hz)
        *                        length should equal to the number of pulses
        * @param[in] freq_time timestamp vector for the frequency vector (s)
        *                      freq_time should be an equal-spaced incremental array
        * @param[in] tx_power transmitter std::power (dBm)
        * @param[in] pulse_start_time pulse start time vector (s)
        *                             length should equal to the number of pulses
        * @param[in] frame_start_time frame start time vector (s)
        *                             length should equal to the number of frames
        * @param[in] density ray density, number of rays per wavelength
        * @param[in] phase_noise phase noise vector
        *                        the size should be the same as the full baseband time matrix
        """
        Transmitter(vector[double] freq,
                    vector[double] freq_offset,
                    vector[double] freq_time,
                    T tx_power,
                    vector[double] pulse_start_time,
                    vector[double] frame_start_time,
                    T density,
                    vector[cpp_complex[double]] phase_noise) except +

        """
        * AddChannel
        *  add a channel and calcuate transmitter's dimension
        *
        * @param[in] channel transmitter channel
        """
        void AddChannel(const TxChannel[T] & channel)


"""
receiver
"""
cdef extern from "receiver.hpp":
    cdef cppclass RxChannel[T]:
        RxChannel() except +
        RxChannel(Vec3[T] loc,
                  Vec3[T] pol,
                  vector[T] phi,
                  vector[T] phi_ptn,
                  vector[T] theta,
                  vector[T] theta_ptn,
                  T antenna_gain) except +

    cdef cppclass Receiver[T]:
        Receiver() except +
        Receiver(T fs,
                 T rf_gain,
                 T resistor,
                 T baseband_gain,
                 int samples) except +
        void AddChannel(const RxChannel[T] & channel)


"""
radar
"""
cdef extern from "radar.hpp":
    cdef cppclass Radar[T]:
        Radar() except +
        Radar(const Transmitter[T] & tx,
              const Receiver[T] & rx) except +

        void SetMotion(vector[Vec3[T]] location_array,
                       vector[Vec3[T]] speed_array,
                       vector[Vec3[T]] rotation_array,
                       vector[Vec3[T]] rotation_rate_array)

"""
snapshot
"""
cdef extern from "snapshot.hpp":
    cdef cppclass Snapshot[T]:
        Snapshot() except +
        Snapshot(double time,
                 int frame_idx,
                 int ch_idx,
                 int pulse_idx,
                 int sample_idx) except +
        double time_
        int sample_idx_
        int pulse_idx_
        int ch_idx_


"""
simulator
"""
cdef extern from "simulator.hpp":
    cdef cppclass Simulator[T]:
        Simulator() except +
        void Run(Radar[T] radar,
                 vector[Point[T]] points,
                 double * bb_real,
                 double * bb_imag)


"""
scene interface
"""
cdef extern from "scene.hpp":
    cdef cppclass Scene[T, F]:
        Scene() except +

        void AddTarget(const Target[F] & mesh)
        void SetRadar(const Radar[F] & radar)
        void RunSimulator(int level,
                          bool debug,
                          vector[Snapshot[F]] & snapshots,
                          double * bb_real,
                          double * bb_imag)
