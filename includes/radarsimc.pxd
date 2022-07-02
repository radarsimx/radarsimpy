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


cdef extern from "simpleray.hpp":
    cdef cppclass SimpleRay[T]:
        SimpleRay() except +
        Vec3[T] * direction_
        Vec3[T] * location_
        int reflections_


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

        vector[SimpleRay[T]] cloud_


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
        TxChannel(const Vec3[T] & location,
                  const Vec3[T] & polar,
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

    cdef cppclass Transmitter[T]:
        Transmitter() except +
        Transmitter(const vector[double] & freq,
                    const vector[double] & freq_offset,
                    const vector[double] & freq_time,
                    const T & tx_power,
                    const vector[double] & pulse_start_time,
                    const vector[double] & frame_start_time,
                    const T & density) except +
        Transmitter(const vector[double] & freq,
                    const vector[double] & freq_offset,
                    const vector[double] & freq_time,
                    const T & tx_power,
                    const vector[double] & pulse_start_time,
                    const vector[double] & frame_start_time,
                    const T & density,
                    const vector[cpp_complex[double]] & phase_noise) except +
        void AddChannel(const TxChannel[T] & channel)


"""
receiver
"""
cdef extern from "receiver.hpp":
    cdef cppclass RxChannel[T]:
        RxChannel() except +
        RxChannel(const Vec3[T] & location,
                  const Vec3[T] & polar,
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
                 const int & samples) except +
        void AddChannel(const RxChannel[T] & channel)


"""
radar
"""
cdef extern from "radar.hpp":
    cdef cppclass Radar[T]:
        Radar() except +
        Radar(const Transmitter[T] & tx,
              const Receiver[T] & rx) except +

        void SetMotion(const vector[Vec3[T]] & location_array,
                       const vector[Vec3[T]] & speed_array,
                       const vector[Vec3[T]] & rotation_array,
                       const vector[Vec3[T]] & rotrate_array)


"""
snapshot
"""
cdef extern from "snapshot.hpp":
    cdef cppclass Snapshot[T]:
        Snapshot() except +
        Snapshot(const double & time,
                 const int & frame_idx,
                 const int & ch_idx,
                 const int & pulse_idx,
                 const int & sample_idx) except +


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
