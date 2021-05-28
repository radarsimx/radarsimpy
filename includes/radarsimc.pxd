"""
    A Python module for radar simulation

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - 2021  Zhengyu Peng
    E-mail: zpeng.me@gmail.com
    Website: https://zpeng.me

    `                      `
    -:.                  -#:
    -//:.              -###:
    -////:.          -#####:
    -/:.://:.      -###++##:
    ..   `://:-  -###+. :##:
           `:/+####+.   :##:
    .::::::::/+###.     :##:
    .////-----+##:    `:###:
     `-//:.   :##:  `:###/.
       `-//:. :##:`:###/.
         `-//:+######/.
           `-/+####/.
             `+##+.
              :##:
              :##:
              :##:
              :##:
              :##:
               .+:

"""
import numpy as np
cimport numpy as np

from radarsimpy.includes.type_def cimport vector
from radarsimpy.includes.type_def cimport uint64_t, float_t, int_t
from radarsimpy.includes.zpvector cimport Vec3
from libcpp cimport bool
from libcpp.complex cimport complex as cpp_complex


"""
target interface
"""
cdef extern from "target.hpp":
    cdef cppclass Target[T]:
        Target() except +
        Target(T *points,
               uint64_t* cells,
               int cell_size,
               Vec3[T] origin,
               vector[Vec3[T]] location_array,
               vector[Vec3[T]] speed_array,
               vector[Vec3[T]] rotation_array,
               vector[Vec3[T]] rotation_rate_array,
               bool is_ground) except +
        Target(T* points,
               uint64_t* cells,
               int cell_size) except +
        Target(T* points,
               uint64_t* cells,
               int cell_size,
               Vec3[T] origin,
               Vec3[T] location,
               Vec3[T] speed,
               Vec3[T] rotation,
               Vec3[T] rotation_rate,
               bool is_ground) except +


cdef extern from "ray.hpp":
    cdef cppclass PathNode[T]:
        Vec3[T] dir_
        Vec3[T] loc_

    cdef cppclass Ray[T, Tg=*]:
        Ray() except +
        Vec3[T] *dir_
        Vec3[T] *loc_
        Vec3[T] *pol_
        Vec3[T] *norm_
        T *range_
        T *range_rate_
        int ref_count_
        T d_theta_
        T d_phi_
        Vec3[cpp_complex[T]] E_


cdef extern from "raypool.hpp":
    cdef cppclass RayPool[T, Tg=*]:
        RayPool() except +
        vector[Ray[T]] pool_


"""
rcs
"""
cdef extern from "rcs.hpp":
    cdef cppclass Rcs[T]:
        Rcs() except +
        Rcs(const Target[T]& mesh,
            const Vec3[T]& inc_dir,
            const Vec3[T]& obs_dir,
            const Vec3[T]& polarization,
            const T& frequency,
            const T& density) except +

        T CalculateRcs()


"""
pointcloud
"""
cdef extern from "pointcloud.hpp":
    cdef cppclass PointCloud[T]:
        PointCloud() except +
        void AddTarget(const Target[T]& target)
        void Sbr(const vector[T]& phi,
                 const vector[T]& theta,
                 const Vec3[T]& position)
        
        vector[Ray[T]] cloud_


"""
point
"""
cdef extern from "point.hpp":
    cdef cppclass Point[T]:
        Point() except +
        Point(const vector[Vec3[T]]& loc,
              const Vec3[T]& speed,
              const vector[T]& rcs,
              const vector[T]& phs) except +


"""
transmitter
"""
cdef extern from "transmitter.hpp":
    cdef cppclass TxChannel[T]:
        TxChannel() except +
        TxChannel(Vec3[T] loc,
                  Vec3[T] pol,
                  vector[cpp_complex[T]] pulse_mod,
                  bool mod_enabled,
                  vector[T] mod_t,
                  vector[cpp_complex[T]] mod_var,
                  vector[T] phi,
                  vector[T] phi_ptn,
                  vector[T] theta,
                  vector[T] theta_ptn,
                  T antenna_gain,
                  T delay,
                  T grid) except +

    cdef cppclass Transmitter[T]:
        Transmitter() except +
        Transmitter(T fc,
                    vector[T] freq,
                    vector[T] freq_offset,
                    vector[T] pulse_timing,
                    T tx_power,
                    vector[T] pulse_start_time,
                    vector[T] frame_time,
                    int frames,
                    int pulses,
                    T density) except +
        Transmitter(T fc,
                    vector[T] freq,
                    vector[T] freq_offset,
                    vector[T] pulse_timing,
                    T tx_power,
                    vector[T] pulse_start_time,
                    vector[T] frame_time,
                    int frames,
                    int pulses,
                    T density,
                    vector[cpp_complex[T]]) except +
        void AddChannel(const TxChannel[T]& channel)


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
        void AddChannel(const RxChannel[T]& channel) 


"""
snapshot
"""    
cdef extern from "snapshot.hpp":
    cdef cppclass Snapshot[T]:
        Snapshot() except +
        Snapshot(T time,
                 int frame_idx,
                 int ch_idx,
                 int pulse_idx,
                 int sample_idx,
                 size_t tx_size,
                 size_t rx_size) except +
        T time_
        int sample_idx_
        int pulse_idx_
        int ch_idx_
        vector[Ray[T]] ray_received

"""
simulator
"""
cdef extern from "simulator.hpp":
    cdef cppclass Simulator[T]:
        Simulator() except +
        void Run(Transmitter[T] tx,
                 Receiver[T] rx,
                 vector[Point[T]] points,
                 vector[cpp_complex[T]]& baseband)
        

"""
scene interface
"""
cdef extern from "scene.hpp":
    cdef cppclass Scene[T]:
        Scene() except +

        void AddTarget(const Target[T]& mesh)
        void SetTransmitter(const Transmitter[T]& tx)
        void AddTxChannel(const TxChannel[T]& channel)
        void SetReceiver(const Receiver[T]& rx)
        void AddRxChannel(const RxChannel[T]& channel)
        void RunSimulator(int level,
                          bool debug,
                          vector[Snapshot[T]]& snapshots,
                          vector[cpp_complex[T]]& baseband)

        vector[Snapshot[T]] snapshots_
