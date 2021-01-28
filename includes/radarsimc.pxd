"""
    A Python module for radar simulation

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - 2020  Zhengyu Peng
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
        Target(T *mesh,
               int mesh_size,
               Vec3[T] origin,
               vector[Vec3[T]] location_array,
               vector[Vec3[T]] speed_array,
               vector[Vec3[T]] rotation_array,
               vector[Vec3[T]] rotation_rate_array,
               bool is_ground) except +
        Target(T* mesh,
               int mesh_size) except +
        Target(T* mesh,
               int mesh_size,
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
        Vec3[T] dir_
        Vec3[T] loc_
        Vec3[T] pol_
        T range_
        T range_rate_
        T area_
        int ref_count_
        vector[PathNode[T]] path_


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
            const T& phi,
            const T& theta,
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


cdef inline Point[float_t] cp_Point(location, speed, rcs, phase, shape):
    cdef vector[Vec3[float_t]] loc_vect
    cdef vector[float_t] rcs_vect
    cdef vector[float_t] phs_vect

    if np.size(location[0]) > 1 or \
        np.size(location[1])  > 1 or \
        np.size(location[2]) > 1 or \
        np.size(rcs) > 1 or \
        np.size(phase) > 1:

        if np.size(location[0]) > 1:
            tgx_t = location[0]
        else:
            tgx_t = np.full(shape, location[0])

        if np.size(location[1]) > 1:
            tgy_t = location[1]
        else:
            tgy_t = np.full(shape, location[1])
        
        if np.size(location[2]) > 1:
            tgz_t = location[2]
        else:
            tgz_t = np.full(shape, location[2])

        if np.size(rcs) > 1:
            rcs_t = rcs
        else:
            rcs_t = np.full(shape, rcs)
        
        if np.size(phase) > 1:
            phs_t = phase
        else:
            phs_t = np.full(shape, phase)

        for ch_idx in range(0, shape[0]):
            for ps_idx in range(0, shape[1]):
                for sp_idx in range(0, shape[2]):
                    loc_vect.push_back(Vec3[float_t](
                        <float_t> tgx_t[ch_idx, ps_idx, sp_idx],
                        <float_t> tgy_t[ch_idx, ps_idx, sp_idx],
                        <float_t> tgz_t[ch_idx, ps_idx, sp_idx]
                    ))
                    rcs_vect.push_back(<float_t> rcs_t[ch_idx, ps_idx, sp_idx])
                    phs_vect.push_back(<float_t> (phs_t[ch_idx, ps_idx, sp_idx]/180*np.pi))
    else:
        loc_vect.push_back(Vec3[float_t](
            <float_t> location[0],
            <float_t> location[1],
            <float_t> location[2]
        ))
        rcs_vect.push_back(<float_t> rcs)
        phs_vect.push_back(<float_t> (phase/180*np.pi))
    return Point[float_t](
                loc_vect,
                Vec3[float_t](
                    <float_t> speed[0],
                    <float_t> speed[1],
                    <float_t> speed[2]
                ),
                rcs_vect,
                phs_vect
            )


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


cdef inline TxChannel[float_t] cp_TxChannel(tx, tx_idx):
    cdef int_t pulses = tx.pulses

    cdef vector[float_t] az_ang_vect, az_ptn_vect
    # cdef float_t[:] az_ang_mem, az_ptn_mem
    cdef vector[float_t] el_ang_vect, el_ptn_vect
    # cdef float_t[:] el_ang_mem, el_ptn_mem

    cdef vector[cpp_complex[float_t]] pulse_mod_vect

    cdef bool mod_enabled
    cdef vector[cpp_complex[float_t]] mod_var_vect
    cdef vector[float_t] mod_t_vect
    # cdef float_t[:] mod_t_mem

    # az_ang_mem = tx.az_angles[tx_idx].astype(np.float64)/180*np.pi
    # az_ptn_mem = tx.az_patterns[tx_idx].astype(np.float64)
    az_ang_vect.reserve(len(tx.az_angles[tx_idx]))
    for idx in range(0, len(tx.az_angles[tx_idx])):
        az_ang_vect.push_back(<float_t> (tx.az_angles[tx_idx][idx]/180*np.pi))
    az_ptn_vect.reserve(len(tx.az_patterns[tx_idx]))
    for idx in range(0, len(tx.az_patterns[tx_idx])):
        az_ptn_vect.push_back(<float_t> (tx.az_patterns[tx_idx][idx]))

    el_ang_mem = np.flip(90-tx.el_angles[tx_idx].astype(np.float64))/180*np.pi
    el_ptn_mem = np.flip(tx.el_patterns[tx_idx].astype(np.float64))
    el_ang_vect.reserve(len(tx.el_angles[tx_idx]))
    for idx in range(0, len(tx.el_angles[tx_idx])):
        el_ang_vect.push_back(el_ang_mem[idx])
    el_ptn_vect.reserve(len(tx.el_patterns[tx_idx]))
    for idx in range(0, len(tx.el_patterns[tx_idx])):
        el_ptn_vect.push_back(el_ptn_mem[idx])

    for idx in range(0, pulses):
        pulse_mod_vect.push_back(cpp_complex[float_t](np.real(tx.pulse_mod[tx_idx, idx]), np.imag(tx.pulse_mod[tx_idx, idx])))

    mod_enabled = tx.mod[tx_idx]['enabled']
    if mod_enabled:
        for idx in range(0, len(tx.mod[tx_idx]['var'])):
            mod_var_vect.push_back(cpp_complex[float_t](np.real(tx.mod[tx_idx]['var'][idx]), np.imag(tx.mod[tx_idx]['var'][idx])))

        mod_t_mem = tx.mod[tx_idx]['t'].astype(np.float64)
        mod_t_vect.reserve(len(tx.mod[tx_idx]['t']))
        for idx in range(0, len(tx.mod[tx_idx]['t'])):
            mod_t_vect.push_back(mod_t_mem[idx])
    
    return TxChannel[float_t](
        Vec3[float_t](
            <float_t> tx.locations[tx_idx, 0],
            <float_t> tx.locations[tx_idx, 1],
            <float_t> tx.locations[tx_idx, 2]
            ),
        Vec3[float_t](
            <float_t> tx.polarization[tx_idx, 0],
            <float_t> tx.polarization[tx_idx, 1],
            <float_t> tx.polarization[tx_idx, 2]
            ),
        pulse_mod_vect,
        mod_enabled,
        mod_t_vect,
        mod_var_vect,
        az_ang_vect,
        az_ptn_vect,
        el_ang_vect,
        el_ptn_vect,
        <float_t> tx.antenna_gains[tx_idx],
        <float_t> tx.delay[tx_idx],
        0.0
        )


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
aperture
"""
cdef extern from "aperture.hpp":
    cdef cppclass Aperture[T]:
        Aperture() except +
        Aperture(const T& phi,
                 const T& theta,
                 const Vec3[T]& location,
                 T* extension) except +
        Aperture(T* aperture, int size) except +

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
                 int sample_idx) except +
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
        void SetAperture(const Aperture[T]& aperture)
        void SetTransmitter(const Transmitter[T]& tx)
        void AddTxChannel(const TxChannel[T]& channel)
        void SetReceiver(const Receiver[T]& rx)
        void AddRxChannel(const RxChannel[T]& channel)
        void RunSimulator(int level,
                          T correction,
                          vector[Snapshot[T]]& snapshots,
                          vector[cpp_complex[T]]& baseband)

        vector[Snapshot[T]] snapshots_
