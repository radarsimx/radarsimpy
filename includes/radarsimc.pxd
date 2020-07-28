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

from radarsimpy.includes.type_def cimport vector, uint64_t
from radarsimpy.includes.zpvector cimport Vec3, Vec2
from libcpp cimport bool
from libcpp cimport complex

"""
radarsimc classes
"""
cdef extern from "radarsimc_global.hpp":
    # PathPy
    #  Ray's path
    cdef cppclass PathPy[T]:
        Vec3[T] dir_
        Vec3[T] loc_

    # RayPy
    #  Ray's properties
    cdef cppclass RayPy[T]:
        Vec3[T] dir_
        Vec3[T] loc_
        Vec3[T] pol_
        T range_
        T range_rate
        T area
        int refCount
        vector[PathPy[T]] path
    
cdef extern from "snapshot.hpp":
    # Snapshot
    #  Scene's snapshot
    cdef cppclass Snapshot[T]:
        T time_
        int sample_idx_
        int pulse_idx_
        int ch_idx_
        vector[RayPy[T]] ray_received

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


"""
scene interface
"""
cdef extern from "scene.hpp":
    cdef cppclass Scene[T]:
        Scene() except +

        void AddTarget(const Target[T]& mesh)


"""
radarsimc interface
"""
cdef extern from "radarsimc.hpp":
    cdef cppclass Radarsimc[T]:
        Radarsimc() except +

        # Target's RCS simulation with ray tracing
        T TargetRcs(T *mesh,
                    int mesh_size,
                    T phi,
                    T theta,
                    Vec3[T] pol,
                    T f,
                    T density)

        # LiDAR point cloud simulation
        void AddLidarTarget(T *mesh,
                            int mesh_size,
                            Vec3[T] origin,
                            Vec3[T] location,
                            Vec3[T] speed,
                            Vec3[T] rotation,
                            Vec3[T] rotation_rate,
                            bool is_ground)

        void LidarScene(Vec3[T] location,
                        vector[T] phi,
                        vector[T] theta,
                        vector[RayPy[T]] &ray_received)

        # Radar scene simulation for target's models
        void AddSnapshot(T time,
                         int frame_idx,
                         int tx_idx,
                         int pulse_idx,
                         int sample_idx,
                         vector[Snapshot[T]] &snapshots)

        void AddSceneTarget(T *mesh,
                            int mesh_size,
                            Vec3[T] origin,
                            vector[Vec3[T]] location_array,
                            vector[Vec3[T]] speed_array,
                            vector[Vec3[T]] rotation_array,
                            vector[Vec3[T]] rotation_rate_array,
                            bool is_ground)

        void SetSceneApertureMesh(T *aperture,
                                  int size)

        void SetSceneAperture(T phi,
                              T theta,
                              Vec3[T] location,
                              T *extension)

        void SetSceneTransmitter(vector[T] fc,
                                 T slope,
                                 T tx_power,
                                 vector[T] pulse_start_time,
                                 vector[T] frame_time,
                                 int frames,
                                 int pulses,
                                 T density)

        void AddSceneTxChannel(Vec3[T] location,
                               Vec3[T] polarization,
                               vector[T] mod_amp,
                               vector[T] mod_phs,
                               T chip_length,
                               vector[T] phi,
                               vector[T] phi_ptn,
                               vector[T] theta,
                               vector[T] theta_ptn,
                               T antenna_gain,
                               T delay,
                               T grid)

        void SetSceneReceiver(T fs,
                              T rf_gain,
                              T resistor,
                              T baseband_gain,
                              int samples)

        void AddSceneRxChannel(Vec3[T] location,
                               Vec3[T] polarization,
                               vector[T] az_angle,
                               vector[T] az,
                               vector[T] el_angle,
                               vector[T] el,
                               T antenna_gain)

        void RadarScene(int level,
                        vector[Snapshot[T]] &snapshots,
                        T correction,
                        T *baseband_re,
                        T *baseband_im)

        # Radar baseband simulation for point targets
        void AddRadarTarget(vector[Vec3[T]] location,
                            Vec3[T] speed,
                            vector[T] rcs,
                            vector[T] phs)

        void SetRadarTransmitter(vector[T] fc,
                                 T slope,
                                 T tx_power,
                                 vector[T] pulse_start_time,
                                 vector[T] frame_time,
                                 int frames,
                                 int pulses)

        void AddRadarTxChannel(Vec3[T] location, 
                               Vec3[T] polarization, 
                               vector[T] mod_amp,
                               vector[T] mod_phs,
                               T chip_length, 
                               vector[T] phi,
                               vector[T] phi_ptn,
                               vector[T] theta,
                               vector[T] theta_ptn,
                               T antenna_gain,
                               T delay)

        void SetRadarReceiver(T fs,
                              T rf_gain,
                              T resistor,
                              T baseband_gain,
                              int samples)

        void AddRadarRxChannel(Vec3[T] location, 
                               Vec3[T] polarization, 
                               vector[T] phi,
                               vector[T] phi_ptn,
                               vector[T] theta,
                               vector[T] theta_ptn,
                               T antenna_gain)

        void RadarSimulator(T *baseband_re,
                            T *baseband_im)
