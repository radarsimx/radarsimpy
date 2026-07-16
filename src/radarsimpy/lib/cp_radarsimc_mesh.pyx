# distutils: language = c++
"""
Mesh-target wrappers for cp_radarsimc.

This file is textually merged into the ``cp_radarsimc`` extension module via
``include`` (see cp_radarsimc.pyx). It converts Python mesh-target
configurations into the C++ targets manager representation for radar
simulation and RCS calculation, and provides helpers to retrieve transformed
meshes and scene state.

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

# Standard imports
import numpy as np

# Cython imports
cimport cython
cimport numpy as np
from libcpp.complex cimport complex as cpp_complex
from libcpp cimport bool
from radarsimpy.includes.radarsimc cimport (
    TargetsManager,
    Mem_Copy_Vec3,
    Target, Triangle, Rotate,
)
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t, float_t, vector


# ============================================================================
# Mesh Targets (Radar Simulation and RCS)
# ============================================================================

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_AddTarget(radar,
                  target,
                  timestamp,
                  mesh_module,
                  TargetsManager[float_t] * targets_manager):
    """
    Add a complex target to the radar simulation.

    :param Radar radar:
        Radar object containing system configuration
    :param dict target:
        Target properties (model, location, speed, rotation, materials, etc.)
    :param timestamp:
        Time array for simulation frames
    :param mesh_module:
        Mesh loading module for processing 3D models
    :param TargetsManager[float_t] * targets_manager:
        Pointer to C++ targets manager
    :raises: ValueError for invalid target config, RuntimeError for mesh issues
    """
    # vector of location, speed, rotation, rotation rate
    cdef vector[Vec3[float_t]] loc_vt
    cdef vector[Vec3[float_t]] spd_vt
    cdef vector[Vec3[float_t]] rot_vt
    cdef vector[Vec3[float_t]] rrt_vt

    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] spdx_mv, spdy_mv, spdz_mv
    cdef float_t[:, :, :] rotx_mv, roty_mv, rotz_mv
    cdef float_t[:, :, :] rrtx_mv, rrty_mv, rrtz_mv

    cdef cpp_complex[float_t] ep_c, mu_c

    ts_shape = np.shape(timestamp)
    cdef int_t bbsize_c = <int_t>(ts_shape[0] * ts_shape[1] * ts_shape[2])

    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv

    _validate_target_keys(target)

    # Load and validate mesh
    points_arr, cells_arr = _load_and_validate_mesh(target, mesh_module)
    points_mv = points_arr
    cells_mv = cells_arr

    cdef float_t[:] origin_mv = np.asarray(target.get("origin", (0, 0, 0)), dtype=np_float)

    location = list(target.get("location", [0, 0, 0]))
    speed = list(target.get("speed", [0, 0, 0]))
    rotation = list(target.get("rotation", [0, 0, 0]))
    rotation_rate = list(target.get("rotation_rate", [0, 0, 0]))

    cdef float_t[:] location_mv, speed_mv, rotation_mv, rotation_rate_mv

    _parse_material_properties(target, &ep_c, &mu_c)

    if any(np.size(var) > 1 for var in location + speed + rotation + rotation_rate):
        if np.size(location[0]) > 1:
            locx_mv = location[0].astype(np_float)
        else:
            locx_mv = (location[0] + speed[0]*timestamp).astype(np_float)

        if np.size(location[1]) > 1:
            locy_mv = location[1].astype(np_float)
        else:
            locy_mv = (location[1] + speed[1]*timestamp).astype(np_float)

        if np.size(location[2]) > 1:
            locz_mv = location[2].astype(np_float)
        else:
            locz_mv = (location[2] + speed[2]*timestamp).astype(np_float)

        if np.size(speed[0]) > 1:
            spdx_mv = speed[0].astype(np_float)
        else:
            spdx_mv = np.full(ts_shape, speed[0], dtype=np_float)

        if np.size(speed[1]) > 1:
            spdy_mv = speed[1].astype(np_float)
        else:
            spdy_mv = np.full(ts_shape, speed[1], dtype=np_float)

        if np.size(speed[2]) > 1:
            spdz_mv = speed[2].astype(np_float)
        else:
            spdz_mv = np.full(ts_shape, speed[2], dtype=np_float)

        if np.size(rotation[0]) > 1:
            rotx_mv = np.radians(rotation[0]).astype(np_float)
        else:
            rotx_mv = np.radians(
                rotation[0] + rotation_rate[0]*timestamp).astype(np_float)

        if np.size(rotation[1]) > 1:
            roty_mv = np.radians(rotation[1]).astype(np_float)
        else:
            roty_mv = np.radians(
                rotation[1] + rotation_rate[1]*timestamp).astype(np_float)

        if np.size(rotation[2]) > 1:
            rotz_mv = np.radians(rotation[2]).astype(np_float)
        else:
            rotz_mv = np.radians(
                rotation[2] + rotation_rate[2]*timestamp).astype(np_float)

        if np.size(rotation_rate[0]) > 1:
            rrtx_mv = np.radians(rotation_rate[0]).astype(np_float)
        else:
            rrtx_mv = np.full(ts_shape, np.radians(rotation_rate[0]), dtype=np_float)

        if np.size(rotation_rate[1]) > 1:
            rrty_mv = np.radians(rotation_rate[1]).astype(np_float)
        else:
            rrty_mv = np.full(ts_shape, np.radians(rotation_rate[1]), dtype=np_float)

        if np.size(rotation_rate[2]) > 1:
            rrtz_mv = np.radians(rotation_rate[2]).astype(np_float)
        else:
            rrtz_mv = np.full(ts_shape, np.radians(rotation_rate[2]), dtype=np_float)

        Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)
        Mem_Copy_Vec3(&spdx_mv[0,0,0], &spdy_mv[0,0,0], &spdz_mv[0,0,0], bbsize_c, spd_vt)
        Mem_Copy_Vec3(&rotx_mv[0,0,0], &roty_mv[0,0,0], &rotz_mv[0,0,0], bbsize_c, rot_vt)
        Mem_Copy_Vec3(&rrtx_mv[0,0,0], &rrty_mv[0,0,0], &rrtz_mv[0,0,0], bbsize_c, rrt_vt)

    else:
        location_mv = np.array(location, dtype=np_float)
        loc_vt.push_back(Vec3[float_t](&location_mv[0]))

        speed_mv = np.array(speed, dtype=np_float)
        spd_vt.push_back(Vec3[float_t](&speed_mv[0]))

        rotation_mv = np.radians(np.array(rotation, dtype=np_float)).astype(np_float)
        rot_vt.push_back(Vec3[float_t](&rotation_mv[0]))

        rotation_rate_mv = np.radians(np.array(rotation_rate, dtype=np_float)).astype(np_float)
        rrt_vt.push_back(Vec3[float_t](&rotation_rate_mv[0]))

    _handle_deprecated_target_params(target)

    targets_manager[0].AddTarget(&points_mv[0, 0],
                           &cells_mv[0, 0],
                           <int_t> cells_mv.shape[0],
                           Vec3[float_t](&origin_mv[0]),
                           loc_vt,
                           spd_vt,
                           rot_vt,
                           rrt_vt,
                           ep_c,
                           mu_c,
                           <bool> target.get("skip_diffusion", False),
                           <float_t> target.get("density", 0.0),
                           <bool> target.get("environment", False))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void cp_RCS_Target(target, mesh_module, TargetsManager[float_t] * targets_manager):
    """
    Create Target object in Cython for RCS calculation.

    :param dict target:
        Target properties dictionary containing model, location, materials
    :param mesh_module:
        Mesh loading module for processing 3D models
    :param TargetsManager[float_t] * targets_manager:
        Pointer to C++ targets manager
    :raises: RuntimeError on mesh limitations, ValueError on invalid target
    """
    # Vector declarations
    cdef vector[Vec3[float_t]] loc_vt, spd_vt, rot_vt, rrt_vt
    cdef cpp_complex[float_t] ep_c, mu_c
    cdef float_t[:, :] points_mv
    cdef int_t[:, :] cells_mv

    _validate_target_keys(target)

    # Load and validate mesh
    points_arr, cells_arr = _load_and_validate_mesh(target, mesh_module)
    points_mv = points_arr
    cells_mv = cells_arr

    cdef float_t[:] origin_mv = np.asarray(target.get("origin", (0, 0, 0)), dtype=np_float)

    location = np.asarray(target.get("location", (0, 0, 0)), dtype=object)
    speed = np.asarray(target.get("speed", (0, 0, 0)), dtype=object)
    rotation = np.asarray(target.get("rotation", (0, 0, 0)), dtype=object)
    rotation_rate = np.asarray(target.get("rotation_rate", (0, 0, 0)), dtype=object)

    cdef float_t[:] location_mv, speed_mv, rotation_mv, rotation_rate_mv

    _parse_material_properties(target, &ep_c, &mu_c)

    location_mv = location.astype(np_float)
    loc_vt.push_back(Vec3[float_t](&location_mv[0]))

    speed_mv = speed.astype(np_float)
    spd_vt.push_back(Vec3[float_t](&speed_mv[0]))

    rotation_mv = np.radians(rotation.astype(np_float)).astype(np_float)
    rot_vt.push_back(Vec3[float_t](&rotation_mv[0]))

    rotation_rate_mv = np.radians(rotation_rate.astype(np_float)).astype(np_float)
    rrt_vt.push_back(Vec3[float_t](&rotation_rate_mv[0]))

    _handle_deprecated_target_params(target)

    targets_manager[0].AddTarget(&points_mv[0, 0],
                           &cells_mv[0, 0],
                           <int_t> cells_mv.shape[0],
                           Vec3[float_t](&origin_mv[0]),
                           loc_vt,
                           spd_vt,
                           rot_vt,
                           rrt_vt,
                           ep_c,
                           mu_c,
                           <bool> target.get("skip_diffusion", False),
                           <float_t> target.get("density", 0.0),
                           <bool> target.get("environment", False))

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def cp_GetTargetMesh(target, timestamp, mesh_module, sim_timestamp=None):
    """
    Get transformed target mesh at query timestamps using C++ Target directly.
    """
    cdef float_t scale = _safe_unit_conversion(target.get("unit", "m"))
    points_arr, cells_arr = _load_and_validate_mesh(target, mesh_module)

    cdef float_t[:, :] points_mv = points_arr
    cdef int_t[:, :] cells_mv = cells_arr
    cdef int_t cell_size = <int_t>cells_mv.shape[0]
    cdef int_t point_size = <int_t>points_mv.shape[0]

    cdef float_t[:] origin_mv = np.asarray(target.get("origin", (0, 0, 0)), dtype=np_float)

    location = list(target.get("location", [0, 0, 0]))
    speed = list(target.get("speed", [0, 0, 0]))
    rotation = list(target.get("rotation", [0, 0, 0]))
    rotation_rate = list(target.get("rotation_rate", [0, 0, 0]))

    cdef float_t[:] location_mv, speed_mv, rotation_mv, rotation_rate_mv
    cdef cpp_complex[float_t] ep_c, mu_c

    _parse_material_properties(target, &ep_c, &mu_c)

    cdef Target[float_t] *target_c = NULL

    cdef vector[Vec3[float_t]] loc_vt
    cdef vector[Vec3[float_t]] spd_vt
    cdef vector[Vec3[float_t]] rot_vt
    cdef vector[Vec3[float_t]] rrt_vt

    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] spdx_mv, spdy_mv, spdz_mv
    cdef float_t[:, :, :] rotx_mv, roty_mv, rotz_mv
    cdef float_t[:, :, :] rrtx_mv, rrty_mv, rrtz_mv

    cdef int_t bbsize_c = 0
    cdef int_t sim_idx

    cdef double[:] timestamp_arr = np.atleast_1d(timestamp).astype(np.float64)
    cdef int_t K = <int_t>timestamp_arr.shape[0]

    cdef object ts = sim_timestamp if sim_timestamp is not None else timestamp
    ts_shape = np.shape(ts)

    if any(np.size(var) > 1 for var in location + speed + rotation + rotation_rate):
        bbsize_c = <int_t>(ts_shape[0] * ts_shape[1] * ts_shape[2])
        if np.size(location[0]) > 1:
            locx_mv = location[0].astype(np_float)
        else:
            locx_mv = (location[0] + speed[0]*ts).astype(np_float)

        if np.size(location[1]) > 1:
            locy_mv = location[1].astype(np_float)
        else:
            locy_mv = (location[1] + speed[1]*ts).astype(np_float)

        if np.size(location[2]) > 1:
            locz_mv = location[2].astype(np_float)
        else:
            locz_mv = (location[2] + speed[2]*ts).astype(np_float)

        if np.size(speed[0]) > 1:
            spdx_mv = speed[0].astype(np_float)
        else:
            spdx_mv = np.full(ts_shape, speed[0], dtype=np_float)

        if np.size(speed[1]) > 1:
            spdy_mv = speed[1].astype(np_float)
        else:
            spdy_mv = np.full(ts_shape, speed[1], dtype=np_float)

        if np.size(speed[2]) > 1:
            spdz_mv = speed[2].astype(np_float)
        else:
            spdz_mv = np.full(ts_shape, speed[2], dtype=np_float)

        if np.size(rotation[0]) > 1:
            rotx_mv = np.radians(rotation[0]).astype(np_float)
        else:
            rotx_mv = np.radians(
                rotation[0] + rotation_rate[0]*ts).astype(np_float)

        if np.size(rotation[1]) > 1:
            roty_mv = np.radians(rotation[1]).astype(np_float)
        else:
            roty_mv = np.radians(
                rotation[1] + rotation_rate[1]*ts).astype(np_float)

        if np.size(rotation[2]) > 1:
            rotz_mv = np.radians(rotation[2]).astype(np_float)
        else:
            rotz_mv = np.radians(
                rotation[2] + rotation_rate[2]*ts).astype(np_float)

        if np.size(rotation_rate[0]) > 1:
            rrtx_mv = np.radians(rotation_rate[0]).astype(np_float)
        else:
            rrtx_mv = np.full(ts_shape, np.radians(rotation_rate[0]), dtype=np_float)

        if np.size(rotation_rate[1]) > 1:
            rrty_mv = np.radians(rotation_rate[1]).astype(np_float)
        else:
            rrty_mv = np.full(ts_shape, np.radians(rotation_rate[1]), dtype=np_float)

        if np.size(rotation_rate[2]) > 1:
            rrtz_mv = np.radians(rotation_rate[2]).astype(np_float)
        else:
            rrtz_mv = np.full(ts_shape, np.radians(rotation_rate[2]), dtype=np_float)

        Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)
        Mem_Copy_Vec3(&spdx_mv[0,0,0], &spdy_mv[0,0,0], &spdz_mv[0,0,0], bbsize_c, spd_vt)
        Mem_Copy_Vec3(&rotx_mv[0,0,0], &roty_mv[0,0,0], &rotz_mv[0,0,0], bbsize_c, rot_vt)
        Mem_Copy_Vec3(&rrtx_mv[0,0,0], &rrty_mv[0,0,0], &rrtz_mv[0,0,0], bbsize_c, rrt_vt)

        target_c = new Target[float_t](&points_mv[0, 0],
                                       &cells_mv[0, 0],
                                       cell_size,
                                       Vec3[float_t](&origin_mv[0]),
                                       loc_vt,
                                       spd_vt,
                                       rot_vt,
                                       rrt_vt,
                                       ep_c,
                                       mu_c,
                                       <bool> target.get("skip_diffusion", False),
                                       <float_t> target.get("density", 0.0),
                                       <bool> target.get("environment", False))
    else:
        location_mv = np.array(location, dtype=np_float)
        speed_mv = np.array(speed, dtype=np_float)
        rotation_mv = np.radians(np.array(rotation, dtype=np_float)).astype(np_float)
        rotation_rate_mv = np.radians(np.array(rotation_rate, dtype=np_float)).astype(np_float)

        target_c = new Target[float_t](&points_mv[0, 0],
                                       &cells_mv[0, 0],
                                       cell_size,
                                       Vec3[float_t](&origin_mv[0]),
                                       Vec3[float_t](&location_mv[0]),
                                       Vec3[float_t](&speed_mv[0]),
                                       Vec3[float_t](&rotation_mv[0]),
                                       Vec3[float_t](&rotation_rate_mv[0]),
                                       <bool> target.get("skip_diffusion", False),
                                       <float_t> target.get("density", 0.0),
                                       <bool> target.get("environment", False))

    points_out = np.zeros((K, point_size, 3), dtype=np.float64)
    cdef double[:, :, :] points_out_mv = points_out

    cdef int_t vtx1_idx, vtx2_idx, vtx3_idx
    cdef Triangle[float_t] trig
    cdef double q_time
    cdef double[:] sim_ts_flat

    if sim_timestamp is not None:
        sim_ts_flat = np.asarray(sim_timestamp).ravel().astype(np.float64)

    cdef int_t[:] sort_indices = np.argsort(timestamp_arr).astype(np.int32)

    cdef int_t idx, orig_idx, i

    try:
        for idx in range(K):
            orig_idx = sort_indices[idx]
            q_time = timestamp_arr[orig_idx]

            if target_c[0].array_size_ > 1:
                sim_idx = <int_t>np.argmin(np.abs(np.asarray(sim_ts_flat) - q_time))
                target_c[0].Move(sim_idx, q_time)
            else:
                target_c[0].Move(0, q_time)

            for i in range(cell_size):
                vtx1_idx = cells_mv[i, 0]
                vtx2_idx = cells_mv[i, 1]
                vtx3_idx = cells_mv[i, 2]
                trig = target_c[0].vect_mesh_[i]

                points_out_mv[orig_idx, vtx1_idx, 0] = <double>trig.vertex_[0][0]
                points_out_mv[orig_idx, vtx1_idx, 1] = <double>trig.vertex_[0][1]
                points_out_mv[orig_idx, vtx1_idx, 2] = <double>trig.vertex_[0][2]

                points_out_mv[orig_idx, vtx2_idx, 0] = <double>trig.vertex_[1][0]
                points_out_mv[orig_idx, vtx2_idx, 1] = <double>trig.vertex_[1][1]
                points_out_mv[orig_idx, vtx2_idx, 2] = <double>trig.vertex_[1][2]

                points_out_mv[orig_idx, vtx3_idx, 0] = <double>trig.vertex_[2][0]
                points_out_mv[orig_idx, vtx3_idx, 1] = <double>trig.vertex_[2][1]
                points_out_mv[orig_idx, vtx3_idx, 2] = <double>trig.vertex_[2][2]
    finally:
        if target_c != NULL:
            del target_c

    out_shape = list(np.shape(timestamp)) + [point_size, 3]
    return {"points": points_out.reshape(out_shape), "cells": cells_arr}


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def cp_GetSceneStateChannels(
    float_t[:, :] tx_local_locs,
    float_t[:, :] rx_local_locs,
    float_t[:, :] q_locs,
    float_t[:, :] q_rots,
):
    """
    Compute global Tx/Rx locations and radar boresights using C++ Rotate function.
    """
    cdef int_t K = <int_t>q_locs.shape[0]
    cdef int_t M = <int_t>tx_local_locs.shape[0]
    cdef int_t N_rx = <int_t>rx_local_locs.shape[0]

    # Allocating outputs
    tx_global_locs = np.zeros((K, M, 3), dtype=np.float32)
    rx_global_locs = np.zeros((K, N_rx, 3), dtype=np.float32)
    radar_boresights = np.zeros((K, 3), dtype=np.float32)

    cdef float_t[:, :, :] tx_global_mv = tx_global_locs
    cdef float_t[:, :, :] rx_global_mv = rx_global_locs
    cdef float_t[:, :] boresight_mv = radar_boresights

    cdef int_t k, m, n
    cdef Vec3[float_t] rot_v, loc_v, local_v, global_v, bore_v
    cdef Vec3[float_t] base_bore = Vec3[float_t](<float_t>1.0, <float_t>0.0, <float_t>0.0)

    for k in range(K):
        rot_v = Vec3[float_t](q_rots[k, 0], q_rots[k, 1], q_rots[k, 2])
        loc_v = Vec3[float_t](q_locs[k, 0], q_locs[k, 1], q_locs[k, 2])

        # Radar boresight
        bore_v = Rotate[float_t](base_bore, rot_v)
        boresight_mv[k, 0] = bore_v[0]
        boresight_mv[k, 1] = bore_v[1]
        boresight_mv[k, 2] = bore_v[2]

        # Tx global locations
        for m in range(M):
            local_v = Vec3[float_t](tx_local_locs[m, 0], tx_local_locs[m, 1], tx_local_locs[m, 2])
            global_v = Rotate[float_t](local_v, rot_v)
            tx_global_mv[k, m, 0] = loc_v[0] + global_v[0]
            tx_global_mv[k, m, 1] = loc_v[1] + global_v[1]
            tx_global_mv[k, m, 2] = loc_v[2] + global_v[2]

        # Rx global locations
        for n in range(N_rx):
            local_v = Vec3[float_t](rx_local_locs[n, 0], rx_local_locs[n, 1], rx_local_locs[n, 2])
            global_v = Rotate[float_t](local_v, rot_v)
            rx_global_mv[k, n, 0] = loc_v[0] + global_v[0]
            rx_global_mv[k, n, 1] = loc_v[1] + global_v[1]
            rx_global_mv[k, n, 2] = loc_v[2] + global_v[2]

    return tx_global_locs, rx_global_locs, radar_boresights
