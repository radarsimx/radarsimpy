"""
Module for retrieving and visualizing the scene state including target meshes
and radar platform locations/boresights over time.
"""

from typing import Union, List, Tuple, Any
import numpy as np
from radarsimpy.mesh_kit import get_target_mesh


def get_scene_state(
    targets: Union[dict, List[dict], Tuple[dict, ...]],
    radar: Any,
    timestamp: Union[float, np.ndarray] = 0.0,
) -> dict:
    """
    Get the state of the simulation scene, including target meshes, radar platform,
    transmitter, and receiver locations and boresight directions at query timestamps.

    :param targets: A target dict or list/tuple of target dicts.
    :param radar: Radar object containing system configuration and motion.
    :param timestamp: Float or numpy array of query timestamp(s). Default: ``0.0``.

    :return: A dictionary containing:

        * **targets** (*dict*): Dictionary of merged target meshes at query timestamps:

          * **points** (*numpy.ndarray*): Transformed vertex coordinates.
            If `timestamp` is a scalar, shape is ``[N, 3]``.
            If `timestamp` is an array of shape ``[...]``, shape is ``[..., N, 3]``.
          * **cells** (*numpy.ndarray*): Face indices with shape ``[M, 3]``.

        * **tx_locations** (*numpy.ndarray*): Global locations of Transmitter channels.
          Shape is ``[M, 3]`` if `timestamp` is scalar, or ``[..., M, 3]`` if array.
        * **rx_locations** (*numpy.ndarray*): Global locations of Receiver channels.
          Shape is ``[N_rx, 3]`` if `timestamp` is scalar, or ``[..., N_rx, 3]`` if array.
        * **radar_boresight** (*numpy.ndarray*): Global boresight direction of the radar platform.
          Shape is ``[3]`` if `timestamp` is scalar, or ``[..., 3]`` if array.
    """
    from radarsimpy.lib import cp_GetSceneStateChannels

    t = np.asarray(timestamp)
    t_shape = t.shape
    t_flat = t.ravel()

    # 1. Retrieve the target mesh
    targets_mesh = get_target_mesh(targets, radar, timestamp)

    # 2. Retrieve platform location and rotation at query timestamps
    if radar.radar_prop["location"].ndim > 1:
        # Time-varying motion (need to interpolate from simulation timeline)
        sim_times = radar.time_prop["timestamp"]
        sim_locs = radar.radar_prop["location"]
        sim_rots = radar.radar_prop["rotation"]

        flat_times = sim_times.ravel()
        sort_idx = np.argsort(flat_times)
        flat_times = flat_times[sort_idx]

        flat_locs = sim_locs.reshape(-1, 3)[sort_idx]
        flat_rots = sim_rots.reshape(-1, 3)[sort_idx]

        q_locs = np.zeros((t_flat.size, 3), dtype=np.float32)
        q_rots = np.zeros((t_flat.size, 3), dtype=np.float32)
        for i in range(3):
            q_locs[:, i] = np.interp(t_flat, flat_times, flat_locs[:, i])
            q_rots[:, i] = np.interp(t_flat, flat_times, flat_rots[:, i])
    else:
        # Static/constant velocity motion
        t_expand = t_flat[..., np.newaxis]
        q_locs = (radar.radar_prop["location"] + radar.radar_prop["speed"] * t_expand).astype(np.float32)
        q_rots = (radar.radar_prop["rotation"] + radar.radar_prop["rotation_rate"] * t_expand).astype(np.float32)

    # 3. Call C++ Rotate wrapper
    tx_local_locs = radar.radar_prop["transmitter"].txchannel_prop["locations"].astype(np.float32)
    rx_local_locs = radar.radar_prop["receiver"].rxchannel_prop["locations"].astype(np.float32)

    tx_global_flat, rx_global_flat, radar_boresight_flat = cp_GetSceneStateChannels(
        tx_local_locs,
        rx_local_locs,
        q_locs,
        q_rots,
    )

    # 4. Reshape outputs to match query shape
    M = tx_local_locs.shape[0]
    N_rx = rx_local_locs.shape[0]

    if t.ndim == 0:
        tx_global_locs = np.squeeze(tx_global_flat, axis=0)
        rx_global_locs = np.squeeze(rx_global_flat, axis=0)
        radar_boresight = np.squeeze(radar_boresight_flat, axis=0)
    else:
        tx_global_locs = tx_global_flat.reshape(t_shape + (M, 3))
        rx_global_locs = rx_global_flat.reshape(t_shape + (N_rx, 3))
        radar_boresight = radar_boresight_flat.reshape(t_shape + (3,))

    return {
        "targets": targets_mesh,
        "tx_locations": tx_global_locs,
        "rx_locations": rx_global_locs,
        "radar_boresight": radar_boresight,
    }
