"""
Script for loading 3D mesh files

This script provides utilities to load 3D mesh files using various available Python
mesh processing libraries.

---

- Copyright (C) 2025 - PRESENT  radarsimx.com
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

import importlib
from typing import Union, List, Tuple, Dict, Any, Optional
import numpy as np


def check_module_installed(module_name: str) -> bool:
    """
    Check if a Python module is installed

    :param str module_name: Name of the module to check

    :return: True if module is installed, False otherwise
    :rtype: bool
    """
    try:
        # Try using importlib.util first (more precise)
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            return True
    except (ImportError, AttributeError, ValueError, ModuleNotFoundError):
        pass

    # Fallback to direct import attempt
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def safe_import(module_name: str) -> object:
    """
    Safely import a module without raising an exception if not found

    :param str module_name: Name of the module to import

    :return: Imported module object or None if import fails
    :rtype: object
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def import_mesh_module() -> object:
    """
    Import the first available mesh processing module from a predefined list

    Tries to import modules in this order: pyvista, pymeshlab, trimesh, meshio

    :return: The mesh processing module object
    :rtype: object
    :raises ImportError: If no valid mesh processing module is found
    """
    module_list = ["trimesh", "pyvista", "pymeshlab", "meshio"]

    for _, module_name in enumerate(module_list):
        if check_module_installed(module_name):
            module = safe_import(module_name)
            return module

    raise ImportError(
        "\nMesh Processing Module Required\n"
        "-----------------------------\n"
        "No valid module was found to process 3D model files.\n\n"
        "Please install one of the following modules:\n"
        "1. PyVista (Recommended)\n"
        "    • pip install pyvista\n"
        "2. PyMeshLab\n"
        "    • pip install pymeshlab\n"
        "3. Trimesh\n"
        "    • pip install trimesh\n"
        "4. meshio\n"
        "    • pip install meshio\n"
    )


def load_mesh(mesh_file_name: str, scale: float, mesh_module: object) -> dict:
    """
    Load a 3D mesh file using the specified module

    :param str mesh_file_name: Path to the mesh file
    :param float scale: Scale factor to apply to the mesh vertices
    :param object mesh_module: The mesh processing module object

    :return: Dictionary containing mesh points and cells
    :rtype: dict with keys:
        - points (numpy.ndarray): Array of vertex coordinates
        - cells (numpy.ndarray): Array of face indices
    """
    if mesh_module.__name__ == "pyvista":
        mesh_data = mesh_module.read(mesh_file_name)
        points = np.array(mesh_data.points) / scale
        cells = mesh_data.faces.reshape(-1, 4)[:, 1:]
        return {"points": points, "cells": cells}

    if mesh_module.__name__ == "pymeshlab":
        ms = mesh_module.MeshSet()
        ms.load_new_mesh(mesh_file_name)
        mesh_data = ms.current_mesh()
        points = np.array(mesh_data.vertex_matrix()) / scale
        cells = np.array(mesh_data.face_matrix())
        if np.isfortran(points):
            points = np.ascontiguousarray(points)
            cells = np.ascontiguousarray(cells)
        ms.clear()
        return {"points": points, "cells": cells}

    if mesh_module.__name__ == "trimesh":
        mesh_data = mesh_module.load(mesh_file_name)
        points = np.array(mesh_data.vertices) / scale
        cells = np.array(mesh_data.faces)
        return {"points": points, "cells": cells}

    if mesh_module.__name__ == "meshio":
        mesh_data = mesh_module.read(mesh_file_name)
        points = mesh_data.points / scale
        cells = mesh_data.cells[0].data
        return {"points": points, "cells": cells}

    raise ImportError(
        "\nMesh Processing Module Required\n"
        "-----------------------------\n"
        "No valid module was found to process 3D model files.\n\n"
        "Please install one of the following modules:\n"
        "1. PyVista (Recommended)\n"
        "    • pip install pyvista\n"
        "2. PyMeshLab\n"
        "    • pip install pymeshlab\n"
        "3. Trimesh\n"
        "    • pip install trimesh\n"
        "4. meshio\n"
        "    • pip install meshio\n"
    )


def get_target_mesh(
    targets: Union[dict, List[dict], Tuple[dict, ...]],
    radar: Any,
    timestamp: Union[float, np.ndarray] = 0.0,
) -> dict:
    """
    Get the transformed target mesh at query timestamps using C++ Target directly.

    :param targets: A target dict or list/tuple of target dicts.
    :param radar: Radar object containing system configuration.
    :param timestamp: Float or numpy array of query timestamp(s). Default: ``0.0``.

    :return: A dictionary containing:

        * **points** (*numpy.ndarray*): Array of transformed vertex coordinates.
          If `timestamp` is a scalar, shape is ``[N, 3]``.
          If `timestamp` is an array of shape ``[...]``, shape is ``[..., N, 3]``.
        * **cells** (*numpy.ndarray*): Array of face indices with shape ``[M, 3]``.
    :rtype: dict
    """
    if isinstance(targets, (list, tuple)):
        meshes = []
        for t in targets:
            if "model" in t:
                meshes.append(
                    get_target_mesh(t, radar, timestamp)
                )
        return merge_meshes(meshes)

    if not isinstance(targets, dict):
        raise TypeError("targets must be a dictionary or a list/tuple of dictionaries.")

    if "model" not in targets:
        raise ValueError("Target dictionary must contain the 'model' key for mesh loading.")

    mesh_module = import_mesh_module()

    sim_timestamp = None
    if radar is not None:
        sim_timestamp = radar.time_prop.get("timestamp", None)

    from radarsimpy.lib.cp_radarsimc import cp_GetTargetMesh
    return cp_GetTargetMesh(targets, timestamp, mesh_module, sim_timestamp)


def merge_meshes(meshes: List[dict]) -> dict:
    """
    Merge multiple meshes into a single mesh.

    :param list meshes: A list of dictionaries containing points and cells.
    :return: A dictionary containing the merged points and cells.
    :rtype: dict
    """
    if not meshes:
        return {"points": np.zeros((0, 3)), "cells": np.zeros((0, 3), dtype=np.int32)}

    total_points = []
    total_cells = []
    offset = 0

    for m in meshes:
        pts = m["points"]
        cells = m["cells"]

        N = pts.shape[-2]
        total_cells.append(cells + offset)
        total_points.append(pts)
        offset += N

    merged_points = np.concatenate(total_points, axis=-2)
    merged_cells = np.concatenate(total_cells, axis=0)
    return {"points": merged_points, "cells": merged_cells}
