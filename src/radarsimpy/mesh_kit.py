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
import numpy as np


def check_module_installed(module_name: str) -> bool:
    """
    Check if a Python module is installed

    :param str module_name: Name of the module to check

    :return: True if module is installed, False otherwise
    :rtype: bool
    """
    return importlib.util.find_spec(module_name) is not None


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
        "No valid module was found to process the 3D model file, "
        + "please install one of the following modules: "
        + "`pyvista`, `pymeshlab`, `trimesh`, `meshio`"
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
        "No valid module was found to process the 3D model file, "
        + "please install one of the following modules: "
        + "`pyvista`, `pymeshlab`, `trimesh`, `meshio`"
    )
