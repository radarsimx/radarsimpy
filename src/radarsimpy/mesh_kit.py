import importlib
import numpy as np


def check_module_installed(module_name):
    return importlib.util.find_spec(module_name) is not None


def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def import_mesh_module():
    module_list = ["pyvista", "pymeshlab", "trimesh", "meshio"]

    for _, module_name in enumerate(module_list):
        if check_module_installed(module_name) is not None:
            module = safe_import(module_name)
            return {"name": module_name, "module": module}

    raise ImportError(
        "No valid module was found to process the 3D model file, please install one of the following modules: `pyvista`, `pymeshlab`, `trimesh`, `meshio`"
    )


def load_mesh(mesh_file_name, scale, module, module_name):
    if module_name == "pyvista":
        module.read(mesh_file_name)
        # return {"points":,
        #         "cells":}

    elif module_name == "pymeshlab":
        ms = module.MeshSet()
        ms.load_new_mesh(mesh_file_name)
        t_mesh = ms.current_mesh()
        points = np.array(t_mesh.vertex_matrix()) / scale
        cells = np.array(t_mesh.face_matrix())
        if np.isfortran(points):
            points = np.ascontiguousarray(points)
            cells = np.ascontiguousarray(cells)
        ms.clear()
        return {"points": points, "cells": cells}

    elif module_name == "meshio":
        t_mesh = module.read(mesh_file_name)
        points = t_mesh.points / scale
        cells = t_mesh.cells[0].data
        return {"points": points, "cells": cells}
