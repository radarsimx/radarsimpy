"""
Tests for ``radarsimpy.mesh_kit``

Covers module discovery (``check_module_installed``, ``safe_import``,
``import_mesh_module``), the per-backend branches of ``load_mesh``,
``merge_meshes`` and the input handling of ``get_target_mesh``.

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

import types

import numpy as np
import numpy.testing as npt
import pytest

from radarsimpy import mesh_kit

from .conftest import MESH_MODULES


class TestModuleDiscovery:
    """``check_module_installed`` / ``safe_import`` / ``import_mesh_module``."""

    def test_check_module_installed_for_stdlib(self):
        """A module that is always importable is reported as installed."""
        assert mesh_kit.check_module_installed("json") is True

    def test_check_module_installed_for_missing_module(self):
        """A module that does not exist is reported as missing."""
        assert mesh_kit.check_module_installed("no_such_module_abc123") is False

    def test_check_module_installed_falls_back_to_import(self, monkeypatch):
        """When ``find_spec`` raises, the direct-import fallback still answers."""

        def boom(_name):
            raise ValueError("simulated find_spec failure")

        monkeypatch.setattr(mesh_kit.importlib.util, "find_spec", boom)

        assert mesh_kit.check_module_installed("json") is True
        assert mesh_kit.check_module_installed("no_such_module_abc123") is False

    def test_safe_import_returns_module(self):
        """A valid name yields the module object."""
        module = mesh_kit.safe_import("json")
        assert isinstance(module, types.ModuleType)
        assert module.__name__ == "json"

    def test_safe_import_returns_none_for_missing(self):
        """An invalid name yields ``None`` instead of raising."""
        assert mesh_kit.safe_import("no_such_module_abc123") is None

    @pytest.mark.mesh
    def test_import_mesh_module_returns_installed_backend(self):
        """The first installed backend is returned."""
        module = mesh_kit.import_mesh_module()
        assert module.__name__ in MESH_MODULES

    def test_import_mesh_module_raises_when_none_installed(self, monkeypatch):
        """With no backend available, a helpful ImportError is raised."""
        monkeypatch.setattr(mesh_kit, "check_module_installed", lambda _name: False)

        with pytest.raises(ImportError, match="Mesh Processing Module Required"):
            mesh_kit.import_mesh_module()


class TestLoadMesh:
    """``load_mesh`` dispatches on the backend's module name."""

    @staticmethod
    def _stub(name, **attrs):
        """Build a fake mesh module named ``name`` carrying ``attrs``."""
        module = types.SimpleNamespace(**attrs)
        module.__name__ = name
        return module

    @pytest.mark.mesh
    def test_load_mesh_with_real_backend(self, mesh_module, model_path):
        """A real .stl loads into contiguous points/cells arrays."""
        mesh = mesh_kit.load_mesh(model_path("plate5x5.stl"), 1.0, mesh_module)

        assert set(mesh) == {"points", "cells"}
        assert mesh["points"].ndim == 2 and mesh["points"].shape[1] == 3
        assert mesh["cells"].ndim == 2 and mesh["cells"].shape[1] == 3
        assert mesh["cells"].max() < mesh["points"].shape[0]

    @pytest.mark.mesh
    def test_scale_divides_vertex_coordinates(self, mesh_module, model_path):
        """The ``scale`` argument divides the vertex coordinates."""
        path = model_path("plate5x5.stl")
        unscaled = mesh_kit.load_mesh(path, 1.0, mesh_module)
        scaled = mesh_kit.load_mesh(path, 2.0, mesh_module)

        npt.assert_allclose(scaled["points"], unscaled["points"] / 2.0)
        npt.assert_array_equal(scaled["cells"], unscaled["cells"])

    def test_pyvista_branch(self):
        """The pyvista branch reads ``points`` and unpacks the ``faces`` array."""
        points = np.array([[0.0, 0, 0], [2.0, 0, 0], [0, 2.0, 0]])
        # pyvista stores faces as [n, i0, i1, i2, ...] runs.
        faces = np.array([3, 0, 1, 2])
        data = types.SimpleNamespace(points=points, faces=faces)
        module = self._stub("pyvista", read=lambda _path: data)

        mesh = mesh_kit.load_mesh("ignored.stl", 2.0, module)

        npt.assert_allclose(mesh["points"], points / 2.0)
        npt.assert_array_equal(mesh["cells"], [[0, 1, 2]])

    def test_pymeshlab_branch_makes_arrays_contiguous(self):
        """Fortran-ordered pymeshlab output is converted to C order."""
        points = np.asfortranarray(np.array([[0.0, 0, 0], [4.0, 0, 0], [0, 4.0, 0]]))
        cells = np.asfortranarray(np.array([[0, 1, 2]]))
        cleared = []

        mesh_data = types.SimpleNamespace(
            vertex_matrix=lambda: points, face_matrix=lambda: cells
        )
        mesh_set = types.SimpleNamespace(
            load_new_mesh=lambda _path: None,
            current_mesh=lambda: mesh_data,
            clear=lambda: cleared.append(True),
        )
        module = self._stub("pymeshlab", MeshSet=lambda: mesh_set)

        mesh = mesh_kit.load_mesh("ignored.stl", 4.0, module)

        npt.assert_allclose(mesh["points"], np.array(points) / 4.0)
        npt.assert_array_equal(mesh["cells"], [[0, 1, 2]])
        assert mesh["points"].flags["C_CONTIGUOUS"]
        assert cleared == [True], "the MeshSet should be cleared after loading"

    def test_meshio_branch(self):
        """The meshio branch takes the first cell block."""
        points = np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]])
        cell_block = types.SimpleNamespace(data=np.array([[0, 1, 2]]))
        data = types.SimpleNamespace(points=points, cells=[cell_block])
        module = self._stub("meshio", read=lambda _path: data)

        mesh = mesh_kit.load_mesh("ignored.stl", 1.0, module)

        npt.assert_allclose(mesh["points"], points)
        npt.assert_array_equal(mesh["cells"], [[0, 1, 2]])

    def test_unknown_backend_raises(self):
        """An unrecognised module name is rejected with the install hint."""
        module = self._stub("not_a_mesh_library")

        with pytest.raises(ImportError, match="Mesh Processing Module Required"):
            mesh_kit.load_mesh("ignored.stl", 1.0, module)


class TestMergeMeshes:
    """``merge_meshes`` concatenates points and re-bases the face indices."""

    def test_empty_list_returns_empty_arrays(self):
        """Merging nothing gives correctly shaped empty arrays."""
        merged = mesh_kit.merge_meshes([])

        assert merged["points"].shape == (0, 3)
        assert merged["cells"].shape == (0, 3)

    def test_single_mesh_is_unchanged(self):
        """A single mesh passes through untouched."""
        mesh = {
            "points": np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]),
            "cells": np.array([[0, 1, 2]]),
        }

        merged = mesh_kit.merge_meshes([mesh])

        npt.assert_allclose(merged["points"], mesh["points"])
        npt.assert_array_equal(merged["cells"], mesh["cells"])

    def test_face_indices_are_offset_per_mesh(self):
        """The second mesh's faces are shifted by the first mesh's point count."""
        first = {
            "points": np.zeros((3, 3)),
            "cells": np.array([[0, 1, 2]]),
        }
        second = {
            "points": np.ones((4, 3)),
            "cells": np.array([[0, 1, 2], [1, 2, 3]]),
        }

        merged = mesh_kit.merge_meshes([first, second])

        assert merged["points"].shape == (7, 3)
        npt.assert_array_equal(merged["cells"], [[0, 1, 2], [3, 4, 5], [4, 5, 6]])
        assert merged["cells"].max() < merged["points"].shape[0]

    def test_time_varying_points_merge_on_the_vertex_axis(self):
        """Meshes with a leading time axis concatenate on the vertex axis."""
        first = {
            "points": np.zeros((5, 3, 3)),  # [timestamps, vertices, xyz]
            "cells": np.array([[0, 1, 2]]),
        }
        second = {
            "points": np.ones((5, 4, 3)),
            "cells": np.array([[0, 1, 2]]),
        }

        merged = mesh_kit.merge_meshes([first, second])

        assert merged["points"].shape == (5, 7, 3)
        npt.assert_array_equal(merged["cells"], [[0, 1, 2], [3, 4, 5]])


class TestGetTargetMesh:
    """Input validation and list handling in ``get_target_mesh``."""

    def test_rejects_non_dict_target(self):
        """Scalars and strings are not valid targets."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            mesh_kit.get_target_mesh("plate5x5.stl", None)

    def test_rejects_dict_without_model_key(self):
        """A point target (no ``model``) cannot produce a mesh."""
        with pytest.raises(ValueError, match="'model' key"):
            mesh_kit.get_target_mesh({"location": (10, 0, 0)}, None)

    def test_list_without_models_returns_empty_mesh(self):
        """A list of point targets merges down to an empty mesh."""
        merged = mesh_kit.get_target_mesh(
            [{"location": (10, 0, 0)}, {"location": (20, 0, 0)}], None
        )

        assert merged["points"].shape == (0, 3)
        assert merged["cells"].shape == (0, 3)

    @pytest.mark.mesh
    def test_single_target_without_radar(self, model_path):
        """A mesh target resolves without a radar object."""
        mesh = mesh_kit.get_target_mesh(
            {"model": model_path("plate5x5.stl"), "location": (10, 0, 0)}, None
        )

        assert mesh["points"].shape[-1] == 3
        assert mesh["points"].shape[0] > 0
        # The plate is placed 10 m down-range.
        npt.assert_allclose(mesh["points"][:, 0].mean(), 10.0, atol=1e-3)

    @pytest.mark.mesh
    def test_tuple_of_targets_is_merged(self, model_path):
        """Tuples are accepted alongside lists and merge the same way."""
        target = {"model": model_path("plate5x5.stl"), "location": (10, 0, 0)}

        single = mesh_kit.get_target_mesh(target, None)
        merged = mesh_kit.get_target_mesh((target, target), None)

        assert merged["points"].shape[0] == 2 * single["points"].shape[0]
        assert merged["cells"].max() < merged["points"].shape[0]

    @pytest.mark.mesh
    def test_point_targets_are_skipped_in_mixed_list(self, model_path):
        """Point targets in a mixed list are ignored rather than raising."""
        target = {"model": model_path("plate5x5.stl"), "location": (10, 0, 0)}

        single = mesh_kit.get_target_mesh(target, None)
        mixed = mesh_kit.get_target_mesh(
            [target, {"location": (20, 0, 0), "rcs": 10}], None
        )

        npt.assert_allclose(mixed["points"], single["points"])


@pytest.mark.mesh
class TestTargetMeshTimestampShapes:
    """
    ``get_target_mesh`` accepts a timestamp of any rank and returns points
    shaped ``np.shape(timestamp) + (num_points, 3)``.
    """

    @pytest.fixture
    def moving_target(self, model_path):
        """A plate sliding along +X at 10 m/s from 10 m down-range."""
        return {
            "model": model_path("plate5x5.stl"),
            "location": (10, 0, 0),
            "speed": (10, 0, 0),
        }

    @staticmethod
    def _num_points(target):
        """Vertex count of the target's mesh."""
        return mesh_kit.get_target_mesh(target, None, 0.0)["points"].shape[0]

    def test_scalar_timestamp_has_no_leading_axis(self, moving_target):
        """A scalar query stays 2-D -- it must not gain a length-1 time axis."""
        mesh = mesh_kit.get_target_mesh(moving_target, None, 0.0)

        assert mesh["points"].ndim == 2
        assert mesh["points"].shape[-1] == 3

    @pytest.mark.parametrize(
        "shape", [(1,), (4,), (2, 2), (1, 3), (2, 2, 2), (2, 1, 3)]
    )
    def test_points_take_the_timestamp_shape(self, moving_target, shape):
        """Every timestamp rank produces ``shape + (num_points, 3)``."""
        num_points = self._num_points(moving_target)
        timestamp = np.linspace(0.0, 1.0, int(np.prod(shape))).reshape(shape)

        mesh = mesh_kit.get_target_mesh(moving_target, None, timestamp)

        assert mesh["points"].shape == shape + (num_points, 3)
        assert mesh["cells"].shape[-1] == 3
        assert mesh["cells"].max() < num_points

    def test_nd_values_match_the_flat_query(self, moving_target):
        """Reshaping the query reshapes the result without changing it."""
        times = np.array([0.0, 0.5, 1.0, 1.5])

        flat = mesh_kit.get_target_mesh(moving_target, None, times)
        square = mesh_kit.get_target_mesh(moving_target, None, times.reshape(2, 2))

        npt.assert_allclose(
            square["points"], flat["points"].reshape(square["points"].shape)
        )
        npt.assert_array_equal(square["cells"], flat["cells"])

    def test_motion_is_resolved_per_timestamp(self, moving_target):
        """A 2-D query moves the plate to the right place at each instant."""
        times = np.array([[0.0, 1.0], [2.0, 3.0]])

        mesh = mesh_kit.get_target_mesh(moving_target, None, times)

        # Starts at x = 10 m and advances 10 m per second.
        mean_x = mesh["points"][..., 0].mean(axis=-1)
        npt.assert_allclose(mean_x, 10.0 + 10.0 * times, atol=1e-3)

    def test_python_list_timestamps_are_accepted(self, moving_target):
        """Nested lists work as well as arrays."""
        num_points = self._num_points(moving_target)

        mesh = mesh_kit.get_target_mesh(moving_target, None, [[0.0, 1.0], [2.0, 3.0]])

        assert mesh["points"].shape == (2, 2, num_points, 3)

    def test_merged_targets_keep_the_timestamp_shape(self, moving_target):
        """Merging several targets concatenates on the vertex axis only."""
        num_points = self._num_points(moving_target)
        times = np.array([[0.0, 1.0], [2.0, 3.0]])

        merged = mesh_kit.get_target_mesh([moving_target, moving_target], None, times)

        assert merged["points"].shape == (2, 2, 2 * num_points, 3)
        assert merged["cells"].max() < 2 * num_points
