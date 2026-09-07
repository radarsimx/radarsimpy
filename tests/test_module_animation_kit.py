"""
Tests for ``radarsimpy.animation_kit``

Covers glTF parsing (accessors, scene graph, keyframe samplers), the mapping
from animated nodes to RadarSimPy target dictionaries, the Y-up to Z-up frame
conversion, and the unsupported-input errors. The end-to-end cases drive the
real C++ ``Target`` transform through ``mesh_kit.get_target_mesh`` and compare
``sim_radar`` output against hand-written constant-motion targets.

Fixture models are built in ``tmp_path`` with ``pygltflib`` rather than being
committed as binary assets.

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

import numpy as np
import numpy.testing as npt
import pytest

from radarsimpy import animation_kit, mesh_kit

pytestmark = pytest.mark.gltf


# =============================================================================
# glTF fixture construction
# =============================================================================


def build_glb(path, nodes, channels=None, animation_name="clip"):
    """
    Write a minimal binary glTF built from plain numpy arrays.

    ``nodes`` entries accept ``name``, ``points``, ``cells``, ``children``,
    ``translation``, ``rotation`` (xyzw) and ``scale``. ``channels`` entries
    accept ``node``, ``path``, ``times``, ``values`` and ``interpolation``.
    """
    import pygltflib as pg

    blob = bytearray()
    views, accessors = [], []

    def add(array, component_type, acc_type, target=None):
        arr = np.ascontiguousarray(array)
        data = arr.tobytes()
        offset = len(blob)
        blob.extend(data)
        while len(blob) % 4:
            blob.append(0)
        views.append(
            pg.BufferView(
                buffer=0, byteOffset=offset, byteLength=len(data), target=target
            )
        )
        accessor = pg.Accessor(
            bufferView=len(views) - 1,
            byteOffset=0,
            componentType=component_type,
            count=arr.shape[0],
            type=acc_type,
        )
        if component_type == pg.FLOAT:
            flat = arr.reshape(arr.shape[0], -1)
            accessor.max = flat.max(axis=0).tolist()
            accessor.min = flat.min(axis=0).tolist()
        accessors.append(accessor)
        return len(accessors) - 1

    meshes, gltf_nodes = [], []
    for spec in nodes:
        mesh_index = None
        if spec.get("points") is not None:
            position = add(
                np.asarray(spec["points"], np.float32),
                pg.FLOAT,
                pg.VEC3,
                pg.ARRAY_BUFFER,
            )
            indices = add(
                np.asarray(spec["cells"], np.uint32).reshape(-1),
                pg.UNSIGNED_INT,
                pg.SCALAR,
                pg.ELEMENT_ARRAY_BUFFER,
            )
            primitive = pg.Primitive(
                attributes=pg.Attributes(POSITION=position),
                indices=indices,
                mode=spec.get("mode", pg.TRIANGLES),
            )
            if spec.get("skinned"):
                primitive.attributes.JOINTS_0 = position
            meshes.append(pg.Mesh(primitives=[primitive]))
            mesh_index = len(meshes) - 1
        gltf_nodes.append(
            pg.Node(
                name=spec.get("name"),
                mesh=mesh_index,
                children=spec.get("children") or [],
                translation=spec.get("translation"),
                rotation=spec.get("rotation"),
                scale=spec.get("scale"),
            )
        )

    animations = []
    if channels:
        samplers, gltf_channels = [], []
        for channel in channels:
            time_acc = add(
                np.asarray(channel["times"], np.float32).reshape(-1, 1),
                pg.FLOAT,
                pg.SCALAR,
            )
            values = np.asarray(channel["values"], np.float32)
            acc_type = {1: pg.SCALAR, 3: pg.VEC3, 4: pg.VEC4}[values.shape[-1]]
            value_acc = add(values, pg.FLOAT, acc_type)
            samplers.append(
                pg.AnimationSampler(
                    input=time_acc,
                    output=value_acc,
                    interpolation=channel.get("interpolation", "LINEAR"),
                )
            )
            gltf_channels.append(
                pg.AnimationChannel(
                    sampler=len(samplers) - 1,
                    target=pg.AnimationChannelTarget(
                        node=channel["node"], path=channel["path"]
                    ),
                )
            )
        animations.append(
            pg.Animation(name=animation_name, samplers=samplers, channels=gltf_channels)
        )

    children = {c for spec in nodes for c in (spec.get("children") or [])}
    roots = [i for i in range(len(nodes)) if i not in children]

    gltf = pg.GLTF2(
        asset=pg.Asset(version="2.0"),
        scene=0,
        scenes=[pg.Scene(nodes=roots)],
        nodes=gltf_nodes,
        meshes=meshes,
        accessors=accessors,
        bufferViews=views,
        buffers=[pg.Buffer(byteLength=len(blob))],
        animations=animations,
    )
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(path))
    return str(path)


def horizontal_quad(size=1.0):
    """A square in the glTF X-Z plane, i.e. horizontal under glTF's Y-up frame."""
    half = size / 2.0
    points = np.array(
        [[-half, 0, -half], [half, 0, -half], [half, 0, half], [-half, 0, half]],
        np.float32,
    )
    return points, np.array([[0, 1, 2], [0, 2, 3]], np.uint32)


def vertical_quad(size=1.0):
    """A square in the glTF Y-Z plane, which becomes the Y-Z plane in RadarSimPy."""
    half = size / 2.0
    points = np.array(
        [[0, -half, -half], [0, half, -half], [0, half, half], [0, -half, half]],
        np.float32,
    )
    return points, np.array([[0, 1, 2], [0, 2, 3]], np.uint32)


def spin_keys(revs_per_second, duration, steps=64):
    """LINEAR quaternion keyframes spinning about the glTF +Y axis."""
    times = np.linspace(0.0, duration, steps + 1)
    angles = 2 * np.pi * revs_per_second * times
    zeros = np.zeros_like(angles)
    quats = np.stack((zeros, np.sin(angles / 2), zeros, np.cos(angles / 2)), axis=-1)
    return times, quats


class FakeRadar:
    """Minimal stand-in exposing only the timestamp ``animation_kit`` reads."""

    def __init__(self, timestamp):
        self.time_prop = {"timestamp": np.asarray(timestamp, dtype=np.float64)}


@pytest.fixture
def spinning_model(tmp_path):
    """A single node spinning once per second about the glTF up axis."""
    points, cells = horizontal_quad(2.0)
    times, quats = spin_keys(1.0, 1.0)
    return build_glb(
        tmp_path / "spin.glb",
        nodes=[{"name": "rotor", "points": points, "cells": cells}],
        channels=[{"node": 0, "path": "rotation", "times": times, "values": quats}],
    )


@pytest.fixture
def rising_model(tmp_path):
    """A single node translating 10 m along the glTF up axis over one second."""
    points, cells = horizontal_quad(2.0)
    return build_glb(
        tmp_path / "rise.glb",
        nodes=[{"name": "lift", "points": points, "cells": cells}],
        channels=[
            {
                "node": 0,
                "path": "translation",
                "times": np.array([0.0, 1.0]),
                "values": np.array([[0, 0, 0], [0, 10.0, 0]]),
            }
        ],
    )


# =============================================================================
# Module discovery
# =============================================================================


class TestModuleDiscovery:
    """``import_gltf_module``."""

    def test_returns_pygltflib(self, gltf_module):
        """The discovered module is the one the tests build fixtures with."""
        assert animation_kit.import_gltf_module() is gltf_module

    def test_missing_module_raises_with_install_hint(self, monkeypatch):
        """A missing library produces an actionable ImportError."""
        monkeypatch.setattr(
            animation_kit, "check_module_installed", lambda _name: False
        )

        with pytest.raises(ImportError, match="pip install pygltflib"):
            animation_kit.import_gltf_module()


# =============================================================================
# Model parsing
# =============================================================================


class TestLoadAnimatedModel:
    """``load_animated_model``."""

    def test_reports_clip_metadata(self, spinning_model):
        """Animation name, list and duration come back with the parts."""
        model = animation_kit.load_animated_model(spinning_model)

        assert model["animation"] == "clip"
        assert model["animations"] == ["clip"]
        assert model["duration"] == pytest.approx(1.0)
        assert model["start_time"] == pytest.approx(0.0)

    def test_animated_node_becomes_a_part(self, spinning_model):
        """The one animated node yields one part carrying the geometry."""
        parts = animation_kit.load_animated_model(spinning_model)["parts"]

        assert len(parts) == 1
        assert parts[0]["name"] == "rotor"
        assert parts[0]["animated"] is True
        assert parts[0]["points"].shape == (4, 3)
        assert parts[0]["cells"].shape == (2, 3)

    def test_static_geometry_is_merged(self, tmp_path):
        """Meshes with no animated ancestor collapse into a single static part."""
        points, cells = horizontal_quad()
        times, quats = spin_keys(1.0, 1.0)
        path = build_glb(
            tmp_path / "mixed.glb",
            nodes=[
                {"name": "spin", "points": points, "cells": cells},
                {
                    "name": "body_a",
                    "points": points,
                    "cells": cells,
                    "translation": [5.0, 0, 0],
                },
                {
                    "name": "body_b",
                    "points": points,
                    "cells": cells,
                    "translation": [-5.0, 0, 0],
                },
            ],
            channels=[{"node": 0, "path": "rotation", "times": times, "values": quats}],
        )

        parts = animation_kit.load_animated_model(path)["parts"]

        assert [(part["name"], part["animated"]) for part in parts] == [
            ("spin", True),
            ("static", False),
        ]
        assert len(parts[1]["cells"]) == 4

    def test_static_geometry_can_stay_split(self, tmp_path):
        """``merge_static=False`` keeps one part per static mesh."""
        points, cells = horizontal_quad()
        times, quats = spin_keys(1.0, 1.0)
        path = build_glb(
            tmp_path / "mixed.glb",
            nodes=[
                {"name": "spin", "points": points, "cells": cells},
                {"name": "body_a", "points": points, "cells": cells},
                {"name": "body_b", "points": points, "cells": cells},
            ],
            channels=[{"node": 0, "path": "rotation", "times": times, "values": quats}],
        )

        parts = animation_kit.load_animated_model(path, merge_static=False)["parts"]

        assert [part["name"] for part in parts] == ["spin", "static_0", "static_1"]

    def test_unit_scales_geometry_to_meters(self, spinning_model):
        """A model authored in millimeters is converted to meters."""
        model = animation_kit.load_animated_model(spinning_model, unit="mm")

        assert model["parts"][0]["points"].max() == pytest.approx(1e-3)

    def test_invalid_unit_rejected(self, spinning_model):
        """An unknown unit is reported rather than silently ignored."""
        with pytest.raises(ValueError, match="Invalid unit"):
            animation_kit.load_animated_model(spinning_model, unit="furlong")

    def test_invalid_up_axis_rejected(self, spinning_model):
        """Only 'y' and 'z' up axes are accepted."""
        with pytest.raises(ValueError, match="Invalid up_axis"):
            animation_kit.load_animated_model(spinning_model, up_axis="x")

    def test_unknown_animation_name_rejected(self, spinning_model):
        """Selecting a missing animation lists what is available."""
        with pytest.raises(ValueError, match="Available animations"):
            animation_kit.load_animated_model(spinning_model, animation="walk")

    def test_animation_index_out_of_range_rejected(self, spinning_model):
        """An out-of-range animation index is reported."""
        with pytest.raises(ValueError, match="out of range"):
            animation_kit.load_animated_model(spinning_model, animation=7)


# =============================================================================
# Sampled motion
# =============================================================================


class TestRotationSampling:
    """Rotation tracks turn into ``rotation`` / ``rotation_rate`` arrays."""

    def test_yaw_follows_the_keyframes(self, spinning_model):
        """A spin about the glTF up axis becomes pure yaw in RadarSimPy."""
        radar = FakeRadar([[[0.0, 0.125, 0.25, 0.5]]])

        target = animation_kit.load_animated_targets(spinning_model, radar)[0]

        npt.assert_allclose(
            target["rotation"][0].ravel(), [0.0, 45.0, 90.0, 180.0], atol=1e-3
        )
        npt.assert_allclose(target["rotation"][1].ravel(), 0.0, atol=1e-4)
        npt.assert_allclose(target["rotation"][2].ravel(), 0.0, atol=1e-4)

    def test_rotation_rate_matches_the_spin(self, spinning_model):
        """One revolution per second is emitted as 360 deg/s of yaw rate."""
        radar = FakeRadar([[[0.0, 0.125, 0.25, 0.5]]])

        target = animation_kit.load_animated_targets(spinning_model, radar)[0]

        npt.assert_allclose(target["rotation_rate"][0].ravel(), 360.0, rtol=1e-4)
        npt.assert_allclose(target["rotation_rate"][1].ravel(), 0.0, atol=1e-4)
        npt.assert_allclose(target["rotation_rate"][2].ravel(), 0.0, atol=1e-4)

    def test_angles_stay_wrapped(self, spinning_model):
        """
        Angles are left in ``(-180, 180]``.

        ``MoveIndex`` turns consecutive angles into ``Rz(dyaw) Ry(-dpitch)
        Rx(droll)``, each factor 360-degree periodic, so wrapping is safe and
        keeps the float32 arrays from growing without bound.
        """
        radar = FakeRadar([[np.linspace(0.0, 3.0, 64)]])

        target = animation_kit.load_animated_targets(spinning_model, radar)[0]

        assert np.abs(target["rotation"][0]).max() <= 180.0 + 1e-3

    def test_gimbal_pole_warns(self, tmp_path):
        """Reaching +/-90 deg of pitch warns that yaw and roll become ambiguous."""
        points, cells = horizontal_quad()
        angles = np.array([0.0, np.pi / 2])
        zeros = np.zeros(2)
        quats = np.stack(
            (zeros, zeros, np.sin(angles / 2), np.cos(angles / 2)), axis=-1
        )
        path = build_glb(
            tmp_path / "pole.glb",
            nodes=[{"points": points, "cells": cells}],
            channels=[
                {
                    "node": 0,
                    "path": "rotation",
                    "times": np.array([0.0, 1.0]),
                    "values": quats,
                }
            ],
        )
        radar = FakeRadar([[[0.0, 1.0]]])

        with pytest.warns(UserWarning, match="pitch"):
            target = animation_kit.load_animated_targets(path, radar, loop=False)[0]

        npt.assert_allclose(target["rotation"][1].ravel()[-1], 90.0, atol=1e-3)


class TestTranslationSampling:
    """Translation tracks turn into ``location`` / ``speed`` arrays."""

    def test_up_axis_conversion(self, rising_model):
        """Motion along glTF +Y lands on RadarSimPy +Z."""
        radar = FakeRadar([[[0.0, 0.25, 0.5]]])

        target = animation_kit.load_animated_targets(rising_model, radar)[0]

        npt.assert_allclose(target["location"][0].ravel(), 0.0, atol=1e-6)
        npt.assert_allclose(target["location"][1].ravel(), 0.0, atol=1e-6)
        npt.assert_allclose(target["location"][2].ravel(), [0.0, 2.5, 5.0], atol=1e-5)

    def test_speed_is_the_analytic_derivative(self, rising_model):
        """
        Velocity comes from the sampler, not from differencing the grid.

        ``radar.time_prop["timestamp"]`` restarts at every channel, so a
        difference over the flattened array would spike at each boundary.
        """
        timestamp = np.stack([np.linspace(0.0, 0.4, 8).reshape(2, 4)] * 3, axis=0)
        radar = FakeRadar(timestamp)

        target = animation_kit.load_animated_targets(rising_model, radar)[0]

        npt.assert_allclose(target["speed"][2], 10.0, rtol=1e-5)
        npt.assert_allclose(target["speed"][0], 0.0, atol=1e-5)

    def test_z_up_models_are_not_rotated(self, rising_model):
        """``up_axis='z'`` leaves an already Z-up asset alone."""
        radar = FakeRadar([[[0.5]]])

        target = animation_kit.load_animated_targets(rising_model, radar, up_axis="z")[
            0
        ]

        npt.assert_allclose(target["location"][1].ravel(), 5.0, atol=1e-5)
        npt.assert_allclose(target["location"][2].ravel(), 0.0, atol=1e-6)


class TestHierarchy:
    """Ancestor transforms compose into the sampled world pose."""

    def test_child_geometry_is_baked_into_the_animated_parent(self, tmp_path):
        """A static child under an animated parent joins the parent's part."""
        points, cells = horizontal_quad(0.2)
        times, quats = spin_keys(1.0, 1.0)
        path = build_glb(
            tmp_path / "hier.glb",
            nodes=[
                {"name": "hub", "children": [1], "translation": [0, 0, -10.0]},
                {
                    "name": "blade",
                    "points": points,
                    "cells": cells,
                    "translation": [3.0, 0, 0],
                },
            ],
            channels=[{"node": 0, "path": "rotation", "times": times, "values": quats}],
        )
        radar = FakeRadar([[[0.0]]])

        model = animation_kit.load_animated_model(path)
        target = animation_kit.load_animated_targets(path, radar)[0]

        assert [part["name"] for part in model["parts"]] == ["hub"]
        # The hub sits 10 m along glTF -Z, which is +Y in RadarSimPy.
        npt.assert_allclose(target["location"][1].ravel(), 10.0, atol=1e-5)
        # The blade's 3 m offset is baked into the vertices, not the location.
        npt.assert_allclose(
            target["model"]["points"].mean(axis=0), [3.0, 0.0, 0.0], atol=1e-5
        )

    def test_uniform_parent_scale_reaches_the_vertices(self, tmp_path):
        """A uniform scale above an animated node is folded into its geometry."""
        points, cells = horizontal_quad(2.0)
        times, quats = spin_keys(1.0, 1.0)
        path = build_glb(
            tmp_path / "scaled.glb",
            nodes=[
                {"name": "root", "children": [1], "scale": [3.0, 3.0, 3.0]},
                {"name": "spin", "points": points, "cells": cells},
            ],
            channels=[{"node": 1, "path": "rotation", "times": times, "values": quats}],
        )

        parts = animation_kit.load_animated_model(path)["parts"]

        assert parts[0]["points"].max() == pytest.approx(3.0)


class TestPlacement:
    """The outer ``location`` / ``speed`` / ``rotation`` arguments."""

    def test_location_offsets_every_part(self, rising_model):
        """The model placement adds to each part's sampled position."""
        radar = FakeRadar([[[0.5]]])

        target = animation_kit.load_animated_targets(
            rising_model, radar, location=(20.0, -3.0, 1.0)
        )[0]

        npt.assert_allclose(target["location"][0].ravel(), 20.0, atol=1e-5)
        npt.assert_allclose(target["location"][1].ravel(), -3.0, atol=1e-5)
        npt.assert_allclose(target["location"][2].ravel(), 6.0, atol=1e-5)

    def test_speed_adds_to_the_sampled_velocity(self, rising_model):
        """Model velocity and part velocity superpose."""
        radar = FakeRadar([[[0.0, 0.5]]])

        target = animation_kit.load_animated_targets(
            rising_model, radar, speed=(4.0, 0.0, 1.0)
        )[0]

        npt.assert_allclose(target["speed"][0].ravel(), 4.0, atol=1e-5)
        npt.assert_allclose(target["speed"][2].ravel(), 11.0, rtol=1e-5)
        npt.assert_allclose(target["location"][0].ravel(), [0.0, 2.0], atol=1e-5)

    def test_rotation_turns_the_whole_model(self, rising_model):
        """A 90 deg model yaw rotates each part's motion into the new frame."""
        radar = FakeRadar([[[0.5]]])

        target = animation_kit.load_animated_targets(
            rising_model, radar, rotation=(90.0, 0.0, 0.0)
        )[0]

        npt.assert_allclose(target["rotation"][0].ravel(), 90.0, atol=1e-3)
        # Motion is along the model's up axis, which yaw leaves untouched.
        npt.assert_allclose(target["location"][2].ravel(), 5.0, atol=1e-5)


class TestTimeMapping:
    """``loop``, ``time_offset`` and ``time_scale``."""

    def test_looping_wraps_past_the_clip(self, rising_model):
        """Simulation time beyond the clip restarts it."""
        radar = FakeRadar([[[0.0, 0.5, 1.0, 1.5, 2.5]]])

        target = animation_kit.load_animated_targets(rising_model, radar)[0]

        npt.assert_allclose(
            target["location"][2].ravel(), [0.0, 5.0, 0.0, 5.0, 5.0], atol=1e-4
        )

    def test_clamping_holds_the_last_pose(self, rising_model):
        """``loop=False`` holds the final keyframe and zeroes the rate."""
        radar = FakeRadar([[[0.0, 0.5, 1.5, 2.5]]])

        target = animation_kit.load_animated_targets(rising_model, radar, loop=False)[0]

        npt.assert_allclose(
            target["location"][2].ravel(), [0.0, 5.0, 10.0, 10.0], atol=1e-4
        )
        npt.assert_allclose(target["speed"][2].ravel()[-2:], 0.0, atol=1e-5)

    def test_time_scale_speeds_the_clip_up(self, rising_model):
        """``time_scale=2`` doubles both the progress and the velocity."""
        radar = FakeRadar([[[0.0, 0.25]]])

        target = animation_kit.load_animated_targets(
            rising_model, radar, time_scale=2.0, loop=False
        )[0]

        npt.assert_allclose(target["location"][2].ravel(), [0.0, 5.0], atol=1e-4)
        npt.assert_allclose(target["speed"][2].ravel(), 20.0, rtol=1e-5)

    def test_time_offset_shifts_the_start(self, rising_model):
        """``time_offset`` picks the animation time at simulation time zero."""
        radar = FakeRadar([[[0.0]]])

        target = animation_kit.load_animated_targets(
            rising_model, radar, time_offset=0.75
        )[0]

        npt.assert_allclose(target["location"][2].ravel(), 7.5, atol=1e-4)


class TestInterpolationModes:
    """``STEP``, ``LINEAR`` and ``CUBICSPLINE`` samplers."""

    def test_step_holds_each_key(self, tmp_path):
        """STEP jumps between keys and reports a zero rate."""
        points, cells = horizontal_quad()
        path = build_glb(
            tmp_path / "step.glb",
            nodes=[{"points": points, "cells": cells}],
            channels=[
                {
                    "node": 0,
                    "path": "translation",
                    "times": np.array([0.0, 0.5, 1.0]),
                    "values": np.array([[0, 0, 0], [0, 4.0, 0], [0, 0, 0]]),
                    "interpolation": "STEP",
                }
            ],
        )
        radar = FakeRadar([[[0.0, 0.25, 0.5, 0.75]]])

        target = animation_kit.load_animated_targets(path, radar, loop=False)[0]

        npt.assert_allclose(
            target["location"][2].ravel(), [0.0, 0.0, 4.0, 4.0], atol=1e-5
        )
        npt.assert_allclose(target["speed"][2].ravel(), 0.0, atol=1e-6)

    def test_cubicspline_matches_the_hermite_basis(self, tmp_path):
        """CUBICSPLINE values and rates follow the analytic Hermite curve."""
        points, cells = horizontal_quad()
        keys = np.zeros((3, 3, 3))
        keys[:, 1, :] = np.array([[0, 0, 0], [0, 4.0, 0], [0, 0, 0]])
        path = build_glb(
            tmp_path / "cubic.glb",
            nodes=[{"points": points, "cells": cells}],
            channels=[
                {
                    "node": 0,
                    "path": "translation",
                    "times": np.array([0.0, 0.5, 1.0]),
                    "values": keys.reshape(9, 3),
                    "interpolation": "CUBICSPLINE",
                }
            ],
        )
        radar = FakeRadar([[[0.0, 0.25, 0.5, 0.75]]])

        target = animation_kit.load_animated_targets(path, radar, loop=False)[0]

        # Zero tangents make each segment a smoothstep: half the rise at the
        # midpoint, and a peak rate of 1.5 * 4 / 0.5 there.
        npt.assert_allclose(
            target["location"][2].ravel(), [0.0, 2.0, 4.0, 2.0], atol=1e-4
        )
        npt.assert_allclose(
            target["speed"][2].ravel(), [0.0, 12.0, 0.0, -12.0], atol=1e-3
        )


class TestStaticSnapshot:
    """``at_time`` for the simulators that reject time-varying motion."""

    def test_at_time_emits_scalar_motion(self, rising_model):
        """A snapshot has plain tuples, which ``sim_rcs``/``sim_lidar`` accept."""
        target = animation_kit.load_animated_targets(rising_model, at_time=0.5)[0]

        assert isinstance(target["location"], tuple)
        assert all(
            np.isscalar(value) or np.ndim(value) == 0 for value in target["location"]
        )
        npt.assert_allclose(target["location"][2], 5.0, atol=1e-6)

    def test_missing_radar_and_time_is_rejected(self, rising_model):
        """Neither sampling mode selected is an error, not a silent default."""
        with pytest.raises(ValueError, match="at_time"):
            animation_kit.load_animated_targets(rising_model)

    def test_non_3d_timestamp_is_rejected(self, rising_model):
        """The C++ motion arrays are indexed as [channels, pulses, samples]."""
        with pytest.raises(ValueError, match="3-D"):
            animation_kit.load_animated_targets(rising_model, FakeRadar([0.0, 1.0]))


class TestTargetDictShape:
    """The emitted dictionaries match what ``cp_AddTarget`` expects."""

    def test_only_recognised_keys_are_emitted(self, spinning_model):
        """Unknown keys would make ``sim_radar`` warn, so none are produced."""
        valid = {
            "model",
            "unit",
            "origin",
            "location",
            "speed",
            "rotation",
            "rotation_rate",
            "permittivity",
            "permeability",
            "skip_diffusion",
            "density",
            "environment",
        }
        radar = FakeRadar([[[0.0]]])

        targets = animation_kit.load_animated_targets(
            spinning_model, radar, permittivity="PEC", skip_diffusion=True
        )

        assert set(targets[0]) <= valid
        assert targets[0]["permittivity"] == "PEC"
        assert targets[0]["skip_diffusion"] is True

    def test_motion_arrays_match_the_timestamp_shape_and_dtype(self, spinning_model):
        """Each axis binds to a C-contiguous float32 [ch, pulses, samples] view."""
        timestamp = np.linspace(0.0, 1e-3, 24).reshape(2, 3, 4)
        radar = FakeRadar(timestamp)

        target = animation_kit.load_animated_targets(spinning_model, radar)[0]

        for key in ("location", "speed", "rotation", "rotation_rate"):
            for axis in target[key]:
                assert axis.shape == timestamp.shape
                assert axis.dtype == np.float32
                assert axis.flags["C_CONTIGUOUS"]

    def test_model_is_handed_over_in_memory(self, spinning_model):
        """Geometry travels as a dict, which ``mesh_kit.load_mesh`` accepts."""
        radar = FakeRadar([[[0.0]]])

        target = animation_kit.load_animated_targets(spinning_model, radar)[0]

        assert set(target["model"]) == {"points", "cells"}
        assert target["model"]["cells"].dtype == np.int32
        assert target["unit"] == "m"
        assert target["origin"] == (0.0, 0.0, 0.0)


# =============================================================================
# Unsupported inputs
# =============================================================================


class TestUnsupportedInputs:
    """Deforming geometry has no rigid-target representation."""

    def test_animated_scale(self, tmp_path):
        """A scale channel deforms the mesh."""
        points, cells = horizontal_quad()
        path = build_glb(
            tmp_path / "scale.glb",
            nodes=[{"points": points, "cells": cells}],
            channels=[
                {
                    "node": 0,
                    "path": "scale",
                    "times": np.array([0.0, 1.0]),
                    "values": np.array([[1.0, 1, 1], [2.0, 2, 2]]),
                }
            ],
        )

        with pytest.raises(NotImplementedError, match="Animated scale"):
            animation_kit.load_animated_model(path)

    def test_non_uniform_scale_on_an_animated_chain(self, tmp_path):
        """A non-uniform scale cannot be split into a rotation and translation."""
        points, cells = horizontal_quad()
        times, quats = spin_keys(1.0, 1.0)
        path = build_glb(
            tmp_path / "nonuniform.glb",
            nodes=[
                {
                    "name": "squashed",
                    "points": points,
                    "cells": cells,
                    "scale": [1.0, 2.0, 1.0],
                }
            ],
            channels=[{"node": 0, "path": "rotation", "times": times, "values": quats}],
        )

        with pytest.raises(NotImplementedError, match="non-uniform scale"):
            animation_kit.load_animated_model(path)

    def test_skinned_mesh(self, tmp_path):
        """Skinning moves individual vertices."""
        points, cells = horizontal_quad()
        path = build_glb(
            tmp_path / "skinned.glb",
            nodes=[{"points": points, "cells": cells, "skinned": True}],
        )

        with pytest.raises(NotImplementedError, match="Skinned"):
            animation_kit.load_animated_model(path)

    def test_non_triangle_primitive(self, tmp_path):
        """Only triangles can be ray traced."""
        import pygltflib as pg

        points, cells = horizontal_quad()
        path = build_glb(
            tmp_path / "lines.glb",
            nodes=[{"points": points, "cells": cells, "mode": pg.LINES}],
        )

        with pytest.raises(NotImplementedError, match="primitive mode"):
            animation_kit.load_animated_model(path)


# =============================================================================
# Accessor decoding
# =============================================================================


class TestAccessorDecoding:
    """``_read_accessor`` handles the buffer layouts exporters produce."""

    def test_interleaved_byte_stride(self, gltf_module):
        """Attributes packed with a byteStride are de-interleaved correctly."""
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], np.float32)
        padding = np.array([[9.0], [9.0]], np.float32)
        blob = np.hstack((expected, padding)).tobytes()

        gltf = gltf_module.GLTF2(
            accessors=[
                gltf_module.Accessor(
                    bufferView=0,
                    byteOffset=0,
                    componentType=gltf_module.FLOAT,
                    count=2,
                    type=gltf_module.VEC3,
                )
            ],
            bufferViews=[
                gltf_module.BufferView(
                    buffer=0, byteOffset=0, byteLength=len(blob), byteStride=16
                )
            ],
        )

        values = animation_kit._read_accessor(gltf, blob, 0)

        npt.assert_allclose(values, expected)

    def test_missing_buffer_view_reads_as_zeros(self, gltf_module):
        """The spec defines an absent bufferView as all-zero data."""
        gltf = gltf_module.GLTF2(
            accessors=[
                gltf_module.Accessor(
                    componentType=gltf_module.FLOAT,
                    count=3,
                    type=gltf_module.VEC3,
                )
            ]
        )

        values = animation_kit._read_accessor(gltf, b"", 0)

        npt.assert_array_equal(values, np.zeros((3, 3)))


# =============================================================================
# End to end, through the compiled transform
# =============================================================================


@pytest.mark.mesh
class TestAgainstTheCppTransform:
    """The emitted motion drives the real ``Target::Move`` correctly."""

    def test_orbiting_part_traces_the_expected_circle(self, tmp_path, make_radar):
        """
        A blade offset from a spinning hub orbits at the analytic angle.

        This validates the yaw/pitch/roll convention against RadarSimCpp rather
        than against the extraction formula that produced it.
        """
        radar = make_radar(tx_kwargs={"pulses": 4})
        timestamp = radar.time_prop["timestamp"]

        revs_per_second = 625.0
        points, cells = horizontal_quad(0.2)
        times, quats = spin_keys(revs_per_second, 0.01, steps=400)
        path = build_glb(
            tmp_path / "orbit.glb",
            nodes=[
                {"name": "hub", "children": [1]},
                {
                    "name": "blade",
                    "points": points,
                    "cells": cells,
                    "translation": [3.0, 0, 0],
                },
            ],
            channels=[{"node": 0, "path": "rotation", "times": times, "values": quats}],
        )

        targets = animation_kit.load_animated_targets(path, radar)
        query = timestamp[0, :, 0]
        mesh = mesh_kit.get_target_mesh(targets, radar, timestamp=query)

        centre = mesh["points"].mean(axis=1)
        npt.assert_allclose(np.linalg.norm(centre[:, :2], axis=1), 3.0, rtol=1e-4)
        npt.assert_allclose(centre[:, 2], 0.0, atol=1e-4)

        angle = np.degrees(np.arctan2(centre[:, 1], centre[:, 0]))
        expected = np.degrees(2 * np.pi * revs_per_second * query)
        npt.assert_allclose(angle % 360.0, expected % 360.0, atol=1e-2)


@pytest.mark.mesh
@pytest.mark.slow
class TestSimRadarEquivalence:
    """
    The glTF path reproduces the hand-written constant-motion path.

    This is the check that the Doppler is right and not just the geometry:
    RadarSimCpp derives hit-point velocity from ``speed`` and ``rotation_rate``,
    which the sampler has to supply independently of ``location``.
    """

    @staticmethod
    def _radar(make_radar):
        return make_radar(
            tx_kwargs={
                "f": [24.0e9, 24.2e9],
                "t": 40e-6,
                "tx_power": 15,
                "prp": 50e-6,
                "pulses": 8,
            },
            rx_kwargs={"fs": 4e6},
        )

    @staticmethod
    def _reference_plate(size=1.0):
        """The RadarSimPy-frame twin of :func:`vertical_quad`."""
        half = size / 2.0
        points = np.array(
            [[0, half, -half], [0, half, half], [0, -half, half], [0, -half, -half]],
            float,
        )
        return points, np.array([[0, 1, 2], [0, 2, 3]], np.int32)

    def test_rotation_matches_a_constant_rotation_rate(self, tmp_path, make_radar):
        """A spinning glTF node equals a target with the same ``rotation_rate``."""
        from radarsimpy.simulator import sim_radar

        radar = self._radar(make_radar)
        revs_per_second = 100.0
        points, cells = vertical_quad(1.0)
        times, quats = spin_keys(revs_per_second, 0.02, steps=2000)
        path = build_glb(
            tmp_path / "spin.glb",
            nodes=[{"name": "plate", "points": points, "cells": cells}],
            channels=[{"node": 0, "path": "rotation", "times": times, "values": quats}],
        )

        animated = animation_kit.load_animated_targets(
            path, radar, location=(20, 0, 0), permittivity="PEC"
        )
        ref_points, ref_cells = self._reference_plate(1.0)
        reference = [
            {
                "model": {"points": ref_points, "cells": ref_cells},
                "origin": (0, 0, 0),
                "location": (20, 0, 0),
                "rotation": (0, 0, 0),
                "rotation_rate": (revs_per_second * 360.0, 0, 0),
                "permittivity": "PEC",
            }
        ]

        from_gltf = sim_radar(radar, animated, density=1, level="sample", device="cpu")[
            "baseband"
        ]
        from_dict = sim_radar(
            radar, reference, density=1, level="sample", device="cpu"
        )["baseband"]

        peak = np.abs(from_dict).max()
        assert peak > 0.0
        assert np.abs(from_gltf - from_dict).max() / peak < 1e-3

    def test_translation_matches_a_constant_speed(self, tmp_path, make_radar):
        """A translating glTF node equals a target with the same ``speed``."""
        from radarsimpy.simulator import sim_radar

        radar = self._radar(make_radar)
        points, cells = vertical_quad(1.0)
        path = build_glb(
            tmp_path / "move.glb",
            nodes=[{"name": "plate", "points": points, "cells": cells}],
            channels=[
                {
                    "node": 0,
                    "path": "translation",
                    "times": np.array([0.0, 1.0]),
                    # glTF -Z is RadarSimPy +Y.
                    "values": np.array([[0, 0, 0], [0, 0, -30.0]]),
                }
            ],
        )

        animated = animation_kit.load_animated_targets(
            path, radar, location=(20, 0, 0), permittivity="PEC"
        )
        ref_points, ref_cells = self._reference_plate(1.0)
        reference = [
            {
                "model": {"points": ref_points, "cells": ref_cells},
                "origin": (0, 0, 0),
                "location": (20, 0, 0),
                "speed": (0, 30.0, 0),
                "permittivity": "PEC",
            }
        ]

        from_gltf = sim_radar(radar, animated, density=1, level="sample", device="cpu")[
            "baseband"
        ]
        from_dict = sim_radar(
            radar, reference, density=1, level="sample", device="cpu"
        )["baseband"]

        peak = np.abs(from_dict).max()
        assert peak > 0.0
        assert np.abs(from_gltf - from_dict).max() / peak < 1e-3
