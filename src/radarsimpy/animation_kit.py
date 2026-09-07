"""
Script for loading animated 3D models (glTF 2.0 / GLB)

This script turns a keyframe-animated glTF 2.0 or GLB file into ordinary
RadarSimPy target dictionaries. Each animated node of the model becomes its own
target carrying per-timestamp ``location``/``speed``/``rotation``/
``rotation_rate`` arrays sampled at ``radar.time_prop["timestamp"]``, so a
spinning rotor, a turning wheel or a keyframed flight path is simulated with
the motion authored in the 3D file rather than re-derived by hand.

Only rigid node animation is supported. Skinning, morph targets and animated
scale would require per-timestep vertex geometry, which the ray tracer does not
model; those inputs raise a descriptive error.

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

# pylint: disable=too-many-lines

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .mesh_kit import check_module_installed, safe_import

__all__ = [
    "import_gltf_module",
    "load_animated_model",
    "load_animated_targets",
]


# =============================================================================
# glTF constants
# =============================================================================

#: glTF ``componentType`` enum to numpy dtype.
_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

#: Normalisation divisor for integer component types read as ``normalized``.
_COMPONENT_NORM = {
    5120: 127.0,
    5121: 255.0,
    5122: 32767.0,
    5123: 65535.0,
}

#: glTF accessor ``type`` to component count.
_TYPE_COUNT = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

#: glTF primitive mode for triangles; the only mode a facet ray tracer accepts.
_MODE_TRIANGLES = 4

#: File-unit name to the divisor that converts stored values to meters.
UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}

#: Rotation carrying glTF's Y-up frame into RadarSimPy's Z-up frame (+90 deg
#: about x), as a quaternion in ``(x, y, z, w)`` order.
_Q_YUP_TO_ZUP = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

#: Identity quaternion in ``(x, y, z, w)`` order.
_Q_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


# =============================================================================
# Module discovery
# =============================================================================


def import_gltf_module() -> object:
    """
    Import the glTF parsing module used to read animated models

    :return: The ``pygltflib`` module object
    :rtype: object
    :raises ImportError: If ``pygltflib`` is not installed
    """
    if check_module_installed("pygltflib"):
        module = safe_import("pygltflib")
        if module is not None:
            return module

    raise ImportError(
        "\nglTF Processing Module Required\n"
        "-------------------------------\n"
        "Reading animated 3D models requires the 'pygltflib' package.\n\n"
        "Please install it with:\n"
        "    • pip install pygltflib\n"
    )


# =============================================================================
# Quaternion helpers
#
# Quaternions are stored in glTF's ``(x, y, z, w)`` order throughout. Every
# helper broadcasts over leading axes, so a whole time series is handled in one
# call.
# =============================================================================


def _quat_multiply(quat_a: NDArray, quat_b: NDArray) -> NDArray:
    """
    Hamilton product ``quat_a * quat_b``

    :param numpy.ndarray quat_a: Left quaternion(s), shape ``[..., 4]``
    :param numpy.ndarray quat_b: Right quaternion(s), shape ``[..., 4]``

    :return: The product, shape ``[..., 4]``
    :rtype: numpy.ndarray
    """
    a_x, a_y, a_z, a_w = (quat_a[..., i] for i in range(4))
    b_x, b_y, b_z, b_w = (quat_b[..., i] for i in range(4))
    return np.stack(
        (
            a_w * b_x + a_x * b_w + a_y * b_z - a_z * b_y,
            a_w * b_y - a_x * b_z + a_y * b_w + a_z * b_x,
            a_w * b_z + a_x * b_y - a_y * b_x + a_z * b_w,
            a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z,
        ),
        axis=-1,
    )


def _quat_conjugate(quat: NDArray) -> NDArray:
    """
    Conjugate (inverse, for unit quaternions)

    :param numpy.ndarray quat: Quaternion(s), shape ``[..., 4]``

    :return: The conjugate, shape ``[..., 4]``
    :rtype: numpy.ndarray
    """
    out = np.array(quat, dtype=np.float64, copy=True)
    out[..., :3] *= -1.0
    return out


def _quat_normalize(quat: NDArray) -> NDArray:
    """
    Normalise to unit length, leaving degenerate quaternions as identity

    :param numpy.ndarray quat: Quaternion(s), shape ``[..., 4]``

    :return: Unit quaternion(s), shape ``[..., 4]``
    :rtype: numpy.ndarray
    """
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.where(norm > 0.0, quat / np.where(norm > 0.0, norm, 1.0), _Q_IDENTITY)


def _quat_rotate(quat: NDArray, vec: NDArray) -> NDArray:
    """
    Rotate vectors by unit quaternions

    :param numpy.ndarray quat: Unit quaternion(s), shape ``[..., 4]``
    :param numpy.ndarray vec: Vector(s), shape ``[..., 3]``

    :return: The rotated vector(s), shape ``[..., 3]``
    :rtype: numpy.ndarray
    """
    axis = quat[..., :3]
    scalar = quat[..., 3:4]
    cross1 = np.cross(axis, vec)
    return vec + 2.0 * (scalar * cross1 + np.cross(axis, cross1))


def _quat_to_rsx_euler(quat: NDArray) -> NDArray:
    """
    Convert quaternions to RadarSimPy ``[yaw, pitch, roll]`` angles

    RadarSimPy's ``GetRotateParams`` builds ``R = Rz(yaw) Ry(-pitch) Rx(roll)``,
    which puts ``sin(pitch)`` at ``R[2][0]``. This inverts that exact form, so
    the angles round-trip through the C++ transform.

    :param numpy.ndarray quat: Unit quaternion(s), shape ``[..., 4]``

    :return: Angles in radians, shape ``[..., 3]`` as ``[yaw, pitch, roll]``
    :rtype: numpy.ndarray
    """
    q_x, q_y, q_z, q_w = (quat[..., i] for i in range(4))

    r20 = 2.0 * (q_x * q_z - q_y * q_w)
    r10 = 2.0 * (q_x * q_y + q_z * q_w)
    r00 = 1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    r21 = 2.0 * (q_y * q_z + q_x * q_w)
    r22 = 1.0 - 2.0 * (q_x * q_x + q_y * q_y)

    pitch = np.arcsin(np.clip(r20, -1.0, 1.0))
    yaw = np.arctan2(r10, r00)
    roll = np.arctan2(r21, r22)

    if np.any(np.abs(r20) > 0.9999):
        warnings.warn(
            "Animated model reaches a pitch of +/-90 degrees, where the "
            "yaw/pitch/roll representation is singular. Yaw and roll may be "
            "split arbitrarily between the two axes at those instants.",
            UserWarning,
            stacklevel=3,
        )

    return np.stack((yaw, pitch, roll), axis=-1)


def _rsx_euler_to_quat(rotation: Sequence[float]) -> NDArray:
    """
    Convert RadarSimPy ``[yaw, pitch, roll]`` angles to a quaternion

    :param rotation: Angles in radians as ``[yaw, pitch, roll]``

    :return: Unit quaternion in ``(x, y, z, w)`` order, shape ``[3 -> 4]``
    :rtype: numpy.ndarray
    """
    yaw, pitch, roll = (float(value) for value in rotation)

    half_yaw, half_pitch, half_roll = yaw / 2.0, -pitch / 2.0, roll / 2.0
    quat_z = np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)])
    quat_y = np.array([0.0, np.sin(half_pitch), 0.0, np.cos(half_pitch)])
    quat_x = np.array([np.sin(half_roll), 0.0, 0.0, np.cos(half_roll)])

    return _quat_multiply(_quat_multiply(quat_z, quat_y), quat_x)


def _angular_velocity(quat: NDArray, quat_dot: NDArray) -> NDArray:
    """
    Body-frame angular velocity from a quaternion and its time derivative

    :param numpy.ndarray quat: Unit quaternion(s), shape ``[..., 4]``
    :param numpy.ndarray quat_dot: Time derivative(s), shape ``[..., 4]``

    :return: Angular velocity in the body frame, shape ``[..., 3]``
    :rtype: numpy.ndarray
    """
    return 2.0 * _quat_multiply(_quat_conjugate(quat), quat_dot)[..., :3]


# =============================================================================
# Accessor decoding
# =============================================================================


def _resolve_blob(gltf: Any, gltf_module: Any) -> bytes:
    """
    Return the model's binary data as a single blob

    Normalises GLB payloads, embedded data URIs and external ``.bin`` files to
    one buffer so a single accessor decoder covers every variant.

    :param gltf: The loaded ``GLTF2`` object
    :param gltf_module: The ``pygltflib`` module

    :return: The binary blob backing every buffer view
    :rtype: bytes
    """
    if not gltf.buffers:
        return b""

    if len(gltf.buffers) > 1:
        raise NotImplementedError(
            "Multi-buffer glTF files are not supported. Re-export the model as "
            "a single-buffer .glb."
        )

    gltf.convert_buffers(gltf_module.BufferFormat.BINARYBLOB)
    blob = gltf.binary_blob()
    if blob is None:
        raise RuntimeError("glTF file contains no readable binary buffer data.")
    return blob


def _read_accessor(gltf: Any, blob: bytes, index: int) -> NDArray:
    """
    Decode one glTF accessor into a numpy array

    :param gltf: The loaded ``GLTF2`` object
    :param bytes blob: Binary data from :func:`_resolve_blob`
    :param int index: Accessor index

    :return: Decoded values, shape ``[count, components]``
    :rtype: numpy.ndarray
    """
    accessor = gltf.accessors[index]

    if getattr(accessor, "sparse", None) is not None:
        raise NotImplementedError(
            "Sparse glTF accessors are not supported. Re-export the model "
            "without sparse accessors."
        )

    n_comp = _TYPE_COUNT[accessor.type]
    dtype = np.dtype(_COMPONENT_DTYPE[accessor.componentType]).newbyteorder("<")

    if accessor.bufferView is None:
        # The spec says a missing bufferView means all-zero data.
        return np.zeros((accessor.count, n_comp), dtype=np.float64)

    view = gltf.bufferViews[accessor.bufferView]
    start = (view.byteOffset or 0) + (accessor.byteOffset or 0)
    packed = dtype.itemsize * n_comp
    stride = view.byteStride or packed

    if stride == packed:
        values = np.frombuffer(
            blob, dtype=dtype, count=accessor.count * n_comp, offset=start
        ).reshape(accessor.count, n_comp)
    else:
        span = (accessor.count - 1) * stride + packed
        raw = np.frombuffer(blob, dtype=np.uint8, count=span, offset=start)
        rows = np.lib.stride_tricks.as_strided(
            raw, shape=(accessor.count, packed), strides=(stride, 1)
        )
        values = np.ascontiguousarray(rows).view(dtype).reshape(accessor.count, n_comp)

    values = values.astype(np.float64)
    if getattr(accessor, "normalized", False):
        divisor = _COMPONENT_NORM.get(accessor.componentType)
        if divisor is not None:
            values = np.maximum(values / divisor, -1.0)

    return values


# =============================================================================
# Keyframe tracks
# =============================================================================


class _Track:  # pylint: disable=too-few-public-methods
    """
    One animation channel: keyframe times, values and an interpolation mode

    Evaluation returns both the value and its analytic time derivative. The
    derivative is taken from the interpolation itself rather than by
    differencing the sampled grid, because ``radar.time_prop["timestamp"]``
    is not monotonic across channels or frames and differencing it would spike
    at every boundary.

    :param numpy.ndarray times: Keyframe times (s), shape ``[n]``
    :param numpy.ndarray values: Keyframe values, shape ``[n, c]`` (``[3n, c]``
        for ``CUBICSPLINE``, which stores in-tangent/value/out-tangent triples)
    :param str interpolation: ``STEP``, ``LINEAR`` or ``CUBICSPLINE``
    :param bool is_quaternion: Whether the values are rotation quaternions
    """

    def __init__(
        self,
        times: NDArray,
        values: NDArray,
        interpolation: str,
        is_quaternion: bool,
    ):
        self.times = np.asarray(times, dtype=np.float64).reshape(-1)
        self.interpolation = (interpolation or "LINEAR").upper()
        self.is_quaternion = is_quaternion

        values = np.asarray(values, dtype=np.float64)
        if self.interpolation == "CUBICSPLINE":
            values = values.reshape(len(self.times), 3, -1)
        else:
            values = values.reshape(len(self.times), -1)

        if is_quaternion:
            # Make consecutive keys take the shorter arc, so slerp does not
            # spin the long way around and the derived rate keeps its sign.
            values = self._align_quaternion_signs(values)

        self.values = values

    @staticmethod
    def _align_quaternion_signs(values: NDArray) -> NDArray:
        """
        Flip keyframe quaternions so consecutive keys stay on the same hemisphere

        :param numpy.ndarray values: Keyframe values, shape ``[n, 4]`` or
            ``[n, 3, 4]``

        :return: Sign-aligned values with the same shape
        :rtype: numpy.ndarray
        """
        values = values.copy()
        spline = values.ndim == 3
        for idx in range(1, values.shape[0]):
            previous = values[idx - 1, 1] if spline else values[idx - 1]
            current = values[idx, 1] if spline else values[idx]
            if float(np.dot(previous, current)) < 0.0:
                values[idx] *= -1.0
        return values

    def _segments(self, times: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Locate each query time inside the keyframe list

        :param numpy.ndarray times: Query times (s), shape ``[k]``

        :return: ``(left_index, right_index, normalised_position)``
        :rtype: tuple
        """
        n_keys = len(self.times)
        if n_keys == 1:
            zeros = np.zeros(times.shape, dtype=np.intp)
            return zeros, zeros, np.zeros(times.shape)

        right = np.clip(np.searchsorted(self.times, times, side="right"), 1, n_keys - 1)
        left = right - 1
        span = self.times[right] - self.times[left]
        span = np.where(span > 0.0, span, 1.0)
        # Clamping outside the keyframe range holds the end pose, per the spec.
        frac = np.clip((times - self.times[left]) / span, 0.0, 1.0)
        return left, right, frac

    def evaluate(self, times: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Sample the track and its time derivative

        :param numpy.ndarray times: Query times (s), shape ``[k]``

        :return: ``(value, d value / d t)``, each shaped ``[k, c]``
        :rtype: tuple
        """
        times = np.asarray(times, dtype=np.float64)
        left, right, frac = self._segments(times)

        if len(self.times) == 1 or self.interpolation == "STEP":
            value = self.values[left, 1] if self.values.ndim == 3 else self.values[left]
            return np.array(value), np.zeros_like(value)

        span = (self.times[right] - self.times[left])[..., None]
        span = np.where(span > 0.0, span, 1.0)
        # Outside the keyframe range the pose is held, so every rate is zero.
        held = self._inside(times)[..., None]

        if self.interpolation == "CUBICSPLINE":
            value, value_dot = self._evaluate_cubicspline(left, right, frac, span)
            value_dot = np.where(held, value_dot, 0.0)
            if self.is_quaternion:
                return self._normalize_with_derivative(value, value_dot)
            return value, value_dot

        if self.is_quaternion:
            return self._evaluate_slerp(left, right, frac, span, held)

        value_0 = self.values[left]
        value_1 = self.values[right]
        value = value_0 + (value_1 - value_0) * frac[..., None]
        value_dot = np.where(held, (value_1 - value_0) / span, 0.0)
        return value, value_dot

    def _inside(self, times: NDArray) -> NDArray:
        """
        Mask of query times that fall within the keyframe range

        :param numpy.ndarray times: Query times (s), shape ``[k]``

        :return: Boolean mask, shape ``[k]``
        :rtype: numpy.ndarray
        """
        return (times >= self.times[0]) & (times <= self.times[-1])

    def _evaluate_cubicspline(
        self, left: NDArray, right: NDArray, frac: NDArray, span: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Evaluate a cubic Hermite segment and its derivative

        :param numpy.ndarray left: Left keyframe indices
        :param numpy.ndarray right: Right keyframe indices
        :param numpy.ndarray frac: Normalised position inside the segment
        :param numpy.ndarray span: Segment duration (s), shape ``[k, 1]``

        :return: ``(value, d value / d t)``
        :rtype: tuple
        """
        pos = frac[..., None]
        pos_sq = pos * pos
        pos_cu = pos_sq * pos

        value_0 = self.values[left, 1]
        out_tangent_0 = self.values[left, 2] * span
        value_1 = self.values[right, 1]
        in_tangent_1 = self.values[right, 0] * span

        value = (
            (2.0 * pos_cu - 3.0 * pos_sq + 1.0) * value_0
            + (pos_cu - 2.0 * pos_sq + pos) * out_tangent_0
            + (-2.0 * pos_cu + 3.0 * pos_sq) * value_1
            + (pos_cu - pos_sq) * in_tangent_1
        )
        value_dot = (
            (6.0 * pos_sq - 6.0 * pos) * value_0
            + (3.0 * pos_sq - 4.0 * pos + 1.0) * out_tangent_0
            + (-6.0 * pos_sq + 6.0 * pos) * value_1
            + (3.0 * pos_sq - 2.0 * pos) * in_tangent_1
        ) / span

        return value, value_dot

    def _evaluate_slerp(  # pylint: disable=too-many-locals
        self,
        left: NDArray,
        right: NDArray,
        frac: NDArray,
        span: NDArray,
        held: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Spherical linear interpolation of quaternions with its derivative

        :param numpy.ndarray left: Left keyframe indices
        :param numpy.ndarray right: Right keyframe indices
        :param numpy.ndarray frac: Normalised position inside the segment
        :param numpy.ndarray span: Segment duration (s), shape ``[k, 1]``
        :param numpy.ndarray held: Mask of times inside the keyframe range,
            shape ``[k, 1]``; the rate is zero outside it

        :return: ``(quaternion, d quaternion / d t)``
        :rtype: tuple
        """
        quat_0 = self.values[left]
        quat_1 = self.values[right]

        # Relative rotation, expressed in quat_0's body frame. Slerp advances
        # about a fixed body axis, so the body-frame rate is constant.
        relative = _quat_multiply(_quat_conjugate(quat_0), quat_1)
        vec_norm = np.linalg.norm(relative[..., :3], axis=-1)
        angle = 2.0 * np.arctan2(vec_norm, np.abs(relative[..., 3]))
        safe_norm = np.where(vec_norm > 1e-12, vec_norm, 1.0)
        axis = relative[..., :3] / safe_norm[..., None]
        axis = np.where((vec_norm > 1e-12)[..., None], axis, 0.0)
        # ``relative`` already takes the short arc via the keyframe sign
        # alignment, but guard the per-segment sign too.
        axis = axis * np.sign(
            np.where(relative[..., 3:4] == 0.0, 1.0, relative[..., 3:4])
        )

        half = (angle * frac / 2.0)[..., None]
        step = np.concatenate((axis * np.sin(half), np.cos(half)), axis=-1)
        quat = _quat_multiply(quat_0, step)

        rate_body = np.where(held, axis * (angle / span[..., 0])[..., None], 0.0)
        quat_dot = 0.5 * _quat_multiply(
            quat,
            np.concatenate((rate_body, np.zeros_like(rate_body[..., :1])), axis=-1),
        )
        return _quat_normalize(quat), quat_dot

    @staticmethod
    def _normalize_with_derivative(
        quat: NDArray, quat_dot: NDArray
    ) -> Tuple[NDArray, NDArray]:
        """
        Normalise a quaternion and correct its derivative for the normalisation

        ``CUBICSPLINE`` quaternion output is not unit length, so the derivative
        of ``q/|q|`` must account for the changing magnitude.

        :param numpy.ndarray quat: Raw quaternion(s), shape ``[..., 4]``
        :param numpy.ndarray quat_dot: Raw derivative(s), shape ``[..., 4]``

        :return: ``(unit quaternion, corrected derivative)``
        :rtype: tuple
        """
        norm = np.linalg.norm(quat, axis=-1, keepdims=True)
        norm = np.where(norm > 0.0, norm, 1.0)
        unit = quat / norm
        projection = np.sum(quat * quat_dot, axis=-1, keepdims=True)
        return unit, quat_dot / norm - quat * projection / norm**3


# =============================================================================
# Scene graph
# =============================================================================


def _node_rest_pose(node: Any) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Rest-pose translation, rotation and scale of one node

    :param node: A glTF node

    :return: ``(translation[3], quaternion[4], scale[3])``
    :rtype: tuple
    """
    matrix = getattr(node, "matrix", None)
    if matrix:
        # glTF stores matrices column-major.
        mat = np.asarray(matrix, dtype=np.float64).reshape(4, 4).T
        translation = mat[:3, 3]
        basis = mat[:3, :3]
        scale = np.linalg.norm(basis, axis=0)
        safe = np.where(scale > 0.0, scale, 1.0)
        return translation, _quat_from_matrix(basis / safe), scale

    translation = np.asarray(node.translation or (0.0, 0.0, 0.0), dtype=np.float64)
    rotation = np.asarray(node.rotation or (0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    scale = np.asarray(node.scale or (1.0, 1.0, 1.0), dtype=np.float64)
    return translation, _quat_normalize(rotation), scale


def _quat_from_matrix(basis: NDArray) -> NDArray:
    """
    Convert an orthonormal 3x3 rotation matrix to a quaternion

    :param numpy.ndarray basis: Rotation matrix, shape ``[3, 3]``

    :return: Unit quaternion in ``(x, y, z, w)`` order
    :rtype: numpy.ndarray
    """
    trace = float(np.trace(basis))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (basis[2, 1] - basis[1, 2]) / scale,
                (basis[0, 2] - basis[2, 0]) / scale,
                (basis[1, 0] - basis[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(basis)))
        idx_1, idx_2 = (axis + 1) % 3, (axis + 2) % 3
        scale = (
            np.sqrt(1.0 + basis[axis, axis] - basis[idx_1, idx_1] - basis[idx_2, idx_2])
            * 2.0
        )
        quat = np.zeros(4)
        quat[axis] = 0.25 * scale
        quat[idx_1] = (basis[idx_1, axis] + basis[axis, idx_1]) / scale
        quat[idx_2] = (basis[idx_2, axis] + basis[axis, idx_2]) / scale
        quat[3] = (basis[idx_2, idx_1] - basis[idx_1, idx_2]) / scale
    return _quat_normalize(quat)


def _build_hierarchy(gltf: Any) -> Tuple[Dict[int, Optional[int]], List[int]]:
    """
    Map every node to its parent and return the default scene's roots

    :param gltf: The loaded ``GLTF2`` object

    :return: ``(parent_of, roots)``
    :rtype: tuple
    :raises ValueError: If the node graph contains a cycle
    """
    parent_of: Dict[int, Optional[int]] = {}
    scene_index = gltf.scene if gltf.scene is not None else 0
    if gltf.scenes:
        roots = list(gltf.scenes[scene_index].nodes or [])
    else:
        roots = list(range(len(gltf.nodes)))

    stack = [(root, None) for root in reversed(roots)]
    while stack:
        index, parent = stack.pop()
        if index in parent_of:
            raise ValueError(
                f"glTF node graph is not a tree: node {index} has more than one parent."
            )
        parent_of[index] = parent
        for child in gltf.nodes[index].children or []:
            stack.append((child, index))

    return parent_of, roots


def _select_animation(gltf: Any, animation: Optional[Union[int, str]]) -> Optional[int]:
    """
    Resolve an animation name or index to an index

    :param gltf: The loaded ``GLTF2`` object
    :param animation: Animation name, index, or ``None`` for the first one

    :return: The animation index, or ``None`` if the file has no animations
    :rtype: int or None
    :raises ValueError: If the requested animation does not exist
    """
    if not gltf.animations:
        return None

    if animation is None:
        return 0

    if isinstance(animation, (int, np.integer)):
        index = int(animation)
        if not 0 <= index < len(gltf.animations):
            raise ValueError(
                f"Animation index {index} is out of range; the model has "
                f"{len(gltf.animations)} animation(s)."
            )
        return index

    names = [clip.name for clip in gltf.animations]
    if animation not in names:
        raise ValueError(
            f"Animation '{animation}' not found. Available animations: {names}"
        )
    return names.index(animation)


def _collect_tracks(
    gltf: Any, blob: bytes, animation_index: Optional[int]
) -> Dict[int, Dict[str, _Track]]:
    """
    Decode every animation channel into per-node tracks

    :param gltf: The loaded ``GLTF2`` object
    :param bytes blob: Binary data from :func:`_resolve_blob`
    :param animation_index: Index of the animation to read, or ``None``

    :return: ``{node index: {path: track}}``
    :rtype: dict
    :raises NotImplementedError: If a channel animates weights or scale
    """
    tracks: Dict[int, Dict[str, _Track]] = {}
    if animation_index is None:
        return tracks

    clip = gltf.animations[animation_index]
    for channel in clip.channels:
        node = channel.target.node
        path = channel.target.path
        if node is None:
            continue

        if path == "weights":
            raise NotImplementedError(
                "Morph-target ('weights') animation is not supported: it "
                "deforms the mesh, which the ray tracer treats as rigid. "
                "Bake the deformation into separate rigid parts instead."
            )
        if path == "scale":
            raise NotImplementedError(
                "Animated scale is not supported: it deforms the mesh, which "
                "the ray tracer treats as rigid."
            )

        sampler = clip.samplers[channel.sampler]
        tracks.setdefault(node, {})[path] = _Track(
            times=_read_accessor(gltf, blob, sampler.input),
            values=_read_accessor(gltf, blob, sampler.output),
            interpolation=sampler.interpolation,
            is_quaternion=(path == "rotation"),
        )

    return tracks


def _clip_time_range(tracks: Dict[int, Dict[str, _Track]]) -> Tuple[float, float]:
    """
    Earliest and latest keyframe time across every track

    :param dict tracks: Tracks from :func:`_collect_tracks`

    :return: ``(start, end)`` in seconds
    :rtype: tuple
    """
    starts, ends = [], []
    for node_tracks in tracks.values():
        for track in node_tracks.values():
            starts.append(track.times[0])
            ends.append(track.times[-1])
    if not starts:
        return 0.0, 0.0
    return float(min(starts)), float(max(ends))


# =============================================================================
# Geometry extraction
# =============================================================================


def _node_geometry(gltf: Any, blob: bytes, node: Any) -> Tuple[NDArray, NDArray]:
    """
    Concatenate the triangle geometry of one node's mesh

    :param gltf: The loaded ``GLTF2`` object
    :param bytes blob: Binary data from :func:`_resolve_blob`
    :param node: A glTF node

    :return: ``(points[n, 3], cells[m, 3])`` in the node's own frame
    :rtype: tuple
    :raises NotImplementedError: If the mesh is skinned or not triangulated
    """
    if node.mesh is None:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    if getattr(node, "skin", None) is not None:
        raise NotImplementedError(
            "Skinned meshes are not supported: skinning moves individual "
            "vertices, while the ray tracer transforms each target rigidly. "
            "Split the model into rigid parts and animate their nodes instead."
        )

    points_list, cells_list, offset = [], [], 0
    for primitive in gltf.meshes[node.mesh].primitives:
        mode = _MODE_TRIANGLES if primitive.mode is None else primitive.mode
        if mode != _MODE_TRIANGLES:
            raise NotImplementedError(
                f"glTF primitive mode {mode} is not supported; only triangles "
                "(mode 4) can be ray traced. Triangulate the model on export."
            )

        attributes = primitive.attributes
        if getattr(attributes, "JOINTS_0", None) is not None:
            raise NotImplementedError(
                "Skinned meshes (JOINTS_0/WEIGHTS_0 attributes) are not "
                "supported: skinning moves individual vertices, while the ray "
                "tracer transforms each target rigidly."
            )
        if primitive.targets:
            raise NotImplementedError(
                "Morph targets are not supported: they deform the mesh, which "
                "the ray tracer treats as rigid."
            )

        points = _read_accessor(gltf, blob, attributes.POSITION)
        if primitive.indices is None:
            cells = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
        else:
            cells = _read_accessor(gltf, blob, primitive.indices)
            cells = cells.astype(np.int64).reshape(-1, 3)

        points_list.append(points)
        cells_list.append(cells + offset)
        offset += len(points)

    if not points_list:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

    return np.concatenate(points_list, axis=0), np.concatenate(cells_list, axis=0)


def _apply_affine(
    points: NDArray, translation: NDArray, quat: NDArray, scale: NDArray
) -> NDArray:
    """
    Apply a TRS transform to a point cloud

    :param numpy.ndarray points: Points, shape ``[n, 3]``
    :param numpy.ndarray translation: Translation, shape ``[3]``
    :param numpy.ndarray quat: Rotation quaternion, shape ``[4]``
    :param numpy.ndarray scale: Scale, shape ``[3]``

    :return: The transformed points, shape ``[n, 3]``
    :rtype: numpy.ndarray
    """
    if len(points) == 0:
        return points
    return _quat_rotate(quat, points * scale) + translation


# =============================================================================
# Model loading
# =============================================================================


def load_animated_model(  # pylint: disable=too-many-locals
    filename: str,
    *,
    animation: Optional[Union[int, str]] = None,
    unit: str = "m",
    up_axis: str = "y",
    merge_static: bool = True,
) -> Dict[str, Any]:
    """
    Load an animated glTF 2.0 / GLB model and split it into rigid parts

    Geometry is grouped by the nearest animated ancestor node, so each returned
    part moves as one rigid body. Parts are expressed in their own node frame,
    which is the frame their sampled pose applies to.

    :param str filename: Path to a ``.gltf`` or ``.glb`` file
    :param animation: Animation name or index to read. Default: the first one.
    :param str unit: Unit the file is authored in (``'m'``, ``'cm'``, ``'mm'``).
        Default: ``'m'``, which is what the glTF specification mandates.
    :param str up_axis: Up axis of the file, ``'y'`` (glTF default) or ``'z'``.
        Geometry and motion are rotated into RadarSimPy's Z-up frame.
    :param bool merge_static: Merge all non-animated geometry into a single
        part. Default: ``True``.

    :return: Dictionary containing:

        * **parts** (*list*): One entry per rigid part, each a dict with
          ``name``, ``points``, ``cells``, ``node`` and ``animated``.
        * **animation** (*str or None*): Name of the selected animation.
        * **animations** (*list*): Names of every animation in the file.
        * **duration** (*float*): Length of the selected animation in seconds.
        * **start_time** (*float*): Time of its first keyframe in seconds.
    :rtype: dict
    :raises NotImplementedError: For skinning, morph targets, animated scale,
        non-uniform scale on an animated node, or non-triangle primitives.
    """
    if unit not in UNIT_SCALE:
        raise ValueError(
            f"Invalid unit '{unit}'. Supported units: {list(UNIT_SCALE.keys())}"
        )
    if up_axis not in ("y", "z"):
        raise ValueError(f"Invalid up_axis '{up_axis}'. Supported: 'y', 'z'.")

    gltf_module = import_gltf_module()
    gltf = gltf_module.GLTF2().load(filename)
    if gltf is None:
        raise RuntimeError(f"Failed to load glTF model '{filename}'.")

    blob = _resolve_blob(gltf, gltf_module)
    parent_of, roots = _build_hierarchy(gltf)
    animation_index = _select_animation(gltf, animation)
    tracks = _collect_tracks(gltf, blob, animation_index)
    animated_nodes = set(tracks)

    scale_divisor = UNIT_SCALE[unit]
    rest_poses = {index: _node_rest_pose(gltf.nodes[index]) for index in parent_of}

    _validate_animated_scale(gltf, parent_of, rest_poses, animated_nodes)

    # Walk the tree, accumulating each mesh into the bucket of its nearest
    # animated ancestor. ``local`` is the transform from that ancestor down to
    # the current node, which is constant and can be baked into the vertices.
    # ``cumulative`` is the scale from the root, tracked separately because an
    # animated node's own scale still has to reach its vertices even though its
    # rotation and translation come from the sampled pose instead.
    buckets: Dict[Optional[int], List[Tuple[NDArray, NDArray]]] = {}
    stack = [
        (root, None, (np.zeros(3), _Q_IDENTITY.copy(), np.ones(3)), np.ones(3))
        for root in reversed(roots)
    ]
    while stack:
        index, owner, local, cumulative = stack.pop()
        translation, quat, scale = rest_poses[index]
        cumulative = cumulative * scale

        if index in animated_nodes:
            owner = index
            local = (np.zeros(3), _Q_IDENTITY.copy(), cumulative.copy())
        else:
            local = _compose_affine(local, (translation, quat, scale))

        node = gltf.nodes[index]
        if node.mesh is not None:
            points, cells = _node_geometry(gltf, blob, node)
            if len(cells):
                buckets.setdefault(owner, []).append(
                    (_apply_affine(points, *local) / scale_divisor, cells)
                )

        for child in node.children or []:
            stack.append((child, owner, local, cumulative))

    parts = _assemble_parts(gltf, buckets, animated_nodes, merge_static)

    start_time, end_time = _clip_time_range(tracks)
    return {
        "parts": parts,
        "animation": (
            gltf.animations[animation_index].name
            if animation_index is not None
            else None
        ),
        "animations": [clip.name for clip in gltf.animations],
        "duration": end_time - start_time,
        "start_time": start_time,
        "_tracks": tracks,
        "_parent_of": parent_of,
        "_rest_poses": rest_poses,
        "_scale_divisor": scale_divisor,
        "_up_axis": up_axis,
    }


def _compose_affine(
    outer: Tuple[NDArray, NDArray, NDArray], inner: Tuple[NDArray, NDArray, NDArray]
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compose two TRS transforms, outer applied after inner

    :param tuple outer: ``(translation, quaternion, scale)`` of the parent
    :param tuple inner: ``(translation, quaternion, scale)`` of the child

    :return: The composed ``(translation, quaternion, scale)``
    :rtype: tuple
    """
    out_t, out_q, out_s = outer
    in_t, in_q, in_s = inner
    return (
        out_t + _quat_rotate(out_q, in_t * out_s),
        _quat_multiply(out_q, in_q),
        out_s * in_s,
    )


def _validate_animated_scale(
    gltf: Any,
    parent_of: Dict[int, Optional[int]],
    rest_poses: Dict[int, Tuple[NDArray, NDArray, NDArray]],
    animated_nodes: set,
) -> None:
    """
    Reject non-uniform scale anywhere on an animated node's ancestor chain

    A part's pose is emitted as a rotation plus a translation, so a non-uniform
    scale above it cannot be represented and would silently distort the mesh.

    :param gltf: The loaded ``GLTF2`` object
    :param dict parent_of: Node to parent map
    :param dict rest_poses: Node to rest ``(translation, quaternion, scale)``
    :param set animated_nodes: Indices of animated nodes

    :raises NotImplementedError: If such a scale is found
    """
    for node_index in animated_nodes:
        index: Optional[int] = node_index
        while index is not None:
            scale = rest_poses[index][2]
            if not np.allclose(scale, scale[0], rtol=1e-6, atol=1e-9):
                name = gltf.nodes[index].name or f"node {index}"
                raise NotImplementedError(
                    f"Node '{name}' has a non-uniform scale {tuple(scale)} on the "
                    "transform chain of an animated node. Only uniform scale is "
                    "supported there, because a part's pose is emitted as a "
                    "rotation plus a translation. Apply the scale to the mesh on "
                    "export."
                )
            index = parent_of[index]


def _assemble_parts(
    gltf: Any,
    buckets: Dict[Optional[int], List[Tuple[NDArray, NDArray]]],
    animated_nodes: set,
    merge_static: bool,
) -> List[Dict[str, Any]]:
    """
    Turn per-owner geometry buckets into part records

    :param gltf: The loaded ``GLTF2`` object
    :param dict buckets: ``{owner node or None: [(points, cells), ...]}``
    :param set animated_nodes: Indices of animated nodes
    :param bool merge_static: Merge every static mesh into one part

    :return: Part records
    :rtype: list
    """
    parts: List[Dict[str, Any]] = []

    for owner in sorted(index for index in buckets if index is not None):
        points, cells = _merge_geometry(buckets[owner])
        parts.append(
            {
                "name": gltf.nodes[owner].name or f"node_{owner}",
                "points": points,
                "cells": cells,
                "node": owner,
                "animated": owner in animated_nodes,
            }
        )

    static = buckets.get(None, [])
    if static:
        if merge_static:
            points, cells = _merge_geometry(static)
            parts.append(
                {
                    "name": "static",
                    "points": points,
                    "cells": cells,
                    "node": None,
                    "animated": False,
                }
            )
        else:
            for order, (points, cells) in enumerate(static):
                parts.append(
                    {
                        "name": f"static_{order}",
                        "points": points,
                        "cells": cells,
                        "node": None,
                        "animated": False,
                    }
                )

    return parts


def _merge_geometry(chunks: List[Tuple[NDArray, NDArray]]) -> Tuple[NDArray, NDArray]:
    """
    Concatenate ``(points, cells)`` chunks, offsetting the face indices

    :param list chunks: ``[(points, cells), ...]``

    :return: ``(points, cells)``
    :rtype: tuple
    """
    points_list, cells_list, offset = [], [], 0
    for points, cells in chunks:
        points_list.append(points)
        cells_list.append(cells + offset)
        offset += len(points)
    return np.concatenate(points_list, axis=0), np.concatenate(cells_list, axis=0)


# =============================================================================
# Pose sampling
# =============================================================================


def _sample_node_pose(  # pylint: disable=too-many-locals
    node_index: int,
    model: Dict[str, Any],
    anim_times: NDArray,
    time_scale: float,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    World pose and velocity of one node, sampled at the given animation times

    Composes the ancestor chain in ``(quaternion, translation)`` space and
    propagates the exact rigid-body velocity recursions

    .. math::

        v_w &= v_p + \\omega_p \\times (R_p\\, s_p\\, t_l) + R_p\\, s_p\\, \\dot{t}_l \\\\
        \\omega_w &= \\omega_p + R_w\\, \\omega_{body}

    so linear and angular rates stay exact rather than being differenced off
    the sample grid.

    :param int node_index: Node to sample
    :param dict model: Model from :func:`load_animated_model`
    :param numpy.ndarray anim_times: Animation times (s), shape ``[k]``
    :param float time_scale: ``d(animation time) / d(simulation time)``

    :return: ``(translation[k,3], quaternion[k,4], velocity[k,3],
        angular_velocity[k,3])`` in the file's own frame and units
    :rtype: tuple
    """
    chain: List[int] = []
    index: Optional[int] = node_index
    while index is not None:
        chain.append(index)
        index = model["_parent_of"][index]
    chain.reverse()

    n_times = len(anim_times)
    world_t = np.zeros((n_times, 3))
    world_q = np.tile(_Q_IDENTITY, (n_times, 1))
    world_v = np.zeros((n_times, 3))
    world_w = np.zeros((n_times, 3))
    world_scale = 1.0

    for index in chain:
        rest_t, rest_q, rest_s = model["_rest_poses"][index]
        node_tracks = model["_tracks"].get(index, {})

        if "translation" in node_tracks:
            local_t, local_t_dot = node_tracks["translation"].evaluate(anim_times)
            local_t_dot = local_t_dot * time_scale
        else:
            local_t = np.broadcast_to(rest_t, (n_times, 3))
            local_t_dot = np.zeros((n_times, 3))

        if "rotation" in node_tracks:
            local_q, local_q_dot = node_tracks["rotation"].evaluate(anim_times)
            local_q = _quat_normalize(local_q)
            rate_body = _angular_velocity(local_q, local_q_dot) * time_scale
        else:
            local_q = np.broadcast_to(rest_q, (n_times, 4))
            rate_body = np.zeros((n_times, 3))

        offset = _quat_rotate(world_q, local_t * world_scale)
        world_v = (
            world_v
            + np.cross(world_w, offset)
            + _quat_rotate(world_q, local_t_dot * world_scale)
        )
        world_t = world_t + offset
        world_q = _quat_multiply(world_q, local_q)
        world_w = world_w + _quat_rotate(world_q, rate_body)
        world_scale = world_scale * float(rest_s[0])

    return world_t, world_q, world_v, world_w


def _to_zup(
    translation: NDArray, quat: NDArray, velocity: NDArray, omega: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Rotate a pose and its rates from glTF's Y-up frame into RadarSimPy's Z-up frame

    A world point is ``C (R p + t) = (C R C^T)(C p) + C t``, so vectors are
    simply rotated by ``C`` while the orientation takes the similarity
    transform ``C R C^T`` — the part's own vertices are rotated by ``C``
    separately in :func:`_orient_points`.

    :param numpy.ndarray translation: Translation, shape ``[k, 3]``
    :param numpy.ndarray quat: Rotation, shape ``[k, 4]``
    :param numpy.ndarray velocity: Linear velocity, shape ``[k, 3]``
    :param numpy.ndarray omega: Angular velocity, shape ``[k, 3]``

    :return: The same quantities in the Z-up frame
    :rtype: tuple
    """
    return (
        _quat_rotate(_Q_YUP_TO_ZUP, translation),
        _quat_multiply(
            _quat_multiply(_Q_YUP_TO_ZUP, quat), _quat_conjugate(_Q_YUP_TO_ZUP)
        ),
        _quat_rotate(_Q_YUP_TO_ZUP, velocity),
        _quat_rotate(_Q_YUP_TO_ZUP, omega),
    )


def _map_time(
    timestamp: NDArray,
    model: Dict[str, Any],
    time_offset: float,
    time_scale: float,
    loop: bool,
) -> NDArray:
    """
    Map simulation time to animation time, wrapping or clamping to the clip

    :param numpy.ndarray timestamp: Simulation timestamps (s)
    :param dict model: Model from :func:`load_animated_model`
    :param float time_offset: Animation time at simulation time zero (s)
    :param float time_scale: Playback rate
    :param bool loop: Wrap past the end of the clip instead of holding the last pose

    :return: Animation times (s), same shape as ``timestamp``
    :rtype: numpy.ndarray
    """
    anim_times = np.asarray(timestamp, dtype=np.float64) * time_scale + time_offset
    duration = model["duration"]
    if loop and duration > 0.0:
        anim_times = model["start_time"] + np.mod(
            anim_times - model["start_time"], duration
        )
    return anim_times


# =============================================================================
# Target generation
# =============================================================================


def load_animated_targets(  # pylint: disable=too-many-arguments, too-many-locals
    filename: Union[str, Dict[str, Any]],
    radar: Any = None,
    *,
    at_time: Optional[float] = None,
    animation: Optional[Union[int, str]] = None,
    time_offset: float = 0.0,
    time_scale: float = 1.0,
    loop: bool = True,
    location: Sequence[float] = (0, 0, 0),
    speed: Sequence[float] = (0, 0, 0),
    rotation: Sequence[float] = (0, 0, 0),
    rotation_rate: Sequence[float] = (0, 0, 0),
    unit: str = "m",
    up_axis: str = "y",
    merge_static: bool = True,
    **target_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Build RadarSimPy target dictionaries from an animated glTF 2.0 / GLB model

    Each animated node of the model becomes one target whose ``location``,
    ``speed``, ``rotation`` and ``rotation_rate`` are arrays sampled at
    ``radar.time_prop["timestamp"]``, so the motion authored in the file drives
    both the geometry and the Doppler. Non-animated geometry becomes a single
    static target.

    The returned list is passed straight to :func:`radarsimpy.sim_radar`:

    >>> targets = load_animated_targets("turbine.glb", radar, location=(50, 0, 0))
    >>> data = sim_radar(radar, targets)

    Only rigid node animation is supported. Skinned meshes, morph targets and
    animated scale raise :class:`NotImplementedError`, because the ray tracer
    transforms each target rigidly and cannot move individual vertices.

    .. note::
        Every animated part allocates twelve ``float32`` arrays the size of
        ``radar.time_prop["timestamp"]``. Models with many independently moving
        parts, simulated over long time records, use a correspondingly large
        amount of memory. Merge parts that do not move independently to reduce it.

    :param filename: Path to a ``.gltf``/``.glb`` file, or a model already
        returned by :func:`load_animated_model`.
    :param radar: :class:`radarsimpy.Radar` whose timestamps the animation is
        sampled at. Required unless ``at_time`` is given.
    :param at_time: Sample a single instant instead, returning static targets.
        Use this for :func:`radarsimpy.sim_rcs` and :func:`radarsimpy.sim_lidar`,
        which do not accept time-varying motion.
    :param animation: Animation name or index. Default: the first one.
    :param float time_offset: Animation time at simulation time zero (s).
    :param float time_scale: Playback rate; ``2.0`` plays twice as fast.
    :param bool loop: Wrap the animation when the simulation outlasts the clip.
        When ``False`` the first and last poses are held. Default: ``True``.
    :param location: Position of the whole model in the global frame [x, y, z] (m).
    :param speed: Velocity of the whole model [vx, vy, vz] (m/s).
    :param rotation: Orientation of the whole model [yaw, pitch, roll] (deg).
    :param rotation_rate: Rotation rate of the whole model
        [yaw rate, pitch rate, roll rate] (deg/s).
    :param str unit: Unit the file is authored in. Default: ``'m'``.
    :param str up_axis: Up axis of the file, ``'y'`` or ``'z'``. Default: ``'y'``.
    :param bool merge_static: Merge all non-animated geometry into one target.
    :param target_kwargs: Extra keys copied into every target dictionary, such
        as ``permittivity``, ``permeability``, ``skip_diffusion``, ``density``
        and ``environment``.

    :return: Target dictionaries ready for :func:`radarsimpy.sim_radar`
    :rtype: list
    :raises ValueError: If neither ``radar`` nor ``at_time`` is provided
    """
    if isinstance(filename, dict):
        model = filename
    else:
        model = load_animated_model(
            filename,
            animation=animation,
            unit=unit,
            up_axis=up_axis,
            merge_static=merge_static,
        )

    if radar is not None:
        timestamp = np.asarray(radar.time_prop["timestamp"], dtype=np.float64)
        if timestamp.ndim != 3:
            raise ValueError(
                "radar.time_prop['timestamp'] must be 3-D "
                f"[channels, pulses, samples]; got shape {timestamp.shape}."
            )
    elif at_time is not None:
        timestamp = np.asarray([[[float(at_time)]]], dtype=np.float64)
    else:
        raise ValueError(
            "Provide 'radar' to sample the animation at the simulation "
            "timestamps, or 'at_time' to take a single static snapshot."
        )

    static_only = radar is None
    ts_shape = timestamp.shape
    anim_times = _map_time(timestamp, model, time_offset, time_scale, loop).reshape(-1)
    flat_time = timestamp.reshape(-1)

    outer = _outer_transform(location, speed, rotation, rotation_rate, flat_time)

    targets: List[Dict[str, Any]] = []
    for part in model["parts"]:
        if len(part["cells"]) == 0:
            continue

        points = _orient_points(part["points"], model["_up_axis"])

        if part["animated"]:
            part_t, part_q, part_v, part_w = _sample_node_pose(
                part["node"], model, anim_times, time_scale
            )
            # ``_sample_node_pose`` works in the file's own units and frame.
            part_t = part_t / model["_scale_divisor"]
            part_v = part_v / model["_scale_divisor"]
            if model["_up_axis"] == "y":
                part_t, part_q, part_v, part_w = _to_zup(part_t, part_q, part_v, part_w)
        else:
            part_t = np.zeros((len(anim_times), 3))
            part_q = np.tile(_Q_IDENTITY, (len(anim_times), 1))
            part_v = np.zeros((len(anim_times), 3))
            part_w = np.zeros((len(anim_times), 3))

        targets.append(
            _build_target(
                points=points,
                cells=part["cells"],
                part_pose=(part_t, part_q, part_v, part_w),
                outer=outer,
                ts_shape=ts_shape,
                static_only=static_only,
                target_kwargs=target_kwargs,
            )
        )

    return targets


def _orient_points(points: NDArray, up_axis: str) -> NDArray:
    """
    Rotate part geometry into RadarSimPy's Z-up frame

    :param numpy.ndarray points: Points, shape ``[n, 3]``
    :param str up_axis: Up axis of the source file

    :return: Points in the Z-up frame, shape ``[n, 3]``
    :rtype: numpy.ndarray
    """
    if up_axis == "z" or len(points) == 0:
        return points
    return _quat_rotate(_Q_YUP_TO_ZUP, points)


def _outer_transform(
    location: Sequence[float],
    speed: Sequence[float],
    rotation: Sequence[float],
    rotation_rate: Sequence[float],
    flat_time: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Sample the user's placement transform for the whole model

    The angular velocity is taken as ``[roll_rate, pitch_rate, yaw_rate]`` in
    the world frame, matching how ``CalculateSpeed`` in RadarSimCpp interprets
    a constant ``rotation_rate``. That reading is exact for rotation about a
    single axis, which is the case this placement argument is meant for.

    :param location: Model position [x, y, z] (m)
    :param speed: Model velocity [vx, vy, vz] (m/s)
    :param rotation: Model orientation [yaw, pitch, roll] (deg)
    :param rotation_rate: Model rotation rate [yaw, pitch, roll] rate (deg/s)
    :param numpy.ndarray flat_time: Simulation timestamps, shape ``[k]``

    :return: ``(translation[k,3], quaternion[k,4], velocity[k,3],
        angular_velocity[k,3])``
    :rtype: tuple
    """
    location = np.asarray(location, dtype=np.float64).reshape(3)
    speed = np.asarray(speed, dtype=np.float64).reshape(3)
    rotation = np.radians(np.asarray(rotation, dtype=np.float64).reshape(3))
    rotation_rate = np.radians(np.asarray(rotation_rate, dtype=np.float64).reshape(3))

    n_times = len(flat_time)
    translation = location + np.outer(flat_time, speed)
    velocity = np.broadcast_to(speed, (n_times, 3))

    if np.any(rotation_rate):
        angles = rotation + np.outer(flat_time, rotation_rate)
        quat = np.stack([_rsx_euler_to_quat(angle) for angle in angles])
    else:
        quat = np.tile(_rsx_euler_to_quat(rotation), (n_times, 1))

    omega = np.broadcast_to(
        np.array([rotation_rate[2], rotation_rate[1], rotation_rate[0]]), (n_times, 3)
    )
    return translation, quat, velocity, omega


def _build_target(  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    points: NDArray,
    cells: NDArray,
    part_pose: Tuple[NDArray, NDArray, NDArray, NDArray],
    outer: Tuple[NDArray, NDArray, NDArray, NDArray],
    ts_shape: Tuple[int, int, int],
    static_only: bool,
    target_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compose a part's pose with the model placement and emit a target dictionary

    :param numpy.ndarray points: Part geometry, shape ``[n, 3]``
    :param numpy.ndarray cells: Part faces, shape ``[m, 3]``
    :param tuple part_pose: ``(translation, quaternion, velocity, omega)`` of the part
    :param tuple outer: The same four quantities for the model placement
    :param tuple ts_shape: Shape of ``radar.time_prop["timestamp"]``
    :param bool static_only: Emit scalar motion instead of per-timestamp arrays
    :param dict target_kwargs: Extra target keys to copy through

    :return: A RadarSimPy target dictionary
    :rtype: dict
    """
    part_t, part_q, part_v, part_w = part_pose
    outer_t, outer_q, outer_v, outer_w = outer

    offset = _quat_rotate(outer_q, part_t)
    world_t = outer_t + offset
    world_q = _quat_multiply(outer_q, part_q)
    world_v = outer_v + np.cross(outer_w, offset) + _quat_rotate(outer_q, part_v)
    world_w = outer_w + _quat_rotate(outer_q, part_w)

    euler = np.degrees(_quat_to_rsx_euler(world_q))
    rates = np.degrees(np.stack((world_w[:, 2], world_w[:, 1], world_w[:, 0]), axis=-1))

    target: Dict[str, Any] = {
        "model": {
            "points": np.ascontiguousarray(points, dtype=np.float64),
            "cells": np.ascontiguousarray(cells, dtype=np.int32),
        },
        "unit": "m",
        "origin": (0.0, 0.0, 0.0),
    }

    if static_only:
        target["location"] = tuple(world_t[0])
        target["speed"] = tuple(world_v[0])
        target["rotation"] = tuple(euler[0])
        target["rotation_rate"] = tuple(rates[0])
    else:
        # ``MoveIndex`` advances the mesh by ``rotation[i] - rotation[i-1]``, and
        # RadarSimCpp turns that difference into Rz(dyaw) Ry(-dpitch) Rx(droll).
        # Each of those factors is 360-degree periodic, so a branch cut in the
        # wrapped angles yields exactly the same rotation and the angles are
        # deliberately left wrapped: accumulating unwrapped turns would grow
        # without bound and lose float32 precision over a long time record.
        target["location"] = _split_axes(world_t, ts_shape)
        target["speed"] = _split_axes(world_v, ts_shape)
        target["rotation"] = _split_axes(euler, ts_shape)
        target["rotation_rate"] = _split_axes(rates, ts_shape)

    target.update(target_kwargs)
    return target


def _split_axes(values: NDArray, ts_shape: Tuple[int, int, int]) -> Tuple[NDArray, ...]:
    """
    Split an ``[k, 3]`` array into three timestamp-shaped ``float32`` arrays

    The time-varying kinematics path in ``cp_radarsimc`` binds each axis to a
    C-contiguous ``float32[:, :, :]`` memoryview, so both the shape and the
    dtype have to match ``radar.time_prop["timestamp"]``.

    :param numpy.ndarray values: Values, shape ``[k, 3]``
    :param tuple ts_shape: Shape of ``radar.time_prop["timestamp"]``

    :return: One array per axis, each shaped ``ts_shape``
    :rtype: tuple
    """
    return tuple(
        np.ascontiguousarray(values[:, axis].reshape(ts_shape), dtype=np.float32)
        for axis in range(3)
    )
