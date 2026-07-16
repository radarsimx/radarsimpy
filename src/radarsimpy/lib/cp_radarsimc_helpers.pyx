# distutils: language = c++
"""
Shared helpers, constants, and module-level setup for cp_radarsimc.

This file is textually merged into the ``cp_radarsimc`` extension module via
``include`` (see cp_radarsimc.pyx). It provides the module-level NumPy setup,
shared constants, and the private helper functions (validation, deprecation
handling, unit conversion, mesh loading, and material parsing) used by the
point, radar, and mesh wrappers.

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

# ============================================================================
# Imports
# ============================================================================

# Standard imports
import numpy as np
import warnings

# Cython imports
cimport numpy as np
from libcpp.complex cimport complex as cpp_complex
from radarsimpy.includes.radarsimc cimport Mem_Copy_Vec3
from radarsimpy.includes.rsvector cimport Vec3
from radarsimpy.includes.type_def cimport int_t, float_t, vector


np.import_array()
np_float = np.float32


# ============================================================================
# Constants
# ============================================================================

cdef dict UNIT_SCALE = {"m": 1.0, "cm": 100.0, "mm": 1000.0}
cdef int_t MAX_FREE_TIER_FACES = 8
cdef frozenset _VALID_TARGET_KEYS = frozenset({
    "model", "unit", "origin",
    "location", "speed", "rotation", "rotation_rate",
    "permittivity", "permeability",
    "skip_diffusion", "is_ground",  # is_ground is a deprecated alias
    "density", "environment",
})

# ============================================================================
# Private Helpers (validation, deprecation, unit conversion)
# ============================================================================

cdef inline float_t _safe_unit_conversion(str unit) except *:
    """
    Safely convert unit string to scale factor.

    :param str unit:
        Unit string ('m', 'cm', 'mm')
    :return: Scale factor for converting to meters
    :rtype: float_t
    :raises: ValueError for unsupported units
    """
    if unit not in UNIT_SCALE:
        raise ValueError(f"Invalid unit '{unit}'. Supported units: {list(UNIT_SCALE.keys())}")
    return <float_t>UNIT_SCALE[unit]

cdef inline void _validate_mesh_for_free_tier(int_t num_faces) except *:
    """
    Check mesh size limits for free tier.

    :param int_t num_faces:
        Number of mesh faces in the target model
    :raises: RuntimeError if FreeTier limitations are exceeded
    """
    from radarsimpy.license import is_licensed
    if not is_licensed() and num_faces > MAX_FREE_TIER_FACES:
        raise RuntimeError(
            f"\nTrial Version Limitation - Mesh Size\n"
            f"----------------------------------------\n"
            f"Current limitation: Maximum {MAX_FREE_TIER_FACES} mesh faces\n"
            f"Your model: {num_faces} faces\n\n"
            f"To simulate larger meshes, please purchase a license:\n"
            f"→ https://radarsimx.com/product/radarsimpy/\n"
        )

cdef inline void _warn_deprecated_parameter(str old_param, str new_param) except *:
    """
    Issue standardized deprecation warnings for renamed parameters.

    :param str old_param:
        Name of the deprecated parameter
    :param str new_param:
        Name of the new parameter to use instead
    :raises: DeprecationWarning with migration guidance
    """
    warnings.warn(
        f"Deprecated: '{old_param}' parameter has been replaced with '{new_param}'. "
        f"Please update your code to use '{new_param}' instead. "
        f"Support for '{old_param}' will be removed in a future version.",
        DeprecationWarning,
        stacklevel=3
    )

cdef inline void _validate_target_keys(target) except *:
    """
    Warn about unrecognized keys in a target dict to catch typos.

    :param dict target:
        Target properties dictionary to validate
    :raises: UserWarning for any keys not in the known set
    """
    unknown = set(target.keys()) - _VALID_TARGET_KEYS
    if unknown:
        warnings.warn(
            f"Unrecognized key(s) in target dict: {sorted(unknown)}. "
            f"These will be ignored. Valid keys: {sorted(_VALID_TARGET_KEYS)}",
            UserWarning,
            stacklevel=3
        )

cdef inline tuple _load_and_validate_mesh(target, mesh_module):
    """
    Load mesh model and validate against free tier limits.

    :param dict target:
        Target properties containing 'model' and optional 'unit'
    :param mesh_module:
        Mesh loading module
    :return: Tuple of (points_array, cells_array) as numpy arrays
    :rtype: tuple
    """
    cdef float_t scale = _safe_unit_conversion(target.get("unit", "m"))
    from radarsimpy.mesh_kit import load_mesh
    try:
        mesh_data = load_mesh(target["model"], scale, mesh_module)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load mesh model '{target.get('model', 'unknown')}': {e}"
        )
    points = mesh_data["points"].astype(np_float)
    cells = mesh_data["cells"].astype(np.int32)
    _validate_mesh_for_free_tier(<int_t>cells.shape[0])
    return points, cells

cdef inline void _parse_material_properties(
    target,
    cpp_complex[float_t] * ep_out,
    cpp_complex[float_t] * mu_out,
) except *:
    """
    Parse permittivity and permeability from target dict.

    :param dict target:
        Target properties containing optional 'permittivity' and 'permeability'
    :param cpp_complex[float_t] * ep_out:
        Output pointer for permittivity
    :param cpp_complex[float_t] * mu_out:
        Output pointer for permeability
    """
    permittivity = target.get("permittivity", 1e38)
    permeability = target.get("permeability", 1)
    if permittivity == "PEC":
        ep_out[0] = cpp_complex[float_t](<float_t>1e38, <float_t>0.0)
        mu_out[0] = cpp_complex[float_t](<float_t>1.0, <float_t>0.0)
    else:
        ep_out[0] = cpp_complex[float_t](<float_t>np.real(permittivity), <float_t>np.imag(permittivity))
        mu_out[0] = cpp_complex[float_t](<float_t>np.real(permeability), <float_t>np.imag(permeability))

cdef inline void _handle_deprecated_target_params(target) except *:
    """
    Handle deprecated parameter names in target dict.

    :param dict target:
        Target properties dictionary (modified in-place)
    """
    if "is_ground" in target:
        target["skip_diffusion"] = target["is_ground"]
        _warn_deprecated_parameter("is_ground", "skip_diffusion")

# ============================================================================
# Private Helpers (time-varying kinematics expansion)
# ============================================================================

cdef inline float_t[:, :, :] _broadcast_time_varying(value, shape) except *:
    """
    Resolve a per-axis parameter that is either already time-varying (an
    array matching ``shape``) or a scalar constant to broadcast across it.

    :param value:
        Either an array already shaped like ``shape``, or a scalar constant
    :param tuple shape:
        Target shape (channels, frames, pulses) to broadcast scalars to
    :return: Value cast to np_float with the given shape
    :rtype: float_t[:, :, :]
    """
    if np.size(value) > 1:
        return value.astype(np_float)
    return np.full(shape, value, dtype=np_float)

cdef inline void _expand_time_varying_kinematics(
    location,
    speed,
    rotation,
    rotation_rate,
    t,
    vector[Vec3[float_t]] &loc_vt,
    vector[Vec3[float_t]] &spd_vt,
    vector[Vec3[float_t]] &rot_vt,
    vector[Vec3[float_t]] &rrt_vt,
) except *:
    """
    Expand per-axis location/speed/rotation/rotation_rate into time-varying
    Vec3 vectors sampled at ``t``.

    Each of the three elements of location/speed/rotation/rotation_rate is
    either a scalar or an array already matching the shape of ``t``. Axes
    that are still scalar are derived from the constant-motion equations
    (location = location + speed * t, rotation = rotation + rotation_rate * t)
    or broadcast to ``t``'s shape. Assumes the caller has already determined
    that at least one input is time-varying.

    :param list location, speed, rotation, rotation_rate:
        3-element lists; each element is a scalar or an array shaped like ``t``
    :param t:
        Reference time array used to broadcast scalars and evaluate constant-
        motion equations
    :param vector[Vec3[float_t]] &loc_vt/spd_vt/rot_vt/rrt_vt:
        Output vectors populated with one Vec3 per sample of ``t``
    """
    cdef float_t[:, :, :] locx_mv, locy_mv, locz_mv
    cdef float_t[:, :, :] spdx_mv, spdy_mv, spdz_mv
    cdef float_t[:, :, :] rotx_mv, roty_mv, rotz_mv
    cdef float_t[:, :, :] rrtx_mv, rrty_mv, rrtz_mv

    ts_shape = np.shape(t)
    cdef int_t bbsize_c = <int_t>(ts_shape[0] * ts_shape[1] * ts_shape[2])

    if np.size(location[0]) > 1:
        locx_mv = location[0].astype(np_float)
    else:
        locx_mv = (location[0] + speed[0]*t).astype(np_float)

    if np.size(location[1]) > 1:
        locy_mv = location[1].astype(np_float)
    else:
        locy_mv = (location[1] + speed[1]*t).astype(np_float)

    if np.size(location[2]) > 1:
        locz_mv = location[2].astype(np_float)
    else:
        locz_mv = (location[2] + speed[2]*t).astype(np_float)

    spdx_mv = _broadcast_time_varying(speed[0], ts_shape)
    spdy_mv = _broadcast_time_varying(speed[1], ts_shape)
    spdz_mv = _broadcast_time_varying(speed[2], ts_shape)

    if np.size(rotation[0]) > 1:
        rotx_mv = np.radians(rotation[0]).astype(np_float)
    else:
        rotx_mv = np.radians(
            rotation[0] + rotation_rate[0]*t).astype(np_float)

    if np.size(rotation[1]) > 1:
        roty_mv = np.radians(rotation[1]).astype(np_float)
    else:
        roty_mv = np.radians(
            rotation[1] + rotation_rate[1]*t).astype(np_float)

    if np.size(rotation[2]) > 1:
        rotz_mv = np.radians(rotation[2]).astype(np_float)
    else:
        rotz_mv = np.radians(
            rotation[2] + rotation_rate[2]*t).astype(np_float)

    rrtx_mv = _broadcast_time_varying(np.radians(rotation_rate[0]), ts_shape)
    rrty_mv = _broadcast_time_varying(np.radians(rotation_rate[1]), ts_shape)
    rrtz_mv = _broadcast_time_varying(np.radians(rotation_rate[2]), ts_shape)

    Mem_Copy_Vec3(&locx_mv[0,0,0], &locy_mv[0,0,0], &locz_mv[0,0,0], bbsize_c, loc_vt)
    Mem_Copy_Vec3(&spdx_mv[0,0,0], &spdy_mv[0,0,0], &spdz_mv[0,0,0], bbsize_c, spd_vt)
    Mem_Copy_Vec3(&rotx_mv[0,0,0], &roty_mv[0,0,0], &rotz_mv[0,0,0], bbsize_c, rot_vt)
    Mem_Copy_Vec3(&rrtx_mv[0,0,0], &rrty_mv[0,0,0], &rrtz_mv[0,0,0], bbsize_c, rrt_vt)
