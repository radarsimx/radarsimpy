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
from radarsimpy.includes.type_def cimport int_t, float_t


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
