"""
RadarSimPy Library Module - Low-Level C++ Interface

This module contains the low-level Cython interface code that bridges Python
to the high-performance C++ radar simulation backend. It provides the compiled
extensions and wrapper functions that enable efficient data transfer and
computation between Python and C++.

Components:
-----------
- **cp_radarsimc**: Core Cython module containing C++ wrapper functions
  - Point target creation and management
  - Radar system configuration and setup
  - 3D mesh target processing and conversion
  - RCS target preparation for simulation

Purpose:
--------
This module serves as the performance-critical interface layer that:
- **Converts Python objects** to C++ data structures
- **Manages memory efficiently** between Python and C++
- **Provides type-safe wrappers** for C++ simulation functions
- **Handles data marshaling** for complex radar configurations

Internal Architecture:
---------------------
- Uses Cython for Python-C++ interoperability
- Implements zero-copy data transfer where possible
- Provides exception-safe memory management
- Optimized for real-time simulation performance

Developer Notes:
---------------
- This is an internal module - not intended for direct user access
- Functions prefixed with 'cp_' are Cython wrapper functions
- Memory management is handled automatically
- Type conversions follow RadarSimPy conventions (float32 for efficiency)

Dependencies:
------------
- Compiled C++ simulation backend (radarsimcpp)
- NumPy for array operations and type definitions
- Cython runtime for Python-C++ bridge

Performance Notes:
-----------------
- Single-precision floating point (float32) used for memory efficiency
- Vectorized operations where possible
- Direct memory access to avoid Python overhead
- Optimized data structures for cache efficiency

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

# Import the compiled Cython module if available
try:
    from .cp_radarsimc import cp_GetTargetMesh, cp_GetSceneStateChannels

    _lib_available = True
    _import_error = None
except ImportError as e:
    _lib_available = False
    _import_error = str(e)

# Module metadata
__author__ = "RadarSimX"
__description__ = "Low-level C++ interface for RadarSimPy"

# Public API - Internal functions (not typically used directly by end users)
__all__ = (
    [
        "cp_GetTargetMesh",
        "cp_GetSceneStateChannels",
        "is_available",
        "get_lib_info",
    ]
    if _lib_available
    else ["is_available", "get_lib_info"]
)


def is_available():
    """
    Check if the compiled library module is available.

    Returns:
        bool: True if compiled extensions are available, False otherwise
    """
    return _lib_available


def get_lib_info():
    """
    Get information about the library module status and capabilities.

    Returns:
        dict: Information about the library module
    """
    info = {
        "module": "radarsimpy.lib",
        "description": __description__,
        "available": _lib_available,
        "compiled_extensions": _lib_available,
    }

    if _lib_available:
        info["functions"] = {
            "cp_GetTargetMesh": "Retrieve target meshes at given timestamps",
            "cp_GetSceneStateChannels": "Compute global locations and boresights using C++ Rotate",
        }
        info["data_types"] = {
            "precision": "float32 (single precision for efficiency)",
            "complex_support": "Native C++ complex number support",
            "vector_types": "2D and 3D vector operations",
            "memory_management": "Automatic with exception safety",
        }
    else:
        info["error"] = (
            _import_error if "_import_error" in globals() else "Unknown import error"
        )
        info["possible_causes"] = [
            "Compiled extensions not built",
            "Missing C++ runtime dependencies",
            "Architecture mismatch (32-bit vs 64-bit)",
            "Missing Visual C++ redistributables (Windows)",
            "Missing shared libraries (Linux/macOS)",
        ]
        info["solutions"] = [
            "Rebuild the package with: python setup.py build_ext --inplace",
            "Check compiler installation and configuration",
            "Verify all dependencies are installed",
            "Contact support if issues persist",
        ]

    return info


def print_lib_status():
    """Print formatted information about the library module status."""
    info = get_lib_info()

    print(f"\n📚 {info['module']} - {info['description']}")

    if info["available"]:
        print("Status: ✅ Available")
        print("\n🔧 Available Functions:")
        for func_name, func_desc in info["functions"].items():
            print(f"  {func_name}: {func_desc}")

        print("\n💾 Data Type Support:")
        for dtype, dtype_desc in info["data_types"].items():
            print(f"  {dtype.replace('_', ' ').title()}: {dtype_desc}")
    else:
        print("Status: ❌ Not Available")
        print(f"Error: {info.get('error', 'Unknown')}")

        print("\n🔍 Possible Causes:")
        for cause in info["possible_causes"]:
            print(f"  - {cause}")

        print("\n🛠️ Possible Solutions:")
        for solution in info["solutions"]:
            print(f"  - {solution}")


def check_compilation():
    """
    Check if the Cython compilation was successful and provide diagnostics.

    Returns:
        bool: True if compilation appears successful, False otherwise
    """
    if not _lib_available:
        print("❌ Compilation Check Failed:")
        print("  The cp_radarsimc module could not be imported.")
        print("  This indicates the Cython extensions were not compiled successfully.")
        print("\n🔧 To fix this issue:")
        print("  1. Ensure you have a C++ compiler installed")
        print("  2. Run: python setup.py build_ext --inplace")
        print("  3. Check for any compilation errors")
        print("  4. Verify all dependencies are installed")
        return False
    else:
        print("✅ Compilation Check Passed:")
        print("  The cp_radarsimc module is available and functional.")
        print("  All low-level C++ interface functions are accessible.")
        return True


# Development helpers
def get_function_signatures():
    """
    Get the function signatures for the available library functions.

    Returns:
        dict: Function signatures and descriptions
    """
    if not _lib_available:
        return {"error": "Library not available"}

    signatures = {
        "cp_GetTargetMesh": {
            "signature": "cp_GetTargetMesh(target, timestamp, mesh_module, sim_timestamp=None)",
            "parameters": {
                "target": "Target configuration dict",
                "timestamp": "Query timestamp(s)",
                "mesh_module": "Mesh processing module reference",
                "sim_timestamp": "Simulation timestamps in the radar object",
            },
            "returns": "Transformed target points and cells",
        },
        "cp_GetSceneStateChannels": {
            "signature": "cp_GetSceneStateChannels(tx_local_locs, rx_local_locs, q_locs, q_rots)",
            "parameters": {
                "tx_local_locs": "Local coordinates of Transmitter channels",
                "rx_local_locs": "Local coordinates of Receiver channels",
                "q_locs": "Query platform locations",
                "q_rots": "Query platform rotations (yaw, pitch, roll in rad)",
            },
            "returns": "Tuple of (tx_global_locs, rx_global_locs, radar_boresights)",
        },
    }

    return signatures


# For debugging and development
if __name__ == "__main__":
    print_lib_status()
    if _lib_available:
        print("\n📋 Function Signatures:")
        sigs = get_function_signatures()
        for function_name, sig_info in sigs.items():
            print(f"\n{function_name}:")
            sig_dict = sig_info if isinstance(sig_info, dict) else {}
            print(f"  Signature: {sig_dict.get('signature', 'N/A')}")
            print(f"  Returns: {sig_dict.get('returns', 'N/A')}")
            print("  Parameters:")
            params = sig_dict.get("parameters", {})
            if isinstance(params, dict):
                for param_name, param_desc in params.items():
                    print(f"    {param_name}: {param_desc}")
            else:
                print("    No parameter information available")
