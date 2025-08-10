"""
RadarSimPy C++ Interface Includes - Cython Header Definitions

This package contains Cython header files (.pxd) that define the interface
between Python and the C++ radar simulation backend. These files provide
type declarations and function signatures for efficient interoperability.

Modules:
--------
- **radarsimc.pxd**: Main radar simulation C++ interface declarations
  Defines radar systems, targets, simulators, and core simulation components

- **rsvector.pxd**: Vector library interface for geometric calculations
  Provides 2D and 3D vector classes for positions, directions, and transformations

- **type_def.pxd**: Type definitions and enhanced C++ standard library interfaces
  Contains primitive type aliases and extended std::vector declarations

Purpose:
--------
These Cython declaration files enable:
- **Type Safety**: Proper C++ type mapping to Python
- **Performance**: Direct C++ function calls without Python overhead
- **Memory Management**: Efficient data transfer between Python and C++
- **Template Support**: Generic programming with C++ templates

Developer Notes:
---------------
- These are declaration files only - no implementation code
- Used by Cython compiler to generate optimized C++ interface code
- Documentation strings inside 'extern' blocks cause compilation errors
- Use regular comments (#) instead of docstrings in extern declarations

For Implementation:
------------------
- See corresponding .pyx files for actual Cython implementations
- C++ source files are located in the radarsimcpp directory
- Python-facing API is in the main radarsimpy modules

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

# Package metadata
__author__ = "RadarSimX"
__email__ = "info@radarsimx.com"
__url__ = "https://radarsimx.com"

# Module information for introspection
__all__ = []  # This is an includes package - no public exports


# Development and debugging helpers
def get_include_info():
    """
    Get information about the Cython include files in this package.

    Returns:
        dict: Information about available .pxd files and their purposes
    """
    import os

    current_dir = os.path.dirname(__file__)
    pxd_files = [f for f in os.listdir(current_dir) if f.endswith(".pxd")]

    info = {
        "package": "radarsimpy.includes",
        "description": "Cython header files for C++ interface",
        "pxd_files": {},
        "location": current_dir,
    }

    # Document each .pxd file
    file_descriptions = {
        "radarsimc.pxd": {
            "description": "Main radar simulation C++ interface",
            "components": [
                "Radar systems",
                "Target models",
                "Simulation engines",
                "Memory management",
            ],
            "primary_use": "Core radar simulation functionality",
        },
        "rsvector.pxd": {
            "description": "Vector library for geometric calculations",
            "components": [
                "Vec3 (3D vectors)",
                "Vec2 (2D vectors)",
                "Coordinate transformations",
            ],
            "primary_use": "Position and direction calculations",
        },
        "type_def.pxd": {
            "description": "Type definitions and standard library extensions",
            "components": [
                "Primitive types",
                "Enhanced std::vector",
                "Iterator support",
            ],
            "primary_use": "Type safety and container operations",
        },
    }

    for pxd_file in pxd_files:
        if pxd_file in file_descriptions:
            info["pxd_files"][pxd_file] = file_descriptions[pxd_file]
        else:
            info["pxd_files"][pxd_file] = {
                "description": "Cython header file",
                "components": ["Unknown"],
                "primary_use": "C++ interface declarations",
            }

    return info


def print_include_summary():
    """Print a summary of the available include files."""
    info = get_include_info()

    print(f"\n{info['package']} - {info['description']}")
    print(f"Location: {info['location']}")
    print("\nAvailable Include Files:")
    print("-" * 50)

    for filename, details in info["pxd_files"].items():
        print(f"\n📄 {filename}")
        print(f"   Description: {details['description']}")
        print(f"   Primary Use: {details['primary_use']}")
        print(f"   Components: {', '.join(details['components'])}")

    print(f"\nTotal files: {len(info['pxd_files'])}")


# For debugging and development
if __name__ == "__main__":
    print_include_summary()
