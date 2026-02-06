"""
RadarSimPy - A Comprehensive Radar Simulator for Python

RadarSimPy is a powerful and versatile Python-based Radar Simulator that models
radar transceivers and simulates baseband data from point targets and 3D models.
Its signal processing tools offer range/Doppler processing, direction of arrival
estimation, and beamforming using various cutting-edge techniques, and you can
even characterize radar detection using Swerling's models. Whether you're a beginner
or an advanced user, RadarSimPy is the perfect tool for anyone looking to develop
new radar technologies or expand their knowledge of radar systems.

Key Capabilities:
- **Radar Modeling**: Transceiver modeling, arbitrary waveforms, phase noise
- **Simulation**: Baseband data from point targets & 3D models, interference simulation
- **Signal Processing**: Range/Doppler processing, DoA estimation, beamforming, CFAR
- **Characterization**: Radar detection characteristics using Swerling's models
- **3D Modeling**: LiDAR point cloud simulation and RCS calculations

Quick Start:
    >>> import radarsimpy as rs
    >>> radar = rs.Radar(transmitter=tx, receiver=rx)
    >>> result = rs.sim_radar(radar, targets)

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

# Core radar system components
from .radar import Radar
from .transmitter import Transmitter
from .receiver import Receiver

# Simulation engines
# try:
from .simulator import sim_radar
from .simulator import sim_lidar
from .simulator import sim_rcs
from .license import initialize_license, is_licensed, get_license_info

_simulation_available = True

# Automatically initialize license on module import
# Searches for license_RadarSimPy_*.lic in module directory
import os
import glob

_module_dir = os.path.dirname(os.path.abspath(__file__))
_license_pattern = os.path.join(_module_dir, "license_RadarSimPy_*.lic")
_license_files = glob.glob(_license_pattern)

if _license_files:
    # Use the first found license file
    initialize_license(_license_files[0])
else:
    # No license file found, initialize without path (free tier mode)
    initialize_license()
# except ImportError:
#     _simulation_available = False

# Signal processing and analysis tools
from . import processing
from . import tools

# 3D mesh utilities
from . import mesh_kit

# Package metadata
__version__ = "14.2.0"
__author__ = "RadarSimX"
__email__ = "info@radarsimx.com"
__url__ = "https://radarsimx.com"
__license__ = "Proprietary"
__description__ = "A comprehensive radar simulation library for Python"

# Public API - modules and functions available for import
__all__ = [
    # Core Components
    "Radar",
    "Transmitter",
    "Receiver",
    # Simulation Functions (if available)
    "sim_radar",
    "sim_lidar",
    "sim_rcs",
    # License Functions (if available)
    "initialize_license",
    "is_licensed",
    "is_free_tier",
    "get_license_info",
    # Processing and Analysis
    "processing",
    "tools",
    "mesh_kit",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__url__",
]

# Remove simulation and license functions from __all__ if not available
if not _simulation_available:
    for func in ["sim_radar", "sim_lidar", "sim_rcs", 
                 "initialize_license", "is_licensed", "is_free_tier", "get_license_info"]:
        if func in __all__:
            __all__.remove(func)


def get_version():
    """
    Get the current version of RadarSimPy.

    Returns:
        str: Version string in semantic versioning format
    """
    return __version__


def get_info():
    """
    Get comprehensive information about the RadarSimPy installation.

    Returns:
        dict: Dictionary containing package information, capabilities, and dependencies
    """
    import sys
    import platform

    info = {
        "package": "RadarSimPy",
        "version": __version__,
        "author": __author__,
        "website": __url__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "capabilities": {
            "core_components": True,
            "simulation_engines": _simulation_available,
            "signal_processing": True,
            "mesh_processing": True,
            "analysis_tools": True,
        },
        "modules": {
            "radar": "Core radar system modeling",
            "transmitter": "Radar transmitter configuration",
            "receiver": "Radar receiver configuration",
            "processing": "Signal processing algorithms",
            "tools": "Analysis and characterization tools",
            "mesh_kit": "3D mesh file loading utilities",
        },
    }

    if _simulation_available:
        info["simulation_engines"] = {
            "sim_radar": "Radar baseband simulation",
            "sim_lidar": "LiDAR point cloud simulation",
            "sim_rcs": "Radar cross-section calculation",
        }

    # Check for optional dependencies
    optional_deps = {}
    try:
        import numpy as np

        optional_deps["numpy"] = np.__version__
    except ImportError:
        optional_deps["numpy"] = "Not installed"

    try:
        import scipy

        optional_deps["scipy"] = scipy.__version__
    except ImportError:
        optional_deps["scipy"] = "Not installed"

    try:
        import pymeshlab

        optional_deps["pymeshlab"] = pymeshlab.__version__
    except ImportError:
        optional_deps["pymeshlab"] = "Not installed"

    try:
        import pyvista

        optional_deps["pyvista"] = pyvista.__version__
    except ImportError:
        optional_deps["pyvista"] = "Not installed"

    info["dependencies"] = optional_deps

    return info


def print_info():
    """Print formatted information about the RadarSimPy installation."""
    info = get_info()

    print(f"\n{info['package']} v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"Website: {info['website']}")
    print(f"Python: {info['python_version']}")
    print(f"Platform: {info['platform']}")

    print("\n📡 Capabilities:")
    for capability, available in info["capabilities"].items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"  {capability.replace('_', ' ').title()}: {status}")

    print("\n📦 Core Modules:")
    for module, description in info["modules"].items():
        print(f"  {module}: {description}")

    if "simulation_engines" in info and isinstance(info["simulation_engines"], dict):
        print("\n🎯 Simulation Engines:")
        for engine, description in info["simulation_engines"].items():
            print(f"  {engine}: {description}")

    print("\n🔧 Dependencies:")
    for dep, version in info["dependencies"].items():
        if version == "Not installed":
            print(f"  {dep}: ❌ {version}")
        else:
            print(f"  {dep}: ✅ v{version}")


def check_installation():
    """
    Check if RadarSimPy is properly installed and functional.

    Returns:
        bool: True if installation appears complete, False otherwise
    """
    issues = []

    # Check core components (already imported at module level)
    # No need to re-import here

    # Check simulation engines
    if not _simulation_available:
        issues.append(
            "Simulation engines not available - compiled extensions may be missing"
        )

    # Check required dependencies
    try:
        import numpy  # noqa: F401
    except ImportError:
        issues.append("NumPy is required but not installed")

    try:
        import scipy  # noqa: F401
    except ImportError:
        issues.append("SciPy is required but not installed")

    if issues:
        print("❌ Installation Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ RadarSimPy installation appears complete")
        return True


# Convenience function for getting started
def hello():
    """Print a welcome message and basic usage information."""
    print(
        """
    🎯 Welcome to RadarSimPy!
    
    A comprehensive radar simulation library for Python.
    
    Quick Start:
    1. Create radar components:
       >>> import radarsimpy as rs
       >>> tx = rs.Transmitter(...)
       >>> rx = rs.Receiver(...)
       >>> radar = rs.Radar(transmitter=tx, receiver=rx)
    
    2. Run simulations:
       >>> result = rs.sim_radar(radar, targets)
    
    3. Process results:
       >>> range_doppler = rs.processing.range_doppler_fft(result['baseband'])
    
    💡 License: Automatically detected from module directory (license_RadarSimPy_*.lic)
       Or manually specify: rs.initialize_license('/path/to/license.lic')
    
    �📚 Documentation: https://radarsimx.github.io/radarsimpy/
    💬 Support: info@radarsimx.com
    """
    )


# For development and debugging
if __name__ == "__main__":
    print_info()
