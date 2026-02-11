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

# =============================================================================
# Imports
# =============================================================================

# Core radar system components
from .radar import Radar
from .transmitter import Transmitter
from .receiver import Receiver

# Simulation engines
from .simulator import sim_radar, sim_lidar, sim_rcs

# License management
from .license import set_license, is_licensed, get_license_info

# Signal processing and analysis modules
from . import processing
from . import tools

# 3D mesh utilities
from . import mesh_kit

# =============================================================================
# License Initialization
# =============================================================================

# Automatically initialize license from module directory
set_license()

# =============================================================================
# Package Metadata
# =============================================================================

__version__ = "15.0.1"
__author__ = "RadarSimX"
__email__ = "info@radarsimx.com"
__url__ = "https://radarsimx.com"
__license__ = "Proprietary"
__description__ = "A comprehensive radar simulation library for Python"

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core Components
    "Radar",
    "Transmitter",
    "Receiver",
    # Simulation Functions
    "sim_radar",
    "sim_lidar",
    "sim_rcs",
    # License Functions
    "set_license",
    "is_licensed",
    "get_license_info",
    # Processing and Analysis Modules
    "processing",
    "tools",
    "mesh_kit",
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__url__",
    # Utility Functions
    "get_version",
    "get_info",
    "print_info",
    "check_installation",
    "hello",
]

# =============================================================================
# Utility Functions
# =============================================================================


def get_version():
    """
    Get the current version of RadarSimPy.

    Returns
    -------
    str
        Version string in semantic versioning format (e.g., "15.0.1")

    Examples
    --------
    >>> import radarsimpy as rs
    >>> rs.get_version()
    '15.0.1'
    """
    return __version__


def get_info():
    """
    Get comprehensive information about the RadarSimPy installation.

    This function collects information about the package version, platform,
    available modules, and installed dependencies.

    Returns
    -------
    dict
        Dictionary containing:

        - **package** (str): Package name
        - **version** (str): Current version
        - **author** (str): Package author
        - **website** (str): Official website URL
        - **python_version** (str): Python interpreter version
        - **platform** (str): Operating system and platform information
        - **modules** (dict): Available RadarSimPy modules and their descriptions
        - **simulation_engines** (dict): Available simulation engines and descriptions
        - **dependencies** (dict): Installed optional dependencies and their versions

    Examples
    --------
    >>> import radarsimpy as rs
    >>> info = rs.get_info()
    >>> print(f"RadarSimPy version: {info['version']}")
    >>> print(f"NumPy installed: {info['dependencies']['numpy']}")
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
        "modules": {
            "radar": "Core radar system modeling",
            "transmitter": "Radar transmitter configuration",
            "receiver": "Radar receiver configuration",
            "processing": "Signal processing algorithms",
            "tools": "Analysis and characterization tools",
            "mesh_kit": "3D mesh file loading utilities",
        },
        "simulation_engines": {
            "sim_radar": "Radar baseband simulation",
            "sim_lidar": "LiDAR point cloud simulation",
            "sim_rcs": "Radar cross-section calculation",
        },
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
    """
    Print formatted information about the RadarSimPy installation.

    This function displays package version, platform details, core modules,
    simulation engines, and dependency status in a readable format.

    Examples
    --------
    >>> import radarsimpy as rs
    >>> rs.print_info()
    RadarSimPy v15.0.1
    Author: RadarSimX
    Website: https://radarsimx.com
    ...
    """
    info = get_info()

    print(f"\n{info['package']} v{info['version']}")
    print(f"Author: {info['author']}")
    print(f"Website: {info['website']}")
    print(f"Python: {info['python_version']}")
    print(f"Platform: {info['platform']}")

    print("\n📦 Core Modules:")
    for module, description in info["modules"].items():
        print(f"  {module}: {description}")

    print("\n🎯 Simulation Engines:")
    for engine, description in info["simulation_engines"].items():
        print(f"  {engine}: {description}")

    print("\n🔧 Dependencies:")
    for dep, version in info["dependencies"].items():
        if version == "Not installed":
            print(f"  {dep}: ❌ {version}")
        else:
            print(f"  {dep}: ✅ v{version}")
    print()


def check_installation():
    """
    Check if RadarSimPy is properly installed and functional.

    Verifies that core components are importable and required dependencies
    are available. Reports any installation issues found.

    Returns
    -------
    bool
        True if installation appears complete and functional, False if issues detected

    Examples
    --------
    >>> import radarsimpy as rs
    >>> if rs.check_installation():
    ...     print("Ready to use!")
    ✅ RadarSimPy installation appears complete
    Ready to use!
    """
    issues = []

    # Check core components (already imported at module level)
    # No need to re-import here

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


def hello():
    """
    Print a welcome message and basic usage information.

    Displays a friendly introduction to RadarSimPy with quick start examples
    and helpful resources for getting started.

    Examples
    --------
    >>> import radarsimpy as rs
    >>> rs.hello()
    🎯 Welcome to RadarSimPy!
    ...
    """
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
    
    📚 Documentation: https://radarsimx.github.io/radarsimpy/
    💬 Support: info@radarsimx.com
    """
    )


# =============================================================================
# Module Entry Point (for development and debugging)
# =============================================================================

if __name__ == "__main__":
    print_info()
