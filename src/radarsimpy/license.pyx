# distutils: language = c++

"""
RadarSimPy License Management Module

This module provides Python access to the RadarSimCpp license verification system.
It allows checking license status and accessing license information.

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

from libcpp.string cimport string
from radarsimpy.includes.radarsimc cimport LicenseManager, IsFreeTier as cpp_IsFreeTier


def initialize_license(license_file_path=None):
    """
    Initialize the license manager with a license file.
    
    Args:
        license_file_path (str, optional): Path to license file. If None, runs in free tier mode.
    
    Example:
        >>> import radarsimpy
        >>> # Explicit path
        >>> radarsimpy.initialize_license("/path/to/license_RadarSimPy_customer.lic")
        >>> if radarsimpy.is_licensed():
        ...     print("Full license active")
        
    Note:
        Typically called automatically by the package during import with auto-detected license path.
    """
    cdef string cpp_license_path
    
    if license_file_path is None:
        # No license file provided, pass empty string (free tier mode)
        cpp_license_path = b""
    else:
        # Use provided path
        cpp_license_path = license_file_path.encode('utf-8')
    
    LicenseManager.getInstance().initialize(cpp_license_path)


def is_licensed():
    """
    Check if a valid license is active.
    
    Returns:
        bool: True if a valid license is loaded, False if running in free tier mode.
    
    Example:
        >>> if radarsimpy.is_licensed():
        ...     print("Full license active")
        ... else:
        ...     print("Running in free tier mode")
    """
    return LicenseManager.getInstance().isLicensed()


def is_free_tier():
    """
    Check if running in free tier mode (no valid license).
    
    Returns:
        bool: True if no valid license is present, False if licensed.
    
    Example:
        >>> if radarsimpy.is_free_tier():
        ...     print("Free tier limitations apply")
    """
    return cpp_IsFreeTier()


def get_license_info():
    """
    Get license information string.
    
    Returns:
        str: License information (customer name, days remaining) or empty string if not licensed.
    
    Example:
        >>> info = radarsimpy.get_license_info()
        >>> print(info)
        'Licensed to: Customer Name (30 days remaining)'
    """
    cdef string info = LicenseManager.getInstance().getLicenseInfo()
    return info.decode('utf-8')
