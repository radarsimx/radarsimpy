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


def initialize_license(product_name="RadarSimPy"):
    """
    Initialize the license manager with a specific product name.
    
    This searches for license files matching the pattern:
    license_<product_name>_*.lic in the library directory.
    
    Args:
        product_name (str): Product name to search for (default: "RadarSimPy")
                           Common values: "RadarSimPy", "RadarSimM"
    
    Example:
        >>> import radarsimpy
        >>> radarsimpy.initialize_license("RadarSimPy")
        >>> if radarsimpy.is_licensed():
        ...     print("Full license active")
    """
    cdef string cpp_product_name = product_name.encode('utf-8')
    LicenseManager.getInstance().initialize(cpp_product_name)


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
