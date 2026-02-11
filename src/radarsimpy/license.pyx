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
from libcpp.vector cimport vector
from radarsimpy.includes.radarsimc cimport LicenseManager
import os
import glob


def set_license(license_file_path=None):
    """
    Set or update the license for RadarSimPy.
    
    Args:
        license_file_path (str, optional): Path to license file. If None, automatically 
            searches for all license_RadarSimPy_*.lic files in the module directory.
    
    Example:
        >>> import radarsimpy
        >>> # Explicit path
        >>> radarsimpy.set_license("/path/to/license_RadarSimPy_customer.lic")
        >>> if radarsimpy.is_licensed():
        ...     print("Full license active")
        
    Note:
        Can be called multiple times to update or recheck license files.
        Typically called automatically by the package during import.
    """
    cdef string cpp_license_path
    cdef vector[string] cpp_license_paths
    cdef string cpp_product = b"RadarSimPy"
    
    if license_file_path is None:
        # No license file provided, search for all license_RadarSimPy_*.lic files
        _module_dir = os.path.dirname(os.path.abspath(__file__))
        license_pattern = os.path.join(_module_dir, "license_RadarSimPy_*.lic")
        license_files = glob.glob(license_pattern)
        
        if license_files:
            # Pass all found license files to C++
            for lic_file in license_files:
                cpp_license_paths.push_back(lic_file.encode('utf-8'))
            LicenseManager.GetInstance().SetLicense(cpp_license_paths, cpp_product)
        else:
            # No license files found, use empty string (free tier mode)
            cpp_license_path = b""
            LicenseManager.GetInstance().SetLicense(cpp_license_path, cpp_product)
    else:
        # Use provided path
        cpp_license_path = license_file_path.encode('utf-8')
        LicenseManager.GetInstance().SetLicense(cpp_license_path, cpp_product)


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
    return LicenseManager.GetInstance().IsLicensed()


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
    cdef string info = LicenseManager.GetInstance().GetLicenseInfo()
    return info.decode('utf-8')
