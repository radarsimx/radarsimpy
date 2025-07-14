"""
Build configuration utilities for RadarSimPy.

This module provides utilities for managing build configurations,
checking dependencies, and validating the build environment.
"""

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are available.

    Returns:
        Tuple of (success, missing_packages)
    """
    required_packages = [
        "numpy",
        "cython",
        "setuptools",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    return len(missing) == 0, missing


def find_msvc_with_vswhere() -> Optional[str]:
    """
    Use vswhere.exe to find the latest installed MSVC compiler (cl.exe).
    Returns the path to cl.exe if found, else None.
    """
    import json

    # Try to find vswhere.exe
    program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get(
        "ProgramFiles"
    )
    if not program_files:
        return None
    vswhere_path = os.path.join(
        program_files, "Microsoft Visual Studio", "Installer", "vswhere.exe"
    )
    if not os.path.exists(vswhere_path):
        return None

    # Query for latest Visual Studio instance with VC tools
    try:
        result = subprocess.run(
            [
                vswhere_path,
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
                "-format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        # vswhere -format json returns a JSON array
        paths = json.loads(result.stdout)
        if isinstance(paths, list) and paths:
            install_path = (
                paths[0]
                if isinstance(paths[0], str)
                else paths[0].get("installationPath")
            )
        else:
            install_path = result.stdout.strip()
        if not install_path:
            return None
        # Look for cl.exe in VC\Tools\MSVC\*\bin\Hostx64\x64
        vc_tools = os.path.join(install_path, "VC", "Tools", "MSVC")
        if not os.path.exists(vc_tools):
            return None
        # Find the latest version directory
        versions = sorted(glob.glob(os.path.join(vc_tools, "*")), reverse=True)
        for version_dir in versions:
            cl_path = os.path.join(version_dir, "bin", "Hostx64", "x64", "cl.exe")
            if os.path.exists(cl_path):
                return cl_path
        return None
    except Exception:
        return None


def check_compiler() -> Tuple[bool, str]:
    """
    Check if a C++ compiler is available.

    Returns:
        Tuple of (success, compiler_info)
    """
    compilers = []

    if sys.platform == "win32":
        # Check for MSVC in PATH
        try:
            result = subprocess.run(["cl"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0 or "Microsoft" in result.stderr:
                compilers.append("Microsoft Visual C++ (PATH)")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        # Try to find MSVC with vswhere if not found in PATH
        if not any("Microsoft Visual C++" in c for c in compilers):
            cl_path = find_msvc_with_vswhere()
            if cl_path:
                compilers.append(f"Microsoft Visual C++ (vswhere): {cl_path}")

    else:
        # Check for GCC/Clang on Unix-like systems
        for compiler in ["gcc", "clang"]:
            try:
                result = subprocess.run(
                    [compiler, "--version"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    compilers.append(compiler.upper())
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    if compilers:
        return True, ", ".join(compilers)
    else:
        return False, "No C++ compiler found"


def check_cuda_availability() -> Dict[str, str]:
    """
    Check CUDA availability and version.

    Returns:
        Dictionary with CUDA information
    """
    cuda_info = {
        "available": False,
        "version": None,
        "path": None,
        "nvcc_path": None,
    }

    # Check CUDA_PATH environment variable
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        cuda_info["path"] = cuda_path
        cuda_info["available"] = True

        # Find nvcc
        nvcc_name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
        nvcc_path = os.path.join(cuda_path, "bin", nvcc_name)
        if os.path.exists(nvcc_path):
            cuda_info["nvcc_path"] = nvcc_path

            # Get CUDA version
            try:
                result = subprocess.run(
                    [nvcc_path, "--version"], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # Extract version from output
                    lines = result.stdout.split("\n")
                    for line in lines:
                        if "release" in line.lower():
                            version_part = line.split("release")[1].strip()
                            cuda_info["version"] = version_part.split(",")[0].strip()
                            break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

    # If CUDA_PATH not set, try to find nvcc in PATH
    if not cuda_info["available"]:
        nvcc_name = "nvcc.exe" if sys.platform == "win32" else "nvcc"
        nvcc_path = shutil.which(nvcc_name)
        if nvcc_path:
            cuda_info["available"] = True
            cuda_info["nvcc_path"] = nvcc_path
            # Infer CUDA path from nvcc location
            cuda_info["path"] = os.path.dirname(os.path.dirname(nvcc_path))

    return cuda_info


def check_cmake(min_version: str = "3.18") -> Tuple[bool, str]:
    """
    Check if CMake is available and meets the minimum version.

    Returns:
        Tuple of (success, version or error message)
    """

    def version_tuple(v):
        return tuple(int(x) for x in v.split(".") if x.isdigit())

    try:
        result = subprocess.run(
            ["cmake", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            # Parse version from output
            first_line = result.stdout.splitlines()[0]
            if "cmake version" in first_line.lower():
                version = first_line.strip().split()[-1]
                if version_tuple(version) >= version_tuple(min_version):
                    return True, version
                else:
                    return (
                        False,
                        f"CMake version {version} found, but >= {min_version} required",
                    )
            return False, "Could not parse CMake version"
        else:
            return False, result.stderr.strip() or "CMake returned nonzero exit code"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "CMake not found"


def validate_build_environment(tier: str, arch: str) -> List[str]:
    """
    Validate the build environment for the given configuration.

    Args:
        tier: Build tier (free or standard)
        arch: Build architecture (cpu or gpu)

    Returns:
        List of validation errors
    """
    errors = []

    # Check Python version
    if sys.version_info < (3, 9):
        errors.append(f"Python 3.9+ required, found {sys.version}")

    # Check dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        errors.append(f"Missing required packages: {', '.join(missing)}")

    # Check compiler
    compiler_ok, compiler_info = check_compiler()
    if not compiler_ok:
        errors.append(f"C++ compiler not found: {compiler_info}")

    # Check CUDA if GPU build
    if arch == "gpu":
        cuda_info = check_cuda_availability()
        if not cuda_info["available"]:
            errors.append("CUDA not found but GPU build requested")
        elif cuda_info["version"] is None:
            errors.append("CUDA found but version could not be determined")

    # Check CMake
    cmake_ok, cmake_info = check_cmake()
    if not cmake_ok:
        errors.append(f"CMake not found or too old: {cmake_info}")

    # Check if source directories exist
    required_dirs = [
        "src/radarsimcpp/includes",
        "src/radarsimpy",
    ]

    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            errors.append(f"Required directory not found: {dir_path}")

    return errors


def get_build_info(tier: str, arch: str) -> Dict[str, str]:
    """
    Get comprehensive build information.

    Args:
        tier: Build tier
        arch: Build architecture

    Returns:
        Dictionary with build information
    """
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "tier": tier,
        "arch": arch,
    }

    # Add compiler info
    compiler_ok, compiler_info = check_compiler()
    info["compiler"] = compiler_info if compiler_ok else "Not found"

    # Add CUDA info for GPU builds
    if arch == "gpu":
        cuda_info = check_cuda_availability()
        info["cuda_available"] = str(cuda_info["available"])
        info["cuda_version"] = cuda_info["version"] or "Unknown"
        info["cuda_path"] = cuda_info["path"] or "Not found"

    # Add CMake info
    cmake_ok, cmake_info = check_cmake()
    info["cmake"] = cmake_info if cmake_ok else f"Not found or too old: {cmake_info}"

    return info


if __name__ == "__main__":
    # Simple test of the configuration utilities
    print("RadarSimPy Build Configuration Check")
    print("=" * 40)

    # Check dependencies
    deps_ok, missing = check_dependencies()
    print(f"Dependencies: {'OK' if deps_ok else 'MISSING: ' + ', '.join(missing)}")

    # Check compiler
    compiler_ok, compiler_info = check_compiler()
    print(f"Compiler: {compiler_info}")

    # Check CUDA
    cuda_info = check_cuda_availability()
    print(f"CUDA: {'Available' if cuda_info['available'] else 'Not found'}")
    if cuda_info["available"]:
        print(f"  Version: {cuda_info['version'] or 'Unknown'}")
        print(f"  Path: {cuda_info['path']}")

    # Check CMake
    cmake_ok, cmake_info = check_cmake()
    print(f"CMake: {cmake_info if cmake_ok else 'Not found or too old: ' + cmake_info}")

    # Test validation
    print("\nValidation for standard CPU build:")
    errors = validate_build_environment("standard", "cpu")
    if errors:
        for error in errors:
            print(f"  ERROR: {error}")
    else:
        print("  All checks passed!")
