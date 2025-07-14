"""
Setup script for a Python package "radarsimpy"

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

# Import required system and platform modules
import platform
import argparse
import sys
import os
from os.path import join as pjoin
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

# Set up command line argument parser for build configuration
ap = argparse.ArgumentParser()
ap.add_argument(
    "-t", "--tier", required=False, help="Build tier, choose `free` or `standard`"
)
ap.add_argument(
    "-a", "--arch", required=False, help="Build architecture, choose `cpu` or `gpu`"
)

# Parse command line arguments, separating setup.py specific args from pass-through args
args, unknown = ap.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

# Determine build tier (free vs standard)
# Default to standard if not specified
if args.tier is None:
    ARG_TIER = "standard"
elif args.tier.lower() == "free":
    ARG_TIER = "free"
elif args.tier.lower() == "standard":
    ARG_TIER = "standard"
else:
    raise ValueError("Invalid --tier parameters, please choose 'free' or 'standard'")

# Determine build architecture (CPU vs GPU)
# Default to CPU if not specified
if args.arch is None:
    ARG_ARCH = "cpu"
elif args.arch.lower() == "cpu":
    ARG_ARCH = "cpu"
elif args.arch.lower() == "gpu":
    ARG_ARCH = "gpu"
else:
    raise ValueError("Invalid --arch parameters, please choose 'cpu' or 'gpu'")

# Detect operating system for platform-specific configurations
os_type = platform.system()  # 'Linux', 'Windows', 'macOS'

# Set platform-specific build configurations
if os_type == "Linux":
    # Linux-specific: Set rpath to find shared libraries in the same directory
    LINK_ARGS = ["-Wl,-rpath,$ORIGIN"]
    COMPILE_ARGS = ["-std=c++20"]
    LIB_DIRS = [
        "src/radarsimcpp/build",
        "src/radarsimcpp/hdf5-lib-build/libs/lib_linux_gcc11_x86_64/lib",
    ]
    LIBS = ["hdf5", "hdf5_cpp", "hdf5_hl", "hdf5_hl_cpp"]
    if args.arch == "gpu":
        NVCC = "nvcc"
        CUDALIB = "lib64"
        LIBS = LIBS + ["cudart"]
elif os_type == "Darwin":  # macOS
    LIBS = []
    COMPILE_ARGS = ["-std=c++20"]
    if platform.processor() == "arm":  # M1/M2 processors
        LINK_ARGS = ["-Wl,-rpath,@loader_path"]
        LIB_DIRS = ["src/radarsimcpp/build"]
    else:  # Intel processors
        LINK_ARGS = ["-Wl,-rpath,@loader_path"]
        LIB_DIRS = ["src/radarsimcpp/build"]
elif os_type == "Windows":
    LINK_ARGS = []
    COMPILE_ARGS = ["/std:c++20"]
    LIB_DIRS = ["src/radarsimcpp/build/Release"]
    LIBS = []
    if args.arch == "gpu":
        NVCC = "nvcc.exe"
        CUDALIB = "lib\\x64"
        LIBS = ["cudart"]

def find_in_path(name, path):
    """
    Iterates over the directories in the search path by splitting the path string using
    os.pathsep as the delimiter. os.pathsep is a string that represents the separator
    used in the PATH environment variable on the current operating system
    (e.g., : on Unix-like systems and ; on Windows).

    :param name: The name of the file
    :type name: str
    :param path: The search path
    :type path: str
    :return: The absolute path of the file
    :rtype: str
    """
    for path_name in path.split(os.pathsep):
        binpath = pjoin(path_name, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None

def locate_cuda():
    """
    Locate the CUDA installation on the system

    :raises EnvironmentError: The nvcc binary could not be located in your $PATH.
        Either add it to your path, or set $CUDA_PATH
    :raises EnvironmentError: The CUDA <key> path could not be located in <val>
    :return: dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    """
    # The code first checks if the CUDA_PATH environment variable is set.
    # If it is, it uses the value of CUDA_PATH as the CUDA installation directory
    # and constructs the path to the nvcc binary (NVIDIA CUDA Compiler) inside that directory.
    if "CUDA_PATH" in os.environ:
        home = os.environ["CUDA_PATH"]
        nvcc = pjoin(home, "bin", NVCC)
    else:
        # If the CUDA_PATH environment variable is not set, it searches for the nvcc
        # binary in the system's PATH environment variable. If nvcc is not found in
        # the PATH, it raises an EnvironmentError. Otherwise, it sets the home variable
        # to the parent directory of nvcc.
        default_path = pjoin(os.sep, "usr", "local", "cuda", "bin")
        nvcc = find_in_path(NVCC, os.environ["PATH"] + os.pathsep + default_path)
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be located in your $PATH. "
                "Either add it to your path, or set $CUDA_PATH"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, CUDALIB),
    }
    for key, val in cudaconfig.items():
        if not os.path.exists(val):
            raise EnvironmentError(
                "The CUDA " + key + " path could not be located in " + val
            )

    return cudaconfig

# Configure build macros based on tier and architecture
if ARG_TIER == "free":
    if ARG_ARCH == "gpu":
        MACROS = [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),  # NumPy API version
            ("_FREETIER_", 1),                                 # Enable free tier features
            ("_CUDA_", None),                                  # Enable CUDA support
        ]
    elif ARG_ARCH == "cpu":
        MACROS = [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("_FREETIER_", 1)
        ]
else:  # standard tier
    if ARG_ARCH == "gpu":
        MACROS = [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("_CUDA_", None)
        ]
    elif ARG_ARCH == "cpu":
        MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# Set include directories for header files
INCLUDE_DIRS = ["src/radarsimcpp/includes", "src/radarsimcpp/includes/rsvector"]

# Add CUDA-specific configurations if GPU build is selected
if ARG_ARCH == "gpu":
    CUDA = locate_cuda()
    INCLUDE_DIRS = INCLUDE_DIRS + [CUDA["include"]]
    LIB_DIRS = LIB_DIRS + [CUDA["lib64"]]

# Define Cython extension modules to be built
ext_modules = [
    # Core radar simulation C++ wrapper
    Extension(
        "radarsimpy.lib.cp_radarsimc",
        ["src/radarsimpy/lib/cp_radarsimc.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=["radarsimcpp"] + LIBS,
        library_dirs=LIB_DIRS,
        extra_link_args=LINK_ARGS,
    ),
    # High-level simulator interface
    Extension(
        "radarsimpy.simulator",
        ["src/radarsimpy/simulator.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS,
        extra_compile_args=COMPILE_ARGS,
        libraries=["radarsimcpp"] + LIBS,
        library_dirs=LIB_DIRS,
        extra_link_args=LINK_ARGS,
    ),
]

# Configure and run setup
setup(
    name="radarsimpy",
    cmdclass={"build_ext": build_ext},  # Use Cython's build_ext
    ext_modules=cythonize(
        ext_modules,
        annotate=False,  # Don't generate HTML annotation files
        compiler_directives={"language_level": "3"},  # Use Python 3 syntax
    ),
    include_dirs=[numpy.get_include()],  # Include NumPy headers
)
