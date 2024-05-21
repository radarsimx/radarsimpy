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


ap = argparse.ArgumentParser()
ap.add_argument(
    "-t", "--tier", required=False, help="Build tier, choose `free` or `standard`"
)
ap.add_argument(
    "-a", "--arch", required=False, help="Build architecture, choose `cpu` or `gpu`"
)

args, unknown = ap.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

if args.tier is None:
    ARG_TIER = "standard"
elif args.tier.lower() == "free":
    ARG_TIER = "free"
elif args.tier.lower() == "standard":
    ARG_TIER = "standard"
else:
    raise ValueError("Invalid --tier parameters, please choose 'free' or 'standard'")

if args.arch is None:
    ARG_ARCH = "cpu"
elif args.arch.lower() == "cpu":
    ARG_ARCH = "cpu"
elif args.arch.lower() == "gpu":
    ARG_ARCH = "gpu"
else:
    raise ValueError("Invalid --arch parameters, please choose 'cpu' or 'gpu'")


os_type = platform.system()  # 'Linux', 'Windows', 'macOS'

if os_type == "Linux":
    LINK_ARGS = ["-Wl,-rpath,-lhdf5,-hdf5_cpp,-hdf5_hl,-hdf5_hl_cpp,$ORIGIN"]
    LIBRARY_DIRS = ["src/radarsimcpp/build", "src/radarsimcpp/hdf5/lib_linux_x86_64/lib"]
    if args.arch == "gpu":
        NVCC = "nvcc"
        CUDALIB = "lib64"
elif os_type == "Darwin":
    if platform.processor() == "arm":
        LINK_ARGS = ["-Wl,-ld_classic,-rpath,$ORIGIN"]
        LIBRARY_DIRS = ["src/radarsimcpp/build"]
    else:
        LINK_ARGS = ["-Wl,-ld_classic,-rpath,$ORIGIN"]
        LIBRARY_DIRS = ["src/radarsimcpp/build"]
elif os_type == "Windows":
    LINK_ARGS = []
    LIBRARY_DIRS = ["src/radarsimcpp/build/Release"]
    if args.arch == "gpu":
        NVCC = "nvcc.exe"
        CUDALIB = "lib\\x64"


def find_in_path(name, path):
    """Iterates over the directories in the search path by splitting the path string using
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
    """Locate the CUDA installation on the system

    :raises EnvironmentError: The nvcc binary could not be located in your $PATH.
        Either add it to your path, or set $CUDA_PATH
    :raises EnvironmentError: The CUDA <key> path could not be located in <val>
    :return: dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    :rtype: dict
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


if ARG_TIER == "free":
    if ARG_ARCH == "gpu":
        MACROS = [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("_FREETIER_", 1),
            ("_CUDA_", None),
        ]
    elif ARG_ARCH == "cpu":
        MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("_FREETIER_", 1)]
else:
    if ARG_ARCH == "gpu":
        MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("_CUDA_", None)]
    elif ARG_ARCH == "cpu":
        MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

INCLUDE_DIRS = ["src/radarsimcpp/includes", "src/radarsimcpp/includes/zpvector"]

if ARG_ARCH == "gpu":
    CUDA = locate_cuda()
    INCLUDE_DIRS = INCLUDE_DIRS + [CUDA["include"]]
    LIBRARY_DIRS = LIBRARY_DIRS + [CUDA["lib64"]]


ext_modules = [
    Extension(
        "radarsimpy.lib.cp_radarsimc",
        ["src/radarsimpy/lib/cp_radarsimc.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS,
        libraries=["radarsimcpp"],
        library_dirs=LIBRARY_DIRS,
        extra_link_args=LINK_ARGS,
    ),
    Extension(
        "radarsimpy.rt",
        ["src/radarsimpy/raytracing/rt.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS,
        libraries=["radarsimcpp"],
        library_dirs=LIBRARY_DIRS,
        extra_link_args=LINK_ARGS,
    ),
    Extension(
        "radarsimpy.simulator",
        ["src/radarsimpy/simulator.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS,
        libraries=["radarsimcpp"],
        library_dirs=LIBRARY_DIRS,
        extra_link_args=LINK_ARGS,
    ),
]

setup(
    name="radarsimpy",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        ext_modules,
        annotate=False,
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()],
)
