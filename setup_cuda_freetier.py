"""
Setup script for a Python package "radarsimpy" with CUDA

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

import os
from os.path import join as pjoin
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


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


os_type = platform.system()  # 'Linux', 'Windows'


if os_type == "Linux":
    LINK_ARGS = ["-Wl,-rpath,$ORIGIN"]
    NVCC = "nvcc"
    CUDALIB = "lib64"
    LIBRARY_DIRS = ["src/radarsimcpp/build"]
elif os_type == "Windows":
    LINK_ARGS = []
    NVCC = "nvcc.exe"
    CUDALIB = "lib\\x64"
    LIBRARY_DIRS = ["src/radarsimcpp/build/Release"]

CUDA = locate_cuda()

MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("_CUDA_", None)]
INCLUDE_DIRS = ["src/radarsimcpp/includes", "src/radarsimcpp/includes/zpvector"]


ext_modules = [
    Extension(
        "radarsimpy.lib.cp_radarsimc",
        ["src/radarsimpy/lib/cp_radarsimc.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS + [CUDA["include"]],
        libraries=["radarsimcpp"],
        library_dirs=LIBRARY_DIRS + [CUDA["lib64"]],
        extra_link_args=LINK_ARGS,
    ),
    Extension(
        "radarsimpy.rt",
        ["src/radarsimpy/raytracing/rt.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS + [CUDA["include"]],
        libraries=["radarsimcpp"],
        library_dirs=LIBRARY_DIRS + [CUDA["lib64"]],
        extra_link_args=LINK_ARGS,
    ),
    Extension(
        "radarsimpy.simulator",
        ["src/radarsimpy/simulator.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS + [CUDA["include"]],
        libraries=["radarsimcpp"],
        library_dirs=LIBRARY_DIRS + [CUDA["lib64"]],
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
