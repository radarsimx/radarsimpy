#!python

import platform

import os
from os.path import join as pjoin
from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


os_type = platform.system()  # 'Linux', 'Windows'

if os_type == "Linux":
    LINK_ARGS = ["-Wl,-rpath,$ORIGIN"]
    NVCC = "nvcc"
    CUDALIB = "lib64"
    LIBRARY_DIRS = ["src/radarsimc/build"]
elif os_type == "Windows":
    LINK_ARGS = []
    NVCC = "nvcc.exe"
    CUDALIB = "lib\\x64"
    LIBRARY_DIRS = ["src/radarsimc/build/Release"]

MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("_CUDA_", None)]
INCLUDE_DIRS = ["src/radarsimc/includes", "src/radarsimc/includes/zpvector"]


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDA_PATH env variable is in use
    if "CUDA_PATH" in os.environ:
        home = os.environ["CUDA_PATH"]
        nvcc = pjoin(home, "bin", NVCC)
    else:
        # otherwise, search the PATH for NVCC
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
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(
                "The CUDA %s path could not be located in %s" % (k, v)
            )

    return cudaconfig


CUDA = locate_cuda()

ext_modules = [
    Extension("radarsimpy.radar", ["src/radarsimpy/radar.py"], define_macros=MACROS),
    Extension("radarsimpy.util", ["src/radarsimpy/util.py"], define_macros=MACROS),
    Extension(
        "radarsimpy.processing", ["src/radarsimpy/processing.py"], define_macros=MACROS
    ),
    Extension("radarsimpy.tools", ["src/radarsimpy/tools.py"], define_macros=MACROS),
    Extension(
        "radarsimpy.lib.cp_radarsimc",
        ["src/radarsimpy/lib/cp_radarsimc.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS + [CUDA["include"]],
        libraries=["radarsimc"],
        library_dirs=LIBRARY_DIRS + [CUDA["lib64"]],
        extra_link_args=LINK_ARGS,
    ),
    Extension(
        "radarsimpy.rt",
        ["src/radarsimpy/raytracing/rt.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS + [CUDA["include"]],
        libraries=["radarsimc"],
        library_dirs=LIBRARY_DIRS + [CUDA["lib64"]],
        extra_link_args=LINK_ARGS,
    ),
    Extension(
        "radarsimpy.simulator",
        ["src/simulator.pyx"],
        define_macros=MACROS,
        include_dirs=INCLUDE_DIRS + [CUDA["include"]],
        libraries=["radarsimc"],
        library_dirs=LIBRARY_DIRS + [CUDA["lib64"]],
        extra_link_args=LINK_ARGS,
    ),
]

setup(
    name="radarsimpy",
    cmdclass={"build_ext": build_ext},
    ext_modules=cythonize(
        ext_modules, annotate=False, compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()],
)
