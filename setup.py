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

from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tier", required=False, help="`free` or `standard`")


args, unknown = ap.parse_known_args()
sys.argv = [sys.argv[0]] + unknown
# print(args.tier)

os_type = platform.system()  # 'Linux', 'Windows', 'macOS'

if os_type == "Linux":
    LINK_ARGS = ["-Wl,-rpath,$ORIGIN"]
    LIBRARY_DIRS = ["src/radarsimcpp/build"]
elif os_type == "Darwin":
    LINK_ARGS = ["-Wl,-rpath,$ORIGIN"]
    LIBRARY_DIRS = ["src/radarsimcpp/build"]
elif os_type == "Windows":
    LINK_ARGS = []
    LIBRARY_DIRS = ["src/radarsimcpp/build/Release"]

if args.tier == "free":
    MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"), ("_FREETIER_", 1)]
else:
    MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

INCLUDE_DIRS = ["src/radarsimcpp/includes", "src/radarsimcpp/includes/zpvector"]

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
