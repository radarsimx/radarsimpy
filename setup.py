#!python

import platform

from setuptools import setup
from setuptools import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy


os_type = platform.system()  # 'Linux', 'Windows'

if os_type == "Linux":
    LINK_ARGS = ["-Wl,-rpath,$ORIGIN"]
    LIBRARY_DIRS = ["src/radarsimc/build"]
elif os_type == "Windows":
    LINK_ARGS = []
    LIBRARY_DIRS = ["src/radarsimc/build/Release"]

MACROS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
INCLUDE_DIRS = ["src/radarsimc/includes", "src/radarsimc/includes/zpvector"]

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
        ext_modules, annotate=False, compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()],
)
