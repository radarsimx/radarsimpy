"""
Setup script for RadarSimPy - A Python-based Radar Simulator

This setup script builds the radarsimpy package with C++ extensions and CUDA support.
It supports both CPU and GPU architectures.

Author: RadarSimX (info@radarsimx.com)
Website: https://radarsimx.com
License: See LICENSE file

::

    ██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝
    ██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝
    ██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗
    ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

"""

import argparse
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    from setuptools import Extension, find_packages, setup
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required dependencies: pip install numpy cython setuptools")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Package metadata
PACKAGE_NAME = "radarsimpy"
PACKAGE_DESCRIPTION = "A Python-based Radar Simulator"
PACKAGE_URL = "https://github.com/radarsimx/radarsimpy"
AUTHOR = "RadarSimX"
AUTHOR_EMAIL = "info@radarsimx.com"

# Build configuration constants
VALID_ARCHS = ["cpu", "gpu"]
DEFAULT_ARCH = "cpu"


def get_version() -> str:
    """Read version from package __init__.py file.

    :return: Package version string
    :rtype: str
    :raises RuntimeError: If version cannot be read from __init__.py
    """
    init_path = Path("src") / PACKAGE_NAME / "__init__.py"
    if not init_path.exists():
        raise RuntimeError(f"Could not find {init_path}")

    try:
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    # Extract version from line like: __version__ = "1.2.3"
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    return version
        raise RuntimeError(f"Could not find __version__ in {init_path}")
    except (OSError, UnicodeDecodeError) as e:
        raise RuntimeError(f"Could not read version from {init_path}: {e}") from e


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for build configuration.

    :return: Parsed command line arguments containing arch and verbose flags
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Build script for RadarSimPy with configurable options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--arch",
        choices=VALID_ARCHS,
        default=DEFAULT_ARCH,
        help=f"Build architecture (default: {DEFAULT_ARCH})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Parse only known args to allow setuptools args to pass through
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown

    return args


class BuildConfig:
    """Configuration class for build settings."""

    def __init__(self, arch: str) -> None:
        """Initialize BuildConfig with architecture settings.

        :param arch: Build architecture ('cpu' or 'gpu')
        :type arch: str
        """
        self.arch = arch
        self.os_type = platform.system()
        self.is_gpu = arch == "gpu"

        # Platform-specific settings
        self._configure_platform()
        self._configure_macros()
        self._configure_cuda()

    def _configure_platform(self) -> None:
        """Configure platform-specific build settings.

        Sets up compile args, link args, library directories, and libraries
        based on the current operating system (Linux, macOS, or Windows).
        """
        if self.os_type == "Linux":
            self.link_args = ["-Wl,-rpath,$ORIGIN"]
            self.compile_args = ["-std=c++20"]
            self.lib_dirs = ["src/radarsimcpp/build"]
            self.libs = []
            self.nvcc_name = "nvcc"
            self.cuda_lib_dir = "lib64"

        elif self.os_type == "Darwin":  # macOS
            self.compile_args = ["-std=c++20"]
            self.link_args = ["-Wl,-rpath,@loader_path"]
            self.lib_dirs = ["src/radarsimcpp/build"]
            self.libs = []
            self.nvcc_name = "nvcc"
            self.cuda_lib_dir = "lib64"

        elif self.os_type == "Windows":
            self.link_args = []
            self.compile_args = ["/std:c++20"]
            self.lib_dirs = ["src/radarsimcpp/build/Release"]
            self.libs = []
            self.nvcc_name = "nvcc.exe"
            self.cuda_lib_dir = "lib\\x64"

        else:
            raise OSError(f"Unsupported operating system: {self.os_type}")

        # Add CUDA libraries if GPU build
        if self.is_gpu:
            self.libs.append("cudart")

    def _configure_macros(self) -> None:
        """Configure preprocessor macros.

        Sets up macros for compilation including NumPy compatibility
        and CUDA support based on configuration.
        """
        self.macros: List[tuple[str, Optional[str]]] = [
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
        ]

        if self.is_gpu:
            self.macros.append(("_CUDA_", None))

    def _configure_cuda(self) -> None:
        """Configure CUDA-specific settings.

        Locates CUDA installation and configures CUDA-related settings
        if GPU architecture is specified.
        """
        self.cuda_config = None
        if self.is_gpu:
            try:
                self.cuda_config = self._locate_cuda()
                logger.info("Found CUDA at: %s", self.cuda_config["home"])
            except (EnvironmentError, OSError) as e:
                logger.error("Failed to locate CUDA: %s", e)
                raise

    def _locate_cuda(self) -> Dict[str, str]:
        """Locate CUDA installation.

        Searches for CUDA installation in standard locations and environment
        variables, then validates the installation paths.

        :return: Dictionary containing CUDA installation paths
        :rtype: Dict[str, str]
        :raises EnvironmentError: If CUDA installation cannot be found or is invalid
        """
        if "CUDA_PATH" in os.environ:
            home = os.environ["CUDA_PATH"]
            nvcc = os.path.join(home, "bin", self.nvcc_name)
        else:
            default_path = os.path.join(os.sep, "usr", "local", "cuda", "bin")
            nvcc = self._find_in_path(
                self.nvcc_name, os.environ.get("PATH", "") + os.pathsep + default_path
            )
            if nvcc is None:
                raise EnvironmentError(
                    f"The {self.nvcc_name} binary could not be located in your $PATH. "
                    "Either add it to your path, or set $CUDA_PATH"
                )
            home = os.path.dirname(os.path.dirname(nvcc))

        # Build include paths, only adding cccl if it exists
        include_paths = [os.path.join(home, "include")]
        cccl_path = os.path.join(home, "include", "cccl")
        if os.path.exists(cccl_path):
            include_paths.append(cccl_path)

        cuda_config = {
            "home": home,
            "nvcc": nvcc,
            "include": include_paths,
            "lib64": os.path.join(home, self.cuda_lib_dir),
        }

        # Verify all paths exist
        for key, path in cuda_config.items():
            if key == "include":
                # Handle include paths array
                for include_path in path:
                    if not os.path.exists(include_path):
                        raise EnvironmentError(
                            f"CUDA include path not found: {include_path}"
                        )
            else:
                # Handle single paths
                if not os.path.exists(path):
                    raise EnvironmentError(f"CUDA {key} path not found: {path}")

        return cuda_config

    def _find_in_path(self, name: str, path: str) -> Optional[str]:
        """Find executable in PATH.

        Searches for an executable file in the given PATH string.

        :param name: Name of the executable to find
        :type name: str
        :param path: PATH string to search in
        :type path: str
        :return: Absolute path to the executable if found, None otherwise
        :rtype: Optional[str]
        """
        for path_dir in path.split(os.pathsep):
            if not path_dir:
                continue
            full_path = os.path.join(path_dir, name)
            if os.path.exists(full_path):
                return os.path.abspath(full_path)
        return None

    def get_include_dirs(self) -> List[str]:
        """Get include directories.

        Returns a list of include directories for compilation,
        including C++ source includes, NumPy includes, and CUDA includes if applicable.

        :return: List of include directory paths
        :rtype: List[str]
        """
        include_dirs = [
            "src/radarsimcpp/includes",
            "src/radarsimcpp/includes/rsvector",
            numpy.get_include(),
        ]

        if self.cuda_config:
            include_dirs.extend(self.cuda_config["include"])

        return include_dirs

    def get_library_dirs(self) -> List[str]:
        """Get library directories.

        Returns a list of library directories for linking,
        including platform-specific lib directories and CUDA lib directories if applicable.

        :return: List of library directory paths
        :rtype: List[str]
        """
        lib_dirs = self.lib_dirs.copy()

        if self.cuda_config:
            lib_dirs.append(self.cuda_config["lib64"])

        return lib_dirs


def create_extension(name: str, sources: List[str], config: BuildConfig) -> Extension:
    """Create a Cython extension with the given configuration.

    :param name: Name of the extension module
    :type name: str
    :param sources: List of source files for the extension
    :type sources: List[str]
    :param config: Build configuration object
    :type config: BuildConfig
    :return: Configured Cython extension
    :rtype: Extension
    """
    return Extension(
        name,
        sources,
        define_macros=config.macros,
        include_dirs=config.get_include_dirs(),
        extra_compile_args=config.compile_args,
        libraries=["radarsimcpp"] + config.libs,
        library_dirs=config.get_library_dirs(),
        extra_link_args=config.link_args,
    )


def get_long_description() -> str:
    """Get long description from README file.

    Reads the README.md file and returns its content as the long description.
    Falls back to the package description if README.md is not available or readable.

    :return: Long description text for the package
    :rtype: str
    """
    readme_path = Path("README.md")
    if readme_path.exists():
        try:
            return readme_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Could not read README.md: %s", e)
    return PACKAGE_DESCRIPTION


def main() -> None:
    """Main setup function.

    Parses command line arguments, creates build configuration,
    defines extension modules, and runs the setup process.
    """
    # Parse arguments
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Building %s with arch=%s", PACKAGE_NAME, args.arch)

    # Get package version
    try:
        package_version = get_version()
        logger.info("Package version: %s", package_version)
    except RuntimeError as e:
        logger.error("Failed to get package version: %s", e)
        sys.exit(1)

    # Create build configuration
    try:
        config = BuildConfig(args.arch)
    except (EnvironmentError, OSError, ValueError) as e:
        logger.error("Failed to create build configuration: %s", e)
        sys.exit(1)

    # Define extension modules
    ext_modules = [
        create_extension(
            "radarsimpy.lib.cp_radarsimc",
            ["src/radarsimpy/lib/cp_radarsimc.pyx"],
            config,
        ),
        create_extension(
            "radarsimpy.simulator", ["src/radarsimpy/simulator.pyx"], config
        ),
        create_extension(
            "radarsimpy.license", ["src/radarsimpy/license.pyx"], config
        ),
    ]

    # Read package requirements
    requirements = []
    requirements_path = Path("requirements.txt")
    if requirements_path.exists():
        try:
            requirements = (
                requirements_path.read_text(encoding="utf-8").strip().split("\n")
            )
            requirements = [req.strip() for req in requirements if req.strip()]
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Could not read requirements.txt: %s", e)

    # Setup configuration
    setup(
        name=PACKAGE_NAME,
        version=package_version,
        description=PACKAGE_DESCRIPTION,
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=PACKAGE_URL,
        packages=find_packages(),
        install_requires=requirements,
        python_requires=">=3.10",
        cmdclass={"build_ext": build_ext},
        ext_modules=cythonize(
            ext_modules,
            annotate=False,
            compiler_directives={"language_level": "3"},
        ),
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: C++",
            "Programming Language :: Cython",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development :: Libraries :: Python Modules",
        ],
        keywords="radar simulation signal-processing python cython",
        project_urls={
            "Bug Reports": f"{PACKAGE_URL}/issues",
            "Source": PACKAGE_URL,
            "Documentation": "https://radarsimx.github.io/radarsimpy/",
        },
    )


if __name__ == "__main__":
    main()
