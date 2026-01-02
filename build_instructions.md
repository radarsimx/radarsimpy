# RadarSimPy Build Instructions

## Overview

RadarSimPy is a radar simulation library that provides both CPU and GPU acceleration capabilities. This document provides comprehensive build instructions for all supported platforms with detailed configuration options and troubleshooting guidance.

> Building `radarsimpy` requires to access the source code of `radarsimcpp`. If you don't have access to `radarsimcpp`, please use the [pre-built module](https://radarsimx.com/product/radarsimpy/). For organizations seeking full source code access for customization or advanced integration, please submit [Quote for Source Code](https://radarsimx.com/quote-for-source-code/).

## Project Structure

The RadarSimPy repository has the following structure:

```text
radarsimpy/
├── assets/                       # Documentation assets and diagrams
├── gen_docs/                     # Documentation generation files
├── models/                       # 3D model files for simulation
├── references/                   # Research papers and documentation
├── src/                          # Source code
│   ├── radarsimcpp/              # C++ source code
│   │   ├── gtest/                # Google Test framework (RadarSimCpp)
│   │   ├── hdf5-lib-build/       # HDF5 library build files
│   │   │   ├── hdf5/             # HDF5 source code (HDF Group)
│   │   │   ├── libs/             # Platform-specific precompiled libraries
│   │   │   │   ├── lib_linux_gcc11_x86_64/
│   │   │   │   ├── lib_macos_arm64/
│   │   │   │   ├── lib_macos_x86_64/
│   │   │   │   └── lib_win_x86_64/
│   │   │   ├── build.bat         # Windows build script
│   │   │   ├── build.sh          # Linux/macOS build script
│   │   │   └── README.md
│   │   │   # Note: RadarSimCpp uses precompiled HDF5 libraries.
│   │   │   # To build HDF5 from source, see: https://github.com/radarsimx/hdf5-lib-build
│   │   ├── includes/             # Header files
│   │   │   ├── libs/             # Core library headers
│   │   │   └── rsvector/         # Custom vector implementations
│   │   ├── src/                  # C++/CUDA implementation files
│   │   ├── CMakeLists.txt        # CMake configuration (Config path to precomipled HDF5 library)
│   │   └── README.md
│   └── radarsimpy/               # Python source code
│       ├── includes/             # Cython declaration files
│       └── lib/                  # Cython wrapper library
├── tests/                        # Test suite (RadarSimPy)
├── batch_build.bat               # Windows batch build script
├── batch_build.sh                # Linux/macOS batch build script
├── build.bat                     # Windows build script
├── build.sh                      # Linux/macOS build script
├── build_config.py               # Build configuration validation
├── build_instructions.md         # This file
├── LICENSE                       # License file (RadarSimPy)
├── README.md                     # Project overview
├── requirements-dev.txt          # Development dependencies
├── requirements.txt              # Runtime dependencies
└── setup.py                      # Python package setup
```

### Key Directories

- **`src/`**: Contains the source code for both Python and C++ components
- **`tests/`**: Comprehensive test suite covering all functionality
- **`gen_docs/`**: Sphinx documentation configuration and source files

## Prerequisites for All Platforms

### System Requirements

- **Python**: 3.10 or higher
- **CMake**: 3.20 or higher
- **C++ Compiler**: Support for C++20 or higher

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements-dev.txt
```

## Validate Build Environment

Before building, validate your environment to catch missing dependencies, outdated CMake, or compiler issues:

```bash
python build_config.py
```

- If you see `All checks passed!`, you are ready to build.
- If you see errors, follow the suggestions to resolve them (e.g., install missing packages, update CMake, or fix compiler setup).

## Windows (MSVC)

### Windows Prerequisites

1. **Visual Studio 2022 or later** with "Desktop development with C++"
2. **CMake** (Windows x64 Installer) - [Download here](https://cmake.org/download/)
3. **CUDA Toolkit 12** (Optional, required only for GPU builds) - [Download here](https://developer.nvidia.com/cuda-downloads)

### Windows Build Steps

1. **Open Command Prompt or PowerShell**

2. **Navigate to the project directory**:

   ```cmd
   cd path\to\radarsimpy
   ```

3. **Build the project**:

   ```batch
   # Basic CPU build
   build.bat

   # CPU build with custom options
   build.bat --arch=cpu --test=on

   # GPU build (requires CUDA)
   build.bat --arch=gpu --test=on
   ```

### Windows Build Options

- `--arch`: Architecture (`cpu` or `gpu`)
- `--test`: Enable testing (`on` or `off`)
- `--tier`: Build tier (`standard` or `free`)

## Ubuntu 22.04/24.04 and Other Linux Distributions

### Linux Prerequisites

1. **Build tools and dependencies**:

   ```bash
   # Update package lists
   sudo apt-get update

   # Install essential build tools
   sudo apt-get install -y build-essential cmake python3-dev python3-pip

   # Install CMake (if system version is too old)
   sudo snap install cmake --classic
   ```

2. **For GPU builds** (Optional):

   ```bash
   # Install CUDA following NVIDIA's official guide:
   # https://developer.nvidia.com/cuda-downloads

   # After CUDA installation, set up environment variables
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc

   # Verify CUDA installation
   nvcc --version
   ```

### Linux Build Steps

1. **Make the build script executable**:

   ```bash
   chmod +x build.sh
   ```

2. **Build the project**:

   ```bash
   # Basic CPU build
   ./build.sh

   # CPU build with custom options
   ./build.sh --arch=cpu --test=on

   # GPU build (requires CUDA)
   ./build.sh --arch=gpu --test=on
   ```

### Linux Build Options

- `--arch`: Architecture (`cpu` or `gpu`)
- `--test`: Enable testing (`on` or `off`)
- `--tier`: Build tier (`standard` or `free`)

## macOS

### macOS Prerequisites

1. **Xcode Command Line Tools**:

   ```bash
   # Install Xcode Command Line Tools
   xcode-select --install
   ```

2. **Homebrew** (recommended package manager):

   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Build tools**:

   ```bash
   # Install required tools
   brew install cmake python3
   ```

4. **OpenMP support (recommended for better performance)**:

   ```bash
   # Install OpenMP for parallel processing
   brew install libomp
   ```

   **Note**: Without OpenMP, the build will succeed but with reduced performance. The CMake configuration will automatically detect OpenMP availability and provide fallback options.

### macOS Build Steps

1. **Make the build script executable**:

   ```bash
   chmod +x build.sh
   ```

2. **Build the project**:

   ```bash
   # Basic CPU build
   ./build.sh

   # CPU build with custom options
   ./build.sh --arch=cpu --test=on
   ```

### macOS Build Options

- `--arch`: Architecture (`cpu`)
- `--test`: Enable testing (`on` or `off`)
- `--jobs`: Number of parallel build jobs (auto-detected by default)
- `--verbose`: Enable verbose output
- `--clean`: Clean build artifacts (`true` or `false`)
- `--tier`: Build tier (`standard` or `free`)
- `--cmake-args`: Additional CMake arguments

## Building Documentation

RadarSimPy includes comprehensive documentation built with Sphinx. The documentation source files are located in the `gen_docs/` directory and include API references, user guides, and examples.

### Documentation Prerequisites

Install the required documentation dependencies:

```bash
# Install all development dependencies (includes Sphinx and extensions)
pip install -r requirements-dev.txt
```

The `requirements-dev.txt` file includes all necessary documentation tools:

- `sphinx`: Documentation generation framework
- `pydata-sphinx-theme`: PyData community theme for professional documentation
- Other required extensions and dependencies

### Documentation Structure

The documentation system includes:

```text
gen_docs/
├── conf.py                       # Sphinx configuration
├── index.rst                     # Main documentation index
├── make.bat                      # Windows build script
├── Makefile                      # Linux/macOS build script
├── api/                          # API documentation
├── user_guide/                   # User guides and tutorials
├── _build/                       # Generated documentation output
└── _static/                      # Static assets (images, CSS, etc.)
```

### Building Documentation

#### Prerequisites Check

Before building documentation, ensure RadarSimPy is properly built:

```bash
# Validate build environment
python build_config.py

# Build the project first (required for API documentation)
./build.sh --tier=standard --arch=cpu --test=off    # Linux/macOS
# OR
build.bat --tier=standard --arch=cpu --test=off     # Windows
```

#### Linux/macOS Documentation Build

1. **Navigate to the documentation directory**:

   ```bash
   cd gen_docs
   ```

2. **Build HTML documentation**:

   ```bash
   make html
   ```

3. **Build other formats** (optional):

   ```bash
   # Clean previous builds
   make clean

   # Build PDF documentation (requires LaTeX)
   make latexpdf

   # Build EPUB format
   make epub

   # Build for single-page HTML
   make singlehtml

   # Check external links
   make linkcheck

   # Run doctests
   make doctest
   ```

#### Windows Documentation Build

1. **Navigate to the documentation directory**:

   ```cmd
   cd gen_docs
   ```

2. **Build HTML documentation**:

   ```cmd
   make.bat html
   ```

3. **Build other formats** (optional):

   ```cmd
   # Clean previous builds
   make.bat clean

   # Build PDF documentation (requires LaTeX)
   make.bat latexpdf

   # Build EPUB format
   make.bat epub

   # Build for single-page HTML
   make.bat singlehtml
   ```

### Viewing the Documentation

After building, the documentation will be available in:

- **HTML**: `gen_docs/_build/html/index.html`
- **PDF**: `gen_docs/_build/latex/radarsimpy.pdf` (if built)
- **EPUB**: `gen_docs/_build/epub/RadarSimPy.epub` (if built)

Open the HTML version in your web browser:

```bash
# Linux/macOS
open gen_docs/_build/html/index.html

# Windows
start gen_docs/_build/html/index.html
```

## Build Output

### Output Structure

After a successful build, the following structure will be created:

```text
radarsimpy/
  ├── lib
  │   ├── __init__.py
  │   └── cp_radarsimc.**.pyd
  ├── __init__.py
  ├── [platform-specific binaries]
  ├── radar.py
  ├── processing.py
  └── ...
```

**Platform-specific binaries:**

- **Windows:** `radarsimcpp.dll`, `simulator.xxx.pyd`
- **Linux:** `libradarsimcpp.so`, `simulator.xxx.so`
- **MacOS:** `libradarsimcpp.dylib`, `simulator.xxx.so`

---

**RadarSimPy** - A Radar Simulator Built with Python
Copyright © 2018 - PRESENT radarsimx.com
