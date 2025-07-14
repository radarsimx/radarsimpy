# RadarSimPy Build Instructions

## Overview

RadarSimPy is a radar simulation library that provides both CPU and GPU acceleration capabilities. This document provides comprehensive build instructions for all supported platforms with detailed configuration options and troubleshooting guidance.

> Building `radarsimpy` requires to access the source code of `radarsimcpp`. If you don't have access to `radarsimcpp`, please use the [pre-built module](https://radarsimx.com/product/radarsimpy/). For organizations seeking full source code access for customization or advanced integration, please submit [Quote for Source Code](https://radarsimx.com/quote-for-source-code/).

## Prerequisites for All Platforms

### System Requirements

- **Python**: 3.9 or higher
- **CMake**: 3.20 or higher
- **C++ Compiler**: Support for C++20 or higher

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
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

### macOS-Specific Notes

- **Apple Silicon (M1/M2/M3)**: Fully supported for CPU builds
- **Intel Macs**: Fully supported for CPU builds
- **Compiler**: Uses Clang/Clang++ by default (provided by Xcode Command Line Tools)

## Build Output

### Output Structure

After a successful build, the following structure will be created:

```text
radarsimpy/
  ├── lib
    ├── __init__.py
    └── cp_radarsimc.**.pyd
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
Copyright (C) 2018 - PRESENT radarsimx.com
