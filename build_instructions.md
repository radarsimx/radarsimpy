# RadarSimPy Build Instructions

## Overview

RadarSimPy is a radar simulation library that provides both CPU and GPU acceleration capabilities. This document provides comprehensive build instructions for all supported platforms with detailed configuration options and troubleshooting guidance.

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
   build_win.bat
   
   # CPU build with custom options
   build_win.bat --arch=cpu --test=on
   
   # GPU build (requires CUDA)
   build_win.bat --arch=gpu --test=on
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
   chmod +x build_linux.sh
   ```

2. **Build the project**:

   ```bash
   # Basic CPU build
   ./build_linux.sh
   
   # CPU build with custom options
   ./build_linux.sh --arch=cpu --test=on
   
   # GPU build (requires CUDA)
   ./build_linux.sh --arch=gpu --test=on
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
   brew install gcc
   brew install cmake python3
   ```

### macOS Build Steps

1. **Make the build script executable**:

   ```bash
   chmod +x build_macos.sh
   ```

2. **Build the project**:

   ```bash
   # Basic CPU build
   ./build_macos.sh
   
   # CPU build with custom options
   ./build_macos.sh --arch=cpu --test=on
   ```

### macOS Build Options

- `--arch`: Architecture (`cpu` or `gpu`)
- `--test`: Enable testing (`on` or `off`)
- `--jobs`: Number of parallel build jobs (auto-detected by default)
- `--verbose`: Enable verbose output
- `--clean`: Clean build artifacts (`true` or `false`)
- `--tier`: Build tier (`standard` or `free`)
- `--cmake-args`: Additional CMake arguments

### macOS-Specific Notes

- **Apple Silicon (M1/M2/M3)**: Fully supported for CPU builds
- **Intel Macs**: Fully supported for CPU builds
- **GPU Support**: Limited CUDA support on macOS, CPU builds recommended
- **Compiler**: Uses Clang by default, GCC available via Homebrew

## Build Output and Verification

### Output Structure

After a successful build, the following structure will be created:

```text
radarsimpy/
├── __init__.py              # Main module initialization
├── radar.py                 # Radar simulation core
├── transmitter.py           # Transmitter configurations
├── receiver.py              # Receiver configurations
├── processing.py            # Signal processing utilities
├── tools.py                 # Utility functions
├── mesh_kit.py             # Mesh handling utilities
├── lib/                    # Library modules
│   └── __init__.py
├── libradarsimcpp.so       # C++ library (Linux)
├── libradarsimcpp.dylib    # C++ library (macOS)
├── radarsimcpp.dll         # C++ library (Windows)
└── simulator.*.pyd         # Python extension (platform-specific)
```

### Verification

1. **Test the installation**:

   ```python
   import radarsimpy as rs
   print(f"RadarSimPy version: {rs.__version__}")
   ```

2. **Run basic functionality test**:

   ```python
   import radarsimpy as rs
   import numpy as np
   
   # Create a simple radar configuration
   radar = rs.Radar(
       frequency=24e9,
       tx_power=10,
       noise_figure=8
   )
   print("Radar created successfully!")
   ```

3. **Check build logs**:

   - **Linux**: `build_YYYYMMDD_HHMMSS.log`
   - **macOS**: `build_macos_YYYYMMDD_HHMMSS.log`
   - **Windows**: `build_win_YYYYMMDD_HHMMSS.log`

---

**RadarSimPy** - A Radar Simulator Built with Python
Copyright (C) 2018 - PRESENT radarsimx.com
