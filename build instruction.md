# RadarSimPy Build Instructions

## Prerequisites for All Platforms

- Python 3.9 or higher
- CMake 3.20 or higher
- C++ compiler with C++20 support
- Python dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Windows (MSVC)

1. Install required tools:
   - [Microsoft Visual Studio 2022](https://visualstudio.microsoft.com/) with "Desktop development with C++" workload
   - [CMake](https://cmake.org/download/) (Windows x64 Installer)
   - [CUDA Toolkit 12](https://developer.nvidia.com/cuda-downloads) (Required only for GPU version)

2. Build the project:

   ```batch
   # For CPU version
   build_win.bat --arch=cpu --test=on

   # For GPU version (requires CUDA)
   build_win.bat --arch=gpu --test=on
   ```

## Ubuntu 22.04/24.04

1. Install system dependencies:

   ```bash
   # Basic development tools
   sudo apt-get update
   sudo apt-get install -y build-essential
   sudo snap install cmake --classic

   # For GPU version only
   # Install CUDA following NVIDIA's official guide:
   # https://developer.nvidia.com/cuda-downloads

   # Set up CUDA environment variables
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export CUDA_PATH=/usr/local/cuda' >> ~/.bashrc
   source ~/.bashrc
   ```

2. Build the project:

   ```bash
   # For CPU version
   ./build_linux.sh --arch=cpu --test=on

   # For GPU version
   ./build_linux.sh --arch=gpu --test=on
   ```

## MacOS

1. Install build tools:

   ```bash
   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install required tools
   brew install cmake gcc
   ```

2. Build the project:

   ```bash
   ./build_macos.sh --arch=cpu --test=on
   ```

## Build Output

The compiled module will be available in the `radarsimpy` folder.

## Build Options

- `--arch`: Build architecture (`cpu` or `gpu`)
- `--test`: Enable testing (`on` or `off`)

## Troubleshooting

- If CMake fails to find CUDA, ensure CUDA_PATH environment variable is set correctly
- For Windows GPU builds, ensure you have compatible NVIDIA drivers installed
- For Linux GPU builds, ensure nvidia-smi works and shows your GPU
