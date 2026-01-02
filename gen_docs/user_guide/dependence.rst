Dependencies
============

RadarSimPy requires both Python packages and system-level dependencies to function properly. This page outlines all requirements for successful installation and operation.

Python Requirements
-------------------

**Core Dependencies**

RadarSimPy requires the following Python packages:

* **Python** >= 3.10
* **NumPy** >= 2.0 - Numerical computing and array operations
* **SciPy** >= 1.11.0 - Scientific computing and signal processing algorithms

**Mesh Processing Libraries**

For 3D object simulation and RCS calculation, at least one of the following mesh processing libraries is required:

* **PyMeshLab** >= 2022.2 - Advanced mesh processing with MeshLab functionality
* **PyVista** >= 0.43.0 - 3D visualization and mesh analysis
* **trimesh** >= 4.0.0 - Lightweight mesh processing library
* **meshio** >= 5.3.0 - Mesh input/output for various formats

.. note::
   You only need to install one mesh processing library. ``trimesh`` is recommended for most users due to its simplicity and minimal dependencies.

**Installation**

Install all required dependencies using:

.. code-block:: bash

    pip install -r requirements.txt

Or install individually:

.. code-block:: bash

    pip install numpy scipy trimesh

System Requirements
-------------------

RadarSimPy includes pre-compiled native libraries that require specific system dependencies.

Windows
^^^^^^^

**Required Components**

* **Operating System**: Windows 10 (version 1809 or later) or Windows 11
* **Architecture**: x64 (64-bit)
* **Visual C++ Runtime**: `Microsoft Visual C++ Redistributable <https://aka.ms/vs/16/release/vc_redist.x64.exe>`_
  
  * Required for running the C++ simulation engine
  * Download and install if you encounter DLL errors

**GPU-Accelerated Version**

If using the CUDA-enabled version for GPU acceleration:

* **CUDA Version**: CUDA Toolkit 13.x compatible
* **GPU**: NVIDIA GPU with Compute Capability 5.0 or higher
* **Drivers**: Latest NVIDIA drivers for your GPU
* **Compatibility**: See `NVIDIA CUDA Compatibility Guide <https://docs.nvidia.com/deploy/cuda-compatibility/#id1>`_

Linux
^^^^^

**Ubuntu 22.04/24.04 (Recommended)**

* **Operating System**: Ubuntu 22.04 LTS or Ubuntu 24.04 LTS
* **Architecture**: x86_64 (64-bit)
* **Compiler Runtime**: GCC standard library (included by default)

If you encounter library issues, ensure system libraries are up to date:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install libstdc++6

**GPU-Accelerated Version**

For CUDA support on Ubuntu:

* **CUDA Version**: CUDA Toolkit 13.x
* **GPU**: NVIDIA GPU with Compute Capability 5.0 or higher
* **Drivers**: Install latest NVIDIA drivers:

  .. code-block:: bash

      sudo apt-get install nvidia-driver-XXX

  Replace ``XXX`` with the latest driver version number for your GPU.

**Other Linux Distributions**

RadarSimPy pre-built binaries are primarily tested on Ubuntu. For other distributions:

1. **Try Ubuntu builds first** - They often work on compatible distributions (Debian, Mint, etc.)
2. **Check dependencies** - Ensure compatible versions of ``glibc`` and ``libstdc++``
3. **Request custom build** - If the Ubuntu build doesn't work, `request a custom build <https://radarsimx.com/request-a-custom-build/>`_ for your specific distribution

MacOS
^^^^^

**Supported Versions**

* **Operating System**: MacOS 10.15 (Catalina) or later
* **Architecture**: 
  
  * Intel x64 (Intel-based Macs)
  * ARM64 (Apple Silicon M1/M2/M3)

**Platform-Specific Requirements**

* **Intel Macs**: Require GCC 14 compiler runtime
  
  .. code-block:: bash

      brew install gcc@14

* **Apple Silicon Macs**: Use default Clang runtime (no additional dependencies required)

.. note::
   Apple Silicon users benefit from native ARM64 performance without needing Rosetta 2.

Hardware Recommendations
------------------------

**Minimum Requirements**

* **CPU**: Multi-core processor (4 cores recommended)
* **RAM**: 8 GB
* **Storage**: 500 MB for RadarSimPy installation

**Recommended for Large Simulations**

* **CPU**: 8+ cores for parallel processing
* **RAM**: 16 GB or more
* **GPU**: NVIDIA GPU with 4+ GB VRAM (for GPU-accelerated version)
* **Storage**: SSD for faster data I/O

See Also
--------

* :doc:`installation` - Installation instructions
* :doc:`build` - Building from source (for developers with source access)
