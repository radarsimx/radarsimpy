Build Instructions
==================

This page provides instructions for building RadarSimPy from source. Building from source is only necessary if you have access to the radarsimcpp source code and need custom modifications.

.. warning::
   Building ``radarsimpy`` requires access to the ``radarsimcpp`` source code. If you don't have access, 
   please use the `pre-built module <https://radarsimx.com/product/radarsimpy/>`_. 
   
   For organizations seeking full source code access for customization or advanced integration, 
   please `submit a Quote for Source Code <https://radarsimx.com/quote-for-source-code/>`_.

Prerequisites
-------------

Before building RadarSimPy, ensure you have the following installed:

**Common Requirements**

- Python >= 3.10 (tested on 3.10 – 3.14)
- CMake >= 3.18
- Git (for cloning repositories)
- C++ compiler with C++20 support

**Platform-Specific Requirements**

- **Windows**: Visual Studio 2022 or later with "Desktop development with C++" workload
- **Linux**: GCC 11 (Ubuntu 22.04), GCC 13 (Ubuntu 24.04), or GCC 15 (Ubuntu 26.04) — the distribution's default compiler
- **MacOS**: Xcode Command Line Tools (Clang)

**For GPU Version**

- NVIDIA CUDA Toolkit 13.x
- NVIDIA GPU with Compute Capability 7.5 (Turing) or higher

.. note::
   CUDA Toolkit 13 dropped offline compilation support for pre-Turing architectures (Maxwell, Pascal, Volta). GPUs older than Turing are not supported by CUDA13 builds.

**Python Dependencies**

Install required Python packages:

.. code-block:: bash

    pip install -r requirements-dev.txt

Validate Build Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before building, validate your environment to catch missing dependencies or configuration issues:

.. code-block:: bash

    python build_config.py

If all checks pass, you're ready to build. If errors occur, follow the suggestions to resolve them.

Build Commands
--------------

Windows
^^^^^^^

Navigate to the RadarSimPy root directory and run the appropriate build script.

**Basic CPU Build**

.. code-block:: batch

    build.bat

**CPU Build with Options**

.. code-block:: batch

    build.bat --arch=cpu --test=on

**GPU Build (CUDA)**

.. code-block:: batch

    build.bat --arch=gpu --test=on

**Build Options**

- ``--arch`` - Architecture: ``cpu`` or ``gpu`` (default: ``cpu``)
- ``--test`` - Enable testing: ``on`` or ``off`` (default: ``on``)
- ``--license`` - Enable license verification: ``on`` or ``off`` (default: ``off``)
- ``--jobs`` - Number of parallel build jobs (auto-detected by default)

Linux
^^^^^

Make the build script executable (first time only):

.. code-block:: bash

    chmod +x build.sh

**Basic CPU Build**

.. code-block:: bash

    ./build.sh

**CPU Build with Options**

.. code-block:: bash

    ./build.sh --arch=cpu --test=on

**GPU Build (CUDA)**

Ensure CUDA Toolkit is installed and in your PATH:

.. code-block:: bash

    export PATH=/usr/local/cuda/bin:$PATH
    export CUDA_PATH=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ./build.sh --arch=gpu --test=on

Verify CUDA installation:

.. code-block:: bash

    nvcc --version

**Build Options**

- ``--arch`` - Architecture: ``cpu`` or ``gpu`` (default: ``cpu``)
- ``--test`` - Enable testing: ``on`` or ``off`` (default: ``on``)
- ``--license`` - Enable license verification: ``on`` or ``off`` (default: ``off``)
- ``--jobs`` - Number of parallel build jobs (auto-detected by default)

MacOS
^^^^^

**Install Build Tools**

.. code-block:: bash

    # Install Xcode Command Line Tools
    xcode-select --install
    
    # Install Homebrew (if not already installed)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Install required tools
    brew install cmake python3
    
    # Install OpenMP (recommended for better performance)
    brew install libomp

.. note::
   Without OpenMP, the build will succeed but with reduced performance. CMake will automatically detect OpenMP availability.

Make the build script executable (first time only):

.. code-block:: bash

    chmod +x build.sh

**Basic CPU Build**

.. code-block:: bash

    ./build.sh

**CPU Build with Options**

.. code-block:: bash

    ./build.sh --arch=cpu --test=on

.. note::
   GPU (CUDA) builds are not supported on MacOS. Apple Silicon Macs use native ARM64 CPU optimization.

**Build Options**

- ``--arch`` - Architecture: ``cpu`` (only option for MacOS)
- ``--test`` - Enable testing: ``on`` or ``off`` (default: ``on``)
- ``--license`` - Enable license verification: ``on`` or ``off`` (default: ``off``)
- ``--jobs`` - Number of parallel build jobs (auto-detected by default)
- ``--verbose`` - Enable verbose output
- ``--clean`` - Clean build artifacts: ``true`` or ``false``
- ``--cmake-args`` - Additional CMake arguments

Build Process
-------------

The build scripts perform the following steps:

1. **Validate Dependencies** - Check for required tools and libraries
2. **Configure CMake** - Generate build files with specified options
3. **Compile C++ Code** - Build the radarsimcpp simulation engine
4. **Build Python Extensions** - Compile Cython extensions and create Python bindings
5. **Run Tests** (if ``--test=on``) - Execute unit tests to verify the build
6. **Create Package** - Package the built files into the radarsimpy module

Build Output
^^^^^^^^^^^^

After a successful build, the following structure will be created:

.. code-block:: none

    radarsimpy/
      ├── lib/
      │   ├── __init__.py
      │   └── cp_radarsimc.**.pyd
      ├── __init__.py
      ├── [platform-specific binaries]
      ├── radar.py
      ├── processing.py
      └── ...

**Platform-specific binaries:**

- **Windows**: ``radarsimcpp.dll``, ``simulator.xxx.pyd``
- **Linux**: ``libradarsimcpp.so``, ``simulator.xxx.so``
- **MacOS**: ``libradarsimcpp.dylib``, ``simulator.xxx.so``

Building Documentation
----------------------

RadarSimPy includes comprehensive documentation built with Sphinx.

**Prerequisites**

Install documentation dependencies:

.. code-block:: bash

    pip install -r requirements-dev.txt

**Build Documentation**

Linux/MacOS:

.. code-block:: bash

    cd gen_docs
    make html

Windows:

.. code-block:: batch

    cd gen_docs
    make.bat html

**View Documentation**

After building, open ``gen_docs/_build/html/index.html`` in your browser.

**Other Documentation Formats**

.. code-block:: bash

    make clean        # Clean previous builds
    make latexpdf     # Build PDF (requires LaTeX)
    make epub         # Build EPUB format
    make singlehtml   # Single-page HTML
    make linkcheck    # Check external links
    make doctest      # Run doctests

See Also
--------

- :doc:`installation` - Installation instructions for pre-built modules
- :doc:`dependence` - Detailed dependency information
- :doc:`examples` - Usage examples and tutorials
