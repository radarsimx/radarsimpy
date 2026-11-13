Build Instructions
==================

This page explains how to build RadarSimPy from source. You only need to do
this if you have the ``radarsimcpp`` source code and want to modify it.

.. warning::
   Building ``radarsimpy`` requires the ``radarsimcpp`` source code. If you do
   not have it, use the `pre-built module
   <https://radarsimx.com/product/radarsimpy/>`_ instead.

   Organizations that need the full source code can `request a quote for
   source code <https://radarsimx.com/quote-for-source-code/>`_.


Prerequisites
-------------

All platforms
^^^^^^^^^^^^^

- Python >= 3.10 (tested on 3.10 – 3.14)
- CMake >= 3.18
- Git
- A C++ compiler with C++20 support

Per platform
^^^^^^^^^^^^

- **Windows:** Visual Studio 2022 or later, with the "Desktop development with
  C++" workload.
- **Linux** (x86_64 or ARM64): the distribution's default GCC — GCC 11 on
  Ubuntu 22.04, GCC 13 on Ubuntu 24.04, GCC 15 on Ubuntu 26.04.
- **macOS:** Xcode Command Line Tools (Clang).

GPU build
^^^^^^^^^

- NVIDIA CUDA Toolkit 13.x.

See :doc:`dependencies` for the GPU and driver that the resulting build needs
at run time.

Python packages
^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install -r requirements-dev.txt

Check your environment
^^^^^^^^^^^^^^^^^^^^^^

Run this script before building. It reports missing tools and configuration
problems, and suggests how to fix them:

.. code-block:: bash

    python build_config.py


Building
--------

Run the build script from the RadarSimPy root directory.

Windows
^^^^^^^

.. code-block:: batch

    REM CPU build
    build.bat

    REM CPU build, explicit options
    build.bat --arch=cpu --test=on

    REM GPU (CUDA) build
    build.bat --arch=gpu --test=on

Linux
^^^^^

The same commands work on x86_64 and ARM64 (aarch64).

Make the script executable the first time:

.. code-block:: bash

    chmod +x build.sh

Then build:

.. code-block:: bash

    # CPU build
    ./build.sh

    # CPU build, explicit options
    ./build.sh --arch=cpu --test=on

For a GPU (CUDA) build, put the CUDA Toolkit on your ``PATH`` first:

.. code-block:: bash

    export PATH=/usr/local/cuda/bin:$PATH
    export CUDA_PATH=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    ./build.sh --arch=gpu --test=on

To check that CUDA is installed:

.. code-block:: bash

    nvcc --version

macOS
^^^^^

Install the build tools:

.. code-block:: bash

    # Xcode Command Line Tools
    xcode-select --install

    # Homebrew (if not already installed)
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Build tools
    brew install cmake python3

    # OpenMP (recommended, for multi-threading)
    brew install libomp

.. note::
   The build also works without OpenMP, but the simulator then runs on a single
   thread and is slower. CMake detects OpenMP automatically.

Make the script executable the first time, then build:

.. code-block:: bash

    chmod +x build.sh

    # CPU build
    ./build.sh

    # CPU build, explicit options
    ./build.sh --arch=cpu --test=on

.. note::
   GPU (CUDA) builds are not supported on macOS.


Build Options
-------------

Both scripts accept the same core options. ``build.sh`` has three extra
options that ``build.bat`` does not.

.. list-table::
   :header-rows: 1
   :widths: 20 45 20 15

   * - Option
     - Description
     - Default
     - Available in
   * - ``--arch``
     - Build architecture: ``cpu`` or ``gpu``
     - ``cpu``
     - Both
   * - ``--test``
     - Run the unit tests after building: ``on`` or ``off``
     - ``on``
     - Both
   * - ``--license``
     - Enable license verification: ``on`` or ``off``
     - ``off``
     - Both
   * - ``--jobs``
     - Number of parallel build jobs
     - auto-detect
     - Both
   * - ``--deps``
     - Where to get the prebuilt third-party libraries. ``repo`` uses the
       ``libs/`` tree committed in the ``radarsimx-deps`` submodule;
       ``release`` downloads the ``radarsimx-deps`` GitHub release archives.
     - ``repo``
     - Both
   * - ``--clean``
     - Delete previous build artifacts first: ``true`` or ``false``
     - ``true``
     - ``build.sh``
   * - ``--verbose``
     - Show detailed build output
     - ``false``
     - ``build.sh``
   * - ``--cmake-args``
     - Extra arguments passed to CMake
     - none
     - ``build.sh``

Run ``build.bat --help`` or ``./build.sh --help`` for the full, current list.


What the Build Does
-------------------

The build scripts:

1. check that the required tools and libraries are installed,
2. configure CMake with the options you chose,
3. compile the ``radarsimcpp`` C++ simulation engine,
4. compile the Cython extensions that connect Python to the engine,
5. run the unit tests (if ``--test=on``),
6. collect the results into the ``radarsimpy`` module.

The output looks like this:

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

The platform-specific binaries are:

- **Windows:** ``radarsimcpp.dll``, ``simulator.xxx.pyd``
- **Linux:** ``libradarsimcpp.so``, ``simulator.xxx.so``
- **macOS:** ``libradarsimcpp.dylib``, ``simulator.xxx.so``


Building the Documentation
--------------------------

The documentation is built with Sphinx. Install the dependencies:

.. code-block:: bash

    pip install -r requirements-dev.txt

Then build the HTML pages.

Linux/macOS:

.. code-block:: bash

    cd gen_docs
    make html

Windows:

.. code-block:: batch

    cd gen_docs
    make.bat html

Open ``gen_docs/_build/html/index.html`` in your browser to view the result.

Other targets:

.. code-block:: bash

    make clean        # Delete previous builds
    make latexpdf     # PDF (requires LaTeX)
    make epub         # EPUB
    make singlehtml   # A single HTML page
    make linkcheck    # Check external links
    make doctest      # Run doctests


See Also
--------

* :doc:`installation` - Installing the pre-built module
* :doc:`dependencies` - Runtime platform and package requirements
* :doc:`examples` - Worked examples to run once built
