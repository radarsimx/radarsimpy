Installation
============

RadarSimPy is distributed as a pre-built module for easy installation across Windows, Linux, and MacOS platforms.

Quick Start
-----------

Follow these steps to install RadarSimPy:

**Installation Steps**

1. **Download** the `pre-built module <https://radarsimx.com/product/radarsimpy/>`_ for your platform (Windows, Linux, or MacOS)
2. **Extract** the downloaded package to a temporary location
3. **Copy** the ``radarsimpy/`` folder into your project directory

**Supported Platforms**

- Windows 10/11 (x64)
- Linux (x64) - Ubuntu 20.04+ or equivalent
- MacOS 10.15+ (x64 and ARM64/M1/M2)

Directory Structure
-------------------

After installation, your project directory should have the following structure:

**Project Layout**

.. code-block:: none

    your_project/
    ├── your_script.py          # Your radar simulation script
    ├── your_notebook.ipynb     # Or Jupyter notebook
    └── radarsimpy/             # RadarSimPy package
        ├── __init__.py         # Package initialization
        ├── radar.py            # Radar configuration
        ├── transmitter.py      # Transmitter definitions
        ├── receiver.py         # Receiver definitions
        ├── processing.py       # Signal processing utilities
        ├── tools.py            # Helper functions
        └── lib/                # Native library bindings

Platform-Specific Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The package includes platform-specific compiled libraries:

**Windows**

.. code-block:: none

    radarsimpy/
    ├── radarsimcpp.dll         # C++ simulation engine
    └── simulator.*.pyd         # Python extension module

**Linux**

.. code-block:: none

    radarsimpy/
    ├── libradarsimcpp.so       # C++ simulation engine
    └── simulator.*.so          # Python extension module

**MacOS**

.. code-block:: none

    radarsimpy/
    ├── libradarsimcpp.dylib    # C++ simulation engine
    └── simulator.*.so          # Python extension module

.. note::
   The ``simulator.*.pyd`` or ``simulator.*.so`` filename includes Python version information (e.g., ``simulator.cpython-39-x86_64-linux-gnu.so``).

Verification
------------

Verify your installation by importing the package and checking the version:

.. code-block:: python

    import radarsimpy
    print(f"RadarSimPy version: {radarsimpy.__version__}")

You should see output similar to:

.. code-block:: none

    RadarSimPy version: 14.x.x

Building from Source
---------------------

Building ``radarsimpy`` requires access to the source code of ``radarsimcpp``. If you don't have access to ``radarsimcpp``, please use the `pre-built module <https://radarsimx.com/product/radarsimpy/>`_. 

For organizations seeking full source code access for customization or advanced integration, please submit a `Quote for Source Code <https://radarsimx.com/quote-for-source-code/>`_.
