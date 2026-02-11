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

License Configuration
---------------------

RadarSimPy supports both free tier and licensed operation modes. License files enable access to advanced features and remove free tier limitations.

License File Placement
^^^^^^^^^^^^^^^^^^^^^^

**Automatic Detection**

The simplest way to activate your license is to place the license file in the ``radarsimpy/`` package directory:

.. code-block:: none

    your_project/
    ├── your_script.py
    └── radarsimpy/
        ├── __init__.py
        ├── license_RadarSimPy_customer.lic    # Your license file
        ├── radar.py
        └── ...

License files must follow the naming pattern: ``license_RadarSimPy_*.lic``

.. note::
   RadarSimPy automatically searches for and validates all ``license_RadarSimPy_*.lic`` files in the package directory. As long as one valid license is found, full functionality is enabled.

**Multiple License Files**

You can place multiple license files in the directory. The system will try each one until a valid license is found:

.. code-block:: none

    radarsimpy/
    ├── license_RadarSimPy_company.lic
    ├── license_RadarSimPy_backup.lic
    └── license_RadarSimPy_trial.lic

Checking License Status
^^^^^^^^^^^^^^^^^^^^^^^^

You can verify your license status at any time:

.. code-block:: python

    import radarsimpy
    
    # Check license status
    if radarsimpy.is_licensed():
        # Get license information
        info = radarsimpy.get_license_info()
        print(f"License info: {info}")
    else:
        print("Running in free tier mode with limitations")

**Example Output**

.. code-block:: none

    License info: Licensed to: Company Name (365 days remaining)

Free Tier Mode
^^^^^^^^^^^^^^

If no valid license file is found, RadarSimPy automatically operates in free tier mode with certain limitations:

- Limited target complexity
- Reduced simulation fidelity options
- Other feature restrictions as documented

.. tip::
   To obtain a license file, visit `radarsimx.com <https://radarsimx.com/product/radarsimpy/>`_ or contact info@radarsimx.com.

Building from Source
---------------------

Building ``radarsimpy`` requires access to the source code of ``radarsimcpp``. If you don't have access to ``radarsimcpp``, please use the `pre-built module <https://radarsimx.com/product/radarsimpy/>`_. 

For organizations seeking full source code access for customization or advanced integration, please submit a `Quote for Source Code <https://radarsimx.com/quote-for-source-code/>`_.
