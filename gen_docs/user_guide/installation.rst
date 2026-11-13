Installation
============

RadarSimPy is distributed as a pre-built module for Windows, Linux and macOS.
You do not install it with ``pip``; you copy the module folder into your
project.


Install the Module
------------------

1. Download the `pre-built module <https://radarsimx.com/product/radarsimpy/>`_
   for your platform.
2. Extract the downloaded archive.
3. Copy the ``radarsimpy/`` folder into your project directory.

Pre-built modules are available for Windows (x86_64), Linux (x86_64 and ARM64)
and macOS (Intel and Apple Silicon). See :doc:`dependencies` for the supported
operating system versions and the packages you need to install first.


Directory Structure
-------------------

After you copy the folder, your project should look like this:

.. code-block:: none

    your_project/
    ├── your_script.py          # Your radar simulation script
    ├── your_notebook.ipynb     # Or a Jupyter notebook
    └── radarsimpy/             # The RadarSimPy package
        ├── __init__.py         # Package initialization
        ├── radar.py            # Radar configuration
        ├── transmitter.py      # Transmitter definitions
        ├── receiver.py         # Receiver definitions
        ├── processing.py       # Signal processing utilities
        ├── tools.py            # Helper functions
        └── lib/                # Native library bindings

The ``radarsimpy/`` folder also contains compiled libraries for your platform:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Platform
     - C++ simulation engine
     - Python extension module
   * - Windows
     - ``radarsimcpp.dll``
     - ``simulator.*.pyd``
   * - Linux
     - ``libradarsimcpp.so``
     - ``simulator.*.so``
   * - macOS
     - ``libradarsimcpp.dylib``
     - ``simulator.*.so``

.. note::
   The extension module's file name includes the Python version and platform,
   for example ``simulator.cpython-312-x86_64-linux-gnu.so``, or
   ``simulator.cpython-312-aarch64-linux-gnu.so`` on Linux ARM64. The module
   only loads in the Python version it was built for.


Check the Installation
----------------------

Import the package and print its version:

.. code-block:: python

    import radarsimpy
    print(f"RadarSimPy version: {radarsimpy.__version__}")

You should see something like:

.. code-block:: none

    RadarSimPy version: 15.x.x


License
-------

Without a license, RadarSimPy runs in **free tier** mode, which has some
limits. A license file removes those limits.

Where to put the license file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The easiest way is to put the license file in the ``radarsimpy/`` folder.
RadarSimPy finds it automatically:

.. code-block:: none

    your_project/
    ├── your_script.py
    └── radarsimpy/
        ├── __init__.py
        ├── license_RadarSimPy_customer.lic    # Your license file
        ├── radar.py
        └── ...

The file name must match ``license_RadarSimPy_*.lic``.

You can put several license files in the folder. RadarSimPy tries each one, and
enables full functionality as soon as one of them is valid:

.. code-block:: none

    radarsimpy/
    ├── license_RadarSimPy_company.lic
    ├── license_RadarSimPy_backup.lic
    └── license_RadarSimPy_trial.lic

Loading a license from another location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the license file is stored somewhere else, such as a shared network drive,
load it at run time with ``set_license(path)``:

.. code-block:: python

    import radarsimpy

    radarsimpy.set_license("/path/to/license_RadarSimPy_customer.lic")

Checking the license status
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import radarsimpy

    if radarsimpy.is_licensed():
        info = radarsimpy.get_license_info()
        print(f"License info: {info}")
    else:
        print("Running in free tier mode with limitations")

Example output:

.. code-block:: none

    License info: Licensed to: Company Name (365 days remaining)

Free tier limits
^^^^^^^^^^^^^^^^

In free tier mode, RadarSimPy limits:

- the complexity of targets,
- the number of radar channels,
- some other features, as described in the documentation.

.. tip::
   To get a license file, visit `radarsimx.com
   <https://radarsimx.com/product/radarsimpy/>`_ or contact info@radarsimx.com.


Building from Source
--------------------

Most users should use the pre-built module. If you have access to the
``radarsimcpp`` source code and need to change it, see :doc:`build`.


See Also
--------

* :doc:`dependencies` - Platform requirements and Python packages
* :doc:`overview` - What RadarSimPy can simulate
* :doc:`examples` - Worked examples to run once installed
* :doc:`build` - Building from source (requires ``radarsimcpp`` access)
