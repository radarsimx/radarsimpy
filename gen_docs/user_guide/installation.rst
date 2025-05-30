Installation
============

Quick Start
-----------

1. Download the `pre-built module <https://radarsimx.com/product/radarsimpy/>`_
2. Extract the package
3. Place the radarsimpy folder in your project directory

Directory Structure
-------------------

The following shows the typical project structure after installation:

Common Files
^^^^^^^^^^^^
.. code-block:: none

    your_project/
    ├── your_project.py
    ├── your_project.ipynb
    └── radarsimpy/
        ├── __init__.py
        ├── radar.py
        ├── processing.py
        └── ...

Platform-Specific Files
^^^^^^^^^^^^^^^^^^^^^^^

**Windows**::

    radarsimpy/
    ├── radarsimcpp.dll
    └── simulator.xxx.pyd

**Linux**::

    radarsimpy/
    ├── libradarsimcpp.so
    └── simulator.xxx.so

**MacOS**::

    radarsimpy/
    ├── libradarsimcpp.dylib
    └── simulator.xxx.so

Verification
------------

To verify the installation, run:

.. code-block:: python

    import radarsimpy
    print(radarsimpy.__version__)
