Installation
=============

.. image:: https://img.shields.io/github/v/tag/radarsimx/radarsimpy?label=Download
  :height: 20
  :target: https://radarsimx.com/product/radarsimpy/

To use the module, please put the radarsimpy folder within your project folder as shown below.

**Windows**

::

    - your_project.py
    - your_project.ipynb
    - radarsimpy
        - __init__.py
        - radarsimcpp.dll
        - simulator.xxx.pyd
        - rt.xxx.pyd
        - radar.py
        - processing.py
        - ...


**Linux**

::

    - your_project.py
    - your_project.ipynb
    - radarsimpy
        - __init__.py
        - libradarsimcpp.so
        - simulator.xxx.so
        - rt.xxx.so
        - radar.py
        - processing.py
        - ...

**MacOS**

::

    - your_project.py
    - your_project.ipynb
    - radarsimpy
        - __init__.py
        - libradarsimcpp.dylib
        - simulator.xxx.so
        - rt.xxx.so
        - radar.py
        - processing.py
        - ...