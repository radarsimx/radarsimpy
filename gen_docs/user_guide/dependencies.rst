Dependencies
============

This page lists the Python packages, operating systems and hardware that
RadarSimPy needs.


Python Packages
---------------

Required
^^^^^^^^

* **Python**
* **NumPy**
* **SciPy**

Mesh library (for 3D models)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To simulate 3D mesh targets or compute RCS, install **at least one** of these
libraries:

* **trimesh** — recommended: small and has few dependencies
* **PyMeshLab**
* **PyVista**
* **meshio**

Animated models (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^

To load keyframe-animated glTF 2.0 / GLB models with
:mod:`radarsimpy.animation_kit`, also install:

* **pygltflib**

You do not need it for static ``.stl``, ``.ply`` or ``.obj`` files. See
:doc:`animated_targets`.

Installing them
^^^^^^^^^^^^^^^

Install everything in ``requirements.txt``:

.. code-block:: bash

    pip install -r requirements.txt

Or install the packages one by one:

.. code-block:: bash

    pip install numpy scipy trimesh

Add this for animated glTF models:

.. code-block:: bash

    pip install pygltflib


Operating System
----------------

RadarSimPy ships with compiled native libraries, so each platform has its own
requirements.

Windows
^^^^^^^

* **Operating system:** Windows 10 (version 1809 or later) or Windows 11
* **Architecture:** x64 (64-bit)
* **Visual C++ runtime:** `Microsoft Visual C++ Redistributable
  <https://aka.ms/vs/16/release/vc_redist.x64.exe>`_. Install it if you see a
  DLL error when importing RadarSimPy.

Linux
^^^^^

* **Operating system:** Ubuntu 22.04 LTS, 24.04 LTS or 26.04 LTS
* **Architecture:** x86_64 or ARM64 (aarch64). Download the package that
  matches your machine; ``uname -m`` prints ``x86_64`` or ``aarch64``.
* **Compiler runtime:** the GCC standard library, which Ubuntu installs by
  default.

If you see errors about missing system libraries, update them:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get install libstdc++6

**Other Linux distributions.** The pre-built modules are tested on Ubuntu only.
On other distributions:

1. Try the Ubuntu build first. It often works on related distributions such as
   Debian and Mint.
2. If it fails, check that your ``glibc`` and ``libstdc++`` versions are
   compatible.
3. If it still does not work, `request a custom build
   <https://radarsimx.com/request-a-custom-build/>`_ for your distribution.

macOS
^^^^^

* **Operating system:** a recent macOS release. Builds are produced and tested
  on macOS 26 "Tahoe" with Xcode 26.4.1. Older versions are not tested.
* **Architecture:** Intel x64 or Apple Silicon (ARM64).
* No extra runtime is needed.


GPU Acceleration
----------------

The Windows and Linux modules support NVIDIA GPUs through CUDA. The same module
works with or without a GPU:

* If a usable GPU is present, the simulation runs on it.
* Otherwise, it runs on the CPU, using all cores through OpenMP.

The CUDA runtime is built into the module, so you do **not** need to install
the CUDA Toolkit. To run on the GPU you need:

* **GPU:** NVIDIA, compute capability 7.5 (Turing) or newer. CUDA 13 no longer
  supports the older Maxwell, Pascal and Volta GPUs.
* **Driver:** an NVIDIA driver that supports CUDA 13.x. Use the latest driver
  for your GPU. On Ubuntu:

  .. code-block:: bash

      sudo apt-get install nvidia-driver-XXX

  Replace ``XXX`` with the latest driver version for your GPU.
* See the `NVIDIA CUDA Compatibility Guide
  <https://docs.nvidia.com/deploy/cuda-compatibility/#id1>`_ for details.

GPU acceleration is not available on macOS.


Hardware
--------

**Minimum**

* **CPU:** multi-core processor (4 cores recommended)
* **RAM:** 8 GB
* **Disk:** 500 MB for RadarSimPy

**Recommended for large simulations**

* **CPU:** 8 or more cores
* **RAM:** 16 GB or more
* **GPU:** NVIDIA GPU with 4 GB or more of memory
* **Disk:** SSD


See Also
--------

* :doc:`installation` - Installation instructions
* :doc:`animated_targets` - Simulating keyframe-animated 3D models
* :doc:`build` - Building from source (for developers with source access)
