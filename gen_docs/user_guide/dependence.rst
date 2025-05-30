Dependencies
============

Python Requirements
-------------------

* ``Python`` >= 3.9
* ``NumPy`` >= 2.0
* ``SciPy`` >= 1.11.0
* One of the following mesh processing libraries:

  * ``PyMeshLab`` >= 2022.2
  * ``PyVista`` >= 0.43.0
  * ``trimesh`` >= 4.0.0
  * ``meshio`` >= 5.3.0

System Requirements
-------------------

Windows
^^^^^^^
* `Visual C++ Runtime <https://aka.ms/vs/16/release/vc_redist.x64.exe/>`_
* For GPU version (CUDA 12): Latest NVIDIA drivers - See `compatibility guide <https://docs.nvidia.com/deploy/cuda-compatibility/#id1>`_

Linux
^^^^^

**Ubuntu 22.04/24.04**

* GCC (included by default)
* For GPU version (CUDA 12): Latest NVIDIA drivers

**Other Linux distributions**

* Try the Ubuntu builds first
* `Request a custom build <https://radarsimx.com/request-a-custom-build/>`_ if needed

MacOS
^^^^^

**Intel/Apple Silicon**

* GCC 14 installation::

    brew install gcc@14
