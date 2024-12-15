Dependence
===========

- ``Python`` >= 3.9
- ``NumPy`` >= 2.0
- ``SciPy``
- ``PyMeshLab`` (*preferred*) or ``meshio``

**Windows**

    - `Visual C++ Runtime <https://aka.ms/vs/16/release/vc_redist.x64.exe/>`_
    - GPU version (CUDA12) - Check `Minimum Required Driver Versions <https://docs.nvidia.com/deploy/cuda-compatibility/#id1>`_

**Ubuntu 22.04**

    - ``GCC 11`` *(Included by default, no additional installation required)*
    - GPU version (CUDA12) - Check `Minimum Required Driver Versions <https://docs.nvidia.com/deploy/cuda-compatibility/#id1>`_

**Ubuntu 24.04**

    - ``GCC 13`` *(Included by default, no additional installation required)*
    - GPU version (CUDA12) - Check `Minimum Required Driver Versions <https://docs.nvidia.com/deploy/cuda-compatibility/#id1>`_

**Generic Linux x86-64**

    - Try the module for Ubuntu 22.04 or Ubuntu 24.04
    - `Request a Custom Build <https://radarsimx.com/request-a-custom-build/>`_ if it doesn't work

**MacOS Intel**

    - ``GCC 13``

    .. code-block:: bash

        brew install gcc@13

**MacOS Apple Silicon**

    - ``GCC 14``

    .. code-block:: bash
        
        brew install gcc@14
