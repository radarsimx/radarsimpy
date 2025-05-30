Build Instructions
==================

.. warning::
   Building radarsimpy requires access to the radarsimcpp source code. If you don't have access, 
   please use the pre-built module. For organizations seeking full source code access for customization 
   or advanced integration, please `submit Quote for Source Code <https://radarsimx.com/quote-for-source-code/>`_.

Windows
-------

**CPU Version**

.. code-block:: batch

    build_win.bat --arch cpu --test=on

**GPU Version (CUDA)**

.. code-block:: batch

    build_win.bat --arch gpu --test=on

Linux
-----

**CPU Version**

.. code-block:: bash

    ./build_linux.sh --arch=cpu --test=on

**GPU Version (CUDA)**

.. code-block:: bash

    ./build_linux.sh --arch=gpu --test=on

MacOS
-----

.. code-block:: bash

    ./build_macos.sh --arch=cpu --test=on
