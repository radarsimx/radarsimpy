.. RadarSimPy documentation master file, created by
   sphinx-quickstart on Wed Dec 16 08:59:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to RadarSimPy's Documentation!
======================================

|

.. image:: https://raw.githubusercontent.com/radarsimx/.github/main/profile/radarsimx_main.svg
   :width: 60%
   :alt: RadarSimX
   :target: https://radarsimx.com

|

User Guide
----------

Step-by-step guides to help you get started with RadarSimPy and understand key concepts.

.. toctree::
   :maxdepth: 1

   user_guide/index

**Getting Started**

* :doc:`user_guide/overview` - What RadarSimPy can model, simulate and process
* :doc:`user_guide/dependencies` - Python packages, platform and hardware requirements
* :doc:`user_guide/installation` - Installing the pre-built module and configuring a license

**Concepts & Conventions**

* :doc:`user_guide/system_model` - Transmitter, Receiver and Radar, and the shape of the output
* :doc:`user_guide/coordinate_systems` - Global and local frames, Euler angles, azimuth and elevation
* :doc:`user_guide/doppler_convention` - The sign of the Doppler frequency, and how to convert it

**Simulation Guides**

* :doc:`user_guide/transmitter` - Waveform, pulse train, modulation and the transmit array
* :doc:`user_guide/receiver` - Sampling, baseband type, noise budget, range gate and the receive array
* :doc:`user_guide/noise` - Receiver thermal noise and transmitter phase noise
* :doc:`user_guide/ray_tracing_simulation` - Ray density, fidelity level and target flags for 3D meshes
* :doc:`user_guide/animated_targets` - Driving targets from keyframed glTF 2.0 / GLB motion
* :doc:`user_guide/stretch_processing` - Range gating for long-range FMCW and stretch radars

**Resources & Development**

* :doc:`user_guide/examples` - Worked examples on radarsimx.com
* :doc:`user_guide/build` - Building from source, for developers with ``radarsimcpp`` access

API
---

Complete reference documentation for all RadarSimPy classes and functions.

.. toctree::
   :maxdepth: 2

   api/index
