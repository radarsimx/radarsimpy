[![Windows x64](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_windows_x64.yml/badge.svg)](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_windows_x64.yml)
[![Ubuntu 22.04 x64](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_ubuntu_22_x64.yml/badge.svg)](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_ubuntu_22_x64.yml)
[![Ubuntu 24.04 x64](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_ubuntu_24_x64.yml/badge.svg)](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_ubuntu_24_x64.yml)
[![MacOS x64](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_macos_x64.yml/badge.svg)](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_macos_x64.yml)
[![MacOS Apple Silicon](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_macos_arm64.yml/badge.svg)](https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_macos_arm64.yml)

[![Documentations](https://img.shields.io/github/v/tag/radarsimx/radarsimpy?label=Documentation&logo=read-the-docs)](https://radarsimx.github.io/radarsimpy/)
[![Download](https://img.shields.io/github/v/tag/radarsimx/radarsimpy?label=Download&logo=python)](https://radarsimx.com/product/radarsimpy/)

# RadarSimPy

<img src="./assets/radarsimpy.svg" alt="logo" width="200"/>

A **Radar** **Sim**ulator for **Py**thon

RadarSimPy is a powerful and versatile Python-based Radar Simulator that models radar transceivers and simulates baseband data from point targets and 3D models. Its signal processing tools offer range/Doppler processing, direction of arrival estimation, and beamforming using various cutting-edge techniques, and you can even characterize radar detection using Swerling’s models. Whether you’re a beginner or an advanced user, RadarSimPy is the perfect tool for anyone looking to develop new radar technologies or expand their knowledge of radar systems.

## Key Features

- ### Radar Modeling

  - Radar transceiver modeling
  - Arbitrary waveform (CW, FMCW, PMCW, Pulse, ...)
  - Phase noise
  - Phase/amplitude modulation (CDM, FDM, DDM, TDM, ...)
  - Fast-time/slow-time modulation

- ### Simulation

  - Simulation of radar baseband data from point targets
  - Simulation of radar baseband data from 3D modeled objects/environment
  - Simulation of interference
  - Simulation of target's RCS
  - Simulation of LiDAR point cloud from 3D modeled objects/environment

- ### Signal Processing

  - Range/Doppler processing
  - Direction of arrival (DoA) estimation
    - **MU**ltiple **SI**gnal **C**lassification (MUSIC) DoA estimations for a uniform linear array (ULA)
    - Root-MUSIC DoA estimation for a ULA
    - **E**stimation of **S**ignal **P**arameters via **R**ational **I**nvariance **T**echniques (ESPRIT) DoA estimation for a ULA
    - **I**terative **A**daptive **A**pproach (IAA) for amplitude and phase estimation
  - Beamformer
    - Capon beamformer
    - Bartlett beamformer
  - Constant false alarm rate (CFAR)
    - 1D/2D cell-averaging CFAR (CA-CFAR)
    - 1D/2D ordered-statistic CFAR (OS-CFAR)

- ### Characterization

  - Radar detection characteristics based on Swerling's models

## Dependence

- `Python` >= 3.9
- `NumPy` >= 2.0
- `SciPy`
- `PyMeshLab` (*preferred*) or `meshio`

- ### Windows

  - [`Visual C++ Runtime`](https://aka.ms/vs/16/release/vc_redist.x64.exe/)
  - GPU version (CUDA12) - Check [Minimum Required Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/#id1)

- ### Ubuntu 22.04

  - `GCC 11` *(Included by default, no additional installation required)*
  - GPU version (CUDA12) - Check [Minimum Required Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/#id1)

- ### Ubuntu 24.04

  - `GCC 13` *(Included by default, no additional installation required)*
  - GPU version (CUDA12) - Check [Minimum Required Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/#id1)

- ### Generic Linux x86-64

  - Try the module for Ubuntu 22.04 or Ubuntu 24.04
  - [Request a Custom Build](https://radarsimx.com/request-a-custom-build/) if it doesn't work

- ### MacOS Intel

  - `GCC 13`

    ```bash
    brew install gcc@13
    ```

- ### MacOS Apple Silicon

  - `GCC 14`

    ```bash
    brew install gcc@14
    ```

## Installation

Download the [pre-built module](https://radarsimx.com/product/radarsimpy/), and put the radarsimpy folder within your project folder as shown below:

---

- ### Windows

  - your_project.py
  - your_project.ipynb
  - radarsimpy
    - \_\_init__.py
    - radarsimcpp.dll
    - simulator.xxx.pyd
    - rt.xxx.pyd
    - radar.py
    - processing.py
    - ...

---

- ### Linux

  - your_project.py
  - your_project.ipynb
  - radarsimpy
    - \_\_init__.py
    - libradarsimcpp.so
    - simulator.xxx.so
    - rt.xxx.so
    - radar.py
    - processing.py
    - ...

---

- ### MacOS

  - your_project.py
  - your_project.ipynb
  - radarsimpy
    - \_\_init__.py
    - libradarsimcpp.dylib
    - simulator.xxx.so
    - rt.xxx.so
    - radar.py
    - processing.py
    - ...

---

## Acceleration

This module supports CPU/GPU parallelization.
CPU parallelization is implemented through OpenMP.
GPU parallelization (CUDA) has been added since v6.0.0.

|         | CPU (x86-64) | CPU (ARM64) | GPU (CUDA) |
|---------|--------------|-------------|------------|
| Windows | ✔️           | ❌️           | ✔️        |
| Linux   | ✔️           | ❌️           | ✔️        |
| MacOS   | ✔️           | ✔️          | ❌️         |

![performance](https://github.com/radarsimx/radarsimpy/raw/master/assets/performance.png)

## Coordinate Systems

### Global Coordinate

- **axis** (m): `[x, y, z]`
- **phi** (deg): angle on the x-y plane. 0 deg is the positive x-axis, 90 deg is the positive y-axis
- **theta** (deg): angle on the z-x plane. 0 deg is the positive z-axis, 90 deg is the x-y plane

<img src="./assets/phi_theta.svg" alt="phi_theta" width="400"/>

### Local Coordinate

- **yaw** (deg): rotation along the z-axis. Positive yaw rotates the object from the positive x-axis to the positive y-axis
- **pitch** (deg): rotation along the y-axis. Positive pitch rotates the object from the positive x-axis to the positive z-axis
- **roll** (deg): rotation along the x-axis. Positive roll rotates the object from the positive y-axis to the positive z-axis
- **origin** (m): `[x, y, z]`, the rotation centor of the object. Radar's origin is always at `[0, 0, 0]`

<img src="./assets/yaw_pitch_roll.svg" alt="yaw_pitch_roll" width="400"/>

- **azimuth** (deg): azimuth -90 ~ 90 deg equal to phi -90 ~ 90 deg
- **elevation** (deg): elevation -90 ~ 90 deg equal to theta 180 ~ 0 deg

<img src="./assets/azimuth_elevation.svg" alt="azimuth_elevation" width="400"/>

## Usage Examples

Check all the usage examples on [radarsimx.com](https://radarsimx.com/category/examples/). The source files of these examples are available at [radarsimnb repository](https://github.com/radarsimx/radarsimnb).

- ### Waveform

  - [Pulsed Radar](https://radarsimx.com/2024/09/13/pulsed-radar/)
  - [Interferometric Radar](https://radarsimx.com/2023/08/31/interferometric-radar/)
  - [Arbitrary waveform](https://radarsimx.com/2021/05/10/arbitrary-waveform/)
  - [PMCW radar](https://radarsimx.com/2019/05/24/pmcw-radar/)
  - [Doppler radar](https://radarsimx.com/2019/05/16/doppler-radar/)
  - [FMCW radar](https://radarsimx.com/2018/10/11/fmcw-radar/)

- ### MIMO

  - [Imaging radar](https://radarsimx.com/2022/12/02/imaging-radar/)
  - [DoA estimation](https://radarsimx.com/2022/12/12/doa-estimation/)
  - [TDM MIMO FMCW radar](https://radarsimx.com/2019/04/07/tdm-mimo-fmcw-radar/)

- ### Angle Estimation

  - [Imaging radar](https://radarsimx.com/2022/12/02/imaging-radar/)
  - [DoA estimation](https://radarsimx.com/2022/12/12/doa-estimation/)

- ### Ray Tracing

  - [Imaging radar](https://radarsimx.com/2022/12/02/imaging-radar/)
  - [Multi-path effect](https://radarsimx.com/2021/05/10/multi-path-effect/)
  - [Micro-Doppler](https://radarsimx.com/2021/05/10/micro-doppler/)
  - [Doppler of a turbine](https://radarsimx.com/2021/05/10/doppler-of-a-turbine/)
  - [FMCW radar with a car](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-car/)
  - [FMCW radar with a plate](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-plate/)
  - [FMCW radar with a corner reflector](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-corner-reflector/)
  - [Cross-Polarization and Co-Polarization RCS](https://radarsimx.com/2024/04/19/cross-polarization-and-co-polarization-rcs/)
  - [Car RCS](https://radarsimx.com/2021/05/10/car-rcs/)
  - [Plate RCS](https://radarsimx.com/2021/05/10/plate-rcs/)
  - [Corner reflector RCS](https://radarsimx.com/2021/05/10/corner-reflector-rcs/)
  - [LIDAR point cloud](https://radarsimx.com/2020/02/05/lidar-point-cloud/)

- ### CFAR

  - [FMCW radar with a corner reflector](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-corner-reflector/)
  - [CFAR](https://radarsimx.com/2021/01/10/cfar/)

- ### Interference

  - [Interference](https://radarsimx.com/2023/01/13/interference/)

- ### System Characterization

  - [FMCW Radar Link Budget - Ideal Point Target](https://radarsimx.com/2024/10/11/fmcw-radar-link-budget-ideal-point-target/)
  - [Phase noise](https://radarsimx.com/2021/01/13/phase-noise/)
  - [Receiver operating characteristic (ROC)](https://radarsimx.com/2019/10/06/receiver-operating-characteristic/)

- ### Radar Cross Section

  - [Cross-Polarization and Co-Polarization RCS](https://radarsimx.com/2024/04/19/cross-polarization-and-co-polarization-rcs/)
  - [Car RCS](https://radarsimx.com/2021/05/10/car-rcs/)
  - [Plate RCS](https://radarsimx.com/2021/05/10/plate-rcs/)
  - [Corner reflector RCS](https://radarsimx.com/2021/05/10/corner-reflector-rcs/)

- ### LiDAR

  - [LIDAR point cloud](https://radarsimx.com/2020/02/05/lidar-point-cloud/)

## Build

> **Building `radarsimpy` requires to access the source code of `radarsimcpp`. If you don't have access to `radarsimcpp`, please use the [pre-built module](https://radarsimx.com/product/radarsimpy/). For organizations seeking full source code access for customization or advanced integration, please submit [Quote for Source Code](https://radarsimx.com/quote-for-source-code/).**

- ### Windows (MSVC)

  ```batch
  build_win.bat --arch cpu --test on
  ```

  Build for GPU (CUDA)

  ```batch
  build_win.bat --arch gpu --test on
  ```

- ### Linux (GCC)

  ```bash
  ./build_linux.sh --arch=cpu --test=on
  ```

  Build for GPU (CUDA)

  ```bash
  ./build_linux.sh --arch=gpu --test=on
  ```

- ### MacOS (GCC)

  ```bash
  ./build_macos.sh --arch=cpu --test=on
  ```

## API Reference

Please check the [Documentation](https://radarsimx.github.io/radarsimpy/)
