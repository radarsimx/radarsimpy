![Windows Test](https://github.com/radarsimx/radarsimpy/workflows/Windows%20Test/badge.svg)
![Linux Test](https://github.com/radarsimx/radarsimpy/workflows/Linux%20Test/badge.svg)
![Python support](https://img.shields.io/badge/Python-3.7%7C3.8%7C3.9%7C3.10%7C3.11-blue?style=flat&logo=python)
[![Documentations](https://img.shields.io/badge/Documentation-latest-brightgree?style=flat&logo=read-the-docs)](https://radarsimx.github.io/radarsimpy/)
[![Download](https://img.shields.io/badge/Download-latest-brightgree?style=flat&logo=python)](https://radarsimx.com/product/radarsimpy/)

**`!!! This module needs to be built/used together with RadarSimC (the C++ engine for radar simulator)`**

# RadarSimPy

<img src="./assets/radarsimpy.svg" alt="logo" width="200"/>

A **Radar** **Sim**ulator for **Py**thon

RadarSimPy is a powerful and versatile Python-based Radar Simulator that models radar transceivers and simulates baseband data from point targets and 3D models. Its signal processing tools offer range/Doppler processing, direction of arrival estimation, and beamforming using various cutting-edge techniques, and you can even characterize radar detection using Swerling’s models. Whether you’re a beginner or an advanced user, RadarSimPy is the perfect tool for anyone looking to develop new radar technologies or expand their knowledge of radar systems.

## Key Features

- ### Radar Modeling

  - Radar transceiver modeling
  - Arbitrary waveform
  - Phase noise
  - Phase/amplitude modulation
  - Fast-time/slow-time modulation

- ### Simulation

  - Simulation of radar baseband data from point targets
  - Simulation of interference
  - Simulation of radar baseband data from 3D modeled objects/environment (**`#raytracing`**)
  - Simulation of target's RCS (**`#raytracing`**)
  - Simulation of LiDAR point cloud from 3D modeled objects/environment (**`#raytracing`**)

- ### Signal Processing

  - Range/Doppler processing
  - Direction of arrival (DoA) estimation
    - **MU**ltiple **SI**gnal **C**lassification (MUSIC) DoA estimations for a uniform linear array (ULA)
    - Root-MUSIC DoA estimation for a ULA
    - **E**stimation of **S**ignal **P**arameters via **R**ational **I**nvariance **T**echniques (ESPRIT) DoA estimation for a ULA
  - Beamformer
    - Capon beamformer
    - Bartlett beamformer
  - Constant false alarm rate (CFAR)
    - 1D/2D cell-averaging CFAR (CA-CFAR)
    - 1D/2D ordered-statistic CFAR (OS-CFAR)

- ### Characterization

  - Radar detection characteristics based on Swerling's models

## Dependence

- numpy
- scipy
- pymeshlab (preferred) or meshio
- [Visual C++ Runtime](https://aka.ms/vs/16/release/vc_redist.x64.exe/) (`Windows`)

## Installation

To use the module, please put the radarsimpy folder within your project folder as shown below.

---

- ### Windows

  - your_project.py
  - your_project.ipynb
  - radarsimpy
    - \_\_init__.py
    - radarsimc.dll
    - scene.xxx.pyd
    - ...

---

- ### Linux

  - your_project.py
  - your_project.ipynb
  - radarsimpy
    - \_\_init__.py
    - libradarsimc.so
    - scene.xxx.so
    - ...

---

## Acceleration

This module supports CPU/GPU parallelization.
CPU parallelization is implemented through OpenMP.
GPU parallelization (CUDA) has been added since v6.0.0.

|         | CPU | GPU (CUDA) |
|---------|-----|------------|
| Windows | ✅  | ✅         |
| Linux   | ✅  | ✅         |
| macOS   | ❌   | ❌          |

![performance](https://github.com/radarsimx/radarsimpy/raw/master/assets/performance.png)

## Coordinate Systems

- ### Scene Coordinate

  - axis (m): `[x, y, z]`
  - phi (deg): angle on the x-y plane. 0 deg is the positive x-axis, 90 deg is the positive y-axis
  - theta (deg): angle on the z-x plane. 0 deg is the positive z-axis, 90 deg is the x-y plane
  - azimuth (deg): azimuth -90 ~ 90 deg equal to phi -90 ~ 90 deg
  - elevation (deg): elevation -90 ~ 90 deg equal to theta 180 ~ 0 deg

- ### Object's Local Coordinate

  - axis (m): `[x, y, z]`
  - yaw (deg): rotation along the z-axis. Positive yaw rotates the object from the positive x-axis to the positive y-axis
  - pitch (deg): rotation along the y-axis. Positive pitch rotates the object from the positive x-axis to the positive z-axis
  - roll (deg): rotation along the x-axis. Positive roll rotates the object from the positive z-axis to the negative y-axis
  - origin (m): `[x, y, z]`
  - rotation (deg): `[yaw, pitch, roll]`
  - rotation (deg/s): rate `[yaw rate, pitch rate, roll rate]`

## Usage Examples

The source files of these Jupyter notebooks are available [here](https://github.com/radarsimx/radarsimnb).

- ### Radar modeling and point target simulation

  - [Doppler radar](https://radarsimx.com/2019/05/16/doppler-radar/)
  - [FMCW radar](https://radarsimx.com/2018/10/11/fmcw-radar/)
  - [TDM MIMO FMCW radar](https://radarsimx.com/2019/04/07/tdm-mimo-fmcw-radar/)
  - [PMCW radar](https://radarsimx.com/2019/05/24/pmcw-radar/)
  - [Arbitrary waveform](https://radarsimx.com/2021/05/10/arbitrary-waveform/)
  - [Phase noise](https://radarsimx.com/2021/01/13/phase-noise/)
  - [CFAR](https://radarsimx.com/2021/01/10/cfar/)
  - [DoA estimation](https://radarsimx.com/2022/12/12/doa-estimation/)
  - [Interference](https://radarsimx.com/2023/01/13/interference/)

- ### Radar modeling and 3D scene simulation with raytracing

  - [Imaging radar](https://radarsimx.com/2022/12/02/imaging-radar/)
  - [FMCW radar with a corner reflector](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-corner-reflector/)
  - [FMCW radar with a plate](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-plate/)
  - [FMCW radar with a car](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-car/)
  - [Doppler of a turbine](https://radarsimx.com/2021/05/10/doppler-of-a-turbine/)
  - [Micro-Doppler](https://radarsimx.com/2021/05/10/micro-doppler/)
  - [Multi-path effect](https://radarsimx.com/2021/05/10/multi-path-effect/)

- ### 3D modeled target's RCS simulation

  - [Corner reflector RCS](https://radarsimx.com/2021/05/10/corner-reflector-rcs/)
  - [Plate RCS](https://radarsimx.com/2021/05/10/plate-rcs/)
  - [Car RCS](https://radarsimx.com/2021/05/10/car-rcs/)

- ### LiDAR point cloud

  - [LIDAR point cloud](https://radarsimx.com/2020/02/05/lidar-point-cloud/)

- ### Receiver characterization

  - [Receiver operating characteristic (ROC)](https://radarsimx.com/2019/10/06/receiver-operating-characteristic/)

## API Reference

- **Radar Model**: Classes to define a radar system

  - [`radarsimpy.Transmitter`](https://radarsimx.github.io/radarsimpy/radar.html#radarsimpy-transmitter): Radar transmitter
  - [`radarsimpy.Receiver`](https://radarsimx.github.io/radarsimpy/radar.html#radarsimpy-receiver): Radar receiver
  - [`radarsimpy.Radar`](https://radarsimx.github.io/radarsimpy/radar.html#radarsimpy-radar): Radar system

- **Simulator**: Radar baseband signal simulator

  - [`radarsimpy.simulator.simpy`](https://radarsimx.github.io/radarsimpy/sim.html#radarsimpy.simulator.simpy): Simulates and generates raw time domain baseband data (Python engine)
  - [`radarsimpy.simulator.simc`](https://radarsimx.github.io/radarsimpy/sim.html#radarsimpy.simulator.simc): Simulates and generates raw time domain baseband data (C++ engine)

- **Raytracing**: Raytracing module for radar scene simulation

  - [`radarsimpy.rt.lidar_scene`](https://radarsimx.github.io/radarsimpy/rt.html#radarsimpy.rt.lidar_scene): Simulates LiDAR's point cloud based on a 3D environment model with ray tracing
  - [`radarsimpy.rt.rcs_sbr`](https://radarsimx.github.io/radarsimpy/rt.html#radarsimpy.rt.rcs_sbr): Simulates target's radar cross section (RCS) based on the 3D model with ray tracing
  - [`radarsimpy.rt.scene`](https://radarsimx.github.io/radarsimpy/rt.html#radarsimpy.rt.scene): Simulates radar's response signal in a 3D environment model with ray tracing

- **Processing**: Basic radar signal processing module

  - [`radarsimpy.processing.range_fft`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.range_fft): Calculate range profile matrix
  - [`radarsimpy.processing.doppler_fft`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.doppler_fft): Calculate range-Doppler matrix
  - [`radarsimpy.processing.range_doppler_fft`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.range_doppler_fft): Range-Doppler processing
  - [`radarsimpy.processing.cfar_ca_1d`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_ca_1d): 1D Cell Averaging CFAR (CA-CFAR)
  - [`radarsimpy.processing.cfar_ca_2d`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_ca_2d): 2D Cell Averaging CFAR (CA-CFAR)
  - [`radarsimpy.processing.cfar_os_1d`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_os_1d): 1D Ordered Statistic CFAR (OS-CFAR)
  - [`radarsimpy.processing.cfar_os_2d`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_os_2d): 2D Ordered Statistic CFAR (OS-CFAR)
  - [`radarsimpy.processing.doa_music`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.doa_music): Estimate DoA using MUSIC for a ULA
  - [`radarsimpy.processing.doa_root_music`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.doa_root_music): Estimate DoA using Root-MUSIC for a ULA
  - [`radarsimpy.processing.doa_esprit`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.doa_esprit): Estimate DoA using ESPRIT for a ULA
  - [`radarsimpy.processing.doa_capon`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.doa_capon): Capon (MVDR) beamforming for a ULA
  - [`radarsimpy.processing.doa_bartlett`](https://radarsimx.github.io/radarsimpy/process.html#radarsimpy.processing.doa_bartlett): Bartlett beamforming for a ULA

- **Tools**: Receiver operating characteristic analysis

  - [`radarsimpy.tools.roc_pd`](https://radarsimx.github.io/radarsimpy/tools.html#radarsimpy.tools.roc_pd): Calculate probability of detection (Pd) in receiver operating characteristic (ROC)
  - [`radarsimpy.tools.roc_snr`](https://radarsimx.github.io/radarsimpy/tools.html#radarsimpy.tools.roc_snr): Calculate the minimal SNR for a certain probability of detection (Pd) and probability of false alarm (Pfa) in receiver operating characteristic (ROC)
