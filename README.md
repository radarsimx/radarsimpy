[![Build Status](https://dev.azure.com/rookiepeng/radarsimc/_apis/build/status/rookiepeng.radarsimc?branchName=master)](https://dev.azure.com/rookiepeng/radarsimc/_build/latest?definitionId=3&branchName=master)
<a href="#" target="_blank" rel="nofollow"><img src="https://img.shields.io/badge/Python-3.7%7C3.8%7C3.9%7C3.10-blue?style=flat&logo=python" height="20" ></a>
<a href="https://rookiepeng.github.io/radarsimpy/" target="_blank" rel="nofollow"><img src="https://img.shields.io/badge/Documentation-latest-brightgree?style=flat&logo=read-the-docs" height="20" ></a>
[![DOI](https://zenodo.org/badge/282958664.svg)](https://zenodo.org/badge/latestdoi/282958664)

**`!!! This module needs to be built/used together with RadarSimC (the C++ engine for radar simulator)`**

**Please fill out this [form](https://zpeng.me/#contact) to request the module if you are interested in using RadarSimPy**

# RadarSimPy

<img src="./assets/radarsimpy.svg" alt="logo" width="200"/>

A **Radar** **Sim**ulator for **Py**thon

## Key Features

- ### Radar Modeling

  - Radar transceiver modeling
  - Arbitrary waveform
  - Phase noise
  - Phase/Amplitude modulation
  - Fast-time/Slow-time modulation

- ### Simulation

  - Simulation of radar baseband data from point targets
  - Simulation of radar baseband data from 3D modeled objects/environment (**`#raytracing`**)
  - Simulation of target's RCS (**`#raytracing`**)
  - Simulation of LiDAR point cloud from 3D modeled objects/environment (**`#raytracing`**)

- ### Signal Processing

  - Range/Doppler processing
  - MUltiple SIgnal Classication (MUSIC) DoA estimation for a uniform linear array (ULA)
  - Root-MUSIC DoA estimation for a ULA
  - Estimation of Signal Parameters via Rational Invariance Techniques (ESPRIT) DoA estimation for a ULA
  - Capon beamformer
  - Bartlett beamformer
  - 1D/2D cell-averaging CFAR (CA-CFAR)
  - 1D/2D ordered-statistic CFAR (OS-CFAR)

- ### Characterization

  - Radar detection characteristics based on Swerling's models

## Dependence

- numpy
- scipy
- meshio
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

![performance](https://github.com/rookiepeng/radarsimpy/raw/master/assets/performance.png)

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

The source files of these Jupyter notebooks are available [here](https://github.com/rookiepeng/radar-notebooks).

- ### Radar modeling and point target simulation

  - [Doppler radar](https://zpeng.me/2019/05/16/doppler-radar/)
  - [FMCW radar](https://zpeng.me/2018/10/11/fmcw-radar/)
  - [TDM MIMO FMCW radar](https://zpeng.me/2019/04/07/tdm-mimo-fmcw-radar/)
  - [PMCW radar](https://zpeng.me/2019/05/24/pmcw-radar/)
  - [Arbitrary waveform](https://zpeng.me/2021/05/10/arbitrary-waveform/)
  - [Phase noise](https://zpeng.me/2021/01/13/phase-noise/)
  - [CFAR](https://zpeng.me/2021/01/10/cfar/)
  - [DoA estimation](https://zpeng.me/2022/12/12/doa-estimation/)

- ### Radar modeling and 3D scene simulation with raytracing

  - [Imaging radar](https://zpeng.me/2022/12/02/imaging-radar/)
  - [FMCW radar with a corner reflector](https://zpeng.me/2021/05/10/fmcw-radar-with-a-corner-reflector/)
  - [FMCW radar with a plate](https://zpeng.me/2021/05/10/fmcw-radar-with-a-plate/)
  - [FMCW radar with a car](https://zpeng.me/2021/05/10/fmcw-radar-with-a-car/)
  - [Doppler of a turbine](https://zpeng.me/2021/05/10/doppler-of-a-turbine/)
  - [Micro-Doppler](https://zpeng.me/2021/05/10/micro-doppler/)
  - [Multi-path effect](https://zpeng.me/2021/05/10/multi-path-effect/)

- ### 3D modeled target's RCS simulation

  - [Corner reflector RCS](https://zpeng.me/2021/05/10/corner-reflector-rcs/)
  - [Plate RCS](https://zpeng.me/2021/05/10/plate-rcs/)
  - [Car RCS](https://zpeng.me/2021/05/10/car-rcs/)

- ### LiDAR point cloud

  - [LIDAR point cloud](https://zpeng.me/2020/02/05/lidar-point-cloud/)

- ### Characterization

  - [Receiver operating characteristic (ROC)](https://zpeng.me/2019/10/06/receiver-operating-characteristic/)

## API Reference

- **Radar Model**: Classes to define a radar system

  - [`radarsimpy.Transmitter`](https://rookiepeng.github.io/radarsimpy/radar.html#radarsimpy-transmitter): Radar transmitter
  - [`radarsimpy.Receiver`](https://rookiepeng.github.io/radarsimpy/radar.html#radarsimpy-receiver): Radar receiver
  - [`radarsimpy.Radar`](https://rookiepeng.github.io/radarsimpy/radar.html#radarsimpy-radar): Radar system

- **Simulator**: Radar baseband signal simulator

  - [`radarsimpy.simulator.simpy`](https://rookiepeng.github.io/radarsimpy/sim.html#radarsimpy.simulator.simpy): Simulates and generates raw time domain baseband data (Python engine)
  - [`radarsimpy.simulator.simc`](https://rookiepeng.github.io/radarsimpy/sim.html#radarsimpy.simulator.simc): Simulates and generates raw time domain baseband data (C++ engine)

- **Raytracing**: Raytracing module for radar scene simulation

  - [`radarsimpy.rt.lidar_scene`](https://rookiepeng.github.io/radarsimpy/rt.html#radarsimpy.rt.lidar_scene): Simulates LiDAR's point cloud based on a 3D environment model with ray tracing
  - [`radarsimpy.rt.rcs_sbr`](https://rookiepeng.github.io/radarsimpy/rt.html#radarsimpy.rt.rcs_sbr): Simulates target's radar cross section (RCS) based on the 3D model with ray tracing
  - [`radarsimpy.rt.scene`](https://rookiepeng.github.io/radarsimpy/rt.html#radarsimpy.rt.scene): Simulates radar's response signal in a 3D environment model with ray tracing

- **Processing**: Basic radar signal processing module

  - [`radarsimpy.processing.range_fft`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.range_fft): Calculate range profile matrix
  - [`radarsimpy.processing.doppler_fft`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.doppler_fft): Calculate range-Doppler matrix
  - [`radarsimpy.processing.range_doppler_fft`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.range_doppler_fft): Range-Doppler processing
  - [`radarsimpy.processing.cfar_ca_1d`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_ca_1d): 1D Cell Averaging CFAR (CA-CFAR)
  - [`radarsimpy.processing.cfar_ca_2d`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_ca_2d): 2D Cell Averaging CFAR (CA-CFAR)
  - [`radarsimpy.processing.cfar_os_1d`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_os_1d): 1D Ordered Statistic CFAR (OS-CFAR)
  - [`radarsimpy.processing.cfar_os_2d`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.cfar_os_2d): 2D Ordered Statistic CFAR (OS-CFAR)
  - [`radarsimpy.processing.doa_music`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.doa_music): Estimate DoA using MUSIC for a ULA
  - [`radarsimpy.processing.doa_root_music`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.doa_root_music): Estimate DoA using Root-MUSIC for a ULA
  - [`radarsimpy.processing.doa_esprit`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.doa_esprit): Estimate DoA using ESPRIT for a ULA
  - [`radarsimpy.processing.doa_capon`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.doa_capon): Capon (MVDR) beamforming for a ULA
  - [`radarsimpy.processing.doa_bartlett`](https://rookiepeng.github.io/radarsimpy/process.html#radarsimpy.processing.doa_bartlett): Bartlett beamforming for a ULA

- **Tools**: Receiver operating characteristic analysis

  - [`radarsimpy.tools.roc_pd`](https://rookiepeng.github.io/radarsimpy/tools.html#radarsimpy.tools.roc_pd): Calculate probability of detection (Pd) in receiver operating characteristic (ROC)
  - [`radarsimpy.tools.roc_snr`](https://rookiepeng.github.io/radarsimpy/tools.html#radarsimpy.tools.roc_snr): Calculate the minimal SNR for a certain probability of detection (Pd) and probability of false alarm (Pfa) in receiver operating characteristic (ROC)
