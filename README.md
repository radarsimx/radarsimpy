<a href="#" target="_blank" rel="nofollow"><img src="https://img.shields.io/badge/Python-3.7%7C3.8%7C3.9-blue?style=flat&logo=python" height="20" ></a>
<a href="https://rookiepeng.github.io/radarsimpy/" target="_blank" rel="nofollow"><img src="https://img.shields.io/badge/Documentation-latest-brightgree?style=flat&logo=read-the-docs" height="20" ></a>

# RadarSimPy
                                                            
<img src="./assets/radarsimpy.svg" alt="logo" width="200"/>

A **Radar** **Sim**ulator for **Py**thon

***This module needs to be built/used together with RadarSimC (the C++ engine for radar simulator)***

There are 5 modules in this package:

1. **Radar**: Classes to define a radar system
   - `radarsimpy.Transmitter`: Radar transmitter
   - `radarsimpy.Receiver`: Radar receiver
   - `radarsimpy.Radar`: Radar system
2. **Simulator**: Radar baseband signal simulator
   - `radarsimpy.simulator`: Simulates and generates raw time domain baseband data (Python engine)
   - `radarsimpy.simulatorcpp`: Simulates and generates raw time domain baseband data (C++ engine)
3. **Raytracing**: Raytracing module for radar scene simulation
   - `radarsimpy.rt.lidar_scene`: Simulates LiDAR's point cloud based on a 3D environment model with ray tracing
   - `radarsimpy.rt.rcs`: Simulates target's radar cross section (RCS) based on the 3D model with ray tracing
   - `radarsimpy.rt.scene`: Simulates radar's response signal in a 3D environment model with ray tracing
4. **Processing**: Basic radar signal processing module
5. **Tools**: Receiver operating characteristic analysis

## Dependence

- numpy
- scipy
- numpy-stl
- [Visual C++ Runtime](https://aka.ms/vs/16/release/vc_redist.x64.exe/) (*Windows only*)

## Installation

[Contact me](https://zpeng.me/#contact) if you are interested in this module.

To use the module, please put the radarsimpy folder within your project folder as shown below.

> Windows
>
> - your_project.py
> - your_project.ipynb
> - radarsimpy
>   - \_\_init__.py
>   - radarsimc.dll
>   - scene.xxx.pyd
>   - ...
>

> Linux
>
> - your_project.py
> - your_project.ipynb
> - radarsimpy
>   - \_\_init__.py
>   - libradarsimc.so
>   - scene.xxx.so
>   - ...
>

## Coordinate Systems

### Scene Coordinate

- axis (m): `[x, y, z]`
- phi (deg): angle on x-y plane. Positive x-axis is 0 deg, positive y-axis is 90 deg
- theta (deg): angle on z-x plane. Positive z-axis is 0 deg, x-y plane is 90 deg
- azimuth (deg): azimuth -90 ~ 90 deg equal to phi -90 ~ 90 deg
- elevation (deg): elevation -90 ~ 90 deg equal to theta 180 ~ 0 deg

### Object's Local Coordinate

- axis (m): `[x, y, z]`
- yaw (deg): rotation along z-axis. Positive yaw rotates object from positive x-axis to positive y-axis
- pitch (deg): rotation along y-axis. Positive pitch rotates object from positive x-axis to positive z-axis
- roll (deg): rotation along x-axis. Positive roll rotates object from positive z-axis to negative y-axis
- origin (m): `[x, y, z]`
- rotation (deg): `[yaw, pitch, roll]`
- rotation (deg/s): rate `[yaw rate, pitch rate, roll rate]`

## Usage

- Radar system simulation
  - [Doppler radar simulation](https://zpeng.me/index.php/2019/05/16/doppler-radar/)
  - [FMCW radar simulation](https://zpeng.me/index.php/2018/10/11/fmcw-radar/)
  - [TDM MIMO FMCW radar simulation](https://zpeng.me/index.php/2019/04/07/tdm-mimo-fmcw-radar/)
  - [PMCW radar simulation](https://zpeng.me/index.php/2019/05/24/pmcw-radar/)

- Target simulation
  - [Target RCS evaluation with ray tracing](https://zpeng.me/index.php/2019/11/13/rcs-calculation-with-ray-tracing/)

- Radar system and scene simulation with ray tracing
  - [FMCW radar ray tracing](https://zpeng.me/index.php/2020/03/20/fmcw-radar-ray-tracing/)
  - [Multi-path effect with ray tracing](https://zpeng.me/index.php/2020/03/20/multi-path-effect-with-ray-tracing/)
  - [Micro-Doppler effect of a rotating object with a Doppler radar](https://zpeng.me/index.php/2020/05/24/micro-doppler-effect-of-a-rotating-object-with-a-doppler-radar/)
  - [ISAR with raytracing](https://zpeng.me/index.php/2020/05/28/isar-with-raytracing/)

- LIDAR (Experimental)
  - [LIDAR point cloud](https://zpeng.me/index.php/2020/02/05/lidar-point-cloud/)

- Characterization
  - [Receiver operating characteristic (ROC)](https://zpeng.me/index.php/2019/10/06/receiver-operating-characteristic/)
