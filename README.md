<p align="center">
  <img src="https://raw.githubusercontent.com/radarsimx/.github/refs/heads/main/profile/radarsimpy.svg" alt="RadarSimPy logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_windows.yml"><img src="https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_windows.yml/badge.svg" alt="Windows Unit Tests"/></a>
  <a href="https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_ubuntu.yml"><img src="https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_ubuntu.yml/badge.svg" alt="Ubuntu Unit Tests"/></a>
  <a href="https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_macos.yml"><img src="https://github.com/radarsimx/radarsimpy/actions/workflows/unit_test_macos.yml/badge.svg" alt="MacOS Unit Tests"/></a>
  <a href="https://radarsimx.github.io/radarsimpy/"><img src="https://img.shields.io/github/v/tag/radarsimx/radarsimpy?label=Documentation&logo=read-the-docs" alt="Documentation"/></a>
  <a href="https://radarsimx.com/product/radarsimpy/"><img src="https://img.shields.io/github/v/tag/radarsimx/radarsimpy?label=Download&logo=python" alt="Download"/></a>
</p>

# RadarSimPy

A **Radar** **Sim**ulator for **Py**thon

RadarSimPy is a powerful and versatile Python-based Radar Simulator that models radar transceivers and simulates baseband data from point targets and 3D models. Its signal processing tools offer range/Doppler processing, direction of arrival estimation, and beamforming using various cutting-edge techniques, and you can even characterize radar detection using Swerling’s models. Whether you’re a beginner or an advanced user, RadarSimPy is the perfect tool for anyone looking to develop new radar technologies or expand their knowledge of radar systems.

---

## Key Features

- :satellite: **Radar Modeling**
  - Transceiver modeling
  - Arbitrary waveform (CW, FMCW, PMCW, Pulse, ...)
  - Phase noise
  - Phase/amplitude modulation (CDM, FDM, DDM, TDM, ...)
  - Fast-time/slow-time modulation
- :video_game: **Simulation**
  - Baseband data from point targets & 3D models
  - Interference simulation
  - Target RCS simulation
  - LiDAR point cloud simulation
- :signal_strength: **Signal Processing**
  - Range/Doppler processing
  - DoA estimation (MUSIC, Root-MUSIC, ESPRIT, IAA)
  - Beamforming (Capon, Bartlett)
  - CFAR (CA-CFAR, OS-CFAR)
- :chart_with_upwards_trend: **Characterization**
  - Radar detection characteristics (Swerling’s models)

---

## Dependencies

- Python >= 3.10
- NumPy >= 2.0
- SciPy
- One of PyMeshLab, PyVista, trimesh or meshio, for 3D model support (`trimesh` is installed by default)
- pygltflib, only for keyframed glTF 2.0 / GLB targets (`radarsimpy.animation_kit`)

```bash
pip install -r requirements.txt
```

Full detail, including platform and hardware requirements, is on the [Dependencies page](https://radarsimx.github.io/radarsimpy/user_guide/dependencies.html).

**Platform-specific requirements:**

- **Windows**
  - [Visual C++ Runtime](https://aka.ms/vs/16/release/vc_redist.x64.exe/)
  - GPU version (CUDA13) – requires a GPU with Compute Capability 7.5 (Turing) or higher; see [Minimum Required Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/#id1)
- **Ubuntu 22.04**
  - GCC 11 (default)
  - GPU version (CUDA13) – requires a GPU with Compute Capability 7.5 (Turing) or higher; see [Minimum Required Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/#id1)
- **Ubuntu 24.04**
  - GCC 13 (default)
  - GPU version (CUDA13) – requires a GPU with Compute Capability 7.5 (Turing) or higher; see [Minimum Required Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/#id1)
- **Ubuntu 26.04**
  - GCC 15 (default)
  - GPU version (CUDA13) – requires a GPU with Compute Capability 7.5 (Turing) or higher; see [Minimum Required Driver Versions](https://docs.nvidia.com/deploy/cuda-compatibility/#id1)
- **Generic Linux x86-64**
  - Try Ubuntu 22.04/24.04 module, or [request a custom build](https://radarsimx.com/request-a-custom-build/)
- **MacOS**
  - Intel: use default Clang (no extra dependency)
  - Apple Silicon: use default Clang (no extra dependency)

---

## Installation

Download the [pre-built module](https://radarsimx.com/product/radarsimpy/) and place the `radarsimpy` folder in your project directory:

```text
your_project.py
your_project.ipynb
radarsimpy/
  ├── __init__.py
  ├── [platform-specific binaries]
  ├── radar.py
  ├── processing.py
  └── ...
```

**Platform-specific binaries:**

- **Windows:** `radarsimcpp.dll`, `simulator.xxx.pyd`
- **Linux:** `libradarsimcpp.so`, `simulator.xxx.so`
- **MacOS:** `libradarsimcpp.dylib`, `simulator.xxx.so`

---

## Acceleration

This module supports CPU/GPU parallelization:

- **CPU:** via OpenMP
- **GPU:** via CUDA (since v6.0.0)

|         | CPU (x86-64) | CPU (ARM64) | GPU (CUDA) |
| ------- | ------------ | ----------- | ---------- |
| Windows | ✔️           | ❌️         | ✔️         |
| Linux   | ✔️           | ❌️         | ✔️         |
| MacOS   | ✔️           | ✔️          | ❌️        |

<img src="https://github.com/radarsimx/radarsimpy/raw/master/assets/performance.svg" alt="performance" width="500"/>

---

## Coordinate Systems

Everything lives in a single right-handed, z-up frame. Angles are in degrees, distances in metres.

### Global frame

- **axis** (m): `[x, y, z]` — x forward, y to the left, z up
- **phi** (deg): azimuthal angle in the x-y plane. 0° at +x, 90° at +y
- **theta** (deg): polar angle from +z. 0° at zenith, 90° at the horizon, 180° at nadir

<img src="https://github.com/radarsimx/radarsimpy/raw/master/assets/phi_theta.svg" alt="phi and theta" width="700"/>

### Radar-centric angles

- **azimuth** (deg): the same angle as phi. 0° at boresight (+x), positive toward +y
- **elevation** (deg): angle above the x-y plane, `elevation = 90° - theta`

<img src="https://github.com/radarsimx/radarsimpy/raw/master/assets/azimuth_elevation.svg" alt="azimuth and elevation" width="700"/>

### Orientation

Objects are oriented with `[yaw, pitch, roll]`, applied in that order:

- **yaw** (deg): about +z. Turns +x toward +y
- **pitch** (deg): about **-y**. Turns +x toward +z
- **roll** (deg): about +x. Turns +y toward +z
- **origin** (m): `[x, y, z]`, the centre that rotation and translation act about. A radar's origin is always `[0, 0, 0]`

<img src="https://github.com/radarsimx/radarsimpy/raw/master/assets/yaw_pitch_roll.svg" alt="yaw, pitch and roll" width="700"/>

> **Note** — pitch is not a right-handed rotation about +y; a right-handed one would turn +x toward *-z*. The composed rotation is `Rz(yaw) · Ry(-pitch) · Rx(roll)`, the aerospace "nose up is positive" convention. Expect a sign flip when importing orientations from a toolchain that uses the strict right-handed sense.

Conversions, the rotation order in full, and the boresight shortcut `rotation = [azimuth, elevation, roll]` are covered in the [Coordinate Systems guide](https://radarsimx.github.io/radarsimpy/user_guide/coordinate_systems.html).

---

## Usage Examples

Find more usage examples at [radarsimx.com](https://radarsimx.com/category/examples/). Source files are available in the [radarsimnb repository](https://github.com/radarsimx/radarsimnb).

- ### **Radar Systems & Waveforms**

  - [FMCW radar](https://radarsimx.com/2018/10/11/fmcw-radar/)
  - [Pulsed Radar](https://radarsimx.com/2024/09/13/pulsed-radar/)
  - [Doppler radar](https://radarsimx.com/2019/05/16/doppler-radar/)
  - [PMCW radar](https://radarsimx.com/2019/05/24/pmcw-radar/)
  - [Interferometric Radar](https://radarsimx.com/2023/08/31/interferometric-radar/)
  - [Arbitrary waveform](https://radarsimx.com/2021/05/10/arbitrary-waveform/)

- ### **MIMO & Multi-Channel Systems**

  - [TDM MIMO FMCW radar](https://radarsimx.com/2019/04/07/tdm-mimo-fmcw-radar/)
  - [Imaging radar](https://radarsimx.com/2022/12/02/imaging-radar/)
  - [DoA estimation](https://radarsimx.com/2022/12/12/doa-estimation/)

- ### **3D Scene Simulation & Ray Tracing**

  - [FMCW radar with a car](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-car/)
  - [FMCW radar with a plate](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-plate/)
  - [FMCW radar with a corner reflector](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-corner-reflector/)
  - [Multi-path effect](https://radarsimx.com/2021/05/10/multi-path-effect/)
  - [Micro-Doppler](https://radarsimx.com/2021/05/10/micro-doppler/)
  - [Doppler of a turbine](https://radarsimx.com/2021/05/10/doppler-of-a-turbine/)

- ### **Radar Cross Section (RCS) Analysis**

  - [Cross-Polarization and Co-Polarization RCS](https://radarsimx.com/2024/04/19/cross-polarization-and-co-polarization-rcs/)
  - [Car RCS](https://radarsimx.com/2021/05/10/car-rcs/)
  - [Plate RCS](https://radarsimx.com/2021/05/10/plate-rcs/)
  - [Corner reflector RCS](https://radarsimx.com/2021/05/10/corner-reflector-rcs/)

- ### **Signal Processing & Detection**

  - [CFAR](https://radarsimx.com/2021/01/10/cfar/)
  - [CFAR with corner reflector](https://radarsimx.com/2021/05/10/fmcw-radar-with-a-corner-reflector/)

- ### **System Performance & Characterization**

  - [FMCW Radar Link Budget - Ideal Point Target](https://radarsimx.com/2024/10/11/fmcw-radar-link-budget-ideal-point-target/)
  - [Phase noise](https://radarsimx.com/2021/01/13/phase-noise/)
  - [Receiver operating characteristic (ROC)](https://radarsimx.com/2019/10/06/receiver-operating-characteristic/)
  - [Interference](https://radarsimx.com/2023/01/13/interference/)

- ### **LiDAR Simulation**

  - [LIDAR point cloud](https://radarsimx.com/2020/02/05/lidar-point-cloud/)

---

## Build

Check [Build Instructions](./build_instructions.md)

---

## Documentation

Full documentation lives at [radarsimx.github.io/radarsimpy](https://radarsimx.github.io/radarsimpy/).

**Getting started**

- [Overview](https://radarsimx.github.io/radarsimpy/user_guide/overview.html) — what RadarSimPy can model, simulate and process
- [Dependencies](https://radarsimx.github.io/radarsimpy/user_guide/dependencies.html) and [Installation](https://radarsimx.github.io/radarsimpy/user_guide/installation.html)

**Concepts and conventions**

- [System model](https://radarsimx.github.io/radarsimpy/user_guide/system_model.html) — how `Transmitter`, `Receiver` and `Radar` fit together, the virtual array, and the shape of the simulated output
- [Coordinate systems](https://radarsimx.github.io/radarsimpy/user_guide/coordinate_systems.html) — frames, angles and orientation
- [Doppler convention](https://radarsimx.github.io/radarsimpy/user_guide/doppler_convention.html) — the sign of the Doppler frequency

**Configuring a simulation**

- [Transmitter and waveform](https://radarsimx.github.io/radarsimpy/user_guide/transmitter.html) — waveform, pulse train, modulation and the transmit array
- [Receiver and baseband](https://radarsimx.github.io/radarsimpy/user_guide/receiver.html) — sampling, baseband type, the noise budget and the range gate
- [Noise](https://radarsimx.github.io/radarsimpy/user_guide/noise.html) — receiver thermal noise and transmitter phase noise
- [Interference](https://radarsimx.github.io/radarsimpy/user_guide/interference.html) — mutual interference from another radar sharing the band
- [Ray-tracing simulation](https://radarsimx.github.io/radarsimpy/user_guide/ray_tracing_simulation.html) — ray density, fidelity level and target flags for 3D meshes
- [Animated targets](https://radarsimx.github.io/radarsimpy/user_guide/animated_targets.html) — driving targets from keyframed glTF motion
- [Long-range stretch processing](https://radarsimx.github.io/radarsimpy/user_guide/stretch_processing.html) — range gating for long-range FMCW

**[API reference](https://radarsimx.github.io/radarsimpy/api/index.html)** — complete class and function documentation

---

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request on GitHub.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
