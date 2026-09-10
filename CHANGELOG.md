# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

### Added

- `tests/test_system_back_propagation.py`, which states back propagation and the reflection filter as properties rather than stored numbers: the flag is inert without multipath, what it adds lands at the path length it stands for, a scene of only skipped surfaces is silent, and every reflection band summed reconstructs the unfiltered run. `benchmarks/scenes.py` gained multi-target scenes to go with them, and the reference capture two cases that are sensitive to back propagation by measurement -- the pre-existing `sphere_backprop` moves by exactly zero and is the negative control
- `radarsimpy.animation_kit`, which turns a keyframe-animated glTF 2.0 / GLB model into ordinary target dictionaries. Each animated node becomes its own target whose `location`, `speed`, `rotation` and `rotation_rate` are sampled at `radar.time_prop["timestamp"]`, so a spinning rotor, a turning wheel or a keyframed flight path is simulated from the motion authored in the file instead of being re-derived by hand. Velocity and angular rate come from the analytic derivative of the keyframe interpolation, not from differencing the sample grid, because the timestamp restarts at every channel and frame. `STEP`, `LINEAR` and `CUBICSPLINE` samplers, nested node hierarchies and the glTF Y-up to RadarSimPy Z-up conversion are handled; skinning, morph targets and animated scale raise a descriptive `NotImplementedError`, since the ray tracer transforms each target rigidly. Requires the optional `pygltflib` package, discovered at runtime the same way the mesh backends are
- `radarsimpy.mesh_kit.load_mesh` accepts an in-memory `{"points": ..., "cells": ...}` dictionary in place of a file path, so generated geometry can be simulated without a temporary file. This is how `animation_kit` hands over the parts it extracts from an animated model
- `gltf` pytest marker, which skips the animated-model tests when `pygltflib` is not installed
- `benchmarks/bench_sbr.py`, a timing harness for the mesh (SBR) simulator. Sweeps the parameters that drive ray-tracing cost -- `density`, `level`, transmit/receive channel count, model, pulse count -- writes a self-describing JSON record, and turns two such records into a speedup table with `--compare`. `--omp-scaling` re-runs the sweeps in a subprocess per `OMP_NUM_THREADS` value, which is the only way to vary it (the OpenMP runtime reads it at load time) and the measurement that distinguishes real parallel speedup from lock contention
- `benchmarks/capture_reference.py`, which records the baseband of six scenes chosen to exercise distinct code paths and diffs a fresh capture against a saved one, so a change to the tracer can be reviewed as a numerical delta instead of a wall of failing asserts
- `benchmarks/baseline/`, holding the pre-optimization CPU and GPU sweeps and reference captures
- `RADARSIMX_CUDA_ARCHITECTURES` environment variable, honored by `build.sh` and `build.bat`, restricting a GPU build to the given CUDA architectures instead of all eleven. Unset -- as in CI and release builds -- still produces the complete fat binary
- `gate_delay` parameter in `Receiver` for range-gated deramp (stretch) processing. The receive window opens at this delay after the chirp start and the deramp reference is the transmit chirp delayed by the same amount, so a target at `c * gate_delay / 2` beats at DC. This makes long-range FMCW/stretch radars representable: previously the reference sat at zero delay, so a 111 km target at a 6e12 Hz/s chirp slope produced a 4.45 GHz beat tone that aliased past any practical ADC rate. Defaults to `0`, which reproduces the previous zero-delay behavior bit-for-bit
- `Receiver.gate_delay` and `Receiver.gate_range` properties
- `Radar.chirp_slope`, `Radar.unambiguous_range_span` and `Radar.unambiguous_range_window` properties, describing deramp (stretch) processing of a linear FM waveform. The window is one-sided `[0, span]` when un-gated (all beat tones are positive, so the full `[0, fs)` band is usable) and two-sided `gate_range +/- span/2` when gated (the residual delay is signed). All three report `None` when the waveform is not a linear FM ramp
- Range gate test suite (`test_system_range_gate.py`)
- `--deps` build option (`build.sh` / `build.bat`) selecting where the prebuilt third-party libraries come from: `repo` (default) reads the committed `libs/` tree in the `radarsimx-deps` submodule and needs no network, `release` downloads the checksum-pinned `radarsimx-deps` GitHub release archives into a cache outside the build directory. All GitHub Actions workflows now build with `--deps=release`
- Runtime CPU fallback for machines without a GPU. The execution policies resolve at compile time, so a GPU-enabled build would attempt CUDA execution even where no device exists; `sim_radar`, `sim_rcs` and `sim_lidar` now probe for a usable CUDA device and dispatch to the CPU policy when there is none. `sim_radar(device="gpu")` reports the fallback as a `RuntimeWarning`. A CPU-only build is unaffected, and an explicit `device="cpu"` request is unchanged
- CPU fallback test suite (`test_system_cpu_fallback.py`)
- `radarsimpy.simulator.gpu_available()`, reporting whether a simulation can actually run on the GPU. It answers the conjunction that matters -- the module was built with CUDA support *and* the machine exposes a usable CUDA device -- which is exactly what `sim_radar(device="auto")` branches on, so it also answers what `"auto"` will pick. The device probe is cached for the lifetime of the process and follows `CUDA_VISIBLE_DEVICES`. Its first use is a `skipif` guard on `test_cpu_gpu_parity`, which previously ran on CPU-only builds and compared the CPU result against itself

### Fixed

- `sim_radar(..., back_propagating=True)` no longer reports multi-bounce returns at too short a range. Back propagation follows a ray out, scatters at a hit point and sends the field back down the same chain, so the round trip includes that chain twice; the return leg was being measured from the ray's first hit instead, which dropped the stretch between there and the scattering point. On two plates 30 m and 20 m from the radar and 20 m apart, the back-propagated return appeared at 40 m rather than its true 50 m. Only affects `back_propagating=True`; everything else is bit-identical
- `radarsimpy.mesh_kit.import_mesh_module` docstring listed the backend search order as pyvista first; the code has always tried trimesh first

### Changed

- `sim_radar(..., back_propagating=True)` is redesigned. It used to run a second ray-tracing and baseband pass that rewrote the traced rays in place; a return path is now described by the scattering point it belongs to, so both kinds of return are computed together. Three things change for a user: a return that reflects on its way out is checked against the surfaces it claims to reflect off, instead of being assumed to close; `ray_filter` counts such a path at its true length, where a seven-bounce return was previously filtered as if it were four; and returns are only built for rays that left the scene, not for rays the trace depth stopped. Scenes that leave the flag off are bit-identical, and so is any scene where rays do not bounce -- a single convex target returns exactly what it did before
- `ray_filter` counts reflections end to end, including those a back-propagated return takes on its way out. `ray_filter[1]` continues to cap trace depth
- Substantially faster mesh (SBR) radar simulation, with baseband output unchanged. Measured against the previous release on the benchmark sweep: **GPU 1.1x-8.2x** (RTX 3050, CUDA 13.3) and **CPU 2.9x-6.3x** (16 logical cores), with `sim_radar(..., level="sample")` dropping from 10.7 s to 1.7 s on the CPU path. The CPU path's output is bit-identical to before and reproducible from run to run; the GPU path moves by ~1e-15 relative, which is atomicAdd ordering. See the `radarsimcpp` changelog for the individual changes
- A GPU-enabled build now has OpenMP for its host code paths, so `sim_radar(..., device="cpu")` on such a build is no longer single-threaded. It was 64.9 s on a scene a CPU-only build ran in 3.1 s; it is now 0.64 s. The `device` docstring no longer carries the performance caveat, and the no-CUDA-device `RuntimeWarning` no longer mentions it
- `sim_rcs` is substantially faster on scenes large enough to exceed the ray-pool budget. The per-tile ray count was capped by a constant that did not account for the size of a ray, so a 5x5 m plate at 77 GHz with density 1 asked for a single 4.97 GB tile; on a 4 GB card that oversubscribes and pages over PCIe rather than failing. 2.69 s to 0.23 s on an RTX 3050, 2.16 s to 1.00 s on the host. Host RCS is also 1.2x-1.5x faster from parallelizing its ray pool, and stays bit-reproducible run to run
- Simulation output changes slightly, from five corrections in `radarsimcpp` to geometry and material handling that previously produced silently wrong results: triangles whose centres shared a Morton voxel were deleted from the scene, a one-triangle scene was invisible, the azimuth ray footprint was missing its `sin(theta)` Jacobian, two paths could put a NaN into the baseband, and the perpendicular Fresnel coefficient was discarded for media with `|ep| < |mu|`. Baseband moves by 2e-6 to 2e-3 relative, dominated by the Jacobian: azimuth-only cases are unchanged, and a target at 45 degrees elevation drops 2.6-2.8 dB. Test expectations are updated accordingly
- `Radar.timestamp` now starts at `gate_delay` rather than 0, reflecting when the receive window actually opens. Unchanged for the default un-gated receiver
- Phase-noise range correlation is now governed by the residual delay `tau - gate_delay` instead of the full round-trip delay. At long range with a gate, close-in phase noise correctly cancels; previously it decorrelated according to absolute range and the LUT index wrapped
- `sim_radar`, `sim_rcs` and `sim_lidar` all take `device` and default it to `"auto"`, selecting the GPU when one is usable and the CPU otherwise. `sim_rcs` and `sim_lidar` previously probed for a device with no way for the caller to override it; they now accept `"auto"`, `"gpu"` and `"cpu"` on the same terms as `sim_radar`, from one shared resolver so the three cannot disagree. `"auto"` is silent, because choosing the CPU there is the documented behaviour rather than a request that could not be honoured; `device="gpu"` still warns when it falls back, and that warning is now meaningful because it only fires when a caller asked for the GPU by name. `sim_radar` callers who passed nothing previously got `"gpu"`, so on a CPU-only build every call raised a fallback `RuntimeWarning`
- Updated `radarsimcpp` submodule for range gate support

---

## [15.3.0] - 2026-07-25

### Added

- `get_scene_state` function (`radarsimpy.scene`) to compute target meshes plus transmitter/receiver channel locations and radar boresight direction at one or more query timestamps, using C++-accelerated rotation
- Ubuntu 26.04 (GCC-15) support across CI, release workflows, and build documentation
- CUDA 13.3.1 toolkit support in CI/release workflows

### Changed

- Split the monolithic `cp_radarsimc.pyx` Cython module into `cp_radarsimc_helpers.pyx`, `cp_radarsimc_mesh.pyx`, `cp_radarsimc_points.pyx`, and `cp_radarsimc_radar.pyx` for maintainability
- Improved handling of time-varying location/speed/rotation/rotation_rate parameters: per-axis scalars are now broadcast and combined with time-varying axes instead of requiring all axes to share the same shape
- Centralized CI build/test matrices into a shared setup job and refactored artifact packaging (dynamic zip/tar packing, generated READMEs, build-summary output) across release and unit-test workflows
- macOS CI now builds with Clang on macOS 26 (Xcode 26.4.1) instead of GCC
- Updated documentation: CUDA 13 GPU compute-capability requirement (7.5+/Turing), supported Python range (3.10-3.14), lowered minimum CMake version to 3.18, `--tier` renamed to `--license` in build docs, added `--jobs`, refreshed macOS/Ubuntu platform support notes
- Updated `radarsimcpp` submodule to v15.3.0 (conditional license-flag initialization, MSVC `/NODEFAULTLIB` linker fix, mbedTLS 4.2.0)

---

## [15.2.0] - 2026-04-19

### Added

- Normals, range, and intensity fields to LiDAR simulation output; intensity is computed using a Lambertian model ($\cos(\theta_i) / r^2$)
- SSB (single-sideband) phase-noise parameters (`pn_f`, `pn_power`, `pn_fs`, `pn_seed`, `pn_validation`) in `Transmitter` for C++-side per-frame phase noise generation
- Receiver noise simulator with deterministic noise generation
- Phase noise generator with spectral noise shaping support
- License data now includes organization and license type fields
- Noise simulation test suite (`test_noise_simulation.py`)
- Receiver noise and transmitter phase noise documentation pages
- Warning when GPU execution is selected but CUDA is unavailable at runtime
- CUDA error detection after kernel launches and device initialization
- GPU memory reservation during device initialization

### Changed

- Replaced Python/NumPy receiver noise generation in `sim_radar` with C++ `NoiseSimulator` for performance and accuracy
- Refactored `sim_radar`: centralized parameter validation, preallocated contiguous baseband/noise buffers, simplified CPU/GPU execution paths
- Refactored Cython radar bindings with dedicated helpers for mesh loading, material parsing, and deprecated parameter handling
- Reorganized Cython interface headers for readability; no API changes
- Optimized antenna pattern precomputation and modulation handling to reduce per-sample overhead
- Optimized BVH traversal, waveform phase calculation, and ray initialization for improved performance
- Optimized phase noise LUT indexing and table sizes
- Refactored interference simulator for improved clarity and performance
- mbedTLS now built from source with static linking; removed vcpkg CI build steps

### Removed

- Batch build scripts (`batch_build.bat`, `batch_build.sh`)

### Fixed

- Error code propagation from C++ simulators (`RadarSim::Run`, `NoiseSimulator::Run`, `LidarSimulator::Run`, `InterferenceSimulator::Run`, `RcsSimulator::Run`) to Python
- RCS simulator now raises `RuntimeError` on non-zero error codes
- GPU availability check is now compile-guarded for non-CUDA builds
- Fixed Cython exception handling in mesh loading helper

---

## [15.1.0] - 2026-03-04

### Added

- `density` parameter to target APIs for per-target ray density control (0.0 uses global density)
- `environment` flag to target APIs to mark large surrounding surfaces, using reduced ray density to improve simulation efficiency
- `dry_run` in `sim_radar`, When enabled, the simulation will skip actual ray tracing while still performing setup and validation.
- Runtime validation of target dictionary keys with `UserWarning` for unrecognized keys to catch typos and silently ignored properties
- Ray-tracing simulation documentation page covering `density`, `level`, `ray_filter`, `back_propagating`, `skip_diffusion`, and `environment` parameters

### Changed

- Updated trial/license messaging across the codebase to prompt users to "purchase a license"
- Simplified trial mesh size error message formatting
- Updated CI GitHub Actions versions

### Removed

- Deprecated `frame_time` and `interf_frame_time` parameters from `sim_radar`

### Fixed

- Interference simulation now uses separate baseband buffers (`bb_real_interf`/`bb_imag_interf`) to prevent overwriting primary signal buffers

---

## [15.0.1] - 2026-02-11

### Changed

- Refactored `__init__.py` to improve module initialization structure

### Removed

- `initialize_license` alias and optional gating from license module

---

## [15.0.0] - 2026-02-09

### Added

- License management system with mbedTLS integration
- `set_license()` API for license configuration
- Support for multiple license files
- Automatic license initialization on module import
- Packaging scripts for Windows and Linux/macOS platforms

### Changed

- Consolidated simulator imports from unified module
- Simplified CI packaging by removing build tiers
- Updated license API usage and free-tier checks
- Enhanced build scripts for better library handling
- Updated radarsimcpp submodule with licensing support

### Removed

- Ubuntu CUDA 12 GPU workflow from CI
- Vehicle STL model files from repository
- Build tier system

---

## [14.2.0] - 2026-01-09

### Added

- Python 3.14 support across all build and test matrices
- Doppler sign convention documentation page
- Sample size consistency check between Python and C++ implementations
- Comprehensive documentation expansion for build, dependencies, features, and coordinate systems

### Changed

- Updated minimum Python requirement to 3.10+ (dropped Python 3.9 support)
- Updated CUDA version to 13.1.0 in existing GPU workflows
- Updated macOS x64 CI to use Xcode 16.4 and macos-15-intel
- Expanded and clarified installation guide with platform-specific details
- Improved coordinate system and Doppler sign convention documentation
- Allow `prp` and `f_offset` parameters in Transmitter to accept List types
- Updated HDF5 libs to v2.0.0

### Removed

- Python 3.9 from all CI build and test matrices

### Fixed

- Validation error when Python `samples_per_pulse` mismatches C++ `sample_size_`

---

## [14.1.0] - 2025-11-12

### Added

- Device selection support for `sim_radar` with `device` parameter ("cpu" or "gpu")
- Execution policy support in Cython bindings for CPU/GPU execution
- CPU device tests for mesh-based radar simulation

### Changed

- Reorganized README usage examples for better clarity
- Improved docstrings for `sim_radar` and `sim_rcs` parameters
- Consolidated artifact packing and build summary jobs

### Removed

- Deprecated `frame_time` and `interf_frame_time` parameters from `sim_radar`
- Redundant build summary steps from release workflows
- Unused `vector` import from `simulator_radar.pyx`

### Fixed

- Motion plan in `Radar` not being properly loaded
- Error handling for `PointSimulator::Run` in `sim_radar` (CPU/GPU)
- Error code enum import path to use `core/enums.hpp`
- `_FREETIER_` macro type in setup.py

---

## [14.0.0] - 2025-09-29

### Added

- Smart pointer-based memory management across all radar components
- Automatic resource handling with RAII patterns
- Modern C++ architecture for safer GPU memory usage

### Changed

- Upgraded to `std::shared_ptr` for transmitter, receiver, and radar objects
- Improved API design for radar configuration and channel setup
- Enhanced performance and modularity in simulation components
- Optimized simulation loops for better efficiency
- Improved internal testing structure and CI configurations

### Deprecated

- `frame_time` parameter in `sim_radar()` (use new timestamp logic via `Radar` constructor)

### Removed

- Manual memory cleanup routines (replaced by automatic RAII)
- Redundant code in simulation engine

### Fixed

- Error code handling in Python bindings
- Memory management issues in GPU operations
- Resource leaks in radar component lifecycle
