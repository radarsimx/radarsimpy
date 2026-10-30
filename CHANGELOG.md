# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]

### Changed

- Windows and Linux now ship a single CUDA-enabled module instead of separate CPU and GPU (CUDA) modules. It runs on the GPU when a usable one is present and on the CPU with OpenMP otherwise. The CPU-only release workflows (`release_windows.yml`, `release_ubuntu.yml`) are removed, and the Windows and Ubuntu unit tests now build the CUDA module for every entry. MacOS stays CPU-only
- The Python extensions link the static CUDA runtime (`cudart_static`) instead of the shared one, so the CUDA module imports on a machine with no CUDA Toolkit installed
- `gpu_available()` now also returns `False` when the GPU is older than every architecture the module was built for (for example a Pascal or Volta card with the CUDA 13 build). Before, it returned `True` and every kernel launch then failed; now the simulation falls back to the CPU
- CPU runs of the CUDA module are faster: up to 1.4x for `level="sample"` and 1.2x at high `density`, which brings them in line with a CPU-only build. Host code in a CUDA build called CUDA's host-side `sin`/`cos`/`sincos` instead of the standard library's
- Windows CI caches the installed CUDA toolkit (`.github/actions/setup-cuda-windows`), so NVIDIA's installer runs only when the CUDA version or sub-package list changes. It took about 210 s per job with the network installer, and about 600 s with the local one that the CUDA 12 release workflow used. The unused cuRAND runtime library is no longer installed; only its headers are
- The Windows unit tests build and run the C++ (Google Test) suite in one matrix entry instead of all five, since it does not depend on the Python version; the other entries build the module and run pytest. CI installs `requirements-test.txt` (new: build and test dependencies) instead of `requirements-dev.txt`, which now includes it and adds the lint and docs tools. The C++ test binary follows `RADARSIMX_CUDA_ARCHITECTURES` when it is set, instead of always compiling for sm_75 and sm_86
- Linux ARM64 builds in CI now download the prebuilt HDF5 and mbedTLS from the radarsimx-deps release (`--deps=release`), the same way every other platform does. Before, they used the copies in the `deps` submodule, because the release had no ARM64 archives
- The README and the Dependencies, Installation and Build pages now list Linux ARM64 as supported, for both the CPU and GPU (CUDA) modules

## [15.5.0] - 2026-10-13

### Added

- Linux ARM64 (aarch64) packages for Ubuntu 22.04, 24.04 and 26.04, in both CPU and CUDA 13 versions, for Python 3.10-3.14. The Ubuntu unit tests also run on ARM64 (CPU only)
- New User Guide pages: System Model, Transmitter, Receiver and Mutual Interference
- New User Guide sections on pulsed waveforms, and on simultaneous transmit channels (how to combine the separated M x N channels into the signal a real receiver sees)
- New figures for the animated-target, receiver-noise and phase-noise guides
- `RADARSIMX_DEVICE_PARITY` build option. Run `set RADARSIMX_DEVICE_PARITY=ON` before `build.bat` to get a GPU build whose results match the CPU exactly, bit for bit. It runs about 1.7x slower, so it is off by default
- `sim_radar` now raises a `RuntimeError` (`[ERROR_PO_LUT_OVERFLOW]`) when a scene produces more scattering points than it can store, instead of returning an incomplete result. The message explains how to fix it: lower `density`, narrow `ray_filter`, or set `skip_diffusion=True` on large surrounding surfaces
- `radarsimpy.lib.np_float`, the NumPy dtype that matches the floating-point type the package was built with
- `benchmarks/precision_ab.py`, a script that compares the accuracy and speed of two builds
- Back-propagation test suite (`tests/test_system_back_propagation.py`) and multi-target benchmark scenes to go with it
- Regression tests for the pulsed-radar, modulation and pulse-index fixes listed below
- Dependabot now also checks the Python requirements files

### Fixed

- `sim_radar(..., back_propagating=True)` reported multi-bounce returns at too short a range. For example, with two plates 30 m and 20 m from the radar and 20 m apart, the return showed up at 40 m instead of the correct 50 m. Simulations without back propagation are not affected
- Pulsed radar: a target beyond the unambiguous range (`c * prp / 2`) could add a ghost return at the wrong sample, with a random and sometimes huge amplitude. The folded echo now appears only where it should. This affected point targets, mesh targets and interference
- Waveform modulation was applied one sample too early, just before the echo arrived. For example, a 21-sample pulse came out 22 samples long
- `sim_radar` with the default `level=None` could get the first sample of a pulse wrong for some numbers of samples per pulse (41, 47, 55, 61, ...). Common sizes such as 100, 128, 200 and 256 were not affected
- Mesh simulation on the GPU with phase noise enabled failed with a CUDA error. This was a regression in 15.4.0. CPU runs were not affected
- Documentation errors:
  - The User Guide said that transmitting on several channels at once gives a combined signal. It does not: the simulator always returns the channels separated
  - The README described `theta` and `pitch` incorrectly. `theta` is measured from the +z axis, and a positive `pitch` turns +x toward +z
  - `mod_t` must be equally spaced. This was not stated before

### Changed

- Faster simulation through mixed-precision math. Values that need double precision (range, delay, phase) keep it, and everything else is now computed in single precision. On an RTX 3050 the baseband step of a mesh simulation is about **4x** faster. Point-target and interference simulation is **1.1x-2.6x** faster on the GPU and **1.05x-4.4x** faster on the CPU
- Simulation results change very slightly because of the mixed-precision math. The difference is below -113 dBc for mesh scenes and below -145 dBc for point targets and interference
- Interference simulation on the CPU now uses all cores; it was single-threaded before. Point-target simulation on Windows also makes better use of multiple cores
- `sim_radar(..., back_propagating=True)` is redesigned. Direct and back-propagated returns are now computed together in one pass. For users this means:
  - Each back-propagated return is checked against the surfaces it reflects off. A return that misses a surface or is blocked is dropped
  - `ray_filter` counts all the reflections of a back-propagated return. Before, a seven-bounce return was counted as four
  - Back-propagated returns are built only for rays that left the scene, not for rays stopped by the trace-depth limit
  - Results are unchanged when the flag is off, and for scenes where rays do not bounce between surfaces, such as a single convex target
- `ray_filter` counts the reflections of the whole path, including those on the way back with `back_propagating=True`. `ray_filter[1]` still limits how deep rays are traced
- `float_t` in `type_def.pxd` now sets the precision of the whole package: `float` (default) gives the mixed-precision build and `double` gives an all-double build. Building with `double` did not compile before. Note that `double` also changes the precision of the mesh geometry, so mesh results differ from the default build
- Stored test reference values are re-recorded. Tests now allow an error relative to the peak of the expected signal (1e-6 for the ideal and interference tests, 3e-5 for the mesh tests) instead of comparing element by element
- Documentation:
  - The coordinate systems page is rewritten, with new figures, a summary table and worked examples
  - The ray-tracing guide is updated for the new back propagation
  - The README has a new section on choosing the CPU or GPU, a benchmark table across releases, an index of the User Guide, and lists `pygltflib` as an optional dependency
  - All figures are redrawn in one consistent style
- Updated `radarsimcpp` submodule to v15.5.0

---

## [15.4.0] - 2026-09-09

### Added

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

- `radarsimpy.mesh_kit.import_mesh_module` docstring listed the backend search order as pyvista first; the code has always tried trimesh first

### Changed

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
