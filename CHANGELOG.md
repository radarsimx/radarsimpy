# Changelog

All notable changes to this project will be documented in this file.

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
