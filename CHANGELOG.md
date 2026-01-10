# Changelog

All notable changes to this project will be documented in this file.

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
