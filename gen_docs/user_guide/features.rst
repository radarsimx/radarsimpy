Key Features
============

RadarSimPy provides comprehensive tools for radar system modeling, simulation, and signal processing. This page outlines the core capabilities available in the library.

Radar System Modeling
---------------------

**Transceiver Configuration**

RadarSimPy supports flexible radar transceiver modeling with:

* **Arbitrary Waveforms** - Full support for various radar waveforms:
  
  * Continuous Wave (CW)
  * Frequency Modulated Continuous Wave (FMCW)
  * Phase Modulated Continuous Wave (PMCW)
  * Pulsed radar waveforms
  * Custom user-defined waveforms

* **Phase Noise Modeling** - Simulate realistic oscillator phase noise effects on radar performance

* **Modulation Schemes** - Multiple modulation techniques for MIMO and multi-channel systems:
  
  * Code Division Multiplexing (CDM)
  * Frequency Division Multiplexing (FDM)
  * Doppler Division Multiplexing (DDM)
  * Time Division Multiplexing (TDM)
  * Hybrid modulation schemes

* **Signal Modulation** - Advanced modulation capabilities:
  
  * Fast-time modulation (pulse-to-pulse variation)
  * Slow-time modulation (across radar frames)
  * Amplitude and phase control

Radar Simulation Capabilities
------------------------------

**Target Simulation**

* **Point Target Simulation** - Generate radar baseband data from point scatterers with configurable:
  
  * Position, velocity, and acceleration
  * Radar cross-section (RCS)
  * Multi-path and reflection effects

* **3D Object Simulation** - High-fidelity simulation from 3D mesh models:
  
  * Support for common 3D formats (STL, OBJ, PLY, etc.)
  * Ray-tracing based electromagnetic scattering
  * Dynamic object motion and articulation
  * Complex scene environments with multiple objects

* **RCS Calculation** - Compute monostatic and bistatic radar cross-sections for 3D models across:
  
  * Multiple frequencies
  * Various aspect angles
  * Polarization configurations

**Interference Simulation**

* Model radar-to-radar interference scenarios
* Evaluate mutual interference effects in dense radar environments
* Support for both intra-vehicle and inter-vehicle interference

**LiDAR Simulation**

* Generate realistic LiDAR point clouds from 3D environments
* Configurable sensor parameters (resolution, field of view)

Signal Processing Toolkit
--------------------------

**Range-Doppler Processing**

RadarSimPy includes optimized algorithms for standard radar signal processing:

* **FFT-based Range Processing** - Fast Fourier Transform for range compression
* **Doppler Processing** - Coherent integration and velocity estimation
* **2D Range-Doppler Maps** - Generate and visualize range-Doppler spectra

**Direction of Arrival (DoA) Estimation**

Advanced DoA estimation for uniform linear arrays (ULA):

* **MUSIC Algorithm** - MUltiple SIgnal Classification for super-resolution angle estimation
* **Root-MUSIC** - Polynomial-rooting variant for improved computational efficiency
* **ESPRIT Algorithm** - Estimation of Signal Parameters via Rotational Invariance Techniques
* **Iterative Adaptive Approach (IAA)** - High-resolution amplitude and phase estimation with excellent sidelobe suppression

**Beamforming Techniques**

* **Capon Beamformer** - Minimum variance distortionless response (MVDR) for optimal interference rejection
* **Bartlett Beamformer** - Conventional delay-and-sum beamforming

**CFAR Detection**

Constant False Alarm Rate (CFAR) detectors for automatic target detection:

* **CA-CFAR** - Cell-Averaging CFAR for homogeneous clutter
  
  * 1D implementation for range or Doppler
  * 2D implementation for range-Doppler maps

* **OS-CFAR** - Ordered-Statistic CFAR for heterogeneous environments
  
  * 1D implementation for range or Doppler
  * 2D implementation for range-Doppler maps
  * Improved performance in multi-target scenarios and clutter edges

Radar Performance Characterization
-----------------------------------

**Detection Analysis**

* **Swerling Target Models** - Evaluate radar detection performance using statistical target models:
  
  * Swerling Case I - Constant RCS (one scan)
  * Swerling Case II - Variable RCS (pulse-to-pulse)
  * Swerling Case III - Dominant constant scatterer
  * Swerling Case IV - Dominant variable scatterer
  * Swerling Case V - Non-fluctuating targets

* **Probability of Detection (Pd)** - Calculate detection probabilities for given:
  
  * Signal-to-noise ratio (SNR)
  * Probability of false alarm (Pfa)
  * Number of pulses integrated
  * Target fluctuation model

Performance Considerations
--------------------------

RadarSimPy leverages optimized C++ implementations for computationally intensive operations, providing:

* **High-speed simulations** suitable for Monte Carlo analysis
* **Multi-threaded processing** for parallel computation
* **Efficient memory management** for large-scale scenarios
* **GPU acceleration** support (where applicable)

.. note::
   For detailed API documentation and usage examples, refer to the :doc:`../api/index` and :doc:`examples` sections.
