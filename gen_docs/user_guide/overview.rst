Overview
========

RadarSimPy is a radar simulator for Python. You describe a radar and a scene,
and it returns the baseband signal the radar would sample. It also includes
the signal-processing and detection tools you need to analyse that signal.

This page lists what the library can do and where to read more about each
part.


Describing the Radar
--------------------

A radar is built from a :class:`radarsimpy.Transmitter` and a
:class:`radarsimpy.Receiver`, combined into a :class:`radarsimpy.Radar`. See
:doc:`system_model` for how the three fit together.

* **Waveforms.** Continuous wave (CW), frequency-modulated continuous wave
  (FMCW), phase-modulated continuous wave (PMCW), pulsed waveforms, and any
  custom frequency-versus-time sweep. See :doc:`transmitter`.
* **Modulation.** Amplitude and phase can change from pulse to pulse (slow
  time) and within each pulse (fast time).
* **MIMO.** Multiple transmit and receive channels. You can separate the
  transmit channels by time (TDM), code (CDM), frequency (FDM), Doppler (DDM),
  or any mix of these.
* **Receiver chain.** Sampling rate, gains, noise figure and range gating. See
  :doc:`receiver`.
* **Noise.** Receiver thermal noise and oscillator phase noise. See
  :doc:`noise`.


Simulating a Scene
------------------

* **Point targets.** Ideal scatterers with a position, velocity, acceleration
  and radar cross-section (RCS). This is the fastest kind of simulation.
* **3D mesh targets.** Models loaded from common mesh formats (STL, OBJ, PLY,
  and others). The simulator uses ray tracing and Physical Optics (PO) to
  compute the scattered signal, including multipath between objects. See
  :doc:`ray_tracing_simulation`.
* **Moving and animated targets.** Targets can move, rotate, and follow
  keyframed glTF animations. See :doc:`animated_targets`.
* **RCS.** Monostatic and bistatic RCS of a 3D model, for any frequency,
  aspect angle and polarization.
* **Interference.** The signal one radar receives from another radar. See
  :doc:`interference`.
* **LiDAR.** Point clouds from a 3D scene.


Processing the Signal
---------------------

The :mod:`radarsimpy.processing` module covers the standard processing chain:

* **Range and Doppler FFTs**, and the 2D range-Doppler map.
* **Direction of arrival (DoA)** for uniform linear arrays: MUSIC, Root-MUSIC,
  ESPRIT and the iterative adaptive approach (IAA).
* **Beamforming:** Bartlett (delay-and-sum) and Capon (minimum variance
  distortionless response, MVDR).
* **CFAR detection:** cell-averaging (CA-CFAR) and ordered-statistic
  (OS-CFAR), each in 1D and 2D.


Detection Performance
---------------------

The :mod:`radarsimpy.tools` module computes the probability of detection for a
given signal-to-noise ratio (SNR), probability of false alarm and number of
integrated pulses. It supports the Swerling 0 (non-fluctuating) and Swerling
I–IV target models, and can generate receiver operating characteristic (ROC)
curves.


Performance
-----------

The simulation engine is written in C++ and runs in parallel on all CPU cores.
On Windows and Linux it also runs on an NVIDIA GPU when one is available. See
:doc:`dependencies` for the hardware requirements.


See Also
--------

* :doc:`dependencies` - What you need installed to use these features
* :doc:`installation` - Installing the pre-built module
* :doc:`examples` - Worked examples for the capabilities listed above
* :doc:`../api/index` - Complete API reference
