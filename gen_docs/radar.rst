Radar Model
===========

Radar Configuration and Phase Noise Modeling in Python

This module contains classes and functions to define and simulate
the parameters and behavior of radar systems. It includes tools for
configuring radar system properties, modeling oscillator phase noise,
and simulating radar motion and noise characteristics. A major focus
of the module is on accurately modeling radar signal properties,
including phase noise and noise amplitudes.

radarsimpy.Transmitter
-----------------------

.. autoclass:: radarsimpy.Transmitter
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: validate_rf_prop, validate_waveform_prop, process_waveform_modulation, process_pulse_modulation, process_txchannel_prop

radarsimpy.Receiver
--------------------

.. autoclass:: radarsimpy.Receiver
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: validate_bb_prop, process_rxchannel_prop

radarsimpy.Radar
-----------------

.. autoclass:: radarsimpy.Radar
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:
    :exclude-members: cal_noise, gen_timestamp, validate_radar_motion, process_radar_motion
