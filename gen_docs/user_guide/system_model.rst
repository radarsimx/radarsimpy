Radar System Model
==================

Every simulation in RadarSimPy is built from the same three objects: a
:class:`radarsimpy.Transmitter`, a :class:`radarsimpy.Receiver`, and a
:class:`radarsimpy.Radar` that binds them to a platform. This page describes
what each object owns, what the ``Radar`` derives from the pair, and how that
determines the shape of the array you get back.


The Transceiver
---------------

RadarSimPy models a coherent transceiver in which the receiver deramps against
the same oscillator that drives the transmitter:

.. figure:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/architecture.svg
    :width: 100%
    :alt: Coherent radar transceiver, transmitter above and receiver below

    The transmit chain generates the waveform, modulates it, amplifies it and
    radiates it. The receive chain amplifies the echo and mixes it against a
    tap of the same source, so the baseband output carries the *difference*
    between the transmitted and the received signal. The mixer feeds an I and a
    Q branch — that pair is what ``bb_type="complex"`` returns, and a real
    baseband keeps only one of them.

Because the local oscillator is shared, the baseband signal is fully coherent
with the transmit waveform. That single fact is what makes range appear as a
beat frequency in an FMCW system, and what makes phase noise partially cancel
for close-in targets — see :doc:`noise`.

Two consequences are worth stating up front:

* **There is no separate IF stage to configure.** The mixer output *is* the
  simulated baseband. Everything from the antenna to the ADC is described by
  :class:`radarsimpy.Receiver`.
* **The deramp reference is a delayed copy of the transmit waveform.** By
  default the delay is zero; ``gate_delay`` moves it, which is what makes
  long-range stretch processing possible (:doc:`stretch_processing`).


From Objects to Baseband
------------------------

.. figure:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/system_model.svg
    :width: 100%
    :alt: Transmitter and Receiver combine into a Radar, which sim_radar turns into baseband

    ``Transmitter`` and ``Receiver`` are pure configuration. ``Radar`` binds
    them to a platform and derives the timing and array geometry. ``sim_radar``
    adds the scene and produces the output arrays.

A minimal end-to-end configuration:

.. code-block:: python

   from radarsimpy import Radar, Transmitter, Receiver
   from radarsimpy.simulator import sim_radar

   wavelength = 3e8 / 79e9

   tx = Transmitter(
       f=[77e9, 81e9],      # 4 GHz sweep
       t=[0, 40e-6],        # 40 us chirp
       tx_power=15,
       pulses=128,
       prp=50e-6,
       channels=[{"location": (0, 0, 0)}],
   )

   rx = Receiver(
       fs=20e6,
       noise_figure=8,
       rf_gain=20,
       baseband_gain=30,
       load_resistor=500,
       channels=[
           {"location": (0, i * wavelength / 2, 0)} for i in range(4)
       ],
   )

   radar = Radar(transmitter=tx, receiver=rx)

   result = sim_radar(radar, targets=[{"location": (50, 0, 0), "rcs": 10}])
   baseband = result["baseband"] + result["noise"]

The division of responsibility is strict:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Object
     - Owns
   * - ``Transmitter``
     - The waveform in time and frequency, how many pulses there are and how
       they are spaced, transmit power and phase noise, and the transmit
       array. See :doc:`transmitter`.
   * - ``Receiver``
     - Sampling rate and baseband type, the gain and noise budget from antenna
       to ADC, the range gate, and the receive array. See :doc:`receiver`.
   * - ``Radar``
     - Where the platform is and how it moves, when frames start, and the
       random seed. Everything else on ``Radar`` is *derived* from the
       transmitter and the receiver.
   * - ``sim_radar``
     - The scene: targets, ray-tracing settings, interference and the
       execution device. See :doc:`ray_tracing_simulation`.


What the Radar Derives
----------------------

Constructing a ``Radar`` computes three things that neither the transmitter nor
the receiver could know on its own.

Samples per pulse
~~~~~~~~~~~~~~~~~

The receive window is open for one pulse length, so

.. math::

   \text{samples\_per\_pulse} = \text{pulse\_length} \times f_s

This is the only place the transmitter's timing and the receiver's sampling
rate meet, and it is a hard constraint: the product must be at least 1, or the
constructor raises. Read it back with ``radar.samples_per_pulse``.

.. note::

   Choose ``fs`` and ``pulse_length`` so that the product is an exact integer.
   Values that land a hair below an integer in binary — ``fs = N / 20e-6`` for
   power-of-two ``N`` is the classic case — are snapped to the nearest integer,
   but keeping the product clean avoids the question entirely.

The virtual array
~~~~~~~~~~~~~~~~~

With :math:`M` transmit channels and :math:`N` receive channels the simulator
produces :math:`M \times N` virtual elements, each at the vector sum of a
transmit and a receive location:

.. figure:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/baseband_cube.svg
    :width: 100%
    :alt: Virtual array element ordering and the shape of the baseband array

    Left: two transmit elements spaced by the full receive aperture give eight
    uniformly spaced virtual elements out of six physical ones. Right: the
    returned arrays are indexed ``[channel, pulse, sample]``.

The ordering is fixed, with the receive index varying fastest:

.. code-block:: text

   ch[0]       = Tx0 -> Rx0
   ch[1]       = Tx0 -> Rx1
   ...
   ch[N-1]     = Tx0 -> Rx(N-1)
   ch[N]       = Tx1 -> Rx0
   ...
   ch[M*N-1]   = Tx(M-1) -> Rx(N-1)

so channel ``n`` corresponds to ``Tx[n // N]`` and ``Rx[n % N]``. The element
positions are available as ``radar.virtual_array_locations``, an ``[M*N, 3]``
array.

.. warning::

   A virtual array is only a valid stand-in for a real one when the transmit
   channels are separable — by time, frequency or code. Placing several
   transmitters at different locations and firing them simultaneously with
   identical waveforms produces a superposition, not :math:`M \times N`
   independent measurements. :doc:`transmitter` describes the modulation
   controls that make the channels separable.

The timestamp
~~~~~~~~~~~~~

``radar.time_prop["timestamp"]`` holds the absolute time of every sample,
shaped ``[channels, pulses, samples]`` like the baseband. It is assembled from
four contributions:

.. code-block:: text

   timestamp[ch, p, s] = frame_start_time            # Radar(frame_time=...)
                       + delay[ch // N]              # Tx channel delay
                       + pulse_start_time[p]         # cumsum(prp) - prp[0]
                       + gate_delay + s / fs         # receive window

This array is also the time base for time-varying motion. Target and radar
trajectories are written as functions of ``radar.time_prop["timestamp"]`` and
indexed positionally by the simulator, so anything that shifts the timestamps —
a transmit delay, a range gate, a non-uniform PRP — shifts the motion sampling
with them.

Noise amplitude
~~~~~~~~~~~~~~~

The thermal-noise standard deviation follows from the receiver's gain and
bandwidth budget alone. It is computed once at construction and stored in
``radar.sample_prop["noise"]``; the chain is laid out in :doc:`receiver`.


Frames
------

``frame_time`` repeats the whole configuration at a list of start times. The
per-frame blocks stack along the same first axis as the channels, with the
frame index varying slowest:

.. code-block:: python

   radar = Radar(transmitter=tx, receiver=rx, frame_time=[0, 0.1, 0.2])

   result = sim_radar(radar, targets)
   # result["baseband"].shape == (3 * M * N, pulses, samples)

   cube = result["baseband"].reshape(3, M * N, pulses, samples)

Each frame gets an independent thermal-noise realization. Within a frame, all
virtual channels that share a physical receiver and a timestamp get *identical*
noise; :doc:`noise` covers what that means for covariance estimation.


See Also
--------

* :doc:`transmitter` — waveform, pulse train and transmit array
* :doc:`receiver` — sampling, gain, noise budget and receive array
* :doc:`coordinate_systems` — how ``location`` and ``rotation`` are interpreted
* :doc:`noise` — thermal noise and phase noise
* :doc:`ray_tracing_simulation` — the scene side of ``sim_radar``
* :doc:`../api/radar` — full API reference for the three classes
