Receiver Noise
==============

RadarSimPy simulates thermal noise at the receiver. This section describes the
statistical properties of the generated noise and how it behaves in
single-channel and MIMO radar configurations.

Noise Properties
----------------

The receiver noise has the following properties:

* **Gaussian white noise** — zero-mean, with standard deviation determined by
  the receiver noise parameters (noise figure, gains, bandwidth, and load
  resistance).
* **Independent across frames** — each frame receives a fresh, independent
  noise realization.
* **Independent across Rx channels** — different physical Rx channels have
  statistically independent noise.
* **Deterministic given a seed** — when a random seed is provided, the noise
  is fully reproducible.

Complex vs. Real Baseband
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Complex baseband** (``bb_type="complex"``): noise has both real and
  imaginary components. Each component is an independent Gaussian with
  standard deviation :math:`\sigma / \sqrt{2}`, so the total RMS amplitude
  equals :math:`\sigma`.
* **Real baseband** (``bb_type="real"``): noise is purely real with standard
  deviation :math:`\sigma`.

Same Timestamp, Same Rx → Same Noise
-------------------------------------

A key property of the noise simulation is:

.. important::

   If two output samples belong to the **same physical Rx channel** and
   correspond to the **same timestamp**, they receive **exactly identical**
   noise values — regardless of how many virtual channels share that
   combination.

The noise value for each sample is a deterministic function of three
identifiers:

1. **Frame index** — which frame the sample belongs to.
2. **Physical Rx channel** — determined by ``rx_ch = ch_idx % rx_size``.
3. **Absolute sample index** — the sample's position on the discrete time grid
   (timestamp offset from the minimum, quantized by the sampling rate).

Because the function is deterministic, any two samples with the same
``(frame_idx, rx_ch, absolute_sample_index)`` tuple always produce the same
noise value. No lookup table or shared buffer is required.

Noise in MIMO Configurations
-----------------------------

In a MIMO radar with :math:`M` Tx and :math:`N` Rx channels, the simulator
produces :math:`M \times N` virtual channels per frame. The virtual channels
are ordered as:

.. code-block:: text

   ch[0]         = Tx0 → Rx0
   ch[1]         = Tx0 → Rx1
   ...
   ch[N-1]       = Tx0 → Rx(N-1)
   ch[N]         = Tx1 → Rx0
   ch[N+1]       = Tx1 → Rx1
   ...
   ch[M*N - 1]   = Tx(M-1) → Rx(N-1)

Each virtual channel maps to a physical Rx channel via:

.. math::

   \text{rx\_ch} = \text{ch\_idx} \mod N

Virtual Channel Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~

All virtual channels that share the **same physical Rx** and have the **same
transmit timing** (i.e., identical timestamps) will contain identical noise.
For example, with 2 Tx and 2 Rx:

.. code-block:: text

   ch[0] = Tx0 → Rx0    ─┐
   ch[2] = Tx1 → Rx0    ─┘  Same Rx0, same timing → identical noise

   ch[1] = Tx0 → Rx1    ─┐
   ch[3] = Tx1 → Rx1    ─┘  Same Rx1, same timing → identical noise

This matches the physical reality: the noise originates at the receiver
front-end, so all signal paths through the same Rx channel at the same time
experience the same thermal noise.

.. note::

   If the Tx channels have different timing offsets (e.g., TDM-MIMO with
   staggered pulse timing), each Tx fires at a different time. In that case,
   even though two virtual channels share the same Rx, they sample the
   noise at different time indices and thus receive **different** noise
   values.

Impact on MIMO Signal Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When performing MIMO beamforming or angle estimation:

* **Noise correlation**: virtual channels sharing the same Rx have fully
  correlated noise. The noise covariance matrix across virtual channels is
  **not** a scaled identity — it has block structure reflecting the Rx
  grouping.
* **Effective independent noise sources**: only :math:`N` independent noise
  realizations exist (one per physical Rx), duplicated across the :math:`M`
  Tx paths that share the same timing.
* **SNR estimation**: the noise power per virtual channel equals the physical
  Rx noise power. Averaging across virtual channels that share the same Rx
  does **not** reduce noise because the samples are identical.

Example
~~~~~~~

.. code-block:: python

   import numpy as np
   from radarsimpy import Radar, Transmitter, Receiver
   from radarsimpy.simulator import sim_radar

   # 2 Tx, 2 Rx MIMO radar
   tx = Transmitter(
       f=[24.075e9, 24.175e9],
       t=80e-6,
       tx_power=10,
       prp=100e-6,
       pulses=4,
       channels=[{"location": (0, 0, 0)}, {"location": (0, 0, 0)}],
   )
   rx = Receiver(
       fs=2e6,
       noise_figure=10,
       rf_gain=20,
       load_resistor=500,
       baseband_gain=30,
       bb_type="complex",
       channels=[{"location": (0, 0, 0)}, {"location": (0, 0, 0)}],
   )
   radar = Radar(transmitter=tx, receiver=rx)

   result = sim_radar(radar, targets=[{"location": (200, 0, 0), "rcs": 10}])
   noise = result["noise"]

   # Virtual channels: ch0=Tx0-Rx0, ch1=Tx0-Rx1, ch2=Tx1-Rx0, ch3=Tx1-Rx1
   # ch0 and ch2 share Rx0 with same timing → identical noise
   assert np.array_equal(noise[0], noise[2])

   # ch1 and ch3 share Rx1 with same timing → identical noise
   assert np.array_equal(noise[1], noise[3])

   # Different Rx channels have independent noise
   assert not np.array_equal(noise[0], noise[1])

Multi-Frame Behavior
---------------------

When simulating multiple frames, each frame generates an independent noise
realization. Virtual channels in different frames always have different noise,
even if they share the same physical Rx channel:

.. code-block:: python

   # With 2 frames, 1 Tx, 1 Rx:
   # ch[0] = Frame 0, ch[1] = Frame 1
   assert not np.array_equal(noise[0], noise[1])

This ensures that frame-to-frame processing (e.g., Doppler estimation) sees
independent noise across slow-time snapshots.
