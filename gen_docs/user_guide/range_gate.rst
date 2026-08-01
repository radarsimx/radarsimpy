Long-Range Stretch Processing
=============================

FMCW and stretch radars recover range by mixing the received echo against a
reference chirp, producing a beat tone proportional to the delay difference.
By default RadarSimPy deramps against a **zero-delay reference**: the beat
frequency is :math:`k \tau`, where :math:`k` is the chirp slope and
:math:`\tau = 2R/c` the full round-trip delay.

That is fine at short range, but it breaks down long before the ranges real
long-range radars operate at.

Why a Range Gate Is Needed
--------------------------

Consider an X-band stretch radar imaging ships at 60 nautical miles:

.. list-table::
   :header-rows: 1
   :widths: 40 30

   * - Parameter
     - Value
   * - Carrier :math:`f_c`
     - 9 GHz
   * - Bandwidth :math:`B`
     - 300 MHz
   * - Chirp length :math:`T`
     - 50 µs
   * - Chirp slope :math:`k = B/T`
     - 6 × 10\ :sup:`12` Hz/s
   * - IF/ADC rate :math:`f_s` (complex)
     - 40 MHz
   * - Max un-gated range :math:`f_s c / 2k`
     - 999 m
   * - Target range :math:`R`
     - 111.12 km
   * - Round-trip delay :math:`\tau`
     - 741.4 µs
   * - Beat frequency :math:`k \tau`
     - **4.45 GHz**

The beat tone is over 200 times the Nyquist limit of ±20 MHz. The target
aliases into an essentially arbitrary range bin and is unrecoverable. Note also
that :math:`\tau` is nearly 15 chirp lengths, so the echo arrives long after the
chirp that produced it has ended.

Real long-range stretch radars avoid this by mixing against a reference chirp
delayed to a **range gate**, so the beat depends only on the target's offset
from the gate rather than on its absolute range.

Using ``gate_delay``
--------------------

Set :code:`Receiver(gate_delay=...)` to the two-way delay of the range you want
to gate on:

.. code-block:: python

   import scipy.constants as const
   from radarsimpy import Radar, Transmitter, Receiver

   gate_range = 111.12e3  # m

   tx = Transmitter(
       f=[8.85e9, 9.15e9],   # 300 MHz sweep about 9 GHz
       t=50e-6,              # 50 us chirp
       tx_power=40,
       prp=200e-6,
       pulses=256,
   )

   rx = Receiver(
       fs=40e6,
       noise_figure=8,
       rf_gain=20,
       baseband_gain=30,
       gate_delay=2 * gate_range / const.c,   # ~741.4 us
   )

   radar = Radar(transmitter=tx, receiver=rx)

The receive window now opens at the gate and the deramp reference is the
transmit chirp delayed by the same amount. A target at exactly ``gate_range``
beats at DC; a target offset by :math:`\Delta R` beats at
:math:`2 k \Delta R / c`.

.. note::

   ``gate_delay`` defaults to ``0``, which reproduces the zero-delay behavior
   exactly. Existing simulations are unaffected.

Swath and Resolution
--------------------

The gate does not change resolution, only where the usable window sits:

.. math::

   \text{swath about the gate} = \pm \frac{f_s c}{4 k}
   \qquad
   \text{range resolution} = \frac{c}{2B}

For the configuration above that is a **±500 m** swath at **0.5 m** resolution,
i.e. 2000 range bins — exactly ``pulse_length * fs``, the number of samples per
pulse. A 200–300 m ship fits comfortably inside one gate.

Both are available from the radar object:

.. code-block:: python

   radar.chirp_slope                # 6e12 Hz/s
   radar.unambiguous_range_span     # ~999.3 m
   radar.unambiguous_range_window   # (110620.4, 111619.7)
   radar.receiver.gate_range        # 111120.0 m

.. note::

   Where the window sits depends on whether a gate is configured. Un-gated,
   every target has a positive round-trip delay, so all beat tones are positive
   and the whole ``[0, fs)`` band is usable — the window is ``[0, span]``. Once
   a gate makes the residual delay signed, the usable band becomes
   ``(-fs/2, +fs/2)`` and the window straddles the gate at ``+/- span/2``.

Converting Beat Bins to Range
-----------------------------

Range is measured relative to the gate:

.. math::

   R = R_\text{gate} + \frac{f_\text{beat}\, c}{2 k}

.. code-block:: python

   import numpy as np
   import scipy.constants as const

   profile = np.fft.fft(baseband[0, 0, :])
   freqs = np.fft.fftfreq(profile.size, d=1 / rx.bb_prop["fs"])
   ranges = radar.receiver.gate_range + freqs * const.c / (2 * radar.chirp_slope)

Negative beat frequencies correspond to targets **inside** the gate, so with
complex baseband the swath is two-sided about the gate.

Covering More Than One Swath
----------------------------

A single simulation has a single gate. To cover a wider area, run the
simulation once per gate and stitch the results:

.. code-block:: python

   for gate_range in np.arange(100e3, 120e3, 1e3):
       rx = Receiver(..., gate_delay=2 * gate_range / const.c)
       ...

Targets outside the swath alias exactly as they would without a gate. Sizing
the gate is up to you; the simulator does not check target ranges against the
window, since the beat-frequency relationship only applies to deramp processing
of a linear FM waveform and not to the pulsed, CW, or arbitrary-waveform
configurations RadarSimPy also supports.

Effect on Phase Noise
---------------------

Transmitter phase noise partially cancels in a coherent radar because the
reference and the echo are drawn from the same oscillator a short time apart —
the *range correlation* effect. With a gate, the two are separated by the
residual delay :math:`\tau - \text{gate\_delay}` rather than by the full round
trip, which is the correct model for a digitally generated delayed reference
(the usual long-range stretch architecture).

.. important::

   Phase-noise results at long range are only meaningful with a gate
   configured. Without one, the reference and echo are separated by the full
   round-trip delay and the phase-noise lookup wraps, so close-in phase noise
   does not cancel as it would in real hardware.

Limitations
-----------

.. note::

   The echo of a gated return physically belongs to the chirp transmitted
   :math:`\text{round}(\text{gate\_delay} / \text{prp})` pulses earlier.
   Per-pulse ``f_offset`` and ``pulse_amp`` / ``pulse_phs`` are applied using
   the current pulse index, so frequency-hopped or phase-coded pulse trains are
   not modeled exactly at long gate delays. Identical chirps — the common FMCW
   case — are unaffected.

Near the edges of the window the reference and echo chirps only partially
overlap. The simulator extrapolates the chirp there, modeling an ideal infinite
chirp, which keeps the full window usable. Real hardware would see reduced
correlation gain over roughly :math:`|\Delta \tau| f_s` samples.
