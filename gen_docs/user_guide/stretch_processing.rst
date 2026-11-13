Long-Range Stretch Processing
=============================

An FMCW or stretch radar measures range by mixing the received echo with a
reference chirp. The result is a *beat* tone whose frequency is proportional to
the delay between the echo and the reference.

By default, RadarSimPy uses a reference with **zero delay**. The beat frequency
is then :math:`k \tau`, where :math:`k` is the chirp slope and
:math:`\tau = 2R/c` is the round-trip delay to the target.

This works at short range. At long range, the beat frequency becomes far too
high to sample. This page shows how to fix that with a range gate.


Why a Range Gate Is Needed
--------------------------

Take an X-band stretch radar looking at ships 60 nautical miles away:

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
   * - ADC rate :math:`f_s` (complex)
     - 40 MHz
   * - Max range without a gate, :math:`f_s c / 2k`
     - 999 m
   * - Target range :math:`R`
     - 111.12 km
   * - Round-trip delay :math:`\tau`
     - 741.4 µs
   * - Beat frequency :math:`k \tau`
     - **4.45 GHz**

The beat frequency is more than 200 times higher than the ±20 MHz the ADC can
sample. The target aliases into a meaningless range bin and is lost. Also,
:math:`\tau` is almost 15 chirp lengths, so the echo arrives long after its
chirp has ended.

Real long-range stretch radars solve this by delaying the reference chirp to a
**range gate**. The beat frequency then depends only on how far the target is
from the gate, not on its total range.


Using ``gate_delay``
--------------------

Set ``Receiver(gate_delay=...)`` to the round-trip delay of the range you want
to look at:

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

Now the receive window opens at the gate, and the reference is the transmitted
chirp delayed by the same amount. A target exactly at ``gate_range`` gives a
beat frequency of zero. A target :math:`\Delta R` away from the gate gives
:math:`2 k \Delta R / c`.

.. note::

   ``gate_delay`` defaults to ``0``, which gives exactly the zero-delay
   behaviour. Existing simulations are not affected.


Swath and Resolution
--------------------

The gate does not change the range resolution. It only moves the range window
you can see (the *swath*):

.. math::

   \text{swath around the gate} = \pm \frac{f_s c}{4 k}
   \qquad
   \text{range resolution} = \frac{c}{2B}

For the example above, the swath is **±500 m** at **0.5 m** resolution. That is
2000 range bins, which equals ``pulse_length * fs``, the number of samples per
pulse. A 200–300 m ship fits easily in one gate.

The radar object reports these values:

.. code-block:: python

   radar.chirp_slope                # 6e12 Hz/s
   radar.unambiguous_range_span     # ~999.3 m
   radar.unambiguous_range_window   # (110620.4, 111619.7)
   radar.receiver.gate_range        # 111120.0 m

.. note::

   The window's position depends on whether a gate is set. **Without a gate**,
   every target has a positive delay, so all beat frequencies are positive.
   The usable band is ``[0, fs)`` and the window is ``[0, span]``. **With a
   gate**, targets can be before or after the gate, so beat frequencies can be
   negative. The usable band is ``(-fs/2, +fs/2)`` and the window extends
   ``span/2`` on each side of the gate.


Converting Beat Frequency to Range
----------------------------------

Range is measured from the gate:

.. math::

   R = R_\text{gate} + \frac{f_\text{beat}\, c}{2 k}

.. code-block:: python

   import numpy as np
   import scipy.constants as const

   profile = np.fft.fft(baseband[0, 0, :])
   freqs = np.fft.fftfreq(profile.size, d=1 / rx.bb_prop["fs"])
   ranges = radar.receiver.gate_range + freqs * const.c / (2 * radar.chirp_slope)

Negative beat frequencies are targets **closer** than the gate. With a complex
baseband, the swath therefore covers both sides of the gate.


Covering More Than One Swath
----------------------------

Each simulation has one gate. To cover a larger area, run one simulation per
gate and combine the results:

.. code-block:: python

   for gate_range in np.arange(100e3, 120e3, 1e3):
       rx = Receiver(..., gate_delay=2 * gate_range / const.c)
       ...

Targets outside the swath alias, just as they would without a gate. Choosing
the gate is up to you. The simulator does not check whether targets fall inside
the window, because the beat-frequency relationship only applies to a linear
FM chirp, not to the pulsed, CW or arbitrary waveforms RadarSimPy also
supports.


Effect on Phase Noise
---------------------

In a coherent radar, the reference and the echo come from the same oscillator,
so transmitter phase noise partly cancels. This is the *range correlation*
effect (see :doc:`noise`). With a gate, the reference and the echo are
separated by the remaining delay :math:`\tau - \text{gate\_delay}` instead of
the full round trip. This is the correct model for a radar that generates the
delayed reference digitally, which is the usual design for long-range stretch
radars.

.. important::

   Phase noise results at long range are only meaningful with a gate. Without
   one, the reference and the echo are separated by the full round-trip delay,
   and the close-in phase noise does not cancel the way it would in real
   hardware.


Limitations
-----------

.. note::

   With a gate, an echo actually comes from a chirp sent
   :math:`\text{round}(\text{gate\_delay} / \text{prp})` pulses earlier.
   RadarSimPy applies per-pulse ``f_offset``, ``pulse_amp`` and ``pulse_phs``
   using the *current* pulse index. So frequency-hopped or phase-coded pulse
   trains are not modelled exactly at long gate delays. Identical chirps — the
   usual FMCW case — are not affected.

Near the edges of the window, the reference chirp and the echo chirp only
partly overlap. The simulator extends the chirp there, as if it were
infinitely long, which keeps the whole window usable. Real hardware would have
less correlation gain over about :math:`|\Delta \tau| f_s` samples at the
edges.


See Also
--------

* :doc:`noise` - The phase noise model this page's range gate interacts with
* :doc:`../api/radar` - ``Receiver.gate_delay`` and the derived range properties
