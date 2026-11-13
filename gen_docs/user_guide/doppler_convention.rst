Doppler Sign Convention
=======================

Radar systems do not all agree on the sign of the Doppler frequency.
RadarSimPy uses the convention that is common in automotive radar:

.. important::

   * A target moving **towards** the radar (closing) has a **negative** Doppler
     frequency.
   * A target moving **away from** the radar (opening) has a **positive**
     Doppler frequency.

Many defense and aerospace systems use the opposite sign. If you compare
RadarSimPy results with such a system, see `Converting to the Other
Convention`_.


Definition
----------

.. math::

   f_d = \frac{2 v_r}{\lambda}, \qquad \lambda = \frac{c}{f_c}

where:

* :math:`f_d` is the Doppler frequency (Hz),
* :math:`v_r` is the radial velocity (m/s): **negative** when the target
  approaches the radar, positive when it moves away,
* :math:`\lambda` is the wavelength (m),
* :math:`f_c` is the carrier frequency (Hz) and :math:`c` is the speed of light
  (about 3×10⁸ m/s).

Because :math:`v_r` is negative for an approaching target, :math:`f_d` is
negative too.

Examples
~~~~~~~~

**Approaching vehicle.** A 77 GHz radar sees a vehicle approaching at 30 m/s
(108 km/h):

* :math:`v_r = -30` m/s
* :math:`\lambda = 3 \times 10^8 / 77 \times 10^9 \approx 3.9` mm
* :math:`f_d = 2 \times (-30) / 0.0039 \approx -15.4` kHz

**Receding vehicle.** A 24 GHz radar sees a vehicle moving away at 20 m/s:

* :math:`v_r = +20` m/s
* :math:`\lambda = 3 \times 10^8 / 24 \times 10^9 = 12.5` mm
* :math:`f_d = 2 \times 20 / 0.0125 = +3.2` kHz


Where the Sign Comes From
-------------------------

The sign is set by how the receiver mixes the echo down to baseband.

Write the transmitted and received signals as

.. math::

   s(t) &= e^{j(2\pi f_c t + \phi(t))} \\
   r(t) &= e^{j(2\pi f_c t + \phi(t) + 2\pi f_D t)}

where :math:`\phi(t)` is the waveform's phase modulation and :math:`f_D` is the
physical Doppler shift, which is positive for an approaching target (the echo
comes back at a higher frequency).

There are two ways to mix the signals:

* **TX × RX\*** (transmit times the conjugate of receive):

  .. math::

     b(t) = s(t) \cdot r^*(t) = e^{-j 2\pi f_D t}

  An approaching target gives a **negative** baseband frequency.

* **RX × TX\*** (receive times the conjugate of transmit):

  .. math::

     b(t) = r(t) \cdot s^*(t) = e^{j 2\pi f_D t}

  An approaching target gives a **positive** baseband frequency.

RadarSimPy uses **TX × RX\***. It is the mixer arrangement found in many
automotive radar chips, where the beat frequency of an up-chirp FMCW waveform is
:math:`f_\text{tx} - f_\text{rx}`, a positive number. As a result, an
approaching target has negative Doppler.


Converting to the Other Convention
----------------------------------

To get positive Doppler for approaching targets, take the complex conjugate of
the baseband signal:

.. math::

   b_\text{converted}(t) = b^*(t)

This flips the sign of every Doppler frequency. Apply it to the raw baseband,
before any processing:

.. code-block:: python

   baseband = np.conj(result["baseband"])

.. note::

   Conjugating also flips the sign of every other frequency in the baseband,
   so it changes how you read range from the beat frequency in a complex FMCW
   baseband.

Both conventions describe the same physics. They differ only in the sign of
:math:`v_r`: RadarSimPy defines it as negative when the target approaches,
and other systems define it as positive.


See Also
--------

* :doc:`coordinate_systems` - How position and velocity vectors are defined
* :doc:`noise` - Phase noise and its effect on the Doppler dimension
* :doc:`../api/radar` - ``Radar``, ``Transmitter`` and ``Receiver`` reference
