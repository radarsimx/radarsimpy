Sign Conventions
=================

Doppler Sign Convention
-----------------------

RadarSimPy follows the automotive radar convention for Doppler frequency sign:

* **Negative Doppler frequency**: Target moving **towards** the radar (closing velocity)
* **Positive Doppler frequency**: Target moving **away from** the radar (opening velocity)

This convention is commonly used in automotive radar applications and differs from some other radar systems.

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~

The Doppler frequency shift is defined as:

.. math::

   f_d = \frac{2 v_r}{\lambda}

where:

* :math:`f_d` is the Doppler frequency (Hz)
* :math:`v_r` is the radial velocity (m/s), negative when approaching the radar
* :math:`\lambda` is the wavelength (m)

With this definition, negative radial velocity (approaching) produces negative Doppler frequency.

For a carrier frequency :math:`f_c`:

.. math::

   \lambda = \frac{c}{f_c}

where :math:`c` is the speed of light (approximately 3×10⁸ m/s).

Examples
~~~~~~~~

**Example 1: Approaching Vehicle**

A vehicle moving towards the radar at 30 m/s (108 km/h) with a 77 GHz radar:

* Radial velocity: :math:`v_r = -30` m/s
* Wavelength: :math:`\lambda = \frac{3 \times 10^8}{77 \times 10^9} \approx 3.9` mm
* Doppler frequency: :math:`f_d = \frac{2 \times (-30)}{0.0039} \approx -15.4` kHz (negative)

**Example 2: Receding Vehicle**

A vehicle moving away from the radar at 20 m/s with a 24 GHz radar:

* Radial velocity: :math:`v_r = +20` m/s
* Wavelength: :math:`\lambda = \frac{3 \times 10^8}{24 \times 10^9} = 12.5` mm
* Doppler frequency: :math:`f_d = \frac{2 \times 20}{0.0125} = +3.2` kHz (positive)

Why Automotive Radar Uses This Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sign convention in automotive radar stems from the baseband mixing process used in coherent radar systems. RadarSimPy models this same mixing process in its backend to accurately reflect real-world automotive radar behavior.

**Baseband Mixing**

Automotive radars typically use coherent waveforms (such as FMCW) with homodyne (IQ) mixing. The transmitted and received signals can be expressed as:

.. math::

   s(t) &= e^{j(2\pi f_c t + \phi(t))} \\
   r(t) &= e^{j(2\pi f_c t + \phi(t) - 2\pi f_D t)}

where :math:`f_c` is the carrier frequency, :math:`\phi(t)` is the phase modulation, and :math:`f_D` is the physical Doppler shift.

After mixing, the baseband signal depends on the mixer implementation:

* **RX × TX\*** (receive × transmit conjugate):

  .. math::

     b(t) = r(t) \cdot s^*(t) = e^{-j(2\pi f_D t)}

  This produces **negative Doppler for approaching targets**.

* **TX × RX\*** (transmit × receive conjugate):

  .. math::

     b(t) = s(t) \cdot r^*(t) = e^{j(2\pi f_D t)}

  This produces **positive Doppler for approaching targets**.

Different radar MMIC vendors adopt different mixing conventions. Many automotive radar chipsets use conjugate baseband generation (I - jQ instead of I + jQ), which results in the RX × TX\* convention, producing negative Doppler for approaching targets. This has become the de facto standard in the automotive radar industry.

**Phase Convention**

The received signal phase evolves as:

.. math::

   \phi(t) = 2\pi f_d t = \frac{4\pi v_r}{\lambda} t

For approaching targets (:math:`v_r < 0`), the phase decreases with time, resulting in a negative frequency shift in the spectrum.

Converting to Other Conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For radar applications that require positive Doppler for approaching targets (such as defense or aerospace systems), you can convert the sign convention by taking the complex conjugate of the baseband signal:

.. math::

   b_{converted}(t) = b^*(t)

This flips the sign of all Doppler frequencies, converting negative Doppler (approaching) to positive Doppler. Apply this conjugation to the raw baseband signal before any signal processing.

Summary
~~~~~~~

.. note::

   **Both conventions are mathematically correct** — the difference lies only in the choice of coordinate system and how radial velocity is defined. In RadarSimPy's convention, radial velocity is negative when approaching (closing) and positive when receding (opening). Other systems may define radial velocity with the opposite sign, resulting in positive Doppler for approaching targets. The physical phenomenon remains the same; only the sign convention differs.
