Interference Simulation
=======================

RadarSimPy can simulate mutual interference: the signal another radar
transmits, received directly by the radar under test. This is the dominant
impairment wherever several radars share a band, as automotive radars do at
77 GHz and 60 GHz.

The model is not tied to a particular waveform. Both radars are described by
ordinary ``Transmitter`` objects, so any waveform that can be configured there —
CW, FMCW, pulsed, phase-coded, stepped frequency, an arbitrary :math:`f(t)`
sweep, with or without pulse and fast-time modulation — can act as the victim,
the interferer, or both, and the two need not match.

Interference reaches the victim by a one-way path, so its power falls as
:math:`1/R^2` against :math:`1/R^4` for a target echo. A nearby interferer can
therefore rival or exceed genuine returns even at modest transmit power.


Features
--------

* **Any waveform, on either side** — the victim and the interferer are
  configured independently, each with its own ``f`` and ``t``, ``pulses``,
  ``prp``, ``f_offset`` hopping, pulse modulation (``pulse_amp``,
  ``pulse_phs``) and fast-time modulation (``mod_t``, ``amp``, ``phs``). See
  :doc:`transmitter`.
* **Both antenna patterns** — the interferer's transmit pattern toward the
  victim and the victim's receive pattern toward the interferer, including
  their fields of view.
* **Platform geometry and motion** — both radars have their own ``location``,
  ``speed``, ``rotation`` and ``rotation_rate``, evaluated per sample.
* **Timing** — the interferer contributes only while it is actually
  transmitting: not before its first frame, not between its pulses, and not
  where its ``amp`` gate is off.
* **Receiver passband** — only the part of the interfering signal that falls
  inside the victim's baseband bandwidth reaches the output.
* **A separate output** — interference comes back as its own array, next to
  ``baseband`` and ``noise``, so it can be analysed, suppressed or combined as
  needed.


Adding an Interferer
--------------------

1. Describe the interfering radar's waveform with a ``Transmitter``.
2. Give it a ``Receiver`` — one is required to build a ``Radar``, but it takes
   no part in the interference calculation, so any valid settings will do.
3. Place and orient it with ``Radar(location=..., rotation=...)``.
4. Pass it to ``sim_radar`` as ``interf``.

The configuration below is the one used in the
`interference example <https://radarsimx.com/2023/01/13/interference/>`_: two
60 GHz radars 30 m apart and facing each other, one down-chirping and one
up-chirping across the same 200 MHz, each hopping carrier from chirp to chirp.

.. code-block:: python

   import numpy as np
   from radarsimpy import Radar, Transmitter, Receiver
   from radarsimpy.simulator import sim_radar

   # the radar under test: down-chirp, 4 chirps hopping in 90 MHz steps
   tx = Transmitter(
       f=[60.6e9, 60.4e9], t=[0, 16e-6], tx_power=25, prp=20e-6, pulses=4,
       f_offset=np.arange(4) * 90e6,
       channels=[{"location": (0, 0, 0), "pulse_phs": np.array([180, 0, 0, 0])}],
   )
   rx = Receiver(fs=40e6, noise_figure=2, rf_gain=20, load_resistor=500,
                 baseband_gain=60, channels=[{"location": (0, 0, 0)}])
   radar = Radar(transmitter=tx, receiver=rx)

   # the interferer: up-chirp, 8 chirps hopping in 70 MHz steps, 30 m away
   int_tx = Transmitter(
       f=[60.4e9, 60.6e9], t=[0, 8e-6], tx_power=15, prp=11e-6, pulses=8,
       f_offset=np.arange(8) * 70e6,
       channels=[{"location": (0, 0.1, 0),
                  "pulse_phs": np.array([0, 0, 180, 0, 0, 0, 0, 0])}],
   )
   int_rx = Receiver(fs=20e6, channels=[{"location": (0, 0.1, 0)}])
   interferer = Radar(transmitter=int_tx, receiver=int_rx,
                      location=(30, 0, 0), rotation=(180, 0, 0))

   targets = [
       {"location": (30, 0, 0), "speed": (0, 0, 0), "rcs": 10},
       {"location": (20, 1, 0), "speed": (-10, 0, 0), "rcs": 10},
   ]

   result = sim_radar(radar, targets, interf=interferer)

``rotation=(180, 0, 0)`` turns the interferer to face back along −x toward the
victim; see :doc:`coordinate_systems` for the angle conventions.


Other Waveforms
---------------

Nothing in the setup above is specific to FMCW. Swapping the interferer's
``Transmitter`` is all it takes to study a different kind of neighbour — here a
CW tone, and the same tone phase-coded at 100 ns per chip, both aimed at the
FMCW victim above:

.. code-block:: python

   # a continuous tone in the middle of the victim's band
   cw_tx = Transmitter(f=60.5e9, t=100e-6, tx_power=15,
                       channels=[{"location": (0, 0, 0)}])

   # the same tone, phase-coded: a PMCW neighbour
   chip = 100e-9
   mod_t = np.arange(0, 100e-6, chip)
   code = np.random.default_rng(0).choice([0, 180], mod_t.size)
   pmcw_tx = Transmitter(f=60.5e9, t=100e-6, tx_power=15,
                         channels=[{"location": (0, 0, 0),
                                    "mod_t": mod_t, "phs": code}])

   for neighbour_tx in (cw_tx, pmcw_tx):
       neighbour = Radar(transmitter=neighbour_tx, receiver=int_rx,
                         location=(30, 0, 0), rotation=(180, 0, 0))
       result = sim_radar(radar, targets, interf=neighbour)

Pulsed, stepped-frequency or arbitrary-sweep neighbours are built the same way,
using the patterns in :doc:`transmitter`. The victim can be any of these too.


Reading the Output
------------------

``result["interference"]`` has the same ``[channels, pulses, samples]`` shape
and channel order as ``baseband`` and ``noise``. Nothing is pre-mixed, so the
received signal is composed explicitly:

.. code-block:: python

   measured = result["baseband"] + result["noise"] + result["interference"]

Keeping the components apart makes common analyses direct:

.. code-block:: python

   # signal-to-interference ratio
   sir_db = 20 * np.log10(np.abs(result["baseband"]).max()
                          / np.abs(result["interference"]).max())

   # which pulses were hit at all, [channels, pulses]
   hit = np.abs(result["interference"]).max(axis=-1) > 0

   # which samples were hit — the mask a blanking scheme needs
   mask = np.abs(result["interference"]) > 0

Targets are optional. ``sim_radar(radar, [], interf=interferer)`` returns an
all-zero ``baseband`` and a fully populated ``interference``, which is the
quickest way to study the impairment on its own.


How Interference Appears
------------------------

For every sample, the simulator works out the victim's own instantaneous
frequency, :math:`f_\text{lo}(t)`, and the frequency the interferer was
radiating when the arriving energy left it, :math:`f_\text{interf}(t)`. Both
come straight from each ``Transmitter``'s ``f``, ``t`` and ``f_offset``, so the
rule is the same whatever the two waveforms are. The interferer reaches the
output only while

.. math::

   \lvert f_\text{lo}(t) - f_\text{interf}(t) \rvert < f_s

where :math:`f_s` is the victim receiver's sampling rate ``fs``. For a real
baseband receiver (``bb_type="real"``) the passband is half as wide,
:math:`f_s / 2` (:doc:`receiver`).

Inside that window the interference carries the interferer's own amplitude and
phase modulation — its pulse codes, its ``amp`` gate, its fast-time phase code
— and oscillates at the beat frequency
:math:`f_\text{lo} - f_\text{interf}`. The victim's own modulation does not
gate reception: its receiver listens for the whole sample window whatever its
transmitter is doing.

.. figure:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/interference_crossing.svg
    :width: 100%
    :alt: An interfering chirp only reaches the baseband while it crosses the victim receiver's passband

    The rule applied to two linear chirps of different slope. Each time the
    interferer crosses the victim's passband it leaves a burst in the baseband.
    The burst has constant amplitude; its beat frequency sweeps through zero at
    the moment the two chirps cross, so it oscillates quickly at its edges and
    slowly at its centre.

What the rule produces depends on how the two frequency curves meet:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Victim and interferer
     - What appears in the baseband
   * - Two chirps, different slopes
     - A short burst at each crossing, as in the figure
   * - Two chirps, the same slope
     - Persistent interference across the chirp; it behaves like a false
       target rather than a transient
   * - A chirp and a CW tone
     - One burst per chirp, when the chirp sweeps past the tone
   * - Two CW tones
     - A steady beat at their frequency offset, or nothing if the offset is
       beyond the passband
   * - A phase-coded interferer
     - Interference with the interferer's code imprinted on its phase, which
       spreads it into a noise-like floor
   * - A pulsed interferer
     - Only the parts of the above that coincide with its pulses
   * - Arbitrary sweeps
     - A burst wherever the two curves come within the passband of each other

For two linear sweeps, a burst lasts as long as the interferer takes to
traverse the passband, :math:`2 f_s / \lvert k_\text{victim} -
k_\text{interf} \rvert`, where :math:`k` is the sweep rate and a CW tone has
:math:`k = 0`.

Pulse to pulse, the bursts move, appear and disappear as the two radars'
repetition periods and frequency hops walk past each other.


Several Interferers
-------------------

``interf`` accepts a single ``Radar``. For several interferers, simulate each
one separately and sum the interference — contributions add linearly, and
``baseband`` is unaffected by the choice of interferer:

.. code-block:: python

   result = sim_radar(radar, targets, interf=interferers[0])
   interference = result["interference"]

   for other in interferers[1:]:
       interference = interference + sim_radar(radar, [], interf=other)["interference"]

   measured = result["baseband"] + result["noise"] + interference

Passing ``[]`` as the targets for the additional runs skips work that would
only reproduce the same ``baseband``. The interferers can each use a different
waveform.


Shaping the Scenario
--------------------

The same parameters that define each radar control how strongly, how often and
where interference lands:

* **Geometry** — ``location`` sets the path length; ``rotation`` points each
  antenna toward or away from the other; ``speed`` and ``rotation_rate`` make
  the coupling evolve through the frame.
* **Antenna patterns** — ``azimuth_pattern`` and ``elevation_pattern`` on
  either side scale the coupling directly, and angles outside a pattern's span
  receive nothing (:doc:`transmitter`).
* **Power** — the interferer's ``tx_power`` scales the interference linearly in
  dB; the victim's scales only its own echoes.
* **Waveform** — the two frequency curves decide when and for how long they
  overlap; ``f_offset`` hopping and a staggered ``prp`` change which pulses
  collide; the interferer's pulse and fast-time modulation are carried into
  the interference.
* **Receiver passband** — the victim's ``fs`` sets the passband width, halved
  for real baseband, and with it how long each overlap lasts.
* **Timing** — each radar's ``frame_time`` aligns or offsets the two
  transmissions.


Limitations
-----------

* Only the **direct path** is modelled. The interferer's signal reflecting off
  targets or the environment before reaching the victim is not included.
* One interferer per ``sim_radar`` call; combine several as shown above.
* The interferer's ``Receiver`` is not used.


See Also
--------

* `Interference example <https://radarsimx.com/2023/01/13/interference/>`_ —
  the FMCW configuration above as a notebook, with time-frequency and baseband
  plots
* :doc:`transmitter` — configuring CW, FMCW, pulsed, phase-coded and arbitrary
  waveforms
* :doc:`system_model` — why the output components come back separately
* :doc:`receiver` — ``fs`` and ``bb_type``, which set the passband
* :doc:`coordinate_systems` — placing and orienting each radar
* :doc:`noise` — the other additive impairment
* :doc:`../api/sim` — the ``sim_radar`` reference
