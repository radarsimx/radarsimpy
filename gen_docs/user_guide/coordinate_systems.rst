Coordinate Systems
==================

RadarSimPy places everything — radars, targets, antenna patterns, motion — in a
single right-handed, z-up Cartesian frame. Angles are in degrees, distances in
metres, and no part of the library uses a different convention internally.

This page defines that frame, the two angle pairs used to point directions
inside it, and the Euler angles that orient objects within it.


At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 56

   * - Quantity
     - Symbol
     - Unit
     - Meaning
   * - position
     - ``[x, y, z]``
     - m
     - Right-handed, z pointing up
   * - phi
     - :math:`\phi`
     - °
     - Azimuthal angle in the x–y plane, 0° at +x
   * - theta
     - :math:`\theta`
     - °
     - Polar angle from +z, 0° at zenith
   * - azimuth
     - —
     - °
     - Same angle as :math:`\phi`, radar-centric name
   * - elevation
     - —
     - °
     - Angle above the x–y plane, :math:`90° - \theta`
   * - orientation
     - ``[yaw, pitch, roll]``
     - °
     - Rotation about z, −y and x respectively


The Global Frame
----------------

The axes carry no built-in geographic meaning — you choose what they represent
— but the handedness is fixed:

- **x** — forward, the boresight direction for a radar at zero orientation
- **y** — to the left, when looking along +x
- **z** — up

Any direction can be named either by a unit vector or by the spherical pair
:math:`(\phi, \theta)`:

.. figure:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/phi_theta.svg
    :width: 100%
    :alt: Spherical angles phi and theta in the global right-handed frame

    :math:`\phi` sweeps within the x–y plane starting from +x; :math:`\theta`
    opens downward from +z. Both arrows show the direction of increase.

**phi** (:math:`\phi`) is the azimuthal angle in the x–y plane. It is 0° along
+x and 90° along +y, increasing counter-clockwise when viewed from above, and
spans 0°…360° (or equivalently −180°…180°).

**theta** (:math:`\theta`) is the polar angle measured from +z. It is 0° at the
zenith, 90° in the x–y plane, and 180° at the nadir, spanning 0°…180°.

Converting between the two descriptions:

.. math::

   x &= r \sin\theta \cos\phi \\
   y &= r \sin\theta \sin\phi \\
   z &= r \cos\theta

.. math::

   r &= \sqrt{x^2 + y^2 + z^2} \\
   \phi &= \operatorname{atan2}(y, x) \\
   \theta &= \arccos(z / r)

.. code-block:: python

   import numpy as np

   def to_spherical(v):
       x, y, z = v
       r = np.linalg.norm(v)
       return r, np.degrees(np.arctan2(y, x)), np.degrees(np.arccos(z / r))

   to_spherical([1, 1, 1])     # (1.732, 45.0, 54.74)


Radar-Centric Angles
--------------------

Antenna patterns and beam geometry are more naturally described relative to
boresight than to the zenith, so the same directions get a second pair of
names:

.. figure:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/azimuth_elevation.svg
    :width: 100%
    :alt: Radar-centric azimuth and elevation angles referenced to boresight along plus x

    Boresight is +x, where azimuth and elevation are both zero. Azimuth turns
    within the x–y plane toward +y; elevation lifts out of that plane toward +z.

**azimuth** is the horizontal angle in the x–y plane. It is 0° at +x — the
boresight — and positive toward +y, which is to the left as seen from behind
the radar looking forward. It is the same angle as :math:`\phi`.

**elevation** is the vertical angle away from the x–y plane, 0° at the horizon
and positive toward +z.

.. math::

   \text{azimuth} &= \phi \\
   \text{elevation} &= 90° - \theta

The difference is only where zero sits and which way is positive; nothing is
lost or gained by switching between the pairs.

.. note::

   Transmit and receive channels default to patterns spanning ``[-90, 90]`` in
   both cuts (``azimuth_angle`` and ``elevation_angle``). That is the default
   extent of the *pattern arrays*, not a restriction on the angles themselves.
   See :doc:`transmitter` and :doc:`receiver` for how patterns are specified,
   including the convention that the azimuth cut carries the absolute gain.


Orientation
-----------

Objects are oriented with three Euler angles given as ``[yaw, pitch, roll]``:

.. figure:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/yaw_pitch_roll.svg
    :width: 100%
    :alt: Yaw, pitch and roll Euler angles, each coloured by the axis it turns about

    Each arc is drawn in the colour of the axis it turns about. Pitch is the
    exception worth noting: it turns about −y, not +y.

.. list-table::
   :header-rows: 1
   :widths: 14 14 36 36

   * - Angle
     - About
     - Positive sense
     - Typical range
   * - ``yaw``
     - +z
     - Turns +x toward +y
     - −180°…180°
   * - ``pitch``
     - −y
     - Turns +x toward +z
     - −90°…90°
   * - ``roll``
     - +x
     - Turns +y toward +z
     - −180°…180°

The composed rotation is

.. math::

   R = R_z(\text{yaw}) \cdot R_y(-\text{pitch}) \cdot R_x(\text{roll})

.. important::

   **Pitch is the odd one out.** Yaw and roll are ordinary right-handed
   rotations about +z and +x. Pitch is not: a right-handed rotation about +y
   would turn +x toward *−z*, whereas positive pitch turns +x toward *+z*. That
   is the aerospace "nose up is positive" convention, and it is why the matrix
   above carries :math:`R_y(-\text{pitch})` rather than :math:`R_y(\text{pitch})`.
   Expect a sign flip on pitch when importing orientations from a toolchain
   that uses the strict right-handed sense.

Order of application
~~~~~~~~~~~~~~~~~~~~

The three angles are an intrinsic **yaw → pitch → roll** sequence: yaw about
the global z, then pitch about the *already yawed* y, then roll about the
*already yawed and pitched* x. Written as a matrix acting on a column vector,
that same sequence reads right to left — roll reaches the vector first.

Either way the order is not negotiable, because rotations do not commute:

.. code-block:: python

   import numpy as np
   from radarsimpy.animation_kit import _rsx_euler_to_quat, _quat_rotate

   def rotate(rotation_deg, vec):
       q = _rsx_euler_to_quat(np.radians(rotation_deg))
       return _quat_rotate(q, np.array(vec, dtype=float))

   rotate([90, 0, 90], [0, 1, 0])      # -> [0, 0, 1]

   # roll first, then yaw, about the fixed global axes: same answer
   rotate([90, 0, 0], rotate([0, 0, 90], [0, 1, 0]))    # -> [0, 0, 1]

   # yaw first, then roll: a different direction entirely
   rotate([0, 0, 90], rotate([90, 0, 0], [0, 1, 0]))    # -> [-1, 0, 0]

Pointing the boresight
~~~~~~~~~~~~~~~~~~~~~~

One consequence is worth having to hand. Roll turns about the body's own x
axis, which *is* the boresight, so it never moves it. Yaw and pitch are then
exactly the azimuth and elevation of the resulting boresight direction:

.. code-block:: python

   rotation = [azimuth, elevation, roll]

To aim a radar at 30° azimuth and 20° below the horizon, set
``rotation=[30, -20, 0]``; the third angle is free to spin the antenna pattern
about the beam without changing where it points.

Origin
~~~~~~

- ``origin`` (m) is the point that rotation and translation act about.
- A radar's origin is always ``[0, 0, 0]`` — position it with ``location``
  instead.
- Targets may set an arbitrary ``origin``, which is what lets a mesh rotate
  about a hinge, an axle or its own centre of mass rather than about the model
  file's zero.


Notes
-----

- Angles are degrees and distances metres everywhere, including motion rates
  (°/s and m/s).
- Right-handedness holds throughout; ``pitch`` is a sign convention on top of
  it, not a departure from it.
- glTF assets arrive Y-up and are converted on import — see
  :doc:`animated_targets`.


See Also
--------

* :doc:`system_model` — where the radar sits in this frame, and its virtual array
* :doc:`transmitter` — antenna pattern cuts and the gain convention
* :doc:`receiver` — the receive array in the same frame
* :doc:`ray_tracing_simulation` — placing and orienting 3D mesh targets
* :doc:`animated_targets` — frame conversion for glTF assets (Y-up to Z-up)
* :doc:`doppler_convention` — the sign convention for radial velocity
