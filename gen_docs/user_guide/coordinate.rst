Coordinate Systems
===================

RadarSimPy uses right-handed coordinate systems for all spatial representations. This page describes the global and local coordinate systems used throughout the library.

Global Coordinate System
-------------------------

The global coordinate system defines the absolute reference frame for all simulations.

**Cartesian Coordinates**

- **axis** (m): ``[x, y, z]`` - Position vector in meters
  
  - ``x``: East-West axis
  - ``y``: North-South axis  
  - ``z``: Vertical axis (up)

**Spherical Angles**

- **phi** (φ, deg): Azimuthal angle in the x-y plane
  
  - Range: 0° to 360° (or -180° to 180°)
  - 0° corresponds to the positive x-axis
  - 90° corresponds to the positive y-axis
  - Measured counter-clockwise when viewed from above

- **theta** (θ, deg): Polar angle from the z-axis
  
  - Range: 0° to 180°
  - 0° corresponds to the positive z-axis (zenith)
  - 90° corresponds to the x-y plane (horizon)
  - 180° corresponds to the negative z-axis (nadir)

.. image:: https://raw.githubusercontent.com/radarsimx/radarsimpy/refs/heads/master/assets/phi_theta.svg
    :width: 400
    :alt: Phi and Theta angle definitions in spherical coordinates

Local Coordinate System
------------------------

The local coordinate system defines object-specific reference frames using Euler angles and origin translations.

**Euler Angle Rotations**

Rotations are applied in the order: yaw → pitch → roll (Z-Y-X convention).

- **yaw** (deg): Rotation about the z-axis
  
  - Positive yaw rotates counter-clockwise from the positive x-axis toward the positive y-axis
  - Range: -180° to 180° (or 0° to 360°)

- **pitch** (deg): Rotation about the y-axis
  
  - Positive pitch rotates the positive x-axis toward the positive z-axis
  - Range: -90° to 90°

- **roll** (deg): Rotation about the x-axis
  
  - Positive roll rotates the positive y-axis toward the positive z-axis
  - Range: -180° to 180°

**Origin**

- **origin** (m): ``[x, y, z]`` - The center point for rotation and translation operations
  
  - All rotations are performed about this point
  - The radar's origin is always fixed at ``[0, 0, 0]``
  - Target objects can have arbitrary origins

.. image:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/yaw_pitch_roll.svg
    :width: 400
    :alt: Yaw, pitch, and roll angle definitions

Radar-Centric Angles
--------------------

For radar applications, azimuth and elevation angles provide an intuitive alternative to phi and theta.

**Angle Definitions**

- **azimuth** (deg): Horizontal angle in the local x-y plane
  
  - Range: -90° to 90°
  - Equivalent to φ in the range [-90°, 90°]
  - 0° is boresight (forward direction)
  - Positive values are to the left, negative values are to the right

- **elevation** (deg): Vertical angle from the horizon
  
  - Range: -90° to 90°
  - Related to θ by: elevation = 90° - θ
  - 0° is the horizon (x-y plane)
  - Positive values are above the horizon, negative values are below

**Relationship to Global Coordinates**

.. math::

   \text{azimuth} &= \phi \quad \text{for } \phi \in [-90°, 90°] \\
   \text{elevation} &= 90° - \theta

.. image:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/azimuth_elevation.svg
    :width: 400
    :alt: Azimuth and elevation angle definitions

Notes
-----

- All angles use degrees unless otherwise specified
- All distances use meters as the base unit
- Right-handed coordinate systems ensure consistency with standard conventions
- The coordinate transformation order matters: always apply yaw, then pitch, then roll
