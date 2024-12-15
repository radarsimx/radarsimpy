Coordinate Systems
===================

**Global Coordinate**

- **axis** (m): ``[x, y, z]``
- **phi** (deg): angle on the x-y plane. 0 deg is the positive x-axis, 90 deg is the positive y-axis
- **theta** (deg): angle on the z-x plane. 0 deg is the positive z-axis, 90 deg is the x-y plane

.. image:: https://raw.githubusercontent.com/radarsimx/radarsimpy/refs/heads/master/assets/phi_theta.svg
    :width: 400
    :alt: Alternative text

**Local Coordinate**

- **yaw** (deg): rotation along the z-axis. Positive yaw rotates the object from the positive x-axis to the positive y-axis
- **pitch** (deg): rotation along the y-axis. Positive pitch rotates the object from the positive x-axis to the positive z-axis
- **roll** (deg): rotation along the x-axis. Positive roll rotates the object from the positive y-axis to the positive z-axis
- **origin** (m): ``[x, y, z]``, the rotation centor of the object. Radar's origin is always at ``[0, 0, 0]``

.. image:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/yaw_pitch_roll.svg
    :width: 400
    :alt: Alternative text

- **azimuth** (deg): azimuth -90 ~ 90 deg equal to phi -90 ~ 90 deg
- **elevation** (deg): elevation -90 ~ 90 deg equal to theta 180 ~ 0 deg

.. image:: https://raw.githubusercontent.com/radarsimx/radarsimpy/master/assets/azimuth_elevation.svg
    :width: 400
    :alt: Alternative text
