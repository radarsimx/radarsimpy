Coordinate Systems
===================

Scene Coordinate
-----------------

    - axis (m): ``[x, y, z]``
    - phi (deg): angle on the x-y plane. 0 deg is the positive x-axis, 90 deg is the positive y-axis
    - theta (deg): angle on the z-x plane. 0 deg is the positive z-axis, 90 deg is the x-y plane
    - azimuth (deg): azimuth -90 ~ 90 deg equal to phi -90 ~ 90 deg
    - elevation (deg): elevation -90 ~ 90 deg equal to theta 180 ~ 0 deg

Object's Local Coordinate
--------------------------

    - axis (m): ``[x, y, z]``
    - yaw (deg): rotation along the z-axis. Positive yaw rotates the object from the positive x-axis to the positive y-axis
    - pitch (deg): rotation along the y-axis. Positive pitch rotates the object from the positive x-axis to the positive z-axis
    - roll (deg): rotation along the x-axis. Positive roll rotates the object from the positive z-axis to the negative y-axis
    - origin (m): ``[x, y, z]``
    - rotation (deg): ``[yaw, pitch, roll]``
    - rotation rate (deg/s): ``[yaw rate, pitch rate, roll rate]``