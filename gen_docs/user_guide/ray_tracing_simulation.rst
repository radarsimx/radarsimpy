Ray-Tracing Simulation
======================

This page describes the parameters of ``sim_radar`` that control the
ray-tracing and Physical Optics (PO) based simulation for 3D mesh targets,
as well as the target-level flags that affect how individual objects are
handled during the simulation.


``sim_radar`` Parameters
------------------------

The following parameters of ``sim_radar`` control the mesh simulation
behavior:

``density``
~~~~~~~~~~~

:Type: ``float``
:Default: ``1.0``

Ray density, defined as the **number of rays per wavelength**. This parameter
directly controls how many rays are launched towards each occupied grid cell
during the pyramid ray generation stage.

- **Higher values** produce more rays, leading to finer spatial sampling of
  the target surface and more accurate PO results, at the cost of increased
  computation time and GPU/CPU memory usage.
- **Lower values** reduce the number of rays for faster computation, but may
  miss small geometric features or produce less accurate scattering results.

A value of ``1.0`` means one ray per wavelength, which is generally a
reasonable starting point. Increase the density for scenes requiring higher
fidelity, or decrease it below ``1.0`` for faster computation when high
spatial resolution is not needed or when the target mesh faces are large
relative to the wavelength.

.. tip::

   Individual targets can override the global density by setting a per-target
   ``density`` value in the target dictionary. When a target's density is
   ``0.0`` (the default), the global density from ``sim_radar`` is used.

``level``
~~~~~~~~~

:Type: ``str`` or ``None``
:Default: ``None``

Controls the **simulation fidelity** by determining how often ray-tracing is
performed across the radar frame:

- ``None`` or ``"frame"`` — Perform **one** ray-tracing simulation for the
  entire frame. Target positions are evaluated once. This is the fastest
  option and is suitable for static or slowly moving targets.
- ``"pulse"`` — Perform ray-tracing **for each pulse**. Target positions are
  updated at each pulse time, capturing intra-frame motion. This provides a
  good balance between accuracy and speed for moving targets.
- ``"sample"`` — Perform ray-tracing **for each sample**. Target positions
  are updated at every ADC sample time. This is the highest-fidelity mode and
  captures rapid target dynamics, as well as non-linear motion. For
  micro-Doppler effect simulation, ``level="sample"`` is recommended.
  This mode is significantly more computationally expensive.

The choice of level affects both simulation accuracy and runtime. For most
scenarios involving moving targets, ``"pulse"`` level is recommended.

``ray_filter``
~~~~~~~~~~~~~~

:Type: ``list`` or ``None``
:Default: ``None``

Filters rays based on their **number of reflections** (bounces). When set,
only rays whose reflection count falls within the range
``[ray_filter[0], ray_filter[1]]`` are included in the baseband calculation.

- ``ray_filter[0]`` — Minimum number of reflections (inclusive).
- ``ray_filter[1]`` — Maximum number of reflections (inclusive).

When ``None``, no filtering is applied and all rays from the minimum default
to the maximum allowed reflections are included.

This is useful for isolating specific scattering mechanisms. For example:

- ``ray_filter=[1, 1]`` — Include only single-bounce (direct) reflections.
- ``ray_filter=[2, 3]`` — Include only double- and triple-bounce reflections.

``back_propagating``
~~~~~~~~~~~~~~~~~~~~

:Type: ``bool``
:Default: ``False``

Enables **backward ray propagation** analysis. When set to ``True``, the
simulator performs an additional backtracing pass after the forward
ray-tracing stage.

In the forward pass, rays are traced from the transmitter to targets. In the
backtracing pass, rays at their final hit points are traced back towards the
radar to check for additional multi-bounce paths that scatter energy back to
the receiver through intermediate reflections.

This is important for capturing indirect scattering paths in scenes with
multiple reflections, such as inside a tunnel where rays bounce between
walls, ceiling, and floor before returning to the radar.

.. note::

   Enabling back-propagation increases computation time since an additional
   ray-tracing pass and baseband calculation are performed for each snapshot.


Target Flags
------------

The following flags can be set in the target dictionary for 3D mesh targets
to modify how the simulator handles them.

``skip_diffusion``
~~~~~~~~~~~~~~~~~~

:Type: ``bool``
:Default: ``False``

When set to ``True``, the simulator **skips the diffusion (diffraction)
calculation** for this target.

This flag is primarily intended for **large flat reflectors** such as ground
planes, walls, or other surfaces where diffusion effects are negligible
compared to specular reflection. Enabling it reduces the computational load
for these targets.

``environment``
~~~~~~~~~~~~~~~

:Type: ``bool``
:Default: ``False``

Marks a target as an **environment object** rather than a primary scattering
target. Environment objects define the physical surroundings of the scene
(e.g., a ground plane, building wall, or terrain surface) and participate
in multi-bounce reflections, but are not the primary radar targets of
interest.

Environment objects such as ground planes or building walls typically have
very large meshes. Without this flag, the simulator would allocate a large
number of rays to these surfaces based on their size, which is
computationally wasteful. Setting ``environment=True`` reduces the ray
density for these objects, significantly improving simulation efficiency
while still allowing them to participate in multi-bounce reflections.
