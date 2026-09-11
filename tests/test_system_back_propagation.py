"""
Properties of back propagation and the reflection filter in the mesh simulator

Back propagation is covered today by one test holding 160 hardcoded baseband
values on a single scene, and ``ray_filter`` is not covered at all. Neither says
anything a reader can check against the physics, so neither survives a redesign
of the pass: regenerating the numbers makes any change look correct.

These tests assert properties instead. Each one states something that has to be
true of the simulator whatever the implementation does, so they keep their
meaning across a rewrite and can be read as the specification the rewrite has to
meet.

---

- Copyright (C) 2018 - PRESENT  radarsimx.com
- E-mail: info@radarsimx.com
- Website: https://radarsimx.com

::

    ██████╗  █████╗ ██████╗  █████╗ ██████╗ ███████╗██╗███╗   ███╗██╗  ██╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██║████╗ ████║╚██╗██╔╝
    ██████╔╝███████║██║  ██║███████║██████╔╝███████╗██║██╔████╔██║ ╚███╔╝
    ██╔══██╗██╔══██║██║  ██║██╔══██║██╔══██╗╚════██║██║██║╚██╔╝██║ ██╔██╗
    ██║  ██║██║  ██║██████╔╝██║  ██║██║  ██║███████║██║██║ ╚═╝ ██║██╔╝ ██╗
    ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝

"""

import numpy as np
import pytest

from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module

pytestmark = pytest.mark.mesh

#: Kept low throughout: these tests assert relationships between runs, not
#: converged absolute levels, and every one of them runs the simulator at least
#: twice.
DENSITY = 0.5

#: Bit-identity is a CPU property. The GPU path accumulates the baseband with
#: `atomicAdd`, so summation order varies between runs and two identical calls
#: differ at around 1e-16 relative -- measured, not assumed. Tests that compare
#: runs exactly therefore pin the device; tests that compare magnitudes do not.
EXACT = {"device": "cpu"}


def _plate(centre, u_axis, v_axis, half=2.0):
    """A square plate as an in-memory mesh, spanned by two unit axes.

    Built here rather than loaded from ``models/`` so the geometry is exact and
    the path lengths below can be derived by hand rather than measured.
    """
    centre = np.asarray(centre, dtype=float)
    u_vec = np.asarray(u_axis, dtype=float) * half
    v_vec = np.asarray(v_axis, dtype=float) * half
    return {
        "points": np.array(
            [
                centre - u_vec - v_vec,
                centre + u_vec - v_vec,
                centre + u_vec + v_vec,
                centre - u_vec + v_vec,
            ]
        ),
        "cells": np.array([[0, 1, 2], [0, 2, 3]]),
    }


# ---------------------------------------------------------------------------
# 1. Back propagation is a no-op where there is nothing to bounce off
# ---------------------------------------------------------------------------


def test_back_propagation_changes_nothing_without_multipath(
    make_radar, model_path, mesh_module
):
    """One plate at normal incidence has no second bounce, so the flag is inert.

    The cheapest statement of what back propagation is for: it may only ever
    *add* returns that leave a scattering point and bounce again. A lone plate
    facing the radar offers nothing to bounce off, so turning the flag on has to
    leave the baseband untouched -- bit for bit, not merely close, because the
    forward pass runs identically in both cases.
    """
    del mesh_module  # fixture is a skip guard

    radar = make_radar()
    targets = [{"model": model_path("plate5x5.stl"), "location": (20, 0, 0)}]

    off = sim_radar(radar, targets, density=DENSITY, back_propagating=False, **EXACT)
    on = sim_radar(radar, targets, density=DENSITY, back_propagating=True, **EXACT)

    assert np.any(off["baseband"] != 0.0), "scene returned nothing; test is vacuous"
    assert np.array_equal(off["baseband"], on["baseband"])


# ---------------------------------------------------------------------------
# 2. What back propagation adds lands at the right range
# ---------------------------------------------------------------------------


def test_back_propagated_return_lands_at_its_true_path_length(
    make_radar, mesh_module
):
    """The added return must sit at the length of the path it stands for.

    Two plates, each turning the ray 45 degrees, so a ray goes
    ``Tx -> A -> B`` and then leaves upwards and never comes back. The only way
    that geometry returns energy is by scattering at B and going back down the
    chain it arrived on::

        Tx (0,0,0) --30m--> A (30,0,0) --20m--> B (30,20,0) --> up and away

    so the path is 30 + 20 + 20 + 30 = 100 m, i.e. a range of 50 m. Measuring
    the *first* hit's distance to the receiver instead -- the natural mistake,
    and one this simulator has made -- puts it at 40 m. 200 MHz of bandwidth
    separates those by more than ten range bins.

    The parameters are sized for cost, which is rays times samples: 24 GHz
    rather than 77 GHz keeps the ray count down, since density is per
    wavelength, and 200 samples cover the 50 m with room to spare. The 77 GHz /
    2 GHz / 2000-sample version gave the same answer in six minutes on CPU.
    """
    del mesh_module  # fixture is a skip guard

    root2 = np.sqrt(2.0)
    plate_a = _plate((30.0, 0.0, 0.0), (1 / root2, 1 / root2, 0.0), (0.0, 0.0, 1.0))
    plate_b = _plate((30.0, 20.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1 / root2, 1 / root2))
    targets = [
        {"model": plate_a, "location": (0, 0, 0)},
        {"model": plate_b, "location": (0, 0, 0)},
    ]

    samples, pulse_length = 200, 20e-6
    radar = make_radar(
        tx_kwargs={"f": [24e9, 24.2e9], "t": pulse_length, "tx_power": 20},
        rx_kwargs={"fs": (samples + 0.5) / pulse_length, "baseband_gain": 60},
    )

    off = sim_radar(radar, targets, density=1.0, back_propagating=False)
    on = sim_radar(radar, targets, density=1.0, back_propagating=True)

    added = np.abs(np.fft.fft(on["baseband"][0, 0, :] - off["baseband"][0, 0, :]))
    assert added.max() > 0.0, "back propagation added nothing; test is vacuous"

    bandwidth = radar.radar_prop["transmitter"].waveform_prop["bandwidth"]
    n_bins = added.size
    bin_metres = (
        3e8 / 2 / bandwidth * radar.sample_prop["samples_per_pulse"] / n_bins
    )
    peak_range = float(np.argmax(added)) * bin_metres

    assert peak_range == pytest.approx(50.0, abs=2.5)


# ---------------------------------------------------------------------------
# 3. The skip_diffusion promise
# ---------------------------------------------------------------------------


def test_a_scene_of_only_skipped_surfaces_returns_nothing(
    make_radar, model_path, mesh_module
):
    """Documented: energy comes back only once a *non*-skipped surface is touched.

    From the user guide: "A ray sends energy back to the receiver only once it
    has touched a surface that is **not** skipped. Bounces on skipped surfaces
    before that point redirect the ray and do nothing else." A scene containing
    nothing but skipped surfaces therefore has to return silence, however many
    times a ray bounces around inside it.

    Worth knowing *where* that holds. The back-tracing pass does not read the
    flag at all, so it is not what upholds this -- the occupancy stage is:
    `_Core_CheckGridOccupancy` launches no rays into a direction whose probe
    finds only skipped surfaces. The promise survives because the rays that
    would expose the gap are never created. Any redesign that moves or relaxes
    that occupancy rejection has to take this over.
    """
    del mesh_module  # fixture is a skip guard

    radar = make_radar()
    targets = [
        {
            "model": model_path("surface_400x400.stl"),
            "location": (0, 0, -1.0),
            "skip_diffusion": True,
        },
        {
            "model": model_path("plate5x5.stl"),
            "location": (20, 0, 0),
            "skip_diffusion": True,
        },
    ]

    result = sim_radar(radar, targets, density=DENSITY, back_propagating=True, **EXACT)

    assert np.all(result["baseband"] == 0.0)


# ---------------------------------------------------------------------------
# 4. ray_filter partitions the return
# ---------------------------------------------------------------------------


def test_reflection_filter_partitions_the_whole_return(
    make_radar, model_path, mesh_module
):
    """Every bounce order, summed, must reconstruct the unfiltered run.

    ``ray_filter`` selects a band of reflection counts, so the bands have to
    tile: running each one alone and adding the results reproduces the whole.
    This is what makes the filter usable for decomposing a scene by scattering
    mechanism, which is what the documentation offers it for.

    Not bit-exact -- the summation order differs from the simulator's own -- so
    this compares against the amplitude of the total.
    """
    del mesh_module  # fixture is a skip guard

    radar = make_radar()
    targets = [{"model": model_path("cr.stl"), "location": (10, 0, 0)}]

    whole = sim_radar(radar, targets, density=DENSITY)["baseband"]
    assert np.any(whole != 0.0), "scene returned nothing; test is vacuous"

    parts = sum(
        sim_radar(radar, targets, density=DENSITY, ray_filter=[k, k])["baseband"]
        for k in range(0, 11)
    )

    assert np.max(np.abs(parts - whole)) < 1e-6 * np.max(np.abs(whole))


def test_reflection_filter_edges(make_radar, model_path, mesh_module):
    """``None`` is the full band, and an inverted band selects nothing."""
    del mesh_module  # fixture is a skip guard

    radar = make_radar()
    targets = [{"model": model_path("cr.stl"), "location": (10, 0, 0)}]

    default = sim_radar(radar, targets, density=DENSITY, ray_filter=None, **EXACT)
    explicit = sim_radar(radar, targets, density=DENSITY, ray_filter=[0, 10], **EXACT)
    inverted = sim_radar(radar, targets, density=DENSITY, ray_filter=[5, 3], **EXACT)

    assert np.array_equal(default["baseband"], explicit["baseband"])
    assert np.all(inverted["baseband"] == 0.0)


def test_reflection_filter_counts_a_back_propagated_path_in_full(
    make_radar, model_path, mesh_module
):
    """A path is as long as the path, not as long as its way in.

    A back-propagated return traverses the chain on the way out as well as the
    way in, so it has more bounces than the hit it scattered at. Filtering to a
    band that excludes the true count must therefore drop it.

    This was the specification for the redesign: the entry used to be filtered
    on its incident index alone, so it survived a band it did not belong to and
    enabling back propagation changed a run that had asked only for short paths.
    """
    del mesh_module  # fixture is a skip guard

    radar = make_radar()
    targets = [{"model": model_path("cr.stl"), "location": (10, 0, 0)}]

    off = sim_radar(radar, targets, density=DENSITY, ray_filter=[1, 2], **EXACT)
    on = sim_radar(
        radar, targets, density=DENSITY, ray_filter=[1, 2],
        back_propagating=True, **EXACT
    )

    # Every back-propagated path through a trihedral is at least three bounces,
    # so none of them belongs in a [1, 2] band.
    assert np.array_equal(off["baseband"], on["baseband"])


# ---------------------------------------------------------------------------
# 5. Robustness and reproducibility
# ---------------------------------------------------------------------------


def test_long_bounce_chains_stay_finite(make_radar, mesh_module):
    """Two plates facing each other drive the reflection count to its ceiling.

    Rays entering the gap bounce until the trace depth stops them, which is the
    regime where the lookup table is fullest and where grazing footprints go
    degenerate. Nothing here should produce a NaN or run off the end of a buffer.
    """
    del mesh_module  # fixture is a skip guard

    left = _plate((10.0, -1.5, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), half=3.0)
    right = _plate((10.0, 1.5, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), half=3.0)
    targets = [
        {"model": left, "location": (0, 0, 0)},
        {"model": right, "location": (0, 0, 0)},
    ]

    radar = make_radar()
    result = sim_radar(radar, targets, density=1.0, back_propagating=True)

    assert np.all(np.isfinite(result["baseband"]))


def test_the_same_scene_twice_gives_the_same_answer(
    make_radar, model_path, mesh_module
):
    """Reproducibility, with back propagation in the picture.

    The lookup table's order fixes the order of every later summation, so a pass
    that appends to it under a shared counter would drift between runs. The
    forward pass is built to avoid that; this asserts the property end to end
    with the back pass enabled too.
    """
    del mesh_module  # fixture is a skip guard

    radar = make_radar()
    targets = [{"model": model_path("cr.stl"), "location": (10, 0, 0)}]

    first = sim_radar(radar, targets, density=DENSITY, back_propagating=True, **EXACT)
    second = sim_radar(radar, targets, density=DENSITY, back_propagating=True, **EXACT)

    assert np.any(first["baseband"] != 0.0), "scene returned nothing; test is vacuous"
    assert np.array_equal(first["baseband"], second["baseband"])
