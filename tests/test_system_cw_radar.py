"""
System level test for raytracing-based scene simulation

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
from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import simc  # pylint: disable=no-name-in-module
from radarsimpy.rt import scene  # pylint: disable=no-name-in-module


def test_sim_cw():
    """
    Test the CW radar simulator.
    """
    tx = Transmitter(
        f=24.125e9, t=20, tx_power=10, pulses=1, channels=[{"location": (0, 0, 0)}]
    )
    rx = Receiver(
        fs=0.5,
        noise_figure=12,
        rf_gain=20,
        baseband_gain=50,
        load_resistor=1000,
        channels=[{"location": (0, 0, 0)}],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    target = {
        "location": (
            1.4 + 1e-3 * np.sin(2 * np.pi * 0.1 * radar.time_prop["timestamp"]),
            0,
            0,
        ),
        "rcs": -10,
        "phase": 0,
    }
    targets = [target]

    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.28017324 + 0.57151868j,
                        -0.62818491 + 0.09703288j,
                        -0.55171423 + 0.31632635j,
                        0.08803159 + 0.63092172j,
                        0.30893151 + 0.55748964j,
                        -0.28016539 + 0.57152253j,
                        -0.62820815 + 0.09688231j,
                        -0.55177488 + 0.31622055j,
                        0.08815256 + 0.63090483j,
                        0.3088246 + 0.55754887j,
                    ]
                ]
            ]
        ),
        atol=1e-05,
    )

    assert np.allclose(
        result["timestamp"],
        np.array([[[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]]]),
    )


def test_sim_cw_raytracing():
    """
    Test the CW radar simulator.
    """
    tx = Transmitter(
        f=24.125e9, t=20, tx_power=10, pulses=1, channels=[{"location": (0, 0, 0)}]
    )
    rx = Receiver(
        fs=0.5,
        noise_figure=12,
        rf_gain=20,
        baseband_gain=50,
        load_resistor=1000,
        channels=[{"location": (0, 0, 0)}],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    target = {
        "model": "./models/cr.stl",
        "location": (
            1.4 + 1e-3 * np.sin(2 * np.pi * 0.1 * radar.time_prop["timestamp"]),
            0,
            0,
        ),
    }
    targets = [target]

    result = scene(radar, targets, density=1, level="sample", noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        2.1923784 + 2.21549872j,
                        -0.5635882 + 3.06396141j,
                        0.57478124 + 3.06246209j,
                        3.05812674 + 0.60727408j,
                        3.0727276 - 0.53192075j,
                        2.19233565 + 2.21551733j,
                        -0.56361637 + 3.06393772j,
                        0.57475453 + 3.06248653j,
                        3.0581141 + 0.60719965j,
                        3.07272899 - 0.53186788j,
                    ]
                ]
            ]
        ),
        atol=1e-05,
    )

    assert np.allclose(
        result["timestamp"],
        np.array([[[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]]]),
    )
