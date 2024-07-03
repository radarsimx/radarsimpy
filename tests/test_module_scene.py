"""
A Python module for radar simulation

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
from radarsimpy.rt import scene  # pylint: disable=no-name-in-module


def test_scene_single_target():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": (10, 0, 0),
            "speed": (0, 0, 0),
            "rotation_rate": (0, 0, 0),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.0487828 + 0.00210325j,
                        -0.03809988 + 0.02922872j,
                        0.01007497 - 0.04623383j,
                    ],
                    [
                        -0.03682247 - 0.0333487j,
                        0.0487828 + 0.00210325j,
                        -0.03809988 + 0.02922872j,
                        0.01007497 - 0.04623383j,
                    ],
                    [
                        -0.03682247 - 0.0333487j,
                        0.0487828 + 0.00210325j,
                        -0.03809988 + 0.02922872j,
                        0.01007497 - 0.04623383j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_varing_prp():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=[100e-6, 110e-6, 130e-6],
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": (10, 0, 0),
            "speed": (-10, 0, 0),
            "rotation_rate": (0, 0, 0),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.04842967 - 0.00610054j,
                        -0.02626628 + 0.04016538j,
                        -0.0136109 - 0.04528237j,
                    ],
                    [
                        -0.0461478 + 0.0181102j,
                        0.01596997 - 0.04601663j,
                        0.02437224 + 0.04123385j,
                        -0.04657203 - 0.00772459j,
                    ],
                    [
                        0.00567273 + 0.04912324j,
                        -0.04034068 - 0.02708163j,
                        0.04594947 - 0.01313626j,
                        -0.01912976 + 0.04306489j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.10000000e-04, 1.26666667e-04, 1.43333333e-04, 1.60000000e-04],
                    [2.40000000e-04, 2.56666667e-04, 2.73333333e-04, 2.90000000e-04],
                ]
            ]
        ),
    )


def test_scene_tx_delay():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
                "delay": 10e-6,
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": (10, 0, 0),
            "speed": (10, 0, 0),
            "rotation_rate": (0, 0, 0),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.04775567 + 0.01025273j,
                        -0.04565249 + 0.01498431j,
                        0.03126375 - 0.03556718j,
                    ],
                    [
                        0.00862125 - 0.0490222j,
                        0.01673449 + 0.04598848j,
                        -0.03698449 - 0.03080911j,
                        0.04679602 + 0.00771201j,
                    ],
                    [
                        0.04616583 - 0.01885925j,
                        -0.03011282 + 0.03869784j,
                        0.00651008 - 0.04778263j,
                        0.01825622 + 0.04385368j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [1.00000000e-05, 2.66666667e-05, 4.33333333e-05, 6.00000000e-05],
                    [1.10000000e-04, 1.26666667e-04, 1.43333333e-04, 1.60000000e-04],
                    [2.10000000e-04, 2.26666667e-04, 2.43333333e-04, 2.60000000e-04],
                ]
            ]
        ),
    )


def test_scene_tx_offset():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (5, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.11497509 + 0.06548127j,
                        0.09333607 + 0.09305221j,
                        0.06546423 + 0.11384928j,
                        0.03336043 + 0.12663272j,
                    ],
                    [
                        0.11497509 + 0.06548127j,
                        0.09333607 + 0.09305221j,
                        0.06546423 + 0.11384928j,
                        0.03336043 + 0.12663272j,
                    ],
                    [
                        0.11497509 + 0.06548127j,
                        0.09333607 + 0.09305221j,
                        0.06546423 + 0.11384928j,
                        0.03336043 + 0.12663272j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_rx_offset():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (5, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00975333 + 0.04633475j,
                        -0.01028245 + 0.04103925j,
                        -0.02399137 + 0.02663358j,
                        -0.02706399 + 0.00815449j,
                    ],
                    [
                        0.00975333 + 0.04633475j,
                        -0.01028245 + 0.04103925j,
                        -0.02399137 + 0.02663358j,
                        -0.02706399 + 0.00815449j,
                    ],
                    [
                        0.00975333 + 0.04633475j,
                        -0.01028245 + 0.04103925j,
                        -0.02399137 + 0.02663358j,
                        -0.02706399 + 0.00815449j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_multiple_targets():
    """
    Test with multiple targets.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 10, 0]),
        },
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, -10, 0]),
        },
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -4.33301396e-04 + 0.00023905j,
                        3.28193149e-04 - 0.00026169j,
                        3.43735356e-04 - 0.00052881j,
                        -7.74455443e-05 + 0.00015481j,
                    ],
                    [
                        -4.33301396e-04 + 0.00023905j,
                        3.28193149e-04 - 0.00026169j,
                        3.43735356e-04 - 0.00052881j,
                        -7.74455443e-05 + 0.00015481j,
                    ],
                    [
                        -4.33301396e-04 + 0.00023905j,
                        3.28193149e-04 - 0.00026169j,
                        3.43735356e-04 - 0.00052881j,
                        -7.74455443e-05 + 0.00015481j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_single_target_speed():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([-10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.04842967 - 0.00610054j,
                        -0.02626628 + 0.04016538j,
                        -0.0136109 - 0.04528237j,
                    ],
                    [
                        -0.04774657 + 0.01337209j,
                        0.02053424 - 0.04417949j,
                        0.02008607 + 0.04349243j,
                        -0.04555844 - 0.01239713j,
                    ],
                    [
                        -0.01408809 + 0.04744038j,
                        -0.02645933 - 0.04079583j,
                        0.04744233 + 0.00602393j,
                        -0.03457864 + 0.03205154j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_radar_location():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx, location=[5, 0, 0])

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.09150579 - 0.03280507j,
                        0.00038708 + 0.09759141j,
                        0.09141052 - 0.03330407j,
                        -0.06232657 - 0.0736831j,
                    ],
                    [
                        -0.09150579 - 0.03280507j,
                        0.00038708 + 0.09759141j,
                        0.09141052 - 0.03330407j,
                        -0.06232657 - 0.0736831j,
                    ],
                    [
                        -0.09150579 - 0.03280507j,
                        0.00038708 + 0.09759141j,
                        0.09141052 - 0.03330407j,
                        -0.06232657 - 0.0736831j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_radar_moving():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx, speed=[10, 0, 0])

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.04842967 - 0.00610054j,
                        -0.02626628 + 0.04016538j,
                        -0.0136109 - 0.04528237j,
                    ],
                    [
                        -0.04774657 + 0.01337209j,
                        0.02053424 - 0.04417949j,
                        0.02008607 + 0.04349243j,
                        -0.04555844 - 0.01239713j,
                    ],
                    [
                        -0.01408809 + 0.04744038j,
                        -0.02645933 - 0.04079583j,
                        0.04744233 + 0.00602393j,
                        -0.03457864 + 0.03205154j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_2_frames_moving_target():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx, time=[0, 1])

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([-5, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.04877901 - 0.00200641j,
                        -0.03264436 + 0.03519788j,
                        -0.0018307 - 0.04726577j,
                    ],
                    [
                        -0.04830821 - 0.01138446j,
                        0.04167432 - 0.02533914j,
                        -0.01148895 + 0.04656663j,
                        -0.02451444 - 0.04041226j,
                    ],
                    [
                        -0.04774657 + 0.01337209j,
                        0.02418099 - 0.04230276j,
                        0.0125051 + 0.04626023j,
                        -0.04101305 - 0.02342634j,
                    ],
                ],
                [
                    [
                        -0.09457635 - 0.03042372j,
                        0.0120404 + 0.0999181j,
                        0.08722919 - 0.05291579j,
                        -0.0848036 - 0.05888089j,
                    ],
                    [
                        -0.09761531 + 0.01914424j,
                        0.05894103 + 0.08174985j,
                        0.05085144 - 0.0886054j,
                        -0.10279241 - 0.01059069j,
                    ],
                    [
                        -0.07627671 + 0.06405112j,
                        0.09124954 + 0.04311862j,
                        0.00169646 - 0.10227992j,
                        -0.09520085 + 0.04041931j,
                    ],
                ],
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ],
                [
                    [1.00000000e00, 1.00001667e00, 1.00003333e00, 1.00005000e00],
                    [1.00010000e00, 1.00011667e00, 1.00013333e00, 1.00015000e00],
                    [1.00020000e00, 1.00021667e00, 1.00023333e00, 1.00025000e00],
                ],
            ]
        ),
    )


def test_scene_2_frames_moving_radar():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx, speed=[5, 0, 0], time=[0, 1])

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([0, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.04877901 - 0.00200641j,
                        -0.03264436 + 0.03519788j,
                        -0.0018307 - 0.04726577j,
                    ],
                    [
                        -0.04830821 - 0.01138446j,
                        0.04167432 - 0.02533914j,
                        -0.01148895 + 0.04656663j,
                        -0.02451444 - 0.04041226j,
                    ],
                    [
                        -0.04774657 + 0.01337209j,
                        0.02418099 - 0.04230276j,
                        0.0125051 + 0.04626023j,
                        -0.04101305 - 0.02342634j,
                    ],
                ],
                [
                    [
                        -0.09150579 - 0.03280507j,
                        0.00859095 + 0.09721411j,
                        0.08450881 - 0.04815913j,
                        -0.07879691 - 0.05567319j,
                    ],
                    [
                        -0.09599106 + 0.0155922j,
                        0.05455584 + 0.08092158j,
                        0.0505607 - 0.08301941j,
                        -0.09587046 - 0.01037787j,
                    ],
                    [
                        -0.07648733 + 0.06012592j,
                        0.08689346 + 0.04442122j,
                        0.00399634 - 0.09705506j,
                        -0.08879757 + 0.03749432j,
                    ],
                ],
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ],
                [
                    [1.00000000e00, 1.00001667e00, 1.00003333e00, 1.00005000e00],
                    [1.00010000e00, 1.00011667e00, 1.00013333e00, 1.00015000e00],
                    [1.00020000e00, 1.00021667e00, 1.00023333e00, 1.00025000e00],
                ],
            ]
        ),
    )


def test_scene_tx_az_pattern():
    """
    Basic test case with a single target and simple radar setup.
    """
    az_angle = np.array([-46, 0, 46])
    az_pattern = np.array([-10, -10, 10])
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
                "azimuth_angle": az_angle,
                "azimuth_pattern": az_pattern,
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, 10, 0]),
            "rotation": [45, 0, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00428461 + 0.00488591j,
                        0.00523453 + 0.00385215j,
                        0.00594037 + 0.00263836j,
                        0.00636913 + 0.00130113j,
                    ],
                    [
                        0.00428461 + 0.00488591j,
                        0.00523453 + 0.00385215j,
                        0.00594037 + 0.00263836j,
                        0.00636913 + 0.00130113j,
                    ],
                    [
                        0.00428461 + 0.00488591j,
                        0.00523453 + 0.00385215j,
                        0.00594037 + 0.00263836j,
                        0.00636913 + 0.00130113j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, -10, 0]),
            "rotation": [-45, 0, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00042846 + 0.00048859j,
                        0.00052345 + 0.00038521j,
                        0.00059404 + 0.00026384j,
                        0.00063691 + 0.00013011j,
                    ],
                    [
                        0.00042846 + 0.00048859j,
                        0.00052345 + 0.00038521j,
                        0.00059404 + 0.00026384j,
                        0.00063691 + 0.00013011j,
                    ],
                    [
                        0.00042846 + 0.00048859j,
                        0.00052345 + 0.00038521j,
                        0.00059404 + 0.00026384j,
                        0.00063691 + 0.00013011j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_rx_az_pattern():
    """
    Basic test case with a single target and simple radar setup.
    """
    az_angle = np.array([-46, 0, 46])
    az_pattern = np.array([-10, -10, 10])
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
                "azimuth_angle": az_angle,
                "azimuth_pattern": az_pattern,
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, 10, 0]),
            "rotation": [45, 0, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00428461 + 0.00488591j,
                        0.00523453 + 0.00385215j,
                        0.00594037 + 0.00263836j,
                        0.00636913 + 0.00130113j,
                    ],
                    [
                        0.00428461 + 0.00488591j,
                        0.00523453 + 0.00385215j,
                        0.00594037 + 0.00263836j,
                        0.00636913 + 0.00130113j,
                    ],
                    [
                        0.00428461 + 0.00488591j,
                        0.00523453 + 0.00385215j,
                        0.00594037 + 0.00263836j,
                        0.00636913 + 0.00130113j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, -10, 0]),
            "rotation": [-45, 0, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00042846 + 0.00048859j,
                        0.00052345 + 0.00038521j,
                        0.00059404 + 0.00026384j,
                        0.00063691 + 0.00013011j,
                    ],
                    [
                        0.00042846 + 0.00048859j,
                        0.00052345 + 0.00038521j,
                        0.00059404 + 0.00026384j,
                        0.00063691 + 0.00013011j,
                    ],
                    [
                        0.00042846 + 0.00048859j,
                        0.00052345 + 0.00038521j,
                        0.00059404 + 0.00026384j,
                        0.00063691 + 0.00013011j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_tx_el_pattern():
    """
    Basic test case with a single target and simple radar setup.
    """
    el_angle = np.array([-46, 0, 46])
    el_pattern = np.array([-10, 10, 10])
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=20,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
                "elevation_angle": el_angle,
                "elevation_pattern": el_pattern,
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=30,
        load_resistor=1000,
        baseband_gain=40,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, 0, 10]),
            "rotation": [0, 45, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.03199762 + 0.03529157j,
                        0.03882344 + 0.02757869j,
                        0.04382921 + 0.01858206j,
                        0.04678241 + 0.00872352j,
                    ],
                    [
                        0.03199762 + 0.03529157j,
                        0.03882344 + 0.02757869j,
                        0.04382921 + 0.01858206j,
                        0.04678241 + 0.00872352j,
                    ],
                    [
                        0.03199762 + 0.03529157j,
                        0.03882344 + 0.02757869j,
                        0.04382921 + 0.01858206j,
                        0.04678241 + 0.00872352j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, 0, -10]),
            "rotation": [0, -45, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00296381 + 0.003689j,
                        0.0036944 + 0.00296905j,
                        0.00425376 + 0.00210706j,
                        0.00461489 + 0.00114307j,
                    ],
                    [
                        0.00296381 + 0.003689j,
                        0.0036944 + 0.00296905j,
                        0.00425376 + 0.00210706j,
                        0.00461489 + 0.00114307j,
                    ],
                    [
                        0.00296381 + 0.003689j,
                        0.0036944 + 0.00296905j,
                        0.00425376 + 0.00210706j,
                        0.00461489 + 0.00114307j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_rx_el_pattern():
    """
    Basic test case with a single target and simple radar setup.
    """
    el_angle = np.array([-46, 0, 46])
    el_pattern = np.array([-10, 10, 10])
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=20,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=30,
        load_resistor=1000,
        baseband_gain=40,
        channels=[
            {
                "location": (0, 0, 0),
                "elevation_angle": el_angle,
                "elevation_pattern": el_pattern,
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, 0, 10]),
            "rotation": [0, 45, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.03199762 + 0.03529157j,
                        0.03882344 + 0.02757869j,
                        0.04382921 + 0.01858206j,
                        0.04678241 + 0.00872352j,
                    ],
                    [
                        0.03199762 + 0.03529157j,
                        0.03882344 + 0.02757869j,
                        0.04382921 + 0.01858206j,
                        0.04678241 + 0.00872352j,
                    ],
                    [
                        0.03199762 + 0.03529157j,
                        0.03882344 + 0.02757869j,
                        0.04382921 + 0.01858206j,
                        0.04678241 + 0.00872352j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )

    targets = [
        {
            "model": "./models/cr.stl",
            "location": np.array([10, 0, -10]),
            "rotation": [0, -45, 0],
        }
    ]
    result = scene(radar, targets, density=1, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00296381 + 0.003689j,
                        0.0036944 + 0.00296905j,
                        0.00425376 + 0.00210706j,
                        0.00461489 + 0.00114307j,
                    ],
                    [
                        0.00296381 + 0.003689j,
                        0.0036944 + 0.00296905j,
                        0.00425376 + 0.00210706j,
                        0.00461489 + 0.00114307j,
                    ],
                    [
                        0.00296381 + 0.003689j,
                        0.0036944 + 0.00296905j,
                        0.00425376 + 0.00210706j,
                        0.00461489 + 0.00114307j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_freq_offset():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        f_offset=[0, 1e6, 2e6],
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.0487828 + 0.00210325j,
                        -0.03809988 + 0.02922872j,
                        0.01007497 - 0.04623383j,
                    ],
                    [
                        -0.02004394 - 0.04541358j,
                        0.04367325 + 0.02175098j,
                        -0.04665996 + 0.01120219j,
                        0.02798143 - 0.03812524j,
                    ],
                    [
                        0.00017622 - 0.04960085j,
                        0.0310226 + 0.03760778j,
                        -0.04715005 - 0.0087329j,
                        0.04103874 - 0.02344732j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_pulse_modulation():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
                "pulse_amp": (0, 1, 2),
                "pulse_phs": (0, 180, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [
                        0.03682247 + 0.0333487j,
                        -0.0487828 - 0.00210325j,
                        0.03809988 - 0.02922872j,
                        -0.01007497 + 0.04623383j,
                    ],
                    [
                        -0.07364494 - 0.0666974j,
                        0.0975656 + 0.00420649j,
                        -0.07619976 + 0.05845745j,
                        0.02014995 - 0.09246767j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_waveform_modulation():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
                "mod_t": (0, 10e-6, 20e-6, 30e-6, 40e-6),
                "amp": (0, 1, 0, 3, 4),
                "phs": (0, 90, 180, -90, -180),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.0333487 - 0.03682247j,
                        0.0 + 0.0j,
                        0.15239952 - 0.1169149j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0333487 - 0.03682247j,
                        0.0 + 0.0j,
                        0.15239952 - 0.1169149j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.0333487 - 0.03682247j,
                        0.0 + 0.0j,
                        0.15239952 - 0.1169149j,
                        0.0 + 0.0j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


def test_scene_arbitrary_waveform():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9, 26e9, 28e9, 26e9],
        t=[0, 20e-6, 40e-6, 60e-6, 80e-6],
        tx_power=10,
        prp=100e-6,
        pulses=3,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e4,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = scene(radar, targets, density=0.4, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.04171989 - 0.02771336j,
                        -0.03233525 + 0.03985125j,
                        0.03863578 - 0.02269049j,
                        -0.04420436 - 0.00311879j,
                    ],
                    [
                        -0.04171989 - 0.02771336j,
                        -0.03233525 + 0.03985125j,
                        0.03863578 - 0.02269049j,
                        -0.04420436 - 0.00311879j,
                    ],
                    [
                        -0.04171989 - 0.02771336j,
                        -0.03233525 + 0.03985125j,
                        0.03863578 - 0.02269049j,
                        -0.04420436 - 0.00311879j,
                    ],
                ]
            ]
        ),
    )

    assert np.allclose(
        result["timestamp"],
        np.array(
            [
                [
                    [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
                    [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
                    [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
                ]
            ]
        ),
    )


# def test_scene_interference():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
#         t=80e-6,
#         tx_power=10,
#         prp=100e-6,
#         pulses=1,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#             }
#         ],
#     )
#     rx = Receiver(
#         fs=6e5,
#         noise_figure=12,
#         rf_gain=20,
#         load_resistor=500,
#         baseband_gain=30,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#             }
#         ],
#     )
#     interference_tx = Transmitter(
#         f=[24.175e9, 24.075e9],
#         t=80e-6,
#         tx_power=10,
#         prp=100e-6,
#         pulses=3,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#             }
#         ],
#     )
#     interference_radar = Radar(
#         transmitter=interference_tx,
#         receiver=rx,
#         location=(20, 0, 0),
#         rotation=(180, 0, 0),
#     )

#     radar = Radar(transmitter=tx, receiver=rx, interf=interference_radar)

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["interference"],
#         np.array(
#             [
#                 [
#                     [
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         -0.01325275 + 0.00434837j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                         0.0 + 0.0j,
#                     ]
#                 ]
#             ]
#         ),
#     )
