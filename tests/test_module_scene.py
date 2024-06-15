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
            "model": "../models/plate5x5.stl",
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
            "model": "../models/plate5x5.stl",
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


# def test_scene_radar_moving():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
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
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx, speed=[10, 0, 0])

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         0.02167872 + 0.01755585j,
#                         -0.02744623 + 0.00499312j,
#                         0.01410064 - 0.02407178j,
#                         0.00905038 + 0.02638979j,
#                     ],
#                     [
#                         0.02640622 - 0.00901099j,
#                         -0.01038519 + 0.0258976j,
#                         -0.01289609 - 0.0247443j,
#                         0.02718049 + 0.00631372j,
#                     ],
#                     [
#                         0.00645317 - 0.02715058j,
#                         0.01642037 + 0.02256592j,
#                         -0.02782165 - 0.00220395j,
#                         0.01978305 - 0.01968716j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_2_frames_moving_target():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
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
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx, time=[0, 1])

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "speed": np.array([-5, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         0.02167872 + 0.01755585j,
#                         -0.02776995 + 0.00265147j,
#                         0.01794177 - 0.02136164j,
#                         0.00216204 + 0.02781327j,
#                     ],
#                     [
#                         0.02746562 + 0.00489594j,
#                         -0.02301209 + 0.01577339j,
#                         0.00536318 - 0.02737916j,
#                         0.01536405 + 0.02328849j,
#                     ],
#                     [
#                         0.02640622 - 0.00901099j,
#                         -0.01251624 + 0.02493704j,
#                         -0.00855785 - 0.0265575j,
#                         0.02473615 + 0.01291073j,
#                     ],
#                 ],
#                 [
#                     [
#                         0.10501874 + 0.03770757j,
#                         -0.00954112 - 0.11117821j,
#                         -0.09705694 + 0.05506731j,
#                         0.09052647 + 0.06525521j,
#                     ],
#                     [
#                         0.11017892 - 0.01778723j,
#                         -0.06217132 - 0.09268947j,
#                         -0.05829531 + 0.09517927j,
#                         0.11082761 + 0.01324784j,
#                     ],
#                     [
#                         0.08785306 - 0.06886658j,
#                         -0.09926082 - 0.05107721j,
#                         -0.00495222 + 0.11152531j,
#                         0.10337779 - 0.04214599j,
#                     ],
#                 ],
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ],
#                 [
#                     [1.00000000e00, 1.00001667e00, 1.00003333e00, 1.00005000e00],
#                     [1.00010000e00, 1.00011667e00, 1.00013333e00, 1.00015000e00],
#                     [1.00020000e00, 1.00021667e00, 1.00023333e00, 1.00025000e00],
#                 ],
#             ]
#         ),
#     )


# def test_scene_2_frames_moving_radar():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
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
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx, speed=[5, 0, 0], time=[0, 1])

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "speed": np.array([0, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         0.02167872 + 0.01755585j,
#                         -0.02776995 + 0.00265147j,
#                         0.01794177 - 0.02136164j,
#                         0.00216204 + 0.02781327j,
#                     ],
#                     [
#                         0.02746562 + 0.00489594j,
#                         -0.02301209 + 0.01577339j,
#                         0.00536318 - 0.02737916j,
#                         0.01536405 + 0.02328849j,
#                     ],
#                     [
#                         0.02640622 - 0.00901099j,
#                         -0.01251624 + 0.02493704j,
#                         -0.00855785 - 0.0265575j,
#                         0.02473615 + 0.01291073j,
#                     ],
#                 ],
#                 [
#                     [
#                         0.10501874 + 0.03770757j,
#                         -0.00954112 - 0.11117821j,
#                         -0.09705694 + 0.05506731j,
#                         0.09052647 + 0.06525521j,
#                     ],
#                     [
#                         0.11017892 - 0.01778723j,
#                         -0.06212666 - 0.09271938j,
#                         -0.05829531 + 0.09517927j,
#                         0.11083401 + 0.01319437j,
#                     ],
#                     [
#                         0.08788617 - 0.06882428j,
#                         -0.09926082 - 0.05107721j,
#                         -0.00489847 + 0.11152771j,
#                         0.10337779 - 0.04214599j,
#                     ],
#                 ],
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ],
#                 [
#                     [1.00000000e00, 1.00001667e00, 1.00003333e00, 1.00005000e00],
#                     [1.00010000e00, 1.00011667e00, 1.00013333e00, 1.00015000e00],
#                     [1.00020000e00, 1.00021667e00, 1.00023333e00, 1.00025000e00],
#                 ],
#             ]
#         ),
#     )


# def test_scene_tx_az_pattern():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     az_angle = np.array([-46, 0, 46])
#     az_pattern = np.array([-10, -10, 10])
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
#         t=80e-6,
#         tx_power=10,
#         prp=100e-6,
#         pulses=3,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#                 "azimuth_angle": az_angle,
#                 "azimuth_pattern": az_pattern,
#             }
#         ],
#     )
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 10, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.03187867 + 0.03048257j,
#                         -0.02458697 + 0.03661854j,
#                         -0.01614748 + 0.04104505j,
#                         -0.0069541 + 0.04355545j,
#                     ],
#                     [
#                         -0.03187867 + 0.03048257j,
#                         -0.02458697 + 0.03661854j,
#                         -0.01614748 + 0.04104505j,
#                         -0.0069541 + 0.04355545j,
#                     ],
#                     [
#                         -0.03187867 + 0.03048257j,
#                         -0.02458697 + 0.03661854j,
#                         -0.01614748 + 0.04104505j,
#                         -0.0069541 + 0.04355545j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )

#     targets = [
#         {
#             "location": np.array([10, -10, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.00318787 + 0.00304826j,
#                         -0.0024587 + 0.00366185j,
#                         -0.00161475 + 0.00410451j,
#                         -0.00069541 + 0.00435555j,
#                     ],
#                     [
#                         -0.00318787 + 0.00304826j,
#                         -0.0024587 + 0.00366185j,
#                         -0.00161475 + 0.00410451j,
#                         -0.00069541 + 0.00435555j,
#                     ],
#                     [
#                         -0.00318787 + 0.00304826j,
#                         -0.0024587 + 0.00366185j,
#                         -0.00161475 + 0.00410451j,
#                         -0.00069541 + 0.00435555j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_rx_az_pattern():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     az_angle = np.array([-46, 0, 46])
#     az_pattern = np.array([-10, -10, 10])
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
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
#     rx = Receiver(
#         fs=6e4,
#         noise_figure=12,
#         rf_gain=20,
#         load_resistor=500,
#         baseband_gain=30,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#                 "azimuth_angle": az_angle,
#                 "azimuth_pattern": az_pattern,
#             }
#         ],
#     )
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 10, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.03187867 + 0.03048257j,
#                         -0.02458697 + 0.03661854j,
#                         -0.01614748 + 0.04104505j,
#                         -0.0069541 + 0.04355545j,
#                     ],
#                     [
#                         -0.03187867 + 0.03048257j,
#                         -0.02458697 + 0.03661854j,
#                         -0.01614748 + 0.04104505j,
#                         -0.0069541 + 0.04355545j,
#                     ],
#                     [
#                         -0.03187867 + 0.03048257j,
#                         -0.02458697 + 0.03661854j,
#                         -0.01614748 + 0.04104505j,
#                         -0.0069541 + 0.04355545j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )

#     targets = [
#         {
#             "location": np.array([10, -10, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.00318787 + 0.00304826j,
#                         -0.0024587 + 0.00366185j,
#                         -0.00161475 + 0.00410451j,
#                         -0.00069541 + 0.00435555j,
#                     ],
#                     [
#                         -0.00318787 + 0.00304826j,
#                         -0.0024587 + 0.00366185j,
#                         -0.00161475 + 0.00410451j,
#                         -0.00069541 + 0.00435555j,
#                     ],
#                     [
#                         -0.00318787 + 0.00304826j,
#                         -0.0024587 + 0.00366185j,
#                         -0.00161475 + 0.00410451j,
#                         -0.00069541 + 0.00435555j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_tx_el_pattern():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     el_angle = np.array([-46, 0, 46])
#     el_pattern = np.array([-10, 10, 10])
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
#         t=80e-6,
#         tx_power=10,
#         prp=100e-6,
#         pulses=3,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#                 "elevation_angle": el_angle,
#                 "elevation_pattern": el_pattern,
#             }
#         ],
#     )
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 0, 10]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.01008092 + 0.00963944j,
#                         -0.00777508 + 0.0115798j,
#                         -0.00510628 + 0.01297958j,
#                         -0.00219908 + 0.01377344j,
#                     ],
#                     [
#                         -0.01008092 + 0.00963944j,
#                         -0.00777508 + 0.0115798j,
#                         -0.00510628 + 0.01297958j,
#                         -0.00219908 + 0.01377344j,
#                     ],
#                     [
#                         -0.01008092 + 0.00963944j,
#                         -0.00777508 + 0.0115798j,
#                         -0.00510628 + 0.01297958j,
#                         -0.00219908 + 0.01377344j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )

#     targets = [
#         {
#             "location": np.array([10, 0, -10]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.00100809 + 0.00096394j,
#                         -0.00077751 + 0.00115798j,
#                         -0.00051063 + 0.00129796j,
#                         -0.00021991 + 0.00137734j,
#                     ],
#                     [
#                         -0.00100809 + 0.00096394j,
#                         -0.00077751 + 0.00115798j,
#                         -0.00051063 + 0.00129796j,
#                         -0.00021991 + 0.00137734j,
#                     ],
#                     [
#                         -0.00100809 + 0.00096394j,
#                         -0.00077751 + 0.00115798j,
#                         -0.00051063 + 0.00129796j,
#                         -0.00021991 + 0.00137734j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_rx_el_pattern():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     el_angle = np.array([-46, 0, 46])
#     el_pattern = np.array([-10, 10, 10])
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
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
#     rx = Receiver(
#         fs=6e4,
#         noise_figure=12,
#         rf_gain=20,
#         load_resistor=500,
#         baseband_gain=30,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#                 "elevation_angle": el_angle,
#                 "elevation_pattern": el_pattern,
#             }
#         ],
#     )
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 0, 10]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.01008092 + 0.00963944j,
#                         -0.00777508 + 0.0115798j,
#                         -0.00510628 + 0.01297958j,
#                         -0.00219908 + 0.01377344j,
#                     ],
#                     [
#                         -0.01008092 + 0.00963944j,
#                         -0.00777508 + 0.0115798j,
#                         -0.00510628 + 0.01297958j,
#                         -0.00219908 + 0.01377344j,
#                     ],
#                     [
#                         -0.01008092 + 0.00963944j,
#                         -0.00777508 + 0.0115798j,
#                         -0.00510628 + 0.01297958j,
#                         -0.00219908 + 0.01377344j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )

#     targets = [
#         {
#             "location": np.array([10, 0, -10]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.00100809 + 0.00096394j,
#                         -0.00077751 + 0.00115798j,
#                         -0.00051063 + 0.00129796j,
#                         -0.00021991 + 0.00137734j,
#                     ],
#                     [
#                         -0.00100809 + 0.00096394j,
#                         -0.00077751 + 0.00115798j,
#                         -0.00051063 + 0.00129796j,
#                         -0.00021991 + 0.00137734j,
#                     ],
#                     [
#                         -0.00100809 + 0.00096394j,
#                         -0.00077751 + 0.00115798j,
#                         -0.00051063 + 0.00129796j,
#                         -0.00021991 + 0.00137734j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_freq_offset():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
#         t=80e-6,
#         tx_power=10,
#         prp=100e-6,
#         pulses=3,
#         f_offset=[0, 1e6, 2e6],
#         channels=[
#             {
#                 "location": (0, 0, 0),
#             }
#         ],
#     )
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         0.02167872 + 0.01755585j,
#                         -0.02789397 + 0.00031774j,
#                         0.02127319 - 0.01804511j,
#                         -0.00486305 + 0.02746863j,
#                     ],
#                     [
#                         0.01265615 + 0.02485824j,
#                         -0.02560738 - 0.0110622j,
#                         0.0267748 - 0.00782435j,
#                         -0.01562117 + 0.02311037j,
#                     ],
#                     [
#                         0.00144308 + 0.02785612j,
#                         -0.01888737 - 0.02052591j,
#                         0.02764021 + 0.00375025j,
#                         -0.02367378 + 0.0147512j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_pulse_modulation():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
#         t=80e-6,
#         tx_power=10,
#         prp=100e-6,
#         pulses=3,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#                 "pulse_amp": (0, 1, 2),
#                 "pulse_phs": (0, 180, 0),
#             }
#         ],
#     )
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
#                     [
#                         -0.02167872 - 0.01755585j,
#                         0.02789397 - 0.00031774j,
#                         -0.02127319 + 0.01804511j,
#                         0.00486305 - 0.02746863j,
#                     ],
#                     [
#                         0.04335744 + 0.0351117j,
#                         -0.05578795 + 0.00063547j,
#                         0.04254638 - 0.03609022j,
#                         -0.0097261 + 0.05493725j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_waveform_modulation():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9],
#         t=80e-6,
#         tx_power=10,
#         prp=100e-6,
#         pulses=3,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#                 "mod_t": (0, 10e-6, 20e-6, 30e-6, 40e-6),
#                 "amp": (0, 1, 0, 3, 4),
#                 "phs": (0, 90, 180, -90, -180),
#             }
#         ],
#     )
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         -0.01755585 + 0.02167872j,
#                         0.0 + 0.0j,
#                         -0.08509277 + 0.07218044j,
#                         0.0 + 0.0j,
#                     ],
#                     [
#                         -0.01755585 + 0.02167872j,
#                         0.0 + 0.0j,
#                         -0.08509277 + 0.07218044j,
#                         0.0 + 0.0j,
#                     ],
#                     [
#                         -0.01755585 + 0.02167872j,
#                         0.0 + 0.0j,
#                         -0.08509277 + 0.07218044j,
#                         0.0 + 0.0j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


# def test_scene_arbitrary_waveform():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     tx = Transmitter(
#         f=[24.075e9, 24.175e9, 26e9, 28e9, 26e9],
#         t=[0, 20e-6, 40e-6, 60e-6, 80e-6],
#         tx_power=10,
#         prp=100e-6,
#         pulses=3,
#         channels=[
#             {
#                 "location": (0, 0, 0),
#             }
#         ],
#     )
#     rx = Receiver(
#         fs=6e4,
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
#     radar = Radar(transmitter=tx, receiver=rx)

#     targets = [
#         {
#             "location": np.array([10, 0, 0]),
#             "rcs": 20,
#         }
#     ]
#     result = simc(radar, targets, noise=False)

#     assert np.allclose(
#         result["baseband"],
#         np.array(
#             [
#                 [
#                     [
#                         0.02091127 + 0.01519129j,
#                         0.01652503 - 0.0198741j,
#                         -0.02224128 + 0.01316747j,
#                         0.02556374 + 0.0038147j,
#                     ],
#                     [
#                         0.02091127 + 0.01519129j,
#                         0.01652503 - 0.0198741j,
#                         -0.02224128 + 0.01316747j,
#                         0.02556374 + 0.0038147j,
#                     ],
#                     [
#                         0.02091127 + 0.01519129j,
#                         0.01652503 - 0.0198741j,
#                         -0.02224128 + 0.01316747j,
#                         0.02556374 + 0.0038147j,
#                     ],
#                 ]
#             ]
#         ),
#     )

#     assert np.allclose(
#         result["timestamp"],
#         np.array(
#             [
#                 [
#                     [0.00000000e00, 1.66666667e-05, 3.33333333e-05, 5.00000000e-05],
#                     [1.00000000e-04, 1.16666667e-04, 1.33333333e-04, 1.50000000e-04],
#                     [2.00000000e-04, 2.16666667e-04, 2.33333333e-04, 2.50000000e-04],
#                 ]
#             ]
#         ),
#     )


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
