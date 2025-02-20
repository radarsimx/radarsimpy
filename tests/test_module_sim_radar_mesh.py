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
from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module


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
    result = sim_radar(radar, targets, density=0.4)

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
    result = sim_radar(radar, targets, density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.0484297 - 0.00610006j,
                        -0.02626685 + 0.04016461j,
                        -0.01360955 - 0.04528188j,
                    ],
                    [
                        -0.04614945 + 0.0181075j,
                        0.01597327 - 0.04601508j,
                        0.02436816 + 0.04123445j,
                        -0.04656862 - 0.00772764j,
                    ],
                    [
                        0.00566604 + 0.04912494j,
                        -0.04033571 - 0.0270871j,
                        0.04594827 - 0.01312849j,
                        -0.01913325 + 0.04305736j,
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
    result = sim_radar(radar, targets, density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.04775579 + 0.01025226j,
                        -0.0456525 + 0.01498527j,
                        0.0312634 - 0.03556857j,
                    ],
                    [
                        0.00861838 - 0.049022j,
                        0.01673768 + 0.04598741j,
                        -0.03698771 - 0.030807j,
                        0.04679892 + 0.00770881j,
                    ],
                    [
                        0.04616229 - 0.01886378j,
                        -0.03010787 + 0.03870164j,
                        0.00650388 - 0.04778526j,
                        0.01826333 + 0.04385472j,
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
    result = sim_radar(radar, targets, density=0.4)

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
    result = sim_radar(radar, targets, density=0.4)

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
    result = sim_radar(radar, targets, density=0.4)

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
    result = sim_radar(radar, targets, density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.0484297 - 0.00610006j,
                        -0.02626685 + 0.04016461j,
                        -0.01360955 - 0.04528188j,
                    ],
                    [
                        -0.04774782 + 0.01336951j,
                        0.02053713 - 0.04417777j,
                        0.02008224 + 0.04349261j,
                        -0.04555498 - 0.01239967j,
                    ],
                    [
                        -0.01409378 + 0.04743956j,
                        -0.02645369 - 0.0407985j,
                        0.04743885 + 0.00602968j,
                        -0.03457894 + 0.03204439j,
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
    result = sim_radar(radar, targets, density=0.4)

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
    result = sim_radar(radar, targets, density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.0484297 - 0.00610006j,
                        -0.02626685 + 0.04016461j,
                        -0.01360955 - 0.04528188j,
                    ],
                    [
                        -0.04774782 + 0.01336951j,
                        0.02053713 - 0.04417777j,
                        0.02008224 + 0.04349261j,
                        -0.04555498 - 0.01239967j,
                    ],
                    [
                        -0.01409378 + 0.04743956j,
                        -0.02645369 - 0.0407985j,
                        0.04743885 + 0.00602968j,
                        -0.03457894 + 0.03204439j,
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
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([-5, 0, 0]),
        }
    ]
    result = sim_radar(radar, targets, frame_time=[0, 1], density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.048779 - 0.00200617j,
                        -0.03264458 + 0.03519746j,
                        -0.00183011 - 0.04726537j,
                    ],
                    [
                        -0.04830816 - 0.0113859j,
                        0.04167507 - 0.02533764j,
                        -0.01149053 + 0.04656554j,
                        -0.0245123 - 0.04041204j,
                    ],
                    [
                        -0.04774782 + 0.01336951j,
                        0.02418353 - 0.04230094j,
                        0.01250176 + 0.04625984j,
                        -0.04100973 - 0.02342769j,
                    ],
                ],
                [
                    [
                        -0.09457635 - 0.03042372j,
                        0.01204567 + 0.09991933j,
                        0.08722993 - 0.0529265j,
                        -0.08481945 - 0.05887941j,
                    ],
                    [
                        -0.09761251 + 0.01917677j,
                        0.05897776 + 0.08174065j,
                        0.05083447 - 0.0886448j,
                        -0.10283258 - 0.01056495j,
                    ],
                    [
                        -0.07624196 + 0.06410639j,
                        0.09130233 + 0.04307221j,
                        0.0016385 - 0.10232754j,
                        -0.09524058 + 0.04048807j,
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
    radar = Radar(transmitter=tx, receiver=rx, speed=[5, 0, 0])

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([0, 0, 0]),
        }
    ]
    result = sim_radar(radar, targets, frame_time=[0, 1], density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03682247 - 0.0333487j,
                        0.048779 - 0.00200617j,
                        -0.03264458 + 0.03519746j,
                        -0.00183011 - 0.04726537j,
                    ],
                    [
                        -0.04830816 - 0.0113859j,
                        0.04167507 - 0.02533764j,
                        -0.01149053 + 0.04656554j,
                        -0.0245123 - 0.04041204j,
                    ],
                    [
                        -0.04774782 + 0.01336951j,
                        0.02418353 - 0.04230094j,
                        0.01250176 + 0.04625984j,
                        -0.04100973 - 0.02342769j,
                    ],
                ],
                [
                    [
                        -0.0915058 - 0.03280507j,
                        0.00859017 + 0.09721339j,
                        0.08450722 - 0.04815743j,
                        -0.07879424 - 0.05567041j,
                    ],
                    [
                        -0.09598922 + 0.01558639j,
                        0.05454866 + 0.08091951j,
                        0.05055801 - 0.08301045j,
                        -0.09585946 - 0.01037386j,
                    ],
                    [
                        -0.07648922 + 0.06011385j,
                        0.08687972 + 0.0444237j,
                        0.00399905 - 0.09703877j,
                        -0.08877797 + 0.03749229j,
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
    result = sim_radar(radar, targets, density=1)

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
    result = sim_radar(radar, targets, density=1)

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
    az_pattern = np.array([-10, 10, 10])
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
    result = sim_radar(radar, targets, density=1)

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
    result = sim_radar(radar, targets, density=1)

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
    result = sim_radar(radar, targets, density=1)

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
    result = sim_radar(radar, targets, density=1)

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
    result = sim_radar(radar, targets, density=1)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.03199767 + 0.03529163j,
                        0.0388235 + 0.02757874j,
                        0.04382929 + 0.0185821j,
                        0.04678249 + 0.00872353j,
                    ],
                    [
                        0.03199767 + 0.03529163j,
                        0.0388235 + 0.02757874j,
                        0.04382929 + 0.0185821j,
                        0.04678249 + 0.00872353j,
                    ],
                    [
                        0.03199767 + 0.03529163j,
                        0.0388235 + 0.02757874j,
                        0.04382929 + 0.0185821j,
                        0.04678249 + 0.00872353j,
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
    result = sim_radar(radar, targets, density=1)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02963792 + 0.03688986j,
                        0.03694382 + 0.02969039j,
                        0.0425374 + 0.02107055j,
                        0.0461487 + 0.01143068j,
                    ],
                    [
                        0.02963792 + 0.03688986j,
                        0.03694382 + 0.02969039j,
                        0.0425374 + 0.02107055j,
                        0.0461487 + 0.01143068j,
                    ],
                    [
                        0.02963792 + 0.03688986j,
                        0.03694382 + 0.02969039j,
                        0.0425374 + 0.02107055j,
                        0.0461487 + 0.01143068j,
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
    result = sim_radar(radar, targets, density=0.4)

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
                        -0.02004311 - 0.0454117j,
                        0.04367144 + 0.02175008j,
                        -0.04665802 + 0.01120173j,
                        0.02798027 - 0.03812366j,
                    ],
                    [
                        0.0001762 - 0.04959674j,
                        0.03102003 + 0.03760466j,
                        -0.04714614 - 0.00873217j,
                        0.04103534 - 0.02344538j,
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
    result = sim_radar(radar, targets, density=0.4)

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
    result = sim_radar(radar, targets, density=0.4)

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
    result = sim_radar(radar, targets, density=0.4)

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


def test_scene_interference():
    """
    Basic test case with a single target and simple radar setup.
    """
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=1,
        channels=[
            {
                "location": (0, 0, 0),
            }
        ],
    )
    rx = Receiver(
        fs=6e5,
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
    interference_tx = Transmitter(
        f=[24.175e9, 24.075e9],
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
    interference_radar = Radar(
        transmitter=interference_tx,
        receiver=rx,
        location=(20, 0, 0),
        rotation=(180, 0, 0),
    )

    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
        }
    ]
    result = sim_radar(radar, targets, density=0.4, interf=interference_radar)

    assert np.allclose(
        result["interference"],
        np.array(
            [
                [
                    [
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        -0.01325275 + 0.00434837j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                        0.0 + 0.0j,
                    ]
                ]
            ]
        ),
    )
