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
from radarsimpy.simulator import simc  # pylint: disable=no-name-in-module


def test_simc_single_target():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02789397 + 0.00031774j,
                        0.02127319 - 0.01804511j,
                        -0.00486305 + 0.02746863j,
                    ],
                    [
                        0.02167872 + 0.01755585j,
                        -0.02789397 + 0.00031774j,
                        0.02127319 - 0.01804511j,
                        -0.00486305 + 0.02746863j,
                    ],
                    [
                        0.02167872 + 0.01755585j,
                        -0.02789397 + 0.00031774j,
                        0.02127319 - 0.01804511j,
                        -0.00486305 + 0.02746863j,
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


def test_simc_varing_prp():
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
            "location": np.array([10, 0, 0]),
            "speed": np.array([-10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02744623 + 0.00499312j,
                        0.01410064 - 0.02407178j,
                        0.00905038 + 0.02638979j,
                    ],
                    [
                        0.02537417 - 0.01160469j,
                        -0.00771748 + 0.02681435j,
                        -0.01533089 - 0.02331491j,
                        0.02768024 + 0.00353236j,
                    ],
                    [
                        -0.00474528 - 0.02750281j,
                        0.02396522 + 0.01430533j,
                        -0.0264509 + 0.00890933j,
                        0.01041974 - 0.02589415j,
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


def test_simc_tx_delay():
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
            "location": np.array([10, 0, 0]),
            "speed": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.0197967 + 0.01965284j,
                        -0.02696882 - 0.00712562j,
                        0.02691385 - 0.00732697j,
                        -0.01965863 + 0.01978702j,
                    ],
                    [
                        -0.00607401 + 0.02722019j,
                        -0.00832206 - 0.02661811j,
                        0.02049043 + 0.01891748j,
                        -0.02719311 - 0.00618155j,
                    ],
                    [
                        -0.02627203 + 0.00934354j,
                        0.01812033 - 0.02119252j,
                        -0.0051565 + 0.02740124j,
                        -0.00919453 - 0.0263216j,
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


def test_simc_tx_offset():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.04858788 - 0.0274211j,
                        -0.03965778 - 0.03924231j,
                        -0.02793127 - 0.04829641j,
                        -0.01423527 - 0.05394494j,
                    ],
                    [
                        -0.04858788 - 0.0274211j,
                        -0.03965778 - 0.03924231j,
                        -0.02793127 - 0.04829641j,
                        -0.01423527 - 0.05394494j,
                    ],
                    [
                        -0.04858788 - 0.0274211j,
                        -0.03965778 - 0.03924231j,
                        -0.02793127 - 0.04829641j,
                        -0.01423527 - 0.05394494j,
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


def test_simc_rx_offset():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.04858788 - 0.0274211j,
                        -0.03965778 - 0.03924231j,
                        -0.02793127 - 0.04829641j,
                        -0.01423527 - 0.05394494j,
                    ],
                    [
                        -0.04858788 - 0.0274211j,
                        -0.03965778 - 0.03924231j,
                        -0.02793127 - 0.04829641j,
                        -0.01423527 - 0.05394494j,
                    ],
                    [
                        -0.04858788 - 0.0274211j,
                        -0.03965778 - 0.03924231j,
                        -0.02793127 - 0.04829641j,
                        -0.01423527 - 0.05394494j,
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


def test_simc_multiple_targets():
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
            "location": np.array([10, 10, 0]),
            "rcs": 20,
        },
        {
            "location": np.array([10, -10, 0]),
            "rcs": 20,
        },
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.02016184 + 0.01927887j,
                        -0.01555017 + 0.0231596j,
                        -0.01021256 + 0.02595917j,
                        -0.00439816 + 0.02754689j,
                    ],
                    [
                        -0.02016184 + 0.01927887j,
                        -0.01555017 + 0.0231596j,
                        -0.01021256 + 0.02595917j,
                        -0.00439816 + 0.02754689j,
                    ],
                    [
                        -0.02016184 + 0.01927887j,
                        -0.01555017 + 0.0231596j,
                        -0.01021256 + 0.02595917j,
                        -0.00439816 + 0.02754689j,
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


def test_simc_single_target_speed():
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
            "location": np.array([10, 0, 0]),
            "speed": np.array([-10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02744623 + 0.00499312j,
                        0.01410064 - 0.02407178j,
                        0.00905038 + 0.02638979j,
                    ],
                    [
                        0.02640622 - 0.00901099j,
                        -0.01038519 + 0.0258976j,
                        -0.01289609 - 0.0247443j,
                        0.02718049 + 0.00631372j,
                    ],
                    [
                        0.00645317 - 0.02715058j,
                        0.01642037 + 0.02256592j,
                        -0.02782165 - 0.00220395j,
                        0.01978305 - 0.01968716j,
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


def test_simc_single_target_phase():
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
            "location": np.array([10, 0, 0]),
            "phase": 180,
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.02167872 - 0.01755585j,
                        0.02789397 - 0.00031773j,
                        -0.02127319 + 0.01804511j,
                        0.00486305 - 0.02746863j,
                    ],
                    [
                        -0.02167872 - 0.01755585j,
                        0.02789397 - 0.00031773j,
                        -0.02127319 + 0.01804511j,
                        0.00486305 - 0.02746863j,
                    ],
                    [
                        -0.02167872 - 0.01755585j,
                        0.02789397 - 0.00031773j,
                        -0.02127319 + 0.01804511j,
                        0.00486305 - 0.02746863j,
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


def test_simc_radar_location():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.10501874 + 0.03770757j,
                        -0.00014794 - 0.11158303j,
                        -0.10491838 + 0.03798591j,
                        0.07132042 + 0.08581488j,
                    ],
                    [
                        0.10501874 + 0.03770757j,
                        -0.00014794 - 0.11158303j,
                        -0.10491838 + 0.03798591j,
                        0.07132042 + 0.08581488j,
                    ],
                    [
                        0.10501874 + 0.03770757j,
                        -0.00014794 - 0.11158303j,
                        -0.10491838 + 0.03798591j,
                        0.07132042 + 0.08581488j,
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


def test_simc_radar_moving():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02744623 + 0.00499312j,
                        0.01410064 - 0.02407178j,
                        0.00905038 + 0.02638979j,
                    ],
                    [
                        0.02640622 - 0.00901099j,
                        -0.01038519 + 0.0258976j,
                        -0.01289609 - 0.0247443j,
                        0.02718049 + 0.00631372j,
                    ],
                    [
                        0.00645317 - 0.02715058j,
                        0.01642037 + 0.02256592j,
                        -0.02782165 - 0.00220395j,
                        0.01978305 - 0.01968716j,
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


def test_simc_2_frames_moving_target():
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
            "location": np.array([10, 0, 0]),
            "speed": np.array([-5, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02776995 + 0.00265147j,
                        0.01794177 - 0.02136164j,
                        0.00216204 + 0.02781327j,
                    ],
                    [
                        0.02746562 + 0.00489594j,
                        -0.02301209 + 0.01577339j,
                        0.00536318 - 0.02737916j,
                        0.01536405 + 0.02328849j,
                    ],
                    [
                        0.02640622 - 0.00901099j,
                        -0.01251624 + 0.02493704j,
                        -0.00855785 - 0.0265575j,
                        0.02473615 + 0.01291073j,
                    ],
                ],
                [
                    [
                        0.10501874 + 0.03770757j,
                        -0.00954112 - 0.11117821j,
                        -0.09705694 + 0.05506731j,
                        0.09052647 + 0.06525521j,
                    ],
                    [
                        0.11017892 - 0.01778723j,
                        -0.06217132 - 0.09268947j,
                        -0.05829531 + 0.09517927j,
                        0.11082761 + 0.01324784j,
                    ],
                    [
                        0.08785306 - 0.06886658j,
                        -0.09926082 - 0.05107721j,
                        -0.00495222 + 0.11152531j,
                        0.10337779 - 0.04214599j,
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


def test_simc_2_frames_moving_radar():
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
            "location": np.array([10, 0, 0]),
            "speed": np.array([0, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02776995 + 0.00265147j,
                        0.01794177 - 0.02136164j,
                        0.00216204 + 0.02781327j,
                    ],
                    [
                        0.02746562 + 0.00489594j,
                        -0.02301209 + 0.01577339j,
                        0.00536318 - 0.02737916j,
                        0.01536405 + 0.02328849j,
                    ],
                    [
                        0.02640622 - 0.00901099j,
                        -0.01251624 + 0.02493704j,
                        -0.00855785 - 0.0265575j,
                        0.02473615 + 0.01291073j,
                    ],
                ],
                [
                    [
                        0.10501874 + 0.03770757j,
                        -0.00954112 - 0.11117821j,
                        -0.09705694 + 0.05506731j,
                        0.09052647 + 0.06525521j,
                    ],
                    [
                        0.11017892 - 0.01778723j,
                        -0.06212666 - 0.09271938j,
                        -0.05829531 + 0.09517927j,
                        0.11083401 + 0.01319437j,
                    ],
                    [
                        0.08788617 - 0.06882428j,
                        -0.09926082 - 0.05107721j,
                        -0.00489847 + 0.11152771j,
                        0.10337779 - 0.04214599j,
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


def test_simc_tx_az_pattern():
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
            "location": np.array([10, 10, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03187867 + 0.03048257j,
                        -0.02458697 + 0.03661854j,
                        -0.01614748 + 0.04104505j,
                        -0.0069541 + 0.04355545j,
                    ],
                    [
                        -0.03187867 + 0.03048257j,
                        -0.02458697 + 0.03661854j,
                        -0.01614748 + 0.04104505j,
                        -0.0069541 + 0.04355545j,
                    ],
                    [
                        -0.03187867 + 0.03048257j,
                        -0.02458697 + 0.03661854j,
                        -0.01614748 + 0.04104505j,
                        -0.0069541 + 0.04355545j,
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
            "location": np.array([10, -10, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.00318787 + 0.00304826j,
                        -0.0024587 + 0.00366185j,
                        -0.00161475 + 0.00410451j,
                        -0.00069541 + 0.00435555j,
                    ],
                    [
                        -0.00318787 + 0.00304826j,
                        -0.0024587 + 0.00366185j,
                        -0.00161475 + 0.00410451j,
                        -0.00069541 + 0.00435555j,
                    ],
                    [
                        -0.00318787 + 0.00304826j,
                        -0.0024587 + 0.00366185j,
                        -0.00161475 + 0.00410451j,
                        -0.00069541 + 0.00435555j,
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


def test_simc_rx_az_pattern():
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
            "location": np.array([10, 10, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03187867 + 0.03048257j,
                        -0.02458697 + 0.03661854j,
                        -0.01614748 + 0.04104505j,
                        -0.0069541 + 0.04355545j,
                    ],
                    [
                        -0.03187867 + 0.03048257j,
                        -0.02458697 + 0.03661854j,
                        -0.01614748 + 0.04104505j,
                        -0.0069541 + 0.04355545j,
                    ],
                    [
                        -0.03187867 + 0.03048257j,
                        -0.02458697 + 0.03661854j,
                        -0.01614748 + 0.04104505j,
                        -0.0069541 + 0.04355545j,
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
            "location": np.array([10, -10, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.00318787 + 0.00304826j,
                        -0.0024587 + 0.00366185j,
                        -0.00161475 + 0.00410451j,
                        -0.00069541 + 0.00435555j,
                    ],
                    [
                        -0.00318787 + 0.00304826j,
                        -0.0024587 + 0.00366185j,
                        -0.00161475 + 0.00410451j,
                        -0.00069541 + 0.00435555j,
                    ],
                    [
                        -0.00318787 + 0.00304826j,
                        -0.0024587 + 0.00366185j,
                        -0.00161475 + 0.00410451j,
                        -0.00069541 + 0.00435555j,
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


def test_simc_tx_el_pattern():
    """
    Basic test case with a single target and simple radar setup.
    """
    el_angle = np.array([-46, 0, 46])
    el_pattern = np.array([-10, 10, 10])
    tx = Transmitter(
        f=[24.075e9, 24.175e9],
        t=80e-6,
        tx_power=10,
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
            "location": np.array([10, 0, 10]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.01008092 + 0.00963944j,
                        -0.00777508 + 0.0115798j,
                        -0.00510628 + 0.01297958j,
                        -0.00219908 + 0.01377344j,
                    ],
                    [
                        -0.01008092 + 0.00963944j,
                        -0.00777508 + 0.0115798j,
                        -0.00510628 + 0.01297958j,
                        -0.00219908 + 0.01377344j,
                    ],
                    [
                        -0.01008092 + 0.00963944j,
                        -0.00777508 + 0.0115798j,
                        -0.00510628 + 0.01297958j,
                        -0.00219908 + 0.01377344j,
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
            "location": np.array([10, 0, -10]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.00100809 + 0.00096394j,
                        -0.00077751 + 0.00115798j,
                        -0.00051063 + 0.00129796j,
                        -0.00021991 + 0.00137734j,
                    ],
                    [
                        -0.00100809 + 0.00096394j,
                        -0.00077751 + 0.00115798j,
                        -0.00051063 + 0.00129796j,
                        -0.00021991 + 0.00137734j,
                    ],
                    [
                        -0.00100809 + 0.00096394j,
                        -0.00077751 + 0.00115798j,
                        -0.00051063 + 0.00129796j,
                        -0.00021991 + 0.00137734j,
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


def test_simc_rx_el_pattern():
    """
    Basic test case with a single target and simple radar setup.
    """
    el_angle = np.array([-46, 0, 46])
    el_pattern = np.array([-10, 10, 10])
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
                "elevation_angle": el_angle,
                "elevation_pattern": el_pattern,
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "location": np.array([10, 0, 10]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.01008092 + 0.00963944j,
                        -0.00777508 + 0.0115798j,
                        -0.00510628 + 0.01297958j,
                        -0.00219908 + 0.01377344j,
                    ],
                    [
                        -0.01008092 + 0.00963944j,
                        -0.00777508 + 0.0115798j,
                        -0.00510628 + 0.01297958j,
                        -0.00219908 + 0.01377344j,
                    ],
                    [
                        -0.01008092 + 0.00963944j,
                        -0.00777508 + 0.0115798j,
                        -0.00510628 + 0.01297958j,
                        -0.00219908 + 0.01377344j,
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
            "location": np.array([10, 0, -10]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.00100809 + 0.00096394j,
                        -0.00077751 + 0.00115798j,
                        -0.00051063 + 0.00129796j,
                        -0.00021991 + 0.00137734j,
                    ],
                    [
                        -0.00100809 + 0.00096394j,
                        -0.00077751 + 0.00115798j,
                        -0.00051063 + 0.00129796j,
                        -0.00021991 + 0.00137734j,
                    ],
                    [
                        -0.00100809 + 0.00096394j,
                        -0.00077751 + 0.00115798j,
                        -0.00051063 + 0.00129796j,
                        -0.00021991 + 0.00137734j,
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


def test_simc_freq_offset():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02789397 + 0.00031774j,
                        0.02127319 - 0.01804511j,
                        -0.00486305 + 0.02746863j,
                    ],
                    [
                        0.01265615 + 0.02485824j,
                        -0.02560738 - 0.0110622j,
                        0.0267748 - 0.00782435j,
                        -0.01562117 + 0.02311037j,
                    ],
                    [
                        0.00144308 + 0.02785612j,
                        -0.01888737 - 0.02052591j,
                        0.02764021 + 0.00375025j,
                        -0.02367378 + 0.0147512j,
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


def test_simc_pulse_modulation():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [
                        -0.02167872 - 0.01755585j,
                        0.02789397 - 0.00031774j,
                        -0.02127319 + 0.01804511j,
                        0.00486305 - 0.02746863j,
                    ],
                    [
                        0.04335744 + 0.0351117j,
                        -0.05578795 + 0.00063547j,
                        0.04254638 - 0.03609022j,
                        -0.0097261 + 0.05493725j,
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


def test_simc_waveform_modulation():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.01755585 + 0.02167872j,
                        0.0 + 0.0j,
                        -0.08509277 + 0.07218044j,
                        0.0 + 0.0j,
                    ],
                    [
                        -0.01755585 + 0.02167872j,
                        0.0 + 0.0j,
                        -0.08509277 + 0.07218044j,
                        0.0 + 0.0j,
                    ],
                    [
                        -0.01755585 + 0.02167872j,
                        0.0 + 0.0j,
                        -0.08509277 + 0.07218044j,
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


def test_simc_arbitrary_waveform():
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
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02091127 + 0.01519129j,
                        0.01652503 - 0.0198741j,
                        -0.02224128 + 0.01316747j,
                        0.02556374 + 0.0038147j,
                    ],
                    [
                        0.02091127 + 0.01519129j,
                        0.01652503 - 0.0198741j,
                        -0.02224128 + 0.01316747j,
                        0.02556374 + 0.0038147j,
                    ],
                    [
                        0.02091127 + 0.01519129j,
                        0.01652503 - 0.0198741j,
                        -0.02224128 + 0.01316747j,
                        0.02556374 + 0.0038147j,
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


def test_simc_interference():
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

    radar = Radar(transmitter=tx, receiver=rx, interf=interference_radar)

    targets = [
        {
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = simc(radar, targets, noise=False)

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


# def test_sim_tdm_fmcw():
#     """
#     Test the TDM-FMCW radar simulator.
#     """
#     radar = tdm_fmcw_radar()
#     target_1 = {"location": (120, 0, 0), "speed": (0, 0, 0), "rcs": 25, "phase": 0}
#     target_2 = {"location": (80, -80, 0), "speed": (0, 0, 0), "rcs": 20, "phase": 0}
#     target_3 = {"location": (30, 20, 0), "speed": (0, 0, 0), "rcs": 8, "phase": 0}

#     rng_targets = np.sort(
#         np.array(
#             [
#                 np.sqrt(target_1["location"][0] ** 2 + target_1["location"][1] ** 2),
#                 np.sqrt(target_2["location"][0] ** 2 + target_2["location"][1] ** 2),
#                 np.sqrt(target_3["location"][0] ** 2 + target_3["location"][1] ** 2),
#             ]
#         )
#     )

#     targets = [target_1, target_2, target_3]

#     data = simc(radar, targets, noise=False)
#     timestamp = data["timestamp"]
#     baseband = data["baseband"]

#     assert np.array_equal(
#         (
#             radar.array_prop["size"],
#             radar.radar_prop["transmitter"].waveform_prop["pulses"],
#             radar.sample_prop["samples_per_pulse"],
#         ),
#         np.shape(timestamp),
#     )
#     assert np.array_equal(
#         (
#             radar.array_prop["size"],
#             radar.radar_prop["transmitter"].waveform_prop["pulses"],
#             radar.sample_prop["samples_per_pulse"],
#         ),
#         np.shape(baseband),
#     )

#     npt.assert_almost_equal(
#         timestamp[0, 0, :],
#         (
#             np.arange(0, radar.sample_prop["samples_per_pulse"])
#             / radar.radar_prop["receiver"].bb_prop["fs"]
#         ),
#     )
#     npt.assert_almost_equal(
#         timestamp[0, :, 0],
#         (
#             np.arange(0, radar.radar_prop["transmitter"].waveform_prop["pulses"])
#             * radar.radar_prop["transmitter"].waveform_prop["prp"][0]
#         ),
#     )
#     npt.assert_almost_equal(
#         timestamp[:, 0, 0],
#         np.array(
#             [
#                 0,
#                 0,
#                 0,
#                 0,
#                 0,
#                 0,
#                 0,
#                 0,
#                 100e-6,
#                 100e-6,
#                 100e-6,
#                 100e-6,
#                 100e-6,
#                 100e-6,
#                 100e-6,
#                 100e-6,
#             ]
#         ),
#     )
#     npt.assert_almost_equal(
#         timestamp[:, 1, 0],
#         np.array(
#             [
#                 200e-6,
#                 200e-6,
#                 200e-6,
#                 200e-6,
#                 200e-6,
#                 200e-6,
#                 200e-6,
#                 200e-6,
#                 300e-6,
#                 300e-6,
#                 300e-6,
#                 300e-6,
#                 300e-6,
#                 300e-6,
#                 300e-6,
#                 300e-6,
#             ]
#         ),
#     )

#     range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
#     range_profile = proc.range_fft(baseband, range_window)

#     rng_nci = 20 * np.log10(np.mean(np.abs(range_profile[:, 0, :]), axis=0))
#     rng_nci = rng_nci - np.max(rng_nci)
#     rng_peaks = signal.find_peaks(rng_nci, height=-10)[0]

#     max_range = (
#         const.c
#         * radar.radar_prop["receiver"].bb_prop["fs"]
#         * radar.radar_prop["transmitter"].waveform_prop["pulse_length"]
#         / radar.radar_prop["transmitter"].waveform_prop["bandwidth"]
#         / 2
#     )

#     range_axis = np.linspace(
#         0, max_range, radar.sample_prop["samples_per_pulse"], endpoint=False
#     )

#     rng_dets = np.sort(range_axis[rng_peaks])

#     npt.assert_almost_equal(rng_targets, rng_dets, decimal=0)


# def test_sim_pmcw():
#     """
#     Test the PMCW radar simulator.
#     """
#     code1 = np.array(
#         [
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#         ]
#     )
#     code2 = np.array(
#         [
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#             1,
#             -1,
#             1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             -1,
#             -1,
#             -1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             -1,
#             1,
#             1,
#             1,
#             -1,
#             -1,
#         ]
#     )
#     radar = pmcw_radar()

#     target_1 = {"location": (20, 0, 0), "speed": (-185, 0, 0), "rcs": 20, "phase": 0}

#     target_2 = {"location": (70, 0, 0), "speed": (0, 0, 0), "rcs": 35, "phase": 0}

#     target_3 = {"location": (33, 10, 0), "speed": (97, 0, 0), "rcs": 20, "phase": 0}

#     rng_targets = np.sort(
#         np.array(
#             [
#                 np.sqrt(target_1["location"][0] ** 2 + target_1["location"][1] ** 2),
#                 np.sqrt(target_2["location"][0] ** 2 + target_2["location"][1] ** 2),
#                 np.sqrt(target_3["location"][0] ** 2 + target_3["location"][1] ** 2),
#             ]
#         )
#     )
#     dop_targets = np.sort(
#         np.array(
#             [
#                 target_1["speed"][0]
#                 * np.cos(np.arctan(target_1["location"][1] / target_1["location"][0])),
#                 target_2["speed"][0]
#                 * np.cos(np.arctan(target_2["location"][1] / target_2["location"][0])),
#                 target_3["speed"][0]
#                 * np.cos(np.arctan(target_3["location"][1] / target_3["location"][0])),
#             ]
#         )
#     )

#     targets = [target_1, target_2, target_3]

#     data = simc(radar, targets, noise=False)
#     timestamp = data["timestamp"]
#     baseband = data["baseband"]

#     assert np.array_equal(
#         (
#             radar.array_prop["size"],
#             radar.radar_prop["transmitter"].waveform_prop["pulses"],
#             radar.sample_prop["samples_per_pulse"],
#         ),
#         np.shape(timestamp),
#     )
#     assert np.array_equal(
#         (
#             radar.array_prop["size"],
#             radar.radar_prop["transmitter"].waveform_prop["pulses"],
#             radar.sample_prop["samples_per_pulse"],
#         ),
#         np.shape(baseband),
#     )

#     npt.assert_almost_equal(
#         timestamp[0, 0, :],
#         (
#             np.arange(0, radar.sample_prop["samples_per_pulse"])
#             / radar.radar_prop["receiver"].bb_prop["fs"]
#         ),
#     )
#     npt.assert_almost_equal(
#         timestamp[0, :, 0],
#         (
#             np.arange(0, radar.radar_prop["transmitter"].waveform_prop["pulses"])
#             * radar.radar_prop["transmitter"].waveform_prop["prp"][0]
#         ),
#     )

#     code_length = 255
#     range_profile = np.zeros(
#         (
#             radar.array_prop["size"],
#             radar.radar_prop["transmitter"].waveform_prop["pulses"],
#             code_length,
#         ),
#         dtype=complex,
#     )

#     for pulse_idx in range(0, radar.radar_prop["transmitter"].waveform_prop["pulses"]):
#         for bin_idx in range(0, code_length):
#             range_profile[:, pulse_idx, bin_idx] = np.sum(
#                 code2 * baseband[1, pulse_idx, bin_idx : (bin_idx + code_length)]
#             )

#     bin_size = const.c / 2 * 4e-9
#     range_bin = np.arange(0, code_length, 1) * bin_size

#     doppler_window = signal.windows.chebwin(
#         radar.radar_prop["transmitter"].waveform_prop["pulses"], at=50
#     )

#     range_doppler = np.zeros(np.shape(range_profile), dtype=complex)
#     for ii in range(0, radar.array_prop["size"]):
#         for jj in range(0, code_length):
#             range_doppler[ii, :, jj] = np.fft.fftshift(
#                 np.fft.fft(
#                     range_profile[ii, :, jj] * doppler_window,
#                     n=radar.radar_prop["transmitter"].waveform_prop["pulses"],
#                 )
#             )
#     unambiguous_speed = (
#         const.c / radar.radar_prop["transmitter"].waveform_prop["prp"][0] / 24.125e9 / 2
#     )

#     rng_dop = 20 * np.log10(np.abs(range_doppler[1, :, :]))
#     rng_dop = rng_dop - np.max(rng_dop)

#     max_rng = np.max(rng_dop, axis=0)
#     max_dop = np.max(rng_dop, axis=1)

#     rng_peaks = signal.find_peaks(max_rng, height=-15)[0]
#     dop_peaks = signal.find_peaks(max_dop, height=-15)[0]

#     range_axis = np.arange(0, code_length, 1) * bin_size
#     rng_dets = np.sort(range_axis[rng_peaks])
#     npt.assert_almost_equal(rng_targets, rng_dets, decimal=0)

#     doppler_axis = np.linspace(
#         -unambiguous_speed / 2,
#         unambiguous_speed / 2,
#         radar.radar_prop["transmitter"].waveform_prop["pulses"],
#         endpoint=False,
#     )
#     dop_dets = np.sort(doppler_axis[dop_peaks])
#     npt.assert_almost_equal(dop_targets, dop_dets, decimal=0)
