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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02744737 + 0.00498684j,
                        0.01411165 - 0.02406532j,
                        0.00905771 + 0.02638727j,
                    ],
                    [
                        0.02536932 - 0.01161528j,
                        -0.00771241 + 0.02681581j,
                        -0.01532997 - 0.02331551j,
                        0.02767929 + 0.0035398j,
                    ],
                    [
                        -0.00473423 - 0.02750472j,
                        0.02396997 + 0.01429737j,
                        -0.02644997 + 0.00891208j,
                        0.01042298 - 0.02589285j,
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
    result = sim_radar(radar, targets)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.01979939 + 0.01965013j,
                        -0.02697143 - 0.00711575j,
                        0.02691655 - 0.00731703j,
                        -0.01966141 + 0.01978426j,
                    ],
                    [
                        -0.00608537 + 0.02721766j,
                        -0.00831702 - 0.02661969j,
                        0.02049118 + 0.01891667j,
                        -0.02719477 - 0.00617424j,
                    ],
                    [
                        -0.02627212 + 0.0093433j,
                        0.01811568 - 0.0211965j,
                        -0.00514422 + 0.02740355j,
                        -0.00918697 - 0.02632424j,
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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02744737 + 0.00498684j,
                        0.01411165 - 0.02406532j,
                        0.00905771 + 0.02638727j,
                    ],
                    [
                        0.02640989 - 0.00900021j,
                        -0.01037674 + 0.02590099j,
                        -0.01289851 - 0.02474305j,
                        0.02717966 + 0.00631729j,
                    ],
                    [
                        0.0064492 - 0.02715153j,
                        0.01641852 + 0.02256727j,
                        -0.02782096 - 0.0022126j,
                        0.01977469 - 0.01969556j,
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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02744737 + 0.00498684j,
                        0.01411165 - 0.02406532j,
                        0.00905771 + 0.02638727j,
                    ],
                    [
                        0.02640989 - 0.00900021j,
                        -0.01037674 + 0.02590099j,
                        -0.01289851 - 0.02474305j,
                        0.02717966 + 0.00631729j,
                    ],
                    [
                        0.0064492 - 0.02715153j,
                        0.01641852 + 0.02256727j,
                        -0.02782096 - 0.0022126j,
                        0.01977469 - 0.01969556j,
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
    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "location": np.array([10, 0, 0]),
            "speed": np.array([-5, 0, 0]),
            "rcs": 20,
        }
    ]
    result = sim_radar(radar, targets, frame_time=[0, 1])

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02776898 + 0.00266167j,
                        0.01794666 - 0.02135753j,
                        0.0021659 + 0.02781297j,
                    ],
                    [
                        0.02746697 + 0.00488833j,
                        -0.02301712 + 0.01576606j,
                        0.00536185 - 0.02737943j,
                        0.01537376 + 0.02328209j,
                    ],
                    [
                        0.02640989 - 0.00900021j,
                        -0.01251727 + 0.02493652j,
                        -0.00856652 - 0.02655471j,
                        0.02473266 + 0.01291742j,
                    ],
                ],
                [
                    [
                        0.10501874 + 0.03770757j,
                        -0.00952841 - 0.11117929j,
                        -0.09706953 + 0.0550451j,
                        0.09053554 + 0.06524264j,
                    ],
                    [
                        0.11018254 - 0.01776475j,
                        -0.06214179 - 0.09270925j,
                        -0.05829066 + 0.09518213j,
                        0.11083315 + 0.01320164j,
                    ],
                    [
                        0.08788114 - 0.06883071j,
                        -0.09925872 - 0.05108129j,
                        -0.00491581 + 0.11152694j,
                        0.10336885 - 0.04216794j,
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
    radar = Radar(transmitter=tx, receiver=rx, speed=[5, 0, 0])

    targets = [
        {
            "location": np.array([10, 0, 0]),
            "speed": np.array([0, 0, 0]),
            "rcs": 20,
        }
    ]
    result = sim_radar(radar, targets, frame_time=[0, 1])

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167872 + 0.01755585j,
                        -0.02776898 + 0.00266167j,
                        0.01794666 - 0.02135753j,
                        0.0021659 + 0.02781297j,
                    ],
                    [
                        0.02746697 + 0.00488833j,
                        -0.02301712 + 0.01576606j,
                        0.00536185 - 0.02737943j,
                        0.01537376 + 0.02328209j,
                    ],
                    [
                        0.02640989 - 0.00900021j,
                        -0.01251727 + 0.02493652j,
                        -0.00856652 - 0.02655471j,
                        0.02473266 + 0.01291742j,
                    ],
                ],
                [
                    [
                        0.10501874 + 0.03770757j,
                        -0.00952841 - 0.11117929j,
                        -0.09706953 + 0.0550451j,
                        0.09053554 + 0.06524264j,
                    ],
                    [
                        0.11018254 - 0.01776475j,
                        -0.06214179 - 0.09270925j,
                        -0.05829066 + 0.09518213j,
                        0.11083315 + 0.01320164j,
                    ],
                    [
                        0.08788114 - 0.06883071j,
                        -0.09925872 - 0.05108129j,
                        -0.00491581 + 0.11152694j,
                        0.10336885 - 0.04216794j,
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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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
    result = sim_radar(radar, targets)

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

    radar = Radar(transmitter=tx, receiver=rx)

    targets = [
        {
            "location": np.array([10, 0, 0]),
            "rcs": 20,
        }
    ]
    result = sim_radar(radar, targets, interf=interference_radar)

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
