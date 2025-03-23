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
                        -0.09182092 - 0.03251328j,
                        0.00070606 + 0.09790686j,
                        0.09172928 - 0.03364269j,
                        -0.0626775 - 0.07401067j,
                    ],
                    [
                        -0.09182092 - 0.03251328j,
                        0.00070606 + 0.09790686j,
                        0.09172928 - 0.03364269j,
                        -0.0626775 - 0.07401067j,
                    ],
                    [
                        -0.09182092 - 0.03251328j,
                        0.00070606 + 0.09790686j,
                        0.09172928 - 0.03364269j,
                        -0.0626775 - 0.07401067j,
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
                        -0.03856109 - 0.03410628j,
                        0.05035437 - 0.00639387j,
                        -0.02776542 + 0.04146667j,
                        -0.0130364 - 0.04720693j,
                    ],
                    [
                        -0.04935347 + 0.01439099j,
                        0.02137024 - 0.04594344j,
                        0.02032826 + 0.04546646j,
                        -0.04683668 - 0.01394894j,
                    ],
                    [
                        -0.01414471 + 0.04934835j,
                        -0.02745609 - 0.04247996j,
                        0.04921802 + 0.00692745j,
                        -0.03658137 + 0.03224719j,
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
                        -0.03856109 - 0.03410628j,
                        0.05072075 - 0.00214221j,
                        -0.03433426 + 0.03623831j,
                        -0.0008072 - 0.04899328j,
                    ],
                    [
                        -0.05020219 - 0.01123817j,
                        0.04332742 - 0.02637167j,
                        -0.01249301 + 0.04828057j,
                        -0.02442343 - 0.04241919j,
                    ],
                    [
                        -0.04935347 + 0.01439099j,
                        0.02515747 - 0.04399251j,
                        0.01242375 + 0.04824679j,
                        -0.0418769 - 0.02524122j,
                    ],
                ],
                [
                    [
                        -0.09182092 - 0.03251328j,
                        0.0089338 + 0.09750314j,
                        0.08476977 - 0.04854338j,
                        -0.07921425 - 0.05590829j,
                    ],
                    [
                        -0.0961366 + 0.01599552j,
                        0.05499201 + 0.08102092j,
                        0.05061641 - 0.08347884j,
                        -0.09634703 - 0.01039543j,
                    ],
                    [
                        -0.0764313 + 0.06055032j,
                        0.08732548 + 0.0443097j,
                        0.00383588 - 0.09748728j,
                        -0.08922718 + 0.03769643j,
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
                        -0.03856109 - 0.03410628j,
                        0.05072075 - 0.00214221j,
                        -0.03433426 + 0.03623831j,
                        -0.0008072 - 0.04899328j,
                    ],
                    [
                        -0.05020219 - 0.01123817j,
                        0.04332742 - 0.02637167j,
                        -0.01249301 + 0.04828057j,
                        -0.02442343 - 0.04241919j,
                    ],
                    [
                        -0.04935347 + 0.01439099j,
                        0.02515747 - 0.04399251j,
                        0.01242375 + 0.04824679j,
                        -0.0418769 - 0.02524122j,
                    ],
                ],
                [
                    [
                        -0.09182092 - 0.03251328j,
                        0.0089338 + 0.09750314j,
                        0.08476977 - 0.04854338j,
                        -0.07921425 - 0.05590829j,
                    ],
                    [
                        -0.0961366 + 0.01599552j,
                        0.05499201 + 0.08102092j,
                        0.05061641 - 0.08347884j,
                        -0.09634703 - 0.01039543j,
                    ],
                    [
                        -0.0764313 + 0.06055032j,
                        0.08732548 + 0.0443097j,
                        0.00383588 - 0.09748728j,
                        -0.08922718 + 0.03769643j,
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
                        0.00484525 + 0.00551935j,
                        0.00591751 + 0.00434994j,
                        0.00671357 + 0.00297759j,
                        0.00719632 + 0.00146636j,
                    ],
                    [
                        0.00484525 + 0.00551935j,
                        0.00591751 + 0.00434994j,
                        0.00671357 + 0.00297759j,
                        0.00719632 + 0.00146636j,
                    ],
                    [
                        0.00484525 + 0.00551935j,
                        0.00591751 + 0.00434994j,
                        0.00671357 + 0.00297759j,
                        0.00719632 + 0.00146636j,
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
                        0.00047128 + 0.00053808j,
                        0.00057584 + 0.00042431j,
                        0.00065352 + 0.00029074j,
                        0.00070071 + 0.00014361j,
                    ],
                    [
                        0.00047128 + 0.00053808j,
                        0.00057584 + 0.00042431j,
                        0.00065352 + 0.00029074j,
                        0.00070071 + 0.00014361j,
                    ],
                    [
                        0.00047128 + 0.00053808j,
                        0.00057584 + 0.00042431j,
                        0.00065352 + 0.00029074j,
                        0.00070071 + 0.00014361j,
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
                        0.00484213 + 0.00551604j,
                        0.00591375 + 0.00434738j,
                        0.00670935 + 0.0029759j,
                        0.00719184 + 0.00146562j,
                    ],
                    [
                        0.00484213 + 0.00551604j,
                        0.00591375 + 0.00434738j,
                        0.00670935 + 0.0029759j,
                        0.00719184 + 0.00146562j,
                    ],
                    [
                        0.00484213 + 0.00551604j,
                        0.00591375 + 0.00434738j,
                        0.00670935 + 0.0029759j,
                        0.00719184 + 0.00146562j,
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
                        0.00048421 + 0.0005516j,
                        0.00059138 + 0.00043474j,
                        0.00067094 + 0.00029759j,
                        0.00071918 + 0.00014656j,
                    ],
                    [
                        0.00048421 + 0.0005516j,
                        0.00059138 + 0.00043474j,
                        0.00067094 + 0.00029759j,
                        0.00071918 + 0.00014656j,
                    ],
                    [
                        0.00048421 + 0.0005516j,
                        0.00059138 + 0.00043474j,
                        0.00067094 + 0.00029759j,
                        0.00071918 + 0.00014656j,
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
                        0.03579173 + 0.0401197j,
                        0.04361241 + 0.03147846j,
                        0.04938885 + 0.02134942j,
                        0.05284682 + 0.01020823j,
                    ],
                    [
                        0.03579173 + 0.0401197j,
                        0.04361241 + 0.03147846j,
                        0.04938885 + 0.02134942j,
                        0.05284682 + 0.01020823j,
                    ],
                    [
                        0.03579173 + 0.0401197j,
                        0.04361241 + 0.03147846j,
                        0.04938885 + 0.02134942j,
                        0.05284682 + 0.01020823j,
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
                        0.00342355 + 0.00368116j,
                        0.00412664 + 0.00286632j,
                        0.00464013 + 0.00192124j,
                        0.00494085 + 0.00088919j,
                    ],
                    [
                        0.00342355 + 0.00368116j,
                        0.00412664 + 0.00286632j,
                        0.00464013 + 0.00192124j,
                        0.00494085 + 0.00088919j,
                    ],
                    [
                        0.00342355 + 0.00368116j,
                        0.00412664 + 0.00286632j,
                        0.00464013 + 0.00192124j,
                        0.00494085 + 0.00088919j,
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
                        0.03579162 + 0.04011977j,
                        0.04361232 + 0.03147856j,
                        0.04938879 + 0.02134955j,
                        0.05284679 + 0.01020837j,
                    ],
                    [
                        0.03579162 + 0.04011977j,
                        0.04361232 + 0.03147856j,
                        0.04938879 + 0.02134955j,
                        0.05284679 + 0.01020837j,
                    ],
                    [
                        0.03579162 + 0.04011977j,
                        0.04361232 + 0.03147856j,
                        0.04938879 + 0.02134955j,
                        0.05284679 + 0.01020837j,
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
                        0.03423574 + 0.0368114j,
                        0.04126665 + 0.02866299j,
                        0.04640145 + 0.01921216j,
                        0.04940858 + 0.0088916j,
                    ],
                    [
                        0.03423574 + 0.0368114j,
                        0.04126665 + 0.02866299j,
                        0.04640145 + 0.01921216j,
                        0.04940858 + 0.0088916j,
                    ],
                    [
                        0.03423574 + 0.0368114j,
                        0.04126665 + 0.02866299j,
                        0.04640145 + 0.01921216j,
                        0.04940858 + 0.0088916j,
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
                        -0.03856109 - 0.03410628j,
                        0.05072873 + 0.0021258j,
                        -0.03993564 + 0.02998089j,
                        0.01148623 - 0.04766143j,
                    ],
                    [
                        -0.02130391 - 0.04683149j,
                        0.04542939 + 0.022589j,
                        -0.0486411 + 0.01111299j,
                        0.02986106 - 0.03882663j,
                    ],
                    [
                        -0.00037187 - 0.05141711j,
                        0.03226388 + 0.039106j,
                        -0.04890948 - 0.00964638j,
                        0.04303788 - 0.02329335j,
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
                        0.03856109 + 0.03410628j,
                        -0.05072873 - 0.0021258j,
                        0.03993564 - 0.02998089j,
                        -0.01148623 + 0.04766143j,
                    ],
                    [
                        -0.07712218 - 0.06821255j,
                        0.10145745 + 0.00425159j,
                        -0.07987128 + 0.05996178j,
                        0.02297245 - 0.09532286j,
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
                        0.03410628 - 0.03856109j,
                        0.0 + 0.0j,
                        0.15974256 - 0.11992356j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.03410628 - 0.03856109j,
                        0.0 + 0.0j,
                        0.15974256 - 0.11992356j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.03410628 - 0.03856109j,
                        0.0 + 0.0j,
                        0.15974256 - 0.11992356j,
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
                        -0.04125997 - 0.03144338j,
                        -0.03359122 + 0.03620961j,
                        0.04004945 - 0.02654158j,
                        -0.04529067 - 0.00788687j,
                    ],
                    [
                        -0.04125997 - 0.03144338j,
                        -0.03359122 + 0.03620961j,
                        0.04004945 - 0.02654158j,
                        -0.04529067 - 0.00788687j,
                    ],
                    [
                        -0.04125997 - 0.03144338j,
                        -0.03359122 + 0.03620961j,
                        0.04004945 - 0.02654158j,
                        -0.04529067 - 0.00788687j,
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


def test_sim_radar_back_propagating():
    tx_channel = dict(
        location=(0, 0, 0),
    )
    tx = Transmitter(
        f=[1e9 - 50e6, 1e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=1,
        channels=[tx_channel],
    )
    rx_channel = dict(
        location=(0, 0, 0),
    )

    rx = Receiver(
        fs=5e5,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )
    radar = Radar(transmitter=tx, receiver=rx)
    target_1 = {
        "model": "./models/plate5x5.stl",
        "unit": "m",
        "location": (20, -4, 0),
        "speed": (0, 0, 0),
        "rotation_rate": (0, 0, 0),
    }

    target_2 = {
        "model": "./models/plate5x5.stl",
        "unit": "m",
        "location": (21, -2.5, 0),
        "speed": (0, 0, 0),
        "rotation_rate": (0, 0, 0),
    }

    targets = [target_1, target_2]
    data = sim_radar(radar, targets, density=1, back_propagating=True)

    baseband = data["baseband"]

    assert np.allclose(
        np.real(baseband[0, 0, :]),
        np.array(
            [
                -0.61427594,
                0.55812842,
                0.01918325,
                -0.64880974,
                0.79179322,
                -0.31224823,
                -0.39358803,
                0.73453841,
                -0.43702868,
                -0.21755588,
                0.64139605,
                -0.45666401,
                -0.17442796,
                0.69443259,
                -0.6396284,
                0.04264089,
                0.59991126,
                -0.75851556,
                0.32005467,
                0.33000778,
                -0.64990646,
                0.40103341,
                0.16877218,
                -0.55823795,
                0.44401603,
                0.0607404,
                -0.5239286,
                0.56634984,
                -0.16620272,
                -0.34512174,
                0.56819196,
                -0.35245825,
                -0.1055814,
                0.43875585,
                -0.40030594,
                0.04087284,
                0.3484548,
                -0.46495565,
                0.21756568,
                0.21240697,
            ]
        ),
    )

    assert np.allclose(
        np.imag(baseband[0, 0, :]),
        np.array(
            [
                -0.18856427,
                -0.43920477,
                0.76803665,
                -0.47944721,
                -0.21642821,
                0.74796364,
                -0.67347119,
                0.06772982,
                0.54014952,
                -0.62746659,
                0.13362268,
                0.49012179,
                -0.67961622,
                0.25314993,
                0.43110496,
                -0.78906628,
                0.51623585,
                0.1562131,
                -0.66931063,
                0.6166616,
                -0.07857762,
                -0.46756551,
                0.56577337,
                -0.16023915,
                -0.38042723,
                0.59276761,
                -0.30650019,
                -0.23226708,
                0.58457028,
                -0.48206686,
                0.02831972,
                0.40791739,
                -0.49703725,
                0.1977592,
                0.23309925,
                -0.45495379,
                0.30303197,
                0.09914297,
                -0.44285344,
                0.46534106,
            ]
        ),
    )
