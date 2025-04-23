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
                        -0.03540304 - 0.03133887j,
                        0.04679733 + 0.00072464j,
                        -0.03577328 + 0.02984913j,
                        0.00765144 - 0.04602434j,
                    ],
                    [
                        -0.03540304 - 0.03133887j,
                        0.04679733 + 0.00072464j,
                        -0.03577328 + 0.02984913j,
                        0.00765144 - 0.04602434j,
                    ],
                    [
                        -0.03540304 - 0.03133887j,
                        0.04679733 + 0.00072464j,
                        -0.03577328 + 0.02984913j,
                        0.00765144 - 0.04602434j,
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
                        -0.03540304 - 0.03133887j,
                        0.04625038 - 0.00714302j,
                        -0.0238634 + 0.04001857j,
                        -0.01566209 - 0.04397004j,
                    ],
                    [
                        -0.0437161 + 0.0178537j,
                        0.01400299 - 0.04462831j,
                        0.02541567 + 0.03906531j,
                        -0.04642346 - 0.00528753j,
                    ],
                    [
                        0.00618424 + 0.04674899j,
                        -0.03963782 - 0.02478723j,
                        0.04421292 - 0.0148035j,
                        -0.01675736 + 0.04368097j,
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
                        -0.0320607 - 0.03473966j,
                        0.04491183 + 0.01316545j,
                        -0.0450605 + 0.01182365j,
                        0.03255919 - 0.03339039j,
                    ],
                    [
                        0.01243482 - 0.04566802j,
                        0.01264253 + 0.04509077j,
                        -0.03387295 - 0.03197584j,
                        0.04551896 + 0.00998337j,
                    ],
                    [
                        0.04536072 - 0.01372567j,
                        -0.03156033 + 0.03464001j,
                        0.00920943 - 0.04566147j,
                        0.01557146 + 0.0438878j,
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
                        0.01225709 + 0.10392048j,
                        -0.01345596 + 0.08654514j,
                        -0.02670977 + 0.06631701j,
                        -0.02981075 + 0.04923867j,
                    ],
                    [
                        0.01225709 + 0.10392048j,
                        -0.01345596 + 0.08654514j,
                        -0.02670977 + 0.06631701j,
                        -0.02981075 + 0.04923867j,
                    ],
                    [
                        0.01225709 + 0.10392048j,
                        -0.01345596 + 0.08654514j,
                        -0.02670977 + 0.06631701j,
                        -0.02981075 + 0.04923867j,
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
                        -0.00949312 + 0.00354618j,
                        -0.0061834 - 0.00358195j,
                        0.00046128 - 0.00751956j,
                        0.00840201 - 0.0069371j,
                    ],
                    [
                        -0.00949312 + 0.00354618j,
                        -0.0061834 - 0.00358195j,
                        0.00046128 - 0.00751956j,
                        0.00840201 - 0.0069371j,
                    ],
                    [
                        -0.00949312 + 0.00354618j,
                        -0.0061834 - 0.00358195j,
                        0.00046128 - 0.00751956j,
                        0.00840201 - 0.0069371j,
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
            "model": "./models/cr.stl",
            "location": np.array([10, 10, 0]),
            "rotation": np.array([45, 0, 0]),
        },
        {
            "model": "./models/cr.stl",
            "location": np.array([10, -10, 0]),
            "rotation": np.array([-45, 0, 0]),
        },
    ]
    result = sim_radar(radar, targets, density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.0023177 + 0.00256185j,
                        0.00281368 + 0.00200431j,
                        0.00317835 + 0.00135329j,
                        0.0033947 + 0.00063919j,
                    ],
                    [
                        0.0023177 + 0.00256185j,
                        0.00281368 + 0.00200431j,
                        0.00317835 + 0.00135329j,
                        0.0033947 + 0.00063919j,
                    ],
                    [
                        0.0023177 + 0.00256185j,
                        0.00281368 + 0.00200431j,
                        0.00317835 + 0.00135329j,
                        0.0033947 + 0.00063919j,
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
                        -0.03540304 - 0.03133887j,
                        0.04625038 - 0.00714302j,
                        -0.0238634 + 0.04001857j,
                        -0.01566209 - 0.04397004j,
                    ],
                    [
                        -0.04530013 + 0.01335092j,
                        0.01844213 - 0.04298666j,
                        0.02133356 + 0.04143447j,
                        -0.04564559 - 0.00995776j,
                    ],
                    [
                        -0.01271831 + 0.0454289j,
                        -0.02667136 - 0.03840332j,
                        0.04646124 + 0.00382609j,
                        -0.03261587 + 0.03351443j,
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
                        -0.09791079 - 0.02653906j,
                        0.00663815 + 0.10235053j,
                        0.09479425 - 0.03925132j,
                        -0.0675829 - 0.07611783j,
                    ],
                    [
                        -0.09791079 - 0.02653906j,
                        0.00663815 + 0.10235053j,
                        0.09479425 - 0.03925132j,
                        -0.0675829 - 0.07611783j,
                    ],
                    [
                        -0.09791079 - 0.02653906j,
                        0.00663815 + 0.10235053j,
                        0.09479425 - 0.03925132j,
                        -0.0675829 - 0.07611783j,
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
                        -0.03540304 - 0.03133887j,
                        0.04625038 - 0.00714302j,
                        -0.0238634 + 0.04001857j,
                        -0.01566209 - 0.04397004j,
                    ],
                    [
                        -0.04530013 + 0.01335092j,
                        0.01844213 - 0.04298666j,
                        0.02133356 + 0.04143447j,
                        -0.04564559 - 0.00995776j,
                    ],
                    [
                        -0.01271831 + 0.0454289j,
                        -0.02667136 - 0.03840332j,
                        0.04646124 + 0.00382609j,
                        -0.03261587 + 0.03351443j,
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
                        -0.03540304 - 0.03133887j,
                        0.04668985 - 0.00322081j,
                        -0.03024872 + 0.03543774j,
                        -0.00413497 - 0.04648244j,
                    ],
                    [
                        -0.04612484 - 0.0102655j,
                        0.03926417 - 0.02544608j,
                        -0.00925605 + 0.04566796j,
                        -0.02619322 - 0.03864651j,
                    ],
                    [
                        -0.04530013 + 0.01335092j,
                        0.02199936 - 0.04128146j,
                        0.01406603 + 0.04442852j,
                        -0.04168061 - 0.02108009j,
                    ],
                ],
                [
                    [
                        -0.09791079 - 0.02653906j,
                        0.01518764 + 0.10143689j,
                        0.08687339 - 0.05452907j,
                        -0.08440337 - 0.05677608j,
                    ],
                    [
                        -0.09867485 + 0.02398333j,
                        0.06221883 + 0.08154832j,
                        0.04969976 - 0.0896141j,
                        -0.10120215 - 0.00882054j,
                    ],
                    [
                        -0.07495844 + 0.06864948j,
                        0.09382466 + 0.04144032j,
                        0.00024459 - 0.10237003j,
                        -0.09270037 + 0.04121612j,
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
                        -0.03540304 - 0.03133887j,
                        0.04668985 - 0.00322081j,
                        -0.03024872 + 0.03543774j,
                        -0.00413497 - 0.04648244j,
                    ],
                    [
                        -0.04612484 - 0.0102655j,
                        0.03926417 - 0.02544608j,
                        -0.00925605 + 0.04566796j,
                        -0.02619322 - 0.03864651j,
                    ],
                    [
                        -0.04530013 + 0.01335092j,
                        0.02199936 - 0.04128146j,
                        0.01406603 + 0.04442852j,
                        -0.04168061 - 0.02108009j,
                    ],
                ],
                [
                    [
                        -0.09791079 - 0.02653906j,
                        0.01518764 + 0.10143689j,
                        0.08687339 - 0.05452907j,
                        -0.08440337 - 0.05677608j,
                    ],
                    [
                        -0.09867485 + 0.02398333j,
                        0.06221883 + 0.08154832j,
                        0.04969976 - 0.0896141j,
                        -0.10120215 - 0.00882054j,
                    ],
                    [
                        -0.07495844 + 0.06864948j,
                        0.09382466 + 0.04144032j,
                        0.00024459 - 0.10237003j,
                        -0.09270037 + 0.04121612j,
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
                        -0.03540304 - 0.03133887j,
                        0.04679733 + 0.00072464j,
                        -0.03577328 + 0.02984913j,
                        0.00765144 - 0.04602434j,
                    ],
                    [
                        -0.01959625 - 0.04299209j,
                        0.04244585 + 0.01966617j,
                        -0.04480454 + 0.01274007j,
                        0.02569066 - 0.03894865j,
                    ],
                    [
                        -0.00043171 - 0.04721252j,
                        0.03078007 + 0.03519858j,
                        -0.04610703 - 0.00655824j,
                        0.0393008 - 0.02515413j,
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
                        0.03540304 + 0.03133887j,
                        -0.04679733 - 0.00072464j,
                        0.03577328 - 0.02984913j,
                        -0.00765144 + 0.04602434j,
                    ],
                    [
                        -0.07080608 - 0.06267773j,
                        0.09359466 + 0.00144928j,
                        -0.07154657 + 0.05969827j,
                        0.01530288 - 0.09204868j,
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
                        0.03133887 - 0.03540304j,
                        0.0 + 0.0j,
                        0.14309313 - 0.11939654j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.03133887 - 0.03540304j,
                        0.0 + 0.0j,
                        0.14309313 - 0.11939654j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.03133887 - 0.03540304j,
                        0.0 + 0.0j,
                        0.14309313 - 0.11939654j,
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
                        -0.04585536 - 0.03628072j,
                        -0.03983022 + 0.03443156j,
                        0.03388717 - 0.02047655j,
                        -0.0430065 + 0.00116208j,
                    ],
                    [
                        -0.04585536 - 0.03628072j,
                        -0.03983022 + 0.03443156j,
                        0.03388717 - 0.02047655j,
                        -0.0430065 + 0.00116208j,
                    ],
                    [
                        -0.04585536 - 0.03628072j,
                        -0.03983022 + 0.03443156j,
                        0.03388717 - 0.02047655j,
                        -0.0430065 + 0.00116208j,
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
