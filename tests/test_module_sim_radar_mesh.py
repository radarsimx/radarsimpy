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
import pytest

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module

# Every test here loads an .stl model.
pytestmark = pytest.mark.mesh


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
                        -0.03540167 - 0.03127643j,
                        0.04677352 + 0.00066665j,
                        -0.03572978 + 0.02989377j,
                        0.00759479 - 0.04604838j,
                    ],
                    [
                        -0.03540167 - 0.03127643j,
                        0.04677352 + 0.00066665j,
                        -0.03572978 + 0.02989377j,
                        0.00759479 - 0.04604838j,
                    ],
                    [
                        -0.03540167 - 0.03127643j,
                        0.04677352 + 0.00066665j,
                        -0.03572978 + 0.02989377j,
                        0.00759479 - 0.04604838j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04677352 + 0.00066665j,
                        -0.03572978 + 0.02989377j,
                        0.00759479 - 0.04604838j,
                    ],
                    [
                        -0.03540167 - 0.03127643j,
                        0.04677352 + 0.00066665j,
                        -0.03572978 + 0.02989377j,
                        0.00759479 - 0.04604838j,
                    ],
                    [
                        -0.03540167 - 0.03127643j,
                        0.04677352 + 0.00066665j,
                        -0.03572978 + 0.02989377j,
                        0.00759479 - 0.04604838j,
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
                        -0.03540167 - 0.03127643j,
                        0.04621743 - 0.00719637j,
                        -0.0238078 + 0.04004681j,
                        -0.01572339 - 0.0439644j,
                    ],
                    [
                        -0.04366022 + 0.01788213j,
                        0.01394025 - 0.04462434j,
                        0.02546675 + 0.03902939j,
                        -0.04644711 - 0.00523064j,
                    ],
                    [
                        0.00622795 + 0.04670366j,
                        -0.03965221 - 0.02472586j,
                        0.04419312 - 0.01486283j,
                        -0.01670951 + 0.04371985j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04621743 - 0.00719637j,
                        -0.0238078 + 0.04004681j,
                        -0.01572339 - 0.0439644j,
                    ],
                    [
                        -0.04366022 + 0.01788213j,
                        0.01394025 - 0.04462434j,
                        0.02546675 + 0.03902939j,
                        -0.04644711 - 0.00523064j,
                    ],
                    [
                        0.00622795 + 0.04670366j,
                        -0.03965221 - 0.02472586j,
                        0.04419312 - 0.01486283j,
                        -0.01670951 + 0.04371985j,
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
                        -0.03206592 - 0.03467743j,
                        0.04490432 + 0.01310326j,
                        -0.04503979 + 0.01188239j,
                        0.03252597 - 0.03344213j,
                    ],
                    [
                        0.01238044 - 0.04563781j,
                        0.01268999 + 0.04505009j,
                        -0.03391038 - 0.03192616j,
                        0.04554376 + 0.00992716j,
                    ],
                    [
                        0.04530554 - 0.01375389j,
                        -0.0315003 + 0.03465687j,
                        0.00914742 - 0.04566514j,
                        0.01563198 + 0.04387747j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03206592 - 0.03467743j,
                        0.04490432 + 0.01310326j,
                        -0.04503979 + 0.01188239j,
                        0.03252597 - 0.03344213j,
                    ],
                    [
                        0.01238044 - 0.04563781j,
                        0.01268999 + 0.04505009j,
                        -0.03391038 - 0.03192616j,
                        0.04554376 + 0.00992716j,
                    ],
                    [
                        0.04530554 - 0.01375389j,
                        -0.0315003 + 0.03465687j,
                        0.00914742 - 0.04566514j,
                        0.01563198 + 0.04387747j,
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
                        0.01557409 + 0.10299858j,
                        -0.0101409 + 0.08737491j,
                        -0.02430757 + 0.06864029j,
                        -0.02894874 + 0.05233632j,
                    ],
                    [
                        0.01557409 + 0.10299858j,
                        -0.0101409 + 0.08737491j,
                        -0.02430757 + 0.06864029j,
                        -0.02894874 + 0.05233632j,
                    ],
                    [
                        0.01557409 + 0.10299858j,
                        -0.0101409 + 0.08737491j,
                        -0.02430757 + 0.06864029j,
                        -0.02894874 + 0.05233632j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.01557409 + 0.10299858j,
                        -0.0101409 + 0.08737491j,
                        -0.02430757 + 0.06864029j,
                        -0.02894874 + 0.05233632j,
                    ],
                    [
                        0.01557409 + 0.10299858j,
                        -0.0101409 + 0.08737491j,
                        -0.02430757 + 0.06864029j,
                        -0.02894874 + 0.05233632j,
                    ],
                    [
                        0.01557409 + 0.10299858j,
                        -0.0101409 + 0.08737491j,
                        -0.02430757 + 0.06864029j,
                        -0.02894874 + 0.05233632j,
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
                        -0.00934485 + 0.00320899j,
                        -0.00585369 - 0.00381428j,
                        0.00089323 - 0.00755166j,
                        0.00881122 - 0.0067338j,
                    ],
                    [
                        -0.00934485 + 0.00320899j,
                        -0.00585369 - 0.00381428j,
                        0.00089323 - 0.00755166j,
                        0.00881122 - 0.0067338j,
                    ],
                    [
                        -0.00934485 + 0.00320899j,
                        -0.00585369 - 0.00381428j,
                        0.00089323 - 0.00755166j,
                        0.00881122 - 0.0067338j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.00934485 + 0.00320899j,
                        -0.00585369 - 0.00381428j,
                        0.00089323 - 0.00755166j,
                        0.00881122 - 0.0067338j,
                    ],
                    [
                        -0.00934485 + 0.00320899j,
                        -0.00585369 - 0.00381428j,
                        0.00089323 - 0.00755166j,
                        0.00881122 - 0.0067338j,
                    ],
                    [
                        -0.00934485 + 0.00320899j,
                        -0.00585369 - 0.00381428j,
                        0.00089323 - 0.00755166j,
                        0.00881122 - 0.0067338j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

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
                        -0.03540167 - 0.03127643j,
                        0.04621743 - 0.00719637j,
                        -0.0238078 + 0.04004681j,
                        -0.01572339 - 0.0439644j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.01837932 - 0.04298887j,
                        0.02138791 + 0.04140375j,
                        -0.04567472 - 0.00990348j,
                    ],
                    [
                        -0.01266066 + 0.04540374j,
                        -0.02670813 - 0.03835219j,
                        0.04646572 + 0.00376373j,
                        -0.03258665 + 0.03356871j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04621743 - 0.00719637j,
                        -0.0238078 + 0.04004681j,
                        -0.01572339 - 0.0439644j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.01837932 - 0.04298887j,
                        0.02138791 + 0.04140375j,
                        -0.04567472 - 0.00990348j,
                    ],
                    [
                        -0.01266066 + 0.04540374j,
                        -0.02670813 - 0.03835219j,
                        0.04646572 + 0.00376373j,
                        -0.03258665 + 0.03356871j,
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
                        -0.09755995 - 0.02757895j,
                        0.00554437 + 0.10182749j,
                        0.09399094 - 0.03817976j,
                        -0.06669428 - 0.07496794j,
                    ],
                    [
                        -0.09755995 - 0.02757895j,
                        0.00554437 + 0.10182749j,
                        0.09399094 - 0.03817976j,
                        -0.06669428 - 0.07496794j,
                    ],
                    [
                        -0.09755995 - 0.02757895j,
                        0.00554437 + 0.10182749j,
                        0.09399094 - 0.03817976j,
                        -0.06669428 - 0.07496794j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.09755995 - 0.02757895j,
                        0.00554437 + 0.10182749j,
                        0.09399094 - 0.03817976j,
                        -0.06669428 - 0.07496794j,
                    ],
                    [
                        -0.09755995 - 0.02757895j,
                        0.00554437 + 0.10182749j,
                        0.09399094 - 0.03817976j,
                        -0.06669428 - 0.07496794j,
                    ],
                    [
                        -0.09755995 - 0.02757895j,
                        0.00554437 + 0.10182749j,
                        0.09399094 - 0.03817976j,
                        -0.06669428 - 0.07496794j,
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
                        -0.03540167 - 0.03127643j,
                        0.04621743 - 0.00719637j,
                        -0.0238078 + 0.04004681j,
                        -0.01572339 - 0.0439644j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.01837932 - 0.04298887j,
                        0.02138791 + 0.04140375j,
                        -0.04567472 - 0.00990348j,
                    ],
                    [
                        -0.01266066 + 0.04540374j,
                        -0.02670813 - 0.03835219j,
                        0.04646572 + 0.00376373j,
                        -0.03258665 + 0.03356871j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04621743 - 0.00719637j,
                        -0.0238078 + 0.04004681j,
                        -0.01572339 - 0.0439644j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.01837932 - 0.04298887j,
                        0.02138791 + 0.04140375j,
                        -0.04567472 - 0.00990348j,
                    ],
                    [
                        -0.01266066 + 0.04540374j,
                        -0.02670813 - 0.03835219j,
                        0.04646572 + 0.00376373j,
                        -0.03258665 + 0.03356871j,
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
    radar = Radar(transmitter=tx, receiver=rx, frame_time=[0, 1])

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([-5, 0, 0]),
        }
    ]
    result = sim_radar(radar, targets, density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04666138 - 0.00327668j,
                        -0.0301985 + 0.03547467j,
                        -0.00419578 - 0.04649192j,
                    ],
                    [
                        -0.04609418 - 0.01021096j,
                        0.03921265 - 0.02548195j,
                        -0.0091943 + 0.04567678j,
                        -0.0262513 - 0.03862607j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.02193695 - 0.0412888j,
                        0.01412466 + 0.04440709j,
                        -0.0417221 - 0.02103458j,
                    ],
                ],
                [
                    [
                        -0.09755995 - 0.02757895j,
                        0.01405578 + 0.10099723j,
                        0.08623647 - 0.05334727j,
                        -0.08328192 - 0.055846j,
                    ],
                    [
                        -0.09882261 + 0.0228863j,
                        0.06099774 + 0.08164928j,
                        0.04963519 - 0.08826354j,
                        -0.09978416 - 0.00845569j,
                    ],
                    [
                        -0.07558215 + 0.06772331j,
                        0.09276116 + 0.04207044j,
                        0.00077243 - 0.10111513j,
                        -0.09125515 + 0.04094099j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04666138 - 0.00327668j,
                        -0.0301985 + 0.03547467j,
                        -0.00419578 - 0.04649192j,
                    ],
                    [
                        -0.04609418 - 0.01021096j,
                        0.03921265 - 0.02548195j,
                        -0.0091943 + 0.04567678j,
                        -0.0262513 - 0.03862607j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.02193695 - 0.0412888j,
                        0.01412466 + 0.04440709j,
                        -0.0417221 - 0.02103458j,
                    ],
                ],
                [
                    [
                        -0.09755995 - 0.02757895j,
                        0.01405578 + 0.10099723j,
                        0.08623647 - 0.05334727j,
                        -0.08328192 - 0.055846j,
                    ],
                    [
                        -0.09882261 + 0.0228863j,
                        0.06099774 + 0.08164928j,
                        0.04963519 - 0.08826354j,
                        -0.09978416 - 0.00845569j,
                    ],
                    [
                        -0.07558215 + 0.06772331j,
                        0.09276116 + 0.04207044j,
                        0.00077243 - 0.10111513j,
                        -0.09125515 + 0.04094099j,
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
    radar = Radar(transmitter=tx, receiver=rx, speed=[5, 0, 0], frame_time=[0, 1])

    targets = [
        {
            "model": "./models/plate5x5.stl",
            "location": np.array([10, 0, 0]),
            "speed": np.array([0, 0, 0]),
        }
    ]
    result = sim_radar(radar, targets, density=0.4)

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04666138 - 0.00327668j,
                        -0.0301985 + 0.03547467j,
                        -0.00419578 - 0.04649192j,
                    ],
                    [
                        -0.04609418 - 0.01021096j,
                        0.03921265 - 0.02548195j,
                        -0.0091943 + 0.04567678j,
                        -0.0262513 - 0.03862607j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.02193695 - 0.0412888j,
                        0.01412466 + 0.04440709j,
                        -0.0417221 - 0.02103458j,
                    ],
                ],
                [
                    [
                        -0.09755995 - 0.02757895j,
                        0.01405578 + 0.10099723j,
                        0.08623647 - 0.05334727j,
                        -0.08328192 - 0.055846j,
                    ],
                    [
                        -0.09882261 + 0.0228863j,
                        0.06099774 + 0.08164928j,
                        0.04963519 - 0.08826354j,
                        -0.09978416 - 0.00845569j,
                    ],
                    [
                        -0.07558215 + 0.06772331j,
                        0.09276116 + 0.04207044j,
                        0.00077243 - 0.10111513j,
                        -0.09125515 + 0.04094099j,
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

    result = sim_radar(radar, targets, density=0.4, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03540167 - 0.03127643j,
                        0.04666138 - 0.00327668j,
                        -0.0301985 + 0.03547467j,
                        -0.00419578 - 0.04649192j,
                    ],
                    [
                        -0.04609418 - 0.01021096j,
                        0.03921265 - 0.02548195j,
                        -0.0091943 + 0.04567678j,
                        -0.0262513 - 0.03862607j,
                    ],
                    [
                        -0.04524732 + 0.01338468j,
                        0.02193695 - 0.0412888j,
                        0.01412466 + 0.04440709j,
                        -0.0417221 - 0.02103458j,
                    ],
                ],
                [
                    [
                        -0.09755995 - 0.02757895j,
                        0.01405578 + 0.10099723j,
                        0.08623647 - 0.05334727j,
                        -0.08328192 - 0.055846j,
                    ],
                    [
                        -0.09882261 + 0.0228863j,
                        0.06099774 + 0.08164928j,
                        0.04963519 - 0.08826354j,
                        -0.09978416 - 0.00845569j,
                    ],
                    [
                        -0.07558215 + 0.06772331j,
                        0.09276116 + 0.04207044j,
                        0.00077243 - 0.10111513j,
                        -0.09125515 + 0.04094099j,
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

    result = sim_radar(radar, targets, density=1, device="cpu")

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

    result = sim_radar(radar, targets, density=1, device="cpu")

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

    result = sim_radar(radar, targets, density=1, device="cpu")

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

    result = sim_radar(radar, targets, density=1, device="cpu")

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
                        0.02491453 + 0.02870054j,
                        0.03053584 + 0.02266598j,
                        0.03472517 + 0.01555537j,
                        0.03728249 + 0.00770294j,
                    ],
                    [
                        0.02491453 + 0.02870054j,
                        0.03053584 + 0.02266598j,
                        0.03472517 + 0.01555537j,
                        0.03728249 + 0.00770294j,
                    ],
                    [
                        0.02491453 + 0.02870054j,
                        0.03053584 + 0.02266598j,
                        0.03472517 + 0.01555537j,
                        0.03728249 + 0.00770294j,
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

    result = sim_radar(radar, targets, density=1, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02491453 + 0.02870054j,
                        0.03053584 + 0.02266598j,
                        0.03472517 + 0.01555537j,
                        0.03728249 + 0.00770294j,
                    ],
                    [
                        0.02491453 + 0.02870054j,
                        0.03053584 + 0.02266598j,
                        0.03472517 + 0.01555537j,
                        0.03728249 + 0.00770294j,
                    ],
                    [
                        0.02491453 + 0.02870054j,
                        0.03053584 + 0.02266598j,
                        0.03472517 + 0.01555537j,
                        0.03728249 + 0.00770294j,
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
                        0.00239357 + 0.00259641j,
                        0.00288837 + 0.00202527j,
                        0.00325007 + 0.00136287j,
                        0.00346258 + 0.00063961j,
                    ],
                    [
                        0.00239357 + 0.00259641j,
                        0.00288837 + 0.00202527j,
                        0.00325007 + 0.00136287j,
                        0.00346258 + 0.00063961j,
                    ],
                    [
                        0.00239357 + 0.00259641j,
                        0.00288837 + 0.00202527j,
                        0.00325007 + 0.00136287j,
                        0.00346258 + 0.00063961j,
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

    result = sim_radar(radar, targets, density=1, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00239357 + 0.00259641j,
                        0.00288837 + 0.00202527j,
                        0.00325007 + 0.00136287j,
                        0.00346258 + 0.00063961j,
                    ],
                    [
                        0.00239357 + 0.00259641j,
                        0.00288837 + 0.00202527j,
                        0.00325007 + 0.00136287j,
                        0.00346258 + 0.00063961j,
                    ],
                    [
                        0.00239357 + 0.00259641j,
                        0.00288837 + 0.00202527j,
                        0.00325007 + 0.00136287j,
                        0.00346258 + 0.00063961j,
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
                        0.02491447 + 0.02870062j,
                        0.03053581 + 0.02266607j,
                        0.03472516 + 0.01555547j,
                        0.0372825 + 0.00770304j,
                    ],
                    [
                        0.02491447 + 0.02870062j,
                        0.03053581 + 0.02266607j,
                        0.03472516 + 0.01555547j,
                        0.0372825 + 0.00770304j,
                    ],
                    [
                        0.02491447 + 0.02870062j,
                        0.03053581 + 0.02266607j,
                        0.03472516 + 0.01555547j,
                        0.0372825 + 0.00770304j,
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

    result = sim_radar(radar, targets, density=1, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02491447 + 0.02870062j,
                        0.03053581 + 0.02266607j,
                        0.03472516 + 0.01555547j,
                        0.0372825 + 0.00770304j,
                    ],
                    [
                        0.02491447 + 0.02870062j,
                        0.03053581 + 0.02266607j,
                        0.03472516 + 0.01555547j,
                        0.0372825 + 0.00770304j,
                    ],
                    [
                        0.02491447 + 0.02870062j,
                        0.03053581 + 0.02266607j,
                        0.03472516 + 0.01555547j,
                        0.0372825 + 0.00770304j,
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
                        0.00239359 + 0.0025964j,
                        0.00288839 + 0.00202525j,
                        0.00325008 + 0.00136285j,
                        0.00346259 + 0.00063959j,
                    ],
                    [
                        0.00239359 + 0.0025964j,
                        0.00288839 + 0.00202525j,
                        0.00325008 + 0.00136285j,
                        0.00346259 + 0.00063959j,
                    ],
                    [
                        0.00239359 + 0.0025964j,
                        0.00288839 + 0.00202525j,
                        0.00325008 + 0.00136285j,
                        0.00346259 + 0.00063959j,
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

    result = sim_radar(radar, targets, density=1, device="cpu")

    assert np.allclose(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.00239359 + 0.0025964j,
                        0.00288839 + 0.00202525j,
                        0.00325008 + 0.00136285j,
                        0.00346259 + 0.00063959j,
                    ],
                    [
                        0.00239359 + 0.0025964j,
                        0.00288839 + 0.00202525j,
                        0.00325008 + 0.00136285j,
                        0.00346259 + 0.00063959j,
                    ],
                    [
                        0.00239359 + 0.0025964j,
                        0.00288839 + 0.00202525j,
                        0.00325008 + 0.00136285j,
                        0.00346259 + 0.00063959j,
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
                        -0.03540167 - 0.03127643j,
                        0.04677352 + 0.00066665j,
                        -0.03572978 + 0.02989377j,
                        0.00759479 - 0.04604838j,
                    ],
                    [
                        -0.0196215 - 0.04293502j,
                        0.04244887 + 0.01960362j,
                        -0.04478404 + 0.01279884j,
                        0.02564954 - 0.03899429j,
                    ],
                    [
                        -0.00047877 - 0.04717159j,
                        0.03080927 + 0.03514327j,
                        -0.0461133 - 0.00649641j,
                        0.03928284 - 0.02521275j,
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
                        0.03540167 + 0.03127643j,
                        -0.04677352 - 0.00066665j,
                        0.03572978 - 0.02989377j,
                        -0.00759479 + 0.04604838j,
                    ],
                    [
                        -0.07080333 - 0.06255285j,
                        0.09354705 + 0.00133329j,
                        -0.07145955 + 0.05978754j,
                        0.01518959 - 0.09209675j,
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
                        0.03127643 - 0.03540167j,
                        0.0 + 0.0j,
                        0.1429191 - 0.11957508j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.03127643 - 0.03540167j,
                        0.0 + 0.0j,
                        0.1429191 - 0.11957508j,
                        0.0 + 0.0j,
                    ],
                    [
                        0.03127643 - 0.03540167j,
                        0.0 + 0.0j,
                        0.1429191 - 0.11957508j,
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
                        -0.04584232 - 0.03613872j,
                        -0.03978771 + 0.03449701j,
                        0.03400982 - 0.02045927j,
                        -0.04280288 + 0.0008302j,
                    ],
                    [
                        -0.04584232 - 0.03613872j,
                        -0.03978771 + 0.03449701j,
                        0.03400982 - 0.02045927j,
                        -0.04280288 + 0.0008302j,
                    ],
                    [
                        -0.04584232 - 0.03613872j,
                        -0.03978771 + 0.03449701j,
                        0.03400982 - 0.02045927j,
                        -0.04280288 + 0.0008302j,
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

    result = sim_radar(
        radar, targets, density=0.4, interf=interference_radar, device="cpu"
    )

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
                -0.61914586,
                0.59318893,
                -0.01438821,
                -0.62321187,
                0.75998025,
                -0.26260408,
                -0.44854804,
                0.75758219,
                -0.38920601,
                -0.34693133,
                0.82103944,
                -0.62779194,
                -0.06443096,
                0.66264698,
                -0.66294523,
                0.07980941,
                0.57731321,
                -0.74993517,
                0.30599996,
                0.36153403,
                -0.68371335,
                0.39978552,
                0.23679762,
                -0.69011452,
                0.59642604,
                -0.05395880,
                -0.48006074,
                0.57679262,
                -0.17680538,
                -0.38319726,
                0.65519232,
                -0.43104678,
                -0.11242055,
                0.56436924,
                -0.58903841,
                0.15705666,
                0.45460573,
                -0.87060617,
                0.87294265,
                -0.53022747,
            ]
        ),
    )

    assert np.allclose(
        np.imag(baseband[0, 0, :]),
        np.array(
            [
                -0.34044920,
                -0.33958327,
                0.70991723,
                -0.44016340,
                -0.24578526,
                0.75234344,
                -0.62786871,
                -0.03583217,
                0.67511494,
                -0.74067671,
                0.17524406,
                0.53547790,
                -0.78321593,
                0.36112956,
                0.36387409,
                -0.77470453,
                0.53577832,
                0.12844688,
                -0.63979801,
                0.56672295,
                0.01660447,
                -0.61076177,
                0.72477027,
                -0.28356579,
                -0.32844358,
                0.60390814,
                -0.33098121,
                -0.25148677,
                0.66923062,
                -0.60064566,
                0.11993786,
                0.38110024,
                -0.50616029,
                0.14802364,
                0.44972382,
                -0.88431841,
                0.87968218,
                -0.46171300,
                -0.08375912,
                0.42229769,
            ]
        ),
    )

    data = sim_radar(radar, targets, density=1, back_propagating=True, device="cpu")

    baseband = data["baseband"]

    assert np.allclose(
        np.real(baseband[0, 0, :]),
        np.array(
            [
                -0.61914586,
                0.59318893,
                -0.01438821,
                -0.62321187,
                0.75998025,
                -0.26260408,
                -0.44854804,
                0.75758219,
                -0.38920601,
                -0.34693133,
                0.82103944,
                -0.62779194,
                -0.06443096,
                0.66264698,
                -0.66294523,
                0.07980941,
                0.57731321,
                -0.74993517,
                0.30599996,
                0.36153403,
                -0.68371335,
                0.39978552,
                0.23679762,
                -0.69011452,
                0.59642604,
                -0.05395880,
                -0.48006074,
                0.57679262,
                -0.17680538,
                -0.38319726,
                0.65519232,
                -0.43104678,
                -0.11242055,
                0.56436924,
                -0.58903841,
                0.15705666,
                0.45460573,
                -0.87060617,
                0.87294265,
                -0.53022747,
            ]
        ),
    )

    assert np.allclose(
        np.imag(baseband[0, 0, :]),
        np.array(
            [
                -0.34044920,
                -0.33958327,
                0.70991723,
                -0.44016340,
                -0.24578526,
                0.75234344,
                -0.62786871,
                -0.03583217,
                0.67511494,
                -0.74067671,
                0.17524406,
                0.53547790,
                -0.78321593,
                0.36112956,
                0.36387409,
                -0.77470453,
                0.53577832,
                0.12844688,
                -0.63979801,
                0.56672295,
                0.01660447,
                -0.61076177,
                0.72477027,
                -0.28356579,
                -0.32844358,
                0.60390814,
                -0.33098121,
                -0.25148677,
                0.66923062,
                -0.60064566,
                0.11993786,
                0.38110024,
                -0.50616029,
                0.14802364,
                0.44972382,
                -0.88431841,
                0.87968218,
                -0.46171300,
                -0.08375912,
                0.42229769,
            ]
        ),
    )
