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

from functools import partial

import numpy as np

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module

from tests.conftest import IDEAL_BASEBAND_PEAK_ATOL
from tests.conftest import assert_baseband_close as _assert_baseband_close

#: Compare baseband against the recorded values relative to their peak rather
#: than element-wise: a fixed absolute bound is simultaneously too tight on the
#: samples near a null and too loose on the ones carrying the signal.
#:
#: The arrays below are recorded at full round-trip precision, so they
#: reproduce exactly on the device they were captured on. See
#: IDEAL_BASEBAND_PEAK_ATOL for what the tolerance is actually covering.
assert_baseband_close = partial(
    _assert_baseband_close, atol_frac=IDEAL_BASEBAND_PEAK_ATOL
)


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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027893975377082825 + 0.00031773850787431j,
                        0.021273192018270493 - 0.018045110628008842j,
                        -0.004863050766289234 + 0.02746862918138504j,
                    ],
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027893975377082825 + 0.00031773850787431j,
                        0.021273192018270493 - 0.018045110628008842j,
                        -0.004863050766289234 + 0.02746862918138504j,
                    ],
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027893975377082825 + 0.00031773850787431j,
                        0.021273192018270493 - 0.018045110628008842j,
                        -0.004863050766289234 + 0.02746862918138504j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027447368949651718 + 0.004986839834600687j,
                        0.014111651107668877 - 0.024065323173999786j,
                        0.009057712741196156 + 0.026387274265289307j,
                    ],
                    [
                        0.025369323790073395 - 0.011615276336669922j,
                        -0.007712406571954489 + 0.02681581676006317j,
                        -0.015329970046877861 - 0.023315511643886566j,
                        0.027679286897182465 + 0.0035397966857999563j,
                    ],
                    [
                        -0.0047342246398329735 - 0.02750471606850624j,
                        0.023969972506165504 + 0.01429736614227295j,
                        -0.02644996903836727 + 0.008912081830203533j,
                        0.010422983206808567 - 0.025892846286296844j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.01979939453303814 + 0.019650127738714218j,
                        -0.026971425861120224 - 0.007115752901881933j,
                        0.02691655419766903 - 0.00731703219935298j,
                        -0.01966141164302826 + 0.01978425867855549j,
                    ],
                    [
                        -0.006085369735956192 + 0.02721765637397766j,
                        -0.008317025378346443 - 0.0266196858137846j,
                        0.020491179078817368 + 0.018916668370366096j,
                        -0.02719477377831936 - 0.006174241192638874j,
                    ],
                    [
                        -0.0262721199542284 + 0.009343295358121395j,
                        0.018115684390068054 - 0.021196499466896057j,
                        -0.005144218914210796 + 0.02740355394780636j,
                        -0.009186972863972187 - 0.026324236765503883j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.04858788475394249 - 0.027421100065112114j,
                        -0.03965778648853302 - 0.03924231231212616j,
                        -0.027931272983551025 - 0.04829641059041023j,
                        -0.014235264621675014 - 0.05394493788480759j,
                    ],
                    [
                        -0.04858788475394249 - 0.027421100065112114j,
                        -0.03965778648853302 - 0.03924231231212616j,
                        -0.027931272983551025 - 0.04829641059041023j,
                        -0.014235264621675014 - 0.05394493788480759j,
                    ],
                    [
                        -0.04858788475394249 - 0.027421100065112114j,
                        -0.03965778648853302 - 0.03924231231212616j,
                        -0.027931272983551025 - 0.04829641059041023j,
                        -0.014235264621675014 - 0.05394493788480759j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.04858788475394249 - 0.027421100065112114j,
                        -0.03965778648853302 - 0.03924231231212616j,
                        -0.027931272983551025 - 0.04829641059041023j,
                        -0.014235264621675014 - 0.05394493788480759j,
                    ],
                    [
                        -0.04858788475394249 - 0.027421100065112114j,
                        -0.03965778648853302 - 0.03924231231212616j,
                        -0.027931272983551025 - 0.04829641059041023j,
                        -0.014235264621675014 - 0.05394493788480759j,
                    ],
                    [
                        -0.04858788475394249 - 0.027421100065112114j,
                        -0.03965778648853302 - 0.03924231231212616j,
                        -0.027931272983551025 - 0.04829641059041023j,
                        -0.014235264621675014 - 0.05394493788480759j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.020161913707852364 + 0.019278796389698982j,
                        -0.015550252050161362 + 0.02315954491496086j,
                        -0.010212653316557407 + 0.025959130376577377j,
                        -0.00439826212823391 + 0.027546871453523636j,
                    ],
                    [
                        -0.020161913707852364 + 0.019278796389698982j,
                        -0.015550252050161362 + 0.02315954491496086j,
                        -0.010212653316557407 + 0.025959130376577377j,
                        -0.00439826212823391 + 0.027546871453523636j,
                    ],
                    [
                        -0.020161913707852364 + 0.019278796389698982j,
                        -0.015550252050161362 + 0.02315954491496086j,
                        -0.010212653316557407 + 0.025959130376577377j,
                        -0.00439826212823391 + 0.027546871453523636j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027447368949651718 + 0.004986839834600687j,
                        0.014111651107668877 - 0.024065323173999786j,
                        0.009057712741196156 + 0.026387274265289307j,
                    ],
                    [
                        0.02640989050269127 - 0.009000211954116821j,
                        -0.010376740247011185 + 0.025900989770889282j,
                        -0.012898511253297329 - 0.02474304474890232j,
                        0.0271796565502882 + 0.006317287217825651j,
                    ],
                    [
                        0.006449196022003889 - 0.027151526883244514j,
                        0.016418518498539925 + 0.022567272186279297j,
                        -0.02782095968723297 - 0.0022126047406345606j,
                        0.01977468840777874 - 0.01969556137919426j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.02167871780693531 - 0.017555853351950645j,
                        0.027893975377082825 - 0.00031773606315255165j,
                        -0.021273193880915642 + 0.018045108765363693j,
                        0.004863053094595671 - 0.02746862731873989j,
                    ],
                    [
                        -0.02167871780693531 - 0.017555853351950645j,
                        0.027893975377082825 - 0.00031773606315255165j,
                        -0.021273193880915642 + 0.018045108765363693j,
                        0.004863053094595671 - 0.02746862731873989j,
                    ],
                    [
                        -0.02167871780693531 - 0.017555853351950645j,
                        0.027893975377082825 - 0.00031773606315255165j,
                        -0.021273193880915642 + 0.018045108765363693j,
                        0.004863053094595671 - 0.02746862731873989j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.1050187349319458 + 0.03770757094025612j,
                        -0.00014793184527661651 - 0.11158303916454315j,
                        -0.10491838306188583 + 0.0379859134554863j,
                        0.07132042199373245 + 0.08581487834453583j,
                    ],
                    [
                        0.1050187349319458 + 0.03770757094025612j,
                        -0.00014793184527661651 - 0.11158303916454315j,
                        -0.10491838306188583 + 0.0379859134554863j,
                        0.07132042199373245 + 0.08581487834453583j,
                    ],
                    [
                        0.1050187349319458 + 0.03770757094025612j,
                        -0.00014793184527661651 - 0.11158303916454315j,
                        -0.10491838306188583 + 0.0379859134554863j,
                        0.07132042199373245 + 0.08581487834453583j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027447368949651718 + 0.004986839834600687j,
                        0.014111651107668877 - 0.024065323173999786j,
                        0.009057712741196156 + 0.026387274265289307j,
                    ],
                    [
                        0.02640989050269127 - 0.009000211954116821j,
                        -0.010376740247011185 + 0.025900989770889282j,
                        -0.012898511253297329 - 0.02474304474890232j,
                        0.0271796565502882 + 0.006317287217825651j,
                    ],
                    [
                        0.006449196022003889 - 0.027151526883244514j,
                        0.016418518498539925 + 0.022567272186279297j,
                        -0.02782095968723297 - 0.0022126047406345606j,
                        0.01977468840777874 - 0.01969556137919426j,
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
    radar = Radar(transmitter=tx, receiver=rx, frame_time=[0, 1])

    targets = [
        {
            "location": np.array([10, 0, 0]),
            "speed": np.array([-5, 0, 0]),
            "rcs": 20,
        }
    ]
    result = sim_radar(radar, targets)

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027768978849053383 + 0.0026616728864610195j,
                        0.017946656793355942 - 0.02135753259062767j,
                        0.002165901707485318 + 0.02781297266483307j,
                    ],
                    [
                        0.027466973289847374 + 0.004888326395303011j,
                        -0.023017114028334618 + 0.015766063705086708j,
                        0.005361844319850206 - 0.02737942524254322j,
                        0.01537376083433628 + 0.023282090201973915j,
                    ],
                    [
                        0.02640989050269127 - 0.009000211954116821j,
                        -0.01251726783812046 + 0.02493651956319809j,
                        -0.008566522970795631 - 0.026554713025689125j,
                        0.02473265863955021 + 0.012917415238916874j,
                    ],
                ],
                [
                    [
                        0.1050187349319458 + 0.03770757094025612j,
                        -0.009528405964374542 - 0.11117929220199585j,
                        -0.09706953167915344 + 0.05504509434103966j,
                        0.09053554385900497 + 0.06524264067411423j,
                    ],
                    [
                        0.11018253862857819 - 0.017764749005436897j,
                        -0.062141794711351395 - 0.0927092507481575j,
                        -0.058290671557188034 + 0.09518212080001831j,
                        0.11083314567804337 + 0.013201641850173473j,
                    ],
                    [
                        0.08788114041090012 - 0.0688307136297226j,
                        -0.09925872087478638 - 0.051081299781799316j,
                        -0.004915804136544466 + 0.11152693629264832j,
                        0.1033688485622406 - 0.042167942970991135j,
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
    radar = Radar(transmitter=tx, receiver=rx, speed=[5, 0, 0], frame_time=[0, 1])

    targets = [
        {
            "location": np.array([10, 0, 0]),
            "speed": np.array([0, 0, 0]),
            "rcs": 20,
        }
    ]
    result = sim_radar(radar, targets)

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027768978849053383 + 0.0026616728864610195j,
                        0.017946656793355942 - 0.02135753259062767j,
                        0.002165901707485318 + 0.02781297266483307j,
                    ],
                    [
                        0.027466973289847374 + 0.004888326395303011j,
                        -0.023017114028334618 + 0.015766063705086708j,
                        0.005361844319850206 - 0.02737942524254322j,
                        0.01537376083433628 + 0.023282090201973915j,
                    ],
                    [
                        0.02640989050269127 - 0.009000211954116821j,
                        -0.01251726783812046 + 0.02493651956319809j,
                        -0.008566522970795631 - 0.026554713025689125j,
                        0.02473265863955021 + 0.012917415238916874j,
                    ],
                ],
                [
                    [
                        0.1050187349319458 + 0.03770757094025612j,
                        -0.009528405964374542 - 0.11117929220199585j,
                        -0.09706953167915344 + 0.05504509434103966j,
                        0.09053554385900497 + 0.06524264067411423j,
                    ],
                    [
                        0.11018253862857819 - 0.017764749005436897j,
                        -0.062141794711351395 - 0.0927092507481575j,
                        -0.058290671557188034 + 0.09518212080001831j,
                        0.11083314567804337 + 0.013201641850173473j,
                    ],
                    [
                        0.08788114041090012 - 0.0688307136297226j,
                        -0.09925872087478638 - 0.051081299781799316j,
                        -0.004915804136544466 + 0.11152693629264832j,
                        0.1033688485622406 - 0.042167942970991135j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03187878429889679 + 0.030482452362775803j,
                        -0.024587105959653854 + 0.03661845251917839j,
                        -0.01614762283861637 + 0.04104498773813248j,
                        -0.006954262498766184 + 0.043555427342653275j,
                    ],
                    [
                        -0.03187878429889679 + 0.030482452362775803j,
                        -0.024587105959653854 + 0.03661845251917839j,
                        -0.01614762283861637 + 0.04104498773813248j,
                        -0.006954262498766184 + 0.043555427342653275j,
                    ],
                    [
                        -0.03187878429889679 + 0.030482452362775803j,
                        -0.024587105959653854 + 0.03661845251917839j,
                        -0.01614762283861637 + 0.04104498773813248j,
                        -0.006954262498766184 + 0.043555427342653275j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.0031878785230219364 + 0.0030482453294098377j,
                        -0.002458710689097643 + 0.0036618453450500965j,
                        -0.0016147622372955084 + 0.004104498773813248j,
                        -0.0006954262498766184 + 0.004355542827397585j,
                    ],
                    [
                        -0.0031878785230219364 + 0.0030482453294098377j,
                        -0.002458710689097643 + 0.0036618453450500965j,
                        -0.0016147622372955084 + 0.004104498773813248j,
                        -0.0006954262498766184 + 0.004355542827397585j,
                    ],
                    [
                        -0.0031878785230219364 + 0.0030482453294098377j,
                        -0.002458710689097643 + 0.0036618453450500965j,
                        -0.0016147622372955084 + 0.004104498773813248j,
                        -0.0006954262498766184 + 0.004355542827397585j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.03187878802418709 + 0.030482454225420952j,
                        -0.024587107822299004 + 0.03661845624446869j,
                        -0.01614762470126152 + 0.041044991463422775j,
                        -0.006954262964427471 + 0.043555427342653275j,
                    ],
                    [
                        -0.03187878802418709 + 0.030482454225420952j,
                        -0.024587107822299004 + 0.03661845624446869j,
                        -0.01614762470126152 + 0.041044991463422775j,
                        -0.006954262964427471 + 0.043555427342653275j,
                    ],
                    [
                        -0.03187878802418709 + 0.030482454225420952j,
                        -0.024587107822299004 + 0.03661845624446869j,
                        -0.01614762470126152 + 0.041044991463422775j,
                        -0.006954262964427471 + 0.043555427342653275j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.0031878785230219364 + 0.0030482453294098377j,
                        -0.002458710689097643 + 0.0036618453450500965j,
                        -0.0016147622372955084 + 0.004104498773813248j,
                        -0.0006954262498766184 + 0.004355542827397585j,
                    ],
                    [
                        -0.0031878785230219364 + 0.0030482453294098377j,
                        -0.002458710689097643 + 0.0036618453450500965j,
                        -0.0016147622372955084 + 0.004104498773813248j,
                        -0.0006954262498766184 + 0.004355542827397585j,
                    ],
                    [
                        -0.0031878785230219364 + 0.0030482453294098377j,
                        -0.002458710689097643 + 0.0036618453450500965j,
                        -0.0016147622372955084 + 0.004104498773813248j,
                        -0.0006954262498766184 + 0.004355542827397585j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.010080956853926182 + 0.009639398194849491j,
                        -0.007775126025080681 + 0.01157977245748043j,
                        -0.005106326658278704 + 0.012979565188288689j,
                        -0.002199131064116955 + 0.013773435726761818j,
                    ],
                    [
                        -0.010080956853926182 + 0.009639398194849491j,
                        -0.007775126025080681 + 0.01157977245748043j,
                        -0.005106326658278704 + 0.012979565188288689j,
                        -0.002199131064116955 + 0.013773435726761818j,
                    ],
                    [
                        -0.010080956853926182 + 0.009639398194849491j,
                        -0.007775126025080681 + 0.01157977245748043j,
                        -0.005106326658278704 + 0.012979565188288689j,
                        -0.002199131064116955 + 0.013773435726761818j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.0010080956853926182 + 0.0009639398194849491j,
                        -0.0007775126141496003 + 0.0011579772690311074j,
                        -0.0005106327007524669 + 0.0012979565653949976j,
                        -0.00021991309768054634 + 0.0013773435493931174j,
                    ],
                    [
                        -0.0010080956853926182 + 0.0009639398194849491j,
                        -0.0007775126141496003 + 0.0011579772690311074j,
                        -0.0005106327007524669 + 0.0012979565653949976j,
                        -0.00021991309768054634 + 0.0013773435493931174j,
                    ],
                    [
                        -0.0010080956853926182 + 0.0009639398194849491j,
                        -0.0007775126141496003 + 0.0011579772690311074j,
                        -0.0005106327007524669 + 0.0012979565653949976j,
                        -0.00021991309768054634 + 0.0013773435493931174j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.010080956853926182 + 0.009639398194849491j,
                        -0.007775126025080681 + 0.01157977245748043j,
                        -0.005106326658278704 + 0.012979565188288689j,
                        -0.002199131064116955 + 0.013773435726761818j,
                    ],
                    [
                        -0.010080956853926182 + 0.009639398194849491j,
                        -0.007775126025080681 + 0.01157977245748043j,
                        -0.005106326658278704 + 0.012979565188288689j,
                        -0.002199131064116955 + 0.013773435726761818j,
                    ],
                    [
                        -0.010080956853926182 + 0.009639398194849491j,
                        -0.007775126025080681 + 0.01157977245748043j,
                        -0.005106326658278704 + 0.012979565188288689j,
                        -0.002199131064116955 + 0.013773435726761818j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.0010080956853926182 + 0.0009639398194849491j,
                        -0.0007775126141496003 + 0.0011579772690311074j,
                        -0.0005106327007524669 + 0.0012979565653949976j,
                        -0.00021991309768054634 + 0.0013773435493931174j,
                    ],
                    [
                        -0.0010080956853926182 + 0.0009639398194849491j,
                        -0.0007775126141496003 + 0.0011579772690311074j,
                        -0.0005106327007524669 + 0.0012979565653949976j,
                        -0.00021991309768054634 + 0.0013773435493931174j,
                    ],
                    [
                        -0.0010080956853926182 + 0.0009639398194849491j,
                        -0.0007775126141496003 + 0.0011579772690311074j,
                        -0.0005106327007524669 + 0.0012979565653949976j,
                        -0.00021991309768054634 + 0.0013773435493931174j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.02167871966958046 + 0.017555851489305496j,
                        -0.027893975377082825 + 0.00031773850787431j,
                        0.021273192018270493 - 0.018045110628008842j,
                        -0.004863050766289234 + 0.02746862918138504j,
                    ],
                    [
                        0.012656153179705143 + 0.02485823817551136j,
                        -0.025607381016016006 - 0.011062202043831348j,
                        0.026774795725941658 - 0.007824353873729706j,
                        -0.015621167607605457 + 0.023110372945666313j,
                    ],
                    [
                        0.0014430786250159144 + 0.027856115251779556j,
                        -0.018887367099523544 - 0.020525911822915077j,
                        0.027640212327241898 + 0.0037502485793083906j,
                        -0.023673780262470245 + 0.014751197770237923j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                    [
                        -0.02167871966958046 - 0.017555851489305496j,
                        0.027893975377082825 - 0.00031773850787431j,
                        -0.021273192018270493 + 0.018045110628008842j,
                        0.004863050766289234 - 0.02746862918138504j,
                    ],
                    [
                        0.04335743933916092 + 0.03511170297861099j,
                        -0.05578795075416565 + 0.00063547701574862j,
                        0.042546384036540985 - 0.036090221256017685j,
                        -0.009726101532578468 + 0.05493725836277008j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        -0.017555851489305496 + 0.02167871966958046j,
                        0.0 + 0.0j,
                        -0.08509276807308197 + 0.07218044251203537j,
                        0.0 + 0.0j,
                    ],
                    [
                        -0.017555851489305496 + 0.02167871966958046j,
                        0.0 + 0.0j,
                        -0.08509276807308197 + 0.07218044251203537j,
                        0.0 + 0.0j,
                    ],
                    [
                        -0.017555851489305496 + 0.02167871966958046j,
                        0.0 + 0.0j,
                        -0.08509276807308197 + 0.07218044251203537j,
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

    assert assert_baseband_close(
        result["baseband"],
        np.array(
            [
                [
                    [
                        0.020911267027258873 + 0.015191296115517616j,
                        0.0165250264108181 - 0.019874105229973793j,
                        -0.022241275757551193 + 0.013167467899620533j,
                        0.025563737377524376 + 0.0038147002924233675j,
                    ],
                    [
                        0.020911267027258873 + 0.015191296115517616j,
                        0.0165250264108181 - 0.019874105229973793j,
                        -0.022241275757551193 + 0.013167467899620533j,
                        0.025563737377524376 + 0.0038147002924233675j,
                    ],
                    [
                        0.020911267027258873 + 0.015191296115517616j,
                        0.0165250264108181 - 0.019874105229973793j,
                        -0.022241275757551193 + 0.013167467899620533j,
                        0.025563737377524376 + 0.0038147002924233675j,
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

    assert assert_baseband_close(
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
                        -0.013252748176455498 + 0.004348371643573046j,
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
