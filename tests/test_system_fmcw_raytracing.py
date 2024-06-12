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
import numpy.testing as npt
from scipy import signal

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.rt import scene  # pylint: disable=no-name-in-module
import radarsimpy.processing as proc


# def test_scene_basic():
#     """
#     Basic test case with a single target and simple radar setup.
#     """
#     radar = Radar(
#         location=np.array([0, 0, 0]),
#         speed=np.array([0, 0, 0]),
#         rotation=np.array([0, 0, 0]),
#         rotation_rate=np.array([0, 0, 0]),
#         frequency=1e9,
#         bandwidth=1e6,
#         pulses=10,
#         samples_per_pulse=100,
#         frame_size=1,
#         rxchannel_prop={"size": 1},
#         txchannel_prop={"size": 1},
#     )
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     result = scene(radar, targets)
#     assert "baseband" in result
#     assert "timestamp" in result
#     assert result["baseband"].shape == (1, 10, 100)  # Check baseband shape


# def test_scene_multiple_targets():
#     """
#     Test with multiple targets.
#     """
#     radar = Radar(
#         location=np.array([0, 0, 0]),
#         speed=np.array([0, 0, 0]),
#         rotation=np.array([0, 0, 0]),
#         rotation_rate=np.array([0, 0, 0]),
#         frequency=1e9,
#         bandwidth=1e6,
#         pulses=10,
#         samples_per_pulse=100,
#         frame_size=1,
#         rxchannel_prop={"size": 1},
#         txchannel_prop={"size": 1},
#     )
#     targets = [
#         {
#             "model": "path/to/model1.obj",
#             "location": np.array([10, 0, 0]),
#         },
#         {
#             "model": "path/to/model2.obj",
#             "location": np.array([0, 10, 0]),
#         },
#     ]
#     result = scene(radar, targets)
#     assert "baseband" in result
#     assert "timestamp" in result
#     assert result["baseband"].shape == (1, 10, 100)  # Check baseband shape


# def test_scene_different_angles():
#     """
#     Test with different radar angles.
#     """
#     radar = Radar(
#         location=np.array([0, 0, 0]),
#         speed=np.array([0, 0, 0]),
#         rotation=np.array([45, 30, 0]),
#         rotation_rate=np.array([0, 0, 0]),
#         frequency=1e9,
#         bandwidth=1e6,
#         pulses=10,
#         samples_per_pulse=100,
#         frame_size=1,
#         rxchannel_prop={"size": 1},
#         txchannel_prop={"size": 1},
#     )
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     result = scene(radar, targets)
#     assert "baseband" in result
#     assert "timestamp" in result
#     assert result["baseband"].shape == (1, 10, 100)  # Check baseband shape


# def test_scene_different_levels():
#     """
#     Test with different fidelity levels.
#     """
#     radar = Radar(
#         location=np.array([0, 0, 0]),
#         speed=np.array([0, 0, 0]),
#         rotation=np.array([0, 0, 0]),
#         rotation_rate=np.array([0, 0, 0]),
#         frequency=1e9,
#         bandwidth=1e6,
#         pulses=10,
#         samples_per_pulse=100,
#         frame_size=1,
#         rxchannel_prop={"size": 1},
#         txchannel_prop={"size": 1},
#     )
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     result_none = scene(radar, targets, level=None)
#     result_pulse = scene(radar, targets, level="pulse")
#     result_sample = scene(radar, targets, level="sample")
#     assert "baseband" in result_none
#     assert "timestamp" in result_none
#     assert "baseband" in result_pulse
#     assert "timestamp" in result_pulse
#     assert "baseband" in result_sample
#     assert "timestamp" in result_sample


# def test_scene_noise():
#     """
#     Test with noise enabled and disabled.
#     """
#     radar = Radar(
#         location=np.array([0, 0, 0]),
#         speed=np.array([0, 0, 0]),
#         rotation=np.array([0, 0, 0]),
#         rotation_rate=np.array([0, 0, 0]),
#         frequency=1e9,
#         bandwidth=1e6,
#         pulses=10,
#         samples_per_pulse=100,
#         frame_size=1,
#         rxchannel_prop={"size": 1},
#         txchannel_prop={"size": 1},
#     )
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     result_noise = scene(radar, targets, noise=True)
#     result_no_noise = scene(radar, targets, noise=False)
#     assert not np.allclose(result_noise["baseband"], result_no_noise["baseband"])


# def test_scene_log_path():
#     """
#     Test with log_path specified.
#     """
#     radar = Radar(
#         location=np.array([0, 0, 0]),
#         speed=np.array([0, 0, 0]),
#         rotation=np.array([0, 0, 0]),
#         rotation_rate=np.array([0, 0, 0]),
#         frequency=1e9,
#         bandwidth=1e6,
#         pulses=10,
#         samples_per_pulse=100,
#         frame_size=1,
#         rxchannel_prop={"size": 1},
#         txchannel_prop={"size": 1},
#     )
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     result = scene(radar, targets, log_path="test.log")
#     assert "baseband" in result
#     assert "timestamp" in result
#     assert result["baseband"].shape == (1, 10, 100)  # Check baseband shape


# def test_scene_debug():
#     """
#     Test with debug mode enabled.
#     """
#     radar = Radar(
#         location=np.array([0, 0, 0]),
#         speed=np.array([0, 0, 0]),
#         rotation=np.array([0, 0, 0]),
#         rotation_rate=np.array([0, 0, 0]),
#         frequency=1e9,
#         bandwidth=1e6,
#         pulses=10,
#         samples_per_pulse=100,
#         frame_size=1,
#         rxchannel_prop={"size": 1},
#         txchannel_prop={"size": 1},
#     )
#     targets = [
#         {
#             "model": "path/to/model.obj",
#             "location": np.array([10, 0, 0]),
#         }
#     ]
#     result = scene(radar, targets, debug=True)
#     assert "baseband" in result
#     assert "timestamp" in result
#     assert result["baseband"].shape == (1, 10, 100)  # Check baseband shape


def test_fmcw_raytracing():
    """
    This function tests an FMCW radar using raytracing.
    """
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01)

    tx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
    }

    tx = Transmitter(
        f=[1e9 - 50e6, 1e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=2,
        channels=[tx_channel],
    )

    rx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
    }

    rx = Receiver(
        fs=2e6,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx, time=[0, 1])

    target_1 = {
        "model": "./models/plate5x5.stl",
        "location": (200, 0, 0),
        "speed": (-50, 0, 0),
        "rotation_rate": (0, 10, 0),
    }

    targets = [target_1]

    data = scene(radar, targets, density=0.4, noise=False, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.array_equal(np.shape(range_profile), np.array([2, 2, 160]))

    assert (
        np.argmax(
            np.abs(range_profile[0, 0, :])  # pylint: disable=invalid-sequence-index
        )
        == 133
    )
    assert (
        np.argmax(
            np.abs(range_profile[0, 1, :])  # pylint: disable=invalid-sequence-index
        )
        == 117
    )
    assert (
        np.argmax(
            np.abs(range_profile[1, 0, :])  # pylint: disable=invalid-sequence-index
        )
        == 100
    )
    assert (
        np.argmax(
            np.abs(range_profile[1, 1, :])  # pylint: disable=invalid-sequence-index
        )
        == 83
    )

    amp1 = 20 * np.log10(
        np.abs(range_profile[0, 0, 133])  # pylint: disable=invalid-sequence-index
    )
    phs1 = (
        np.angle(range_profile[0, 0, 133])  # pylint: disable=invalid-sequence-index
        / np.pi
        * 180
    )

    amp2 = 20 * np.log10(
        np.abs(range_profile[0, 1, 117])  # pylint: disable=invalid-sequence-index
    )
    phs2 = (
        np.angle(range_profile[0, 1, 117])  # pylint: disable=invalid-sequence-index
        / np.pi
        * 180
    )

    amp3 = 20 * np.log10(
        np.abs(range_profile[1, 0, 100])  # pylint: disable=invalid-sequence-index
    )
    phs3 = (
        np.angle(range_profile[1, 0, 100])  # pylint: disable=invalid-sequence-index
        / np.pi
        * 180
    )

    amp4 = 20 * np.log10(
        np.abs(range_profile[1, 1, 83])  # pylint: disable=invalid-sequence-index
    )
    phs4 = (
        np.angle(range_profile[1, 1, 83])  # pylint: disable=invalid-sequence-index
        / np.pi
        * 180
    )

    npt.assert_almost_equal(amp1, 16.23, decimal=1)
    npt.assert_almost_equal(amp2, -7.61, decimal=1)
    npt.assert_almost_equal(amp3, -5.93, decimal=1)
    npt.assert_almost_equal(amp4, -8.48, decimal=1)

    npt.assert_almost_equal(phs1, -17.71, decimal=0)
    npt.assert_almost_equal(phs2, 170.59, decimal=0)
    npt.assert_almost_equal(phs3, -9.69, decimal=0)
    npt.assert_almost_equal(phs4, 157.27, decimal=0)


def test_fmcw_raytracing_tx_azimuth():
    """
    This function tests the transmitter's azimuth pattern in an FMCW radar using raytracing.
    """
    tx_az_angle = np.arange(-90, 91, 1)
    tx_az_pattern = tx_az_angle / 9

    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": tx_az_angle,
        "azimuth_pattern": tx_az_pattern,
        "elevation_angle": tx_el_angle,
        "elevation_pattern": tx_el_pattern,
    }

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=1,
        channels=[tx_channel],
    )

    rx_channel = {
        "location": (0, 0, 0),
    }

    rx = Receiver(
        fs=2e6,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx)

    target_1 = {
        "model": "./models/cr.stl",
        "location": (50, 50, 0),
        "speed": (0, 0, 0),
        "rotation": (45, 0, 0),
    }

    targets = [target_1]

    data = scene(radar, targets, density=1, noise=False, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -48.01, decimal=1)
    npt.assert_almost_equal(phs, 154.20, decimal=0)


def test_fmcw_raytracing_tx_elevation():
    """
    This function tests the transmitter's elevation pattern in an FMCW radar using raytracing.
    """
    tx_az_angle = np.arange(-90, 91, 1)
    tx_az_pattern = tx_az_angle / 9

    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": tx_az_angle,
        "azimuth_pattern": tx_az_pattern,
        "elevation_angle": tx_el_angle,
        "elevation_pattern": tx_el_pattern,
    }

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=1,
        channels=[tx_channel],
    )

    rx_channel = {
        "location": (0, 0, 0),
    }

    rx = Receiver(
        fs=2e6,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx)

    target_1 = {
        "model": "./models/cr.stl",
        "location": (50, 0, 50),
        "speed": (0, 0, 0),
        "rotation": (0, 45, 0),
    }

    targets = [target_1]

    data = scene(radar, targets, density=1, noise=False, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -51.03, decimal=1)
    npt.assert_almost_equal(phs, 151.81, decimal=0)


def test_fmcw_raytracing_rx_azimuth():
    """
    This function tests the receiver's azimuth pattern in an FMCW radar using raytracing.
    """
    tx_channel = {
        "location": (0, 0, 0),
    }

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=1,
        channels=[tx_channel],
    )

    rx_az_angle = np.arange(-90, 91, 1)
    rx_az_pattern = rx_az_angle / 9

    rx_el_angle = np.arange(-90, 91, 1)
    rx_el_pattern = rx_el_angle / 9

    rx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": rx_az_angle,
        "azimuth_pattern": rx_az_pattern,
        "elevation_angle": rx_el_angle,
        "elevation_pattern": rx_el_pattern,
    }

    rx = Receiver(
        fs=2e6,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx)

    target_1 = {
        "model": "./models/cr.stl",
        "location": (50, 50, 0),
        "speed": (0, 0, 0),
        "rotation": (45, 0, 0),
    }

    targets = [target_1]

    data = scene(radar, targets, density=1, noise=False, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -48.01, decimal=1)
    npt.assert_almost_equal(phs, 154.20, decimal=0)


def test_fmcw_raytracing_rx_elevation():
    """
    This function tests the receiver's elevation pattern in an FMCW radar using raytracing.
    """
    tx_channel = {"location": (0, 0, 0)}

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=1,
        channels=[tx_channel],
    )

    rx_az_angle = np.arange(-90, 91, 1)
    rx_az_pattern = rx_az_angle / 9

    rx_el_angle = np.arange(-90, 91, 1)
    rx_el_pattern = rx_el_angle / 9

    rx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": rx_az_angle,
        "azimuth_pattern": rx_az_pattern,
        "elevation_angle": rx_el_angle,
        "elevation_pattern": rx_el_pattern,
    }

    rx = Receiver(
        fs=2e6,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx)

    target_1 = {
        "model": "./models/cr.stl",
        "location": (50, 0, 50),
        "speed": (0, 0, 0),
        "rotation": (0, 45, 0),
    }

    targets = [target_1]

    data = scene(radar, targets, density=1, noise=False, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -51.04, decimal=1)
    npt.assert_almost_equal(phs, 151.81, decimal=0)


def test_fmcw_raytracing_radar_rotation():
    """
    This function tests the rotation of an FMCW radar using raytracing.
    """
    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = {
        "location": (0, 0, 0),
        "elevation_angle": tx_el_angle,
        "elevation_pattern": tx_el_pattern,
    }

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=1,
        channels=[tx_channel],
    )

    rx_channel = {
        "location": (0, 0, 0),
    }

    rx = Receiver(
        fs=2e6,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx, rotation=(0, 45, 0))

    target_1 = {
        "model": "./models/cr.stl",
        "location": (50, 0, 0),
        "speed": (0, 0, 0),
        "rotation": (0, 0, 0),
    }

    targets = [target_1]

    data = scene(radar, targets, density=1, noise=False, level="pulse", debug=False)

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 33

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 33]))
    phs = np.angle(range_profile[0, 0, 33]) / np.pi * 180

    npt.assert_almost_equal(amp, -47.18, decimal=1)
    npt.assert_almost_equal(phs, -65.29, decimal=0)


def test_fmcw_raytracing_radar_speed():
    """
    This function tests the speed of an FMCW radar using raytracing.
    """
    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = {
        "location": (0, 0, 0),
        "elevation_angle": tx_el_angle,
        "elevation_pattern": tx_el_pattern,
    }

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=1,
        channels=[tx_channel],
    )

    rx_channel = {
        "location": (0, 0, 0),
    }

    rx = Receiver(
        fs=2e6,
        noise_figure=8,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(
        transmitter=tx,
        receiver=rx,
        time=[0, 1],
        location=(-20, 0, 0),
        speed=(20, 0, 0),
        rotation=(0, 45, 0),
    )

    target_1 = {
        "model": "./models/cr.stl",
        "location": (50, 0, 0),
        "speed": (0, 0, 0),
        "rotation": (0, 0, 0),
    }

    targets = [target_1]

    data = scene(radar, targets, density=1, noise=False, level="pulse", debug=False)

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47
    assert np.argmax(np.abs(range_profile[1, 0, :])) == 33

    amp1 = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs1 = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp1, -54.36, decimal=1)
    npt.assert_almost_equal(phs1, -14.57, decimal=0)

    amp2 = 20 * np.log10(np.abs(range_profile[1, 0, 33]))
    phs2 = np.angle(range_profile[1, 0, 33]) / np.pi * 180

    npt.assert_almost_equal(amp2, -46.82, decimal=1)
    npt.assert_almost_equal(phs2, -84.39, decimal=0)
