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

import scipy.constants as const
import numpy as np
import numpy.testing as npt
from scipy import signal

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module
import radarsimpy.processing as proc


def test_sim_fmcw():
    """
    Test the FMCW radar simulator.
    """
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6

    tx_channel = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
    }

    tx = Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=256,
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
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx)
    target_1 = {"location": (200, 0, 0), "speed": (-5, 0, 0), "rcs": 20, "phase": 0}
    target_2 = {"location": (95, 20, 0), "speed": (-50, 0, 0), "rcs": 15, "phase": 0}
    target_3 = {"location": (30, -5, 0), "speed": (-22, 0, 0), "rcs": 5, "phase": 0}

    rng_targets = np.sort(
        np.array(
            [
                np.sqrt(target_1["location"][0] ** 2 + target_1["location"][1] ** 2),
                np.sqrt(target_2["location"][0] ** 2 + target_2["location"][1] ** 2),
                np.sqrt(target_3["location"][0] ** 2 + target_3["location"][1] ** 2),
            ]
        )
    )
    dop_targets = np.sort(
        np.array(
            [
                target_1["speed"][0]
                * np.cos(np.arctan(target_1["location"][1] / target_1["location"][0])),
                target_2["speed"][0]
                * np.cos(np.arctan(target_2["location"][1] / target_2["location"][0])),
                target_3["speed"][0]
                * np.cos(np.arctan(target_3["location"][1] / target_3["location"][0])),
            ]
        )
    )

    targets = [target_1, target_2, target_3]

    data = sim_radar(radar, targets, frame_time=[0, 1])
    timestamp = data["timestamp"]
    baseband = data["baseband"]

    assert np.array_equal(
        (
            radar.array_prop["size"] * 2,
            radar.radar_prop["transmitter"].waveform_prop["pulses"],
            radar.sample_prop["samples_per_pulse"],
        ),
        np.shape(timestamp),
    )
    assert np.array_equal(
        (
            radar.array_prop["size"] * 2,
            radar.radar_prop["transmitter"].waveform_prop["pulses"],
            radar.sample_prop["samples_per_pulse"],
        ),
        np.shape(baseband),
    )

    npt.assert_almost_equal(
        timestamp[0, 0, :],
        (
            np.arange(0, radar.sample_prop["samples_per_pulse"])
            / radar.radar_prop["receiver"].bb_prop["fs"]
        ),
    )
    npt.assert_almost_equal(
        timestamp[0, :, 0],
        (
            np.arange(0, radar.radar_prop["transmitter"].waveform_prop["pulses"])
            * radar.radar_prop["transmitter"].waveform_prop["prp"][0]
        ),
    )

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)
    doppler_window = signal.windows.chebwin(
        radar.radar_prop["transmitter"].waveform_prop["pulses"], at=60
    )
    range_doppler = proc.doppler_fft(range_profile, doppler_window)
    rng_dop = 20 * np.log10(np.abs(range_doppler))
    rng_dop = rng_dop - np.max(rng_dop[0, :, :])

    max_rng = np.max(rng_dop[0, :, :], axis=0)
    max_dop = np.max(rng_dop[0, :, :], axis=1)

    rng_peaks = signal.find_peaks(max_rng, height=-20)[0]
    dop_peaks = signal.find_peaks(max_dop, height=-20)[0]

    max_range = (
        const.c
        * radar.radar_prop["receiver"].bb_prop["fs"]
        * radar.radar_prop["transmitter"].waveform_prop["pulse_length"]
        / radar.radar_prop["transmitter"].waveform_prop["bandwidth"]
        / 2
    )

    unambiguous_speed = (
        const.c / radar.radar_prop["transmitter"].waveform_prop["prp"][0] / 24.125e9 / 2
    )

    range_axis = np.linspace(
        0, max_range, radar.sample_prop["samples_per_pulse"], endpoint=False
    )

    rng_dets = np.sort(range_axis[rng_peaks])
    npt.assert_almost_equal(rng_targets, rng_dets, decimal=0)

    doppler_axis = np.linspace(
        -unambiguous_speed,
        0,
        radar.radar_prop["transmitter"].waveform_prop["pulses"],
        endpoint=False,
    )

    dop_dets = np.sort(doppler_axis[dop_peaks])
    npt.assert_almost_equal(dop_targets, dop_dets, decimal=0)

    # frame 2
    rng_dop = rng_dop - np.max(rng_dop[1, :, :])

    max_rng = np.max(rng_dop[1, :, :], axis=0)
    max_dop = np.max(rng_dop[1, :, :], axis=1)

    rng_peaks = signal.find_peaks(max_rng, height=-40)[0]
    dop_peaks = signal.find_peaks(max_dop, height=-40)[0]

    range_axis = np.linspace(
        0, max_range, radar.sample_prop["samples_per_pulse"], endpoint=False
    )

    rng_dets = np.sort(range_axis[rng_peaks])
    npt.assert_almost_equal(np.array([9.0, 48.0, 195.0]), rng_dets, decimal=0)

    doppler_axis = np.linspace(
        -unambiguous_speed,
        0,
        radar.radar_prop["transmitter"].waveform_prop["pulses"],
        endpoint=False,
    )

    dop_dets = np.sort(doppler_axis[dop_peaks])
    npt.assert_almost_equal(
        np.array([-45.66062176, -18.45854922, -5.1003886]), dop_dets, decimal=0
    )


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

    radar = Radar(transmitter=tx, receiver=rx)

    target_1 = {
        "model": "./models/plate5x5.stl",
        "location": (200, 0, 0),
        "speed": (-50, 0, 0),
        "rotation_rate": (0, 10, 0),
    }

    targets = [target_1]

    data = sim_radar(radar, targets, frame_time=[0, 1], density=0.4, level="pulse")

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

    npt.assert_almost_equal(amp1, 17.22933324317141, decimal=1)
    npt.assert_almost_equal(amp2, -7.61, decimal=1)
    npt.assert_almost_equal(amp3, -5.93, decimal=1)
    npt.assert_almost_equal(amp4, -8.48, decimal=1)

    npt.assert_almost_equal(phs1, -13.08918953855478, decimal=0)
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

    data = sim_radar(radar, targets, density=1, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -47.116, decimal=1)
    npt.assert_almost_equal(phs, 152.265, decimal=0)


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

    data = sim_radar(radar, targets, density=1, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -50.246, decimal=1)
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

    data = sim_radar(radar, targets, density=1, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -47.122, decimal=1)
    npt.assert_almost_equal(phs, 152.258, decimal=0)


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

    data = sim_radar(radar, targets, density=1, level="pulse")

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -50.262, decimal=1)
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

    data = sim_radar(radar, targets, density=1, level="pulse", debug=False)

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 33

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 33]))
    phs = np.angle(range_profile[0, 0, 33]) / np.pi * 180

    npt.assert_almost_equal(amp, -52.851766416352596, decimal=1)
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

    data = sim_radar(
        radar, targets, frame_time=[0, 1], density=1, level="pulse", debug=False
    )

    baseband = data["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47
    assert np.argmax(np.abs(range_profile[1, 0, :])) == 33

    amp1 = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs1 = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp1, -59.086, decimal=1)
    npt.assert_almost_equal(phs1, -14.57, decimal=0)

    amp2 = 20 * np.log10(np.abs(range_profile[1, 0, 33]))
    phs2 = np.angle(range_profile[1, 0, 33]) / np.pi * 180

    npt.assert_almost_equal(amp2, -52.48597283140286, decimal=1)
    npt.assert_almost_equal(phs2, -84.39, decimal=0)
