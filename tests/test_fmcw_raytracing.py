#  ____           _            ____  _          __  __
# |  _ \ __ _  __| | __ _ _ __/ ___|(_)_ __ ___ \ \/ /
# | |_) / _` |/ _` |/ _` | '__\___ \| | '_ ` _ \ \  /
# |  _ < (_| | (_| | (_| | |   ___) | | | | | | |/  \
# |_| \_\__,_|\__,_|\__,_|_|  |____/|_|_| |_| |_/_/\_\

"""
A Python module for radar simulation

----------
RadarSimPy - A Radar Simulator Built with Python
Copyright (C) 2018 - PRESENT  radarsimx.com
E-mail: info@radarsimx.com
Website: https://radarsimx.com

"""

import numpy as np
import numpy.testing as npt

from radarsimpy import Radar, Transmitter, Receiver
from scipy import signal
from radarsimpy.rt import scene
import radarsimpy.processing as proc


def test_fmcw_raytracing():
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01)

    tx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=angle,
        azimuth_pattern=pattern,
        elevation_angle=angle,
        elevation_pattern=pattern,
    )

    tx = Transmitter(
        f=[1e9 - 50e6, 1e9 + 50e6],
        t=[0, 80e-6],
        tx_power=15,
        prp=0.5,
        pulses=2,
        channels=[tx_channel],
    )

    rx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=angle,
        azimuth_pattern=pattern,
        elevation_angle=angle,
        elevation_pattern=pattern,
    )

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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.array_equal(np.shape(range_profile), np.array([2, 2, 160]))

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 133
    assert np.argmax(np.abs(range_profile[0, 1, :])) == 117
    assert np.argmax(np.abs(range_profile[1, 0, :])) == 100
    assert np.argmax(np.abs(range_profile[1, 1, :])) == 83

    amp1 = 20 * np.log10(np.abs(range_profile[0, 0, 133]))
    phs1 = np.angle(range_profile[0, 0, 133]) / np.pi * 180

    amp2 = 20 * np.log10(np.abs(range_profile[0, 1, 117]))
    phs2 = np.angle(range_profile[0, 1, 117]) / np.pi * 180

    amp3 = 20 * np.log10(np.abs(range_profile[1, 0, 100]))
    phs3 = np.angle(range_profile[1, 0, 100]) / np.pi * 180

    amp4 = 20 * np.log10(np.abs(range_profile[1, 1, 83]))
    phs4 = np.angle(range_profile[1, 1, 83]) / np.pi * 180

    npt.assert_almost_equal(amp1, 16.23, decimal=1)
    npt.assert_almost_equal(amp2, -7.61, decimal=1)
    npt.assert_almost_equal(amp3, -5.93, decimal=1)
    npt.assert_almost_equal(amp4, -8.48, decimal=1)

    npt.assert_almost_equal(phs1, -17.71, decimal=0)
    npt.assert_almost_equal(phs2, 170.59, decimal=0)
    npt.assert_almost_equal(phs3, -9.69, decimal=0)
    npt.assert_almost_equal(phs4, 157.27, decimal=0)


def test_fmcw_raytracing_tx_azimuth():
    tx_az_angle = np.arange(-90, 91, 1)
    tx_az_pattern = tx_az_angle / 9

    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=tx_az_angle,
        azimuth_pattern=tx_az_pattern,
        elevation_angle=tx_el_angle,
        elevation_pattern=tx_el_pattern,
    )

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -48.01, decimal=1)
    npt.assert_almost_equal(phs, 154.20, decimal=0)


def test_fmcw_raytracing_tx_elevation():
    tx_az_angle = np.arange(-90, 91, 1)
    tx_az_pattern = tx_az_angle / 9

    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=tx_az_angle,
        azimuth_pattern=tx_az_pattern,
        elevation_angle=tx_el_angle,
        elevation_pattern=tx_el_pattern,
    )

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -51.03, decimal=1)
    npt.assert_almost_equal(phs, 151.81, decimal=0)


def test_fmcw_raytracing_rx_azimuth():
    tx_channel = dict(
        location=(0, 0, 0),
    )

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

    rx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=rx_az_angle,
        azimuth_pattern=rx_az_pattern,
        elevation_angle=rx_el_angle,
        elevation_pattern=rx_el_pattern,
    )

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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -48.01, decimal=1)
    npt.assert_almost_equal(phs, 154.20, decimal=0)


def test_fmcw_raytracing_rx_elevation():
    tx_channel = dict(location=(0, 0, 0))

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

    rx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=rx_az_angle,
        azimuth_pattern=rx_az_pattern,
        elevation_angle=rx_el_angle,
        elevation_pattern=rx_el_pattern,
    )

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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 47

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 47]))
    phs = np.angle(range_profile[0, 0, 47]) / np.pi * 180

    npt.assert_almost_equal(amp, -51.04, decimal=1)
    npt.assert_almost_equal(phs, 151.81, decimal=0)


def test_fmcw_raytracing_radar_rotation():
    tx_az_angle = np.arange(-90, 91, 1)
    tx_az_pattern = np.abs(tx_az_angle) / 9
    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = dict(
        location=(0, 0, 0),
        # azimuth_angle=tx_az_angle,
        # azimuth_pattern=tx_az_pattern,
        elevation_angle=tx_el_angle,
        elevation_pattern=tx_el_pattern,
    )

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)

    assert np.argmax(np.abs(range_profile[0, 0, :])) == 33

    amp = 20 * np.log10(np.abs(range_profile[0, 0, 33]))
    phs = np.angle(range_profile[0, 0, 33]) / np.pi * 180

    npt.assert_almost_equal(amp, -47.18, decimal=1)
    npt.assert_almost_equal(phs, -65.29, decimal=0)


def test_fmcw_raytracing_radar_speed():
    tx_az_angle = np.arange(-90, 91, 1)
    tx_az_pattern = np.abs(tx_az_angle) / 9
    tx_el_angle = np.arange(-90, 91, 1)
    tx_el_pattern = tx_el_angle / 9

    tx_channel = dict(
        location=(0, 0, 0),
        # azimuth_angle=tx_az_angle,
        # azimuth_pattern=tx_az_pattern,
        elevation_angle=tx_el_angle,
        elevation_pattern=tx_el_pattern,
    )

    tx = Transmitter(
        f=[10e9 - 50e6, 10e9 + 50e6],
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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
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
