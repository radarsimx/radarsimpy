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
import numpy.testing as npt

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import sim_radar  # pylint: disable=no-name-in-module
from scipy import constants


def test_pulsed_radar():
    """
    This function tests pulsed radar simulation
    """
    antenna_gain = 20  # dBi

    az_angle = np.arange(-20, 21, 1)
    az_pattern = 20 * np.log10(np.cos(az_angle / 180 * np.pi) ** 500) + antenna_gain

    el_angle = np.arange(-20, 21, 1)
    el_pattern = 20 * np.log10((np.cos(el_angle / 180 * np.pi)) ** 400) + antenna_gain

    light_speed = constants.c

    max_range = 5000  # Maximum unambiguous range
    range_res = 500  # Required range resolution
    pulse_bw = light_speed / (2 * range_res)  # Pulse bandwidth
    pulse_width = 1 / pulse_bw  # Pulse width
    prf = light_speed / (2 * max_range)  # Pulse repetition frequency
    prp = 1 / prf
    fs = 6e6  # Sampling rate 6 msps
    fc = 35e9  # Carrier frequency
    num_pulse = 1

    total_samples = int(prp * num_pulse * fs)

    mod_t = np.arange(0, total_samples) / fs  # Time series of the modulation
    amp = np.zeros_like(mod_t)  # Amplitude of the modulation
    amp[mod_t <= pulse_width] = 1  # Create a rectangular pulse

    tx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=az_angle,
        azimuth_pattern=az_pattern,
        elevation_angle=el_angle,
        elevation_pattern=el_pattern,
        amp=amp,
        mod_t=mod_t,
    )

    tx = Transmitter(
        f=fc,
        t=1 / prf,
        tx_power=67,  # dBm
        pulses=num_pulse,
        channels=[tx_channel],
    )

    rx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=az_angle,
        azimuth_pattern=az_pattern,
        elevation_angle=el_angle,
        elevation_pattern=el_pattern,
    )

    rx = Receiver(
        fs=fs,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx)

    target = dict(location=(1000, 0, 0), speed=(150, 0, 0), rcs=10, phase=0)

    data = sim_radar(radar, [target])
    baseband = data["baseband"]  # noise is left out on purpose

    npt.assert_allclose(
        np.arctan2(np.real(baseband[0, 0, :]), np.imag(baseband[0, 0, :])),
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.97831584,
                0.94163854,
                0.90496124,
                0.86828393,
                0.83160663,
                0.79492933,
                0.75825213,
                0.72157483,
                0.68489753,
                0.64822023,
                0.61154293,
                0.57486563,
                0.53818833,
                0.50151103,
                0.46483373,
                0.42815643,
                0.39147913,
                0.35480183,
                0.31812473,
                0.28144743,
                0.24477013,
                0.20809283,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


def test_pulsed_radar_with_doppler():
    antenna_gain = 20  # dBi

    az_angle = np.arange(-20, 21, 1)
    az_pattern = 20 * np.log10(np.cos(az_angle / 180 * np.pi) ** 500) + antenna_gain

    el_angle = np.arange(-20, 21, 1)
    el_pattern = 20 * np.log10((np.cos(el_angle / 180 * np.pi)) ** 400) + antenna_gain

    light_speed = constants.c

    max_range = 5000  # Maximum unambiguous range
    range_res = 500  # Required range resolution
    pulse_bw = light_speed / (2 * range_res)  # Pulse bandwidth
    pulse_width = 1 / pulse_bw  # Pulse width
    prf = light_speed / (2 * max_range)  # Pulse repetition frequency
    prp = 1 / prf
    fs = 6e6  # Sampling rate 6 msps
    fc = 35e9  # Carrier frequency
    num_pulse = 1

    total_samples = int(prp * num_pulse * fs)
    sample_per_pulse = int(total_samples / num_pulse)

    mod_t = np.arange(0, total_samples) / fs  # Time series of the modulation
    amp = np.zeros_like(mod_t)  # Amplitude of the modulation
    amp[mod_t <= pulse_width] = 1  # Create a rectangular pulse

    tx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=az_angle,
        azimuth_pattern=az_pattern,
        elevation_angle=el_angle,
        elevation_pattern=el_pattern,
        amp=amp,
        mod_t=mod_t,
    )

    tx = Transmitter(
        f=fc,
        t=1 / prf,
        tx_power=67,  # dBm
        pulses=num_pulse,
        channels=[tx_channel],
    )

    rx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=az_angle,
        azimuth_pattern=az_pattern,
        elevation_angle=el_angle,
        elevation_pattern=el_pattern,
    )

    rx = Receiver(
        fs=fs,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar = Radar(transmitter=tx, receiver=rx)

    target = dict(location=(2000, 0, 0), speed=(150, 0, 0), rcs=10, phase=0)

    data = sim_radar(radar, [target])
    baseband = data["baseband"]  # noise is left out on purpose

    npt.assert_allclose(
        np.arctan2(np.real(baseband[0, 0, 79:100]), np.imag(baseband[0, 0, 79:100])),
        np.array(
            [
                0.34915804,
                0.31248074,
                0.27580344,
                0.23912614,
                0.20244884,
                0.16577154,
                0.12909424,
                0.09241694,
                0.05573964,
                0.01906234,
                -0.01761476,
                -0.05429206,
                -0.09096936,
                -0.12764666,
                -0.16432396,
                -0.20100126,
                -0.23767856,
                -0.27435586,
                -0.31103316,
                -0.34771046,
                -0.38438776,
            ]
        ),
        1e-6,
    )
