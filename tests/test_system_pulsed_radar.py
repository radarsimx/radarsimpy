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
                0.97831574,
                0.94163844,
                0.90496114,
                0.86828384,
                0.83160654,
                0.79492924,
                0.75825204,
                0.72157474,
                0.68489744,
                0.64822014,
                0.61154284,
                0.57486554,
                0.53818824,
                0.50151094,
                0.46483363,
                0.42815633,
                0.39147903,
                0.35480173,
                0.31812463,
                0.28144733,
                0.24477003,
                0.20809273,
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
                0.34915785,
                0.31248055,
                0.27580325,
                0.23912595,
                0.20244865,
                0.16577135,
                0.12909405,
                0.09241675,
                0.05573945,
                0.01906215,
                -0.01761495,
                -0.05429225,
                -0.09096955,
                -0.12764685,
                -0.16432415,
                -0.20100145,
                -0.23767875,
                -0.27435605,
                -0.31103335,
                -0.34771065,
                -0.38438795,
            ]
        ),
        1e-6
    )
