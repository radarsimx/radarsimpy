"""
A Python module for radar simulation

----------
RadarSimPy - A Radar Simulator Built with Python
Copyright (C) 2018 - PRESENT  radarsimx.com
E-mail: info@radarsimx.com
Website: https://radarsimx.com

"""

from radarsimpy import Receiver
import numpy as np
import numpy.testing as npt

import scipy.constants as const


def cw_rx():
    return Receiver(
        fs=20,
        noise_figure=12,
        rf_gain=20,
        baseband_gain=50,
        load_resistor=1000,
        channels=[dict(location=(0, 0, 0))],
    )


def test_cw_rx():
    print("#### CW receiver ####")
    cw = cw_rx()

    print("# CW receiver parameters #")
    assert cw.fs == 20
    assert cw.noise_figure == 12
    assert cw.rf_gain == 20
    assert cw.load_resistor == 1000
    assert cw.baseband_gain == 50
    assert cw.noise_bandwidth == cw.fs

    print("# CW receiver channel #")
    assert cw.channel_size == 1
    assert np.array_equal(cw.locations, np.array([[0, 0, 0]]))
    assert np.array_equal(cw.az_angles, [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.az_patterns, [np.zeros(2)])
    assert np.array_equal(cw.el_angles, [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.el_patterns, [np.zeros(2)])


def fmcw_rx():
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6

    rx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=angle,
        azimuth_pattern=pattern,
        elevation_angle=angle,
        elevation_pattern=pattern,
    )

    return Receiver(
        fs=2e6,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )


def test_fmcw_rx():
    print("#### FMCW receiver ####")
    fmcw = fmcw_rx()

    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6
    pattern = pattern - np.max(pattern)

    print("# FMCW receiver parameters #")
    assert fmcw.fs == 2e6
    assert fmcw.noise_figure == 12
    assert fmcw.rf_gain == 20
    assert fmcw.load_resistor == 500
    assert fmcw.baseband_gain == 30
    assert fmcw.noise_bandwidth == fmcw.fs

    print("# FMCW receiver channel #")
    assert fmcw.channel_size == 1
    assert np.array_equal(fmcw.locations, np.array([[0, 0, 0]]))
    assert np.array_equal(fmcw.az_angles, [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.az_patterns, [pattern])
    assert np.array_equal(fmcw.el_angles, [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.el_patterns, [pattern])


def tdm_fmcw_rx():
    wavelength = const.c / 24.125e9
    channels = []
    for idx in range(0, 8):
        channels.append(dict(location=(0, wavelength / 2 * idx, 0)))

    return Receiver(
        fs=2e6,
        noise_figure=4,
        rf_gain=20,
        baseband_gain=50,
        load_resistor=500,
        channels=channels,
    )


def test_tdm_fmcw_rx():
    print("#### TDM FMCW receiver ####")
    tdm = tdm_fmcw_rx()

    print("# TDM FMCW receiver parameters #")
    assert tdm.fs == 2e6
    assert tdm.noise_figure == 4
    assert tdm.rf_gain == 20
    assert tdm.load_resistor == 500
    assert tdm.baseband_gain == 50
    assert tdm.noise_bandwidth == tdm.fs

    print("# TDM FMCW receiver channel #")
    half_wavelength = const.c / 24.125e9 / 2
    assert tdm.channel_size == 8
    assert np.array_equal(
        tdm.locations,
        np.array(
            [
                [0, 0, 0],
                [0, half_wavelength, 0],
                [0, half_wavelength * 2, 0],
                [0, half_wavelength * 3, 0],
                [0, half_wavelength * 4, 0],
                [0, half_wavelength * 5, 0],
                [0, half_wavelength * 6, 0],
                [0, half_wavelength * 7, 0],
            ]
        ),
    )
    assert np.array_equal(
        tdm.az_angles,
        [
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
        ],
    )
    assert np.array_equal(
        tdm.az_patterns,
        [
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
        ],
    )
    assert np.array_equal(
        tdm.el_angles,
        [
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
            np.arange(-90, 91, 180),
        ],
    )
    assert np.array_equal(
        tdm.el_patterns,
        [
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
        ],
    )


def pmcw_rx():
    angle = np.arange(-90, 91, 1)
    pattern = np.ones(181) * 12

    return Receiver(
        fs=250e6,
        noise_figure=10,
        rf_gain=20,
        baseband_gain=30,
        load_resistor=1000,
        channels=[
            dict(
                location=(0, 0, 0),
                azimuth_angle=angle,
                azimuth_pattern=pattern,
                elevation_angle=angle,
                elevation_pattern=pattern,
            )
        ],
    )


def test_pmcw_rx():
    print("#### PMCW receiver ####")
    pmcw = pmcw_rx()

    print("# PMCW receiver parameters #")
    assert pmcw.fs == 250e6
    assert pmcw.noise_figure == 10
    assert pmcw.rf_gain == 20
    assert pmcw.load_resistor == 1000
    assert pmcw.baseband_gain == 30
    assert pmcw.noise_bandwidth == pmcw.fs

    print("# PMCW receiver channel #")
    assert pmcw.channel_size == 1
    assert np.array_equal(pmcw.locations, np.array([[0, 0, 0]]))
    assert np.array_equal(pmcw.az_angles, [np.arange(-90, 91, 1)])
    assert np.array_equal(pmcw.az_patterns, [np.zeros(181)])
    assert np.array_equal(pmcw.el_angles, [np.arange(-90, 91, 1)])
    assert np.array_equal(pmcw.el_patterns, [np.zeros(181)])
