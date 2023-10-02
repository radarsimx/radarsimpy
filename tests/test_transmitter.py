r"""
A Python module for radar simulation

----------
RadarSimPy - A Radar Simulator Built with Python
Copyright (C) 2018 - PRESENT  radarsimx.com
E-mail: info@radarsimx.com
Website: https://radarsimx.com

 ____           _            ____  _          __  __
|  _ \ __ _  __| | __ _ _ __/ ___|(_)_ __ ___ \ \/ /
| |_) / _` |/ _` |/ _` | '__\___ \| | '_ ` _ \ \  /
|  _ < (_| | (_| | (_| | |   ___) | | | | | | |/  \
|_| \_\__,_|\__,_|\__,_|_|  |____/|_|_| |_| |_/_/\_\

"""

from radarsimpy import Transmitter
import scipy.constants as const
import numpy as np
import numpy.testing as npt


def cw_tx():
    return Transmitter(f=24e9, t=10, tx_power=10, pulses=2)


def test_cw_tx():
    print("#### CW transmitter ####")
    cw = cw_tx()

    print("# CW transmitter parameters #")
    # assert np.array_equal(cw.fc_vect, np.ones(2)*24e9)
    # assert cw.pulse_length == 10
    # assert cw.bandwidth == 0
    assert cw.tx_power == 10
    assert cw.prp[0] == 10
    assert cw.pulses == 2

    print("# CW transmitter channel #")
    assert cw.channel_size == 1
    assert np.array_equal(cw.locations, np.array([[0, 0, 0]]))
    assert np.array_equal(cw.az_angles, [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.az_patterns, [np.zeros(2)])
    assert np.array_equal(cw.el_angles, [np.arange(-90, 91, 180)])
    assert np.array_equal(cw.el_patterns, [np.zeros(2)])

    print("# CW transmitter modulation #")
    assert np.array_equal(cw.pulse_mod, [np.ones(cw.pulses)])


def fmcw_tx():
    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6

    tx_channel = dict(
        location=(0, 0, 0),
        azimuth_angle=angle,
        azimuth_pattern=pattern,
        elevation_angle=angle,
        elevation_pattern=pattern,
    )

    return Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=10,
        prp=100e-6,
        pulses=256,
        channels=[tx_channel],
    )


def test_fmcw_tx():
    print("#### FMCW transmitter ####")
    fmcw = fmcw_tx()

    angle = np.arange(-90, 91, 1)
    pattern = 20 * np.log10(np.cos(angle / 180 * np.pi) + 0.01) + 6
    pattern = pattern - np.max(pattern)

    print("# FMCW transmitter parameters #")
    assert np.array_equal(fmcw.fc_vect, np.ones(256) * 24.125e9)
    assert fmcw.pulse_length == 80e-6
    assert fmcw.bandwidth == 100e6
    assert fmcw.tx_power == 10
    assert fmcw.prp[0] == 100e-6
    assert fmcw.pulses == 256

    print("# FMCW transmitter channel #")
    assert fmcw.channel_size == 1
    assert np.array_equal(fmcw.locations, np.array([[0, 0, 0]]))
    assert np.array_equal(fmcw.az_angles, [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.az_patterns, [pattern])
    assert np.array_equal(fmcw.el_angles, [np.arange(-90, 91, 1)])
    assert np.array_equal(fmcw.el_patterns, [pattern])

    print("# FMCW transmitter modulation #")
    assert np.array_equal(fmcw.pulse_mod, [np.ones(fmcw.pulses)])


def tdm_fmcw_tx():
    wavelength = const.c / 24.125e9

    tx_channel_1 = dict(location=(0, -4 * wavelength, 0), delay=0)
    tx_channel_2 = dict(location=(0, 0, 0), delay=100e-6)

    return Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=20,
        prp=200e-6,
        pulses=2,
        channels=[tx_channel_1, tx_channel_2],
    )


def test_tdm_fmcw_tx():
    print("#### TDM FMCW transmitter ####")
    tdm = tdm_fmcw_tx()

    print("# TDM FMCW transmitter parameters #")
    assert np.array_equal(tdm.fc_vect, np.ones(2) * 24.125e9)
    assert tdm.pulse_length == 80e-6
    assert tdm.bandwidth == 100e6
    assert tdm.tx_power == 20
    assert tdm.prp[0] == 200e-6
    assert tdm.pulses == 2

    print("# TDM FMCW transmitter channel #")
    assert tdm.channel_size == 2
    assert np.array_equal(
        tdm.locations, np.array([[0, -4 * const.c / 24.125e9, 0], [0, 0, 0]])
    )
    assert np.array_equal(
        tdm.az_angles, [np.arange(-90, 91, 180), np.arange(-90, 91, 180)]
    )
    assert np.array_equal(tdm.az_patterns, [np.zeros(2), np.zeros(2)])
    assert np.array_equal(
        tdm.el_angles, [np.arange(-90, 91, 180), np.arange(-90, 91, 180)]
    )
    assert np.array_equal(tdm.el_patterns, [np.zeros(2), np.zeros(2)])

    print("# TDM FMCW transmitter modulation #")
    assert np.array_equal(tdm.pulse_mod, [np.ones(tdm.pulses), np.ones(tdm.pulses)])


def pmcw_tx(code1, code2):
    angle = np.arange(-90, 91, 1)
    pattern = np.ones(181) * 12

    pulse_phs1 = np.zeros(np.shape(code1))
    pulse_phs2 = np.zeros(np.shape(code2))
    pulse_phs1[np.where(code1 == 1)] = 0
    pulse_phs1[np.where(code1 == -1)] = 180
    pulse_phs2[np.where(code2 == 1)] = 0
    pulse_phs2[np.where(code2 == -1)] = 180

    mod_t1 = np.arange(0, len(code1)) * 4e-9
    mod_t2 = np.arange(0, len(code2)) * 4e-9

    tx_channel_1 = dict(
        location=(0, 0, 0),
        azimuth_angle=angle,
        azimuth_pattern=pattern,
        elevation_angle=angle,
        elevation_pattern=pattern,
        mod_t=mod_t1,
        phs=pulse_phs1,
    )

    tx_channel_2 = dict(
        location=(0, 0, 0),
        azimuth_angle=angle,
        azimuth_pattern=pattern,
        elevation_angle=angle,
        elevation_pattern=pattern,
        mod_t=mod_t2,
        phs=pulse_phs2,
    )

    return Transmitter(
        f=24.125e9,
        t=2.1e-6,
        tx_power=20,
        pulses=256,
        channels=[tx_channel_1, tx_channel_2],
    )


def test_pmcw_tx():
    code1 = np.array(
        [
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
        ]
    )
    code2 = np.array(
        [
            1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            -1,
            -1,
            1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            -1,
            1,
            1,
            1,
            1,
            1,
            -1,
            -1,
            1,
            -1,
            1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            -1,
            -1,
            -1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            -1,
            1,
            1,
            1,
            -1,
            -1,
        ]
    )

    print("#### PMCW transmitter ####")
    pmcw = pmcw_tx(code1, code2)

    print("# PMCW transmitter parameters #")
    assert np.array_equal(pmcw.fc_vect, np.ones(256) * 24.125e9)
    assert pmcw.pulse_length == 2.1e-6
    assert pmcw.bandwidth == 0
    assert pmcw.tx_power == 20
    assert pmcw.prp[0] == 2.1e-6
    assert pmcw.pulses == 256

    print("# PMCW transmitter channel #")
    assert pmcw.channel_size == 2
    assert np.array_equal(pmcw.locations, np.array([[0, 0, 0], [0, 0, 0]]))
    assert np.array_equal(
        pmcw.az_angles, [np.arange(-90, 91, 1), np.arange(-90, 91, 1)]
    )
    assert np.array_equal(pmcw.az_patterns, [np.zeros(181), np.zeros(181)])
    assert np.array_equal(
        pmcw.el_angles, [np.arange(-90, 91, 1), np.arange(-90, 91, 1)]
    )
    assert np.array_equal(pmcw.el_patterns, [np.zeros(181), np.zeros(181)])

    print("# PMCW transmitter modulation #")
    npt.assert_almost_equal(pmcw.waveform_mod[0]["var"], code1)
    npt.assert_almost_equal(pmcw.waveform_mod[1]["var"], code2)

    npt.assert_almost_equal(pmcw.waveform_mod[0]["t"], np.arange(0, len(code1)) * 4e-9)
    npt.assert_almost_equal(pmcw.waveform_mod[1]["t"], np.arange(0, len(code2)) * 4e-9)

    # assert np.array_equal(pmcw.chip_length, [4e-9, 4e-9])


def test_fsk_tx():
    print("#### FSK transmitter ####")


def test_bpm_fmcw_tx():
    print("#### BPM FMCW transmitter ####")
