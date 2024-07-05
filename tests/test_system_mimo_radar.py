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
from radarsimpy.simulator import simc  # pylint: disable=no-name-in-module
import radarsimpy.processing as proc


def test_sim_tdm_fmcw():
    """
    Test the TDM-FMCW radar simulator.
    """
    wavelength = const.c / 24.125e9

    tx_channel_1 = {"location": (0, -4 * wavelength, 0), "delay": 0}
    tx_channel_2 = {"location": (0, 0, 0), "delay": 100e-6}

    tx = Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=20,
        prp=200e-6,
        pulses=2,
        channels=[tx_channel_1, tx_channel_2],
    )

    channels = []
    for idx in range(0, 8):
        channels.append({"location": (0, wavelength / 2 * idx, 0)})

    rx = Receiver(
        fs=2e6,
        noise_figure=4,
        rf_gain=20,
        baseband_gain=50,
        load_resistor=500,
        channels=channels,
    )
    radar = Radar(transmitter=tx, receiver=rx)
    target_1 = {"location": (120, 0, 0), "speed": (0, 0, 0), "rcs": 25, "phase": 0}
    target_2 = {"location": (80, -80, 0), "speed": (0, 0, 0), "rcs": 20, "phase": 0}
    target_3 = {"location": (30, 20, 0), "speed": (0, 0, 0), "rcs": 8, "phase": 0}

    rng_targets = np.sort(
        np.array(
            [
                np.sqrt(target_1["location"][0] ** 2 + target_1["location"][1] ** 2),
                np.sqrt(target_2["location"][0] ** 2 + target_2["location"][1] ** 2),
                np.sqrt(target_3["location"][0] ** 2 + target_3["location"][1] ** 2),
            ]
        )
    )

    targets = [target_1, target_2, target_3]

    data = simc(radar, targets, noise=False)
    timestamp = data["timestamp"]
    baseband = data["baseband"]

    assert np.array_equal(
        (
            radar.array_prop["size"],
            radar.radar_prop["transmitter"].waveform_prop["pulses"],
            radar.sample_prop["samples_per_pulse"],
        ),
        np.shape(timestamp),
    )
    assert np.array_equal(
        (
            radar.array_prop["size"],
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
    npt.assert_almost_equal(
        timestamp[:, 0, 0],
        np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                100e-6,
                100e-6,
                100e-6,
                100e-6,
                100e-6,
                100e-6,
                100e-6,
                100e-6,
            ]
        ),
    )
    npt.assert_almost_equal(
        timestamp[:, 1, 0],
        np.array(
            [
                200e-6,
                200e-6,
                200e-6,
                200e-6,
                200e-6,
                200e-6,
                200e-6,
                200e-6,
                300e-6,
                300e-6,
                300e-6,
                300e-6,
                300e-6,
                300e-6,
                300e-6,
                300e-6,
            ]
        ),
    )

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile = proc.range_fft(baseband, range_window)

    rng_nci = 20 * np.log10(np.mean(np.abs(range_profile[:, 0, :]), axis=0))
    rng_nci = rng_nci - np.max(rng_nci)
    rng_peaks = signal.find_peaks(rng_nci, height=-10)[0]

    max_range = (
        const.c
        * radar.radar_prop["receiver"].bb_prop["fs"]
        * radar.radar_prop["transmitter"].waveform_prop["pulse_length"]
        / radar.radar_prop["transmitter"].waveform_prop["bandwidth"]
        / 2
    )

    range_axis = np.linspace(
        0, max_range, radar.sample_prop["samples_per_pulse"], endpoint=False
    )

    rng_dets = np.sort(range_axis[rng_peaks])

    npt.assert_almost_equal(rng_targets, rng_dets, decimal=0)


def test_sim_pmcw():
    """
    Test the PMCW radar simulator.
    """
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

    tx_channel_1 = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
        "mod_t": mod_t1,
        "phs": pulse_phs1,
    }

    tx_channel_2 = {
        "location": (0, 0, 0),
        "azimuth_angle": angle,
        "azimuth_pattern": pattern,
        "elevation_angle": angle,
        "elevation_pattern": pattern,
        "mod_t": mod_t2,
        "phs": pulse_phs2,
    }

    tx = Transmitter(
        f=24.125e9,
        t=2.1e-6,
        tx_power=20,
        pulses=256,
        channels=[tx_channel_1, tx_channel_2],
    )

    rx = Receiver(
        fs=250e6,
        noise_figure=10,
        rf_gain=20,
        baseband_gain=30,
        load_resistor=1000,
        channels=[
            {
                "location": (0, 0, 0),
                "azimuth_angle": angle,
                "azimuth_pattern": pattern,
                "elevation_angle": angle,
                "elevation_pattern": pattern,
            }
        ],
    )
    radar = Radar(transmitter=tx, receiver=rx)

    target_1 = {"location": (20, 0, 0), "speed": (-185, 0, 0), "rcs": 20, "phase": 0}

    target_2 = {"location": (70, 0, 0), "speed": (0, 0, 0), "rcs": 35, "phase": 0}

    target_3 = {"location": (33, 10, 0), "speed": (97, 0, 0), "rcs": 20, "phase": 0}

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

    data = simc(radar, targets, noise=False)
    timestamp = data["timestamp"]
    baseband = data["baseband"]

    assert np.array_equal(
        (
            radar.array_prop["size"],
            radar.radar_prop["transmitter"].waveform_prop["pulses"],
            radar.sample_prop["samples_per_pulse"],
        ),
        np.shape(timestamp),
    )
    assert np.array_equal(
        (
            radar.array_prop["size"],
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

    code_length = 255
    range_profile = np.zeros(
        (
            radar.array_prop["size"],
            radar.radar_prop["transmitter"].waveform_prop["pulses"],
            code_length,
        ),
        dtype=complex,
    )

    for pulse_idx in range(0, radar.radar_prop["transmitter"].waveform_prop["pulses"]):
        for bin_idx in range(0, code_length):
            range_profile[:, pulse_idx, bin_idx] = np.sum(
                code2 * baseband[1, pulse_idx, bin_idx : (bin_idx + code_length)]
            )

    bin_size = const.c / 2 * 4e-9
    range_bin = np.arange(0, code_length, 1) * bin_size

    doppler_window = signal.windows.chebwin(
        radar.radar_prop["transmitter"].waveform_prop["pulses"], at=50
    )

    range_doppler = np.zeros(np.shape(range_profile), dtype=complex)
    for ii in range(0, radar.array_prop["size"]):
        for jj in range(0, code_length):
            range_doppler[ii, :, jj] = np.fft.fftshift(
                np.fft.fft(
                    range_profile[ii, :, jj] * doppler_window,
                    n=radar.radar_prop["transmitter"].waveform_prop["pulses"],
                )
            )
    unambiguous_speed = (
        const.c / radar.radar_prop["transmitter"].waveform_prop["prp"][0] / 24.125e9 / 2
    )

    rng_dop = 20 * np.log10(np.abs(range_doppler[1, :, :]))
    rng_dop = rng_dop - np.max(rng_dop)

    max_rng = np.max(rng_dop, axis=0)
    max_dop = np.max(rng_dop, axis=1)

    rng_peaks = signal.find_peaks(max_rng, height=-15)[0]
    dop_peaks = signal.find_peaks(max_dop, height=-15)[0]

    range_axis = np.arange(0, code_length, 1) * bin_size
    rng_dets = np.sort(range_axis[rng_peaks])
    npt.assert_almost_equal(rng_targets, rng_dets, decimal=0)

    doppler_axis = np.linspace(
        -unambiguous_speed / 2,
        unambiguous_speed / 2,
        radar.radar_prop["transmitter"].waveform_prop["pulses"],
        endpoint=False,
    )
    dop_dets = np.sort(doppler_axis[dop_peaks])
    npt.assert_almost_equal(dop_targets, dop_dets, decimal=0)
