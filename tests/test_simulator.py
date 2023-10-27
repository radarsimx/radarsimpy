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

from .test_radar import cw_radar, fmcw_radar, tdm_fmcw_radar, pmcw_radar

import numpy as np
import numpy.testing as npt
import scipy.constants as const
from scipy import signal

from radarsimpy.simulator import simc
import radarsimpy.processing as proc


def test_sim_cw():
    radar = cw_radar()

    target = dict(
        location=(1.5 + 1e-3 * np.sin(2 * np.pi * 1 * radar.timestamp), 0, 0),
        rcs=0,
        phase=0,
    )
    targets = [target]

    data = simc(radar, targets, noise=False)
    timestamp = data["timestamp"]
    baseband = data["baseband"]
    demod = np.angle(baseband[0, 0, :])

    nfft = 2048
    spectrum = np.abs(np.fft.fft(demod - np.mean(demod), nfft))
    fft_length = np.shape(spectrum)[0]
    f = np.linspace(0, radar.receiver.bb_prop["fs"], nfft)

    npt.assert_almost_equal(f[np.argmax(spectrum[0 : int(nfft / 2)])], 1, decimal=2)
    npt.assert_almost_equal(
        timestamp[0, 0, :],
        np.arange(0, radar.samples_per_pulse) / radar.receiver.bb_prop["fs"],
    )


def test_sim_fmcw():
    radar = fmcw_radar()
    target_1 = dict(location=(200, 0, 0), speed=(-5, 0, 0), rcs=20, phase=0)
    target_2 = dict(location=(95, 20, 0), speed=(-50, 0, 0), rcs=15, phase=0)
    target_3 = dict(location=(30, -5, 0), speed=(-22, 0, 0), rcs=5, phase=0)

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
            radar.channel_size * radar.frames,
            radar.transmitter.waveform_prop["pulses"],
            radar.samples_per_pulse,
        ),
        np.shape(timestamp),
    )
    assert np.array_equal(
        (
            radar.channel_size * radar.frames,
            radar.transmitter.waveform_prop["pulses"],
            radar.samples_per_pulse,
        ),
        np.shape(baseband),
    )

    npt.assert_almost_equal(
        timestamp[0, 0, :],
        (np.arange(0, radar.samples_per_pulse) / radar.receiver.bb_prop["fs"]),
    )
    npt.assert_almost_equal(
        timestamp[0, :, 0],
        (
            np.arange(0, radar.transmitter.waveform_prop["pulses"])
            * radar.transmitter.waveform_prop["prp"][0]
        ),
    )

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)
    doppler_window = signal.windows.chebwin(
        radar.transmitter.waveform_prop["pulses"], at=60
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
        * radar.receiver.bb_prop["fs"]
        * radar.transmitter.waveform_prop["pulse_length"]
        / radar.transmitter.waveform_prop["bandwidth"]
        / 2
    )

    unambiguous_speed = (
        const.c / radar.transmitter.waveform_prop["prp"][0] / 24.125e9 / 2
    )

    range_axis = np.linspace(0, max_range, radar.samples_per_pulse, endpoint=False)

    rng_dets = np.sort(range_axis[rng_peaks])
    npt.assert_almost_equal(rng_targets, rng_dets, decimal=0)

    doppler_axis = np.linspace(
        -unambiguous_speed, 0, radar.transmitter.waveform_prop["pulses"], endpoint=False
    )

    dop_dets = np.sort(doppler_axis[dop_peaks])
    npt.assert_almost_equal(dop_targets, dop_dets, decimal=0)

    # frame 2
    rng_dop = rng_dop - np.max(rng_dop[1, :, :])

    max_rng = np.max(rng_dop[1, :, :], axis=0)
    max_dop = np.max(rng_dop[1, :, :], axis=1)

    rng_peaks = signal.find_peaks(max_rng, height=-40)[0]
    dop_peaks = signal.find_peaks(max_dop, height=-40)[0]

    range_axis = np.linspace(0, max_range, radar.samples_per_pulse, endpoint=False)

    rng_dets = np.sort(range_axis[rng_peaks])
    npt.assert_almost_equal(np.array([9.0, 48.0, 195.0]), rng_dets, decimal=0)

    doppler_axis = np.linspace(
        -unambiguous_speed, 0, radar.transmitter.waveform_prop["pulses"], endpoint=False
    )

    dop_dets = np.sort(doppler_axis[dop_peaks])
    npt.assert_almost_equal(
        np.array([-45.66062176, -18.45854922, -5.1003886]), dop_dets, decimal=0
    )


def test_sim_tdm_fmcw():
    radar = tdm_fmcw_radar()
    target_1 = dict(location=(120, 0, 0), speed=(0, 0, 0), rcs=25, phase=0)
    target_2 = dict(location=(80, -80, 0), speed=(0, 0, 0), rcs=20, phase=0)
    target_3 = dict(location=(30, 20, 0), speed=(0, 0, 0), rcs=8, phase=0)

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
            radar.channel_size,
            radar.transmitter.waveform_prop["pulses"],
            radar.samples_per_pulse,
        ),
        np.shape(timestamp),
    )
    assert np.array_equal(
        (
            radar.channel_size,
            radar.transmitter.waveform_prop["pulses"],
            radar.samples_per_pulse,
        ),
        np.shape(baseband),
    )

    npt.assert_almost_equal(
        timestamp[0, 0, :],
        (np.arange(0, radar.samples_per_pulse) / radar.receiver.bb_prop["fs"]),
    )
    npt.assert_almost_equal(
        timestamp[0, :, 0],
        (
            np.arange(0, radar.transmitter.waveform_prop["pulses"])
            * radar.transmitter.waveform_prop["prp"][0]
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

    range_window = signal.windows.chebwin(radar.samples_per_pulse, at=60)
    range_profile = proc.range_fft(baseband, range_window)

    rng_nci = 20 * np.log10(np.mean(np.abs(range_profile[:, 0, :]), axis=0))
    rng_nci = rng_nci - np.max(rng_nci)
    rng_peaks = signal.find_peaks(rng_nci, height=-10)[0]

    max_range = (
        const.c
        * radar.receiver.bb_prop["fs"]
        * radar.transmitter.waveform_prop["pulse_length"]
        / radar.transmitter.waveform_prop["bandwidth"]
        / 2
    )

    range_axis = np.linspace(0, max_range, radar.samples_per_pulse, endpoint=False)

    rng_dets = np.sort(range_axis[rng_peaks])

    npt.assert_almost_equal(rng_targets, rng_dets, decimal=0)


def test_sim_pmcw():
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
    radar = pmcw_radar()

    target_1 = dict(location=(20, 0, 0), speed=(-185, 0, 0), rcs=20, phase=0)

    target_2 = dict(location=(70, 0, 0), speed=(0, 0, 0), rcs=35, phase=0)

    target_3 = dict(location=(33, 10, 0), speed=(97, 0, 0), rcs=20, phase=0)

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
            radar.channel_size,
            radar.transmitter.waveform_prop["pulses"],
            radar.samples_per_pulse,
        ),
        np.shape(timestamp),
    )
    assert np.array_equal(
        (
            radar.channel_size,
            radar.transmitter.waveform_prop["pulses"],
            radar.samples_per_pulse,
        ),
        np.shape(baseband),
    )

    npt.assert_almost_equal(
        timestamp[0, 0, :],
        (np.arange(0, radar.samples_per_pulse) / radar.receiver.bb_prop["fs"]),
    )
    npt.assert_almost_equal(
        timestamp[0, :, 0],
        (
            np.arange(0, radar.transmitter.waveform_prop["pulses"])
            * radar.transmitter.waveform_prop["prp"][0]
        ),
    )

    code_length = 255
    range_profile = np.zeros(
        (radar.channel_size, radar.transmitter.waveform_prop["pulses"], code_length),
        dtype=complex,
    )

    for pulse_idx in range(0, radar.transmitter.waveform_prop["pulses"]):
        for bin_idx in range(0, code_length):
            range_profile[:, pulse_idx, bin_idx] = np.sum(
                code2 * baseband[1, pulse_idx, bin_idx : (bin_idx + code_length)]
            )

    bin_size = const.c / 2 * 4e-9
    range_bin = np.arange(0, code_length, 1) * bin_size

    doppler_window = signal.windows.chebwin(
        radar.transmitter.waveform_prop["pulses"], at=50
    )

    range_doppler = np.zeros(np.shape(range_profile), dtype=complex)
    for ii in range(0, radar.channel_size):
        for jj in range(0, code_length):
            range_doppler[ii, :, jj] = np.fft.fftshift(
                np.fft.fft(
                    range_profile[ii, :, jj] * doppler_window,
                    n=radar.transmitter.waveform_prop["pulses"],
                )
            )
    unambiguous_speed = (
        const.c / radar.transmitter.waveform_prop["prp"][0] / 24.125e9 / 2
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
        radar.transmitter.waveform_prop["pulses"],
        endpoint=False,
    )
    dop_dets = np.sort(doppler_axis[dop_peaks])
    npt.assert_almost_equal(dop_targets, dop_dets, decimal=0)
