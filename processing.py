#!python
# cython: language_level=3

# Script for radar signal processing

# This script requires that `numpy` and be installed within the Python
# environment you are running this script in.

# This file can be imported as a module and contains the following
# functions:

# * cal_range_profile - calculate range profile matrix
# * cal_range_doppler - range-Doppler processing
# * get_polar_image - convert cartesian coordinate to polar

# ----------
# RadarSimPy - A Radar Simulator Built with Python
# Copyright (C) 2018 - 2020  Zhengyu Peng
# E-mail: zpeng.me@gmail.com
# Website: https://zpeng.me

# `                      `
# -:.                  -#:
# -//:.              -###:
# -////:.          -#####:
# -/:.://:.      -###++##:
# ..   `://:-  -###+. :##:
#        `:/+####+.   :##:
# .::::::::/+###.     :##:
# .////-----+##:    `:###:
#  `-//:.   :##:  `:###/.
#    `-//:. :##:`:###/.
#      `-//:+######/.
#        `-/+####/.
#          `+##+.
#           :##:
#           :##:
#           :##:
#           :##:
#           :##:
#            .+:


import numpy as np


def cal_range_profile(radar, baseband, range_window=1, n=None):
    """
    Calculate range profile matrix

    :param Radar radar:
        A well defined radar system
    :param numpy.3darray baseband:
        Baseband data, ``[channels, pulses, adc_samples]``
    :param numpy.1darray range_window:
        Window for FFT, length should be equal to adc_samples. (default is
        a square window)
    :param int n:
        FFT size, if n > adc_samples, zero-padding will be applied.
        (default is None)

    :return: A 3D array of range profile, ``[channels, pulses, range]``
    :rtype: numpy.3darray
    """

    if n is None:
        rng_profile = np.zeros(np.shape(baseband), dtype=complex)
    else:
        rng_profile = np.zeros((
            radar.channel_size*radar.frames,
            radar.transmitter.pulses,
            n,
        ),
            dtype=complex)
    for ii in range(0, radar.channel_size*radar.frames):
        for jj in range(0, radar.transmitter.pulses):
            rng_profile[ii, jj, :] = np.fft.fft(
                baseband[ii, jj, :] * range_window,
                n=n,
            )

    return rng_profile


def cal_range_doppler(radar, range_profile, doppler_window=1, fft_shift=False, n=None):
    """
    Calculate range-Doppler matrix

    :param Radar radar:
        A well defined radar system
    :param numpy.3darray range_profile: 
        Range profile matrix, ``[channels, pulses, adc_samples]``
    :param numpy.1darray doppler_window:
        Window for FFT, length should be equal to adc_samples. (default is
        a square window)
    :param bool fft_shift:
        Perform FFT shift.  (default is False)

    :return: A 3D array of range profile, ``[channels, Doppler, range]``
    :rtype: numpy.3darray
    """

    if n is None:
        rng_doppler = np.zeros(np.shape(range_profile), dtype=complex)
    else:
        rng_doppler = np.zeros(
            (np.shape(range_profile)[0], n, np.shape(range_profile)[2]), dtype=complex)

    for ii in range(0, radar.channel_size*radar.frames):
        for jj in range(0, np.shape(rng_doppler)[2]):
            if fft_shift:
                rng_doppler[ii, :, jj] = np.fft.fftshift(
                    np.fft.fft(
                        range_profile[ii, :, jj] * doppler_window,
                        n=n,
                    ))
            else:
                rng_doppler[ii, :, jj] = np.fft.fft(
                    range_profile[ii, :, jj] * doppler_window,
                    n=n,
                )

    return rng_doppler


def get_polar_image(image, range_bins, angle_bins, fov_deg):
    """
    Convert cartesian coordinate to polar

    :param numpy.2darray image:
        Data with cartesian coordinate, [range, angle]
    :param int range_bins:
        Number of range bins
    :param int angle_bins:
        Number of angle bins
    :param float fov_deg:
        Field of view (deg)

    :return: A 2D image with polar coordinate
    :rtype: numpy.2darray
    """

    angle_bin_res = fov_deg / angle_bins

    latitude_bins = int(range_bins * np.sin(fov_deg / 360 * np.pi) + 1)
    polar = np.zeros((range_bins, latitude_bins * 2), dtype=complex)

    x = np.arange(1, range_bins, 1, dtype=int)
    y = np.arange(0, latitude_bins * 2, 1, dtype=int)
    X_data, Y_data = np.meshgrid(x, y)

    b = 180 * np.arctan(
        (Y_data - latitude_bins) /
        X_data) / angle_bin_res / np.pi + fov_deg / angle_bin_res / 2
    a = X_data / (np.cos((angle_bin_res * b - fov_deg / 2) / 180 * np.pi))
    b = b.astype(int)
    a = a.astype(int)

    idx = np.where(
        np.logical_and(
            np.logical_and(np.less(b, angle_bins), np.greater_equal(b, 0)),
            np.logical_and(np.less(a, range_bins), np.greater_equal(a, 0))))
    b = b[idx]
    a = a[idx]
    xx = X_data[idx]
    yy = Y_data[idx]

    polar[xx, yy] = image[a, b]
    return polar
