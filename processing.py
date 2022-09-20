#!python
# distutils: language = c++

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
# Copyright (C) 2018 - PRESENT  Zhengyu Peng
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
from scipy.signal import convolve2d
from .tools import log_factorial


def range_fft(data, rwin=None, n=None):
    """
    Calculate range profile matrix

    :param numpy.3darray data:
        Baseband data, ``[channels, pulses, adc_samples]``
    :param numpy.1darray rwin:
        Window for FFT, length should be equal to adc_samples. (default is
        a square window)
    :param int n:
        FFT size, if n > adc_samples, zero-padding will be applied.
        (default is None)

    :return: A 3D array of range profile, ``[channels, pulses, range]``
    :rtype: numpy.3darray
    """

    shape = np.shape(data)

    if rwin is None:
        rwin = 1
    else:
        rwin = np.tile(rwin[np.newaxis, np.newaxis, ...],
                       (shape[0], shape[1], 1))

    return np.fft.fft(data * rwin, n=n, axis=2)


def doppler_fft(data, dwin=None, n=None):
    """
    Calculate range-Doppler matrix

    :param numpy.3darray data:
        Range profile matrix, ``[channels, pulses, adc_samples]``
    :param numpy.1darray dwin:
        Window for FFT, length should be equal to adc_samples. (default is
        a square window)
    :param int n:
        FFT size, if n > adc_samples, zero-padding will be applied.
        (default is None)

    :return: A 3D array of range-Doppler map, ``[channels, Doppler, range]``
    :rtype: numpy.3darray
    """

    shape = np.shape(data)

    if dwin is None:
        dwin = 1
    else:
        dwin = np.tile(dwin[np.newaxis, ..., np.newaxis],
                       (shape[0], 1, shape[2]))

    return np.fft.fft(data * dwin, n=n, axis=1)


def range_doppler_fft(data, rwin=None, dwin=None, rn=None, dn=None):
    """
    Range-Doppler processing

    :param numpy.3darray data:
        Baseband data, ``[channels, pulses, adc_samples]``
    :param numpy.1darray rwin:
        Range window for FFT, length should be equal to adc_samples.
        (default is a square window)
    :param numpy.1darray dwin:
        Doppler window for FFT, length should be equal to adc_samples.
        (default is a square window)
    :param int rn:
        Range FFT size, if n > adc_samples, zero-padding will be applied.
        (default is None)
    :param int dn:
        Doppler FFT size, if n > adc_samples, zero-padding will be applied.
        (default is None)

    :return: A 3D array of range-Doppler map, ``[channels, Doppler, range]``
    :rtype: numpy.3darray
    """

    return doppler_fft(range_fft(data, rwin=rwin, n=rn), dwin=dwin, n=dn)


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


def cfar_ca_1d(data, guard, trailing, pfa=1e-5, axis=0, offset=None):
    """
    1-D Cell Averaging CFAR (CA-CFAR)

    :param data:
        Radar data
    :type data: numpy.1darray or numpy.2darray
    :param int guard:
        Number of guard cells on one side, total guard cells are ``2*guard``
    :param int trailing:
        Number of trailing cells on one side, total trailing cells are
        ``2*trailing``
    :param float pfa:
        Probability of false alarm. ``default 1e-5``
    :param int axis:
        The axis to calculat CFAR. ``default 0``
    :param float offset:
        CFAR threshold offset. If offect is None, threshold offset is
        ``2*trailing(pfa^(-1/2/trailing)-1)``. ``default None``

    :return: CFAR threshold. The dimension is the same as ``data``
    :rtype: numpy.1darray or numpy.2darray
    """

    data = np.abs(data)
    data_shape = np.shape(data)
    cfar = np.zeros_like(data)

    if offset is None:
        a = trailing*2*(pfa**(-1/trailing/2)-1)
    else:
        a = offset

    cfar_win = np.ones((guard+trailing)*2+1)
    cfar_win[trailing:(trailing+guard*2+1)] = 0
    cfar_win = cfar_win/np.sum(cfar_win)

    if axis == 0:
        if data.ndim == 1:
            cfar = a*np.convolve(data, cfar_win, mode='same')
        elif data.ndim == 2:
            for idx in range(0, data_shape[1]):
                cfar[:, idx] = a * \
                    np.convolve(data[:, idx], cfar_win, mode='same')
    elif axis == 1:
        for idx in range(0, data_shape[0]):
            cfar[idx, :] = a*np.convolve(data[idx, :], cfar_win, mode='same')

    return cfar


def cfar_ca_2d(data, guard, trailing, pfa=1e-5, offset=None):
    """
    2-D Cell Averaging CFAR (CA-CFAR)

    :param data:
        Radar data
    :type data: numpy.1darray or numpy.2darray
    :param guard:
        Number of guard cells on one side, total guard cells are ``2*guard``
    :type guard: int or list[int]
    :param trailing:
        Number of trailing cells on one side, total trailing cells are
        ``2*trailing``
    :type trailing: int or list[int]
    :param float pfa:
        Probability of false alarm. ``default 1e-5``
    :param float offset:
        CFAR threshold offset. If offect is None, threshold offset is
        ``2*trailing(pfa^(-1/2/trailing)-1)``. ``default None``

    :return: CFAR threshold. The dimension is the same as ``data``
    :rtype: numpy.1darray or numpy.2darray
    """

    data = np.abs(data)

    if offset is None:
        a = trailing*2*(pfa**(-1/trailing/2)-1)
    else:
        a = offset

    guard = np.array(guard)
    if guard.size == 1:
        guard = np.tile(guard, 2)
    trailing = np.array(trailing)
    if trailing.size == 1:
        trailing = np.tile(trailing, 2)

    cfar_win = np.ones(((guard+trailing)*2+1))
    cfar_win[trailing[0]:(trailing[0]+guard[0]*2+1),
             trailing[1]:(trailing[1]+guard[1]*2+1)] = 0
    cfar_win = cfar_win/np.sum(cfar_win)

    return a*convolve2d(data, cfar_win, mode='same', boundary='wrap')


def os_cfar_threshold(k, n, pfa):
    """
    Use Secant method to calculate OS-CFAR's threshold

    :param int n:
        Number of cells around CUT (cell under test) for calculating
    :param int k:
        Rank in the order
    :param float pfa:
        Probability of false alarm

    :return: CFAR threshold
    :rtype: float

    *Reference*

    Rohling, Hermann. "Radar CFAR thresholding in clutter and multiple target
    situations." IEEE transactions on aerospace and electronic systems 4
    (1983): 608-621.
    """

    def fun(k, n, Tos, pfa):
        return log_factorial(n)-log_factorial(n-k) - \
            np.sum(np.log(np.arange(n, n-k, -1)+Tos))-np.log(pfa)

    max_iter = 10000

    t_max = 1e32
    t_min = 1

    for idx in range(0, max_iter):

        m_n = t_max-fun(k, n, t_max, pfa)*(t_min-t_max) / \
            (fun(k, n, t_min, pfa) -
             fun(k, n, t_max, pfa))
        f_m_n = fun(k, n, m_n, pfa)
        if f_m_n == 0:
            return m_n
        elif np.abs(f_m_n) < 0.0001:
            return m_n
        elif fun(k, n, t_max, pfa)*f_m_n < 0:
            t_max = t_max
            t_min = m_n
        elif fun(k, n, t_min, pfa)*f_m_n < 0:
            t_max = m_n
            t_min = t_min
        else:
            # print("Secant method fails.")
            break

    return None


def cfar_os_1d(
        data,
        guard,
        trailing,
        k,
        pfa=1e-5,
        axis=0,
        offset=None):
    """
    1-D Ordered Statistic CFAR (OS-CFAR)

    For edge cells, use rollovered cells to fill the missing cells.

    :param data:
        Radar data
    :type data: numpy.1darray or numpy.2darray
    :param int guard:
        Number of guard cells on one side, total guard cells are ``2*guard``
    :param int trailing:
        Number of trailing cells on one side, total trailing cells are
        ``2*trailing``
    :param int k:
        Rank in the order
    :param float pfa:
        Probability of false alarm. ``default 1e-5``
    :param int axis:
        The axis to calculat CFAR. ``default 0``
    :param float offset:
        CFAR threshold offset. If offect is None, threshold offset is
        calculated from ``pfa``. ``default None``

    :return: CFAR threshold. The dimension is the same as ``data``
    :rtype: numpy.1darray or numpy.2darray

    *Reference*

    [1] H. Rohling, “Radar CFAR Thresholding in Clutter and Multiple Target
    Situations,” IEEE Trans. Aerosp. Electron. Syst., vol. AES-19, no. 4,
    pp. 608-621, 1983.
    """

    data = np.abs(data)
    data_shape = np.shape(data)
    cfar = np.zeros_like(data)
    leading = trailing
    # trailing = n-leading

    if offset is None:
        a = os_cfar_threshold(k, trailing*2, pfa)
    else:
        a = offset

    if axis == 0:
        for idx in range(0, data_shape[0]):
            win_idx = np.mod(
                np.concatenate(
                    [np.arange(idx-leading-guard, idx-guard, 1),
                        np.arange(idx+1+guard, idx+1+trailing+guard, 1)]
                ), data_shape[0])
            if data.ndim == 1:
                samples = np.sort(data[win_idx.astype(int)])
                cfar[idx] = a*samples[k]
            elif data.ndim == 2:
                samples = np.sort(data[win_idx.astype(int), :], axis=0)
                cfar[idx, :] = a*samples[k, :]

    elif axis == 1:
        for idx in range(0, data_shape[1]):
            win_idx = np.mod(
                np.concatenate(
                    [np.arange(idx-leading-guard, idx-guard, 1),
                     np.arange(idx+1+guard, idx+1+trailing+guard, 1)]
                ), data_shape[1])
            samples = np.sort(data[:, win_idx.astype(int)], axis=1)

            cfar[:, idx] = a*samples[:, k]

    return cfar


def cfar_os_2d(
        data,
        guard,
        trailing,
        k,
        pfa=1e-5,
        offset=None):
    """
    2-D Ordered Statistic CFAR (OS-CFAR)

    For edge cells, use rollovered cells to fill the missing cells.

    :param data:
        Radar data
    :type data: numpy.1darray or numpy.2darray
    :param guard:
        Number of guard cells on one side, total guard cells are ``2*guard``
    :type guard: int or list[int]
    :param trailing:
        Number of trailing cells on one side, total trailing cells are
        ``2*trailing``
    :type trailing: int or list[int]
    :param int k:
        Rank in the order
    :param float pfa:
        Probability of false alarm. ``default 1e-5``
    :param float offset:
        CFAR threshold offset. If offect is None, threshold offset is
        calculated from ``pfa``. ``default None``

    :return: CFAR threshold. The dimension is the same as ``data``
    :rtype: numpy.1darray or numpy.2darray

    *Reference*

    [1] H. Rohling, “Radar CFAR Thresholding in Clutter and Multiple Target
    Situations,” IEEE Trans. Aerosp. Electron. Syst., vol. AES-19, no. 4,
    pp. 608-621, 1983.
    """

    data = np.abs(data)
    data_shape = np.shape(data)
    cfar = np.zeros_like(data)

    guard = np.array(guard)
    if guard.size == 1:
        guard = np.tile(guard, 2)
    trailing = np.array(trailing)
    if trailing.size == 1:
        trailing = np.tile(trailing, 2)

    if offset is None:
        a = os_cfar_threshold(k, np.sum(trailing)*2, pfa)
    else:
        a = offset

    for idx_0 in range(0, data_shape[0]):
        for idx_1 in range(0, data_shape[1]):
            win_idx_0 = np.mod(
                np.concatenate(
                    [np.arange(idx_0-trailing[0]-guard[0],
                               idx_0-guard[0],
                               1),
                        np.arange(idx_0+1+guard[0],
                                  idx_0+1+trailing[0]+guard[0],
                                  1)]
                ), data_shape[0])
            win_idx_1 = np.mod(
                np.concatenate(
                    [np.arange(idx_1-trailing[1]-guard[1],
                               idx_1-guard[1],
                               1),
                        np.arange(idx_1+1+guard[1],
                                  idx_1+1+trailing[1]+guard[1],
                                  1)]
                ), data_shape[1])

            x, y = np.meshgrid(win_idx_0, win_idx_1)
            samples = np.sort(data[x.astype(int), y.astype(int)].flatten())

            cfar[idx_0, idx_1] = a*samples[k]

    return cfar
