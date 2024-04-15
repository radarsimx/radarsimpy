"""
Script for radar signal processing

This script requires that `numpy` and `scipy` be installed within the Python
environment you are running this script in.

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

from warnings import warn
import numpy as np
from scipy.signal import convolve, find_peaks
from scipy import linalg
from scipy import fft
from .tools import log_factorial  # pylint: disable=no-name-in-module


def range_fft(data, rwin=None, n=None) -> np.ndarray:
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
        rwin = np.tile(rwin[np.newaxis, np.newaxis, ...], (shape[0], shape[1], 1))

    return fft.fft(data * rwin, n=n, axis=2)


def doppler_fft(data, dwin=None, n=None) -> np.ndarray:
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
        dwin = np.tile(dwin[np.newaxis, ..., np.newaxis], (shape[0], 1, shape[2]))

    return fft.fft(data * dwin, n=n, axis=1)


def range_doppler_fft(data, rwin=None, dwin=None, rn=None, dn=None) -> np.ndarray:
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


def cfar_ca_1d(
    data, guard, trailing, pfa=1e-5, axis=0, detector="squarelaw", offset=None
):
    """
    1-D Cell Averaging CFAR (CA-CFAR)

    :param data:
        Amplitude/Power data. Amplitude data for ``linear`` detector,
        Power data for ``squarelaw`` detector
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
    :param str detector:
        Detector type, ``linear`` or ``squarelaw``. ``default squarelaw``
    :param float offset:
        CFAR threshold offset. If offect is None, threshold offset is
        ``2*trailing(pfa^(-1/2/trailing)-1)``. ``default None``

    :return: CFAR threshold. The dimension is the same as ``data``
    :rtype: numpy.1darray or numpy.2darray
    """

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    data_shape = np.shape(data)
    cfar = np.zeros_like(data)

    if offset is None:
        if detector == "squarelaw":
            a = trailing * 2 * (pfa ** (-1 / (trailing * 2)) - 1)
        elif detector == "linear":
            a = np.sqrt(trailing * 2 * (pfa ** (-1 / (trailing * 2)) - 1))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    cfar_win = np.ones((guard + trailing) * 2 + 1)
    cfar_win[trailing : (trailing + guard * 2 + 1)] = 0
    cfar_win = cfar_win / np.sum(cfar_win)

    if axis == 0:
        if data.ndim == 1:
            cfar = a * convolve(data, cfar_win, mode="same")
        elif data.ndim == 2:
            for idx in range(0, data_shape[1]):
                cfar[:, idx] = a * convolve(data[:, idx], cfar_win, mode="same")
    elif axis == 1:
        for idx in range(0, data_shape[0]):
            cfar[idx, :] = a * convolve(data[idx, :], cfar_win, mode="same")

    return cfar


def cfar_ca_2d(data, guard, trailing, pfa=1e-5, detector="squarelaw", offset=None):
    """
    2-D Cell Averaging CFAR (CA-CFAR)

    :param data:
        Amplitude/Power data. Amplitude data for ``linear`` detector,
        Power data for ``squarelaw`` detector
    :type data: numpy.1darray or numpy.2darray
    :param guard:
        Number of guard cells on one side, total guard cells are ``2*guard``.
        When ``guard`` is a list, ``guard[0]`` is for axis 0, and ``guard[1]``
        is for axis 1. If ``guard`` is a number, axis 0 and axis 1 are the same
    :type guard: int or list[int]
    :param trailing:
        Number of trailing cells on one side, total trailing cells are
        ``2*trailing``. When ``trailing`` is a list, ``trailing[0]`` is for
        axis 0, and ``trailing[1]`` is for axis 1. If ``trailing`` is a number,
        axis 0 and axis 1 are the same
    :type trailing: int or list[int]
    :param float pfa:
        Probability of false alarm. ``default 1e-5``
    :param str detector:
        Detector type, ``linear`` or ``squarelaw``. ``default squarelaw``
    :param float offset:
        CFAR threshold offset. If offect is None, threshold offset is
        ``2*trailing(pfa^(-1/2/trailing)-1)``. ``default None``

    :return: CFAR threshold. The dimension is the same as ``data``
    :rtype: numpy.1darray or numpy.2darray
    """

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    guard = np.array(guard)
    if guard.size == 1:
        guard = np.tile(guard, 2)
    trailing = np.array(trailing)
    if trailing.size == 1:
        trailing = np.tile(trailing, 2)

    if offset is None:
        tg_sum = trailing + guard
        t_num = (2 * tg_sum[0] + 1) * (2 * tg_sum[1] + 1)
        g_num = (2 * guard[0] + 1) * (2 * guard[1] + 1)

        if t_num == g_num:
            raise ValueError("No trailing bins!")

        if detector == "squarelaw":
            a = (t_num - g_num) * (pfa ** (-1 / (t_num - g_num)) - 1)
        elif detector == "linear":
            a = np.sqrt((t_num - g_num) * (pfa ** (-1 / (t_num - g_num)) - 1))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    cfar_win = np.ones(((guard + trailing) * 2 + 1))
    cfar_win[
        trailing[0] : (trailing[0] + guard[0] * 2 + 1),
        trailing[1] : (trailing[1] + guard[1] * 2 + 1),
    ] = 0
    cfar_win = cfar_win / np.sum(cfar_win)

    return a * convolve(data, cfar_win, mode="same")


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

    def fun(k, n, t_os, pfa):
        return (
            log_factorial(n)
            - log_factorial(n - k)
            - np.sum(np.log(np.arange(n, n - k, -1) + t_os))
            - np.log(pfa)
        )

    max_iter = 10000

    t_max = 1e32
    t_min = 1

    for _ in range(0, max_iter):
        m_n = t_max - fun(k, n, t_max, pfa) * (t_min - t_max) / (
            fun(k, n, t_min, pfa) - fun(k, n, t_max, pfa)
        )
        f_m_n = fun(k, n, m_n, pfa)
        if f_m_n == 0:
            return m_n
        if np.abs(f_m_n) < 0.0001:
            return m_n

        if fun(k, n, t_max, pfa) * f_m_n < 0:
            # t_max = t_max
            t_min = m_n
        elif fun(k, n, t_min, pfa) * f_m_n < 0:
            t_max = m_n
            # t_min = t_min
        else:
            # print("Secant method fails.")
            break

    return None


def cfar_os_1d(
    data, guard, trailing, k, pfa=1e-5, axis=0, detector="squarelaw", offset=None
):
    """
    1-D Ordered Statistic CFAR (OS-CFAR)

    For edge cells, use rollovered cells to fill the missing cells.

    :param data:
        Amplitude/Power data. Amplitude data for ``linear`` detector,
        Power data for ``squarelaw`` detector
    :type data: numpy.1darray or numpy.2darray
    :param int guard:
        Number of guard cells on one side, total guard cells are ``2*guard``
    :param int trailing:
        Number of trailing cells on one side, total trailing cells are
        ``2*trailing``
    :param int k:
        Rank in the order. ``k`` is usuall chosen to satisfy ``N/2 < k < N``.
        Typically, ``k`` is on the order of ``0.75N``
    :param float pfa:
        Probability of false alarm. ``default 1e-5``
    :param int axis:
        The axis to calculat CFAR. ``default 0``
    :param str detector:
        Detector type, ``linear`` or ``squarelaw``. ``default squarelaw``
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

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    data_shape = np.shape(data)
    cfar = np.zeros_like(data)
    leading = trailing
    # trailing = n-leading

    if offset is None:
        if detector == "squarelaw":
            a = os_cfar_threshold(k, trailing * 2, pfa)
        elif detector == "linear":
            a = np.sqrt(os_cfar_threshold(k, trailing * 2, pfa))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    if k < trailing or k > trailing * 2:
        warn(
            "``k`` is usuall chosen to satisfy ``N/2 < k < N "
            "(N = " + str(trailing * 2) + ")``. "
            "Typically, ``k`` is on the order of ``0.75N``"
        )

    if axis == 0:
        for idx in range(0, data_shape[0]):
            win_idx = np.mod(
                np.concatenate(
                    [
                        np.arange(idx - leading - guard, idx - guard, 1),
                        np.arange(idx + 1 + guard, idx + 1 + trailing + guard, 1),
                    ]
                ),
                data_shape[0],
            )
            if data.ndim == 1:
                samples = np.sort(data[win_idx.astype(int)])
                cfar[idx] = a * samples[k]
            elif data.ndim == 2:
                samples = np.sort(data[win_idx.astype(int), :], axis=0)
                cfar[idx, :] = a * samples[k, :]

    elif axis == 1:
        for idx in range(0, data_shape[1]):
            win_idx = np.mod(
                np.concatenate(
                    [
                        np.arange(idx - leading - guard, idx - guard, 1),
                        np.arange(idx + 1 + guard, idx + 1 + trailing + guard, 1),
                    ]
                ),
                data_shape[1],
            )
            samples = np.sort(data[:, win_idx.astype(int)], axis=1)

            cfar[:, idx] = a * samples[:, k]

    return cfar


def cfar_os_2d(data, guard, trailing, k, pfa=1e-5, detector="squarelaw", offset=None):
    """
    2-D Ordered Statistic CFAR (OS-CFAR)

    For edge cells, use rollovered cells to fill the missing cells.

    :param data:
        Amplitude/Power data. Amplitude data for ``linear`` detector,
        Power data for ``squarelaw`` detector
    :type data: numpy.1darray or numpy.2darray
    :param guard:
        Number of guard cells on one side, total guard cells are ``2*guard``.
        When ``guard`` is a list, ``guard[0]`` is for axis 0, and ``guard[1]``
        is for axis 1. If ``guard`` is a number, axis 0 and axis 1 are the same
    :type guard: int or list[int]
    :param trailing:
        Number of trailing cells on one side, total trailing cells are
        ``2*trailing``. When ``trailing`` is a list, ``trailing[0]`` is for
        axis 0, and ``trailing[1]`` is for axis 1. If ``trailing`` is a number,
        axis 0 and axis 1 are the same
    :type trailing: int or list[int]
    :param int k:
        Rank in the order. ``k`` is usuall chosen to satisfy ``N/2 < k < N``.
        Typically, ``k`` is on the order of ``0.75N``
    :param float pfa:
        Probability of false alarm. ``default 1e-5``
    :param str detector:
        Detector type, ``linear`` or ``squarelaw``. ``default squarelaw``
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

    if np.iscomplexobj(data):
        raise ValueError("Input data should not be complex.")

    data_shape = np.shape(data)
    cfar = np.zeros_like(data)

    guard = np.array(guard)
    if guard.size == 1:
        guard = np.tile(guard, 2)
    trailing = np.array(trailing)
    if trailing.size == 1:
        trailing = np.tile(trailing, 2)

    tg_sum = trailing + guard
    if offset is None:
        t_num = (2 * tg_sum[0] + 1) * (2 * tg_sum[1] + 1)
        g_num = (2 * guard[0] + 1) * (2 * guard[1] + 1)

        if t_num == g_num:
            raise ValueError("No trailing bins!")

        if detector == "squarelaw":
            a = os_cfar_threshold(k, t_num - g_num, pfa)
        elif detector == "linear":
            a = np.sqrt(os_cfar_threshold(k, t_num - g_num, pfa))
        else:
            raise ValueError("`detector` can only be `linear` or `squarelaw`.")
    else:
        a = offset

    if k < (t_num - g_num) / 2 or k > t_num - g_num:
        warn(
            "``k`` is usuall chosen to satisfy ``N/2 < k < N "
            "(N = " + str(t_num - g_num) + ")``. "
            "Typically, ``k`` is on the order of ``0.75N``"
        )

    cfar_win = np.ones((tg_sum * 2 + 1), dtype=bool)
    cfar_win[
        trailing[0] : (trailing[0] + guard[0] * 2 + 1),
        trailing[1] : (trailing[1] + guard[1] * 2 + 1),
    ] = False

    for idx_0 in range(0, data_shape[0]):
        for idx_1 in range(0, data_shape[1]):
            win_idx_0 = np.mod(
                np.arange(idx_0 - tg_sum[0], idx_0 + 1 + tg_sum[0], 1), data_shape[0]
            )
            win_idx_1 = np.mod(
                np.arange(idx_1 - tg_sum[1], idx_1 + 1 + tg_sum[1], 1), data_shape[1]
            )

            x, y = np.meshgrid(win_idx_0, win_idx_1, indexing="ij")
            sample_cube = data[x, y]
            samples = np.sort(sample_cube[cfar_win].flatten())

            cfar[idx_0, idx_1] = a * samples[k]

    return cfar


def doa_music(covmat, nsig, spacing=0.5, scanangles=range(-90, 91)):
    """
    Estimate arrival directions of signals using MUSIC for a uniform linear
    array (ULA)

    :param numpy.2darray covmat:
        Sensor covariance matrix, specified as a complex-valued, positive-
        definite M-by-M matrix. The quantity M is the number of elements
        in the ULA array
    :param int nsig:
        Number of arriving signals, specified as a positive integer. The
        number of signals must be smaller than the number of elements in
        the ULA array
    :param float spacing:
        Distance (wavelength) between array elements. ``default 0.5``
    :param numpy.1darray scanangles:
        Broadside search angles, specified as a real-valued vector in degrees.
        Angles must lie in the range [-90°,90°] and must be in increasing
        order. ``default [-90°,90°] ``

    :return: doa angles in degrees, doa index, pseudo spectrum (dB)
    :rtype: list, list, numpy.1darray
    """
    n_array = np.shape(covmat)[0]
    array = np.linspace(0, (n_array - 1) * spacing, n_array)
    scanangles = np.array(scanangles)

    # `eigh` guarantees the eigen values are sorted
    _, eig_vects = linalg.eigh(covmat)
    noise_subspace = eig_vects[:, :-nsig]

    array_grid, angle_grid = np.meshgrid(array, np.radians(scanangles), indexing="ij")
    steering_vect = np.exp(1j * 2 * np.pi * array_grid * np.sin(angle_grid)) / np.sqrt(
        n_array
    )

    pseudo_spectrum = 1 / linalg.norm((noise_subspace.T.conj() @ steering_vect), axis=0)

    ps_db = 10 * np.log10(pseudo_spectrum / pseudo_spectrum.min())
    doa_idx, _ = find_peaks(ps_db)
    doa_idx = doa_idx[np.argsort(ps_db[doa_idx])[-nsig:]]

    return scanangles[doa_idx], doa_idx, ps_db


def doa_root_music(covmat, nsig, spacing=0.5):
    """
    Estimate arrival directions of signals using root-MUSIC for a uniform
    linear array (ULA)

    :param numpy.2darray covmat:
        Sensor covariance matrix, specified as a complex-valued, positive-
        definite M-by-M matrix. The quantity M is the number of elements
        in the ULA array
    :param int nsig:
        Number of arriving signals, specified as a positive integer. The
        number of signals must be smaller than the number of elements in
        the ULA array
    :param float spacing:
        Distance (wavelength) between array elements. ``default 0.5``

    :return: doa angles in degrees
    :rtype: list
    """

    n_covmat = np.shape(covmat)[0]

    _, eig_vects = linalg.eigh(covmat)
    noise_subspace = eig_vects[:, :-nsig]

    # Compute the coefficients for the polynomial.
    noise_mat = noise_subspace @ noise_subspace.T.conj()
    coeff = np.zeros((n_covmat - 1,), dtype=np.complex_)
    for i in range(1, n_covmat):
        coeff[i - 1] = np.trace(noise_mat, i)
    coeff = np.hstack((coeff[::-1], np.trace(noise_mat), coeff.conj()))

    roots = np.roots(coeff)

    # Find k points inside the unit circle that are also closest to the unit
    # circle.
    mask = np.abs(roots) <= 1
    # On the unit circle. Need to find the closest point and remove it.
    for _, i in enumerate(np.where(np.abs(roots) == 1)[0]):
        mask_idx = np.argsort(np.abs(roots - roots[i]))[1]
        mask[mask_idx] = False

    roots = roots[mask]
    sorted_indices = np.argsort(1.0 - np.abs(roots))
    sin_vals = np.angle(roots[sorted_indices[:nsig]]) / (2 * np.pi * spacing)

    return np.degrees(np.arcsin(sin_vals))


def doa_esprit(covmat, nsig, spacing=0.5):
    """
    Estimate arrival directions of signals using ESPRIT for a uniform linear
    array (ULA)

    :param numpy.2darray covmat:
        Sensor covariance matrix, specified as a complex-valued, positive-
        definite M-by-M matrix. The quantity M is the number of elements
        in the ULA array
    :param int nsig:
        Number of arriving signals, specified as a positive integer. The
        number of signals must be smaller than the number of elements in
        the ULA array
    :param float spacing:
        Distance (wavelength) between array elements. ``default 0.5``

    :return: doa angles in degrees
    :rtype: list
    """

    _, eig_vects = linalg.eigh(covmat)
    signal_subspace = eig_vects[:, -nsig:]

    # the original array is divided into two subarrays
    # [0,1,...,N-2] and [1,2,...,N-1]
    phi = linalg.pinv(signal_subspace[0:-1]) @ signal_subspace[1:]
    eigs = linalg.eigvals(phi)
    return np.degrees(np.arcsin(np.angle(eigs) / np.pi / (spacing / 0.5)))


def doa_iaa(beam_vect, steering_vect, num_it=15, p_init=None):
    """
    IAA-APES follows Source Localization and Sensing: A Nonparametric Iterative Adaptive
    Approach Based on Weighted Least Square and its notation

    IAA-APES: iterative adaptive approach for amplitude and phase estimation
        y(n) = A*s(n) + e(n)   (n = 1,..,N snapshots)

    :param numpy.2darray beam_vect:
        num_array X num_snap with num_array being the number of array elements and num_snap
        being the number of pulses/snap shots. When num_snap>1,
        beam_vect = [y(1),...,y(num_snap))] with y(n) - num_array X 1
    :param numpy.2darray steering_vect:
        num_array X num_grid is the steering vectors matrix from array manifold.
        num_grid is the number of sources or the number of scanning points/grids
    :param int num_it:
        number of iterations. According to the paper, IAA-APES does not
        provide significant improvements in performance after about
        15 iterations. ``default 15``
    :param numpy.1darray p_init:
        Initial estimation. ``default None``

    :return: power (in dB) at each angle on the scanning grid
    :rtype: numpy.1darray
    """

    # Initialization
    num_grid = np.shape(steering_vect)[1]

    if p_init is None:
        spectrum_k = np.zeros(num_grid, dtype=complex)
        for ik in range(0, num_grid):
            a_vect = steering_vect[:, ik]
            a_vect = np.conj(a_vect[np.newaxis, :])
            spectrum_k[ik] = (
                1
                / ((a_vect @ a_vect.conj().T) ** 2)
                * np.mean(np.abs(a_vect @ beam_vect) ** 2)
            ).item()
    else:
        spectrum_k = p_init

    # iteration
    for _ in range(0, num_it - 1):
        p_diag = np.diag(spectrum_k.flatten())
        r_mat = steering_vect @ p_diag @ steering_vect.conj().T
        r_mat_inv = np.linalg.inv(r_mat)
        for ik in range(0, num_grid):
            a_vect = steering_vect[:, ik]
            a_vect = np.conj(a_vect[np.newaxis, :])
            spec = (
                a_vect @ r_mat_inv @ beam_vect / (a_vect @ r_mat_inv @ a_vect.conj().T)
            )
            spectrum_k[ik] = np.mean(np.abs(spec) ** 2)
    return 10 * np.log10(np.real(spectrum_k))


def doa_bartlett(covmat, spacing=0.5, scanangles=range(-90, 91)):
    """
    Bartlett beamforming for a uniform linear array (ULA)

    :param numpy.2darray covmat:
        Sensor covariance matrix, specified as a complex-valued, positive-
        definite M-by-M matrix. The quantity M is the number of elements
        in the ULA array
    :param float spacing:
        Distance (wavelength) between array elements. ``default 0.5``
    :param numpy.1darray scanangles:
        Broadside search angles, specified as a real-valued vector in degrees.
        Angles must lie in the range [-90°,90°] and must be in increasing
        order. ``default [-90°,90°] ``

    :return: spectrum in dB
    :rtype: numpy.1darray
    """

    n_array = np.shape(covmat)[0]
    array = np.linspace(0, (n_array - 1) * spacing, n_array)
    scanangles = np.array(scanangles)

    array_grid, angle_grid = np.meshgrid(array, np.radians(scanangles), indexing="ij")
    steering_vect = np.exp(1j * 2 * np.pi * array_grid * np.sin(angle_grid)) / np.sqrt(
        n_array
    )

    ps = np.sum(steering_vect.conj() * (covmat @ steering_vect), axis=0).real

    return 10 * np.log10(ps)


def doa_capon(covmat, spacing=0.5, scanangles=range(-90, 91)):
    """
    Capon (MVDR) beamforming for a uniform linear array (ULA)

    :param numpy.2darray covmat:
        Sensor covariance matrix, specified as a complex-valued, positive-
        definite M-by-M matrix. The quantity M is the number of elements
        in the ULA array
    :param float spacing:
        Distance (wavelength) between array elements. ``default 0.5``
    :param numpy.1darray scanangles:
        Broadside search angles, specified as a real-valued vector in degrees.
        Angles must lie in the range [-90°,90°] and must be in increasing
        order. ``default [-90°,90°] ``

    :return: spectrum in dB
    :rtype: numpy.1darray
    """

    n_array = np.shape(covmat)[0]
    array = np.linspace(0, (n_array - 1) * spacing, n_array)
    scanangles = np.array(scanangles)

    array_grid, angle_grid = np.meshgrid(array, np.radians(scanangles), indexing="ij")
    steering_vect = np.exp(1j * 2 * np.pi * array_grid * np.sin(angle_grid)) / np.sqrt(
        n_array
    )

    covmat = covmat + np.eye(n_array) * 0.000000001
    inv_covmat = linalg.pinv(covmat)

    ps = np.zeros(scanangles.shape)
    for idx, _ in enumerate(scanangles):
        s_vect = steering_vect[:, idx]

        weight = inv_covmat @ s_vect / (s_vect.T.conj() @ inv_covmat @ s_vect)
        ps[idx] = np.abs(weight.T.conj() @ covmat @ weight)

    return 10 * np.log10(ps)


if __name__ == "__main__":
    # data = np.arange(1, 10000)
    # cfar = cfar_ca_1d(data,
    #                   1,
    #                   100,
    #                   pfa=1e-3,
    #                   axis=0,
    #                   detector='squarelaw',
    #                   offset=None)

    # print(cfar[2000])
    # print(cfar[3000])

    scale = os_cfar_threshold(18, 32, 1e-6)
    print(scale)
