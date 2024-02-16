"""
Useful tools for radar system analysis

This script requires that 'numpy' and 'scipy' be installed within the
Python environment you are running this script in.

This file can be imported as a module and contains the following
functions:

* roc_pd - Calculate probability of detection (Pd) in receiver operating
           characteristic (ROC)
* roc_snr - Calculate the minimal SNR for certain probability of
            detection (Pd) and probability of false alarm (Pfa) in
            receiver operating characteristic (ROC)

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

import warnings
import numpy as np
from scipy.special import (  # pylint: disable=no-name-in-module
    erfc,
    erfcinv,
    gammainc,
    gammaincinv,
    iv,
)
from scipy.stats import distributions


def marcumq(a, x, m=1):
    """
    Calculates the generalized Marcum Q function.

    The Marcum Q function is defined as:
        Q_m(a, x) = 1 - F_ncx2(m * 2, a^2, x^2)

    :param float a: Non-centrality parameter.
    :param float x: Threshold value.
    :param int m: Order of the function, positive integer (default is 1).

    :return: Generalized Marcum Q function value.
    :rtype: float

    :references:
        - `Wikipedia - Marcum Q-function <https://en.wikipedia.org/wiki/Marcum_Q-function>`_
        - `SciPy Documentation - scipy.stats.ncx2
            <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html>`_
    """
    return 1 - distributions.ncx2.cdf(df=m * 2, nc=a**2, x=x**2)


def log_factorial(n):
    """
    Compute the factorial of 'n' using logarithms to avoid overflow

    :param int n:
        Integer number

    :return:
        log(n!)
    :rtype: float
    """

    if np.isscalar(n):
        return np.sum(np.log(np.arange(1, n + 1)))

    val = np.zeros_like(n, dtype=float)
    for idx, n_item in enumerate(n):
        val[idx] = np.sum(np.log(np.arange(1, n_item + 1)))

    return val

    # n = n + 9.0
    # n2 = n**2
    # return (
    #     (n - 1) * np.log(n)
    #     - n
    #     + np.log(np.sqrt(2 * np.pi * n))
    #     + ((1 - (1 / 30 + (1 / 105) / n2) / n2) / 12) / n
    #     - np.log(
    #         (n - 1)
    #         * (n - 2)
    #         * (n - 3)
    #         * (n - 4)
    #         * (n - 5)
    #         * (n - 6)
    #         * (n - 7)
    #         * (n - 8)
    #     )
    # )


def threshold(pfa, npulses):
    """
    Threshold ratio

    :param float pfa:
        Probability of false alarm
    :param int npulses:
        Number of pulses for integration

    :return:
        Threshod ratio
    :rtype: float

    :references:
        - Mahafza, Bassem R. Radar systems analysis and design using MATLAB.
            Chapman and Hall/CRC, 2005.
    """

    return gammaincinv(npulses, 1 - pfa)


def pd_swerling0(npulses, snr, thred):
    sum_array = np.arange(2, npulses + 1)

    return marcumq(np.sqrt(2 * npulses * snr), np.sqrt(2 * thred)) + np.exp(
        -(thred + npulses * snr)
    ) * np.sum(
        (thred / (npulses * snr)) ** ((sum_array - 1) / 2)
        * iv(sum_array - 1, 2 * np.sqrt(npulses * snr * thred))
    )


def pd_swerling1(npulses, snr, thred):
    if npulses == 1:
        return np.exp(-thred / (1 + snr))
    else:
        temp_sw1 = 1 + 1 / (npulses * snr)
        igf1 = gammainc(npulses - 1, thred)
        igf2 = gammainc(npulses - 1, thred / temp_sw1)
        return (
            1
            - igf1
            + (temp_sw1 ** (npulses - 1)) * igf2 * np.exp(-thred / (1 + npulses * snr))
        )


def pd_swerling2(npulses, snr, thred):
    return 1 - gammainc(npulses, (thred / (1 + snr)))


def pd_swerling3(npulses, snr, thred):
    temp_1 = thred / (1 + 0.5 * npulses * snr)
    ko = (
        np.exp(-temp_1)
        * (1 + 2 / (npulses * snr)) ** (npulses - 2)
        * (1 + temp_1 - 2 * (npulses - 2) / (npulses * snr))
    )
    if npulses <= 2:
        return ko

    # c_var = 1 / (1 + 0.5 * npulses * snr)
    # sum_array = np.arange(0, npulses - 1)

    # var_1 = (
    #     thred ** (npulses - 1)
    #     * np.exp(-thred)
    #     * c_var
    #     / np.exp(log_factorial(npulses - 2))
    # )

    # var_2 = np.sum(np.exp(-thred) * thred**sum_array / np.exp(log_factorial(sum_array)))

    # var_3_1 = np.exp(-c_var * thred) / ((1 - c_var) ** (npulses - 2))
    # var_3_2 = 1 - (npulses - 2) * c_var / (1 - c_var) + c_var * thred
    # var_3_3 = 1 - np.sum(
    #     np.exp(-(1 - c_var) * thred)
    #     * (thred**sum_array)
    #     * ((1 - c_var) ** sum_array)
    #     / np.exp(log_factorial(sum_array))
    # )

    # pd = var_1 + var_2 + var_3_1 * var_3_2 * var_3_3

    var_1 = (
        thred ** (npulses - 1)
        * np.exp(-thred)
        / ((1 + 0.5 * npulses * snr) * np.exp(log_factorial(npulses - 2.0)))
    )

    pd = (
        var_1
        + 1
        - gammainc(npulses - 1, thred)
        + ko * gammainc(npulses - 1, thred / (1 + 2 / (npulses * snr)))
    )

    return pd

    #     warnings.filterwarnings("ignore", category=RuntimeWarning)
    #     temp4 = (
    #         thred ** (npulses - 1)
    #         * np.exp(-thred)
    #         / (temp_1 * np.exp(log_factorial(npulses - 2.0)))
    #     )
    #     warnings.filterwarnings("default", category=RuntimeWarning)

    #     if np.isscalar(temp4):
    #         if np.isnan(temp4) or np.isinf(temp4):
    #             temp4 = 0
    #     else:
    #         temp4[np.isnan(temp4)] = 0
    #         temp4[np.isinf(temp4)] = 0

    #     pd[it_pfa.index, :] = (
    #         temp4
    #         + 1
    #         - gammainc(npulses - 1, thred)
    #         + ko * gammainc(npulses - 1, thred / (1 + 2 / (npulses * snr)))
    #     )
    # if np.size(pd[it_pfa.index, :]) == 1:
    #     if pd[it_pfa.index, :] > 1:
    #         pd[it_pfa.index, :] = 1
    # else:
    #     neg_idx = np.where(pd[it_pfa.index, :] > 1)
    #     pd[it_pfa.index, :][neg_idx[0]] = 1


def pd_swerling4(npulses, snr, thred):
    c_var = 1 / (1 + 0.5 * snr)
    if thred >= npulses * (2 - c_var):
        pd = 0
        for k_idx in range(0, npulses + 1):
            l_array = np.arange(0, 2 * npulses - k_idx)
            pd += (
                np.exp(log_factorial(npulses))
                / np.exp(log_factorial(k_idx))
                / np.exp(log_factorial(npulses - k_idx))
                * (((1 - c_var) / c_var) ** (npulses - k_idx))
                * np.sum(
                    np.exp(-c_var * thred)
                    * ((c_var * thred) ** l_array)
                    / np.exp(log_factorial(l_array))
                )
            )
        return pd * (c_var**npulses)
    else:
        pd = 0
        factor_overflow_val = 160
        for k_idx in range(0, npulses + 1):
            if (2 * npulses - k_idx) <= factor_overflow_val:
                l_array = np.arange((2 * npulses - k_idx), factor_overflow_val + 1)
                pd += (
                    np.exp(log_factorial(npulses))
                    / np.exp(log_factorial(k_idx))
                    / np.exp(log_factorial(npulses - k_idx))
                    * (((1 - c_var) / c_var) ** (npulses - k_idx))
                    * np.sum(
                        np.exp(-c_var * thred)
                        * ((c_var * thred) ** l_array)
                        / np.exp(log_factorial(l_array))
                    )
                )
        return 1 - pd * (c_var**npulses)


def roc_pd(pfa, snr, npulses=1, stype="Coherent"):
    """
    Calculate probability of detection (Pd) in receiver operating
    characteristic (ROC)

    :param pfa:
        Probability of false alarm (Pfa)
    :type pfa: float or numpy.1darray
    :param snr:
        Signal to noise ratio in decibel (dB)
    :type snr: float or numpy.1darray
    :param int npulses:
        Number of pulses for integration (default is 1)
    :param str stype:
        Signal type (default is ``Coherent``)

        - ``Coherent``: Non-fluctuating coherent
        - ``Real``: Non-fluctuating real signal
        - ``Swerling 0``: Non-coherent Swerling 0, Non-fluctuating non-coherent
        - ``Swerling 1``: Non-coherent Swerling 1
        - ``Swerling 2``: Non-coherent Swerling 2
        - ``Swerling 3``: Non-coherent Swerling 3
        - ``Swerling 4``: Non-coherent Swerling 4
        - ``Swerling 5``: Non-coherent Swerling 5, Non-fluctuating non-coherent

    :return: probability of detection (Pd).
        if both ``pfa`` and ``snr`` are floats, ``pd`` is a float
        if ``pfa`` or ``snr`` is a 1-D array, ``pd`` is a 1-D array
        if both ``pfa`` and ``snr`` are 1-D arrays, ``pd`` is a 2-D array
    :rtype: float or 1-D array or 2-D array

    *Reference*

    Mahafza, Bassem R. Radar systems analysis and design using MATLAB.
    Chapman and Hall/CRC, 2005.
    """
    snr_db = snr
    snr = 10.0 ** (snr_db / 10.0)

    size_pfa = np.size(pfa)
    size_snr = np.size(snr)

    pd = np.zeros((size_pfa, size_snr))

    it_pfa = np.nditer(pfa, flags=["f_index"])
    while not it_pfa.finished:
        thred = threshold(it_pfa[0], npulses)

        if stype == "Swerling 1":
            pd[it_pfa.index, :] = pd_swerling1(npulses, snr, thred)

        elif stype == "Swerling 2":
            pd[it_pfa.index, :] = pd_swerling2(npulses, snr, thred)

        elif stype == "Swerling 3":
            pd[it_pfa.index, :] = pd_swerling3(npulses, snr, thred)

        elif stype == "Swerling 4":
            pd[it_pfa.index, :] = pd_swerling4(npulses, snr, thred)

        elif stype in ("Swerling 5", "Swerling 0"):
            pd[it_pfa.index, :] = pd_swerling0(npulses, snr, thred)

        elif stype == "Coherent":
            snr = snr * npulses
            pd[it_pfa.index, :] = erfc(erfcinv(2 * it_pfa[0]) - np.sqrt(snr)) / 2

        elif stype == "Real":
            snr = snr * npulses / 2
            pd[it_pfa.index, :] = erfc(erfcinv(2 * it_pfa[0]) - np.sqrt(snr)) / 2

        else:
            return None

        it_pfa.iternext()

    if size_pfa == 1 and size_snr == 1:
        return pd[0, 0]

    if size_pfa == 1 and size_snr > 1:
        return pd[0, :]

    if size_pfa > 1 and size_snr == 1:
        return pd[:, 0]

    return pd


def roc_snr(pfa, pd, npulses=1, stype="Coherent"):
    """
    Calculate the minimal SNR for certain probability of
    detection (Pd) and probability of false alarm (Pfa) in
    receiver operating characteristic (ROC) with Secant method

    :param pfa:
        Probability of false alarm (Pfa)
    :type pfa: float or numpy.1darray
    :param pd:
         Probability of detection (Pd)
    :type pd: float or numpy.1darray
    :param int npulses:
        Number of pulses for integration (default is 1)
    :param str stype:
        Signal type (default is ``Coherent``)

        - ``Coherent`` : Non-fluctuating coherent
        - ``Real`` : Non-fluctuating real signal
        - ``Swerling 0`` : Non-fluctuating non-coherent
        - ``Swerling 1`` : Non-coherent Swerling 1
        - ``Swerling 2`` : Non-coherent Swerling 2
        - ``Swerling 3`` : Non-coherent Swerling 3
        - ``Swerling 4`` : Non-coherent Swerling 4
        - ``Swerling 5`` : Same as ``Swerling 0``

    :return: Minimal signal to noise ratio in decibel (dB)
        if both ``pfa`` and ``pd`` are floats, ``SNR`` is a float
        if ``pfa`` or ``pd`` is a 1-D array, ``SNR`` is a 1-D array
        if both ``pfa`` and ``pd`` are 1-D arrays, ``SNR`` is a 2-D array
    :rtype: float or 1-D array or 2-D array

    *Reference*

    Secant method:

        The x intercept of the secant line on the the Nth interval

        .. math:: m_n = a_n - f(a_n)*(b_n - a_n)/(f(b_n) - f(a_n))

        The initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0
        for some intercept m_n then the function returns this solution.
        If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
        iterations, the secant method fails and return None.
    """

    def fun(pfa, pd, snr):
        return roc_pd(pfa, snr, npulses, stype) - pd

    max_iter = 1000
    snra = 40
    snrb = -20

    if stype == "Coherent" or stype == "Real":
        snrb = -40

    size_pd = np.size(pd)
    size_pfa = np.size(pfa)
    snr = np.zeros((size_pfa, size_pd))

    it_pfa = np.nditer(pfa, flags=["f_index"])
    while not it_pfa.finished:
        it_pd = np.nditer(pd, flags=["f_index"])
        while not it_pd.finished:
            if fun(it_pfa[0], it_pd[0], snra) * fun(it_pfa[0], it_pd[0], snrb) >= 0:
                # print("Initializing Secant method fails.")
                return None
            a_n = snra
            b_n = snrb
            for _ in range(1, max_iter + 1):
                m_n = a_n - fun(it_pfa[0], it_pd[0], a_n) * (b_n - a_n) / (
                    fun(it_pfa[0], it_pd[0], b_n) - fun(it_pfa[0], it_pd[0], a_n)
                )
                f_m_n = fun(it_pfa[0], it_pd[0], m_n)
                if f_m_n == 0:
                    # print("Found exact solution.")
                    snr[it_pfa.index, it_pd.index] = m_n
                    break
                if np.abs(f_m_n) < 0.00001:
                    # print("Reach threshold.")
                    snr[it_pfa.index, it_pd.index] = m_n
                    break
                if fun(it_pfa[0], it_pd[0], a_n) * f_m_n < 0:
                    # a_n = a_n
                    b_n = m_n
                elif fun(it_pfa[0], it_pd[0], b_n) * f_m_n < 0:
                    a_n = m_n
                    # b_n = b_n
                else:
                    # print("Secant method fails.")
                    # return None
                    snr[it_pfa.index, it_pd.index] = float("nan")
                    break
            # return a_n-fun(a_n)*(b_n-a_n)/(fun(b_n)-fun(a_n))
            snr[it_pfa.index, it_pd.index] = a_n - fun(it_pfa[0], it_pd[0], a_n) * (
                b_n - a_n
            ) / (fun(it_pfa[0], it_pd[0], b_n) - fun(it_pfa[0], it_pd[0], a_n))
            it_pd.iternext()
        it_pfa.iternext()

    if size_pfa == 1 and size_pd == 1:
        return snr[0, 0]

    if size_pfa == 1 and size_pd > 1:
        return snr[0, :]

    if size_pfa > 1 and size_pd == 1:
        return snr[:, 0]

    return snr
