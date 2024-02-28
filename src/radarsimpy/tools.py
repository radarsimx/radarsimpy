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
    """
    Calculates the probability of detection (Pd) for Swerling 0 target model.

    :param npulses: Number of pulses.
    :type npulses: int
    :param snr: Signal-to-noise ratio.
    :type snr: float
    :param thred: Detection threshold.
    :type thred: float
    :return: Probability of detection (Pd).
    :rtype: float

    :Notes:
        - For npulses <= 50, uses the Marcum Q function and modified Bessel functions.
        - For npulses > 50, employs an approximation based on statistical parameters.

    :References:
        - Swerling, P. (1953). Probability of Detection for Fluctuating Targets.
          IRE Transactions on Information Theory, 6(3), 269-308.
    """
    if npulses <= 50:

        if np.isscalar(snr):
            sum_array = np.arange(2, npulses + 1)

            warnings.filterwarnings("ignore", category=RuntimeWarning)
            var_1 = np.exp(-(thred + npulses * snr)) * np.sum(
                (thred / (npulses * snr)) ** ((sum_array - 1) / 2)
                * iv(sum_array - 1, 2 * np.sqrt(npulses * snr * thred))
            )
            warnings.filterwarnings("default", category=RuntimeWarning)

            if np.isnan(var_1):
                var_1 = 0

        else:
            snr_len = np.size(snr)
            sum_array = np.arange(2, npulses + 1)

            sum_array = np.repeat(sum_array[np.newaxis, :], snr_len, axis=0)

            snr_mat = np.repeat(snr[:, np.newaxis], np.shape(sum_array)[1], axis=1)

            warnings.filterwarnings("ignore", category=RuntimeWarning)
            var_1 = np.exp(-(thred + npulses * snr)) * np.sum(
                (thred / (npulses * snr_mat)) ** ((sum_array - 1) / 2)
                * iv(sum_array - 1, 2 * np.sqrt(npulses * snr_mat * thred)),
                axis=1,
            )
            warnings.filterwarnings("default", category=RuntimeWarning)

            var_1[np.isnan(var_1)] = 0

        return marcumq(np.sqrt(2 * npulses * snr), np.sqrt(2 * thred)) + var_1

    temp_1 = 2 * snr + 1
    omegabar = np.sqrt(npulses * temp_1)
    c3 = -(snr + 1 / 3) / (np.sqrt(npulses) * temp_1**1.5)
    c4 = (snr + 0.25) / (npulses * temp_1**2.0)
    c6 = c3 * c3 / 2
    v_var = (thred - npulses * (1 + snr)) / omegabar
    v_sqr = v_var**2
    val1 = np.exp(-v_sqr / 2) / np.sqrt(2 * np.pi)
    val2 = (
        c3 * (v_sqr - 1)
        + c4 * v_var * (3 - v_sqr)
        - c6 * v_var * (v_var**4 - 10 * v_sqr + 15)
    )
    q = 0.5 * erfc(v_var / np.sqrt(2))
    return q - val1 * val2


def pd_swerling1(npulses, snr, thred):
    """
    Calculates the probability of detection (Pd) for Swerling 1 target model.

    :param npulses: Number of pulses.
    :type npulses: int
    :param snr: Signal-to-noise ratio.
    :type snr: float
    :param thred: Detection threshold.
    :type thred: float
    :return: Probability of detection (Pd).
    :rtype: float

    :Notes:
        - Swerling 1 assumes a target made up of many independent scatterers of roughly equal areas.
        - The RCS varies according to a chi-squared probability density function with two degrees
            of freedom (m = 1).
        - The radar cross section is constant from pulse-to-pulse but varies independently from
            scan to scan.

    :References:
        - Swerling, P. (1953). Probability of Detection for Fluctuating Targets.
          IRE Transactions on Information Theory, 6(3), 269-308.
    """
    if npulses == 1:
        return np.exp(-thred / (1 + snr))

    temp_sw1 = 1 + 1 / (npulses * snr)
    igf1 = gammainc(npulses - 1, thred)
    igf2 = gammainc(npulses - 1, thred / temp_sw1)
    return (
        1
        - igf1
        + (temp_sw1 ** (npulses - 1)) * igf2 * np.exp(-thred / (1 + npulses * snr))
    )


def pd_swerling2(npulses, snr, thred):
    """
    Calculates the probability of detection (Pd) for Swerling 2 target model.

    :param npulses: Number of pulses.
    :type npulses: int
    :param snr: Signal-to-noise ratio.
    :type snr: float
    :param thred: Detection threshold.
    :type thred: float
    :return: Probability of detection (Pd).
    :rtype: float

    :Notes:
        - Swerling 2 assumes a target made up of many independent scatterers of roughly equal areas.
        - The radar cross section (RCS) varies from pulse to pulse.
        - Statistics follow a chi-squared probability density function with two degrees of freedom.

    :References:
        - Swerling, P. (1953). Probability of Detection for Fluctuating Targets.
          IRE Transactions on Information Theory, 6(3), 269-308.
    """
    return 1 - gammainc(npulses, (thred / (1 + snr)))


def pd_swerling3(npulses, snr, thred):
    """
    Calculates the probability of detection (Pd) for Swerling 3 target model.

    :param npulses: Number of pulses.
    :type npulses: int
    :param snr: Signal-to-noise ratio.
    :type snr: float
    :param thred: Detection threshold.
    :type thred: float
    :return: Probability of detection (Pd).
    :rtype: float

    :Notes:
        - Swerling 3 assumes a target made up of one dominant isotropic reflector superimposed
            by several small reflectors.
        - The radar cross section (RCS) varies from pulse to pulse but remains constant within
            a single scan.
        - The statistical properties follow a density of probability based on the Chi-squared
            distribution with four degrees of freedom (m = 2).

    :References:
        - Swerling, P. (1953). Probability of Detection for Fluctuating Targets.
          IRE Transactions on Information Theory, 6(3), 269-308.
    """
    temp_1 = thred / (1 + 0.5 * npulses * snr)
    ko = (
        np.exp(-temp_1)
        * (1 + 2 / (npulses * snr)) ** (npulses - 2)
        * (1 + temp_1 - 2 * (npulses - 2) / (npulses * snr))
    )
    if npulses <= 2:
        return ko

    var_1 = np.exp(
        (npulses - 1) * np.log(thred) - thred - log_factorial(npulses - 2.0)
    ) / (1 + 0.5 * npulses * snr)

    pd = (
        var_1
        + 1
        - gammainc(npulses - 1, thred)
        + ko * gammainc(npulses - 1, thred / (1 + 2 / (npulses * snr)))
    )

    return pd


def pd_swerling4(npulses, snr, thred):
    """
    Calculates the probability of detection (Pd) for Swerling 4 target model.

    :param npulses: Number of pulses.
    :type npulses: int
    :param snr: Signal-to-noise ratio.
    :type snr: float
    :param thred: Detection threshold.
    :type thred: float
    :return: Probability of detection (Pd).
    :rtype: float

    :Notes:
        - Swerling 4 assumes a target made up of one dominant isotropic reflector
            superimposed by several small reflectors.
        - The radar cross section (RCS) varies from pulse to pulse rather than from scan to scan.
        - The statistical properties follow a density of probability based on the Chi-squared
            distribution with four degrees of freedom (m = 2).

    :References:
        - Swerling, P. (1953). Probability of Detection for Fluctuating Targets.
          IRE Transactions on Information Theory, 6(3), 269-308.
    """
    beta = 1 + snr / 2
    if npulses >= 50:
        omegabar = np.sqrt(npulses * (2 * beta**2 - 1))
        c3 = (2 * beta**3 - 1) / (3 * (2 * beta**2 - 1) * omegabar)
        c4 = (2 * beta**4 - 1) / (4 * npulses * (2 * beta**2 - 1) ** 2)
        c6 = c3**2 / 2
        v_var = (thred - npulses * (1 + snr)) / omegabar
        v_sqr = v_var**2
        val1 = np.exp(-v_sqr / 2) / np.sqrt(2 * np.pi)
        val2 = (
            c3 * (v_sqr - 1)
            + c4 * v_var * (3 - v_sqr)
            - c6 * v_var * (v_var**4 - 10 * v_sqr + 15)
        )
        return 0.5 * erfc(v_var / np.sqrt(2)) - val1 * val2

    gamma0 = gammainc(npulses, thred / beta)
    a1 = (thred / beta) ** npulses / (
        np.exp(log_factorial(npulses)) * np.exp(thred / beta)
    )
    sum_var = gamma0
    for idx_1 in range(1, npulses + 1, 1):

        if idx_1 == 1:
            ai = a1
        else:
            ai = (thred / beta) * a1 / (npulses + idx_1 - 1)
        a1 = ai
        gammai = gamma0 - ai
        gamma0 = gammai
        # a1 = ai

        temp_sw4 = np.sum(np.log(npulses + 1 - np.arange(1, idx_1 + 1)))

        try:
            term = (snr / 2) ** idx_1 * gammai * np.exp(temp_sw4 - log_factorial(idx_1))
        except OverflowError:
            term = 0

        sum_var = sum_var + term
    return 1 - sum_var / beta**npulses


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
