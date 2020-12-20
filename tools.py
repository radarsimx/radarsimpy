#!python
# cython: language_level=3

# Useful tools for radar system analysis

# This script requires that 'numpy' and 'scipy' be installed within the
# Python environment you are running this script in.

# This file can be imported as a module and contains the following
# functions:

# * threshold - Threshold ratio
# * roc_pd - Calculate probability of detection (Pd) in receiver operating
#            characteristic (ROC)
# * roc_snr - Calculate the minimal SNR for certain probability of
#             detection (Pd) and probability of false alarm (Pfa) in
#             receiver operating characteristic (ROC)

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
from scipy.special import erfc, erfcinv, gammainc


def log_factorial(n):
    """
    Compute the factorial of 'n' using logarithms to avoid overflow

    :param int n:
        Integer number

    :return:
        log(n!)
    :rtype: float
    """

    n = n+9.0
    n2 = n**2
    return (n-1)*np.log(n)-n+np.log(np.sqrt(2*np.pi*n)) + \
        ((1-(1/30+(1/105)/n2)/n2)/12)/n - \
        np.log((n-1)*(n-2)*(n-3)*(n-4)*(n-5)*(n-6)*(n-7)*(n-8))


def threshold(pfa, N):
    """
    Threshold ratio

    :param float pfa:
        Probability of false alarm
    :param int N:
        Number of pulses for integration

    :return:
        Threshod ratio
    :rtype: float

    *Reference*

    Mahafza, Bassem R. Radar systems analysis and design using MATLAB.
        Chapman and Hall/CRC, 2005.
    """

    eps = 0.00000001
    delta = 10000.
    nfa = N * np.log(2)/pfa
    sqrtpfa = np.sqrt(-np.log10(pfa))
    sqrtnp = np.sqrt(N)
    thred0 = N-sqrtnp+2.3*sqrtpfa*(sqrtpfa+sqrtnp-1.0)
    thred = thred0
    while (delta >= thred0):
        igf = gammainc(N, thred0)
        deno = np.exp((N-1) * np.log(thred0+eps) - thred0 - log_factorial(N-1))
        thred = thred0+((0.5**(N/nfa)-igf)/(deno+eps))

        delta = np.abs(thred-thred0)*10000.0
        thred0 = thred

    return thred


def roc_pd(pfa, snr, N=1, stype='Coherent'):
    """
    Calculate probability of detection (Pd) in receiver operating
    characteristic (ROC)

    :param pfa:
        Probability of false alarm (Pfa)
    :type pfa: float or numpy.1darray
    :param snr:
        Signal to noise ratio in decibel (dB)
    :type snr: float or numpy.1darray
    :param int N:
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
        if ``pfa`` is a 1-D array or ``snr`` is a 1-D array, ``pd`` is a 1-D array
        if both ``pfa`` and ``snr`` are 1-D arrays, ``pd`` is a 2-D array
    :rtype: float or 1-D array or 2-D array

    *Reference*

    Mahafza, Bassem R. Radar systems analysis and design using MATLAB.
    Chapman and Hall/CRC, 2005.
    """
    snr_db = snr
    snr = 10.0**(snr_db/10.)

    size_pfa = np.size(pfa)
    size_snr = np.size(snr)

    pd = np.zeros((size_pfa, size_snr))

    it_pfa = np.nditer(pfa, flags=['f_index'])
    while not it_pfa.finished:
        thred = threshold(it_pfa[0], N)

        if stype == 'Swerling 1':
            if (N == 1):
                pd[it_pfa.index, :] = np.exp(-thred/(1+snr))
            else:
                temp_sw1 = 1+1/(N*snr)
                igf1 = gammainc(N-1, thred)
                igf2 = gammainc(N-1, thred/temp_sw1)
                pd[it_pfa.index, :] = 1-igf1 + \
                    (temp_sw1**(N-1))*igf2*np.exp(-thred/(1+N*snr))
        elif stype == 'Swerling 2':
            if (N <= 50):
                pd[it_pfa.index, :] = 1-gammainc(N, (thred/(1+snr)))
            else:
                V = (thred-N*(snr+1))/(np.sqrt(N)*(snr+1))
                Vsqr = V**2
                val1 = np.exp(-Vsqr/2)/np.sqrt(2*np.pi)
                val2 = -1/np.sqrt(9*N)*(Vsqr-1)+0.25*V * \
                    (3-Vsqr)/N-V*(V**4-10*Vsqr+15)/(18*N)
                pd[it_pfa.index, :] = 0.5*erfc(V/np.sqrt(2))-val1*val2
        elif stype == 'Swerling 3':
            temp_1 = thred/(1+0.5*N*snr)
            ko = np.exp(-temp_1)*(1+2/(N*snr))**(N-2) * \
                (1+temp_1-2*(N-2)/(N*snr))
            if (N <= 2):
                pd[it_pfa.index, :] = ko
            else:
                if log_factorial(N-2.) <= 500:
                    temp4 = thred**(N-1)*np.exp(-thred) / \
                        (temp_1*np.exp(log_factorial(N-2.)))
                else:
                    temp4 = 0
                pd[it_pfa.index, :] = temp4+1-gammainc(N-1, thred) + \
                    ko*gammainc(N-1, thred/(1+2/(N*snr)))
            if np.size(pd[it_pfa.index, :]) == 1:
                if pd[it_pfa.index, :] > 1:
                    pd[it_pfa.index, :] = 1
            else:
                neg_idx = np.where(pd[it_pfa.index, :] > 1)
                pd[it_pfa.index, :][neg_idx[0]] = 1
        elif stype == 'Swerling 4':
            beta = 1+snr/2
            if (N >= 50):
                omegabar = np.sqrt(N*(2*beta**2-1))
                c3 = (2*beta**3-1)/(3*(2*beta**2-1)*omegabar)
                c4 = (2*beta**4-1)/(4*N*(2*beta**2-1)**2)
                c6 = c3**2/2
                V = (thred-N*(1+snr))/omegabar
                Vsqr = V**2
                val1 = np.exp(-Vsqr/2)/np.sqrt(2*np.pi)
                val2 = c3*(V**2-1)+c4*V*(3-V**2)-c6*V*(V**4-10*V**2+15)
                pd[it_pfa.index, :] = 0.5*erfc(V/np.sqrt(2))-val1*val2
            else:
                gamma0 = gammainc(N, thred/beta)
                a1 = (thred/beta)**N / \
                    (np.exp(log_factorial(N))*np.exp(thred/beta))
                sum_var = gamma0
                for i in range(1, N+1, 1):
                    temp_sw4 = 1
                    if (i == 1):
                        ai = a1
                    else:
                        ai = (thred/beta)*a1/(N+i-1)
                    a1 = ai
                    gammai = gamma0-ai
                    gamma0 = gammai
                    a1 = ai

                    for ii in range(1, i+1, 1):
                        temp_sw4 = temp_sw4*(N+1-ii)

                    term = (snr/2)**i*gammai*temp_sw4/np.exp(log_factorial(i))
                    sum_var = sum_var+term
                pd[it_pfa.index, :] = 1-sum_var/beta**N
            if np.size(pd[it_pfa.index, :]) == 1:
                if pd[it_pfa.index, :] < 0:
                    pd[it_pfa.index, :] = 0
            else:
                neg_idx = np.where(pd[it_pfa.index, :] < 0)
                pd[it_pfa.index, :][neg_idx[0]] = 0
        elif stype == 'Swerling 5' or stype == 'Swerling 0':
            temp_1 = 2*snr+1
            omegabar = np.sqrt(N*temp_1)
            c3 = -(snr+1/3)/(np.sqrt(N)*temp_1**1.5)
            c4 = (snr+0.25)/(N*temp_1**2.)
            c6 = c3*c3/2
            V = (thred-N*(1+snr))/omegabar
            Vsqr = V**2
            val1 = np.exp(-Vsqr/2)/np.sqrt(2*np.pi)
            val2 = c3*(V**2-1)+c4*V*(3-V**2) - \
                c6*V*(V**4-10*V**2+15)
            q = 0.5*erfc(V/np.sqrt(2))
            pd[it_pfa.index, :] = q-val1*val2
        elif stype == 'Coherent':
            snr = snr*N
            pd[it_pfa.index, :] = erfc(erfcinv(2*it_pfa[0])-np.sqrt(snr))/2
        elif stype == 'Real':
            snr = snr*N/2
            pd[it_pfa.index, :] = erfc(erfcinv(2*it_pfa[0])-np.sqrt(snr))/2
        else:
            return None

        it_pfa.iternext()

    if size_pfa == 1 and size_snr == 1:
        return pd[0, 0]
    elif size_pfa == 1 and size_snr > 1:
        return pd[0, :]
    elif size_pfa > 1 and size_snr == 1:
        return pd[:, 0]
    else:
        return pd


def roc_snr(pfa, pd, N=1, stype='Coherent'):
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
    :param int N:
        Number of pulses for integration (default is 1)
    :param str stype:
        Signal type (default is ``Coherent``)

        - ``Coherent`` : Non-fluctuating coherent
        - ``Real`` : Non-fluctuating real signal
        - ``Swerling 0`` : Non-coherent Swerling 0, Non-fluctuating non-coherent
        - ``Swerling 1`` : Non-coherent Swerling 1
        - ``Swerling 2`` : Non-coherent Swerling 2
        - ``Swerling 3`` : Non-coherent Swerling 3
        - ``Swerling 4`` : Non-coherent Swerling 4
        - ``Swerling 5`` : Non-coherent Swerling 5, Non-fluctuating non-coherent

    :return: Minimal signal to noise ratio in decibel (dB)
        if both ``pfa`` and ``pd`` are floats, ``SNR`` is a float
        if ``pfa`` is a 1-D array or ``pd`` is a 1-D array, ``SNR`` is a 1-D array
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
        return roc_pd(pfa, snr, N, stype)-pd

    max_iter = 1000
    snra = 40
    snrb = -30

    size_pd = np.size(pd)
    size_pfa = np.size(pfa)
    snr = np.zeros((size_pfa, size_pd))

    it_pfa = np.nditer(pfa, flags=['f_index'])
    while not it_pfa.finished:

        it_pd = np.nditer(pd, flags=['f_index'])
        while not it_pd.finished:
            if fun(it_pfa[0], it_pd[0], snra) *\
                    fun(it_pfa[0], it_pd[0], snrb) >= 0:
                # print("Initializing Secant method fails.")
                return None
            a_n = snra
            b_n = snrb
            for n in range(1, max_iter+1):
                m_n = a_n-fun(it_pfa[0], it_pd[0], a_n)*(b_n-a_n) / \
                    (fun(it_pfa[0], it_pd[0], b_n) -
                     fun(it_pfa[0], it_pd[0], a_n))
                f_m_n = fun(it_pfa[0], it_pd[0], m_n)
                if f_m_n == 0:
                    # print("Found exact solution.")
                    snr[it_pfa.index, it_pd.index] = m_n
                    break
                elif np.abs(f_m_n) < 0.00001:
                    # print("Reach threshold.")
                    snr[it_pfa.index, it_pd.index] = m_n
                    break
                elif fun(it_pfa[0], it_pd[0], a_n)*f_m_n < 0:
                    a_n = a_n
                    b_n = m_n
                elif fun(it_pfa[0], it_pd[0], b_n)*f_m_n < 0:
                    a_n = m_n
                    b_n = b_n
                else:
                    # print("Secant method fails.")
                    # return None
                    snr[it_pfa.index, it_pd.index] = float('nan')
                    break
            # return a_n-fun(a_n)*(b_n-a_n)/(fun(b_n)-fun(a_n))
            snr[it_pfa.index, it_pd.index] = a_n - \
                fun(it_pfa[0], it_pd[0], a_n)*(b_n-a_n) / \
                (fun(it_pfa[0], it_pd[0], b_n)-fun(it_pfa[0], it_pd[0], a_n))
            it_pd.iternext()
        it_pfa.iternext()

    if size_pfa == 1 and size_pd == 1:
        return snr[0, 0]
    elif size_pfa == 1 and size_pd > 1:
        return snr[0, :]
    elif size_pfa > 1 and size_pd == 1:
        return snr[:, 0]
    else:
        return snr
