#!python
# cython: language_level=3

# This script contains classes that define all the parameters for
# a radar system

# This script requires that 'numpy' be installed within the Python
# environment you are running this script in.

# This file can be imported as a module and contains the following
# class:

# * Transmitter - A class defines parameters of a radar transmitter
# * Receiver - A class defines parameters of a radar receiver
# * Radar - A class defines basic parameters of a radar system

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


def cal_phase_noise(signal, fs, freq, power, seed=None, validation=False):
    """
    Oscillator Phase Noise Model

    :param numpy.2darray signal:
        Input signal
    :param float fs:
        Sampling frequency
    :param numpy.1darray freq:
        Frequency of the phase noise
    :param numpy.1darray power:
        Power of the phase noise
    :param int seed:
        Seed for noise generator
    :param boolean validation:
        Validate phase noise

    :return:
        Signal with phase noise
    :rtype: numpy.2darray

    **NOTES**

    - The presented model is a simple VCO phase noise model based
    on the following consideration:
        If the output of an oscillator is given as
        V(t) = V0 * cos( w0*t + phi(t) ), then phi(t) is defined
        as the phase noise.  In cases of small noise sources (a valid
        assumption in any usable system), a narrowband modulatio
        approximation can be used to express the oscillator output as:

        V(t) = V0 * cos( w0*t + phi(t) )
            = V0 * [cos(w0*t)*cos(phi(t)) - signal(w0*t)*signal(phi(t)) ]
            ~ V0 * [cos(w0*t) - signal(w0*t)*phi(t)]

        This shows that phase noise will be mixed with the carrier
        to produce sidebands around the carrier.

    - In other words, exp(j*x) ~ (1+j*x) for small x

    - Phase noise = 0 dBc/Hz at freq. offset of 0 Hz

    - The lowest phase noise level is defined by the input SSB phase
    noise power at the maximal freq. offset from DC.
    (IT DOES NOT BECOME EQUAL TO ZERO )

    The generation process is as follows:

    First of all we interpolate (in log-scale) SSB phase noise power
    spectrum in M equally spaced points
    (on the interval [0 fs/2] including bounds ).

    After that we calculate required frequency shape of the phase
    noise by X(m) = sqrt(P(m)*dF(m)) and after that complement it
    by the symmetrical negative part of the spectrum.

    After that we generate AWGN of power 1 in the freq domain and
    multiply it sample-by-sample to the calculated shape

    Finally we perform  2*M-2 points IFFT to such generated noise

    ::

        |  0 dBc/Hz
        | \\                                                    /
        |  \\                                                  /
        |   \\                                                /
        |    \\P dBc/Hz                                      /
        |    .\\                                            /
        |    . \\                                          /
        |    .  \\                                        /
        |    .   \\______________________________________/ <- This level
        |    .              is defined by the power at the maximal freq
        |  |__| _|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__  (N)
        |  0   dF                    fs/2                       fs
        |  DC
        |
    """

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    signal = signal.astype(complex)

    # Sort freq and power
    sort_idx = np.argsort(freq)
    freq = freq[sort_idx]
    power = power[sort_idx]

    cut_idx = np.where(freq < fs/2)
    freq = freq[cut_idx]
    power = power[cut_idx]

    # Add 0 dBc/Hz @ DC
    if not np.any(np.isin(freq, 0)):
        freq = np.concatenate(([0], freq))
        power = np.concatenate(([0], power))

    # Calculate input length
    [row, N] = np.shape(signal)
    # Define M number of points (frequency resolution) in the
    # positive spectrum (M equally spaced points on the interval
    # [0 fs/2] including bounds), then the number of points in the
    # negative spectrum will be M-2 ( interval (fs/2, fs) not
    # including bounds )
    #
    # The total number of points in the frequency domain will be
    # 2*M-2, and if we want to get the same length as the input
    # signal, then
    #   2*M-2 = N
    #   M-1 = N/2
    #   M = N/2 + 1
    #
    # So, if N is even then M = N/2 + 1, and if N is odd we will take
    # M = (N+1)/2 + 1
    #
    if np.remainder(N, 2):
        M = int((N+1)/2 + 1)
    else:
        M = int(N/2 + 1)

    # Equally spaced partitioning of the half spectrum
    F = np.linspace(0, fs/2, int(M))    # Freq. Grid
    dF = np.concatenate((np.diff(F), [F[-1]-F[-2]]))  # Delta F

    realmin = np.finfo(float).tiny

    # Perform interpolation of power in log-scale
    intrvlNum = len(freq)
    logP = np.zeros(int(M))
    # for intrvlIndex = 1 : intrvlNum,
    for intrvlIndex in range(0, intrvlNum):
        leftBound = freq[intrvlIndex]
        t1 = power[intrvlIndex]
        if intrvlIndex == intrvlNum-1:
            rightBound = fs/2
            t2 = power[-1]
            inside = np.where(np.logical_and(
                F >= leftBound, F <= rightBound))
        else:
            rightBound = freq[intrvlIndex+1]
            t2 = power[intrvlIndex+1]
            inside = np.where(np.logical_and(
                F >= leftBound, F < rightBound))

        logP[inside] = t1 + (np.log10(F[inside] + realmin) -
                             np.log10(leftBound + realmin)) / \
            (np.log10(rightBound + 2*realmin) -
                np.log10(leftBound + realmin)) * (t2-t1)

    # Interpolated P ( half spectrum [0 fs/2] ) [ dBc/Hz ]
    P = 10**(np.real(logP)/10)

    # Now we will generate AWGN of power 1 in frequency domain and shape
    # it by the desired shape as follows:
    #
    #    At the frequency offset F(m) from DC we want to get power Ptag(m)
    #    such that P(m) = Ptag/dF(m), that is we have to choose
    #    X(m) = sqrt( P(m)*dF(m) );
    #
    # Due to the normalization factors of FFT and IFFT defined as follows:
    #     For length K input vector x, the DFT is a length K vector X,
    #     with elements
    #                K
    #      X(k) =   sum  x(n)*exp(-j*2*pi*(k-1)*(n-1)/K), 1 <= k <= K.
    #               n=1
    #     The inverse DFT (computed by IFFT) is given by
    #                      K
    #      x(n) = (1/K) sum  X(k)*exp( j*2*pi*(k-1)*(n-1)/K), 1 <= n <= K.
    #                     k=1
    #
    # we have to compensate normalization factor (1/K) multiplying X(k)
    # by K. In our case K = 2*M-2.

    # Generate AWGN of power 1
    if validation:
        awgn_P1 = (np.sqrt(0.5)*(np.ones((row, M)) +
                                 1j*np.ones((row, M))))
    else:
        awgn_P1 = (np.sqrt(0.5)*(rng.standard_normal((row, M)) +
                                 1j*rng.standard_normal((row, M))))

    # Shape the noise on the positive spectrum [0, fs/2] including bounds
    # ( M points )
    X = (2*M-2) * np.sqrt(dF * P) * awgn_P1

    # X = np.transpose(X)
    # Complete symmetrical negative spectrum  (fs/2, fs) not including
    # bounds (M-2 points)
    tmp_X = np.zeros((row, int(M*2-2)), dtype=complex)
    tmp_X[:, 0:M] = X
    tmp_X[:, M:(2*M-2)] = np.fliplr(np.conjugate(X[:, 1:-1]))

    X = tmp_X

    # Remove DC
    X[:, 0] = 0

    # Perform IFFT
    x_t = np.fft.ifft(X, axis=1)

    # Calculate phase noise
    phase_noise = np.exp(1j * np.real(x_t[:, 0:N]))

    # Add phase noise
    return signal * phase_noise
