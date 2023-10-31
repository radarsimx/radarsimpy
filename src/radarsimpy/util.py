"""
This script contains classes that define all the parameters for
a radar system

This script requires that 'numpy' be installed within the Python
environment you are running this script in.

This file can be imported as a module and contains the following
class:

* Transmitter - A class defines parameters of a radar transmitter
* Receiver - A class defines parameters of a radar receiver
* Radar - A class defines basic parameters of a radar system

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


import numpy as np


def cal_phase_noise(  # pylint: disable=too-many-arguments, too-many-locals
    signal, fs, freq, power, seed=None, validation=False
):
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
    spectrum in num_f_points equally spaced points
    (on the interval [0 fs/2] including bounds ).

    After that we calculate required frequency shape of the phase
    noise by spec_noise(m) = sqrt(P(m)*delta_f(m)) and after that complement it
    by the symmetrical negative part of the spectrum.

    After that we generate AWGN of power 1 in the freq domain and
    multiply it sample-by-sample to the calculated shape

    Finally we perform  2*num_f_points-2 points IFFT to such generated noise

    ::

        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █
        █ 0 dBc/Hz                                                        █
        █ \\                                                    /         █
        █  \\                                                  /          █
        █   \\                                                /           █
        █    \\P dBc/Hz                                      /            █
        █    .\\                                            /             █
        █    . \\                                          /              █
        █    .  \\                                        /               █
        █    .   \\______________________________________/ <- This level  █
        █    .              is defined by the power at the maximal freq   █
        █  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__  (N) █
        █  0   delta_f                    fs/2                       fs   █
        █  DC                                                             █
        █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █ █

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

    cut_idx = np.where(freq < fs / 2)
    freq = freq[cut_idx]
    power = power[cut_idx]

    # Add 0 dBc/Hz @ DC
    if not np.any(np.isin(freq, 0)):
        freq = np.concatenate(([0], freq))
        power = np.concatenate(([0], power))

    # Calculate input length
    [row, num_samples] = np.shape(signal)
    # Define num_f_points number of points (frequency resolution) in the
    # positive spectrum (num_f_points equally spaced points on the interval
    # [0 fs/2] including bounds), then the number of points in the
    # negative spectrum will be num_f_points-2 ( interval (fs/2, fs) not
    # including bounds )
    #
    # The total number of points in the frequency domain will be
    # 2*num_f_points-2, and if we want to get the same length as the input
    # signal, then
    #   2*num_f_points-2 = num_samples
    #   num_f_points-1 = num_samples/2
    #   num_f_points = num_samples/2 + 1
    #
    # So, if num_samples is even then num_f_points = num_samples/2 + 1,
    # and if num_samples is odd we will take
    # num_f_points = (num_samples+1)/2 + 1
    #
    if np.remainder(num_samples, 2):
        num_f_points = int((num_samples + 1) / 2 + 1)
    else:
        num_f_points = int(num_samples / 2 + 1)

    # Equally spaced partitioning of the half spectrum
    f_grid = np.linspace(0, fs / 2, int(num_f_points))  # Freq. Grid
    delta_f = np.concatenate((np.diff(f_grid), [f_grid[-1] - f_grid[-2]]))  # Delta F

    realmin = np.finfo(np.float64).tiny
    # realmin = 1e-30

    # Perform interpolation of power in log-scale
    intrvl_num = len(freq)
    log_p = np.zeros(int(num_f_points))
    # for intrvl_index = 1 : intrvl_num,
    for intrvl_index in range(0, intrvl_num):
        left_bound = freq[intrvl_index]
        t1 = power[intrvl_index]
        if intrvl_index == intrvl_num - 1:
            right_bound = fs / 2
            t2 = power[-1]
            inside = np.where(
                np.logical_and(f_grid >= left_bound, f_grid <= right_bound)
            )
        else:
            right_bound = freq[intrvl_index + 1]
            t2 = power[intrvl_index + 1]
            inside = np.where(
                np.logical_and(f_grid >= left_bound, f_grid < right_bound)
            )

        log_p[inside] = t1 + (
            np.log10(f_grid[inside] + realmin) - np.log10(left_bound + realmin)
        ) / (np.log10(right_bound + 2 * realmin) - np.log10(left_bound + realmin)) * (
            t2 - t1
        )

    # Interpolated P ( half spectrum [0 fs/2] ) [ dBc/Hz ]
    p_interp = 10 ** (np.real(log_p) / 10)

    # Now we will generate AWGN of power 1 in frequency domain and shape
    # it by the desired shape as follows:
    #
    #    At the frequency offset f_grid(m) from DC we want to get power Ptag(m)
    #    such that p_interp(m) = Ptag/delta_f(m), that is we have to choose
    #    spec_noise(m) = sqrt( p_interp(m)*delta_f(m) );
    #
    # Due to the normalization factors of FFT and IFFT defined as follows:
    #     For length K input vector x, the DFT is a length K vector spec_noise,
    #     with elements
    #                K
    #      spec_noise(k) =   sum  x(n)*exp(-j*2*pi*(k-1)*(n-1)/K), 1 <= k <= K.
    #               n=1
    #     The inverse DFT (computed by IFFT) is given by
    #                      K
    #      x(n) = (1/K) sum  spec_noise(k)*exp( j*2*pi*(k-1)*(n-1)/K), 1 <= n <= K.
    #                     k=1
    #
    # we have to compensate normalization factor (1/K) multiplying spec_noise(k)
    # by K. In our case K = 2*num_f_points-2.

    # Generate AWGN of power 1
    if validation:
        awgn_p1 = np.sqrt(0.5) * (
            np.ones((row, num_f_points)) + 1j * np.ones((row, num_f_points))
        )
    else:
        awgn_p1 = np.sqrt(0.5) * (
            rng.standard_normal((row, num_f_points))
            + 1j * rng.standard_normal((row, num_f_points))
        )

    # Shape the noise on the positive spectrum [0, fs/2] including bounds
    # ( num_f_points points )
    # spec_noise = (2*num_f_points-2) * np.sqrt(delta_f * p_interp) * awgn_p1
    spec_noise = num_f_points * np.sqrt(delta_f * p_interp) * awgn_p1
    ## NOTE: this normalization should be num_f_points vs. (2*num_f_points-2)
    # since on line 222 he creates the two-sided spectrum by adding the negative frequency spectrum.

    # spec_noise = np.transpose(spec_noise)
    # Complete symmetrical negative spectrum  (fs/2, fs) not including
    # bounds (num_f_points-2 points)
    tmp_spec_noise = np.zeros((row, int(num_f_points * 2 - 2)), dtype=complex)
    tmp_spec_noise[:, 0:num_f_points] = spec_noise
    tmp_spec_noise[:, num_f_points : (2 * num_f_points - 2)] = np.fliplr(
        np.conjugate(spec_noise[:, 1:-1])
    )

    spec_noise = tmp_spec_noise

    # Remove DC
    spec_noise[:, 0] = 0

    # Perform IFFT
    x_t = np.fft.ifft(spec_noise, axis=1)

    # Calculate phase noise
    phase_noise = np.exp(-1j * np.real(x_t[:, 0:num_samples]))

    # Add phase noise
    return signal * phase_noise
