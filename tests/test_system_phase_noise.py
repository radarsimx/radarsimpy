"""
A Python module for radar simulation

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

from scipy import signal
import numpy as np
import numpy.testing as npt

from radarsimpy import Radar, Transmitter, Receiver
from radarsimpy.simulator import simc  # pylint: disable=no-name-in-module
from radarsimpy.radar import cal_phase_noise
import radarsimpy.processing as proc


def test_phase_noise():
    """
    This function tests phase noise calculation
    """
    sig = np.ones((1, 256))
    pn_f = np.array([1000, 10000, 100000, 1000000])
    pn_power_db_per_hz = np.array([-84, -100, -96, -109])
    fs = 4e6

    pn = cal_phase_noise(sig, fs, pn_f, pn_power_db_per_hz, validation=True)

    # f = np.linspace(0, fs, 256)
    spec = 20 * np.log10(np.abs(np.fft.fft(pn[0, :] / 256)))

    # pn_power_db = pn_power_db_per_hz+10*np.log10(fs/256)

    npt.assert_almost_equal(spec[1], -63.4, decimal=2)
    npt.assert_almost_equal(spec[6], -60.21, decimal=2)
    npt.assert_almost_equal(spec[64], -73.09, decimal=2)


def test_fmcw_phase_noise():
    """
    This function tests the phase noise simulation
    """
    tx_channel = {"location": (0, 0, 0)}

    pn_f = np.array([1000, 10000, 100000, 1000000])
    pn_power = np.array([-65, -70, -65, -90])

    tx_pn = Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=40,
        prp=100e-6,
        pulses=1,
        pn_f=pn_f,
        pn_power=pn_power,
        channels=[tx_channel],
    )

    tx = Transmitter(
        f=[24.125e9 - 50e6, 24.125e9 + 50e6],
        t=80e-6,
        tx_power=40,
        prp=100e-6,
        pulses=1,
        channels=[tx_channel],
    )

    rx_channel = {"location": (0, 0, 0)}

    rx = Receiver(
        fs=2e6,
        noise_figure=12,
        rf_gain=20,
        load_resistor=500,
        baseband_gain=30,
        channels=[rx_channel],
    )

    radar_pn = Radar(transmitter=tx_pn, receiver=rx, seed=1234, validation=True)
    radar = Radar(transmitter=tx, receiver=rx, seed=1234, validation=True)

    target_1 = {"location": (150, 20, 0), "speed": (0, 0, 0), "rcs": 60, "phase": 0}

    targets = [target_1]

    data_cpp_pn = simc(radar_pn, targets)
    data_matrix_cpp_pn = data_cpp_pn["baseband"]
    data_cpp = simc(radar, targets)
    data_matrix_cpp = data_cpp["baseband"]

    range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
    range_profile_pn = proc.range_fft(data_matrix_cpp_pn, range_window)
    range_profile = proc.range_fft(data_matrix_cpp, range_window)

    range_profile_pn = 20 * np.log10(np.abs(range_profile_pn[0, 0, :]))
    range_profile = 20 * np.log10(np.abs(range_profile[0, 0, :]))

    profile_diff = range_profile_pn - range_profile

    npt.assert_allclose(
        profile_diff,
        np.array(
            [
                8.15236683e00,
                8.62450253e00,
                9.94566903e00,
                8.41230320e00,
                1.35971821e01,
                1.62558002e01,
                1.93212483e01,
                1.31870147e01,
                1.75406669e01,
                1.38916956e01,
                8.49327223e00,
                1.86175487e01,
                1.40706202e01,
                1.76409000e01,
                2.25048424e01,
                1.45015542e01,
                2.05210282e01,
                1.86531317e01,
                1.71046366e01,
                4.10641684e00,
                3.79405756e01,
                2.75389544e01,
                3.90926646e01,
                6.27310149e00,
                2.16279769e01,
                2.52661677e01,
                3.36658700e01,
                2.88195604e01,
                3.91194415e01,
                4.59002235e01,
                5.04732280e01,
                3.94599231e01,
                3.24503013e01,
                2.96344893e01,
                3.03904606e01,
                2.83176478e01,
                3.11181814e01,
                2.53430024e01,
                2.30302068e01,
                1.72725494e01,
                1.87363003e01,
                1.71670574e01,
                1.69392597e01,
                1.04715126e01,
                1.59758951e01,
                7.97696140e00,
                1.11580555e01,
                1.44807343e01,
                1.40912099e01,
                1.41879078e01,
                1.29359464e01,
                1.20614666e01,
                1.08958080e01,
                8.03589480e00,
                1.04874421e01,
                6.87794869e00,
                8.70538605e00,
                1.02700420e01,
                -1.50547286e01,
                -1.89459088e00,
                4.42629745e-01,
                1.06804309e00,
                7.46382310e00,
                -9.28290884e00,
                7.02103212e00,
                9.03193911e00,
                4.89482735e00,
                3.28958093e00,
                5.25774634e00,
                1.72219960e00,
                -7.17070791e-01,
                4.87554162e00,
                1.26008143e00,
                4.53897983e00,
                -1.31826831e01,
                -3.97219210e-01,
                -1.88513576e01,
                -1.00752817e00,
                5.18454718e-01,
                -6.54762623e-01,
                -9.00964293e-01,
                9.64784119e-01,
                4.68541185e00,
                8.86610661e00,
                1.16768714e01,
                1.40424314e01,
                1.23302152e01,
                1.20765995e01,
                1.62397170e01,
                1.68698397e01,
                1.63122911e01,
                1.63895149e01,
                2.22444422e01,
                3.04330686e01,
                3.34724037e01,
                1.52160781e01,
                1.19215742e01,
                1.53405891e01,
                2.20231153e01,
                -1.05176631e00,
                8.30268334e-02,
                -3.95634294e-02,
                5.80602668e-02,
                -6.59517939e-01,
                1.34651845e01,
                1.53163491e01,
                2.45530362e01,
                6.18439387e01,
                2.27776171e01,
                1.74963002e01,
                1.77674211e01,
                1.50284897e01,
                1.25935400e01,
                6.74313391e00,
                6.07564392e00,
                1.00810249e01,
                4.78584396e00,
                -4.26958606e00,
                -5.58157923e00,
                -2.45911840e00,
                4.72325979e00,
                7.08286461e00,
                3.99875291e00,
                6.58567146e00,
                7.95576358e00,
                4.24917591e00,
                4.43187106e00,
                6.59002957e00,
                5.28506440e00,
                -1.07894150e00,
                -2.71490104e00,
                8.49118834e00,
                7.33305209e00,
                7.58354325e00,
                3.00461831e00,
                1.45074678e00,
                9.11779594e00,
                6.87048797e00,
                3.04951896e-01,
                8.08362341e00,
                6.77752185e00,
                6.09439334e00,
                8.00496874e00,
                9.93519129e00,
                9.16553127e00,
                9.02552562e00,
                1.17643448e01,
                4.90491193e00,
                -1.86823144e00,
                6.35799399e00,
                1.27685095e01,
                1.34158011e01,
                1.15192466e01,
                1.08706573e01,
                7.58792202e00,
                -8.31226080e00,
                6.27604552e00,
                7.86567283e00,
                1.32263561e01,
                -6.63679916e00,
            ]
        ),
        atol=1,
    )
