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
from radarsimpy.simulator import simc
from radarsimpy.radar import cal_phase_noise
import radarsimpy.processing as proc


def test_phase_noise():
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


def test_fmcw_phase_noise_cpp():
    tx_channel = dict(location=(0, 0, 0))

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

    rx_channel = dict(location=(0, 0, 0))

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

    target_1 = dict(location=(150, 20, 0), speed=(0, 0, 0), rcs=60, phase=0)

    targets = [target_1]

    data_cpp_pn = simc(radar_pn, targets, noise=False)
    data_matrix_cpp_pn = data_cpp_pn["baseband"]
    data_cpp = simc(radar, targets, noise=False)
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
                8.79798637e00,
                8.92852677e00,
                9.10244790e00,
                9.23691427e00,
                9.39637374e00,
                9.55030817e00,
                9.75201879e00,
                9.99391771e00,
                1.02540629e01,
                1.05633725e01,
                1.09709745e01,
                1.14634761e01,
                1.20602118e01,
                1.27221889e01,
                1.34656013e01,
                1.43745850e01,
                1.53798577e01,
                1.65158326e01,
                1.76935265e01,
                1.90055719e01,
                1.94698268e01,
                2.74742955e01,
                2.27652179e01,
                2.41057326e01,
                2.51006053e01,
                2.64592428e01,
                2.80574979e01,
                3.02969827e01,
                3.35404161e01,
                4.03730894e01,
                5.10139042e01,
                3.61690121e01,
                3.05100242e01,
                2.68250805e01,
                2.38824627e01,
                2.14787115e01,
                1.94539696e01,
                1.76038801e01,
                1.58218269e01,
                1.41671421e01,
                1.25263458e01,
                1.09833725e01,
                9.36974652e00,
                7.78373012e00,
                6.14752450e00,
                4.45653221e00,
                2.69558824e00,
                8.92489685e-01,
                -9.88604711e-01,
                -2.77310247e00,
                -4.34194299e00,
                -5.34248482e00,
                -5.58878960e00,
                -5.17162741e00,
                -4.43632751e00,
                -3.59143526e00,
                -2.79504101e00,
                -2.08075069e00,
                -1.44908234e00,
                -9.12383849e-01,
                -4.49804341e-01,
                -5.21983807e-02,
                2.85803922e-01,
                5.75951386e-01,
                8.28473424e-01,
                1.04610451e00,
                1.24001324e00,
                1.41735362e00,
                1.58596369e00,
                1.74589347e00,
                1.91689014e00,
                2.07782510e00,
                2.28588639e00,
                2.49213422e00,
                2.72855815e00,
                3.00658524e00,
                3.31348973e00,
                3.67142504e00,
                4.04902702e00,
                4.50009494e00,
                4.97741334e00,
                5.50357818e00,
                6.07309727e00,
                6.73237712e00,
                7.41145104e00,
                8.18894268e00,
                9.01696097e00,
                1.00455593e01,
                1.11374080e01,
                1.25478502e01,
                1.42728844e01,
                1.66367794e01,
                2.05593639e01,
                2.74899059e01,
                3.13456951e01,
                2.00634893e01,
                1.50121223e01,
                1.21829281e01,
                1.69154068e01,
                -5.77896226e-01,
                4.88479938e-02,
                -2.41179037e-02,
                4.39818875e-02,
                -7.01442870e-01,
                1.22746498e01,
                1.25900205e01,
                1.88463218e01,
                5.52312297e01,
                2.00740761e01,
                1.58423063e01,
                1.25294146e01,
                1.08581000e01,
                9.58197592e00,
                8.57942283e00,
                7.79418262e00,
                7.06103782e00,
                6.46613664e00,
                5.85610944e00,
                5.30366351e00,
                4.74076635e00,
                4.21611510e00,
                3.66232718e00,
                3.10619495e00,
                2.53497122e00,
                1.98027816e00,
                1.38543959e00,
                8.18703781e-01,
                2.49900259e-01,
                -2.76135665e-01,
                -7.58744341e-01,
                -1.16041714e00,
                -1.40698611e00,
                -1.54860906e00,
                -1.50330449e00,
                -1.31265337e00,
                -9.75833813e-01,
                -5.23414842e-01,
                6.43666620e-03,
                5.50886150e-01,
                1.16029829e00,
                1.73313607e00,
                2.33306103e00,
                2.87948578e00,
                3.42592455e00,
                3.93953357e00,
                4.41939616e00,
                4.87874672e00,
                5.32239162e00,
                5.70678798e00,
                6.09397684e00,
                6.43869764e00,
                6.77035391e00,
                7.08163966e00,
                7.34420969e00,
                7.61606210e00,
                7.84589533e00,
                8.07298677e00,
                8.27414041e00,
                8.45476601e00,
                8.62785532e00,
            ]
        ),
        atol=1,
    )
