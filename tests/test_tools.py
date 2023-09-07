"""
    A Python module for radar simulation

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - PRESENT  Zhengyu Peng
    E-mail: zpeng.me@gmail.com
    Website: https://zpeng.me

    `                      `
    -:.                  -#:
    -//:.              -###:
    -////:.          -#####:
    -/:.://:.      -###++##:
    ..   `://:-  -###+. :##:
           `:/+####+.   :##:
    .::::::::/+###.     :##:
    .////-----+##:    `:###:
     `-//:.   :##:  `:###/.
       `-//:. :##:`:###/.
         `-//:+######/.
           `-/+####/.
             `+##+.
              :##:
              :##:
              :##:
              :##:
              :##:
               .+:

"""

import numpy as np
import numpy.testing as npt

from radarsimpy.tools import roc_pd, roc_snr, threshold


def test_roc_pd():
    npt.assert_almost_equal(roc_pd(1e-8, 13, 1, "Swerling 5"), 0.6290, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-8, 11, 1, "Swerling 5"), 0.1681, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-8, -3.2, 256, "Swerling 5"), 0.8424, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-8, -4.8, 256, "Swerling 5"), 0.2266, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-9, -10, 256, "Coherent"), 0.8765, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-9, -12, 256, "Coherent"), 0.3767, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-6, 12.4, 1, "Coherent"), 0.8733, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-6, 8.8, 1, "Coherent"), 0.1953, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, 16, 1, "Swerling 1"), 0.7980, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 1"), 0.2036, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 3.6, 256, "Swerling 1"), 0.8959, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -7.6, 256, "Swerling 1"), 0.2560, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, -4.4, 256, "Swerling 2"), 0.9120, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -7.2, 256, "Swerling 2"), 0.2125, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 16, 1, "Swerling 2"), 0.7980, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 2"), 0.2036, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, 15.2, 1, "Swerling 3"), 0.8846, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 3"), 0.1931, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -0.4, 256, "Swerling 3"), 0.8889, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -8.4, 256, "Swerling 3"), 0.1772, decimal=4)

    npt.assert_almost_equal(roc_pd(1e-4, 15.2, 1, "Swerling 4"), 0.8846, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, 6.8, 1, "Swerling 4"), 0.1931, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -4.8, 256, "Swerling 4"), 0.8413, decimal=4)
    npt.assert_almost_equal(roc_pd(1e-4, -7.2, 256, "Swerling 4"), 0.2155, decimal=4)


def test_threshold():
    npt.assert_almost_equal(threshold(1e-4, 1), 9.21, decimal=2)
    npt.assert_almost_equal(threshold(1e-4, 10), 26.19, decimal=2)
    npt.assert_almost_equal(threshold(1e-4, 20), 41.03, decimal=2)
    npt.assert_almost_equal(threshold(1e-4, 40), 67.89, decimal=2)


def test_roc_snr():
    npt.assert_almost_equal(roc_snr(1e-8, 0.6290, 1, "Swerling 5"), 13, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-8, 0.1681, 1, "Swerling 5"), 11, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-8, 0.8424, 256, "Swerling 5"), -3.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-8, 0.2266, 256, "Swerling 5"), -4.8, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-9, 0.8765, 256, "Coherent"), -10, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-9, 0.3767, 256, "Coherent"), -12, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-6, 0.8733, 1, "Coherent"), 12.4, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-6, 0.1953, 1, "Coherent"), 8.8, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.7980, 1, "Swerling 1"), 16, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2036, 1, "Swerling 1"), 6.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.8959, 256, "Swerling 1"), 3.6, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2560, 256, "Swerling 1"), -7.6, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.9120, 256, "Swerling 2"), -4.4, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2125, 256, "Swerling 2"), -7.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.7980, 1, "Swerling 2"), 16, decimal=0)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2036, 1, "Swerling 2"), 6.8, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.8846, 1, "Swerling 3"), 15.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.1931, 1, "Swerling 3"), 6.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.8889, 256, "Swerling 3"), -0.4, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.1772, 256, "Swerling 3"), -8.4, decimal=1)

    npt.assert_almost_equal(roc_snr(1e-4, 0.8846, 1, "Swerling 4"), 15.2, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.1931, 1, "Swerling 4"), 6.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.8413, 256, "Swerling 4"), -4.8, decimal=1)
    npt.assert_almost_equal(roc_snr(1e-4, 0.2155, 256, "Swerling 4"), -7.2, decimal=1)
