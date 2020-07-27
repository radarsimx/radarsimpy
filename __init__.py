"""
    A Python module for radar simulation

    ----------
    RadarSimPy - A Radar Simulator Built with Python
    Copyright (C) 2018 - 2020  Zhengyu Peng
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

import numpy

# radar
from .radar import Radar
from .radar import Transmitter
from .radar import Receiver

from .simulator import run_simulator as simulator
from .rt.simulatorcpp import run_simulator as simulatorcpp
# from .processing import cal_range_profile
# from .processing import cal_range_doppler
# from .processing import get_polar_image

# # roc
# from .tools import roc_pd, roc_snr, threshold

__version__ = '2.4.1'
