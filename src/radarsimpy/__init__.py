# distutils: language = c++
# cython: language_level=3

"""
A Python module for radar simulation

----------
RadarSimPy - A Radar Simulator Built with Python
Copyright (C) 2018 - PRESENT  radarsimx.com
E-mail: info@radarsimx.com
Website: https://radarsimx.com

"""

# radar
from .radar import Radar
from .radar import Transmitter
from .radar import Receiver

__version__ = "11.3.0"
