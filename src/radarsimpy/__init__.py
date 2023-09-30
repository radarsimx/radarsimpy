# distutils: language = c++
# cython: language_level=3

#  ____           _            ____  _          __  __
# |  _ \ __ _  __| | __ _ _ __/ ___|(_)_ __ ___ \ \/ /
# | |_) / _` |/ _` |/ _` | '__\___ \| | '_ ` _ \ \  /
# |  _ < (_| | (_| | (_| | |   ___) | | | | | | |/  \
# |_| \_\__,_|\__,_|\__,_|_|  |____/|_|_| |_| |_/_/\_\

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
