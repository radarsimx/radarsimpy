# distutils: language = c++

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

"""
zpvector

vector library
"""
cdef extern from "zpvector.hpp" namespace "zpv" nogil:
    # 3D vector
    cdef cppclass Vec3[T]:
        Vec3()
        Vec3(const T & ...)
        Vec3(T * )

        inline Vec3 & operator = (const Vec3 &)
        inline T & operator[](const unsigned int &)
