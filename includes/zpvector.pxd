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

"""
zpvector

vector library
"""
cdef extern from "zpvector.hpp" namespace "zpv" nogil:
    # 3D vector
    cdef cppclass Vec3[T]:
        Vec3()
        Vec3(const T &...)
        Vec3(T *)

        inline Vec3 &operator=(const Vec3 &)
        inline T &operator[](const unsigned int &)
        