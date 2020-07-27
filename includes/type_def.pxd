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

ctypedef int int_t
ctypedef float float_t
ctypedef unsigned int uint_t
ctypedef float complex complex_t

IF UNAME_SYSNAME == "Windows":
    ctypedef unsigned long long uint64_t
ELSE:
    ctypedef unsigned long uint64_t

from libcpp cimport bool

"""
C++ vector
"""
cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T]:
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        vector()
        vector(vector&)
        vector(size_type)
        vector(size_type, T&)
        T& operator[](size_type)
        bint operator==(vector&, vector&)
        bint operator!=(vector&, vector&)
        bint operator<(vector&, vector&)
        bint operator>(vector&, vector&)
        bint operator<=(vector&, vector&)
        bint operator>=(vector&, vector&)
        void assign(size_type, const T&)
        T& at(size_type)
        T& back()
        size_type capacity()
        void clear()
        bint empty()
        T& front()
        size_type max_size()
        void pop_back()
        void push_back(T&)
        void reserve(size_type)
        void resize(size_type)
        void resize(size_type, T&)
        size_type size()
        void swap(vector&)

        T* data()
        const T* const_data "data"()
        void shrink_to_fit()
