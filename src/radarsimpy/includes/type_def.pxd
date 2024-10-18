# distutils: language = c++
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


ctypedef int int_t
ctypedef unsigned int uint_t

ctypedef float float_t


"""
C++ vector
"""
cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T]:
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        vector() except +
        vector(vector&) except +
        vector(size_type) except +
        vector(size_type, T&) except +
        #vector[InputIt](InputIt, InputIt)
        T& operator[](size_type)
        #vector& operator=(vector&)
        bint operator==(vector&, vector&)
        bint operator!=(vector&, vector&)
        bint operator<(vector&, vector&)
        bint operator>(vector&, vector&)
        bint operator<=(vector&, vector&)
        bint operator>=(vector&, vector&)
        void assign(size_type, const T&)
        void assign[InputIt](InputIt, InputIt) except +
        T& at(size_type) except +
        T& back()
        # iterator begin()
        # const_iterator const_begin "begin"()
        # const_iterator cbegin()
        size_type capacity()
        void clear()
        bint empty()
        # iterator end()
        # const_iterator const_end "end"()
        # const_iterator cend()
        # iterator erase(iterator)
        # iterator erase(iterator, iterator)
        T& front()
        # iterator insert(iterator, const T&) except +
        # iterator insert(iterator, size_type, const T&) except +
        # iterator insert[InputIt](iterator, InputIt, InputIt) except +
        size_type max_size()
        void pop_back()
        void push_back(T&) except +
        # reverse_iterator rbegin()
        # const_reverse_iterator const_rbegin "rbegin"()
        # const_reverse_iterator crbegin()
        # reverse_iterator rend()
        # const_reverse_iterator const_rend "rend"()
        # const_reverse_iterator crend()
        void reserve(size_type) except +
        void resize(size_type) except +
        void resize(size_type, T&) except +
        size_type size()
        void swap(vector&)

        # C++11 methods
        T* data()
        const T* const_data "data"()
        void shrink_to_fit() except +
        # iterator emplace(const_iterator, ...) except +
        T& emplace_back(...) except +
