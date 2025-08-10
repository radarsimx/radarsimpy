# distutils: language = c++
"""
RadarSimPy Type Definitions and Standard Library Extensions

This module provides essential type definitions and extended C++ standard library 
interfaces optimized for radar simulation applications.

Key Components:
- **Primitive type aliases**: Consistent naming for integer and floating-point types
- **Extended vector interface**: Enhanced std::vector with full iterator support
- **Memory-efficient containers**: Optimized for high-performance numerical computations
- **Cross-platform compatibility**: Consistent behavior across different architectures

Type Safety Features:
- Clear distinction between signed/unsigned integers
- Explicit floating-point precision control
- Template-based generic programming support
- Exception-safe memory management

Performance Optimizations:
- Custom allocator support removed for better const reference handling
- Direct memory access patterns for numerical arrays
- Iterator-based algorithms for efficient data processing

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


#------------------------------------------------------------------------------
# Primitive Type Definitions
#------------------------------------------------------------------------------

# Integer types for indexing and counting
ctypedef int int_t              # Signed integer for general indexing
ctypedef unsigned int uint_t    # Unsigned integer for sizes and counts

# Floating-point types for numerical precision control  
ctypedef float float_t          # Single precision (32-bit) for memory efficiency


#------------------------------------------------------------------------------
# Enhanced C++ Standard Library - std::vector Interface
#------------------------------------------------------------------------------

"""
Extended std::vector Interface for RadarSimPy

This is an enhanced version of Cython's libcpp/vector.pxd with the following
key modifications for RadarSimPy compatibility:

1. **Custom Allocator Removal**: ALLOCATOR=* parameter removed to enable 
   proper 'const &' reference handling in C++ function signatures

2. **Complete Iterator Support**: Full iterator hierarchy including:
   - Forward iterators (iterator, const_iterator)
   - Reverse iterators (reverse_iterator, const_reverse_iterator)
   - All comparison and arithmetic operators

3. **Memory Safety**: Exception-safe operations with proper exception 
   propagation from C++ to Python

4. **Performance Optimized**: Direct memory access methods for numerical
   array operations common in radar signal processing

Usage Notes:
- Use for storing large arrays of numerical data (samples, coordinates, etc.)
- Prefer push_back() for dynamic array construction
- Use data() for direct memory access when interfacing with C++ algorithms
- Reserve capacity with reserve() for known array sizes to prevent reallocations
"""

cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T]:
        # Type definitions for size and iterator arithmetic
        ctypedef size_t size_type
        ctypedef ptrdiff_t difference_type

        # Iterator Classes - For traversing container elements
        cppclass const_iterator
        cppclass iterator:
            # Mutable forward iterator for vector elements
            # Supports random access and arithmetic operations
            iterator() except +
            iterator(iterator&) except +
            T& operator*()                              # Dereference to get element
            iterator operator++()                       # Pre-increment
            iterator operator--()                       # Pre-decrement  
            iterator operator++(int)                    # Post-increment
            iterator operator--(int)                    # Post-decrement
            iterator operator+(size_type)               # Forward jump
            iterator operator-(size_type)               # Backward jump
            difference_type operator-(iterator)         # Distance between iterators
            difference_type operator-(const_iterator)
            # Comparison operators for iterator positioning
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)
            
        cppclass const_iterator:
            # Immutable forward iterator for vector elements
            # Provides read-only access with full random access capabilities
            const_iterator() except +
            const_iterator(iterator&) except +
            const_iterator(const_iterator&) except +
            operator=(iterator&) except +
            const T& operator*()                        # Dereference to get const element
            const_iterator operator++()                 # Pre-increment
            const_iterator operator--()                 # Pre-decrement
            const_iterator operator++(int)              # Post-increment
            const_iterator operator--(int)              # Post-decrement
            const_iterator operator+(size_type)         # Forward jump
            const_iterator operator-(size_type)         # Backward jump
            difference_type operator-(iterator)         # Distance calculations
            difference_type operator-(const_iterator)
            # Comparison operators
            bint operator==(iterator)
            bint operator==(const_iterator)
            bint operator!=(iterator)
            bint operator!=(const_iterator)
            bint operator<(iterator)
            bint operator<(const_iterator)
            bint operator>(iterator)
            bint operator>(const_iterator)
            bint operator<=(iterator)
            bint operator<=(const_iterator)
            bint operator>=(iterator)
            bint operator>=(const_iterator)

        cppclass const_reverse_iterator
        cppclass reverse_iterator:
            reverse_iterator() except +
            reverse_iterator(reverse_iterator&) except +
            T& operator*()
            reverse_iterator operator++()
            reverse_iterator operator--()
            reverse_iterator operator++(int)
            reverse_iterator operator--(int)
            reverse_iterator operator+(size_type)
            reverse_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(reverse_iterator)
            bint operator==(const_reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator!=(const_reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator<(const_reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator>(const_reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator<=(const_reverse_iterator)
            bint operator>=(reverse_iterator)
            bint operator>=(const_reverse_iterator)
        cppclass const_reverse_iterator:
            const_reverse_iterator() except +
            const_reverse_iterator(reverse_iterator&) except +
            operator=(reverse_iterator&) except +
            const T& operator*()
            const_reverse_iterator operator++()
            const_reverse_iterator operator--()
            const_reverse_iterator operator++(int)
            const_reverse_iterator operator--(int)
            const_reverse_iterator operator+(size_type)
            const_reverse_iterator operator-(size_type)
            difference_type operator-(iterator)
            difference_type operator-(const_iterator)
            bint operator==(reverse_iterator)
            bint operator==(const_reverse_iterator)
            bint operator!=(reverse_iterator)
            bint operator!=(const_reverse_iterator)
            bint operator<(reverse_iterator)
            bint operator<(const_reverse_iterator)
            bint operator>(reverse_iterator)
            bint operator>(const_reverse_iterator)
            bint operator<=(reverse_iterator)
            bint operator<=(const_reverse_iterator)
            bint operator>=(reverse_iterator)
            bint operator>=(const_reverse_iterator)

        # ================================================================
        # Vector Constructors and Core Operations
        # ================================================================
        
        # Constructors
        vector() except +                                # Default empty vector
        vector(vector&) except +                         # Copy constructor
        vector(size_type) except +                       # Size constructor (default values)
        vector(size_type, T&) except +                   # Size + value constructor
        #vector[InputIt](InputIt, InputIt)               # Iterator range constructor
        
        # Element Access Methods
        T& operator[](size_type)                         # Direct element access (unchecked)
        T& at(size_type) except +                        # Bounds-checked element access
        T& front()                                       # First element reference
        T& back()                                        # Last element reference
        
        # Memory and Direct Access  
        T* data()                                        # Pointer to underlying array
        const T* const_data "data"()                     # Const pointer to data
        
        # Comparison Operators
        #vector& operator=(vector&)                      # Assignment operator
        bint operator==(vector&, vector&)                # Equality comparison
        bint operator!=(vector&, vector&)                # Inequality comparison  
        bint operator<(vector&, vector&)                 # Lexicographic less than
        bint operator>(vector&, vector&)                 # Lexicographic greater than
        bint operator<=(vector&, vector&)                # Less than or equal
        bint operator>=(vector&, vector&)                # Greater than or equal
        # ================================================================
        # Container Modification Methods
        # ================================================================
        
        # Assignment and Population
        void assign(size_type, const T&)                 # Fill with repeated value
        void assign[InputIt](InputIt, InputIt) except +  # Assign from iterator range
        
        # Iterator Access
        iterator begin()                                 # Iterator to first element
        const_iterator const_begin "begin"()             # Const iterator to first
        const_iterator cbegin()                          # C++11 const begin iterator
        iterator end()                                   # Iterator past last element
        const_iterator const_end "end"()                 # Const iterator past last
        const_iterator cend()                            # C++11 const end iterator
        
        # Reverse Iterator Access (for backwards traversal)
        reverse_iterator rbegin()                        # Reverse iterator to last
        const_reverse_iterator const_rbegin "rbegin"()   # Const reverse begin
        const_reverse_iterator crbegin()                 # C++11 const reverse begin
        reverse_iterator rend()                          # Reverse iterator before first
        const_reverse_iterator const_rend "rend"()       # Const reverse end
        const_reverse_iterator crend()                   # C++11 const reverse end
        
        # Element Insertion and Removal
        iterator erase(iterator)                         # Erase single element
        iterator erase(iterator, iterator)               # Erase range of elements
        iterator insert(iterator, const T&) except +     # Insert single element
        iterator insert(iterator, size_type, const T&) except +  # Insert multiple copies
        iterator insert[InputIt](iterator, InputIt, InputIt) except +  # Insert range
        void pop_back()                                  # Remove last element
        void push_back(T&) except +                      # Add element to end
        
        # Size and Capacity Management
        bint empty()                                     # Check if container is empty
        size_type size()                                 # Number of elements
        size_type max_size()                             # Maximum possible size
        size_type capacity()                             # Current allocated capacity
        void clear()                                     # Remove all elements
        void reserve(size_type) except +                 # Reserve memory capacity
        void resize(size_type) except +                  # Resize container
        void resize(size_type, T&) except +              # Resize with fill value
        void swap(vector&)                               # Swap contents with another vector
        
        # C++11 Enhanced Methods
        void shrink_to_fit() except +                    # Reduce capacity to fit size
        iterator emplace(const_iterator, ...) except +   # Construct element in-place
        T& emplace_back(...) except +                    # Construct element at end

#------------------------------------------------------------------------------
# End of RadarSimPy Type Definitions
#------------------------------------------------------------------------------
