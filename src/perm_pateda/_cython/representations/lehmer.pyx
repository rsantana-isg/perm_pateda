# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

ctypedef cnp.int64_t INT


cdef class LehmerRepresentation:
    
    cdef public bint left

    def __init__(self, bint left=False):
        self.left = left

    def encode(self, cnp.ndarray[INT, ndim=2] perm):
        cdef int batch_size = perm.shape[0]
        cdef int n = perm.shape[1]
        cdef cnp.ndarray[INT, ndim=2] result = perm.copy()
        cdef int i, b, count, j

        if self.left:
            for i in range(n - 1, 0, -1):
                for b in range(batch_size):
                    count = 0
                    for j in range(i):
                        if result[b, i] > result[b, j]:
                            count += 1
                    result[b, i] = count
            result[:, 0] = 0
        else:
            for i in range(n - 1):
                for b in range(batch_size):
                    count = 0
                    for j in range(i + 1, n):
                        if result[b, i] > result[b, j]:
                            count += 1
                    result[b, i] = count
            result[:, -1] = 0

        return result

    def decode(self, cnp.ndarray[INT, ndim=2] code):
        cdef int batch_size = code.shape[0]
        cdef int n = code.shape[1]
        cdef cnp.ndarray[INT, ndim=2] perm = code.copy()
        cdef int i, j, b

        for i in range(1, n):
            if self.left:
                for b in range(batch_size):
                    for j in range(i):
                        if perm[b, i] <= perm[b, j]:
                            perm[b, j] += 1
            else:
                j = n - i - 1
                for b in range(batch_size):
                    for k in range(j + 1, n):
                        if perm[b, j] <= perm[b, k]:
                            perm[b, k] += 1

        return perm