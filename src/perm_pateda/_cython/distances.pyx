# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.int64_t INT


# ===========================================================================
# Kendall
# ===========================================================================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray[INT, ndim=1] _v_vector(cnp.ndarray[INT, ndim=1] perm):
    cdef int n = perm.shape[0]
    cdef cnp.ndarray[INT, ndim=1] v = np.zeros(n, dtype=np.int64)
    cdef int i, j, count

    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                count += 1
        v[i] = count
    return v


def v_vector(cnp.ndarray[INT, ndim=1] perm):
    """Public wrapper of the v-vector (Lehmer code)."""
    return _v_vector(perm)


def kendall_distance(cnp.ndarray[INT, ndim=1] perm1,
                     cnp.ndarray[INT, ndim=1] perm2):
    cdef cnp.ndarray[INT, ndim=1] inv = np.argsort(perm2)
    cdef cnp.ndarray[INT, ndim=1] comp = perm1[inv]
    cdef cnp.ndarray[INT, ndim=1] v = _v_vector(comp)

    cdef int n = v.shape[0]
    cdef int total = 0
    cdef int i
    for i in range(n):
        total += v[i]
    return total


# ===========================================================================
# Hamming
# ===========================================================================

def hamming_distance(cnp.ndarray[INT, ndim=1] perm1,
                     cnp.ndarray[INT, ndim=1] perm2):
    cdef int n = perm1.shape[0]
    cdef int count = 0
    cdef int i
    for i in range(n):
        if perm1[i] != perm2[i]:
            count += 1
    return count


# ===========================================================================
# Cayley
# ===========================================================================

def cayley_distance(cnp.ndarray[INT, ndim=1] perm1,
                    cnp.ndarray[INT, ndim=1] perm2):
    cdef cnp.ndarray[INT, ndim=1] inv = np.argsort(perm2)
    cdef cnp.ndarray[INT, ndim=1] comp = perm1[inv]
    return _n_minus_cycles(comp)


cdef int _n_minus_cycles(cnp.ndarray[INT, ndim=1] perm):
    cdef int n = perm.shape[0]
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] visited = np.zeros(n, dtype=np.uint8)
    cdef int cycles = 0
    cdef int i, j

    for i in range(n):
        if not visited[i]:
            cycles += 1
            j = i
            while not visited[j]:
                visited[j] = 1
                j = perm[j]
    return n - cycles


# ===========================================================================
# Ulam (LIS)
# ===========================================================================

def ulam_distance(cnp.ndarray[INT, ndim=1] perm1,
                  cnp.ndarray[INT, ndim=1] perm2):
    cdef cnp.ndarray[INT, ndim=1] inv = np.argsort(perm2)
    cdef cnp.ndarray[INT, ndim=1] comp = perm1[inv]
    return perm1.shape[0] - _lis_length(comp)


cdef int _lis_length(cnp.ndarray[INT, ndim=1] seq):
    cdef int n = seq.shape[0]
    if n == 0:
        return 0

    cdef list tail = []
    cdef int num, left, right, mid

    for num in seq:
        left = 0
        right = len(tail)
        while left < right:
            mid = (left + right) // 2
            if tail[mid] < num:
                left = mid + 1
            else:
                right = mid
        if left == len(tail):
            tail.append(num)
        else:
            tail[left] = num
    return len(tail)