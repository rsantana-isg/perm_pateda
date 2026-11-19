# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.int64_t INT


def find_consensus_borda(cnp.ndarray[INT, ndim=2] population):
    cdef int pop_size = population.shape[0]
    cdef int n_vars = population.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] scores = np.zeros(n_vars, dtype=np.float64)
    cdef int i, pos, item
    cdef bint one_indexed = (np.min(population) == 1)

    for i in range(pop_size):
        for pos in range(n_vars):
            item = population[i, pos]
            if one_indexed:
                item -= 1
            scores[item] += (n_vars - pos)

    cdef cnp.ndarray[INT, ndim=1] consensus = np.argsort(-scores).astype(np.int64)

    if one_indexed:
        consensus += 1

    return consensus


def find_consensus_median(object distance_func,
                          cnp.ndarray[INT, ndim=2] population):
    cdef int pop_size = population.shape[0]
    cdef int i, j
    cdef double total_dist
    cdef double min_total = 1e300
    cdef int best_idx = 0

    for i in range(pop_size):
        total_dist = 0.0
        for j in range(pop_size):
            if i != j:
                total_dist += distance_func(population[i], population[j])
        if total_dist < min_total:
            min_total = total_dist
            best_idx = i

    return population[best_idx].copy()


def find_consensus_best(cnp.ndarray[INT, ndim=2] population,
                        cnp.ndarray[cnp.float64_t, ndim=1] fitness):
    cdef int best_idx = np.argmax(fitness)
    return population[best_idx].copy()


def get_consensus(str method,
                  cnp.ndarray[INT, ndim=2] population,
                  cnp.ndarray[cnp.float64_t, ndim=1] fitness=None,
                  object distance_func=None):
    method = method.lower()

    if method == "borda":
        return find_consensus_borda(population)
    elif method == "setmedian":
        if distance_func is None:
            raise ValueError("distance_func must be provided for 'setmedian'")
        return find_consensus_median(distance_func, population)
    elif method == "best":
        if fitness is None:
            raise ValueError("fitness must be provided for 'best'")
        return find_consensus_best(population, fitness)
    else:
        raise ValueError(f"Unknown consensus method: {method}")

def compose_permutations(cnp.ndarray[INT, ndim=1] perm1,
                         cnp.ndarray[INT, ndim=1] perm2):
   
    return perm2[perm1]