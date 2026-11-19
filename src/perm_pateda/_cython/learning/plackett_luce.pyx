# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.int64_t INT
ctypedef cnp.float64_t FLOAT


cdef class LearnPlackettLuce:
    """Cython version of LearnPlackettLuce (MM algorithm)"""

    def __call__(self,
                 int generation,
                 int n_vars,
                 object cardinality,
                 cnp.ndarray[INT, ndim=2] selected_pop,
                 object selected_fitness,
                 int max_iter=100,
                 double tol=1e-6,
                 **kwargs):

        cdef int K = selected_pop.shape[0]
        cdef int n = selected_pop.shape[1]
        cdef cnp.ndarray[FLOAT, ndim=1] weights = np.ones(n, dtype=np.float64) / n
        cdef cnp.ndarray[FLOAT, ndim=1] wins = np.zeros(n, dtype=np.float64)
        cdef int k, p

        # Win counts
        for k in range(K):
            for p in range(n - 1):
                wins[selected_pop[k, p]] += 1

        cdef cnp.ndarray[FLOAT, ndim=1] old_weights
        cdef cnp.ndarray[FLOAT, ndim=1] denominators
        cdef cnp.ndarray[INT, ndim=1] perm
        cdef cnp.ndarray[FLOAT, ndim=1] suffix_sums
        cdef cnp.ndarray[FLOAT, ndim=1] inv_suffix_sums
        cdef cnp.ndarray[FLOAT, ndim=1] cum_inv
        cdef int i

        for _ in range(max_iter):
            old_weights = weights.copy()
            denominators = np.zeros(n, dtype=np.float64)

            for k in range(K):
                perm = selected_pop[k]

                # suffix_sums[p] = sum of weights of items at positions p..n-1
                suffix_sums = np.cumsum(weights[perm][::-1])[::-1]

                inv_suffix_sums = np.zeros(n, dtype=np.float64)
                for i in range(n - 1):
                    if suffix_sums[i] > 0:
                        inv_suffix_sums[i] = 1.0 / suffix_sums[i]

                cum_inv = np.cumsum(inv_suffix_sums)
                denominators[perm] += cum_inv

            weights = wins / np.maximum(denominators, 1e-12)
            weights = np.maximum(weights, 1e-9)
            weights /= np.sum(weights)

            if np.sum(np.abs(weights - old_weights)) < tol:
                break

        return {
            "weights": weights,
            "model_type": "plackett_luce",
        }