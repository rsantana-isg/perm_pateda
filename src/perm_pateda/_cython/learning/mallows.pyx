# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython
from scipy.optimize import fminbound

from perm_pateda._cython.distances import kendall_distance, v_vector
from perm_pateda._cython.consensus import get_consensus

ctypedef cnp.int64_t INT


cdef class LearnMallowsKendall:
    """Cython version of LearnMallowsKendall"""

    def __call__(self,
                 int generation,
                 int n_vars,
                 cnp.ndarray cardinality,
                 cnp.ndarray[INT, ndim=2] selected_pop,
                 cnp.ndarray[cnp.float64_t, ndim=1] selected_fitness,
                 double initial_theta=0.1,
                 double upper_theta=10.0,
                 int max_iter=100,
                 str consensus_method="borda"):

        consensus = get_consensus(
            consensus_method,
            selected_pop,
            selected_fitness,
            kendall_distance
        )

        theta = self._calculate_theta(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        psis = self._calculate_psi_constants(theta, n_vars)
        v_probs = self._calculate_v_prob_matrix(n_vars, psis, theta)

        return {
            "v_probs": v_probs,
            "consensus": consensus,
            "theta": theta,
            "psis": psis,
            "model_type": "mallows_kendall",
        }

    cdef double _calculate_theta(self,
                                 cnp.ndarray[INT, ndim=1] consensus,
                                 cnp.ndarray[INT, ndim=2] population,
                                 double initial_theta,
                                 double upper_theta,
                                 int max_iter,
                                 int n_vars):

        inv_consensus = np.argsort(consensus)
        n_pop = population.shape[0]
        v_vectors = np.zeros((n_pop, n_vars), dtype=np.float64)

        for i in range(n_pop):
            composition = population[i, inv_consensus]
            v_vectors[i] = v_vector(composition)

        v_mean = np.mean(v_vectors, axis=0)

        def objective(theta):
            expected = self._expected_v_vector(theta, n_vars)
            return np.sum((expected - v_mean) ** 2)

        theta_opt = fminbound(objective, 0.001, upper_theta, xtol=1e-6, maxfun=max_iter)
        return float(theta_opt)

    def _expected_v_vector(self, double theta, int n):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] expected_v = np.zeros(n, dtype=np.float64)
        cdef int j, r
        cdef double psi_j, expected_val, prob_r

        for j in range(n - 1):
            psi_j = (1.0 - np.exp(-(n - j) * theta)) / (1.0 - np.exp(-theta))
            expected_val = 0.0
            for r in range(n - j):
                prob_r = np.exp(-r * theta) / psi_j
                expected_val += r * prob_r
            expected_v[j] = expected_val

        return expected_v

    def _calculate_psi_constants(self, double theta, int n):
        cdef cnp.ndarray[cnp.int64_t, ndim=1] j = np.arange(1, n, dtype=np.int64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] psis = (
            (1.0 - np.exp(-(n - j + 1) * theta)) / (1.0 - np.exp(-theta))
        )
        return psis

    def _calculate_v_prob_matrix(self, int n_vars, cnp.ndarray[cnp.float64_t, ndim=1] psis, double theta):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] v_probs = np.zeros((n_vars - 1, n_vars), dtype=np.float64)
        cdef int j, r

        for j in range(n_vars - 1):
            for r in range(n_vars - j):
                v_probs[j, r] = np.exp(-r * theta) / psis[j]

        return v_probs


cdef class LearnGeneralizedMallowsKendall:
    """Cython version of LearnGeneralizedMallowsKendall"""

    def __call__(self,
                 int generation,
                 int n_vars,
                 object cardinality,
                 cnp.ndarray[INT, ndim=2] selected_pop,
                 object selected_fitness,
                 double initial_theta=0.1,
                 double upper_theta=10.0,
                 int max_iter=100,
                 str consensus_method="borda",
                 **kwargs):

        consensus = get_consensus(
            consensus_method,
            selected_pop,
            selected_fitness,
            kendall_distance
        )

        thetas = self._calculate_thetas(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        psis = self._calculate_psi_constants(thetas, n_vars)
        v_probs = self._calculate_v_prob_matrix(n_vars, psis, thetas)

        return {
            "v_probs": v_probs,
            "consensus": consensus,
            "theta": thetas,
            "psis": psis,
            "model_type": "generalized_mallows_kendall",
        }

    def _calculate_thetas(self,
                          cnp.ndarray[INT, ndim=1] consensus,
                          cnp.ndarray[INT, ndim=2] population,
                          double initial_theta,
                          double upper_theta,
                          int max_iter,
                          int n_vars):
        cdef cnp.ndarray[INT, ndim=1] inv_consensus = np.argsort(consensus)
        cdef int n_pop = population.shape[0]
        cdef cnp.ndarray[cnp.float64_t, ndim=2] v_vectors = np.zeros((n_pop, n_vars), dtype=np.float64)
        cdef int i

        for i in range(n_pop):
            composition = population[i, inv_consensus]
            v_vectors[i] = v_vector(composition)

        v_mean = np.mean(v_vectors, axis=0)          # ← sin cdef
        thetas = np.zeros(n_vars - 1, dtype=np.float64)
        cdef int j

        for j in range(n_vars - 1):
            def theta_function(theta, j=j):
                m = n_vars - j - 1
                if theta < 1e-12:
                    return m / 2.0 - v_mean[j]
                x = np.exp(-theta)
                r = np.arange(m + 1)
                w = x ** r
                expected_vj = float(np.dot(r, w) / w.sum())
                return expected_vj - v_mean[j]

            try:
                theta_j = fminbound(
                    lambda t: abs(theta_function(t)),
                    0.001,
                    upper_theta,
                    xtol=1e-6,
                    maxfun=max_iter
                )
            except:
                theta_j = initial_theta

            thetas[j] = theta_j

        return thetas

    def _calculate_psi_constants(self, cnp.ndarray[cnp.float64_t, ndim=1] thetas, int n):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] psis = np.zeros(n - 1, dtype=np.float64)
        cdef int j, n_j
        cdef double theta_j

        for j in range(n - 1):
            theta_j = thetas[j]
            n_j = n - j - 1
            psis[j] = np.sum(np.exp(-np.arange(n_j + 1) * theta_j))

        return psis

    def _calculate_v_prob_matrix(self, int n_vars,
                                 cnp.ndarray[cnp.float64_t, ndim=1] psis,
                                 cnp.ndarray[cnp.float64_t, ndim=1] thetas):
        cdef cnp.ndarray[cnp.float64_t, ndim=2] v_probs = np.zeros((n_vars - 1, n_vars), dtype=np.float64)
        cdef int j, r
        cdef double theta_j

        for j in range(n_vars - 1):
            theta_j = thetas[j]
            for r in range(n_vars - j):
                v_probs[j, r] = np.exp(-r * theta_j) / psis[j]

        return v_probs