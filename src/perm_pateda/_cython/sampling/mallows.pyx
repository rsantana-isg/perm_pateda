# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython

from perm_pateda._cython.consensus import compose_permutations

ctypedef cnp.int64_t INT


cdef class SampleMallowsKendall:
    """Cython version of SampleMallowsKendall (optimizada)"""

    def __call__(self,
                 int n_vars,
                 dict model,
                 cnp.ndarray cardinality,
                 cnp.ndarray population,
                 cnp.ndarray fitness,
                 int sample_size,
                 object rng=None):

        if rng is None:
            rng = np.random.default_rng()

        cdef cnp.ndarray[cnp.float64_t, ndim=2] v_probs = model["v_probs"]
        cdef cnp.ndarray[INT, ndim=1] consensus = model["consensus"]

        cdef cnp.ndarray[INT, ndim=2] new_pop = np.zeros((sample_size, n_vars), dtype=np.int64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] rand_values = rng.random((sample_size, n_vars - 1))

        cdef int i
        cdef cnp.ndarray[INT, ndim=1] v_vec
        cdef cnp.ndarray[INT, ndim=1] perm

        for i in range(sample_size):
            v_vec = self._sample_v_vector(v_probs, rand_values[i], n_vars)
            perm = self._generate_perm_from_v(v_vec, n_vars)
            new_pop[i] = compose_permutations(perm, consensus)

        return new_pop

    cdef cnp.ndarray[INT, ndim=1] _sample_v_vector(self,
                                                   cnp.ndarray[cnp.float64_t, ndim=2] v_probs,
                                                   cnp.ndarray[cnp.float64_t, ndim=1] rand_values,
                                                   int n_vars):
        cdef cnp.ndarray[INT, ndim=1] v_vec = np.zeros(n_vars, dtype=np.int64)
        cdef int j
        cdef int index
        cdef double rand_val
        cdef cnp.ndarray[cnp.float64_t, ndim=1] probs
        cdef cnp.ndarray[cnp.float64_t, ndim=1] cumsum

        for j in range(n_vars - 1):
            probs = v_probs[j, :n_vars - j]
            cumsum = np.cumsum(probs)
            rand_val = rand_values[j]
            index = np.searchsorted(cumsum, rand_val, side="right")
            if index >= n_vars - j:
                index = n_vars - j - 1
            v_vec[j] = index

        v_vec[n_vars - 1] = 0
        return v_vec

    cdef cnp.ndarray[INT, ndim=1] _generate_perm_from_v(self,
                                                        cnp.ndarray[INT, ndim=1] v,
                                                        int n_vars):
        """
        Versión más rápida: usa un array de índices disponibles
        y lo va compactando (evita el bucle while lento).
        """
        cdef cnp.ndarray[INT, ndim=1] available = np.arange(n_vars, dtype=np.int64)
        cdef cnp.ndarray[INT, ndim=1] perm = np.zeros(n_vars, dtype=np.int64)
        cdef int i, val, idx, remaining

        remaining = n_vars

        for i in range(n_vars - 1):
            val = v[i]
            if val >= remaining:
                val = remaining - 1

            # El elemento que queremos está en la posición 'val' del array available
            perm[i] = available[val]

            # Compactar: mover los elementos posteriores una posición hacia atrás
            for idx in range(val, remaining - 1):
                available[idx] = available[idx + 1]

            remaining -= 1

        # Último elemento
        perm[n_vars - 1] = available[0]
        return perm

cdef class SampleGeneralizedMallowsKendall:
    

    def __call__(self,
                 int n_vars,
                 dict model,
                 object cardinality,
                 object population,
                 object fitness,
                 int sample_size,
                 object rng=None,
                 **kwargs):

        if rng is None:
            rng = np.random.default_rng()

        cdef cnp.ndarray[cnp.float64_t, ndim=2] v_probs = model["v_probs"]
        cdef cnp.ndarray[INT, ndim=1] consensus = model["consensus"]

        cdef cnp.ndarray[INT, ndim=2] new_pop = np.zeros((sample_size, n_vars), dtype=np.int64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] rand_values = rng.random((sample_size, n_vars - 1))

        cdef int i
        cdef cnp.ndarray[INT, ndim=1] v_vec
        cdef cnp.ndarray[INT, ndim=1] perm

        for i in range(sample_size):
            v_vec = self._sample_v_vector(v_probs, rand_values[i], n_vars)
            perm = self._generate_perm_from_v(v_vec, n_vars)
            new_pop[i] = compose_permutations(perm, consensus)

        return new_pop

    cdef cnp.ndarray[INT, ndim=1] _sample_v_vector(self,
                                                   cnp.ndarray[cnp.float64_t, ndim=2] v_probs,
                                                   cnp.ndarray[cnp.float64_t, ndim=1] rand_values,
                                                   int n_vars):
        cdef cnp.ndarray[INT, ndim=1] v_vec = np.zeros(n_vars, dtype=np.int64)
        cdef int j, index
        cdef double rand_val
        cdef cnp.ndarray[cnp.float64_t, ndim=1] cumsum

        for j in range(n_vars - 1):
            cumsum = np.cumsum(v_probs[j, :n_vars - j])
            rand_val = rand_values[j]
            index = np.searchsorted(cumsum, rand_val)
            if index > n_vars - j - 1:
                index = n_vars - j - 1
            v_vec[j] = index

        v_vec[n_vars - 1] = 0
        return v_vec

    cdef cnp.ndarray[INT, ndim=1] _generate_perm_from_v(self,
                                                        cnp.ndarray[INT, ndim=1] v,
                                                        int n_vars):
        # Reutilizamos la misma implementación optimizada que en SampleMallowsKendall
        cdef cnp.ndarray[INT, ndim=1] available = np.arange(n_vars, dtype=np.int64)
        cdef cnp.ndarray[INT, ndim=1] perm = np.zeros(n_vars, dtype=np.int64)
        cdef int i, val, idx, remaining = n_vars

        for i in range(n_vars - 1):
            val = v[i]
            if val >= remaining:
                val = remaining - 1
            perm[i] = available[val]
            for idx in range(val, remaining - 1):
                available[idx] = available[idx + 1]
            remaining -= 1

        perm[n_vars - 1] = available[0]
        return perm