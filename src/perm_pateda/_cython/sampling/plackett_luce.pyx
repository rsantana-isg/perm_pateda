# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp
cimport cython

ctypedef cnp.int64_t INT
ctypedef cnp.float64_t FLOAT


cdef class SamplePlackettLuce:
    """Cython version of SamplePlackettLuce (Gumbel-max trick)"""

    def __call__(self,
                 int n_vars,
                 dict model,
                 cnp.ndarray cardinality,
                 cnp.ndarray population,
                 cnp.ndarray fitness,
                 int sample_size,
                 object rng,
                 **kwargs):

        cdef cnp.ndarray[FLOAT, ndim=1] weights = model["weights"]
        cdef cnp.ndarray[FLOAT, ndim=1] log_w = np.log(np.maximum(weights, 1e-12))

        # Gumbel noise
        cdef cnp.ndarray[FLOAT, ndim=2] u = rng.uniform(0.0, 1.0, size=(sample_size, n_vars))
        cdef cnp.ndarray[FLOAT, ndim=2] gumbel_noise = -np.log(-np.log(u + 1e-12))

        cdef cnp.ndarray[FLOAT, ndim=2] scores = log_w + gumbel_noise

        # argsort descending
        cdef cnp.ndarray[INT, ndim=2] samples = np.argsort(scores, axis=1)[:, ::-1].astype(np.int64)

        return samples