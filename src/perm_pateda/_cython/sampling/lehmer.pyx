# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from perm_pateda._cython.representations.lehmer import LehmerRepresentation

ctypedef cnp.int64_t INT


cdef class SampleLehmerUMDA:
    
    cdef public object rep
    
    def __init__(self):
        self.rep = LehmerRepresentation(left=False)

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

        marginals = model["marginals"]
        n = model["n_vars"]

        codes = np.zeros((sample_size, n), dtype=np.int64)
        for i in range(n):
            probs = marginals[i]
            codes[:, i] = rng.choice(len(probs), size=sample_size, p=probs)

        return self.rep.decode(codes)