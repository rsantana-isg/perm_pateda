# cython: boundscheck=False
# cython: wraparound=False

import numpy as np
cimport numpy as cnp
from perm_pateda._cython.representations.lehmer import LehmerRepresentation

ctypedef cnp.int64_t INT


cdef class LearnLehmerUMDA:

    cdef public object rep
    cdef public double laplace_smoothing

    def __init__(self, double laplace_smoothing=0.01):
        self.rep = LehmerRepresentation(left=False)
        self.laplace_smoothing = laplace_smoothing

    def __call__(self,
                 int generation,
                 int n_vars,
                 object cardinality,
                 cnp.ndarray[INT, ndim=2] selected_pop,
                 object selected_fitness,
                 **kwargs):


        codes = self.rep.encode(selected_pop)
        K, n = codes.shape


        marginals = []
        alpha = self.laplace_smoothing

        for i in range(n):
            domain = n - i if not self.rep.left else i + 1
            counts = np.bincount(codes[:, i], minlength=domain).astype(np.float64)
            probs = (counts + alpha) / (counts.sum() + alpha * domain)
            marginals.append(probs)

        return {
            "marginals": marginals,
            "n_vars": n_vars,
            "model_type": "lehmer_umda",
        }