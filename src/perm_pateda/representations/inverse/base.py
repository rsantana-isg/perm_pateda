from __future__ import annotations

import numpy as np
from perm_pateda.representations.base import PermutationRepresentation


class InverseRepresentation(PermutationRepresentation):

    def get_domain(self, n_vars: int) -> list[int]:

        return [n_vars] * n_vars

    def encode(self, permutation: np.ndarray) -> np.ndarray:
        return self._invert(permutation)

    def decode(self, code: np.ndarray) -> np.ndarray:
        return self._invert(code)

    @staticmethod
    def _invert(perm: np.ndarray) -> np.ndarray:
        if perm.ndim == 1:
            inv = np.empty_like(perm)
            inv[perm] = np.arange(len(perm))
            return inv
        else:
            pop_size, n = perm.shape
            inv = np.empty_like(perm)
            for i in range(pop_size):
                inv[i, perm[i]] = np.arange(n)
            return inv