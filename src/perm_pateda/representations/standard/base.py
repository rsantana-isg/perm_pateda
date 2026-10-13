from __future__ import annotations

import numpy as np
from perm_pateda.representations.base import PermutationRepresentation


class StandardRepresentation(PermutationRepresentation):
    
    def get_domain(self, n_vars: int) -> list[int]:
        return [n_vars] * n_vars

    def encode(self, permutation: np.ndarray) -> np.ndarray:
        return permutation

    def decode(self, code: np.ndarray) -> np.ndarray:
        return code