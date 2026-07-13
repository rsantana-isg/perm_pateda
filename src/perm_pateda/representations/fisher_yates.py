import numpy as np
from .base import PermutationRepresentation

class FisherYatesRepresentation(PermutationRepresentation):
    def encode(self, perm: np.ndarray) -> np.ndarray:
        original_num_dims = perm.ndim
        perm = np.atleast_2d(perm)
        B, n = perm.shape
        
        perm_base = np.tile(np.arange(n), (B, 1))
        fisher_yates = np.zeros_like(perm)
        batch_idx = np.arange(B)
        
        for i in range(n):
            matches = (perm_base[:, i:] == perm[:, [i]])
            j = np.argmax(matches, axis=1)
            fisher_yates[batch_idx, i] = j
            
            
            swap_idx = i + j
            temp = perm_base[batch_idx, i].copy()
            perm_base[batch_idx, i] = perm_base[batch_idx, swap_idx]
            perm_base[batch_idx, swap_idx] = temp
            
        return fisher_yates.squeeze() if original_num_dims == 1 else fisher_yates

    def decode(self, code: np.ndarray) -> np.ndarray:
        original_num_dims = code.ndim
        code = np.atleast_2d(code)
        B, n = code.shape
        
        perm = np.tile(np.arange(n), (B, 1))
        batch_idx = np.arange(B)
        
        for i in range(n):
            j = code[:, i] + i
            # Swap vectorial inverso
            temp = perm[:, i].copy()
            perm[:, i] = perm[batch_idx, j]
            perm[batch_idx, j] = temp
            
        return perm.squeeze() if original_num_dims == 1 else perm

    def get_domain(self, n: int) -> list:
        return [n - 1 - i for i in range(n)]