import numpy as np
from .base import PermutationRepresentation

class LehmerRepresentation(PermutationRepresentation):
    def __init__(self, left: bool = False):
        self.left = left  

    def encode(self, perm: np.ndarray) -> np.ndarray:
        original_num_dims = perm.ndim
        perm = np.atleast_2d(perm.copy())
        batch_size, n = perm.shape
        
        if self.left:
            for i in reversed(range(1, n)):
                perm[:, i] = np.sum(perm[:, [i]] > perm[:, :i], axis=-1)
            perm[:, 0] = 0  
        else:
            for i in range(n - 1):
                perm[:, i] = np.sum(perm[:, [i]] > perm[:, i+1:], axis=-1)
            perm[:, -1] = 0 
            
        return perm.squeeze() if original_num_dims == 1 else perm

    def decode(self, code: np.ndarray) -> np.ndarray:
        original_num_dims = code.ndim
        perm = np.atleast_2d(code.copy())
        batch_size, n = perm.shape
        
        for i in range(1, n):
            if self.left:
                perm[:, :i] += (perm[:, [i]] <= perm[:, :i]).astype(int)
            else:
                j = n - i - 1
                perm[:, j+1:] += (perm[:, [j]] <= perm[:, j+1:]).astype(int)
                
        return perm.squeeze() if original_num_dims == 1 else perm

    def get_domain(self, n: int) -> list:
        return [i for i in range(n)] if self.left else [n - 1 - i for i in range(n)]