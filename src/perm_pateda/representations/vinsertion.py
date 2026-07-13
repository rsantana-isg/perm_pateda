import numpy as np
from .base import PermutationRepresentation
from .lehmer import LehmerRepresentation

class InsertionVectorRepresentation(PermutationRepresentation):
    def __init__(self):
        self.lehmer_left = LehmerRepresentation(left=True)

    def encode(self, perm: np.ndarray) -> np.ndarray:
        inv_perm = np.argsort(perm, axis=-1)
        return self.lehmer_left.encode(inv_perm)

    def decode(self, code: np.ndarray) -> np.ndarray:
        inv_perm = self.lehmer_left.decode(code)
        return np.argsort(inv_perm, axis=-1)

    def get_domain(self, n: int) -> list:
        return [i for i in range(n)]