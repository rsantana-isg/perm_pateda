from abc import ABC, abstractmethod
import numpy as np

class PermutationRepresentation(ABC):
    """Abstract base class for factorized permutation representations in perm_pateda."""
    
    @abstractmethod
    def encode(self, perm: np.ndarray) -> np.ndarray:
        """std-perm-repres --> factorized-repres"""
        pass

    @abstractmethod
    def decode(self, code: np.ndarray) -> np.ndarray:
        """factorized-repres --> std-perm-repres"""
        pass
    
    @abstractmethod
    def get_domain(self, n: int) -> list:
        """Return the maximum allowed value (inclusive) at each position i."""
        pass