from abc import ABC, abstractmethod
import numpy as np

class PermutationRepresentation(ABC):
    """Clase base abstracta para representaciones factorizadas en perm_pateda."""
    
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
        """Devuelve el valor máximo permitido (inclusivo) en cada posición i."""
        pass