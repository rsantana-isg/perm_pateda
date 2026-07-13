"""Permutation factorized representations.

Alternative bijective representations for permutations (Lehmer codes, 
Fisher-Yates draws, and Insertion-Vectors) mapping the symmetric group 
to independent bounded integer domains.
"""

from perm_pateda.representations.base import PermutationRepresentation
from perm_pateda.representations.lehmer import LehmerRepresentation
from perm_pateda.representations.fisher_yates import FisherYatesRepresentation
from perm_pateda.representations.vinsertion import InsertionVectorRepresentation

__all__ = [
    "PermutationRepresentation",
    "LehmerRepresentation",
    "FisherYatesRepresentation",
    "InsertionVectorRepresentation",
]