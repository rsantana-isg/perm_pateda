"""Permutation model learning methods.

Histogram models (edge / node), distance-based exponential models
(Mallows and Generalized Mallows under Kendall's-tau and Cayley distances),
and Doubly Stochastic Matrix (DSM) models.
"""

from perm_pateda.learning.histogram import LearnEHM, LearnNHM, learn_ehm, learn_nhm
from perm_pateda.learning.dsm import LearnDSM, learn_dsm
from perm_pateda.learning.mallows import (
    LearnMallowsKendall,
    LearnMallowsCayley,
    LearnMallowsUlam,
    LearnGeneralizedMallowsKendall,
    LearnGeneralizedMallowsCayley,
    learn_mallows_kendall,
    learn_mallows_cayley,
    learn_mallows_ulam,
)

from perm_pateda.learning.plackett_luce import LearnPlackettLuce
from perm_pateda.learning.mixture_plackett_luce import LearnPlackettLuceMixture
from perm_pateda.learning.hamming_kmm import LearnHammingKMM

from perm_pateda.learning.umda import LearnUMDA
from perm_pateda.learning.tree import LearnTree
from perm_pateda.learning.markov import LearnMarkov
from perm_pateda.learning.lehmer import LearnLehmerUMDA, LearnLehmerTree, LearnLehmerMarkov
from perm_pateda.learning.fisher_yates import (
    LearnFisherYatesUMDA, LearnFisherYatesTree, LearnFisherYatesMarkov,
)
from perm_pateda.learning.vinsertion import (
    LearnInsertionVectorUMDA, LearnInsertionVectorChain, LearnInsertionVectorTree,
)

__all__ = [
    "LearnEHM",
    "LearnNHM",
    "learn_ehm",
    "learn_nhm",
    "LearnDSM",
    "learn_dsm",
    "LearnMallowsKendall",
    "LearnMallowsCayley",
    "LearnMallowsUlam",
    "LearnGeneralizedMallowsKendall",
    "LearnGeneralizedMallowsCayley",
    "learn_mallows_kendall",
    "learn_mallows_cayley",
    "learn_mallows_ulam",
    "LearnPlackettLuce",
    "LearnPlackettLuceMixture",
    "LearnHammingKMM",
    "LearnUMDA",
    "LearnTree",
    "LearnMarkov",
    "LearnLehmerUMDA",
    "LearnLehmerTree",
    "LearnLehmerMarkov",
    "LearnFisherYatesUMDA",
    "LearnFisherYatesTree",
    "LearnFisherYatesMarkov",
    "LearnInsertionVectorUMDA",
    "LearnInsertionVectorChain",
    "LearnInsertionVectorTree",
]
