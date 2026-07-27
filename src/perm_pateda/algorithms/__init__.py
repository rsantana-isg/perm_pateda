"""
Plug-and-play permutation EDA wrappers::

    from perm_pateda import EHMEDA, MallowsKendallEDA, DSMPSEDA, DSMASEDA, ...
"""

from perm_pateda.algorithms.permutation import (
    EHMEDA,
    NHMEDA,
    MallowsKendallEDA,
    MallowsCayleyEDA,
    MallowsUlamEDA,
    GMallowsKendallEDA,
    GMallowsCayleyEDA,
    HammingKMMEDA,
    PlackettLuceEDA,
    PlackettLuceMixtureEDA,
    DSMPSEDA,
    DSMASEDA,
    LehmerUmdaEDA,
    LehmerTreeEDA,
    LehmerMarkovEDA,
    FisherYatesUmdaEDA,
    FisherYatesTreeEDA,
    FisherYatesMarkovEDA,
    InsertionVectorUmdaEDA,
    InsertionVectorMarkovEDA,
    InsertionVectorTreeEDA,
)
from perm_pateda.algorithms.random_keys import (
    RKGaussianUMDAEDA,
    RKGaussianFullEDA,
    RKCopulaVinesEDA,
)

__all__ = [
    "EHMEDA",
    "NHMEDA",
    "MallowsKendallEDA",
    "MallowsCayleyEDA",
    "MallowsUlamEDA",
    "GMallowsKendallEDA",
    "GMallowsCayleyEDA",
    "HammingKMMEDA",
    "PlackettLuceEDA",
    "PlackettLuceMixtureEDA",
    "DSMPSEDA",
    "DSMASEDA",
    "RKGaussianUMDAEDA",
    "RKGaussianFullEDA",
    "RKCopulaVinesEDA",
    "LehmerUmdaEDA",
    "LehmerTreeEDA",
    "LehmerMarkovEDA",
    "FisherYatesUmdaEDA",
    "FisherYatesTreeEDA",
    "FisherYatesMarkovEDA",
    "InsertionVectorUmdaEDA",
    "InsertionVectorMarkovEDA",
    "InsertionVectorTreeEDA",
]
