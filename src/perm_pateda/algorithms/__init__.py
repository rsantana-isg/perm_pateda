"""
Plug-and-play permutation EDA wrappers::

    from perm_pateda import EHMEDA, MallowsKendallEDA, DSMPSEDA, DSMASEDA, ...
"""

from perm_pateda.algorithms.permutation import (
    EHMEDA,
    NHMEDA,
    MallowsKendallEDA,
    MallowsCayleyEDA,
    GMallowsKendallEDA,
    GMallowsCayleyEDA,
    DSMPSEDA,
    DSMASEDA,
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
    "GMallowsKendallEDA",
    "GMallowsCayleyEDA",
    "DSMPSEDA",
    "DSMASEDA",
    "RKGaussianUMDAEDA",
    "RKGaussianFullEDA",
    "RKCopulaVinesEDA",
]
