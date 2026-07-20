"""Permutation sampling methods.

Histogram samplers (edge / node), distance-based samplers for the Mallows
and Generalized Mallows models under Kendall's-tau and Cayley distances,
and Doubly Stochastic Matrix (DSM) samplers (PS and AS variants).
"""

from perm_pateda.sampling.histogram import SampleEHM, SampleNHM, sample_ehm, sample_nhm
from perm_pateda.sampling.dsm import (
    SampleDSMPS,
    SampleDSMAS,
    sample_dsm_ps,
    sample_dsm_as,
)
from perm_pateda.sampling.mallows import (
    SampleMallowsKendall,
    SampleMallowsCayley,
    SampleMallowsUlam,
    SampleGeneralizedMallowsKendall,
    SampleGeneralizedMallowsCayley,
    sample_mallows_kendall,
    sample_mallows_cayley,
    sample_mallows_ulam,
)

from perm_pateda.sampling.plackett_luce import SamplePlackettLuce
from perm_pateda.sampling.mixture_plackett_luce import SamplePlackettLuceMixture
from perm_pateda.sampling.hamming_kmm import SampleHammingKMM 

from perm_pateda.sampling.lehmer import SampleLehmerUMDA, SampleLehmerTree
from perm_pateda.sampling.fisher_yates import SampleFisherYatesUMDA, SampleFisherYatesTree
from perm_pateda.sampling.vinsertion import  SampleInsertionVectorUMDA, SampleInsertionVectorChain


__all__ = [
    "SampleEHM",
    "SampleNHM",
    "sample_ehm",
    "sample_nhm",
    "SampleDSMPS",
    "SampleDSMAS",
    "sample_dsm_ps",
    "sample_dsm_as",
    "SampleMallowsKendall",
    "SampleMallowsCayley",
    "SampleMallowsUlam",
    "SampleGeneralizedMallowsKendall",
    "SampleGeneralizedMallowsCayley",
    "sample_mallows_kendall",
    "sample_mallows_cayley",
    "sample_mallows_ulam"
    "SamplePlackettLuce"
    "SamplePlackettLuceMixture",
    "SampleHammingKMM",
    "SampleLehmerUMDA",
    "SampleLehmerTree",
    "SampleFisherYatesUMDA",
    "SampleFisherYatesTree",
    "SampleInsertionVectorUMDA",
    "SampleInsertionVectorChain",

]
