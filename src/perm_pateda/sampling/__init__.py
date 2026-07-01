"""Permutation sampling methods.

Histogram samplers (edge / node) and distance-based samplers for the Mallows
and Generalized Mallows models under Kendall's-tau and Cayley distances.
"""

from perm_pateda.sampling.histogram import SampleEHM, SampleNHM, sample_ehm, sample_nhm
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
__all__ = [
    "SampleEHM",
    "SampleNHM",
    "sample_ehm",
    "sample_nhm",
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
]
