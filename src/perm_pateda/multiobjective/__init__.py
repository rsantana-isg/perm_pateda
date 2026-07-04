"""
Multi-objective optimization for permutation problems.

Provides:
- Scalarization functions (weighted sum, Tchebycheff)
- Pareto dominance and archive management
- MEDA/D-MK: Decomposition-based EDA with Kernels of Mallows Models (Cayley)
- MEDA/D-KENDALL: Decomposition-based EDA with Mallows Kendall
- MEDA/D-ULAM: Decomposition-based EDA with Mallows Ulam
- MEDA/D-GMKENDALL: Decomposition-based EDA with Generalized Mallows Kendall
- MEDA/D-GMCAYLEY: Decomposition-based EDA with Generalized Mallows Cayley
- MEDA/D-PLACKETT_LUCE: Decomposition-based EDA with the Plackett-Luce model
- MEDA/D-MIXTURE_PLACKETT_LUCE: Decomposition-based EDA with a mixture of
  Plackett-Luce models
- MEDA/D-NHM: Decomposition-based EDA with the Node Histogram Model
- MEDA/D-EHM: Decomposition-based EDA with the Edge Histogram Model

All nine bi-objective algorithms share a common interface: ``__init__``
accepts ``objectives``, ``n``, ``n_subproblems``, ``neighbourhood_size``,
``nr``, ``scalarization``, ``shake_threshold``, ``shake_strength``,
``minimize``, ``seed``, and an optional external mutation operator via
``mutation_fn`` (a callable ``(perm, rng) -> perm``) and ``mutation_rate``
(probability of applying it to a freshly sampled candidate before
evaluation, default 0.0 = disabled). ``run()`` accepts ``n_generations``,
``verbose``, ``initial_population``, and ``generation_callback``.
"""

from perm_pateda.multiobjective.scalarization import weighted_sum, tchebycheff
from perm_pateda.multiobjective.pareto import dominates, pareto_front, ParetoArchive
from perm_pateda.multiobjective.meda_d_mk import (
    MEDA_D_MK,
    MEDA_D_KENDALL,
    MEDA_D_ULAM,
    MEDA_D_GMKENDALL,
    MEDA_D_GMCAYLEY,
)
from perm_pateda.multiobjective.meda_d_pl import (
    MEDA_D_PLACKETT_LUCE,
    MEDA_D_MIXTURE_PLACKETT_LUCE,
)
from perm_pateda.multiobjective.meda_d_hm import (
    MEDA_D_NHM,
    MEDA_D_EHM,
)

__all__ = [
    "weighted_sum", "tchebycheff",
    "dominates", "pareto_front", "ParetoArchive",
    "MEDA_D_MK",
    "MEDA_D_KENDALL",
    "MEDA_D_ULAM",
    "MEDA_D_GMKENDALL",
    "MEDA_D_GMCAYLEY",
    "MEDA_D_PLACKETT_LUCE",
    "MEDA_D_MIXTURE_PLACKETT_LUCE",
    "MEDA_D_NHM",
    "MEDA_D_EHM",
]
