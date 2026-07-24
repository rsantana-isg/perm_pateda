# perm_pateda

**Estimation of Distribution Algorithms (EDAs) for permutation-based combinatorial
optimization problems.**

`perm_pateda` is an independent, permutation-focused companion to
[`pateda`](https://github.com/rsantana-isg/pateda). It contributes everything that
is specific to the permutation space — distances, consensus estimators, probability
models over the symmetric group, their learning and sampling methods, permutation
seeding, benchmark problems and instance readers — and reuses `pateda` for
everything common to all EDAs (the core EDA engine, component interfaces,
selection, replacement, statistics, stopping conditions and visualization).

The five distance-based EDAs of the **`perm_mateda`** Matlab toolbox
(Irurozki, Ceberio, Santamaria, Santana & Mendiburu, 2018, *Algorithm 989*,
ACM TOMS 44(4), Article 47) are all available, and the library extends them with a
substantially wider catalogue of permutation models, four additional problems, a
decomposition-based multi-objective framework, and reproducibility tooling.

A full description of the library is given in the user guide,
[`paper/perm_pateda_user_guide.tex`](paper/perm_pateda_user_guide.tex).

---

## Relationship to `pateda`

```
pateda  ─────────────────────────────►  perm_pateda
(core EDA engine, selection,            (permutation models: learning + sampling,
 replacement, statistics, viz,           distances, consensus, representations,
 discrete & continuous EDAs)             permutation problems, MEDA/D)
```

`pateda` is a **dependency** of `perm_pateda`. All permutation-related code has
been removed from `pateda` and now lives here, so the two packages have a clean
separation of concerns:

| Concern | Package |
|---|---|
| Core EDA loop, components, model containers | `pateda` |
| Selection / replacement / statistics / stopping / visualization | `pateda` |
| Discrete & continuous EDAs (UMDA, EBNA, Gaussian, vine copulas, …) | `pateda` |
| Permutation distances (Kendall, Cayley, Ulam, Hamming) | `perm_pateda` |
| Consensus (central-permutation) estimators | `perm_pateda` |
| All permutation model learning / sampling | `perm_pateda` |
| Bijective codings (Lehmer, Fisher–Yates, insertion vector) and random keys | `perm_pateda` |
| Permutation problems (TSP, PFSP, LOP, QAP, MIS, MaxCut, MVC) | `perm_pateda` |
| Decomposition-based multi-objective permutation EDAs (MEDA/D) | `perm_pateda` |

---

## Installation

`pateda` is an in-development package (not yet on PyPI), so install it first
from the local checkout, then install `perm_pateda`:

```bash
# from the repository root (…/github/pateda)
pip install -e packages/pateda
pip install -e packages/perm_pateda
```

For development tooling (pytest, ruff, black, mypy):

```bash
pip install -e "packages/perm_pateda[dev]"
```

Requires Python ≥ 3.9, `numpy`, `scipy` and `pateda`. The mixture of
Plackett–Luce models additionally uses `scikit-learn`; the statistical
comparison utilities use `pandas` and `matplotlib`.

---

## Quick start

```python
import numpy as np
from perm_pateda import MallowsKendallEDA
from perm_pateda.functions import create_random_lop

# A Linear Ordering Problem instance on 15 items
lop = create_random_lop(15, seed=0)

alg = MallowsKendallEDA(
    n_vars=15,
    fitness_func=lop,        # callable: permutation -> scalar (higher is better)
    pop_size=100,
    n_gen=50,
    selection_ratio=0.3,
    random_seed=0,
)
stats, _ = alg.run()
print("Best fitness:", stats.best_fitness_overall)
print("Best permutation:", stats.best_individual)
```

Every algorithm class takes the same core parameters (`n_vars`, `fitness_func`,
`pop_size`, `n_gen`, `selection_ratio`, `elitism`, `random_seed`) plus the
model-specific extras listed below, and `run(verbose=False)` returns the
`(stats, cache)` pair of `pateda` objects.

**Fitness is always maximized.** The problem classes follow this convention
internally (`TSP`, `QAP`, `PFSP` and `MVC` return negated costs) and expose an
`evaluate_*` method returning the natural positive objective value.

---

## Available algorithms

### Distance-based exponential models

Import from `perm_pateda`.

| Class | Model | Distance | Extra parameters |
|---|---|---|---|
| `MallowsKendallEDA`  | Mallows             | Kendall's-τ | — |
| `MallowsCayleyEDA`   | Mallows             | Cayley      | — |
| `MallowsUlamEDA`     | Mallows (MCMC sampler) | Ulam     | `burn_in`, `step_size` |
| `GMallowsKendallEDA` | Generalized Mallows | Kendall's-τ | — |
| `GMallowsCayleyEDA`  | Generalized Mallows | Cayley      | — |
| `HammingKMMEDA`      | Kernels of Mallows models | Hamming | `expected_dist_start`, `expected_dist_end`, `gamma` |

All Mallows/GM learners accept `consensus_method` ∈ `{"borda", "setmedian", "best"}`.

### Histogram, multistage and matrix models

Import from `perm_pateda`.

| Class | Model | Extra parameters |
|---|---|---|
| `EHMEDA`                 | Edge Histogram Model | — |
| `NHMEDA`                 | Node Histogram Model | — |
| `PlackettLuceEDA`        | Plackett–Luce, MM algorithm | — |
| `PlackettLuceMixtureEDA` | Mixture of *K* Plackett–Luce models, spectral EM | `n_components` |
| `DSMPSEDA`               | Doubly stochastic matrix, probabilistic sampling | `alpha` |
| `DSMASEDA`               | Doubly stochastic matrix, algebraic sampling | `alpha` |

### Models over bijective codings of the symmetric group

Import from `perm_pateda.algorithms`. Every code decodes to a valid permutation,
so no repair operator is ever needed.

| Class | Coding | Model | Extra parameters |
|---|---|---|---|
| `LehmerUmdaEDA`            | Lehmer code       | Univariate marginals | `laplace_smoothing` |
| `LehmerTreeEDA`            | Lehmer code       | Chow–Liu tree | `laplace_smoothing`, `root` |
| `FisherYatesUmdaEDA`       | Fisher–Yates draws| Univariate marginals | `laplace_smoothing` |
| `FisherYatesTreeEDA`       | Fisher–Yates draws| Chow–Liu tree | `laplace_smoothing`, `root` |
| `InsertionVectorUmdaEDA`   | Insertion vector  | Univariate marginals | `laplace_smoothing` |
| `InsertionVectorMarkovEDA` | Insertion vector  | First-order Markov chain | `laplace_smoothing` |

### Random-key EDAs

Import from `perm_pateda`. Permutations are searched through a continuous
relaxation in `[0,1]^n`, using the continuous models of `pateda`.

| Class | Model | Extra parameters |
|---|---|---|
| `RKGaussianUMDAEDA` | Univariate Gaussian | `diminishing`, `cooling`, `initial_sigma`, `min_sigma` |
| `RKGaussianFullEDA` | Full multivariate Gaussian | idem |
| `RKCopulaVinesEDA`  | Vine copulas | idem + `truncation_level` |

---

## Multi-objective optimization

`perm_pateda.multiobjective` implements the MEDA/D framework (MOEA/D-style
decomposition combined with permutation probability models), instantiated with
nine models:

`MEDA_D_MK` (Mallows kernels, Cayley), `MEDA_D_KENDALL`, `MEDA_D_ULAM`,
`MEDA_D_GMKENDALL`, `MEDA_D_GMCAYLEY`, `MEDA_D_PLACKETT_LUCE`,
`MEDA_D_MIXTURE_PLACKETT_LUCE`, `MEDA_D_NHM`, `MEDA_D_EHM`.

```python
from perm_pateda.multiobjective import MEDA_D_GMCAYLEY
from perm_pateda.functions import create_random_qap, create_random_lop

n = 20
qap, lop = create_random_qap(n, seed=1), create_random_lop(n, seed=2)
objectives = [lambda p: -qap(p), lambda p: -lop(p)]   # both minimized

alg = MEDA_D_GMCAYLEY(objectives=objectives, n=n, n_subproblems=50,
                      neighbourhood_size=10, nr=2,
                      scalarization="tchebycheff",
                      minimize=(True, True), seed=0)
result = alg.run(n_generations=200)
print("Pareto front size:", len(result["pareto_solutions"]))
```

All nine classes share the same constructor signature (`objectives`, `n`,
`n_subproblems`, `neighbourhood_size`, `nr`, `scalarization`, `shake_threshold`,
`shake_strength`, `minimize`, `seed`, optional `mutation_fn` / `mutation_rate`)
and `run(n_generations, verbose, initial_population, generation_callback)`
returns a dict with `pareto_solutions`, `pareto_objectives`,
`final_population` and `final_objectives`.

Supporting utilities: `weighted_sum`, `tchebycheff`, `dominates`,
`pareto_front`, `ParetoArchive`.

---

## Problems

| Class | Problem | Generators | Instance reader |
|---|---|---|---|
| `TSP`    | Travelling Salesman | `create_random_tsp`, `create_tsp_from_coordinates` | `parse_tsplib` |
| `PFSP`   | Permutation Flowshop Scheduling (makespan / flowtime) | `create_random_pfsp` | `parse_taillard_pfsp`, `load_taillard_instance` |
| `LOP`    | Linear Ordering | `create_random_lop`, `create_tournament_lop`, `create_triangular_lop`, `create_sparse_lop`, `feedback_arc_set_to_lop` | `parse_lolib`, `load_lolib_instance` |
| `QAP`    | Quadratic Assignment | `create_random_qap`, `create_uniform_qap`, `create_grid_qap` | `parse_qaplib`, `load_qaplib_instance` |
| `MIS`    | Maximum Independent Set | `create_random_mis`, `create_mis_from_edges` | — |
| `MaxCut` | Maximum Cut | `create_random_max_cut`, `create_max_cut_from_edges` | — |
| `MVC`    | Minimum Vertex Cover | `create_random_mvc`, `create_mvc_from_edges` | — |

MIS, MaxCut and MVC use the *permutation picture* formulation of graph
combinatorial problems (Min, 2024), in which a permutation of the vertices plus a
cut point encodes a vertex subset.

```python
from perm_pateda import parse_qaplib, GMallowsCayleyEDA
from perm_pateda.functions import load_qaplib_instance

flow, dist = parse_qaplib("instances/tai40b.dat")
qap = load_qaplib_instance(flow, dist)
alg = GMallowsCayleyEDA(n_vars=qap.n, fitness_func=qap,
                        pop_size=10 * qap.n, n_gen=500,
                        selection_ratio=0.1, random_seed=111)
stats, _ = alg.run()
print("Best cost:", -stats.best_fitness_overall)
```

---

## Statistical comparison

`perm_pateda.utils.stats_utils` (re-exported at the top level) implements the
comparison protocol used in the permutation EDA literature, taking a `pandas`
DataFrame whose rows are instances and whose columns are algorithms:

`summary_table`, `friedman_test`, `wilcoxon_pairwise`, `critical_difference_plot`.

---

## Package layout

```
perm_pateda/
├── distances.py          # Kendall, Cayley, Ulam, Hamming (+ V and X decompositions,
│                         #   derangement numbers)
├── consensus.py          # find_consensus_{borda,median,best}, get_consensus dispatcher
├── random_keys.py        # permutation <-> random keys <-> ranks, rank rescaling
├── representations/      # Lehmer, Fisher-Yates and insertion-vector codings
├── learning/
│   ├── mallows.py            # Mallows & Generalized Mallows (Kendall, Cayley, Ulam)
│   ├── hamming_kmm.py        # Kernels of Mallows models (Hamming)
│   ├── histogram.py          # LearnEHM, LearnNHM
│   ├── plackett_luce.py      # MM algorithm
│   ├── mixture_plackett_luce.py  # spectral EM + weighted LSR
│   ├── dsm.py                # doubly stochastic matrices
│   ├── umda.py, tree.py      # generic models over a bijective coding
│   └── lehmer.py, fisher_yates.py, vinsertion.py
├── sampling/             # one sampler per learner (same module names)
├── seeding/
│   └── permutation_init.py   # PermutationInit (uniform random permutations)
├── functions/            # tsp, pfsp, lop, qap, mis, maxcut, mvc
├── multiobjective/       # scalarization, pareto, meda_d_mk, meda_d_pl, meda_d_hm
├── utils/
│   ├── benchmark_parsers.py  # TSPLIB, Taillard, LOLIB, QAPLIB
│   └── stats_utils.py        # Friedman, Wilcoxon, critical-difference plots
└── algorithms/
    ├── permutation.py    # plug-and-play EDA wrappers
    ├── random_keys.py    # random-key EDA wrappers
    └── base.py           # adapters to the pateda component interfaces
```

There are no `selection/`, `replacement/`, `stop_conditions/`, `statistics/` or
`knowledge_extraction/` modules: those components are representation agnostic and
are imported directly from `pateda`.

---

## Using the components directly

Because the learners and samplers implement the `pateda` component interfaces
(directly, or through the adapters in `algorithms/base.py`), a permutation EDA
can also be assembled component by component:

```python
import numpy as np
from pateda.core.eda import EDA
from pateda.core.components import EDAComponents
from pateda.selection import TruncationSelection
from pateda.replacement import ElitistReplacement
from pateda.stop_conditions import MaxGenerations

from perm_pateda.seeding import PermutationInit
from perm_pateda.learning.histogram import LearnEHM
from perm_pateda.sampling.histogram import SampleEHM
from perm_pateda.functions import create_random_tsp

n_cities, pop_size, n_gen = 15, 80, 40
tsp = create_random_tsp(n_cities, seed=42)

components = EDAComponents(
    seeding=PermutationInit(),
    selection=TruncationSelection(ratio=0.5),
    learning=LearnEHM(),
    sampling=SampleEHM(),
    replacement=ElitistReplacement(n_elite=int(pop_size * 0.1)),
    stop_condition=MaxGenerations(n_gen),
)
components.learning_params = {"symmetric": True, "beta_ratio": 0.01}
components.sampling_params = {"sample_size": pop_size}

eda = EDA(n_vars=n_cities,
          cardinality=np.full(n_cities, n_cities, dtype=int),
          fitness_func=tsp, pop_size=pop_size,
          components=components, random_seed=42)
stats, cache = eda.run()
print("Best tour length:", -stats.best_fitness_overall)
```

---

## Examples and tests

```bash
python3 examples/ehm_tsp_example.py
python3 examples/mallows_tsp_example.py

pytest tests/
```

---

## Documentation

* [`paper/perm_pateda_user_guide.tex`](paper/perm_pateda_user_guide.tex) — full
  user guide: models, learning and sampling algorithms, components, problems,
  multi-objective framework, and the correspondence with `perm_mateda`.
* [`ROADMAP.md`](ROADMAP.md) — development status and planned work.
* `docs/` — supplementary literature on permutation representations, distances,
  distributions over permutations and permutation optimization.

---

## Citation

If you use the distance-based permutation EDAs implemented here, please cite the
original toolbox:

> E. Irurozki, J. Ceberio, J. Santamaria, R. Santana and A. Mendiburu (2018).
> *Algorithm 989: perm_mateda — A Matlab Toolbox of Estimation of Distribution
> Algorithms for Permutation-based Combinatorial Optimization Problems.*
> ACM Transactions on Mathematical Software, 44(4), Article 47.

References for each individual model are given in the docstring of its
implementation and in the user guide bibliography (`paper/perm_pateda.bib`).

## License

MIT — see [LICENSE](LICENSE).
