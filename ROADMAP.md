# perm_pateda — Roadmap

The original target of this roadmap was **feature parity with the `perm_mateda`
toolbox** described in:

> E. Irurozki, J. Ceberio, J. Santamaria, R. Santana, A. Mendiburu (2018).
> *Algorithm 989: perm_mateda — A Matlab Toolbox of EDAs for Permutation-based
> Combinatorial Optimization Problems.* ACM TOMS 44(4), Article 47.

**That target has been reached** (Phases 0–4 below), and the library has since
grown well beyond it: seven additional model families, three additional problems,
a decomposition-based multi-objective framework, and reproducibility tooling.
The remaining phases cover consolidation, validation and the outstanding gaps.

Current version: **0.2.0**.

---

## Phase 0 — Distance-based EDAs and the four classical problems (✅ done)

* Distances: `kendall_distance`, `cayley_distance`, `ulam_distance`,
  `hamming_distance`, plus the inversion vector **V**, the cycle-based
  decomposition vector **X**, and derangement numbers.
* Consensus: `find_consensus_borda`, `find_consensus_median` (SetMedian),
  `find_consensus_best` (BestPermutation), and the `get_consensus` dispatcher.
  Every Mallows/GM learner accepts `consensus_method ∈ {"borda", "setmedian",
  "best"}`.
* The five paper EDAs: `MallowsKendallEDA`, `MallowsCayleyEDA`,
  `MallowsUlamEDA`, `GMallowsKendallEDA`, `GMallowsCayleyEDA`
  (GM-Ulam is intentionally excluded — the Ulam distance has no distance
  decomposition vector).
* Problems: TSP, PFSP, LOP, QAP, with random-instance generators.
* Instance readers for the four standard libraries: `parse_tsplib`,
  `parse_taillard_pfsp`, `parse_lolib`, `parse_qaplib`.
* Seeding (`PermutationInit`), plug-and-play wrappers, and `pateda`
  component adapters (`algorithms/base.py`).

**Known deviations from the reference toolbox**

* `SampleMallowsUlam` is a Metropolis–Hastings chain (random-transposition
  proposal, `burn_in` / `step_size` parameters), *not* the exact "distances
  sampler" based on random standard Young tableaux. See Phase 6.
* `LearnMallowsUlam` counts the permutations at each Ulam distance exactly only
  for `n ≤ 8`; for larger `n` it uses a Gaussian approximation centred at
  `n − 2√n`. See Phase 6.

---

## Phase 1 — Additional model families (✅ done)

Models with no counterpart in `perm_mateda`:

* **Kernels of Mallows models under the Hamming distance** (`HammingKMMEDA`):
  one kernel per selected solution, with θ derived from an exponentially
  decaying expected-distance schedule `(expected_dist_start, expected_dist_end,
  gamma)`.
* **Histogram models** (`EHMEDA`, `NHMEDA`): edge and node histograms with a
  `beta_ratio` prior; the EHM supports symmetric and asymmetric variants.
* **Plackett–Luce** (`PlackettLuceEDA`): MM algorithm for the MLE, Gumbel-max
  sampling.
* **Mixture of Plackett–Luce models** (`PlackettLuceMixtureEDA`): spectral
  initialization (pairwise embedding + SVD + k-means) followed by EM with
  weighted Luce Spectral Ranking.
* **Doubly stochastic matrix models** (`DSMPSEDA`, `DSMASEDA`): smoothed convex
  combination of permutation matrices, with probabilistic and algebraic
  samplers.

---

## Phase 2 — Alternative representations (✅ done)

* **Bijective codings** (`representations/`): `LehmerRepresentation`,
  `FisherYatesRepresentation`, `InsertionVectorRepresentation`, all vectorized
  over a population, behind the `PermutationRepresentation` interface.
* **Generic models over a coding**: `LearnUMDA` (independent marginals with
  Laplace smoothing) and `LearnTree` (Chow–Liu tree with a selectable root),
  wired into six wrappers: `LehmerUmdaEDA`, `LehmerTreeEDA`,
  `FisherYatesUmdaEDA`, `FisherYatesTreeEDA`, `InsertionVectorUmdaEDA`,
  `InsertionVectorMarkovEDA` (first-order Markov chain).
* **Random keys** (`random_keys.py`, `algorithms/random_keys.py`): conversions
  between permutations, keys and ranks, plus `RKGaussianUMDAEDA`,
  `RKGaussianFullEDA` and `RKCopulaVinesEDA`, implementing the RK-EDA
  diminishing-variance rescaling and cooling schedule on top of the continuous
  learners of `pateda`.

---

## Phase 3 — Additional problems (✅ done)

* **PFSP** (`functions/pfsp.py`): makespan and total flowtime objectives,
  `create_random_pfsp`, `load_taillard_instance`.
* **Graph problems under the permutation picture** (Min, 2024):
  `MIS` (maximum independent set), `MaxCut` (maximum cut) and
  `MVC` (minimum vertex cover), each with a random generator and an
  edge-list constructor.

---

## Phase 4 — Multi-objective framework and experimental tooling (✅ done)

* **MEDA/D** (`multiobjective/`): MOEA/D-style decomposition combined with
  permutation models, instantiated nine ways — `MEDA_D_MK` (Mallows kernels,
  Cayley), `MEDA_D_KENDALL`, `MEDA_D_ULAM`, `MEDA_D_GMKENDALL`,
  `MEDA_D_GMCAYLEY`, `MEDA_D_PLACKETT_LUCE`,
  `MEDA_D_MIXTURE_PLACKETT_LUCE`, `MEDA_D_NHM`, `MEDA_D_EHM` — all sharing one
  constructor signature, with weight-vector neighbourhoods, normalized
  Tchebycheff / weighted-sum scalarization, an external `ParetoArchive`, a
  shaking mechanism and an optional external mutation hook.
* **Pareto tooling**: `dominates`, `pareto_front`, `ParetoArchive`.
* **Statistical comparison** (`utils/stats_utils.py`): `summary_table`,
  `friedman_test`, `wilcoxon_pairwise`, `critical_difference_plot`.
* **Numerical hardening**: the histogram samplers fall back to a uniform
  distribution over the remaining items when a row has zero mass.

---

## Phase 5 — Documentation (🔶 in progress)

* ✅ Full user guide (`paper/perm_pateda_user_guide.tex`) covering the models,
  their learning and sampling algorithms, the components, the problems, the
  multi-objective framework and the correspondence with `perm_mateda`, with a
  bibliography (`paper/perm_pateda.bib`) of the primary source for each
  implementation.
* ✅ `README.md` reflecting the full API surface.
* ⏳ Sphinx API documentation (the `docs` extra is declared in
  `pyproject.toml` but no Sphinx project exists yet).
* ⏳ One worked example per model family; currently only `examples/ehm_tsp_example.py`
  and `examples/mallows_tsp_example.py` exist.

---

## Phase 6 — Outstanding gaps (⏳ planned)

**Fourier-decomposition models — missing dependency.**
`multiobjective/meda_d_f.py`, `multiobjective/fourier_moead.py` and
`multiobjective/moead.py` import `perm_pateda.fourier`,
`perm_pateda.distributions` and `perm_pateda.utils.permutations`, none of which
exist in this repository — they belong to the companion `permutation_fourier`
project. These three modules are therefore **not importable** and are not
re-exported by `multiobjective/__init__.py`. Either vendor the Fourier package
here, or declare it as an optional dependency and guard the imports.

**Exact Ulam sampling and normalization.**
Replace the Metropolis–Hastings sampler by the exact distances sampler (sample a
target Ulam distance, then a uniform permutation at that distance via random
standard Young tableaux / RSK), and replace the Gaussian approximation of the
per-distance permutation counts by the exact Ferrers/RSK shape counts for
`n > 8`.

**Experimental harness.**
There is no `scripts/` directory yet. Add
`scripts/run_permutation_eda.py` (positional args, seed first),
`scripts/compare_permutation_edas.py` and the matching `slurm/` launcher, so
that a full study over the four benchmark libraries can be reproduced with the
conventions used across the other projects.

**Validation against `perm_mateda`.**
Compare parameter estimates (σ₀, θ, **θ**) and sampled-distance histograms
against the reference outputs of the Matlab toolbox, for each distance.

**Test coverage.**
`tests/` currently contains only `test_mallows_cayley.py` and
`test_generalized_mallows.py`. Every learner/sampler pair needs at least an
encode/decode or learn→sample round-trip test, a validity check (output rows are
permutations), and a convergence smoke test on a small LOP/TSP instance. The
bijective codings in particular should be tested for `decode(encode(σ)) == σ`
over random populations.

**Permutation local search.**
`pateda` local search operators assume a Cartesian discrete representation and
do not apply here. Add swap / insertion / 2-opt neighbourhood operators as
first-class `LocalOptimizationMethod` components, plus permutation crossover
(OX, PMX, CX) for the hybrid algorithms.

**Housekeeping.**
* `perm_pateda/__init__.py`: `__all__` lists `"hamming_distnce"` (typo) while
  the imported name is `hamming_distance`.
* `learning/__init__.py`: `__all__` lists `"learn_mallos_cayley"` (typo), and
  two missing commas silently concatenate entries
  (`"LearnPlackettLuce" "LearnPlackettLuceMixture"`); `sampling/__init__.py` has
  the same two concatenation bugs.
* The module docstring of `perm_pateda/__init__.py` shows
  `fitness_func=lop.evaluate`; `LOP` has no `evaluate` method — the correct
  usage is `fitness_func=lop` (or `lop.evaluate_objective` for the raw value).
* `algorithms/__init__.py` does not re-export `MallowsUlamEDA`,
  `PlackettLuceEDA`, `PlackettLuceMixtureEDA` or `HammingKMMEDA`, although the
  top-level package does; the two lists should be made consistent.
* A stray file `src/perm_pateda/1` is tracked in the source tree.
* Several learner/sampler docstrings and inline comments are in Spanish; unify
  on English.

---

## Summary: coverage of `perm_mateda`

| Component | `perm_mateda` | `perm_pateda` |
|---|---|---|
| MEDA-Kendall   | ✓ | ✅ `MallowsKendallEDA` |
| MEDA-Cayley    | ✓ | ✅ `MallowsCayleyEDA` |
| MEDA-Ulam      | ✓ | ✅ `MallowsUlamEDA` (MCMC sampler, see Phase 6) |
| GMEDA-Kendall  | ✓ | ✅ `GMallowsKendallEDA` |
| GMEDA-Cayley   | ✓ | ✅ `GMallowsCayleyEDA` |
| Borda consensus           | ✓ | ✅ `find_consensus_borda` |
| SetMedian consensus       | ✓ | ✅ `find_consensus_median` |
| BestPermutation consensus | ✓ | ✅ `find_consensus_best` |
| TSP / PFSP / LOP / QAP    | ✓ | ✅ `functions/` |
| TSPLIB / Taillard / LOLIB / QAPLIB readers | ✓ | ✅ `utils/benchmark_parsers.py` |
| Kernels of Mallows models (Hamming) | — | ✅ extra |
| Edge / node histogram EDAs          | — | ✅ extra |
| Plackett–Luce and mixtures          | — | ✅ extra |
| Doubly stochastic matrix models     | — | ✅ extra |
| Bijective codings + UMDA/tree/Markov| — | ✅ extra |
| Random-key EDAs                     | — | ✅ extra |
| MIS / MaxCut / MVC                  | — | ✅ extra |
| Multi-objective MEDA/D (nine models)| — | ✅ extra |
| Statistical comparison utilities    | — | ✅ extra |
| Fourier-decomposition models        | — | ⏳ dependency missing (Phase 6) |
