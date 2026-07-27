# perm_pateda examples

Runnable scripts covering the library. Run any of them with, e.g.:

```bash
python3 examples/verify_distances.py
```

The `verify_*` scripts are **self-checking**: they print `PASS`/`FAIL` per check
and exit with a non-zero status on failure, so they double as lightweight
regression guards (they can be wired into CI).

## Verification guards

| Script | What it checks |
|---|---|
| `verify_distances.py` | Kendall/Cayley/Ulam/Hamming against the JSS worked examples, right-invariance, and `cayley_distance == n - #cycles`. |
| `verify_decomposition_vectors.py` | `sum(V)=d_k`, `sum(X)=d_c`, V ranges, and V/X ↔ permutation round-trips. |
| `verify_reproducibility.py` | Every EDA returns identical results under a fixed `random_seed` (RK-CopulaVines reported as a known pateda-side caveat). |
| `verify_mallows_roundtrip.py` | Each Mallows/GM model reproduces the selected mean distance-to-consensus, and θ grows with concentration. |

## Model-family worked examples

| Script | Contents |
|---|---|
| `mallows_all_distances_lop.py` | The five distance-based EDAs on one LOP, sweeping `consensus_method ∈ {borda, setmedian, best}`. |
| `hamming_kmm_qap.py` | Kernels of Mallows (Hamming) on QAP; plots the E[K]/θ schedule. |
| `plackett_luce_and_mixture.py` | PL and mixture-PL EDAs, with weight/mixing-proportion inspection. |
| `dsm_ps_vs_as.py` | DSM-PS vs DSM-AS on QAP; verifies the learned matrix is doubly stochastic. |
| `bijective_codings.py` | Lehmer/Fisher–Yates/insertion UMDA/Tree/Markov EDAs on PFSP; validity + `decode(encode)` round-trips. |
| `random_key_edas.py` | The three RK-EDAs on TSP, showing the `diminishing`/`cooling` flags. |

## Problems, multi-objective, tooling

| Script | Contents |
|---|---|
| `problems_smoke.py` | All seven problems + in-memory readers; maximization convention; MIS/MaxCut/MVC on hand-built instances with known optima. |
| `meda_d_biobjective.py` | The nine MEDA/D models on a QAP+LOP bi-objective problem (front size + hypervolume), plus a mixed minimise/maximise case. |
| `statistical_comparison.py` | `summary_table` / `friedman_test` / `wilcoxon_pairwise` / `critical_difference_plot` on a small study (long **and** wide input). |
| `component_path_vs_wrapper.py` | Reproduces user-guide §3.3: the same EDA via `EDAComponents`+`pateda.EDA` and via the wrapper. |
| `model_inspection.py` | Learns one model per family and prints the key quantities (consensus, θ, matrices, weights). |

## Pre-existing examples

`ehm_tsp_example.py` and `mallows_tsp_example.py` show the two ways of assembling
a permutation EDA (component path and plug-and-play wrapper).

Scripts that produce figures (`hamming_kmm_qap.py`, `statistical_comparison.py`)
save them to the system temp directory and print the path; they degrade
gracefully if matplotlib is unavailable.
