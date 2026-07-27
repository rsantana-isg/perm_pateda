# sampling/hamming_kmm.py

import numpy as np
from scipy.special import comb
from typing import Dict, Any
from perm_pateda.distances import hamming_distance, compute_derangements


def _sample_at_hamming_distance(center: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a permutation at exactly Hamming distance k from `center`.
    Choose k positions and derange them (none stays in its original place).
    """
    n = len(center)
    result = center.copy()

    # Choose k positions to permute
    positions = rng.choice(n, size=k, replace=False)
    items = result[positions].copy()

    # Derange the items: none may stay in its original position
    # Fisher-Yates with rejection until a valid derangement is obtained
    # (a random shuffle is a derangement with probability ~1/e, so 1000
    # attempts make failure practically impossible)
    for _ in range(1000):
        rng.shuffle(items)
        if all(items[i] != result[positions[i]] for i in range(k)):
            break

    result[positions] = items
    return result


class SampleHammingKMM:
    """
    Sampler for the Kernels of Mallows Model under the Hamming distance.

    Algorithm (Algorithm 2 of the paper):
      1. Choose a centre sigma_0 from the set uniformly at random
      2. Compute p(k) proportional to S(n,k) * e^(-theta*k), k in {0} u [n]\{1}
      3. Sample k according to those probabilities
      4. Build sigma at Hamming distance k from sigma_0
    """

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:

        centers = model["centers"]
        theta = model["theta"]
        n = n_vars

        # Compute S(n,k) = C(n,k) * D(k)
        derangements = compute_derangements(n)
        s_nk = np.array([
            int(comb(n, k, exact=True)) * int(derangements[k])
            for k in range(n + 1)
        ], dtype=float)

        # p(k) proportional to S(n,k) * e^(-theta*k); k=1 impossible under Hamming
        log_weights = np.log(s_nk + 1e-300) - theta * np.arange(n + 1)
        log_weights[1] = -np.inf
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        weights /= weights.sum()

        new_pop = np.empty((sample_size, n), dtype=int)

        for i in range(sample_size):
            # Step 1: choose a random centre
            center = centers[rng.integers(len(centers))]

            # Steps 2-3: sample distance k
            k = int(rng.choice(n + 1, p=weights))

            # Step 4: build a permutation at distance k
            if k == 0:
                new_pop[i] = center.copy()
            else:
                new_pop[i] = _sample_at_hamming_distance(center, k, rng)

        return new_pop