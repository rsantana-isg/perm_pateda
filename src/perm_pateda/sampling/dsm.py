"""
Doubly Stochastic Matrix (DSM) Sampling for Permutation-based EDAs

This module implements two sampling strategies from [1] for drawing
permutations from a Doubly Stochastic Matrix (DSM):

DSM-PS (Probabilistic Sampling)
    Iteratively selects a random remaining row of D, draws the column
    assignment from the corresponding (renormalised) probability vector,
    and removes the matched row/column until a complete permutation is
    built.  Complexity: O(n²) per permutation.

DSM-AS (Algebraic Sampling)
    Generates a uniform random vector v, multiplies D by v, and derives
    the permutation from the ranking relationship between v and D·v.
    Complexity: O(n²) per permutation (dominated by the matrix–vector
    product), but benefits from highly optimised BLAS routines in
    practice.

References:
    [1] V. Santucci, J. Ceberio: Doubly Stochastic Matrix Models for
        Estimation of Distribution Algorithms. arXiv:2304.02458, 2023.
"""

import numpy as np
from typing import Dict, Any, Optional


class SampleDSMPS:
    """Sample permutations from a DSM using Probabilistic Sampling (PS).

    The PS algorithm (Sect. 4.2 of [1]) builds a permutation
    sequentially: at each step, a remaining row is selected uniformly at
    random and the column assignment is drawn from the (renormalised)
    row probabilities over the still-unassigned columns.

    Probability of sampling permutation σ under PS [1, Eq. 4]:

        Pr(σ | D) = ∏ᵢ d_{i, σ(i)} / Perm(D)

    where Perm(D) is the permanent of D.
    """

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> np.ndarray:
        """Entry point to match EDA interface.  Calls ``__call__`` internally."""
        return self.__call__(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            sample_size=sample_size,
            rng=rng,
        )

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample permutations from the DSM using Probabilistic Sampling.

        Args:
            n_vars:      Permutation length n.
            model:       Model dictionary produced by :class:`LearnDSM`
                         containing key ``"dsm"`` (shape n × n).
            cardinality: Not used for permutations.
            population:  Not used.
            fitness:     Not used.
            sample_size: Number of permutations to generate.
            rng:         NumPy random generator.  A fresh generator is
                         created if ``None``.

        Returns:
            Integer array of shape (sample_size, n_vars) containing the
            sampled permutations (0-indexed).
        """
        if rng is None:
            rng = np.random.default_rng()

        D = model["dsm"]
        n = n_vars
        new_pop = np.empty((sample_size, n), dtype=int)

        for s in range(sample_size):
            new_pop[s] = self._sample_one(D, n, rng)

        return new_pop

    @staticmethod
    def _sample_one(D: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw one permutation using the PS procedure."""
        remaining_rows = list(range(n))
        remaining_cols = list(range(n))
        sigma = np.empty(n, dtype=int)

        for _ in range(n):
            # Select a row uniformly at random from the remaining rows
            r_idx = int(rng.integers(len(remaining_rows)))
            i = remaining_rows[r_idx]

            # Probabilities over remaining columns (renormalise for safety)
            col_arr = np.array(remaining_cols, dtype=int)
            probs = D[i, col_arr]
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(len(remaining_cols)) / len(remaining_cols)

            # Draw column index
            c_idx = int(rng.choice(len(remaining_cols), p=probs))
            j = remaining_cols[c_idx]

            sigma[i] = j
            remaining_rows.pop(r_idx)
            remaining_cols.pop(c_idx)

        return sigma


class SampleDSMAS:
    """Sample permutations from a DSM using Algebraic Sampling (AS).

    The AS algorithm (Sect. 4.2 of [1]) derives a permutation from the
    relationship between a uniform random vector v and the product D·v:

        1. Draw v ∈ [0, 1]ⁿ uniformly at random.
        2. Compute π = rank(D·v)  [i.e. argsort(argsort(D·v))].
        3. Compute ρ = argsort(v).
        4. Return σ such that σ(i) = ρ(π(i)).

    This is equivalent to finding the permutation matrix P* closest (in
    Frobenius norm) to D·v [1, Eq. 6]:

        P* = argmin_{P ∈ Pₙ} ‖D·v − P·v‖²_F

    Complexity is O(n²) per permutation (matrix–vector product), making
    AS faster in practice than PS due to BLAS acceleration.
    """

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: Optional[np.random.Generator] = None,
        **kwargs,
    ) -> np.ndarray:
        """Entry point to match EDA interface.  Calls ``__call__`` internally."""
        return self.__call__(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            sample_size=sample_size,
            rng=rng,
        )

    def __call__(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        sample_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample permutations from the DSM using Algebraic Sampling.

        Vectorised over the full batch: all ``sample_size`` random
        vectors are generated and processed simultaneously.

        Args:
            n_vars:      Permutation length n.
            model:       Model dictionary produced by :class:`LearnDSM`
                         containing key ``"dsm"`` (shape n × n).
            cardinality: Not used for permutations.
            population:  Not used.
            fitness:     Not used.
            sample_size: Number of permutations to generate.
            rng:         NumPy random generator.  A fresh generator is
                         created if ``None``.

        Returns:
            Integer array of shape (sample_size, n_vars) containing the
            sampled permutations (0-indexed).
        """
        if rng is None:
            rng = np.random.default_rng()

        D = model["dsm"]  # (n, n)
        n = n_vars

        # Draw all random vectors at once: shape (sample_size, n)
        V = rng.uniform(0.0, 1.0, size=(sample_size, n))

        # Compute D·v for every v simultaneously.
        # For row s: (D · V[s])[i] = sum_j D[i,j] * V[s,j] = (V[s] @ D.T)[i]
        # Stacking over all s: DV = V @ D.T, shape (sample_size, n)
        DV = V @ D.T  # DV[s, i] = (D · V[s])[i]

        # ρ[s] = argsort(V[s])  — permutation that sorts v ascending
        rho = np.argsort(V, axis=1)   # (sample_size, n)

        # π[s] = rank(D·v[s]) = argsort(argsort(D·v[s]))
        pi = np.argsort(np.argsort(DV, axis=1), axis=1)  # (sample_size, n)

        # σ[s, i] = ρ[s, π[s, i]]
        # Fancy indexing: for each row s, gather rho[s] at positions pi[s]
        row_idx = np.arange(sample_size)[:, None]
        sigma = rho[row_idx, pi]  # (sample_size, n)

        return sigma.astype(int)


def sample_dsm_ps(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Convenience function to sample from a DSM using Probabilistic Sampling.

    See :class:`SampleDSMPS` for parameter details.
    """
    sampler = SampleDSMPS()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng=rng)


def sample_dsm_as(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Convenience function to sample from a DSM using Algebraic Sampling.

    See :class:`SampleDSMAS` for parameter details.
    """
    sampler = SampleDSMAS()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng=rng)
