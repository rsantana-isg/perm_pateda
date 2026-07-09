"""
Doubly Stochastic Matrix (DSM) Learning for Permutation-based EDAs

This module implements the smoothed learning scheme that estimates a DSM
from a set of permutations.  The DSM is a convex combination of the
permutation matrices corresponding to the selected individuals, plus a
uniform DSM scaled by a smoothing factor α:

    D = (1 - α)/m * (P₁ + … + Pₘ) + α * U

where U[i,j] = 1/n for all i,j (the uniform DSM).  The smoothing
factor α prevents zero entries, which would make sampling impossible.

References:
    [1] V. Santucci, J. Ceberio: Doubly Stochastic Matrix Models for
        Estimation of Distribution Algorithms. arXiv:2304.02458, 2023.
"""

import numpy as np
from typing import Dict, Any, Optional


class LearnDSM:
    """Learn a Doubly Stochastic Matrix from a population of permutations.

    Uses the smoothed learning scheme of [1] (Sect. 4.1): the selected
    permutations are converted to permutation matrices, averaged with
    equal weights, and mixed with the uniform DSM to prevent zero
    entries.
    """

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Entry point to match EDA interface.  Calls ``__call__`` internally."""
        return self.__call__(
            generation=generation,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_pop=population,
            selected_fitness=fitness,
            **kwargs,
        )

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Learn a Doubly Stochastic Matrix from the selected population.

        The learning equation is (Eq. 3 of [1]):

            D = (1-α)/m * Σᵢ Pᵢ + α * U

        where Pᵢ is the permutation matrix of the i-th selected
        permutation, U = (1/n) * **1** is the uniform DSM, m is the
        number of selected permutations, and α ∈ (0, 1] is the
        smoothing factor.

        Args:
            generation:      Current generation number (unused, for interface
                             compatibility).
            n_vars:          Permutation length n.
            cardinality:     Not used for permutations.
            selected_pop:    Selected population of permutations,
                             shape (m, n_vars).  May be 0- or 1-indexed.
            selected_fitness: Fitness values (unused).
            alpha:           Smoothing factor.  Defaults to 1/n_vars² as
                             suggested in [1].

        Returns:
            Model dictionary containing:
                - dsm:        Learned DSM, shape (n_vars, n_vars).
                - alpha:      Smoothing factor used.
                - model_type: ``"dsm"``.
        """
        n = n_vars
        if alpha is None:
            alpha = 1.0 / (n ** 2)

        pop = np.asarray(selected_pop, dtype=int)
        m = pop.shape[0]

        # Normalise to 0-indexed items
        if pop.min() == 1:
            pop = pop - 1

        # Weighted sum of permutation matrices
        D = np.zeros((n, n), dtype=float)
        w_perm = (1.0 - alpha) / m
        for perm in pop:
            D[np.arange(n), perm] += w_perm

        # Add uniform DSM scaled by alpha
        D += alpha / n

        return {
            "dsm": D,
            "alpha": alpha,
            "model_type": "dsm",
        }


def learn_dsm(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    """
    Convenience function to learn a Doubly Stochastic Matrix.

    See :class:`LearnDSM` for parameter details.
    """
    learner = LearnDSM()
    return learner(generation, n_vars, cardinality, selected_pop, selected_fitness, **params)
