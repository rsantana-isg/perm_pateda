import numpy as np
from typing import Dict, Any


class LearnMarkov:
    """First-order Markov chain over a bijective coding.

    Generic learner (analogous to :class:`LearnUMDA` and :class:`LearnTree`): it
    models the code as a first-order chain

        P(c) = P(c_0) * prod_{i>=1} P(c_i | c_{i-1}),

    with Laplace-smoothed tables whose sizes follow the representation's
    per-position domain (``representation.get_domain(n)``).  Unlike the univariate
    UMDA it captures dependencies between consecutive code positions; unlike the
    Chow-Liu tree the dependency structure is fixed (the natural chain over
    positions 0, 1, ..., n-1) rather than learned.
    """

    def __init__(self, representation, model_type: str):
        self._repr = representation
        self._model_type = model_type

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
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
        laplace_alpha: float = 1.0,
    ) -> Dict[str, Any]:
        selected_pop = np.atleast_2d(selected_pop)
        m, n = selected_pop.shape

        codes = self._repr.encode(selected_pop)
        domain_sizes = np.array(self._repr.get_domain(n), dtype=int) + 1

        # Marginal of the first code position P(c_0).
        counts0 = np.full(domain_sizes[0], laplace_alpha, dtype=float)
        for v in codes[:, 0]:
            counts0[int(v)] += 1.0
        marginal_0 = counts0 / counts0.sum()

        # Conditional tables P(c_i | c_{i-1}) for i = 1 .. n-1.
        conditionals = {}
        for i in range(1, n):
            counts = np.full((domain_sizes[i - 1], domain_sizes[i]), laplace_alpha, dtype=float)
            for k in range(m):
                counts[int(codes[k, i - 1]), int(codes[k, i])] += 1.0
            row_sums = counts.sum(axis=1, keepdims=True)
            conditionals[i] = counts / row_sums

        return {
            "marginal_0": marginal_0,
            "conditionals": conditionals,
            "domain_sizes": domain_sizes,
            "n_vars": n,
            "model_type": self._model_type,
        }
