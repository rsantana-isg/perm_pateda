import numpy as np
from typing import Dict, Any


class LearnUMDA:


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

        marginals = []
        for i in range(n):
            domain_size = domain_sizes[i]
            counts = np.zeros(domain_size, dtype=float)
            for v in codes[:, i]:
                counts[int(v)] += 1.0
            counts += laplace_alpha
            marginals.append(counts / counts.sum())

        return {
            "marginals": marginals,
            "domain_sizes": domain_sizes,
            "n_vars": n,
            "model_type": self._model_type,
        }