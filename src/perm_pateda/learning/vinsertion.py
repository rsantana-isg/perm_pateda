import numpy as np
from typing import Dict, Any

from perm_pateda.representations.vinsertion import InsertionVectorRepresentation
from perm_pateda.learning.umda import LearnUMDA


class LearnInsertionVectorUMDA:
    def __init__(self, laplace_smoothing: float = 0.01, **kwargs):
        self.rep = InsertionVectorRepresentation()
        self.laplace_smoothing = laplace_smoothing
        self.learner = LearnUMDA(representation=self.rep, model_type="vinsertion_umda")

    def learn(self, generation: int, n_vars: int, cardinality: np.ndarray, population: np.ndarray, fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.__call__(generation, n_vars, cardinality, population, fitness, **kwargs)

    def __call__(self, generation: int, n_vars: int, cardinality: np.ndarray, selected_pop: np.ndarray, selected_fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.learner(
            generation, n_vars, cardinality, selected_pop, selected_fitness, 
            laplace_alpha=self.laplace_smoothing, **kwargs
        )


def learn_insertion_vector_umda(*args, **kwargs) -> Dict[str, Any]:
    return LearnInsertionVectorUMDA()(*args, **kwargs)

class LearnInsertionVectorChain:
    """
    EDA basado en Cadenas de Markov sobre el Vector de Inserción.
    """
    def __init__(self, laplace_smoothing: float = 0.01, **kwargs):
        self._repr = InsertionVectorRepresentation()
        self.laplace_smoothing = laplace_smoothing
 
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
        laplace_alpha: float = None, # Valor de fallback
        **kwargs,
    ) -> Dict[str, Any]:
        # Usamos laplace_smoothing guardado en el init si laplace_alpha no se ha sobrescrito
        alpha = laplace_alpha if laplace_alpha is not None else self.laplace_smoothing

        selected_pop = np.atleast_2d(selected_pop)
        m, n = selected_pop.shape
 
        iv_codes = self._repr.encode(selected_pop)
        domain_sizes = np.array([i + 1 for i in range(n)], dtype=int)
        marginal_0 = np.array([1.0])
 
        conditionals = {}
        for i in range(1, n):
            prev_domain = domain_sizes[i - 1]  
            curr_domain = domain_sizes[i]       
 
            # Usamos alpha (el suavizado)
            counts = np.full((prev_domain, curr_domain), alpha, dtype=float)
 
            for k in range(m):
                v_prev = int(iv_codes[k, i - 1])
                v_curr = int(iv_codes[k, i])
                counts[v_prev, v_curr] += 1.0
 
            row_sums = counts.sum(axis=1, keepdims=True)
            conditionals[i] = counts / row_sums
 
        return {
            "marginal_0": marginal_0,
            "conditionals": conditionals,
            "domain_sizes": domain_sizes,
            "n_vars": n,
            "model_type": "insertion_vector_chain",
        }
 
 
def learn_insertion_vector_chain(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    # Pasamos params al init si es necesario
    learner = LearnInsertionVectorChain(**params)
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )