import numpy as np
from typing import Dict, Any

from perm_pateda.representations.vinsertion import InsertionVectorRepresentation


class LearnInsertionVectorUMDA:


    def __init__(self):
        self._repr = InsertionVectorRepresentation()

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

        iv_codes = self._repr.encode(selected_pop)  # shape (m, n)

        domain_sizes = np.array([i + 1 for i in range(n)], dtype=int)

        marginals = []
        for i in range(n):
            domain_size = domain_sizes[i]  

            counts = np.zeros(domain_size, dtype=float)
            for v in iv_codes[:, i]:
                counts[int(v)] += 1.0

            
            counts += laplace_alpha
            marginals.append(counts / counts.sum())

        return {
            "marginals": marginals,
            "domain_sizes": domain_sizes,
            "n_vars": n,
            "model_type": "insertion_vector_umda",
        }


def learn_insertion_vector_umda(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    
    learner = LearnInsertionVectorUMDA()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )

class LearnInsertionVectorChain:

    def __init__(self):
        self._repr = InsertionVectorRepresentation()
 
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
 
        iv_codes = self._repr.encode(selected_pop)  # shape (m, n)
 
        domain_sizes = np.array([i + 1 for i in range(n)], dtype=int)
 
        marginal_0 = np.array([1.0])
 
        conditionals = {}
        for i in range(1, n):
            prev_domain = domain_sizes[i - 1]  
            curr_domain = domain_sizes[i]       
 
            counts = np.full((prev_domain, curr_domain), laplace_alpha, dtype=float)
 
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

    learner = LearnInsertionVectorChain()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )
