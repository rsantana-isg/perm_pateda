import numpy as np
from typing import Dict, Any, Optional

from perm_pateda.representations.vinsertion import InsertionVectorRepresentation
from perm_pateda.sampling.tree import SampleTree


class SampleInsertionVectorUMDA:

    def __init__(self):
        self._repr = InsertionVectorRepresentation()

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray = None,
        fitness: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        if population is None:
            population = np.array([])
        if fitness is None:
            fitness = np.array([])

        sample_size = kwargs.get("sample_size", 100)
        rng = kwargs.get("rng", None)

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
        
        if rng is None:
            rng = np.random.default_rng()

        marginals = model["marginals"]
        domain_sizes = model["domain_sizes"]
        n = model["n_vars"]

        iv_codes = np.zeros((sample_size, n), dtype=int)

        for i in range(n):
            probs = marginals[i]
            domain_size = domain_sizes[i]  # = i + 1
            iv_codes[:, i] = rng.choice(domain_size, size=sample_size, p=probs)

        perms = self._repr.decode(iv_codes)

        return perms


def sample_insertion_vector_umda(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    sampler = SampleInsertionVectorUMDA()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng)

class SampleInsertionVectorChain:

    def __init__(self):
        self._repr = InsertionVectorRepresentation()
 
    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray = None,
        fitness: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
        if population is None:
            population = np.array([])
        if fitness is None:
            fitness = np.array([])
 
        sample_size = kwargs.get("sample_size", 100)
        rng = kwargs.get("rng", None)
 
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

        if rng is None:
            rng = np.random.default_rng()
 
        conditionals = model["conditionals"]
        domain_sizes = model["domain_sizes"]
        n = model["n_vars"]
 
        iv_codes = np.zeros((sample_size, n), dtype=int)
 
        iv_codes[:, 0] = 0
 
        for i in range(1, n):
            cond_table = conditionals[i]    
            curr_domain = domain_sizes[i]   
            prev_domain = domain_sizes[i-1] 
            prev_vals = iv_codes[:, i - 1]
 
            for v_prev in range(prev_domain):
                mask = prev_vals == v_prev
                count = mask.sum()
                if count == 0:
                    continue
                probs = cond_table[v_prev]  # shape (i+1,)
                iv_codes[mask, i] = rng.choice(curr_domain, size=count, p=probs)
 
        perms = self._repr.decode(iv_codes)
 
        return perms
 
 
def sample_insertion_vector_chain(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    sampler = SampleInsertionVectorChain()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng)


class SampleInsertionVectorTree(SampleTree):
    """Sample a Chow-Liu tree over the insertion-vector coding."""
    def __init__(self):
        super().__init__(InsertionVectorRepresentation())


def sample_insertion_vector_tree(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    sampler = SampleInsertionVectorTree()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng)
