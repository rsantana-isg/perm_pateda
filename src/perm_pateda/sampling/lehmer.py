"""
LehmerUMDA, LehmerTree — Sampling

"""

import numpy as np
from typing import Dict, Any, Optional

from perm_pateda.representations.lehmer import LehmerRepresentation
from perm_pateda.sampling.markov import SampleMarkov


class SampleLehmerUMDA:

    def __init__(self):
        self._repr = LehmerRepresentation(left=False)  # right-Lehmer

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
        n = model["n_vars"]

        lehmer_codes = np.zeros((sample_size, n), dtype=int)
        for i in range(n):
            probs = marginals[i]
            domain_size = len(probs)
            lehmer_codes[:, i] = rng.choice(domain_size, size=sample_size, p=probs)

        perms = self._repr.decode(lehmer_codes)

        return perms


def sample_lehmer_umda(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    
    sampler = SampleLehmerUMDA()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng)


class SampleLehmerTree:
 
    def __init__(self):
        self._repr = LehmerRepresentation(left=False)  # right-Lehmer
 
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
 
        marginal_root = model["marginal_root"]
        conditionals = model["conditionals"]
        parents = model["parents"]
        tree_order = model["tree_order"]
        domain_sizes = model["domain_sizes"]
        root = model["root"]
        n = model["n_vars"]
 
        lehmer_codes = np.zeros((sample_size, n), dtype=int)
 
        for node in tree_order:
            if node == root:
                probs = marginal_root
                lehmer_codes[:, node] = rng.choice(
                    len(probs), size=sample_size, p=probs
                )
            else:
                parent = parents[node]
                cond_table = conditionals[node]
                child_domain = domain_sizes[node]
 
                parent_vals = lehmer_codes[:, parent]
 
                for vp in range(domain_sizes[parent]):
                    mask = parent_vals == vp
                    count = mask.sum()
                    if count == 0:
                        continue
                    probs = cond_table[vp]
                    lehmer_codes[mask, node] = rng.choice(
                        child_domain, size=count, p=probs
                    )
 
        perms = self._repr.decode(lehmer_codes)
 
        return perms
 
 
def sample_lehmer_tree(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:

    sampler = SampleLehmerTree()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng)


class SampleLehmerMarkov(SampleMarkov):
    """Sample a first-order Markov chain over the right-Lehmer code."""
    def __init__(self):
        super().__init__(LehmerRepresentation(left=False))


def sample_lehmer_markov(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    sampler = SampleLehmerMarkov()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng)
