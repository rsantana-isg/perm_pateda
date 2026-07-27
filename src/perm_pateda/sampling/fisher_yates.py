import numpy as np
from typing import Dict, Any, Optional

from perm_pateda.representations.fisher_yates import FisherYatesRepresentation
from perm_pateda.sampling.markov import SampleMarkov


class SampleFisherYatesUMDA:

    def __init__(self):
        self._repr = FisherYatesRepresentation()

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
        cyclic = kwargs.get("cyclic", False)

        return self.__call__(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            sample_size=sample_size,
            rng=rng,
            cyclic=cyclic,
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
        cyclic: bool = False,
    ) -> np.ndarray:
        
        if rng is None:
            rng = np.random.default_rng()

        marginals = model["marginals"]
        domain_sizes = model["domain_sizes"]
        n = model["n_vars"]

        fy_codes = np.zeros((sample_size, n), dtype=int)

        for i in range(n):
            probs = marginals[i].copy()
            domain_size = domain_sizes[i]

            if cyclic and domain_size > 1:
                
                probs[0] = 0.0
                total = probs.sum()
                if total > 0:
                    probs = probs / total
                else:
                    
                    probs = np.zeros(domain_size)
                    probs[1:] = 1.0 / (domain_size - 1)

            fy_codes[:, i] = rng.choice(domain_size, size=sample_size, p=probs)

        perms = self._repr.decode(fy_codes)

        return perms


def sample_fisher_yates_umda(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
    cyclic: bool = False,
) -> np.ndarray:
    
    sampler = SampleFisherYatesUMDA()
    return sampler(
        n_vars, model, cardinality, population, fitness, sample_size, rng, cyclic
    )



class SampleFisherYatesTree:
 
    def __init__(self):
        self._repr = FisherYatesRepresentation()
 
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
        cyclic = kwargs.get("cyclic", False)
 
        return self.__call__(
            n_vars=n_vars,
            model=model,
            cardinality=cardinality,
            population=population,
            fitness=fitness,
            sample_size=sample_size,
            rng=rng,
            cyclic=cyclic,
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
        cyclic: bool = False,
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
 
        fy_codes = np.zeros((sample_size, n), dtype=int)
 
        for node in tree_order:
            domain_size = domain_sizes[node]
 
            if node == root:
                probs = self._apply_cyclic(marginal_root.copy(), domain_size, cyclic)
                fy_codes[:, node] = rng.choice(domain_size, size=sample_size, p=probs)
            else:
                parent = parents[node]
                cond_table = conditionals[node]
                parent_vals = fy_codes[:, parent]
 
                for vp in range(domain_sizes[parent]):
                    mask = parent_vals == vp
                    count = mask.sum()
                    if count == 0:
                        continue
                    probs = self._apply_cyclic(
                        cond_table[vp].copy(), domain_size, cyclic
                    )
                    fy_codes[mask, node] = rng.choice(domain_size, size=count, p=probs)
 
        perms = self._repr.decode(fy_codes)
 
        return perms
 
    @staticmethod
    def _apply_cyclic(probs: np.ndarray, domain_size: int, cyclic: bool) -> np.ndarray:
        
        if not cyclic or domain_size <= 1:
            return probs
 
        probs[0] = 0.0
        total = probs.sum()
        if total > 0:
            return probs / total
        else:
            probs = np.zeros(domain_size)
            probs[1:] = 1.0 / (domain_size - 1)
            return probs
 
 
def sample_fisher_yates_tree(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
    cyclic: bool = False,
) -> np.ndarray:
    
    sampler = SampleFisherYatesTree()
    return sampler(
        n_vars, model, cardinality, population, fitness, sample_size, rng, cyclic
    )


class SampleFisherYatesMarkov(SampleMarkov):
    """Sample a first-order Markov chain over the Fisher-Yates draws."""
    def __init__(self):
        super().__init__(FisherYatesRepresentation())


def sample_fisher_yates_markov(
    n_vars: int,
    model: Dict[str, Any],
    cardinality: np.ndarray,
    population: np.ndarray,
    fitness: np.ndarray,
    sample_size: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    sampler = SampleFisherYatesMarkov()
    return sampler(n_vars, model, cardinality, population, fitness, sample_size, rng)
 
