import numpy as np
from typing import Dict, Any, Optional


class SampleTree:
    """Sampler for a Chow-Liu tree model over a bijective coding.

    Generic counterpart of :class:`~perm_pateda.learning.tree.LearnTree`: it
    samples the root code position from its marginal and every other position
    from its conditional table given its parent (following ``tree_order``), then
    decodes the resulting code into a permutation.
    """

    def __init__(self, representation):
        self._repr = representation

    def sample(
        self,
        n_vars: int,
        model: Dict[str, Any],
        cardinality: np.ndarray,
        population: np.ndarray = None,
        fitness: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:
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

        codes = np.zeros((sample_size, n), dtype=int)

        for node in tree_order:
            if node == root:
                probs = marginal_root
                codes[:, node] = rng.choice(len(probs), size=sample_size, p=probs)
            else:
                parent = parents[node]
                cond_table = conditionals[node]
                child_domain = domain_sizes[node]
                parent_vals = codes[:, parent]
                for vp in range(domain_sizes[parent]):
                    mask = parent_vals == vp
                    count = int(mask.sum())
                    if count == 0:
                        continue
                    codes[mask, node] = rng.choice(child_domain, size=count, p=cond_table[vp])

        return self._repr.decode(codes)
