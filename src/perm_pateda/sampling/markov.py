import numpy as np
from typing import Dict, Any, Optional


class SampleMarkov:
    """Sampler for a first-order Markov chain over a bijective coding.

    Generic counterpart of :class:`~perm_pateda.learning.markov.LearnMarkov`.
    The first code position is drawn from the learned marginal ``marginal_0``,
    each subsequent position from the conditional table given the previous one,
    and the resulting code is decoded into a permutation by the representation.
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

        marginal_0 = model["marginal_0"]
        conditionals = model["conditionals"]
        domain_sizes = model["domain_sizes"]
        n = model["n_vars"]

        codes = np.zeros((sample_size, n), dtype=int)
        codes[:, 0] = rng.choice(len(marginal_0), size=sample_size, p=marginal_0)

        for i in range(1, n):
            cond_table = conditionals[i]
            curr_domain = domain_sizes[i]
            prev_vals = codes[:, i - 1]
            for v_prev in range(domain_sizes[i - 1]):
                mask = prev_vals == v_prev
                count = int(mask.sum())
                if count == 0:
                    continue
                codes[mask, i] = rng.choice(curr_domain, size=count, p=cond_table[v_prev])

        return self._repr.decode(codes)
