"""Random-key EDAs for permutation optimization."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional, Tuple

import numpy as np

from pateda.core.eda import Cache, Statistics

from perm_pateda.random_keys import random_keys_to_permutation, rescale_random_keys


@dataclass
class _RKModelComponents:
    learner: Any
    sampler: Any
    model_name: str


def _load_first(candidates: Tuple[Tuple[str, str], ...]) -> Any:
    for module_name, attr_name in candidates:
        try:
            module = import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError):
            continue
    names = ", ".join(f"{m}.{a}" for m, a in candidates)
    raise ImportError(
        "Could not load any of the required pateda symbols: "
        f"{names}. This usually means the installed 'pateda' version is "
        "incompatible (the symbol was renamed/moved). Check that a compatible "
        "pateda is installed and expose one of the names listed above."
    )


def _instantiate_sampler(sampler_cls: Any, n_samples: int) -> Any:
    for kw in ({"n_samples": n_samples}, {"sample_size": n_samples}, {}):
        try:
            return sampler_cls(**kw)
        except TypeError:
            continue
    return sampler_cls()


def _covariance_to_fixed_variance(covariance: np.ndarray, sigma_g: float) -> np.ndarray:
    """Preserve correlation structure while forcing every diagonal entry to ``sigma_g**2``."""
    cov = np.asarray(covariance, dtype=float)
    std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr * (sigma_g ** 2)


class _CopulaVinesLearner:
    def __init__(self, truncation_level: Optional[int] = None):
        self.truncation_level = truncation_level

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **params: Any,
    ) -> Any:
        learn_auto = _load_first((
            ("pateda.learning.vine_copula", "learn_vine_copula_auto"),
        ))
        model_params = {"truncation_level": self.truncation_level}
        model_params.update(params)
        return learn_auto(population, fitness, model_params)


class _CopulaVinesSampler:
    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def sample(
        self,
        n_vars: int,
        model: Any,
        cardinality: np.ndarray,
        aux_pop: Optional[np.ndarray] = None,
        aux_fitness: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
        **params: Any,
    ) -> np.ndarray:
        sample_vine = _load_first((
            ("pateda.sampling.vine_copula", "sample_vine_copula"),
        ))
        return sample_vine(model, self.n_samples, bounds=cardinality, params=params, rng=rng)


def _gaussian_umda_components(pop_size: int) -> _RKModelComponents:
    learner_cls = _load_first((
        ("pateda.learning", "LearnGaussianUMDA"),
        ("pateda.learning.basic_gaussian", "LearnGaussianUnivariate"),
    ))
    sampler_cls = _load_first((
        ("pateda.sampling", "SampleGaussianUMDA"),
        ("pateda.sampling.basic_gaussian", "SampleGaussianUnivariate"),
    ))
    return _RKModelComponents(learner=learner_cls(), sampler=_instantiate_sampler(sampler_cls, pop_size), model_name="gaussian_umda")


def _gaussian_full_components(pop_size: int) -> _RKModelComponents:
    learner_cls = _load_first((
        ("pateda.learning", "LearnGaussianFull"),
        ("pateda.learning.basic_gaussian", "LearnGaussianFull"),
    ))
    sampler_cls = _load_first((
        ("pateda.sampling", "SampleGaussianFull"),
        ("pateda.sampling.basic_gaussian", "SampleGaussianFull"),
    ))
    return _RKModelComponents(learner=learner_cls(), sampler=_instantiate_sampler(sampler_cls, pop_size), model_name="gaussian_full")


def _copula_components(pop_size: int, truncation_level: Optional[int]) -> _RKModelComponents:
    return _RKModelComponents(
        learner=_CopulaVinesLearner(truncation_level=truncation_level),
        sampler=_CopulaVinesSampler(n_samples=pop_size),
        model_name="copula_vines",
    )


class _RandomKeyEDA:
    """Common RK-EDA loop using random keys as internal representation."""

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        diminishing: bool = True,
        cooling: bool = True,
        initial_sigma: Optional[float] = None,
        min_sigma: float = 1e-6,
    ):
        if n_vars < 1:
            raise ValueError("n_vars must be at least 1")
        self.n_vars = n_vars
        self.fitness_func = fitness_func
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.selection_ratio = selection_ratio
        self.elitism = elitism
        self.diminishing = diminishing
        self.cooling = cooling
        self.min_sigma = min_sigma
        self.rng = np.random.default_rng(random_seed)
        self._bounds = np.vstack([np.zeros(n_vars), np.ones(n_vars)])
        if initial_sigma is not None:
            self.initial_sigma = float(initial_sigma)
        elif n_vars >= 2:
            # RK-EDA paper default: sigma = 1 / (pi * log10(n)); clamp for numerical safety.
            self.initial_sigma = float(1.0 / (np.pi * max(np.log10(n_vars), 1e-6)))
        else:
            self.initial_sigma = 0.1

    def _evaluate(self, random_key_population: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        perm_population = random_keys_to_permutation(random_key_population)
        fitness = np.array([self.fitness_func(perm) for perm in perm_population])
        return perm_population, fitness

    def _learn(self, gen: int, selected_keys: np.ndarray, selected_fitness: np.ndarray) -> Any:
        learner = self._learner  # type: ignore[attr-defined]
        if hasattr(learner, "learn"):
            return learner.learn(
                generation=gen,
                n_vars=self.n_vars,
                cardinality=self._bounds,
                population=selected_keys,
                fitness=selected_fitness,
            )
        return learner(
            generation=gen,
            n_vars=self.n_vars,
            cardinality=self._bounds,
            population=selected_keys,
            fitness=selected_fitness,
        )

    def _sigma_at_generation(self, gen: int) -> float:
        if not self.cooling:
            return max(self.initial_sigma, self.min_sigma)
        cooling_rate = 1.0 - (gen / max(1, self.n_gen))
        return max(self.initial_sigma * max(cooling_rate, 0.0), self.min_sigma)

    def _apply_variance_control(self, model: Any, sigma_g: float) -> Any:
        if not self.cooling:
            return model
        scaled = deepcopy(model)
        if hasattr(scaled, "parameters") and isinstance(scaled.parameters, dict):
            params = scaled.parameters
            if "stds" in params:
                params["stds"] = np.full_like(params["stds"], sigma_g, dtype=float)
            elif "cov" in params:
                params["cov"] = _covariance_to_fixed_variance(params["cov"], sigma_g)
            return scaled
        if isinstance(scaled, dict):
            if "stds" in scaled:
                scaled["stds"] = np.full_like(np.asarray(scaled["stds"], dtype=float), sigma_g)
            elif "cov" in scaled:
                scaled["cov"] = _covariance_to_fixed_variance(scaled["cov"], sigma_g)
        return scaled

    def _postprocess_samples(self, sampled: np.ndarray, sigma_g: float) -> np.ndarray:
        return sampled

    def _sample(
        self,
        model: Any,
        selected_keys: np.ndarray,
        selected_fitness: np.ndarray,
        sigma_g: float,
    ) -> np.ndarray:
        sampler = self._sampler  # type: ignore[attr-defined]
        model_to_sample = self._apply_variance_control(model, sigma_g)
        if hasattr(sampler, "sample"):
            sampled = sampler.sample(
                n_vars=self.n_vars,
                model=model_to_sample,
                cardinality=self._bounds,
                aux_pop=selected_keys,
                aux_fitness=selected_fitness,
                rng=self.rng,
            )
        else:
            sampled = sampler(
                n_vars=self.n_vars,
                model=model_to_sample,
                cardinality=self._bounds,
                aux_pop=selected_keys,
                aux_fitness=selected_fitness,
                rng=self.rng,
            )

        sampled = np.asarray(sampled, dtype=float)
        if sampled.shape != (self.pop_size, self.n_vars):
            sampled = sampled.reshape(self.pop_size, self.n_vars)

        sampled = self._postprocess_samples(sampled, sigma_g)
        return np.clip(sampled, 0.0, 1.0)

    def run(self, verbose: bool = False) -> Tuple[Statistics, Cache]:
        stats = Statistics()
        cache = Cache()
        n_select = max(2, int(self.pop_size * self.selection_ratio))

        rk_population = self.rng.random((self.pop_size, self.n_vars))
        if self.diminishing:
            rk_population = rescale_random_keys(rk_population)

        perm_population, fitness = self._evaluate(rk_population)
        best_idx = int(np.argmax(fitness))
        best_key = rk_population[best_idx].copy()
        best_fit = float(fitness[best_idx])

        for gen in range(self.n_gen):
            stats.update(gen, perm_population, fitness.reshape(-1, 1))
            if verbose:
                print(f"Gen {gen}: best={stats.best_fitness[-1]:.4f} mean={stats.mean_fitness[-1]:.4f}")

            sorted_idx = np.argsort(fitness)[::-1]
            selected_keys = rk_population[sorted_idx[:n_select]]
            selected_fitness = fitness[sorted_idx[:n_select]]

            learning_keys = rescale_random_keys(selected_keys) if self.diminishing else selected_keys
            model = self._learn(gen, learning_keys, selected_fitness)
            sigma_g = self._sigma_at_generation(gen)

            new_rk_population = self._sample(model, learning_keys, selected_fitness, sigma_g)
            if self.diminishing:
                new_rk_population = rescale_random_keys(new_rk_population)
            new_perm_population, new_fitness = self._evaluate(new_rk_population)
            new_best_idx = int(np.argmax(new_fitness))
            new_best_fit = float(new_fitness[new_best_idx])

            if self.elitism:
                if best_fit > new_best_fit:
                    worst_idx = int(np.argmin(new_fitness))
                    new_rk_population[worst_idx] = best_key.copy()
                    new_perm_population[worst_idx] = random_keys_to_permutation(best_key)
                    new_fitness[worst_idx] = best_fit
                else:
                    best_fit = new_best_fit
                    best_key = new_rk_population[new_best_idx].copy()

            rk_population = new_rk_population
            perm_population = new_perm_population
            fitness = new_fitness

        stats.update(self.n_gen, perm_population, fitness.reshape(-1, 1))
        return stats, cache


class RKGaussianUMDAEDA(_RandomKeyEDA):
    """
    RK-EDA variant with GaussianUMDA learning on random keys.

    Reference:
    Ayodele, M., McCall, J., Regnier-Coudert, O. (2016).
    RK-EDA: A Novel Random Key Based Estimation of Distribution Algorithm.
    PPSN XIV, LNCS 9921, 849-858.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        diminishing: bool = True,
        cooling: bool = True,
        initial_sigma: Optional[float] = None,
        min_sigma: float = 1e-6,
    ):
        super().__init__(
            n_vars=n_vars,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            random_seed=random_seed,
            diminishing=diminishing,
            cooling=cooling,
            initial_sigma=initial_sigma,
            min_sigma=min_sigma,
        )
        components = _gaussian_umda_components(pop_size)
        self._learner = components.learner
        self._sampler = components.sampler
        self._model_name = components.model_name


class RKGaussianFullEDA(_RandomKeyEDA):
    """
    RK-EDA variant with full Gaussian learning on random keys.

    Reference:
    Ayodele, M., McCall, J., Regnier-Coudert, O. (2016).
    RK-EDA: A Novel Random Key Based Estimation of Distribution Algorithm.
    PPSN XIV, LNCS 9921, 849-858.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        diminishing: bool = True,
        cooling: bool = True,
        initial_sigma: Optional[float] = None,
        min_sigma: float = 1e-6,
    ):
        super().__init__(
            n_vars=n_vars,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            random_seed=random_seed,
            diminishing=diminishing,
            cooling=cooling,
            initial_sigma=initial_sigma,
            min_sigma=min_sigma,
        )
        components = _gaussian_full_components(pop_size)
        self._learner = components.learner
        self._sampler = components.sampler
        self._model_name = components.model_name


class RKCopulaVinesEDA(_RandomKeyEDA):
    """
    RK-EDA variant with vine-copula learning on random keys.

    Reference:
    Ayodele, M., McCall, J., Regnier-Coudert, O. (2016).
    RK-EDA: A Novel Random Key Based Estimation of Distribution Algorithm.
    PPSN XIV, LNCS 9921, 849-858.
    """

    def __init__(
        self,
        n_vars: int,
        fitness_func: Callable,
        pop_size: int = 100,
        n_gen: int = 50,
        selection_ratio: float = 0.5,
        elitism: bool = True,
        random_seed: Optional[int] = None,
        diminishing: bool = True,
        cooling: bool = True,
        initial_sigma: Optional[float] = None,
        min_sigma: float = 1e-6,
        truncation_level: Optional[int] = None,
    ):
        super().__init__(
            n_vars=n_vars,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            random_seed=random_seed,
            diminishing=diminishing,
            cooling=cooling,
            initial_sigma=initial_sigma,
            min_sigma=min_sigma,
        )
        components = _copula_components(pop_size, truncation_level=truncation_level)
        self._learner = components.learner
        self._sampler = components.sampler
        self._model_name = components.model_name

    def _postprocess_samples(self, sampled: np.ndarray, sigma_g: float) -> np.ndarray:
        if self.cooling:
            sampled = sampled + self.rng.normal(0.0, sigma_g, size=sampled.shape)
        return sampled


__all__ = [
    "RKGaussianUMDAEDA",
    "RKGaussianFullEDA",
    "RKCopulaVinesEDA",
]
