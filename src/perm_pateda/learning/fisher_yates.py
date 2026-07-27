from typing import Dict, Any
import numpy as np

from perm_pateda.representations.fisher_yates import FisherYatesRepresentation
from perm_pateda.learning.umda import LearnUMDA
from perm_pateda.learning.tree import LearnTree
from perm_pateda.learning.markov import LearnMarkov


class LearnFisherYatesUMDA:
    """
    Aprende un modelo UMDA sobre la secuencia de extracciones de Fisher-Yates.
    """
    def __init__(self, laplace_smoothing: float = 0.01, **kwargs):
        self.rep = FisherYatesRepresentation()
        self.laplace_smoothing = laplace_smoothing
        self.learner = LearnUMDA(representation=self.rep, model_type="fisher_yates_umda")

    def learn(self, generation: int, n_vars: int, cardinality: np.ndarray, population: np.ndarray, fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.__call__(generation, n_vars, cardinality, population, fitness, **kwargs)

    def __call__(self, generation: int, n_vars: int, cardinality: np.ndarray, selected_pop: np.ndarray, selected_fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.learner(
            generation, n_vars, cardinality, selected_pop, selected_fitness, 
            laplace_alpha=self.laplace_smoothing, **kwargs
        )

class LearnFisherYatesTree:
    def __init__(self, laplace_smoothing: float = 0.01, root: int = 0, **kwargs):
        self.rep = FisherYatesRepresentation()
        self.laplace_smoothing = laplace_smoothing
        self.root = root
        self.learner = LearnTree(
            representation=self.rep, 
            model_type="fisher_yates_tree"
        )

    def learn(self, generation: int, n_vars: int, cardinality: np.ndarray, population: np.ndarray, fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.__call__(generation, n_vars, cardinality, population, fitness, **kwargs)

    def __call__(self, generation: int, n_vars: int, cardinality: np.ndarray, selected_pop: np.ndarray, selected_fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.learner(
            generation, n_vars, cardinality, selected_pop, selected_fitness, 
            laplace_alpha=self.laplace_smoothing, root=self.root, **kwargs
        )


class LearnFisherYatesMarkov:
    def __init__(self, laplace_smoothing: float = 0.01, **kwargs):
        self.rep = FisherYatesRepresentation()
        self.laplace_smoothing = laplace_smoothing
        self.learner = LearnMarkov(representation=self.rep, model_type="fisher_yates_markov")

    def learn(self, generation: int, n_vars: int, cardinality: np.ndarray, population: np.ndarray, fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.__call__(generation, n_vars, cardinality, population, fitness, **kwargs)

    def __call__(self, generation: int, n_vars: int, cardinality: np.ndarray, selected_pop: np.ndarray, selected_fitness: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.learner(
            generation, n_vars, cardinality, selected_pop, selected_fitness,
            laplace_alpha=self.laplace_smoothing, **kwargs
        )


def learn_fisher_yates_umda(*args, **kwargs) -> Dict[str, Any]:
    return LearnFisherYatesUMDA()(*args, **kwargs)


def learn_fisher_yates_tree(*args, **kwargs) -> Dict[str, Any]:
    return LearnFisherYatesTree()(*args, **kwargs)


def learn_fisher_yates_markov(*args, **kwargs) -> Dict[str, Any]:
    return LearnFisherYatesMarkov()(*args, **kwargs)