import numpy as np
cimport numpy as np

from perm_pateda.algorithms.permutation import _PermEDA
from perm_pateda._cython.learning.mallows import LearnMallowsKendall, LearnGeneralizedMallowsKendall
from perm_pateda._cython.sampling.mallows import SampleMallowsKendall, SampleGeneralizedMallowsKendall
from perm_pateda._cython.learning.plackett_luce import LearnPlackettLuce 
from perm_pateda._cython.sampling.plackett_luce import SamplePlackettLuce
from perm_pateda._cython.learning.lehmer import LearnLehmerUMDA
from perm_pateda._cython.sampling.lehmer import SampleLehmerUMDA



class MallowsKendallEDA(_PermEDA):
    """
    Versión Cython de MallowsKendallEDA.
    Usa el learner y sampler implementados en Cython.
    """

    def __init__(self,
                 int n_vars,
                 object fitness_func,
                 int pop_size=100,
                 int n_gen=50,
                 double selection_ratio=0.5,
                 bint elitism=True,
                 object random_seed=None,
                 str consensus_method="borda",
                 double initial_theta=0.1,
                 double upper_theta=10.0,
                 int max_iter=100,
                 object representation=None):

        super().__init__(
            n_vars=n_vars,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            random_seed=random_seed,
            representation=representation,
        )

        self._learner = LearnMallowsKendall()
        self._sampler = SampleMallowsKendall()
        self._learn_params = {
            "consensus_method": consensus_method,
            "initial_theta": initial_theta,
            "upper_theta": upper_theta,
            "max_iter": max_iter,
        }

class GMallowsKendallEDA(_PermEDA):
    """Versión Cython de GMallowsKendallEDA"""

    def __init__(self,
                 int n_vars,
                 object fitness_func,
                 int pop_size=100,
                 int n_gen=50,
                 double selection_ratio=0.5,
                 bint elitism=True,
                 object random_seed=None,
                 str consensus_method="borda",
                 double initial_theta=0.1,
                 double upper_theta=10.0,
                 int max_iter=100,
                 object representation=None):

        super().__init__(
            n_vars=n_vars,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            random_seed=random_seed,
            representation=representation,
        )

        self._learner = LearnGeneralizedMallowsKendall()
        self._sampler = SampleGeneralizedMallowsKendall()
        self._learn_params = {
            "consensus_method": consensus_method,
            "initial_theta": initial_theta,
            "upper_theta": upper_theta,
            "max_iter": max_iter,
        }

class PlackettLuceEDA(_PermEDA):
    """
    Versión Cython de PlackettLuceEDA.
    Usa el learner y sampler implementados en Cython.
    """

    def __init__(self,
                 int n_vars,
                 object fitness_func,
                 int pop_size=100,
                 int n_gen=50,
                 double selection_ratio=0.5,
                 bint elitism=True,
                 object random_seed=None,
                 int max_iter=100,
                 double tol=1e-6,
                 object representation=None):

        super().__init__(
            n_vars=n_vars,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            random_seed=random_seed,
            representation=representation,
        )

        self._learner = LearnPlackettLuce()
        self._sampler = SamplePlackettLuce()
        self._learn_params = {
            "max_iter": max_iter,
            "tol": tol,
        }

class LehmerUmdaEDA(_PermEDA):
    """Versión Cython de LehmerUmdaEDA"""

    def __init__(self,
                 int n_vars,
                 object fitness_func,
                 int pop_size=100,
                 int n_gen=50,
                 double selection_ratio=0.5,
                 bint elitism=True,
                 object random_seed=None,
                 double laplace_smoothing=0.01,
                 object representation=None):

        super().__init__(
            n_vars=n_vars,
            fitness_func=fitness_func,
            pop_size=pop_size,
            n_gen=n_gen,
            selection_ratio=selection_ratio,
            elitism=elitism,
            random_seed=random_seed,
            representation=representation,
        )

        self._learner = LearnLehmerUMDA(laplace_smoothing=laplace_smoothing)
        self._sampler = SampleLehmerUMDA()