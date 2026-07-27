"""
Mallows Model Learning for Permutation-based EDAs

This module implements learning methods for Mallows models with different
distance metrics (Kendall, Cayley, Ulam).

References:
    [1] C. L. Mallows: Non-null ranking models. Biometrika, 1957
    [2] J. Ceberio, A. Mendiburu, J.A Lozano: Introducing the Mallows Model
        on Estimation of Distribution Algorithms. ICONIP 2011
"""

import numpy as np
from typing import Dict, Any, Callable, Optional
from scipy.optimize import fminbound, newton
from perm_pateda.distances import (
    kendall_distance,
    cayley_distance,
    ulam_distance,
    _x_vector_cycles,
)
from perm_pateda.consensus import compose_permutations, get_consensus


class LearnMallowsKendall:
    """Learn Mallows model with Kendall distance"""

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Learn method to match EDA interface. Calls __call__ internally."""
        return self.__call__(
            generation=generation,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_pop=population,
            selected_fitness=fitness,
            **kwargs
        )

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        initial_theta: float = 0.1,
        upper_theta: float = 10.0,
        max_iter: int = 100,
        consensus_method: str = "borda",
    ) -> Dict[str, Any]:
        """
        Learn Mallows model with Kendall distance.

        Args:
            generation: Current generation number
            n_vars: Number of variables (permutation length)
            cardinality: Not used for permutations
            selected_pop: Selected population of permutations
            selected_fitness: Fitness values (not used in learning)
            initial_theta: Initial theta parameter value
            upper_theta: Upper bound for theta
            max_iter: Maximum iterations for optimization
            consensus_method: Method to find consensus ("borda" or "median")

        Returns:
            Model dictionary containing:
                - v_probs: Probability matrix for v-vector
                - consensus: Consensus ranking
                - theta: Learned theta parameter
                - psis: Normalization constants
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        consensus = get_consensus(
            method=consensus_method,
            population=selected_pop,
            fitness=selected_fitness,        
            distance_func=kendall_distance 
        )

        # 2. Calculate theta parameter
        theta = self._calculate_theta(
            consensus, selected_pop, initial_theta, upper_theta, max_iter
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(theta, n_vars)

        # 4. Calculate v-vector probability matrix
        v_probs = self._calculate_v_prob_matrix(n_vars, psis, theta)

        return {
            "v_probs": v_probs,
            "consensus": consensus,
            "theta": theta,
            "psis": psis,
            "model_type": "mallows_kendall",
        }
 


    def _calculate_theta(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
    ) -> float:
        """Calculate theta parameter using maximum likelihood estimation."""
        inv_consensus = np.argsort(consensus)

        # Use all selected individuals to estimate theta (matching the MLE
        # described in the user guide, which matches the observed mean inversion
        # vector over the whole selected set).
        max_samples_for_theta = population.shape[0]
        sample_pop = population[:max_samples_for_theta]
        n_vars = population.shape[1]
        
        v_vectors = []

        for i in range(max_samples_for_theta):
            composition = sample_pop[i][inv_consensus]
            v_vec = self._v_vector(composition)
            v_vectors.append(v_vec)

        v_vectors_array = np.array(v_vectors)
        v_mean = np.mean(v_vectors_array, axis=0)

        # Use optimization to find theta that matches expected v-vector mean
        def objective(theta):
            expected_v = self._expected_v_vector(theta, n_vars)
            return np.sum((expected_v - v_mean) ** 2)

        # Use bounded optimization
        theta_opt = fminbound(objective, 0.001, upper_theta, xtol=1e-6, maxfun=max_iter)

        return float(theta_opt)

    def _v_vector(self, perm: np.ndarray) -> np.ndarray:
        """Calculate v-vector (Lehmer code) for a permutation."""
        n = len(perm)
        v = np.zeros(n, dtype=int)

        for i in range(n):
            v[i] = np.sum(perm[i] > perm[i + 1 :])

        return v

    def _expected_v_vector(self, theta: float, n: int) -> np.ndarray:
        """Calculate expected v-vector under Mallows model with given theta.

        For 0-indexed position ``j`` the inversion count V_j takes values in
        {0, ..., n-j-1} (support size n-j), the *same* support used by the
        sampling probability matrix (``_calculate_v_prob_matrix``).  Matching the
        two supports keeps theta estimation and sampling consistent.
        """
        expected_v = np.zeros(n)

        for j in range(n - 1):
            # Support of V_j is {0, ..., n-j-1}:  psi_j = sum_{r=0}^{n-j-1} e^{-r*theta}
            psi_j = (1 - np.exp(-(n - j) * theta)) / (1 - np.exp(-theta))

            expected_val = 0.0
            for r in range(n - j):
                prob_r = np.exp(-r * theta) / psi_j
                expected_val += r * prob_r

            expected_v[j] = expected_val

        return expected_v

    def _calculate_psi_constants(self, theta: float, n: int) -> np.ndarray:
        """Calculate psi normalization constants."""
        j = np.arange(1, n)  # j from 1 to n-1
        psis = (1 - np.exp(-(n - j + 1) * theta)) / (1 - np.exp(-theta))
        return psis

    def _calculate_v_prob_matrix(
        self, n_vars: int, psis: np.ndarray, theta: float
    ) -> np.ndarray:
        """Calculate probability matrix for v-vector values."""
        v_probs = np.zeros((n_vars - 1, n_vars))

        for j in range(n_vars - 1):
            for r in range(n_vars - j):
                v_probs[j, r] = np.exp(-r * theta) / psis[j]

        return v_probs


# For consistency with other learning methods
def learn_mallows_kendall(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    """
    Convenience function to learn Mallows model with Kendall distance.

    See LearnMallowsKendall for parameter details.
    """
    learner = LearnMallowsKendall()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )


class LearnMallowsCayley:
    """Learn Mallows model with Cayley distance"""

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        initial_theta: float = 0.1,
        upper_theta: float = 10.0,
        max_iter: int = 100,
        consensus_method: str = "borda",
    ) -> Dict[str, Any]:
        """
        Learn Mallows model with Cayley distance.

        Args:
            generation: Current generation number
            n_vars: Number of variables (permutation length)
            cardinality: Not used for permutations
            selected_pop: Selected population of permutations
            selected_fitness: Fitness values (not used in learning)
            initial_theta: Initial theta parameter value
            upper_theta: Upper bound for theta
            max_iter: Maximum iterations for optimization
            consensus_method: Method to find consensus ("borda" or "median")

        Returns:
            Model dictionary containing:
                - x_probs: Probability vector for x-vector
                - consensus: Consensus ranking
                - theta: Learned theta parameter
                - psis: Normalization constants
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        consensus = get_consensus(
            method=consensus_method,
            population=selected_pop,
            fitness=selected_fitness,        
            distance_func=cayley_distance   
        )

        # 2. Calculate theta parameter
        theta = self._calculate_theta(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(theta, n_vars)

        # 4. Calculate x-vector probability vector
        x_probs = self._calculate_x_prob_vector(psis)

        return {
            "x_probs": x_probs,
            "consensus": consensus,
            "theta": theta,
            "psis": psis,
            "model_type": "mallows_cayley",
        }

    def _calculate_theta(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
        n_vars: int,
    ) -> float:
        """
        Calculate theta parameter using Newton-Raphson method.

        References:
            [1] E. Irurozki, B. Calvo, J.A Lozano: Sampling and learning mallows
                and generalized mallows models under the cayley distance. Tech. Rep., 2013
        """
        # Get inverse of consensus
        inv_consensus = np.argsort(consensus)

        # Compose each permutation with inverse of consensus and calculate x-vectors
        n_pop, _ = population.shape
        x_vectors = []

        for i in range(n_pop):
            composition = population[i][inv_consensus]
            x_vec = _x_vector_cycles(composition)
            x_vectors.append(x_vec)

        x_vectors_array = np.array(x_vectors)
        x_mean = np.mean(x_vectors_array, axis=0)

        # Define the theta function and its derivative for Newton-Raphson
        def theta_function(theta):
            """Function to find root: f(theta) = 0"""
            j = np.arange(1, n_vars)  # j from 1 to n-1
            return np.sum(j / (j + np.exp(theta))) - np.sum(x_mean)

        def theta_derivative(theta):
            """Derivative of theta function"""
            j = np.arange(1, n_vars)  # j from 1 to n-1
            return np.sum((-j * np.exp(theta)) / ((np.exp(theta) + j) ** 2))

        # Use Newton-Raphson to find theta
        try:
            theta_opt = newton(
                theta_function,
                initial_theta,
                fprime=theta_derivative,
                maxiter=max_iter,
                tol=1e-6,
            )
            # Clip to valid range
            theta_opt = np.clip(theta_opt, 0.001, upper_theta)
        except:
            # Fallback to simple search if Newton-Raphson fails
            theta_opt = fminbound(
                lambda t: abs(theta_function(t)), 0.001, upper_theta, xtol=1e-6
            )

        return float(theta_opt)

    def _calculate_psi_constants(self, theta: float, n: int) -> np.ndarray:
        """
        Calculate psi normalization constants for Cayley distance.

        Psi_j = (n-j) * exp(-theta) + 1
        """
        j = np.arange(1, n)  # j from 1 to n-1
        psis = (n - j) * np.exp(-theta) + 1
        return psis

    def _calculate_x_prob_vector(self, psis: np.ndarray) -> np.ndarray:
        """
        Probability vector consumed by SampleMallowsCayley.

        NOTE ON CONVENTION: this returns ``1 / Psi_j``, which is **P(X_j = 0)**
        (not P(X_j = 1)).  The sampler compensates by setting ``X_j = 1`` when
        ``rand >= x_probs[j]``, so the effective sampled probability is
        ``P(X_j = 1) = 1 - 1/Psi_j = (n-1-j) e^{-theta} / Psi_j`` — the correct
        Mallows-Cayley marginal.  (The Generalized-Mallows-Cayley learner instead
        stores P(X_j = 1) directly in column 1 of its matrix; see
        LearnGeneralizedMallowsCayley._calculate_x_prob_matrix.)
        """
        x_probs = 1.0 / psis
        return x_probs


def learn_mallows_cayley(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    """
    Convenience function to learn Mallows model with Cayley distance.

    See LearnMallowsCayley for parameter details.
    """
    learner = LearnMallowsCayley()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )


class LearnGeneralizedMallowsKendall:
    """Learn Generalized Mallows model with Kendall distance

    The Generalized Mallows model uses a position-dependent spread parameter theta,
    where theta is a vector of length n-1 instead of a single value.

    References:
        [1] M.A. Fligner, J.S. Verducci: Distance based ranking models. JRSS, 1986
        [2] J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: A Distance-based
            Ranking Model Estimation of Distribution Algorithm for the Flowshop
            Scheduling Problem. IEEE TEVC, 2014
    """

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        initial_theta: float = 0.1,
        upper_theta: float = 10.0,
        max_iter: int = 100,
        consensus_method: str = "borda",
    ) -> Dict[str, Any]:
        """
        Learn Generalized Mallows model with Kendall distance.

        Args:
            generation: Current generation number
            n_vars: Number of variables (permutation length)
            cardinality: Not used for permutations
            selected_pop: Selected population of permutations
            selected_fitness: Fitness values (not used in learning)
            initial_theta: Initial theta parameter value
            upper_theta: Upper bound for theta
            max_iter: Maximum iterations for optimization
            consensus_method: Method to find consensus ("borda" or "median")

        Returns:
            Model dictionary containing:
                - v_probs: Probability matrix for v-vector
                - consensus: Consensus ranking
                - theta: Learned theta parameter vector (length n-1)
                - psis: Normalization constants (length n-1)
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        consensus = get_consensus(
            method=consensus_method,
            population=selected_pop,
            fitness=selected_fitness,      
            distance_func=kendall_distance   
        )

        # 2. Calculate theta parameters (vector of length n-1)
        thetas = self._calculate_thetas(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(thetas, n_vars)

        # 4. Calculate v-vector probability matrix
        v_probs = self._calculate_v_prob_matrix(n_vars, psis, thetas)

        return {
            "v_probs": v_probs,
            "consensus": consensus,
            "theta": thetas,
            "psis": psis,
            "model_type": "generalized_mallows_kendall",
        }

    def _calculate_thetas(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
        n_vars: int,
    ) -> np.ndarray:
        """Calculate theta parameters (one for each position) using MLE."""
        inv_consensus = np.argsort(consensus)

        # Use all selected individuals to estimate the per-position thetas.
        max_samples_for_theta = population.shape[0]
        sample_pop = population[:max_samples_for_theta]
        
        v_vectors = []
        for i in range(max_samples_for_theta):
            composition = sample_pop[i][inv_consensus]
            v_vec = self._v_vector(composition)
            v_vectors.append(v_vec)

        v_vectors_array = np.array(v_vectors)
        v_mean = np.mean(v_vectors_array, axis=0)

        thetas = np.zeros(n_vars - 1)

        for j in range(n_vars - 1):
            def theta_function(theta, j=j):
                # V_j has support {0, ..., m} with m = n-j-1 (the same support
                # used by the sampler's probability matrix).  Compute E[V_j]
                # exactly over that support so estimation matches sampling.
                m = n_vars - j - 1
                if theta < 1e-12:
                    return m / 2.0 - v_mean[j]
                x = np.exp(-theta)
                r = np.arange(m + 1)
                w = x ** r
                expected_vj = float(np.dot(r, w) / w.sum())
                return expected_vj - v_mean[j]

            
            try:
                theta_j = fminbound(
                    lambda t: abs(theta_function(t)),
                    0.001,
                    upper_theta,
                    xtol=1e-6,
                    maxfun=max_iter
                )
            except:
                theta_j = initial_theta
            
            thetas[j] = theta_j

        return thetas

    def _v_vector(self, perm: np.ndarray) -> np.ndarray:
        """Calculate v-vector (Lehmer code) for a permutation."""
        n = len(perm)
        v = np.zeros(n, dtype=int)

        for i in range(n):
            v[i] = np.sum(perm[i] > perm[i + 1 :])

        return v

    def _calculate_psi_constants(self, thetas: np.ndarray, n: int) -> np.ndarray:
        """Calculate psi normalization constants for each position."""
        psis = np.zeros(n - 1)

        for j in range(n - 1):
            # V_j takes values 0..n-j-1, so psi_j = sum_{r=0}^{n-j-1} exp(-r*theta_j)
            theta_j = thetas[j]
            n_j = n - j - 1
            psi_j = np.sum(np.exp(-np.arange(n_j + 1) * theta_j))
            psis[j] = psi_j

        return psis

    def _calculate_v_prob_matrix(
        self, n_vars: int, psis: np.ndarray, thetas: np.ndarray
    ) -> np.ndarray:
        """Calculate probability matrix for v-vector values."""
        v_probs = np.zeros((n_vars - 1, n_vars))

        for j in range(n_vars - 1):
            theta_j = thetas[j]
            for r in range(n_vars - j):
                v_probs[j, r] = np.exp(-r * theta_j) / psis[j]

        return v_probs


class LearnGeneralizedMallowsCayley:
    """Learn Generalized Mallows model with Cayley distance

    The Generalized Mallows model uses a position-dependent spread parameter theta,
    where theta is a vector of length n-1 instead of a single value.

    References:
        [1] M.A. Fligner, J.S. Verducci: Distance based ranking models. JRSS, 1986
        [2] J. Ceberio, E. Irurozki, A. Mendiburu, J.A Lozano: Extending Distance-based
            Ranking Models in EDAs. CEC 2014
    """

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        initial_theta: float = 0.1,
        upper_theta: float = 10.0,
        max_iter: int = 100,
        consensus_method: str = "borda",
    ) -> Dict[str, Any]:
        """
        Learn Generalized Mallows model with Cayley distance.

        Args:
            generation: Current generation number
            n_vars: Number of variables (permutation length)
            cardinality: Not used for permutations
            selected_pop: Selected population of permutations
            selected_fitness: Fitness values (not used in learning)
            initial_theta: Initial theta parameter value
            upper_theta: Upper bound for theta
            max_iter: Maximum iterations for optimization
            consensus_method: Method to find consensus ("borda" or "median")

        Returns:
            Model dictionary containing:
                - x_probs: Probability matrix for x-vector (n-1 x 2)
                - consensus: Consensus ranking
                - theta: Learned theta parameter vector (length n-1)
                - psis: Normalization constants (length n-1)
        """
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        consensus = get_consensus(
            method=consensus_method,
            population=selected_pop,
            fitness=selected_fitness,        
            distance_func=cayley_distance   
        )

        # 2. Calculate theta parameters (vector of length n-1)
        thetas = self._calculate_thetas(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        # 3. Calculate psi normalization constants
        psis = self._calculate_psi_constants(thetas, n_vars)

        # 4. Calculate x-vector probability matrix
        x_probs = self._calculate_x_prob_matrix(psis)

        return {
            "x_probs": x_probs,
            "consensus": consensus,
            "theta": thetas,
            "psis": psis,
            "model_type": "generalized_mallows_cayley",
        }

    def _calculate_thetas(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
        n_vars: int,
    ) -> np.ndarray:
        """Calculate theta parameters (one for each position) using MLE."""
        # Get inverse of consensus
        inv_consensus = np.argsort(consensus)

        # Compose each permutation with inverse of consensus
        n_pop = population.shape[0]
        x_vectors = []

        for i in range(n_pop):
            composition = population[i][inv_consensus]
            x_vec = _x_vector_cycles(composition)
            x_vectors.append(x_vec)

        x_vectors_array = np.array(x_vectors)
        x_mean = np.mean(x_vectors_array, axis=0)

        # Calculate theta_j for each position j independently
        thetas = np.zeros(n_vars - 1)

        for j in range(n_vars - 1):
            # For the Generalized Mallows model under the Cayley distance, the
            # j-th decomposition term X_j can be set to 1 in m_j = (n - 1 - j)
            # ways (this matches the number of choices made by the sampler in
            # _generate_perm_from_x at position j), so
            #     Psi_j       = 1 + m_j * exp(-theta_j)
            #     P(X_j = 1)  = m_j * exp(-theta_j) / Psi_j
            #     P(X_j = 0)  = 1 / Psi_j
            # The MLE matches E[X_j] = P(X_j = 1) to the observed mean x_mean[j]:
            #     x_mean = m_j e^{-theta} / (1 + m_j e^{-theta})
            #   =>  theta_j = log( m_j * (1 - x_mean) / x_mean )
            m_j = n_vars - 1 - j  # number of ways X_j = 1 (>= 1 for j = 0..n-2)

            xm = float(x_mean[j])
            if xm <= 0.0:
                # No dispersion observed at this position -> maximal concentration.
                theta_j = upper_theta
            elif xm >= 1.0:
                # Fully dispersed -> minimal spread parameter.
                theta_j = 0.001
            else:
                inner_val = m_j * (1.0 - xm) / xm
                theta_j = np.log(inner_val)
                theta_j = np.clip(theta_j, 0.001, upper_theta)

            thetas[j] = theta_j

        return thetas

    def _calculate_psi_constants(self, thetas: np.ndarray, n: int) -> np.ndarray:
        """Calculate psi normalization constants for each position."""
        psis = np.zeros(n - 1)

        for j in range(n - 1):
            # Psi_j = 1 + m_j * exp(-theta_j), with m_j = n - 1 - j the number
            # of ways X_j = 1 (consistent with _calculate_thetas and the sampler).
            m_j = n - 1 - j
            psis[j] = 1.0 + m_j * np.exp(-thetas[j])

        return psis

    def _calculate_x_prob_matrix(self, psis: np.ndarray) -> np.ndarray:
        """Calculate probability matrix for x-vector values.

        Returns a matrix of shape (n-1, 2) where:
        - Column 0 is P(X_j = 0)
        - Column 1 is P(X_j = 1)
        """
        n = len(psis)
        x_probs = np.zeros((n, 2))

        for j in range(n):
            # Psi_j = 1 + m_j e^{-theta_j}, so P(X_j = 0) = 1 / Psi_j and
            # P(X_j = 1) = m_j e^{-theta_j} / Psi_j = 1 - 1 / Psi_j.
            prob_0 = 1.0 / psis[j]
            prob_1 = 1.0 - prob_0
            x_probs[j, 0] = prob_0
            x_probs[j, 1] = prob_1

        return x_probs
    
class LearnMallowsUlam:
    """Learn Mallows model with Ulam distance
    
    Note: Unlike Kendall and Cayley, Ulam distance does not decompose into
    independent positional probabilities. Therefore, this learner only returns
    the consensus and theta parameter. Sampling requires MCMC methods.
    
    References:
        [1] E. Irurozki, J. Ceberio, B. Calvo, J.A. Lozano: Sampling and learning
            the Mallows model under the Ulam distance. Tech. Rep., 2014
    """

    def learn(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        population: np.ndarray,
        fitness: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        return self.__call__(
            generation=generation,
            n_vars=n_vars,
            cardinality=cardinality,
            selected_pop=population,
            selected_fitness=fitness,
            **kwargs
        )

    def __call__(
        self,
        generation: int,
        n_vars: int,
        cardinality: np.ndarray,
        selected_pop: np.ndarray,
        selected_fitness: np.ndarray,
        initial_theta: float = 0.1,
        upper_theta: float = 10.0,
        max_iter: int = 100,
        consensus_method: str = "borda",
    ) -> Dict[str, Any]:
        
        n_selected = selected_pop.shape[0]

        # 1. Calculate consensus ranking
        consensus = get_consensus(
            method=consensus_method,
            population=selected_pop,
            fitness=selected_fitness,      
            distance_func=ulam_distance   
        )

        theta = self._calculate_theta(
            consensus, selected_pop, initial_theta, upper_theta, max_iter, n_vars
        )

        
        return {
            "consensus": consensus,
            "theta": theta,
            "model_type": "mallows_ulam",
        }

    def _calculate_theta(
        self,
        consensus: np.ndarray,
        population: np.ndarray,
        initial_theta: float,
        upper_theta: float,
        max_iter: int,
        n_vars: int,
    ) -> float:
        """
        Estimate theta so that the model's expected Ulam distance matches the
        mean Ulam distance observed in the selected population.
        """
        
        d_avg = np.mean([ulam_distance(p, consensus) for p in population])

        if d_avg < 1e-5:
            return upper_theta

        Nd = self._get_ulam_distribution(n_vars)
        d_values = np.arange(len(Nd))
        
        def expected_distance_diff(theta):
            
            weights = Nd * np.exp(-theta * d_values)
            Z = np.sum(weights)
            
            if Z == 0 or np.isinf(Z):
                return float('inf')

            expected_d = np.sum(d_values * weights) / Z
            return expected_d - d_avg

        from scipy.optimize import fminbound
        theta_opt = fminbound(
            lambda t: abs(expected_distance_diff(t)), 
            0.001, 
            upper_theta, 
            xtol=1e-6, 
            maxfun=max_iter
        )

        return float(theta_opt)

    def _get_ulam_distribution(self, n: int) -> np.ndarray:
        """
        Return the number of permutations at each Ulam distance (Nd).

        Exact by enumeration for n <= 8; for larger n a Gaussian profile
        centred at n - 2*sqrt(n) with variance n/4 is used as an approximation.
        """
        
        import math
        if n <= 8:
            from itertools import permutations
            Nd = np.zeros(n)
            ref = np.arange(n)
            for p in permutations(range(n)):
                d = ulam_distance(ref, p)
                Nd[int(d)] += 1
            return Nd
        else:

            d_center = n - 2 * math.sqrt(n)
            d_values = np.arange(n)
            variance = n / 4.0 
            approx_Nd = np.exp(-0.5 * ((d_values - d_center) ** 2) / variance)
            return approx_Nd

def learn_mallows_ulam(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    learner = LearnMallowsUlam()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )
