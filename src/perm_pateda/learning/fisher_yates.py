
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from perm_pateda.representations.fisher_yates import FisherYatesRepresentation


class LearnFisherYatesUMDA:

    def __init__(self):
        self._repr = FisherYatesRepresentation()

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

        fy_codes = self._repr.encode(selected_pop)  # shape (m, n)

        domain_sizes = np.array([n - i for i in range(n)], dtype=int)

        marginals = []
        for i in range(n):
            domain_size = domain_sizes[i]

            counts = np.zeros(domain_size, dtype=float)
            for v in fy_codes[:, i]:
                counts[int(v)] += 1.0

            counts += laplace_alpha
            marginals.append(counts / counts.sum())

        return {
            "marginals": marginals,
            "domain_sizes": domain_sizes,
            "n_vars": n,
            "model_type": "fisher_yates_umda",
        }


def learn_fisher_yates_umda(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    
    learner = LearnFisherYatesUMDA()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )

class LearnFisherYatesTree:
 
    def __init__(self):
        self._repr = FisherYatesRepresentation()
 
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
        root: int = 0,
    ) -> Dict[str, Any]:
        
        selected_pop = np.atleast_2d(selected_pop)
        m, n = selected_pop.shape
 
        fy_codes = self._repr.encode(selected_pop)  # shape (m, n)
 
        domain_sizes = np.array([n - i for i in range(n)], dtype=int)
 
        mi_matrix = self._compute_mutual_information(fy_codes, domain_sizes, laplace_alpha)
 
        tree_edges = self._maximum_spanning_tree(mi_matrix, n)
 
        parents, tree_order = self._orient_tree(tree_edges, root, n)
 
        marginal_root = self._learn_marginal(
            fy_codes[:, root], domain_sizes[root], laplace_alpha
        )
 
        conditionals = {}
        for node in tree_order:
            if node == root:
                continue
            parent = parents[node]
            conditionals[node] = self._learn_conditional(
                fy_codes[:, parent],
                fy_codes[:, node],
                domain_sizes[parent],
                domain_sizes[node],
                laplace_alpha,
            )
 
        return {
            "marginal_root": marginal_root,
            "conditionals": conditionals,
            "parents": parents,
            "tree_order": tree_order,
            "domain_sizes": domain_sizes,
            "root": root,
            "n_vars": n,
            "model_type": "fisher_yates_tree",
        }
 
    def _learn_marginal(
        self, values: np.ndarray, domain_size: int, alpha: float
    ) -> np.ndarray:
        counts = np.zeros(domain_size, dtype=float)
        for v in values:
            counts[int(v)] += 1.0
        counts += alpha
        return counts / counts.sum()
 
    def _learn_conditional(
        self,
        parent_vals: np.ndarray,
        child_vals: np.ndarray,
        parent_domain: int,
        child_domain: int,
        alpha: float,
    ) -> np.ndarray:
        counts = np.full((parent_domain, child_domain), alpha, dtype=float)
        for vp, vc in zip(parent_vals, child_vals):
            counts[int(vp), int(vc)] += 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        return counts / row_sums
 
    def _compute_mutual_information(
        self,
        codes: np.ndarray,
        domain_sizes: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        m, n = codes.shape
        mi = np.zeros((n, n), dtype=float)
 
        for i in range(n):
            for j in range(i + 1, n):
                di, dj = domain_sizes[i], domain_sizes[j]
 
                joint = np.full((di, dj), alpha, dtype=float)
                for k in range(m):
                    joint[int(codes[k, i]), int(codes[k, j])] += 1.0
                joint /= joint.sum()
 
                p_i = joint.sum(axis=1)
                p_j = joint.sum(axis=0)
                outer = np.outer(p_i, p_j)
 
                mask = (joint > 0) & (outer > 0)
                mi_val = np.sum(joint[mask] * np.log(joint[mask] / outer[mask]))
                mi[i, j] = mi_val
                mi[j, i] = mi_val
 
        return mi
 
    def _maximum_spanning_tree(
        self, mi_matrix: np.ndarray, n: int
    ) -> List[Tuple[int, int]]:
        in_tree = [False] * n
        in_tree[0] = True
        edges = []
 
        for _ in range(n - 1):
            best_mi = -np.inf
            best_i, best_j = -1, -1
 
            for i in range(n):
                if not in_tree[i]:
                    continue
                for j in range(n):
                    if in_tree[j]:
                        continue
                    if mi_matrix[i, j] > best_mi:
                        best_mi = mi_matrix[i, j]
                        best_i, best_j = i, j
 
            if best_i == -1:
                break
 
            edges.append((best_i, best_j))
            in_tree[best_j] = True
 
        return edges
 
    def _orient_tree(
        self, edges: List[Tuple[int, int]], root: int, n: int
    ) -> Tuple[Dict[int, int], List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
 
        parents: Dict[int, int] = {}
        tree_order: List[int] = []
        visited = set()
        queue = [root]
        visited.add(root)
 
        while queue:
            node = queue.pop(0)
            tree_order.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    parents[neighbor] = node
                    queue.append(neighbor)
 
        return parents, tree_order
 
 
def learn_fisher_yates_tree(
    generation: int,
    n_vars: int,
    cardinality: np.ndarray,
    selected_pop: np.ndarray,
    selected_fitness: np.ndarray,
    **params,
) -> Dict[str, Any]:
    
    learner = LearnFisherYatesTree()
    return learner(
        generation, n_vars, cardinality, selected_pop, selected_fitness, **params
    )
