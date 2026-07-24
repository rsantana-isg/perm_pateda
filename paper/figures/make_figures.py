"""
Generate the figures used in the pateda user guide (v3).

Schematic figures are drawn directly; data-driven figures are produced from
short EDA runs.  Run with the interpreter that has ``pateda`` + ``bayes_nets``
installed (python3.11 in the development environment)::

    python3.11 paper/figures/make_figures.py

All figures are written as PDF into paper/figures/.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx

warnings.filterwarnings("ignore")
np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))
def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# 1. How pateda stores an arbitrary (ordered) factorization
# ---------------------------------------------------------------------------
def fig_factorization_storage():
    fig, (axg, axs) = plt.subplots(1, 2, figsize=(10, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.35]})
    # --- left: a junction-graph style factorization over 6 variables ---
    G = nx.DiGraph()
    edges = [(0, 1), (1, 2), (1, 3), (3, 4), (3, 5)]
    G.add_edges_from(edges)
    pos = {0: (0, 2), 1: (1, 2), 2: (2, 2.7), 3: (2, 1.3), 4: (3, 2), 5: (3, 0.7)}
    nx.draw_networkx_nodes(G, pos, ax=axg, node_color="#cfe3f7",
                           edgecolors="#2b6cb0", node_size=650)
    nx.draw_networkx_edges(G, pos, ax=axg, arrowstyle="-|>", arrowsize=14,
                           edge_color="#4a5568", width=1.4)
    nx.draw_networkx_labels(G, pos, {i: f"$X_{i+1}$" for i in range(6)}, ax=axg,
                            font_size=11)
    axg.set_title("A factorization (ancestral order)", fontsize=11)
    axg.axis("off")

    # --- right: the FactorizedModel storage ---
    axs.axis("off")
    axs.set_title("FactorizedModel storage", fontsize=11)
    rows = [
        (r"factor 1", r"new $\{X_1\}$",       r"sep $\emptyset$",       r"$P(X_1)$"),
        (r"factor 2", r"new $\{X_2\}$",       r"sep $\{X_1\}$",         r"$P(X_2\!\mid\!X_1)$"),
        (r"factor 3", r"new $\{X_3\}$",       r"sep $\{X_2\}$",         r"$P(X_3\!\mid\!X_2)$"),
        (r"factor 4", r"new $\{X_4\}$",       r"sep $\{X_2\}$",         r"$P(X_4\!\mid\!X_2)$"),
        (r"factor 5", r"new $\{X_5,X_6\}$",   r"sep $\{X_4\}$",         r"$P(X_5,X_6\!\mid\!X_4)$"),
    ]
    y = 0.9
    axs.text(0.02, 1.0, "structure (cliques matrix)", fontsize=9, color="#2b6cb0")
    axs.text(0.74, 1.0, "parameters", fontsize=9, color="#c05621")
    for name, new, sep, tab in rows:
        axs.add_patch(FancyBboxPatch((0.0, y - 0.07), 0.72, 0.13,
                      boxstyle="round,pad=0.01", fc="#eef5fc", ec="#2b6cb0"))
        axs.text(0.02, y, f"{name}:  {new},  {sep}", fontsize=9, va="center")
        axs.add_patch(FancyBboxPatch((0.74, y - 0.07), 0.26, 0.13,
                      boxstyle="round,pad=0.01", fc="#fdf0e6", ec="#c05621"))
        axs.text(0.87, y, tab, fontsize=9, va="center", ha="center")
        y -= 0.19
    axs.text(0.0, y + 0.02,
             r"row format: $[\,|S_i|,\ |C_i|,\ \mathrm{sep\ idx}\ldots,\ \mathrm{new\ idx}\ldots\,]$",
             fontsize=8.5, color="#4a5568")
    axs.set_xlim(0, 1.02); axs.set_ylim(y - 0.05, 1.08)
    save(fig, "fig_factorization_storage.pdf")


# ---------------------------------------------------------------------------
# 2. Interaction structure of representative benchmark problems
# ---------------------------------------------------------------------------
def fig_problem_structures():
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    def style(ax, G, pos, title):
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=120,
                               node_color="#cfe3f7", edgecolors="#2b6cb0")
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#718096", width=1.0)
        ax.set_title(title, fontsize=10); ax.axis("off")

    # OneMax: no edges
    G = nx.Graph(); G.add_nodes_from(range(12))
    pos = {i: (i % 6, -(i // 6)) for i in range(12)}
    style(axes[0, 0], G, pos, "OneMax (separable)")

    # Trap/ADF: disjoint order-3 blocks
    G = nx.Graph()
    for b in range(4):
        nodes = [3 * b, 3 * b + 1, 3 * b + 2]
        G.add_nodes_from(nodes)
        for i in nodes:
            for j in nodes:
                if i < j: G.add_edge(i, j)
    pos = {}
    for b in range(4):
        cx = b
        pos[3 * b] = (cx, 0.3); pos[3 * b + 1] = (cx - 0.2, -0.3); pos[3 * b + 2] = (cx + 0.2, -0.3)
    style(axes[0, 1], G, pos, "Trap / ADF (order-3 blocks)")

    # Ising 2D lattice
    G = nx.grid_2d_graph(4, 4)
    pos = {n: (n[0], n[1]) for n in G.nodes()}
    style(axes[0, 2], G, pos, "Ising (2D lattice)")

    # NK circular (k=2)
    n = 12; G = nx.Graph()
    for i in range(n):
        G.add_edge(i, (i + 1) % n); G.add_edge(i, (i + 2) % n)
    pos = nx.circular_layout(G)
    style(axes[1, 0], G, pos, "NK landscape (circular, k=2)")

    # UBQP: dense random Q
    n = 10; G = nx.gnp_random_graph(n, 0.35, seed=3)
    pos = nx.circular_layout(G)
    style(axes[1, 1], G, pos, "UBQP (dense Q)")

    # HP protein: chain
    G = nx.path_graph(12)
    pos = {i: (i, 0) for i in range(12)}
    style(axes[1, 2], G, pos, "HP folding (sequential chain)")

    save(fig, "fig_problem_structures.pdf")


# ---------------------------------------------------------------------------
# Helpers for real runs
# ---------------------------------------------------------------------------
def _bn_edges(model):
    """Return list of directed edges (i,j) from a BayesianNetworkModel."""
    S = np.asarray(model.structure)
    return [(i, j) for i in range(S.shape[0]) for j in range(S.shape[1]) if S[i, j]]


def _run_ebna_cached(n_vars=15, pop=400, gens=12, seed=1):
    from pateda.core.eda import EDA
    from pateda.core.components import EDAComponents, CacheConfig
    from pateda.seeding.random_init import RandomInit
    from pateda.selection.truncation import TruncationSelection
    from pateda.learning.ebna import LearnEBNA
    from pateda.sampling.bayesian_network import SampleBayesianNetwork
    from pateda.replacement.elitist import ElitistReplacement
    from pateda.stop_conditions.max_generations import MaxGenerations
    from pateda.functions import deceptive3

    comp = EDAComponents(
        seeding=RandomInit(), selection=TruncationSelection(ratio=0.5),
        learning=LearnEBNA(), sampling=SampleBayesianNetwork(n_samples=pop),
        replacement=ElitistReplacement(n_elite=1), stop_condition=MaxGenerations(gens))
    eda = EDA(pop, n_vars, deceptive3, np.full(n_vars, 2), comp, random_seed=seed)
    cc = CacheConfig(cache_populations=True, cache_fitness=True, cache_models=True,
                     cache_statistics=True, cache_selections=True)
    stats, cache = eda.run(cache_config=cc, verbose=False)
    return stats, cache


# ---------------------------------------------------------------------------
# 3. A Bayesian network learned by EBNA on a deceptive problem
# ---------------------------------------------------------------------------
def fig_bn_learned(cache):
    model = cache.models[len(cache.models) // 2]
    edges = _bn_edges(model)
    n = np.asarray(model.structure).shape[0]
    G = nx.DiGraph(); G.add_nodes_from(range(n)); G.add_edges_from(edges)
    # circular layout grouped by trap block of 3
    pos = nx.circular_layout(G)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    block_colors = ["#cfe3f7", "#c6f6d5", "#fed7d7", "#fefcbf", "#e9d8fd"]
    node_colors = [block_colors[(i // 3) % len(block_colors)] for i in range(n)]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           edgecolors="#2b6cb0", node_size=560)
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle="-|>", arrowsize=13,
                           edge_color="#4a5568", width=1.3)
    nx.draw_networkx_labels(G, pos, {i: f"$X_{{{i+1}}}$" for i in range(n)}, ax=ax,
                            font_size=10)
    ax.set_title("Bayesian network learned by EBNA (deceptive-3, blocks of 3 shaded)",
                 fontsize=10)
    ax.axis("off")
    save(fig, "fig_bn_learned.pdf")


# ---------------------------------------------------------------------------
# 4. Edge-frequency matrix accumulated over generations
# ---------------------------------------------------------------------------
def fig_edge_frequency(cache):
    models = cache.models
    n = np.asarray(models[0].structure).shape[0]
    freq = np.zeros((n, n))
    for m in models:
        S = np.asarray(m.structure)
        und = ((S + S.T) > 0).astype(float)
        freq += und
    freq /= len(models)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(freq, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1), fontsize=7)
    ax.set_yticklabels(range(1, n + 1), fontsize=7)
    ax.set_xlabel("variable"); ax.set_ylabel("variable")
    ax.set_title("Edge-frequency matrix over generations (EBNA, deceptive-3)", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of generations")
    save(fig, "fig_edge_frequency.pdf")


# ---------------------------------------------------------------------------
# 5. Structure snapshots along the run (evolution of the model)
# ---------------------------------------------------------------------------
def fig_structure_evolution(cache):
    models = cache.models
    idxs = [0, len(models) // 2, len(models) - 1]
    n = np.asarray(models[0].structure).shape[0]
    pos = nx.circular_layout(nx.complete_graph(n))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, gi in zip(axes, idxs):
        G = nx.DiGraph(); G.add_nodes_from(range(n))
        G.add_edges_from(_bn_edges(models[gi]))
        node_colors = ["#cfe3f7"] * n
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               edgecolors="#2b6cb0", node_size=260)
        nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle="-|>", arrowsize=9,
                               edge_color="#4a5568", width=1.0)
        ax.set_title(f"generation {gi}  ({G.number_of_edges()} edges)", fontsize=10)
        ax.axis("off")
    fig.suptitle("Evolution of the learned structure along generations", fontsize=11)
    save(fig, "fig_structure_evolution.pdf")


# ---------------------------------------------------------------------------
# 6. A-priori interaction matrix restricting the model
# ---------------------------------------------------------------------------
def fig_apriori_interaction():
    n = 12
    M = np.zeros((n, n))
    for b in range(4):  # block-diagonal a-priori structure
        for i in range(3 * b, 3 * b + 3):
            for j in range(3 * b, 3 * b + 3):
                if i != j: M[i, j] = 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.imshow(M, cmap="Blues")
    ax1.set_title("interaction_matrix (allowed edges)", fontsize=10)
    ax1.set_xticks(range(n)); ax1.set_yticks(range(n))
    ax1.set_xticklabels(range(1, n + 1), fontsize=6); ax1.set_yticklabels(range(1, n + 1), fontsize=6)
    G = nx.Graph((M + M.T > 0))
    pos = nx.spring_layout(G, seed=2)
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=200, node_color="#cfe3f7",
                           edgecolors="#2b6cb0")
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color="#718096")
    nx.draw_networkx_labels(G, pos, {i: i + 1 for i in range(n)}, ax=ax2, font_size=8)
    ax2.set_title("model restricted to allowed edges", fontsize=10); ax2.axis("off")
    save(fig, "fig_apriori_interaction.pdf")


# ---------------------------------------------------------------------------
# 7. Vine copula structure (schematic R-vine over 5 variables)
# ---------------------------------------------------------------------------
def fig_vine_structure():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    labels = {i: f"$u_{i+1}$" for i in range(5)}
    # T1: path 1-2-3-4-5
    T1 = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4)])
    pos1 = {i: (i, 0) for i in range(5)}
    # T2 over edges of T1
    T2 = nx.Graph([(0, 1), (1, 2), (2, 3)])
    pos2 = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}
    lab2 = {0: "12", 1: "23", 2: "34", 3: "45"}
    # T3
    T3 = nx.Graph([(0, 1), (1, 2)])
    pos3 = {0: (0, 0), 1: (1, 0), 2: (2, 0)}
    lab3 = {0: "13|2", 1: "24|3", 2: "35|4"}
    for ax, (T, pos, lab, title) in zip(axes, [
        (T1, pos1, labels, "Tree $T_1$: bivariate copulas"),
        (T2, pos2, lab2, "Tree $T_2$: conditional pairs"),
        (T3, pos3, lab3, "Tree $T_3$")]):
        nx.draw_networkx_nodes(T, pos, ax=ax, node_size=520, node_color="#e9d8fd",
                               edgecolors="#6b46c1")
        nx.draw_networkx_edges(T, pos, ax=ax, edge_color="#718096", width=1.3)
        nx.draw_networkx_labels(T, pos, lab, ax=ax, font_size=9)
        ax.set_title(title, fontsize=10); ax.axis("off")
    fig.suptitle("Regular-vine decomposition of a 5-variable copula", fontsize=11)
    save(fig, "fig_vine_structure.pdf")


if __name__ == "__main__":
    fig_factorization_storage()
    fig_problem_structures()
    fig_apriori_interaction()
    fig_vine_structure()
    print("running EBNA for data-driven figures ...")
    _stats, cache = _run_ebna_cached()
    fig_bn_learned(cache)
    fig_edge_frequency(cache)
    fig_structure_evolution(cache)
    print("all figures done")
