from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from perm_pateda.seeding import PermutationInit


# ---------------------------------------------------------------------------
# Folder setup
# ---------------------------------------------------------------------------

def ensure_evolution_dirs(root: str | Path = "evolution") -> dict:
    root = Path(root)
    subdirs = {
        "diversity": root / "diversity",
        "convergence": root / "convergence",
    }
    for path in subdirs.values():
        os.makedirs(path, exist_ok=True)
    return subdirs



def positional_entropy(population: np.ndarray) -> float:
    pop_size, n = population.shape
    if n <= 1 or pop_size <= 1:
        return 0.0
    log_n = np.log(n)
    total = 0.0
    for j in range(n):
        _, counts = np.unique(population[:, j], return_counts=True)
        probs = counts / pop_size
        h = -np.sum(probs * np.log(probs))
        total += h / log_n if log_n > 0 else 0.0
    return float(total / n)


def run_with_history(
    cls,
    n_vars: int,
    fitness_func: Callable,
    pop_size: int = 100,
    n_gen: int = 60,
    selection_ratio: float = 0.5,
    random_seed: Optional[int] = None,
    extra_kwargs: Optional[dict] = None,
) -> dict:
    eda = cls(n_vars=n_vars, fitness_func=fitness_func, pop_size=pop_size,
              n_gen=n_gen, selection_ratio=selection_ratio,
              random_seed=random_seed, **(extra_kwargs or {}))

    n_select = max(2, int(pop_size * selection_ratio))
    seeder = PermutationInit()
    population = seeder.seed(n_vars, pop_size, eda._card, rng=eda.rng)
    fitness = eda._evaluate(population)

    best_idx = int(np.argmax(fitness))
    best_ind = population[best_idx].copy()
    best_fit = float(fitness[best_idx])

    history = {"best_fitness": [], "mean_fitness": [], "diversity": []}

    for gen in range(n_gen):
        history["best_fitness"].append(best_fit)
        history["mean_fitness"].append(float(np.mean(fitness)))
        history["diversity"].append(positional_entropy(population))

        sorted_idx = np.argsort(fitness)[::-1]
        selected = population[sorted_idx[:n_select]]
        sel_fit = fitness[sorted_idx[:n_select]]

        model = eda._learn(gen, selected, sel_fit)
        new_pop = eda._sample(model, current_pop=population)
        new_fit = eda._evaluate(new_pop)

        if eda.elitism:
            new_best_idx = int(np.argmax(new_fit))
            if best_fit > new_fit[new_best_idx]:
                worst_idx = int(np.argmin(new_fit))
                new_pop[worst_idx] = best_ind.copy()
                new_fit[worst_idx] = best_fit
            else:
                best_fit = float(new_fit[new_best_idx])
                best_ind = new_pop[new_best_idx].copy()

        population = new_pop
        fitness = new_fit

    history["best_fitness"].append(best_fit)
    history["mean_fitness"].append(float(np.mean(fitness)))
    history["diversity"].append(positional_entropy(population))
    return history



def plot_diversity(histories: dict, title: str, out_dir: Path, filename: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in histories.items():
        ax.plot(hist["diversity"], label=label, linewidth=1.4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean positional entropy (normalized)")
    ax.set_title(f"Population diversity --- {title}")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_convergence(histories: dict, title: str, out_dir: Path, filename: str, maximize: bool = True) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, hist in histories.items():
        vals = hist["best_fitness"] if maximize else [-v for v in hist["best_fitness"]]
        ax.plot(vals, label=label, linewidth=1.4)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness so far")
    ax.set_title(f"Convergence --- {title}")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _normalize_convergence(best_fitness: list) -> np.ndarray:
    arr = np.asarray(best_fitness, dtype=float)
    return arr - arr[0]


def _aggregate_instance(histories: dict) -> tuple[dict, dict]:
    shifted = {label: _normalize_convergence(h["best_fitness"]) for label, h in histories.items()}
    global_max = max(arr[-1] for arr in shifted.values())
    conv = {label: (arr / global_max if global_max > 0 else arr) for label, arr in shifted.items()}
    div = {label: np.asarray(h["diversity"], dtype=float) for label, h in histories.items()}
    return conv, div


def plot_aggregated(curves_by_algo: dict, ylabel: str, title: str, out_path: Path, ylim: Optional[tuple] = None) -> Path:

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, curves in curves_by_algo.items():
        min_len = min(len(c) for c in curves)
        stacked = np.stack([c[:min_len] for c in curves])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        x = np.arange(min_len)
        line, = ax.plot(x, mean, label=label, linewidth=1.6)
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.15)
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path



def track_evolution(problems: list, algorithms: list, args, root: str | Path = "evolution") -> None:

    dirs = ensure_evolution_dirs(root)
    mode = getattr(args, "evolution_mode", "aggregate")
    print(f"=== Tracking search evolution (mode={mode}) ===")

    conv_by_type: dict = {}
    div_by_type: dict = {}

    for p in problems:
        histories = {}
        for label, cls, extra in algorithms:
            try:
                histories[label] = run_with_history(
                    cls, p["n_vars"], p["fitness"],
                    pop_size=args.pop, n_gen=args.gen,
                    selection_ratio=args.selection_ratio,
                    random_seed=args.base_seed,
                    extra_kwargs=extra,
                )
            except Exception as exc:
                print(f"  ! {p['name']} / {label} FAILED: {exc}")

        if not histories:
            continue

        conv, div = _aggregate_instance(histories)
        ptype = p["ptype"]
        conv_by_type.setdefault(ptype, {})
        div_by_type.setdefault(ptype, {})
        for label in histories:
            conv_by_type[ptype].setdefault(label, []).append(conv[label])
            div_by_type[ptype].setdefault(label, []).append(div[label])

        if mode == "per-instance":
            fname = f"{ptype}_{p['name']}.png"
            dpath = plot_diversity(histories, p["name"], dirs["diversity"], fname)
            cpath = plot_convergence(histories, p["name"], dirs["convergence"], fname,
                                      maximize=(p["better"] == "max"))
            print(f"  [per-instance] {p['name']}: {dpath}  |  {cpath}")

    for ptype in conv_by_type:
        n_inst = len(next(iter(conv_by_type[ptype].values())))
        dpath = plot_aggregated(
            div_by_type[ptype], "Mean positional entropy (normalized)",
            f"Population diversity --- {ptype}  (avg. over {n_inst} instances)",
            dirs["diversity"] / f"{ptype}.png", ylim=(0, 1.02),
        )
        cpath = plot_aggregated(
            conv_by_type[ptype], "Relative improvement over initial population",
            f"Convergence --- {ptype}  (avg. over {n_inst} instances)",
            dirs["convergence"] / f"{ptype}.png",
        )
        print(f"  [aggregate] {ptype} ({n_inst} instances): {dpath}  |  {cpath}")