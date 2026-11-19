import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from perm_pateda.functions import create_random_lop

from perm_pateda.algorithms.permutation import (
    MallowsKendallEDA as MallowsPy,
    GMallowsKendallEDA as GMallowsPy,
    PlackettLuceEDA as PL_Py,
    LehmerUmdaEDA as LehmerPy,
)

from perm_pateda._cython.algorithms.permutation import (
    MallowsKendallEDA as MallowsCy,
    GMallowsKendallEDA as GMallowsCy,
    PlackettLuceEDA as PL_Cy,
    LehmerUmdaEDA as LehmerCy,
)

ALGORITHMS = {
    "MallowsKendall": (MallowsPy, MallowsCy),
    "GMallowsKendall": (GMallowsPy, GMallowsCy),
    "PlackettLuce": (PL_Py, PL_Cy),
    "LehmerUMDA": (LehmerPy, LehmerCy),
}

CONFIGS = [
    {"n_vars": 20, "pop_size": 50, "n_gen": 20},
    {"n_vars": 30, "pop_size": 80, "n_gen": 25},
    {"n_vars": 40, "pop_size": 100, "n_gen": 20},
    {"n_vars": 50, "pop_size": 120, "n_gen": 15},
]


def run_algorithm(AlgClass, n_vars, pop_size, n_gen, seed=42):
    fitness = create_random_lop(n_vars, seed=0)
    alg = AlgClass(
        n_vars=n_vars,
        fitness_func=fitness,
        pop_size=pop_size,
        n_gen=n_gen,
        random_seed=seed,
        selection_ratio=0.5,
        elitism=True,
    )
    start = time.perf_counter()
    stats, _ = alg.run(verbose=False)
    elapsed = time.perf_counter() - start
    return elapsed, stats.best_fitness_overall


def main():
    results = []
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("BENCHMARK COMPLETO: Python vs Cython")
    print("=" * 80)

    for algo_name, (PyClass, CyClass) in ALGORITHMS.items():
        print(f"\n>>> {algo_name}")
        for cfg in CONFIGS:
            n, pop, gen = cfg["n_vars"], cfg["pop_size"], cfg["n_gen"]
            print(f"  n={n}, pop={pop}, gen={gen}")

            t_py, best_py = run_algorithm(PyClass, n, pop, gen)
            t_cy, best_cy = run_algorithm(CyClass, n, pop, gen)
            speedup = t_py / t_cy if t_cy > 0 else np.nan

            results.append({
                "algorithm": algo_name,
                "n_vars": n,
                "pop_size": pop,
                "n_gen": gen,
                "time_python": t_py,
                "time_cython": t_cy,
                "speedup": speedup,
                "best_python": best_py,
                "best_cython": best_cy,
            })

            print(f"    Python: {t_py:.2f}s | Cython: {t_cy:.2f}s | Speedup: {speedup:.2f}x")

    df = pd.DataFrame(results)
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, algo_name in enumerate(ALGORITHMS.keys()):
        ax = axes[idx]
        sub = df[df["algorithm"] == algo_name]
        x = np.arange(len(sub))
        width = 0.35

        ax.bar(x - width/2, sub["time_python"], width, label="Python", color="#4C72B0")
        ax.bar(x + width/2, sub["time_cython"], width, label="Cython", color="#55A868")
        ax.set_title(algo_name)
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={r.n_vars}" for r in sub.itertuples()], rotation=45)
        ax.set_ylabel("Tiempo (s)")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "tiempos_comparacion.png", dpi=150)
    plt.close()


    plt.figure(figsize=(10, 6))
    for algo_name in ALGORITHMS.keys():
        sub = df[df["algorithm"] == algo_name]
        plt.plot(sub["n_vars"], sub["speedup"], marker="o", label=algo_name)
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.xlabel("Número de variables (n)")
    plt.ylabel("Speedup (Python / Cython)")
    plt.title("Speedup de Cython vs Python")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "speedup.png", dpi=150)
    plt.close()

    print(f"Gráficos guardados en: {output_dir}/")
    print("\nResumen:")
    print(df.groupby("algorithm")[["speedup"]].mean().round(2))


if __name__ == "__main__":
    main()