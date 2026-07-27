"""
stats_utils.py
--------------
Statistical comparison helpers following the conventions of Ceberio et al. (2015)
and the wider pateda experimental framework.

Public API
----------
summary_table(df)          → pd.DataFrame  (mean ± std, best, rank)
friedman_test(df)          → dict
wilcoxon_pairwise(df)      → pd.DataFrame
critical_difference_plot(df, alpha, filepath)  → None
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, rankdata


# ---------------------------------------------------------------------------
# Input-format handling
# ---------------------------------------------------------------------------

def _ensure_long(
    df: pd.DataFrame,
    value_col: str = "Best Fitness",
    problem_col: str = "Problem",
    algorithm_col: str = "Algorithm",
    seed_col: str = "Seed",
) -> pd.DataFrame:
    """Return the results in the LONG format used internally by these helpers.

    Two input layouts are accepted:

    * **Long** — one row per (problem, algorithm, seed) with the named columns
      ``problem_col``/``algorithm_col``/``seed_col``/``value_col`` (missing
      ``Problem``/``Seed`` columns are filled with a single default block).
    * **Wide** — the layout documented in the user guide (Section 9.2): rows are
      problem instances (the DataFrame index) and columns are algorithms, each
      cell holding the value.  It is melted to long form with one seed block.

    The two are told apart by whether the long-format value/algorithm columns are
    present.
    """
    if algorithm_col in df.columns and value_col in df.columns:
        out = df.copy()
        if problem_col not in out.columns:
            out[problem_col] = 0
        if seed_col not in out.columns:
            out[seed_col] = 0
        return out

    # Treat as wide: rows = instances, columns = algorithms.
    wide = df.copy()
    index_name = wide.index.name if wide.index.name is not None else "index"
    long = (
        wide.reset_index()
        .melt(id_vars=index_name, var_name=algorithm_col, value_name=value_col)
        .rename(columns={index_name: problem_col})
    )
    long[seed_col] = 0
    return long


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(
    df: pd.DataFrame,
    value_col: str = "Best Fitness",
    group_cols: Sequence[str] = ("Problem", "Algorithm"),
) -> pd.DataFrame:
    """Return a summary table with mean ± std, best value, and rank per problem.

    Parameters
    ----------
    df:
        Raw results DataFrame with one row per (problem, algorithm, seed).
    value_col:
        Column containing the fitness / objective value.
    group_cols:
        Columns to group by (problem first, then algorithm).

    Returns
    -------
    pd.DataFrame indexed by (Problem, Algorithm) with columns:
        Mean, Std, Best, Rank

    Accepts either the long format (columns Problem/Algorithm/Seed/value_col) or
    the wide format documented in the user guide (rows=instances, columns=algorithms).
    """
    df = _ensure_long(
        df, value_col=value_col, problem_col=group_cols[0], algorithm_col=group_cols[1]
    )
    agg = (
        df.groupby(list(group_cols))[value_col]
        .agg(Mean="mean", Std="std", Best="max")
        .round(4)
    )
    # Rank within each problem (higher = better by default, flip if minimisation)
    agg["Rank"] = (
        agg.groupby(level=0)["Mean"]
        .rank(ascending=False, method="average")
    )
    agg["Mean±Std"] = agg.apply(
        lambda r: f"{r['Mean']:.2f} ± {r['Std']:.2f}", axis=1
    )
    return agg[["Mean±Std", "Best", "Rank", "Mean", "Std"]]


# ---------------------------------------------------------------------------
# Friedman test
# ---------------------------------------------------------------------------

def friedman_test(
    df: pd.DataFrame,
    value_col: str = "Best Fitness",
    problem_col: str = "Problem",
    algorithm_col: str = "Algorithm",
    seed_col: str = "Seed",
) -> dict:
    """Perform the Friedman test across algorithms.

    For each problem the algorithms are ranked per seed; the Friedman
    statistic is computed on those ranks (over problems × seeds as blocks).

    Returns
    -------
    dict with keys: statistic, p_value, reject_H0 (at α=0.05)

    Accepts either the long or the wide (UG §9.2) input format.
    """
    df = _ensure_long(
        df, value_col=value_col, problem_col=problem_col,
        algorithm_col=algorithm_col, seed_col=seed_col,
    )
    algorithms = df[algorithm_col].unique()
    # Build matrix: rows = blocks (problem × seed), cols = algorithm ranks
    records = []
    for prob in df[problem_col].unique():
        for seed in df[seed_col].unique():
            row = df[(df[problem_col] == prob) & (df[seed_col] == seed)]
            if len(row) == len(algorithms):
                records.append(
                    [row[row[algorithm_col] == a][value_col].values[0] for a in algorithms]
                )
    if not records:
        return {"statistic": np.nan, "p_value": np.nan, "reject_H0": False}

    matrix = np.array(records)  # (n_blocks, n_algorithms)
    stat, p = friedmanchisquare(*matrix.T)
    return {
        "statistic": round(stat, 4),
        "p_value": round(p, 6),
        "reject_H0": p < 0.05,
        "algorithms": list(algorithms),
    }


# ---------------------------------------------------------------------------
# Wilcoxon pairwise post-hoc
# ---------------------------------------------------------------------------

def wilcoxon_pairwise(
    df: pd.DataFrame,
    value_col: str = "Best Fitness",
    algorithm_col: str = "Algorithm",
    seed_col: str = "Seed",
    problem_col: str = "Problem",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Wilcoxon signed-rank test for each pair of algorithms.

    Values are pooled across seeds and problems (one observation per
    problem × seed block), so differences are evaluated on paired blocks.

    Returns
    -------
    pd.DataFrame with columns: AlgA, AlgB, statistic, p_value, significant

    Accepts either the long or the wide (UG §9.2) input format.
    """
    df = _ensure_long(
        df, value_col=value_col, problem_col=problem_col,
        algorithm_col=algorithm_col, seed_col=seed_col,
    )
    algorithms = sorted(df[algorithm_col].unique())
    rows = []
    for a, b in itertools.combinations(algorithms, 2):
        paired_a, paired_b = [], []
        for prob in df[problem_col].unique():
            for seed in df[seed_col].unique():
                va = df[(df[problem_col] == prob) & (df[seed_col] == seed) & (df[algorithm_col] == a)][value_col].values
                vb = df[(df[problem_col] == prob) & (df[seed_col] == seed) & (df[algorithm_col] == b)][value_col].values
                if len(va) == 1 and len(vb) == 1:
                    paired_a.append(va[0])
                    paired_b.append(vb[0])
        try:
            stat, p = wilcoxon(paired_a, paired_b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = np.nan, np.nan
        rows.append(
            {
                "AlgA": a,
                "AlgB": b,
                "statistic": round(stat, 4) if not np.isnan(stat) else np.nan,
                "p_value": round(p, 6) if not np.isnan(p) else np.nan,
                "significant": bool(p < alpha) if not np.isnan(p) else False,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Critical-difference diagram (Demšar 2006)
# ---------------------------------------------------------------------------

def critical_difference_plot(
    df: pd.DataFrame,
    alpha: float = 0.05,
    filepath: str | None = None,
    value_col: str = "Best Fitness",
    algorithm_col: str = "Algorithm",
    problem_col: str = "Problem",
    seed_col: str = "Seed",
    maximize: bool = True,
) -> None:
    """Draw a critical-difference diagram (Demšar, 2006).

    Algorithms that are **not** significantly different are connected by a
    horizontal bar.

    Parameters
    ----------
    df:
        Raw results DataFrame.
    alpha:
        Significance level for the Nemenyi test approximation.
    filepath:
        If provided, save figure to this path; otherwise show interactively.
    maximize:
        If True, higher rank = better (rank 1 is best).

    Accepts either the long or the wide (UG §9.2) input format.
    """
    df = _ensure_long(
        df, value_col=value_col, problem_col=problem_col,
        algorithm_col=algorithm_col, seed_col=seed_col,
    )
    algorithms = sorted(df[algorithm_col].unique())
    k = len(algorithms)
    problems = df[problem_col].unique()
    seeds = df[seed_col].unique()

    # Average rank per algorithm
    avg_ranks: dict[str, float] = {a: 0.0 for a in algorithms}
    n_blocks = 0
    for prob in problems:
        for seed in seeds:
            block = {}
            for a in algorithms:
                vals = df[
                    (df[problem_col] == prob)
                    & (df[seed_col] == seed)
                    & (df[algorithm_col] == a)
                ][value_col].values
                if len(vals) == 1:
                    block[a] = vals[0]
            if len(block) == k:
                # Proper Friedman/Nemenyi ranking with average (mid) ranks for
                # ties.  Rank 1 = best: for maximisation the largest value is
                # best, so rank the negated values ascending.
                names = list(block.keys())
                vals = np.array([block[a] for a in names], dtype=float)
                block_ranks = rankdata(-vals if maximize else vals, method="average")
                for alg, r in zip(names, block_ranks):
                    avg_ranks[alg] += float(r)
                n_blocks += 1

    if n_blocks == 0:
        print("CD plot: no complete blocks found, skipping.")
        return

    for a in algorithms:
        avg_ranks[a] /= n_blocks

    # Nemenyi critical difference:  CD = q_alpha * sqrt(k(k+1) / (6*N)).
    # The Nemenyi q_alpha is the Studentized-range critical value divided by
    # sqrt(2); compute it exactly for any k via scipy (added in scipy 1.7),
    # falling back to the Demsar (2006) Table 5 values for small k.
    try:
        from scipy.stats import studentized_range
        q = float(studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2.0)
    except Exception:
        q_alpha_table = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
                         6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}
        q = q_alpha_table.get(k, 3.164)
    CD = q * math.sqrt(k * (k + 1) / (6 * n_blocks))

    sorted_algs = sorted(algorithms, key=lambda a: avg_ranks[a])
    ranks = [avg_ranks[a] for a in sorted_algs]

    # ── Maximal non-significant cliques (contiguous runs within CD) ─────────
    # Because the average ranks are sorted, a set of pairwise non-significant
    # algorithms (all within CD of each other) is always a *contiguous* run whose
    # extreme ranks differ by less than CD.  Draw only the MAXIMAL such runs: a
    # run is skipped when it is fully contained in a wider one, so e.g. B-C is
    # NOT drawn separately once A-B-C is already connected.
    n_alg = len(ranks)
    reach = []
    for i in range(n_alg):
        j = i
        while j + 1 < n_alg and (ranks[j + 1] - ranks[i]) < CD:
            j += 1
        reach.append(j)
    bars = []
    last_reach = -1
    for i in range(n_alg):
        j = reach[i]
        if j > i and j > last_reach:      # non-trivial and not subsumed by a wider run
            bars.append((i, j))
            last_reach = j

    # ── Layout (Demsar 2006 style; labels staggered left/right) ─────────────
    # Best half of the algorithms have their names on the left, the worst half on
    # the right, each name on its own row with a leader line to its marker, so
    # names never overlap even when their average ranks are nearly identical.
    lowv, highv = 1, n_alg
    mid = (n_alg + 1) // 2
    row = 0.5                                     # vertical spacing between labels
    bar_gap = 0.13
    bars_span = 0.15 + len(bars) * bar_gap
    label_gap = 0.35
    axis_y = 0.5 + mid * row + label_gap + bars_span

    fig_w = max(9.0, 0.65 * n_alg + 4.0)
    fig_h = max(3.5, 0.55 * axis_y + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(lowv - 0.6, highv + 0.6)
    ax.set_ylim(0.0, axis_y + 0.9)
    ax.axis("off")
    ax.set_title(f"Critical Difference Diagram   (CD = {CD:.3f},  α = {alpha})", fontsize=11)

    # Rank axis (numbers above the line)
    ax.plot([lowv, highv], [axis_y, axis_y], color="black", lw=1.2, zorder=1)
    for t in range(lowv, highv + 1):
        ax.plot([t, t], [axis_y, axis_y + 0.08], color="black", lw=1.0, zorder=1)
        ax.text(t, axis_y + 0.12, str(t), ha="center", va="bottom", fontsize=8)
    ax.text((lowv + highv) / 2.0, axis_y + 0.40, "Average Rank",
            ha="center", va="bottom", fontsize=9)

    # CD-length indicator, just below the axis on the left
    ax.annotate("", xy=(lowv + CD, axis_y - 0.12), xytext=(lowv, axis_y - 0.12),
                arrowprops=dict(arrowstyle="|-|", color="black", lw=1.2))
    ax.text(lowv + CD / 2.0, axis_y - 0.10, "CD", ha="center", va="bottom", fontsize=8)

    # Markers on the axis
    for r in ranks:
        ax.plot(r, axis_y, "o", color="#2c7bb6", markersize=6, zorder=4)

    # Clique bars, stacked just below the axis
    for m, (i, j) in enumerate(bars):
        y = axis_y - 0.22 - m * bar_gap
        ax.plot([ranks[i] - 0.05, ranks[j] + 0.05], [y, y],
                lw=4.0, color="#d7191c", alpha=0.9, solid_capstyle="round", zorder=3)

    # Staggered labels with leader lines
    label_top = axis_y - bars_span - label_gap
    x_left, x_right = lowv - 0.5, highv + 0.5
    for l in range(mid):                          # left column: best ranks
        idx = l
        y = label_top - l * row
        ax.plot([ranks[idx], ranks[idx], x_left], [axis_y, y, y],
                color="grey", lw=0.8, zorder=2)
        ax.text(x_left - 0.05, y, sorted_algs[idx], ha="right", va="center",
                fontsize=8, clip_on=False)
    for l in range(n_alg - mid):                  # right column: worst ranks
        idx = n_alg - 1 - l
        y = label_top - l * row
        ax.plot([ranks[idx], ranks[idx], x_right], [axis_y, y, y],
                color="grey", lw=0.8, zorder=2)
        ax.text(x_right + 0.05, y, sorted_algs[idx], ha="left", va="center",
                fontsize=8, clip_on=False)

    if filepath:
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# needed inside this module
import math
