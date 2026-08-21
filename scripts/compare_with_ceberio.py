#!/usr/bin/env python3
"""
compare_with_ceberio.py
--------------------------

    python3 compare_with_ceberio.py results.csv ceberio_table2_reference.csv

"""


import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

##Algorithms we can compare with ceberio
TO_COMPARE = {
    "Mallows-K": "Mk",
    "Mallows-C": "Mc",
    "Mallows-U": "Mu",
    "GM-K": "GMk",
    "GM-C": "GMc",
}
BETTER = {"LOP": "max", "QAP": "min", "PFSP": "min", "TSP": "min"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_csv", type=Path, help="output of parse_summary_results.py")
    p.add_argument("ceberio_csv", type=Path, help="ceberio_table2_reference.csv")
    p.add_argument("--flag-threshold", type=float, default=0.05, help="flag instance/algorithm pairs whose |our_ARPD - Ceberio_ARPD| " 
                   "exceeds this value (default: 0.05)")
    p.add_argument("--output", type=Path, default=Path("comparison_with_ceberio.csv"))
    return p.parse_args()


def load_ceberio_reference(path: Path) -> dict:
    """Returns {(problem, instance): {Mk: v, Mc: v, Mu: v, GMk: v, GMc: v}}."""
    ref = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["problem"], row["instance"])
            values = {}
            for col in ("Mk", "Mc", "Mu", "GMk", "GMc"):
                v = row[col].strip()
                values[col] = float(v) if v else None
            ref[key] = values
    return ref


def load_our_results(path: Path) -> dict:
    """Returns {(problem, instance, algorithm): [best_fitness, ...]} for the three overlapping algorithms only."""
    data = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            algo = row["algorithm"]
            if algo not in TO_COMPARE:
                continue
            problem = row["problem"]
            instance = row["instance"]
            prefix = f"{problem}-"
            if instance.startswith(prefix):
                instance = instance[len(prefix):]
            try:
                fit = float(row["best_fitness"])
            except ValueError:
                continue
            data[(problem, instance, algo)].append(fit)
    return data


def main():
    args = parse_args()
    ceberio_ref = load_ceberio_reference(args.ceberio_csv)
    ours = load_our_results(args.results_csv)

    per_instance_all = defaultdict(list)
    for (problem, instance, algo), fits in ours.items():
        per_instance_all[(problem, instance)].extend(fits)

    rows = []
    for (problem, instance, algo), fits in ours.items():
        if problem not in BETTER:
            continue
        key = (problem, instance)
        if key not in ceberio_ref:
            continue  
        ceberio_col = TO_COMPARE[algo]
        ceberio_val = ceberio_ref[key][ceberio_col]
        if ceberio_val is None:
            continue

        avg_res = mean(fits)
        all_fits_this_instance = per_instance_all[key]
        best_restricted = (max(all_fits_this_instance) if BETTER[problem] == "max"
                            else min(all_fits_this_instance))
        if best_restricted == 0:
            continue
        our_arpd = abs(avg_res - best_restricted) / abs(best_restricted)

        rows.append({
            "problem": problem, "instance": instance, "algorithm": algo,
            "our_arpd": our_arpd, "ceberio_arpd": ceberio_val,
            "diff": our_arpd - ceberio_val,
        })

    if not rows:
        print("No matching (problem, instance, algorithm) rows found -- check that "
              "results.csv contains Mallows-K / Mallows-C / GM-C runs on the exact "
              "Ceberio-benchmark instance names.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["problem", "instance", "algorithm",
                                           "our_arpd", "ceberio_arpd", "diff"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}\n")

    print("=== Summary: mean |our_ARPD - Ceberio_ARPD| and correlation, per problem x algorithm ===")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["problem"], r["algorithm"])].append(r)

    for (problem, algo), grp in sorted(groups.items()):
        diffs = [abs(r["diff"]) for r in grp]
        ours_vals = [r["our_arpd"] for r in grp]
        ceb_vals = [r["ceberio_arpd"] for r in grp]
        n = len(grp)
        mean_abs_diff = mean(diffs)

        mx, my = mean(ours_vals), mean(ceb_vals)
        cov = sum((a - mx) * (b - my) for a, b in zip(ours_vals, ceb_vals))
        sx = sum((a - mx) ** 2 for a in ours_vals) ** 0.5
        sy = sum((b - my) ** 2 for b in ceb_vals) ** 0.5
        corr = cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")

        print(f"{problem:6s} {algo:12s} n={n:3d}  mean|diff|={mean_abs_diff:.4f}  "
              f"corr={corr:.3f}")

    flagged = [r for r in rows if abs(r["diff"]) > args.flag_threshold]
    print(f"\n=== Flagged instance/algorithm pairs (|diff| > {args.flag_threshold}) ===")
    if not flagged:
        print("None -- our implementation matches Ceberio's published results closely.")
    else:
        for r in sorted(flagged, key=lambda r: -abs(r["diff"])):
            print(f"  {r['problem']:6s} {r['instance']:16s} {r['algorithm']:12s} "
                  f"ours={r['our_arpd']:.4f}  ceberio={r['ceberio_arpd']:.4f}  "
                  f"diff={r['diff']:+.4f}")


if __name__ == "__main__":
    main()