#!/usr/bin/env python3
"""Smoke-test all seven problems and the fitness-maximization convention.

Instantiates TSP, PFSP, LOP, QAP, MIS, MaxCut, MVC (random generators and the
in-memory `load_*` readers), checks the evaluate_* helpers return the natural
positive objective, and verifies the graph problems on hand-built instances with
known optima.

Run:  python3 examples/problems_smoke.py
"""
import sys
import numpy as np

from perm_pateda.functions import (
    create_random_tsp, create_random_pfsp, create_random_lop, create_random_qap,
    create_mis_from_edges, create_max_cut_from_edges, create_mvc_from_edges,
    load_qaplib_instance, load_lolib_instance, load_taillard_instance,
)

failures = []


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        failures.append(name)


print("Classical problems: fitness = maximised, evaluate_* = natural value")
tsp = create_random_tsp(10, seed=1)
p = np.arange(10)
check("TSP __call__ == -evaluate_distance", np.isclose(tsp(p), -tsp.evaluate_distance(p)))

qap = create_random_qap(10, seed=1)
check("QAP __call__ == -evaluate_cost", np.isclose(qap(p), -qap.evaluate_cost(p)))

pfsp = create_random_pfsp(10, 5, seed=1)
check("PFSP __call__ == -evaluate_makespan (makespan objective)",
      np.isclose(pfsp(p), -pfsp.evaluate_makespan(p)))

lop = create_random_lop(10, seed=1)
check("LOP __call__ == evaluate_objective (already maximised)",
      np.isclose(lop(p), lop.evaluate_objective(p)))

print("\nIn-memory benchmark readers (load_*)")
flow = np.array([[0, 2], [2, 0]]); dist = np.array([[0, 3], [3, 0]])
qap2 = load_qaplib_instance(flow, dist)
check("load_qaplib_instance builds a working QAP", qap2.evaluate_cost(np.array([0, 1])) >= 0)
lop2 = load_lolib_instance(np.array([[0, 5], [1, 0]]))
check("load_lolib_instance builds a working LOP", lop2.evaluate_objective(np.array([0, 1])) >= 0)
pf2 = load_taillard_instance(np.array([[1, 2], [3, 4]]))
check("load_taillard_instance builds a working PFSP", pf2.evaluate_makespan(np.array([0, 1])) > 0)

print("\nGraph problems (permutation picture) on hand-built instances")
# path 0-1-2-3-4: max independent set = {0,2,4} (size 3)
mis = create_mis_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
check("MIS path-5 optimum == 3", mis(np.array([0, 2, 4, 1, 3])) == 3.0)
# 4-cycle 0-1-2-3-0: max cut = 4 (bipartite)
mc = create_max_cut_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
check("MaxCut C4 optimum == 4", mc(np.array([0, 2, 1, 3])) == 4.0)
# 4-cycle: min vertex cover = 2  (fitness is -k)
mvc = create_mvc_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
check("MVC C4 optimum == -2", mvc(np.array([0, 2, 1, 3])) == -2.0)

print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
sys.exit(1 if failures else 0)
