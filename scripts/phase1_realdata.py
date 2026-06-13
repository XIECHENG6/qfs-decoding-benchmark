"""
phase1_realdata.py  --  the honest mRMR-QUBO benchmark on REAL datasets.

Question: on real data, does the same picture hold?
  - naive relevance (MI top-k) and best-of-N random leave a gap, BUT
  - classical mean-field / SA / greedy already match the exact optimum
  => quantum optimisation (QAOA or VQC-QFS) would add nothing.

Suite: 10 standard ML benchmarks (data/sets/*.npz, fetched once by
fetch_datasets.py from scikit-learn and OpenML). Each is MI-pre-filtered to
N_PRE features so the exact optimum over C(N_PRE, K) subsets is tractable.
Downstream metric: SVM-RBF 5-fold CV accuracy (gap to the exact-J subset).

Output: ../results/realdata_benchmark.json + console table.
"""
import json
import os
import numpy as np
import qfs_common as q
from datasets import load_all_sets

LAM = 1.0
N_PRE = 18          # MI pre-filter -> N_PRE qubits (C(18,6) exact-able)
K = 6
SEEDS = [0, 1, 2, 3, 4]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "realdata_benchmark.json")
METHODS = ["rel", "rand", "greedy", "mf", "sa"]


def mi_prefilter(X, y, n_pre):
    Xb = q._discretize(X)
    r_full = np.array([q._mi(Xb[:, i], y) for i in range(X.shape[1])])
    keep = np.argsort(-r_full)[:n_pre]
    return X[:, keep]


def bench(X, y):
    Xp = mi_prefilter(X, y, min(N_PRE, X.shape[1]))
    r, R = q.build_qubo(Xp, y)
    _, combos, best = q.exact_solve(r, R, K, LAM)
    Sx = combos[best]
    gaps = {m: [] for m in METHODS}
    ex_accs = []
    for sd in SEEDS:
        ex = q.downstream_svm(Xp, y, Sx, seed=sd)
        ex_accs.append(ex)
        subs = {
            "rel": q.relevance_topk(r, K),
            "rand": q.random_mrmr(r, R, K, LAM, seed=sd),
            "greedy": q.greedy_mrmr(r, R, K, LAM),
            "mf": q.classical_meanfield(r, R, K, LAM, seed=sd),
            "sa": q.simulated_annealing(r, R, K, LAM, seed=sd),
        }
        for m in METHODS:
            gaps[m].append(ex - q.downstream_svm(Xp, y, subs[m], seed=sd))
    return {
        "n_features_used": int(Xp.shape[1]), "n_samples": int(Xp.shape[0]),
        "svm_exact": float(np.mean(ex_accs)),
        "gap_mean": {m: float(np.mean(v)) for m, v in gaps.items()},
        "gap_std": {m: float(np.std(v)) for m, v in gaps.items()},
    }


def main():
    print(f"mRMR-QUBO on 10 real datasets (MI pre-filter -> {N_PRE}, k={K}, "
          f"lambda={LAM}, {len(SEEDS)} seeds)\n")
    hdr = f"{'dataset':<13}{'feat':>5}{'SVM*':>8}" + "".join(f"{'d_'+m:>9}" for m in METHODS)
    print(hdr); print("-" * len(hdr))
    results = {"config": {"lam": LAM, "n_pre": N_PRE, "k": K, "seeds": SEEDS,
                          "methods": METHODS, "metric": "SVM-RBF 5-fold CV"},
               "datasets": {}}
    for name, (X, y) in load_all_sets().items():
        res = bench(X, y)
        results["datasets"][name] = res
        g = res["gap_mean"]
        print(f"{name:<13}{res['n_features_used']:>5}{res['svm_exact']:>8.3f}" +
              "".join(f"{g[m]:>+9.3f}" for m in METHODS))
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {os.path.relpath(OUT, os.path.join(HERE, '..'))}")
    print("Reading: d_method = SVM(exact-J subset) - SVM(method); + => method worse.")
    print("Claim: d_rel/d_rand > 0 (relevance/random leave a gap) while "
          "d_sa ~ 0 and d_mf ~ 0 (classical already optimal) on essentially every set.")


if __name__ == "__main__":
    main()
