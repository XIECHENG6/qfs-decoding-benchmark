"""
phase4_multiclf.py -- classifier-robustness check for the real-data benchmark
(paper Sec. "Robustness across classifiers").

Question: is the no-advantage conclusion an artefact of the SVM probe?
Protocol: identical to phase1_realdata.py (MI pre-filter -> 18 features, k=6,
lambda=1, same solvers, same 5 CV seeds), but the downstream accuracy is
measured with FOUR classifiers: SVM-RBF, k-NN, logistic regression, and
random forest. The SVM column therefore reproduces the real-data table of the
paper, validating the script against the frozen phase1 numbers.

Expected picture (paper claims):
  - method ordering preserved under every classifier: relevance-only worst,
    greedy / mean-field / SA cluster near the exact optimum;
  - mean-field tracks SA under all four classifiers (any residual gap to the
    exact-J subset appears equally for SA, which provably reaches J*);
  - hence no quantum variant (bounded by mean-field, phase2) can become
    advantageous under any classifier.

Output: ../results/phase4_multiclf.json + console table.
"""
import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import qfs_common as q
from datasets import load_all_sets

LAM = 1.0
N_PRE = 18
K = 6
SEEDS = [0, 1, 2, 3, 4]

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "phase4_multiclf.json")


def make_clf(name, seed):
    if name == "svm":
        return SVC(kernel="rbf")
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "logreg":
        return LogisticRegression(max_iter=2000)
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, random_state=seed)
    raise ValueError(name)


def downstream(X, y, S, clf_name, seed):
    """5-fold stratified CV accuracy of the given classifier on subset S
    (same protocol as qfs_common.downstream_svm)."""
    Xs = StandardScaler().fit_transform(X[:, list(S)])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(make_clf(clf_name, seed), Xs, y, cv=cv).mean())


def mi_prefilter(X, y, n_pre):
    Xb = q._discretize(X)
    r_full = np.array([q._mi(Xb[:, i], y) for i in range(X.shape[1])])
    keep = np.argsort(-r_full)[:n_pre]
    return X[:, keep]


CLFS = ["svm", "knn", "logreg", "rf"]
METHODS = ["rel", "rand", "greedy", "mf", "sa"]


def bench(X, y):
    Xp = mi_prefilter(X, y, min(N_PRE, X.shape[1]))
    r, R = q.build_qubo(Xp, y)
    _, combos, best = q.exact_solve(r, R, K, LAM)
    Sx = list(combos[best])

    out = {}
    for clf in CLFS:
        ex_accs, gaps = [], {m: [] for m in METHODS}
        for sd in SEEDS:
            subsets = {
                "rel": q.relevance_topk(r, K),
                "rand": q.random_mrmr(r, R, K, LAM, seed=sd),
                "greedy": q.greedy_mrmr(r, R, K, LAM),
                "mf": q.classical_meanfield(r, R, K, LAM, seed=sd),
                "sa": q.simulated_annealing(r, R, K, LAM, seed=sd),
            }
            ex = downstream(Xp, y, Sx, clf, sd)
            ex_accs.append(ex)
            for m in METHODS:
                gaps[m].append(ex - downstream(Xp, y, subsets[m], clf, sd))
        out[clf] = {
            "exact_acc": float(np.mean(ex_accs)),
            "gap_mean": {m: float(np.mean(v)) for m, v in gaps.items()},
            "gap_std": {m: float(np.std(v)) for m, v in gaps.items()},
        }
    return out, Sx


def main():
    print(f"phase4: real-data benchmark x 4 classifiers "
          f"(MI pre-filter -> {N_PRE}, k={K}, lambda={LAM}, seeds={SEEDS})\n")
    results = {"config": {"lam": LAM, "n_pre": N_PRE, "k": K, "seeds": SEEDS,
                          "classifiers": CLFS, "methods": METHODS},
               "datasets": {}}
    for name, (X, y) in load_all_sets().items():
        res, Sx = bench(X, y)
        results["datasets"][name] = {"exact_subset": [int(i) for i in Sx], **res}
        print(f"=== {name} ===")
        print(f"{'clf':<8}{'acc_exact':>10}" +
              "".join(f"{'d_' + m:>10}" for m in METHODS))
        for clf in CLFS:
            row = res[clf]
            print(f"{clf:<8}{row['exact_acc']:>10.3f}" +
                  "".join(f"{row['gap_mean'][m]:>10.3f}" for m in METHODS))
        print()
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved -> {os.path.relpath(OUT, os.path.join(HERE, '..'))}")
    print("\nReading: d_<method> = acc(exact-J subset) - acc(method subset); "
          "positive => method worse than exact.")
    print("Claim check: d_rel largest in every row; d_greedy/d_mf/d_sa ~ 0; "
          "d_mf ~ d_sa per row (mean-field tracks SA).")


if __name__ == "__main__":
    main()
