"""
phase1_realdata.py  --  extend the honest mRMR-QUBO benchmark to REAL datasets.

Question: on real data, does the same picture hold?
  - naive relevance (MI top-k) and best-of-N random leave a gap, BUT
  - classical mean-field / SA / greedy already match the exact optimum
  => quantum optimisation (QAOA or VQC-QFS) would add nothing.

WDBC ships with scikit-learn (offline). Ionosphere / Sonar are loaded from UCI via
uci_loader (downloaded once, then cached offline under ../data/uci/).
"""
import numpy as np
import qfs_common as q
import uci_loader

LAM = 1.0
N_PRE = 18          # MI pre-filter original features -> N_PRE qubits (C(18,k) exact-able)
K = 6
SEEDS = [0, 1, 2, 3, 4]


def load_datasets():
    return uci_loader.load_all()


def mi_prefilter(X, y, n_pre):
    Xb = q._discretize(X)
    r_full = np.array([q._mi(Xb[:, i], y) for i in range(X.shape[1])])
    keep = np.argsort(-r_full)[:n_pre]
    return X[:, keep]


def bench(X, y):
    Xp = mi_prefilter(X, y, min(N_PRE, X.shape[1]))
    r, R = q.build_qubo(Xp, y)
    Ja, combos, best = q.exact_solve(r, R, K, LAM)
    Sx = combos[best]
    res = {m: [] for m in ["rel", "rand", "greedy", "mf", "sa"]}
    for sd in SEEDS:
        ex = q.downstream_svm(Xp, y, Sx, seed=sd)
        res["rel"].append(ex - q.downstream_svm(Xp, y, q.relevance_topk(r, K), seed=sd))
        res["rand"].append(ex - q.downstream_svm(Xp, y, q.random_mrmr(r, R, K, LAM, seed=sd), seed=sd))
        res["greedy"].append(ex - q.downstream_svm(Xp, y, q.greedy_mrmr(r, R, K, LAM), seed=sd))
        res["mf"].append(ex - q.downstream_svm(Xp, y, q.classical_meanfield(r, R, K, LAM, seed=sd), seed=sd))
        res["sa"].append(ex - q.downstream_svm(Xp, y, q.simulated_annealing(r, R, K, LAM, seed=sd), seed=sd))
    ex_acc = np.mean([q.downstream_svm(Xp, y, Sx, seed=sd) for sd in SEEDS])
    return ex_acc, {m: float(np.mean(v)) for m, v in res.items()}


def main():
    print(f"mRMR-QUBO on real data (MI pre-filter -> {N_PRE} features, k={K}, lambda={LAM})\n")
    print(f"{'dataset':<12}{'SVM_exact':>10}{'dSVM_rel':>10}{'dSVM_rand':>11}"
          f"{'dSVM_greedy':>12}{'dSVM_mf':>9}{'dSVM_sa':>9}")
    print("-" * 73)
    for name, (X, y) in load_datasets().items():
        ex, d = bench(X, y)
        print(f"{name:<12}{ex:>10.3f}{d['rel']:>10.3f}{d['rand']:>11.3f}"
              f"{d['greedy']:>12.3f}{d['mf']:>9.3f}{d['sa']:>9.3f}")
    print("\nReading: dSVM = exact - method (positive => method is worse than exact).")
    print("If mf/sa ~ 0 while rel/rand > 0  ->  classical mean-field already solves it"
          " => no room for quantum.")


if __name__ == "__main__":
    main()
