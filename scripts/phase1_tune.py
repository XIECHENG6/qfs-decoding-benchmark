"""
phase1_tune.py  --  find a CONFIG where best-of-N random is convincingly beaten
(dSVM_rand large) while classical mean-field still solves it (dSVM_mf ~ 0).
That combination is the honest headline: "random fails -> real optimisation is
needed, but a TRIVIAL classical mean-field optimiser -- not quantum -- suffices."

Lever: problem size n and the random shot budget (coverage). Bigger n / fewer
shots -> random covers less of the space -> larger random gap.
"""
import numpy as np
import qfs_common as q

LAM = 1.0
SEEDS = [0, 1, 2, 3, 4, 5]

# (n_A, n_B, n_C, d_B, k)  -> n = n_A+n_B+n_C
CONFIGS = [
    (6, 10, 6, 6, 7),    # n=22  (baseline)
    (7, 11, 6, 7, 8),    # n=24
    (8, 12, 6, 8, 9),    # n=26
]
SHOTS = [4096, 1024]


def gaps_for(gen_cfg, k, shots):
    n_A, n_B, n_C, d_B = gen_cfg
    out = {m: [] for m in ["rel", "rand", "mf", "sa"]}
    for sd in SEEDS:
        X, y, g = q.make_hard_instance(n_A=n_A, n_B=n_B, n_C=n_C, d_B=d_B,
                                       n_samples=800, seed=sd)
        r, R = q.build_qubo(X, y)
        Ja, combos, best = q.exact_solve(r, R, k, LAM)
        ex = q.downstream_svm(X, y, combos[best], seed=sd)
        out["rel"].append(ex - q.downstream_svm(X, y, q.relevance_topk(r, k), seed=sd))
        out["rand"].append(ex - q.downstream_svm(X, y, q.random_mrmr(r, R, k, LAM, shots=shots, seed=sd), seed=sd))
        out["mf"].append(ex - q.downstream_svm(X, y, q.classical_meanfield(r, R, k, LAM, seed=sd), seed=sd))
        out["sa"].append(ex - q.downstream_svm(X, y, q.simulated_annealing(r, R, k, LAM, seed=sd), seed=sd))
    return {m: float(np.mean(v)) for m, v in out.items()}


def main():
    print(f"{'n':>4}{'k':>4}{'shots':>7}{'dSVM_rel':>10}{'dSVM_rand':>11}{'dSVM_mf':>9}{'dSVM_sa':>9}")
    print("-" * 54)
    for (n_A, n_B, n_C, d_B, k) in CONFIGS:
        n = n_A + n_B + n_C
        for shots in SHOTS:
            d = gaps_for((n_A, n_B, n_C, d_B), k, shots)
            flag = "  <== good" if (d["rand"] >= 0.04 and d["mf"] <= 0.01) else ""
            print(f"{n:>4}{k:>4}{shots:>7}{d['rel']:>10.3f}{d['rand']:>11.3f}"
                  f"{d['mf']:>9.3f}{d['sa']:>9.3f}{flag}")


if __name__ == "__main__":
    main()
