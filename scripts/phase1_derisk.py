"""
phase1_derisk.py  --  Pure-classical de-risk for the mRMR-QUBO rework (Path A).

Run this FIRST (locally or Colab; deps: numpy / scikit-learn / matplotlib).
It answers two go/no-go questions BEFORE we touch any quantum code:

  CHECK 1  (difficulty): On the new mRMR-QUBO instance, do relevance-only (MI top-k)
           and best-of-N random BOTH leave a real gap vs the exact optimum -- in
           QUBO objective AND in downstream SVM accuracy? If yes, the problem is
           genuinely hard and "can't beat random search" is FIXED.

  CHECK 2  (room for quantum): How close does the CLASSICAL mean-field optimiser
           (= VQC-QFS minus the quantum circuit) get to exact? And does a strong
           classical solver (SA) basically solve it? This bounds what the quantum
           ansatz could possibly add in Phase 2.

Outputs (under ../):  data/instance_seed0.npz , results/phase1_summary.json ,
                      figures/phase1_lambda_sweep.png
"""

import os, json
import numpy as np

import qfs_common as q   # same folder

# ----------------------------------------------------------------------------
CONFIG = dict(
    n_A=6, n_B=10, n_C=6,      # -> n = 22 features
    n_samples=800,
    k=7,                       # cardinality (C(22,7)=170544 -> exact is fast)
    shots=4096,                # random / sampling budget (matches the paper)
    lam_grid=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    sweep_seeds=[0, 1, 2, 3, 4],   # instances averaged for the lambda sweep
    detail_seed=0,             # instance used for the detailed per-method report
)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for d in ("data", "results", "figures"):
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)


def run_all_methods(X, y, r, R, k, lam, groups, shots, seed=0):
    J_all, combos, best = q.exact_solve(r, R, k, lam)
    sets = {
        "EXACT":         np.array(sorted(combos[best])),
        "MI-top-k":      q.relevance_topk(r, k),
        "Random-mRMR":   q.random_mrmr(r, R, k, lam, shots=shots, seed=seed),
        "Greedy-mRMR":   q.greedy_mrmr(r, R, k, lam),
        "SA":            q.simulated_annealing(r, R, k, lam, seed=seed),
        "MeanField(cl)": q.classical_meanfield(r, R, k, lam, seed=seed),
    }
    Jstar = float(J_all[best])
    rows = {}
    for name, S in sets.items():
        Jv = q.eval_J(S, r, R, lam)
        rows[name] = dict(
            J=Jv,
            gap=Jstar - Jv,
            pct=q.percentile_of(Jv, J_all),
            svm=q.downstream_svm(X, y, S, seed=seed),
            comp=q.composition(S, groups),
            S=[int(i) for i in S],
        )
    return rows, Jstar


def detailed_report(lam):
    n = CONFIG["n_A"] + CONFIG["n_B"] + CONFIG["n_C"]
    print("\n" + "=" * 78)
    print(f" DETAILED REPORT  (instance seed={CONFIG['detail_seed']}, "
          f"n={n}, k={CONFIG['k']}, lambda={lam})")
    print("=" * 78)
    X, y, groups = q.make_hard_instance(CONFIG["n_A"], CONFIG["n_B"], CONFIG["n_C"],
                                        CONFIG["n_samples"], seed=CONFIG["detail_seed"])
    print(f" class balance: y=1 -> {y.mean():.2f}")
    r, R = q.build_qubo(X, y)
    rows, Jstar = run_all_methods(X, y, r, R, CONFIG["k"], lam, groups,
                                  CONFIG["shots"], seed=CONFIG["detail_seed"])
    print(f"\n J* (exact optimum) = {Jstar:.4f}\n")
    hdr = f" {'method':<14}{'J':>9}{'gap':>9}{'pct':>7}{'SVM':>8}   A/B/b/C"
    print(hdr); print(" " + "-" * (len(hdr) - 1))
    for name, d in rows.items():
        c = d["comp"]
        print(f" {name:<14}{d['J']:>9.4f}{d['gap']:>9.4f}{d['pct']:>6.1f}%{d['svm']:>8.3f}"
              f"   {c['A']}/{c['B']}/{c['b']}/{c['C']}")
    return rows, Jstar, X, y, r, R, groups


def lambda_sweep():
    print("\n" + "=" * 78)
    print(f" LAMBDA SWEEP  (averaged over {len(CONFIG['sweep_seeds'])} instances)")
    print("=" * 78)
    print(f" {'lambda':>7}{'dSVM(rel)':>11}{'dSVM(rand)':>12}{'dSVM(MF)':>10}"
          f"{'dSVM(SA)':>10}{'rand_pct':>10}")
    sweep = []
    for lam in CONFIG["lam_grid"]:
        acc = {kk: [] for kk in ["EXACT", "MI-top-k", "Random-mRMR", "MeanField(cl)", "SA"]}
        randpct = []
        for sd in CONFIG["sweep_seeds"]:
            X, y, groups = q.make_hard_instance(CONFIG["n_A"], CONFIG["n_B"], CONFIG["n_C"],
                                                CONFIG["n_samples"], seed=sd)
            r, R = q.build_qubo(X, y)
            rows, _ = run_all_methods(X, y, r, R, CONFIG["k"], lam, groups,
                                      CONFIG["shots"], seed=sd)
            for kk in acc:
                acc[kk].append(rows[kk]["svm"])
            randpct.append(rows["Random-mRMR"]["pct"])
        m = {kk: float(np.mean(v)) for kk, v in acc.items()}
        rec = dict(lam=lam,
                   dSVM_rel=m["EXACT"] - m["MI-top-k"],
                   dSVM_rand=m["EXACT"] - m["Random-mRMR"],
                   dSVM_mf=m["EXACT"] - m["MeanField(cl)"],
                   dSVM_sa=m["EXACT"] - m["SA"],
                   rand_pct=float(np.mean(randpct)),
                   svm_exact=m["EXACT"])
        sweep.append(rec)
        print(f" {lam:>7.1f}{rec['dSVM_rel']:>11.3f}{rec['dSVM_rand']:>12.3f}"
              f"{rec['dSVM_mf']:>10.3f}{rec['dSVM_sa']:>10.3f}{rec['rand_pct']:>9.1f}%")
    return sweep


def verdicts(sweep):
    best = max(sweep, key=lambda s: s["dSVM_rel"])   # lambda with strongest trap
    lam_star = best["lam"]
    print("\n" + "=" * 78)
    print(" VERDICTS")
    print("=" * 78)
    print(f" recommended lambda* = {lam_star}  (max relevance-trap gap)")

    c1a = best["dSVM_rel"] >= 0.03
    c1b = best["dSVM_rand"] >= 0.02
    print("\n CHECK 1  (problem is hard / beats random):")
    print(f"   relevance-only gap dSVM={best['dSVM_rel']:.3f}  -> {'PASS' if c1a else 'FAIL'}")
    print(f"   random       gap dSVM={best['dSVM_rand']:.3f}  -> {'PASS' if c1b else 'FAIL'}")
    if c1a and c1b:
        print("   => mRMR-QUBO instance is genuinely hard. 'Can't beat random' is FIXED.")
    else:
        print("   => TOO EASY. Increase n / lower k / raise rho_A,w_A (stronger trap),")
        print("      or lower 'shots' so random covers less of the space; then rerun.")

    print(f"\n CHECK 2  (room for the quantum ansatz, at lambda*={lam_star}):")
    print(f"   classical mean-field gap dSVM={best['dSVM_mf']:.3f}")
    print(f"   strong classical SA gap  dSVM={best['dSVM_sa']:.3f}")
    if best["dSVM_mf"] <= 0.005:
        print("   => classical mean-field already ~matches exact. The quantum circuit can")
        print("      only win via EXPRESSIBILITY (entanglement ablation must show a gap),")
        print("      not via better optimisation. Plan Phase-2 around that.")
    else:
        print("   => mean-field leaves a gap -> Phase-2 question is whether the quantum")
        print("      ring-entangled ansatz closes it vs the no-entanglement ablation.")
    if best["dSVM_sa"] <= 0.005:
        print("   NOTE: SA ~solves it -> do NOT expect quantum > classical; the honest")
        print("         contribution is O(n) mean-field competitiveness + decoding theory.")
    return lam_star


def save_artifacts(lam_star, sweep, detail_rows, Jstar, X, y, r, R, groups):
    np.savez(os.path.join(ROOT, "data", "instance_seed0.npz"),
             X=X, y=y, r=r, R=R, groups=groups,
             k=CONFIG["k"], lam=lam_star,
             exact_S=np.array(detail_rows["EXACT"]["S"]))
    with open(os.path.join(ROOT, "results", "phase1_summary.json"), "w") as f:
        json.dump(dict(config=CONFIG, lam_star=lam_star, Jstar=Jstar,
                       detail=detail_rows, sweep=sweep), f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lams = [s["lam"] for s in sweep]
        plt.figure(figsize=(6, 4))
        plt.plot(lams, [s["dSVM_rel"] for s in sweep], "o-", label="EXACT - MI-top-k")
        plt.plot(lams, [s["dSVM_rand"] for s in sweep], "s-", label="EXACT - Random-mRMR")
        plt.plot(lams, [s["dSVM_mf"] for s in sweep], "^-", label="EXACT - MeanField(cl)")
        plt.plot(lams, [s["dSVM_sa"] for s in sweep], "d-", label="EXACT - SA")
        plt.axhline(0.0, color="grey", lw=0.8)
        plt.xlabel(r"redundancy weight $\lambda$")
        plt.ylabel("downstream SVM gap vs exact")
        plt.title("Phase-1: how hard is the mRMR-QUBO instance?")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(ROOT, "figures", "phase1_lambda_sweep.png"), dpi=150)
        print("\n saved figure -> figures/phase1_lambda_sweep.png")
    except Exception as e:
        print(f"\n (figure skipped: {e})")
    print(" saved data    -> data/instance_seed0.npz")
    print(" saved summary -> results/phase1_summary.json")


def main():
    sweep = lambda_sweep()
    lam_star = max(sweep, key=lambda s: s["dSVM_rel"])["lam"]
    detail_rows, Jstar, X, y, r, R, groups = detailed_report(lam_star)
    lam_star = verdicts(sweep)
    save_artifacts(lam_star, sweep, detail_rows, Jstar, X, y, r, R, groups)
    print("\n done.\n")


if __name__ == "__main__":
    main()
