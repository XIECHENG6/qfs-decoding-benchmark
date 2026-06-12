"""
phase_d_hardness_control.py  --  Tier-1 (D): hardness control / "is the quantum
implementation just bad, or is the problem too easy?"

Rebuttal + delineation. We run the SAME solver suite on two problem families at the
same sizes:
  * mRMR   : the redundancy-aware QUBO of the paper (positive R) -- mean-field-easy.
  * FRUST  : a frustrated cardinality-constrained QUBO, max sum r_i x_i
             - lam sum_{i<j} W_ij x_i x_j with W RANDOM-SIGN (frustration) -- the
             regime where a product-of-marginals (mean-field) relaxation provably
             struggles. (Verified: free mean-field leaves a real gap here.)

Reusing qfs_common's generic solvers (they take any symmetric coupling matrix), the
multi-restart VQC of phase2, and the FAIR (normalised + multistart, per Tier-1 A) QAOA
of phase_a. We report the optimality gap dJ of each method on each family.

Reading:
  - mean-field gap: ~0 on mRMR, >0 on FRUST  => mRMR is mean-field-solvable, FRUST is not.
  - the VQC (a mean-field ansatz) tracks mean-field on BOTH => its limit is the ansatz,
    not a bug; the full-cost-layer QAOA is the test of whether an entangling cost layer
    recovers the frustration that mean-field misses -- the direction a useful quantum FS
    method would have to take (cf. Section 6 delineation).
  - SA (strong classical) anchors the exact optimum on both.

Local-friendly at n=12,14. Saves incrementally to results/hardness_control.json.
"""
import os, json, time
import numpy as np
import qfs_common as q
import phase2_ablation_strong as p2
import phase_a_qaoa_depth as A

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "hardness_control.json")

CONFIG = dict(
    ns=[12, 14],
    seeds=list(range(8)),
    lam=1.0,
    k_frac=0.42,
    n_samples=600,
    qaoa=dict(p=3, mu=2.0, steps=150, lr=0.04, restarts=6),  # FAIR setting (Tier-1 A: A1)
)


def make_frustrated(n, seed):
    """Frustrated cardinality QUBO coefficients: relevance r in [0,1], symmetric
    random-SIGN coupling W normalised to [-1,1]. eval_J(S,r,W,lam) is then frustrated."""
    rng = np.random.default_rng(1000 + seed)
    r = rng.uniform(0, 1, n)
    W = rng.standard_normal((n, n)); W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0); W /= (np.abs(W).max() + 1e-12)
    return r, W


def make_mrmr(n, seed, cfg):
    g = p2.scaled_groups(n)
    X, y, _ = q.make_hard_instance(n_A=g["n_A"], n_B=g["n_B"], n_C=g["n_C"],
                                   d_B=g["d_B"], n_samples=cfg["n_samples"], seed=seed)
    return q.build_qubo(X, y)


def solve_all(r, R, k, lam, seed, cfg):
    """dJ for SA, MeanField, VQC(ring), QAOA(fair) on a generic (r,R) instance."""
    Ja, combos, best = q.exact_solve(r, R, k, lam)
    Jx = float(Ja[best])
    out = {}
    out["SA"] = Jx - q.eval_J(q.simulated_annealing(r, R, k, lam, seed=seed), r, R, lam)
    out["MeanField"] = Jx - q.eval_J(q.classical_meanfield(r, R, k, lam, seed=seed), r, R, lam)
    out["VQC(ring)"] = Jx - q.eval_J(p2.vqc_select(r, R, k, lam, "ring", seed, p2.CONFIG["vqc"]), r, R, lam)
    qc = cfg["qaoa"]
    res = A.optimise_qaoa(r, R, k, lam, qc["p"], qc["mu"], qc["steps"], qc["lr"],
                          qc["restarts"], "random", True, seed, 4096)
    out["QAOA(fair)"] = Jx - q.eval_J(res["_S"], r, R, lam)
    return {m: float(v) for m, v in out.items()}


def main():
    t0 = time.time(); c = CONFIG
    records = []
    if os.path.exists(OUT):
        try:
            records = json.load(open(OUT, encoding="utf-8")).get("records", [])
            print(f"resuming: {len(records)} records", flush=True)
        except Exception:
            pass
    done = {(r["problem"], r["n"], r["seed"]) for r in records}

    def save():
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(dict(config=c, records=records), f, indent=2)

    print(f"Phase-D hardness control  ns={c['ns']}  seeds={len(c['seeds'])}  "
          f"(mRMR vs frustrated)\n", flush=True)
    for problem in ["mRMR", "frustrated"]:
        for n in c["ns"]:
            k = max(2, round(c["k_frac"] * n))
            for sd in c["seeds"]:
                if (problem, n, sd) in done:
                    continue
                if problem == "mRMR":
                    r, R = make_mrmr(n, sd, c)
                else:
                    r, R = make_frustrated(n, sd)
                gaps = solve_all(r, R, k, c["lam"], sd, c)
                rec = dict(problem=problem, n=n, k=k, seed=sd, **gaps)
                records.append(rec); save()
                print(f"  {problem:<10} n={n} seed={sd}  " +
                      "  ".join(f"{m}={gaps[m]:.3f}" for m in gaps) +
                      f"   [{time.time()-t0:.0f}s]", flush=True)

    # summary
    import collections
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        for m in ("SA", "MeanField", "VQC(ring)", "QAOA(fair)"):
            if m in r:
                agg[(r["problem"], r["n"])][m].append(r[m])
    print("\n" + "=" * 76)
    print(" mean optimality gap dJ  (smaller=better, 0=exact)")
    print(" " + "-" * 74)
    print(f" {'problem':<11}{'n':>3}  {'SA':>8}{'MeanField':>11}{'VQC(ring)':>11}{'QAOA(fair)':>12}")
    for (problem, n) in sorted(agg):
        a = agg[(problem, n)]
        print(f" {problem:<11}{n:>3}  " +
              "".join(f"{np.mean(a[m]):>{w}.3f}" for m, w in
                      [("SA", 8), ("MeanField", 11), ("VQC(ring)", 11), ("QAOA(fair)", 12)]))
    save()
    print("=" * 76)
    print(" READING: MeanField gap ~0 on mRMR but >0 on frustrated => mRMR is mean-field-")
    print(" solvable; the gap quantum methods must beat only exists off the mRMR family.")
    print(f"\nsaved -> {OUT}\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
