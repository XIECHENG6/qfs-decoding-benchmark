"""
qaoa_fair_rerun.py  --  OPTIONAL: re-run the scaling QAOA-mRMR row with the FAIR
optimisation setting identified in Tier-1 A (Hamiltonian-normalised, 6 restarts), so the
tab:scaling QAOA row reflects comparable effort to the multi-restart VQC rows rather than
a single random start.

Only run this if you choose option A ("fair re-run") for the QAOA-row decision; the paper
is already self-consistent with the single-start row + the app:qaoa caveat (option B).

Same instances as the scaling run (scaled_groups + make_hard_instance + build_qubo),
same seeds/lambda. QAOA via phase_a.optimise_qaoa with init='random', normalize=True,
restarts=6, p=3, steps=150, mu=2.0 (= setting A1). SA recorded as the exact anchor.

n=16 is feasible locally (~minutes/instance); n=18-22 want a GPU (Colab: set DEV in
phase2 to lightning.gpu, or run there). Saves incrementally to results/scaling_qaoa_fair.json
with the SAME schema as scaling_qaoa.json so it can drop into tab:scaling.

Usage:  python qaoa_fair_rerun.py            # default ns below
        (edit CONFIG['ns'] to add 18,20,22 when on GPU)
"""
import os, json, time
import numpy as np
import qfs_common as q
import phase2_ablation_strong as p2
import phase_a_qaoa_depth as A

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "scaling_qaoa_fair.json")

CONFIG = dict(
    ns=[16],                       # add 18,20,22 on a GPU
    seeds=list(range(8)),
    lams=[1.0, 2.0],
    k_frac=0.42,                   # matches the scaling run (k=round(0.42 n))
    n_samples=600,
    qaoa=dict(p=3, mu=2.0, steps=150, lr=0.04, restarts=6, normalize=True),
)


def main():
    t0 = time.time(); c = CONFIG; qc = c["qaoa"]
    records = []
    if os.path.exists(OUT):
        try:
            records = json.load(open(OUT, encoding="utf-8")).get("records", [])
            print(f"resuming: {len(records)} records", flush=True)
        except Exception:
            pass
    done = {(r["n"], r["lam"], r["seed"]) for r in records}

    def save():
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(dict(config=c, records=records), f, indent=2)

    print(f"FAIR QAOA re-run  ns={c['ns']} seeds={len(c['seeds'])} lams={c['lams']} "
          f"(p={qc['p']}, normalize={qc['normalize']}, restarts={qc['restarts']})\n", flush=True)
    for n in c["ns"]:
        g = p2.scaled_groups(n); k = max(2, round(c["k_frac"] * n))
        for lam in c["lams"]:
            for sd in c["seeds"]:
                if (n, lam, sd) in done:
                    continue
                X, y, _ = q.make_hard_instance(n_A=g["n_A"], n_B=g["n_B"], n_C=g["n_C"],
                                               d_B=g["d_B"], n_samples=c["n_samples"], seed=sd)
                r, R = q.build_qubo(X, y)
                Ja, combos, best = q.exact_solve(r, R, k, lam); Jx = float(Ja[best])
                exsvm = q.downstream_svm(X, y, combos[best], seed=sd)
                res = A.optimise_qaoa(r, R, k, lam, qc["p"], qc["mu"], qc["steps"], qc["lr"],
                                      qc["restarts"], "random", qc["normalize"], sd, 4096)
                S = res["_S"]
                dJ = Jx - q.eval_J(S, r, R, lam)
                dS = exsvm - q.downstream_svm(X, y, S, seed=sd)
                sa = q.simulated_annealing(r, R, k, lam, seed=sd)
                dJ_sa = Jx - q.eval_J(sa, r, R, lam)
                records.append(dict(n=n, k=k, lam=lam, seed=sd, method="QAOA-fair",
                                    dJ=float(dJ), dSVM=float(dS), dJ_sa=float(dJ_sa)))
                save()
                print(f"  n={n} lam={lam} seed={sd}  QAOA-fair dJ={dJ:.3f} (SA={dJ_sa:.3f})  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    import collections
    agg = collections.defaultdict(list)
    for r in records:
        agg[r["n"]].append(r["dJ"])
    print("\n n | QAOA-fair mean dJ (8 seeds x 2 lambda)")
    for n in sorted(agg):
        print(f" {n} | {np.mean(agg[n]):.3f}")
    save()
    print(f"\nsaved -> {OUT}\ntotal {time.time()-t0:.0f}s  "
          f"(compare to single-start tab:scaling: 0.170/0.163/0.231/0.201)", flush=True)


if __name__ == "__main__":
    main()
