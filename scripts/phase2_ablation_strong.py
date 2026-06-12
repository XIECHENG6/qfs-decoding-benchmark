"""
phase2_ablation_strong.py  --  hardened entanglement ablation (defeats the
"you just under-trained / too small / cherry-picked" referee critiques).

On the SAME hard mRMR instances, against the EXACT optimum, compare:
  classical:  Random-mRMR, Greedy-mRMR, SA, MeanField(cl, multi-restart)
  quantum  :  VQC(no-ent), VQC(ring, O(n) CNOT), VQC(all-to-all, O(n^2) CNOT)
              -- each multi-restart, best restart chosen by TRAINING loss (no peeking)
  full enc :  QAOA-mRMR on the complete O(n^2) Ising encoding (the "correct" quantum
              way) -- shows even the faithful encoding does not beat classical.

Robustness knobs: many seeds, multiple n, multiple lambda, real training budget,
restarts. Restart selection uses the optimisation objective only (fair).

Outputs: results/phase2_ablation_strong.json  +  printed summary.
Designed to run LOCALLY at moderate n (12,14). For n>=16 use the Colab notebook.
"""
import os, json, time
import numpy as np
import qfs_common as q
import pennylane as qml
from pennylane import numpy as pnp

# ------------------------------ config --------------------------------------
CONFIG = dict(
    ns=[12, 14],                 # local-friendly; Colab does 16..22
    k_frac=0.42,                 # k = round(k_frac * n)  (=>5 at 12, 6 at 14)
    lams=[1.0, 2.0],
    seeds=[0, 1, 2, 3, 4, 5, 6, 7],
    n_samples=600,
    vqc=dict(L=4, steps=220, restarts=3, lr=0.06, tau0=1.0, tau1=10.0, alpha=5.0),
    qaoa=dict(p=3, steps=150, lr=0.04, mu=2.0),
)

# fast device
try:
    qml.device("lightning.qubit", wires=2)
    DEV, DIFF = "lightning.qubit", "adjoint"
except Exception:
    DEV, DIFF = "default.qubit", "backprop"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def scaled_groups(n):
    """Scale the A/B/C trap to size n; keep an informative B core."""
    n_A = max(2, round(n * 0.25))
    n_C = max(2, round(n * 0.22))
    n_B = n - n_A - n_C
    d_B = max(3, round(n_B * 0.7))
    return dict(n_A=n_A, n_B=n_B, n_C=n_C, d_B=d_B)


# ------------------------------ VQC ----------------------------------------
def _qnode(n, L, entangle):
    dev = qml.device(DEV, wires=n)

    def circ(theta, r):
        for i in range(n):
            qml.RY(r[i] * np.pi, wires=i)
        for l in range(L):
            for i in range(n):
                qml.RY(theta[l, i, 0], wires=i)
                qml.RZ(theta[l, i, 1], wires=i)
            if entangle == "ring":
                for i in range(n):
                    qml.CNOT(wires=[i, (i + 1) % n])
            elif entangle == "all":
                for i in range(n):
                    for j in range(i + 1, n):
                        qml.CNOT(wires=[i, j])
            if l < L - 1:
                for i in range(n):
                    qml.RY(r[i] * np.pi / 2, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return qml.QNode(circ, dev, diff_method=DIFF, interface="autograd")


def vqc_select(r, R, k, lam, entangle, seed, cfg):
    n = len(r); L = cfg["L"]
    qn = _qnode(n, L, entangle)
    r_p = pnp.array(r, requires_grad=False)
    R_p = pnp.array(R, requires_grad=False)
    best_loss, best_S = np.inf, None
    for rs in range(cfg["restarts"]):
        rng = np.random.default_rng(1000 * seed + rs)
        theta = pnp.array(rng.uniform(-np.pi, np.pi, (L, n, 2)), requires_grad=True)
        opt = qml.AdamOptimizer(cfg["lr"])
        final = None
        for t in range(cfg["steps"]):
            tau = cfg["tau0"] + (cfg["tau1"] - cfg["tau0"]) * t / max(cfg["steps"] - 1, 1)

            def cost(th):
                z = pnp.stack(qn(th, r_p))
                P = 1.0 / (1.0 + pnp.exp(tau * z))
                return (-pnp.sum(r_p * P)
                        + lam * 0.5 * pnp.sum(R_p * pnp.outer(P, P))
                        + cfg["alpha"] * (pnp.sum(P) - k) ** 2)

            theta, final = opt.step_and_cost(cost, theta)
        z = np.array(qn(theta, r_p))
        P = 1.0 / (1.0 + np.exp(cfg["tau1"] * z))
        S = np.array(sorted(np.argsort(-P)[:k]))
        if float(final) < best_loss:        # pick restart by TRAINING loss only
            best_loss, best_S = float(final), S
    return best_S


# ------------------------------ QAOA on full mRMR encoding -------------------
def qaoa_mrmr_select(r, R, k, lam, seed, cfg):
    n = len(r); p = cfg["p"]; mu = cfg["mu"]
    # QUBO f(x) = -sum r x + lam sum_{i<j} R x x + mu (sum x - k)^2 ; x=(1-Z)/2
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -r[i] + mu * (1 - 2 * k)
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = lam * R[i, j] + 2 * mu
    h = np.zeros(n); J = {}
    for i in range(n):
        h[i] += -Q[i, i] / 2.0
        for j in range(i + 1, n):
            J[(i, j)] = Q[i, j] / 4.0
            h[i] += -Q[i, j] / 4.0
            h[j] += -Q[i, j] / 4.0
    dev = qml.device(DEV, wires=n)

    def _ansatz(g, b):
        for i in range(n):
            qml.Hadamard(wires=i)
        for layer in range(p):
            for i in range(n):
                if abs(h[i]) > 1e-12:
                    qml.RZ(2 * g[layer] * h[i], wires=i)
            for (i, j), Jij in J.items():
                if abs(Jij) > 1e-12:
                    qml.CNOT(wires=[i, j]); qml.RZ(2 * g[layer] * Jij, wires=j); qml.CNOT(wires=[i, j])
            for i in range(n):
                qml.RX(2 * b[layer], wires=i)

    def circ_energy(g, b):                       # gradient path (expvals)
        _ansatz(g, b)
        terms = [qml.expval(qml.PauliZ(i)) for i in range(n)]
        terms += [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)) for (i, j) in J]
        return terms

    def circ_probs(g, b):                         # readout path (probs, no grad)
        _ansatz(g, b)
        return qml.probs(wires=range(n))

    qn = qml.QNode(circ_energy, dev,
                   diff_method=("backprop" if DEV == "default.qubit" else "adjoint"),
                   interface="autograd")
    qn_probs = qml.QNode(circ_probs, dev, diff_method=None)

    hvec = pnp.array(h, requires_grad=False)
    Jvals = pnp.array([J[key] for key in J], requires_grad=False)
    rng = np.random.default_rng(seed)
    g = pnp.array(rng.uniform(0, np.pi, p), requires_grad=True)
    b = pnp.array(rng.uniform(0, np.pi, p), requires_grad=True)
    opt = qml.AdamOptimizer(cfg["lr"])
    for t in range(cfg["steps"]):
        def energy(gg, bb):
            vals = qn(gg, bb)
            z = pnp.stack(vals[:n]); zz = pnp.stack(vals[n:])
            return pnp.sum(hvec * z) + pnp.sum(Jvals * zz)
        (g, b), _ = opt.step_and_cost(energy, g, b)

    # decode: sample from probs, post-select |S|=k, rank by true mRMR J
    probs = np.array(qn_probs(g, b))
    rngs = np.random.default_rng(seed + 7)
    idx = rngs.choice(len(probs), size=4096, p=probs / probs.sum())
    bestJ, bestS = -np.inf, None
    for v in np.unique(idx):
        S = [i for i in range(n) if (v >> (n - 1 - i)) & 1]
        if len(S) == k:
            Jv = q.eval_J(S, r, R, lam)
            if Jv > bestJ:
                bestJ, bestS = Jv, np.array(sorted(S))
    if bestS is None:                          # fallback: top-k by marginal prob
        marg = np.zeros(n)
        for v, c in zip(*np.unique(idx, return_counts=True)):
            for i in range(n):
                if (v >> (n - 1 - i)) & 1:
                    marg[i] += c
        bestS = np.array(sorted(np.argsort(-marg)[:k]))
    return bestS


# ------------------------------ driver --------------------------------------
def main():
    t0 = time.time()
    methods_q = ["VQC(no-ent)", "VQC(ring)", "VQC(all)"]
    ent_map = {"VQC(no-ent)": "none", "VQC(ring)": "ring", "VQC(all)": "all"}
    methods_c = ["Random-mRMR", "Greedy-mRMR", "SA", "MeanField(cl)"]
    allm = methods_c + methods_q + ["QAOA-mRMR"]
    agg = {(n, lam): {m: {"dJ": [], "dsvm": []} for m in allm}
           for n in CONFIG["ns"] for lam in CONFIG["lams"]}
    records = []

    print(f"device={DEV}/{DIFF}  ns={CONFIG['ns']}  lams={CONFIG['lams']}  "
          f"seeds={len(CONFIG['seeds'])}", flush=True)
    for n in CONFIG["ns"]:
        g = scaled_groups(n)
        k = max(2, round(CONFIG["k_frac"] * n))
        for lam in CONFIG["lams"]:
            for sd in CONFIG["seeds"]:
                X, y, grp = q.make_hard_instance(n_A=g["n_A"], n_B=g["n_B"], n_C=g["n_C"],
                                                 d_B=g["d_B"], n_samples=CONFIG["n_samples"], seed=sd)
                r, R = q.build_qubo(X, y)
                Ja, combos, best = q.exact_solve(r, R, k, lam)
                Jx = float(Ja[best]); ex = q.downstream_svm(X, y, combos[best], seed=sd)
                sel = {
                    "Random-mRMR":   q.random_mrmr(r, R, k, lam, seed=sd),
                    "Greedy-mRMR":   q.greedy_mrmr(r, R, k, lam),
                    "SA":            q.simulated_annealing(r, R, k, lam, seed=sd),
                    "MeanField(cl)": q.classical_meanfield(r, R, k, lam, seed=sd),
                }
                for mq in methods_q:
                    sel[mq] = vqc_select(r, R, k, lam, ent_map[mq], sd, CONFIG["vqc"])
                sel["QAOA-mRMR"] = qaoa_mrmr_select(r, R, k, lam, sd, CONFIG["qaoa"])
                for m, S in sel.items():
                    dJ = Jx - q.eval_J(S, r, R, lam)
                    ds = ex - q.downstream_svm(X, y, S, seed=sd)
                    agg[(n, lam)][m]["dJ"].append(dJ)
                    agg[(n, lam)][m]["dsvm"].append(ds)
                    records.append(dict(n=n, k=k, lam=lam, seed=sd, method=m,
                                        dJ=float(dJ), dsvm=float(ds)))
                print(f"  n={n} lam={lam} seed={sd}  done  [{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 74)
    print(f" {'(n,lam)':>10} {'method':<15}{'dJ mean':>10}{'dJ>0 frac':>11}"
          f"{'dSVM mean':>11}{'dSVM std':>10}")
    print(" " + "-" * 72)
    for (n, lam) in agg:
        for m in allm:
            dJ = np.array(agg[(n, lam)][m]["dJ"]); ds = np.array(agg[(n, lam)][m]["dsvm"])
            print(f" {str((n,lam)):>10} {m:<15}{dJ.mean():>10.3f}{(dJ>1e-6).mean():>11.2f}"
                  f"{ds.mean():>11.3f}{ds.std():>10.3f}")
    out = os.path.join(ROOT, "results", "phase2_ablation_strong.json")
    with open(out, "w") as f:
        json.dump(dict(config=CONFIG, device=f"{DEV}/{DIFF}", records=records), f, indent=2)
    print(f"\n saved -> {out}\n total {time.time()-t0:.0f}s")
    print("\n INTERPRETATION: if every quantum row (incl. VQC(all) and QAOA-mRMR) has")
    print(" dJ>=0 and dJ mean >= MeanField(cl), then NO entanglement structure and NOT")
    print(" EVEN the full O(n^2) encoding beats a free classical mean-field optimiser.")


if __name__ == "__main__":
    main()
