"""
phase_a_qaoa_depth.py  --  Tier-1 (A): QAOA depth / optimiser / warm-start ablation.

Purpose: defend the claim (Section 5) that QAOA on the full O(n^2) mRMR-QUBO is
*intrinsically* hard to optimise at scale -- NOT merely under-tuned. A referee will
say "p=3 with one random init and no Hamiltonian normalisation is under-optimised."
So we give QAOA every reasonable advantage and show the optimality gap does not close:

  * depth        : p in {1,2,3,5,8}
  * multistart   : best-of-`restarts` random inits (chosen by training energy; fair)
  * warm-start   : Trotterised-annealing (INTERP-style) schedule init
  * conditioning : Hamiltonian normalisation so gamma in [0,pi] is well-scaled
  * diagnostic   : gradient variance Var[d<H>/d gamma_0] vs n (barren-plateau probe)

Anchor: classical SA reaches the exact optimum on every instance (dJ=0). If QAOA's
best-effort dJ stays well above 0 and roughly flat in p, the gap is structural.

Same instances/QUBO as the scaling run (reuses phase2_ablation_strong.scaled_groups,
qfs_common). Decode = sample probs, post-select |S|=k, rank by true mRMR J (identical
to the ablation), so dJ is directly comparable.

Local-friendly at n=14,16. Saves incrementally to results/qaoa_depth_ablation.json.
"""
import os, json, time
import numpy as np
import qfs_common as q
import phase2_ablation_strong as p2          # reuse scaled_groups + device choice
import pennylane as qml
from pennylane import numpy as pnp

DEV, DIFF = p2.DEV, p2.DIFF
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "qaoa_depth_ablation.json")

CONFIG = dict(
    ns=[14, 16],
    seeds=[0, 1, 2],
    lam=1.0,
    k_frac=0.42,
    n_samples=600,
    steps=150,
    lr=0.04,
    mu=2.0,
    shots_decode=4096,
    gradvar_ns=[10, 12, 14, 16],   # barren-plateau probe (no optimisation)
    gradvar_p=3,
    gradvar_samples=40,
)

# Decisive variants (p, init, normalize, restarts). A0 reproduces the paper's QAOA
# setting (1 random init, un-normalised H, p=3); A1-A3 give it every reasonable
# advantage. If A1-A3 close the gap that A0 leaves, the paper's scaling-QAOA row is
# an under-optimisation artefact and must be re-run -- and the "densest encoding
# hardest to optimise" sentence revised.
VARIANTS = [
    (3, "random", False, 1),   # A0: reproduces paper (1 init, no normalization)
    (3, "random", True,  6),   # A1: best-effort, same depth (multistart + normalization)
    (3, "anneal", True,  1),   # A2: annealing warm-start
    (5, "anneal", True,  1),   # A3: warm-start + more depth
]


# ---------------------------------------------------------------------------
# Ising mapping (identical to phase2_ablation_strong.qaoa_mrmr_select)
# ---------------------------------------------------------------------------
def ising_from_qubo(r, R, k, lam, mu):
    n = len(r)
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
    return h, J


def make_qnodes(n, h, J, p):
    dev = qml.device(DEV, wires=n)
    # Cost as a SINGLE Hamiltonian expectation -> lightning adjoint is ~10-50x faster
    # than returning the n + |J| individual Pauli expvals and summing in autograd.
    coeffs, ops = [], []
    for i in range(n):
        if abs(h[i]) > 1e-12:
            coeffs.append(float(h[i])); ops.append(qml.PauliZ(i))
    for (i, j), Jij in J.items():
        if abs(Jij) > 1e-12:
            coeffs.append(float(Jij)); ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    H = qml.Hamiltonian(coeffs, ops)

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

    def circ_energy(g, b):
        _ansatz(g, b)
        return qml.expval(H)

    def circ_probs(g, b):
        _ansatz(g, b)
        return qml.probs(wires=range(n))

    qn_e = qml.QNode(circ_energy, dev,
                     diff_method=("backprop" if DEV == "default.qubit" else "adjoint"),
                     interface="autograd")
    qn_p = qml.QNode(circ_probs, dev, diff_method=None)
    return qn_e, qn_p


def anneal_init(p):
    """Trotterised-annealing (INTERP-style) schedule: gamma ramps up, beta ramps down.
    Scaled small because the normalised Hamiltonian has O(1) couplings."""
    l = (np.arange(p) + 0.5) / p
    g0 = 0.4 * l                       # 0 -> ~0.4
    b0 = 0.8 * (1.0 - l)               # ~0.8 -> 0
    return g0, b0


def optimise_qaoa(r, R, k, lam, p, mu, steps, lr, restarts, init, normalize, seed,
                  shots_decode):
    """Returns dict with best dJ over restarts (chosen by training energy), the
    best final training energy, and the energy of the exact optimum (for reference)."""
    n = len(r)
    h, J = ising_from_qubo(r, R, k, lam, mu)
    scale = 1.0
    if normalize:
        Jarr = np.array(list(J.values()))
        scale = float(np.sqrt(np.mean(h**2) + (np.mean(Jarr**2) if len(Jarr) else 0.0))) + 1e-12
    hh = h / scale
    JJ = {key: val / scale for key, val in J.items()}
    qn_e, qn_p = make_qnodes(n, hh, JJ, p)

    def energy(gg, bb):
        return qn_e(gg, bb)

    if init == "anneal":
        inits = [anneal_init(p)]
    else:
        rng = np.random.default_rng(seed)
        inits = [(rng.uniform(0, np.pi, p), rng.uniform(0, np.pi, p)) for _ in range(restarts)]

    best = dict(loss=np.inf, S=None, dJ=np.inf)
    for (g0, b0) in inits:
        g = pnp.array(g0, requires_grad=True); b = pnp.array(b0, requires_grad=True)
        opt = qml.AdamOptimizer(lr)
        final = None
        for t in range(steps):
            (g, b), final = opt.step_and_cost(energy, g, b)
        probs = np.array(qn_p(g, b))
        rngs = np.random.default_rng(seed + 7)
        idx = rngs.choice(len(probs), size=shots_decode, p=probs / probs.sum())
        bJ, bS = -np.inf, None
        for v in np.unique(idx):
            S = [i for i in range(n) if (v >> (n - 1 - i)) & 1]
            if len(S) == k:
                Jv = q.eval_J(S, r, R, lam)
                if Jv > bJ:
                    bJ, bS = Jv, np.array(sorted(S))
        if bS is None:                      # fallback: marginal top-k
            marg = np.zeros(n)
            for v, c in zip(*np.unique(idx, return_counts=True)):
                for i in range(n):
                    if (v >> (n - 1 - i)) & 1:
                        marg[i] += c
            bS = np.array(sorted(np.argsort(-marg)[:k]))
        if float(final) < best["loss"]:     # pick restart by training energy only
            best = dict(loss=float(final), S=bS, dJ=None)
            best["_S"] = bS
    return best


def grad_variance(r, R, k, lam, p, mu, n_samples, seed):
    """Var over random params of d<H>/d gamma_0 (normalised H). Barren-plateau probe."""
    n = len(r)
    h, J = ising_from_qubo(r, R, k, lam, mu)
    Jarr = np.array(list(J.values()))
    scale = float(np.sqrt(np.mean(h**2) + (np.mean(Jarr**2) if len(Jarr) else 0.0))) + 1e-12
    hh = h / scale
    JJ = {key: val / scale for key, val in J.items()}
    qn_e, _ = make_qnodes(n, hh, JJ, p)

    def energy(gg, bb):
        return qn_e(gg, bb)

    rng = np.random.default_rng(seed)
    grads = []
    for _ in range(n_samples):
        g = pnp.array(rng.uniform(0, np.pi, p), requires_grad=True)
        b = pnp.array(rng.uniform(0, np.pi, p), requires_grad=True)
        gg, _gb = qml.grad(energy)(g, b)    # grads wrt (gamma, beta); both trainable
        grads.append(float(gg[0]))          # d<H>/d gamma_0
    return float(np.var(grads))


def main():
    t0 = time.time()
    records = []
    gradvar = {}
    # resume if partial
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            records = prev.get("records", [])
            gradvar = prev.get("gradvar", {})
            print(f"resuming: {len(records)} records already present", flush=True)
        except Exception:
            pass
    done = {(rec["n"], rec["seed"], rec["p"], rec["init"], rec["normalize"]) for rec in records}

    def save():
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(dict(config=CONFIG, device=f"{DEV}/{DIFF}",
                           records=records, gradvar=gradvar), f, indent=2)

    print(f"device={DEV}/{DIFF}  ns={CONFIG['ns']}  seeds={CONFIG['seeds']}  "
          f"variants={VARIANTS}", flush=True)

    for n in CONFIG["ns"]:
        g = p2.scaled_groups(n)
        k = max(2, round(CONFIG["k_frac"] * n))
        lam = CONFIG["lam"]
        for sd in CONFIG["seeds"]:
            X, y, grp = q.make_hard_instance(n_A=g["n_A"], n_B=g["n_B"], n_C=g["n_C"],
                                             d_B=g["d_B"], n_samples=CONFIG["n_samples"], seed=sd)
            r, R = q.build_qubo(X, y)
            Ja, combos, best = q.exact_solve(r, R, k, lam)
            Jx = float(Ja[best])
            sa = q.simulated_annealing(r, R, k, lam, seed=sd)
            dJ_sa = Jx - q.eval_J(sa, r, R, lam)
            # variants: full p-sweep (random, normalized) + p=3 reference variants
            for (p, init, norm, rs) in VARIANTS:
                if (n, sd, p, init, norm) in done:
                    continue
                res = optimise_qaoa(r, R, k, lam, p, CONFIG["mu"], CONFIG["steps"],
                                    CONFIG["lr"], rs, init, norm, sd,
                                    CONFIG["shots_decode"])
                S = res["_S"]
                dJ = Jx - q.eval_J(S, r, R, lam)
                records.append(dict(n=n, k=k, lam=lam, seed=sd, p=p, init=init,
                                    normalize=norm, restarts=rs, dJ=float(dJ),
                                    train_energy=res["loss"], dJ_sa=float(dJ_sa)))
                save()
                print(f"  n={n} seed={sd} p={p} init={init} norm={norm}  "
                      f"dJ={dJ:.3f} (SA={dJ_sa:.3f})  [{time.time()-t0:.0f}s]", flush=True)
        # gradient-variance probe at this point too (cheap), independent of seed loop
    # barren-plateau probe over n (uses seed 0 instance)
    for n in CONFIG["gradvar_ns"]:
        if str(n) in gradvar:
            continue
        g = p2.scaled_groups(n); k = max(2, round(CONFIG["k_frac"] * n))
        X, y, grp = q.make_hard_instance(n_A=g["n_A"], n_B=g["n_B"], n_C=g["n_C"],
                                         d_B=g["d_B"], n_samples=CONFIG["n_samples"], seed=0)
        r, R = q.build_qubo(X, y)
        gv = grad_variance(r, R, k, CONFIG["lam"], CONFIG["gradvar_p"], CONFIG["mu"],
                           CONFIG["gradvar_samples"], seed=0)
        gradvar[str(n)] = gv
        save()
        print(f"  gradvar n={n}: Var[dH/dgamma0]={gv:.4e}  [{time.time()-t0:.0f}s]", flush=True)

    save()
    print(f"\nsaved -> {OUT}\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
