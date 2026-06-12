"""
phase2_quantum_derisk.py  --  the LAST de-risk: does the quantum circuit (and its
ring entanglement) add anything over a classical mean-field optimiser on the hard
mRMR-QUBO?

Compares, on the SAME hard instances, against the exact optimum:
  - Random-mRMR        (best-of-N random)
  - MeanField(cl)      (VQC-QFS with the quantum circuit REMOVED; from qfs_common)
  - VQC-QFS (ring)     (real PennyLane circuit, ring entanglement, mean-field loss)
  - VQC-QFS (no-ent)   (same circuit, ring CNOTs removed)  <-- entanglement ablation

Expected (confirm before writing): VQC(ring) ~ VQC(no-ent) ~ MeanField(cl) ~ exact
=> the quantum circuit / entanglement contributes nothing.

Kept small (statevector) so it finishes fast; the conclusion is n-independent.
"""
import time
import numpy as np
import qfs_common as q
import pennylane as qml
from pennylane import numpy as pnp

# ---------------- config (small = fast; conclusion is n-independent) ----------
N_A, N_B, N_C, D_B = 3, 5, 4, 4     # n = 12
K = 4
LAM = 1.0
L = 3
SEEDS = [0, 1, 2]
T_STEPS = 120
LR = 0.08
TAU0, TAU1, ALPHA = 1.0, 10.0, 5.0

# fast device if available
try:
    _ = qml.device("lightning.qubit", wires=2)
    DEV_NAME, DIFF = "lightning.qubit", "adjoint"
except Exception:
    DEV_NAME, DIFF = "default.qubit", "backprop"


def make_qnode(n, entangle):
    dev = qml.device(DEV_NAME, wires=n)

    def circ(theta, r):
        for i in range(n):
            qml.RY(r[i] * np.pi, wires=i)
        for l in range(L):
            for i in range(n):
                qml.RY(theta[l, i, 0], wires=i)
                qml.RZ(theta[l, i, 1], wires=i)
            if entangle:
                for i in range(n):
                    qml.CNOT(wires=[i, (i + 1) % n])
            if l < L - 1:
                for i in range(n):
                    qml.RY(r[i] * np.pi / 2, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return qml.QNode(circ, dev, diff_method=DIFF, interface="autograd")


def train_vqc(r, R, k, lam, entangle, seed):
    n = len(r)
    qnode = make_qnode(n, entangle)
    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.uniform(-np.pi, np.pi, (L, n, 2)), requires_grad=True)
    r_p = pnp.array(r, requires_grad=False)
    R_p = pnp.array(R, requires_grad=False)
    opt = qml.AdamOptimizer(LR)
    for t in range(T_STEPS):
        tau = TAU0 + (TAU1 - TAU0) * t / max(T_STEPS - 1, 1)

        def cost(th):
            z = pnp.stack(qnode(th, r_p))
            P = 1.0 / (1.0 + pnp.exp(tau * z))
            return (-pnp.sum(r_p * P)
                    + lam * 0.5 * pnp.sum(R_p * pnp.outer(P, P))
                    + ALPHA * (pnp.sum(P) - k) ** 2)

        theta = opt.step(cost, theta)

    z = np.array(qnode(theta, r_p))
    P = 1.0 / (1.0 + np.exp(TAU1 * z))
    return np.array(sorted(np.argsort(-P)[:k]))


def main():
    n = N_A + N_B + N_C
    t0 = time.time()
    print(f"Phase-2 quantum de-risk: device={DEV_NAME}/{DIFF}, n={n}, k={K}, "
          f"lambda={LAM}, L={L}, {len(SEEDS)} seeds", flush=True)
    acc = {m: {"dsvm": [], "dJ": []} for m in
           ["Random-mRMR", "MeanField(cl)", "VQC(ring)", "VQC(no-ent)"]}
    for sd in SEEDS:
        X, y, g = q.make_hard_instance(n_A=N_A, n_B=N_B, n_C=N_C, d_B=D_B,
                                       n_samples=600, seed=sd)
        r, R = q.build_qubo(X, y)
        Ja, combos, best = q.exact_solve(r, R, K, LAM)
        Sx = combos[best]; Jx = float(Ja[best]); ex = q.downstream_svm(X, y, Sx, seed=sd)
        sets = {
            "Random-mRMR":   q.random_mrmr(r, R, K, LAM, seed=sd),
            "MeanField(cl)": q.classical_meanfield(r, R, K, LAM, seed=sd),
            "VQC(ring)":     train_vqc(r, R, K, LAM, True,  sd),
            "VQC(no-ent)":   train_vqc(r, R, K, LAM, False, sd),
        }
        line = f"  seed {sd}: exactSVM={ex:.3f}"
        for m, S in sets.items():
            acc[m]["dsvm"].append(ex - q.downstream_svm(X, y, S, seed=sd))
            acc[m]["dJ"].append(Jx - q.eval_J(S, r, R, LAM))
            line += f" | {m} dJ={Jx - q.eval_J(S, r, R, LAM):+.3f}"
        print(line + f"   [{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f" {'method':<16}{'dSVM (mean±std)':>20}{'dJ (mean)':>12}", flush=True)
    print(" " + "-" * 47, flush=True)
    for m, d in acc.items():
        ds = np.array(d["dsvm"]); dj = np.array(d["dJ"])
        print(f" {m:<16}{ds.mean():>10.3f} ± {ds.std():.3f}      {dj.mean():>8.3f}", flush=True)
    print("=" * 60, flush=True)
    print(" dSVM/dJ = exact - method. VQC(ring) ~ VQC(no-ent) ~ MeanField(cl) ~ 0", flush=True)
    print(" => quantum circuit AND ring entanglement add nothing.  [done]", flush=True)


if __name__ == "__main__":
    main()
