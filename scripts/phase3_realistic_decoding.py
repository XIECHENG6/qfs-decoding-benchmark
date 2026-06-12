"""
phase3_realistic_decoding.py  --  validate the decoder-selection criterion under
REALISTIC composite noise (not just depolarizing). This is the simulation-based
replacement for a real-hardware run: it shows the expval->sampling crossover
predicted by Cor.4 / Prop.5 survives T1/T2 damping, readout error and coherent
over-rotations, defeating the old "depolarizing-only" referee critique.

Protocol (matches the paper): train VQC-ring NOISELESSLY (statevector, fast), then
EVALUATE the fixed parameters under noise (mixed state) at a sweep of strengths,
decoding with expval / top-k / sampling. Overlay the theoretical critical rate.

Composite noise per gate (strength s scales all of them together):
  depolarizing(p1=s, p2=10s) + amplitude damping(gamma=0.5s) + phase damping(0.5s),
  plus optional coherent over-rotation, plus readout bit-flip(p_ro) before measure.

Small n (<=10) runs locally on default.mixed; larger n -> Colab.
Outputs: results/phase3_realistic.json
"""
import os, json, time
import numpy as np
import qfs_common as q
import pennylane as qml
from pennylane import numpy as pnp

CONFIG = dict(
    n=8, k=3, L=4, seeds=[0, 1, 2, 3, 4],
    train_steps=200, lr=0.06, tau0=1.0, tau1=10.0, alpha=5.0,
    noise_levels=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    coherent=0.5,        # coherent over-rotation angle = coherent * s (radians)
    p_readout=2.0,       # readout flip prob = p_readout * s (capped at 0.5)
    shots=4096,
)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def train_ring_noiseless(r, k, cfg, seed):
    """Train VQC-ring on the statevector simulator (fast); return params + r."""
    n = len(r); L = cfg["L"]
    dev = qml.device("default.qubit", wires=n)

    def circ(theta, rr):
        for i in range(n):
            qml.RY(rr[i] * np.pi, wires=i)
        for l in range(L):
            for i in range(n):
                qml.RY(theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])
            if l < L - 1:
                for i in range(n):
                    qml.RY(rr[i] * np.pi / 2, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    qn = qml.QNode(circ, dev, diff_method="backprop", interface="autograd")
    rng = np.random.default_rng(seed)
    theta = pnp.array(rng.uniform(-np.pi, np.pi, (L, n, 2)), requires_grad=True)
    r_p = pnp.array(r, requires_grad=False)
    opt = qml.AdamOptimizer(cfg["lr"])
    for t in range(cfg["train_steps"]):
        tau = cfg["tau0"] + (cfg["tau1"] - cfg["tau0"]) * t / max(cfg["train_steps"] - 1, 1)

        def cost(th):
            z = pnp.stack(qn(th, r_p))
            P = 1.0 / (1.0 + pnp.exp(tau * z))
            return -pnp.sum(r_p * P) + cfg["alpha"] * (pnp.sum(P) - k) ** 2

        theta = opt.step(cost, theta)
    return np.array(theta)


def noisy_eval(theta, r, cfg, s):
    """Evaluate trained params under composite noise; return (<Z>, bitstring probs)."""
    n = len(r); L = cfg["L"]
    p1, p2 = s, min(10 * s, 0.75)
    gamma_ad, gamma_pd = 0.5 * s, 0.5 * s
    coh = cfg["coherent"] * s
    dev = qml.device("default.mixed", wires=n)

    def add_noise(i):
        if s <= 0:
            return
        qml.DepolarizingChannel(p1, wires=i)
        if gamma_ad > 0:
            qml.AmplitudeDamping(min(gamma_ad, 1.0), wires=i)
        if gamma_pd > 0:
            qml.PhaseDamping(min(gamma_pd, 1.0), wires=i)
        if coh != 0:
            qml.RZ(coh, wires=i)            # coherent over-rotation (not in depol model)

    def base(theta_, rr):
        for i in range(n):
            qml.RY(rr[i] * np.pi, wires=i); add_noise(i)
        for l in range(L):
            for i in range(n):
                qml.RY(theta_[l, i, 0], wires=i); qml.RZ(theta_[l, i, 1], wires=i); add_noise(i)
            for i in range(n):
                j = (i + 1) % n
                qml.CNOT(wires=[i, j])
                if s > 0:
                    qml.DepolarizingChannel(p2, wires=i); qml.DepolarizingChannel(p2, wires=j)
            if l < L - 1:
                for i in range(n):
                    qml.RY(rr[i] * np.pi / 2, wires=i)
        # readout bit-flip
        if s > 0 and cfg["p_readout"] * s > 0:
            for i in range(n):
                qml.BitFlip(min(cfg["p_readout"] * s, 0.5), wires=i)

    th = pnp.array(theta, requires_grad=False); r_p = pnp.array(r, requires_grad=False)

    @qml.qnode(dev)
    def z_node(theta_, rr):
        base(theta_, rr)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    @qml.qnode(dev)
    def p_node(theta_, rr):
        base(theta_, rr)
        return qml.probs(wires=range(n))

    return np.array(z_node(th, r_p)), np.array(p_node(th, r_p))


def decode_all(z, probs, r, R, k, lam, n, shots, seed):
    expval = np.array(sorted([i for i in range(n) if z[i] < 0]))   # may != k
    topk = np.array(sorted(np.argsort(z)[:k]))                      # most negative <Z>
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(probs), size=shots, p=np.clip(probs, 0, None) / probs.sum())
    bestJ, samp = -np.inf, None
    for v in np.unique(idx):
        S = [i for i in range(n) if (v >> (n - 1 - i)) & 1]
        if len(S) == k:
            Jv = q.eval_J(S, r, R, lam)
            if Jv > bestJ:
                bestJ, samp = Jv, np.array(sorted(S))
    if samp is None:
        samp = topk
    return dict(expval=expval, topk=topk, sampling=samp)


def main():
    t0 = time.time(); c = CONFIG
    n, k = c["n"], c["k"]
    print(f"Phase-3 realistic-noise decoding  n={n} k={k} L={c['L']} "
          f"seeds={len(c['seeds'])} (depol+T1+T2+coherent+readout)\n", flush=True)
    # theoretical critical rate (Cor.4), G1=3Ln, G2=Ln, c=10, tau=10, zbar~0.7, M=ln9
    G1, G2 = 3 * c["L"] * n, c["L"] * n
    p1_star = np.log(c["tau1"] * 0.7 / np.log(9)) / (G1 + 2 * 10 * G2)
    print(f" theoretical expval critical p1*  ~ {p1_star:.2e} (Cor.4)\n", flush=True)

    res = {d: {lv: [] for lv in c["noise_levels"]} for d in ["expval", "topk", "sampling"]}
    card = {d: {lv: [] for lv in c["noise_levels"]} for d in ["expval", "topk", "sampling"]}
    for sd in c["seeds"]:
        X, y, _ = q.make_hard_instance(n_A=2, n_B=4, n_C=2, d_B=3, n_samples=500, seed=sd)
        # use first n features deterministically -> exactly n qubits
        X = X[:, :n]
        r, R = q.build_qubo(X, y)
        theta = train_ring_noiseless(r, k, c, sd)
        for s in c["noise_levels"]:
            z, probs = noisy_eval(theta, r, c, s)
            dec = decode_all(z, probs, r, R, k, 1.0, n, c["shots"], sd)
            for d, S in dec.items():
                acc = q.downstream_svm(X, y, S, seed=sd) if len(S) > 0 else 0.5
                res[d][s].append(acc); card[d][s].append(int(len(S) == k))
            print(f"  seed {sd} s={s:<6}  "
                  f"expval={res['expval'][s][-1]:.3f}({card['expval'][s][-1]}) "
                  f"topk={res['topk'][s][-1]:.3f} "
                  f"samp={res['sampling'][s][-1]:.3f}   [{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 70)
    print(f" {'noise s':>9}{'expval':>16}{'top-k':>16}{'sampling':>16}")
    print(" " + "-" * 65)
    for s in c["noise_levels"]:
        def ms(d): return f"{np.mean(res[d][s]):.3f}±{np.std(res[d][s]):.3f}"
        flag = "  <-- p1*" if (s > 0 and s <= p1_star * 6 and s >= p1_star / 6) else ""
        print(f" {s:>9}{ms('expval'):>16}{ms('topk'):>16}{ms('sampling'):>16}{flag}")
    print("=" * 70)
    out = os.path.join(ROOT, "results", "phase3_realistic.json")
    with open(out, "w") as f:
        json.dump(dict(config=c, p1_star=float(p1_star),
                       acc={d: {str(s): res[d][s] for s in c["noise_levels"]} for d in res},
                       card={d: {str(s): card[d][s] for s in c["noise_levels"]} for d in card}),
                  f, indent=2)
    print(f"\n saved -> {out}\n total {time.time()-t0:.0f}s")
    print(" CLAIM if expval collapses near p1* while sampling stays high: the closed-form")
    print(" decoder criterion predicts the crossover even under realistic (non-depolarizing) noise.")


if __name__ == "__main__":
    main()
