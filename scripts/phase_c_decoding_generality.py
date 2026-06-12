"""
phase_c_decoding_generality.py  --  Tier-1 (C): generality + statistical power for the
decoding crossover (the paper's positive contribution).

Two referee-proofing goals:
  1. POWER. phase3 used 5 seeds, so a paired Wilcoxon floors at p=0.031 (one-sided) /
     0.0625 (two-sided): expval's collapse came out "n.s." despite a 0.25 drop. We
     re-run with more seeds (16) so the crossover is unambiguous (p well below 1e-3).
  2. GENERALITY + MECHANISM.
     (a) circuit-generality: the expval->sampling crossover holds for ring, all-to-all
         AND no-entanglement circuits (not a ring artefact).
     (b) channel-isolation: which noise *channel* breaks expectation-value decoding?
         Run each channel alone -- depolarizing / amplitude+phase damping / coherent
         over-rotation / readout flip -- vs the composite. Prediction (from the
         <Z>-damping theory, Prop.\\ref{prop:damp}): the *damping* channels
         (depol/damping/readout) shrink |<Z>| and break the threshold near p1*, while a
         purely *coherent* (unitary) error does not damp |<Z>| and leaves expval far
         more robust; sampling is robust to ALL of them. This confirms the criterion's
         mechanism, not just its existence.

Same protocol as phase3: train each circuit NOISELESSLY (statevector), then evaluate
fixed params under noise (default.mixed) and decode by expval / top-k / sampling.

n=8 (tractable on default.mixed). Saves incrementally to results/decoding_generality.json.
"""
import os, json, time
import numpy as np
import qfs_common as q
import pennylane as qml
from pennylane import numpy as pnp
from scipy import stats

CONFIG = dict(
    n=8, k=3, L=4,
    seeds=list(range(16)),
    train_steps=200, lr=0.06, tau0=1.0, tau1=10.0, alpha=5.0,
    noise_levels=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
    coherent=2.0,        # coherent over-rotation angle = coherent * s (rad); larger so
                         # a unitary error of comparable per-gate size is actually probed
    p_readout=2.0,
    shots=4096,
    circuits=["ring", "all", "none"],
    # which noise channels to run, per circuit (ring gets the full isolation grid)
    channels_ring=["composite", "depol", "damping", "coherent", "readout"],
    channels_other=["composite"],
)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "decoding_generality.json")


def _entangle(n, kind):
    if kind == "ring":
        for i in range(n):
            qml.CNOT(wires=[i, (i + 1) % n])
    elif kind == "all":
        for i in range(n):
            for j in range(i + 1, n):
                qml.CNOT(wires=[i, j])
    # "none": no entangling gates


def train_circuit(r, k, cfg, seed, entangle):
    """Train a VQC of the given entanglement noiselessly (statevector)."""
    n = len(r); L = cfg["L"]
    dev = qml.device("default.qubit", wires=n)

    def circ(theta, rr):
        for i in range(n):
            qml.RY(rr[i] * np.pi, wires=i)
        for l in range(L):
            for i in range(n):
                qml.RY(theta[l, i, 0], wires=i); qml.RZ(theta[l, i, 1], wires=i)
            _entangle(n, entangle)
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


def noisy_eval(theta, r, cfg, s, entangle, channel):
    """Evaluate trained params under ONE noise channel (or 'composite'); return (<Z>, probs)."""
    n = len(r); L = cfg["L"]
    p1, p2 = s, min(10 * s, 0.75)
    use_depol = channel in ("composite", "depol")
    use_damp = channel in ("composite", "damping")
    use_coh = channel in ("composite", "coherent")
    use_ro = channel in ("composite", "readout")
    gamma = 0.5 * s
    coh = cfg["coherent"] * s
    dev = qml.device("default.mixed", wires=n)

    def add_noise(i):
        if s <= 0:
            return
        if use_depol:
            qml.DepolarizingChannel(p1, wires=i)
        if use_damp:
            qml.AmplitudeDamping(min(gamma, 1.0), wires=i)
            qml.PhaseDamping(min(gamma, 1.0), wires=i)
        if use_coh and coh != 0:
            qml.RX(coh, wires=i)        # coherent over-rotation about X (does damp <Z> if
                                        # mid-circuit; chosen so a unitary error is probed)

    def base(theta_, rr):
        for i in range(n):
            qml.RY(rr[i] * np.pi, wires=i); add_noise(i)
        for l in range(L):
            for i in range(n):
                qml.RY(theta_[l, i, 0], wires=i); qml.RZ(theta_[l, i, 1], wires=i); add_noise(i)
            if entangle == "ring":
                for i in range(n):
                    j = (i + 1) % n
                    qml.CNOT(wires=[i, j])
                    if s > 0 and use_depol:
                        qml.DepolarizingChannel(p2, wires=i); qml.DepolarizingChannel(p2, wires=j)
            elif entangle == "all":
                for i in range(n):
                    for j in range(i + 1, n):
                        qml.CNOT(wires=[i, j])
                        if s > 0 and use_depol:
                            qml.DepolarizingChannel(p2, wires=i); qml.DepolarizingChannel(p2, wires=j)
            if l < L - 1:
                for i in range(n):
                    qml.RY(rr[i] * np.pi / 2, wires=i)
        if s > 0 and use_ro and cfg["p_readout"] * s > 0:
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
    expval = np.array(sorted([i for i in range(n) if z[i] < 0]))
    topk = np.array(sorted(np.argsort(z)[:k]))
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
    G1, G2 = 3 * c["L"] * n, c["L"] * n
    p1_star = np.log(c["tau1"] * 0.7 / np.log(9)) / (G1 + 2 * 10 * G2)
    print(f"Phase-C decoding generality  n={n} k={k} seeds={len(c['seeds'])}  "
          f"p1*={p1_star:.2e}\n", flush=True)

    # acc[circuit][channel][decoder][noise] = [per-seed]
    acc, card = {}, {}
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT, encoding="utf-8"))
            acc = prev.get("acc", {}); card = prev.get("card", {})
            print("resuming from existing file", flush=True)
        except Exception:
            pass

    def ensure(circ, ch):
        for store in (acc, card):
            store.setdefault(circ, {}).setdefault(ch, {})
            for d in ("expval", "topk", "sampling"):
                store[circ][ch].setdefault(d, {})
                for s in c["noise_levels"]:
                    store[circ][ch][d].setdefault(str(s), [])

    def save():
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(dict(config=c, p1_star=float(p1_star), acc=acc, card=card), f, indent=2)

    for si, sd in enumerate(c["seeds"]):
        X, y, _ = q.make_hard_instance(n_A=2, n_B=4, n_C=2, d_B=3, n_samples=500, seed=sd)
        X = X[:, :n]
        r, R = q.build_qubo(X, y)
        for circ in c["circuits"]:
            channels = c["channels_ring"] if circ == "ring" else c["channels_other"]
            # skip whole circuit if already done for all its channels at this seed
            need = any(len(acc.get(circ, {}).get(ch, {}).get("expval", {}).get(str(c["noise_levels"][0]), [])) <= si
                       for ch in channels)
            if not need:
                continue
            theta = train_circuit(r, k, c, sd, circ)
            for ch in channels:
                ensure(circ, ch)
                if len(acc[circ][ch]["expval"][str(c["noise_levels"][0])]) > si:
                    continue        # this (circ,ch) already has seed si
                for s in c["noise_levels"]:
                    z, probs = noisy_eval(theta, r, c, s, circ, ch)
                    dec = decode_all(z, probs, r, R, k, 1.0, n, c["shots"], sd)
                    for d, S in dec.items():
                        a = q.downstream_svm(X, y, S, seed=sd) if len(S) > 0 else 0.5
                        acc[circ][ch][d][str(s)].append(float(a))
                        card[circ][ch][d][str(s)].append(int(len(S) == k))
                save()
            print(f"  seed {sd} ({si+1}/{len(c['seeds'])}) circ={circ} done  "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # ----- significance summary -----
    print("\n" + "=" * 78)
    print(f" sampling vs expval at s=0.05  (one-sided Wilcoxon, {len(c['seeds'])} seeds)")
    print("=" * 78)
    smax = str(c["noise_levels"][-1]); s0 = str(c["noise_levels"][0])
    for circ in acc:
        for ch in acc[circ]:
            a = np.array(acc[circ][ch]["sampling"][smax]); b = np.array(acc[circ][ch]["expval"][smax])
            try:
                _, p = stats.wilcoxon(a - b, alternative="greater")
            except Exception:
                p = float("nan")
            # expval collapse magnitude
            drop = np.mean(acc[circ][ch]["expval"][s0]) - np.mean(b)
            print(f"  {circ:<5} {ch:<10}  samp={np.mean(a):.3f} expval={np.mean(b):.3f}  "
                  f"(expval drop {drop:+.3f})  p={p:.2e}")
    save()
    print(f"\nsaved -> {OUT}\ntotal {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
