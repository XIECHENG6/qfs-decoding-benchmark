# =============================================================================
# Colab cell: FAIR QAOA re-run for the scaling table (n = 18, 20, 22 on GPU)
# =============================================================================
# Paste this whole block into ONE Colab cell and run it. It is SELF-CONTAINED
# (defines its own engine; does not depend on the other notebook cells), mounts
# Drive, probes for a GPU, and writes results/scaling_qaoa_fair.json to Drive.
#
# Purpose: the paper's tab:scaling QAOA-mRMR row was a SINGLE-start p=3 run
# (an upper bound). Tier-1 A showed that with Hamiltonian normalisation + 6
# restarts the QAOA gap drops to the ring-VQC level. This re-runs QAOA with that
# FAIR setting at the large sizes so the row reflects effort comparable to the
# multi-restart VQC rows. n=16 is done locally; do 18/20/22 here on GPU.
#
# Expected runtime: minutes/instance on a T4/A100; 3 sizes x 8 seeds x 2 lambda.
# Resumable: re-running skips finished (n,lam,seed). Compare output to the
# single-start row 0.170/0.163/0.231/0.201.
# =============================================================================
!pip -q install pennylane pennylane-lightning-gpu custatevec-cu12 >/dev/null 2>&1 || echo "gpu plugin optional"
import os, json, time, itertools
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from google.colab import drive

# ---- Drive + device ---------------------------------------------------------
try:
    drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print("drive mount:", e)
BASE = None
for cand in ['/content/drive/MyDrive/mRMR_rework', '/content/drive/My Drive/mRMR_rework']:
    if os.path.isdir(cand):
        BASE = cand; break
if BASE is None:
    BASE = '/content/drive/MyDrive/mRMR_rework'; os.makedirs(os.path.join(BASE, 'results'), exist_ok=True)
OUT = os.path.join(BASE, 'results', 'scaling_qaoa_fair.json')
DEV, DIFF = 'default.qubit', 'backprop'
for name in ['lightning.gpu', 'lightning.qubit', 'default.qubit']:
    try:
        qml.device(name, wires=2); DEV = name; DIFF = 'adjoint' if name.startswith('lightning') else 'backprop'; break
    except Exception:
        pass
print('device =', DEV, '/', DIFF, '| out =', OUT)

CONFIG = dict(ns=[18, 20, 22], seeds=list(range(8)), lams=[1.0, 2.0], k_frac=0.42,
              n_samples=600, qaoa=dict(p=3, mu=2.0, steps=150, lr=0.04, restarts=6))

# ---- engine (verbatim from qfs_common.py) -----------------------------------
def make_hard_instance(n_A=6, n_B=10, n_C=6, n_samples=800, rho_A=0.92, w_A=2.2,
                       w_B=1.0, d_B=6, noise_B=0.35, label_noise=0.5, seed=0):
    rng = np.random.default_rng(seed); N = n_samples
    s_A = rng.standard_normal(N); s_B = rng.standard_normal((n_B, N))
    logit = w_A * s_A + w_B * s_B[:d_B].sum(0)
    logit = (logit - logit.mean()) / (logit.std() + 1e-9)
    y = (logit + label_noise * rng.standard_normal(N) > 0).astype(int)
    cols = []
    for _ in range(n_A):
        cols.append(rho_A * s_A + np.sqrt(max(1 - rho_A**2, 0.0)) * rng.standard_normal(N))
    for j in range(n_B):
        cols.append(np.sqrt(max(1 - noise_B**2, 0.0)) * s_B[j] + noise_B * rng.standard_normal(N))
    for _ in range(n_C):
        cols.append(rng.standard_normal(N))
    return np.column_stack(cols), y

def _discretize(X, n_bins=8):
    Xb = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        edges = np.quantile(X[:, i], np.linspace(0, 1, n_bins + 1)[1:-1])
        Xb[:, i] = np.digitize(X[:, i], edges)
    return Xb

def _mi(a, b):
    Na, Nb = int(a.max()) + 1, int(b.max()) + 1
    joint = np.zeros((Na, Nb)); np.add.at(joint, (a, b), 1.0); joint /= joint.sum()
    pa = joint.sum(1, keepdims=True); pb = joint.sum(0, keepdims=True); nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])))

def build_qubo(X, y, n_bins=8):
    Xb = _discretize(X, n_bins); n = X.shape[1]
    r = np.array([_mi(Xb[:, i], y) for i in range(n)])
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = R[j, i] = _mi(Xb[:, i], Xb[:, j])
    r = r / (r.max() + 1e-12); R = R / (R.max() + 1e-12); np.fill_diagonal(R, 0.0)
    return r, R

def eval_J(S, r, R, lam):
    S = list(S); return float(r[S].sum() - lam * R[np.ix_(S, S)].sum() / 2.0)

def exact_solve(r, R, k, lam):
    n = len(r)
    flat = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n), k)), dtype=np.int64)
    combos = flat.reshape(-1, k); J = r[combos].sum(1)
    for a in range(k):
        for b in range(a + 1, k):
            J = J - lam * R[combos[:, a], combos[:, b]]
    return J, combos, int(np.argmax(J))

def scaled_groups(n):
    n_A = max(2, round(n * 0.25)); n_C = max(2, round(n * 0.22)); n_B = n - n_A - n_C
    return dict(n_A=n_A, n_B=n_B, n_C=n_C, d_B=max(3, round(n_B * 0.7)))

# ---- FAIR QAOA (normalised Hamiltonian + multistart; from phase_a) ----------
def fair_qaoa(r, R, k, lam, p, mu, steps, lr, restarts, seed):
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
            J[(i, j)] = Q[i, j] / 4.0; h[i] += -Q[i, j] / 4.0; h[j] += -Q[i, j] / 4.0
    Jarr = np.array(list(J.values()))
    scale = float(np.sqrt(np.mean(h**2) + (np.mean(Jarr**2) if len(Jarr) else 0.0))) + 1e-12
    hh = h / scale; JJ = {key: val / scale for key, val in J.items()}
    dev = qml.device(DEV, wires=n)
    coeffs, ops = [], []
    for i in range(n):
        if abs(hh[i]) > 1e-12:
            coeffs.append(float(hh[i])); ops.append(qml.PauliZ(i))
    for (i, j), v in JJ.items():
        if abs(v) > 1e-12:
            coeffs.append(float(v)); ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
    H = qml.Hamiltonian(coeffs, ops)

    def ansatz(g, b):
        for i in range(n):
            qml.Hadamard(wires=i)
        for layer in range(p):
            for i in range(n):
                if abs(hh[i]) > 1e-12:
                    qml.RZ(2 * g[layer] * hh[i], wires=i)
            for (i, j), v in JJ.items():
                if abs(v) > 1e-12:
                    qml.CNOT(wires=[i, j]); qml.RZ(2 * g[layer] * v, wires=j); qml.CNOT(wires=[i, j])
            for i in range(n):
                qml.RX(2 * b[layer], wires=i)

    def c_e(g, b): ansatz(g, b); return qml.expval(H)
    def c_p(g, b): ansatz(g, b); return qml.probs(wires=range(n))
    qn_e = qml.QNode(c_e, dev, diff_method=DIFF, interface='autograd')
    qn_p = qml.QNode(c_p, dev, diff_method=None)
    rng = np.random.default_rng(seed); best = (np.inf, None)
    for _ in range(restarts):
        g = pnp.array(rng.uniform(0, np.pi, p), requires_grad=True)
        b = pnp.array(rng.uniform(0, np.pi, p), requires_grad=True)
        opt = qml.AdamOptimizer(lr); final = None
        for _t in range(steps):
            (g, b), final = opt.step_and_cost(qn_e, g, b)
        probs = np.array(qn_p(g, b)); rs = np.random.default_rng(seed + 7)
        idx = rs.choice(len(probs), size=4096, p=probs / probs.sum())
        bJ, bS = -np.inf, None
        for v in np.unique(idx):
            S = [i for i in range(n) if (v >> (n - 1 - i)) & 1]
            if len(S) == k:
                Jv = eval_J(S, r, R, lam)
                if Jv > bJ:
                    bJ, bS = Jv, sorted(S)
        if bS is None:
            bS = sorted(np.argsort(-probs.reshape(-1))[:1])  # degenerate fallback
        if float(final) < best[0]:
            best = (float(final), bS)
    return best[1]

# ---- driver -----------------------------------------------------------------
records = []
if os.path.exists(OUT):
    try: records = json.load(open(OUT)).get('records', [])
    except Exception: pass
done = {(r['n'], r['lam'], r['seed']) for r in records}
qc = CONFIG['qaoa']; t0 = time.time()
for n in CONFIG['ns']:
    g = scaled_groups(n); k = max(2, round(CONFIG['k_frac'] * n))
    for lam in CONFIG['lams']:
        for sd in CONFIG['seeds']:
            if (n, lam, sd) in done: continue
            X, y = make_hard_instance(n_A=g['n_A'], n_B=g['n_B'], n_C=g['n_C'],
                                      d_B=g['d_B'], n_samples=CONFIG['n_samples'], seed=sd)
            r, R = build_qubo(X, y)
            Ja, combos, best = exact_solve(r, R, k, lam); Jx = float(Ja[best])
            S = fair_qaoa(r, R, k, lam, qc['p'], qc['mu'], qc['steps'], qc['lr'], qc['restarts'], sd)
            dJ = Jx - eval_J(S, r, R, lam)
            records.append(dict(n=n, k=k, lam=lam, seed=sd, method='QAOA-fair', dJ=float(dJ)))
            json.dump(dict(config=CONFIG, records=records), open(OUT, 'w'), indent=2)
            print(f"  n={n} lam={lam} seed={sd}  QAOA-fair dJ={dJ:.3f}  [{time.time()-t0:.0f}s]", flush=True)

agg = {}
for r in records: agg.setdefault(r['n'], []).append(r['dJ'])
print("\n n | QAOA-fair mean dJ   (single-start row: 16=0.170 18=0.163 20=0.231 22=0.201)")
for n in sorted(agg): print(f" {n} | {np.mean(agg[n]):.3f}")
print(f"\nsaved -> {OUT}")
