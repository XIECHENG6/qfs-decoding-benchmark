"""
qfs_common.py  --  Shared engine for the mRMR-QUBO feature-selection rework (Path A).

Pure classical: numpy + scikit-learn only (NO quantum dependency).
Phase-2 (quantum / PennyLane) MUST import this module and load the SAME saved
instances from ../data so the comparison is on identical problems.

Problem (mRMR-QUBO), maximised over k-subsets S (|S| = k):
        J(S) = sum_{i in S} r_i  -  lambda * sum_{i<j in S} R_ij
    r_i  = normalised mutual information MI(f_i ; y)        (relevance)
    R_ij = normalised mutual information MI(f_i ; f_j)      (redundancy)

This is genuinely NP-hard (dense R), unlike the paper's current linear-relevance
QUBO whose optimum is just "sort by r_i". Making the problem hard is the whole
point of Path A: it is what lets the method beat a random baseline.
"""

import os
# Pin native thread pools BEFORE numpy/scipy/sklearn load: avoids a Windows
# OpenMP crash after many repeated SVC/cross_val_score calls.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------
# 1. Synthetic "relevance-redundancy trap" instance
# ----------------------------------------------------------------------------
def make_hard_instance(n_A=6, n_B=10, n_C=6, n_samples=800,
                       rho_A=0.92, w_A=2.2, w_B=1.0, d_B=6,
                       noise_B=0.35, label_noise=0.5, seed=0):
    """
    Plants a trap that defeats relevance-only / random selection:

      Group A (n_A): redundant near-copies of ONE strong latent signal s_A.
                     Each A-feature has the HIGHEST relevance to y, but they are
                     mutually redundant -> taking >1 of them wastes the budget.
      Group B (n_B): each tied to a DISTINCT independent latent signal. The first
                     d_B of them genuinely drive y (moderate relevance, ~zero
                     mutual redundancy). The remaining (n_B - d_B) are weak.
      Group C (n_C): pure noise, irrelevant.

    The label y depends on s_A AND the d_B informative B-signals, so a GOOD subset
    must be DIVERSE: ~1 from A + the informative B's. Sorting by relevance grabs
    many A-copies (redundant) and misses B -> structurally sub-optimal. Hence real
    combinatorial optimisation (mRMR) is required.

    Returns
        X      : (n_samples, n) feature matrix,  n = n_A + n_B + n_C
        y      : (n_samples,) binary labels (~balanced)
        groups : (n,) array of tags 'A' / 'B' (informative) / 'b' (weak) / 'C'
    """
    rng = np.random.default_rng(seed)
    N = n_samples
    s_A = rng.standard_normal(N)
    s_B = rng.standard_normal((n_B, N))

    logit = w_A * s_A + w_B * s_B[:d_B].sum(0)
    logit = (logit - logit.mean()) / (logit.std() + 1e-9)
    y = (logit + label_noise * rng.standard_normal(N) > 0).astype(int)

    cols, groups = [], []
    for _ in range(n_A):                                   # redundant relevant block
        cols.append(rho_A * s_A + np.sqrt(max(1 - rho_A**2, 0.0)) * rng.standard_normal(N))
        groups.append('A')
    for j in range(n_B):                                   # independent features
        cols.append(np.sqrt(max(1 - noise_B**2, 0.0)) * s_B[j] + noise_B * rng.standard_normal(N))
        groups.append('B' if j < d_B else 'b')
    for _ in range(n_C):                                   # pure noise
        cols.append(rng.standard_normal(N))
        groups.append('C')

    X = np.column_stack(cols)
    return X, y, np.array(groups)


# ----------------------------------------------------------------------------
# 2. Mutual-information QUBO coefficients (relevance r_i, redundancy R_ij)
# ----------------------------------------------------------------------------
def _discretize(X, n_bins=8):
    Xb = np.zeros_like(X, dtype=int)
    for i in range(X.shape[1]):
        edges = np.quantile(X[:, i], np.linspace(0, 1, n_bins + 1)[1:-1])
        Xb[:, i] = np.digitize(X[:, i], edges)
    return Xb


def _mi(a, b):
    """Discrete mutual information (nats) between integer-coded vectors a, b."""
    Na, Nb = int(a.max()) + 1, int(b.max()) + 1
    joint = np.zeros((Na, Nb))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])))


def build_qubo(X, y, n_bins=8):
    """Return normalised relevance r (n,) and redundancy R (n,n), both in [0,1]."""
    Xb = _discretize(X, n_bins)
    n = X.shape[1]
    r = np.array([_mi(Xb[:, i], y) for i in range(n)])
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            R[i, j] = R[j, i] = _mi(Xb[:, i], Xb[:, j])
    r = r / (r.max() + 1e-12)
    R = R / (R.max() + 1e-12)
    np.fill_diagonal(R, 0.0)
    return r, R


# ----------------------------------------------------------------------------
# 3. Objective + exact solver
# ----------------------------------------------------------------------------
def eval_J(S, r, R, lam):
    """Constrained mRMR objective for a k-subset S (|S| = k enforced by caller)."""
    S = list(S)
    return float(r[S].sum() - lam * R[np.ix_(S, S)].sum() / 2.0)


def exact_solve(r, R, k, lam):
    """
    Vectorised exhaustive optimum over all C(n,k) subsets (feasible n <= ~24).
    Returns (J_all, combos, best_idx): the FULL objective distribution (for
    percentiles), the combo index array, and the argmax.
    """
    n = len(r)
    flat = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(n), k)),
                       dtype=np.int64)
    combos = flat.reshape(-1, k)
    J = r[combos].sum(1)
    for a in range(k):
        for b in range(a + 1, k):
            J = J - lam * R[combos[:, a], combos[:, b]]
    return J, combos, int(np.argmax(J))


# ----------------------------------------------------------------------------
# 4. Baselines  (classical_meanfield is the no-quantum analog of VQC-QFS)
# ----------------------------------------------------------------------------
def relevance_topk(r, k):
    """MI top-k: sort by relevance only (the paper's current implicit solution)."""
    return np.argsort(-r)[:k]


def random_mrmr(r, R, k, lam, shots=4096, seed=0):
    """Best-of-N random k-subsets ranked by the mRMR objective (honest random baseline)."""
    rng = np.random.default_rng(seed)
    n = len(r)
    bestJ, bestS = -np.inf, None
    for _ in range(shots):
        S = rng.choice(n, k, replace=False)
        v = eval_J(S, r, R, lam)
        if v > bestJ:
            bestJ, bestS = v, S
    return np.array(sorted(bestS))


def greedy_mrmr(r, R, k, lam):
    """Standard mRMR forward selection (Peng et al. MID-style marginal gain)."""
    n = len(r)
    S = []
    for _ in range(k):
        best, bi = -np.inf, None
        for i in range(n):
            if i in S:
                continue
            gain = r[i] - lam * (R[i, S].sum() if S else 0.0)
            if gain > best:
                best, bi = gain, i
        S.append(bi)
    return np.array(sorted(S))


def simulated_annealing(r, R, k, lam, iters=15000, T0=0.25, seed=0):
    """Strong classical optimiser (swap-move Metropolis). Expected near-optimal."""
    rng = np.random.default_rng(seed)
    n = len(r)
    S = set(rng.choice(n, k, replace=False).tolist())
    cur = eval_J(list(S), r, R, lam)
    best, bestS = cur, set(S)
    for t in range(iters):
        T = T0 * (1 - t / iters) + 1e-4
        out = [o for o in range(n) if o not in S]
        i = rng.choice(list(S))
        j = rng.choice(out)
        S2 = set(S); S2.discard(i); S2.add(j)
        nv = eval_J(list(S2), r, R, lam)
        if nv > cur or rng.random() < np.exp((nv - cur) / T):
            S, cur = S2, nv
            if nv > best:
                best, bestS = nv, set(S)
    return np.array(sorted(bestS))


def classical_meanfield(r, R, k, lam, T=500, lr=0.05, alpha=5.0,
                        tau0=1.0, tau1=10.0, restarts=8, seed=0):
    """
    *** The critical ablation: VQC-QFS with the quantum circuit removed. ***
    Free per-feature parameters theta -> p_i = sigmoid(tau * theta_i), temperature
    annealed 1->10 (identical relaxation to VQC-QFS), Adam ascent on the SOFT
    mRMR objective, then top-k decode. If VQC-QFS cannot beat THIS, the quantum
    circuit is contributing nothing.
    """
    rng = np.random.default_rng(seed)
    n = len(r)
    bestJ, bestS = -np.inf, None
    for _ in range(restarts):
        theta = rng.normal(0, 0.1, n)
        m = np.zeros(n); v = np.zeros(n)
        for t in range(T):
            tau = tau0 + (tau1 - tau0) * t / max(T - 1, 1)
            p = 1.0 / (1.0 + np.exp(-tau * theta))
            # g(p) = r.p - 0.5 lam p^T R p - alpha (sum p - k)^2  (maximise)
            dg_dp = r - lam * (R @ p) - 2 * alpha * (p.sum() - k)
            grad = dg_dp * (tau * p * (1 - p))          # chain rule d p / d theta
            g = -grad                                   # Adam minimises
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** (t + 1))
            vh = v / (1 - 0.999 ** (t + 1))
            theta -= lr * mh / (np.sqrt(vh) + 1e-8)
        p = 1.0 / (1.0 + np.exp(-tau1 * theta))
        S = np.argsort(-p)[:k]
        vJ = eval_J(S, r, R, lam)
        if vJ > bestJ:
            bestJ, bestS = vJ, S
    return np.array(sorted(bestS))


# ----------------------------------------------------------------------------
# 5. Downstream evaluation + helpers
# ----------------------------------------------------------------------------
def downstream_svm(X, y, S, seed=0):
    """5-fold stratified CV accuracy of an SVM-RBF on the selected subset."""
    Xs = StandardScaler().fit_transform(X[:, list(S)])
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(SVC(kernel='rbf'), Xs, y, cv=cv).mean())


def composition(S, groups):
    """Count how many selected features fall in each group (A/B/b/C)."""
    g = groups[list(S)]
    return {t: int((g == t).sum()) for t in ['A', 'B', 'b', 'C']}


def percentile_of(value, J_all):
    """Percentile rank of an objective value within the exact distribution."""
    return float((J_all < value).mean() * 100.0)
