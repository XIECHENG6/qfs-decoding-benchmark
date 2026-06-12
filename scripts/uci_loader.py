"""
uci_loader.py  --  robust UCI dataset loader with local caching.

Downloads Ionosphere and Sonar from the UCI ML repository ONCE, caches the raw
files under ../data/uci/, and thereafter reads offline. Labels are mapped to {0,1}.
WDBC stays on scikit-learn (ships offline). Makes the repo self-contained and
removes the OpenML 504 dependency.

API:
    load_uci(name) -> (X float[n,d], y int[n] in {0,1})   name in {ionosphere,sonar}
    load_all()     -> dict name -> (X, y)   incl. WDBC from sklearn
"""
import os, urllib.request
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uci")
os.makedirs(DATA, exist_ok=True)

SOURCES = {
    "ionosphere": "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
    "sonar":      "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
}
POS_LABEL = {"ionosphere": "g", "sonar": "R"}   # fixed mapping to class 1


def _cache_path(name):
    return os.path.join(DATA, name + ".data")


def _ensure(name, timeout=30):
    p = _cache_path(name)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    req = urllib.request.Request(SOURCES[name], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    with open(p, "wb") as f:
        f.write(data)
    return p


def load_uci(name):
    p = _ensure(name)
    rows = [ln.strip().split(",") for ln in open(p) if ln.strip()]
    X = np.array([[float(v) for v in r[:-1]] for r in rows], dtype=float)
    y = np.array([1 if r[-1].strip() == POS_LABEL[name] else 0 for r in rows], dtype=int)
    return X, y


def load_all():
    from sklearn.datasets import load_breast_cancer
    out = {}
    Xw, yw = load_breast_cancer(return_X_y=True)
    out["WDBC"] = (Xw, yw.astype(int))
    for name, disp in [("ionosphere", "Ionosphere"), ("sonar", "Sonar")]:
        try:
            out[disp] = load_uci(name)
        except Exception as e:
            print(f"  (skipped {disp}: {e})")
    return out


if __name__ == "__main__":
    for name in ["ionosphere", "sonar"]:
        try:
            X, y = load_uci(name)
            print(f"{name:12s} X={X.shape} classes={sorted(set(y.tolist()))} "
                  f"pos_frac={y.mean():.3f}")
        except Exception as e:
            print(f"{name:12s} FAILED: {e}")
