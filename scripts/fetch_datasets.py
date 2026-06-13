"""
fetch_datasets.py -- one-time fetch + repackage of the real-world benchmark
datasets into offline, reproducible .npz files under ../data/sets/.

Sources: scikit-learn built-ins (WDBC, Wine, Digits) and the OpenML repository
(the rest). Each dataset is coerced to a float feature matrix X and an
integer-coded label vector y, then saved as data/sets/<name>.npz with arrays
X, y and string attrs source/openml_id. Reproduction reads these .npz files
offline -- no network or ARFF parser needed at run time.

Run ONCE (needs internet, DIRECT connection -- the OpenML/UCI hosts fail through
the local Clash proxy, so we clear proxy env vars here). Datasets that fail to
fetch are skipped with a warning; the benchmark uses whatever succeeded.
"""
import os
# OpenML/UCI hosts handshake-fail through the local proxy; force direct.
for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(k, None)

import socket
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, fetch_openml
from sklearn.preprocessing import LabelEncoder

socket.setdefaulttimeout(40)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "sets")
os.makedirs(OUT, exist_ok=True)

# display name -> (n_features, n_classes) is filled in after load.
# OpenML numeric, no-missing classification datasets (verified standard FS benchmarks).
OPENML = {
    "Ionosphere":  59,
    "Sonar":       40,
    "Parkinsons":  1488,
    "Spambase":    44,
    "QSAR-biodeg": 1494,
    "Vehicle":     54,
    "Hill-Valley": 1479,
}


def _clean(X, y):
    """Coerce X to float, integer-encode y, drop non-finite rows."""
    X = np.asarray(X, dtype=float)
    y = LabelEncoder().fit_transform(np.asarray(y).ravel())
    finite = np.isfinite(X).all(axis=1)
    return X[finite], y[finite].astype(int)


def save(name, X, y, source, oid=""):
    X, y = _clean(X, y)
    path = os.path.join(OUT, name + ".npz")
    np.savez_compressed(path, X=X, y=y, source=source, openml_id=str(oid))
    frac = float((y == y.max()).mean())
    print(f"  {name:14s} X={X.shape} classes={len(set(y.tolist()))} "
          f"maj/total={frac:.2f}  [{source} {oid}]")


def main():
    print("Saving datasets to data/sets/ (offline .npz) ...\n[sklearn built-ins]")
    for name, loader in [("WDBC", load_breast_cancer), ("Wine", load_wine),
                         ("Digits", load_digits)]:
        d = loader()
        save(name, d.data, d.target, "sklearn")

    print("[OpenML]")
    for name, oid in OPENML.items():
        try:
            d = fetch_openml(data_id=oid, as_frame=False, parser="liac-arff")
            save(name, d.data, d.target, "openml", oid)
        except Exception as e:
            print(f"  {name:14s} SKIPPED ({type(e).__name__}: {str(e)[:70]})")

    got = sorted(f[:-4] for f in os.listdir(OUT) if f.endswith(".npz"))
    print(f"\n{len(got)} datasets ready: {', '.join(got)}")


if __name__ == "__main__":
    main()
