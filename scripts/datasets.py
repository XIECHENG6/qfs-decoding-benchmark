"""
datasets.py -- offline loader for the expanded real-world benchmark suite.

Reads every data/sets/<name>.npz produced once by fetch_datasets.py (each holds
float X and integer-coded y). No network or ARFF parser needed at run time.

API:
    load_all_sets() -> dict  display_name -> (X float[n,d], y int[n])
    DATASET_ORDER   -> preferred display order (skips any that are absent)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = os.path.join(HERE, "..", "data", "sets")

# Preferred presentation order (small -> large feature count, binary then multi).
DATASET_ORDER = ["WDBC", "Ionosphere", "Sonar", "Parkinsons", "Spambase",
                 "QSAR-biodeg", "Hill-Valley", "Wine", "Vehicle", "Digits"]


def load_all_sets():
    out = {}
    present = {f[:-4] for f in os.listdir(SETS) if f.endswith(".npz")}
    ordered = [n for n in DATASET_ORDER if n in present] + \
              sorted(present - set(DATASET_ORDER))
    for name in ordered:
        d = np.load(os.path.join(SETS, name + ".npz"), allow_pickle=True)
        out[name] = (d["X"].astype(float), d["y"].astype(int))
    return out


if __name__ == "__main__":
    for name, (X, y) in load_all_sets().items():
        print(f"{name:14s} X={X.shape} classes={len(set(y.tolist()))} "
              f"maj={float((y==np.bincount(y).argmax()).mean()):.2f}")
