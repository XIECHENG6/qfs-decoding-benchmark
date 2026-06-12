# -*- coding: utf-8 -*-
"""Merge fair-QAOA results (local n=16 + Colab n=18-22), compare against
scaling_ablation.json (ring/no-ent/all/SA/MF) and single-start scaling_qaoa.json.

n=16 per-seed values are reconstructed from results/scaling_qaoa_fair.out
(the original local JSON was overwritten by the Colab download, which only
contained ns=[18,20,22]; the .out log keeps all 16 records at 3 d.p.).

Outputs:
  results/scaling_qaoa_fair_merged.json   (all 4 n, 64 records)
  results/qaoa_fair_analysis.md
"""
import json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

# --- n=16 records recovered from scaling_qaoa_fair.out (local run 2026-06-09) ---
N16 = {
    (1.0, 0): 0.037, (1.0, 1): 0.000, (1.0, 2): 0.159, (1.0, 3): 0.013,
    (1.0, 4): 0.000, (1.0, 5): 0.013, (1.0, 6): 0.040, (1.0, 7): 0.000,
    (2.0, 0): 0.207, (2.0, 1): 0.000, (2.0, 2): 0.132, (2.0, 3): 0.091,
    (2.0, 4): 0.055, (2.0, 5): 0.145, (2.0, 6): 0.150, (2.0, 7): 0.058,
}

fair = json.load(open(os.path.join(RES, "scaling_qaoa_fair.json"), encoding="utf-8"))
recs = [{"n": 16, "k": 7, "lam": lam, "seed": s, "method": "QAOA-fair", "dJ": v,
         "source": "local .out (3 d.p.)"} for (lam, s), v in sorted(N16.items())]
recs += fair["records"]

merged = {"config": fair["config"], "note": "n=16 from local run (scaling_qaoa_fair.out); "
          "n=18-22 from Colab. Same instances/seeds/k/lambda as scaling_ablation.json.",
          "records": recs}
merged["config"]["ns"] = [16, 18, 20, 22]
with open(os.path.join(RES, "scaling_qaoa_fair_merged.json"), "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

abl = json.load(open(os.path.join(RES, "scaling_ablation.json"), encoding="utf-8"))
ss = json.load(open(os.path.join(RES, "scaling_qaoa.json"), encoding="utf-8"))
ss_recs = ss["records"] if isinstance(ss, dict) else ss

def key(r): return (r["n"], r["lam"], r["seed"])
fair_d = {key(r): r["dJ"] for r in recs}
ss_d = {key(r): r["dJ"] for r in ss_recs}
abl_d = {}
for r in abl:
    abl_d.setdefault(r["method"], {})[key(r)] = r["dJ"]

ns = [16, 18, 20, 22]
methods = sorted(abl_d.keys())
lines = ["# Fair QAOA (6-restart, A1 settings) vs baselines\n"]
lines.append("## Per-n mean dJ (8 seeds x 2 lambda = 16 runs)\n")
hdr = "| n | QAOA-fair | QAOA-fair med | QAOA single-start | " + " | ".join(methods) + " |"
lines.append(hdr)
lines.append("|" + "---|" * (4 + len(methods)))
for n in ns:
    fv = [v for (nn, l, s), v in fair_d.items() if nn == n]
    sv = [v for (nn, l, s), v in ss_d.items() if nn == n]
    row = f"| {n} | {np.mean(fv):.3f} ± {np.std(fv):.3f} | {np.median(fv):.3f} | {np.mean(sv):.3f} |"
    for m in methods:
        mv = [v for (nn, l, s), v in abl_d[m].items() if nn == n]
        row += f" {np.mean(mv):.3f} |"
    lines.append(row)

lines.append("\n## Per-lambda fair means\n")
lines.append("| n | lam=1 | lam=2 |")
lines.append("|---|---|---|")
for n in ns:
    l1 = [v for (nn, l, s), v in fair_d.items() if nn == n and l == 1.0]
    l2 = [v for (nn, l, s), v in fair_d.items() if nn == n and l == 2.0]
    lines.append(f"| {n} | {np.mean(l1):.3f} | {np.mean(l2):.3f} |")

lines.append("\n## Paired tests (same instance = n,lam,seed)\n")
def paired(a, b, keys):
    da = np.array([a[k] for k in keys]); db = np.array([b[k] for k in keys])
    diff = da - db
    try:
        w = stats.wilcoxon(da, db)
        p = w.pvalue
    except ValueError:
        p = float("nan")
    return np.mean(diff), np.mean(da < db - 1e-9), p

common_ss = sorted(set(fair_d) & set(ss_d))
m, wf, p = paired(fair_d, ss_d, common_ss)
lines.append(f"- fair vs single-start (all n, N={len(common_ss)}): mean diff {m:+.3f}, "
             f"fair-better frac {wf:.2f}, Wilcoxon p={p:.2g}")
for n in ns:
    kk = [k for k in common_ss if k[0] == n]
    m, wf, p = paired(fair_d, ss_d, kk)
    lines.append(f"  - n={n}: mean diff {m:+.3f}, fair-better {wf:.2f}, p={p:.2g}")

for ref in ["VQC(ring)", "VQC(all)", "VQC(no-ent)", "MeanField"]:
    if ref not in abl_d: continue
    common = sorted(set(fair_d) & set(abl_d[ref]))
    m, wf, p = paired(fair_d, abl_d[ref], common)
    lines.append(f"- fair vs {ref} (all n, N={len(common)}): mean diff {m:+.3f}, "
                 f"fair-better frac {wf:.2f}, Wilcoxon p={p:.2g}")
    for n in ns:
        kk = [k for k in common if k[0] == n]
        m, wf, p = paired(fair_d, abl_d[ref], kk)
        lines.append(f"  - n={n}: mean diff {m:+.3f}, fair-better {wf:.2f}, p={p:.2g}")

# exact-hit rate (dJ <= 1e-3)
lines.append("\n## Exact-hit fraction (dJ <= 1e-3)\n")
for n in ns:
    fv = [v for (nn, l, s), v in fair_d.items() if nn == n]
    lines.append(f"- n={n}: {np.mean(np.array(fv) <= 1e-3):.2f}")

out = "\n".join(lines) + "\n"
with open(os.path.join(RES, "qaoa_fair_analysis.md"), "w", encoding="utf-8") as f:
    f.write(out)
print(out)
print("methods in ablation:", methods)
