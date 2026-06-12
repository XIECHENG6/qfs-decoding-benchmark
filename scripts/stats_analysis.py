# -*- coding: utf-8 -*-
"""
Tier-1 (B): statistical-rigour pass over the existing results/*.json.
No new compute -- pure re-analysis. Produces bootstrap 95% CIs, paired
Wilcoxon signed-rank tests and interpretable effect sizes for every claim
the paper makes, and writes a drop-in markdown report to
results/stats_summary.md (+ a machine-readable results/stats_summary.json).

Claims tested
-------------
Scaling (n=16-22) and ablation (n=12,14):
  * classical SA / mean-field reach the exact optimum (fraction exact, CI)
  * every circuit leaves a gap > 0                       (Wilcoxon vs 0, greater)
  * no quantum advantage: circuit gap > mean-field gap   (paired Wilcoxon, greater)
  * QAOA vs ring, QAOA vs no-ent, entanglement ordering  (paired Wilcoxon, 2-sided)
Decoding (composite + depolarizing):
  * sampling > expval / topk at high noise               (paired Wilcoxon, greater)
  * sampling is flat in noise; expval collapses          (paired Wilcoxon, 2-sided)
  * crossover noise vs the closed-form p1*
"""
import json, os, glob
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
RNG = np.random.default_rng(0)
NBOOT = 10000
EXACT_TOL = 1e-9

out_lines = []          # markdown
J = {}                  # machine-readable


def md(s=""):
    out_lines.append(s)


def boot_ci(x, func=np.mean, nboot=NBOOT, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(nboot, len(x)))
    stat = func(x[idx], axis=1)
    lo, hi = np.percentile(stat, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(func(x)), float(lo), float(hi)


def fmt_ci(m, lo, hi, d=3):
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def wilcoxon_safe(a, b=None, alternative="two-sided"):
    """Return (stat, p, n_used). Handles all-zero / tiny samples gracefully."""
    a = np.asarray(a, float)
    if b is not None:
        b = np.asarray(b, float)
        d = a - b
    else:
        d = a
    nz = d[np.abs(d) > 1e-12]
    if len(nz) < 1:
        return (np.nan, 1.0, 0)         # no differences at all -> indistinguishable
    try:
        st, p = stats.wilcoxon(nz, alternative=alternative, zero_method="wilcox")
        return (float(st), float(p), int(len(nz)))
    except Exception:
        # fall back to sign test
        pos = int(np.sum(nz > 0)); n = len(nz)
        p = stats.binomtest(pos, n, 0.5,
                            alternative={"greater": "greater", "less": "less"}.get(alternative, "two-sided")).pvalue
        return (float(pos), float(p), n)


def win_fraction(a, b):
    """fraction of paired instances where a > b (gap-of-a worse than gap-of-b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    nz = d[np.abs(d) > 1e-12]
    if len(nz) == 0:
        return 0.5
    return float(np.mean(nz > 0))


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# ---------------------------------------------------------------------------
# Load + merge the scaling data (ablation + qaoa share the same instances)
# ---------------------------------------------------------------------------
def load_records():
    abl = json.load(open(os.path.join(RES, "scaling_ablation.json"), encoding="utf-8"))
    qaoa = json.load(open(os.path.join(RES, "scaling_qaoa.json"), encoding="utf-8"))
    scaling = abl + qaoa
    ph2 = json.load(open(os.path.join(RES, "phase2_ablation_strong.json"), encoding="utf-8"))["records"]
    # normalise the downstream-accuracy key name
    for r in ph2:
        if "dSVM" not in r and "dsvm" in r:
            r["dSVM"] = r["dsvm"]
    return scaling, ph2


def key(r):
    return (r["n"], r["lam"], r["seed"])


def paired(records, m_a, m_b, ns=None):
    """Return aligned dJ arrays for two methods over shared (n,lam,seed) keys."""
    A = {key(r): r["dJ"] for r in records if r["method"] == m_a and (ns is None or r["n"] in ns)}
    B = {key(r): r["dJ"] for r in records if r["method"] == m_b and (ns is None or r["n"] in ns)}
    keys = sorted(set(A) & set(B))
    return np.array([A[k] for k in keys]), np.array([B[k] for k in keys]), keys


def method_dJ(records, m, ns=None):
    return np.array([r["dJ"] for r in records if r["method"] == m and (ns is None or r["n"] in ns)])


def analyse_block(records, title, ns_list, classical, circuits, ring_name, noent_name, all_name, tag):
    md(f"## {title}\n")
    methods = sorted(set(r["method"] for r in records))
    J[tag] = {"methods": methods, "per_n": {}, "tests": {}}

    # ---- per-n mean [CI] table ----
    md("### Optimality gap $\\Delta J$ — mean [95% bootstrap CI]\n")
    header = "| Method | " + " | ".join(f"n={n}" for n in ns_list) + " |"
    md(header)
    md("|" + "---|" * (len(ns_list) + 1))
    order = classical + circuits
    for m in order:
        cells = []
        for n in ns_list:
            x = method_dJ(records, m, ns=[n])
            if len(x):
                mlo = boot_ci(x)
                cells.append(fmt_ci(*mlo))
                J[tag]["per_n"].setdefault(m, {})[str(n)] = {
                    "mean": mlo[0], "ci_lo": mlo[1], "ci_hi": mlo[2],
                    "median": float(np.median(x)), "n": int(len(x)),
                    "frac_exact": float(np.mean(np.abs(x) < EXACT_TOL))}
            else:
                cells.append("-")
        md(f"| {m} | " + " | ".join(cells) + " |")
    md("")

    # ---- classical exactness ----
    md("### Classical exactness (fraction of runs at the exact optimum, $|\\Delta J|<10^{-9}$)\n")
    md("| Method | " + " | ".join(f"n={n}" for n in ns_list) + " | pooled |")
    md("|" + "---|" * (len(ns_list) + 2))
    for m in classical:
        cells = []
        for n in ns_list:
            x = method_dJ(records, m, ns=[n])
            cells.append(f"{100*np.mean(np.abs(x)<EXACT_TOL):.0f}%" if len(x) else "-")
        xall = method_dJ(records, m)
        cells.append(f"{100*np.mean(np.abs(xall)<EXACT_TOL):.0f}%")
        md(f"| {m} | " + " | ".join(cells) + " |")
    md("")

    # ---- gap > 0 for every circuit (pooled) ----
    md("### Every circuit leaves a strictly positive gap (Wilcoxon signed-rank vs 0, one-sided `greater`)\n")
    md("| Circuit | mean $\\Delta J$ [CI] | median | p (>0) | sig |")
    md("|---|---|---|---|---|")
    for m in circuits:
        x = method_dJ(records, m)
        if not len(x):
            continue
        mlo = boot_ci(x)
        _, p, _ = wilcoxon_safe(x, alternative="greater")
        md(f"| {m} | {fmt_ci(*mlo)} | {np.median(x):.3f} | {p:.2e} | {stars(p)} |")
        J[tag]["tests"].setdefault("gap_gt_0", {})[m] = {"p": p, "mean": mlo[0]}
    md("")

    # ---- no advantage: circuit gap > mean-field gap (paired) ----
    mf = "MeanField" if "MeanField" in methods else ("MeanField(cl)" if "MeanField(cl)" in methods else None)
    if mf:
        md(f"### No quantum advantage: each circuit's gap **exceeds** the free classical mean-field gap (paired Wilcoxon, one-sided `greater` vs `{mf}`)\n")
        md("| Circuit | median($\\Delta J_\\mathrm{circ}-\\Delta J_\\mathrm{MF}$) | win-frac | p | sig |")
        md("|---|---|---|---|---|")
        for m in circuits:
            a, b, _ = paired(records, m, mf)
            if not len(a):
                continue
            d = a - b
            _, p, _ = wilcoxon_safe(a, b, alternative="greater")
            md(f"| {m} | {np.median(d):+.3f} | {win_fraction(a,b):.2f} | {p:.2e} | {stars(p)} |")
            J[tag]["tests"].setdefault("circuit_gt_mf", {})[m] = {"p": p, "median_diff": float(np.median(d)), "win_frac": win_fraction(a, b)}
    md("")

    # ---- QAOA vs ring / no-ent + entanglement ordering (paired, two-sided) ----
    md("### Paired circuit comparisons (Wilcoxon signed-rank, two-sided) — pooled and per-n\n")
    md("| Comparison (A vs B) | scope | median(A−B) [CI] | A-worse frac | p | sig |")
    md("|---|---|---|---|---|---|")
    qaoa = next((m for m in methods if m.startswith("QAOA")), None)
    comps = []
    if qaoa and ring_name in methods:
        comps.append((qaoa, ring_name))
    if qaoa and noent_name in methods:
        comps.append((qaoa, noent_name))
    if noent_name in methods and ring_name in methods:
        comps.append((noent_name, ring_name))
    if ring_name in methods and all_name in methods:
        comps.append((ring_name, all_name))
    for (ma, mb) in comps:
        for scope, ns in [("pooled", None)] + [(f"n={n}", [n]) for n in ns_list]:
            a, b, _ = paired(records, ma, mb, ns=ns)
            if not len(a):
                continue
            d = a - b
            dlo = boot_ci(d, func=np.median)
            _, p, _ = wilcoxon_safe(a, b, alternative="two-sided")
            md(f"| {ma} vs {mb} | {scope} | {fmt_ci(*dlo)} | {win_fraction(a,b):.2f} | {p:.2e} | {stars(p)} |")
            if scope == "pooled":
                J[tag]["tests"].setdefault("paired", {})[f"{ma}__vs__{mb}"] = {
                    "median_diff": dlo[0], "ci": [dlo[1], dlo[2]], "win_frac": win_fraction(a, b), "p": p}
    md("")


def decoding_block():
    md("## Decoding crossover — statistical robustness\n")
    ph3 = json.load(open(os.path.join(RES, "phase3_realistic.json"), encoding="utf-8"))
    grid = json.load(open(os.path.join(RES, "decoder_grid.json"), encoding="utf-8"))
    J["decoding"] = {}

    for label, data, p1 in [("Composite noise (T1/T2 + coherent + readout)", ph3["acc"], ph3.get("p1_star")),
                            ("Depolarizing only", grid, None)]:
        md(f"### {label}" + (f"  (closed-form $p_1^*={p1:.2e}$)" if p1 else "") + "\n")
        decoders = list(data.keys())
        noises = sorted(data[decoders[0]].keys(), key=lambda s: float(s))
        # mean [CI] table
        md("| Decoder | " + " | ".join(f"s={s}" for s in noises) + " |")
        md("|" + "---|" * (len(noises) + 1))
        store = {}
        for dec in decoders:
            cells = []
            for s in noises:
                x = np.asarray(data[dec][s], float)
                store[(dec, s)] = x
                m, lo, hi = boot_ci(x)
                cells.append(f"{m:.3f} [{lo:.3f},{hi:.3f}]")
            md(f"| {dec} | " + " | ".join(cells) + " |")
        md("")
        s_lo, s_hi = noises[0], noises[-1]
        tests = {}
        # collapse: expval(s_hi) vs expval(s_lo)
        for dec in decoders:
            a, b = store[(dec, s_hi)], store[(dec, s_lo)]
            _, p, _ = wilcoxon_safe(a, b, alternative="two-sided")
            drop = float(np.mean(b) - np.mean(a))
            tests[f"{dec}_drop_{s_lo}_to_{s_hi}"] = {"mean_drop": drop, "p": p}
            md(f"- **{dec}**: accuracy change from s={s_lo} to s={s_hi} = {-drop:+.3f}  (Wilcoxon p={p:.2e}, {stars(p)})")
        # sampling vs expval / topk at highest noise
        if "sampling" in decoders and "expval" in decoders:
            a, b = store[("sampling", s_hi)], store[("expval", s_hi)]
            _, p, _ = wilcoxon_safe(a, b, alternative="greater")
            tests["sampling_gt_expval_at_hi"] = {"p": p, "median_diff": float(np.median(a - b))}
            md(f"- **sampling > expval at s={s_hi}**: median Δacc={np.median(a-b):+.3f}, Wilcoxon one-sided p={p:.2e} ({stars(p)})")
        if "sampling" in decoders and "topk" in decoders:
            a, b = store[("sampling", s_hi)], store[("topk", s_hi)]
            _, p, _ = wilcoxon_safe(a, b, alternative="greater")
            md(f"- **sampling > topk at s={s_hi}**: median Δacc={np.median(a-b):+.3f}, Wilcoxon one-sided p={p:.2e} ({stars(p)})")
        md("")
        J["decoding"][label] = tests


def main():
    md("# Statistical-rigour summary (Tier-1 B)\n")
    md("Generated by `scripts/stats_analysis.py` from `results/*.json`. "
       "Bootstrap CIs use 10 000 resamples (seed 0); paired tests are Wilcoxon "
       "signed-rank over shared (n, λ, seed) instances. `***`<1e-3, `**`<1e-2, `*`<0.05, `ns`≥0.05.\n")

    scaling, ph2 = load_records()

    analyse_block(
        scaling, "Scaling benchmark (n = 16–22)", [16, 18, 20, 22],
        classical=["SA", "MeanField"],
        circuits=["VQC(no-ent)", "VQC(ring)", "VQC(all)", "QAOA-mRMR"],
        ring_name="VQC(ring)", noent_name="VQC(no-ent)", all_name="VQC(all)", tag="scaling")

    # phase2 method names differ (suffix -mRMR on some); discover them
    ph2_methods = sorted(set(r["method"] for r in ph2))
    cl2 = [m for m in ph2_methods if m in ("SA", "MeanField", "MeanField(cl)")]
    circ2 = [m for m in ph2_methods if m.startswith("VQC") or m.startswith("QAOA")]
    ring2 = next((m for m in ph2_methods if "ring" in m), "VQC(ring)")
    noent2 = next((m for m in ph2_methods if "no-ent" in m or "noent" in m), "VQC(no-ent)")
    all2 = next((m for m in ph2_methods if "all" in m), "VQC(all)")
    ns2 = sorted(set(r["n"] for r in ph2))
    analyse_block(
        ph2, "Entanglement ablation (n = 12, 14)", ns2,
        classical=cl2, circuits=circ2,
        ring_name=ring2, noent_name=noent2, all_name=all2, tag="ablation")

    decoding_block()

    rep = os.path.join(RES, "stats_summary.md")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    with open(os.path.join(RES, "stats_summary.json"), "w", encoding="utf-8") as f:
        json.dump(J, f, indent=2)
    print("wrote", rep)
    print("wrote", os.path.join(RES, "stats_summary.json"))
    # echo the headline numbers
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
