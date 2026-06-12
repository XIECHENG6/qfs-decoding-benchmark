"""
make_figures.py  --  Generate the two paper figures from the saved results JSON.

  1) figures/ablation_bars.{png,pdf}   <- results/phase2_ablation_strong.json
       Optimality gap dJ = J* - J per method, grouped by n, classical vs quantum.
       (Visualises Table tab:ablation.)
  2) figures/decoding_crossover.{png,pdf} <- results/phase3_realistic.json
       Downstream accuracy vs composite-noise strength s for the three decoders,
       with the closed-form critical rate p1* and the chance line.
       (Visualises Table tab:decoding.)

Pure matplotlib (Agg). Run: python scripts/make_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})

CLA = "#4C72B0"   # classical
QUA = "#C44E52"   # quantum


def _save(fig, stem):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("  wrote", os.path.relpath(p, ROOT))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: entanglement-ablation optimality gap
# ---------------------------------------------------------------------------
def fig_ablation():
    d = json.load(open(os.path.join(RES, "phase2_ablation_strong.json")))
    recs = d["records"]
    ns = sorted({r["n"] for r in recs})                       # [12, 14]
    # display order + pretty labels; flag which are quantum
    order = [
        ("Random-mRMR", "Random", False), ("Greedy-mRMR", "Greedy", False),
        ("SA", "SA", False), ("MeanField(cl)", "Mean-field", False),
        ("VQC(no-ent)", "VQC\nno-ent", True), ("VQC(ring)", "VQC\nring", True),
        ("VQC(all)", "VQC\nall-to-all", True), ("QAOA-mRMR", "QAOA\nmRMR", True),
    ]

    def agg(method, n):
        v = [r["dJ"] for r in recs if r["method"] == method and r["n"] == n]
        return float(np.mean(v)), float(np.std(v) / np.sqrt(len(v))), len(v)

    print("ablation dJ (mean over seeds x lambda):")
    means = {n: [] for n in ns}; sems = {n: [] for n in ns}
    for key, _lab, _q in order:
        line = f"  {key:<14}"
        for n in ns:
            m, se, cnt = agg(key, n)
            means[n].append(m); sems[n].append(se)
            line += f" n{n}={m:.3f}(N={cnt})"
        print(line)

    labels = [lab for _k, lab, _q in order]
    is_q = [q for _k, _l, q in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    # shade quantum region
    qstart = is_q.index(True)
    ax.axvspan(qstart - 0.5, len(order) - 0.5, color=QUA, alpha=0.06, zorder=0)
    ax.axvspan(-0.5, qstart - 0.5, color=CLA, alpha=0.06, zorder=0)

    for off, n, hatch, alpha in [(-w / 2, ns[0], None, 0.95), (w / 2, ns[1], "//", 0.95)]:
        colors = [QUA if q else CLA for q in is_q]
        ax.bar(x + off, means[n], w, yerr=sems[n], capsize=2.5,
               color=colors, alpha=alpha, hatch=hatch, edgecolor="white",
               linewidth=0.5, error_kw=dict(lw=0.8, ecolor="0.3"),
               label=f"$n={n}$")

    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"optimality gap  $\Delta J = J^\star - J$")
    ax.set_title("Entanglement ablation: every quantum circuit trails the exact optimum")
    ax.text(qstart / 2 - 0.5, ax.get_ylim()[1] * 0.92, "classical\n(reaches exact)",
            ha="center", va="top", color=CLA, fontsize=8.5, fontweight="bold")
    ax.text((qstart + len(order)) / 2 - 0.5, ax.get_ylim()[1] * 0.92,
            "quantum\n(gap > 0)", ha="center", va="top", color=QUA,
            fontsize=8.5, fontweight="bold")
    # legend: n hatch + group colours
    h_n = [Patch(facecolor="0.6", label=f"$n={ns[0]}$"),
           Patch(facecolor="0.6", hatch="//", label=f"$n={ns[1]}$")]
    ax.legend(handles=h_n, loc="upper left", frameon=False, ncol=1)
    _save(fig, "ablation_bars")


# ---------------------------------------------------------------------------
# Figure 2: decoder x noise crossover
# ---------------------------------------------------------------------------
def fig_decoding():
    d = json.load(open(os.path.join(RES, "phase3_realistic.json")))
    acc = d["acc"]; p1 = d["p1_star"]
    levels = [float(s) for s in d["config"]["noise_levels"]]    # [0,.001,..,.05]
    xs = np.arange(len(levels))                                  # categorical x
    styles = {"expval": ("Expval", "o", "#C44E52", "-"),
              "topk":   ("Top-$k$", "s", "#DD8452", "--"),
              "sampling": ("Sampling", "^", "#4C72B0", "-")}

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    print("decoding curves (mean +/- std over seeds):")
    for key in ("sampling", "topk", "expval"):
        lab, mk, col, ls = styles[key]
        m = np.array([np.mean(acc[key][str(s)]) for s in d["config"]["noise_levels"]])
        sd = np.array([np.std(acc[key][str(s)]) for s in d["config"]["noise_levels"]])
        ax.plot(xs, m, mk + ls, color=col, label=lab, lw=1.8, ms=6, zorder=3)
        ax.fill_between(xs, m - sd, m + sd, color=col, alpha=0.15, zorder=1)
        print(f"  {lab:<10} " + " ".join(f"{v:.3f}" for v in m))

    # critical rate p1* placed in categorical-x space by log interpolation
    pos = np.searchsorted(levels, p1)            # between level[pos-1] and level[pos]
    lo, hi = levels[pos - 1], levels[pos]
    xp = (pos - 1) + (np.log(p1) - np.log(lo)) / (np.log(hi) - np.log(lo))
    ax.axvline(xp, color="0.3", ls=":", lw=1.4, zorder=2)
    ax.text(xp + 0.08, 0.60, r"$p_1^\star\approx%.1f\times10^{-3}$" % (p1 * 1e3),
            rotation=90, va="bottom", ha="left", fontsize=8.5, color="0.3")
    ax.axhline(0.5, color="0.6", ls="-", lw=0.8)
    ax.text(len(levels) - 1.0, 0.505, "chance", fontsize=8, color="0.5",
            va="bottom", ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels([("0" if s == 0 else f"{s:g}") for s in levels])
    ax.set_xlabel(r"composite-noise strength  $s$")
    ax.set_ylabel("downstream SVM accuracy")
    ax.set_ylim(0.48, 0.80)
    ax.set_title("Decoder choice governs noise resilience\n"
                 "(expval collapses past $p_1^\\star$; sampling stays flat)")
    ax.legend(loc="lower left", frameon=False)
    _save(fig, "decoding_crossover")


# ---------------------------------------------------------------------------
# Figure 3 (F4): scaling of the optimality gap with problem size n
# ---------------------------------------------------------------------------
def fig_scaling():
    d = json.load(open(os.path.join(RES, "scaling_ablation.json")))
    qf = json.load(open(os.path.join(RES, "scaling_qaoa_fair.json")))["records"]  # fair QAOA, all n
    ns = sorted({r["n"] for r in d})                                       # [16,18,20,22]
    x = np.array(ns)

    def agg(recs, method=None):
        m, e = [], []
        for n in ns:
            v = [r["dJ"] for r in recs if r["n"] == n and (method is None or r["method"] == method)]
            m.append(float(np.mean(v)) if v else np.nan)
            e.append(float(np.std(v) / np.sqrt(len(v))) if v else 0.0)
        return np.array(m), np.array(e)

    # (label, recs, method, colour, linestyle, marker)
    classical = [
        ("SA",           d, "SA",          "#1b3a6b", "-", "o"),
        ("Mean-field",   d, "MeanField",   "#5b82c2", "-", "s"),
        ("Random (ref.)",d, "Random",      "#9a9a9a", ":", "x"),
    ]
    quantum = [
        ("VQC no-ent",      d,  "VQC(no-ent)", "#7a1718", "--", "v"),
        ("VQC ring",        d,  "VQC(ring)",   "#C44E52", "--", "^"),
        ("VQC all-to-all",  d,  "VQC(all)",    "#e8999b", "--", "D"),
        ("QAOA (fair, 6-restart)", qf, None,   "#b5462b", "--", "P"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.axhspan(-0.004, 0.012, color=CLA, alpha=0.07, zorder=0)
    ax.text(22.35, 0.006, r"classical $\approx$ exact", color=CLA, fontsize=8,
            va="center", ha="right", style="italic")

    print("scaling gap dJ (mean over 8 seeds x 2 lambda):")
    for lab, recs, meth, col, ls, mk in classical + quantum:
        m, e = agg(recs, meth)
        ax.fill_between(x, m - e, m + e, color=col, alpha=0.08, zorder=1)
        ax.plot(x, m, marker=mk, ls=ls, color=col, lw=1.8, ms=6, label=lab, zorder=3)
        print(f"  {lab:<15} " + " ".join(f"{v:.3f}" for v in m))

    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(ns)
    ax.set_xlim(15.1, 22.6)
    ax.set_xlabel(r"problem size  $n$")
    ax.set_ylabel(r"optimality gap  $\Delta J = J^\star - J$")
    ax.set_title("Classical solvers stay exact; the quantum gap grows with $n$")
    ax.set_ylim(-0.018, 0.275)
    ax.legend(ncol=2, frameon=False, fontsize=8, loc="upper left",
              columnspacing=1.2, handlelength=2.2)
    _save(fig, "scaling_gap")


# ---------------------------------------------------------------------------
# Figure 4 (F7): redundancy-weight (lambda) sweep
# ---------------------------------------------------------------------------
def fig_lambda_sweep():
    s = json.load(open(os.path.join(RES, "phase1_summary.json")))["sweep"]
    lam = [r["lam"] for r in s]
    # (label, key, colour, marker, linestyle)
    series = [
        ("MI top-$k$ (relevance-only)", "dSVM_rel",  "#C44E52", "o", "-"),
        ("Random-mRMR",                 "dSVM_rand", "#9a9a9a", "x", ":"),
        ("Mean-field (classical)",      "dSVM_mf",   "#5b82c2", "s", "--"),
        ("SA (classical)",              "dSVM_sa",   "#1b3a6b", "^", "-"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    # shade the region where the redundancy trap is present (lambda > 0)
    ax.axvspan(0.25, max(lam) + 0.15, color="#C44E52", alpha=0.05, zorder=0)
    ax.text(max(lam), 0.083, "redundancy trap present", color="#C44E52", fontsize=8,
            ha="right", va="top", style="italic")

    print("lambda sweep (dSVM gap vs exact):")
    for lab, key, col, mk, ls in series:
        y = [r[key] for r in s]
        ax.plot(lam, y, marker=mk, ls=ls, color=col, lw=1.8, ms=6, label=lab, zorder=3)
        print(f"  {lab:<28} " + " ".join(f"{v:+.3f}" for v in y))

    ax.axhline(0, color="0.5", lw=0.8)
    ax.annotate("$\\lambda=0$: prior-work\nrelevance-only QUBO\n(no trap: random ties optimum)",
                xy=(0, 0.0), xytext=(0.35, 0.045), fontsize=7.6, color="0.25",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax.set_xlabel(r"redundancy weight  $\lambda$")
    ax.set_ylabel(r"downstream SVM gap vs. exact  $\Delta$SVM")
    ax.set_title("The relevance trap appears only with redundancy ($\\lambda>0$)")
    ax.set_ylim(-0.02, 0.092)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    _save(fig, "lambda_sweep")


# ---------------------------------------------------------------------------
# Figure 5 (F8): real-data benchmark (values are tab:realdata = phase1_realdata)
# ---------------------------------------------------------------------------
def fig_realdata():
    # Recorded values from Table tab:realdata (downstream SVM gap vs exact optimum).
    cols = ["WDBC", "Ionosphere", "Sonar"]
    svm_star = [0.968, 0.928, 0.743]
    methods = ["MI top-$k$", "Random", "Greedy", "Mean-field", "SA"]
    vals = {
        "WDBC":       [0.018,  0.000,  0.001, -0.001, 0.000],
        "Ionosphere": [0.049,  0.007,  0.003, -0.002, 0.000],
        "Sonar":      [0.007, -0.003, -0.007,  0.000, 0.000],
    }
    colours = ["#C44E52", "#9a9a9a", "#7fa3d6", "#5b82c2", "#1b3a6b"]
    x = np.arange(len(cols)); w = 0.16
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.axhspan(-0.004, 0.004, color="#4C72B0", alpha=0.06, zorder=0)
    for i, m in enumerate(methods):
        ys = [vals[c][i] for c in cols]
        ax.bar(x + (i - 2) * w, ys, w, color=colours[i], label=m,
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(SVM$^\\star$={sv:.3f})" for c, sv in zip(cols, svm_star)])
    ax.set_ylabel(r"downstream SVM gap vs. exact  $\Delta$SVM  (higher $=$ worse)")
    ax.set_title("Relevance-only fails; greedy / mean-field / SA reach the exact optimum")
    ax.text(2.32, 0.0025, "classical\n$\\approx$ exact", color="#4C72B0", fontsize=7.5,
            ha="right", va="bottom", style="italic")
    ax.legend(frameon=False, fontsize=8.5, ncol=5, loc="upper center",
              columnspacing=1.0, handlelength=1.3, handletextpad=0.4)
    ax.set_ylim(-0.012, 0.064)
    print("real-data benchmark (dSVM gap):")
    for c in cols:
        print(f"  {c:<11} " + " ".join(f"{v:+.3f}" for v in vals[c]))
    _save(fig, "realdata_bars")


# ---------------------------------------------------------------------------
# Figure 6 (F9, appendix): noise-channel isolation of the expval collapse
# ---------------------------------------------------------------------------
def fig_channels():
    g = json.load(open(os.path.join(RES, "decoding_generality.json")))
    ring = g["acc"]["ring"]
    s = "0.05"   # the noise level where the composite collapse appears
    # (label, json-key, colour, is_damping)
    chans = [
        ("Composite",            "composite", "#7a1718", True),
        ("Depolarizing",         "depol",     "#C44E52", True),
        ("Amp/phase\ndamping",   "damping",   "#e8999b", True),
        ("Coherent\nover-rot.",  "coherent",  "#5b82c2", False),
        ("Readout\nflip",        "readout",   "#7fa3d6", False),
    ]
    x = np.arange(len(chans))
    ev_m = [float(np.mean(ring[c[1]]["expval"][s])) for c in chans]
    ev_e = [float(np.std(ring[c[1]]["expval"][s]) / np.sqrt(len(ring[c[1]]["expval"][s]))) for c in chans]
    cols = [c[2] for c in chans]
    samp = float(np.mean(ring["composite"]["sampling"][s]))

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.bar(x, ev_m, 0.62, yerr=ev_e, capsize=3, color=cols, edgecolor="white",
           linewidth=0.6, error_kw=dict(lw=0.8, ecolor="0.3"), zorder=3)
    ax.axhline(samp, color="#1b3a6b", ls="--", lw=1.4, zorder=2)
    ax.text(len(chans) - 0.5, samp + 0.005,
            r"sampling (all channels) $\approx%.2f$" % samp,
            color="#1b3a6b", fontsize=8, ha="right", va="bottom")
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.text(0.0, 0.508, "chance", fontsize=8, color="0.5", va="bottom")
    ax.annotate("compounded\ndamping $\\to$ collapse", xy=(0, ev_m[0]),
                xytext=(0.70, 0.585), fontsize=7.8, color="#7a1718", ha="center",
                arrowprops=dict(arrowstyle="->", color="#7a1718", lw=0.8))
    # subtle background grouping: damping-type (depol, damping) vs non-damping (coherent, readout)
    ax.axvspan(0.5, 2.5, color="#C44E52", alpha=0.045, zorder=0)
    ax.axvspan(2.5, 4.5, color="#5b82c2", alpha=0.045, zorder=0)

    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in chans])
    ax.set_ylabel("expval-decoder accuracy")
    ax.set_title("Channel isolation: only compounded damping collapses expval ($s{=}0.05$)")
    ax.set_ylim(0.48, 0.82)
    print("F9 channel isolation (expval @ s=0.05):")
    for (lab, _k, _c, _d), m in zip(chans, ev_m):
        print(f"  {lab.replace(chr(10),' '):<22} {m:.3f}")
    print(f"  sampling(composite) {samp:.3f}")
    _save(fig, "channel_isolation")


# ---------------------------------------------------------------------------
# Figure 7 (F12, appendix): hardness control -- mRMR vs frustrated
# ---------------------------------------------------------------------------
def fig_hardness():
    h = json.load(open(os.path.join(RES, "hardness_control.json")))
    recs = h["records"]; tol = 1e-6
    methods = [("SA", "SA"), ("Mean-field", "MeanField"),
               ("VQC (ring)", "VQC(ring)"), ("QAOA (fair)", "QAOA(fair)")]
    fams = [("mRMR (submodular)", "mRMR", "#4C72B0"),
            ("Frustrated (control)", "frustrated", "#C44E52")]

    def solved(fam, key):
        gv = [r[key] for r in recs if r["problem"] == fam]
        return 100.0 * float(np.mean([abs(v) <= tol for v in gv]))

    x = np.arange(len(methods)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    print("F12 hardness control (% solved to exact, pooled n=12,14):")
    for i, (flab, fkey, fcol) in enumerate(fams):
        ys = [solved(fkey, m[1]) for m in methods]
        ax.bar(x + (i - 0.5) * w, ys, w, color=fcol, label=flab,
               edgecolor="white", linewidth=0.6, zorder=3)
        for xi, yv in zip(x + (i - 0.5) * w, ys):
            ax.text(xi, yv + 1.6, f"{yv:.0f}", ha="center", va="bottom",
                    fontsize=8, color=fcol)
        print(f"  {flab:<22} " + " ".join(f"{v:5.1f}" for v in ys))

    ax.annotate("mean-field fails\noff submodular", xy=(1 + 0.5 * w, 75),
                xytext=(1.55, 44), fontsize=7.6, color="#7a1718", ha="center",
                arrowprops=dict(arrowstyle="->", color="#7a1718", lw=0.8))
    ax.set_xticks(x); ax.set_xticklabels([m[0] for m in methods])
    ax.set_ylabel("% of instances solved to the exact optimum")
    ax.set_title("Hardness control: SA and full-cost QAOA reach beyond mean-field")
    ax.set_ylim(0, 116)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    _save(fig, "hardness_control")


if __name__ == "__main__":
    print("== Figure 1: ablation ==")
    fig_ablation()
    print("== Figure 2: decoding ==")
    fig_decoding()
    print("== Figure 3 (F4): scaling ==")
    fig_scaling()
    print("== Figure 4 (F7): lambda sweep ==")
    fig_lambda_sweep()
    print("== Figure 5 (F8): real-data benchmark ==")
    fig_realdata()
    print("== Figure 6 (F9): channel isolation ==")
    fig_channels()
    print("== Figure 7 (F12): hardness control ==")
    fig_hardness()
    print("done.")
