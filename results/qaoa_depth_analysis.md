# Tier-1 (A): QAOA depth / optimiser / warm-start ablation — analysis

Source: `scripts/phase_a_qaoa_depth.py` → `results/qaoa_depth_ablation.json` (+ `.out`).
Config: ns=[14,16], seeds=[0,1,2], λ=1, steps=150, lr=0.04, μ=2.0, decode=4096 shots.
Anchor: SA reaches the exact optimum (dJ=0) on every instance.

## Variants (p, init, normalize, restarts)
- **A0** = (3, random, False, 1)  → reproduces the paper's QAOA setting
- **A1** = (3, random, True, 6)   → multistart + Hamiltonian normalization
- **A2** = (3, anneal, True, 1)   → annealing-schedule warm-start
- **A3** = (5, anneal, True, 1)   → warm-start + more depth

## Mean dJ (3 seeds, λ=1)
| setting | n=14 | n=16 |
|---|---|---|
| A0 single-start (paper)      | 0.024 | 0.112 |
| A1 6-restart + normalised    | 0.041 | 0.065 |
| A2 anneal warm-start         | 0.000 | 0.041 |
| A3 anneal warm-start p=5     | 0.063 | 0.081 |
| SA (classical)               | 0.000 | 0.000 |

Raw n=16 per seed: A0 [0.155,0.146,0.035]; A1 [0.037,-0.000,0.159]; A2 [0.089,-0.000,0.035]; A3 [0.072,0.093,0.077].

## Gradient-variance (barren-plateau probe, normalised H, p=3, 40 random params)
n=10: 2.00e2 | n=12: 2.25e2 | n=14: 4.88e2 | n=16: 8.12e2  → **INCREASES with n** (not a barren plateau).

## Conclusions (honest)
1. **The paper's single-start QAOA is under-optimised.** At n=16, multistart/normalisation/warm-start cut the gap from ~0.11 to ~0.04–0.07 (≈ ring-VQC level 0.071), NOT the no-ent level (0.149).
2. **Best-effort QAOA still ≠ 0** (SA=0) → **no-advantage conclusion holds**.
3. **The earlier-added claim "QAOA falls back to no-ent / densest encoding hardest to optimise" was an under-optimisation artefact → REMOVED from §5.** Consistent truth: **QAOA ≈ ring at all n**, both >0, both < classical.
4. Difficulty = local-minima ruggedness (gradient variance grows, doesn't vanish), mitigated by restarts/warm-start. NOT barren plateau, NOT representational limit.
5. High variance (3 seeds; seed-2 A1>A0 anomaly from energy↔mRMR-J decode mismatch under the μ penalty) → a fair main-table QAOA row needs a multi-seed (8×2λ) re-run with setting A1/A2.

## Paper edits made (this analysis)
- §5: removed the degradation sentence; replaced with single-start caveat + ref app:qaoa.
- §5 stats sentence: dropped QAOA-specific p-values; kept SA-exact + circuit>MF (p<1e-5).
- appendix app:stats: softened QAOA claim (optimisation-effort, not capacity).
- appendix: NEW \section{QAOA optimisation sensitivity} (app:qaoa) with tab:qaoa-opt (3-seed) + barren-plateau-negative note. CI-table row relabelled "QAOA-mRMR (1-start)".

## OPEN DECISION (for user)
Main-table QAOA-mRMR row (0.170/0.163/0.231/0.201) is single-start = inflated. Options:
- **(rec) Fair Colab re-run** scaling-QAOA (n=16–22, 8 seeds, 2λ) with A1 (6-restart+normalised) → replace row with fair numbers (~ring level), solidify app:qaoa.
- Keep single-start row + the 3-seed app:qaoa caveat (honest, less work).
- Drop QAOA from tab:scaling; keep only in tab:ablation (n=12,14) + app:qaoa.
