# Tier-1 (C): decoding generality + power — analysis

Source: `scripts/phase_c_decoding_generality.py` → `results/decoding_generality.{json,out}`.
n=8, k=3, L=4, **16 seeds**, composite + per-channel noise, circuits {ring, all, none}.

## 1. Statistical power (fixes B's finding: 5 seeds → Wilcoxon floor p=0.031)
ring/composite at s=0.05:
- expval collapse 0.769 → 0.524, one-sided Wilcoxon **p=1.53e-5** (was n.s. at 5 seeds)
- sampling 0.777 → 0.781, two-sided Wilcoxon **p=0.24 (flat / no significant change)**

## 2. ring/composite accuracy curve (matches old 5-seed tab:decoding, now 16-seed)
| s | expval | topk | sampling |
|---|---|---|---|
| 0     | 0.769 | 0.767 | 0.777 |
| 0.001 | 0.769 | 0.767 | 0.781 |
| 0.005 | 0.767 | 0.767 | 0.781 |
| 0.01  | 0.769 | 0.770 | 0.781 |
| 0.02  | 0.757 | 0.749 | 0.781 |
| 0.05  | 0.524 | 0.711 | 0.781 |

## 3. Circuit generality (composite, expval@0→0.05; sampling@0.05; one-sided Wilcoxon samp>expval @0.05)
| circuit | expval 0→0.05 | sampling@0.05 | p |
|---|---|---|---|
| ring | 0.769→0.524 | 0.781 | 1.5e-5 |
| all  | 0.766→0.500 | 0.781 | 2.2e-4 |
| none | 0.762→0.762 (no collapse) | 0.781 | 3.5e-4 |
→ expval collapse scales with gate count/entanglement (none doesn't collapse: few gates → little ⟨Z⟩ damping), matching p1*∝1/(G1+G2). Sampling robust on all.

## 4. Channel isolation (ring, expval drop@0.05) — THE MECHANISM
| channel alone | expval@0.05 | drop |
|---|---|---|
| composite | 0.524 | **0.246** |
| depolarizing | 0.723 | 0.046 |
| amp/phase damping | 0.767 | 0.003 |
| **coherent (unitary)** | 0.769 | **0.000** |
| readout | 0.769 | 0.000 |
→ Collapse is a **damping** phenomenon (depol-driven, compounded). Coherent over-rotation (doesn't shrink |⟨Z⟩|) leaves expval intact. Sampling ~0.78 under every channel. **Confirms the ⟨Z⟩-damping mechanism behind p1* (Prop.\ref{prop:damp}), not just its existence.**

## Paper edits made
- tab:decoding: 5→16 seeds (numbers ~unchanged: expval→0.524, topk→0.711, sampling 0.777→0.781); caption adds Wilcoxon p=1.5e-5 collapse / p=0.24 flat.
- §5.4 prose: added significance (16 seeds), circuit-generality, channel-isolation mechanism, ref app:decoding-gen.
- appendix: NEW \section{Decoding crossover: generality and mechanism} (app:decoding-gen) + tab:decoding-chan (channel-isolation).
