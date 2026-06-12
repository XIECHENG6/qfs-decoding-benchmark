# qfs-decoding-benchmark

Code, data, and results for the paper:

> **Decoding strategy governs noise-resilient quantum feature selection: closed-form criteria and a redundancy-aware benchmark**
> Cheng Xie, Yi Zheng, Chenlei Fang, Ying Zhou, Wei Pan, Haobin Shi
> School of Computer Science, Northwestern Polytechnical University
> (submitted to *Knowledge-Based Systems*, 2026)

The paper makes two contributions, both reproducible from this repository:

1. **An *a priori* decoder-selection rule.** Closed-form, circuit-specific criteria
   (computable from gate counts alone, no noisy simulation) for when
   expectation-value vs. sampling decoding of a noisy quantum state remains reliable.
2. **A rigorous quantum-advantage benchmark.** A redundancy-aware (mRMR) QUBO
   benchmark against exact optima and strong classical baselines (greedy, simulated
   annealing, mean-field), with an entanglement ablation, a hardness control, and a
   structural (submodularity) explanation of why no quantum variant wins at
   NISQ-accessible scales.

## Environment

- Python 3.11
- [PennyLane](https://pennylane.ai) 0.45 with the `lightning.qubit` backend
  (`default.mixed` for noisy-decoding experiments)
- NumPy, SciPy, scikit-learn, matplotlib

```bash
pip install pennylane==0.45 pennylane-lightning scikit-learn scipy matplotlib
```

> **Windows note:** `scripts/qfs_common.py` pins `OMP_NUM_THREADS=1`; without it,
> repeated scikit-learn SVC fits can segfault on Windows.

All experiment drivers are run from the repository root, e.g.
`python scripts/phase1_derisk.py`. Outputs are written to `results/`
(the frozen outputs used in the paper are committed there).

## Reproduction map (paper table/figure → script → output)

| Paper item | Script | Frozen output |
|---|---|---|
| Table 1 (per-method benchmark), Fig. lambda-sweep | `scripts/phase1_derisk.py`, `scripts/phase1_tune.py` | `results/phase1_summary.json` |
| Real-data benchmark (WDBC / Ionosphere / Sonar) | `scripts/phase1_realdata.py` (+ `scripts/uci_loader.py`) | `results/phase1_summary.json` |
| Entanglement ablation (n = 12, 14) | `scripts/phase2_ablation_strong.py` | `results/phase2_ablation_strong.json` |
| Scaling table/figure (n = 16–22, classical + VQC) | `notebooks/colab_scaling.ipynb` (GPU statevector) | `results/scaling_ablation.json` |
| QAOA scaling row, fair multi-restart setting | `notebooks/scaling_qaoa_fair.ipynb`, `scripts/qaoa_fair_rerun.py`, analysis: `scripts/analyze_qaoa_fair.py` | `results/scaling_qaoa_fair.json` (single-start: `results/scaling_qaoa.json`) |
| Decoding under composite realistic noise | `scripts/phase3_realistic_decoding.py` | `results/phase3_realistic.json` |
| Decoding generality (16 seeds, architectures, channel isolation) | `scripts/phase_c_decoding_generality.py` | `results/decoding_generality.json` |
| Decoder grid (pure depolarizing cross-check) | `notebooks/colab_scaling.ipynb` (decoder cell) | `results/decoder_grid.json` |
| QAOA depth/restart/warm-start ablation (appendix) | `scripts/phase_a_qaoa_depth.py` | `results/qaoa_depth_ablation.json` |
| Hardness control (mRMR vs. frustrated, appendix) | `scripts/phase_d_hardness_control.py` | `results/hardness_control.json` |
| Robustness across classifiers (SVM / kNN / LogReg / RF) | `scripts/phase4_multiclf.py` | `results/phase4_multiclf.json` |
| Statistical analysis (bootstrap CIs, paired tests, appendix) | `scripts/stats_analysis.py` | `results/stats_summary.{json,md}` |
| All data figures | `scripts/make_figures.py` (reads `results/*.json`) | `figures/*.{png,pdf}` |
| Concept-figure data (redundancy matrix, ⟨Z⟩-damping demo) | `scripts/make_fig_data.py` | `figures/redundancy_matrix.csv`, `figures/zdamp_demo.json` |

`results/*_analysis.md` are the accompanying analysis notes for the corresponding runs.

## Data

- `data/instance_seed0.npz` — the synthetic "relevance–redundancy trap" instance
  (seed 0) used in the per-method benchmark; the generator is
  `qfs_common.make_hard_instance`.
- `data/uci/` — cached copies of the Ionosphere and Sonar datasets from the
  [UCI Machine Learning Repository](https://archive.ics.uci.edu) (CC BY 4.0);
  WDBC is loaded at runtime from scikit-learn.

## License

MIT — see [LICENSE](LICENSE).
