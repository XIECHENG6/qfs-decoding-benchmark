# Tier-1 (E): why classical wins — structural explanation (submodularity)

Verified numerically (scripts: qfs_common + phase_d generators), light analysis, n=12/14, 8 seeds.

## Core result: mRMR-QUBO is SUBMODULAR; frustration breaks it
J(S) = Σ_{i∈S} r_i − λ Σ_{i<j∈S} R_ij.  With R_ij ≥ 0 the pairwise sum is supermodular,
so −λΣR is submodular and J (modular + submodular) is submodular.

| instance | frac(coupling<0) | min coupling | submodularity-violation rate |
|---|---|---|---|
| mRMR (R=MI redundancy) | 0.000 | +0.034 (n=14), +0.037 (n=12) | **0.000 (perfectly submodular)** |
| frustrated (W random-sign) | ~0.53 | negative | **0.500 (broken)** |

(submodularity-violation = fraction of (S⊂T, e∉T) triples with INCREASING marginal gain
J(S+e)−J(S) < J(T+e)−J(T); submodular ⟺ 0. 2000 random triples per instance.)

## Why this explains the no-advantage result
- R_ij ≥ 0 because mutual-information redundancies are non-negative → J submodular.
- Submodular cardinality-constrained max is the classic regime where greedy ((1−1/e)
  guarantee) and mean-field/continuous relaxations are near-optimal.
- No sign-frustration ⇒ a product-state (mean-field) assignment can simultaneously honour
  all pairwise penalties ⇒ the mean-field relaxation is TIGHT (empirically gap≈0) ⇒ a
  beyond-mean-field (entangled) optimiser has nothing to exploit. This is *why* VQC-QFS
  (a mean-field ansatz) cannot beat free classical mean-field on mRMR.
- Frustration (mixed-sign couplings) breaks submodularity (50% violation) and opens a
  mean-field gap (Tier-1 D: MF gap 0.04–0.07 on frustrated vs ≈0 on mRMR). But the
  redundancy-aware QUBO does not produce such objectives.

## Caveats (honest)
- Submodular max is still NP-hard in general; greedy is only (1−1/e)-approx in the worst
  case (greedy gap 0.08 on one bench instance). The claim is "near-optimal / mean-field-
  tight on these instances", not "exactly poly-time solvable". Phrase carefully.
- The empirical mean-field-tightness (gap≈0) is the operative fact; submodularity +
  no-frustration is the structural reason.

## Where to write (after D completes, with D's MF-gap numbers)
- §6 "why no advantage" subsection: upgrade "pairwise QUBO is mean-field-solvable" →
  "submodular (R≥0) ⇒ mean-field-tight; frustration would break it but mRMR has none".
- Cite D's hardness-control table for the empirical frustrated-vs-mRMR MF-gap contrast.
- Optionally a one-line submodularity-violation metric (0% vs 50%) in §6 or appendix.
