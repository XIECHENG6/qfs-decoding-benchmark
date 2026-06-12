# Fair QAOA (6-restart, A1 settings) vs baselines

## Per-n mean dJ (8 seeds x 2 lambda = 16 runs)

| n | QAOA-fair | QAOA-fair med | QAOA single-start | Greedy | MeanField | Random | SA | VQC(all) | VQC(no-ent) | VQC(ring) |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.069 ± 0.067 | 0.048 | 0.170 | 0.005 | 0.002 | 0.011 | -0.000 | 0.019 | 0.149 | 0.071 |
| 18 | 0.145 ± 0.087 | 0.136 | 0.163 | 0.017 | 0.000 | 0.110 | -0.000 | 0.028 | 0.144 | 0.091 |
| 20 | 0.136 ± 0.066 | 0.128 | 0.231 | 0.007 | 0.008 | 0.137 | 0.000 | 0.042 | 0.219 | 0.120 |
| 22 | 0.236 ± 0.110 | 0.227 | 0.201 | 0.012 | 0.008 | 0.143 | 0.000 | 0.040 | 0.204 | 0.141 |

## Per-lambda fair means

| n | lam=1 | lam=2 |
|---|---|---|
| 16 | 0.033 | 0.105 |
| 18 | 0.139 | 0.151 |
| 20 | 0.142 | 0.129 |
| 22 | 0.222 | 0.250 |

## Paired tests (same instance = n,lam,seed)

- fair vs single-start (all n, N=64): mean diff -0.045, fair-better frac 0.59, Wilcoxon p=0.031
  - n=16: mean diff -0.101, fair-better 0.69, p=0.029
  - n=18: mean diff -0.018, fair-better 0.50, p=0.78
  - n=20: mean diff -0.095, fair-better 0.69, p=0.0092
  - n=22: mean diff +0.035, fair-better 0.50, p=0.53
- fair vs VQC(ring) (all n, N=64): mean diff +0.040, fair-better frac 0.38, Wilcoxon p=0.013
  - n=16: mean diff -0.002, fair-better 0.50, p=0.9
  - n=18: mean diff +0.054, fair-better 0.31, p=0.061
  - n=20: mean diff +0.015, fair-better 0.44, p=0.43
  - n=22: mean diff +0.095, fair-better 0.25, p=0.036
- fair vs VQC(all) (all n, N=64): mean diff +0.114, fair-better frac 0.06, Wilcoxon p=8e-10
  - n=16: mean diff +0.049, fair-better 0.12, p=0.0076
  - n=18: mean diff +0.118, fair-better 0.00, p=0.00098
  - n=20: mean diff +0.094, fair-better 0.12, p=0.0092
  - n=22: mean diff +0.196, fair-better 0.00, p=3.1e-05
- fair vs VQC(no-ent) (all n, N=64): mean diff -0.033, fair-better frac 0.61, Wilcoxon p=0.093
  - n=16: mean diff -0.080, fair-better 0.81, p=0.0092
  - n=18: mean diff +0.002, fair-better 0.44, p=0.91
  - n=20: mean diff -0.084, fair-better 0.69, p=0.065
  - n=22: mean diff +0.031, fair-better 0.50, p=0.5
- fair vs MeanField (all n, N=64): mean diff +0.142, fair-better frac 0.02, Wilcoxon p=6.3e-12
  - n=16: mean diff +0.067, fair-better 0.06, p=0.0015
  - n=18: mean diff +0.145, fair-better 0.00, p=3.1e-05
  - n=20: mean diff +0.127, fair-better 0.00, p=3.1e-05
  - n=22: mean diff +0.227, fair-better 0.00, p=3.1e-05

## Exact-hit fraction (dJ <= 1e-3)

- n=16: 0.25
- n=18: 0.00
- n=20: 0.00
- n=22: 0.00
