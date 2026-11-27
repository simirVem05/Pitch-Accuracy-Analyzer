# Confidence analysis

Generated from `results.json`.

Evidence quantities per beat segment:
- `r`: Pearson fit (absolute)
- `G`: gap to the best template (relative)
- `E`: permutation-null significance
- `c = r − m`: **excess fit** over the chance-level best fit of the same values scrambled
- `I_c`: chroma informativeness

Posterior numbers are **uncalibrated weights**, not probabilities of correctness.

κ (chroma informativeness floor) = 99th percentile of I_c on synthetic noise = **0.0139**. Selected for D: **EXCESS γ=0.15 δ=10.0** (max degraded recall s.t. noise FA <= 0.05 and clean mean |S| <= 3).

## Component medians by synthetic condition

| condition | frames | r_best | excess c_best | E_best | I_c | posterior top-1 |
|---|---|---|---|---|---|---|
| clean | 8064 | 0.971 | 0.190 | 0.825 | 0.448 | 0.998 |
| degraded | 8064 | 0.917 | 0.142 | 0.830 | 0.373 | 0.995 |
| white | 480 | 0.723 | -0.010 | 0.410 | 0.001 | 0.919 |
| pink | 480 | 0.670 | -0.077 | 0.205 | 0.007 | 0.955 |
| cluster | 480 | 0.707 | 0.000 | 0.380 | 0.280 | 0.494 |
| single | 432 | 0.910 | 0.212 | 0.823 | 0.744 | 1.000 |
| mixture | 480 | 0.943 | 0.160 | 0.890 | 0.379 | 0.973 |

## Strategy sweep (synthetic)

*accept* is the share of frames with at least one eligible chord, and *n* is the mean eligible count. Noise FA is the mean of white and pink acceptance. Real acceptance (last columns) is descriptive only: no real labels exist.

| strategy | params | clean recall | clean n | degraded recall | degraded n | noise FA | cluster accept | single-pitch accept (n) | mixture any-acceptable | mixture both | **real accept** | real n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ABS | theta=0.7 | 0.988 | 1.04 | 0.917 | 1.16 | 0.485 | 0.681 | 1.000 (1.0) | 0.850 | 0.000 | 0.874 | 1.27 |
| ABS | theta=0.8 | 0.988 | 1.04 | 0.911 | 1.13 | 0.183 | 0.056 | 1.000 (1.0) | 0.850 | 0.000 | 0.687 | 0.92 |
| ABS | theta=0.9 | 0.976 | 1.00 | 0.601 | 0.72 | 0.034 | 0.000 | 0.667 (0.7) | 0.777 | 0.000 | 0.318 | 0.35 |
| REL | delta=0.03 | 0.987 | 1.01 | 0.903 | 1.09 | 1.000 | 1.000 | 1.000 (1.0) | 0.850 | 0.000 | 1.000 | 1.31 |
| REL | delta=0.05 | 0.988 | 1.04 | 0.917 | 1.16 | 1.000 | 1.000 | 1.000 (1.0) | 0.850 | 0.000 | 1.000 | 1.56 |
| REL | delta=0.1 | 1.000 | 1.41 | 0.979 | 1.48 | 1.000 | 1.000 | 1.000 (1.3) | 0.950 | 0.404 | 1.000 | 2.31 |
| NULL | eps=0.9 | 0.293 | 0.30 | 0.284 | 0.33 | 0.014 | 0.006 | 0.000 (0.0) | 0.454 | 0.000 | 0.114 | 0.11 |
| NULL | eps=0.95 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| NULL | eps=0.99 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| NULL+REL | eps=0.9,delta=0.03 | 0.293 | 0.30 | 0.284 | 0.33 | 0.014 | 0.006 | 0.000 (0.0) | 0.454 | 0.000 | 0.114 | 0.11 |
| NULL+REL | eps=0.9,delta=0.05 | 0.293 | 0.30 | 0.284 | 0.33 | 0.014 | 0.006 | 0.000 (0.0) | 0.454 | 0.000 | 0.114 | 0.11 |
| NULL+REL | eps=0.9,delta=0.1 | 0.293 | 0.30 | 0.284 | 0.33 | 0.014 | 0.006 | 0.000 (0.0) | 0.454 | 0.000 | 0.114 | 0.11 |
| NULL+REL | eps=0.95,delta=0.03 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| NULL+REL | eps=0.95,delta=0.05 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| NULL+REL | eps=0.95,delta=0.1 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| NULL+REL | eps=0.99,delta=0.03 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| NULL+REL | eps=0.99,delta=0.05 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| NULL+REL | eps=0.99,delta=0.1 | 0.007 | 0.01 | 0.002 | 0.00 | 0.000 | 0.000 | 0.000 (0.0) | 0.000 | 0.000 | 0.003 | 0.00 |
| EXCESS | gamma=0.05 | 0.998 | 1.57 | 0.920 | 1.36 | 0.185 | 0.102 | 1.000 (1.3) | 0.850 | 0.412 | 0.684 | 1.13 |
| EXCESS | gamma=0.1 | 0.993 | 1.36 | 0.799 | 1.01 | 0.089 | 0.031 | 1.000 (1.3) | 0.825 | 0.081 | 0.498 | 0.66 |
| EXCESS **(D)** | gamma=0.15 | 0.847 | 0.91 | 0.371 | 0.47 | 0.025 | 0.000 | 1.000 (1.2) | 0.669 | 0.050 | 0.268 | 0.30 |
| EXCESS+REL | gamma=0.05,delta=0.03 | 0.985 | 1.01 | 0.843 | 0.99 | 0.185 | 0.102 | 1.000 (1.0) | 0.850 | 0.000 | 0.684 | 0.81 |
| EXCESS+REL | gamma=0.05,delta=0.05 | 0.986 | 1.03 | 0.856 | 1.04 | 0.185 | 0.102 | 1.000 (1.0) | 0.850 | 0.000 | 0.684 | 0.89 |
| EXCESS+REL | gamma=0.05,delta=0.1 | 0.998 | 1.38 | 0.912 | 1.23 | 0.185 | 0.102 | 1.000 (1.3) | 0.850 | 0.354 | 0.684 | 1.05 |
| EXCESS+REL | gamma=0.1,delta=0.03 | 0.980 | 1.00 | 0.751 | 0.88 | 0.089 | 0.031 | 1.000 (1.0) | 0.825 | 0.000 | 0.498 | 0.56 |
| EXCESS+REL | gamma=0.1,delta=0.05 | 0.981 | 1.02 | 0.762 | 0.91 | 0.089 | 0.031 | 1.000 (1.0) | 0.825 | 0.000 | 0.498 | 0.60 |
| EXCESS+REL | gamma=0.1,delta=0.1 | 0.993 | 1.31 | 0.795 | 1.00 | 0.089 | 0.031 | 1.000 (1.3) | 0.825 | 0.060 | 0.498 | 0.66 |
| EXCESS+REL | gamma=0.15,delta=0.03 | 0.847 | 0.86 | 0.358 | 0.45 | 0.025 | 0.000 | 1.000 (1.0) | 0.669 | 0.000 | 0.268 | 0.29 |
| EXCESS+REL | gamma=0.15,delta=0.05 | 0.847 | 0.87 | 0.363 | 0.46 | 0.025 | 0.000 | 1.000 (1.0) | 0.669 | 0.000 | 0.268 | 0.29 |
| EXCESS+REL | gamma=0.15,delta=0.1 | 0.847 | 0.91 | 0.371 | 0.47 | 0.025 | 0.000 | 1.000 (1.2) | 0.669 | 0.050 | 0.268 | 0.30 |

## Real songs: D's chord selection

Beat segments: 29199. Distribution of eligible-chord count per beat: 0: 73.2%, 1: 24.0%, 2: 2.7%, 3: 0.1%, 4: 0.0%, 5: 0.0%, 6+: 0.0%. Mean count when any qualify: 1.11.

Segment status under D: CHORD 26.8%, CHROMA_ONLY 70.5%, INSUFFICIENT 2.6%. Note status: CHORD 26.3%, CHROMA_ONLY 70.7%, INSUFFICIENT 3.0%.

| real beat segments | accepted | rejected |
|---|---|---|
| Ic_median | 0.248 | 0.151 |
| r_best_median | 0.938 | 0.818 |
| diatonic_share_of_top_chord | 0.891 | 0.852 |
| posterior_top1_median | 0.991 | 0.912 |
| posterior_top1_ge_0.9_share | 0.807 | 0.521 |

Eligible-set stability under CENS noise σ = 0.1: mean Jaccard 0.971, identical sets 96.8%, status agreement 97.3%.

