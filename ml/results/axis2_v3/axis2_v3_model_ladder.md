# Axis 2 V3 model ladder (validation)

Assembled by `ml/experiments/axis2_v3_shared/model_ladder.py` from stored results. Frozen V2 model definitions; fixed-hybrid weight and gate settings re-selected on V3 train OOF only. **test_evaluated: false.**

## Main table

| model | MAE | RMSE | median AE | mean signed | slope |
|---|---|---|---|---|---|
| Zero | 20.5121 | 25.5066 | 20.046 | -0.305 | n/a |
| Full median | 18.6045 | 25.2016 | 14.543 | -0.329 | n/a |
| Core median | 17.4255 | 21.9002 | 14.365 | -0.258 | n/a |
| Ridge | 16.1049 | 19.7494 | 14.062 | +0.104 | 0.420 |
| Natural offset | 14.0508 | 18.4247 | 11.013 | +0.121 | 0.723 |
| Fixed hybrid | 13.9629 | 18.0753 | 11.227 | +0.118 | 0.660 |
| Magnitude gate | 13.6797 | 17.7479 | 10.841 | +0.130 | 0.657 |
| Dynamic hybrid | **13.4557** | **17.3976** | 10.754 | +0.282 | 0.637 |

## MAE by variant

| model | clean | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| Zero | 0.00 | 10.01 | 20.02 | 30.02 | 42.50 |
| Full median | 18.60 | 18.60 | 18.60 | 18.60 | 18.60 |
| Core median | 17.43 | 17.43 | 17.43 | 17.43 | 17.43 |
| Ridge | 11.54 | 12.25 | 14.14 | 18.07 | 24.52 |
| Natural offset | 14.05 | 14.05 | 14.05 | 14.05 | 14.05 |
| Fixed hybrid | 13.29 | 13.38 | 13.60 | 14.18 | 15.35 |
| Magnitude gate | 13.61 | 13.58 | 13.17 | 12.95 | 15.08 |
| Dynamic hybrid | 12.31 | 12.74 | 13.00 | 13.42 | 15.82 |

## Prediction slope (corrupted validation)

| model | slope | intercept | R² |
|---|---|---|---|
| Ridge | 0.420 | +0.300 | 0.471 |
| Natural offset | 0.723 | +0.227 | 0.605 |
| Fixed hybrid | 0.660 | +0.242 | 0.596 |
| Magnitude gate | 0.657 | +0.253 | 0.610 |
| Dynamic hybrid | 0.637 | +0.428 | 0.613 |

## Sign (corrupted validation)

| model | + n | + MAE | + RMSE | + signed | − n | − MAE | − RMSE | − signed |
|---|---|---|---|---|---|---|---|---|
| Ridge | 7536 | 17.068 | 20.642 | -14.706 | 7424 | 17.430 | 21.047 | +15.087 |
| Natural offset | 7536 | 13.948 | 18.328 | -7.024 | 7424 | 14.155 | 18.522 | +7.374 |
| Fixed hybrid | 7536 | 14.014 | 18.147 | -8.637 | 7424 | 14.249 | 18.383 | +8.994 |
| Magnitude gate | 7536 | 13.583 | 17.787 | -8.530 | 7424 | 13.813 | 18.009 | +8.906 |
| Dynamic hybrid | 7536 | 13.619 | 17.620 | -8.923 | 7424 | 13.869 | 17.903 | +9.642 |

## Tail

| model | P90 | P95 | P99 | P99.5 | P99.9 | max | > 25 ¢ | > 50 ¢ |
|---|---|---|---|---|---|---|---|---|
| Zero | 42.46 | 46.27 | 49.24 | 49.61 | 49.89 | 49.99 | 7480 (40.00%) | 0 (0.00%) |
| Full median | 40.07 | 46.44 | 71.83 | 88.36 | 139.90 | 272.13 | 5365 (28.69%) | 515 (2.75%) |
| Core median | 38.26 | 43.35 | 48.81 | 49.32 | 49.91 | 49.98 | 5210 (27.86%) | 0 (0.00%) |
| Ridge | 32.81 | 37.04 | 44.15 | 46.63 | 51.39 | 58.92 | 4440 (23.74%) | 32 (0.17%) |
| Natural offset | 30.13 | 37.64 | 53.92 | 58.75 | 70.80 | 74.64 | 3015 (16.12%) | 305 (1.63%) |
| Fixed hybrid | 29.85 | 36.54 | 50.95 | 55.07 | 63.69 | 69.48 | 2971 (15.89%) | 209 (1.12%) |
| Magnitude gate | 29.19 | 35.91 | 50.08 | 55.59 | 64.01 | 74.38 | 2857 (15.28%) | 192 (1.03%) |
| Dynamic hybrid | 28.76 | 35.26 | 48.24 | 52.82 | 62.86 | 73.73 | 2764 (14.78%) | 145 (0.78%) |

## Dynamic gate Ridge weight

| mean | std | P10 | P25 | P50 | P75 | P90 | min | max |
|---|---|---|---|---|---|---|---|---|
| 0.324 | 0.351 | 0.000 | 0.010 | 0.166 | 0.619 | 0.927 | 0.000 | 1.000 |

| variant | clean | corruption_1 | corruption_2 | corruption_3 | corruption_4 |
|---|---|---|---|---|---|
| w̄ | 0.349 | 0.333 | 0.307 | 0.300 | 0.332 |

| \|core\| | 0-10 | 10-20 | 20-30 | 30-40 | 40-49.999 |
|---|---|---|---|---|---|
| w̄ | 0.312 | 0.245 | 0.271 | 0.381 | 0.523 |

Max |observed core| on V3 train+validation: 49.9988 ¢ (examples ≥ 50 ¢: 0); no 50+ bucket exists.

## Expert disagreement |ridge − natural|

| bucket | n | Ridge | Natural | Fixed | Dynamic | w̄ dynamic |
|---|---|---|---|---|---|---|
| 0-5 | 5931 | 13.833 | 13.708 | 13.697 | 13.515 | 0.480 |
| 5-10 | 5125 | 15.069 | 13.930 | 13.938 | 13.633 | 0.302 |
| 10-20 | 5956 | 17.533 | 13.933 | 13.921 | 13.342 | 0.201 |
| 20+ | 1688 | 22.192 | 16.040 | 15.122 | 13.111 | 0.278 |

## Note groups (3740 validation notes, tie within 0.5 ¢)

- Dynamic vs fixed: dynamic better 48.1%, tie 24.1%, fixed better 27.8%; median gain 1.736 ¢, median loss 1.442 ¢.
- Dynamic vs Ridge: dynamic better 65.1%, tie 7.5%, Ridge better 27.4%.

## V2 → V3 (validation MAE; the task changed, so this is not a ranking of datasets)

| model | V2 | V3 |
|---|---|---|
| Zero | 20.5121 | 20.5121 |
| Full median | 18.6045 | 18.6045 |
| Core median | 17.4255 | 17.4255 |
| Ridge | 13.3153 | 16.1049 |
| Natural offset | 14.0508 | 14.0508 |
| Fixed hybrid | 12.5828 | 13.9629 |
| Magnitude gate | 11.8036 | 13.6797 |
| Dynamic hybrid | 11.5335 | 13.4557 |

Fixed-hybrid weight: V2 0.56 Ridge / 0.44 natural; V3 0.21 / 0.79.
Slopes V2 → V3: Ridge 0.602 → 0.420, Natural offset 1.002 → 0.723, Fixed hybrid 0.778 → 0.660, Magnitude gate 0.789 → 0.657, Dynamic hybrid 0.768 → 0.637.

V2 dynamic-vs-fixed gain: 1.0494 ¢ overall, of which 49.3% came from |core| > 50; on the |core| ≤ 50 subset it was 0.6173 ¢. V3 dynamic-vs-fixed gain: 0.5072 ¢ (3.63%).
