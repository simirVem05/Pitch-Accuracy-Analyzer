# Axis 2 V3 final evaluation — test

Frozen model: 50-tree Random Forest + natural-offset HGB, combined per example by the dynamic gate (w on RF). Artifacts loaded, not refit; RF inference n_jobs=1. 8 songs, 4070 clean notes, 20350 examples. test_evaluated: **true**.

**FINAL TEST MAE 10.7760 c, RMSE 14.5513 c** (validation 12.9703 / 16.7909; Δ MAE -2.1943 c — excellent (within ~+0.5 c of validation)).

## Headline

| model | validation MAE | test MAE | Δ | test/val | validation RMSE | test RMSE | test median AE |
|---|---|---|---|---|---|---|---|
| Zero | 20.5121 | 20.4766 | -0.035 | 0.998 | 25.5066 | 25.4696 | 19.874 |
| Full median | 18.6045 | 14.4636 | -4.141 | 0.777 | 25.2016 | 20.7153 | 9.675 |
| Core median | 17.4255 | 13.9055 | -3.520 | 0.798 | 21.9002 | 18.7689 | 9.611 |
| Random Forest (50) | 13.8488 | 11.5511 | -2.298 | 0.834 | 17.6251 | 15.3714 | 8.637 |
| Natural offset | 14.0508 | 11.6664 | -2.384 | 0.830 | 18.4247 | 15.9414 | 8.231 |
| Dynamic hybrid | 12.9703 | 10.7760 | -2.194 | 0.831 | 16.7909 | 14.5513 | 7.794 |

Dynamic: mean signed -1.874; P75 15.37, P90 24.65, P95 30.92, P99 42.91, P99.5 48.26, P99.9 57.29, max 71.23; > 25 c 1957 (9.62%), > 50 c 81 (0.40%).

## By variant (MAE; validation in parentheses)

| model | clean | c1 | c2 | c3 | c4 |
|---|---|---|---|---|---|
| Random Forest (50) | 8.18 (9.88) | 9.01 (10.73) | 10.29 (12.28) | 12.59 (15.23) | 17.69 (21.12) |
| Natural offset | 11.67 (14.05) | 11.67 (14.05) | 11.67 (14.05) | 11.67 (14.05) | 11.67 (14.05) |
| Dynamic hybrid | 8.62 (10.44) | 9.57 (11.35) | 10.39 (12.33) | 11.35 (13.81) | 13.96 (16.92) |

| dynamic | n | MAE | RMSE | median AE | mean signed |
|---|---|---|---|---|---|
| clean | 4070 | 8.616 | 11.623 | 6.289 | -1.749 |
| corruption_1 | 4070 | 9.572 | 12.489 | 7.478 | -1.878 |
| corruption_2 | 4070 | 10.389 | 13.851 | 7.769 | -1.775 |
| corruption_3 | 4070 | 11.346 | 15.578 | 7.619 | -1.719 |
| corruption_4 | 4070 | 13.958 | 18.251 | 10.716 | -2.247 |

## Sign (corrupted)

| sign | model | n | MAE | RMSE | mean signed |
|---|---|---|---|---|---|
| positive | Random Forest (50) | 8515 | 12.834 | 16.441 | -11.293 |
| positive | Natural offset | 8515 | 11.561 | 15.622 | -6.979 |
| positive | Dynamic hybrid | 8515 | 11.600 | 15.225 | -9.263 |
| negative | Random Forest (50) | 7765 | 11.910 | 16.190 | +8.536 |
| negative | Natural offset | 7765 | 11.782 | 16.284 | +3.333 |
| negative | Dynamic hybrid | 7765 | 11.004 | 15.163 | +6.164 |

## Tails (test; validation in parentheses)

| model | P90 | P95 | P99 | P99.5 | P99.9 | max | > 25 c | > 50 c |
|---|---|---|---|---|---|---|---|---|
| Random Forest (50) | 26.1 (29.2) | 32.2 (35.0) | 43.5 (46.5) | 48.2 (50.4) | 56.6 (59.4) | 67.9 (71.9) | 2281 (2934) | 81 (104) |
| Natural offset | 27.2 (30.1) | 34.1 (37.6) | 48.9 (53.9) | 56.4 (58.7) | 63.0 (70.8) | 71.5 (74.6) | 2380 (3015) | 180 (305) |
| Dynamic hybrid | 24.6 (27.7) | 30.9 (33.8) | 42.9 (46.4) | 48.3 (51.8) | 57.3 (62.6) | 71.2 (73.2) | 1957 (2487) | 81 (111) |

## Per song (dynamic; model-error diagnostic, not singer quality)

| song | clean notes | examples | MAE | RMSE | median AE | signed | P95 | max | RF MAE | natural MAE |
|---|---|---|---|---|---|---|---|---|---|---|
| giveon_heartbreak_anniversary_audio | 476 | 2380 | 7.872 | 10.889 | 5.981 | -2.80 | 21.3 | 60.9 | 8.298 | 8.415 |
| tum_hi_ho_lyrics_arijit_singhaashiqui_2_tseries | 489 | 2445 | 8.177 | 12.140 | 4.950 | -0.30 | 27.7 | 51.7 | 8.705 | 9.129 |
| ariana_grande_into_you_official_video | 550 | 2750 | 8.790 | 12.059 | 6.327 | -1.11 | 25.4 | 62.5 | 9.477 | 9.364 |
| kacey_musgraves_rainbow_official_audio | 393 | 1965 | 9.240 | 12.814 | 6.284 | +0.25 | 27.6 | 56.6 | 10.725 | 9.794 |
| adele_hello | 523 | 2615 | 10.292 | 14.146 | 7.116 | -0.97 | 31.1 | 57.6 | 11.234 | 10.954 |
| prince_royce_corazon_sin_cara_audio | 440 | 2200 | 10.400 | 13.550 | 8.120 | -4.83 | 29.6 | 71.2 | 10.882 | 11.285 |
| mariah_carey_all_i_want_for_christmas_is_you | 647 | 3235 | 14.661 | 18.116 | 12.752 | -4.71 | 33.9 | 60.4 | 15.533 | 16.001 |
| chris_stapleton_tennessee_whiskey_official_audio | 552 | 2760 | 14.860 | 18.601 | 12.431 | +0.08 | 37.0 | 66.8 | 15.698 | 16.243 |

## Notes (4070; per note, mean |error| over its five variants)

Dynamic: mean 10.776, median 8.132, P75 15.003, P90 23.616, P95 28.312, P99 38.189, max 53.139.

| comparison | dynamic better | tie (0.5 c) | expert better | median gain | median loss | mean gain |
|---|---|---|---|---|---|---|
| Dynamic vs RF | 50.9% | 21.2% | 27.9% | 2.109 | 1.838 | +0.775 |
| Dynamic vs Natural | 47.3% | 20.6% | 32.0% | 2.096 | 1.568 | +0.890 |

## Gate (test; validation in parentheses)

| mean | std | P10 | P25 | P50 | P75 | P90 | min | max |
|---|---|---|---|---|---|---|---|---|
| 0.509 (0.501) | 0.346 (0.357) | 0.023 (0.014) | 0.180 (0.145) | 0.522 (0.504) | 0.844 (0.862) | 0.983 (0.986) | 0.000 (0.000) | 1.000 (1.000) |

| variant | clean | corruption_1 | corruption_2 | corruption_3 | corruption_4 |
|---|---|---|---|---|---|
| w̄_RF | 0.590 (0.573) | 0.531 (0.532) | 0.467 (0.490) | 0.456 (0.443) | 0.502 (0.467) |

| \|RF − natural\| | n | RF | Natural | Dynamic | w̄_RF | validation w̄_RF |
|---|---|---|---|---|---|---|
| 0-5 | 10146 | 9.572 | 9.600 | 9.453 | 0.521 | 0.507 |
| 5-10 | 5314 | 11.485 | 11.281 | 10.833 | 0.461 | 0.455 |
| 10-20 | 3662 | 14.786 | 14.472 | 12.914 | 0.508 | 0.489 |
| 20+ | 1228 | 18.542 | 22.039 | 15.087 | 0.628 | 0.652 |

Routing (post hoc): RF lower error on 49.9%, natural on 50.1%; w̄_RF 0.554 where RF was better vs 0.465 where natural was; higher-weighted expert was right 55.3%.
Oracle (not deployable): MAE 8.6233, RMSE 12.2996; dynamic − oracle +2.1527 c (validation oracle 10.5134).

## Slope (corrupted; secondary diagnostic)

| model | slope | intercept | R² |
|---|---|---|---|
| Random Forest (50) | 0.604 | -1.091 | 0.683 |
| Natural offset | 0.799 | -1.682 | 0.703 |
| Dynamic hybrid | 0.694 | -1.330 | 0.719 |

## Bootstrap 95% CIs (10,000 resamples, seed 20260824)

| statistic | point | note-group CI | song-cluster CI (unstable, 8 songs) |
|---|---|---|---|
| dynamic_mae | 10.7760 | [10.5134, 11.0415] | [8.8893, 12.7665] |
| dynamic_rmse | 14.5513 | [14.2139, 14.9000] | [12.3762, 16.5397] |
| rf_mae | 11.5511 | [11.3005, 11.8029] | [9.5845, 13.6087] |
| natural_mae | 11.6664 | [11.3412, 12.0013] | [9.5763, 13.8784] |
| dynamic_minus_rf_mae | -0.7751 | [-0.8831, -0.6643] | [-0.9807, -0.5968] |
| dynamic_minus_natural_mae | -0.8904 | [-1.0060, -0.7753] | [-1.1289, -0.6540] |

## Integrity

- Frozen artifacts/configs/dataset/code changed after evaluation: 0.
- Unlabeled predictions hashed before labels were loaded: `fe6e74fe51bc336cee6ea81ea4c1d3153de381e51adabfc5f9921e07b197840f`.
- Git commit 4af7ec61; uncommitted changes in working tree: True.
- Python 3.11.15, numpy 2.2.6, scikit-learn 1.7.1, torch 2.13.0.
