# Axis 2 V3 — final model on the sealed test split

Hand-written summary of the one sealed-test evaluation of the frozen Axis 2 V3 model. Every number here is copied from the generated record in `ml/experiments/axis2_v3_final_test/`: `final_test_results.json`, `final_test_summary.md`, `bootstrap_results.json` and `per_song_results.csv`. Those files are authoritative, and this document must not be read as a re-evaluation. The validation-only ladder that chose the model is `axis2_v3_model_ladder.md` in this directory.

> **The test split is consumed.** It was opened once, on 2026-11-25 (`TEST_OPENED`, 00:04:56 UTC), with `evaluations_of_this_split_with_frozen_model: 1`. Do not re-run it, and do not use these 8 songs for model selection, tuning, or comparison-driven iteration. Any further model development needs a new held-out set.

## What was evaluated

**Model.** A 50-tree Random Forest on Ridge's 24 features, plus the natural-offset HGB on 71 features, combined per example by a 481-parameter dynamic gate that outputs the RF weight *w*. The artifacts were frozen and loaded, not refit. RF inference ran with `n_jobs=1`.

| Role | Current location | Path recorded at evaluation time |
|---|---|---|
| RF | `data/models/axis2_v3/selected_rf_model.joblib` | `ml/experiments/axis2_v3_rf_compression/selected_rf_model.joblib` |
| natural | `data/models/axis2_v3/natural_offset_model.joblib` | `ml/experiments/axis2_v3_rf_natural_hybrid/natural_offset_model.joblib` |
| gate | `data/models/axis2_v3/selected_dynamic_gate.pt` | `ml/experiments/axis2_v3_rf_compression/selected_dynamic_gate.pt` |

The files were moved byte-for-byte on 2026-11-26. `ml/artifact_relocation_manifest.json` maps old paths to new ones with SHA-256 hashes, and `python ml/artifact_paths.py verify` re-checks them.

**Data.** The test split of `data/ml_dataset/axis2_v3` covers 8 songs, 4,070 clean notes and 20,350 examples. Each note yields one clean copy (label 0) plus four corrupted copies, one per shift band: c1 5–15 ¢, c2 15–25 ¢, c3 25–35 ¢ and c4 35–50 ¢. Each corrupted copy has a constant shift added to every cents frame, and the label is that injected signed shift. Signs are target-preserving, so a corrupted core median stays within ±49.999 ¢ of the unchanged target MIDI. The test split is 52.3% positive corruptions, against 50.4% on validation. The songs are disjoint from train and validation, and their vocalists are too.

## Headline

**Dynamic hybrid: test MAE 10.776 ¢, RMSE 14.551 ¢** (validation 12.970 / 16.791).

| Model | Val MAE | Test MAE | Test − val | Test / val | Val RMSE | Test RMSE | Test median AE |
|---|---|---|---|---|---|---|---|
| Zero (predict no shift) | 20.512 | 20.477 | −0.035 | 0.998 | 25.507 | 25.470 | 19.874 |
| Full median | 18.604 | 14.464 | −4.141 | 0.777 | 25.202 | 20.715 | 9.675 |
| Core median | 17.426 | 13.905 | −3.520 | 0.798 | 21.900 | 18.769 | 9.611 |
| Random Forest (50) | 13.849 | 11.551 | −2.298 | 0.834 | 17.625 | 15.371 | 8.637 |
| Natural offset | 14.051 | 11.666 | −2.384 | 0.830 | 18.425 | 15.941 | 8.231 |
| **Dynamic hybrid (final)** | **12.970** | **10.776** | −2.194 | 0.831 | **16.791** | **14.551** | **7.794** |

Final-model tails (test, |error|): P90 24.6 ¢, P95 30.9 ¢, P99 42.9 ¢, max 71.2 ¢. 9.62% of examples have error > 25 ¢, 0.40% have error > 50 ¢, and none exceed 100 ¢. The mean signed error is −1.87 ¢.

## How to read the headline

**Test was easier than validation, for every method.** The core-median and full-median baselines improved by 3.5 and 4.1 ¢ on test, and every learned model improved by a nearly identical ratio (0.83). The gap reflects these 8 songs, not better generalization. The generated record labels it "excellent (within ~+0.5 c of validation)". That label comes from a mechanical rule and is descriptive only. It isn't evidence that the model generalizes better than it validated.

**The comparison that carries over is relative to the baselines on the same songs:**

| | Validation | Test |
|---|---|---|
| Dynamic vs core median (production-style center), absolute | −4.46 ¢ | −3.13 ¢ |
| Dynamic vs core median, relative | −25.6% | −22.5% |
| Dynamic vs zero, relative | −36.8% | −47.4% |

The advantage over the core-median baseline shrank slightly on test but held.

**The hybrid beats both of its experts, and that holds up under resampling:**

| Statistic (10,000 resamples, seed 20260824) | Point | Note-group 95% CI | Song-cluster 95% CI |
|---|---|---|---|
| Dynamic MAE | 10.776 | [10.513, 11.042] | [8.889, 12.767] |
| Dynamic RMSE | 14.551 | [14.214, 14.900] | [12.376, 16.540] |
| RF MAE | 11.551 | [11.301, 11.803] | [9.585, 13.609] |
| Natural MAE | 11.666 | [11.341, 12.001] | [9.576, 13.878] |
| Dynamic − RF MAE | −0.775 | [−0.883, −0.664] | [−0.981, −0.597] |
| Dynamic − natural MAE | −0.890 | [−1.006, −0.775] | [−1.129, −0.654] |

The note-group CIs treat notes as independent, so they are too narrow for a claim about new songs. The song-cluster CIs come from only 8 songs and the record flags them as unstable. The honest range for the absolute MAE on unseen songs is therefore wide, roughly 9–13 ¢. Both bootstraps exclude zero for the paired differences, though, so "the gate helps over either expert alone" is the firmest result here.

## Where the error is

**By variant** (test MAE; validation in parentheses):

| Model | clean | c1 5–15 ¢ | c2 15–25 ¢ | c3 25–35 ¢ | c4 35–50 ¢ |
|---|---|---|---|---|---|
| Random Forest (50) | 8.18 (9.88) | 9.01 (10.73) | 10.29 (12.28) | 12.59 (15.23) | 17.69 (21.12) |
| Natural offset | 11.67 (14.05) | 11.67 (14.05) | 11.67 (14.05) | 11.67 (14.05) | 11.67 (14.05) |
| **Dynamic hybrid** | 8.62 (10.44) | 9.57 (11.35) | 10.39 (12.33) | 11.35 (13.81) | 13.96 (16.92) |

- **The RF shrinks toward zero**, so it is best on clean and small shifts and worst on large ones. The natural-offset model has the same MAE on every variant by construction: it predicts the clean center N, and observed core = N + shift exactly. The gate sits between the two and leans RF on clean notes (w̄_RF 0.59) and less on c2–c3 (0.46–0.47).
- **Large shifts are still under-predicted.** The corrupted-example slope is 0.694 (RF 0.604, natural 0.799), with R² 0.719. The mean signed error is −9.26 ¢ on positive shifts and +6.16 ¢ on negative shifts. So a singer who is truly 40 ¢ off will tend to be reported as less off than they are.
- **Clean notes are not read as perfect.** On unmodified released vocals the final model's MAE is 8.62 ¢. That is the scale of error the model would attribute to a performance that matches the reference exactly.

**By song** (final model; a model-error diagnostic, not a judgment of the singers):

| Song | Notes | MAE | RMSE | P95 |
|---|---|---|---|---|
| giveon_heartbreak_anniversary_audio | 476 | 7.87 | 10.89 | 21.3 |
| tum_hi_ho_lyrics_arijit_singhaashiqui_2_tseries | 489 | 8.18 | 12.14 | 27.7 |
| ariana_grande_into_you_official_video | 550 | 8.79 | 12.06 | 25.4 |
| kacey_musgraves_rainbow_official_audio | 393 | 9.24 | 12.81 | 27.6 |
| adele_hello | 523 | 10.29 | 14.15 | 31.1 |
| prince_royce_corazon_sin_cara_audio | 440 | 10.40 | 13.55 | 29.6 |
| mariah_carey_all_i_want_for_christmas_is_you | 647 | 14.66 | 18.12 | 33.9 |
| chris_stapleton_tennessee_whiskey_official_audio | 552 | 14.86 | 18.60 | 37.0 |

The spread is nearly 2× (7.9 to 14.9 ¢). Two songs carry much of the error, which is why the song-cluster CI is wide.

**Gate behavior.** Routing is close to a coin flip on which expert is better: the RF was better on 49.9% of examples. The higher-weighted expert was the better one 55.3% of the time. The gate earns most of its gain where the experts disagree by 20 ¢ or more: there its MAE is 15.09, against 18.54 for the RF and 22.04 for the natural model. A per-example oracle choosing the better expert would reach 8.62 ¢ MAE (not deployable), which leaves 2.15 ¢ that better routing could still recover.

## Limits of this result

- **The corruption is synthetic and symbolic.** It is a constant pitch-center shift added to the saved cents contour. It isn't real off-pitch singing, it involves no audio pitch shifting (so no artifact leakage, and no exposure to real artifacts), and it includes no vibrato, timing or timbre errors. The result measures how well the model recovers an injected shift relative to the released vocal. It does not validate the model against human judgments of intonation.
- **The sign of a large shift is partly predictable** from where the clean note sits, because of the target-preserving policy. In band c4, 0.0% (to one decimal, across all splits) of notes with clean core > 15 ¢ were corrupted positively (`ml/experiments/axis2_v3_dataset_audit/axis2_v3_dataset_summary.md`). This is a property of the v3 task definition, but it may let a model learn a prior that real singing doesn't share.
- **The notes are v2-segmented.** Production still segments with v1, so these numbers describe notes production does not produce.
- **The test covers 8 songs and mainstream pop, R&B, country, Latin and Hindi film vocals.** The per-song spread above means one more song could move the headline materially.
- **Nothing here is in production.** `backend/scoring.py` still uses the Phase A curve, and there is no defined mapping from a predicted shift to a user-facing score.

## Integrity and provenance

- The frozen-model manifest was written before any test data was read (`final_model_manifest_before_test.json`, 00:04:56 UTC). It records SHA-256 hashes for the 3 artifacts, 5 configs, 4 dataset files and 9 code files.
- The after-test manifest (00:05:04 UTC) reports `frozen_items_changed_after_evaluation: []`.
- The unlabeled predictions were hashed before labels were loaded: `fe6e74fe51bc336cee6ea81ea4c1d3153de381e51adabfc5f9921e07b197840f`. That file, now at `data/experiment_archives/experiments/axis2_v3_final_test/final_test_predictions_unlabeled.csv`, still matches the hash.
- **Caveat:** the evaluation ran at commit `4af7ec61` with uncommitted changes in the working tree (14 porcelain lines, listed in the manifests). The code that ran is pinned by its recorded file hashes, which still match the tracked files, not by a commit.
- Environment: Python 3.11.15, numpy 2.2.6, scikit-learn 1.7.1, torch 2.13.0, macOS arm64.
