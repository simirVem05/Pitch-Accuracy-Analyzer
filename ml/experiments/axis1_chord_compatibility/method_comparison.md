# Method comparison — all 63 songs, 31,803 notes

Generated from `results.json`. **In-corpus method development; not evidence of generalization.** All values are label-free. A higher "mean H sung" or win rate does **not** mean a method is more correct, since the alternatives are not known to be wrong. The objective metrics are breadth (several pitch classes can fit), specificity (not everything fits), grading and stability.

Column notes: *pcs ≥ x* is the mean number of the 12 pitch classes scoring at least x per note. *Values in (0.25, 0.75)* is the share of all 12×N values that are intermediate, i.e. graded rather than all-or-nothing. *Gap* is the sung note's H minus the H of the same note shifted up by that interval.

## Main methods

| method | pcs ≥ 0.75 | pcs ≥ 0.5 | notes with ≥10 pcs ≥ 0.5 | notes with no pc ≥ 0.75 | max−min | values in (0.25, 0.75) | mean H sung | mean H alt | sung − alt | win rate | P4↑ gap | P5↑ gap | m2↑ gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prod_calibrated | 4.81 | 5.50 | 0.000 | 0.000 | 1.000 | 0.121 | 0.660 | 0.443 | 0.217 | 0.628 | 0.080 | 0.069 | 0.322 |
| prod_harmonic_rank | 4.00 | 7.02 | 0.008 | 0.000 | 0.961 | 0.499 | 0.686 | 0.551 | 0.135 | 0.634 | 0.048 | 0.040 | 0.203 |
| raw_chroma | 2.04 | 3.71 | 0.033 | 0.000 | 0.943 | 0.355 | 0.462 | 0.354 | 0.109 | 0.607 | 0.026 | 0.041 | 0.161 |
| rank_chroma | 3.08 | 5.97 | 0.000 | 0.000 | 0.974 | 0.484 | 0.605 | 0.490 | 0.114 | 0.607 | 0.050 | 0.043 | 0.163 |
| chord_hard | 3.08 | 6.63 | 0.000 | 0.000 | 1.000 | 0.510 | 0.679 | 0.514 | 0.166 | 0.628 | 0.048 | 0.041 | 0.271 |
| chord_dist | 3.20 | 6.53 | 0.000 | 0.000 | 0.992 | 0.496 | 0.679 | 0.514 | 0.165 | 0.640 | 0.048 | 0.040 | 0.270 |
| chord_dist_key | 3.31 | 6.69 | 0.000 | 0.000 | 0.959 | 0.508 | 0.692 | 0.526 | 0.166 | 0.650 | 0.050 | 0.039 | 0.271 |
| chord_dist_fixed_key | 3.26 | 6.55 | 0.000 | 0.000 | 0.939 | 0.502 | 0.687 | 0.524 | 0.163 | 0.648 | 0.050 | 0.037 | 0.269 |
| chord_dist_key_alt_tiers | 5.00 | 7.07 | 0.001 | 0.000 | 0.864 | 0.406 | 0.740 | 0.589 | 0.151 | 0.650 | 0.046 | 0.033 | 0.250 |
| chord_dist_key_ctx | 3.35 | 7.92 | 0.064 | 0.000 | 0.939 | 0.558 | 0.749 | 0.574 | 0.176 | 0.666 | 0.095 | 0.083 | 0.239 |

Method definitions:

- `prod_calibrated`: production key_compliance (rank local+slack ^0.75 · global ^0.25 → clip)
- `prod_harmonic_rank`: production value before calibration
- `raw_chroma`: window-mean CENS / max
- `rank_chroma`: window-mean per-frame rank
- `chord_hard`: tier under the single most likely chord
- `chord_dist`: expected tier over the chord distribution
- `chord_dist_key`: chord_dist, evidence-gated with key context (**H_inst**)
- `chord_dist_fixed_key`: chord_dist with a fixed 0.75/0.25 key blend
- `chord_dist_key_alt_tiers`: H_inst with tier values (1, .75, .4, .1)
- `chord_dist_key_ctx`: H_inst plus melodic/temporal context (**H_ctx**)

## Ablations (each replaces one component of `chord_dist_key`; no context)

| method | pcs ≥ 0.75 | pcs ≥ 0.5 | notes with ≥10 pcs ≥ 0.5 | notes with no pc ≥ 0.75 | max−min | values in (0.25, 0.75) | mean H sung | mean H alt | sung − alt | win rate | P4↑ gap | P5↑ gap | m2↑ gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chord_dist_key | 3.31 | 6.69 | 0.000 | 0.000 | 0.959 | 0.508 | 0.692 | 0.526 | 0.166 | 0.650 | 0.050 | 0.039 | 0.271 |
| abl_template_harmonic | 3.29 | 6.66 | 0.000 | 0.000 | 0.958 | 0.511 | 0.692 | 0.525 | 0.167 | 0.651 | 0.047 | 0.044 | 0.268 |
| abl_temporal_beat | 3.31 | 6.69 | 0.000 | 0.000 | 0.959 | 0.508 | 0.692 | 0.526 | 0.166 | 0.650 | 0.050 | 0.039 | 0.271 |
| abl_temporal_beat_hmm | 3.30 | 6.83 | 0.000 | 0.000 | 0.956 | 0.522 | 0.699 | 0.533 | 0.166 | 0.651 | 0.053 | 0.039 | 0.271 |
| abl_temporal_frame | 3.31 | 6.70 | 0.000 | 0.000 | 0.957 | 0.509 | 0.690 | 0.527 | 0.164 | 0.649 | 0.050 | 0.038 | 0.268 |
| abl_temporal_frame_hmm | 3.26 | 6.74 | 0.000 | 0.000 | 0.967 | 0.516 | 0.692 | 0.527 | 0.165 | 0.647 | 0.050 | 0.038 | 0.269 |
| abl_vocab_triads | 3.33 | 6.66 | 0.000 | 0.000 | 0.945 | 0.525 | 0.699 | 0.529 | 0.170 | 0.654 | 0.057 | 0.040 | 0.277 |
| abl_vocab_triads_7ths | 3.52 | 6.72 | 0.000 | 0.000 | 0.952 | 0.498 | 0.705 | 0.533 | 0.172 | 0.652 | 0.058 | 0.037 | 0.287 |

## Stability

Mean and p95 of |ΔH| over all 12 values per note, under (a) multiplicative CENS noise (lognormal σ = 0.10, re-normalized) and (b) ±50 ms note shifts.

| method | noise mean | noise p95 | shift mean | shift p95 |
|---|---|---|---|---|
| chord_dist_key | 0.0049 | 0.0240 | 0.0085 | 0.0485 |
| prod_calibrated | 0.0145 | 0.0941 | 0.0139 | 0.0970 |
| raw_chroma | 0.0116 | 0.0486 | 0.0174 | 0.0620 |
| rank_chroma | 0.0171 | 0.0682 | 0.0196 | 0.0727 |

## By song group (Axis 2 pool vs the 13 excluded)

| group | method | mean H sung | mean H alt | pcs ≥ 0.75 | pcs ≥ 0.5 | win rate |
|---|---|---|---|---|---|---|
| axis2_50 | prod_calibrated | 0.650 | 0.441 | 4.79 | 5.46 | 0.622 |
| axis2_50 | raw_chroma | 0.441 | 0.334 | 1.98 | 3.47 | 0.603 |
| axis2_50 | rank_chroma | 0.600 | 0.491 | 3.09 | 5.97 | 0.602 |
| axis2_50 | chord_dist | 0.672 | 0.515 | 3.21 | 6.53 | 0.633 |
| axis2_50 | chord_dist_key | 0.683 | 0.524 | 3.33 | 6.64 | 0.644 |
| axis2_50 | chord_dist_key_ctx | 0.746 | 0.572 | 3.37 | 7.87 | 0.662 |
| excluded_13 | prod_calibrated | 0.705 | 0.453 | 4.86 | 5.66 | 0.652 |
| excluded_13 | raw_chroma | 0.547 | 0.435 | 2.24 | 4.68 | 0.626 |
| excluded_13 | rank_chroma | 0.623 | 0.489 | 3.07 | 5.99 | 0.625 |
| excluded_13 | chord_dist | 0.709 | 0.511 | 3.13 | 6.55 | 0.667 |
| excluded_13 | chord_dist_key | 0.728 | 0.536 | 3.25 | 6.92 | 0.677 |
| excluded_13 | chord_dist_key_ctx | 0.764 | 0.582 | 3.28 | 8.10 | 0.682 |

## Chord-sequence behavior on real songs, by temporal estimator (no labels)

| estimator | switches/beat | flicker/beat | N share | diatonic share | beat-grid +50 ms agreement |
|---|---|---|---|---|---|
| frame | 0.559 | 0.072 | 0.310 | 0.880 | — |
| frame_hmm | 0.512 | 0.058 | 0.327 | 0.881 | — |
| beat | 0.529 | 0.060 | 0.305 | 0.882 | 0.904 |
| beat_hmm | 0.183 | 0.001 | 0.494 | 0.899 | 0.949 |

Quality share of beat-level top chords (`beat`, mean over songs): maj 23.1%, power 20.8%, min7 14.1%, maj7 13.7%, min 13.2%, sus2+sus4 11.5%, dom7 2.0%, hdim7 0.8%, dim 0.5%, aug 0.3%

## Selected method by note bucket (sung note)

| variable | bucket | notes | H_inst | H_ctx | production |
|---|---|---|---|---|---|
| duration_s | [0,0.1) | 6399 | 0.580 | 0.692 | 0.515 |
| duration_s | [0.1,0.2) | 10194 | 0.669 | 0.727 | 0.634 |
| duration_s | [0.2,0.4) | 7562 | 0.725 | 0.764 | 0.713 |
| duration_s | [0.4,0.8) | 5001 | 0.768 | 0.799 | 0.757 |
| duration_s | [0.8,inf) | 2647 | 0.814 | 0.837 | 0.782 |
| mean_crepe_conf | [0,0.75) | 4585 | 0.650 | 0.705 | 0.629 |
| mean_crepe_conf | [0.75,0.85) | 10088 | 0.690 | 0.743 | 0.652 |
| mean_crepe_conf | [0.85,0.95) | 16591 | 0.707 | 0.766 | 0.676 |
| mean_crepe_conf | [0.95,1.01) | 539 | 0.619 | 0.734 | 0.592 |
| abs_core_cents | [0,40.0001) | 29558 | 0.703 | 0.757 | 0.673 |
| abs_core_cents | [40.0001,inf) | 2245 | 0.553 | 0.650 | 0.489 |
| evidence | [0,0.5) | 9253 | 0.712 | 0.751 | 0.672 |
| evidence | [0.5,0.8) | 3565 | 0.699 | 0.748 | 0.684 |
| evidence | [0.8,0.95) | 4113 | 0.697 | 0.759 | 0.682 |
| evidence | [0.95,1.0001) | 14872 | 0.677 | 0.746 | 0.641 |
| top1_chord_mass | [0,0.2) | 3 | 0.139 | 0.139 | 0.000 |
| top1_chord_mass | [0.2,0.35) | 146 | 0.755 | 0.796 | 0.740 |
| top1_chord_mass | [0.35,0.5) | 1732 | 0.722 | 0.775 | 0.721 |
| top1_chord_mass | [0.5,0.75) | 7786 | 0.709 | 0.763 | 0.686 |
| top1_chord_mass | [0.75,1.0001) | 22136 | 0.683 | 0.743 | 0.646 |

## Context specificity

Share of notes whose score context raises: **sung pitch class 0.191**, against **alternative pitch classes 0.162** (same rules, same neighbors).

| code | sung notes | per alternative (count/11) |
|---|---|---|
| ANTICIPATION | 153 | 177 |
| CROSSES_CHANGE | 2193 | 2159 |
| NEIGHBOR | 1088 | 574 |
| PASSING | 623 | 173 |
| STEP_RESOLUTION | 2011 | 2063 |
| SUSPENSION | 24 | 14 |

## Runtime

Selected pipeline (posterior, smoothing, 5 note windows, context): mean 0.03 s per song, max 0.07 s; 0.008 s per minute of audio on this machine, excluding Demucs and CREPE.

