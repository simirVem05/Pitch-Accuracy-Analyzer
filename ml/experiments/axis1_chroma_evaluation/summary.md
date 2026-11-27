# Axis 1 chroma evaluation — summary

**What this is.** A label-free evaluation of chroma-based harmonic-fit scoring on the saved feature store: 63 songs, 31,803 v2 notes, `other`-stem CENS chroma. Each sung pitch class is compared against the 11 counterfactual alternatives in the same time window. 17 deterministic scoring rules, including the production rule, were tested under one protocol (`protocol.md`). That protocol was written, and the song split (`axis1_splits.json`) fixed, before any score was computed.

**What it is not.** It is not accuracy against human judgments. No harmonic-fit labels exist. An alternative pitch class is a counterfactual, not a wrong note, and a released vocal is not assumed to fit perfectly on every note. Every rate below is a *proxy*.

| File | Contents |
|---|---|
| `protocol.md`, `axis1_splits.json` | the pre-registered protocol and split |
| `axis1_chroma_evaluation.py` | the evaluation (≈4 s; verifies against `backend/harmony.py`) |
| `results.json`, `tables.md` | all results; generated tables |
| `method_comparison.csv`, `song_level_methods.csv`, `interval_win_rates.csv`, `perturbation_summary.csv`, `transposition_songs.csv`, `diagnostic_buckets.csv` | tables |
| `data_audit_songs.csv` | per-song eligibility and integrity audit |
| `posthoc_checks.py` / `.json` | analyses added **after** seeing results; not used for selection |
| `data/experiment_archives/axis1_chroma_evaluation/note_level.csv.gz` | per-note table, 31,803 rows (Git-ignored; SHA-256 in `results.json`) |

## Verification

- **Production reproduced.** `prod_geo_w25` matches `backend/harmony.py:score_key_compliance` on every note: max abs difference 7.8e-13 on `harmonic_rank` and 8.3e-13 on `local_support`.
- **Stored arrays are consistent.** The stored `other_chroma_rank` equals production `_to_rank(other_chroma_cens)` for all 63 songs, and the global profile and rank are consistent. The stored chroma was produced by the current chroma code: `harmony.py` changed after the store's build commit only by adding a dataclass field.
- **Sources untouched.** The SHA-256 of all 126 source `.npz` files was identical before and after the run, and each matches the `harmony_npz_sha256` in its `metadata.json`.
- **One post-run change.** The score-equality tolerance went from 1e-12 to 1e-9, because cumulative-sum rounding (8e-13) sat close to 1e-12. Re-running changed **no** reported metric (max change 0) and not the selection.

## 1. Data eligibility and integrity

| | Songs | Notes | Eligible | High-confidence subset |
|---|---|---|---|---|
| dev (40 of the 50) | 40 | 19,965 | 19,965 | 4,427 |
| holdout (10 of the 50) | 10 | 5,582 | 5,582 | 1,073 |
| robustness (the 13 `axis2_ml=false`) | 13 | 6,256 | 6,256 | 1,513 |
| **all** | **63** | **31,803** | **31,803 (100%)** | **7,013 (22.1%)** |

**Structurally clean:**
- no invalid, non-finite or empty-window notes
- no `pitch_class ≠ target_midi % 12`
- no all-zero chroma frames, no overlapping or unsorted notes, no contour outside its note bounds
- chroma, vocal and metadata timelines agree to within one chroma hop (−23 to +16 ms); every note ends before the chroma does

**Flagged, but kept and reported by bucket rather than dropped:**

| Flag | Count |
|---|---|
| notes < 0.1 s | 8,036 (25.3%) |
| notes spanning a single chroma frame | 162 |
| notes ≥ 40 ¢ from their semitone (pitch class ambiguous) | 2,245 (7.1%) |
| contour frames below the 0.60 CREPE threshold inside notes | 18,987 of 491,403 (3.9%) — most likely the voicing mask's 1-frame gap fill (inferred, not verified) |

**Tempo, used for the ±1-beat slack.** The range is 49.7–184.6 BPM, and four songs look like tempo-octave errors: `at_last` 172, `jeff_buckley_hallelujah` 184.6, `tum_hi_ho` 184.6 and `dangerously_in_love` 49.7. Their slack is half or double a beat. This can't be verified without audio.

**Tuning.** The range is −23 to +42 ¢. `billiejean` at +42 ¢ is close to a quarter-tone, which makes pitch-class assignment fragile.

**A production defect found here.** `_to_rank` breaks ties by `argsort` order, i.e. by pitch-class **index**. CENS leaves many bins at exactly zero: a median of 28.8% of frames contain a tie (max 76.4%). Among tied bins, C loses 0.12 rank on average and B gains 0.12 (`posthoc_checks.json`). Its effect on scores is small: the sung note's production score moves by 0.0015 on average (p99 0.057, max 0.29), and fixing it (`geo_w25_avgties`) changes no aggregate by more than 0.001. It is a real bias, but not a material one.

**Not measurable from saved data** (needs audio):
- the a cappella guard (stem RMS was not saved)
- the production bleed test (no `bass` or `other+bass` chroma)
- percussion-heavy sections (no percussive stem; CENS is per-frame normalized, so energy is lost)
- alternative chroma types (CQT, HPSS, unsmoothed chroma)
- backing-vocal presence

## 2. The baseline (production Axis 1)

- **Feature:** CENS chroma of the `other` stem at 22,050 Hz, hop 512, tuning-corrected.
- **Rank:** each frame's 12 values are converted to a per-frame rank in [0, 1].
- **Local term:** the mean rank of the sung pitch class over the note's frames, maximized over that window and the same-width window shifted ±1 beat (shifted, not widened).
- **Global term:** the pitch class's rank in the whole-song mean chroma.
- **Combination:** `local^0.75 · global^0.25`, then a linear map of [0.50, 0.75] → [0, 1], clipped. The headline is the duration-weighted mean.
- **Provenance of the constants.** `RANK_LOW`, `RANK_HIGH` and `GLOBAL_WEIGHT` were set in Aug 2026 on five bundled songs. Two of them (`into_you`, `dont`) are, by title, songs in this store. They are different audio files, but they were forced into dev.
- **Known weakness:** perfect-fourth/fifth confusion under transposition (14/20 historically).

## 3. Experiment B — per-note alternatives

The headline metric is the pairwise **win rate**: the share of the 11 alternatives the sung pitch class outscores (ties count half; chance = 0.5). Below it, song-level means (primary) and pooled notes.

| Method | dev song-mean | holdout | robustness | all 63 | all-63 top-1 | top-3 | mean placement /12 | high-conf win (all 63) |
|---|---|---|---|---|---|---|---|---|
| raw_local | 0.603 | 0.613 | 0.635 | 0.611 | 0.128 | 0.378 | 5.32 | 0.635 |
| rank_local | 0.602 | 0.613 | 0.635 | 0.610 | 0.126 | 0.375 | 5.33 | 0.634 |
| rank_local_slack | 0.614 | 0.621 | 0.647 | 0.622 | 0.075 | 0.366 | 5.20 | 0.659 |
| global | 0.634 | 0.638 | 0.671 | 0.642 | 0.144 | 0.408 | 4.97 | 0.700 |
| geo_w25_noslack | 0.616 | 0.625 | 0.653 | 0.625 | 0.135 | 0.392 | 5.17 | 0.659 |
| **prod_geo_w25 (production)** | **0.629** | **0.635** | **0.669** | **0.638** | **0.144** | **0.404** | **5.02** | **0.685** |
| geo_w50 | 0.636 | 0.640 | 0.676 | — | — | — | — | — |
| arith_w50 | 0.637 | 0.640 | 0.676 | — | — | — | — | — |
| chord_local_slack (expl.) | 0.626 | 0.617 | 0.653 | 0.630 | 0.000† | 0.403 | 5.11 | 0.678 |
| chord_geo_w25 (expl.) | 0.643 | 0.641 | 0.680 | 0.650 | 0.152 | 0.418 | 4.89 | 0.708 |

All 17 methods are in `tables.md`. † A triad-template score ties a pitch class with the other tones of its best chord, so a strict top-1 is impossible for `chord_local_slack`.

**How good production is.** On all 63 songs, the sung pitch class beats 63.8% of alternatives, is the single best candidate 14.4% of the time and is in the top 3 for 40.4%. Its median margin over the best alternative is **negative** (−0.21 on dev). Duration-weighted, which matches how the headline is computed, the dev song-mean win rate is 0.695. The placement distribution is broad: all 63 songs, 31,803 notes, 4,566 at #1 and 1,267 at #12.

**Win rate against specific substitutions** (production, all 63; +k = k semitones up mod 12):

| +1 m2↑ | +2 M2↑ | +3 | +4 | **+5 P4↑** | +6 TT | **+7 P5↑** | +8 | +9 | +10 M2↓ | +11 m2↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.688 | 0.621 | 0.608 | 0.687 | **0.509** | 0.712 | **0.595** | 0.611 | 0.624 | 0.587 | 0.736 |

"±5 and ±7" collapse into two pitch-class offsets, because +5 ≡ −7 and +7 ≡ −5. A note relabelled a fourth up is a coin flip for production (0.509, dev 0.500), and a fifth up is barely better (0.595). Semitones (0.69 / 0.74) and the tritone (0.71) are the easiest substitutions to tell apart.

**Where per-note ranking is weakest** (production win rate, all 63):

| Condition | Win rate |
|---|---|
| duration < 0.1 s, 25% of notes | 0.542 |
| 0.1–0.2 s | 0.616 |
| 0.2–0.4 s | 0.668 |
| 0.4–0.8 s | 0.712 |
| ≥ 0.8 s | 0.744 |
| pitch-class-ambiguous notes (> 40 ¢ off), 7% | 0.513 (chance) |
| sung pitch class tied in CENS (≥ half the window frames), 5% | 0.245 |
| within-note SD < 15 ¢ | 0.663 |
| within-note SD ≥ 35 ¢ | 0.628 |

Duration is the strongest effect measured. Tied notes are where the accompaniment has *no* energy at the sung pitch class, which makes them the closest thing to genuinely unsupported notes in this data.

Two buckets are **not** monotone and are not interpreted further. CREPE confidence: 0.616 below 0.75, 0.644 at 0.85–0.95, 0.578 above 0.95 (536 notes). Chroma clarity: the high-clarity tertile scores slightly *lower* (0.620 vs 0.643).

**Local × global split** (production terms, all 63):

| | share of notes |
|---|---|
| local top-3 and in key | 35.2% |
| local weak, in key (passing/non-chord-tone candidates) | 42.9% |
| local top-3, out of key | 1.5% |
| both weak (chromatic / out-of-key / extraction-error candidates) | 20.5% |

Only about 37% of sung notes are among the 3 most prominent accompaniment pitch classes at that moment.

## 4. Experiment C — controlled perturbations

20% of each song's notes are relabelled, with 5 seeded draws per pattern per song (315 draws in all). Timing and chroma are held fixed.

| Production | Draws lowering the song mean | Songs with 5/5 | Affected notes lowered | Headline drop |
|---|---|---|---|---|
| semitone ±1 | 314/315 | 62/63 | 72.1% | **9.2 pp** |
| whole tone ±2 | 308/315 | 57/63 | 60.5% | 5.1 pp |
| **fourth/fifth (+5/+7)** | **256/315** | **39/63** | **55.5%** | **2.4 pp** |
| tritone | 314/315 | 62/63 | 71.5% | 10.2 pp |
| random nonzero | 310/315 | 60/63 | 63.8% | 6.2 pp |

**Within a song, the score almost always falls**, except for fourth/fifth substitutions: in 19% of those draws the song mean is unchanged or *rises*.

**Across songs, the drop is small relative to natural variation.** The production headline for unmodified released songs spans 55–93% with an SD of 8.2 pp. 20% wrong-by-a-semitone notes cost about 9 pp, and 20% fourths/fifths cost about 2.4 pp. A single absolute headline therefore can't tell "this song, sung with 20% wrong notes" from "a different song, sung correctly". The score is informative as a *within-song* comparison, not as an absolute grade.

**When a substitution fails to lower the note's score** (production, all 63):
- It depends strongly on duration. For semitone substitutions the failure rate is 43% for notes under 0.1 s and 13% for notes of 0.8 s or longer.
- Up a fourth (+5) fails on 49% of notes, up a fifth (+7) on 40%.
- Chroma clarity has no effect.

The other methods behave similarly. `chord_geo_w25` lowers 262/315 fourth/fifth draws, against 256 for production (`tables.md`).

## 5. Experiment D — song-level transposition

The original melody's placement among its 12 transpositions (unweighted mean, as in `evaluate.py`):

| Method | dev #1 | holdout #1 | robustness #1 | **all 63 #1** | failures | of which P4/P5 |
|---|---|---|---|---|---|---|
| raw_local | 21/40 | 6/10 | 10/13 | 37/63 | 26 | 23 |
| rank_local | 24/40 | 8/10 | 10/13 | 42/63 | 21 | 17 |
| global | 24/40 | 7/10 | 9/13 | 40/63 | 23 | 21 |
| geo_w25_noslack | 25/40 | 10/10 | 10/13 | 45/63 | 18 | 15 |
| **prod_geo_w25** | **27/40** | **9/10** | **10/13** | **46/63** | **17** | **15** |
| chord_local_slack (expl.) | 36/40 | 9/10 | 12/13 | 57/63 | 6 | 4 |
| chord_geo_w25 (expl.) | 32/40 | 8/10 | 11/13 | 51/63 | 12 | 9 |

- **Production placements** (all 63): 46 at #1, 11 at #2, 4 at #3 and 2 at #4. The median margin over the best wrong key is +0.020.
- **Replication:** production fails on 17 of 63 songs, 15 of them to a perfect fourth or fifth. That reproduces the historical 14/20 with all failures to P4/P5.
- **Chord templates help on this test:** they cut failures from 17 to 6. But that gain is concentrated in dev. On holdout, `chord_local_slack` and production tie at 9/10.

This test is about **key identification**, not individual wrong notes. Uniform transposition preserves the melody's internal intervals and keeps it diatonic, so it doesn't resemble how singers err.

## 6. Pre-registered selection, and holdout

- **Selection on dev** (highest song-mean win rate): **`chord_geo_w25`**, exploratory. Against production it scores **+0.0135** [+0.0085, +0.0183], better on 32/40 songs. The best non-exploratory method was `arith_w50` at +0.0079 [+0.0051, +0.0109].
- **Holdout** (reported once): +0.0061 [+0.0004, +0.0128], better on **7/10** songs. That meets the pre-registered "≥ 7/10" rule **exactly at its threshold**. The improvement is about 0.6 points of win rate, and on holdout transposition `chord_geo_w25` is *worse* than production (8/10 vs 9/10).
- **Robustness group** (13 songs, not used for selection): `chord_geo_w25` 0.680 vs production 0.669.

## 7. Post-hoc checks (added after seeing results; not used for selection)

**Local vs global** (paired song-level win rate against `global` alone, all 63):

| | diff vs `global` | 95% CI | songs better |
|---|---|---|---|
| production | −0.004 | [−0.010, +0.002] | 25/63 |
| geo_w50 | +0.003 | [−0.001, +0.006] | 33/63 |
| arith_w50 | +0.003 | [−0.001, +0.007] | 37/63 |
| chord_geo_w25 | +0.008 | [+0.003, +0.012] | 44/63 (holdout alone: +0.002, CI includes 0) |
| rank_local_slack | −0.021 | [−0.029, −0.012] | 16/63 |

**Slack.** Moving from the note span alone to the shifted ±1-beat windows changes the order of 39,787 sung/alternative pairs out of 349,833:
- 21,912 times it lifts the sung note above an alternative
- 17,875 times it lifts an alternative above the sung note
- net +4,037 pairs (1.2%)

It also raises saturation. The share of alternatives scoring 100% rises from 22.1% to 31.6%, and of sung notes from 37.3% to 51.8%. The gap between sung and alternative mean calibrated score widens only from 0.193 to 0.217.

**Calibration.** In production, 51.8% of sung notes score exactly 100% and 23.3% score 0%. The 100% figure is 31.6% for alternatives. The mean calibrated score is 0.660 for sung notes and 0.443 for alternatives.

**Bleed proxy.** The rank of the sung pitch class during the note, minus its rank in nearby unvoiced, non-note frames: median 0.000, positive for 49.4% of notes (robustness group 52.5%, mean +0.015). This gives **no sign** that vocal leakage props up the sung pitch class. It is a weak proxy, though, and the real bleed test needs audio.

**Song-level correlates** (Spearman, production win rate, all 63): every one is weak (|ρ| ≤ 0.26). Vocal coverage +0.08, clarity −0.15, tie fraction −0.16, note count −0.26.

## 8. Answers

1. **Usable songs: 63 of 63.** All have v2 notes, chroma, global profiles, beats and tuning. None fails a structural check.
2. **Evaluable notes: 31,803**, all of them. That is 25,547 in the 50-song pool (dev 19,965, holdout 5,582) and 6,256 in the 13. 7,013 (22.1%) pass the strict high-confidence filter, and on that subset production's win rate is 0.685.
3. **Current algorithm: weakly.** The sung pitch class beats 63.8% of alternatives (chance 50%), is top-1 for 14% of notes and top-3 for 40%, with a mean placement of 5.0 of 12. It is better on long, stable notes (0.74 at ≥ 0.8 s) and at chance on short or pitch-ambiguous ones. Duration weighting, which the headline uses, lifts the dev song-mean win rate to 0.695. Within a song, scattered wrong notes almost always lower the mean. But the absolute headline cannot separate a 20%-wrong performance from between-song variation (SD 8.2 pp).
4. **Raw vs rank: no per-note difference** (0.611 vs 0.610 on all 63). Rank is better for song-level key identification: 42 vs 37 songs at #1, and a median transposition margin of 0.023 vs 0.004. That matches the earlier "rank doubles separation" claim, which was a transposition measurement. Rank does not help pick the sung note within a moment.
5. **Local vs global: global carries more per-note information.** `global` alone (0.642) beats local alone with slack (0.622; −0.021 [−0.029, −0.012]). Local information is what lets the system identify the key over time: combined methods beat global alone on transposition (46 vs 40 at #1).
6. **Combining: it helps local, but hardly helps global.** Production does not beat `global` alone per note (−0.004, CI includes 0). Raising the global weight to 0.5 makes the combination ≈ global + 0.003 (CI includes 0). On transposition the geometric weights from 0.10 to 0.50 are flat (45–47 of 63 at #1). Geometric and arithmetic forms perform the same at matched weights.
7. **Slack: a small net gain, built on large churn.** It raises the win rate by about 0.013 and the transposition #1 count from 45 to 46. But it lifts an alternative over the sung note in 17,875 pairs and raises the share of alternatives at 100% from 22% to 32%. It produces many false positives and slightly more true positives.
8. **Fourth/fifth confusion: severe and consistent.**
   - Per note, a fourth-up substitution is indistinguishable from the sung note (0.509), and a fifth-up nearly so (0.595).
   - Under perturbation, 19% of 20%-fourth/fifth draws do not lower the song mean, and the headline cost is 2.4 pp.
   - Under transposition, 15 of production's 17 failures are to P4/P5.
   - Chord templates reduce transposition failures (6 vs 17 on all 63), but not on holdout, and not the per-note P4 coin flip (0.521).
9. **Controlled wrong notes: detected within songs, weakly per note.** Production lowers the song mean on 314/315 semitone draws and 310/315 random draws, but only 256/315 fourth/fifth draws. Per affected note, it lowers the score 72% of the time for semitones, 61% for whole tones and 56% for fourths/fifths. Failures concentrate on short notes.
10. **50 vs 13: the 13 score *higher*, not lower.** Production's song-mean win rate is 0.669 for the 13 against 0.630 for the 50, and the rap/speech-like subset scores 0.651. Axis 2 exclusion reasons (rap, multiple leads, processing) do not predict worse Axis 1 proxy scores. A plausible but untested explanation is that those tracks have simpler or loop-based harmony.
11. **Weakest cases:**
    - **Notes:** notes under 0.1 s (25% of notes), pitch-class-ambiguous notes, notes whose pitch class is tied/absent in the accompaniment chroma, and pitch-unstable notes.
    - **Songs:** `at_last` (0.549; also a tempo-octave suspect), `don_t` (0.563; transposition #4, lost to P4), `demi_lovato_stone_cold` (0.567), `jessie_j_who_you_are` (0.569; transposition #4), `if_i_ain_t_got_you` (0.580) and `jennifer_hudson_spotlight` (0.582). Most are ballads with dense or extended harmony.
    - **Strongest songs:** `jingle_bells` (0.752), `gata_only` (0.745), `love_of_my_life`, `sunflower` and `starboy` (0.70–0.71).
12. **Clearly superior methods: none.**
    - `chord_geo_w25` was selected by the pre-registered rule and passed the holdout rule only at its threshold (+0.006, 7/10), while doing slightly worse on holdout transposition.
    - `arith_w50`/`geo_w50` give a small, consistent per-note gain (+0.007–0.008 on dev).
    - `chord_local_slack` is much better at transposition on dev, but ties on holdout.
    
    These are leads, not replacements.
13. **What this cannot establish without human labels:**
    - whether any score tracks perceived harmonic fit
    - whether a "both weak" note is a mistake, a blue note, a chromatic passing tone or an extraction error
    - where `RANK_LOW`/`RANK_HIGH` should sit
    - whether duration weighting matches perception
    - the true wrong-note rate in released vocals, which caps every proxy here
    
    Alternatives are not known to be wrong: an in-key third substituted for a root is often musically fine.
14. **Chord recognition: yes, as the next feature to test. It is not proven.** Even crude triad templates on smoothed CENS cut transposition failures from 17 to 6 and gave the best per-note combination. The P4/P5 problem is a presence-vs-function problem that raw chroma reweighting cannot fix: every chroma method tried leaves the fourth-up win rate at 0.49–0.52 (0.54 at best with chord templates). The holdout did not replicate the transposition gain, so the effect size is unknown.
15. **Next experiment:** see below.

## 9. Recommended next experiment

**Chord-sequence estimation from the saved CENS, pre-registered, with a fresh evaluation set.**

- **Methods:**
  - templates beyond triads: 7ths, sus, power chords and no-chord
  - temporal smoothing (an HMM/Viterbi chord sequence rather than per-window argmax)
  - each note scored by its relation to the estimated chord *and* to the estimated key function
- **Primary endpoints, fixed in advance:**
  - the fourth-up per-note win rate, which no method here has moved above 0.54
  - transposition failures to P4/P5
  - the perturbation detection rate on notes of at least 0.2 s
- **Evaluation set.** This experiment's 10-song holdout has now been used for one selection. Use the 13 robustness songs as the evaluation set (they were reported but never used to select) and declare that before running. Better still, add newly processed songs once audio is available.

**Before or alongside that** — the bottlenecks found here that chroma rules cannot fix:
- **Short notes.** A quarter of all notes are under 0.1 s and score near chance. This should be a scoring-policy decision (a minimum duration, or explicit down-weighting) rather than something left to the duration weighting.
- **Label-free evaluation has run out of resolution.** A small human-labelled pilot is the only way to know whether a 0.006 win-rate gain matters: 200–300 notes across 10 songs, labelled fit / tension / wrong / unclear.
- **Audio-dependent checks.** CENS is smoothed over about 41 frames (≈ 0.95 s at librosa's default), which may be why "local" behaves like a blurred global. Re-extracting chroma with less smoothing, re-checking the a cappella guard and re-running the bleed test all need the source audio.

Production scoring was not changed.
