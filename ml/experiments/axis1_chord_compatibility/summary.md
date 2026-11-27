# Axis 1 Experiment 2 — chord-aware harmonic compatibility: summary

**Question.** *How harmonically compatible is this sung pitch class with the accompaniment at this moment?* For each note the model produces `H[0..11] ∈ [0,1]`, one value per pitch class, and several can be high at once. It is not a probability of correctness, and it does not try to recover the artist's melody.

**Scope.** Method development on all 63 songs (31,803 v2 notes). There is no held-out split, so **in-corpus comparisons are not evidence of generalization**. No human labels exist, so nothing here establishes perceptual validity.

**Discipline.** Every free parameter was fitted on synthetic audio rendered through the exact production chroma call. That call was verified identical to `build_harmonic_context`. The parameters are template type, similarity, posterior temperature, no-chord threshold, temporal estimator and key temperature. Musically motivated constants (tiers, smoothing prior, context thresholds) were fixed in `protocol.md` before any run. Analyses added afterwards are in `posthoc_checks.py` / `.json` and labelled post hoc here. The feature store was read-only: SHA-256 of all 126 source files is unchanged.

| File | Contents |
|---|---|
| `protocol.md` | design, tier table with rationale, and pre-registered synthetic expectations |
| `chord_model.py`, `synthetic_audio.py`, `axis1_chord_compatibility.py`, `report.py` | implementation (≈80 s end to end) |
| `results.json`, `song_level.csv` | all results |
| `synthetic_tests.md`, `method_comparison.md`, `failure_analysis.md` | generated reports |
| `posthoc_checks.py`, `posthoc_checks.json` | post-hoc analyses |
| `data/experiment_archives/axis1_chord_compatibility/note_level.csv.gz` | the per-note table: top-3 chords, evidence, all 12 H values, reason codes, production score (hash in `results.json`) |

## The model that was evaluated

1. **Harmony from instrumental chroma only.** Every beat segment's mean CENS (between saved beats) is compared against 112 distinct chord pitch sets plus "no chord" (N). The 112 sets come from 11 qualities × 12 roots after merging the chroma-identical sets: sus2(X) = sus4(X+7), and augmented is symmetric. Binary templates and Pearson correlation are used, with posterior `softmax(r/0.02)` and N score θ = 0.8.
2. **Chord tiers.** Each chord quality assigns every interval one of four ordinal tiers:
   - **T1** chord tone (1)
   - **T2** stable extension or color (2/3), e.g. 9 and 13 over major, 11 and b7 over minor
   - **T3** tension usable with context (1/3), e.g. the natural 11 over major, b9/#9/b13 over dom7, blue b3/b7/b5
   - **T4** clash (0), e.g. b9 over a plain major triad, maj7 over dom7
   
   The full table and rationale are in `protocol.md`.
3. **`H_chord(pc)`** is the expected tier under the chord distribution of the note window.
4. **`H_inst = e·H_chord + (1−e)·H_key`**, where `e = 1 − p(N)` and `H_key` is the expected tier under a Krumhansl–Kessler key posterior from the global chroma.
5. **`H_ctx`** adds deterministic phrase rules (passing, neighbor, suspension, step resolution, anticipation, chord-change crossing). It is a separate output.

Because the tiers are fixed, **an H of 0.8 means the same thing in every song**: the expected tier under the estimated harmony lies between "chord tone" and "stable extension".

## Key results

### Chord estimation (synthetic, known truth)

**Single frames:**
- With clean voicings and 3 timbres, top-1 accuracy is 98% and top-3 is 100%.
- With one extra non-chord tone at half level and 15 dB noise:

| chord quality | top-1 (degraded) |
|---|---|
| dom7 / maj7 / min7 / hdim7 | 0.99–1.00 |
| aug | 0.92 |
| sus | 0.93 |
| dim | 0.81 |
| maj | 0.77 |
| min | 0.69 |
| power | 0.60 |

  Top-3 stays at 0.99–1.00. Triads under an extra tone are mistaken for the 7th chord containing it, and power chords become sus.
- Pearson correlation beat cosine under degradation (0.866 vs 0.731 top-1), and binary templates beat harmonic templates.

**Temporal models** (48 progressions with 1-, 2- and 4-beat chords):

| estimator | mean frame accuracy | 1-beat chords |
|---|---|---|
| frame | 0.87 | 0.76 |
| frame_hmm | 0.89 | 0.80 |
| **beat (selected)** | **0.97** | **0.92** |
| beat_hmm | 0.93 | 0.80 |

- **The beat-level HMM masks real 1-beat changes:** it detects only 85% of them.
- **CENS smoothing costs frame-level estimators accuracy near chord changes.** Within 0.1 s of a change, frame accuracy is 0.44, against 0.97 for beat-synchronous estimation. The synthetic chords change exactly on the beat grid, which flatters the beat methods.

**Key estimation is weak:** 60% MAP accuracy on synthetic progressions. Of the 19 errors, 9 were relative major/minor and 8 a fourth or fifth away.

### Synthetic musical tests (`synthetic_tests.md`)

Pre-registered tally: template layer 25/28, audio layer 24/28. Case 8 (octave invariance) failed only because of a **harness bug**: it placed three octave notes at the same instant, so the context rules treated them as neighbors. Re-run correctly, it passes in both layers (`posthoc_checks.json`). **Corrected: template 26/28, audio 25/28.**

**What works, in both layers:**
- C, E and G all reach 1.0 over C major, and A, C and E over A minor. G7 and Cmaj7 give all four tones high scores.
- The extensions D, A and B reach 0.67 over C major, above the clashing C#, F# and G#.
- The avoid note F (0.33) sits below the 9th, D (0.67).
- Over Csus4 the third, E, gets 0.33 — "tension", not "out".
- Progression C–Am–F–G: 100% interior accuracy.
- A passing F between E and G is lifted from 0.33 to 0.67.
- The re-articulated 4–3 suspension is recognized.
- An unresolved C# over C stays at 0.
- Results are identical under ±50 ms shifts.
- Chord tones stay at 1.0 under 10 dB noise.

Production, on the same cases, treats extensions inconsistently (over C, audio layer: D = 0, A = 0, B = 0.91) and gives the passing F 0.

**Real failures:**
- **Case 11 (held suspension):** a single F note spanning F→C is half-supported (H_inst 0.67), so it is not a "candidate". Instead the crossing rule gives it H_ctx = 1.0. The behavior is lenient, and the SUSPENSION rule never fires for held suspensions.
- **Case 13b (single sustained pitch):** E and Eb both get about 0.66 ✓, but the posterior names it C5 with 100% confidence. **The posterior is overconfident**, and the ambiguity flag never triggers.
- **Case 13a, audio layer (white noise):** evidence is correctly low (e = 0.10). But the key fallback, fitted to noise's non-flat global profile, gives one pitch class 0.15: a confident low score caused by fallback.

### Real songs (`method_comparison.md`)

| | production (calibrated) | raw chroma | rank chroma | chord_hard | chord_dist | **chord_dist_key (H_inst)** | H_ctx |
|---|---|---|---|---|---|---|---|
| pitch classes ≥ 0.75 per note | 4.81 | 2.04 | 3.08 | 3.08 | 3.20 | **3.31** | 3.35 |
| pitch classes ≥ 0.5 | 5.50 | 3.71 | 5.97 | 6.63 | 6.53 | **6.69** | 7.92 |
| notes with ≥ 10 pcs ≥ 0.5 | 0% | 3.3% | 0% | 0% | 0% | **0%** | 6.4% |
| values that are graded, in (0.25, 0.75) | 12% | 36% | 48% | 51% | 50% | **51%** | 56% |
| \|ΔH\| under CENS noise (mean / p95) | 0.015 / 0.094 | 0.012 / 0.049 | 0.017 / 0.068 | — | 0.007 / 0.035 (post hoc) | **0.005 / 0.024** | — |
| \|ΔH\| under ±50 ms (mean / p95) | 0.014 / 0.097 | 0.017 / 0.062 | 0.020 / 0.073 | — | 0.011 / 0.057 (post hoc) | **0.009 / 0.049** | — |
| *secondary:* mean H sung / alternatives | 0.66 / 0.44 | 0.46 / 0.35 | 0.61 / 0.49 | 0.68 / 0.51 | 0.68 / 0.51 | 0.69 / 0.53 | 0.75 / 0.57 |
| *secondary:* sung-vs-alternative win rate | 0.628 | 0.607 | 0.607 | 0.628 | 0.640 | 0.650 | 0.666 |

- **Breadth and specificity.** A typical note has about 3 pitch classes at chord-tone level and about 6.7 at or above 0.5, and *no* note has everything high.
- **Production is not graded.** It saturates (only 12% of values are intermediate), and its 4.8 "≥ 0.75" classes mostly reflect clipping at 1.0.
- **Stability.** The chord-aware score is 2–3× more stable than production under chroma noise and timing shifts.
- **Sung-note tiers under the top chord:** chord tone 41.1%, extension 33.1%, tension 15.1%, out 10.8%.
- **Duration:** mean H_inst rises from 0.58 for notes under 0.1 s to 0.81 for notes of 0.8 s or longer. Production rises from 0.52 to 0.78 over the same range.
- **50 vs 13:** the 13 excluded songs score slightly higher (0.73 vs 0.68 H_inst), the same direction as experiment 1.
- **Chord sequences on real songs** (beat estimator):
  - 0.53 switches per beat and 0.06 A→B→A flickers per beat
  - 88% of chords diatonic to the estimated key
  - 90% agreement under a 50 ms beat-grid shift
  - quality mix: maj 23%, power 21%, min7 14%, maj7 14%, min 13%, sus 12%, dom7 2%
  
  The large power and sus shares are the estimator's "partial match" classes.
- **Ablations barely move real-song aggregates.** Changing the template, temporal estimator or vocabulary moves mean H and the win rate by ≤ 0.013. The exception is the maj/min + 7ths vocabulary, which raises pitch classes ≥ 0.75 from 3.31 to 3.52. The beat HMM halves switching (0.18/beat) but raises the no-chord share to 49%.
- **No sign of bleed-driven circularity.** Among notes whose top chord is a 7th chord, the sung note is its 7th 11.6% of the time, against 11.0% root, 12.7% third and 15.4% fifth.
- **Runtime:** 0.03 s per song (0.008 s per minute of audio) for the posterior, smoothing, all note windows and context, excluding Demucs and CREPE.

### The biggest problem: evidence calibration does not transfer to real stems

- **29.1% of sung notes have e < 0.5**, and 40% have key weight > 0.2. For those notes the score is mostly the *key* estimate.
- **The cause** (post hoc): the best template correlation on real beat segments has a median of 0.856, and 31% are below θ = 0.8. On degraded synthetic chords only 0.5% are.
- **Re-fitting on degraded synthetic audio does not help:** θ stays at 0.8, and 27% of notes remain low-evidence.
- **The consequence:** every `chord_tone_low` and `extension_low` example in `failure_analysis.md` has e ≈ 0. A plausible local chord (e.g. Gmaj7 at 0.91 renormalized) is ignored, and the key, which is only 60% accurate even on synthetic data, assigns a chord tone about 0.01.
- **Fallback variants** (post hoc):

| fallback | mean sung / alternatives | win rate | noise \|ΔH\| | coverage |
|---|---|---|---|---|
| key (pre-registered) | 0.69 / 0.53 | 0.650 | 0.005 | — |
| chord-only (`H_chord`) | 0.68 / 0.51 | 0.640 | 0.007 | — |
| neutral 0.5 | 0.63 / 0.51 | 0.640 | — | — |
| abstain at e < 0.5 | — | — | — | 71% |

  The aggregates are nearly the same. They differ in *failure character*: key fallback creates confident lows from a weak key estimate, while chord-only trusts the best partial template match.

### Context (`H_ctx`)

| rule | sung notes | per alternative | ratio |
|---|---|---|---|
| PASSING | 623 | 173 | 3.6× |
| NEIGHBOR | 1,088 | 574 | 1.9× |
| SUSPENSION | 24 | 14 | |
| STEP_RESOLUTION | 2,011 | 2,063 | 1.0× |
| ANTICIPATION | 153 | 177 | 0.9× |
| CROSSES_CHANGE | 2,193 | 2,159 | 1.0× |

- **The full rule set is barely specific.** It lifts 19.1% of sung notes and 16.2% of counterfactual alternatives, and raises the share of notes where ≥ 10 pitch classes are acceptable from 0% to 6.4%.
- **PASSING and NEIGHBOR are specific.** STEP_RESOLUTION, ANTICIPATION and CROSSES_CHANGE lift alternatives as often as sung notes, so they are forgiveness, not detection.
- **Restricted to PASSING + NEIGHBOR** (post hoc): 5.4% of sung notes lifted vs 2.3% of alternatives (2.3×), and 1.1% of notes with ≥ 10 high pitch classes.
- **CROSSES_CHANGE's real-song extremes are 4–15 s v2 notes.** These are held or merged segments spanning several chords, where averaging dilutes the posterior.

## Answers

1. **Useful chord information from saved CENS?** Partly. On synthetic audio through the same chroma path, chords are recovered almost perfectly, and beat-level sequences on real songs are mostly diatonic (88%) and stable under a 50 ms grid shift (90%). But real-stem chroma matches *any* template much less well than synthetic chroma, and accuracy on real songs is **unmeasured**, because no chord annotations exist. The real-song chord *quality* mix is heavy on power, sus and 7th "partial matches".
2. **Hard label or distribution?** A distribution, in principle: on degraded synthetic audio top-3 is about 1.00 where top-1 is 0.60–0.77, so the correct chord is usually among the leading hypotheses. In practice the fitted posterior is too sharp on real songs (median top-1 mass 0.95), so `chord_dist` behaves almost like `chord_hard` (real-song metrics within 0.01). The temperature needs calibrating on real stems.
3. **Which chord types are recognizable?** On synthetic audio, all are recognizable when clean. Under degradation, 7th chords (0.99–1.00), aug and sus (≈ 0.93) are robust. Triads are fragile (maj 0.77, min 0.69) and absorb into the 7th chord containing any stray tone, and power chords (0.60) become sus. sus2/sus4 and the inversions of aug are **structurally indistinguishable** by chroma and must be merged. Power chords act as a catch-all for weak evidence on real songs.
4. **Scoring tones and extensions.** Use quality-specific ordinal tiers: chord tone > stable extension (9/13 over major, 9/11/b7 over minor, 9/13 over dominant) > context-dependent tension (avoid 11 over major and dominant, altered dominant tensions, blue notes, the third over sus) > clash. The tiers did what was intended in every synthetic case (extensions 0.67, avoid notes 0.33, clashes 0). Their *numeric spacing* is ordinal. The alternative spacing (1, .75, .4, .1) raised every mean (sung 0.74) without changing the tier order or the win rate (0.650), so the spacing is a calibration choice that needs human data.
5. **Does temporal smoothing help?** Beat-synchronous aggregation helps: it fixes CENS-blurred boundaries (0.44 → 0.97 accuracy within 0.1 s of a change) and gives the best overall accuracy. An HMM on top **masks legitimate one-beat changes** (1-beat accuracy 0.92 → 0.80, change recall 0.85) and, on real songs, gets stuck in no-chord (49%). Frame-level HMM smoothing helped only slightly. Recommendation: beat segmentation, no HMM.
6. **Does melodic context separate intentional tension from unsupported notes?** Only two rules do. Passing and neighbor tones are recognized about 2–4× more often for sung notes than for counterfactuals. The resolution, anticipation and crossing rules forgive alternatives as readily as sung notes, so they mainly forgive. Keep context as a separate, limited output.
7. **Can several pitch classes score high?** Yes: about 3.3 per note at ≥ 0.75 and 6.7 at ≥ 0.5; over a C major chord C, E and G all score 1.0.
8. **Stronger vs weaker relationships?** Yes. 51% of values are intermediate (production: 12%), and the synthetic cases show chord tones > extensions > tensions > clashes.
9. **Does it avoid scoring everything high?** Yes without context (0% of notes with ≥ 10 pitch classes ≥ 0.5). The full context rule set pushes that to 6.4%, which is why it should be restricted.
10. **Compared with the direct chroma approach:**
    - **More graded:** production saturates, so most values are 0 or 1; the chord-aware score is not.
    - **2–3× more stable** under noise and timing shifts.
    - **Musically sensible in controlled cases:** production scores the extensions of a C chord 0 / 0 / 0.91 and passing tones 0.
    - **Interpretable:** every score decomposes into chord hypotheses, tiers, evidence and context codes.
    
    On the secondary original-note diagnostics it is about equal to slightly better: win rate 0.650 vs 0.628. It does **not** fix the P4/P5 gap (sung minus a fourth up is still 0.05). Under the new objective that is largely expected, because a fourth or fifth away from a chord tone is often another chord tone or an extension. Many such substitutions really are compatible.
11. **Failure cases across the 63 songs** (`failure_analysis.md`):
    - **Key-fallback lows when evidence is low** (29% of notes are low-evidence); this drives every `chord_tone_low` (0.4%) and `extension_low` (0.4%) example.
    - **Very long v2 notes** (4–15 s, e.g. `jeff_buckley_hallelujah` 371.6–386.8 s) averaged across many chords.
    - **Short notes:** 3.6% of notes are under 0.15 s and score below 1/3, mostly confident "out" values such as C# over C in `bruno_mars_when_i_was_your_man` at 23.38 s. These could be scoops, grace notes or segmentation errors.
    - **Power/sus partial matches** standing in for unclear harmony.
    - **The weakest songs are dense-harmony ballads and standards** (`at_last` 0.62, `luther_vandross_here_and_now` 0.62, `jennifer_hudson_spotlight` 0.63). The strongest are simple and loop-based (`jingle_bells` 0.82, `sunflower` 0.79).
    - **Noise instability is rare:** 2 notes with |ΔH| ≥ 0.25.
12. **Low-confidence harmony:** report it, and don't fold it silently into the score. Evidence `e` is the only uncertainty signal that works here; the ambiguity flag (top-1 mass) almost never fires because the posterior is overconfident. The key fallback should not be the default: the key estimate is only 60% accurate even on synthetic data and produces confident lows. Use `H_chord` (the best available harmony) plus an explicit evidence/confidence value, and decide whether to abstain once evidence is calibrated on real stems. At the current θ, abstention would drop 29% of notes.
13. **Evidence still missing:**
    - human graded-compatibility judgments for the score and the tier spacing
    - chord annotations on real separated stems, to measure chord accuracy and calibrate τ and θ where they will be used
    - a song set not used for development
    - validation of the context rules against annotated non-chord tones
    - the audio-dependent checks: unsmoothed chroma, vocal bleed, the a cappella guard
14. **Recommended next Axis 1 candidate** (deterministic, about 0.03 s per song):
    - beat-synchronous Pearson matching against binary templates of the 112 merged chord sets plus N
    - quality-specific ordinal tiers
    - **H = expected tier under the chord distribution** (`chord_dist`), reported **with** evidence `e` and the top chord hypotheses as reason codes
    - no HMM, and no default key fallback
    - a separate `H_ctx` limited to PASSING and NEIGHBOR
    - notes over about 2 beats scored per beat segment (duration-weighted) instead of over one averaged window
    - notes under 0.1 s flagged rather than scored at full weight
    
    This is the smallest version that kept every synthetic behavior that passed.
15. **What to evaluate before finalizing Axis 1:**
    1. **Chord annotation of 8–10 real songs at beat level** (public chord charts, verified by ear, no audio processing needed). This measures real-stem chord accuracy and calibrates τ, θ and the evidence threshold where they are used, instead of on synthetic audio.
    2. **A human-labelled pilot:** 300–500 sung notes *and* counterfactual pitch classes, rated fit / color / tension-that-works / clash / unclear. This sets the tier spacing and the 0–100 mapping, and tests whether H tracks perception.
    3. **Re-run this harness on the chosen variant**, with the long-note and short-note policies, and report it on a song set not used here.
    
    Only after (1) and (2) should a 0–100 curve be frozen.

## Calibration note (0–100)

`100·H` already has a consistent meaning: the expected ordinal tier (100 = chord tone, 67 = stable extension, 33 = tension, 0 = clash), independent of song. But the perceptual distance between tiers is unknown, and the observed means (sung notes 69, alternatives 53) are **not** a target. No thresholds were chosen to make released recordings look good. Perceptual calibration cannot be established without human annotations.

Production scoring, the frontend, Axis 2 and the feature store were not modified.
