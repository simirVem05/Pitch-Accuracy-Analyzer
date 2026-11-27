# Axis 1 Experiment 2 — chord-aware harmonic compatibility: protocol

Written 2026-11-27, **before** any code in this directory was run. Its scope is method development on the full 63-song corpus: no held-out split exists, and in-corpus comparisons are **not** evidence of generalization.

## Objective

For a sung note and its instrumental context, compute a compatibility vector `H[0..11] ∈ [0,1]`, one entry per pitch class. It answers "how harmonically compatible is this pitch class with the accompaniment at this moment". It is not "which note did the artist choose", and not the probability that a note is correct. Several entries may be high at once. The sung note's score is `H[pc]`.

## Data

- `data/main_data/<song>/notes.npz` (v2) and `harmony.npz`, read-only, for all 63 songs (31,803 notes).
- **Inputs used:**
  - `other_chroma_cens`, `chroma_frame_times_s`, `global_chroma_profile`, `beat_times_s` and `estimated_tempo_bpm`
  - note times, pitch classes, target MIDI and contours (contours for diagnostics only)
  - `other_chroma_rank`, only for the production comparison
- **Harmony is estimated from the instrumental chroma only.** Vocal note choices never enter chord or key estimation.

## Parameter discipline

**No parameter is fitted to the 63 real songs.** Free parameters are fitted on **synthetic audio** rendered through the exact production chroma call (`librosa.feature.chroma_cens`, sr 22050, hop 512, tuning from `estimate_tuning`):

- Synthetic set A (calibration) and set B (evaluation) use disjoint random voicings and timbres.
- Synthetic set C (progressions) chooses the temporal model.

Musically motivated constants (tiers, smoothing prior, context thresholds) are fixed below with their rationale. Real-song results are reported for the configuration chosen on synthetic data. Any change made after seeing real-song results is labelled post hoc in `summary.md`.

## Step A — harmony estimation

**Vocabulary.** 11 qualities × 12 roots, intervals above the root:

| quality | intervals |
|---|---|
| maj | 0 4 7 |
| min | 0 3 7 |
| dim | 0 3 6 |
| aug | 0 4 8 |
| sus2 | 0 2 7 |
| sus4 | 0 5 7 |
| power | 0 7 |
| dom7 | 0 4 7 10 |
| maj7 | 0 4 7 11 |
| min7 | 0 3 7 10 |
| hdim7 | 0 3 6 10 |

Duplicate pitch sets cannot be told apart by chroma (sus2(X) = sus4(X+7); aug is symmetric), so they are merged into one template. That gives 112 distinct chord sets plus **N** (no chord). The compatibility profile of a merged set is the mean of its interpretations.

**Templates, chosen on set A/B by top-1 pitch-set accuracy:**

- **binary:** unit-norm indicator of the chord tones.
- **harmonic:** each chord tone also contributes its partials 1–6 with weight 0.6^(k−1), folded to chroma.

**Similarity, chosen by the same rule:** cosine or Pearson correlation between the window chroma and the template.

**Posterior:** `p(c) ∝ exp(s_c/τ)` over the chords, plus N with the fixed score θ. τ and θ minimize the log-loss on set A. Set A includes N frames: white noise, pink noise and 6-note clusters. A soft distribution is kept throughout, and a hard label is only an `argmax` of it.

**Temporal estimators** (one is chosen on set C by mean frame accuracy averaged over 1-, 2- and 4-beat chord durations, so an estimator that masks fast changes loses):

- `frame`: per-frame posterior, no smoothing
- `frame_hmm`: forward–backward over frames. Self-transition `1 − 1/D`, where D = 4 beats of frames (one bar, a musical prior); switches are uniform.
- `beat`: per beat-segment posterior, from the mean CENS between consecutive saved beats
- `beat_hmm`: forward–backward over beat segments, self-transition 0.75 (expected duration 4 beats)

**The posterior for a note window** is the frame posteriors averaged over the production note window `[i0, i0+width)`, or the overlap-weighted beat-segment posteriors.

**Real-song diagnostics** (no labels exist):
- top-1 mass and entropy
- switches per beat, and A→B→A flicker within 2 beats
- diatonic share (the top chord's tones are all inside the estimated key's scale)
- quality distribution
- agreement under CENS noise and a 50 ms beat-grid shift

## Step B — compatibility vector

**Chord tiers.** For chord c, each pitch class gets one of four ordinal tiers by its interval above the root. They are scored on an ordinal scale, **T1 = 1, T2 = 2/3, T3 = 1/3, T4 = 0**:

- **T1** chord tone
- **T2** stable extension or color tone
- **T3** tension: an avoid note, an altered tension or a blue note — usable with context
- **T4** clash with no stylistic license

The spacing is ordinal, not perceptual. A sensitivity variant uses (1, 0.75, 0.4, 0.1).

| quality | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| maj | 0 4 7 | 2 9 11 | 3 5 10 | 1 6 8 |
| min | 0 3 7 | 2 5 10 | 6 8 9 11 | 1 4 |
| dim | 0 3 6 | 9 10 | 2 5 8 | 1 4 7 11 |
| aug | 0 4 8 | 2 6 10 | 11 | 1 3 5 7 9 |
| sus2 | 0 2 7 | 4 5 9 10 | 3 11 | 1 6 8 |
| sus4 | 0 5 7 | 2 9 10 | 3 4 11 | 1 6 8 |
| power | 0 7 | 2 3 4 5 10 | 8 9 11 | 1 6 |
| dom7 | 0 4 7 10 | 2 9 | 1 3 5 6 8 | 11 |
| maj7 | 0 4 7 11 | 2 6 9 | 3 5 | 1 8 10 |
| min7 | 0 3 7 10 | 2 5 9 | 6 8 | 1 4 11 |
| hdim7 | 0 3 6 10 | 2 5 8 | 1 | 4 7 9 11 |

Rationale, by chord-scale practice in pop, R&B and jazz:
- **Major:** add9, 6 and maj7 are standard colors. The natural 11 is the textbook avoid note (a semitone above the 3rd), usable as a suspension. b3 and b7 are blue notes over major in R&B and blues. b9, #11 and b13 rub a semitone against the root or fifth with no license over a plain triad.
- **Minor:** 9, 11 and b7 are the minor-pentatonic, dorian and aeolian colors (m9, m11, m7). b5 is a blue note. b6, 6 and maj7 are modal colors. b9 and the major 3rd clash.
- **Dominant 7th:** 9 and 13 are natural tensions. b9, #9, #11 and b13 are altered tensions (legitimate tension). 11 is an avoid note, and the major 7th clashes with the b7.
- **Sus chords:** the 3rd is the resolution target and clashes with the 4th, so it is T3, not T4.
- **Power chord:** with no third, both thirds and the pentatonic colors are T2.
- **Diminished and augmented:** follow the diminished and whole-tone scales.

**Chord-part score:** `H_chord(pc) = Σ_c p(c)·tier_c(pc) / (1 − p(N))` — the expected tier.

**Key context.** Each of the 24 keys is scored by its Krumhansl–Kessler profile's correlation with the global CENS profile. The key posterior is `softmax(r/τ_key)`, with τ_key minimizing the key log-loss on set C. Key tiers, relative to the tonic:
- major key: T1 = 0 4 7, T2 = 2 5 9 11, T3 = 3 6 10 (blue), T4 = 1 8
- minor key: T1 = 0 3 7, T2 = 2 5 8 10 11, T3 = 6 9, T4 = 1 4

`H_key` is the expected key tier.

**Evidence-gated combination:** `H_inst = e·H_chord + (1 − e)·H_key`, where `e = 1 − p(N)` over the note window. Global context fills in only to the extent that local evidence is missing, so it cannot override clear local harmony. A fixed-blend ablation uses `0.75·H_chord + 0.25·H_key`.

**Uncertainty output:** `e`, the top-1 chord mass and the entropy are reported for every note. The flag `LOW_EVIDENCE` is set when e < 0.5 and `AMBIGUOUS_CHORD` when top-1 mass < 0.35. An abstention analysis treats LOW_EVIDENCE notes as unscored.

## Step C — melodic and temporal context (deterministic; heuristics, not ground truth)

**Definitions:**
- **Neighbors:** the previous and next v2 notes, if the gap is ≤ 0.30 s.
- **Step:** |ΔMIDI| ∈ {1, 2}.
- **Supported:** H_inst ≥ 2/3.
- **Candidate:** a note with H_inst < 1/2.

**Patterns** (first match wins, in this order):

| Code | Rule |
|---|---|
| `PASSING` | dur ≤ 0.35 s; prev and next both supported; prev→note and note→next are steps in the same direction |
| `NEIGHBOR` | dur ≤ 0.35 s; prev and next supported; steps away from prev and back to prev's pitch class |
| `SUSPENSION` | dur ≤ 1.0 s; next is supported and a step below; the note's pitch class is supported by the harmony in the 0.5 s before its onset |
| `STEP_RESOLUTION` | as SUSPENSION, but without prior support, and the step may be up or down (appoggiatura-like) |
| `ANTICIPATION` | dur ≤ 0.6 s; the pitch class is supported by the harmony in the 0.5 s after the note ends |

**Effect.** A recognized pattern sets `H_ctx = max(H_inst, 2/3)` — "resolved tension counts like an available extension". Separately, a note whose first and second halves have different top chords is flagged `CROSSES_CHANGE`, and gets `H_ctx = max(H_ctx, H_half1, H_half2)`. `SHORT` (dur < 0.10 s) is a reliability flag only.

**Specificity check.** The same rules are applied to every counterfactual pitch class in each note's place (nearest MIDI with that pitch class). If alternatives are lifted about as often as sung notes, context is only forgiving, and that is reported as such.

## Step D — methods compared (each yields a 12-vector per note)

| ID | method |
|---|---|
| `prod_calibrated` | production `key_compliance` (`rank_local_slack^0.75·global^0.25` → clip) |
| `prod_harmonic_rank` | the pre-calibration production value |
| `raw_chroma` | window-mean CENS / max over the 12 (span only) |
| `rank_chroma` | the production per-frame rank mean (span only) |
| `chord_hard` | tier under the argmax chord of the note-window posterior |
| `chord_dist` | `H_chord` (expected tier) |
| `chord_dist_key` | `H_inst` (evidence-gated) |
| `chord_dist_key_ctx` | `H_ctx` |

**Ablations** (on `chord_dist_key`): each temporal estimator; vocabulary limited to triads (maj/min), to maj/min + 7ths, and full; binary vs harmonic templates; tier spacing; key gating vs fixed blend vs none.

## Step E — synthetic tests (intended behavior fixed now)

Two layers, reported separately:
- **(T) template level:** idealized chroma injected directly. Tests only the scoring rules.
- **(S) audio level:** synthetic audio → production CENS → full pipeline. Also tests harmony estimation.

Melody notes are symbolic (pitch class + times), as with real vocals over the `other` stem. In the table, "chord tones high" means each ≥ 0.75.

| # | case | intended behavior (pass criterion) |
|---|---|---|
| 1 | sustained C major | C, E, G each ≥ 0.75; min(chord tones) > max(T3 ∪ T4 pcs); D, A, B (extensions) in [0.4, 0.95) and each > C#, F#, G# |
| 2 | sustained A minor | A, C, E ≥ 0.75; min(chord tones) > max(T4 pcs) |
| 3 | G7 | G, B, D, F ≥ 0.75 |
| 4 | Cmaj7 | C, E, G, B ≥ 0.75; F (avoid 11) < D (9) |
| 5 | Csus4 | C, F, G ≥ 0.75; E (3rd) ≥ 0.25 (tension, not "out") |
| 6 | C–Am–F–G, 4 beats each, 100 BPM | interior-frame top-chord accuracy ≥ 0.8; within each segment that chord's tones ≥ 0.75 at the segment's centre |
| 7 | E held across C→F (chord tone → maj7) | `CROSSES_CHANGE` set; H_ctx ≥ max(H_half1, H_half2) − 1e-9 |
| 8 | octave and voicing | the same pitch class at MIDI 52/64/76 gives identical H (exact). C major in 3 voicings: chord tones ≥ 0.75 in each; max \|ΔH\| across voicings ≤ 0.15 |
| 9 | extensions | over C: D, A ≥ 0.4. Over G7: A ≥ 0.4. Over C: F < D |
| 10 | passing tone E–F–G over C (F 0.15 s) | F: H_inst < 0.5 and H_ctx ≥ 2/3 with code PASSING |
| 11 | 4–3 suspension: F held from an F chord into C, then E | the F note's code is SUSPENSION and H_ctx ≥ 2/3 |
| 11b | the same, with F re-articulated at the chord change (sung F over F, then F over C, then E). Added during implementation, before any run | the second F's code is SUSPENSION, or its H_inst ≥ 2/3 (still supported because of smoothing — reported); H_ctx ≥ 2/3 |
| 12 | unresolved C# over C, 1.5 s, then a leap to A | H_inst ≤ 1/3 and H_ctx = H_inst (no code) |
| 13a | white noise only | LOW_EVIDENCE set, and no pitch class < 0.25 (no confident low score) |
| 13b | single sustained pitch C | C ≥ 0.75; E and Eb both ≥ 0.4; AMBIGUOUS_CHORD set |
| 13c | C5 power chord | C, G ≥ 0.75; E and Eb both ≥ 0.5 |
| 14 | specificity over cases 1–5 | in each: ≤ 7 pitch classes ≥ 0.5 and ≥ 2 pitch classes ≤ 1/3 |
| 15 | timing: note shifted ±50 ms inside a sustained chord | \|ΔH\| ≤ 0.05 for all 12 |
| 16 | C major + white noise at 10 dB SNR | C, E, G ≥ 0.7 |

Production's scores are shown alongside each case for reference. They are not pass/fail.

## Step F — real songs (all 63; exploratory)

**Per note:** song, index, times, pitch class, top-3 chords with probabilities, e, entropy, `H_inst[0..11]`, H_inst and H_ctx of the sung note, production's score, context and reason codes. The table goes to `data/experiment_archives/axis1_chord_compatibility/`.

**Evaluation criteria** (none is "rank the original note first"):

| criterion | measure |
|---|---|
| multiple compatibility | pitch classes per note with H ≥ 0.75 |
| specificity | pitch classes with H ≥ 0.5; share of notes with ≥ 10 such ("nearly everything fits"); max − min spread |
| grading | share of the 12 values in (0.25, 0.75) |
| stability | mean \|ΔH\| under multiplicative CENS noise (lognormal σ = 0.1, re-normalized) and ±50 ms note shifts |
| chord stability | switches per beat, flicker |
| abstention | LOW_EVIDENCE share |
| failure categories | a chord tone of the top chord with H < 0.5; breadth ≥ 10; T2 of the top chord with H < 1/3; `CROSSES_CHANGE` share; short notes with H_ctx < 1/3 |

**Practicality:** per-song runtime of the chosen pipeline.

**Secondary diagnostics only:** the sung note's mean H against the alternatives' mean, the pairwise win rate, the P4/P5 gap and transposition placement.

## Calibration (to 0–100)

No curve is frozen. H is an **expected ordinal tier**: H = 0.8 means "on average, between a chord tone and a stable extension under the estimated harmony". That reading is the same in every song, because tiers do not depend on the song. Perceptual spacing between tiers is unknown without human labels.
