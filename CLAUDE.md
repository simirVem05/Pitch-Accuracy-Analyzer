# CLAUDE.md

Technical orientation for working in this repo. `README.md` is the high-level overview; this file is the internals. `docs/RESEARCH.md` is the evidence base — when this file and it disagree, it wins, and this file should be corrected.

## What the system is

A two-axis vocal analyzer. Given a full mixed song, it separates the stems, derives the harmony from the instrumental, tracks the vocal f0, groups frames into notes, and scores each note twice:

| Axis | Question | Reference | State |
|---|---|---|---|
| 1 — harmonic fit | Do the chosen notes fit the music? | CENS chroma of the `other` stem, per frame | built, measured |
| 2 — intonation precision | Were those notes sung cleanly? | cents from a tuning-corrected 12-TET target | Phase A in production; Phase B in offline research only (dataset, baselines, first models — nothing shipped) |

**The axes are never averaged.** That is the single most important invariant in the project. Averaging recreates the exact problem the redesign existed to fix: a deliberate ♭5 sung dead-center is low Axis 1 and excellent Axis 2, and averages to "mediocre," which describes nothing. A singer who picked safe notes and sang them all 30 cents flat averages to the same value. The entire diagnostic content is in the split. The axes also differ in reliability — Axis 1's published ceiling is r ≈ 0.611 — so averaging contaminates the firmer number with the softer one.

A third number, **coverage**, is load-bearing rather than cosmetic. The `vocals` stem contains lead plus backing plus harmonies plus doubles, and CREPE is monophonic, so it can follow whichever voice is loudest and jump between lead and harmony mid-phrase. The chosen mitigation is to confidence-gate hard and drop ambiguous frames — score fewer notes rather than score them wrong — which is only honest if the user sees how much was skipped.

## Repo layout

```
backend/           pipeline modules (9 files, ~1,600 lines)
  separation.py      Demucs → Stems
  harmony.py         tuning, CENS chroma, beats, Axis 1 scoring
  preprocess.py      CREPE f0, voicing mask, coverage
  note_segmentation.py  v1: frames → note segments, tuning-aware targets (production)
  note_segmentation_v2.py  v2 segmentation — offline only, builds the canonical notes
  scoring.py         Axis 2 scoring, chart point construction
  main.py            orchestration, metric aggregation, Gemini, CLI
  api.py             FastAPI: POST /analyze, GET /health
  evaluate.py        label-free evaluation harness
frontend/          React 19 + Vite + Tailwind + Recharts SPA
scripts/
  build_main_dataset.py      audio → feature store (schema 1, v1 notes)
  main_data_io.py            shared audio-free note arrays, atomic writers, validation
  resegment_main_dataset.py  harmony.npz → notes (stage/compare-v1/backup/promote/validate)
  build_axis2_ml_dataset.py  canonical notes → synthetic-corruption ML dataset
data/              (gitignored)
  song_mp3s/         source audio — currently EMPTY, see below
  main_data/         canonical feature store, 63 songs + manifest.json, schema 2
  dataset_selections.json   which songs may be used for what, plus Axis 2 splits
  ml_dataset/        axis2_v1/ (v1 notes), axis2_v2/ (v2 notes)
  backups/           verified pre-migration copy of main_data (schema 1)
  migration_v2/      staging dirs and logs from the v2 migration
ml/
  audits/            segmentation tail, stable center, segmentation v2 sweep, v2 migration records
  baselines/         zero, median, core-median, Ridge; v1-vs-v2 comparison
  experiments/       natural-offset feature model (HGB), sequence model (TCN)
  results/           baseline outputs for axis2_v1 and axis2_v2
docs/              RESEARCH.md only (tracked)
```

`data/` is gitignored (`.gitignore:13`), so the feature store and ML datasets are **not under version control**. `docs/` is *not* ignored; `docs/RESEARCH.md` is tracked.

**The source audio is gone.** `data/song_mp3s/` is empty, and the `evaluate.py` stem cache (`/tmp/pitch_eval_cache`) is empty too. Until audio is restored, `evaluate.py` and `build_main_dataset.py` have nothing to run on, and `harmony.npz` (with the full saved vocal tracks) is the only copy of the extracted signal. Everything after the original extraction has been regenerated from it without audio.

## Architecture: one measurement core, four consumers

```
             separation.py  →  Stems
             harmony.py     →  HarmonicContext
             preprocess.py  →  PitchTrack
             note_segmentation.py → segments (list of dicts)
                        │
   ┌────────────┬───────┴────────┬──────────────────┐
   ▼            ▼                ▼                  ▼
main.py      api.py         evaluate.py    build_main_dataset.py
 (CLI)     (HTTP+CORS)     (validation)   (offline features)
```

All four entry points call the *same* primitives. This is deliberate and worth preserving: it is why the audio dataset builder cannot drift from production behavior, and why `evaluate.py` measures the configuration that actually ships. `build_main_dataset.py` imports from `backend/` by injecting it onto `sys.path` (`scripts/build_main_dataset.py:22-23`) rather than duplicating any DSP.

**One deliberate exception: note segmentation.** Production (`main.py`, `api.py`) and `evaluate.py` still segment with **v1** (`note_segmentation.py`). The canonical feature store's `notes.npz` and the `axis2_v2` ML dataset were re-segmented with **v2** (`note_segmentation_v2.py`, `CANONICAL_CONFIG`) by `scripts/resegment_main_dataset.py`, working from `harmony.npz` with no audio. So the notes the Axis 2 research trains on are not the notes production scores. v2 has not been evaluated for Axis 1, and moving production to it would change Axis 1 inputs too.

### Three cross-module contracts

Breaking any of these breaks things far from the edit site.

1. **`NaN` means unvoiced.** `preprocess.py` masks the f0 array rather than returning a separate boolean, and every downstream consumer tests `np.isnan` / `np.isfinite`. Established at `preprocess.py:153-154`, relied on in `note_segmentation.py:109` and in the voiced masks `build_main_dataset.py:118` and `:122` derive. v2 segmentation reads the same NaN-masked f0 back out of `harmony.npz`.

2. **`segments` is a list of plain dicts, mutated in place by scorers.** `score_key_compliance` and `score_intonation` add keys rather than returning new objects. This is what lets `evaluate.py` shallow-copy a segment, swap one integer, and re-score for nearly free — see `_mean_rank` at `evaluate.py:83`. Converting segments to a frozen dataclass would be tidier and would break the harness.

3. **`None` means unmeasured, never zero.** When there is no usable harmony, `key_compliance` is `None`, and that propagates coherently through four layers: `harmony.py:215-218` → `compute_metrics` skips those notes and reports `None` → `build_gemini_prompt` switches to a branch that states the axis was not measured → `MetricsPanel.jsx` renders "—". Never coerce it to 0.0; a zero would read as "every note was wrong."

## Axis 1, precisely (`harmony.py`)

For each note segment with pitch class `pc`:

```
local  = max over 3 window positions of mean(rank[pc, window])
             positions: the note's own span, that span shifted one beat
             earlier, and that span shifted one beat later
global = rank of pc in the whole-song mean chroma profile

combined     = local^0.75 * global^0.25            # GLOBAL_WEIGHT = 0.25
harmonic_fit = clamp01((combined - 0.50) / (0.75 - 0.50))   # RANK_LOW / RANK_HIGH
```

The reported metric is the **duration-weighted** mean over scorable notes, so a grace note does not count as much as a sustained one (`main.py:33`).

**Exactly two features feed the score**: local chroma and global chroma, both from the `other` stem. No bass, no chord labels, no beat-position weighting, no genre, no key, no allowlist.

Five design decisions here are measured rather than assumed, and each has a failure mode if reverted:

- **Rank, not raw salience** (`_to_rank`, `harmony.py:90`, double-`argsort`). Absolute chroma energy tracks level and arrangement density, so no raw threshold is comparable across sections of one song. Rank roughly doubled the separation between real and transposed notes.

- **`other` only, never `other + bass`.** Bass is 2.8–4.6× louder, so summing waveforms lets root notes dominate and drown out the voicings that distinguish one harmony from another. Measured: margin +0.043 → +0.014, and the true key stops ranking first on half the songs. Note that `Stems.harmony` (`separation.py:26`) still sums them — it exists **only** for `evaluate.py`'s bleed test. Production passes `stems.other` (`main.py:150`). Do not "fix" the inconsistency by wiring `.harmony` into production.

- **Slack shifts a fixed-width window; it does not widen it** (`harmony.py:238`). This was a real bug once. Widening dilutes a short note into beats of unrelated harmony and lifts every pitch class equally, forgiving wrong notes exactly as much as anticipated ones. Shifting asks the intended question — was this note supported by the chord just before or just after?

- **Geometric mean, not a weighted sum** (`harmony.py:247-249`). Local support alone cannot separate a note outside the key from an in-key note over a chord that does not contain it (a passing tone or suspension, musically fine). A genuinely wrong note is weak on *both* terms; a passing tone is strong on one. Geometric so the terms **gate** rather than compensate — a sum would let high key membership drag a locally-rejected note up linearly. `GLOBAL_WEIGHT` anywhere in 0.20–0.30 performs the same, so 0.25 is mid-plateau, not a fitted peak.

- **Octave is ignored by design**, permanently (a reversed earlier decision). Harmony is octave-invariant: a chord is a set of pitch classes, so a pitch class supported by the sounding harmony is supported in every octave. The only octave reference available is whichever one the original artist sang in, a property of their range rather than of the music, so checking it would penalize every singer whose range differs — which is most of them, and would silently turn a note-choice score into an unrequested voice-type judgment. Describe harmonic fit as a *pitch-class* measure, not as "octave-blind."

`RANK_LOW = 0.50` and `RANK_HIGH = 0.75` are a **presentation choice, not a measurement**. The same audio reports 66% or 85% depending on where they sit. Any recalibration must be re-validated with `evaluate.py`, never by eye.

Two escape hatches return `None`: zero chroma frames, and the a cappella guard (`MIN_HARMONY_TO_VOCAL_RMS = 0.04`, `harmony.py:51`). An a cappella upload still yields a nominal `other` stem, but it holds only separation residue of the voice itself — scoring against it compares the vocal to its own leakage and returns a confident, meaningless number (measured at 78% before the guard existed). Real mixes measure 0.29–0.60 on this ratio, so there is an order of magnitude of headroom, but the threshold was validated when chroma used `other+bass` and has **not** been re-confirmed against an actual a cappella since.

### Known failure mode: fourths and fifths

On the 20-song evaluation the true key ranks #1 of 12 on 14 songs. **All 6 failures lost to a perfect fourth or fifth, and to no other interval.** The mechanism: the score is a mean over pitch-class ranks with no harmonic-function awareness, and transposing by a fourth or fifth maps most scale degrees onto other degrees of the same key, so the shifted melody still lands on pitch classes the chromagram rates highly. Diatonic set overlap is maximal at ±5 and ±7 semitones (6 of 7 notes shared), which is why exactly those two shifts are confusable.

The chromagram measures pitch-class **presence**, not **role**. That is why chord estimation, not reweighting, is the plausible fix — a chord label distinguishes I from V where raw pitch-class presence cannot — and why the P4/P5 problem should not be debugged as a chroma-quality problem.

A related design position: scoring does not need discrete chord labels, and surfacing them in the UI would add interpretability plus a new failure mode. That is a statement about *display*, and it does not settle whether chord estimation should replace raw chroma as the scoring feature — which remains the open, and most promising, question.

Two things to keep straight when reading eval numbers. Uniform global transposition is not how singers err, so this result bounds *key identification* and says less about catching an individual wrong note; the perturbation test is the better proxy there and passes 20/20 with every draw positive. And **mean margin is a misleading summary** — it was identical (+0.023) across a 5-song set with 0% failures and a 20-song set with 30%. Always report the placement distribution with it.

## Axis 2, currently (`scoring.py`)

Phase A only: `|cents|` from a tuning-corrected target, measured off the core-trimmed median, mapped through a hand-tuned piecewise curve (`intonation_score_from_cents`, `scoring.py:25`) — linear to 25 ¢, steeper to 45 ¢, exponential decay beyond. The result is clipped to `SCORE_FLOOR = 0.05`, so a catastrophically flat note reads as 5% rather than 0%; keep that in mind when reading a low score or comparing against a hand computation.

Phase A **exists to be the baseline Phase B must beat.** Building the model first would leave nothing to compare against. It is a 12-TET grid and therefore wrong for the target genres by construction: a just major third sits −13.7 ¢ from equal temperament, a just minor third +15.6 ¢, skilled singers sharpen leading tones ~+10 ¢, and blue notes are deliberately microtonal. Measured median deviation on this repo's samples is ~22.5 ¢, of which a just major third alone would account for 13.7. The system cannot currently distinguish "sang an expressive third" from "sang a sloppy third."

Tuning correction is what makes the axis defensible at all. Offsets on real recordings span roughly −16 to +35 cents, and the curve's entire "pro zone" is 0–25 ¢ — so a 16-cent offset would consume 64% of that budget before the singer sings a note, and would apply uniformly, making a perfect performance over a sharp backing track read as consistently sharp. The offset is estimated from the *instrumental* (`estimate_tuning_semitones`, `harmony.py:75`) because fixed-pitch instruments define the grid the singer is aiming at, and it feeds both axes.

**Phase B (planned)** trains on synthetic corruption: take a released vocal, apply a known pitch shift, and the shift amount *is* the label. This is why ML fits Axis 2 and not Axis 1 — for note choice no label exists and part of the target is genuinely undetermined, while here labels are free. Because the target becomes *the released vocal* rather than a grid, expressive practice is baked in: a just third appears at 0-shift throughout training, so the model learns it as the target rather than a 14-cent error.

Two traps recorded before anyone starts:

- **Artifact leakage.** Pitch-shifting leaves resampling and phase-vocoder fingerprints. A model can score perfectly by detecting "this frame was shifted" while understanding nothing about pitch, then collapse on real vocals. Defense: include zero-shift controls that still pass through the full shifting pipeline, so artifacts carry a label of 0 cents and are uninformative. Do this from the first run — retrofitting invalidates every prior result.
- **It makes vibrato not-penalized, not assessable.** The model only ever sees good vibrato at 0-shift and pitch-shifted good vibrato at X-shift, so it learns vibrato is irrelevant to its prediction. Judging vibrato quality needs human labels that do not exist here. Keep vibrato and portamento in the reported-not-judged category.

### Known inconsistency: the metrics panel and the chart disagree

`compute_metrics` averages the **unsmoothed** `intonation_score` (`main.py:59`), while `build_graph_points` plots a **median-smoothed** version over a ±0.5 s window (`SMOOTHING_HALF_WINDOW_S`, `scoring.py:64-71`). The headline number and the line in the chart are therefore computed from different values, and a user can see a dip in the graph that the reported score does not reflect — or vice versa.

This is live, not stale. It is the last surviving item from the original known-bugs list; the others have all been fixed (the bare `except` in the scoring path is gone, CREPE's `activation` is explicitly discarded as `_activation`, `/analyze` no longer blocks the event loop, and metrics are now duration-weighted rather than note-count weighted).

Whichever way it gets resolved, both consumers should read from one source. Smoothing exists to make the chart legible across short notes, which is a presentation concern, so averaging the raw per-note scores and smoothing only for display is the more defensible direction — but it is a genuine choice and changing the metric changes a user-visible number.

## Signal-processing constants that matter

| Where | Constant | Value | Why |
|---|---|---|---|
| `preprocess.py:25` | `conf_threshold` | 0.60 | load-bearing, not cosmetic — gates layered-vocal ambiguity |
| `preprocess.py:19` | highpass | 85 Hz, order 5 | removes rumble below sung range |
| `preprocess.py:28-29` | f0 range | 65–1200 Hz | separation bleed often lands outside it |
| `preprocess.py:31-32` | min run / gap fill | 5 frames / 1 frame | morphological voicing cleanup |
| `note_segmentation.py:9` | `DRIFT_CENTS` | 40.0 | hysteresis threshold for a new note |
| `note_segmentation.py:10` | `TRIGGER_FRAMES` | 3 | needs 4 agreeing frames (~80 ms) to commit |
| `note_segmentation.py:12` | core trim | 0.20 | keeps middle 60%, excluding scoops and releases |
| `scoring.py:11` | `SCORE_FLOOR` | 0.05 | worst possible intonation score is 5%, not 0% |
| `scoring.py:9` | `SMOOTHING_HALF_WINDOW_S` | 0.5 | chart-only median smoothing — see the inconsistency above |
| `scoring.py:8` | `SILENCE_BREAK_S` | 0.1 | gap that inserts a `None` so the chart lifts the pen |
| `harmony.py:22-23` | `RANK_LOW` / `RANK_HIGH` | 0.50 / 0.75 | presentation choice, re-validate if touched |
| `harmony.py:44` | `GLOBAL_WEIGHT` | 0.25 | mid-plateau of 0.20–0.30 |
| `harmony.py:51` | `MIN_HARMONY_TO_VOCAL_RMS` | 0.04 | a cappella guard |

Two segmentation subtleties that were bugs and are now fixed — do not regress them:

- **The target note comes from the core-trimmed median, not the first voiced frame** (`_finalize_segment`, `note_segmentation.py:55`). Onsets are exactly where scoops live, so anchoring there can pick the wrong semitone and bias every deviation measured against it, including the core-trimmed median specifically designed to ignore onsets.
- **Segment boundaries are tracked in voiced-frame positions, not raw frame indices** (`note_segmentation.py:155`). The loop skips `NaN` frames, so tracking raw indices put boundaries in the wrong place whenever unvoiced frames interleaved a transition.

### v1's measured failure modes, and v2 (`note_segmentation_v2.py`)

`ml/audits/axis2_segmentation_tail/` found that **12.1% of v1 notes on the 50 Axis 2 songs are clearly invalid** as single-note units, plus 8.4% probably invalid. The main cause: v1 walks voiced frames by position and **never looks at elapsed time**, so it bridges unvoiced gaps of any length (up to 10.4 s). The material after a gap is usually short or gliding, so the 4-agreeing-frames rule rarely fires and the next fragment gets absorbed into the previous note. CREPE errors were rare in that tail.

v2 keeps v1's inputs, voiced-only frames, the 3-frame minimum and the output dict schema, and changes the boundary rules. `CANONICAL_CONFIG` (`mode="gap_v1"`, `hard_gap_s=0.5`, `separation_cents=100`, `gap_before="local"`, `gap_context_frames=10`, `target_estimator="core_median"`) splits unconditionally at any gap over 0.5 s, applies a short-gap rule comparing the local pitch either side of a gap, and otherwise falls back to v1's exact-semitone commit. The `SegmentationV2Config` dataclass defaults are the sweep's *base* configuration, **not** the canonical one; they are left unchanged so every recorded sweep result still reproduces. Use `CANONICAL_CONFIG`.

On the 50 songs (`ml/audits/axis2_segmentation_v2/segmentation_v2_summary.md`): clear-invalid 12.06% → 0.92%, notes bridging a gap over 0.5 s 1,024 → 0, 94.4% of v1-valid notes unchanged, note count +14.9%. A pre-registered check on the other 13 songs passed all 11 criteria. Known residuals: vibrato over-splitting (inherited from v1) and small same-pitch tails (~1.6% of notes). v2 imports v1's helpers (`_core_slice`, `cents_diff`, `MIN_SEGMENT_VOICED_FRAMES`, …), so editing those changes both.

## The dataset

### Why it exists

Separation plus CREPE is 15 s–8 min per song, overwhelmingly dominated by Demucs. Axis 2 Phase B needs to iterate over thousands of notes many times. Re-running separation per experiment is untenable, so `scripts/build_main_dataset.py` extracts features **once** and writes them to disk in a form cheap to reload.

The key property: it is **score-free**. It saves contours, chroma, and timing — never a harmonic-fit or intonation score. Scores depend on constants under active revision (`RANK_LOW`, the intonation curve), so baking them in would stale the whole store the moment one changed. Anything derivable from the saved arrays is deliberately left underived.

### Layout

```
data/main_data/
  manifest.json                    index of every processed song (+ segmentation version/config)
  <song_id>/
    notes.npz          per-note arrays + flattened per-frame contours (v2 notes, schema 2)
    notes_v1.npz       the original v1 notes, preserved verbatim
    harmony.npz        chroma, beats, tuning, full vocal + bass f0 tracks (never rewritten)
    metadata.json      provenance and every parameter used (schema 2)
    metadata_v1.json   the original schema-1 metadata, preserved verbatim
```

`song_id` is an ASCII slug of the source filename (`_song_slug`, `build_main_dataset.py:50`), with an 8-char SHA-256 suffix appended only on a genuine collision.

### Schema versions

- **Schema 1:** `notes.npz` produced by v1 segmentation directly from audio by `build_main_dataset.py`.
- **Schema 2 (current, all 63 songs):** `notes.npz` re-segmented from `harmony.npz` by `resegment_main_dataset.py` with v2 `CANONICAL_CONFIG`. The array layout is identical; schema 2 adds provenance to `metadata.json` (`SCHEMA2_METADATA_KEYS`, `main_data_io.py:44`): segmentation version, algorithm, config, code hash, `harmony_npz_sha256`, content hashes of the new and v1 notes, v1 note count, and the re-segmentation commit and timestamp.

The store holds **31,803 v2 notes** across 63 songs; the v1 notes (`notes_v1.npz`) total **27,554**. The migration ran 2026-10-14 and is recorded in `ml/audits/axis2_migration_v2/migration_report.md`. A verified, hashed schema-1 backup is at `data/backups/main_data_v1_20261014/`.

`build_main_dataset.py` still writes schema 1, and **refuses to write into a store at a different schema** (`build_main_dataset.py:296`), so an audio rebuild cannot silently downgrade the v2 store. To rebuild from audio, use a new `--output-dir`.

### `notes.npz` — the flat-plus-offsets pattern

Per-note scalar arrays, all length `n_notes`: `note_index`, `target_midi_note`, `target_note_name`, `pitch_class`, `start_time_s`, `end_time_s`, `duration_s`, `median_cents_deviation`, `core_median_cents_deviation`, `voiced_frame_count`.

Per-frame contours, all length `contour_offsets[-1]`: `contour_time_s`, `contour_f0_hz`, `contour_cents_deviation`, `contour_crepe_confidence`, `contour_source_frame_index`.

Notes have **variable length**, which normally means an object array and `allow_pickle=True`. That would make the file executable-on-load and a security liability. Instead contours are concatenated into flat arrays plus `contour_offsets` (length `n_notes + 1`), so note *i* is `arr[offsets[i]:offsets[i+1]]` — the CSR-style layout. Every file therefore loads with `allow_pickle=False`, verified across all 63 songs. **Preserve this.** If you add a per-frame field, add it as another flat array sharing the same offsets.

`contour_source_frame_index` indexes back into the full-length vocal arrays in `harmony.npz`, so a note's contour can always be re-derived and cross-checked against its source.

### `harmony.npz`

`other_chroma_cens` and `other_chroma_rank` are `(12, n_frames)` at 22050 Hz / hop 512; `chroma_frame_times_s` aligns them. `global_chroma_profile` and `global_chroma_rank` are the whole-song `(12,)` vectors. `beat_times_s`, plus 0-d scalars `estimated_tempo_bpm`, `tuning_offset_semitones`, `tuning_offset_cents`.

Then the **full** CREPE tracks at 20 ms, not just the segmented parts: `vocal_frame_times_s`, `vocal_f0_hz`, `vocal_crepe_confidence`, `vocal_voiced_mask`, and the same four for `bass`. Full tracks are stored because a future experiment may want a different confidence threshold or segmentation rule, and that must not require re-running Demucs. Note the arrays are stored `f0`-with-`NaN` plus an explicit boolean mask — redundant on purpose, since the mask survives a dtype conversion that would silently lose `NaN`.

The bass track is **exploratory only**. It reuses the production vocal-tuned CREPE config, which is not a claim that it is an optimal bass tracker (see the comment at `build_main_dataset.py:203-205`), and bass currently feeds nothing in either score.

### `metadata.json`

Provenance and full parameter capture: `source_sha256`, `git_commit`, `processing_timestamp_utc`, `dataset_schema_version`, plus every parameter that shaped the output — Demucs model and device, CREPE capacity/step/threshold/viterbi, chroma rate and hop, pitch range, the tuning offset and tempo actually measured, and frame and coverage counts. The point is that any row of the dataset can be traced to the exact code and settings that produced it. Schema 2 adds the segmentation provenance listed above.

### Integrity guarantees

The feature-store tooling is stricter than typical script code, deliberately, because a corrupt feature store is expensive to detect later. The shared pieces live in `scripts/main_data_io.py`, moved verbatim out of `build_main_dataset.py` so the audio builder and the audio-free re-segmenter build `notes.npz` identically:

- **Atomic writes.** Every file goes to a `NamedTemporaryFile` in the destination directory, then `os.replace` (`_atomic_json`, `_atomic_npz` in `main_data_io.py`). No reader ever sees a partial file.
- **Content-addressed idempotency.** Sources are SHA-256 hashed; `_existing_output_is_current` re-validates and skips anything already at the current schema unless `--force`. The hash, not the filename, decides identity.
- **Validation after every write.** `validate_song_output` (`main_data_io.py:207`) re-opens the files and checks schema version, note-count agreement across all three files, strictly-increasing offsets terminating at the contour length, array alignment, finite and positive f0 and durations, and confidence within [0, 1], plus the schema-2 provenance keys. It runs on write and again under `--validate-only`.
- **Crash-safe manifest.** Saved after *every* song, not at the end, and failures are recorded as records with `processing_status: "failed"` rather than vanishing.
- **Never mutates sources.** Audio is read and hashed only; `resegment_main_dataset.py` never writes `harmony.npz`.

### Re-segmentation tooling (`scripts/resegment_main_dataset.py`)

Subcommands, in migration order: `stage` (segment every manifest song into a staging dir, `--segmentation v1|v2`), `compare-v1` (array-for-array and byte-for-byte check of a v1 staging run against the canonical v1 notes), `backup` (copy, hash, re-read and re-hash), `promote` (rename `notes.npz` → `notes_v1.npz` and `metadata.json` → `metadata_v1.json`, verified against the backup, then move staged files in, manifest last; resumable), and `validate`.

It is deterministic: two `stage` runs with the same `--timestamp` are byte-identical. Re-segmenting v1 from `harmony.npz` reproduced all 63 songs' original notes byte-for-byte, which is what licensed regenerating the store without audio.

Both default input paths (`build_main_dataset.py:43`, `evaluate.py:35`) point at a `data/test_songs` / `test_songs` directory that does not exist in this tree. `data/song_mp3s` exists but is currently empty, so pass explicit paths to restored audio.

## `data/dataset_selections.json`

Records **how each processed song may be used**, kept separate from the features themselves.

```json
{
  "schema_version": 1,
  "selection_basis": { "axis2_ml": "...", "harmonic_fit_eval": "...", "axis2_split": "..." },
  "counts": { "songs_total": 63, "axis2_ml": 50,
              "axis2_excluded": 13, "harmonic_fit_eval": 63,
              "axis2_split_train": 35, "axis2_split_validation": 7, "axis2_split_test": 8 },
  "songs": {
    "<song_id>": {
      "source_filename": "...",
      "axis2_ml": true,
      "harmonic_fit_eval": true,
      "axis2_split": "train"
    }
  }
}
```

Songs with `axis2_ml: false` carry an additional `axis2_exclusion_reason` string and `axis2_split: null`.

`axis2_split` is the song-level train/validation/test assignment for the Axis 2 ML dataset. It was locked before any synthetic corruption was generated, all variants of a note stay in their song's split, and splits are vocalist-disjoint (Bryson Tiller is the only repeated vocalist, and both songs are in train). The same assignment is in `data/ml_dataset/axis2_v1/splits.json`, which `axis2_v2` reuses verbatim.

**Why a separate file rather than fields in `metadata.json`.** Selection is a *judgment* about a song; the feature store is a *measurement* of it. Judgments change — a song might leave the Axis 2 pool as criteria tighten — and measurements do not. Writing selection into `metadata.json` would mean rewriting 63 validated files, invalidating their checksums and `git_commit` provenance, to record a decision that has nothing to do with what was measured. Keeping it separate means the canonical store changes only when a measurement changes (as it did once, in the v2 re-segmentation, which preserved the v1 files alongside), and the manifest stays the single source of truth for *what exists* while this file answers *what it is for*.

**The two flags are independent, and that is the point.** `axis2_ml` selects a 50-song clean-reference pool: songs whose vocal is a suitable intonation reference, excluding rap and speech-like delivery, multiple lead vocalists, heavy effects, and the weakest extractions. `harmonic_fit_eval` is currently `true` for all 63, because unsuitability as a clean *intonation* reference says nothing about usefulness for harmonic-fit evaluation — a rap track still has a chord progression, and Axis 1 reads only pitch class and time bounds. Conflating the two would discard perfectly good Axis 1 evaluation material for an unrelated reason.

`harmonic_fit_eval: true` means **eligible**, not audited. No harmonic-fit extraction-quality audit has been run; the flag is a statement of scope, not of verified quality.

Deliberately absent from this file: synthetic-corruption parameters (they live in each ML dataset's `summary.json`) and per-note filtering. Note filtering in particular belongs at training time, not here — song-level selection and note-level confidence gating are different operations at different granularities, and the extraction audit was explicit that a song with modest coverage can still yield hundreds of excellent individual notes.

## `ml/` — Axis 2 research

Everything under `ml/` is offline research. **Nothing here is wired into production**: `scoring.py` still uses the Phase A curve, and no code path loads a trained model.

### The extraction-quality audit that informed song selection (not in the tree)

The per-song Axis 2 candidate audit that `dataset_selections.json`'s exclusion reasons cite (`axis2_candidate_audit.csv` plus a summary) **is not in this checkout and was never committed**. Two of its headline figures can be recomputed from the store and still hold: median vocal coverage 67.5% and median voiced-frame CREPE confidence 0.90 across 63 songs. It covered the 27,554 v1 notes. It was an audit of **extraction quality, not singing quality**: large within-note cents movement usually means vibrato, portamento or scoops, which are expression, not error.

### `ml/audits/`

| Directory | What it is |
|---|---|
| `axis2_stable_center/` | Do v1 notes have stable pitch centers? The naive "middle 50% is 38% more stable" result is mostly an artifact of fewer frames; a same-length-window control leaves a real core effect of ~10%, and only for index-based (not time-based) trimming. |
| `axis2_segmentation_tail/` | The extreme tail of v1 notes is mostly segmentation failure, not expressive singing (12.1% clearly invalid). Defines the clear/likely-invalid classifier the v2 work reuses. |
| `axis2_segmentation_v2/` | The v2 parameter sweep (100 configs plus a 22-config follow-up), the recommendation of `gapv1local_g0.5_s100`, and the pre-registered 13-song shadow check (`shadow_excluded/`; criteria committed before the first run). |
| `axis2_migration_v2/` | Migration report, backup hashes, stage/promote/post validation JSON, and the fault-injection rehearsal script. |

The v1 audits now read `notes_v1.npz` when it exists, so they keep describing v1.

### ML datasets (`data/ml_dataset/`, built by `scripts/build_axis2_ml_dataset.py`)

Each clean note yields 5 examples: the clean contour verbatim (label 0.0) plus 4 copies with a constant shift added to **every cents frame**, one per band (5–15, 15–25, 25–35, 35–50 ¢), exactly 2 positive and 2 negative. The label is the injected signed shift. Confidence, timing and target MIDI are not changed (target MIDI is not recomputed after shifting). Shifts come from an RNG seeded by `sha256(f"20260824:{song_id}:{note_index}")`, so they do not depend on iteration order. No note filtering beyond structural validity. Each dataset's `README.md` has the full label semantics.

| Dataset | Notes from | Clean notes | Examples | Train / val / test songs |
|---|---|---|---|---|
| `axis2_v1` | `notes_v1.npz` | 22,241 | 111,205 | 35 / 7 / 8 |
| `axis2_v2` | `notes.npz` (v2) | 25,547 | 127,735 | 35 / 7 / 8 (same split) |

The original `axis2_v1` builder was never committed; `build_axis2_ml_dataset.py` re-implements it, and `--reproduce-check` rebuilt `axis2_v1` byte-identically before the script was trusted to build `axis2_v2`.

This corruption is **symbolic**: it shifts the saved cents contour and never touches audio. That sidesteps the phase-vocoder artifact-leakage trap described under Axis 2, but it also means no model here has seen real pitch-shifting artifacts.

### Baselines and experiments (validation split only; the test split has never been evaluated)

Hyperparameters are chosen by 5-fold `GroupKFold` by song inside train. Results are on `axis2_v2` unless noted:

| Method | Where | Val MAE / RMSE (¢) |
|---|---|---|
| Zero (predict no shift) | `ml/baselines/zero_baseline.py` | 20.51 / 25.51 |
| Full-contour median | `median_cents_baseline.py` | 18.60 / 25.20 (v1 notes: 19.03 / 28.44) |
| Core median (production `_core_slice`) | `core_median_baseline.py` | 17.43 / 21.90 |
| Ridge, 24 features | `ridge_baseline.py` | 13.32 / 16.51 (clean 10.33, 35–50 ¢ band 18.11) |
| Natural-offset HGB, 71 features | `ml/experiments/axis2_v2_natural_offset/` | 14.05 / 18.42 |
| Natural-offset TCN (contour + context) | `ml/experiments/axis2_v2_sequence_natural_offset/` | 14.34 / 19.61 |

Caveats that are easy to lose when quoting these:

- **The core-median baseline's errors are bounded by ±50 ¢ by construction.** A note's target MIDI is the rounded core median, so its "0 errors over 50 ¢" is structural rather than a robustness result.
- **Ridge shrinks.** Its prediction-vs-true-shift slope is 0.60, so large shifts are under-predicted. Its alpha barely matters: every alpha up to 100 scored within ~0.001 ¢ in CV, and 0.001 was chosen.
- **The natural-offset models predict a note's clean core median (N) from features with the absolute pitch center removed**, then estimate shift = observed core median − N̂. Because the corruption shifts every frame equally, the observed core median is exactly N + shift, so their slope of ~1.0 and their identical MAE on every variant are **properties of that construction, not learned behavior**. They beat Ridge only in the 25–35 and 35–50 ¢ bands, and lose on clean notes and shifts up to 25 ¢.
- **The TCN does not beat the handcrafted model** (CV 13.74 vs 13.57; validation 14.34 vs 14.05).

`ml/experiments/axis2_v2_sequence_natural_offset/best_model.pt` and three ~18.7k-row `validation_predictions.csv` files are committed. Each baseline and experiment script is standalone, with no shared training library.

## API and frontend

`POST /analyze` takes a multipart upload and validates: filename present (400), extension in a 9-item allowlist (415), non-empty (400), ≤60 MB (413). `ValueError` maps to 422 because it carries genuine user-facing conditions such as "no sung notes detected"; everything else becomes a generic 500 with detail kept server-side. The upload is deleted in a `finally` block and nothing is persisted.

The endpoint is intentionally `def`, **not** `async def` (`api.py:44`). Separation is minutes of blocking CPU work, so a sync handler runs on Starlette's threadpool instead of stalling the event loop. Do not "modernize" it to `async` without also moving the work off the loop. There is no job queue, so one request occupies one worker for its full duration.

One rough edge: the size check happens *after* the full stream to disk (`api.py:57-68`), so an oversized upload is written before rejection.

The frontend is a thin presenter — one `fetch`, a four-state machine (`idle → loading → done | error`), and no client-side scoring beyond `toPct` / `clamp01`. It never combines the axes, and its copy explains why. CORS is `allow_origins=["*"]` with `allow_credentials=False`, which is a safe pairing but wide open for deployment. The `/api` proxy in `vite.config.js` is dead config: `api.js` calls the absolute `VITE_API_BASE` and never uses that prefix.

`build_gemini_prompt` (`main.py:77`) interpolates **only numeric metrics** through format specs, never a filename or any user string, so the prompt-injection surface is closed by construction. Keep it that way. The report degrades gracefully — a missing key or any SDK failure returns an explanatory string rather than raising.

## Evaluation

`evaluate.py` is the only automated check of production scoring. It is label-free: released vocals are by definition in-key, so the system should rank the notes actually sung above the same notes relabelled to a wrong pitch class.

Perturbation is **symbolic** — only the integer `pitch_class` changes. This is valid because Axis 1 reads nothing but pitch class and time bounds, so relabelling fully simulates "a different note was sung here," and it avoids re-running separation and CREPE, which would change segmentation and confound the result. All 12 transpositions score in milliseconds.

Three tests: **transposition** (all 11 wrong global shifts; report margin over *best* wrong, not mean — mean flatters the system), **perturbation** (20% of notes shifted ±1–2 semitones, seeded, 5 draws — the closest proxy to real user behavior), and **bleed** (`other+bass` vs `other` vs `bass` alone, checking the signal is real harmony rather than the vocal's own leakage into `other`).

Stems cache to `/tmp/pitch_eval_cache` keyed on **filename stem with no content hash**, so a changed file with an unchanged name serves stale stems. That cache is currently empty and the source audio is gone, so `evaluate.py` cannot run until audio is restored. The 20-song figures above are historical measurements, not something this checkout can reproduce today. `evaluate.py` segments with v1.

## Working in this repo

**There are no unit tests.** Not for the DSP, not for the API, not for the frontend. Every constant above is unguarded, and `evaluate.py` would surface a regression as metric drift rather than a failure. The cheapest high-value tests would be pure-function ones needing no audio: `smooth_voicing_mask`, `_to_rank`, `intonation_score_from_cents`, `build_graph_points`, and `segment_notes` on synthetic f0 arrays.

Script-level checks do exist for the offline side, and are the closest thing to tests: `build_main_dataset.py --validate-only`, `resegment_main_dataset.py validate` / `compare-v1`, `build_axis2_ml_dataset.py --reproduce-check`, and `ml/audits/axis2_segmentation_v2/synthetic_checks.py` (v2 segmentation on synthetic contours). There is no CI, no Docker, and `requirements.txt` is a single flat freeze covering production and ML.

Some conventions the existing code follows:

- Frozen dataclasses for the inter-module payloads (`Stems`, `PitchTrack`, `HarmonicContext`, `PreprocessConfig`); plain dicts for note segments, per the contract above.
- Constants at module top with a comment explaining *why* that value, and what was measured. Several carry the measurement that justified them. Extend that habit rather than dropping bare numbers in.
- `from __future__ import annotations` in every backend module.
- Comments explain non-obvious *why* — the sync endpoint, the shifting window, the geometric mean. The code is unusually well-commented for its size, and the comments carry real decision history.

**The single most useful methodological lesson**, from the work that produced the current scoring: feature choice dominated everything. Removing bass from the chromagram tripled the margin, while *every* combination-level tweak swept — log-compression, HPSS, beat-sync aggregation, CQT-vs-CENS — moved it by ≤0.002. Prefer new information over new ways of combining existing information. This is why chord estimation outranks fitting the local/global weights despite being far more expensive.

## Current state

Built and measured: separation, tuning estimation, CENS chroma, Axis 1 scoring, Axis 2 Phase A, coverage reporting, beat tracking (tempo only), the a cappella guard, the local/global split, the evaluation harness, the 63-song feature store, and the Axis 2 candidate audit (not in the tree).

Built offline for Axis 2, not shipped: the segmentation tail and stable-center audits, v2 segmentation and the schema-2 migration of the feature store, the `axis2_v1` / `axis2_v2` synthetic-corruption datasets with locked song-level splits, four baselines (zero, median, core median, Ridge), and two natural-offset models (HGB, TCN). All are evaluated on validation only.

Next, in the recommended order: rename `key_compliance` → `harmonic_fit` in code (docs renamed it 2026-08-11; the rename must leave `evaluate.py` output byte-identical, and that *is* the check); re-verify the a cappella guard; strong/weak beat weighting, since `beat_times` is computed but unread and beat position is what distinguishes a suspension from a mistake; chord estimation, the direct attack on P4/P5; bass root motion, noting that bass-*in-chroma* is tested and harmful while bass-as-root-motion is untested — do not conflate them; a labelled set, which is what makes "more accurate" mean anything beyond the transposition proxy; then Axis 2 Phase B.

Deliberately not being done: octave checking (reversed by decision, not deferred), a learned note-set generator for Axis 1 (rejected on three grounds — no label exists, part of the target is genuinely undetermined, and feeding the vocal in leaks the answer since the cheapest way to minimize loss is to learn "whatever was sung is allowed"), and a high-accuracy dry-vocal mode (deferred).
