# CLAUDE.md

Technical orientation for working in this repo. `README.md` is the high-level overview; this file is the internals. `docs/NEW_VERSION.md` is the authoritative design document and `docs/RESEARCH.md` is the evidence base — when this file and those disagree, they win, and this file should be corrected.

## What the system is

A two-axis vocal analyzer. Given a full mixed song, it separates the stems, derives the harmony from the instrumental, tracks the vocal f0, groups frames into notes, and scores each note twice:

| Axis | Question | Reference | State |
|---|---|---|---|
| 1 — harmonic fit | Do the chosen notes fit the music? | CENS chroma of the `other` stem, per frame | built, measured |
| 2 — intonation precision | Were those notes sung cleanly? | cents from a tuning-corrected 12-TET target | Phase A built; Phase B (learned) not started |

**The axes are never averaged.** That is the single most important invariant in the project. Averaging recreates the exact problem the redesign existed to fix: a deliberate ♭5 sung dead-center is low Axis 1 and excellent Axis 2, and averages to "mediocre," which describes nothing. A singer who picked safe notes and sang them all 30 cents flat averages to the same value. The entire diagnostic content is in the split. The axes also differ in reliability — Axis 1's published ceiling is r ≈ 0.611 — so averaging contaminates the firmer number with the softer one.

A third number, **coverage**, is load-bearing rather than cosmetic. The `vocals` stem contains lead plus backing plus harmonies plus doubles, and CREPE is monophonic, so it can follow whichever voice is loudest and jump between lead and harmony mid-phrase. The chosen mitigation is to confidence-gate hard and drop ambiguous frames — score fewer notes rather than score them wrong — which is only honest if the user sees how much was skipped.

## Repo layout

```
backend/           production pipeline (8 modules, ~1,100 lines)
  separation.py      Demucs → Stems
  harmony.py         tuning, CENS chroma, beats, Axis 1 scoring
  preprocess.py      CREPE f0, voicing mask, coverage
  note_segmentation.py  frames → note segments, tuning-aware targets
  scoring.py         Axis 2 scoring, chart point construction
  main.py            orchestration, metric aggregation, Gemini, CLI
  api.py             FastAPI: POST /analyze, GET /health
  evaluate.py        label-free evaluation harness
frontend/          React 19 + Vite + Tailwind + Recharts SPA
scripts/
  build_main_dataset.py   offline feature extraction → data/main_data/
data/
  song_mp3s/         source audio (35 files, gitignored)
  main_data/         canonical feature store, 63 songs + manifest.json
  dataset_selections.json   which songs may be used for what
ml/audits/         Axis 2 candidate extraction audit (CSV + summary)
docs/              NEW_VERSION.md (design), RESEARCH.md (evidence)
```

`docs/` and `data/` are gitignored (`.gitignore:13`, `:16`), so the design documents and the dataset are **not under version control**. Worth knowing before assuming a clean clone has them.

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

All four entry points call the *same* primitives. This is deliberate and worth preserving: it is why the dataset builder cannot drift from production behavior, and why `evaluate.py` measures the configuration that actually ships. `build_main_dataset.py` imports from `backend/` by injecting it onto `sys.path` (`scripts/build_main_dataset.py:23-26`) rather than duplicating any DSP.

### Three cross-module contracts

Breaking any of these breaks things far from the edit site.

1. **`NaN` means unvoiced.** `preprocess.py` masks the f0 array rather than returning a separate boolean, and every downstream consumer tests `np.isnan`. Established at `preprocess.py:153-154`, relied on in `note_segmentation.py:109` and `build_main_dataset.py:232`.

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

- **Octave is ignored by design**, permanently (NEW_VERSION §5.3a, a reversed earlier decision). Harmony is octave-invariant: a chord is a set of pitch classes, so a pitch class supported by the sounding harmony is supported in every octave. The only octave reference available is whichever one the original artist sang in, a property of their range rather than of the music, so checking it would penalize every singer whose range differs — which is most of them, and would silently turn a note-choice score into an unrequested voice-type judgment. Describe harmonic fit as a *pitch-class* measure, not as "octave-blind."

`RANK_LOW = 0.50` and `RANK_HIGH = 0.75` are a **presentation choice, not a measurement**. The same audio reports 66% or 85% depending on where they sit. Any recalibration must be re-validated with `evaluate.py`, never by eye.

Two escape hatches return `None`: zero chroma frames, and the a cappella guard (`MIN_HARMONY_TO_VOCAL_RMS = 0.04`, `harmony.py:51`). An a cappella upload still yields a nominal `other` stem, but it holds only separation residue of the voice itself — scoring against it compares the vocal to its own leakage and returns a confident, meaningless number (measured at 78% before the guard existed). Real mixes measure 0.29–0.60 on this ratio, so there is an order of magnitude of headroom, but the threshold was validated when chroma used `other+bass` and has **not** been re-confirmed against an actual a cappella since.

### Known failure mode: fourths and fifths

On the 20-song evaluation the true key ranks #1 of 12 on 14 songs. **All 6 failures lost to a perfect fourth or fifth, and to no other interval.** The mechanism: the score is a mean over pitch-class ranks with no harmonic-function awareness, and transposing by a fourth or fifth maps most scale degrees onto other degrees of the same key, so the shifted melody still lands on pitch classes the chromagram rates highly. Diatonic set overlap is maximal at ±5 and ±7 semitones (6 of 7 notes shared), which is why exactly those two shifts are confusable.

The chromagram measures pitch-class **presence**, not **role**. That is why chord estimation, not reweighting, is the plausible fix — a chord label distinguishes I from V where raw pitch-class presence cannot — and why the P4/P5 problem should not be debugged as a chroma-quality problem.

A caveat when following cross-references: NEW_VERSION's §5.10 ("Discrete chord labels are presentation only") is a **heading with no body**, so anything citing it for chord estimation lands on an empty section. The intended claim is reconstructable from §5.5 and open decision 7: scoring does not need discrete chord labels, and surfacing them in the UI would add interpretability plus a new failure mode. That is a statement about *display*, and it does not settle whether chord estimation should replace raw chroma as the scoring feature — which remains the open, and most promising, question.

Two things to keep straight when reading eval numbers. Uniform global transposition is not how singers err, so this result bounds *key identification* and says less about catching an individual wrong note; the perturbation test is the better proxy there and passes 20/20 with every draw positive. And **mean margin is a misleading summary** — it was identical (+0.023) across a 5-song set with 0% failures and a 20-song set with 30%. Always report the placement distribution with it.

## Axis 2, currently (`scoring.py`)

Phase A only: `|cents|` from a tuning-corrected target, measured off the core-trimmed median, mapped through a hand-tuned piecewise curve (`intonation_score_from_cents`, `scoring.py:25`) — linear to 25 ¢, steeper to 45 ¢, exponential decay beyond. The result is clipped to `SCORE_FLOOR = 0.05`, so a catastrophically flat note reads as 5% rather than 0%; keep that in mind when reading a low score or comparing against a hand computation.

Phase A **exists to be the baseline Phase B must beat.** Building the model first would leave nothing to compare against. It is a 12-TET grid and therefore wrong for the target genres by construction: a just major third sits −13.7 ¢ from equal temperament, a just minor third +15.6 ¢, skilled singers sharpen leading tones ~+10 ¢, and blue notes are deliberately microtonal. Measured median deviation on this repo's samples is ~22.5 ¢, of which a just major third alone would account for 13.7. The system cannot currently distinguish "sang an expressive third" from "sang a sloppy third."

Tuning correction is what makes the axis defensible at all. Offsets on real recordings span roughly −16 to +35 cents, and the curve's entire "pro zone" is 0–25 ¢ — so a 16-cent offset would consume 64% of that budget before the singer sings a note, and would apply uniformly, making a perfect performance over a sharp backing track read as consistently sharp. The offset is estimated from the *instrumental* (`estimate_tuning_semitones`, `harmony.py:75`) because fixed-pitch instruments define the grid the singer is aiming at, and it feeds both axes.

**Phase B (planned, NEW_VERSION §7)** trains on synthetic corruption: take a released vocal, apply a known pitch shift, and the shift amount *is* the label. This is why ML fits Axis 2 and not Axis 1 — for note choice no label exists and part of the target is genuinely undetermined, while here labels are free. Because the target becomes *the released vocal* rather than a grid, expressive practice is baked in: a just third appears at 0-shift throughout training, so the model learns it as the target rather than a 14-cent error.

Two traps recorded before anyone starts:

- **Artifact leakage.** Pitch-shifting leaves resampling and phase-vocoder fingerprints. A model can score perfectly by detecting "this frame was shifted" while understanding nothing about pitch, then collapse on real vocals. Defense: include zero-shift controls that still pass through the full shifting pipeline, so artifacts carry a label of 0 cents and are uninformative. Do this from the first run — retrofitting invalidates every prior result.
- **It makes vibrato not-penalized, not assessable.** The model only ever sees good vibrato at 0-shift and pitch-shifted good vibrato at X-shift, so it learns vibrato is irrelevant to its prediction. Judging vibrato quality needs human labels that do not exist here. Keep vibrato and portamento in the reported-not-judged category.

### Known inconsistency: the metrics panel and the chart disagree

`compute_metrics` averages the **unsmoothed** `intonation_score` (`main.py:59`), while `build_graph_points` plots a **median-smoothed** version over a ±0.5 s window (`SMOOTHING_HALF_WINDOW_S`, `scoring.py:64-71`). The headline number and the line in the chart are therefore computed from different values, and a user can see a dip in the graph that the reported score does not reflect — or vice versa.

This is live, not stale. It is the last surviving item from NEW_VERSION's "known bugs" appendix; the others there have all been fixed (the bare `except` in the scoring path is gone, CREPE's `activation` is explicitly discarded as `_activation`, `/analyze` no longer blocks the event loop, and metrics are now duration-weighted rather than note-count weighted).

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

## The dataset

### Why it exists

Separation plus CREPE is 15 s–8 min per song, overwhelmingly dominated by Demucs. Axis 2 Phase B needs to iterate over thousands of notes many times. Re-running separation per experiment is untenable, so `scripts/build_main_dataset.py` extracts features **once** and writes them to disk in a form cheap to reload.

The key property: it is **score-free**. It saves contours, chroma, and timing — never a harmonic-fit or intonation score. Scores depend on constants under active revision (`RANK_LOW`, the intonation curve), so baking them in would stale the whole store the moment one changed. Anything derivable from the saved arrays is deliberately left underived.

### Layout

```
data/main_data/
  manifest.json                    index of every processed song
  <song_id>/
    notes.npz       per-note arrays + flattened per-frame contours
    harmony.npz     chroma, beats, tuning, full vocal + bass f0 tracks
    metadata.json   provenance and every parameter used
```

`song_id` is an ASCII slug of the source filename (`_song_slug`, `:107`), with an 8-char SHA-256 suffix appended only on a genuine collision.

### `notes.npz` — the flat-plus-offsets pattern

Per-note scalar arrays, all length `n_notes`: `note_index`, `target_midi_note`, `target_note_name`, `pitch_class`, `start_time_s`, `end_time_s`, `duration_s`, `median_cents_deviation`, `core_median_cents_deviation`, `voiced_frame_count`.

Per-frame contours, all length `contour_offsets[-1]`: `contour_time_s`, `contour_f0_hz`, `contour_cents_deviation`, `contour_crepe_confidence`, `contour_source_frame_index`.

Notes have **variable length**, which normally means an object array and `allow_pickle=True`. That would make the file executable-on-load and a security liability. Instead contours are concatenated into flat arrays plus `contour_offsets` (length `n_notes + 1`), so note *i* is `arr[offsets[i]:offsets[i+1]]` — the CSR-style layout. Every file therefore loads with `allow_pickle=False`, verified across all 63 songs. **Preserve this.** If you add a per-frame field, add it as another flat array sharing the same offsets.

`contour_source_frame_index` indexes back into the full-length vocal arrays in `harmony.npz`, so a note's contour can always be re-derived and cross-checked against its source.

### `harmony.npz`

`other_chroma_cens` and `other_chroma_rank` are `(12, n_frames)` at 22050 Hz / hop 512; `chroma_frame_times_s` aligns them. `global_chroma_profile` and `global_chroma_rank` are the whole-song `(12,)` vectors. `beat_times_s`, plus 0-d scalars `estimated_tempo_bpm`, `tuning_offset_semitones`, `tuning_offset_cents`.

Then the **full** CREPE tracks at 20 ms, not just the segmented parts: `vocal_frame_times_s`, `vocal_f0_hz`, `vocal_crepe_confidence`, `vocal_voiced_mask`, and the same four for `bass`. Full tracks are stored because a future experiment may want a different confidence threshold or segmentation rule, and that must not require re-running Demucs. Note the arrays are stored `f0`-with-`NaN` plus an explicit boolean mask — redundant on purpose, since the mask survives a dtype conversion that would silently lose `NaN`.

The bass track is **exploratory only**. It reuses the production vocal-tuned CREPE config, which is not a claim that it is an optimal bass tracker (see the comment at `build_main_dataset.py:420`), and bass currently feeds nothing in either score.

### `metadata.json`

Provenance and full parameter capture: `source_sha256`, `git_commit`, `processing_timestamp_utc`, `dataset_schema_version`, plus every parameter that shaped the output — Demucs model and device, CREPE capacity/step/threshold/viterbi, chroma rate and hop, pitch range, the tuning offset and tempo actually measured, and frame and coverage counts. The point is that any row of the dataset can be traced to the exact code and settings that produced it.

### Integrity guarantees

`build_main_dataset.py` is stricter than typical script code, deliberately, because a corrupt feature store is expensive to detect later:

- **Atomic writes.** Every file goes to a `NamedTemporaryFile` in the destination directory, then `os.replace` (`_atomic_json`, `_atomic_npz`). No reader ever sees a partial file.
- **Content-addressed idempotency.** Sources are SHA-256 hashed; `_existing_output_is_current` re-validates and skips anything already at the current schema unless `--force`. The hash, not the filename, decides identity.
- **Validation after every write.** `validate_song_output` (`:295`) re-opens the files and checks schema version, note-count agreement across all three files, strictly-increasing offsets terminating at the contour length, array alignment, finite and positive f0 and durations, and confidence within [0, 1]. It runs on write and again under `--validate-only`.
- **Crash-safe manifest.** Saved after *every* song, not at the end, and failures are recorded as records with `processing_status: "failed"` rather than vanishing.
- **Never mutates sources.** Audio is read and hashed only.

Both default input paths (`build_main_dataset.py:39`, `evaluate.py:35`) point at a `data/test_songs` / `test_songs` directory that does not exist in this tree; the corpus is at `data/song_mp3s`. Pass explicit paths or repoint the constants.

## `data/dataset_selections.json`

Records **how each processed song may be used**, kept separate from the features themselves.

```json
{
  "schema_version": 1,
  "selection_basis": { "axis2_ml": "...", "harmonic_fit_eval": "..." },
  "counts": { "songs_total": 63, "axis2_ml": 50,
              "axis2_excluded": 13, "harmonic_fit_eval": 63 },
  "songs": {
    "<song_id>": {
      "source_filename": "...",
      "axis2_ml": true,
      "harmonic_fit_eval": true
    }
  }
}
```

Songs with `axis2_ml: false` carry an additional `axis2_exclusion_reason` string.

**Why a separate file rather than fields in `metadata.json`.** Selection is a *judgment* about a song; the feature store is a *measurement* of it. Judgments change — a song might leave the Axis 2 pool as criteria tighten — and measurements do not. Writing selection into `metadata.json` would mean rewriting 63 validated files, invalidating their checksums and `git_commit` provenance, to record a decision that has nothing to do with what was measured. Keeping it separate means the canonical store is append-only in practice, and the manifest stays the single source of truth for *what exists* while this file answers *what it is for*.

**The two flags are independent, and that is the point.** `axis2_ml` selects a 50-song clean-reference pool: songs whose vocal is a suitable intonation reference, excluding rap and speech-like delivery, multiple lead vocalists, heavy effects, and the weakest extractions. `harmonic_fit_eval` is currently `true` for all 63, because unsuitability as a clean *intonation* reference says nothing about usefulness for harmonic-fit evaluation — a rap track still has a chord progression, and Axis 1 reads only pitch class and time bounds. Conflating the two would discard perfectly good Axis 1 evaluation material for an unrelated reason.

`harmonic_fit_eval: true` means **eligible**, not audited. No harmonic-fit extraction-quality audit has been run; the flag is a statement of scope, not of verified quality.

Deliberately absent, and should stay absent until each is actually decided: train/validation/test splits, synthetic-corruption parameters, and per-note filtering. Note filtering in particular belongs at training time, not here — song-level selection and note-level confidence gating are different operations at different granularities, and the audit is explicit that a song with modest coverage can still yield hundreds of excellent individual notes.

## `ml/audits/`

`axis2_candidate_audit.csv` (63 rows, 37 columns) and a markdown summary. Per-song extraction-quality diagnostics: coverage, CREPE confidence distributions (median, P25, P10, per-note medians), note-duration distributions, absolute cents deviations, within-note cents standard deviation, and five boolean flags.

Read it with its own framing: it is an audit of **extraction quality, not singing quality**. Large within-note cents movement usually means vibrato, portamento, or scoops — expression, not error. The flags are provisional prompts for review, not exclusion rules, and the audit explicitly declines to select the final song list. Headline figures: 63 songs, 27,554 notes, median coverage 67.5%, median voiced-frame confidence 0.90, 35 songs flagged for review (mostly on the strictest flag, "fewer than 75% of notes at median confidence ≥ 0.85").

## API and frontend

`POST /analyze` takes a multipart upload and validates: filename present (400), extension in a 9-item allowlist (415), non-empty (400), ≤60 MB (413). `ValueError` maps to 422 because it carries genuine user-facing conditions such as "no sung notes detected"; everything else becomes a generic 500 with detail kept server-side. The upload is deleted in a `finally` block and nothing is persisted.

The endpoint is intentionally `def`, **not** `async def` (`api.py:41`). Separation is minutes of blocking CPU work, so a sync handler runs on Starlette's threadpool instead of stalling the event loop. Do not "modernize" it to `async` without also moving the work off the loop. There is no job queue, so one request occupies one worker for its full duration.

One rough edge: the size check happens *after* the full stream to disk (`api.py:57-68`), so an oversized upload is written before rejection.

The frontend is a thin presenter — one `fetch`, a four-state machine (`idle → loading → done | error`), and no client-side scoring beyond `toPct` / `clamp01`. It never combines the axes, and its copy explains why. CORS is `allow_origins=["*"]` with `allow_credentials=False`, which is a safe pairing but wide open for deployment. The `/api` proxy in `vite.config.js` is dead config: `api.js` calls the absolute `VITE_API_BASE` and never uses that prefix.

`build_gemini_prompt` (`main.py:77`) interpolates **only numeric metrics** through format specs, never a filename or any user string, so the prompt-injection surface is closed by construction. Keep it that way. The report degrades gracefully — a missing key or any SDK failure returns an explanatory string rather than raising.

## Evaluation

`evaluate.py` is the only automated check that exists. It is label-free: released vocals are by definition in-key, so the system should rank the notes actually sung above the same notes relabelled to a wrong pitch class.

Perturbation is **symbolic** — only the integer `pitch_class` changes. This is valid because Axis 1 reads nothing but pitch class and time bounds, so relabelling fully simulates "a different note was sung here," and it avoids re-running separation and CREPE, which would change segmentation and confound the result. All 12 transpositions score in milliseconds.

Three tests: **transposition** (all 11 wrong global shifts; report margin over *best* wrong, not mean — mean flatters the system), **perturbation** (20% of notes shifted ±1–2 semitones, seeded, 5 draws — the closest proxy to real user behavior), and **bleed** (`other+bass` vs `other` vs `bass` alone, checking the signal is real harmony rather than the vocal's own leakage into `other`).

Stems cache to `/tmp/pitch_eval_cache` keyed on **filename stem with no content hash**, so a changed file with an unchanged name serves stale stems.

## Working in this repo

**There are no unit tests.** Not for the DSP, not for the API, not for the frontend. Every constant above is unguarded, and `evaluate.py` would surface a regression as metric drift rather than a failure. The cheapest high-value tests would be pure-function ones needing no audio: `smooth_voicing_mask`, `_to_rank`, `intonation_score_from_cents`, `build_graph_points`, and `segment_notes` on synthetic f0 arrays.

Some conventions the existing code follows:

- Frozen dataclasses for the inter-module payloads (`Stems`, `PitchTrack`, `HarmonicContext`, `PreprocessConfig`); plain dicts for note segments, per the contract above.
- Constants at module top with a comment explaining *why* that value, and what was measured. Several carry the measurement that justified them. Extend that habit rather than dropping bare numbers in.
- `from __future__ import annotations` in every backend module.
- Comments explain non-obvious *why* — the sync endpoint, the shifting window, the geometric mean. The code is unusually well-commented for its size, and the comments carry real decision history.

**The single most useful methodological lesson**, from the work that produced the current scoring: feature choice dominated everything. Removing bass from the chromagram tripled the margin, while *every* combination-level tweak swept — log-compression, HPSS, beat-sync aggregation, CQT-vs-CENS — moved it by ≤0.002. Prefer new information over new ways of combining existing information. This is why chord estimation outranks fitting the local/global weights despite being far more expensive.

## Current state

Built and measured: separation, tuning estimation, CENS chroma, Axis 1 scoring, Axis 2 Phase A, coverage reporting, beat tracking (tempo only), the a cappella guard, the local/global split, the evaluation harness, the 63-song feature store, and the Axis 2 candidate audit.

Next, in the order NEW_VERSION §9 recommends: rename `key_compliance` → `harmonic_fit` in code (docs renamed it 2026-08-11; the rename must leave `evaluate.py` output byte-identical, and that *is* the check); re-verify the a cappella guard; strong/weak beat weighting, since `beat_times` is computed but unread and beat position is what distinguishes a suspension from a mistake; chord estimation, the direct attack on P4/P5; bass root motion, noting that bass-*in-chroma* is tested and harmful while bass-as-root-motion is untested — do not conflate them; a labelled set, which is what makes "more accurate" mean anything beyond the transposition proxy; then Axis 2 Phase B.

Deliberately not being done: octave checking (§5.3a, reversed by decision, not deferred), a learned note-set generator for Axis 1 (rejected on three grounds in §3 warning 4 — no label exists, part of the target is genuinely undetermined, and feeding the vocal in leaks the answer since the cheapest way to minimize loss is to learn "whatever was sung is allowed"), and a high-accuracy dry-vocal mode (§12.2, deferred).
