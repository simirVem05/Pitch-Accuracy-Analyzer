# Pitch Accuracy Analyzer

Vocal analysis that reads a song's harmony from its own instrumental instead of asking you to declare a key.

Upload a full song. The system separates it into stems, works out the harmony from the instrumental, tracks what the singer actually sang, and reports **two independent scores**:

| Axis | Question | Reference |
|---|---|---|
| **Harmonic fit** | Do the notes chosen fit the music? | The song's own instrumental, frame by frame |
| **Intonation precision** | Were those notes sung cleanly? | Tuning-corrected cents deviation |

These are never combined into one number. A singer can nail a deliberate ♭5 dead-center — excellent intonation, low harmonic fit, musically superb. A singer can also drift 20 cents flat on a plain root note — correct note choice, poor execution. One number describes neither, so the system reports both plus a coverage figure and lets them disagree.

## Why not a key and genre picker

The previous version asked for a key and a genre, then built a set of "allowed" notes from a lookup table. That does not work: two R&B songs in B minor with different chord progressions have different correct-note sets, so genre is a weak prior over an enormous space, and the system became a chain of if/else branches each patching the last one's mistakes.

The instrumental answers the question directly, per song, per moment. Deriving the pitch reference from the accompaniment is validated in the literature — Hsieh et al. (Interspeech 2025) report r = 0.611 against human raters versus 0.364 and 0.232 for reference-free approaches. Notably, they used a *global* key; reading harmony per frame is a step beyond published work.

See `docs/NEW_VERSION.md` for the full design and `docs/RESEARCH.md` for the evidence base.

## How it works

```
full song (mp3, wav, flac, m4a, ...)
  │
  ▼
Demucs 4-stem separation → vocals / other / bass / drums
  │
  ├─ tuning offset ── librosa.estimate_tuning on the instrumental
  │                   (feeds both axes; drums excluded — broadband
  │                    transients smear energy across all 12 bins)
  │
  ├─ harmony ──────── CENS chroma of `other`, tuning-aligned
  │                   → per-frame pitch-class prominence
  │                   + beat tracking for slack windows
  │
  └─ vocal ───────── CREPE (full capacity) → f0, confidence-gated
                     → note segments via hysteresis
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      harmonic fit                intonation precision
      pitch class vs. chroma      cents from tuning-
      prominence, ±1 beat         corrected target,
      of slack                    perceptual curve
              └─────────────┬─────────────┘
                            ▼
              two scores + coverage + Gemini report
```

**Chroma, not per-stem pitch tracking.** No separator produces individual instruments — guitars, synths, and keys all land together in one polyphonic `other` stem, which a monophonic tracker cannot read. Chroma never commits to a single note, so chords need no special handling.

**Rank, not raw salience.** Absolute chroma energy depends on level and arrangement density, so it is not comparable across sections. Each pitch class is scored by its rank among the twelve in that frame, which measured roughly twice the separation between notes actually sung and deliberately transposed ones.

**Slack shifts the window, it does not widen it.** Widening a short note's window dilutes it into beats of unrelated harmony and raises the score for every pitch class equally — forgiving wrong notes as much as anticipated ones. Shifting a fixed-width window asks the intended question: was this note supported by the chord just before or just after it?

**Local and global support, gated.** Each note is scored on both how prominent its pitch class is *at that moment* and across *the whole song*, combined as a geometric mean. Local support alone conflates a note outside the key with an in-key note over a chord that doesn't contain it — a passing tone or suspension. A genuinely wrong note is weak on both terms; a passing tone is strong on one. The geometric mean makes the terms gate rather than compensate, so key membership can't fully rescue a note the current harmony rejects.

**Bass is excluded from the chromagram.** It measures 2.8–4.6× louder than `other` in real mixes, so summing them lets root notes drown out the chord voicings that actually distinguish one harmony from another. Measured cost of including it: the margin over the best wrong key falls from +0.043 to +0.014.

**Harmonic fit is a pitch-class measure, by design.** Chroma collapses octaves, and that is intended rather than a gap — a chord is a set of pitch classes, so a note supported by the sounding harmony is supported in every octave. The only octave reference available would be the one the original artist happened to sing in, which is a property of their range, not of the music; checking it would penalize every singer whose range differs from the artist's. See `docs/NEW_VERSION.md` §5.3a.

## Evaluating changes

`backend/evaluate.py` is a label-free harness. Released vocals are by definition in-key, so the system should rank the notes actually sung above the same notes relabelled to any of the 11 wrong pitch classes:

```bash
cd backend && python evaluate.py               # every song in test_songs/
python evaluate.py ../data/song_mp3s/song.mp3  # one file
```

It runs three tests — global transposition (can the true key be identified), scattered perturbation (are individual wrong notes penalized), and a bleed check (is the signal real harmony or the vocal leaking into `other`). Stems are cached to `/tmp` after the first run, so iterating on scoring is fast.

Note that `TEST_DIR` in `evaluate.py` points at a `test_songs/` directory relative to the working directory, which is not present in this tree — the corpus now lives in `data/song_mp3s/`. Pass an explicit path, or repoint the constant.

**Current results, measured on 20 songs:** the true key ranks #1 of 12 on **14 of 20** songs. All 6 failures lose to a perfect fourth or perfect fifth, and to nothing else. This is a systematic and explicable failure mode rather than variance: the score is a mean over pitch-class ranks with no harmonic-function awareness, and transposing by a fourth or fifth maps most scale degrees onto other degrees of the same key, so the shifted melody still lands on pitch classes the chromagram rates highly. Diatonic overlap is maximal at ±5 and ±7 semitones, which is why those two shifts and no others are confusable.

Two caveats that cut in opposite directions. Uniform global transposition is *not* how singers err, so this bounds how well the feature identifies a key and says less about catching an individual wrong note — the perturbation test is the better proxy there, and it passes cleanly on 20/20 songs with every individual draw positive. But an earlier 5-song run reported 5/5 and a mean margin identical to the 20-song set's, so **mean margin is a misleading summary statistic** and five songs was too few to evaluate this axis. Report the placement distribution alongside it.

## The feature dataset

Extract score-free note contours, `other`-stem chroma, beat/tuning information, and an exploratory bass pitch track using the same production preprocessing code:

```bash
python scripts/build_main_dataset.py --limit 5  # next five unprocessed files, sorted by filename
python scripts/build_main_dataset.py            # all remaining/new files
python scripts/build_main_dataset.py --validate-only
```

Outputs are written under `data/main_data/<song_id>/`. Valid current-schema files are skipped unless `--force` is supplied. Note contours use flat numerical arrays plus `contour_offsets`, so `notes.npz` loads with `allow_pickle=False` while still supporting variable-length notes. Source audio is hashed and is never modified or deleted by the builder.

`--input-dir` defaults to `data/test_songs`, which is not present in this tree; point it at `data/song_mp3s`.

63 songs and 27,554 notes are currently extracted. `data/dataset_selections.json` records which of them may be used for what — 50 songs form the Axis 2 clean-reference pool, all 63 are eligible for harmonic-fit evaluation. Selection is stored separately from the features so the canonical feature store never has to be rebuilt when a decision changes. See `CLAUDE.md` for the schema and rationale.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd frontend && npm install && cd ..
```

Add a `backend/.env` with a Gemini key (optional — analysis works without it, only the written report is skipped):

```
GEMINI_API_KEY=your_key_here
```

## Running

```bash
# API
cd backend && uvicorn api:app --reload --port 8000

# UI, in another shell
cd frontend && npm run dev
```

CLI, for a single file:

```bash
cd backend
python main.py ../data/song_mp3s/"Adele - Hello.mp3"
python main.py ../data/song_mp3s/"Adele - Hello.mp3" --no-report   # skip Gemini
```

Writes `outputs/metrics.json`, `outputs/graph_points.json`, and `outputs/report.txt`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Enables the written report |
| `DEMUCS_MODEL` | `htdemucs` | Set `htdemucs_ft` for slightly better stems at ~4× the runtime |
| `DEMUCS_DEVICE` | auto | Forces `cuda` / `mps` / `cpu`; auto-detects by default |
| `VITE_API_BASE` | `http://127.0.0.1:8000` | Frontend → backend URL |

## Runtime

Separation dominates: roughly 15–30 s on a CUDA GPU, 1–2 min on Apple silicon via MPS, and 3–8 min on CPU. The device is auto-selected. `/analyze` is deliberately synchronous so FastAPI runs it on the threadpool rather than blocking the event loop; there is no job queue yet, so one request occupies one worker for its duration.

## Known limitations

- **Harmonic fit is a relative indicator, not an absolute grade.** Released recordings score 66–94%; the same notes transposed to a wrong key score 17–28%. So a 66% does not mean a third of the notes were wrong — it means the accompaniment gave them moderate support. The two constants that set this scale are a presentation choice, not a measurement. Compare within a song, not across songs. Details in `NEW_VERSION.md` §5.6.
- **Fourths and fifths are systematically confusable** (above). The chromagram measures pitch-class *presence*, not *harmonic role*.
- **Cross-song scores overlap.** Across the 20-song set the true-key score band (0.643–0.816) overlaps the best-wrong-key band (0.643–0.747) — one song's *worst* wrong key can outscore another's true key. Arrangement density moves the whole scale.
- **No human ground truth.** The thresholds mapping chroma rank and cents deviation to scores are calibrated against released recordings, not fitted to labelled data. Correlation against human raters has never been measured, and "ranks the true key first" is not the same as "agrees with a listener."
- **Beat slack is a fixed ±1 beat**, with no strong/weak beat weighting, though `beat_times` is already computed. Beat position is what distinguishes a suspension from a mistake, so expect some false dissonance on heavily syncopated phrasing — a timing problem, not a chroma problem.
- **A cappella uploads cannot be scored for harmonic fit.** With no instrumental there is no harmonic reference; the system detects this and reports the axis as unmeasured rather than scoring the vocal against its own separation residue. The detection threshold has not been re-verified since chroma switched to `other`-only.
- **Layered vocals reduce coverage.** The `vocals` stem holds lead, backing, harmonies, and doubles together, and CREPE is monophonic — it can jump between lead and harmony mid-phrase. Ambiguous frames are dropped rather than guessed, and the reported coverage figure shows how much was skipped. Median coverage across the 63-song dataset is 67.5%.
- **Intonation is still measured against a 12-TET grid**, so intentionally microtonal notes are penalized. A just major third sits 13.7 cents below equal temperament and blue notes are deliberately microtonal, which is exactly what the target genres are built on. Replacing this with a learned, style-aware target is the planned next step (`NEW_VERSION.md` §7).
- **No unit tests.** `evaluate.py` measures end-to-end discrimination, but the pure functions have none, so a refactor would surface as metric drift rather than a failure.

## Roadmap

Ordered as in `NEW_VERSION.md` §9. Everything through the local/global support split is built.

| Next | Effort |
|---|---|
| Rename `key_compliance` → `harmonic_fit` in code and UI (docs already use the new name) | small |
| Re-verify the a cappella guard against an actual a cappella | small |
| Strong/weak beat weighting — `beat_times` is computed but unread | medium |
| Chord estimation instead of raw chroma; the direct attack on the P4/P5 mechanism | large |
| Bass root motion as a third feature | medium |
| A labelled set — what makes "more accurate" mean anything beyond the transposition proxy | large |
| Axis 2 Phase B: learned intonation target trained on synthetic pitch corruption | large |

One methodological lesson from the work so far: **feature choice dominated.** Removing bass from the chromagram tripled the margin, while every combination-level tweak swept (log-compression, HPSS, beat-sync aggregation, CQT-vs-CENS) moved it by ≤0.002. Prefer new information over new ways of combining existing information.

**Terminology note.** The docs renamed "key compliance" to "harmonic fit" on 2026-08-11 — the score never measured compliance with a *key*, and "compliance" implied rule-breaking where a low value usually means an adventurous-but-correct choice. The rename through code is pending, so `metrics.json`, the API response, and the frontend still carry `key_compliance`. Scoring math is unaffected.

## Tech stack

Python · Demucs (PyTorch) · CREPE (TensorFlow) · librosa · SciPy · NumPy · FastAPI · React · Vite · Tailwind · Recharts · Gemini
