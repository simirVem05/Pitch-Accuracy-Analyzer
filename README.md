# Pitch Accuracy Analyzer

Vocal analysis that reads a song's harmony from its own instrumental instead of asking you to declare a key.

Upload a full song. The system separates it into stems, works out the harmony from the instrumental, tracks what the singer actually sang, and reports **two independent scores**:

| Axis | Question | Reference |
|---|---|---|
| **Key compliance** | Do the notes chosen fit the music? | The song's own instrumental |
| **Intonation accuracy** | Were those notes sung cleanly? | Tuning-corrected cents deviation |

These are never combined into one number. A singer can nail a deliberate ♭5 dead-center — excellent intonation, low key compliance, musically superb. Averaging those would describe neither.

## Why not a key and genre picker

The previous version asked for a key and a genre, then built a set of "allowed" notes from a lookup table. That does not work: two R&B songs in B minor with different chord progressions have different correct-note sets, so genre is a weak prior over an enormous space, and the system became a chain of if/else branches each patching the last one's mistakes.

The instrumental answers the question directly, per song, per moment. Deriving the pitch reference from the accompaniment is validated in the literature — Hsieh et al. (Interspeech 2025) report r = 0.611 against human raters versus 0.364 and 0.232 for reference-free approaches.

See `backend/docs/NEW_VERSION.md` for the full design and `backend/docs/RESEARCH.md` for the evidence base.

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
  ├─ harmony ──────── CENS chroma of other+bass, tuning-aligned
  │                   → per-frame pitch-class prominence
  │                   + beat tracking for slack windows
  │
  └─ vocal ───────── CREPE (full capacity) → f0, confidence-gated
                     → note segments via hysteresis
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      key compliance              intonation accuracy
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

**Bass is excluded from the chromagram.** It measures 2.8–4.6× louder than `other` in real mixes, so summing them lets root notes drown out the chord voicings that actually distinguish one harmony from another.

## Evaluating changes

`backend/evaluate.py` is a label-free harness. Released vocals are by definition in-key, so the system should rank the notes actually sung above the same notes relabelled to any of the 11 wrong pitch classes:

```bash
cd backend && python evaluate.py          # all songs in test_songs/
python evaluate.py path/to/song.mp3       # one file
```

It runs three tests — global transposition (can the true key be identified), scattered perturbation (are individual wrong notes penalized), and a bleed check (is the signal real harmony or the vocal leaking into `other`). Stems are cached to `/tmp` after the first run, so iterating on scoring is fast. Currently the true key ranks #1 of 12 on all five test songs.

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
python main.py sample_songs/glimpse_of_us.mp3
python main.py sample_songs/glimpse_of_us.mp3 --no-report   # skip Gemini
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

- **Key compliance is octave-blind.** Chroma collapses octaves by construction, so a note sung a full octave off scores as a pitch-class match.
- **Key compliance is a relative indicator, not an absolute grade.** Released recordings score 66–94%; the same notes transposed to a wrong key score 17–28%. So a 66% does not mean a third of the notes were wrong — it means the accompaniment gave them moderate support. Compare within a song, not across songs. Details in `NEW_VERSION.md` §5.6–5.7.
- **No ground truth.** The thresholds mapping chroma rank and cents deviation to scores are calibrated against released recordings on five songs, not fitted to labelled data.
- **Beat slack is a fixed ±1 beat**, with no strong/weak beat weighting. Expect some false dissonance on heavily syncopated phrasing.
- **A cappella uploads cannot be scored for key compliance.** With no instrumental there is no harmonic reference; the system detects this and reports the axis as unmeasured rather than scoring the vocal against its own separation residue.
- **Layered vocals reduce coverage.** The `vocals` stem holds lead, backing, harmonies, and doubles together, and CREPE is monophonic — it can jump between lead and harmony mid-phrase. Ambiguous frames are dropped rather than guessed, and the reported coverage figure shows how much was skipped.
- **Intonation is still measured against a 12-TET grid**, so intentionally microtonal notes are penalized. Replacing this with a learned, style-aware target is the planned next step (`NEW_VERSION.md` §7).

## Tech stack

Python · Demucs (PyTorch) · CREPE (TensorFlow) · librosa · SciPy · NumPy · FastAPI · React · Vite · Tailwind · Recharts · Gemini
