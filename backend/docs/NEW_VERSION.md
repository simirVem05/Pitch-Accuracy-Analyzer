# NEW_VERSION.md — Two-Axis Redesign Plan

Rewritten 2026-08-10. Supersedes the earlier proposal draft. Companion to
`RESEARCH.md`, which holds the evidence base and citations. **Read `RESEARCH.md`
first** — this document assumes it and cites back by section (e.g. "RESEARCH §5.1").

**Status as of 2026-08-10:**

| Part | State |
|---|---|
| Stem separation, tuning, chroma pipeline | **built** (`separation.py`, `harmony.py`) |
| **Axis 1 — harmonic fit** | **built and measured** — see §5, which describes working code |
| **Axis 2 Phase A** — tuning-corrected cents | **built** (`scoring.py`) — see §6 |
| **Axis 2 Phase B** — learned intonation model | **not started** — §7 is still a proposal |
| Evaluation harness | **built** (`evaluate.py`) — see §5.8 |
| Old key/genre/allowlist system | **deleted** |

Sections 5 and 6.1 describe implemented behaviour. Sections 7 and 12 are forward-
looking. Section 2 is retained as the rationale for why the old system was replaced.

The input contract has changed: the user uploads a **full mixed song** (instrumental
+ vocals). Key and genre are no longer asked for or used. The system reports **two
independent scores** — harmonic fit and intonation precision — plus a coverage
figure. It never combines them into one number.

**Terminology change, 2026-08-11: "key compliance" is now "harmonic fit."** The old
name pointed at the wrong reference. The score never measured compliance with a
*key* — it measures how well a note is supported by the harmony sounding at that
moment, which is a per-frame chord-level question, not a global key-membership one.
"Compliance" also carried a rules-and-obedience connotation that actively misleads:
a low value most often means an adventurous-but-correct choice, not a violation.
Rename only — `RANK_LOW`, `RANK_HIGH`, `GLOBAL_WEIGHT` and the scoring math are
untouched. The code identifiers (`key_compliance`, `score_key_compliance`) still
carry the old name and are listed for rename in §9.

---

## Table of Contents

1. [The core reframe](#1-the-core-reframe)
2. [Why the current system fails](#2-why-the-current-system-fails)
3. [What the research validates](#3-what-the-research-validates--and-what-it-warns-about)
4. [Pipeline](#4-pipeline)
5. [Axis 1 — harmonic fit (implemented)](#5-axis-1--harmonic-fit-implemented)
6. [Axis 2 — intonation precision](#6-axis-2--intonation-precision)
7. [The learned intonation model](#7-the-learned-intonation-model)
8. [Reporting](#8-reporting)
9. [Build order](#9-build-order)
10. [What to keep, what to delete](#10-what-to-keep-what-to-delete)
11. [Honest limitations](#11-honest-limitations)
12. [Open decisions](#12-open-decisions)
13. [Appendix: current-system reference](#appendix-current-system-reference)

---

## 1. The core reframe

The project has been collapsing two independent questions into one score:

| Axis | Question | Ground truth source |
|---|---|---|
| **1 — Harmonic fit** | Do the notes chosen fit the music? | The song's own instrumental |
| **2 — Intonation precision** | Were those notes sung cleanly? | Deviation from a tuning-corrected, style-aware target |

These are orthogonal. A singer can nail a deliberate ♭5 dead-center — *excellent*
intonation, *low* harmonic fit, musically superb. A singer can also drift
20 cents flat on a plain root note — *correct* note choice, *poor* execution. One
number cannot express either case, and the current `on_key_score` collapses them
into a value that means neither.

Splitting the axes also **dissolves the architectural problem** that made the
current system brittle. Once "does this note fit" is a continuous harmonic-salience
question answered by the audio, there is nothing left for a hand-maintained
allowlist to decide. And once "was it sung cleanly" is judged against a target
learned from real recordings, there is nothing left for a hand-tuned perceptual
curve to guess at.

Two further consequences of the reframe:

- **Genre and key stop being inputs.** Both were proxies for "what fits here."
  The instrumental answers that directly, per song, per moment.
- **The two axes have different reliability, and that is not a defect.** Axis 1's
  published ceiling is r ≈ 0.611 with genuinely undetermined regions (§11.4), and
  its measured margins are thin (§5.9). Axis 2 should do considerably better.
  Keeping them separate keeps the soft number from contaminating the firm one.

---

## 2. Why the current system fails

Four failure modes, each traced to code.

### 2.1 The allowlist is unfalsifiable guesswork

`note_segmentation.py:57-65` hardcodes borrowed-note sets per genre:

```python
GENRE_EXTRA_INTERVALS = {
    "hip hop": {1, 6, 10},
    "pop": {8, 6},
    "r&b": {6, 1},
    ...
}
```

Three problems:

1. **No provenance.** Nothing documents where these intervals came from or how
   they were validated. They cannot be tuned against anything.
2. **Some are redundant.** `"country": {2, 3, 10}` — interval 2 is already
   diatonic in major, so that entry is partly a no-op.
3. **It answers the wrong question.** Genre does not determine which notes are
   correct in a *specific song*. Two R&B songs in B minor with different chord
   progressions have different correct-note sets. Genre is a weak prior over an
   enormous space.

The literature does not contain this approach (RESEARCH §7.2). Its absence is
informative: it does not work well enough to publish.

### 2.2 The rescue logic is compensating for the wrong model

`note_classification.py:115-135` adds contextual rescue — reclassifying
`dissonant` notes as `passing` / `neighbor` / `leading` from neighboring pitch
classes — plus a 65-cent near-miss window (`NEAR_ALLOWED_CENTS`).

This machinery exists to patch false positives generated by §2.1. The music theory
behind it is real (RESEARCH §6.1), but it is applied without the two inputs that
make non-chord-tone classification meaningful: **the sounding harmony** and **beat
position**. Per RESEARCH §6.1, whether a dissonance is a suspension or a mistake is
*defined* by strong-vs-weak beat placement. The current code has no beat tracker,
so it is guessing.

The result is a chain of if/else branches, each patching the previous one's errors.
Adding branches to fix the remaining cases is a losing strategy.

### 2.3 A440 is assumed, and it is measurably wrong

Nothing in the pipeline estimates tuning. `note_segmentation.py:18-27` hardcodes
the A440 equal-tempered grid:

```python
def hz_to_midi(hz): return int(np.round(12.0 * np.log2(hz / 440.0) + 69.0))
def midi_to_hz(midi): return 440.0 * (2.0 ** ((midi - 69) / 12.0))
```

Measured on this repo's own samples (RESEARCH §9.1), tuning offsets span
**−16 to +16 cents**. The scoring curve's entire "pro zone" is 0–25 cents
(`scoring.py:19`). **A 16-cent offset consumes 64% of that budget before the singer
sings a note** — and it applies uniformly, so a perfectly in-tune performance over a
sharp backing track reads as consistently sharp, indistinguishable from real error
(RESEARCH §4.2).

This is the cheapest high-value fix in the redesign, and it affects **both** axes
(§5.3, §6.1). **Fixed** — tuning is now estimated from the instrumental and fed to
both the chroma bin alignment and the Axis 2 target.

### 2.4 12-TET is the wrong target for the target genres

The scoring curve measures |cents| from the nearest equal-tempered semitone. Per
RESEARCH §5:

- A just major third sits **−13.7 ¢** from 12-TET.
- A just minor third sits **+15.6 ¢**.
- Leading tones get sharpened ~**+10 ¢** by skilled singers.
- Blue notes (♭3, ♭5, ♭7) are **deliberately microtonal** — "flattened by a
  variable microtone."

Measured median deviation on this repo's samples is **~22.5 cents** (RESEARCH §9.2);
a just major third alone accounts for 13.7 of those cents. The system cannot
distinguish "sang an expressive third" from "sang a sloppy third," and
R&B/soul/gospel — the stated target genres — are built precisely on the intervals a
12-TET grid penalizes.

**This is the deepest of the four failures**, and the reason Axis 2 eventually needs
a learned target rather than a better curve. Observed symptom: the current system
works acceptably on singers who sit flatly on top of notes, and degrades on
expressive singers whose inflections, vibrato, and slides produce large deviations
that still sound good. Vibrato and portamento are detected but their *quality* is
undecidable by threshold — a wide expressive vibrato and unstable pitch look alike
in cents.

---

## 3. What the research validates — and what it warns about

### Validated

**Hsieh et al., Interspeech 2025** (RESEARCH §7.1) implements the core idea — pitch
reference inferred from the accompaniment — and reports:

| Method | Pearson r vs. human raters |
|---|---|
| **Accompaniment-derived reference** | **0.611** |
| Pitch-interval, reference-free (Nakano-style) | 0.364 |
| Pitch-histogram, reference-free (Gupta-style) | 0.232 |

Deriving harmony from the instrumental nearly doubles agreement with human judgment.
The space is almost empty: exhaustive DBLP queries for singing-assessment ×
chord-detection return **zero** results (RESEARCH §7.2).

Hsieh et al. used **key** — a global property — not frame-level harmony. A
chroma-level approach is a step *beyond* published work.

For Axis 2, **Deep Autotuner** (Wager et al. 2019, RESEARCH §7.3) validates the
learned approach: it predicts pitch correction "from the relationship between the
spectral contents of the vocal and accompaniment tracks," explicitly where no score
exists. That is the design §7 below adopts.

### Four warnings

**1. The instrumental tells you note-fit, not note-quality.** Zhang et al. (ISMIR
2021, RESEARCH §7.3) used the accompaniment as a metric-learning anchor and it
**underperformed**. Their finding: the accompaniment "does not provide details on
singing, but only helps with judging the rhythm and tonality." The dossier calls
this its most important cautionary result. It argues directly for the two-axis
split — the instrumental feeds Axis 1 *only*, and Axis 2 uses it as conditioning
context, never as the sole reference.

**2. "Separate every instrument" is impossible.** Production separation gives a
fixed 4 stems; `other` stays polyphonic (RESEARCH §2.1). There are no
per-melodic-instrument stems to run CREPE on. Chroma makes this a non-issue rather
than a problem to solve (§5.1).

**3. There is a ceiling on Axis 1.** r = 0.611 is good, not solved. Quartal
voicings have no definite root; pedal points, chromatic mediants, and
sustained-tension gospel technique all produce notes that clash yet are correct
(RESEARCH §6.3). Harmonic context cannot fully determine correctness even when
perfectly identified. Also RESEARCH §6.2: *no current MIR system automatically
classifies "intentional non-chord tone vs. pitch error" in the general case.*

**4. A learned note-set generator is not viable, and this was considered.** The
idea of training a model to emit the set of on-key notes given melody + vocals was
evaluated and rejected on three grounds, recorded here so it is not re-proposed:

- **No label exists.** RESEARCH §10.7: the project has no labeled data and any
  accuracy claim is "currently unfalsifiable." Hsieh et al. resorted to commercial
  Tunebat labels for a *global 12-class* target; a time-varying 12-dimensional
  multilabel target is far harder to annotate. Deriving labels from chroma instead
  means distilling a function you already have, with added error and less
  interpretability.
- **Part of the target is genuinely undetermined**, not merely unknown (RESEARCH
  §6.3) — in quartal harmony the chord-tone distinction *collapses*. A perfect
  annotator could not produce a clean label.
- **Feeding the vocal in leaks the answer.** In released music the sung notes are
  almost always the correct ones, so the cheapest way to minimize loss is to learn
  "whatever was sung is allowed" — systematically forgiving of exactly the errors
  the system exists to catch. Balanced loss on a set-valued target also collapses
  toward "output the diatonic scale," which reproduces the system being replaced.
  *(Inference, not a repo finding — but it should be assumed until measured.)*

Emitting a *set* also reintroduces the binary in/out decision as a learned if/else
that cannot be read or debugged. Chroma's advantage is that it is **continuous**
(RESEARCH §3.2). ML belongs on Axis 2, where labels are free (§7).

---

## 4. Pipeline

Axis 1 is **implemented and measured** — the diagram below is as-built, and §5 is
written as a description of working code. Axis 2 Phase A is implemented; Phase B
(§7) is not.

```
full mixed song (user upload)
  │
  ▼
Demucs htdemucs → 4 stems: vocals / other / bass / drums
  │               (MIT incl. weights; ~3-8 min CPU, 1-2 min MPS)
  │
  ├─ drums ──── used ONLY for beat tracking, never for chroma (§4.2)
  ├─ bass ───── NOT used in scoring at all (§5.2) — separated and cached
  │             only so evaluate.py can run its bleed check
  │
  ├─ TUNING ── librosa.estimate_tuning(other) ──► offset (semitones)
  │            feeds chroma bin alignment AND the Axis 2 target
  │                                                         │
  ├─ HARMONY ─ chroma_cens(other, tuning=offset) ──► (12, n_frames)
  │            │                                            │
  │            ├─► per-frame RANK ──────► local support      │
  │            └─► whole-song mean rank ─► global support    │
  │            + librosa.beat.beat_track(drums) ──► tempo → slack width
  │                                                         │
  └─ VOCAL ─── CREPE(vocals, capacity="full") ──► f0, confidence
               confidence + f0-range gate ──► voicing mask ──► coverage
               hysteresis segmentation ──► note segments
                                                            │
                        ┌───────────────────────────────────┴──────────┐
                        ▼                                              ▼
              AXIS 1 — harmonic fit                     AXIS 2 — intonation precision
              local^0.75 × global^0.25                  |cents| from tuning-corrected
              local = best of 3 window                  target, core-trimmed median
                      positions (±1 beat)               → perceptual curve
              → RANK_LOW/HIGH → 0..1                    Phase A: implemented
              (octave-blind BY DESIGN, §5.3a)           Phase B: not built (§7)
                        └───────────────────────────────┬──────────────┘
                                                        ▼
                            duration-weighted means, reported SEPARATELY,
                            plus coverage and tuning offset (§8)
```

**Module map.** `separation.py` (stems) → `harmony.py` (tuning, chroma, beats,
Axis 1) → `preprocess.py` (CREPE, voicing, coverage) → `note_segmentation.py`
(segments, tuning-aware targets) → `scoring.py` (Axis 2) → `main.py`
(orchestration, metrics, Gemini) → `api.py`. `evaluate.py` is the label-free test
harness (§5.8).

### 4.1 Stems

`demucs` 4.1.0, model `htdemucs_ft`, **MIT-licensed including weights** (RESEARCH
§2.5). Consume `vocals` (the performance), `other` (harmony), `bass` (roots).

### 4.2 Do not split "vocals vs. everything else"

A 2-stem split puts **drums** in the accompaniment. Drums are broadband — a snare or
cymbal dumps energy across the whole spectrum, which folds into all 12 chroma bins
and raises the noise floor uniformly. Every pitch class then looks moderately
present, so every note looks moderately in-key, and Axis 1 loses its
discriminative power.

**Use the 4-stem model and compute chroma on `other` (+ `bass`), discarding
`drums`.** Same runtime, materially cleaner signal.

### 4.3 The two axes are architecturally independent, not temporally parallel

Both axes depend on note segments, which depend on CREPE on the vocals, which
depends on separation. The real graph:

```
Demucs           3–8 min CPU   ← dominates
  ├── CREPE(vocals) → segs      ← both axes wait on this
  └── chroma(other+bass)  ~s    ← the only genuine overlap
            ↓
  Axis 1 (aggregate + lookup)   fast
  Axis 2 (model inference)      fast
```

The expensive stages are **shared prerequisites and strictly sequential**. Only
chroma overlaps CREPE — seconds inside a multi-minute job. Write the axes as two
independent, separately testable functions because that is cleaner, **not** for
speed. Do not add threading.

### 4.4 Dependency and runtime consequences

- **Demucs is PyTorch; CREPE is TensorFlow.** Neither `torch` nor `demucs` is
  currently installed (verified 2026-08-10 — RESEARCH §2.5 only confirmed demucs
  exists on PyPI). Adding torch + torchaudio beside TF 2.20 usually works but
  numpy/CUDA pinning between the two frameworks is a known pain point, and
  `requirements.txt` is a fully-pinned `pip freeze`, so a conflict surfaces as a
  wall of version errors. The venv is already 2.1 GB and this plausibly doubles it.
  **Install into a scratch venv and confirm both frameworks import before
  committing.**
- **Runtime is the main UX cost.** RESEARCH §2.5: ~15–30 s GPU, **1–2 min Apple
  MPS**, **3–8 min CPU**, 10–20 min on entry CPUs. On Apple silicon, verify demucs
  actually uses MPS rather than silently falling back to CPU — that is the
  difference between 90 seconds and 8 minutes.
- **`/analyze` currently blocks the event loop** (`api.py:27`, `async def` around
  synchronous CPU work). At Demucs runtimes this freezes the whole app per request.
  Minimum fix: drop `async` so Starlette uses its threadpool. Job handling remains
  an open decision (§12.3).

---

## 5. Axis 1 — harmonic fit (implemented)

**Status: built, measured, working.** This section describes what the code does,
not what was proposed. Implemented in `harmony.py`.

### 5.0 The score in one place

For each note segment, with `pc` = its pitch class:

```
local  = max over 3 window positions of  mean(rank[pc, window])
             positions: the note's own span, that span shifted one beat
             earlier, and shifted one beat later
global = rank of pc in the whole-song mean chroma profile

combined     = local^0.75 * global^0.25           # GLOBAL_WEIGHT = 0.25
harmonic_fit = clamp01((combined - 0.50) / (0.75 - 0.50))   # RANK_LOW/HIGH
```

Reported metric is the **duration-weighted mean** of `harmonic_fit` over all
scorable notes, so a grace note does not count as much as a sustained one.

`pc` is a **pitch class**, so octave is deliberately not part of this score (§5.3a).

**Exactly two features feed the score: local chroma and global chroma, both from
the `other` stem.** No bass, no chord labels, no beat-position weighting, no
genre, no key input, no allowlist.

Constants live at the top of `harmony.py`: `GLOBAL_WEIGHT`, `RANK_LOW`,
`RANK_HIGH`, `DEFAULT_SLACK_S`, `MIN_HARMONY_TO_VOCAL_RMS`.

### 5.1 Why chroma, and why CENS specifically

A chromagram collapses every octave of a spectrogram onto one 12-element vector
per frame: all C energy (65, 131, 262, 523 Hz…) sums into one bin. Output is
`(12, n_frames)` — "how much of each pitch class is sounding right now."

Three properties, none of which the old allowlist had:

- **Inherently polyphonic.** No separator yields individual instruments — guitars,
  synths, keys and strings all land mixed in `other` (RESEARCH §2.1), which a
  monophonic tracker cannot read. Chroma never commits to a single note, so chords
  need no special handling.
- **Continuous, not binary.** The scorer asks "how much support did this note
  have," not "is it allowed." Nothing to branch on, no table to maintain.
- **Robust to separation artifacts.** Pooling across octaves and time covers
  spectral holes and moderate bleed (RESEARCH §2.3). Vocal f0 has no such
  redundancy, which is why the harmony path tolerates messy stems and the
  intonation path does not.

**CENS over CQT chroma** (`librosa.feature.chroma_cens`): measured better contrast
between prominent and absent pitch classes. Margin +0.043 vs. +0.041 — a small
edge, but free.

### 5.2 Bass is excluded, and this was measured

The `bass` stem is separated and cached but **feeds nothing in the score.**

Summing it into the chroma actively hurts. Bass measures **2.8-4.6x louder than
`other`** across the test mixes, so summing waveforms lets root notes dominate the
chromagram and drowns out the chord voicings that distinguish one harmony from
another:

| Chroma source | Margin over best wrong key | True key ranked #1 |
|---|---|---|
| `other + bass` | +0.014 | 2/5 |
| **`other` only** | **+0.043** | **4/5** |
| `bass` only | −0.010 | 2/5 |

**Bass root motion as a separate feature was never implemented.** RESEARCH §3.2
Tier 2 recommends it — bass is monophonic by construction, so CREPE on it is
correct there, and root motion resolves inversions and rootless voicings that
chroma cannot. It remains a reasonable next feature, but note that bass-only
chroma was the weakest signal measured (−0.010), which tempers the expectation.
Do not confuse the two uses: bass *in the chromagram* is tested and harmful;
bass *as a root-motion feature* is untested.

### 5.3 Tuning correction

`librosa.estimate_tuning` runs on `other` and the offset is passed to
`chroma_cens(tuning=...)`, so bins align to the song's actual grid rather than
A440. Without it, energy lands between bins and smears across neighbours, blurring
the profile exactly where it needs to be sharp. Measured offsets on real songs run
+1 to +35 cents, so this is not hypothetical. The same offset feeds Axis 2.

### 5.3a Octave is deliberately ignored — decided 2026-08-11

**Octave checking will not be implemented.** Earlier revisions of this document
listed it as step 14 and called it a correctness bug. **That was wrong, and the
decision is reversed.**

Chroma is octave-blind by construction, so a note sung an octave away from the
recorded lead scores as a pitch-class match. That is now the *intended* behaviour,
not a gap. The reasoning:

- **Axis 1 asks whether the note fits the harmony, and harmony is octave-invariant.**
  A chord is a set of pitch classes. If a pitch class is supported by the sounding
  harmony, it is supported in every octave — a C over an F major chord is the fifth
  whether it is sung at C3 or C5. Penalizing the octave would answer a question this
  axis was never scoped to ask.
- **There is no correct octave to check against.** The only available reference is
  the octave the original artist happened to sing in, which is a property of *their*
  range, not of the music. A baritone covering a soprano lead sings the whole song an
  octave down and is not wrong; that is transposition, a normal and universal
  practice. An octave check would systematically penalize every singer whose range
  does not match the artist's, which is most of them.
- **It would silently double as an unrequested range judgment.** Marking octave
  displacement as error conflates "sang the wrong note" with "has a different voice
  type." The first is a mistake worth reporting; the second is not a mistake at all.

The narrow case this gives up is a singer who jumps an octave *mid-phrase*
unintentionally. That is real, but it is an execution and phrasing artifact rather
than a note-choice error, and Axis 1 is the note-choice axis. If it is ever worth
surfacing, it belongs as a **separate, separately-reported observation** — in the
same reported-not-judged category as vibrato and portamento (§7.2) — and never
folded into the harmonic-fit score.

Consequence for the documented behaviour: harmonic fit is a **pitch-class** measure,
by design and permanently. It should be described that way rather than caveated as
octave-blind, and §11 no longer lists this as a limitation.

### 5.4 Beat slack: implemented as a shifting window

`librosa.beat.beat_track` runs on the drums stem, and **only the tempo is
consumed** — as `slack_s = 60 / tempo`, one beat. `beat_times` is computed and
stored on `HarmonicContext` but nothing reads it.

The slack **shifts a fixed-width window; it does not widen it.** This matters and
was a real bug in the first implementation. Widening dilutes a short note into
several beats of unrelated harmony and raises the score for *every* pitch class
equally, so it forgives wrong notes exactly as much as anticipated ones. Shifting
asks the intended question: was this note supported by the chord just before, or
just after? That is what accommodates anticipations and melisma across chord
changes (RESEARCH §6.4).

**Strong/weak beat weighting is NOT implemented.** RESEARCH §6.1 holds that beat
position is "definitional, not decorative" — the same pitch is a suspension on
beat 1 and plausibly an error on beat 4. `beat_times` is already available, so
this is the cheapest remaining improvement. Consequence of its absence: expect
false dissonance concentrated on syncopated phrasing, the signature of the target
genres. **Do not debug that as a chroma problem; it is a timing problem.**

### 5.5 Local + global support, gated by a geometric mean

Local prominence alone conflates two musically different cases:

| Global (song key) | Local (this moment) | Interpretation |
|---|---|---|
| high | high | chord tone — clearly correct |
| **high** | **low** | in key, outside the current chord — passing tone, suspension, anticipation. **Musically fine.** |
| low | low | outside the key entirely — likely a real error |
| low | high | chromatic or borrowed-chord moment |

A genuinely wrong note is weak on **both** terms; a passing tone is strong on one.
That is the distinction the second row makes available and local-only scoring
cannot express.

**Geometric mean, not a weighted sum**, so the terms *gate* rather than
compensate: strong key membership cannot fully rescue a note the current harmony
rejects outright. A sum would let a high global score drag a locally-rejected note
up linearly.

| Config | Margin | True key #1 | Scattered-wrong-note sensitivity |
|---|---|---|---|
| local only | +0.043 | 4/5 | +0.0327 |
| **geometric, `GLOBAL_WEIGHT` = 0.25** | +0.041 | **5/5** | **+0.0413** |

`GLOBAL_WEIGHT` in 0.20-0.30 performs equivalently, so 0.25 sits mid-plateau
rather than on a fitted peak. Note the trade: *margin* is flat while *placement*
and *perturbation sensitivity* both improve — the split does not sharpen key
identification, it makes the score better at penalizing individual wrong notes,
which is what a user actually experiences.

### 5.6 Calibration, and why the absolute number means little

`RANK_LOW = 0.50`, `RANK_HIGH = 0.75` map combined support onto 0-1.

Calibrated against the only available anchor: **commercially released tracks,
whose vocals are by definition in-key.** Measured combined support is 0.68-0.82
for the real vocal and ~0.55 for the same notes transposed. That band maps
released recordings to 72-100% and wrong keys to 17-28%.

**These two constants are a presentation choice, not a measurement.** The same
audio reports 66% or 85% depending on where they sit. Consequences:

- Harmonic fit is a **relative** indicator. Compare moments within a song, or
  the same song before and after a change. Do not read "66%" as "a third of the
  notes were wrong."
- Cross-song comparison is weak: arrangement density shifts the whole scale.
- Any recalibration must be re-validated with `evaluate.py`, not by eye.

Current end-to-end scores: Jingle Bells 93.7%, Party in the U.S.A. 74.6%, Work
71.0%, One Dance 67.4%, Billie Jean 66.0%. The ordering is plausible — simple
arrangement highest, densest lowest.

### 5.7 A cappella uploads are refused, not scored

An a cappella still yields a nominal `other` stem, but it holds only separation
residue of the voice — scoring against it compares the vocal to its own leakage
and returns a confident, meaningless number (measured: 78% before the guard).

`MIN_HARMONY_TO_VOCAL_RMS = 0.04` gates on the `other`-to-vocal RMS ratio. When it
fails, harmonic fit is `None` end-to-end: metrics report `null`, the Gemini
prompt switches to a branch that states the axis was not measured, and the UI
shows "—". Axis 2 still scores normally.

Real full mixes measure 0.29-0.60 on this ratio, so there is an order of magnitude
of headroom. **Caveat: the threshold was validated against a cappellas when
chroma used `other+bass`; after the switch to `other`-only it has been confirmed
safe for full mixes but not re-confirmed against an actual a cappella.**

### 5.8 How to evaluate any change: `evaluate.py`

Label-free harness. Released vocals are by definition in-key, so the system should
rank the notes actually sung above the same notes relabelled to a wrong pitch
class. No human annotation needed.

```bash
cd backend && python evaluate.py          # all of test_songs/
python evaluate.py path/to/song.mp3       # one file
```

Perturbation is **symbolic** — only the integer `pitch_class` changes. Axis 1
reads nothing but pitch class and time bounds, so relabelling fully simulates "a
different note was sung here," and it avoids re-running separation and CREPE
(which would change segmentation and confound the result). All 12 transpositions
score in milliseconds. Stems cache to `/tmp` after the first run.

Three tests:

1. **Transposition** — shift every note by the same interval, all 11 wrong
   values. Reports separation from mean-wrong and **margin over best-wrong**.
   *Margin is the honest number*; mean-wrong flatters the system. Billie Jean's
   failure was visible only in margin.
2. **Perturbation** — shift a random 20% of notes by ±1-2 semitones, rest
   correct. Seeded, averaged over 5 draws. Asks whether *individual* wrong notes
   are penalized in an otherwise-correct performance — much closer to a real user
   than a global key change, and arguably the more relevant metric.
3. **Bleed** — score on `other+bass`, `other` alone, `bass` alone. Vocal leaks
   into `other` at ~9.4 dB SDR, raising the worry that notes look supported by
   their own leakage. Bass sits mostly below vocal range so carries far less.
   **Result: separation survives on bass alone (+0.027 to +0.106) on all five
   songs — the signal is real harmony, not circular.**

Transpositions are not equally wrong: nearby keys share scale notes and are
genuinely harder to reject. Measured ordering roughly tracks key relatedness
(tritone worst at 0.55, fifth 0.64, whole-tone 0.68), which is itself mild
evidence the feature reads real harmony.

### 5.9 Current state, honestly

**After dropping bass and adding the global term, the true key ranks #1 of 12 on
all 5 test songs** (from 2/5 at first implementation).

| Song | Real | Wrong (mean) | Margin | Placement |
|---|---|---|---|---|
| Jingle Bells | 0.818 | 0.546 | +0.121 | 1/12 |
| Party in the U.S.A. | 0.736 | 0.569 | +0.041 | 1/12 |
| Work | 0.735 | 0.562 | +0.008 | 1/12 |
| One Dance | 0.695 | 0.543 | +0.006 | 1/12 |
| Billie Jean | 0.680 | 0.552 | +0.031 | 1/12 |

**The margins are thin.** Work and One Dance clear the best wrong key by <0.01.
Placement is perfect but not robust — a different song could easily rank 2nd. This
is consistent with the published r ≈ 0.611 ceiling (§3) and with RESEARCH §6.3's
irreducible ambiguity.

**One methodological lesson worth carrying forward:** feature choice dominated
everything. Dropping bass tripled the margin, while every combination-level tweak
swept (log-compression, HPSS, beat-sync aggregation, CQT-vs-CENS) moved it by
≤0.002. Prefer new information over new ways of combining existing information.

**Ranked next steps for Axis 1:**

1. **Strong/weak beat weighting** (§5.4) — `beat_times` already computed,
   strongest theoretical grounding (RESEARCH §6.1).
2. **Chord estimation** instead of raw chroma (§5.10) — biggest potential gain,
   most likely to fix dense arrangements, new dependency.
3. **Bass root motion** (§5.2) — tier-2 recommended but the one weak signal
   measured.
4. **Fit the local/global weights** (§12.8) — cheap, but combination-level, so
   expect little.
5. **A labelled set** (§12.1) — gates knowing whether any of the above helped in
   human terms rather than transposition terms.

The octave check that previously headed this list has been **removed by decision**,
not deferred — see §5.3a.

### 5.10 Discrete chord labels are presentation only

## 6. Axis 2 — intonation precision

Two phases. **Phase A ships first and exists to be the baseline Phase B must
beat.** Building the model first leaves nothing to compare against — the same
unfalsifiability trap as RESEARCH §10.7.

### 6.1 Phase A — tuning-corrected analytic cents

Cents deviation from a target that is:

- **Tuning-corrected** — the §5.2 offset removed before any measurement.
  Non-negotiable, and per §2.3 the single cheapest real fix in the redesign.
- **Measured off the core-trimmed median**, keeping the existing scoop/release
  exclusion (`note_segmentation.py:136-162`).
- Optionally **interval-aware** (thirds −13.7 ¢, leading tones +10 ¢ per RESEARCH
  §5). Start without it. This is the most overfit-prone piece and Phase B is meant
  to subsume it.

Phase A remains subject to §2.4 — it is a 12-TET grid, wrong for these genres by
construction. It is a *baseline*, not the destination.

### 6.2 Fix the target-note derivation first

Independent of either phase, `segment_notes` (`note_segmentation.py:174`) sets
`current_note = hz_to_midi(freqs[i0])` — the nearest semitone to the segment's
**first voiced frame**, which is exactly where a scoop lives. Every deviation for
that note is then measured against a target that may be the wrong semitone,
including the core-trimmed median specifically designed to ignore onsets.

Derive `target_note` from the segment's core-trimmed median instead. Small change,
outsized effect on measurement validity, and it poisons Phase B's training labels
if left unfixed.

---

## 7. The learned intonation model

### 7.1 Why ML fits Axis 2 when it does not fit Axis 1

The asymmetry is entirely about labels. For note choice, no label exists and part
of the target is undetermined (§3, warning 4). For intonation, **the label is
free**: take a released recording, apply a known pitch shift to the vocal, and the
shift amount *is* the ground truth. You know it because you applied it.

This also addresses §2.4 at the root. Because the target is *the released vocal*
rather than a 12-TET grid, expressive practice is baked into the label: a just
major third sitting 14 cents below equal temperament appears at 0-shift throughout
training, so the model learns that is the target rather than a 14-cent error. Same
for vibrato excursion and portamento — present in the 0-shift data, therefore
learned as *not* deviation.

That dissolves the "expressive singing looks like bad singing" problem without a
single threshold constant, which is the correct response to §2.4 — the variation
across singing styles is too large for any if/else chain to cover.

### 7.2 What it does not do

**It makes vibrato and portamento not-penalized. It does not make them
assessable.** The model only ever sees good vibrato at 0-shift and pitch-shifted
good vibrato at X-shift. What it learns is that vibrato is *irrelevant to its
prediction* — it ignores vibrato rather than judging it. "Was that good vibrato?"
is a separate model requiring human quality labels that do not exist here.

This is still a large win over the current system, which actively penalizes both.
Keep detecting them and reporting them as stylistic observations, unjudged — the
current philosophy is right.

### 7.3 Design

- **Target: regress a continuous offset, not a binary flag.** Corrupted-vs-clean
  classification tells you *whether* something is off, not *how far*, so no score
  can be built from it. Apply a continuous random offset (e.g. uniform ±50 cents),
  **per note segment rather than per song**, and train to regress the value in
  cents. Error = |predicted − actual|.
- **Condition on the accompaniment.** Feed the instrumental context alongside the
  vocal, per Deep Autotuner (RESEARCH §7.3). This absorbs the tuning-offset problem
  for free on this axis, since the model learns the vocal-to-instrumental
  relationship rather than assuming A440.
- **Predict per note segment.** Composes with everything else in the pipeline.
  Per-frame needs its own aggregation layer. **Decide before generating training
  data**, since it changes the generation code.
- Output is **deviation in cents**, which still needs a deviation→score mapping.
  The difference from today is that the input is contextually correct rather than
  measured off a grid that is wrong for these genres, so a simple honest curve
  becomes defensible where it was previously compensating for a broken measurement.

### 7.4 The artifact-leakage trap — get this right from the start

Pitch-shifting leaves its own fingerprint (resampling and phase-vocoder artifacts).
A model can score perfectly by learning "this frame has shift artifacts" while
understanding nothing about pitch, then collapse on real uncorrected vocals.

**Defense: include zero-shift controls that still pass through the full shifting
pipeline**, so artifacts are present with a label of 0 cents. Artifact detection
then carries no information and the model must actually learn pitch.

Do this from the first training run. Retrofitting it invalidates every prior
result.

### 7.5 Training-data caveats

- **The label means "deviation from the released vocal," not "deviation from
  correct."** This works because commercial vocals are pitch-edited, so the release
  approximates the intended target. The flip side: on heavily autotuned references
  the model learns to reward machine-perfect pitch, and an expressive human reads
  as inaccurate. Audit what the training mix is made of.
- **Training on separated stems means training on lead-plus-harmony soup**
  (RESEARCH §2.4 — the `vocals` stem contains lead + backing + doubles + ad-libs,
  and no production-quality separator splits them). Prefer clean multitrack sources
  for training where obtainable.
- **Inference is a domain shift.** Scoring a user's cover applies the model to a
  vocal it has no reference for. That is what accompaniment conditioning is for,
  but it must be *measured*, not assumed.

### 7.6 Evaluation

The corruption setup is itself the test harness this project has never had: known
input, known answer, measurable error. Report mean absolute error in cents on
held-out songs, and separately on **zero-shift controls** (where any nonzero
prediction is pure false-positive rate). Phase B ships only if it beats Phase A on
held-out data.

---

## 8. Reporting

**Report the two axes separately. Do not average them.** Averaging recreates
exactly the problem this redesign exists to fix: a deliberate ♭5 sung dead-center
is low Axis 1 and excellent Axis 2 — musically superb — and averages to
"mediocre," which describes nothing. A singer who picked safe notes and sang them
all 30 cents flat averages to the same value. The entire diagnostic content is in
the split. The axes also differ in reliability (§1), so averaging contaminates the
firm number with the soft one.

Report three things:

1. **Harmonic fit** (Axis 1)
2. **Intonation precision** (Axis 2)
3. **Coverage** — what fraction of the vocal was confidently scorable

**Coverage is load-bearing**, and more so given §12.2. The `vocals` stem contains
lead + backing + harmonies + doubles, and CREPE — a monophonic tracker — will
**jump between lead and harmony mid-phrase** on produced tracks (RESEARCH §2.4,
the redesign's hardest open problem). The mitigation adopted here is to
**confidence-gate hard and drop ambiguous frames**: score fewer notes rather than
score them wrong. That is only honest if the user sees how much was skipped. A
track scored on 40% of its vocal must say so.

If a single headline is needed for UX, show both side by side and let the Gemini
report explain the interaction — "adventurous note choices, clean execution" is
more useful than 74%.

---

## 9. Build order

### Done

| # | Step | Where |
|---|---|---|
| 1 | Tuning-offset estimation | `harmony.py:estimate_tuning_semitones` |
| 2 | Demucs 4-stem integration | `separation.py` |
| 3 | Fix target-note derivation (was anchored on the first voiced frame, where scoops live) | `note_segmentation.py:_finalize_segment` |
| 4 | Raise CREPE capacity off `"tiny"`; delete the `inspect` indirection that made it unreachable | `preprocess.py` |
| 5 | Fix segment-boundary bug (candidate frames miscounted when unvoiced frames interleaved a transition) | `note_segmentation.py:segment_notes` |
| 6 | Chroma harmonic profile | `harmony.py:build_harmonic_context` |
| 7 | Axis 1 scoring; allowlist + rescue chain deleted | `harmony.py:score_key_compliance` |
| 8 | Axis 2 Phase A (tuning-corrected cents) | `scoring.py` |
| 9 | Two-axis reporting + coverage; key/genre inputs dropped front to back | `main.py`, `api.py`, frontend |
| 10 | Beat tracking for ±1 beat slack (tempo only; `beat_times` unused) | `harmony.py` |
| 11 | Evaluation harness | `evaluate.py` |
| 12 | A cappella guard | `harmony.py:MIN_HARMONY_TO_VOCAL_RMS` |
| 13 | Local + global support split | `harmony.py:GLOBAL_WEIGHT` |

### Next, in recommended order

| # | Step | Effort | Why this order |
|---|---|---|---|
| 14 | **Rename `key_compliance` → `harmonic_fit`** through code and UI | small | Doc renamed 2026-08-11; identifiers still carry the old name, so doc and code now disagree |
| 15 | **Re-verify the a cappella guard** with an actual a cappella (§5.7) | small | Threshold was validated pre-`other`-only; a false trip silently drops Axis 1 |
| 16 | **Strong/weak beat weighting** (§5.4) | medium | `beat_times` already computed; strongest theory backing (RESEARCH §6.1) |
| 17 | **Chord estimation** instead of raw chroma (§5.10) | large | Biggest potential gain, most likely to fix dense arrangements; new dependency, benchmark first |
| 18 | **Bass root motion** as a third feature (§5.2) | medium | Tier-2 recommended, but bass-only chroma was the weakest signal measured |
| 19 | **Labelled set** (§12.1) | large | Gates knowing whether 16-18 helped *in human terms* rather than transposition terms |
| 20 | **Axis 2 Phase B** — corruption-trained model (§7) | large | Research project; stays behind the Phase A interface until it beats Phase A |

The octave check formerly listed here as step 14 was **removed by decision**, not
reordered — see §5.3a.

Step 14 is a rename with no scoring change, so `evaluate.py` output must be
**byte-identical** before and after; that is the check. Steps 16-18 are measurable
today with `evaluate.py`. Step 19 is what makes "more accurate" mean anything beyond
the transposition proxy.

**Rename surface for step 14**, so it is done in one pass rather than piecemeal:
`score_key_compliance` and the `key_compliance` segment key (`harmony.py`); the
`key_compliance` metrics field and its Gemini prompt branch (`main.py`); the
`hasKey` / `keyCompliance` locals and the "Key compliance" card label
(`MetricsPanel.jsx`); the "key compliance is reported separately" chart caption
(`PerformanceChart.jsx`); and the `metrics.json` field name, which is a
consumer-visible contract — `outputs/` is gitignored, so stale files on disk will
still carry the old key.

**Methodological lesson from steps 6-13, worth carrying into 14-20:** feature choice
dominated. Removing bass from the chromagram tripled the margin, while every
combination-level tweak swept (log-compression, HPSS, beat-sync aggregation,
CQT-vs-CENS) moved it ≤0.002. **Prefer new information over new ways of combining
existing information** — which is also why fitting the local/global weights (§12.8)
is ranked below chord estimation despite being far cheaper.

---

## 10. What to keep, what to delete

### Keep — the DSP core is sound

- **CREPE** for vocal f0 (RESEARCH §2.3). **But raise `model_capacity` from
  `"tiny"`** (`preprocess.py:159`), the smallest and least accurate variant.
  `_run_preprocess`'s `inspect` indirection (`main.py:160-176`) passes only
  `audio_path`, making the parameter **unreachable** from `run_backend` — delete
  the indirection.
- **Voicing mask + morphological smoothing** (`preprocess.py:110-123`) — the
  fill-gaps → drop-short-runs → fill-gaps pipeline is well-built, and
  `NaN`-as-unvoiced is a clean cross-module contract. Its confidence gate becomes
  load-bearing per §8.
- **Hysteresis note segmentation** (`note_segmentation.py:165`) — prevents jitter
  and spurious flips. Genuinely good. (One latent bug: `finalize(i - cand_count)`
  at `:232` assumes candidate frames were contiguous, but the loop skips `NaN`
  frames without resetting the count, so boundaries land wrong when unvoiced frames
  interleave a transition.)
- **Core-median trimming** — correctly excludes scoops and releases.
- **Vibrato / portamento detection as reported-not-judged** — correct treatment,
  and §7.2 confirms nothing here will grade their quality. Keep the philosophy.
- **Single `run_backend` shared by API and CLI** — good structure.
- **Gemini report as non-essential**, degrading gracefully (`main.py:144-157`).
  Its role grows under §8: explaining the interaction between two axes.

### Delete

- `GENRE_EXTRA_INTERVALS` and `build_allowed_pitch_classes`
  (`note_segmentation.py:57-100`) — replaced by measured salience.
- The classification rescue chain and `NEAR_ALLOWED_CENTS`
  (`note_classification.py:100-135`) — it existed to patch the allowlist's false
  positives.
- The hand-tuned perceptual curve (`scoring.py:16-38`) — replaced by Axis 2.
- Key and genre inputs, front to back: `UploadForm.jsx` selects, `api.py:27` form
  fields, `_parse_key` / `_normalize_genre` (`main.py:20-47`).
- The unused `activation` array (`main.py:192`) — computed, NaN-masked, never read.

Genre may survive as a *weak* prior on blue-note tolerance in Axis 2 Phase A. It
does not survive as a gate.

---

## 11. Honest limitations

1. **No human-referenced ground truth for Axis 1.** `evaluate.py` gives a
   *label-free* proxy — the true key should outrank transpositions — and that proxy
   is currently satisfied on 5/5 songs. But **r against human raters has never been
   measured and cannot be** without labelled data (§12.1). Do not confuse "ranks
   the true key first" with "agrees with a listener."
2. **Axis 1 margins are thin** (§5.9). Two of five songs clear the best wrong key
   by <0.01. Placement is perfect but not robust; a new song could easily rank 2nd.
3. **Harmonic fit is a relative indicator, not an absolute grade** (§5.6). The
   reported percentage moves with two calibration constants that are a presentation
   choice, not a measurement. Compare within a song, not across songs.
4. **Axis 1 ceiling ≈ r 0.6, not 1.0.** Best published accompaniment-derived
   result. Quartal harmony, pedal points, chromatic mediants, and intentional
   sustained tension remain unresolvable (RESEARCH §6.3).
5. **Vocal harmonies degrade Axis 2** on exactly the commercially produced tracks
   users are most likely to upload (RESEARCH §2.4). Mitigated by hard
   confidence-gating and honest coverage reporting (§8), not solved.
6. **CPU runtime is a real UX problem** — 3–8 min per song, against an endpoint
   that currently blocks the event loop (§4.4).
7. **Separation artifacts are borderline for f0.** `htdemucs_ft` vocals ≈ 9.4 dB
   SDR against a ~10 dB literature threshold for reliable pitch tracking (RESEARCH
   §2.3). Unmeasured on this repo's material; RESEARCH §10.3 notes it is cheaply
   measurable locally and nobody has published it.
8. **Harmonic fit is a pitch-class measure and ignores octave by design** (§5.3a).
   This is listed here as *scope*, not as a defect — the axis asks whether a note
   fits the sounding harmony, and harmony is octave-invariant. What it genuinely
   gives up is detecting an unintentional mid-phrase octave jump, which would have
   to be reported separately if it is ever wanted.
9. **No strong/weak beat weighting** (§5.4). ±1 beat of slack is implemented, but
   beat *position* is unused despite `beat_times` being computed. Expect false
   dissonance on syncopated phrasing, the target genres' signature.
10. **The a cappella guard has not been re-verified** since chroma switched to
   `other`-only (§5.7). Full mixes are confirmed safe with an order of magnitude of
   headroom, but a false trip would silently drop Axis 1 to "not measured."
11. **RESEARCH §3 is the weakest-sourced section** of the dossier (the assigned agent
   never reported). Version-check madmom / Basic Pitch / Chordino before depending
   on them.
12. **No high-accuracy dry-vocal mode.** Explicitly out of scope (§12.2). RESEARCH
   §5.2 recommended it as mitigation (3) alongside confidence-gating; only the
   gating is adopted, so the existing a cappella path becomes dead code and
   coverage carries the honesty burden alone.
13. **Two deep-learning frameworks in one venv** (§4.4), neither of the new ones
    yet installed or import-verified.
14. **Axis 2 Phase B (§7) is unimplemented and unvalidated** — a research-backed
    proposal, not a tested design. Axis 1 and Axis 2 Phase A are built and
    validated only against the label-free proxy in §5.8.
15. **No unit tests.** `evaluate.py` measures end-to-end discrimination, but the
    pure functions (segmentation, the intonation curve, chroma ranking) have no
    tests. A refactor could silently change behaviour and the harness would only
    show it as a metric drift, not a failure.

---

## 12. Open decisions

1. **Ground-truth strategy for Axis 1.** Hand-label a small set? Rank-order
   performances by ear and check correlation? Use DAMP / VocalSet (RESEARCH §7.8)?
   **Without this, step 7 cannot be evaluated.** Most important open decision.
   Cheap bootstrap: run the chroma pipeline, then **review and correct its output by
   ear** — fixing a machine's guesses is far cheaper than annotating from scratch,
   and it produces the dataset that makes everything else measurable.
2. **Dry-vocal high-accuracy mode — deferred by decision, 2026-08-10.** Not being
   built now. Revisit only if coverage on produced tracks proves unacceptable. The
   existing a cappella path stays in the tree as dead code until then; decide
   whether to delete or keep it.
3. **Job handling for Demucs.** 3–8 min CPU runtime is incompatible with a blocking
   HTTP request. Options: background job + polling, GPU/MPS host, or accept long
   waits and at minimum stop blocking the event loop.
4. **Per-segment vs. per-frame** for the Phase B model (§7.3). Decide **before**
   building the corruption pipeline — it changes the data generation.
5. **Training corpus for Phase B** (§7.5). What songs, how autotuned, separated
   stems or clean multitracks?
6. **How far to push interval-aware targets in Phase A** (§6.1). Principled per
   RESEARCH §5, most overfit-prone part of the plan, and Phase B may subsume it
   entirely. Recommendation: ship tuning correction only, measure, then decide.
7. **Chord labels in the UI?** Adds interpretability and a failure mode. Scoring
   does not need them (§5.5).
8. **Learn the local/global weights?** Considered 2026-08-10. A *discriminative*
   model — score one note from features — avoids every objection to the
   note-set generator in §3 warning 4: no set-valued target, no degenerate
   collapse, no vocal leakage. Negatives come free from the transposition
   harness (real = 1, transposed = 0).

   Two cautions. First, "this song has no off-key notes" is a **per-song**
   judgment but the label must be **per-note**, and released tracks are full of
   deliberately low-support notes (passing tones, ♭7s, suspensions — RESEARCH
   §6.1). Labelling those 1.0 teaches the model that low local support is fine,
   which is the distinction Axis 1 exists to measure. Second, transposition
   negatives are wrong in a *different way* than humans are wrong: singers slip a
   semitone or pick a plausible wrong scale note, they do not transpose uniformly
   by a tritone.

   With two features this is 2-parameter logistic regression — cheap and
   overfit-resistant, but §5.7 shows feature choice moved the margin 3× while
   combination tweaks moved it ~0. So: worth doing *after* chord estimation, not
   instead of it.

---

## Appendix: current-system reference

Constants and locations, for anyone diffing old against new.

| Location | Constant / behavior | Value |
|---|---|---|
| `preprocess.py:14-15` | sample rate / step | 16 kHz / 20 ms |
| `preprocess.py:20` | CREPE confidence threshold | 0.60 ← load-bearing per §8 |
| `preprocess.py:159` | CREPE model capacity | `"tiny"` ← raise this |
| `preprocess.py:16` | high-pass cutoff | 85 Hz, order 5 |
| `preprocess.py:24-25` | mask min run / gap fill | 5 frames / 1 frame |
| `note_segmentation.py:8` | `DRIFT_CENTS` | 40.0 |
| `note_segmentation.py:9` | `TRIGGER_FRAMES` | 3 (requires 4 frames, ~80 ms) |
| `note_segmentation.py:13-14` | core trim / min voiced | 0.20 (middle 60%) / 10 |
| `note_segmentation.py:57-65` | `GENRE_EXTRA_INTERVALS` | ← delete |
| `note_segmentation.py:97` | blue offsets | {3, 6, 10} |
| `note_segmentation.py:174` | target note from first voiced frame | ← bug, see §6.2 |
| `note_classification.py:28` | `NEAR_ALLOWED_CENTS` | 65.0 ← delete |
| `note_classification.py:12-13` | vibrato band | 4.0–8.5 Hz |
| `note_classification.py:21-22` | portamento slide / R² | 120 ¢ / 0.60 |
| `scoring.py:19-21` | perceptual curve | 0–25 ¢ → 1.00–0.90; 25–45 ¢ → 0.90–0.65; then exp decay ← delete |
| `main.py:96` | score buckets | high ≥ 0.80, mediocre 0.60–0.80, low < 0.60 |

Known bugs independent of this redesign: `/analyze` blocks the event loop
(`api.py:27`); user input errors return HTTP 500 instead of 422 with raw exception
text leaked (`api.py:38`); metrics average *unsmoothed* scores while the chart plots
*smoothed* ones, so the panel and graph disagree (`main.py:84` vs.
`scoring.py:53-59`); all metrics are note-count weighted, not duration weighted, so
a 40 ms grace note counts as much as a 3 s sustain; `activation` computed but never
consumed (`main.py:192`); bare `except:` in the scoring hot path (`scoring.py:12`);
segmentation boundary bug at `note_segmentation.py:232` (§10).
