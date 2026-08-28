# RESEARCH.md — Pitch Accuracy Analyzer

Research dossier compiled 2026-08-09. Covers music source separation, polyphonic
harmonic analysis, intonation psychoacoustics, and prior art in automatic singing
assessment.

**Purpose.** This document is the complete research record behind the proposed
redesign (see `NEW_VERSION.md`). It is written so that a reader with no prior
context can reach the same level of informedness as the session that produced it.

**How to read the confidence labels.** Every substantive claim is tagged:

- **[VERIFIED]** — read directly from a primary source (paper PDF, official
  README, library docs) or measured locally on this machine.
- **[SECONDARY]** — reported consistently by reliable secondary sources, primary
  not accessed.
- **[INFERRED]** — deduced from title/venue/author track record; abstract or full
  text was NOT read. Treat as a lead, not a fact.
- **[KNOWLEDGE]** — from the compiling model's training data, not verified this
  session. Version-check before acting.

Research method and its failures are documented in §8. Read that section before
trusting any single citation.

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [Music source separation](#2-music-source-separation)
3. [Polyphonic harmonic analysis](#3-polyphonic-harmonic-analysis-chords-and-multi-f0)
4. [Tuning reference](#4-tuning-reference-the-a440-assumption)
5. [Intonation psychoacoustics](#5-intonation-psychoacoustics-do-singers-sing-12-tet)
6. [Harmonic correctness theory](#6-harmonic-correctness-vertical-vs-horizontal)
7. [Prior art in singing assessment](#7-prior-art-automatic-singing-assessment)
8. [Research method, gaps, and caveats](#8-research-method-gaps-and-caveats)
9. [Local measurements](#9-local-measurements-on-this-repo)
10. [Open questions](#10-open-questions)

---

## 1. Executive summary

The project's original premise — that the set of musically-correct notes can be
derived from `(vocals, declared key, genre)` — is not supported by the literature
and is not what any working system does. The proposed replacement — derive
harmonic ground truth from the song's own instrumental — **is** validated, by
exactly one published paper, with a measurable accuracy gain.

Five findings that should shape any redesign:

1. **Accompaniment-derived pitch reference works and is nearly unexplored.**
   Hsieh et al. (Interspeech 2025) infer the reference from the backing track and
   report Pearson r = 0.611 against human raters, vs. 0.364 and 0.232 for the two
   dominant reference-free families. §7.1
2. **"Separate every instrument" is not possible today.** Production separation
   yields a fixed 4 stems; the `other` stem stays polyphonic. §2
3. **Therefore harmony must come from a polyphonic representation** (chroma,
   chord estimation, or multi-f0) — not from running a monophonic tracker on
   per-instrument stems, which do not exist. §3
4. **The vocals stem contains lead + backing + harmonies + doubles.** No
   production-quality lead/backing separator exists. This is the hardest
   unsolved piece of the proposed design. §2.4
5. **Three systematic biases corrupt cents-based scoring regardless of
   architecture**: an unestimated tuning offset, the assumption that singers
   target 12-TET, and ignoring beat position. §4, §5, §6

---

## 2. Music source separation

### 2.1 What stems current models actually produce

**[VERIFIED]** — stem lists confirmed from official model documentation.

| Model | Stems |
|---|---|
| `htdemucs`, `htdemucs_ft` | drums, bass, other, vocals |
| `htdemucs_6s` (experimental) | drums, bass, other, vocals, **guitar, piano** |
| `mdx`, `mdx_extra` | drums, bass, other, vocals |
| Spleeter 2stems | vocals, accompaniment |
| Spleeter 4stems | vocals, drums, bass, other |
| Spleeter 5stems | vocals, piano, drums, bass, other |
| Open-Unmix (all) | vocals, drums, bass, other |
| MDX-Net | vocals, drums, bass, other |
| Band-Split RNN (BSRNN) | vocals, drums, bass, other |

The near-universal vocabulary is **4 fixed stems**. The `other` stem is
explicitly a catch-all: guitars, synths, keys, strings, horns, and any pitched
instrument that is not bass or voice are mixed together in it.

**Critical consequence.** There is no "melodic elements, individually separated"
output from any production tool. A design that requires running a monophonic
pitch tracker on each melodic instrument separately has no such stems to consume.

### 2.2 Six-stem and per-instrument separation

**[VERIFIED]** `htdemucs_6s` is the only public model exposing both `guitar` and
`piano`. The official Demucs README states guitar quality is "okay" and piano
shows "a lot of bleeding and artifacts." No published SDR table exists for the
6-stem model. Source: https://github.com/facebookresearch/demucs

**[VERIFIED]** Spleeter 5stems adds `piano` but no `guitar`, and its overall
quality is materially below Demucs: Spleeter 4stem vocals SDR ≈ **6.55 dB** vs.
`htdemucs_ft` ≈ **9.4 dB**.

**[SECONDARY]** 2024–2025 SOTA for 4-stem separation:
- **BS-RoFormer** (ByteDance) — arXiv:2309.02612
- **Mel-Band RoFormer** (2024)
- Both reach ~9–10+ dB average SDR. Neither adds stems.

**[SECONDARY]** Query-conditioned separation — the research frontier for
arbitrary instrument extraction:
- **AudioSep** (2023) — arXiv:2308.11786. Natural-language queries
  ("separate the piano").
- **SAM Audio** (Meta, Dec 2025) — arXiv:2512.18099. Diffusion transformers with
  text/temporal conditioning; SOTA on standard benchmarks.
- **QSCNet** — arXiv:2512.15532. Conditioned U-Net, >1 dB SNR over prior
  query-based baselines.

**[SECONDARY]** Known failure modes for query-based models: closely-related
timbres (electric vs. acoustic guitar; two guitar parts), rare instruments with
sparse training data, and heavily processed/synthesized timbres. They also
require the user to name what to extract, which presupposes instrument
knowledge the app would not have.

None of these ship as a battle-tested pip-installable package comparable to the
`demucs` CLI. **Do not build on them.**

**[VERIFIED/KNOWLEDGE]** Per-*note* audio separation is not feasible. Automatic
music transcription produces piano-roll symbolic output, not separated
waveforms. NMF-based harmonic decomposition works only on sparse/simple mixes.

### 2.3 Separation artifacts and downstream pitch tracking

**[SECONDARY]** Four degradation mechanisms matter for f0 estimation:

- **Spectral holes** — Wiener-filter masking punches gaps in the harmonic
  series. Monophonic trackers depend on harmonic structure; missing partials
  directly raise f0 error.
- **Bleed / inter-stem leakage** — accompaniment leaking into the vocal stem
  makes trackers lock onto instrumental f0 during vocal rests or soft passages.
- **Musical noise** — isolated AM bursts of residual noise create spurious
  harmonic-like content that fools both autocorrelation trackers (pYIN) and
  salience trackers (CREPE).
- **Phase smearing** — reduces waveform periodicity. CREPE operates on
  log-frequency spectrograms so is somewhat more robust to phase damage than
  waveform-domain methods, but magnitude distortion still applies.

**[SECONDARY]** Molina et al. (ISMIR 2014) showed melody extraction degrades
significantly on separated rather than clean sources, even at 7–8 dB SDR.

**[SECONDARY]** The rough working threshold in the AMT / melody-extraction
literature is **>10 dB SDR for reliable monophonic pitch tracking**.
`htdemucs_ft` vocals at ~9.4 dB sits just *below* that line — borderline, not
comfortable.

**Asymmetry worth internalizing.** Artifacts hurt the *instrumental* path far
less than the *vocal* path. Aggregating the instrumental to 12 pitch classes
(chroma) is robust to spectral holes and moderate bleed, because energy is
pooled across octaves and time. Extracting a precise vocal f0 to score cents
deviation is not robust — every artifact translates into scoring error.

### 2.4 The vocals stem contains all vocals

**[VERIFIED]** Every standard model's `vocals` stem contains **all** vocal
content: lead, backing, harmonies, doubles, ad-libs, and any pitched human
voice. No model in the Demucs/Spleeter/MDX/Open-Unmix stack separates lead from
backing.

**[SECONDARY]** Lead-vs-backing separation is an active research area with no
reliable production-quality open-source tool as of mid-2025. Search terms for
follow-up: "lead vocal extraction", "predominant vocal separation", "VoSNet",
"vocal layer separation". Commercial voice-isolation tools (e.g. iZotope RX
Dialogue Isolation) split voice from non-voice, not lead from backing.

**Consequence for the proposed design.** On any commercially produced track with
harmonies or doubles, the `vocals` stem is *polyphonic*. CREPE is explicitly a
monophonic tracker; given polyphonic input it will follow whichever voice is
loudest per frame, jumping between lead and harmony mid-phrase. This is the most
serious open problem in the redesign. Mitigations in `NEW_VERSION.md` §5.

### 2.5 Licensing, runtime, memory

**[VERIFIED]** Licenses:
- **Demucs — MIT**, including pretrained weights. No commercial restriction.
  https://github.com/facebookresearch/demucs/blob/main/LICENSE
- Spleeter — MIT.
- Open-Unmix — `umx`/`umxhq` MIT; **`umxl` is CC BY-NC-SA 4.0 (non-commercial)**.

**[VERIFIED]** `demucs` is on PyPI at **4.1.0** (checked 2026-08-09 against this
repo's venv).

**[SECONDARY]** Runtime for a 3–4 minute song with `htdemucs`:

| Hardware | Time |
|---|---|
| RTX 3090 / A100 (GPU) | ~15–30 s |
| Apple M1/M2 (MPS) | ~1–2 min |
| Modern CPU (8–16 core) | ~3–8 min |
| Entry CPU (2–4 core) | ~10–20 min |

**[SECONDARY]** Memory: ~3–4 GB VRAM at default segment size (reducible via
`--segment`); ~4–6 GB CPU RAM peak; `htdemucs_6s` somewhat higher (~5–6 GB).
Demucs chunks audio into overlapping ~7.8 s segments and cross-fades, so memory
is bounded by segment length, **not** song duration.

---

## 3. Polyphonic harmonic analysis (chords and multi-f0)

> **Coverage warning.** The research agent assigned to this topic did not report
> back. This section is **[KNOWLEDGE]** plus local PyPI verification, and is the
> weakest-sourced part of this dossier. Everything here should be
> version-checked and benchmarked before it is built on. Marked **[UNVERIFIED
> GAP]** where a specific accuracy claim would be needed.

### 3.1 Local availability check

**[VERIFIED]** Checked against PyPI from this repo's venv on 2026-08-09:

| Package | Latest on PyPI | Notes |
|---|---|---|
| `librosa` | 0.11.0 | already installed in this repo |
| `demucs` | 4.1.0 | MIT, 4-stem |
| `madmom` | 0.16.1 | chord/beat/downbeat; last release is old |
| `basic-pitch` | 0.4.0 | Spotify, polyphonic transcription |
| `autochord` | 0.1.4 | chord labels; small project |
| `essentia` | **no matching distribution** | no wheel for this Python; would need conda or source build |

The `essentia` result matters: several tuning/key recommendations in the
literature name Essentia specifically, but it is **not trivially installable
here**. `librosa` covers the tuning-estimation need natively (§4.2).

### 3.2 Three tiers of harmonic representation

**Tier 1 — Chroma / pitch-class salience.** **[KNOWLEDGE]**
`librosa.feature.chroma_cqt`, `chroma_cens`, or a CQT folded to 12 pitch classes.
Produces a time-varying 12-dimensional vector of per-pitch-class energy.

Why this is the right default:
- **Inherently polyphonic.** Chords require no special handling because the
  representation never commits to "which single note is sounding." A C-major
  triad simply lights up pitch classes 0, 4, 7.
- **Continuous, not binary.** Yields a *salience prior* rather than an allowlist,
  which structurally eliminates the if/else branching that made the current
  system brittle.
- **Robust to separation artifacts** (see §2.3 asymmetry).
- Already a dependency of this project.

Harmonic Pitch Class Profile (HPCP) is the established name for the tuned,
normalized variant of this feature; it is standard MIR, not an invention.

**Tier 2 — Bass stem → monophonic f0.** **[KNOWLEDGE]**
Bass is monophonic by construction in nearly all popular music, so CREPE is
*correctly* applied here. Gives root motion, which disambiguates chord
inversions and rootless voicings that chroma alone cannot resolve. This is the
one place the original "run CREPE per stem" instinct works as intended.

**Tier 3 — Discrete chord labels.** **[KNOWLEDGE]** / **[UNVERIFIED GAP]**
Options: `madmom` (deep chord recognition), Chordino/NNLS-Chroma (Vamp plugin,
Mauch & Dixon), `autochord`, and CRNN/BTC-family research models.

Trade-offs: needed only for human-readable output ("you sang a ♭9 over Cmaj7").
Adds a failure mode — ACE vocabularies are typically triads plus sevenths, so
extensions and slash chords get flattened into the nearest supported label.
**[UNVERIFIED GAP]** MIREX-style accuracy figures for these tools on separated
stems were not obtained; treat chord labels as a presentation layer, not as
scoring ground truth, until benchmarked.

**[SECONDARY]** For context on ACE maturity: Pauwels, O'Hanlon, Gómez, Sandler,
"20 Years of Automatic Chord Recognition from Audio," ISMIR 2019.
http://archives.ismir.net/ismir2019/paper/000004.pdf

### 3.3 Polyphonic transcription

**[KNOWLEDGE]** Candidates if discrete note events are wanted rather than
chroma:
- **Basic Pitch** (Spotify) — pip-installable, permissive, instrument-agnostic
  polyphonic transcription. Practical, lightweight.
- **MT3** (Google) — multi-task multi-track transcription; research-grade,
  heavy.
- **Omnizart** — multi-purpose transcription toolkit.

**[UNVERIFIED GAP]** No accuracy comparison of these on Demucs-separated stems
was obtained this session.

### 3.4 Multi-f0 on vocal stems

**[KNOWLEDGE]** / **[UNVERIFIED GAP]** For tracking multiple simultaneous
singing voices, the relevant literature is vocal-harmony transcription and
choral analysis (Deep Salience, and the Dagstuhl ChoirSet line of work — see
§7.6). Whether any of it is production-usable for lead-vs-harmony
disambiguation on pop recordings was **not established**. This remains the
open problem flagged in §2.4.

**Practical alternative** **[KNOWLEDGE]**: salience-based *predominant melody*
extraction (the PreFEst / Melodia lineage, §7.4) is purpose-built for
extracting the leading melodic line from polyphony — a better tool than CREPE
where harmonies are present.

---

## 4. Tuning reference: the A440 assumption

### 4.1 How far real recordings drift

**[SECONDARY]** Causes and magnitudes:

| Cause | Deviation |
|---|---|
| Tape varispeed, ±1% transport error | **±17.3 cents** |
| Tape varispeed, ±0.5% | ±8.7 cents |
| Guitar tuned half-step down | −100 cents |
| Guitar tuned quarter-step down | ≈ −25 cents |
| Pitch-shifted samples (DAW) | arbitrary; common in hip-hop/electronic |
| A=432 Hz | **−31.8 cents** exactly, = 1200·log₂(432/440) |
| Berlin Philharmonic, A=443 Hz | +11.75 cents |
| European orchestras, 441–444 Hz | +3.9 to +15.6 cents |
| Baroque pitch, A=415 Hz | −100 cents |

Pre-2000 recordings are especially affected by varispeed. The A=432 movement has
no credible scientific basis and limited presence in mainstream releases, but the
arithmetic is worth knowing.

### 4.2 Why this is a first-order bug, not a refinement

**[SECONDARY]** A systematic offset of X cents shifts *every* pitch measurement
by exactly X cents. At a +15-cent backing track, a perfectly in-tune singer
measures as uniformly 15 cents sharp — **indistinguishable from genuine
inaccuracy** unless the offset is estimated and removed first.

**[VERIFIED]** Tuning estimation is standard MIR with ready implementations:

- **`librosa.estimate_tuning`** — returns deviation in fractional bins,
  range [−0.5, 0.5) ≈ ±50 cents at default resolution 0.01.
  https://librosa.org/doc/latest/generated/librosa.estimate_tuning.html
- **Essentia `TuningFrequency`** (Gómez 2005/2006, MTG/UPF) — magnitude-weighted
  histogram of fractional semitone offsets over spectral peaks. Output range
  **−35 to +65 cents**; the asymmetry was derived empirically from real audio
  distributions. https://essentia.upf.edu/reference/std_TuningFrequency.html
  *(But see §3.1 — Essentia has no installable wheel here.)*
- **NNLS-Chroma / Chordino** (Mauch & Dixon, ISMIR 2010) — estimates tuning
  internally as preprocessing, then re-aligns chroma bins to the detected
  offset. The QM Vamp suite exposes a `tuningFrequency` parameter.

**Note the dependency direction:** estimating tuning from an *instrumental* is
far more reliable than from a solo voice, because fixed-pitch instruments define
the grid the singer is aiming at. This is an argument *for* the full-mix
redesign, not merely a compatible add-on.

Measured offsets on this repo's own samples: **§9**.

---

## 5. Intonation psychoacoustics: do singers sing 12-TET?

No. Three documented systematic departures.

### 5.1 Just-intonation pull on thirds and sixths

**[SECONDARY]** — strong, widely reproduced:

| Interval | Just | 12-TET | Δ from 12-TET |
|---|---|---|---|
| Major third | 386.3 ¢ | 400 ¢ | **−13.7 ¢** |
| Minor third | 315.6 ¢ | 300 ¢ | **+15.6 ¢** |
| Pythagorean major third | 407.8 ¢ | 400 ¢ | +7.8 ¢ |

The harmonic series places the major third ~14 cents below the 12-TET grid (5th
partial). The pedagogical literature states plainly that singers and string
players gravitate toward pure intervals.

**[SECONDARY]** Shackford (1961/62, *Journal of Music Theory*) documented string
quartet intonation preferences; Loosen (1995, *JASA*) found solo violinists
deviated toward Pythagorean for some intervals and just for others — **neither
consistently matched 12-TET**. Performers appear to navigate a space *between*
just and Pythagorean. Both papers are paywalled; findings here are from
secondary sources.

### 5.2 Leading-tone sharpening

**[SECONDARY]** Pythagorean major seventh = 1109.8 ¢ vs. 12-TET 1100 ¢ = +9.8 ¢.
Singers and pedagogues widely report sharpening the 7th degree when approaching
the octave, roughly **+10 to +15 cents**. Magnitude is less firmly established
than the thirds finding.

**[SECONDARY]** Sundberg et al. (2013) found sharpened intonation at a phrase
climax *increased* perceived expressiveness and excitement — i.e. deviation can
be a feature.

### 5.3 Adaptation to accompaniment

**[SECONDARY]** Singers adjust to the harmonic context actually sounding. Against
a fixed equal-tempered instrument (piano, synth) they pull toward 12-TET; a
cappella or against strings they pull toward just intonation.

**[SECONDARY]** Devaney et al. (2011) found measurable differences between
professional and non-professional vocalists in semitone interval sizes —
intonation is *skilled behavior*, not random drift.

### 5.4 Implication for scoring

Measuring cents deviation from the nearest 12-TET semitone is defensible when the
backing track is dominated by fixed-pitch equal-tempered instruments. It is
**not** neutral otherwise. In R&B, soul, and hip-hop — where harmony is often a
sampled chord, an electric guitar, or has no fixed-pitch anchor — a skilled
singer's major third may legitimately sit at 387–393 cents, scoring as 7–13
cents "flat" against a 12-TET target.

**[SECONDARY]** Blue notes compound this. The flattened 3rd, 5th, and 7th in
blues/R&B/soul/gospel are intentionally microtonal — "flattened by a variable
microtone." Microtonal interval variation is standard practice in
African-American musical forms (spirituals, blues, jazz). **A 12-TET grid
penalizes correct blue notes by construction**, and this project's stated target
genres are exactly the ones built on them.

---

## 6. Harmonic correctness: vertical vs. horizontal

### 6.1 Non-chord tones are a formal category

**[SECONDARY]** — standard common-practice theory (Aldwell/Schachter,
Kostka/Payne):

| Type | Beat position | Motion | Resolution |
|---|---|---|---|
| Passing tone | Weak | Stepwise between chord tones | Continues same direction |
| Neighbor tone | Weak | Step away and back | Returns to origin |
| Suspension | **Strong** | Held over from prior chord | Resolves step down |
| Appoggiatura | **Strong** | Approached by leap | Resolves stepwise |
| Anticipation | Weak, before change | Arrives before its chord | None needed |
| Pedal point | Any | Static while harmony moves | Resolves on return |
| Escape tone | Weak | Step then leap | No conventional resolution |

**Beat position is definitional, not decorative.** The standard taxonomy holds
that the most important distinction is whether a non-chord tone falls on a
strong or weak beat. The same pitch against the same chord is a *suspension* on
beat 1 and plausibly an *error* on beat 4.

### 6.2 Computational consonance models exist but do not solve this

**[SECONDARY]**
- **Hutchinson & Knopoff (1978)**, *JASA* — roughness model summing beating
  contributions across all partial pairs, amplitude-weighted.
- **Parncutt (1989)**, *Harmony: A Psychoacoustical Approach*, Springer —
  extends Terhardt's virtual-pitch theory to chords; predicts perceived root,
  chord salience, consonance from harmonic templates.
- **Huron (1994)**, "Interval-Class Content Investigated"; and *Sweet
  Anticipation* (MIT Press, 2006) — predictive coding of harmonic expectation.
- **Sethares (1998)**, *Tuning, Timbre, Spectrum, Scale*, 2nd ed. — roughness is
  **timbre-dependent**: the same interval is more or less dissonant depending on
  spectral content. A sine-wave major second is nearly smooth.

**Hard limitation.** These compute *acoustic roughness*, not *musical
correctness*. None distinguishes functional/intentional dissonance from a
mistake. That judgment needs style knowledge, voice-leading context, and beat
position — none of which a roughness model has access to.

**[SECONDARY]** No current MIR system automatically classifies "intentional
non-chord tone vs. pitch error" in the general case. Score-informed systems can
(they have the score); for commercial pop without a score this is an **open
research problem**.

### 6.3 Irreducible ambiguity — cases where harmony cannot decide

**[SECONDARY]**

1. **Blue notes** — deliberately microtonal, between semitones by intent (§5.4).
2. **Quartal / quintal harmony** — stacked fourths have no definite root. Per
   Persichetti, any member of a quartal chord can function as the root. The
   chord-tone / non-chord-tone distinction *collapses*.
3. **Pedal points and chromatic mediants** — a note may clash with a static pedal
   yet be correct against the active harmony above it. Chromatic mediant motion
   (e.g. C major → A♭ major) can flip a note's status instantaneously.
4. **Sustained tension in R&B/gospel** — the "cry" technique sustains a dissonant
   pitch that resolves late. Genre-correct behavior that naive scoring flags as
   error.
5. **Modal and atonal passages** — the allowable pitch set depends on correctly
   identifying the mode, which the system may get wrong.

**Conclusion.** Harmonic context, even perfectly identified, cannot fully
determine acceptable vocal notes. Expect a performance ceiling, not a solved
problem.

### 6.4 Timing dependence

**[SECONDARY]** Beat position and chord-boundary alignment are structural
requirements, not polish:

- **Anticipations** arrive one or two beats *before* their chord. Scoring against
  the outgoing chord marks them wrong; the incoming chord hasn't started. Only
  a slack window handles this.
- **Syncopation** — in R&B and hip-hop, harmonic arrivals routinely land off the
  grid, or the vocalist anticipates the change.
- **Melisma across chord changes** — a run may start on chord A's tones and end
  on chord B's; middle notes belong to neither and are understood as passing
  motion.
- **Appoggiatura** — dissonant *at onset on a strong beat* by design.

**[SECONDARY]** Minimum viable requirement: segment notes by onset, align to
chord windows with anticipation slack of roughly **±1 beat** at phrase level in
R&B. Frame-by-frame scoring against the simultaneous chord label will
systematically misjudge syncopated phrasing.

---

## 7. Prior art: automatic singing assessment

### 7.1 The direct match

**[VERIFIED — full PDF text read]**

> **Tonality-Based Accompaniment-Guided Automatic Singing Evaluation**
> Pei-Chin Hsieh, Yih-Liang Shen, Ngoc-Son Tran, Tai-Shih Chi
> Interspeech 2025, pp. 3085–3089 · DOI `10.21437/Interspeech.2025-1015`

This is the paper that validates the proposed direction. Details:

- **Reference is inferred automatically from the backing track.** A 5-layer CNN
  key classifier takes the user's vocal plus accompaniment as input. Stated
  rationale: "the musical key is typically associated with the background music's
  chord progression."
- **Score** = voiced-frame-weighted proportion of sung pitches falling inside the
  detected key's scale.
- **No MIDI, no score, no reference vocal.**
- 24 keys collapsed to 12 scale classes (relative major/minor merged).
- Trained on GTZAN / iKala / MedleyDB using **commercial Tunebat key labels as
  ground truth** — no annotated key dataset existed for the task.
- **Pearson r = 0.611** vs. human raters. Baselines: **0.364** for a
  Nakano-style pitch-interval method, **0.232** for a Gupta-style pitch-histogram
  method.
- Dominant-key confusion is tolerable because the scales largely overlap.
- **Stated limitation: fails a cappella.**

**Interpretation.** Harmony-from-accompaniment nearly doubles correlation with
human judgment versus reference-free methods. But note the ceiling: r = 0.611 is
a good result, not near-perfect agreement — consistent with the irreducible
ambiguity in §6.3. Also note this paper uses *key*, a global/slow-moving
property, not frame-level chords. A chord-level approach is a step beyond
published work.

### 7.2 The field has not combined chord detection with singing assessment

**[VERIFIED — negative result]** Exhaustive DBLP full-text queries returned
**zero** matches for the intersection:

- `singing assessment chord`
- `chord based pitch reference singing evaluation`
- `chord detection pitch singing karaoke accompaniment`
- `automatic pitch feedback singing chord`
- `chord recognition singing accompaniment pitch`
- `singing chord`, `vocal pitch accuracy chord`

ISMIR proceedings 2009–2024 were also checked individually: chord-recognition
papers and singing-assessment papers exist as **two entirely separate clusters,
never combined**.

**[VERIFIED]** The 2026 survey (dos Santos & Masiero, arXiv:2601.12153) covers
karaoke scoring in §4.5; its 66-item bibliography contains **no chord-detection
reference**. The survey confirms the pitch reference in the literature is always
either f0 from a reference recording (PreFEst/Melodia) or a score/MIDI.

**Caveat on this negative result:** DBLP indexes CS venues only. Music-education
and voice-pedagogy journals (*Journal of Voice*, *Music Education Research*,
*Psychology of Music*) are not covered.

### 7.3 Strongest near-misses

**[VERIFIED]**
- **Deep Autotuner** — Wager, Tzanetakis, Wang, Guo, Sivaraman, Kim (2019),
  arXiv:1902.00956. Predicts pitch correction "from the relationship between the
  spectral contents of the vocal and accompaniment tracks," explicitly for cases
  where "no musical score of the vocals nor the accompaniment exists."
  Backing-track-derived pitch targets without a score — but it performs
  *correction*, not *assessment*, and learns the mapping implicitly rather than
  detecting harmony explicitly. Also **KaraTuner** (Interspeech 2022).
- **Ju et al., ISMIR 2024** — "End-to-End Automatic Singing Skill Evaluation
  Using Cross-Attention and Data Augmentation for Solo Singing and Singing With
  Accompaniment," DOI `10.5281/zenodo.14877383`. Despite the title, accompaniment
  is treated as **interference to be robust to** (augmentation + bidirectional
  cross-attention), explicitly avoiding the singing-voice-separation
  preprocessing of prior work. **No harmonic reference extracted.**
- **Zhang et al., ISMIR 2021** — "Learn by Referencing: Towards Deep Metric
  Learning for Singing Assessment,"
  https://archives.ismir.net/ismir2021/paper/000103.pdf. Uses the accompaniment
  track as a metric-learning *anchor*. **The authors' own negative finding:** it
  underperforms because "the accompaniment track as anchor does not provide
  details on singing, but only helps with judging the rhythm and tonality."
  **This is the most important cautionary result in the dossier** — the
  instrumental constrains *which notes fit*, not *how well they were sung*. It
  argues directly for separating the two scoring axes.
- **Ju et al., ICME 2023** — "Improving Automatic Singing Skill Evaluation with
  Timbral Features, Attention, and Singing Voice Separation," DOI
  `10.1109/ICME55011.2023.00111`. Uses separation to *remove* accompaniment
  before evaluation; found "accompaniment removal achieves better performances."

### 7.4 Family A — melody f0 copied from a reference recording

Automatic from audio, but tracks the *melody line*, not harmony.

**[SECONDARY]**
- **Tsai & Lee**, IEEE TASLP 20(4), 2012, DOI `10.1109/TASL.2011.2174224`;
  precursor ICASSP 2011, DOI `10.1109/ICASSP.2011.5946974`. Karaoke evaluation
  from pitch extracted from karaoke VCD audio.
- **MiruSinger** — Nakano, Goto, Hiraga, IEEE ISMW 2007, DOI
  `10.1109/ISMW.2007.61`. Uses commercial CD recordings as pitch reference via
  PreFEst.
- **Goto (2004)**, "A Real-Time Music-Scene-Description System: Predominant-F0
  Estimation…", *Speech Communication* 43(4):311–329. The **PreFEst**
  foundation — predominant-f0 from polyphonic audio. Directly relevant as an
  alternative to CREPE on polyphonic vocal stems (§3.4).
- **[INFERRED]** Tsai, Ma & Hsu (2015), "Automatic Singing Performance Evaluation
  Using Accompanied Vocals as Reference Bases," *J. Inf. Sci. Eng.* 31(6).
  Promising title; **both publisher URLs are dead (404/403)**. Given the authors'
  track record this is likely melody-from-audio, not harmony. Unresolved lead.

### 7.5 Family B — score / MIDI reference (the incumbent paradigm)

**[SECONDARY]**
- **Molina et al., ICASSP 2013** — "Fundamental frequency alignment vs.
  note-based melodic similarity for singing voice assessment," DOI
  `10.1109/ICASSP.2013.6637747`. Best methodological near-miss on metrics.
- **Huang, Hung, Pati, Gururani, Lerch, ISMIR 2020** — "Score-informed Networks
  for Music Performance Assessment," arXiv:2008.00203. The direct methodological
  *opposite* of chord-inferred reference; useful contrast citation.
- **Huang & Lerch, ISMIR 2019** — "Automatic Assessment of Sight-reading
  Exercises," http://archives.ismir.net/ismir2019/paper/000070.pdf
- **Bonada et al., AES 121 (2006)** — "The Singing Tutor"; HMM segmentation with
  MIDI alignment.
- **Bozkurt, Baysal, Yüret, CMMR 2017** — "A Dataset and Baseline System for
  Singing Voice Assessment"; Melodia f0 from separately recorded piano.
- **Yang et al., IEEE TMM 2023**, DOI `10.1109/TMM.2022.3168132` — multi-stage
  sight-singing; F-measure 77.95% on SSVD.

### 7.6 Family C — reference-free

**[SECONDARY]**
- **Nakano, Goto, Hiraga, Interspeech 2006**, DOI
  `10.21437/Interspeech.2006-474` — pitch-interval accuracy + vibrato features
  for *unknown melodies*. This is the **0.364 baseline** in Hsieh et al.
- **Gupta, Li, Wang** — the main reference-free line, pitch-histogram based:
  - APSIPA 2018, "Automatic Evaluation of Singing Quality without a Reference,"
    DOI `10.23919/APSIPA.2018.8659545`. Spearman ρ = **0.716** vs. human
    judgment.
  - IEEE/ACM TASLP 28:13–26 (2020), "Automatic Leaderboard: Evaluation of
    Singing Quality Without a Standard Reference," DOI
    `10.1109/TASLP.2019.2947737`. Journal extension. This family is the
    **0.232 baseline** in Hsieh et al.
  - APSIPA Trans. 2018, DOI `10.1017/ATSIP.2018.10`; APSIPA 2017, DOI
    `10.1109/APSIPA.2017.8282110`.
  - ISMIR 2020 twin-neural rank-ordering,
    http://archives.ismir.net/ismir2020/paper/000165.pdf
- **Zhang, Jiang, Deng, Li, ICASSP 2019** — "Automatic Singing Evaluation without
  Reference Melody Using Bi-dense Neural Network," DOI
  `10.1109/ICASSP.2019.8682665`.
- **Shi, Ai, Lu, Du, Ling, IEEE SLT 2024** — "Pitch-and-Spectrum-Aware Singing
  Quality Assessment…", DOI `10.1109/SLT61566.2024.10832260`, arXiv:2411.11123.
  **[VERIFIED from abstract]** pitch histogram + neural codec on sung audio only;
  no chord/MIDI reference. First place, VoiceMOS Challenge 2024 Track 2.
- **Bohm, Eyben, Schmitt, Kosch, Schuller, IJCNN 2017** — "Seeking the
  SuperStar," DOI `10.1109/IJCNN.2017.7966037`. 84.7% 3-class accuracy from
  acoustic features.
- **[VERIFIED from abstract]** **SingMOS-Pro** (Tang et al., 2025),
  arXiv:2510.01812 — MOS benchmark, 7,981 clips, no chord analysis.

### 7.7 Intonation-reference inference (closest conceptual relatives)

**[SECONDARY]** Work that infers the tonal reference from the audio itself
rather than assuming A440/MIDI — methodologically the nearest neighbors to the
tuning-estimation recommendation in §4:

- **Weiss, Schlecht, Rosenzweig, Müller, ISMIR 2019** — "Towards Measuring
  Intonation Quality of Choir Recordings: A Case Study on Bruckner's Locus
  Iste," http://archives.ismir.net/ismir2019/paper/000032.pdf. Derives the
  intonation reference from the **ensemble's own tuning**. Closest existing work
  to "infer the reference from the performance context."
- **Serrà et al., ISMIR 2011** — "Assessing the Tuning of Sung Indian Classical
  Music," https://archives.ismir.net/ismir2011/paper/000008.pdf. Tonic/tuning
  inferred from audio, no score.
- **Dai, Mauch, Dixon, ISMIR 2015** — "Analysis of Intonation Trajectories in
  Solo Singing," http://ismir2015.uma.es/articles/233_Paper.pdf
- **Devaney, Mandel, Fujinaga, ISMIR 2012** — "A Study of Intonation in
  Three-Part Singing using AMPACT" (score-aligned).
- **Rosenzweig, Scherbaum, Müller, ICASSP 2021** — "Reliability Assessment of
  Singing Voice F0-Estimates Using Multiple Algorithms," DOI
  `10.1109/ICASSP39728.2021.9413372`. Directly relevant to gating unreliable
  CREPE frames.
- **Viraraghavan, Aravind, Murthy, ISMIR 2018** — "Precision of Sung Notes in
  Carnatic Music," https://archives.ismir.net/ismir2018/paper/000120.pdf

### 7.8 Surveys and datasets

**[VERIFIED]** Surveys:
- **dos Santos & Masiero (2026)**, "A Survey on 30+ Years of Automatic Singing
  Assessment and Singing Information Processing," arXiv:2601.12153. §4.5 =
  karaoke scoring. Confirms no chord-based method exists in the field.
- **Gupta, Li, Goto (2022)**, "Deep Learning Approaches in Topics of Singing
  Information Processing," IEEE/ACM TASLP 30, DOI `10.1109/taslp.2022.3190732`.

**[SECONDARY]** Datasets worth knowing:
- **DAMP / Smule DAMP** — large-scale amateur karaoke performances.
- **VocalSet** — professional singer technique recordings.
- **Dagstuhl ChoirSet** — Rosenzweig et al., TISMIR, DOI `10.5334/tismir.48`.
  Multitrack a cappella choral; intonation assessment applications.
- **MAST melody dataset**, **SSVD** (sight-singing), **iKala**, **MedleyDB**
  (used by Hsieh et al.), **GTZAN**.

### 7.9 Commercial systems

**[UNVERIFIED GAP]** The agent tasked with commercial/consumer scoring
(Smule, SingStar, Yousician, Melodyne, Auto-Tune, Waves Tune) did not report
back. **[KNOWLEDGE]** The general expectation — that consumer karaoke scoring
relies on a **pre-authored reference melody/MIDI** rather than inferring targets
from the backing track, and that Melodyne/Auto-Tune snap to a user-specified
scale or nearest semitone rather than doing chord-aware retargeting — is
**not verified this session** and should be treated as an assumption. The
academic negative result in §7.2 does not extend to unpublished commercial
practice.

---

## 8. Research method, gaps, and caveats

**Read this before citing anything above.**

### 8.1 What was actually accessible

Four parallel research agents were dispatched (separation; polyphonic/chord
tooling; theory/psychoacoustics; prior art). Additional agents ran on
prior-art sub-angles.

**Blocked or unusable sources:** Google Scholar (CAPTCHA), Semantic Scholar (WAF
block), OpenAlex (paid quota exhausted), IEEE Xplore (HTTP 403/418), ACM
(403), ResearchGate (403). **DBLP's API plus direct PDF extraction carried
essentially all verification.**

**Consequence:** many abstracts could not be read firsthand. Pitch-reference
classifications for several §7.4–7.6 entries are **[INFERRED]** from title,
venue, and author track record. The items verified from full text or abstract
are explicitly marked.

### 8.2 Coverage gaps in this dossier

1. **Polyphonic/chord tooling (§3)** — assigned agent never reported. Section is
   **[KNOWLEDGE]** + local PyPI checks. No accuracy benchmarks obtained for
   madmom / Basic Pitch / Chordino on separated stems.
2. **Commercial systems (§7.9)** — assigned agent never reported. Treat as
   unverified assumption.
3. **Multi-f0 on vocal harmony (§3.4)** — no production-readiness assessment
   obtained. This is the redesign's hardest open problem and it is the least
   researched part of this document.
4. **Separation → f0 benchmarks (§2.3)** — no head-to-head "CREPE on Demucs
   stems vs. clean audio" benchmark was located. The >10 dB threshold is a
   literature rule of thumb, not a measured result for this pipeline.
5. **DBLP-only negative result (§7.2)** — CS venues only; music-education and
   voice journals uncovered.

### 8.3 Security note — prompt injection encountered

**[VERIFIED]** One research agent reported that **four fetched web pages
contained injected text impersonating system messages and user instructions**,
including a fabricated list of agent types and a fake "message from the user"
claiming a time limit and asking the agent to relay instructions to other
agents. The agent ignored all four and relayed nothing.

Relevant when extending this research: **web content fetched during MIR
literature searches has been observed carrying injection attempts.** Do not let
fetched page text steer tooling decisions or agent behavior.

### 8.4 Corrections made during research

One agent initially mis-reported peer-agent status (claimed results had arrived
that had not) and self-corrected. Noted here only as a reminder to check whether
a cited finding was actually received versus assumed.

---

## 9. Local measurements on this repo

**[VERIFIED — measured 2026-08-09 on this machine]**

### 9.1 Tuning offsets on the bundled samples

`librosa.estimate_tuning`, first 45 s, sr=22050:

| File | Tuning offset | Chroma-argmax pitch class |
|---|---|---|
| `come_and_see_me.mp3` | **−6.0 ¢** | E |
| `dont.mp3` | **+16.0 ¢** | F# |
| `glimpse_of_us.mp3` | +1.0 ¢ | A# |
| `into_you.mp3` | +3.0 ¢ | C# |
| `my_bad_dont_cover.mp3` | **−10.0 ¢** | F# |
| `my_tbh_cover.mp3` | **+14.0 ¢** | C# |
| `whenever_wherever.mp3` | **−16.0 ¢** | G# |

**Spread: −16 to +16 cents, a 32-cent range.** For scale, the current scoring
curve's entire "pro zone" is 0–25 cents. An unestimated 16-cent offset consumes
most of that budget before the singer sings a note.

Two caveats: (a) some of these files are the user's own a cappella covers, and
tuning estimated from a solo voice reflects the singer, not a backing grid —
the measurement is most meaningful on full mixes; (b) chroma-argmax is a crude
proxy for key, shown only to demonstrate the feature is computable, **not** as a
key-detection result.

### 9.2 Current-system score distribution

Full existing pipeline on two samples (key `B minor`, genre `rnb`):

| | `my_tbh_cover` | `my_bad_dont_cover` |
|---|---|---|
| Notes segmented | 204 | 274 |
| Voiced frames | 2629 / 4386 (60%) | 3807 / 5959 (64%) |
| Median abs core deviation | **22.78 ¢** | **22.36 ¢** |
| p75 / p90 deviation | 37.3 / 47.5 ¢ | 33.6 / 48.6 ¢ |
| Mean `on_key_score` | 81.8% | 82.6% |
| Bucket split (high/med/low) | 68% / 23% / 9% | 75% / 15% / 11% |

**Interpretation.** Median deviation ~22.5 cents on both files. This sits inside
the range where §5 predicts *expressive* intonation is indistinguishable from
error under a 12-TET grid — a just major third alone accounts for 13.7 of those
cents.

### 9.3 Environment

**[VERIFIED]** Python 3.11.15 venv at repo root. Installed and importable:
`numpy` 2.2.6, `librosa` 0.11.0, `scipy`, `crepe` 0.0.16, `tensorflow` 2.20.0,
`fastapi` 0.116.1, `uvicorn` 0.35.0, `dotenv`, `google.generativeai` 0.8.5.
CREPE currently runs at `model_capacity="tiny"` (smallest/least accurate),
16 kHz, 20 ms step, confidence threshold 0.60.

---

## 10. Open questions

Genuinely unresolved; each would need its own investigation.

1. **Lead-vs-backing vocal separation** — is there any production-viable
   approach in 2026? This gates accuracy on commercially produced tracks (§2.4).
   Highest-value unknown.
2. **Chroma vs. discrete chords** — does frame-level chord labeling actually
   outperform a continuous chroma salience prior for this scoring task? Hsieh et
   al. used *key*, not chords; nobody has published the chord-level comparison
   (§7.2).
3. **CREPE on separated vocals** — quantify the accuracy loss versus clean
   a cappella, on this repo's own samples. Nobody has published it; it is
   cheaply measurable locally.
4. **Tuning estimation reliability** — how stable is `librosa.estimate_tuning`
   across a full song versus 45 s windows, and on separated stems versus full
   mix?
5. **Expressive-intonation targets** — is it practical to make the cents target
   interval-aware (thirds −14 ¢, leading tones +10 ¢) without overfitting? No
   prior art applies this to automatic scoring.
6. **Beat/chord alignment slack** — what anticipation window actually maximizes
   agreement with human judgment in R&B? §6.4 suggests ±1 beat; unvalidated.
7. **Ground truth** — this project has **no labeled data and no test suite**. Any
   claim that a redesign is "more accurate" is currently unfalsifiable. Hsieh et
   al. resorted to commercial Tunebat labels for exactly this reason.
8. **Commercial practice** (§7.9) — unverified.

---

## Appendix: quick citation index

Highest-value sources, by purpose:

| Need | Source |
|---|---|
| Validates the whole approach | Hsieh et al., Interspeech 2025, `10.21437/Interspeech.2025-1015` |
| Cautionary counter-result | Zhang et al., ISMIR 2021, `archives.ismir.net/ismir2021/paper/000103.pdf` |
| Field overview | dos Santos & Masiero 2026, arXiv:2601.12153 |
| Separation tool + license | github.com/facebookresearch/demucs (MIT) |
| Tuning estimation | `librosa.estimate_tuning`; Gómez 2005 (Essentia) |
| Chroma/tuning preprocessing | Mauch & Dixon, NNLS-Chroma, ISMIR 2010 |
| Predominant-f0 in polyphony | Goto 2004, *Speech Communication* 43(4) |
| Reference-free baselines | Gupta et al., TASLP 2020 `10.1109/TASLP.2019.2947737`; Nakano et al. Interspeech 2006 |
| ACE maturity | Pauwels et al., ISMIR 2019 |
| Consonance modeling | Hutchinson & Knopoff 1978; Parncutt 1989; Sethares 1998 |
| Intonation reference from ensemble | Weiss et al., ISMIR 2019 |
| f0 reliability gating | Rosenzweig et al., ICASSP 2021 |
