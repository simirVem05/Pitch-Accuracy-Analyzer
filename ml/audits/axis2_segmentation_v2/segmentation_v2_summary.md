# Axis 2 — Note segmentation v2: shadow evaluation and promotion recommendation

Shadow-mode comparison of candidate segmenters against production v1 (`backend/note_segmentation.py`) on the
**50 `axis2_ml` songs**. All segmentations are recomputed in memory from the saved full vocal tracks in
`data/main_data/<song>/harmony.npz`. The source audio no longer exists, and none of this needed it: no Demucs, no CREPE,
no audio preprocessing. Nothing under `data/` was written; integrity hashes are at the end.

The invalidity definitions (clear / likely / ambiguous, IQR, gap, and plateau rules) are those of
`ml/audits/axis2_segmentation_tail/`. The classifier is imported from that audit, not re-implemented, and it was
re-asserted to reproduce the tail audit's per-note categories on v1 with **0 mismatches**. v1 itself reproduces
all **22,241** canonical notes (frame indices, targets, medians, times, cents) exactly (`_verification.json`).

## Headline

**Recommend promoting `hybrid_gap_v1` with a local gap reference** (`gapv1local_g0.5_s100`) as the canonical
Axis 2 segmentation:

| | v1 (canonical) | **recommended** |
|---|---|---|
| clear-invalid | 2,683 (12.06%) | **235 (0.92%)** |
| likely-invalid | 4,541 (20.42%) | **2,094 (8.20%)** |
| notes bridging an unvoiced gap > 0.5 s | 1,024 | **0** |
| v1-valid notes left unchanged | — | **94.4%** |
| notes | 22,241 | 25,547 (+14.9%) |

It removes the catastrophic failure mode the tail audit found: v1 never looks at time, so it bridges silences up to 10.4 s.
It leaves nearly all valid v1 notes alone. Its residual problems are either inherited unchanged from v1
(vibrato over-splitting) or small and characterised (same-pitch tiny tails, ~1.6% of notes).

One change was made to the configuration the interrupted session had recommended (`gapv1_g0.5_s100`), and it was
found by inspection rather than by a new sweep. Its short-gap rule measured the "before" pitch over up to 1 s of
material that, in the hybrid modes, spans earlier v1 notes. That made **836** new boundaries split a *held* pitch
across a 60–160 ms dropout (§5). The fix is the opt-in `gap_before="local"`, which mirrors the existing "after"
window. It improves every collateral metric at the same parameters and reproduces the original sweep exactly when
off (§1.2).

---

## 1. Recovered sweep and configurations

### 1.1 Completed sweep (reused, not rerun)

`parameter_comparison.csv` has 100 configurations, 90 columns, no empty or NaN cells. Its v1 row matches the
canonical store, and its hybrid and pure rows were reproduced exactly by `resume_analysis.py` (note counts 22,241 /
26,056 / 22,483; hybrid clear-invalid 1.1053% both ways). The pre-interruption detail dumps for `gapv1_g0.5_s100`
(`v1_vs_v2_notes.csv`, `v2_notes.csv`, `representative_examples.csv`) were complete and internally consistent: 26,056
rows, a single config. They are preserved unmodified in `superseded_gapv1_g0.5_s100/`. The top-level copies were
regenerated for the recommended configuration.

### 1.2 Follow-up (new, small)

`followup_gap_reference.py` runs **22 configurations** through the identical harness and metrics into
`followup/parameter_comparison_gap_reference.csv`:

- v1 and v1+hardgap0.5
- hybrid × {hard gap 0.25, 0.35, 0.5} × {separation 70, 100, 120} × {center_window, local}
- two gap-context lengths

Every configuration that also appears in the original sweep **reproduces it byte-for-byte in all 90 columns**, so
the code change (`gap_before`, default `"center_window"`) does not alter any prior result.

### 1.3 Exact configurations

All three are `SegmentationV2Config` in `backend/note_segmentation_v2.py`. Unlisted fields are at their defaults.

| role | name | mode | hard_gap_s | separation_cents | gap_before | gap_context_frames | center_window_frames | target_estimator | persistence / return |
|---|---|---|---|---|---|---|---|---|---|
| **recommended** | `gapv1local_g0.5_s100` | `gap_v1` | 0.50 | 100 | `local` | 10 | 50 (unused by `local`) | `core_median` | unused in `gap_v1` |
| superseded hybrid | `gapv1_g0.5_s100` | `gap_v1` | 0.50 | 100 | `center_window` | 10 | 50 | `core_median` | unused |
| best pure-sustained | `v2_g0.5_s70_p3` | `sustained` | 0.50 | 70 | `center_window` | 10 | 50 | `core_median` | persistence 3, return 3 |

**Pure-sustained choice.** `v2_g0.5_s70_p3` uses the same hard gap as the hybrid, so the two are comparable. The
lowest clear-invalid rate of any pure config is `v2_g0.25_s70_p3`: 0.62% clear, 4.39% adjacent-merge, +4.2% notes. It
has the same structural problem, and §6 lists both.

### 1.4 The recommended algorithm, exactly

On the voiced frames (`NaN` = unvoiced, the contract is unchanged) of one song:

1. **Hard gap.** Any gap > **0.50 s** between consecutive voiced frames ends the note unconditionally.
2. **Short-gap rule.** Within each hard-gap region, at every gap > 1.5 hops (> 30 ms), compare two medians, in
   tuning-corrected cents:
   - the gap-free run just before it, at most 10 frames and not crossing the last split;
   - the gap-free run just after it, at most 10 frames.

   If they differ by > **100 ¢**, split there.
3. **v1 inside each piece.** Run v1's own commit rule (`segment_notes_with_frame_indices`) separately inside each
   resulting piece: drift ≥ 40 ¢ held for > 3 consecutive voiced frames that round to the same new semitone. Semitone
   steps are handled exactly as v1 handles them.
4. **Drop and target.** Drop pieces with < 3 voiced frames (v1's rule). The target is v1's rule (§8).

Output dicts and frame-index lists have v1's schema, so they slot into `notes.npz` unchanged.

**Why hard gap 0.5 s, not 0.25 s.**
- 0.5 s is the tail audit's own *clear* gap threshold, so the unconditional rule removes only what is definitionally
  invalid.
- Gaps of 0.25–0.5 s go through the pitch-aware short-gap rule instead, so a held note re-articulated across a
  0.3 s breath is not cut.
- 0.25 s looks better on likely-invalid (5.84% vs 8.20%), but most of that gain is circular: "likely D" *is* "gap
  > 0.25 s".
- On clear-invalid the two differ by 0.2 points (0.72% vs 0.92%), and 0.25 s adds 645 more notes.

---

## 2. Recommended v2 vs canonical v1 (detail)

Source: `followup/parameter_comparison_gap_reference.csv`, `_resume_analysis_gapv1local_g0.5_s100.json`.

| metric | v1 | recommended | change |
|---|---|---|---|
| total notes | 22,241 | 25,547 | **+3,306 (+14.86%)** |
| assigned voiced frames | 402,379 | 399,777 | −0.65% (pieces < 3 frames dropped) |
| clear-invalid | 2,683 (12.06%) | **235 (0.92%)** | −91% |
| likely-invalid (B∪C∪D) | 4,541 (20.42%) | **2,094 (8.20%)** | −54% |
| ambiguous (E, flagged) | 920 (4.14%) | 1,351 (5.29%) | +431 |
| IQR > 100 ¢ | 3,801 (17.09%) | 3,698 (14.48%) | |
| IQR > 200 ¢ | 1,887 (8.48%) | 1,373 (5.37%) | |
| IQR > 400 ¢ | 717 (3.22%) | **320 (1.25%)** | |
| internal gap > 0.10 s | 4,664 | 1,963 | |
| internal gap > 0.25 s | 2,291 | 620 | |
| internal gap > 0.50 s | 1,024 | **0** | |
| internal gap > 1.0 s | 379 | **0** | |
| maximum internal gap | 10.38 s | **0.50 s** | |
| duration < 0.06 s | 1 (0.00%) | 519 (2.03%) | |
| duration < 0.08 s | 1,578 (7.10%) | 2,761 (10.81%) | |
| duration < 0.10 s | 3,039 (13.66%) | 5,159 (20.19%) | |
| duration < 0.15 s | 6,874 (30.91%) | 10,977 (42.97%) | |
| median duration | 0.26 s | 0.18 s | |
| P90 / P95 / P99 duration | 1.02 / 1.44 / 2.86 s | 0.72 / 1.02 / 1.84 s | |
| maximum duration | 15.20 s | 15.16 s | see below |
| notes > 2 s / > 5 s / > 10 s | 537 / 62 / 6 | 194 / 9 / 1 | |
| median frame count | 12 | 10 | |

The two maxima are different notes:
- **v1's 15.20 s** is Stapleton #319, six fragments glued across an 8.7 s silence (§7).
- **The recommended 15.16 s** is `jeff_buckley_hallelujah` #713, a genuine held note: 759 *contiguous* frames,
  IQR 17 ¢, confidence 0.95, category A.

Duration is `t[last] − t[first]`, so a 3-frame note is 0.04 s. "< 0.06 s" therefore means exactly 3 frames, which v1's
"> 3 agreeing frames" rule almost never produces.

**Remaining 235 clear-invalid:** B 191, D 38, BD 5, C 1.
- All 43 notes with a clear-D flag carry a 0.25–0.5 s gap *with* a ≥ 100 ¢ pitch step across it (the clear-D rule's
  second clause).
- The remainder are multi-pitch units the v1 commit rule does not see (§9). None contains an octave jump.

## 3. What the fix targets

v1 notes are mapped to recommended notes by shared source frames, using the sweep's rule: same = best IoU ≥ 0.8;
split = ≥ 2 pieces each holding ≥ 3 frames and ≥ 15% of the note; merged = one note holds ≥ 85% but IoU < 0.8;
otherwise partial.

| | count |
|---|---|
| v1 clear-invalid notes | 2,683 |
| — **fixed** (no overlapping v2 note is clear-invalid) | **2,450 (91.3%)** |
| — of those, every overlapping note is fully valid (not even likely-invalid) | 2,074 (77.3%) |
| — **remaining** (still touches a clear-invalid v2 note) | **233 (8.7%)** |
| v1 likely-invalid notes fixed (no overlapping likely-invalid note) | 2,450 of 4,541 (54.0%) |
| previously valid v1 notes that now touch a clear-invalid v2 note | **10 of 17,700 (0.06%)** |
| previously valid v1 notes that now touch a likely-invalid v2 note | 80 (0.45%) |
| v2 clear-invalid notes built *only* from valid v1 notes | **0** |
| v2 likely-invalid notes built only from valid v1 notes | 19 |

**How targeted the fix is.** 2,964 of 22,241 v1 notes (13.3%) change at all. Of those changed:
- **66.6% were v1 likely-invalid**, against a 20.4% base rate (3.3× enrichment);
- **54.1% were v1 clear-invalid**, against 12.1% (4.5×).

**Collateral-change rate.** 989 of the 17,700 v1-valid notes change (**5.59%**): 511 split (2.89%), 263 merged
(1.49%), 215 partial (1.21%). The 549 boundaries cutting a split valid note break down as follows:
- **362 (65.9%)** separate pitches ≥ 80 ¢ apart (348 at a short gap, 14 contiguous). These are real pitch changes the
  tail classifier is too coarse to flag.
- 131 (23.9%) separate the same pitch (< 50 ¢).
- 56 sit in between.

So at most ~1% of valid notes receive a questionable cut. All 10 of the stable-center audit's "extremely stable"
example notes are unchanged. The superseded hybrid split one of them, `teri_ore` #273, a 15-frame MIDI 67 note,
across a 60 ms dropout. The other 24 stable-center examples, drifting or onset/release-differing notes, are all
split; they are multi-pitch units by the tail audit's measures.

For contrast, the superseded `center_window` hybrid changed 7.2% of valid notes. 458 of its 905 cuts in valid notes
separated the same pitch.

## 4. v1 → v2 mapping

| v1 notes | → 0 | → 1 | → 2 | → 3 | → 4+ |
|---|---|---|---|---|---|
| all (22,241) | 9 (0.04%) | 19,612 (88.18%) | 2,331 (10.48%) | 258 (1.16%) | 31 (0.14%) |
| v1-valid (17,700) | 8 | **17,117 (96.71%)** | 557 (3.15%) | 16 (0.09%) | 2 |
| v1 clear-invalid (2,683) | 0 | 1,025 (38.20%) | **1,434 (53.45%)** | 198 (7.38%) | 26 (0.97%) |

"→ 1" counts a note that keeps one substantive piece. It includes the 1,079 clear-invalid notes that are fixed by
detaching a short island while keeping IoU ≥ 0.8.

Outcomes across all v1 notes: same 19,277 (86.67%), split 2,344 (10.54%), merged 283 (1.27%), partial 337 (1.52%).
**Many → one:** 54 recommended notes (0.21%) contain ≥ 2 v1 notes nearly whole (≥ 50% of each). That happens where
v1, restarted fresh inside a gap piece, does not re-commit a boundary it had committed with its carried-over state.

The hybrid is therefore **splitting bad gapped segments, not rewriting segmentation**. Splits concentrate in invalid
notes: 1,658 of 2,683 clear-invalid notes are split into 2+ pieces, against 575 of 17,700 valid ones. The pure
sustained alternative is the opposite: only 47.5% of v1 notes survive unchanged, 29.4% merge, and 10.3% of its notes
contain ≥ 2 whole v1 notes.

## 5. Where the new tiny notes come from

The recommended segmentation has 5,159 notes < 0.10 s, against v1's 3,039.
- 3,076 of them match a v1 note (IoU ≥ 0.8) and are not new.
- **2,083 are new.** Each was traced to its v1 source note(s) and classified by the boundaries on either side. Every
  boundary was labelled hard (> 0.5 s), short (30–500 ms) or contiguous, with its local pitch step taken as the median
  of ≤ 10 gap-free frames on each side.

Rules are applied in this order (`tiny_notes_gapv1local_g0.5_s100.csv` has every note):

| cat | rule | < 0.06 s (518) | < 0.08 s (1,170) | < 0.10 s (2,083) |
|---|---|---|---|---|
| **A** legitimate short pitch after a gap | stable (P5–P95 ≤ 100 ¢), conf ≥ 0.70, short-gap side with step ≥ 80 ¢ | 159 (30.7%) | 355 (30.3%) | **586 (28.1%)** |
| **B** adjacent short note exposed | stable, conf ≥ 0.70, step ≥ 80 ¢ on a contiguous side | 33 (6.4%) | 83 (7.1%) | 111 (5.3%) |
| **C** isolated fragment exposed by hard-gap split | no contiguous neighbour, at least one side a hard gap or the track edge | 0 | 21 (1.8%) | 221 (10.6%) |
| **D** v1 boundary interaction | unstable glide or scoop cut off by v1 restarting fresh in a piece | 107 (20.7%) | 253 (21.6%) | 392 (18.8%) |
| **E** low-confidence fragment | median CREPE confidence < 0.70 | 96 (18.5%) | 179 (15.3%) | 275 (13.2%) |
| **F** likely over-segmentation | same pitch (< 50 ¢) as a non-hard neighbour | 107 (20.7%) | 255 (21.8%) | **404 (19.4%)** |
| **G** ambiguous | none of the above | 16 (3.1%) | 24 (2.1%) | 94 (4.5%) |

**71.8%** of the new tiny notes come out of v1 likely-invalid notes, and 61.0% out of clear-invalid ones.

**Verdict: mostly a natural consequence of un-merging, with one identified minority defect.**

- **A + B + C (44%)** is real material v1 had glued onto an unrelated note. An isolated 60–100 ms island (C) may not
  be a sung note, but it was already an island; v1 just hid it inside a longer note, where it contaminated that
  note's statistics.
- **E (13%)** is marginal-confidence material that v1 had likewise absorbed.
- **D (19%)** are scoop or glide fragments. When v1 restarts at a piece boundary, its first frame sets the reference
  semitone, so an onset scoop can be committed as its own short note. Under v1's carried-over state it was absorbed.
  This is v1's own scoop behaviour (the synthetic "onset scoop" case splits under v1 too), exposed more often.
- **F (19%, 404 notes, 1.6% of all notes) is a new defect, and its mechanism is identified.**
  - The typical case: a same-pitch tail resumes after a 60–160 ms gap, then the pitch changes a few frames later.
  - The 10-frame "after" window reaches past the tail into the next pitch, so the short-gap rule fires at the gap
    even though the change happens later. v1 then commits the real change, leaving a ≤ 5-frame tail that belongs to
    the previous note.
  - 292 of 404 have exactly this short-gap-left / contiguous-right signature.
  - Shortening the context window does not fix it: at 4 frames, clear-invalid rises to 1.00% and vibrato-note
    splits to 3.1%; at 6 frames the result is mixed: clear-invalid 0.94%, vibrato-note splits 1.15% (`followup/`). It is a known residual, not a blocker.

**The superseded hybrid, for comparison.** Its 2,175 new tiny notes have essentially the same category mix, but
there are more of them. Its `center_window` reference was the source of the 836 same-pitch short-gap boundaries,
which the fix cut to 265.

## 6. Side by side

| | v1 | **recommended** `gapv1local_g0.5_s100` | superseded `gapv1_g0.5_s100` | pure `v2_g0.5_s70_p3` | pure `v2_g0.25_s70_p3` |
|---|---|---|---|---|---|
| total notes | 22,241 | 25,547 | 26,056 | 22,483 | 23,170 |
| note-count change | — | +14.9% | +17.2% | +1.1% | +4.2% |
| clear-invalid | 12.06% | **0.92%** | 1.11% | 0.69% | 0.62% |
| likely-invalid | 20.42% | 8.20% | 8.11% | 6.43% | 4.02% |
| IQR > 400 ¢ | 3.22% | 1.25% | 1.30% | 1.01% | 0.99% |
| gap > 0.5 s | 4.60% (1,024) | **0** | 0 | 0 | 0 |
| tiny (< 0.10 s) | 13.66% | 20.19% | 20.12% | 15.70% | 16.09% |
| median duration | 0.26 s | 0.18 s | 0.18 s | 0.20 s | 0.18 s |
| **short adjacent-pitch merge rate** | **0.82%** | **0.42%** | 0.41% | **4.60%** | **4.39%** |
| v1 notes unchanged | 100% | 86.7% | 85.1% | 47.5% | 46.6% |
| v1-valid notes unchanged | 100% | **94.4%** | 92.8% | 52.3% | 52.5% |
| v1 notes merged into another | 0 | 1.27% | 1.22% | **29.4%** | 28.7% |
| v1 vibrato-like notes split | 0 | **0.76%** | 4.78% | 4.97% | 4.97% |

**Adjacent-pitch merge rate.** This is the sweep's `pct_multi_semitone`: a note with two raw plateaus of ≥ 4 frames
whose centres are ≥ 80 ¢ apart, *and* a ~180 ms rolling median that moves ≥ 80 ¢ (vibrato barely moves it). The
previously quoted "0.82% vs ≥ 4.4%" is confirmed exactly: v1 0.8183%, pure 4.39–4.60%. The recommended hybrid
*halves* v1's rate, to 0.42%.

**Why the hybrid wins even though pure sustained has the lower invalid rate.** Pure sustained gets its low rate by
merging:
- 29% of v1 notes get absorbed into neighbours;
- only half of the *valid* v1 notes survive;
- notes holding two adjacent pitches (the merge-rate metric above) become 5.6× more common than in v1.

Those merged notes contain two pitches but are not flagged, because the tail classifier's B rules need two ≥ 6-frame sustained regions, a ≥ 150 ¢ plateau spread with ≥ 200 ¢ of slow
movement, or a ≥ 300 ¢ step across a break, so a 100 ¢ step between two short notes slips under them. Its lower invalid rate
partly reflects a coarser classifier, not better segmentation. The hybrid gets 98% of pure sustained's clear-invalid reduction (11.14 of 11.37 points)
while keeping v1's semitone sensitivity and 94% of valid notes intact.

## 7. Known failure cases

`representative_examples.csv` has these alongside the tail audit's 64 extreme notes and the stable-center
examples. Full traces are in `_detail_gapv1local_g0.5_s100.json`.

### Chris Stapleton #319 (`chris_stapleton_tennessee_whiskey_official_audio`)

| version | segment | start–end (s) | dur (s) | frames | target MIDI | IQR (¢) | largest gap (s) | target share | category |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 319 | 176.28–191.48 | **15.20** | 35 | 63 | **595** | **8.74** | 0.46 | DB, clear-invalid |
| rec. | 370 | 176.28–176.38 | 0.10 | 6 | 63 | 12 | 0.02 | 1.00 | valid |
| rec. | 371 | 177.28–177.84 | 0.56 | 15 | 63 | 40 | 0.30 | 0.67 | D-likely (same-pitch 0.3 s breath) |
| rec. | 372 | 186.58–186.68 | 0.10 | 6 | 57 | 44 | 0.02 | 0.50 | valid |
| rec. | 373 | 188.72–188.82 | 0.10 | 6 | 53 | 66 | 0.02 | 0.67 | A (flagged, plausible) |

**Confirmed: the 15.2 s, six-island unit is no longer one note.** It becomes four notes on three different pitches,
none clear-invalid, plus 2 frames dropped: the final 2-frame island at 191.46 s is below the 3-frame minimum. The
largest gap inside any piece is 0.30 s, at the same pitch.

### Starboy #2 (`the_weeknd_starboy_audio_ft_daft_punk`)

| version | segment | start–end (s) | dur (s) | frames | target MIDI | IQR (¢) | largest gap (s) | target share | category |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 2 | 5.10–16.74 | **11.64** | 20 | **57** | **1,602** | **10.08** | **0.00** | DCB, clear-invalid |
| rec. | 3 | 5.10–5.48 | 0.38 | 9 | 67 | 5 | 0.24 | 1.00 | A |
| rec. | 4 | 15.56–15.66 | 0.10 | 6 | 51 | 168 | 0.02 | 0.50 | A |
| rec. | 5 | 16.66–17.90 | 1.24 | 45 (5 from v1 #2; rest from v1 #3) | 62 | 18 | 0.30 | 0.89 | D-likely (same pitch) |

**Confirmed resolved.** v1's target, MIDI 57, is a pitch that is never sung: 0% of frames lie within ±50 ¢ of it,
because the median of islands at ≈ 67, 51 and 62 falls between them. Once segmentation separates the islands, the
*unchanged* core-median rule yields 67, 51 and 62, each represented. Segmentation fixed it; the target rule did not
need to change.

## 8. Target MIDI

- **v1 rule:** `round(median(core_slice(midi_float)))`, the rounded median of the middle 60% of the note's voiced
  frames in tuning-corrected MIDI (`_finalize_segment`).
- **Recommended v2 rule: unchanged** (`target_estimator="core_median"`). It reads only the note's own f0 frames and
  the tuning offset, never chroma or harmonic fit, so Axis 2 targets stay independent of Harmonic Fit.
  (`pitch_class`, which Axis 1 consumes, is derived *from* the target, not the other way round.)

"Represented" means ≥ 50% of a note's frames lie within ±50 ¢ of its target.

| | represented |
|---|---|
| v1, core median | 90.30% |
| recommended, core median | 89.56% (notes ≥ 10 frames: v1 85.48% → **87.89%**) |
| recommended, modal (tested, not recommended) | 92.00% |

The small overall dip comes from the larger share of short notes; for sustained notes representation improves.

Of the recommended segmentation's 2,668 unrepresented notes:
- **2,045 (77%)** have *no* semitone holding half their frames. These are wide-vibrato, gliding or still-multi-pitch
  notes that no single integer target can represent.
- **623** have a different semitone that does hold ≥ half.

Modal targeting fixes exactly those 623, breaks none, and changes 1,751 targets in total. It is **not recommended in
this promotion**:
- That result is close to tautological: the modal rounded semitone is by construction the one with the most frames
  within ±50 ¢, so it maximises this metric whether or not it is the intended note.
- Changing it would alter both Axis 2 contours and the production Axis 1 `pitch_class` in the same step as the
  segmentation change.

It is a reasonable, separately measurable follow-up. The cases that motivated it, such as Starboy #2, were
segmentation failures, and the recommended segmentation already fixes them.

## 9. Remaining limitation: vibrato over-splitting (inherited, not new)

The hybrid keeps v1's commit rule, so it keeps v1's known weakness. A vibrato wide enough to spend ≥ 4 frames beyond
the ±50 ¢ rounding boundary is cut into alternating semitone "notes".

**Synthetic checks** (`synthetic_checks.py`, new `recommended` column). The recommended configuration fails 6 cases,
**exactly the same 6 v1 fails**, and passes the hard-gap case v1 fails:

| case | v1 | recommended |
|---|---|---|
| ±150 ¢ vibrato | 2 | 2 |
| ±120 ¢ vibrato @ 4.5 Hz | 18 | 18 |
| ±120 ¢ vibrato @ 4 Hz | 16 | 16 |
| vibrato centred +50 ¢ | 22 | 22 |
| slow +80 ¢ drift | 2 | 2 |
| −200 ¢ onset scoop | 2 | 2 |

Pure sustained passes all of them, which was its design goal.

**Real data.** An alternating chain is ≥ 3 consecutive gap-free notes, each ≤ 0.30 s, whose targets alternate
A, B, A one semitone apart. This is an upper bound on vibrato splitting, because trills and fast melismas match too.

| | chains | notes in chains | % of notes | chains ≥ 5 notes |
|---|---|---|---|---|
| v1 | 392 | 1,363 | 6.13% | 40 |
| recommended | 464 | 1,600 | 6.26% | 46 |
| pure `v2_g0.5_s70_p3` | 108 | 343 | 1.53% | 3 |

The hybrid also newly splits only **4 of the 523** v1 vibrato-like notes (0.76%; the superseded hybrid split 25).

**Severity.** The limitation is real, affects at most ~6% of notes, and is **identical in the current canonical
data**, so promoting the hybrid neither creates nor worsens it. It matters for Axis 2: each fragment's cents are
measured against an alternating target, which inflates apparent deviation. That makes it the next segmentation
problem to solve, but **it does not block promotion**. Fixing it means replacing v1's commit rule, which is where pure
sustained over-merged, so it needs its own design work.

## 10. Decision: promote

Promote `gapv1local_g0.5_s100` as the canonical Axis 2 segmentation:

| criterion | assessment |
|---|---|
| clearly invalid notes | 12.06% → 0.92% (−91%); 91.3% of v1's clear-invalid notes fixed |
| long-gap merges | gaps > 0.5 s: 1,024 → 0; max internal gap 10.38 s → 0.50 s |
| collateral on valid notes | 94.4% unchanged; ≤ ~1% receive a questionable cut; 0 clear-invalid notes created from valid material |
| tiny notes | +2,083 new; ~44% recovered real material, ~32% scoop/low-confidence material v1 had absorbed, ~19% a characterised same-pitch-tail defect (1.6% of all notes) |
| adjacent-note merging | halved vs v1 (0.82% → 0.42%); pure sustained ×5.6 |
| vibrato | v1's over-splitting inherited unchanged (~6% upper bound); no new vibrato damage |
| note count | +14.9%, almost entirely splits of invalid notes |

**What promotion does *not* cover.** The evidence is on the 50 `axis2_ml` songs. `data/main_data` holds 63 songs,
and the other 13 (rap, multiple leads, heavy effects) feed `harmonic_fit_eval`. Run the same shadow comparison on
those 13 before regenerating their notes. Promoting into the **production backend** (`main.py`, `api.py`,
`evaluate.py`) is a separate decision: it changes Axis 1's inputs (more and shorter notes, duration weighting) and
needs Axis 1 re-validation. `evaluate.py` needs audio, so on this corpus that means porting it to read the stored
`other_chroma_rank`, beats and tuning from `harmony.npz`.

## 11. Regeneration plan (not executed)

`harmony.npz` is now the highest recoverable source. It holds every input the segmenter uses
(`vocal_frame_times_s`, `vocal_f0_hz`, `vocal_crepe_confidence`, `tuning_offset_semitones`), and v1 notes were
proven exactly reproducible from it.

**Must be regenerated**
1. `data/main_data/<song_id>/notes.npz`, for all 63 songs. The format is unchanged: the same 15 arrays and the same
   CSR offsets, built with the existing `_note_arrays(segments, frame_indices, track, tuning)`.
2. `data/main_data/<song_id>/metadata.json`:
   - `number_of_segmented_notes`
   - new provenance fields: `segmentation_algorithm`, `segmentation_config` (the full `SegmentationV2Config`),
     `harmony_npz_sha256`, `resegmented_at_utc`, `resegmentation_git_commit`
   - keep `source_sha256`, `git_commit` and the processing timestamp: they still describe the extraction that
     produced `harmony.npz`.
   - Bump `dataset_schema_version` 1 → 2 and update `validate_song_output`, so code expecting v1 notes fails loudly
     rather than silently mixing segmentations.
3. `data/main_data/manifest.json`: `number_of_notes` per song, `updated_at_utc`, schema version.
4. `data/ml_dataset/axis2_v2/`, a **new directory**; do not overwrite `axis2_v1`.
   - Reuse `splits.json`'s song assignments (song-level, locked before corruption), with recomputed counts.
   - Seeds are derived from `clean_note_id`, so draws change with note indices, as expected for a new version.
   - **The builder for `axis2_v1` is not in the repo.** It has to be written and committed first, following
     `axis2_v1/README.md` and `summary.json`.

**A new script is required.** `scripts/build_main_dataset.py` cannot do this. It starts from audio, runs Demucs and
CREPE, and its idempotency check is keyed on the hash of source audio that no longer exists. Write
`scripts/resegment_main_dataset.py`:
- read `harmony.npz`, segment, build notes;
- write with the existing `_atomic_npz` / `_atomic_json`;
- re-validate every song with `validate_song_output(expected_source_sha256=None)`;
- save the manifest after every song;
- never touch `harmony.npz`.

**Unchanged**
- `harmony.npz`, all 63.
- `data/dataset_selections.json`. Selection is song-level (delivery style, number of lead vocalists, effects,
  extraction quality), and none of it depends on note boundaries; its counts are song counts.
- `ml/results/axis2_v1/*`, which stays tied to `axis2_v1`. Rerun the baselines on `axis2_v2` afterwards (not part of
  this task).
- The tail and stable-center audits, which are audits of v1 and remain valid as such.

## 12. Preserving v1 (when migrating)

`data/` is gitignored, so **there is no version-control safety net for any of these files.** Do both of the following:

1. **Full verified snapshot before any write.**
   - `cp -Rp data/main_data data/main_data_v1_<date>`, plus a SHA-256 list of every file, re-checked after the copy.
   - Keep a second copy off this machine.
   - `axis2_v1` needs no copy, because the new dataset goes to `axis2_v2`.
2. **In-place sidecar.** Copy each song's current `notes.npz` byte-for-byte to `notes_v1.npz` and record its hash in
   `metadata.json` (`notes_v1_sha256`) before writing the v2 `notes.npz`. Every song directory then keeps both
   segmentations side by side, and the same `validate_song_output` checks can be pointed at either.

**Backstop.** `harmony.npz` plus `backend/note_segmentation.py` regenerate v1 exactly (22,241 / 22,241, 0
mismatches). Keep v1's segmenter in the repo, unmodified, after the switch. Commit this audit,
`note_segmentation_v2.py`, and the resegmentation script before migrating, so the migration's `git_commit` points at
real code.

---

## Files

| file | contents |
|---|---|
| `evaluate_segmentation_v2.py` | the sweep harness: v1 reproduction, tail-classifier reproduction, 100 configs |
| `parameter_comparison.csv` | sweep results (unchanged) |
| `_verification.json` | v1 reproduction (22,241 notes, 0 mismatching songs) and classifier reproduction (0 mismatches) |
| `followup_gap_reference.py`, `followup/` | `gap_before` follow-up (22 configs); its own `_verification.json` |
| `detail_outputs.py` | per-note detail for one config |
| `v1_vs_v2_notes.csv` | one row per v1 note: outcome under the **recommended** config and the overlapping v2 notes |
| `v2_notes.csv` | one row per recommended v2 note (25,547), metrics and audit categories |
| `representative_examples.csv` | Stapleton #319, Starboy #2, the 64 tail extremes, stable-center examples: v1 and v2 segments |
| `_detail_gapv1local_g0.5_s100.json` | outcome counts, target-estimator comparison, headline traces |
| `_detail_gapv1_g0.5_s100.json`, `_detail_v2_g0.5_s70_p3.json` | same for the superseded hybrid and the pure alternative (pre-interruption) |
| `resume_analysis.py` | fix accounting, mapping, boundary causes, tiny-note mechanisms, target rule, vibrato chains |
| `_resume_analysis_<config>.json`, `tiny_notes_<config>.csv` | its outputs, for the recommended and the superseded hybrid |
| `synthetic_checks.py` | behavioural pass/fail cases; now includes the recommended config |
| `superseded_gapv1_g0.5_s100/` | the pre-interruption detail CSVs, unmodified |

Code change outside this directory: `backend/note_segmentation_v2.py` gained `gap_before` (default
`"center_window"`, which reproduces every prior result) and the helper `_contiguous_start`. No production module was
changed.

## Integrity

SHA-256 over every file in `data/main_data/`, `data/ml_dataset/` and `data/dataset_selections.json` (197 files under
the two directories), taken before and after this resumed audit:

```
before  ac5a1882e4d0f41be20b587861bded8b3e9f2307dbdd91dd7c15ce0d61a5df19
after   ac5a1882e4d0f41be20b587861bded8b3e9f2307dbdd91dd7c15ce0d61a5df19
```

Identical: no canonical feature file, ML dataset file or selection was modified. No source audio was read, no
Demucs or CREPE was run, no baseline was rerun, and nothing was trained.
