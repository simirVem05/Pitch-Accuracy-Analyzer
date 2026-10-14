# Axis 2 V1 — The extreme tail of segmented notes: expressive singing or segmentation failure?

Read-only descriptive audit of the **22,241 clean canonical notes** of the 50 `axis2_ml: true` songs.
Run date 2026-10-14. No segmentation code, canonical file, ML dataset file, or selection was modified;
no preprocessing was rerun; nothing was trained. Integrity hashes are at the end.

**Headline.** The extreme tail is **mostly segmentation failure, not expressive singing.** The 3.2% of
notes with IQR > 400 ¢ cannot be single sung notes: a pure sinusoidal vibrato of extent ±A has
IQR = √2·A, so IQR > 400 ¢ would require a vibrato extent of more than ±283 ¢, roughly three times the
commonly reported ±50–100 ¢. 80% of these notes contain an unvoiced break, and 98% have a slow pitch
component (≈180 ms rolling median) spanning ≥ 200 ¢.

**12.1% of all notes (2,683) are clearly invalid as single-note units**, and another 8.4% are
probably invalid. The dominant mechanism is that `segment_notes` **never looks at time**. It bridges
unvoiced gaps of any length, up to 10.4 s. The material after the gap is usually short or gliding, so
the rule "more than 3 consecutive frames agreeing on one new semitone" almost never fires, and the next
phrase fragment is absorbed into the previous note.

**CREPE errors are rare** in this tail: 2 clear cases, 236 possible. A fix needs only re-segmentation
from the saved `harmony.npz` tracks, with no Demucs or CREPE. This was verified by exactly reproducing
all 22,241 saved notes from those tracks.

Files in this directory:

| File | Contents |
|---|---|
| `segmentation_tail_audit.py` | the full analysis; deterministic, read-only, ~25 s |
| `segmentation_tail_audit.csv` | one row per clean note (22,241), every metric, flag, and category |
| `extreme_examples.csv` | 64 deduplicated notes (top-20 by IQR, jump, gap, duration) with classification and reason |
| `extreme_contours.csv` | per-frame `time_s`, `cents_deviation`, `crepe_confidence`, `source_frame_index` for those 64 |
| `_summary_stats.json` | every number in this document, machine-readable |

---

## 0. Verification that the analysis sees what production sees

Before any diagnosis, `segment_notes_with_frame_indices` was re-run on each song's saved full vocal
track (`harmony.npz`: `vocal_frame_times_s`, `vocal_f0_hz`, `tuning_offset_semitones`). For all
50 songs it reproduces the saved notes exactly: note count, every source-frame index, target MIDI,
core median, start time, frame times, and confidence. That is 22,241 notes and 0 mismatches.

The audit uses an **instrumented copy** of the segmentation state machine to record *why* each
candidate boundary was abandoned. The copy is asserted, note by note, to produce boundaries identical to
production, so its explanations describe production behavior. The note count and per-note frame count
also match the ML manifest's clean rows exactly. Only `split` and `num_frames` were read from the
manifest; `injected_shift_cents` was never read.

## 1. Flag counts (diagnostic, nothing excluded)

| Flag | Notes | % of 22,241 |
|---|---:|---:|
| IQR > 100 ¢ | 3,801 | 17.09% |
| IQR > 200 ¢ | 1,887 | 8.48% |
| IQR > 400 ¢ | 717 | 3.22% |
| P5–P95 > 200 ¢ | 6,299 | 28.32% |
| max adjacent jump > 100 ¢ | 8,747 | 39.33% |
| max adjacent jump > 200 ¢ | 4,197 | 18.87% |
| largest inter-frame gap > 0.10 s | 4,664 | 20.97% |
| largest inter-frame gap > 0.25 s | 2,291 | 10.30% |
| duration > 2 s | 537 | 2.41% |
| duration > 5 s | 62 | 0.28% |
| duration > 10 s | 6 | 0.03% |
| **any flag** | **10,515** | **47.28%** |

The flag set is deliberately broad. Being flagged does not mean a note is bad: 5,101 flagged notes end up
as plausible expressive single notes.

## 2. Temporal continuity

Only **14,636 of 22,241 notes (65.8%)** consist of consecutive source frames. The other 7,605 (34.2%)
contain at least one unvoiced break, which matches the stable-center audit's 34.2%. Breaks are always ≥ 2
frames: the voicing gate already fills 1-frame holes.

Distributions over the 10,515 flagged notes:

| Metric | P25 | median | P75 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| largest gap (s) | 0.02 | 0.08 | 0.22 | 0.50 | 0.80 | 2.14 | **10.38** |
| non-consecutive gaps per note | 0 | 1 | 1 | 2 | 3 | 4 | 11 |
| voiced time ÷ elapsed span | 0.63 | 0.86 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 (P5 = 0.28) |

The 2,291 notes with a gap > 0.25 s:
- **Gap size.** 1,024 have a gap > 0.5 s, 379 > 1 s, and 122 > 2 s.
- **Pitch across the gap.** In 1,437 the pitch center across the gap moves by ≥ 100 ¢. In 814 it stays
  within 80 ¢, i.e. the same pitch is re-sung after a rest or after a confidence dropout.
- **Confidence inside the gap.** The median frame confidence is 0.38 (P5 0.17, P95 0.54). Gaps are not
  marginal near-misses at the 0.60 threshold; most of their frames are clearly unvoiced.

**The previously observed 15.2 s / 35-frame note** is `chris_stapleton_tennessee_whiskey #319`. It is six
voiced islands of 2–8 frames each, separated by gaps of 0.90, 0.30, **8.74**, 2.04 and 2.64 s. The island
medians are +34, +27, +31, −560, −1008 and −390 ¢. Its voiced time is 0.70 s of 15.2 s elapsed, a ratio
of 0.046. It is not one note on any reading. Its siblings:

- `no_ordinary_love #471`: 15 frames over 13.34 s, with a single **10.38 s** gap.
- `the_weeknd_starboy #2`: 20 frames over 11.64 s, with a 10.08 s gap. Islands sit at MIDI ≈ 67, 51 and 62,
  yet its stored `target_midi` is **57**, a pitch that is never sung. That is the core median of
  unrelated fragments.

Not every long note is broken. `jeff_buckley_hallelujah #646` (15.16 s, 759 consecutive frames,
IQR 17 ¢, confidence 0.95) is a genuine sustained note. `you_raise_me_up #22` (7.4 s, 10 islands, all
within ±40 ¢, max gap 0.26 s) looks like one sustained note with confidence dropouts. That is why
same-pitch gaps of 0.25–0.5 s are rated only "likely D", not clear.

## 3. Pitch continuity (flagged notes)

| Metric | P25 | median | P75 | P90 | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| largest |adjacent jump| (¢) | 113 | 171 | 271 | 504 | 690 | 1,184 | 3,301 |
| P90 adjacent jump (¢) | 49 | 75 | 112 | 173 | 214 | 364 | 1,147 |
| median adjacent jump (¢) | 11 | 18 | 28 | 39 | 47 | 73 | 216 |
| # jumps > 50 ¢ | 2 | 3 | 5 | 8 | 11 | 17 | 52 |
| # jumps > 100 ¢ | 1 | 1 | 2 | 4 | 5 | 8 | 21 |
| # jumps > 200 ¢ | 0 | 0 | 1 | 1 | 2 | 3 | 8 |
| # jumps > 300 ¢ | 0 | 0 | 0 | 1 | 1 | 2 | 6 |

Of the 8,747 notes with a jump > 100 ¢:
- 2,256 have it **only across an unvoiced gap**, where "adjacent" is not adjacent in time.
- 6,491 have one between truly consecutive 20 ms frames.

A 100 ¢ change in 20 ms is faster than vibrato: ±100 ¢ at 6 Hz moves at most ~75 ¢ per frame. It is
typical of run notes, glide edges, and the occasional tracker blip. A single large jump alone was not
treated as an error.

## 4. Multiple pitch regions

Three transparent detectors were used, each recorded per note:

1. **Sustained regions.** Runs where the 9-frame (~180 ms) rolling median stays on one semitone level
   for ≥ 6 frames (120 ms). Neighboring regions < 80 ¢ apart are merged.
2. **Raw plateaus.** Runs of ≥ 3 raw frames (60 ms) on one semitone level. These catch melisma notes
   shorter than the 4-frame commit rule can ever split. They count only when the **slow-component
   range** (range of the 9-frame rolling median) is ≥ 200 ¢. Vibrato oscillates around its center, so
   the rolling median barely moves: P99 for unflagged notes is 109 ¢.
3. **Islands.** Runs of consecutive source frames, with each island's median. A step of ≥ 300 ¢ between
   two islands of ≥ 3 frames is a different pitch after a break.

`apparent_pitch_region_count` (detector 1) over flagged notes: median 1, P90 2, max 7. Detectors 2 and 3
matter more in the tail: of the 717 notes with IQR > 400 ¢, 528 have an island step ≥ 300 ¢.

## 5. CREPE-tracking indicators

- **Octave jumps.** Jumps of 1000–1400 ¢ **between consecutive frames** occur in only **2 notes**. CREPE
  with Viterbi smoothing almost never flips octave mid-run here.
- **Short excursions.** ≤ 3-frame excursions > 200 ¢ from the local median that return, and account for
  over half the note's range, are rare.
- **Octave-sized steps across a break** (1200 ± 150 ¢) are more common: these make up most of the 236
  "C-likely" notes.
- **Very large steps across a break.** 176 notes step ≥ 1000 ¢ between islands, and 27 step ≥ 1500 ¢.
  Their median confidence is 0.81, against 0.90 overall. Two-octave steps inside "one note" are not
  sung intervals. They are more consistent with CREPE latching onto another source in a short fragment:
  backing vocal, ad-lib, or bleed. This can't be confirmed without audio, so these are labeled B with C
  as a possible co-label, not C alone.

Per-note `extreme_jump_confidence`, `confidence_before_jump` and `confidence_after_jump` are in the
main CSV.

## 6. Why segmentation merged these frames

`backend/note_segmentation.py`, `segment_notes_with_frame_indices` (lines 90–162), read without
modification. The complete rule set:

| Concern | Actual behavior |
|---|---|
| unvoiced frames | dropped before segmentation (`voiced_positions = flatnonzero(~isnan(freqs))`, :113); the loop walks **voiced positions only** |
| **temporal gaps** | **never examined.** No code path compares timestamps or source-frame indices; a 10 s gap is the same as 20 ms |
| reference pitch | `current_note` = rounded semitone of the **first voiced frame** (:118), replaced only on commit |
| allowable difference | a frame within `DRIFT_CENTS = 40` of `current_note` resets any candidate (:138) |
| rounding reset | a frame ≥ 40 ¢ off that still rounds to `current_note` also resets (:143) |
| boundary detection | a new note commits only after **`TRIGGER_FRAMES + 1 = 4` consecutive voiced frames all propose the same new semitone** (:147–152); any different proposal restarts the count at 1 |
| boundary placement | at the first frame of the winning candidate run, in voiced positions (:155) |
| minimum note length | `MIN_SEGMENT_VOICED_FRAMES = 3` voiced frames, else the segment is dropped (:15, :68) |
| smoothing / gap filling | none in segmentation; the 1-frame gap fill and 5-frame minimum run happen earlier, in `preprocess.smooth_voicing_mask` |
| final target | recomputed after the fact as the rounded core-trimmed median of all frames (`_finalize_segment`, :55) |

Across all notes, the instrumented loop abandoned **38,411 candidate boundaries**:
- 23,934 (62%) because the proposed semitone **changed** before reaching 4 frames
- 10,774 (28%) because the pitch returned within 40 ¢
- 3,703 (10%) because the pitch rounded back to the current note

In the 717 IQR > 400 ¢ notes, "proposal changed" is the dominant reason in **699 (97.5%)**.

The failures come from the interaction of two rules:

1. **No gap rule, so any silence is bridged.** Example: `stapleton #319`. After the 8.74 s gap, the
   first frame sits 508 ¢ from `current_note = 63`. The candidate runs go 1, then 3 (abort: proposal
   changed), then 3, 1, 2, 1, 2 frames: the fragment glides through MIDI 57.9 → 56.9, then
   54.4 → 51.8 → 53.0. No four consecutive frames ever agree, so 15 s of unrelated fragments stay in the
   note that started at 176.28 s.
2. **The exact-semitone agreement rule cannot split moving pitch.** Glides, fast runs (each note
   < 80 ms), and fragments whose pitch slides by more than ±50 ¢ change the rounded proposal almost every
   frame, so the counter keeps restarting. `no_ordinary_love #471` shows this after its 10.38 s gap:
   proposals 63 → 63 → 64 → 66 → 66 → 66 → 65 → … abort at counts 1, 2, 1, 3. The same mechanism keeps
   melismatic runs inside one note even with no gap: `bruno_mars #372` goes −270 ¢ (4 frames), glides,
   then sits at +250 ¢, with a maximum candidate run of 3.
3. **Contributing:** the reference is the onset frame, and the stored target is recomputed afterwards.
   The decision rule and the stored contour therefore refer to different pitches. In a merged note the
   recomputed target can be a pitch that is never sung (`starboy #2`). 159 notes have
   |`median_cents`| > 100 ¢.

## 7. Categories

Rules, applied identically to every note; all thresholds are diagnostic, not proposals:

- **D, temporal-gap bridging.** Largest gap > 0.25 s. *Clear*: gap > 0.5 s, or > 0.25 s with a ≥ 100 ¢
  pitch change across it.
- **B, multiple notes merged.** Any of: ≥ 2 sustained regions ≥ 80 ¢ apart; ≥ 2 raw plateaus ≥ 150 ¢
  apart with slow-component range ≥ 200 ¢; or an island step ≥ 300 ¢. *Clear*: two ≥ 8-frame regions
  ≥ 100 ¢ apart, or an island step ≥ 300 ¢, or slow range ≥ 300 ¢ with ≥ 4-frame plateaus ≥ 200 ¢ apart.
- **C, CREPE error.** An octave jump between consecutive frames, or short returning excursions that make
  up more than half the range, or an octave-sized step across a break. *Clear*: a consecutive-frame
  octave jump, or a ≤ 3-frame excursion > 300 ¢ that dominates the range.
- **A, plausible expressive single note.** Flagged, no B/C/D, slow-component range < 200 ¢, and
  IQR ≤ 200 ¢. These are vibrato, scoops, falls and drift around one center.
- **E, ambiguous.** Flagged, with none of the above.

Multi-label, flagged notes only (10,515):

| Category | Count | % of flagged | % of all 22,241 |
|---|---:|---:|---:|
| A plausible expressive | 5,101 | 48.5% | 22.9% |
| B multi-note merge | 3,077 | 29.3% | 13.8% |
| C CREPE error | 236 | 2.2% | 1.1% |
| D temporal-gap bridge | 2,291 | 21.8% | 10.3% |
| E ambiguous | 920 | 8.7% | 4.1% |

Combinations: A 5,101 · B 2,096 · D 1,296 · E 920 · **DB 866** · DCB 79 · C 71 · DC 50 · CB 36.
B and D co-occur heavily, which is the mechanism in §6.

"Clear" subsets, over all notes: B 1,604 (7.21%) · C 2 (0.01%) · D 1,736 (7.81%) ·
**any clear 2,683 (12.06%)**. Among the clear cases: D only 1,077 · B only 945 · D + B 659 · C 2.

Validation of the rules on the other side: among the 11,726 unflagged notes, the B/C/D rules fire on only
47, all weak B hits. Among notes with IQR ≤ 100 ¢, B-clear fires on 488. Those are mostly a sustained note
plus a short fragment at a distant pitch after a break: genuinely two events, but the dominant center
survives.

## 8. Representative examples

All 64 deduplicated examples, with every metric and the full reason string, are in
`extreme_examples.csv`; their per-frame contours are in `extreme_contours.csv`.

Class codes below: uppercase = clear, lowercase = likely only.


#### 20 highest-IQR notes

| song_id | idx | start | end | dur | frames | MIDI | med ¢ | IQR | P5–P95 | max jump | max gap s | med conf | class | reason |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| you_make_me_feel_like_a_natural_wo | 301 | 158.30 | 159.24 | 0.94 | 24 | 54 | -49 | 2869 | 3527 | 1576 | 0.50 | 0.78 | BD | 2 islands over 0.50s max gap, island medians 2013;-617; plateaus 2013; break step 2630c; slow range 3408c |
| john_legend_ordinary_people_hq | 0 | 0.44 | 0.80 | 0.36 | 10 | 55 | -35 | 2727 | 3363 | 1965 | 0.20 | 0.73 | B | plateaus ; break step 2836c; slow range 2836c |
| miley_cyrus_party_in_the_u_s_a_aud | 141 | 89.12 | 89.76 | 0.64 | 16 | 80 | -12 | 1931 | 2111 | 2084 | 0.36 | 0.69 | BD | 2 islands over 0.36s max gap, island medians -7;-2005; plateaus -7;-1935; break step 1998c; slow range 1999c |
| dangerously_in_love | 2 | 2.06 | 3.64 | 1.58 | 16 | 80 | -11 | 1918 | 1965 | 1948 | 1.30 | 0.85 | BD | 2 islands over 1.30s max gap, island medians -5;-1931; plateaus -5;-1902; break step 1926c; slow range 1930c |
| demi_lovato_stone_cold_audio_only | 407 | 164.44 | 165.70 | 1.26 | 32 | 79 | 25 | 1869 | 2323 | 661 | 0.62 | 0.88 | BD | 3 islands over 0.62s max gap, island medians 296;-1853;-1563; plateaus 298; break step 2149c; slow range 2298c |
| christina_aguilera_hurt_audio | 440 | 221.94 | 222.38 | 0.44 | 12 | 69 | 5 | 1858 | 1965 | 1394 | 0.24 | 0.64 | B | plateaus 1421; break step 1831c; slow range 1942c |
| celine_dion_because_you_loved_me_t | 433 | 232.24 | 232.82 | 0.58 | 27 | 82 | -6 | 1738 | 2160 | 399 | 0.08 | 0.71 | B | plateaus 203;-1;-1667; break step 1850c; slow range 2092c |
| christina_aguilera_hurt_audio | 174 | 99.26 | 101.46 | 2.20 | 18 | 62 | -28 | 1711 | 2622 | 2767 | 1.72 | 0.86 | BD | 3 islands over 1.72s max gap, island medians -493;1993;-2; plateaus -493;-9; break step 2486c; slow range 2101c |
| teri_ore_full_audio_singh_is_kinng | 0 | 0.90 | 1.16 | 0.26 | 8 | 44 | -36 | 1688 | 2195 | 1214 | 0.14 | 0.82 | B | plateaus ; break step 1773c; slow range 1590c |
| usher_burn | 28 | 27.36 | 28.40 | 1.04 | 16 | 56 | 35 | 1672 | 2099 | 831 | 0.50 | 0.76 | BcD | 3 islands over 0.50s max gap, island medians -1131;-66;864; plateaus -1121; break step 1064c; slow range 1994c; octave-sized step/excursion |
| jazmine_sullivan_bust_your_windows | 478 | 191.40 | 191.70 | 0.30 | 10 | 59 | -42 | 1661 | 1786 | 1241 | 0.14 | 0.81 | B | plateaus -328; break step 1678c; slow range 1689c |
| chris_stapleton_tennessee_whiskey_ | 2 | 7.18 | 7.88 | 0.70 | 11 | 47 | 8 | 1655 | 1896 | 1640 | 0.52 | 0.65 | BD | 2 islands over 0.52s max gap, island medians 1;1697; plateaus 1; break step 1696c; slow range 1696c |
| romeo_santos_propuesta_indecente_a | 398 | 231.70 | 235.40 | 3.70 | 12 | 59 | 386 | 1639 | 2078 | 2134 | 3.50 | 0.76 | BD | 2 islands over 3.50s max gap, island medians 1490;-114; plateaus 1490; break step 1604c; slow range 1604c |
| if_i_ain_t_got_you_alicia_keys_hq_ | 256 | 107.84 | 109.58 | 1.74 | 28 | 52 | -34 | 1620 | 1758 | 1534 | 1.14 | 0.86 | BD | 3 islands over 1.14s max gap, island medians -108;1593;1256; plateaus -116;1290; break step 1701c; slow range 1712c |
| the_weeknd_starboy_audio_ft_daft_p | 2 | 5.10 | 16.74 | 11.64 | 20 | 57 | 511 | 1602 | 1826 | 1599 | 10.08 | 0.80 | BcD | 4 islands over 10.08s max gap, island medians 1036;1035;-650;447; plateaus 1035;-582;424; break step 1685c; slow range 1620c; octave-sized step/excurs |
| celine_dion_because_you_loved_me_t | 434 | 232.84 | 235.18 | 2.34 | 13 | 64 | -29 | 1540 | 1913 | 1379 | 1.18 | 0.70 | BD | 3 islands over 1.18s max gap, island medians -128;-90;1658; plateaus -108; break step 1748c; slow range 1700c |
| miguel_adorn_audio | 455 | 191.24 | 193.30 | 2.06 | 15 | 54 | -48 | 1499 | 1801 | 1319 | 1.80 | 0.72 | BD | 2 islands over 1.80s max gap, island medians 1317;-140; plateaus 1317;-34; break step 1456c; slow range 1650c |
| luther_vandross_here_and_now | 192 | 81.06 | 81.82 | 0.76 | 11 | 62 | 18 | 1448 | 1497 | 1431 | 0.58 | 0.72 | BD | 2 islands over 0.58s max gap, island medians -1395;52; plateaus -1395; break step 1447c; slow range 1455c |
| you_make_me_feel_like_a_natural_wo | 268 | 145.36 | 146.14 | 0.78 | 29 | 76 | -94 | 1440 | 1553 | 1352 | 0.20 | 0.84 | B | plateaus -1382;9;114;-200;-94; break step 1447c; slow range 1510c |
| luther_vandross_here_and_now | 623 | 272.30 | 272.70 | 0.40 | 17 | 69 | 29 | 1434 | 1610 | 673 | 0.06 | 0.80 | Bc | plateaus 201;-1396; break step 1289c; slow range 1568c; octave-sized step/excursion |

#### 20 largest adjacent-jump notes

| song_id | idx | start | end | dur | frames | MIDI | med ¢ | IQR | P5–P95 | max jump | max gap s | med conf | class | reason |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| prince_royce_corazon_sin_cara_audi | 322 | 171.86 | 174.94 | 3.08 | 38 | 52 | 2 | 221 | 3672 | 3301 | 1.18 | 0.73 | BD | 4 islands over 1.18s max gap, island medians 67;3278;-159;-53; plateaus 89;-87;3365;-115;-313; break step 3437c; slow range 3497c |
| christina_aguilera_hurt_audio | 174 | 99.26 | 101.46 | 2.20 | 18 | 62 | -28 | 1711 | 2622 | 2767 | 1.72 | 0.86 | BD | 3 islands over 1.72s max gap, island medians -493;1993;-2; plateaus -493;-9; break step 2486c; slow range 2101c |
| i_have_nothing | 261 | 146.40 | 148.32 | 1.92 | 48 | 71 | 14 | 49 | 1103 | 2745 | 1.00 | 0.87 | D | 2 islands over 1.00s max gap, island medians 14;-1278 |
| christina_aguilera_hurt_audio | 442 | 223.72 | 224.64 | 0.92 | 12 | 83 | -2 | 18 | 1160 | 2529 | 0.68 | 0.80 | D | 3 islands over 0.68s max gap, island medians -1;2;-2509 |
| don_t | 176 | 77.74 | 79.30 | 1.56 | 11 | 54 | 33 | 32 | 1327 | 2420 | 0.78 | 0.75 | D | 3 islands over 0.78s max gap, island medians 10;38;2464 |
| romeo_santos_propuesta_indecente_a | 398 | 231.70 | 235.40 | 3.70 | 12 | 59 | 386 | 1639 | 2078 | 2134 | 3.50 | 0.76 | BD | 2 islands over 3.50s max gap, island medians 1490;-114; plateaus 1490; break step 1604c; slow range 1604c |
| miley_cyrus_party_in_the_u_s_a_aud | 141 | 89.12 | 89.76 | 0.64 | 16 | 80 | -12 | 1931 | 2111 | 2084 | 0.36 | 0.69 | BD | 2 islands over 0.36s max gap, island medians -7;-2005; plateaus -7;-1935; break step 1998c; slow range 1999c |
| i_have_nothing | 164 | 98.08 | 99.18 | 1.10 | 30 | 69 | -207 | 721 | 2207 | 2073 | 0.46 | 0.84 | BD | 3 islands over 0.46s max gap, island medians -204;1225;-910; plateaus -204;1222; break step 2136c; slow range 2129c |
| john_legend_ordinary_people_hq | 0 | 0.44 | 0.80 | 0.36 | 10 | 55 | -35 | 2727 | 3363 | 1965 | 0.20 | 0.73 | B | plateaus ; break step 2836c; slow range 2836c |
| dangerously_in_love | 2 | 2.06 | 3.64 | 1.58 | 16 | 80 | -11 | 1918 | 1965 | 1948 | 1.30 | 0.85 | BD | 2 islands over 1.30s max gap, island medians -5;-1931; plateaus -5;-1902; break step 1926c; slow range 1930c |
| romeo_santos_propuesta_indecente_a | 1 | 1.36 | 4.28 | 2.92 | 6 | 72 | -18 | 1391 | 1880 | 1890 | 2.84 | 0.84 | D | 2 islands over 2.84s max gap, island medians -3;-1862 |
| you_make_me_feel_like_a_natural_wo | 143 | 85.30 | 86.22 | 0.92 | 25 | 81 | 11 | 30 | 1693 | 1767 | 0.36 | 0.80 | BD | 3 islands over 0.36s max gap, island medians 23;4;-1658; plateaus 12; break step 1662c; slow range 1673c |
| frank_ocean_pink_white | 146 | 94.74 | 95.34 | 0.60 | 14 | 47 | 30 | 1319 | 1769 | 1767 | 0.24 | 0.76 | B | plateaus 1;1719; break step 1690c; slow range 1735c |
| frank_ocean_pink_white | 81 | 58.80 | 59.80 | 1.00 | 17 | 69 | -0 | 10 | 1576 | 1743 | 0.64 | 0.73 | BD | 3 islands over 0.64s max gap, island medians -0;4;-1519; plateaus 3;-1512; break step 1523c; slow range 1515c |
| don_t | 338 | 196.12 | 198.02 | 1.90 | 53 | 52 | -3 | 11 | 22 | 1728 | 0.88 | 0.90 | D | 2 islands over 0.88s max gap, island medians -3;1737 |
| at_last | 2 | 9.28 | 13.54 | 4.26 | 26 | 67 | 16 | 15 | 1274 | 1719 | 3.74 | 0.82 | D | 3 islands over 3.74s max gap, island medians 16;25;-1667 |
| frank_ocean_pink_white | 190 | 122.10 | 122.82 | 0.72 | 20 | 71 | -1 | 9 | 1676 | 1700 | 0.36 | 0.78 | BD | 2 islands over 0.36s max gap, island medians -0;-1666; plateaus -0; break step 1666c; slow range 1574c |
| miley_cyrus_party_in_the_u_s_a_aud | 162 | 101.18 | 101.72 | 0.54 | 11 | 82 | -6 | 5 | 870 | 1696 | 0.36 | 0.77 | D | 2 islands over 0.36s max gap, island medians -6;-1702 |
| justin_bieber_baby_ft_ludacris | 330 | 144.30 | 145.00 | 0.70 | 17 | 47 | 67 | 260 | 720 | 1669 | 0.40 | 0.87 | D | 2 islands over 0.40s max gap, island medians 48;1249 |
| miley_cyrus_party_in_the_u_s_a_aud | 53 | 41.18 | 41.72 | 0.54 | 9 | 82 | -4 | 6 | 1003 | 1659 | 0.40 | 0.81 | D | 2 islands over 0.40s max gap, island medians -4;-1651 |

#### 20 largest time-gap notes

| song_id | idx | start | end | dur | frames | MIDI | med ¢ | IQR | P5–P95 | max jump | max gap s | med conf | class | reason |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| no_ordinary_love | 471 | 404.62 | 417.96 | 13.34 | 15 | 64 | -264 | 767 | 946 | 705 | 10.38 | 0.68 | BD | 3 islands over 10.38s max gap, island medians -727;128;-307; plateaus -725;186; break step 855c; slow range 855c |
| the_weeknd_starboy_audio_ft_daft_p | 2 | 5.10 | 16.74 | 11.64 | 20 | 57 | 511 | 1602 | 1826 | 1599 | 10.08 | 0.80 | BcD | 4 islands over 10.08s max gap, island medians 1036;1035;-650;447; plateaus 1035;-582;424; break step 1685c; slow range 1620c; octave-sized step/excurs |
| giveon_heartbreak_anniversary_audi | 1 | 3.32 | 13.12 | 9.80 | 8 | 48 | 46 | 1109 | 1168 | 1173 | 9.68 | 0.67 | BcD | 2 islands over 9.68s max gap, island medians 22;1118; plateaus 34;1118; break step 1096c; slow range 1093c; octave-sized step/excursion |
| wild_thoughts | 88 | 44.44 | 54.32 | 9.88 | 37 | 58 | -2 | 351 | 1371 | 879 | 8.94 | 0.88 | D | 3 islands over 8.94s max gap, island medians -7;188;2 |
| chris_stapleton_tennessee_whiskey_ | 319 | 176.28 | 191.48 | 15.20 | 35 | 63 | 16 | 595 | 1100 | 601 | 8.74 | 0.69 | BD | 6 islands over 8.74s max gap, island medians 34;27;31;-560;-1008;-390; plateaus 33;73;26;-541;-589; break step 591c; slow range 1052c |
| jessie_j_who_you_are_audio | 348 | 170.12 | 178.34 | 8.22 | 19 | 52 | 12 | 203 | 284 | 198 | 7.88 | 0.73 | BD | 2 islands over 7.88s max gap, island medians 3;226; plateaus 3;211; break step 222c; slow range 245c |
| sam_smith_lay_me_down_lyric_video | 482 | 240.78 | 249.88 | 9.10 | 95 | 64 | -18 | 35 | 113 | 118 | 7.14 | 0.83 | D | 4 islands over 7.14s max gap, island medians -11;10;-54;-24 |
| no_ordinary_love | 450 | 360.98 | 368.66 | 7.68 | 29 | 59 | -19 | 19 | 66 | 586 | 6.76 | 0.75 | D | 4 islands over 6.76s max gap, island medians -28;-11;-13;623 |
| marc_anthony_ahora_quien_cover_aud | 407 | 239.22 | 245.80 | 6.58 | 8 | 65 | 4 | 15 | 917 | 1351 | 6.46 | 0.83 | D | 2 islands over 6.46s max gap, island medians 5;-1360 |
| tum_hi_ho_lyrics_arijit_singhaashi | 235 | 142.86 | 149.64 | 6.78 | 25 | 53 | -5 | 35 | 78 | 105 | 6.28 | 0.74 | D | 3 islands over 6.28s max gap, island medians -33;-10;4 |
| no_ordinary_love | 379 | 295.32 | 301.42 | 6.10 | 10 | 62 | 21 | 306 | 694 | 254 | 5.94 | 0.67 | D | 2 islands over 5.94s max gap, island medians 5;602 |
| prince_royce_corazon_sin_cara_audi | 2 | 7.76 | 15.74 | 7.98 | 23 | 63 | -202 | 537 | 1183 | 882 | 5.84 | 0.78 | BD | 4 islands over 5.84s max gap, island medians -215;62;-37;-976; plateaus -215;81;-37; break step 939c; slow range 993c |
| lately | 183 | 101.60 | 107.92 | 6.32 | 25 | 56 | 14 | 509 | 1218 | 621 | 5.82 | 0.79 | bD | 3 islands over 5.82s max gap, island medians -205;14;139; plateaus -205;484; break step 218c; slow range 739c |
| bruno_mars_when_i_was_your_man_aud | 135 | 75.46 | 82.94 | 7.48 | 88 | 60 | -1 | 35 | 220 | 101 | 5.68 | 0.94 | bD | 4 islands over 5.68s max gap, island medians -4;3;22;242; plateaus -2;242; break step 220c; slow range 265c |
| kacey_musgraves_rainbow_official_a | 1 | 4.72 | 11.18 | 6.46 | 13 | 58 | 2 | 1178 | 1616 | 645 | 5.50 | 0.70 | BD | 3 islands over 5.50s max gap, island medians -705;147;903; plateaus -705; break step 852c; slow range 1434c |
| no_ordinary_love | 409 | 328.62 | 334.12 | 5.50 | 13 | 54 | 9 | 928 | 1028 | 832 | 5.28 | 0.78 | BD | 2 islands over 5.28s max gap, island medians -2;944; plateaus -2;1001; break step 946c; slow range 965c |
| norah_jones_don_t_know_why | 233 | 135.16 | 141.46 | 6.30 | 17 | 65 | 3 | 47 | 116 | 119 | 5.12 | 0.74 | D | 3 islands over 5.12s max gap, island medians 18;-29;23 |
| chris_stapleton_tennessee_whiskey_ | 5 | 17.60 | 22.92 | 5.32 | 15 | 60 | 71 | 435 | 492 | 449 | 4.88 | 0.74 | BD | 3 islands over 4.88s max gap, island medians 393;-56;159; plateaus 393;-72; break step 449c; slow range 436c |
| prince_royce_corazon_sin_cara_audi | 319 | 166.02 | 171.38 | 5.36 | 30 | 59 | -9 | 35 | 752 | 669 | 4.80 | 0.92 | BD | 2 islands over 4.80s max gap, island medians -6;-730; plateaus -6;-713; break step 725c; slow range 728c |
| chris_stapleton_tennessee_whiskey_ | 159 | 93.06 | 98.60 | 5.54 | 18 | 57 | 77 | 141 | 427 | 446 | 4.72 | 0.83 | bD | 3 islands over 4.72s max gap, island medians 85;-28;132; plateaus 93;-62; break step 160c; slow range 223c |

#### 20 longest-duration notes

| song_id | idx | start | end | dur | frames | MIDI | med ¢ | IQR | P5–P95 | max jump | max gap s | med conf | class | reason |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|---|
| chris_stapleton_tennessee_whiskey_ | 319 | 176.28 | 191.48 | 15.20 | 35 | 63 | 16 | 595 | 1100 | 601 | 8.74 | 0.69 | BD | 6 islands over 8.74s max gap, island medians 34;27;31;-560;-1008;-390; plateaus 33;73;26;-541;-589; break step 591c; slow range 1052c |
| jeff_buckley_hallelujah | 646 | 371.62 | 386.78 | 15.16 | 759 | 64 | -3 | 17 | 49 | 168 | 0.02 | 0.95 | A | single sustained region, continuous contour (vibrato/scoop/drift scale) |
| no_ordinary_love | 471 | 404.62 | 417.96 | 13.34 | 15 | 64 | -264 | 767 | 946 | 705 | 10.38 | 0.68 | BD | 3 islands over 10.38s max gap, island medians -727;128;-307; plateaus -725;186; break step 855c; slow range 855c |
| jeff_buckley_hallelujah | 21 | 42.82 | 54.48 | 11.66 | 76 | 60 | 7 | 33 | 119 | 514 | 3.76 | 0.71 | cD | 12 islands over 3.76s max gap, island medians 13;42;5;17;4;-8;50;-6;-3;15;-84;10; octave-sized step/excursion |
| the_weeknd_starboy_audio_ft_daft_p | 2 | 5.10 | 16.74 | 11.64 | 20 | 57 | 511 | 1602 | 1826 | 1599 | 10.08 | 0.80 | BcD | 4 islands over 10.08s max gap, island medians 1036;1035;-650;447; plateaus 1035;-582;424; break step 1685c; slow range 1620c; octave-sized step/excurs |
| no_ordinary_love | 378 | 282.46 | 294.08 | 11.62 | 31 | 61 | -696 | 789 | 1346 | 789 | 4.26 | 0.72 | BD | 5 islands over 4.26s max gap, island medians -715;-68;599;89;-705; plateaus -715;-71;599;92;-705; break step 795c; slow range 1219c |
| wild_thoughts | 88 | 44.44 | 54.32 | 9.88 | 37 | 58 | -2 | 351 | 1371 | 879 | 8.94 | 0.88 | D | 3 islands over 8.94s max gap, island medians -7;188;2 |
| giveon_heartbreak_anniversary_audi | 1 | 3.32 | 13.12 | 9.80 | 8 | 48 | 46 | 1109 | 1168 | 1173 | 9.68 | 0.67 | BcD | 2 islands over 9.68s max gap, island medians 22;1118; plateaus 34;1118; break step 1096c; slow range 1093c; octave-sized step/excursion |
| sam_smith_lay_me_down_lyric_video | 482 | 240.78 | 249.88 | 9.10 | 95 | 64 | -18 | 35 | 113 | 118 | 7.14 | 0.83 | D | 4 islands over 7.14s max gap, island medians -11;10;-54;-24 |
| no_ordinary_love | 408 | 319.64 | 328.34 | 8.70 | 27 | 61 | -399 | 630 | 992 | 1177 | 3.80 | 0.67 | BD | 5 islands over 3.80s max gap, island medians -381;-730;-40;82;-477; plateaus -381;-712;-40;-407; break step 690c; slow range 746c |
| jessie_j_who_you_are_audio | 348 | 170.12 | 178.34 | 8.22 | 19 | 52 | 12 | 203 | 284 | 198 | 7.88 | 0.73 | BD | 2 islands over 7.88s max gap, island medians 3;226; plateaus 3;211; break step 222c; slow range 245c |
| prince_royce_corazon_sin_cara_audi | 2 | 7.76 | 15.74 | 7.98 | 23 | 63 | -202 | 537 | 1183 | 882 | 5.84 | 0.78 | BD | 4 islands over 5.84s max gap, island medians -215;62;-37;-976; plateaus -215;81;-37; break step 939c; slow range 993c |
| teri_ore_full_audio_singh_is_kinng | 347 | 182.78 | 190.70 | 7.92 | 57 | 62 | 5 | 21 | 94 | 316 | 3.44 | 0.91 | D | 3 islands over 3.44s max gap, island medians 5;-22;-437 |
| no_ordinary_love | 450 | 360.98 | 368.66 | 7.68 | 29 | 59 | -19 | 19 | 66 | 586 | 6.76 | 0.75 | D | 4 islands over 6.76s max gap, island medians -28;-11;-13;623 |
| paramore_the_only_exception_audio | 246 | 168.32 | 175.92 | 7.60 | 49 | 64 | -5 | 45 | 782 | 987 | 4.46 | 0.86 | BD | 5 islands over 4.46s max gap, island medians -0;-4;-2;-357;246; plateaus -2;-300; break step 603c; slow range 748c |
| bruno_mars_when_i_was_your_man_aud | 135 | 75.46 | 82.94 | 7.48 | 88 | 60 | -1 | 35 | 220 | 101 | 5.68 | 0.94 | bD | 4 islands over 5.68s max gap, island medians -4;3;22;242; plateaus -2;242; break step 220c; slow range 265c |
| you_raise_me_up | 22 | 19.68 | 27.08 | 7.40 | 321 | 65 | -6 | 22 | 63 | 60 | 0.26 | 0.85 | d | 10 islands over 0.26s max gap, island medians -2;-5;2;-2;-5;-9;-20;-6;-36;-18 |
| demi_lovato_stone_cold_audio_only | 425 | 181.26 | 188.64 | 7.38 | 281 | 48 | -16 | 25 | 62 | 53 | 1.42 | 0.79 | D | 6 islands over 1.42s max gap, island medians -7;-16;-18;-16;-17;-18 |
| you_raise_me_up | 197 | 187.82 | 194.96 | 7.14 | 271 | 53 | -14 | 42 | 90 | 691 | 1.72 | 0.91 | D | 3 islands over 1.72s max gap, island medians -13;-21;675 |
| marc_anthony_ahora_quien_cover_aud | 260 | 140.80 | 147.92 | 7.12 | 28 | 59 | 15 | 649 | 1905 | 1327 | 3.90 | 0.81 | BcD | 3 islands over 3.90s max gap, island medians -106;-1240;463; plateaus -98;478;399; break step 1702c; slow range 1855c; octave-sized step/excursion |


## 9. How many notes are actually segmentation failures

Exclusive partition of all 22,241 clean notes. Each note is counted once, in the first matching row.

| Group | Notes | % of notes | % of frames |
|---|---:|---:|---:|
| unflagged, no rule hit | 11,679 | 52.51% | 35.8% |
| A: flagged but plausible expressive single note | 5,101 | 22.94% | 26.6% |
| **plausibly legitimate, total** | **16,780** | **75.45%** | **62.4%** |
| clearly invalid (any clear B/C/D) | 2,683 | 12.06% | 19.1% |
| likely invalid, not clear (B/C/D likely only) | 1,858 | 8.35% | 13.6% |
| E: ambiguous | 920 | 4.14% | 5.0% |

Requested headline rates, as a percentage of all notes (multi-label, so they do not sum to 100):

| | likely | clear |
|---|---:|---:|
| legitimate expressive (unflagged + A) | 75.4% | — |
| multi-note merges (B) | 14.0% | 7.2% |
| temporal-gap merges (D) | 10.3% | 7.8% |
| CREPE tracking failures (C) | 1.1% | 0.01% |
| ambiguous (E) | 4.1% | — |

**The IQR > 400 ¢ tail (717 notes, 3.22%) is bad segmentation, not single notes.**
- **No plausible single notes.** 0 are A, since the vibrato bound rules it out; 622 are B-likely, 271
  D-likely, and 66 C-likely.
- **Most are clearly invalid.** 547 (76.3%) meet a clear rule.
- **Breaks dominate.** 576 (80.3%) contain an unvoiced break, and 528 step ≥ 300 ¢ across one.
- **What's left.** Only 87 (12.1%) remain ambiguous. These are mostly contiguous glides or falls
  spanning 5–12 semitones.

Much of the *audio* in these units is real vocal behavior: runs, turns, falls, the next phrase.
The *unit* is wrong, because several sung events have been fused into one "note".

The IQR > 100 ¢ tail (3,801 notes) is mixed:
- 961 (25.3%) are A: falls, scoops, and wide oscillation around one center.
- 1,393 (36.6%) are clearly invalid.
- The rest are likely-invalid or ambiguous.

## 10. Effect on the ML dataset if excluded (counts only; nothing removed)

Each clean note yields 5 examples. Split sizes: train 15,416 / validation 3,242 / test 3,583 notes.

| Scenario | Clean notes | Examples | % notes | train notes / ex | validation notes / ex | test notes / ex | frames |
|---|---:|---:|---:|---|---|---|---:|
| S1 clear B only | 1,604 | 8,020 | 7.21% | 1,136 / 5,680 (7.4%) | 236 / 1,180 (7.3%) | 232 / 1,160 (6.5%) | 47,935 |
| S2 clear B + clear D | 2,681 | 13,405 | 12.05% | 1,924 / 9,620 (12.5%) | 396 / 1,980 (12.2%) | 361 / 1,805 (10.1%) | 76,620 |
| S3 clear B + C + D | 2,683 | 13,415 | 12.06% | 1,926 / 9,630 (12.5%) | 396 / 1,980 (12.2%) | 361 / 1,805 (10.1%) | 76,672 |
| S3b (reference) any likely B/C/D | 4,541 | 22,705 | 20.42% | 3,254 / 16,270 (21.1%) | 665 / 3,325 (20.5%) | 622 / 3,110 (17.4%) | 131,248 |

The frames column is per clean note, out of 402,379 total. S2 and S3 are nearly identical because clear C
is only 2 notes. Exclusion would remove a disproportionate share of *frames* and of *duration*. The S3
notes are 12.1% of notes, but 19.1% of voiced frames and **32.8% of total `duration_s`**.

## 11. Concentration

Failures are **systematic with note length and confidence, and broad across songs**. They are not
concentrated in a few bad songs.

| Duration (s) | <0.1 | 0.1–0.25 | 0.25–0.5 | 0.5–1 | 1–2 | 2–5 | 5–10 | ≥10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| notes | 3,890 | 6,873 | 5,235 | 3,906 | 1,791 | 484 | 56 | 6 |
| % clearly invalid | 0.0 | 0.7 | 9.0 | 24.6 | 45.3 | 68.8 | 91.1 | 83.3 |

| Frames | 3–4 | 5–9 | 10–19 | 20–49 | 50–99 | ≥100 |
|---|---:|---:|---:|---:|---:|---:|
| notes | 1,619 | 7,167 | 6,976 | 5,217 | 1,080 | 182 |
| % clearly invalid | 0.2 | 4.2 | 13.5 | 20.9 | 26.2 | 34.6 |

| Median confidence | <0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–0.95 | ≥0.95 |
|---|---:|---:|---:|---:|---:|
| notes | 714 | 2,878 | 7,416 | 9,577 | 1,656 |
| % clearly invalid | 22.5 | 23.5 | 15.4 | 7.0 | 2.2 |

The duration trend is partly mechanical: a bridged gap *creates* duration. Frame count, which gaps do
not inflate, shows the same monotonic rise. So longer real segments are also more often merged runs.

- **By song.** The clearly-invalid rate ranges from **5.4% to 27.3%** (P10 7.5, P25 10.0, median 12.2,
  P75 13.8, P90 15.5). Every one of the 50 songs is affected. Highest: `michael_jackson_chicago` 27.3%,
  `you_make_me_feel_like_a_natural_woman` 19.5%, `john_legend_ordinary_people` 18.6%,
  `luis_miguel_hasta_que_me_olvides` 16.5%. Lowest: `prince_royce_corazon_sin_cara` 5.4%,
  `giveon_heartbreak_anniversary` 5.7%, `amy_winehouse_love_is_a_losing_game` 6.6%,
  `jeff_buckley_hallelujah` 7.2%. The top songs skew toward melismatic or ad-lib-heavy R&B/soul and
  sparse arrangements with long rests. That is an observation, not a tested claim.
- **By split.** Train 12.5%, validation 12.2%, test 10.1%.
- **By artist.** Not reliably available. Most source filenames are not in "Artist – Title" form (13 of 50
  parse to `unknown`), and almost every artist appears in only one song, so artist-level rates would
  just restate song-level rates. A best-effort parse is in the `artist` column, and per-parse rates are in
  `_summary_stats.json` → `by_artist`.

## 12. Production side effect, outside this audit's scope but measured here

The same merged segments feed production. `compute_metrics` weights both axes by `duration_s`
(`backend/main.py:33-66`), and `score_key_compliance` averages chroma over the note's full `start..end`
span (`backend/harmony.py:225-236`). The S3 notes are 12.1% of notes but **32.8% of the duration
weight**. For D-likely notes it is 10.3% of notes and 32.5% of the weight, against 17.4% of voiced time.
A 15.2 s "note" therefore weighs as much as ~58 median-length (0.26 s) notes, and its harmonic fit is measured against
15 s of changing chords. This is noted for the record; nothing was changed.

---

## 13. Conclusions

### A. Expressive singing or segmentation failure?

**The extreme contours are primarily segmentation failures.** The IQR > 400 ¢ tail contains no
plausible single notes:
- the vibrato bound excludes all of it;
- 76% meets a clear invalidity rule;
- 80% bridges unvoiced breaks;
- 12% remains ambiguous, mostly very wide contiguous glides.

Below that, the picture is mixed. Most flagged notes with IQR ≤ 200 ¢ and a single slow center (5,101)
are legitimate expressive notes: vibrato, scoops, falls, drift. Being flagged by IQR or jump size alone
is not evidence of error. CREPE errors are a minor contributor.

### B. What percentage is clearly invalid as single-note units?

**12.06% (2,683 notes; 13,415 ML examples)** by the clear rules. Another 8.35% are likely invalid, so the
upper bound is about 20.4%.

About half of the clear cases (1,290) have IQR ≤ 100 ¢: a sustained note plus a short, distant fragment.
Their dominant pitch center is intact even though the unit holds two events. The most damaging subset,
clear and IQR > 100 ¢, is 1,393 notes (6.3%).

### C. Dominant failure mechanism

**Unvoiced-gap bridging combined with the exact-semitone commit rule.** 96.7% of clearly invalid notes
(2,595 of 2,683) contain at least one unvoiced break; only 88 are contiguous. Segmentation never examines
time. After any break, a new note is created only if 4 consecutive frames round to the same new semitone.
Short, gliding, or fast post-break fragments change their rounded semitone almost every frame, which
accounts for 62% of all abandoned boundaries and 97.5% in the IQR > 400 ¢ tail. They are therefore
absorbed into whatever note preceded the silence. The same rule, with no gap involved, keeps fast melisma
inside one note.

### D. Must segmentation change before Axis 2 training?

**Yes.** At 12–20% contamination the training units are not what the V1 label assumes, a single pitch
center plus a constant shift:
- clean examples include fused multi-note contours;
- `target_midi` is sometimes a pitch that was never sung;
- the per-note median the baselines use is not a pitch-center estimate for these notes.

The contamination is concentrated in the longest notes, which hold 19% of frames. It is also nearly
uniform across splits, so it would bias train and evaluation alike rather than show up as a held-out
gap.

### E. What should change (not implemented)

In priority order:

1. **Add a temporal boundary.** End the current note when consecutive voiced frames are separated by an
   unvoiced gap. The data supports a two-tier rule:
   - split unconditionally above a long gap. Gaps > 0.5 s affect 1,024 notes, almost never belong to one
     sustained note, and have median in-gap confidence of 0.38.
   - for short gaps, split only when the pitch center across the gap moves (e.g. ≥ ~80–100 ¢). This
     keeps cases like `you_raise_me_up #22`, a sustained note with confidence dropouts, intact.

   Thresholds must be chosen by an experiment, not by this audit.
2. **Stop requiring agreement on one exact semitone to commit.** Trigger a boundary on *sustained
   distance* from the current note: N consecutive frames all > DRIFT_CENTS away, regardless of which
   semitone each rounds to. Alternatively, decide on a short rolling-median contour, so glides and runs
   are split while vibrato, which the rolling median cancels, is not. This must be validated against
   vibrato-heavy notes so they are not over-split.
3. **Secondary.** Re-anchor `current_note` once the note stabilizes rather than at the onset frame, and
   make the commit rule and the stored target refer to the same center.
4. **No CREPE or tracking change is indicated by this audit.** A per-segment confidence check might help
   with the ≥ 1000 ¢ fragments, but that is a separate question.

Any change should be judged by re-running this audit (the B/D rates should collapse while A is
preserved) and by `evaluate.py`, since Axis 1 consumes the same segments.

### F. Rebuild `data/main_data`, or re-segment from saved tracks?

**Re-segmentation from the saved tracks is sufficient. Demucs and CREPE do not need to be rerun.** This
was determined from the schema and code, and verified empirically:

- **Inputs are all saved.** Segmentation's only inputs are `(times, f0 with NaN = unvoiced, tuning
  semitones)`. All three are stored per song in `harmony.npz` as `vocal_frame_times_s`, `vocal_f0_hz`
  and `tuning_offset_semitones`.
- **Reproduction is exact.** Running the current `segment_notes_with_frame_indices` on them reproduces
  all 22,241 saved notes exactly (§0).
- **Per-frame confidence is saved too.** `vocal_crepe_confidence` exists for every frame, voiced or not,
  so confidence-aware segmentation rules are also possible from saved data.
- **One limit: voicing can only get stricter.** The saved `vocal_f0_hz` is already NaN-masked
  (`isnan(vocal_f0_hz) == ~vocal_voiced_mask` exactly), so raw f0 for unvoiced frames is not stored. A
  fix that only *re-segments*, or *tightens* voicing, needs no re-extraction. A fix that *loosened*
  voicing would need CREPE again: a lower confidence threshold, a wider f0 range, or a shorter minimum
  run. Nothing in this audit calls for that.

What would need to be regenerated, from saved data alone:
- **Per song in `data/main_data`:** `notes.npz`, plus `metadata.json` (`number_of_segmented_notes`,
  which `validate_song_output` cross-checks, and `git_commit`), plus `manifest.json`. `harmony.npz` is
  untouched.
- **New code path.** `scripts/build_main_dataset.py` has no re-segment-only mode: `_process_song` always
  calls `separate` and `detect_pitch` (`:407`, `:412`).
- **The ML dataset `data/ml_dataset/axis2_v1`.** Note indices and contours change, so every join on
  `source_note_index` breaks, including this audit, the stable-center audit, and the baseline results.
  Note that the script that built `axis2_v1` is not tracked in the repo.

---

## Integrity

- **Before/after hashes.** SHA-256 of every file under `data/main_data` (191), `data/ml_dataset` (6),
  `data/song_mp3s` (1) and `data/dataset_selections.json` was taken before any analysis and again after
  all outputs were written: 199 files, **identical**.
- **Files touched.** `git status` shows only the new directory `ml/audits/axis2_segmentation_tail/`. No
  tracked file changed, and no file under `backend/` or `scripts/` was modified.
- **No preprocessing rerun.** No Demucs or CREPE was run, and nothing was imported from `separation` or
  `preprocess`. Only `note_segmentation`'s pure functions were imported, and bytecode writing was
  disabled.
- **No labels used.** From `data/ml_dataset/axis2_v1/manifest.csv`, only `song_id`,
  `source_note_index`, `split`, `variant` and `num_frames` of clean rows were read. `injected_shift_cents`
  and `examples.npz` were never read, no corrupted variant was analyzed, and nothing was trained or
  evaluated.
- **Test-split notes included descriptively.** Test-split clean notes were analyzed from the canonical
  store along with the rest, because the per-split counts in §10 require it. This exposes no labels or
  metrics.
- **Purely descriptive.** No note was filtered, re-segmented in place, or modified.
