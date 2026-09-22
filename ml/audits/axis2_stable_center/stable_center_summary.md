# Axis 2 V1 — Do segmented notes have stable pitch centers?

Read-only descriptive audit of the **22,241 clean notes** in `data/ml_dataset/axis2_v1`.
No training, no filtering, no dataset modification. Run date 2026-08-29.

**Headline:** the naive comparison says the middle 50% is dramatically more stable than the
full note (median IQR 31.7 ¢ → 20.7 ¢, a 37.9% reduction, better on 75.8% of notes). **That
result is mostly an artifact.** A control against same-length windows shows the middle-50%
window sits at the **40th percentile** of all equally-long windows from the same contour,
with a median IQR ratio of **1.000**. Fewer frames mechanically means smaller IQR; position
contributes very little. There *is* a real core effect, but it is roughly 10%, not 38%, and
it only appears when trimming is done **by frame index rather than by elapsed time**.

---

## Method, and one decision that changes the answer

`examples.npz` stores `cents_flat` and `confidence_flat` but **no per-frame time array**.
34.2% of clean notes (7,605) contain interleaved unvoiced frames, so consecutive stored
frames are *not* uniformly 20 ms apart — the production voicing gate drops frames inside a
note and `segment_notes` bridges them.

So "normalized time" is genuinely ambiguous, and the two readings disagree materially:

- **by elapsed time** — join canonical `contour_time_s`, normalize to [0,1], take frames in
  [0.25, 0.75)
- **by frame index** — take stored positions `[floor(0.25n) : ceil(0.75n)]`

Regions in §1–§8 use **true elapsed time** (the literal reading of "divide the note by
normalized time"), joining `data/main_data/<song_id>/notes.npz` on
`(song_id, source_note_index)`. All 22,241 clean cents and confidence contours were verified
**byte-identical** to canonical before the join, and all time contours strictly increasing.

§9 reports both, because the choice turns out to decide the recommendation. Production's
`_core_slice` trims by index, and on gap-bridging notes that is **13.8 ¢ better** than
trimming by time.

Region metrics require ≥3 frames; drift fits require ≥4. Notes below those are marked
unavailable, never imputed. Only clean notes are analyzed — corruption is a constant
translation, so every shape statistic here is identical for the four siblings.

## Eligibility

| Analysis | Eligible | % of clean |
|---|---:|---:|
| Full-note metrics | 22,241 | 100.00% |
| Middle 50% (≥3 frames in window) | 16,776 | 75.43% |
| First quarter | 14,733 | 66.24% |
| Last quarter | 13,498 | 60.69% |
| Middle third | 13,172 | 59.22% |
| All three of q1 + mid50 + q4 | 12,166 | 54.70% |
| Middle-50% drift fit (≥4 frames) | 14,231 | 63.99% |
| Full-note drift fit | 22,240 | 100.00% |

**24.57% of clean notes (5,465) cannot support a middle-region analysis at all.** Frame
counts: median 12, P10 = 5, P90 = 38, max 759. Every note with ≤4 frames (1,619) is
ineligible.

---

## 1–3. Region statistics and within-region stability

Distributions in cents, `n | min | P10 | P25 | median | P75 | P90 | P95 | max`:

| Metric | n | min | P10 | P25 | **med** | P75 | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full IQR | 22,241 | 0.79 | 10.66 | 17.87 | **31.66** | 62.89 | 172.39 | 292.98 | 2868.56 |
| mid50 IQR | 16,776 | 0.07 | 4.90 | 9.51 | **20.70** | 49.41 | 110.40 | 170.33 | 1201.65 |
| full MAD | 22,241 | 0.17 | 5.10 | 8.61 | **14.88** | 25.92 | 50.14 | 82.25 | 1418.15 |
| mid50 MAD | 16,776 | 0.00 | 2.14 | 4.49 | **10.07** | 23.21 | 48.19 | 73.22 | 716.28 |
| full std | 22,241 | 0.84 | 13.24 | 21.80 | **40.05** | 83.49 | 163.41 | 243.70 | 1416.00 |
| mid50 std | 16,776 | 0.06 | 4.46 | 8.60 | **18.82** | 43.47 | 85.31 | 124.23 | 1247.40 |
| full P90−P10 | 22,241 | 1.75 | 25.70 | 41.90 | **74.16** | 167.86 | 357.74 | 551.53 | 3514.85 |
| mid50 P90−P10 | 16,776 | 0.11 | 9.21 | 17.85 | **39.21** | 94.65 | 193.32 | 285.37 | 2719.49 |

Ratios mid50/full (tolerance ±2% for "approximately equal"):

| Ratio | median | P25 | P75 | more stable | ≈equal | less stable |
|---|---:|---:|---:|---:|---:|---:|
| IQR | 0.6465 | 0.3735 | 0.9663 | 75.8% | 2.7% | 21.5% |
| MAD | 0.6875 | 0.4293 | 1.0000 | 72.7% | 4.3% | 23.0% |
| std | 0.4873 | 0.2320 | 0.8315 | 84.2% | 2.3% | 13.5% |

Note that **the middle is *less* stable on 21.5% of notes** by IQR — this is not a
one-directional effect.

### Center differences (§2)

| Difference | n | P10 | P25 | **med** | P75 | P90 | P95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| \|full − mid50\| | 16,776 | 0.00 | 0.96 | **4.59** | 13.18 | 32.99 | 66.67 | 1622.16 |
| \|onset − center\| | 13,348 | 1.91 | 5.34 | **13.32** | 31.34 | 93.06 | 172.36 | 1894.52 |
| \|release − center\| | 12,558 | 3.41 | 9.83 | **31.62** | 126.05 | 314.18 | 501.87 | 2089.48 |

The full and middle medians agree closely for most notes (median 4.59 ¢), so switching
statistics changes little on the typical note. **Releases move ~2.4× further from center
than onsets** (31.6 ¢ vs 13.3 ¢) — the note's end is far less trustworthy than its start.

## 4. Concentration around the middle center

| Window | median frac | ≥75% of frames | ≥90% | 100% |
|---|---:|---:|---:|---:|
| ±5 ¢ | 0.333 | 11.5% | 4.7% | 4.3% |
| ±10 ¢ | 0.500 | 27.9% | 15.3% | 13.9% |
| ±15 ¢ | 0.667 | 41.1% | 25.6% | 23.5% |
| ±20 ¢ | 0.750 | **50.9%** | 34.9% | 31.7% |
| ±25 ¢ | 0.800 | 58.1% | 41.8% | 38.3% |
| ±30 ¢ | 0.862 | 63.9% | 47.4% | 43.5% |

Only **half** of eligible notes hold 75% of their middle frames within ±20 ¢ of their own
middle median. At ±10 ¢ it is 27.9%. The median note has just 50% of its middle frames
within ±10 ¢ of its own center.

## 5. Drift

| Region | n | P10 | P25 | **med** | P75 | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mid50 \|drift\| | 14,231 | 3.64 | 9.87 | **26.52** | 74.98 | 199.44 | 315.12 |
| full \|drift\| | 22,240 | 7.75 | 21.58 | **60.01** | 158.96 | 385.03 | 607.77 |

Fraction with middle-region \|drift\| below threshold: **<5 ¢ 13.4% · <10 ¢ 25.3% ·
<15 ¢ 34.6% · <20 ¢ 42.1% · <30 ¢ 53.4%**. (Full note: 6.6 / 12.7 / 18.4 / 23.6 / 32.2%.)

Median middle-region drift of 26.5 ¢ is larger than the median middle IQR of 20.7 ¢ —
**the typical note's center is sloping, not sitting.** Trimming halves drift (60 → 27 ¢) but
does not remove it.

## 6. Onset, center, release

| Signed | n | P10 | P25 | **med** | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|
| onset→center | 13,348 | −44.06 | −14.65 | **−0.47** | 12.24 | 35.84 |
| release−center | 12,558 | −193.85 | −37.58 | **−1.77** | 26.89 | 144.40 |

Both are **symmetric and centered near zero**: onset below center 48.8% / above 51.2%;
release above 47.4% / below 52.6%. There is no systematic scoop-from-below in aggregate —
the population contains scoops up and scoops down in equal measure, so a directional
correction would be unjustified.

The plateau question: \|onset−center\| exceeds the middle IQR on **31.4%** of notes, and
\|release−center\| on **54.5%**. So the middle behaves like a plateau relative to the onset
on about two-thirds of notes, but relative to the release on fewer than half.

## 7. Stratification by length

| frames | n | %elig | full IQR | mid IQR | mid MAD | \|drift\| | frac ≤20 ¢ | ratio | %mid better |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3–4 | 1,619 | 0.0 | 21.95 | — | — | — | — | — | — |
| 5–6 | 3,069 | 15.7 | 25.61 | 12.13 | 4.71 | — | 1.00 | 0.500 | 91.5 |
| 7–10 | 5,067 | 87.3 | 32.86 | 12.73 | 6.12 | 22.80 | 1.00 | 0.417 | 92.0 |
| 11–20 | 6,391 | 92.0 | 40.03 | 22.90 | 10.96 | 30.47 | 0.71 | 0.634 | 76.7 |
| 21–50 | 4,891 | 97.9 | 36.47 | 30.77 | 14.62 | 28.59 | 0.62 | 0.847 | 63.7 |
| 51+ | 1,204 | 99.8 | 29.21 | 27.47 | 13.19 | 16.12 | 0.65 | 0.953 | 54.0 |

| duration | n | %elig | full IQR | mid IQR | mid MAD | \|drift\| | frac ≤20 ¢ |
|---|---:|---:|---:|---:|---:|---:|---:|
| <0.10 s | 3,890 | 11.9 | 23.35 | 12.13 | 4.66 | — | 1.00 |
| 0.10–0.20 s | 5,066 | 86.5 | 29.80 | 12.41 | 6.03 | 21.69 | 1.00 |
| 0.20–0.40 s | 5,486 | 91.7 | 36.98 | 21.27 | 10.08 | 28.61 | 0.75 |
| 0.40–0.80 s | 4,508 | 89.8 | 39.37 | 30.49 | 14.35 | 30.46 | 0.62 |
| ≥0.80 s | 3,291 | 86.7 | 36.50 | 30.94 | 14.90 | 23.33 | 0.62 |

**This is the opposite of the expected pattern.** Longer notes are *less* internally stable
and benefit *less* from trimming (ratio 0.42 at 7–10 frames → 0.95 at 51+; "% mid better"
falls 92% → 54%). Stable-center behavior does **not** emerge with length.

The short-note figures are largely mechanical. At 5–10 frames the middle window holds 3–5
samples, so its IQR is small because the sample is small — "frac within ±20 ¢ = 1.00" for
those buckets is a 3-sample statistic. This is what motivated the control.

## 8. Candidate criteria (exploratory — not used to filter)

Denominator = 16,776 middle-50%-eligible notes.

| Criterion | count | % eligible | % all clean | + drift <20 ¢ | % eligible |
|---|---:|---:|---:|---:|---:|
| **A** ≥75% within ±30 ¢ | 10,724 | 63.9% | 48.2% | 5,230 | 31.2% |
| **B** ≥75% within ±20 ¢ | 8,537 | 50.9% | 38.4% | 4,636 | 27.6% |
| **C** ≥75% within ±10 ¢ | 4,682 | 27.9% | 21.1% | 3,225 | 19.2% |

Adding the drift requirement roughly halves A and B. Under the strict-and-stable reading
(C + drift), **19.2% of eligible notes — 14.5% of all clean notes — have a tight,
non-drifting center.**

## 9. Does the core actually improve stability? — the control

The §3 numbers are confounded: a k-frame window has smaller IQR than an n-frame window for
k<n regardless of position. The control compares the middle-50% window against **every
contiguous window of the same length** in the same contour (16,776 notes):

| Statistic | Value |
|---|---|
| Percentile rank of mid50 IQR among same-length windows | **median 40.0** (P25 20.0, P75 60.0, mean 40.0) |
| mid50 IQR ÷ median same-length window IQR | **median 1.0000** (P25 0.797, P75 1.107) |
| Notes where mid50 beats the median same-length window | **43.6%** |
| Notes where mid50 is the single most stable window | **14.2%** |

By frame bucket the median ratio is **1.000 in every bucket**, and the percentile rank rises
from 33 (5–10 frames) toward 48 (51+) — converging on "no positional advantage at all."

So the honest decomposition of the 37.9% median IQR reduction: **almost all of it is the
length effect; roughly a 10th percentile of positional advantage remains.** The middle is
*slightly* better than an arbitrary window of equal length, and it is meaningfully better
than the onset — but it is not the plateau the naive number suggests.

### What the statistic comparison actually shows

For clean examples the label is exactly `0.0`, so `|error| = |statistic|`. This makes the
audit directly comparable to the existing median-cents baseline without training anything:

| Pitch-center statistic | mean \|stat\| (¢) | median \|stat\| (¢) |
|---|---:|---:|
| full median (current baseline) | 16.899 | 10.242 |
| **production `core_median` (index-trim, 20%, n≥10 gate)** | **15.197** | 10.344 |
| index-trim 20% with no gate | 15.102 | — |
| middle 50% by frame index | 16.465 | 10.602 |
| middle 50% by elapsed time | 21.687 | 11.603 |

Two things matter here:

1. **`core_median` beats `median` by 1.70 ¢ (−10.1%) in mean absolute terms**, and it
   replicates on every split: train −10.05%, validation −11.62%, test −8.28%.
2. **Index-based trimming beats time-based trimming**, decisively on gap-bridging notes:

| Note class | n | full | core | idx-mid50 | time-mid50 |
|---|---:|---:|---:|---:|---:|
| No unvoiced gaps | 10,718 | 15.481 | 14.612 | 15.481 | 16.500 |
| Has unvoiced gaps | 6,058 | 18.120 | **14.514** | 17.097 | **30.864** |

On gapped notes, time-based windowing is **+13.8 ¢ worse** than index-based. Because
`segment_notes` bridges unvoiced frames, a time-normalized [0.25, 0.75) window can land
almost entirely inside a gap and select a handful of frames from the wrong side of it. This
is why production's `_core_slice` trimming by index is the right choice, and it should not
be "corrected" to use elapsed time.

Note also the median is essentially unchanged (10.24 → 10.34). **The gain is entirely in the
tail**: core-trimming rescues badly-behaved notes rather than improving typical ones.

## 10. Examples

`example_notes.csv` (40 notes × 19 metrics) and `example_contours.csv` (1,054 frames with
cents *and* CREPE confidence). Plots: `example_contours.png`, `iqr_and_drift.png`,
`concentration_and_length.png`.

The stable examples are real plateaus — e.g. `christina_aguilera_hurt#432` (13 frames,
mid IQR 0.58 ¢, drift −1.9 ¢) and `teri_ore…#88` (18 frames, mid IQR 0.65 ¢). Several show
the classic shape: flat middle, sharp release drop (`ariana_grande_into_you#137`, release
−29 ¢; `kacey_musgraves_rainbow#298`, release −52.7 ¢).

**The unstable examples are not singing.** Middle-region IQRs of 337–1,202 ¢ and drifts up
to 3,854 ¢ are 3–38 *semitones* — octave errors and note-boundary failures, not intonation.
`jazmine_sullivan…#481` has a full IQR of 1,398 ¢ across 22 frames.
`demi_lovato_stone_cold#407` has a middle median of −1,597 ¢ (16 semitones below target).
These are the monophonic-tracker-through-polyphony failures the dataset README warns about,
and they are what drags the mean while leaving the median untouched.

Scale of the problem: **full IQR >100 ¢ (one semitone) on 17.1% of notes; >200 ¢ on 8.5%;
>400 ¢ on 3.2%.** Separately, 7.9% of notes have a time span >2× their contiguous-frame
span, and 201 notes claim durations ≥3 s (max **15.2 s at only 35 frames** — a "note"
spanning 15 seconds of mostly-unvoiced audio). These are segmentation artifacts worth a
follow-up, independent of the core-median question.

## 11. Split robustness

Every conclusion holds in all three splits — the property is corpus-wide, not one split's
quirk. (Descriptive contour geometry only; no test labels or corruption targets read.)

| Split | songs | n | full IQR | mid IQR | ratio | %mid better | ≥75% w/in ±10 ¢ | ±20 ¢ | ±30 ¢ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 35 | 15,416 | 31.78 | 20.97 | 0.652 | 75.6 | 27.8% | 50.6% | 63.8% |
| validation | 7 | 3,242 | 38.05 | 26.96 | 0.657 | 75.9 | 20.9% | 42.6% | 55.5% |
| test | 8 | 3,583 | 26.42 | 15.79 | 0.620 | 76.7 | 34.7% | 59.5% | 72.0% |

Test is the cleanest set and validation the noisiest — worth remembering when reading any
future validation metric, since validation is harder than test here.

---

# Required conclusions

### A. Do the majority of notes have a stable central pitch region?

**No, not under any strict reading.** Only **50.9%** of middle-50%-eligible notes hold ≥75%
of their middle frames within ±20 ¢ of their own middle median, and that is 38.4% of all
clean notes. At ±10 ¢ it falls to 27.9% of eligible / 21.1% of all. Adding a modest
non-drift requirement (<20 ¢) leaves **27.6%** at ±20 ¢ and **19.2%** at ±10 ¢.

Median middle-region drift (26.5 ¢) *exceeds* median middle-region IQR (20.7 ¢), so the
typical note's center is sloping rather than sitting. A loose ±30 ¢ criterion is satisfied by
63.9% of eligible notes — but ±30 ¢ is wider than the largest corruption band in the
dataset, so calling that "stable" would define stability more loosely than the signal the
model must detect.

### B. Is the middle 50% measurably more stable than the full note?

**Yes on the raw comparison, but the effect is far smaller than it appears, and most of the
headline number is an artifact.**

Raw: median IQR 31.66 → 20.70 ¢ (−37.9%), MAD 15.31 → 10.07 ¢ (−34.2%), std 45.47 → 18.82 ¢
(−58.6%); middle more stable on 75.8% / 72.7% / 84.2% of notes.

Controlled: the middle-50% window ranks at the **40th percentile** of same-length windows
from the same contour, median IQR ratio **1.000**, and is the single most stable window on
only 14.2% of notes. Nearly all of the reduction is the mechanical consequence of measuring
fewer frames.

The residual real effect shows up where it counts: predicting the production `core_median`
instead of the full median lowers mean \|deviation\| on clean notes by **1.70 ¢ (−10.1%)**,
replicated across all three splits. Real, worthwhile, an order of magnitude smaller than
38%.

### C. How strongly does stable-center behavior depend on duration/frame count?

**Strongly — and in the opposite direction from the hypothesis.** Longer notes are *less*
internally stable and benefit *less* from trimming: median middle IQR rises 12.7 ¢ (7–10
frames) → 30.8 ¢ (21–50), the mid/full IQR ratio rises 0.42 → 0.95, and "% middle more
stable" falls 92% → 54%. Fraction of middle frames within ±20 ¢ falls 1.00 → 0.62.

Mechanism: long notes carry vibrato, portamento and genuine melodic movement, and are also
where segmentation most often fuses distinct pitches. Apparent short-note stability is
substantially a 3–5-sample artifact. **A stable-center concept does not become more
meaningful as notes lengthen** — the opposite.

### D. What fraction of notes are too short for meaningful center-region analysis?

**24.57% (5,465 of 22,241)** cannot support a middle-50% region of ≥3 frames. Additional
thresholds: ≤4 frames 7.28% (1,619, none eligible); ≤6 frames 21.08%; no middle-region drift
estimate possible (<4 middle frames) **36.01%**; eligible for all three of
onset/middle/release **54.70%**.

A stricter but defensible bar — the ~21% of notes at ≤6 frames — makes any region statistic
a 3-sample estimate, which the control shows carries no positional information. Production's
own `_core_slice` implicitly agrees: it is a **no-op below 10 voiced frames, which is 39.5%
of clean notes** (`core_median` is bit-identical to `median` for 100% of those, and 54.0% of
all clean notes overall).

### E. Does the evidence support trying a core-median baseline next?

**Yes — with the design fixed by these measurements, and with expectations set low.**

Supporting:
- `core_median` reduces mean \|deviation\| on clean notes by 1.70 ¢ (−10.1%) vs `median`,
  replicating on train (−10.05%), validation (−11.62%), and test (−8.28%).
- The statistic is already computed and stored per example, so the baseline is cheap.
- It attacks the right failure. The median-cents baseline scores identical MAE (19.03 ¢)
  on all five variants because a constant shift cancels; its error *is* the natural offset
  of the reference, and that error is tail-dominated by tracker failures — exactly what
  trimming suppresses.

Constraining — three findings that should shape it:
1. **Trim by frame index, not elapsed time.** Time-based windowing is 13.8 ¢ *worse* on the
   34% of notes bridging unvoiced gaps, and 5.2 ¢ worse overall. Reuse production
   `_core_slice`; do not "improve" it to use `contour_time_s`.
2. **Expect a tail-only gain.** Median \|statistic\| barely moves (10.24 → 10.34 ¢) while
   the mean improves 10%. Report median and a trimmed mean alongside MAE, or the mechanism
   will be invisible.
3. **It cannot fix the structural problem.** Both statistics are contour summaries, and
   `clean` labels are 0.0 while clean contours are deliberately *not* recentered. Any
   contour-summary predictor is therefore penalized by the reference's natural offset and
   will plateau near it. The remaining ~15 ¢ is the floor for this family of models, not
   something a better trim will remove — closing it needs harmonic context, which is what
   Phase B's learned target is for.

Recommended framing: run it as the **third reference point** bounding what contour summaries
can achieve (zero → median → core-median), not as a candidate for the shipping Axis 2 score.

---

## Surprises

1. **The middle-50% advantage is ~90% a sample-size artifact** (percentile rank 40, ratio
   1.000). Any future region comparison in this repo needs a same-length control.
2. **Longer notes are less stable, not more** — inverting the natural assumption behind
   "core-median works better on sustained notes."
3. **Time-based vs index-based windowing differ by 13.8 ¢** on a third of the corpus. An
   invisible methodological choice outweighs the entire effect being measured.
4. **Production's core trim is a no-op on 39.5% of clean notes** (n<10 gate) and identical to
   the full median on 54.0% overall — so "core vs full" is really a question about 46% of the
   corpus.
5. **Releases are 2.4× less trustworthy than onsets** (31.6 ¢ vs 13.3 ¢ from center). The
   existing symmetric 20% trim spends equal budget on unequal problems.
6. **Onset and release deviations are symmetric about zero** (48.8/51.2, 47.4/52.6) — no
   population-level scoop direction, so no directional correction is warranted.
7. **A hard core of segmentation failures:** full IQR >1 semitone on 17.1% of notes, >4
   semitones on 3.2%, plus one 15.2-second "note" with 35 frames and 7.9% of notes spanning
   >2× their contiguous frame span. These distort every mean in this audit and deserve their
   own investigation — they are a note-segmentation issue, not an intonation one.
8. **Validation is harder than test** (mid IQR 26.96 vs 15.79 ¢) — relevant to reading any
   future validation-vs-test comparison.

## Integrity

SHA-256 hashed before and after: all 190 files in `data/main_data/`, all 4 in
`data/ml_dataset/axis2_v1/`, and `data/dataset_selections.json` (195 total).
**All hashes unchanged; `unchanged: true` recorded in `_integrity.json`.** The dataset was
opened read-only throughout; no test labels and no corruption targets were used — clean
labels were read only to assert they are exactly `0.0`. All outputs are confined to this
directory.

## Files

| File | Contents |
|---|---|
| `stable_center_audit.csv` | 22,241 rows × 60 metrics, one per clean note |
| `stable_center_summary.md` | this report |
| `example_notes.csv` | 40 example notes across 4 behavior groups |
| `example_contours.csv` | 1,054 frames — cents + confidence for those examples |
| `stable_center_audit.py` | per-note metric extraction |
| `summarize.py` | aggregation |
| `control_subwindow.py` | same-length-window control (the §9 result) |
| `examples_and_plots.py` | example selection + plots |
| `iqr_and_drift.png`, `concentration_and_length.png`, `example_contours.png` | plots |
| `_summary_stats.json`, `_control_subwindow.json`, `_integrity.json` | machine-readable stats |
