# Axis 1 chroma evaluation — protocol

Written 2026-11-26, after structural data checks and **before any harmonic-fit score was computed**. The song split (`axis1_splits.json`) was created before this file. Anything decided after seeing results is labelled as such in `summary.md`.

## Question

How well do the saved `other`-stem CENS chromagrams separate the pitch class actually sung from the 11 alternatives, and which deterministic scoring rule does that best? Every number here is a **label-free proxy**. Released vocals are assumed to be mostly, not entirely, harmonically fitting, and an alternative pitch class is a counterfactual, not a known-wrong note.

## Data

- `data/main_data/<song>/notes.npz` (v2 notes, schema 2) and `harmony.npz`, for all 63 songs. These files are read-only; nothing is recomputed from audio, since no audio exists.
- Inputs used: `other_chroma_cens`, `other_chroma_rank`, `chroma_frame_times_s`, `global_chroma_profile`, `global_chroma_rank` and `estimated_tempo_bpm` (for the slack); note start/end, `pitch_class` and duration; the per-frame CREPE confidence and cents contour (diagnostics only); and the vocal voiced mask (for the bleed proxy only).
- **Not available, so not measured:** chroma of any other stem mix (`other+bass`, bass alone), the percussive stem, stem RMS (needed for the a cappella guard), alternative chroma types (CQT, HPSS), and the audio itself. The production bleed test cannot be re-run.

## Groups (`axis1_splits.json`)

| Group | Songs | Use |
|---|---|---|
| dev | 40 of the 50 `axis2_ml` songs | exploration and method selection |
| holdout | 10 of the 50, chosen by a seeded hash | reported once, after selection |
| robustness | the 13 `axis2_ml=false` songs | reported separately; not used for selection |

Three songs with prior contact were forced into dev: `into_you`, `don_t` and `wild_thoughts`. The first two share titles with the bundled samples that set the current constants; the audio files differ. **Neither holdout nor robustness is a fully blind set.** The historical 20-song check drew from the same corpus and its membership is unrecorded. It ran after the constants were last changed (`harmony.py` unchanged since 2026-08-12), so it did not tune them, but the songs may have been seen.

## Note eligibility

- **Primary set:** every structurally valid note. That means finite times with end > start, a window of at least 1 chroma frame inside the chromagram, no all-zero chroma in the window, and `pitch_class == target_midi % 12`. Confidence is not filtered.
- **High-confidence subset (secondary):** mean contour CREPE confidence ≥ 0.85, duration ≥ 0.10 s, within-note cents SD < 35, and |core median cents| ≤ 40. Coverage is reported.

## Methods (all fixed before scoring)

Windows follow production exactly. `i0 = searchsorted(t, start, 'left')`, `i1 = searchsorted(t, end, 'right')`, `i1 ≥ i0+1`. "Slack" means the max over the note span and the same-width span shifted ±1 beat (`round((60/tempo)/hop)` frames), with shifted spans used only when fully in bounds. For each note, every method produces a score for each of the 12 candidate pitch classes.

| ID | Method |
|---|---|
| `raw_local` | mean raw CENS of the candidate pitch class over the note span |
| `raw_local_slack` | the same, with slack |
| `rank_local` | mean per-frame rank (stored `other_chroma_rank`) over the note span |
| `rank_local_slack` | the same, with slack (= production local term) |
| `rank_avgties_local_slack` | as above, but rank recomputed with **average** tie handling instead of argsort order |
| `global` | whole-song `global_chroma_rank` of the candidate |
| `global_raw` | whole-song `global_chroma_profile` value (differs from `global` only in song-level means) |
| `prod_geo_w25` | `rank_local_slack^0.75 · global^0.25` — **production `harmonic_rank`**, checked against `backend/harmony.py` |
| `geo_w25_noslack` | `rank_local^0.75 · global^0.25` |
| `geo_w25_avgties` | `rank_avgties_local_slack^0.75 · global^0.25` |
| `arith_w25` | `0.75 · rank_local_slack + 0.25 · global` |
| `geo_w10`, `geo_w50`, `arith_w10`, `arith_w50` | sensitivity at w = 0.10 and 0.50 (slack, stored rank) |
| `chord_local_slack` | *exploratory*: max over the 24 major/minor triads containing the candidate of the cosine between the binary triad template and the window-mean CENS, with slack |
| `chord_geo_w25` | *exploratory*: `chord_local_slack^0.75 · global^0.25` |

No other weights, windows or templates are tried. `RANK_LOW`/`RANK_HIGH` do not affect any ranking, because they are a monotone map. They do clip scores at 0 and 1, so the production calibrated score is reported separately for saturation.

## Experiments

- **B — per-note alternatives.** For each note and method, compare the original pitch class with the 11 alternatives, holding the window fixed. Ties within 1e-12 count half.
  - Pairwise win rate = (#alternatives beaten + ½ #ties)/11.
  - Mid-rank placement, top-1 (strict) and top-3.
  - Margin = original − best alternative.
  - Win rate against each signed interval +1 … +11 (+5 = up a fourth, +7 = up a fifth).
- **C — perturbations.** In each song, 20% of notes are relabelled, with 5 seeded draws per pattern (seed `sha256("axis1:{song}:{pattern}:{draw}")`). Patterns: semitone (±1), whole tone (±2), fourth/fifth (+5 or +7), tritone (+6) and random nonzero (1–11). Timing and chroma are unchanged. Reported:
  - the drop in the song mean (unweighted and duration-weighted)
  - the fraction of draws that lower it
  - the per-affected-note score change
  - the drop in the production calibrated headline, in percentage points
- **D — song-level transposition.** Mean score of all notes at each of the 12 global shifts (unweighted, as in `evaluate.py`; duration-weighted as a secondary). Reported: the original's placement among the 12 (ties count half), margin over the best shift, the best competing shift, and the P4/P5 share.

## Pre-registered selection rule

- **Primary metric:** the mean over **dev songs** of the song-mean pairwise win rate, on the primary note set.
- **Uncertainty:** each method's paired difference from `prod_geo_w25`, with a song-cluster bootstrap CI (10,000 resamples, seed 20261126).
- **Choice:** the method with the highest dev primary metric is the selected candidate. The exploratory chord methods are eligible but flagged.
- **Holdout:** the selected method and `prod_geo_w25` are then reported on holdout. Robustness and all-63 results are reported for every method, but **no choice is made from them**.

Reporting a method as better than production on holdout requires the paired holdout difference to be positive on at least 7 of 10 songs. Otherwise the result is "not distinguishable on 10 songs".

## Diagnostics (proxies; none is ground truth)

- **Buckets:** duration (<0.1, 0.1–0.2, 0.2–0.4, 0.4–0.8, ≥0.8 s); mean CREPE confidence (<0.75, 0.75–0.85, 0.85–0.95, ≥0.95); within-note cents SD (<15, 15–35, ≥35); |core median cents| (≤40 vs >40, pitch-class-ambiguous); and chroma clarity tertile.
- **Chroma clarity:** 1 − normalized entropy of the L1-normalized window-mean CENS. Tertile cut points are fixed from the dev notes.
- **Tie exposure:** the fraction of window frames in which the original pitch class is tied with another.
- **Local × global 2×2 (production terms):**
  - locally supported = original placement ≤ 3 on `rank_local_slack`
  - in key = global placement ≤ 7
  - "local-weak, in-key" is the passing-tone/non-chord-tone candidate cell
  - "both weak" is the chromatic, out-of-key or extraction-error candidate cell
- **Bleed proxy:** the original pitch class's mean rank during the note, minus its mean rank over unvoiced frames within ±1 s outside any note, compared with the same quantity averaged over the 11 alternatives. A positive excess is *consistent with* vocal bleed, but sustained harmony that follows the melody produces the same sign, so it is not a measurement of bleed.
- **Song level:** per-song metrics against vocal coverage, note count and Axis 2 exclusion category.
