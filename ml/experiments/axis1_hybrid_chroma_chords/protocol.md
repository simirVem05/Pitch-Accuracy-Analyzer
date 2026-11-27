# Axis 1 Experiment 3 — chroma + confidence-weighted chord hybrid: protocol

Written 2026-11-27, **before** any code in this directory was run.

**Scope:** exploratory method development on all 63 songs (31,803 v2 notes). Results on this corpus are not independent proof of anything, and no human labels exist, so perceptual validity is not measured.

**Inputs:** the stored `other_chroma_cens`, frame times, beats, and v2 notes. Everything is read-only.

**Reused from experiment 2 (not modified):**
- `chord_model.py`: the 112-set vocabulary, binary templates, Pearson similarity, beat segmentation, and the tier table T1 = 1, T2 = 2/3, T3 = 1/3, T4 = 0
- `synthetic_audio.py`: rendering through the exact production chroma call

## The question

For each note, compute `H[0..11] ∈ [0,1]`: the harmonic compatibility of each pitch class with the accompaniment at that moment. Several values can be high at once, and H is not a probability of correctness. **Original-note ranking is a secondary diagnostic only.**

## Temporal unit (all methods)

The unit is a **beat segment**: the mean CENS between consecutive saved beats, as selected in experiment 2. There is no HMM.

Each method produces a per-segment 12-vector. A note's H is the **overlap-duration-weighted mean** of its segments' vectors. Segments where a method abstains are excluded from both the numerator and the denominator; if all of a note's segments abstain, the note abstains. Notes shorter than 0.1 s are flagged `SHORT` and reported separately, never silently dropped.

## Evidence quantities (per segment)

| Symbol | Meaning |
|---|---|
| `r_k` | Pearson correlation of the segment chroma with chord template k — **absolute fit** |
| `r_null` | for P = 200 fixed seeded permutations π of the 12 pitch classes, `max_k r(π(v), T_k)`: the best fit any template achieves on the same values with harmonic structure destroyed |
| `E_k` | `mean_π[r_null(π) < r_k]` — **null-calibrated evidence**. Under "no harmonic structure" the pitch classes are exchangeable, so P(E_best ≥ 0.95) ≈ 5% in any domain, synthetic or real. This is the property experiment 2's fixed θ lacked |
| `m`, `q_ε` | the median and ε-quantile of `r_null` |
| `x_k` | `r_k − m` — **excess fit** over the chance-level best fit; the confidence weight |
| `G_k` | `r_best − r_k` — **relative** gap |
| `I_c` | `1 − H(p)/log 12` for the L1-normalized segment chroma — **chroma informativeness** |
| posterior | experiment 2's `softmax(r/0.02)` with N at 0.8, renormalized without N — kept only for Method B and diagnostics. It is not a probability of correctness |

`κ` (the chroma-informativeness floor) is the 99th percentile of `I_c` over synthetic white- and pink-noise frames. It is fixed on synthetic data only.

## Methods

The tier vector of chord k is `tier_k`. Every method below outputs a 12-vector per segment.

| ID | Definition | Abstains when |
|---|---|---|
| **A** chroma only | `C = v / max(v)`, the raw segment CENS ratio. `A_rank` (rank/11) is reported as a variant | `I_c < κ` |
| **B** chords only | `F_B = Σ_k p̂_k · tier_k`, using experiment 2's posterior over all 112 chords, no key fallback. Flag `LOW_EVIDENCE` if p(N) > 0.5 | never (its evidence gap is measured, not hidden) |
| **C5u** top-5 hybrid | `α·C + (1−α)·mean(tier_k over the top 5 r_k)` — equal weights | never |
| **C5w** (diagnostic) | as C5u, but the top-5 weighted by the renormalized posterior | never |
| **D** confidence hybrid (main hypothesis) | `α·C + (1−α)·F`, with `F = Σ_{k∈S} w_k·tier_k` and `w_k = x_k / Σ x`. S is the set of eligible chords (`E_k ≥ ε` and `G_k ≤ δ`) | S empty and `I_c < κ` |

If S is empty but `I_c ≥ κ`, D uses `H = C` and flags `CHROMA_ONLY`.

**D ablations** (α = 0.5):

| ID | Change |
|---|---|
| D1 | weighting, no threshold: `w ∝ max(x_k, 0)` over all 112 chords |
| D2 | threshold, uniform weights over S |
| D3 | threshold + `x`-weights (= D) |
| D4 | D3 with confidence-aware `α_eff = 0.25 + 0.5·(1 − s)`, `s = clip((r_best − q_ε)/(1 − q_ε), 0, 1)`. Strong, credible chord evidence leans on chords (α → 0.25); evidence just past threshold leans on chroma (α → 0.75) |

**α** ∈ {0.25, 0.50, 0.75} is shared by C5u, C5w and D. The headline tables use the **pre-registered α = 0.5**, and α is never tuned to original-note ranking.

**Context (separate output):** experiment 2's PASSING and NEIGHBOR rules only, applied to the recommended method.

## Revision 1 (2026-11-27, before any experiment was run; only the evidence measure had been sanity-checked)

A check of `E_k` on idealized prototypes showed that the permutation-significance measure has **low power for clean chords**. A perfect C-major chroma gets E_best = 0.74, because rearranging three strong pitch classes often produces another valid triad, so a real chord is not surprising relative to its own rearrangements. It would therefore reject clean chords at ε = 0.95.

The primary evidence measure is therefore the **excess fit `c_k = x_k = r_k − m`**: how much better chord k explains this chroma than the best chord explains the same values scrambled. It is computed from the segment's own values, so it adapts to the overall correlation level of the domain. On prototypes it is 0.17–0.24 for clean chords, ≈ 0 for near-flat noise, and negative for clusters.

Changes:
- **New strategy families:** EXCESS (`c_k ≥ γ`, γ ∈ {0.05, 0.10, 0.15}) and EXCESS+REL (`c_k ≥ γ` and `G_k ≤ δ`, δ ∈ {0.03, 0.05, 0.10}). NULL and NULL+REL stay in the table as reported strategies.
- **D's eligible set S** uses EXCESS+REL, chosen by the same pre-registered rule, now over the EXCESS and EXCESS+REL grid.
- **D's weights** are unchanged (`w ∝ c_k`).
- **D4's strength term** becomes `s = clip((c_best − γ)/(1 − m − γ), 0, 1)`.

Everything else is as originally written.

**Selection fallback (also fixed before any run):** if no EXCESS or EXCESS+REL candidate satisfies both constraints (noise false acceptance ≤ 0.05 and mean |S| on clean chords ≤ 3), choose the candidate with the lowest noise false acceptance. Ties go to higher degraded recall.

## Threshold strategies (all evaluated, on synthetic audio and real songs)

| Strategy | Rule | Grid |
|---|---|---|
| ABS | `r_k ≥ θ` and `G_k ≤ 0.05` | θ ∈ {0.70, 0.80, 0.90} |
| REL | `G_k ≤ δ` only (never rejects) | δ ∈ {0.03, 0.05, 0.10} |
| NULL | `E_k ≥ ε` | ε ∈ {0.90, 0.95, 0.99} |
| NULL+REL | `E_k ≥ ε` and `G_k ≤ δ` | ε ∈ {0.90, 0.95, 0.99} × δ ∈ {0.03, 0.05, 0.10} |
| EXCESS (rev. 1) | `c_k ≥ γ` | γ ∈ {0.05, 0.10, 0.15} |
| EXCESS+REL (rev. 1) | `c_k ≥ γ` and `G_k ≤ δ` | γ × δ ∈ {0.03, 0.05, 0.10} |

**Synthetic evaluation sets** (from experiment 2, regenerated with the same seeds):
- **B clean** and **B degraded** chord frames, with exact labels
- **noise** (white, pink): no chord expected
- **6-note clusters**: no chord expected
- **single sustained pitch**: identity undetermined
- **two-chord mixtures**, intentionally ambiguous; the acceptable set is both chords plus any template equal to their union

**Metrics:**
- true-chord recall (the true set ∈ S)
- mean |S|
- false acceptance (|S| > 0) on noise and on clusters

**Pre-registered choice of D's (ε, δ):** among NULL and NULL+REL, maximize recall on degraded synthetic chords, subject to noise false acceptance ≤ 0.05 and mean |S| on clean chords ≤ 3. Ties go to the smaller mean |S|. ABS and REL are reported only, to show transfer and rejection behavior.

## Real-song evaluation (all methods; nothing is tuned here)

| Area | Measures |
|---|---|
| A. multiple compatibility | pitch classes per note with H ≥ 0.75; with H ≥ 0.5 |
| F. specificity | share of notes with ≥ 10 pitch classes ≥ 0.5; max − min |
| B. grading | share of values in (0.25, 0.75); mean H by tier of each pitch class under a *reference* chord (D's top eligible chord). This is circular for chord methods, informative for A |
| C. stability | mean and p95 \|ΔH\| under multiplicative CENS noise (lognormal σ = 0.1, re-normalized), ±50 ms note shifts and a +50 ms beat-grid shift; also Jaccard of D's eligible sets under noise |
| D. chord selection | distribution of \|S\| per beat; share of beats with S empty; CHROMA_ONLY and INSUFFICIENT rates |
| E. confidence quality (label-free) | acceptance rates per strategy on synthetic degraded vs real (transfer); diatonic share of accepted vs rejected top chords; `I_c` of accepted vs rejected segments; experiment 2 posterior top-1 mass on segments D rejects (overconfidence) |
| G. examples | the largest method disagreements; representative chord-tone, extension, short-note, chord-transition and ambiguous cases |
| H. cost | per-song seconds for each method, excluding Demucs and CREPE |
| Secondary | sung vs alternative mean H, win rate, P4/P5 gap |

All methods are compared on the same eligible notes. Coverage (non-abstained share) is reported, as are metrics on the common subset where every method scores.

## Synthetic musical tests (expectations fixed now)

There are two layers: **T** (idealized harmonic-partial chroma) and **S** (synthetic audio → production CENS). Each check is a musical requirement, applied to **every method** and scored pass/fail per method. Truth types: *exact* (known chord) or *ambiguous* (intentionally undetermined).

| # | case (truth) | requirement |
|---|---|---|
| 1 | C major (exact) | C, E, G ≥ 0.75; D, A ≥ 1/3 (extensions meaningful); C#, F#, G# ≤ 0.25; min(C, E, G) > max(D, A, B) |
| 2 | A minor (exact) | A, C, E ≥ 0.75; A#, C# ≤ 0.25 |
| 3 | G7 (exact) | G, B, D, F ≥ 0.75 |
| 4 | Csus4 (exact) | C, F, G ≥ 0.75; E ≥ 0.25 |
| 5 | C5 power (exact; thirds undetermined) | C, G ≥ 0.75; E ≥ 1/3 and Eb ≥ 1/3 |
| 6 | C + Am sounding together (ambiguous) | C, E, G, A ≥ 0.6; C#, F# ≤ 0.4 |
| 7 | C and E only, 3+ completions plausible (ambiguous) | C, E ≥ 0.75; G ≥ 1/3 and A ≥ 1/3 |
| 8 | C major at 5 dB SNR: weak wrong templates fit (exact) | C, E, G ≥ 0.6; C#, F# ≤ 0.4 |
| 9 | 6-note cluster, no credible chord (ambiguous) | either abstain, or no pitch class > 0.75 without chroma energy (method must not invent a confident chord) — operationalized: abstain OR max over the 6 absent pitch classes ≤ 0.5 |
| 10 | white noise (no harmony) | abstain/insufficient (methods that cannot abstain fail) |
| 11 | single sustained C (identity undetermined) | C ≥ 0.75; \|H(E) − H(Eb)\| ≤ 0.2 (no confident major/minor call) |
| 12 | C → F → G → C, 2 beats each, 100 BPM (exact) | a note centred in each chord: that chord's tones ≥ 0.75 |
| 13 | C major with the third quiet (E at 0.15 amplitude) (exact) | E ≥ 0.6 (compatible despite little energy) |
| 14 | C major + loud non-chord F# (exact chord; F# is foreign) | F# < min(C, E, G) and F# ≤ 0.6 |
| 15 | several compatible pitch classes (over cases 1–4) | ≥ 3 pitch classes ≥ 0.75 **and** ≤ 8 pitch classes ≥ 0.5 in each |
| 16 | top-5 permissiveness (cases 1–4) | the number of pitch classes ≥ 0.5 is reported for every method. Expectation: **C5u > D**, and C5u gives some T4 pitch class of the true chord ≥ 0.5 in at least one case. This is a demonstration, not pass/fail per method |
| 17 | T4 suppression (cases 1–4) | every T4 pitch class of the true chord ≤ 0.25 |
| 18 | stability (case 1) | \|ΔH\| ≤ 0.05 for a ±50 ms note shift and ≤ 0.10 under CENS noise σ = 0.1 |

## Decision procedure (fixed now)

The recommendation weighs, in order:
1. synthetic musical checks passed (both layers)
2. specificity: ≥ 10-pc share ≤ 1% and mean pcs ≥ 0.5 ≤ 8
3. stability no worse than production's experiment-2 figures (noise 0.0145, shift 0.0139)
4. justified abstention (it rejects noise; real coverage reported)
5. interpretability

Original-note ranking can't decide it. Where criteria conflict, the trade-off is stated rather than hidden.
