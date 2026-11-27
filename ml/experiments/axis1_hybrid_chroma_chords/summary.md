# Axis 1 Experiment 3 — chroma vs chords vs top-5 hybrid vs confidence-weighted hybrid: summary

**Question.** How compatible is each of the 12 pitch classes with the accompaniment at the moment a note is sung? The output is `H[0..11] ∈ [0,1]`, and several entries can be high at once. Original-note ranking is a secondary diagnostic only.

**Scope.** Exploratory, on all 63 songs (31,803 v2 notes), with beat-synchronous `other`-stem CENS and no HMM.
- Synthetic results show rule behavior on rendered audio.
- Real-song results are label-free proxies.
- **Nothing here measures perceptual accuracy.**

**Provenance.**
- The feature store was read-only, and SHA-256 hashes of all 126 source files are unchanged.
- Experiment 2's modules were reused unmodified; their hashes are recorded in `results.json`.

**Process notes.**
- **Revision 1 (pre-run).** A sanity check of the confidence measure on idealized chroma, before any experiment, showed that the planned permutation-significance measure rejects clean chords. A perfect C major scores only E = 0.74, because rearranged triads are often other triads. The protocol was revised to use excess fit (below) before anything was run, and the reason is recorded in `protocol.md`.
- **Post hoc.** The γ-sensitivity and chord-only analyses (`posthoc_checks.py` / `.json`) were added after seeing results. They are labelled as such below.

## What "confidence" is here

Per beat segment:
- `r_k` is the Pearson fit of chord template k: **absolute fit**.
- `m` is the best fit any template achieves on the *same chroma values scrambled across pitch classes* (median over 200 fixed permutations).
- **Excess fit `c_k = r_k − m`** is how much better chord k explains this chroma than chance-level explanation of the same values. Because `m` comes from the segment itself, the measure is self-normalizing.
- `I_c` (1 − normalized entropy) is **chroma informativeness**. Its floor κ = 0.0139 is the 99th percentile on synthetic noise.

Median values by synthetic condition (`confidence_analysis.md`):

| condition | r_best | excess c_best | old posterior top-1 |
|---|---|---|---|
| clean chords | 0.971 | 0.190 | 0.998 |
| degraded chords | 0.917 | 0.142 | 0.995 |
| single pitch | 0.910 | 0.212 | 1.000 |
| white noise | 0.723 | −0.010 | **0.919** |
| pink noise | 0.670 | −0.077 | **0.955** |
| 6-note clusters | 0.707 | 0.000 | 0.494 |

Excess fit separates chords from noise. Raw r and the experiment-2 posterior do not: the posterior is about 92–96% "confident" on noise. The posterior weights are uncalibrated and are not probabilities of correctness.

## Threshold strategies (synthetic labels; real acceptance is descriptive)

| strategy | clean recall | degraded recall | noise false-accept | real beats accepted |
|---|---|---|---|---|
| ABS r ≥ 0.8 | 0.99 | 0.91 | 0.18 | 0.69 |
| ABS r ≥ 0.9 | 0.98 | 0.60 | 0.03 | 0.32 |
| REL only (δ = 0.05) | 0.99 | 0.92 | **1.00** (never rejects) | 1.00 |
| NULL significance ε = 0.90 | 0.29 | 0.28 | 0.01 | 0.11 |
| EXCESS γ = 0.05 | 1.00 | 0.92 | 0.19 | 0.68 |
| EXCESS γ = 0.10 | 0.99 | 0.80 | 0.09 | 0.50 |
| **EXCESS γ = 0.15 (pre-registered choice)** | 0.85 | 0.37 | 0.025 | **0.27** |

- **The rule chose γ = 0.15**, the only candidate meeting noise false-acceptance ≤ 5%.
- **The relative constraint δ mattered little at the chosen γ** (degraded recall 0.358–0.371 across δ at γ = 0.15), but more at γ = 0.05 (0.843–0.912). Few chords pass a strict excess-fit gate, so the relative gap rarely binds.
- **Real stems still sit lower than synthetic.** Real beat segments have a median excess fit of 0.10, against 0.14 for degraded synthetic chords. So γ = 0.15 accepts only 27% of real beats: **D's evidence gate does not fully transfer either**. Absolute levels shift, but by less than raw r.

**Label-free plausibility on real beats** (accepted vs rejected):

| | accepted | rejected |
|---|---|---|
| top chord diatonic | 89.1% | 85.2% |
| `I_c` median | 0.248 | 0.151 |
| r_best median | 0.938 | 0.818 |

The experiment-2 posterior still claims ≥ 0.9 confidence on 52% of the *rejected* segments, which confirms its overconfidence.

**Selected-chord counts per real beat** (γ = 0.15): 0 chords 73.2%, 1 chord 24.0%, 2 chords 2.7%, 3 or more 0.1%, so about 1.1 chords when any qualify. Under CENS noise σ = 0.1 the eligible set is identical in 96.8% of beats (mean Jaccard 0.971).

## Synthetic musical tests (`synthetic_tests.md`; 17 requirements per layer, fixed before running)

| method | T | S | fails (both layers unless noted) |
|---|---|---|---|
| A chroma only | 10 | 11 | 1 (T), 4, 5, 7, 13, 14, 17 (T): silent-but-compatible tones and loud foreign tones |
| A_rank | 9 | 10 | |
| **B chords only (posterior)** | **14** | **14** | 6\*, 9 (cluster → names a chord), 10 (noise → confident score) |
| C5u top-5 uniform, α = 0.5 | 11 | 12 | 3 (T), 5, 7, 10, 13, 17 |
| C5w top-5 posterior-weighted, α = 0.5 | 11 | 12 | 1 (T), 4, 5, 7, 10, 13 |
| **D confidence hybrid, α = 0.5 (main hypothesis)** | **13** | **13** | 4, 7, 13, 14 |
| D1 weighting, no threshold | 12 | 11 | |
| D2 threshold, uniform weights | 13 | 13 | |
| D4 adaptive α | 13 | 13 | |

\* Case 6 (C + Am sounding = Am7) required F# ≤ 0.4, but F# is the dorian 13th of Am7, which the tier table marks as an extension (0.67). The expectation contradicted the pre-registered tiers; it is a test-design inconsistency, not a scoring error.

**What the cases show:**
- **The chroma term is the main source of failures.**
  - **Cases 4, 7 and 13.** The third over sus4, the completions of a bare C–E dyad, and a quiet third all fail for A and every hybrid. A pitch class can be harmonically compatible while having little energy, and `α·chroma` halves it.
  - **Case 14 (C major plus a loud foreign F#).** The F# destroys every chord's fit. D then falls back to chroma-only and gives F# full credit, while B correctly keeps it low.
- **Abstention is D's genuine contribution.** Only A and the D variants abstain on white noise (case 10). B and the top-5 methods produce confident scores for noise and clusters.
- **Top-5 "permissiveness" (case 16) was milder than expected.** On clean single chords C5u at α = 0.5 gave 3–4 pitch classes ≥ 0.5, the same as D, because the chroma term dominates. Its real cost is **clash leakage**: case 17 (clash-tier pitch classes ≤ 0.25) fails for C5u and passes for the weighted C5w and for D. On real songs, clash-tier H is 0.143 for C5u, 0.076 for C5w and 0.077 for D.

## Real songs (`method_comparison.md`; α = 0.5)

| | A | B | C5u | C5w | **D** | production |
|---|---|---|---|---|---|---|
| coverage | 0.975 | 1.000 | 1.000 | 1.000 | 0.976 | 1.000 |
| pitch classes ≥ 0.75 | 1.92 | 3.21 | 2.34 | 2.64 | 2.12 | 4.81 |
| pitch classes ≥ 0.5 | 3.51 | 6.53 | 4.39 | 4.34 | 3.65 | 5.50 |
| notes with ≥ 10 pcs ≥ 0.5 | 0.5% | 0.0% | 0.8% | 0.4% | 0.5% | 0.0% |
| graded share | 0.36 | 0.50 | 0.50 | 0.46 | 0.39 | 0.12 |
| H by reference tier (chord tone / extension / tension / clash) | .84/.18/.11/.13 | .99/.66/.33/.03 | .88/.39/.24/.14 | .91/.42/.22/.08 | .91/.41/.22/.08 | .98/.43/.18/.09 |
| \|ΔH\| under CENS noise | 0.0059 | 0.0070 | 0.0067 | 0.0059 | 0.0085 | 0.0145 (exp. 2) |
| \|ΔH\| under ±50 ms note shift | 0.0112 | 0.0112 | 0.0090 | 0.0101 | 0.0118 | 0.0139 (exp. 2) |
| \|ΔH\| under +50 ms beat-grid shift | 0.0201 | 0.0262 | 0.0196 | 0.0205 | 0.0268 | — |
| *secondary:* H sung / alternatives, win rate | .46/.35, .610 | .68/.51, .641 | .57/.44, .642 | .57/.44, .639 | .49/.37, .617 | .66/.44, .628 |

- **D's real-song notes:** chord-based for 26.3%, chroma-only fallback for 70.7%, insufficient (abstain) for 3.0%. **On real songs the pre-registered D is mostly Method A.**
- **Chroma alone does not grade.** Under a reference chord, A gives extensions 0.18 and clashes 0.13, nearly the same. It knows whether a pitch class is sounding, not how it relates to the harmony.
- **α trades grading for chroma-likeness.** For D, extension H is 0.52, 0.41 and 0.30 at α = 0.25, 0.5 and 0.75.
- **Cost per song, excluding Demucs and CREPE:**

| step | time |
|---|---|
| beat segments | 0.0008 s |
| A | < 0.0001 s |
| B / C | 0.0003 s |
| D's evidence (200 permutations × 112 templates × about 463 beats) | 0.033 s |
| all 15 variants plus note aggregation | 0.004 s |

  Every method is trivially fast for a local FastAPI service.

## Ablations within D (`ablation_results.md`)

| variant | synthetic T / S | real pcs ≥ 0.5 | noise \|ΔH\| | what it isolates |
|---|---|---|---|---|
| D1 weighting, no threshold | 12 / 11 | 4.19 | 0.0071 | fails noise in both layers (accepts chance fits); broader and less clash-safe (clash H 0.096) |
| D2 threshold, uniform | 13 / 13 | 3.65 (identical to D) | 0.0085 | |
| D3 threshold + weights (= D) | 13 / 13 | 3.65 | 0.0085 | |
| D4 + adaptive α | 13 / 13 | 3.60 | 0.0079 | extension H 0.374 vs 0.408 |

- **Excluding unsupported chords** (the threshold) is what makes abstention and noise rejection possible. D1 cannot reject noise.
- **Weighting the plausible chords** makes almost no difference inside D, because about 1.1 chords qualify. It does matter whenever several chords are combined: C5w vs C5u cuts clash leakage roughly in half.
- **Combining with chroma** costs grading and "silent but compatible" behavior.
- **Adaptive α** changes little: synthetic results are identical, and on real songs pitch classes ≥ 0.5 go from 3.65 to 3.60 and extension H from 0.408 to 0.374. It does not remove the chroma term's basic problem.

## Post-hoc: γ sensitivity and confidence-weighted *chord-only* (`posthoc_checks.json`)

The chroma term was the main source of failures, so the natural follow-up keeps D's evidence machinery and drops chroma from the score:
- `Dchord_*` uses α = 0, with excess-fit-selected and excess-fit-weighted chords.
- Chroma informativeness is used only to decide abstention.
- When no chord qualifies there are three fallbacks:
  - `Dchord_Bfb`: the posterior chords, flagged LOW_EVIDENCE
  - `_chromafb`: chroma
  - `_abstain`: abstain

| γ | variant | synthetic T / S | real coverage | pcs ≥ .75 / ≥ .5 | H by tier (top-fit reference) | noise \|ΔH\| |
|---|---|---|---|---|---|---|
| 0.05 | Dchord_Bfb | 14 / 14 (accepts noise) | 0.981 | 3.28 / 6.51 | .97/.64/.34/.06 | 0.0073 |
| **0.10** | **Dchord_Bfb** | **15 / 15** | 0.977 | 3.25 / 6.52 | .97/.65/.34/.05 | 0.0069 |
| 0.10 | Dchord_abstain | 14 / 14 | **0.571** | 3.27 / 6.50 | .97/.65/.34/.05 | 0.0044 |
| 0.10 | D (α = 0.5) | 13 / 12 | 0.977 | 2.32 / 3.85 | .82/.35/.24/.16 | 0.0086 |
| 0.15 | Dchord_Bfb | 15 / 15 | 0.976 | 3.22 / 6.53 | .97/.65/.34/.05 | 0.0064 |
| — | B (reference) | 14 / 14 | 1.000 | 3.21 / 6.53 | .97/.65/.34/.05 | 0.0070 |

- **Dchord_Bfb passes the most synthetic checks of any method** (15/17 in both layers). It fails only the case-6 expectation conflict and the cluster case, where the posterior fallback names a chord.
- **On real songs it is within 0.003 of B on every aggregate.** In practice its value is principled abstention (about 2.3–3% of notes, all with near-flat chroma) and an explicit evidence path for the 50% of beats that pass γ = 0.10, not different scores.
- **Abstaining whenever no chord qualifies would be most stable** (0.0044), but it drops 43% of notes at γ = 0.10 and 67% at γ = 0.15.

## Real-song examples (`failure_analysis.md`)

- **Chroma credits whatever is sounding.** `justin_bieber_baby` at 158.6 s, sung D#: the accompaniment chroma is nearly just D# (I_c 0.033, best chord fit r = 0.47, excess −0.27). A and D's chroma fallback give 0.99, B gives 0.00, and the posterior claims Dsus2 with 99.6% confidence. The near-solo pitch class is consistent with vocal bleed into `other`, though that can't be verified without audio. Neither extreme is trustworthy, and this is the kind of case to flag, not score.
- **The chroma term halves a sung chord tone that is quiet in the mix.** `jeff_buckley_hallelujah` at 192.5 s, sung E over Am (eligible: Am 1.00): chroma 0.32, so A gives 0.26 and D 0.47, while B gives 1.00.
- **The extension effect.** `no_ordinary_love` at 82.6 s, sung C# over Gmaj7 (#11, tier T2): B gives 0.81, D 0.54, A 0.40.
- **Ambiguous harmony handled gracefully.** `if_i_ain_t_got_you` at 124.3 s has eligible chords Bm7 0.34, Dmaj7 0.33 and D 0.32. The sung D is a tone of all three and scores 0.91 in D. In `amy_winehouse_love_is_a_losing_game` at 55.4 s, C over {Em 0.38, Cmaj7 0.35, Em7 0.28} is a chord tone only under Cmaj7, so D gives a graded 0.53 and B 0.57.
- **Correct abstention where others are confident.** `mark_ronson_uptown_funk` at 25.7 s has I_c 0.01. D and A abstain; B gives 1.0 (posterior C5 at 100%), and production gives 1.0.
- **Short notes.** 25% of notes are under 0.1 s. Sung-note H is 0.58 for those vs 0.81 for notes of 0.8 s or longer under B, and 0.41 vs 0.59 under D. Many are clash-tier with high-confidence chords, e.g. B over D# for 60 ms in `adele_hello` at 34.7 s, likely a scoop or grace note. They need the SHORT flag and reduced weight, not full-weight scoring.
- **Very long v2 notes** (4–15 s, e.g. Hallelujah 371.6–386.8 s) span several chords. Duration-weighted per-beat aggregation handles them better than one window, but whether a 15 s "note" is one sung note is a segmentation question.

## Answers

1. **Chromagram only (A):** stable (noise 0.006) and specific, but **not a compatibility measure**. It cannot credit silent-but-compatible tones (extensions 0.18), can't tell extensions from clashes (0.18 vs 0.13), and gives full credit to loud foreign or bleed-like pitch classes. Synthetic 10–11/17.
2. **Chords only (B):** the best grading (chord tone 0.99, extension 0.66, tension 0.33, clash 0.03), several compatible notes (3.2 at ≥ 0.75, 6.5 at ≥ 0.5, none near-universal), and synthetic 14/17. Its weakness is confidence: it can't abstain, and it scores noise and clusters with near-certainty.
3. **Top-5 hybrid (C5u):** no improvement. It is less graded than B (extension 0.39), lets clash pitch classes rise (clash H 0.14; synthetic case 17 fails), and cannot abstain. It was not dramatically over-permissive at α = 0.5, because the chroma term pulls everything down.
4. **Confidence-weighted hybrid (D):** better than C5u and A on synthetic checks (13 vs 11–12 and 10–11), clash-safe, and the only hybrid that abstains on noise. But it is **not better than chords-only**. The chroma term costs extension and silent-tone compatibility, and with the pre-registered γ, 71% of real notes fall back to chroma. **The main hypothesis is not supported as formulated.** Its confidence machinery is useful; its chroma blend is not.
5. **Does the dynamic threshold reject unsupported candidates?** On synthetic data, yes. At γ = 0.15, noise false-acceptance is 2.5% and clusters 0%, while clean recall stays at 85%. At γ = 0.10, noise false-acceptance is 9% and degraded recall 80%. On real songs its decisions are plausible (accepted beats are more diatonic and informative) but **unverified**. The real excess-fit distribution sits about 0.04 below synthetic, so no γ is validated for real stems.
6. **How confidence should be calculated:** as three separate parts.
   - **(i) Informativeness:** `I_c`, to decide whether there is harmony to analyze at all.
   - **(ii) Absolute evidence, self-normalized:** excess fit over the segment's own chance-level best fit.
   - **(iii) Relative competitiveness:** the gap to the best chord. It is cheap, but barely mattered here.
   
   Not raw r (noise reaches 0.72–0.83), not the softmax posterior (92–96% "confident" on noise), and not permutation significance (cannot certify clean triads).
7. **How many chords qualify:** usually one. At γ = 0.15, 73% of real beats have none, 24% one, 2.7% two and 0.1% three or more. Multi-chord sets appear in genuinely ambiguous textures and get sensible shared weights.
8. **Does weighting stop low-confidence chords from over-crediting notes?** Yes, where several chords are combined: posterior or excess-fit weighting halves clash leakage relative to uniform top-5 (0.076 vs 0.143 real; case 17). Inside D it matters little because about one chord qualifies; the **threshold** does the protective work.
9. **Which component is more useful?** Chord compatibility, by a wide margin, for the stated objective. Direct chroma adds "is it sounding", which actively conflicts with compatibility for extensions and quiet chord tones. Chroma's one clear value is **informativeness gating** (abstention).
10. **Fixed or confidence-aware α?** Neither rescues the hybrid. Adaptive α changes little (extension H 0.41 → 0.37), and α = 0 (chord-only, with chroma as a gate) is best supported by the synthetic checks and the grading.
11. **Several compatible choices without over-permissiveness?** Chord-based scores (B, Dchord) give about 3.2 pitch classes ≥ 0.75 and 6.5 ≥ 0.5 per note, with 0% of notes having ≥ 10. The hybrids narrow this to 2.1–2.6 and 3.6–4.4, mainly by penalizing silent extensions.
12. **When chord evidence is unreliable:**
    - **B** stays overconfident.
    - **D** falls back to chroma and inherits its blind spots (the loud-foreign-tone and bleed-like cases).
    - **Abstaining** is honest, but at real-song acceptance rates it removes 43–67% of notes.
    
    The defensible compromise is to score with the full chord distribution and **flag** it LOW_EVIDENCE when no chord passes the evidence gate, and to abstain only when the chroma itself is uninformative (≈ 3% of notes).
13. **Most stable:** all methods are far more stable than production (noise ≤ 0.0099 vs 0.0145). The chroma-weighted variants (A, C5w) are lowest under noise (0.0059); B and Dchord are close behind (≈ 0.007). D is slightly higher (0.0085), plausibly because noise moves 2.7% of beats across the chord/chroma-only boundary (status agreement 97.3%).
14. **Remaining failure cases:**
    - **evidence levels** not validated on real stems (no chord annotations)
    - **posterior overconfidence** in the low-evidence fallback
    - **clusters and dense textures** still get named chords under B-type fallbacks
    - a **loud foreign tone** breaks every chord's fit
    - **near-solo, bleed-like** accompaniment chroma
    - **short notes** (25%, often clash-tier scoops)
    - **multi-chord v2 notes** lasting seconds
    - the **tier spacing itself**, which is unvalidated
15. **Recommended next Axis 1 implementation: confidence-gated chord compatibility** (`Dchord_Bfb`, γ ≈ 0.10, provisional; chosen post hoc):
    - **Score:** H = the excess-fit-weighted expected tier over the chords that pass `c_k ≥ γ`, per beat, duration-weighted across a note's beats, **without a chroma term**.
    - **No chord qualifies:** use the full chord distribution and set `LOW_EVIDENCE`.
    - **Chroma uninformative** (`I_c < κ`): report `INSUFFICIENT_HARMONIC_EVIDENCE` and don't score.
    - **Reason codes:** eligible chords with weights, excess fit, `I_c`, SHORT. The PASSING/NEIGHBOR context stays a separate output.
    
    Why: it keeps B's grading and breadth, passed the most synthetic checks (15/17), adds the only principled abstention found here, is stable (0.0069), and costs about 0.04 s per song. The honest caveat is that on real songs it scores almost exactly like B. Its advantages are transparency and abstention, not demonstrated accuracy. γ, κ and the tier spacing all need **real chord annotations** (about 10 songs, at beat level) and a **human-labelled compatibility pilot** before any 0–100 calibration.

Production scoring, the frontend, Axis 2 and the feature store were not modified.
