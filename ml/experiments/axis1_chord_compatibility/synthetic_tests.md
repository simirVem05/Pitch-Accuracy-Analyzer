# Synthetic tests

Generated from `results.json`. The intended behavior of every case was fixed in `protocol.md` before any run.

- **Layer T (template level):** idealized chroma (harmonic-partial chord chroma) injected directly. This tests the scoring rules only.
- **Layer S (audio level):** synthetic audio (summed harmonic partials) → the exact production `chroma_cens` call → the full pipeline. This also tests harmony estimation, but on timbres far simpler than a real `other` stem.

Passing these checks shows that the rules behave as specified. It does **not** show agreement with human perception.

**Passed: layer T 25/28, layer S 24/28.**

| layer | case | check | pass | detail (this method) | production (reference) |
|---|---|---|---|---|---|
| T | 1 | C,E,G >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.01 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.55 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.18 A=0.00 A#=0.00 B=0.91 |
| T | 1 | min(chord tones) > max(T3,T4) | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.01 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.55 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.18 A=0.00 A#=0.00 B=0.91 |
| T | 1 | D,A,B in [0.4,0.95) and each > C#,F#,G# | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.01 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.55 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.18 A=0.00 A#=0.00 B=0.91 |
| T | 2 | A,C,E >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.33 G=0.67 G#=0.33 A=1.00 A#=0.00 B=0.67 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=1.00 F=0.00 F#=0.00 G=0.55 G#=0.18 A=1.00 A#=0.00 B=0.91 |
| T | 2 | min(chord tones) > max(T4: A#, C#) | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.33 G=0.67 G#=0.33 A=1.00 A#=0.00 B=0.67 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=1.00 F=0.00 F#=0.00 G=0.55 G#=0.18 A=1.00 A#=0.00 B=0.91 |
| T | 3 | G,B,D,F >= 0.75 | PASS | C=0.33 C#=0.33 D=1.00 D#=0.33 E=0.67 F=1.00 F#=0.00 G=1.00 G#=0.33 A=0.67 A#=0.33 B=1.00 | C=0.00 C#=0.00 D=1.00 D#=0.00 E=0.00 F=0.91 F#=0.18 G=1.00 G#=0.00 A=0.55 A#=0.00 B=1.00 |
| T | 4 | C,E,G,B >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.67 G=1.00 G#=0.00 A=0.67 A#=0.00 B=1.00 | C=0.91 C#=0.00 D=0.18 D#=0.00 E=1.00 F=0.00 F#=0.55 G=1.00 G#=0.00 A=0.00 A#=0.00 B=1.00 |
| T | 4 | F (avoid 11) < D (9) | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.67 G=1.00 G#=0.00 A=0.67 A#=0.00 B=1.00 | C=0.91 C#=0.00 D=0.18 D#=0.00 E=1.00 F=0.00 F#=0.55 G=1.00 G#=0.00 A=0.00 A#=0.00 B=1.00 |
| T | 5 | C,F,G >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.50 E=0.33 F=1.00 F#=0.00 G=1.00 G#=0.17 A=0.67 A#=0.67 B=0.17 | C=1.00 C#=0.00 D=0.91 D#=0.00 E=0.00 F=1.00 F#=0.00 G=1.00 G#=0.00 A=0.18 A#=0.00 B=0.55 |
| T | 5 | E (3rd) >= 0.25 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.50 E=0.33 F=1.00 F#=0.00 G=1.00 G#=0.17 A=0.67 A#=0.67 B=0.17 | C=1.00 C#=0.00 D=0.91 D#=0.00 E=0.00 F=1.00 F#=0.00 G=1.00 G#=0.00 A=0.18 A#=0.00 B=0.55 |
| T | 6 | interior-frame top-chord accuracy >= 0.8 | PASS | accuracy=0.997 |  |
| T | 6 | each segment's chord tones >= 0.75 at its centre | PASS | min chord-tone H per segment: 1.00, 1.00, 1.00, 1.00 | 1.00, 1.00, 0.83, 1.00 |
| T | 7 | CROSSES_CHANGE set and H_ctx >= max(halves) | PASS | crosses=True H_inst=0.84 half1=1.00 half2=0.67 H_ctx=1.00 code='CROSSES_CHANGE' | 1.00 |
| T | 8 | same pitch class at MIDI 52/64/76 -> identical H | **FAIL** | exact equality of the 12-vectors |  |
| T | 8 | 3 voicings of C: chord tones >= 0.75 in each, max |dH| <= 0.15 | PASS | min chord tone=1.00, max |dH| across voicings=0.000 |  |
| T | 9 | over C: D, A >= 0.4 | PASS | D=0.67 A=0.67 | D=0.55 A=0.00 |
| T | 9 | over G7: A (9) >= 0.4 | PASS | A=0.67 | A=0.55 |
| T | 9 | over C: F (11) < D (9) | PASS | F=0.33 D=0.67 | F=0.00 D=0.55 |
| T | 10 | F: H_inst < 0.5, code PASSING, H_ctx >= 2/3 | PASS | H_inst=0.33 code='PASSING' H_ctx=0.67 | 0.00 |
| T | 11 | held F: code SUSPENSION and H_ctx >= 2/3 | **FAIL** | H_inst=0.67 code='CROSSES_CHANGE' H_ctx=1.00 crosses=True | 1.00 |
| T | 11b | re-articulated F over C: SUSPENSION or H_inst >= 2/3; H_ctx >= 2/3 | PASS | H_inst=0.33 code='SUSPENSION' H_ctx=0.67 | 1.00 |
| T | 12 | C# over C: H_inst <= 1/3 and H_ctx == H_inst | PASS | H_inst=0.00 H_ctx=0.00 code='' | 0.00 |
| T | 13a | noise: LOW_EVIDENCE (e<0.5) and no pc < 0.25 | PASS | e=0.00 min=0.57 max=0.57 | min=0.00 max=1.00 |
| T | 13b | single C: C >= 0.75; E, Eb >= 0.4; AMBIGUOUS_CHORD (top-1 < 0.35) | **FAIL** | C=1.00 E=0.67 Eb=0.64 top1=C5@1.00 | C=1.00 E=1.00 Eb=0.00 |
| T | 13c | C5: C, G >= 0.75; E, Eb >= 0.5 | PASS | C=1.00 G=1.00 E=0.67 Eb=0.67 top1=C5@1.00 | C=1.00 G=1.00 E=0.55 Eb=0.00 |
| T | 14 | cases 1-5: <= 7 pcs >= 0.5 and >= 2 pcs <= 1/3 | PASS | (#>=0.5, #<=1/3) = [(6, 4), (6, 4), (6, 6), (7, 4), (7, 4)] | [(5, 7), (5, 7), (5, 7), (5, 7), (5, 7)] |
| T | 15 | +-50 ms shift: |dH| <= 0.05 | PASS | max |dH|=0.0000 |  |
| T | 16 | C major + noise (10 dB SNR at audio level): C,E,G >= 0.7 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.01 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.55 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.18 A=0.00 A#=0.00 B=0.91 |
| S | 1 | C,E,G >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.00 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.00 B=0.91 |
| S | 1 | min(chord tones) > max(T3,T4) | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.00 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.00 B=0.91 |
| S | 1 | D,A,B in [0.4,0.95) and each > C#,F#,G# | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.00 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.00 B=0.91 |
| S | 2 | A,C,E >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.33 G=0.67 G#=0.33 A=1.00 A#=0.00 B=0.67 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=1.00 F=0.00 F#=0.00 G=0.00 G#=0.18 A=1.00 A#=0.21 B=0.91 |
| S | 2 | min(chord tones) > max(T4: A#, C#) | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.33 G=0.67 G#=0.33 A=1.00 A#=0.00 B=0.67 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=1.00 F=0.00 F#=0.00 G=0.00 G#=0.18 A=1.00 A#=0.21 B=0.91 |
| S | 3 | G,B,D,F >= 0.75 | PASS | C=0.33 C#=0.33 D=1.00 D#=0.33 E=0.67 F=0.99 F#=0.01 G=1.00 G#=0.33 A=0.67 A#=0.33 B=1.00 | C=0.00 C#=0.00 D=1.00 D#=0.00 E=0.00 F=0.91 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.34 B=1.00 |
| S | 4 | C,E,G,B >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.67 G=1.00 G#=0.00 A=0.67 A#=0.00 B=1.00 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=0.91 F=0.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.45 B=1.00 |
| S | 4 | F (avoid 11) < D (9) | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.67 G=1.00 G#=0.00 A=0.67 A#=0.00 B=1.00 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=0.91 F=0.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.45 B=1.00 |
| S | 5 | C,F,G >= 0.75 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.50 E=0.33 F=1.00 F#=0.00 G=1.00 G#=0.17 A=0.67 A#=0.67 B=0.17 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=0.00 F=1.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.06 B=0.81 |
| S | 5 | E (3rd) >= 0.25 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.50 E=0.33 F=1.00 F#=0.00 G=1.00 G#=0.17 A=0.67 A#=0.67 B=0.17 | C=1.00 C#=0.00 D=0.00 D#=0.00 E=0.00 F=1.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.06 B=0.81 |
| S | 6 | interior-frame top-chord accuracy >= 0.8 | PASS | accuracy=1.000 |  |
| S | 6 | each segment's chord tones >= 0.75 at its centre | PASS | min chord-tone H per segment: 1.00, 1.00, 1.00, 1.00 | 1.00, 1.00, 1.00, 1.00 |
| S | 7 | CROSSES_CHANGE set and H_ctx >= max(halves) | PASS | crosses=True H_inst=0.84 half1=1.00 half2=0.67 H_ctx=1.00 code='CROSSES_CHANGE' | 1.00 |
| S | 8 | same pitch class at MIDI 52/64/76 -> identical H | **FAIL** | exact equality of the 12-vectors |  |
| S | 8 | 3 voicings of C: chord tones >= 0.75 in each, max |dH| <= 0.15 | PASS | min chord tone=1.00, max |dH| across voicings=0.000 |  |
| S | 9 | over C: D, A >= 0.4 | PASS | D=0.67 A=0.67 | D=0.00 A=0.00 |
| S | 9 | over G7: A (9) >= 0.4 | PASS | A=0.67 | A=0.00 |
| S | 9 | over C: F (11) < D (9) | PASS | F=0.33 D=0.67 | F=0.00 D=0.00 |
| S | 10 | F: H_inst < 0.5, code PASSING, H_ctx >= 2/3 | PASS | H_inst=0.33 code='PASSING' H_ctx=0.67 | 0.00 |
| S | 11 | held F: code SUSPENSION and H_ctx >= 2/3 | **FAIL** | H_inst=0.67 code='CROSSES_CHANGE' H_ctx=1.00 crosses=True | 1.00 |
| S | 11b | re-articulated F over C: SUSPENSION or H_inst >= 2/3; H_ctx >= 2/3 | PASS | H_inst=0.33 code='SUSPENSION' H_ctx=0.67 | 1.00 |
| S | 12 | C# over C: H_inst <= 1/3 and H_ctx == H_inst | PASS | H_inst=0.00 H_ctx=0.00 code='' | 0.00 |
| S | 13a | noise: LOW_EVIDENCE (e<0.5) and no pc < 0.25 | **FAIL** | e=0.10 min=0.15 max=0.96 | min=0.00 max=1.00 |
| S | 13b | single C: C >= 0.75; E, Eb >= 0.4; AMBIGUOUS_CHORD (top-1 < 0.35) | **FAIL** | C=1.00 E=0.66 Eb=0.66 top1=C5@1.00 | C=1.00 E=0.00 Eb=0.00 |
| S | 13c | C5: C, G >= 0.75; E, Eb >= 0.5 | PASS | C=1.00 G=1.00 E=0.67 Eb=0.67 top1=C5@1.00 | C=1.00 G=1.00 E=0.00 Eb=0.00 |
| S | 14 | cases 1-5: <= 7 pcs >= 0.5 and >= 2 pcs <= 1/3 | PASS | (#>=0.5, #<=1/3) = [(6, 3), (6, 3), (6, 4), (7, 4), (7, 4)] | [(4, 8), (4, 8), (4, 7), (4, 7), (4, 8)] |
| S | 15 | +-50 ms shift: |dH| <= 0.05 | PASS | max |dH|=0.0000 |  |
| S | 16 | C major + noise (10 dB SNR at audio level): C,E,G >= 0.7 | PASS | C=1.00 C#=0.00 D=0.67 D#=0.33 E=1.00 F=0.33 F#=0.00 G=1.00 G#=0.00 A=0.67 A#=0.33 B=0.67 | C=1.00 C#=0.00 D=0.87 D#=0.00 E=1.00 F=0.00 F#=0.00 G=1.00 G#=0.00 A=0.00 A#=0.00 B=0.71 |

## Chord identifiability (synthetic sets A/B)

Frames: A 9264, B clean 9264, B degraded 9264. "Degraded" adds one non-chord tone at half the level of a chord tone, plus white noise at 15 dB SNR.

| template/similarity | τ | θ | log-loss A | top-1 clean | top-3 clean | top-1 degraded | top-3 degraded | N recall | false N |
|---|---|---|---|---|---|---|---|---|---|
| binary/cosine | 0.01 | 0.9 | 0.060 | 0.977 | 1.000 | 0.731 | 0.999 | 1.000 | 0.001 |
| binary/pearson **(selected)** | 0.02 | 0.8 | 0.082 | 0.982 | 1.000 | 0.866 | 0.999 | 0.860 | 0.000 |
| harmonic/cosine | 0.01 | 0.9 | 0.033 | 0.986 | 1.000 | 0.767 | 1.000 | 0.979 | 0.000 |
| harmonic/pearson | 0.01 | 0.9 | 0.041 | 0.985 | 1.000 | 0.493 | 1.000 | 0.978 | 0.004 |

Per-quality top-1 (selected estimator, single frames, no smoothing):

| quality | clean top-1 | clean top-3 | degraded top-1 | degraded top-3 | most common confusions (degraded) |
|---|---|---|---|---|---|
| aug | 1.000 | 1.000 | 0.917 | 1.000 | {'maj': 24} |
| dim | 0.988 | 1.000 | 0.808 | 1.000 | {'hdim7': 78, 'dom7': 32, 'min': 23, 'dim': 21} |
| dom7 | 1.000 | 1.000 | 0.997 | 1.000 | {'hdim7': 3} |
| hdim7 | 1.000 | 1.000 | 0.986 | 1.000 | {'min7': 12} |
| maj | 0.975 | 1.000 | 0.767 | 1.000 | {'min7': 77, 'dom7': 74, 'maj7': 50} |
| maj7 | 1.000 | 1.000 | 0.997 | 1.000 | {'min7': 3} |
| min | 1.000 | 1.000 | 0.685 | 1.000 | {'maj7': 112, 'min7': 96, 'hdim7': 62, 'sus2+sus4': 2} |
| min7 | 1.000 | 1.000 | 0.999 | 1.000 | {'dom7': 1} |
| power | 0.867 | 1.000 | 0.602 | 0.987 | {'sus2+sus4': 205, 'maj': 60, 'min': 36, 'N': 31} |
| sus2+sus4 | 1.000 | 1.000 | 0.934 | 1.000 | {'maj': 36, 'min': 11, 'sus2+sus4': 10} |

## Temporal estimators on synthetic progressions (set C)

48 progressions, 32 beats each at 80–130 BPM. Chords last 1, 2 or 4 beats; diatonic, with some 7ths and sus4.

| estimator | chord length | frame acc | interior acc | switches/beat (true) | flicker/beat | boundary recall (±½ beat) |
|---|---|---|---|---|---|---|
| frame | 1 beat | 0.763 | 0.997 | 1.000 (1.000) | 0.083 | 1.000 |
| frame | 2 beat | 0.888 | 0.989 | 0.486 (0.484) | 0.000 | 1.000 |
| frame | 4 beat | 0.954 | 1.000 | 0.226 (0.226) | 0.000 | 1.000 |
| frame_hmm | 1 beat | 0.802 | 0.997 | 0.996 (1.000) | 0.079 | 0.996 |
| frame_hmm | 2 beat | 0.914 | 0.993 | 0.488 (0.484) | 0.000 | 1.000 |
| frame_hmm | 4 beat | 0.964 | 1.000 | 0.226 (0.226) | 0.000 | 1.000 |
| beat **(selected)** | 1 beat | 0.924 | 0.960 | 0.998 (1.000) | 0.079 | 0.998 |
| beat **(selected)** | 2 beat | 0.988 | 0.989 | 0.492 (0.484) | 0.000 | 1.000 |
| beat **(selected)** | 4 beat | 0.994 | 0.997 | 0.232 (0.226) | 0.000 | 1.000 |
| beat_hmm | 1 beat | 0.797 | 0.894 | 0.851 (1.000) | 0.062 | 0.851 |
| beat_hmm | 2 beat | 0.996 | 0.996 | 0.484 (0.484) | 0.000 | 1.000 |
| beat_hmm | 4 beat | 1.000 | 1.000 | 0.226 (0.226) | 0.000 | 1.000 |

Accuracy by distance to the nearest true chord change (all durations):

| estimator | <0.1 s | 0.1–0.25 s | 0.25–0.5 s | >0.5 s |
|---|---|---|---|---|
| frame | 0.440 | 0.948 | 0.993 | 0.998 |
| frame_hmm | 0.561 | 0.954 | 0.990 | 0.998 |
| beat | 0.965 | 0.966 | 0.971 | 0.998 |
| beat_hmm | 0.930 | 0.930 | 0.954 | 0.978 |

Key estimation (Krumhansl–Kessler, τ_key = 0.1): MAP accuracy 0.604. Of 19 errors, 9 were relative major/minor and 8 were a fourth or fifth away.


## Post-run note (added by hand after the run; not generated)

**Case 8, first check, is a harness bug, not a model failure.** The harness placed the three octave notes (MIDI 52/64/76) at the *same* time. The context rules then treated them as each other's neighbors, so their `H_ctx` differed; real v2 notes never overlap. Re-run with one note per analysis (`posthoc_checks.py`, `case8_octave_invariance_fixed_harness`), both layers pass with exact equality. **Corrected tallies: T 26/28, S 25/28.** The remaining failures (11, 13a-S, 13b) are genuine and are discussed in `summary.md`.
