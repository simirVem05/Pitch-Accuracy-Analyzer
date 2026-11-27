# Synthetic tests

Generated from `results.json`. Each requirement was fixed in `protocol.md` before any run and is applied to **every** method.
- **Layer T:** idealized harmonic-partial chroma; tests the scoring rules.
- **Layer S:** synthetic audio → the production CENS call; also tests estimation.
Passing shows rule behavior on synthetic material only, not perceptual validity.

## Pass counts

| method | T | S |
|---|---|---|
| A | 10/17 | 11/17 |
| A_rank | 9/17 | 10/17 |
| B | 14/17 | 14/17 |
| C5u_a0.25 | 13/17 | 14/17 |
| C5u_a0.5 | 11/17 | 12/17 |
| C5u_a0.75 | 9/17 | 10/17 |
| C5w_a0.25 | 14/17 | 14/17 |
| C5w_a0.5 | 11/17 | 12/17 |
| C5w_a0.75 | 10/17 | 10/17 |
| D_a0.25 | 13/17 | 13/17 |
| D_a0.5 | 13/17 | 13/17 |
| D_a0.75 | 11/17 | 11/17 |
| D1_weight_nothreshold | 12/17 | 11/17 |
| D2_threshold_uniform | 13/17 | 13/17 |
| D4_adaptive_alpha | 13/17 | 13/17 |

## Case × method (✓ pass, ✗ fail)

### Layer T

| case | truth | requirement | A | A_rank | B | C5u_a0.25 | C5u_a0.5 | C5u_a0.75 | C5w_a0.25 | C5w_a0.5 | C5w_a0.75 | D_a0.25 | D_a0.5 | D_a0.75 | D1_weight_nothreshold | D2_threshold_uniform | D4_adaptive_alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exact | C,E,G>=.75; D,A>=1/3; C#,F#,G#<=.25; min(CEG)>max(DAB) | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| 2 | exact | A,C,E>=.75; A#,C#<=.25 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | exact | G,B,D,F>=.75 | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | exact | C,F,G>=.75; E>=.25 | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 5 | exact (thirds undetermined) | C,G>=.75; E>=1/3 and Eb>=1/3 | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| 6 | ambiguous | C,E,G,A>=.6; C#,F#<=.4 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 7 | ambiguous | C,E>=.75; G>=1/3 and A>=1/3 | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | exact (noisy) | C,E,G>=.6; C#,F#<=.4 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | ambiguous (no credible chord) | abstain OR max over the 6 absent pcs <= .5 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | no harmony | abstain / insufficient | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 11 | identity undetermined | C>=.75; |E-Eb|<=.2 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 12 | exact | each chord's tones >= .75 at its centre | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 13 | exact | quiet third E >= .6 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 14 | exact chord + foreign F# | F# < min(C,E,G) and F# <= .6 | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| 15 | exact | each of cases 1-4: >=3 pcs >=.75 and <=8 pcs >=.5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 17 | exact | T4 pcs of the true chord <= .25 (cases 1-4) | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 18 | exact | +-50 ms |dH|<=.05; CENS noise |dH|<=.10 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Layer S

| case | truth | requirement | A | A_rank | B | C5u_a0.25 | C5u_a0.5 | C5u_a0.75 | C5w_a0.25 | C5w_a0.5 | C5w_a0.75 | D_a0.25 | D_a0.5 | D_a0.75 | D1_weight_nothreshold | D2_threshold_uniform | D4_adaptive_alpha |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exact | C,E,G>=.75; D,A>=1/3; C#,F#,G#<=.25; min(CEG)>max(DAB) | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| 2 | exact | A,C,E>=.75; A#,C#<=.25 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 3 | exact | G,B,D,F>=.75 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 4 | exact | C,F,G>=.75; E>=.25 | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |
| 5 | exact (thirds undetermined) | C,G>=.75; E>=1/3 and Eb>=1/3 | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| 6 | ambiguous | C,E,G,A>=.6; C#,F#<=.4 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✗ |
| 7 | ambiguous | C,E>=.75; G>=1/3 and A>=1/3 | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 8 | exact (noisy) | C,E,G>=.6; C#,F#<=.4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 9 | ambiguous (no credible chord) | abstain OR max over the 6 absent pcs <= .5 | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 10 | no harmony | abstain / insufficient | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| 11 | identity undetermined | C>=.75; |E-Eb|<=.2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 12 | exact | each chord's tones >= .75 at its centre | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 13 | exact | quiet third E >= .6 | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 14 | exact chord + foreign F# | F# < min(C,E,G) and F# <= .6 | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| 15 | exact | each of cases 1-4: >=3 pcs >=.75 and <=8 pcs >=.5 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 17 | exact | T4 pcs of the true chord <= .25 (cases 1-4) | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 18 | exact | +-50 ms |dH|<=.05; CENS noise |dH|<=.10 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Details for the main methods

| layer | case | method | pass | detail |
|---|---|---|---|---|
| T | 1 | A | ✗ | C=0.81 E=0.86 G=1.00 D=0.19 A=0.00 B=0.25 C#=0.00 F#=0.00 G#=0.06 |
| T | 1 | B | ✓ | C=1.00 E=1.00 G=1.00 D=0.67 A=0.67 B=0.67 C#=0.00 F#=0.01 G#=0.00 |
| T | 1 | C5u_a0.5 | ✓ | C=0.90 E=0.90 G=1.00 D=0.43 A=0.33 B=0.39 C#=0.03 F#=0.17 G#=0.10 |
| T | 1 | D_a0.5 | ✓ | C=0.90 E=0.93 G=1.00 D=0.43 A=0.33 B=0.46 C#=0.00 F#=0.00 G#=0.03 |
| T | 2 | A | ✓ | C=0.76 C#=0.05 E=1.00 A=0.76 A#=0.00 |
| T | 2 | B | ✓ | C=1.00 C#=0.00 E=1.00 A=1.00 A#=0.00 |
| T | 2 | C5u_a0.5 | ✓ | C=0.85 C#=0.09 E=1.00 A=0.88 A#=0.03 |
| T | 2 | D_a0.5 | ✓ | C=0.88 C#=0.03 E=1.00 A=0.88 A#=0.00 |
| T | 3 | A | ✓ | D=1.00 F=0.81 G=0.81 B=0.86 |
| T | 3 | B | ✓ | D=1.00 F=1.00 G=1.00 B=1.00 |
| T | 3 | C5u_a0.5 | ✗ | D=1.00 F=0.74 G=0.80 B=0.93 |
| T | 3 | D_a0.5 | ✓ | D=1.00 F=0.90 G=0.90 B=0.93 |
| T | 4 | A | ✗ | C=1.00 E=0.06 F=0.81 G=1.00 |
| T | 4 | B | ✓ | C=1.00 E=0.33 F=1.00 G=1.00 |
| T | 4 | C5u_a0.5 | ✓ | C=1.00 E=0.33 F=0.77 G=0.97 |
| T | 4 | D_a0.5 | ✗ | C=1.00 E=0.20 F=0.90 G=1.00 |
| T | 5 | A | ✗ | C=0.81 D#=0.00 E=0.06 G=1.00 |
| T | 5 | B | ✓ | C=1.00 D#=0.67 E=0.67 G=1.00 |
| T | 5 | C5u_a0.5 | ✗ | C=0.90 D#=0.27 E=0.30 G=1.00 |
| T | 5 | D_a0.5 | ✓ | C=0.90 D#=0.33 E=0.36 G=1.00 |
| T | 6 | A | ✓ | C=0.76 C#=0.05 E=1.00 F#=0.00 G=0.95 A=0.76 |
| T | 6 | B | ✗ | C=1.00 C#=0.00 E=1.00 F#=0.67 G=1.00 A=1.00 |
| T | 6 | C5u_a0.5 | ✓ | C=0.81 C#=0.13 E=1.00 F#=0.23 G=0.94 A=0.81 |
| T | 6 | D_a0.5 | ✓ | C=0.88 C#=0.03 E=1.00 F#=0.33 G=0.97 A=0.88 |
| T | 7 | A | ✗ | C=0.93 E=1.00 G=0.22 A=0.00 |
| T | 7 | B | ✓ | C=1.00 E=1.00 G=0.99 A=0.66 |
| T | 7 | C5u_a0.5 | ✗ | C=0.97 E=1.00 G=0.49 A=0.30 |
| T | 7 | D_a0.5 | ✗ | C=0.93 E=1.00 G=0.22 A=0.00 |
| T | 8 | A | ✓ | C=0.85 C#=0.38 E=0.89 F#=0.32 G=1.00 |
| T | 8 | B | ✓ | C=1.00 C#=0.00 E=1.00 F#=0.06 G=1.00 |
| T | 8 | C5u_a0.5 | ✓ | C=0.89 C#=0.22 E=0.91 F#=0.34 G=1.00 |
| T | 8 | D_a0.5 | ✓ | C=0.92 C#=0.19 E=0.94 F#=0.16 G=1.00 |
| T | 9 | A | ✓ | max over absent pcs=0.19 |
| T | 9 | B | ✗ | max over absent pcs=0.58 |
| T | 9 | C5u_a0.5 | ✓ | max over absent pcs=0.35 |
| T | 9 | D_a0.5 | ✓ | max over absent pcs=0.19 |
| T | 10 | A | ✓ | abstain |
| T | 10 | B | ✗ | scored: max=0.86 min=0.33 |
| T | 10 | C5u_a0.5 | ✗ | scored: max=0.92 min=0.64 |
| T | 10 | D_a0.5 | ✓ | abstain |
| T | 11 | A | ✓ | C=1.00 E=0.07 D#=0.00 |
| T | 11 | B | ✓ | C=1.00 E=0.67 D#=0.67 |
| T | 11 | C5u_a0.5 | ✓ | C=1.00 E=0.30 D#=0.27 |
| T | 11 | D_a0.5 | ✓ | C=1.00 E=0.37 D#=0.33 |
| T | 12 | A | ✓ | min chord-tone H per chord: 0.81, 0.81, 0.81, 0.81 |
| T | 12 | B | ✓ | min chord-tone H per chord: 1.00, 1.00, 1.00, 1.00 |
| T | 12 | C5u_a0.5 | ✓ | min chord-tone H per chord: 0.90, 0.90, 0.90, 0.90 |
| T | 12 | D_a0.5 | ✓ | min chord-tone H per chord: 0.90, 0.90, 0.90, 0.90 |
| T | 13 | A | ✗ | C=0.81 E=0.18 G=1.00 |
| T | 13 | B | ✓ | C=1.00 E=0.67 G=1.00 |
| T | 13 | C5u_a0.5 | ✗ | C=0.90 E=0.36 G=1.00 |
| T | 13 | D_a0.5 | ✗ | C=0.90 E=0.42 G=1.00 |
| T | 14 | A | ✗ | C=0.81 E=0.86 G=1.00 F#=0.81 |
| T | 14 | B | ✓ | C=1.00 E=1.00 G=1.00 F#=0.00 |
| T | 14 | C5u_a0.5 | ✓ | C=0.90 E=0.90 G=1.00 F#=0.57 |
| T | 14 | D_a0.5 | ✗ | C=0.81 E=0.86 G=1.00 F#=0.81 |
| T | 15 | A | ✓ | (#>=.75, #>=.5) = [(3, 3), (3, 3), (4, 4), (3, 3)] |
| T | 15 | B | ✓ | (#>=.75, #>=.5) = [(3, 6), (3, 6), (4, 6), (3, 7)] |
| T | 15 | C5u_a0.5 | ✓ | (#>=.75, #>=.5) = [(3, 3), (3, 3), (3, 4), (3, 3)] |
| T | 15 | D_a0.5 | ✓ | (#>=.75, #>=.5) = [(3, 3), (3, 3), (4, 4), (3, 3)] |
| T | 17 | A | ✗ | max T4 per case: 0.06, 0.05, 0.25, 0.00 |
| T | 17 | B | ✓ | max T4 per case: 0.01, 0.00, 0.00, 0.17 |
| T | 17 | C5u_a0.5 | ✗ | max T4 per case: 0.17, 0.09, 0.29, 0.12 |
| T | 17 | D_a0.5 | ✓ | max T4 per case: 0.03, 0.03, 0.13, 0.08 |
| T | 18 | A | ✓ | shift max|dH|=0.000, noise max|dH|=0.030 |
| T | 18 | B | ✓ | shift max|dH|=0.000, noise max|dH|=0.001 |
| T | 18 | C5u_a0.5 | ✓ | shift max|dH|=0.000, noise max|dH|=0.015 |
| T | 18 | D_a0.5 | ✓ | shift max|dH|=0.000, noise max|dH|=0.015 |
| S | 1 | A | ✗ | C=1.00 E=1.00 G=1.00 D=0.00 A=0.00 B=0.00 C#=0.00 F#=0.00 G#=0.00 |
| S | 1 | B | ✓ | C=1.00 E=1.00 G=1.00 D=0.67 A=0.67 B=0.67 C#=0.00 F#=0.00 G#=0.00 |
| S | 1 | C5u_a0.5 | ✓ | C=1.00 E=0.97 G=1.00 D=0.33 A=0.33 B=0.27 C#=0.03 F#=0.17 G#=0.07 |
| S | 1 | D_a0.5 | ✓ | C=1.00 E=1.00 G=1.00 D=0.33 A=0.33 B=0.33 C#=0.00 F#=0.00 G#=0.00 |
| S | 2 | A | ✓ | C=1.00 C#=0.00 E=1.00 A=1.00 A#=0.00 |
| S | 2 | B | ✓ | C=1.00 C#=0.00 E=1.00 A=1.00 A#=0.00 |
| S | 2 | C5u_a0.5 | ✓ | C=0.97 C#=0.07 E=1.00 A=1.00 A#=0.03 |
| S | 2 | D_a0.5 | ✓ | C=1.00 C#=0.00 E=1.00 A=1.00 A#=0.00 |
| S | 3 | A | ✓ | D=0.84 F=1.00 G=1.00 B=1.00 |
| S | 3 | B | ✓ | D=1.00 F=1.00 G=1.00 B=1.00 |
| S | 3 | C5u_a0.5 | ✓ | D=0.92 F=0.88 G=0.90 B=0.95 |
| S | 3 | D_a0.5 | ✓ | D=0.92 F=1.00 G=1.00 B=1.00 |
| S | 4 | A | ✗ | C=1.00 E=0.00 F=1.00 G=1.00 |
| S | 4 | B | ✓ | C=1.00 E=0.33 F=1.00 G=1.00 |
| S | 4 | C5u_a0.5 | ✓ | C=1.00 E=0.30 F=0.90 G=0.93 |
| S | 4 | D_a0.5 | ✗ | C=1.00 E=0.17 F=1.00 G=1.00 |
| S | 5 | A | ✗ | C=1.00 D#=0.00 E=0.00 G=1.00 |
| S | 5 | B | ✓ | C=1.00 D#=0.66 E=0.67 G=1.00 |
| S | 5 | C5u_a0.5 | ✗ | C=1.00 D#=0.27 E=0.27 G=1.00 |
| S | 5 | D_a0.5 | ✓ | C=1.00 D#=0.33 E=0.33 G=1.00 |
| S | 6 | A | ✓ | C=0.76 C#=0.00 E=1.00 F#=0.00 G=1.00 A=1.00 |
| S | 6 | B | ✗ | C=1.00 C#=0.00 E=1.00 F#=0.67 G=1.00 A=1.00 |
| S | 6 | C5u_a0.5 | ✓ | C=0.78 C#=0.17 E=1.00 F#=0.20 G=0.93 A=0.97 |
| S | 6 | D_a0.5 | ✓ | C=0.88 C#=0.00 E=1.00 F#=0.33 G=1.00 A=1.00 |
| S | 7 | A | ✗ | C=0.75 E=1.00 G=0.00 A=0.00 |
| S | 7 | B | ✓ | C=1.00 E=1.00 G=0.65 A=0.57 |
| S | 7 | C5u_a0.5 | ✗ | C=0.81 E=1.00 G=0.34 A=0.30 |
| S | 7 | D_a0.5 | ✗ | C=0.75 E=1.00 G=0.00 A=0.00 |
| S | 8 | A | ✓ | C=0.72 C#=0.00 E=1.00 F#=0.00 G=1.00 |
| S | 8 | B | ✓ | C=1.00 C#=0.00 E=1.00 F#=0.01 G=1.00 |
| S | 8 | C5u_a0.5 | ✓ | C=0.80 C#=0.07 E=1.00 F#=0.23 G=1.00 |
| S | 8 | D_a0.5 | ✓ | C=0.86 C#=0.00 E=1.00 F#=0.00 G=1.00 |
| S | 9 | A | ✓ | max over absent pcs=0.00 |
| S | 9 | B | ✗ | max over absent pcs=0.58 |
| S | 9 | C5u_a0.5 | ✓ | max over absent pcs=0.32 |
| S | 9 | D_a0.5 | ✓ | max over absent pcs=0.00 |
| S | 10 | A | ✓ | abstain |
| S | 10 | B | ✗ | scored: max=1.00 min=0.01 |
| S | 10 | C5u_a0.5 | ✗ | scored: max=0.87 min=0.58 |
| S | 10 | D_a0.5 | ✓ | abstain |
| S | 11 | A | ✓ | C=1.00 E=0.00 D#=0.00 |
| S | 11 | B | ✓ | C=1.00 E=0.67 D#=0.67 |
| S | 11 | C5u_a0.5 | ✓ | C=1.00 E=0.27 D#=0.27 |
| S | 11 | D_a0.5 | ✓ | C=1.00 E=0.33 D#=0.33 |
| S | 12 | A | ✓ | min chord-tone H per chord: 0.94, 0.93, 0.92, 0.94 |
| S | 12 | B | ✓ | min chord-tone H per chord: 1.00, 1.00, 1.00, 1.00 |
| S | 12 | C5u_a0.5 | ✓ | min chord-tone H per chord: 0.94, 0.93, 0.93, 0.94 |
| S | 12 | D_a0.5 | ✓ | min chord-tone H per chord: 0.97, 0.97, 0.96, 0.97 |
| S | 13 | A | ✗ | C=0.75 E=0.25 G=1.00 |
| S | 13 | B | ✓ | C=1.00 E=0.67 G=1.00 |
| S | 13 | C5u_a0.5 | ✗ | C=0.87 E=0.39 G=1.00 |
| S | 13 | D_a0.5 | ✗ | C=0.87 E=0.46 G=1.00 |
| S | 14 | A | ✗ | C=0.67 E=1.00 G=1.00 F#=1.00 |
| S | 14 | B | ✓ | C=1.00 E=1.00 G=1.00 F#=0.00 |
| S | 14 | C5u_a0.5 | ✗ | C=0.83 E=1.00 G=0.93 F#=0.77 |
| S | 14 | D_a0.5 | ✗ | C=0.67 E=1.00 G=1.00 F#=1.00 |
| S | 15 | A | ✓ | (#>=.75, #>=.5) = [(3, 3), (3, 3), (4, 4), (3, 3)] |
| S | 15 | B | ✓ | (#>=.75, #>=.5) = [(3, 6), (3, 6), (4, 6), (3, 7)] |
| S | 15 | C5u_a0.5 | ✓ | (#>=.75, #>=.5) = [(3, 3), (3, 3), (4, 4), (3, 3)] |
| S | 15 | D_a0.5 | ✓ | (#>=.75, #>=.5) = [(3, 3), (3, 3), (4, 4), (3, 3)] |
| S | 17 | A | ✓ | max T4 per case: 0.00, 0.00, 0.00, 0.00 |
| S | 17 | B | ✓ | max T4 per case: 0.00, 0.00, 0.00, 0.17 |
| S | 17 | C5u_a0.5 | ✓ | max T4 per case: 0.17, 0.07, 0.12, 0.15 |
| S | 17 | D_a0.5 | ✓ | max T4 per case: 0.00, 0.00, 0.00, 0.08 |
| S | 18 | A | ✓ | shift max|dH|=0.000, noise max|dH|=0.026 |
| S | 18 | B | ✓ | shift max|dH|=0.000, noise max|dH|=0.000 |
| S | 18 | C5u_a0.5 | ✓ | shift max|dH|=0.000, noise max|dH|=0.013 |
| S | 18 | D_a0.5 | ✓ | shift max|dH|=0.000, noise max|dH|=0.013 |

## Case 16 — top-5 permissiveness (demonstration)

| layer | method | pcs ≥ 0.5 in cases 1–4 | cases with a T4 pc ≥ 0.5 |
|---|---|---|---|
| T | A | [3, 3, 4, 3] | 0 |
| T | A_rank | [6, 6, 6, 6] | 2 |
| T | B | [6, 6, 6, 7] | 0 |
| T | C5u_a0.25 | [5, 6, 5, 4] | 0 |
| T | C5u_a0.5 | [3, 3, 4, 3] | 0 |
| T | C5u_a0.75 | [3, 3, 4, 3] | 0 |
| T | C5w_a0.25 | [5, 6, 5, 5] | 0 |
| T | C5w_a0.5 | [3, 3, 4, 3] | 0 |
| T | C5w_a0.75 | [3, 3, 4, 3] | 0 |
| T | D_a0.25 | [6, 6, 6, 6] | 0 |
| T | D_a0.5 | [3, 3, 4, 3] | 0 |
| T | D_a0.75 | [3, 3, 4, 3] | 0 |
| T | D1_weight_nothreshold | [4, 3, 4, 3] | 0 |
| T | D2_threshold_uniform | [3, 3, 4, 3] | 0 |
| T | D4_adaptive_alpha | [3, 3, 5, 3] | 0 |
| S | A | [3, 3, 4, 3] | 0 |
| S | A_rank | [6, 6, 6, 6] | 1 |
| S | B | [6, 6, 6, 7] | 0 |
| S | C5u_a0.25 | [5, 6, 5, 4] | 0 |
| S | C5u_a0.5 | [3, 3, 4, 3] | 0 |
| S | C5u_a0.75 | [3, 3, 4, 3] | 0 |
| S | C5w_a0.25 | [5, 6, 5, 4] | 0 |
| S | C5w_a0.5 | [3, 3, 4, 3] | 0 |
| S | C5w_a0.75 | [3, 3, 4, 3] | 0 |
| S | D_a0.25 | [6, 6, 6, 6] | 0 |
| S | D_a0.5 | [3, 3, 4, 3] | 0 |
| S | D_a0.75 | [3, 3, 4, 3] | 0 |
| S | D1_weight_nothreshold | [3, 3, 4, 3] | 0 |
| S | D2_threshold_uniform | [3, 3, 4, 3] | 0 |
| S | D4_adaptive_alpha | [6, 6, 4, 6] | 0 |

## D's eligible chords per case (dominant beat)

| layer | case | eligible (weight) | status | fit | top-r chord |
|---|---|---|---|---|---|
| T | 1 | C:1.00 | CHORD | r_best=0.971 | top_r=C |
| T | 2 | Am:1.00 | CHORD | r_best=0.971 | top_r=Am |
| T | 3 | G7:1.00 | CHORD | r_best=0.964 | top_r=G7 |
| T | 4 | Fsus2/Csus4:1.00 | CHORD | r_best=0.984 | top_r=Fsus2/Csus4 |
| T | 5 | C5:1.00 | CHORD | r_best=0.979 | top_r=C5 |
| T | 6 | Am7:1.00 | CHORD | r_best=0.970 | top_r=Am7 |
| T | 7 | — | CHROMA_ONLY | r_best=0.848 | top_r=C |
| T | 8 | C:1.00 | CHORD | r_best=0.958 | top_r=C |
| T | 9 | — | CHROMA_ONLY | r_best=0.571 | top_r=Csus2/Gsus4 |
| T | 10 | — | INSUFFICIENT | r_best=0.722 | top_r=A#m7b5 |
| T | 11 | C5:1.00 | CHORD | r_best=0.826 | top_r=C5 |
| T | 12 | C:1.00 | CHORD | r_best=0.971 | top_r=C |
| T | 13 | C5:1.00 | CHORD | r_best=0.971 | top_r=C5 |
| T | 14 | — | CHROMA_ONLY | r_best=0.824 | top_r=C |
| S | 1 | C:1.00 | CHORD | r_best=1.000 | top_r=C |
| S | 2 | Am:1.00 | CHORD | r_best=1.000 | top_r=Am |
| S | 3 | G7:1.00 | CHORD | r_best=0.989 | top_r=G7 |
| S | 4 | Fsus2/Csus4:1.00 | CHORD | r_best=1.000 | top_r=Fsus2/Csus4 |
| S | 5 | C5:1.00 | CHORD | r_best=0.983 | top_r=C5 |
| S | 6 | Am7:1.00 | CHORD | r_best=0.991 | top_r=Am7 |
| S | 7 | — | CHROMA_ONLY | r_best=0.733 | top_r=C |
| S | 8 | C:1.00 | CHORD | r_best=0.969 | top_r=C |
| S | 9 | — | CHROMA_ONLY | r_best=0.577 | top_r=Csus2/Gsus4 |
| S | 10 | — | INSUFFICIENT | r_best=0.862 | top_r=D5 |
| S | 11 | C5:1.00 | CHORD | r_best=0.836 | top_r=C5 |
| S | 12 | C:1.00 | CHORD | r_best=0.991 | top_r=C |
| S | 13 | C5:1.00 | CHORD | r_best=0.966 | top_r=C5 |
| S | 14 | — | CHROMA_ONLY | r_best=0.765 | top_r=C |
