# Failure analysis — real songs

Generated from `results.json` (selected method `chord_dist_key` / `chord_dist_key_ctx`). Counts are over 31,803 sung notes. Up to six examples per category are shown, one per song, chosen by the most extreme value. "Top chords" are the estimated harmony of the note window with renormalized probabilities. A failure category is a *symptom* to inspect; it is not proof the model is wrong.

| category | definition | notes |
|---|---|---|
| chord_tone_low | the sung pitch class is a chord tone of the top chord, but H_inst < 0.5 | 143 (0.4%) |
| breadth_ge10 | ≥ 10 of the 12 pitch classes have H ≥ 0.5 (nearly everything fits) | 2 (0.0%) |
| extension_low | the sung pitch class is a T2 extension of the top chord, but H_inst < 1/3 | 131 (0.4%) |
| noise_unstable | |ΔH_inst| of the sung pitch class ≥ 0.25 under CENS noise σ = 0.1 | 2 (0.0%) |
| crosses_change | the top chord differs between the note's first and second halves | 2344 (7.4%) |
| short_low | duration < 0.15 s and H_ctx < 1/3 | 1156 (3.6%) |
| low_evidence | evidence e < 0.5 (would abstain) | 9253 (29.1%) |

## chord_tone_low

| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |
|---|---|---|---|---|---|---|---|---|---|
| billiejean_mj_full_song | 63.58–63.78 | 0.2 | G | Gmaj7:0.91, B5:0.08, Esus2/Bsus4:0.00 | 0.0 | 0.011 | 0.011 |  | 1.0 |
| norah_jones_don_t_know_why | 86.98–87.24 | 0.26 | F# | Gmaj7:0.99, G5:0.01, D+/F#+/A#+:0.00 | 0.005 | 0.024 | 0.667 | STEP_RESOLUTION | 1.0 |
| ariana_grande_into_you_official_video | 252.66–254.88 | 2.22 | F | F5:0.42, A#5:0.32, Fmaj7:0.26 | 0.0 | 0.025 | 0.025 |  | 1.0 |
| wild_thoughts | 201.18–201.24 | 0.06 | C# | C#maj7:0.45, F5:0.25, Cmaj7:0.20 | 0.0 | 0.037 | 0.667 | NEIGHBOR | 0.045 |
| i_have_nothing | 90.42–90.52 | 0.1 | D# | D#sus2/A#sus4:0.35, F5:0.32, Dm7:0.26 | 0.032 | 0.054 | 0.054 |  | 0.891 |
| michael_jackson_chicago_official_audio | 45.36–45.52 | 0.16 | D# | D#:0.50, D#maj7:0.20, D#sus2/A#sus4:0.12 | 0.0 | 0.057 | 0.057 |  | 1.0 |

## breadth_ge10

| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |
|---|---|---|---|---|---|---|---|---|---|
| mark_ronson_uptown_funk_official_audio_ft_bruno_mars | 54.92–56.28 | 1.36 | F | C#7:1.00, Fdim:0.00, C#m7:0.00 | 0.357 | 0.919 | 0.919 |  | 1.0 |

## extension_low

| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |
|---|---|---|---|---|---|---|---|---|---|
| billiejean_mj_full_song | 152.98–153.04 | 0.06 | G | Dsus2/Asus4:0.69, D5:0.20, D:0.10 | 0.0 | 0.011 | 0.011 |  | 0.0 |
| john_legend_ordinary_people_hq | 121.26–121.32 | 0.06 | F# | D5:0.73, D#maj7:0.27, G5:0.00 | 0.0 | 0.023 | 0.023 |  | 0.0 |
| michael_jackson_chicago_official_audio | 43.82–43.88 | 0.06 | F# | E5:0.73, A5:0.20, Fmaj7:0.06 | 0.0 | 0.024 | 0.024 |  | 0.0 |
| ariana_grande_into_you_official_video | 20.66–20.7 | 0.04 | F | D5:0.76, Dsus2/Asus4:0.23, Dmaj7:0.01 | 0.0 | 0.025 | 0.667 | STEP_RESOLUTION | 0.672 |
| usher_burn | 187.38–187.54 | 0.16 | D | Fmaj7:0.81, F5:0.12, A#5:0.05 | 0.0 | 0.031 | 0.667 | STEP_RESOLUTION | 0.347 |
| florence_the_machine_never_let_me_go | 83.18–83.24 | 0.06 | A# | F#5:0.94, Esus2/Bsus4:0.05, B:0.01 | 0.0 | 0.038 | 0.667 | PASSING | 0.0 |

## noise_unstable

| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |
|---|---|---|---|---|---|---|---|---|---|
| demi_lovato_stone_cold_audio_only | 175.14–175.28 | 0.14 | C | C#dim:0.98, C#m:0.02, A#dim:0.00 | 0.602 | 0.396 | 0.396 |  | 0.0 |
| sheila_ki_jawani_audio_song_tees_maar_khan_katrina_kaif_vishal_dadlani_sunidhi_chauhan | 41.44–41.52 | 0.08 | B | F#maj7:0.98, A#m:0.01, F#:0.01 | 0.832 | 0.439 | 0.439 |  | 1.0 |

## crosses_change

| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |
|---|---|---|---|---|---|---|---|---|---|
| jeff_buckley_hallelujah | 371.62–386.78 | 15.16 | E | C5:0.32, G5:0.16, F5:0.16 | 0.693 | 0.751 | 0.815 | CROSSES_CHANGE | 0.579 |
| teri_ore_full_audio_singh_is_kinng_akshay_kkatrina_kpritamrahat_fateh_ali_khanshreya_ghoshal | 6.78–12.02 | 5.24 | D | D5:0.58, D:0.37, Bm7:0.03 | 0.997 | 1.0 | 1.0 | CROSSES_CHANGE | 1.0 |
| you_raise_me_up | 257.16–261.92 | 4.76 | F# | B:0.49, D#5:0.24, C#sus2/G#sus4:0.18 | 0.871 | 0.734 | 0.942 | CROSSES_CHANGE | 0.796 |
| no_ordinary_love | 231.3–236.04 | 4.74 | C# | B5:0.44, Bm:0.34, Esus2/Bsus4:0.22 | 0.66 | 0.671 | 0.671 | CROSSES_CHANGE | 0.0 |
| lately | 230.84–235.32 | 4.48 | F# | D#5:0.65, Bmaj7:0.25, G#sus2/D#sus4:0.08 | 0.893 | 0.712 | 0.77 | CROSSES_CHANGE | 0.0 |
| celine_dion_because_you_loved_me_theme_from_up_close_and_personal_audio | 266.22–270.6 | 4.38 | G | D#:0.66, D#5:0.31, D#sus2/A#sus4:0.03 | 0.999 | 0.887 | 0.952 | CROSSES_CHANGE | 0.645 |

## short_low

| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |
|---|---|---|---|---|---|---|---|---|---|
| adele_hello | 119.32–119.42 | 0.1 | E | D#:0.98, D#maj7:0.02, Cm7:0.00 | 1.0 | 0.0 | 0.0 |  | 0.0 |
| black_beatles | 215.94–216.04 | 0.1 | F | Em:1.00, Em7:0.00, C#m7b5:0.00 | 0.983 | 0.0 | 0.0 |  | 0.06 |
| bruno_mars_when_i_was_your_man_audio | 23.38–23.46 | 0.08 | C# | C:1.00, C5:0.00, Cmaj7:0.00 | 1.0 | 0.0 | 0.0 |  | 0.0 |
| celine_dion_because_you_loved_me_theme_from_up_close_and_personal_audio | 53.32–53.46 | 0.14 | D | F#sus2/C#sus4:1.00, F#5:0.00, C#5:0.00 | 1.0 | 0.0 | 0.0 |  | 0.0 |
| chris_stapleton_tennessee_whiskey_official_audio | 93.06–93.16 | 0.1 | A# | A:0.95, Amaj7:0.03, A5:0.01 | 1.0 | 0.0 | 0.0 |  | 0.118 |
| christina_aguilera_hurt_audio | 118.12–118.22 | 0.1 | A# | E5:0.61, Asus2/Esus4:0.39, Em:0.00 | 0.999 | 0.0 | 0.0 |  | 0.0 |

## low_evidence

| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |
|---|---|---|---|---|---|---|---|---|---|
| ariana_grande_into_you_official_video | 4.62–4.74 | 0.12 | D | F#5:0.96, F#sus2/C#sus4:0.02, F#maj7:0.01 | 0.0 | 0.615 | 0.615 |  | 0.045 |
| at_last | 13.52–13.64 | 0.12 | D | D5:0.55, D7:0.43, Csus2/Gsus4:0.02 | 0.0 | 0.749 | 0.749 |  | 1.0 |
| billiejean_mj_full_song | 30.68–30.92 | 0.24 | B | Amaj7:0.57, C#+/F+/A+:0.33, C#:0.06 | 0.0 | 0.667 | 0.667 |  | 0.0 |
| black_beatles | 4.78–4.88 | 0.1 | B | Asus2/Esus4:0.94, Cmaj7:0.06, Am:0.00 | 0.0 | 0.996 | 0.996 |  | 1.0 |
| chris_stapleton_tennessee_whiskey_official_audio | 252.5–252.56 | 0.06 | C | Asus2/Esus4:0.44, F#m7:0.44, Bm7:0.09 | 0.0 | 0.204 | 0.204 |  | 0.0 |
| christina_aguilera_hurt_audio | 39.6–39.66 | 0.06 | A# | E5:0.81, Esus2/Bsus4:0.10, Emaj7:0.08 | 0.0 | 0.351 | 0.667 | STEP_RESOLUTION | 0.0 |

## Songs with the lowest and highest mean H_inst (sung notes)

| song | group | est. key (p) | mean H sung | mean H alt | median top-1 mass | crosses share | sung CT share | sung OUT share | prod mean |
|---|---|---|---|---|---|---|---|---|---|
| at_last | axis2_50 | F major (0.585) | 0.620 | 0.545 | 0.84 | 0.03 | 0.33 | 0.17 | 0.530 |
| luther_vandross_here_and_now | axis2_50 | G major (0.763) | 0.621 | 0.524 | 0.92 | 0.06 | 0.40 | 0.20 | 0.618 |
| jennifer_hudson_spotlight | axis2_50 | B minor (0.462) | 0.625 | 0.523 | 0.93 | 0.08 | 0.37 | 0.16 | 0.595 |
| miguel_adorn_audio | axis2_50 | D major (0.523) | 0.631 | 0.508 | 0.98 | 0.06 | 0.38 | 0.14 | 0.581 |
| john_legend_ordinary_people_hq | axis2_50 | A# major (0.848) | 0.632 | 0.532 | 0.94 | 0.11 | 0.35 | 0.15 | 0.599 |
| demi_lovato_stone_cold_audio_only | axis2_50 | G# major (0.674) | 0.634 | 0.518 | 0.85 | 0.11 | 0.37 | 0.15 | 0.542 |
| miley_cyrus_party_in_the_u_s_a_audio | axis2_50 | D# minor (0.64) | 0.773 | 0.527 | 0.91 | 0.06 | 0.49 | 0.07 | 0.731 |
| the_weeknd_starboy_audio_ft_daft_punk | axis2_50 | A minor (0.688) | 0.775 | 0.523 | 0.84 | 0.12 | 0.54 | 0.04 | 0.781 |
| post_malone_swae_lee_sunflower_spider_man_into_the_spider_verse | excluded_13 | D major (0.786) | 0.791 | 0.514 | 0.89 | 0.06 | 0.47 | 0.02 | 0.798 |
| jingle_bells_with_lyrics_christmas_songs_hd_christmas_songs_and_carols | excluded_13 | G major (0.821) | 0.822 | 0.498 | 0.93 | 0.13 | 0.69 | 0.05 | 0.834 |

## Reason-code frequencies (sung notes)

CT 13060, KEY_FALLBACK 12819, EXT 10511, LOW_EVIDENCE 9253, SHORT 8036, TENSION 4803, OUT 3429, PC_AMBIGUOUS 2245, CROSSES_CHANGE 2193, STEP_RESOLUTION 2011, NEIGHBOR 1088, PASSING 623, ANTICIPATION 153, AMBIGUOUS_CHORD 149, SUSPENSION 24

