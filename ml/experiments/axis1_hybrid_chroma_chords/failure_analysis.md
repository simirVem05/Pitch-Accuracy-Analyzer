# Failure analysis and real-song examples

Generated from `results.json`. One example per song per category, chosen by the most extreme value. *elig* lists D's eligible chords with normalized excess-fit weights. *C* is the direct chroma support of the sung pitch class, and *ref tier* is its tier under D's top eligible chord (−1 = none).

## largest_D_vs_A

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| work_rihanna_full_song | 173.26–173.68 | 0.42 | F# | CHORD | F#maj7:1.00 | 0.823 | 0.157 | 0.006 | 0.982 | 0.925 | 0 | 0.309 | 0.983 | 0.724 | 0.951 | 0.951 | 1.000 |
| bruno_mars_when_i_was_your_man_audio | 168.52–168.72 | 0.2 | E | CHORD | G:1.00 | 0.939 | 0.189 | 0.309 | 0.93 | 0.0 | 1 | 0.000 | 0.645 | 0.300 | 0.334 | 0.334 | 0.000 |
| celine_dion_because_you_loved_me_theme_from_up_close_and_personal_audio | 6.88–7.14 | 0.26 | A# | CHORD | C#:1.00 | 0.983 | 0.19 | 0.457 | 1.0 | 0.0 | 1 | 0.000 | 0.667 | 0.333 | 0.333 | 0.333 | 0.000 |
| chris_stapleton_tennessee_whiskey_official_audio | 220.92–221.16 | 0.24 | E | CHORD | Bm:1.00 | 0.937 | 0.172 | 0.308 | 0.928 | 0.0 | 1 | 0.000 | 0.667 | 0.333 | 0.333 | 0.333 | 0.000 |
| christina_aguilera_hurt_audio | 44.9–45.14 | 0.24 | G | CHORD | E5:1.00 | 0.964 | 0.193 | 0.316 | 1.0 | 0.0 | 1 | 0.000 | 0.667 | 0.267 | 0.333 | 0.333 | 0.000 |

## largest_D_vs_C5u

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| midas_the_jagaban_party_with_a_jagaban_lyrics | 72.86–73.18 | 0.32 | F | INSUFFICIENT | — | 0.618 | -0.086 | 0.014 | 0.972 | 0.99 | -1 | 0.223 | 0.968 | 0.882 | 0.223 | 0.223 | 1.000 |
| sheila_ki_jawani_audio_song_tees_maar_khan_katrina_kaif_vishal_dadlani_sunidhi_chauhan | 254.9–255.1 | 0.2 | A | INSUFFICIENT | — | 0.704 | 0.0 | 0.014 | 0.724 | 0.693 | -1 | 0.066 | 0.721 | 0.709 | 0.066 | 0.066 | 0.284 |
| michael_jackson_chicago_official_audio | 4.38–4.9 | 0.52 | A | INSUFFICIENT | — | 0.691 | -0.026 | 0.009 | 0.763 | 1.0 | -1 | 0.469 | 1.000 | 0.993 | 0.469 | 0.469 | 1.000 |
| work_rihanna_full_song | 132.88–133.2 | 0.32 | F# | INSUFFICIENT | — | 0.777 | 0.059 | 0.008 | 0.99 | 1.0 | -1 | 0.405 | 0.959 | 0.845 | 0.405 | 0.405 | 1.000 |
| ckay_love_nwantinti_official_audio | 36.96–37.36 | 0.4 | F | CHROMA_ONLY | — | 0.649 | -0.139 | 0.227 | 0.508 | 0.103 | -1 | 0.103 | 0.831 | 0.518 | 0.103 | 0.103 | 1.000 |

## largest_D_vs_B

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| justin_bieber_baby_ft_ludacris | 158.6–158.96 | 0.36 | D# | CHROMA_ONLY | — | 0.472 | -0.265 | 0.033 | 0.996 | 1.0 | -1 | 0.993 | 0.004 | 0.877 | 0.993 | 0.993 | 1.000 |
| work_rihanna_full_song | 75.8–76.32 | 0.52 | D# | CHROMA_ONLY | — | 0.431 | -0.261 | 0.018 | 0.989 | 1.0 | -1 | 1.000 | 0.019 | 0.678 | 1.000 | 1.000 | 1.000 |
| sheila_ki_jawani_audio_song_tees_maar_khan_katrina_kaif_vishal_dadlani_sunidhi_chauhan | 144.62–144.88 | 0.26 | C | CHROMA_ONLY | — | 0.609 | -0.061 | 0.09 | 1.0 | 0.96 | -1 | 0.960 | 0.000 | 0.480 | 0.960 | 0.960 | 1.000 |
| post_malone_swae_lee_sunflower_spider_man_into_the_spider_verse | 101.4–101.7 | 0.3 | D | CHROMA_ONLY | — | 0.69 | -0.047 | 0.357 | 1.0 | 0.034 | -1 | 0.048 | 1.000 | 0.407 | 0.048 | 0.048 | 0.732 |
| ariana_grande_into_you_official_video | 109.62–109.98 | 0.36 | A | CHROMA_ONLY | — | 0.442 | -0.326 | 0.187 | 0.992 | 0.108 | -1 | 0.108 | 0.997 | 0.454 | 0.108 | 0.108 | 0.964 |

## chord_tone_low_chroma

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jeff_buckley_hallelujah | 192.46–193.0 | 0.54 | E | CHORD | Am:1.00 | 0.886 | 0.158 | 0.599 | 0.998 | 0.315 | 0 | 0.261 | 0.998 | 0.563 | 0.465 | 0.465 | 1.000 |
| kacey_musgraves_rainbow_official_audio | 208.6–209.7 | 1.1 | D# | CHORD | D#5:1.00 | 0.842 | 0.186 | 0.592 | 0.846 | 0.332 | 0 | 0.302 | 0.954 | 0.586 | 0.635 | 0.635 | 1.000 |
| work_rihanna_full_song | 173.26–173.68 | 0.42 | F# | CHORD | F#maj7:1.00 | 0.823 | 0.157 | 0.006 | 0.982 | 0.925 | 0 | 0.309 | 0.983 | 0.724 | 0.951 | 0.951 | 1.000 |
| no_ordinary_love | 72.68–73.64 | 0.96 | C# | CHORD | F#m7:1.00 | 0.858 | 0.158 | 0.256 | 0.99 | 0.328 | 0 | 0.351 | 0.886 | 0.532 | 0.514 | 0.514 | 1.000 |
| post_malone_swae_lee_sunflower_spider_man_into_the_spider_verse | 113.84–114.64 | 0.8 | F# | CHORD | D:0.51, D5:0.49 | 0.898 | 0.155 | 0.385 | 0.54 | 0.403 | 0 | 0.362 | 0.781 | 0.609 | 0.494 | 0.494 | 1.000 |

## extension

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jeff_buckley_hallelujah | 396.44–402.28 | 5.84 | E | CHORD | F:0.56, Fmaj7:0.44 | 0.965 | 0.2 | 0.305 | 0.886 | 0.373 | 1 | 0.227 | 0.493 | 0.399 | 0.330 | 0.331 | 0.494 |
| lately | 230.84–235.32 | 4.48 | F# | CHORD | D#5:1.00 | 0.957 | 0.192 | 0.392 | 1.0 | 0.0 | 1 | 0.182 | 0.748 | 0.400 | 0.282 | 0.282 | 0.000 |
| no_ordinary_love | 82.58–86.02 | 3.44 | C# | CHORD | Gmaj7:1.00 | 0.974 | 0.235 | 0.423 | 1.0 | 0.061 | 1 | 0.402 | 0.809 | 0.519 | 0.535 | 0.535 | 0.808 |
| sam_smith_lay_me_down_lyric_video | 184.1–187.16 | 3.06 | E | CHORD | Dmaj7:1.00 | 0.963 | 0.213 | 0.103 | 0.988 | 0.115 | 1 | 0.276 | 0.702 | 0.504 | 0.373 | 0.373 | 0.224 |
| adele_hello | 7.08–9.94 | 2.86 | A# | CHORD | G#:1.00 | 0.96 | 0.204 | 0.238 | 0.98 | 0.245 | 1 | 0.261 | 0.674 | 0.493 | 0.444 | 0.444 | 0.604 |

## short_note

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| adele_hello | 34.68–34.74 | 0.06 | B | CHORD | D#:1.00 | 0.939 | 0.176 | 0.247 | 0.997 | 0.0 | 3 | 0.000 | 0.001 | 0.067 | 0.000 | 0.000 | 0.000 |
| amy_winehouse_love_is_a_losing_game_audio | 41.62–41.68 | 0.06 | C# | CHORD | Cmaj7:1.00 | 0.923 | 0.153 | 0.269 | 0.876 | 0.0 | 3 | 0.000 | 0.000 | 0.033 | 0.000 | 0.000 | 0.000 |
| at_last | 53.06–53.12 | 0.06 | G# | CHROMA_ONLY | — | 0.588 | -0.127 | 0.187 | 0.839 | 0.0 | -1 | 0.000 | 0.287 | 0.083 | 0.000 | 0.000 | 0.000 |
| black_beatles | 14.1–14.18 | 0.08 | A# | CHROMA_ONLY | — | 0.77 | 0.046 | 0.153 | 0.83 | 0.0 | -1 | 0.000 | 0.332 | 0.083 | 0.000 | 0.000 | 0.000 |
| bruno_mars_when_i_was_your_man_audio | 8.36–8.4 | 0.04 | B | CHROMA_ONLY | — | 0.814 | 0.047 | 0.19 | 0.753 | 0.0 | -1 | 0.000 | 0.397 | 0.267 | 0.000 | 0.000 | 0.000 |

## chord_transition

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jeff_buckley_hallelujah | 371.62–386.78 | 15.16 | E | CHROMA_ONLY | — | 0.806 | 0.094 | 0.356 | 0.974 | 0.33 | -1 | 0.238 | 0.638 | 0.440 | 0.320 | 0.320 | 0.579 |
| you_raise_me_up | 19.68–27.08 | 7.4 | F | CHROMA_ONLY | — | 0.916 | 0.114 | 0.385 | 0.997 | 1.0 | -1 | 0.973 | 1.000 | 0.977 | 0.986 | 0.986 | 1.000 |
| teri_ore_full_audio_singh_is_kinng_akshay_kkatrina_kpritamrahat_fateh_ali_khanshreya_ghoshal | 6.78–12.02 | 5.24 | D | CHROMA_ONLY | — | 0.898 | 0.113 | 0.14 | 0.688 | 1.0 | -1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| floyymenor_gata_only_ft_cris_mj_letralyrics | 197.0–201.3 | 4.3 | C# | CHROMA_ONLY | — | 0.849 | 0.095 | 0.144 | 0.839 | 0.389 | -1 | 0.494 | 0.804 | 0.649 | 0.611 | 0.611 | 0.961 |
| lady_gaga_million_reasons_official_audio | 105.56–109.48 | 3.92 | E | CHORD | C:1.00 | 0.963 | 0.204 | 0.227 | 0.997 | 0.724 | 0 | 0.774 | 0.905 | 0.844 | 0.821 | 0.821 | 1.000 |

## ambiguous_many_chords

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amy_winehouse_love_is_a_losing_game_audio | 55.38–55.62 | 0.24 | C | CHORD | Em:0.38, Cmaj7:0.35, Em7:0.28 | 0.922 | 0.207 | 0.237 | 0.688 | 0.466 | 2 | 0.489 | 0.570 | 0.451 | 0.529 | 0.526 | 1.000 |
| florence_the_machine_never_let_me_go | 160.24–160.7 | 0.46 | F# | CHORD | F#m:0.39, Dmaj7:0.35, F#m7:0.27 | 0.914 | 0.221 | 0.161 | 0.758 | 0.965 | 0 | 0.938 | 1.000 | 0.969 | 0.969 | 0.969 | 1.000 |
| if_i_ain_t_got_you_alicia_keys_hq_audio | 124.32–124.54 | 0.22 | D | CHORD | Bm7:0.34, Dmaj7:0.33, D:0.32 | 0.885 | 0.167 | 0.16 | 0.407 | 0.817 | 0 | 0.817 | 1.000 | 0.842 | 0.909 | 0.909 | 1.000 |
| mark_ronson_uptown_funk_official_audio_ft_bruno_mars | 229.04–229.3 | 0.26 | G | CHORD | Dm7:0.34, F7:0.33, F:0.33 | 0.873 | 0.197 | 0.019 | 0.363 | 0.469 | 1 | 0.469 | 0.666 | 0.560 | 0.568 | 0.568 | 0.000 |
| midas_the_jagaban_party_with_a_jagaban_lyrics | 168.22–168.58 | 0.36 | C | CHORD | A#maj7:0.36, A#:0.34, Gm7:0.30 | 0.865 | 0.196 | 0.117 | 0.527 | 0.487 | 1 | 0.565 | 0.718 | 0.636 | 0.641 | 0.641 | 1.000 |

## chroma_only_fallback

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jeff_buckley_hallelujah | 371.62–386.78 | 15.16 | E | CHROMA_ONLY | — | 0.806 | 0.094 | 0.356 | 0.974 | 0.33 | -1 | 0.238 | 0.638 | 0.440 | 0.320 | 0.320 | 0.579 |
| you_raise_me_up | 19.68–27.08 | 7.4 | F | CHROMA_ONLY | — | 0.916 | 0.114 | 0.385 | 0.997 | 1.0 | -1 | 0.973 | 1.000 | 0.977 | 0.986 | 0.986 | 1.000 |
| romeo_santos_propuesta_indecente_audio | 235.42–241.28 | 5.86 | F | CHROMA_ONLY | — | 0.585 | -0.145 | 0.649 | 0.638 | 1.0 | -1 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 0.891 |
| love_of_my_life | 142.78–148.36 | 5.58 | C | CHROMA_ONLY | — | 0.845 | 0.101 | 0.272 | 0.864 | 1.0 | -1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| teri_ore_full_audio_singh_is_kinng_akshay_kkatrina_kpritamrahat_fateh_ali_khanshreya_ghoshal | 6.78–12.02 | 5.24 | D | CHROMA_ONLY | — | 0.898 | 0.113 | 0.14 | 0.688 | 1.0 | -1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## insufficient

| song | t (s) | dur | pc | D status | elig | r_best | c_best | I_c | post top-1 | C | ref tier | A | B | C5u | D | D2 | prod |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| michael_jackson_chicago_official_audio | 117.3–120.58 | 3.28 | A | INSUFFICIENT | — | 0.546 | -0.176 | 0.013 | 0.539 | 0.667 | -1 | 0.421 | 0.848 | 0.728 | 0.421 | 0.421 | 1.000 |
| mark_ronson_uptown_funk_official_audio_ft_bruno_mars | 25.66–28.56 | 2.9 | C | INSUFFICIENT | — | 0.729 | -0.043 | 0.01 | 1.0 | 1.0 | -1 | — | 1.000 | 0.972 | — | — | 1.000 |
| miley_cyrus_party_in_the_u_s_a_audio | 134.36–136.98 | 2.62 | F# | INSUFFICIENT | — | 0.856 | 0.074 | 0.012 | 0.974 | 0.474 | -1 | 0.445 | 0.384 | 0.544 | 0.445 | 0.445 | 0.575 |
| gotye_somebody_that_i_used_to_know_2015_remaster | 60.4–62.66 | 2.26 | D | INSUFFICIENT | — | 0.842 | 0.089 | 0.012 | 0.999 | 0.644 | -1 | 1.000 | 0.807 | 0.828 | 1.000 | 1.000 | 1.000 |
| ne_yo_so_sick | 199.34–201.12 | 1.78 | F# | INSUFFICIENT | — | 0.647 | -0.012 | 0.005 | 0.895 | 1.0 | -1 | 1.000 | 0.962 | 0.865 | 1.000 | 1.000 | 1.000 |

