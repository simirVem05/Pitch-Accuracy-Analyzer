"""Generated markdown reports for axis1_chord_compatibility. Numbers only come from results.json; interpretation lives in summary.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

METHOD_ORDER = ["prod_calibrated", "prod_harmonic_rank", "raw_chroma", "rank_chroma", "chord_hard", "chord_dist", "chord_dist_key",
                "chord_dist_fixed_key", "chord_dist_key_alt_tiers", "chord_dist_key_ctx"]
METHOD_DESC = {
    "prod_calibrated": "production key_compliance (rank local+slack ^0.75 · global ^0.25 → clip)",
    "prod_harmonic_rank": "production value before calibration",
    "raw_chroma": "window-mean CENS / max",
    "rank_chroma": "window-mean per-frame rank",
    "chord_hard": "tier under the single most likely chord",
    "chord_dist": "expected tier over the chord distribution",
    "chord_dist_key": "chord_dist, evidence-gated with key context (**H_inst**)",
    "chord_dist_fixed_key": "chord_dist with a fixed 0.75/0.25 key blend",
    "chord_dist_key_alt_tiers": "H_inst with tier values (1, .75, .4, .1)",
    "chord_dist_key_ctx": "H_inst plus melodic/temporal context (**H_ctx**)",
}


def f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def method_table(M: Dict[str, Dict[str, float]], order: List[str]) -> List[str]:
    L = ["| method | pcs ≥ 0.75 | pcs ≥ 0.5 | notes with ≥10 pcs ≥ 0.5 | notes with no pc ≥ 0.75 | max−min | values in (0.25, 0.75) | mean H sung | mean H alt | sung − alt | win rate | P4↑ gap | P5↑ gap | m2↑ gap |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in order:
        if m not in M:
            continue
        e = M[m]
        L.append(f"| {m} | {f(e['breadth_ge_0.75'], 2)} | {f(e['breadth_ge_0.5'], 2)} | {f(e['share_notes_ge10_pcs_ge_0.5'])} | {f(e['share_notes_no_pc_ge_0.75'])} | "
                 f"{f(e['spread_max_minus_min'])} | {f(e['graded_share_in_0.25_0.75'])} | {f(e['mean_H_sung'])} | {f(e['mean_H_alternatives'])} | "
                 f"{f(e['sung_minus_alt'])} | {f(e['win_rate_sung_vs_alt'])} | {f(e['P4_up_gap'])} | {f(e['P5_up_gap'])} | {f(e['semitone_up_gap'])} |")
    return L


def write_reports(R: Dict[str, Any], per_song: List[Dict[str, Any]], out: Path) -> None:
    S = R["stage4_real"]
    # ------------------------------------------------ method_comparison.md
    L = ["# Method comparison — all 63 songs, 31,803 notes", "",
         "Generated from `results.json`. **In-corpus method development; not evidence of generalization.** All values are label-free. "
         "A higher \"mean H sung\" or win rate does **not** mean a method is more correct, since the alternatives are not known to be wrong. "
         "The objective metrics are breadth (several pitch classes can fit), specificity (not everything fits), grading and stability.", "",
         "Column notes: *pcs ≥ x* is the mean number of the 12 pitch classes scoring at least x per note. *Values in (0.25, 0.75)* is the share of all 12×N values that are intermediate, i.e. graded rather than all-or-nothing. "
         "*Gap* is the sung note's H minus the H of the same note shifted up by that interval.", "",
         "## Main methods", ""]
    L += method_table(S["methods_all_63"], METHOD_ORDER)
    L += ["", "Method definitions:", ""] + [f"- `{k}`: {v}" for k, v in METHOD_DESC.items()]
    L += ["", "## Ablations (each replaces one component of `chord_dist_key`; no context)", ""]
    abl = sorted(k for k in S["methods_all_63"] if k.startswith("abl_"))
    L += method_table(S["methods_all_63"], ["chord_dist_key"] + abl)
    L += ["", "## Stability", "",
          "Mean and p95 of |ΔH| over all 12 values per note, under (a) multiplicative CENS noise (lognormal σ = 0.10, re-normalized) and (b) ±50 ms note shifts.", "",
          "| method | noise mean | noise p95 | shift mean | shift p95 |", "|---|---|---|---|---|"]
    for m in ("chord_dist_key", "prod_calibrated", "raw_chroma", "rank_chroma"):
        n, s = S["stability"]["noise"][m], S["stability"]["shift"][m]
        L.append(f"| {m} | {f(n['mean_abs_dH_all12'], 4)} | {f(n['p95_abs_dH'], 4)} | {f(s['mean_abs_dH_all12'], 4)} | {f(s['p95_abs_dH'], 4)} |")
    L += ["", "## By song group (Axis 2 pool vs the 13 excluded)", "", "| group | method | mean H sung | mean H alt | pcs ≥ 0.75 | pcs ≥ 0.5 | win rate |", "|---|---|---|---|---|---|---|"]
    for g, d in S["by_group"].items():
        for m, e in d.items():
            L.append(f"| {g} | {m} | {f(e['mean_H_sung'])} | {f(e['mean_H_alternatives'])} | {f(e['breadth_ge_0.75'], 2)} | {f(e['breadth_ge_0.5'], 2)} | {f(e['win_rate_sung_vs_alt'])} |")
    L += ["", "## Chord-sequence behavior on real songs, by temporal estimator (no labels)", "",
          "| estimator | switches/beat | flicker/beat | N share | diatonic share | beat-grid +50 ms agreement |", "|---|---|---|---|---|---|"]
    for tp, d in S["chord_sequence_diagnostics"].items():
        g = lambda k: f(d[k]["mean"]) if k in d else "—"  # noqa: E731
        L.append(f"| {tp} | {g('switch_per_beat')} | {g('flicker_per_beat')} | {g('N_share')} | {g('diatonic_share')} | {g('beat_shift_50ms_agreement')} |")
    sel = R["selected_config"]["temporal"]
    q = S["chord_sequence_diagnostics"][sel]["quality_share_mean_over_songs"]
    L += ["", f"Quality share of beat-level top chords (`{sel}`, mean over songs): " + ", ".join(f"{k} {v:.1%}" for k, v in q.items()), ""]
    L += ["## Selected method by note bucket (sung note)", "", "| variable | bucket | notes | H_inst | H_ctx | production |", "|---|---|---|---|---|---|"]
    for var, d in S["buckets_selected_method"].items():
        for b, e in d.items():
            L.append(f"| {var} | {b} | {e['notes']} | {f(e['mean_H_inst_sung'])} | {f(e['mean_H_ctx_sung'])} | {f(e['mean_prod_sung'])} |")
    c = S["context_specificity"]
    L += ["", "## Context specificity", "",
          f"Share of notes whose score context raises: **sung pitch class {c['sung_lift_rate']:.3f}**, against **alternative pitch classes {c['alternative_lift_rate']:.3f}** (same rules, same neighbors).", "",
          "| code | sung notes | per alternative (count/11) |", "|---|---|---|"]
    for k in sorted(set(c["by_code_sung"]) | set(c["by_code_alternatives_per_alt"])):
        L.append(f"| {k} | {c['by_code_sung'].get(k, 0)} | {c['by_code_alternatives_per_alt'].get(k, 0):.0f} |")
    rt = S["runtime"]
    L += ["", "## Runtime", "", f"Selected pipeline (posterior, smoothing, 5 note windows, context): mean {rt['selected_pipeline_seconds_per_song']['mean']:.2f} s per song, "
          f"max {rt['selected_pipeline_seconds_per_song']['max']:.2f} s; {rt['seconds_per_minute_of_audio']:.3f} s per minute of audio on this machine, "
          "excluding Demucs and CREPE.", ""]
    (out / "method_comparison.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------ synthetic_tests.md
    syn = R["stage3_synthetic"]
    cal, tmp = R["stage1_calibration"], R["stage2_temporal"]
    L = ["# Synthetic tests", "",
         "Generated from `results.json`. The intended behavior of every case was fixed in `protocol.md` before any run.", "",
         "- **Layer T (template level):** idealized chroma (harmonic-partial chord chroma) injected directly. This tests the scoring rules only.",
         "- **Layer S (audio level):** synthetic audio (summed harmonic partials) → the exact production `chroma_cens` call → the full pipeline. This also tests harmony estimation, but on timbres far simpler than a real `other` stem.",
         "", "Passing these checks shows that the rules behave as specified. It does **not** show agreement with human perception.", "",
         f"**Passed: layer T {syn['passed']['T']}, layer S {syn['passed']['S']}.**", "",
         "| layer | case | check | pass | detail (this method) | production (reference) |", "|---|---|---|---|---|---|"]
    for c in syn["checks"]:
        L.append(f"| {c['layer']} | {c['case']} | {c['check']} | {'PASS' if c['pass'] else '**FAIL**'} | {c['detail']} | {c['production_reference'] or ''} |")
    L += ["", "## Chord identifiability (synthetic sets A/B)", "",
          f"Frames: A {cal['frames']['A']}, B clean {cal['frames']['B_clean']}, B degraded {cal['frames']['B_degraded']}. "
          "\"Degraded\" adds one non-chord tone at half the level of a chord tone, plus white noise at 15 dB SNR.", "",
          "| template/similarity | τ | θ | log-loss A | top-1 clean | top-3 clean | top-1 degraded | top-3 degraded | N recall | false N |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, e in cal["candidates"].items():
        L.append(f"| {k}{' **(selected)**' if k == cal['selected']['template'] + '/' + cal['selected']['sim'] else ''} | {e['tau']} | {e['theta']} | {f(e['logloss_A'])} | "
                 f"{f(e['clean']['top1_chord'])} | {f(e['clean']['top3_chord'])} | {f(e['degraded']['top1_chord'])} | {f(e['degraded']['top3_chord'])} | "
                 f"{f(e['clean']['N_recall'])} | {f(e['clean']['false_N_on_chords'])} |")
    ss = cal["candidates"][cal["selected"]["template"] + "/" + cal["selected"]["sim"]]
    L += ["", "Per-quality top-1 (selected estimator, single frames, no smoothing):", "",
          "| quality | clean top-1 | clean top-3 | degraded top-1 | degraded top-3 | most common confusions (degraded) |", "|---|---|---|---|---|---|"]
    for q in ss["clean"]["per_quality"]:
        c_, d_ = ss["clean"]["per_quality"][q], ss["degraded"]["per_quality"][q]
        L.append(f"| {q} | {f(c_['top1'])} | {f(c_['top3'])} | {f(d_['top1'])} | {f(d_['top3'])} | {d_['confused_with']} |")
    L += ["", "## Temporal estimators on synthetic progressions (set C)", "",
          f"{tmp['progressions']} progressions, 32 beats each at 80–130 BPM. Chords last 1, 2 or 4 beats; diatonic, with some 7ths and sus4.", "",
          "| estimator | chord length | frame acc | interior acc | switches/beat (true) | flicker/beat | boundary recall (±½ beat) |", "|---|---|---|---|---|---|---|"]
    for tp, e in tmp["estimators"].items():
        for d in (1, 2, 4):
            x = e[f"{d}_beat_chords"]
            L.append(f"| {tp}{' **(selected)**' if tp == tmp['selected'] else ''} | {d} beat | {f(x['acc'])} | {f(x['acc_interior'])} | "
                     f"{f(x['switch_per_beat'])} ({f(x['true_switch_per_beat'])}) | {f(x['flicker'])} | {f(x['boundary_recall'])} |")
    L += ["", "Accuracy by distance to the nearest true chord change (all durations):", "", "| estimator | <0.1 s | 0.1–0.25 s | 0.25–0.5 s | >0.5 s |", "|---|---|---|---|---|"]
    for tp, e in tmp["estimators"].items():
        a = e["acc_by_distance_to_true_boundary"]
        L.append(f"| {tp} | {f(a['<0.1s'])} | {f(a['0.1-0.25s'])} | {f(a['0.25-0.5s'])} | {f(a['>0.5s'])} |")
    k = tmp["key"]
    L += ["", f"Key estimation (Krumhansl–Kessler, τ_key = {k['tau_key']}): MAP accuracy {k['map_accuracy']:.3f}. "
          f"Of {k['errors_total']} errors, {k['errors_relative_major_minor']} were relative major/minor and {k['errors_fourth_fifth']} were a fourth or fifth away.", ""]
    (out / "synthetic_tests.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ------------------------------------------------ failure_analysis.md
    L = ["# Failure analysis — real songs", "",
         "Generated from `results.json` (selected method `chord_dist_key` / `chord_dist_key_ctx`). Counts are over 31,803 sung notes. "
         "Up to six examples per category are shown, one per song, chosen by the most extreme value. "
         "\"Top chords\" are the estimated harmony of the note window with renormalized probabilities. "
         "A failure category is a *symptom* to inspect; it is not proof the model is wrong.", "",
         "| category | definition | notes |", "|---|---|---|"]
    defs = {"chord_tone_low": "the sung pitch class is a chord tone of the top chord, but H_inst < 0.5",
            "breadth_ge10": "≥ 10 of the 12 pitch classes have H ≥ 0.5 (nearly everything fits)",
            "extension_low": "the sung pitch class is a T2 extension of the top chord, but H_inst < 1/3",
            "noise_unstable": "|ΔH_inst| of the sung pitch class ≥ 0.25 under CENS noise σ = 0.1",
            "crosses_change": "the top chord differs between the note's first and second halves",
            "short_low": "duration < 0.15 s and H_ctx < 1/3",
            "low_evidence": "evidence e < 0.5 (would abstain)"}
    for kk, d in defs.items():
        L.append(f"| {kk} | {d} | {S['failure_counts'][kk]} ({S['failure_counts'][kk] / R['verification']['notes']:.1%}) |")
    for kk, ex in S["failure_examples"].items():
        L += ["", f"## {kk}", "", "| song | t (s) | dur | pc | top chords | e | H_inst | H_ctx | code | prod |", "|---|---|---|---|---|---|---|---|---|---|"]
        for e in ex:
            L.append(f"| {e['song']} | {e['start_s']}–{e['end_s']} | {e['duration_s']} | {e['pc']} | {e['top_chords']} | {e['evidence']} | "
                     f"{e['H_inst']} | {e['H_ctx']} | {e['code']} | {e['prod_calibrated']} |")
    ps = sorted(per_song, key=lambda r: r["mean_H_sung"])
    L += ["", "## Songs with the lowest and highest mean H_inst (sung notes)", "",
          "| song | group | est. key (p) | mean H sung | mean H alt | median top-1 mass | crosses share | sung CT share | sung OUT share | prod mean |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in ps[:6] + ps[-4:]:
        L.append(f"| {r['song_id']} | {r['group']} | {r['key']} ({r['key_p']}) | {r['mean_H_sung']:.3f} | {r['mean_H_alt']:.3f} | {r['median_top1_mass']:.2f} | "
                 f"{r['crosses_share']:.2f} | {r['share_sung_CT']:.2f} | {r['share_sung_OUT']:.2f} | {r['prod_mean_sung']:.3f} |")
    L += ["", "## Reason-code frequencies (sung notes)", "", ", ".join(f"{k} {v}" for k, v in sorted(S["reason_code_counts"].items(), key=lambda kv: -kv[1])), ""]
    (out / "failure_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")
