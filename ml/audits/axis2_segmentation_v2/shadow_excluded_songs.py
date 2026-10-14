#!/usr/bin/env python3
"""
Shadow evaluation of the recommended segmentation on the 13 songs the 50-song
audit did not cover (dataset_selections.json: axis2_ml == false).

The recommendation in segmentation_v2_summary.md rests on the 50 axis2_ml
songs. Regenerating notes.npz touches all 63, and the other 13 are the songs
excluded for rap/speech-like delivery, multiple leads, heavy effects, or weak
extraction, so they are exactly where segmentation behaviour could differ.

Same harness, same classifier, same metrics, same v1 mapping as the sweep
(evaluate_segmentation_v2) and the follow-up analysis (resume_analysis). The
pass criteria below were fixed, and committed, before this script was first
run; they are derived from the 50-song results with stated tolerances and are
not to be edited after seeing the 13-song numbers.

Writes only into shadow_excluded/:
  parameter_comparison_excluded.csv   sweep-style summary row per config
  per_song.csv                        v1 vs recommended, one row per song
  _verification.json                  v1 reproduction on these 13 songs
  _resume_analysis_<config>.json, tiny_notes_<config>.csv   (resume_analysis)
  _pass_criteria.json                 every criterion, its threshold, value, verdict

Usage: python shadow_excluded_songs.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import evaluate_segmentation_v2 as E
import followup_gap_reference as F
import resume_analysis as RA

RECOMMENDED = "gapv1local_g0.5_s100"
CONFIGS = ("v1", "v1+hardgap0.5", "gapv1_g0.5_s100", RECOMMENDED, "v2_g0.5_s70_p3")
OUT = Path(__file__).resolve().parent / "shadow_excluded"

# ---- pass criteria (fixed before the first run; 50-song value in brackets)
CRITERIA = {
    "C1_v1_reproduced_exactly": "v1 re-run on harmony.npz equals canonical notes.npz for every song [50/50]",
    "C2_no_gap_over_0.5s": "recommended notes with an internal gap > 0.5 s == 0 [0]",
    "C3_clear_invalid_relative_reduction_ge_75pct": "clear-invalid rate falls by >= 75% relative to v1 [92.4%]",
    "C4_no_song_clear_invalid_increase": "no song has more clear-invalid notes than under v1",
    "C5_likely_invalid_decreases": "likely-invalid rate is lower than v1's [20.4% -> 8.2%]",
    "C6_valid_notes_preserved_ge_90pct": ">= 90% of v1-valid notes unchanged (IoU >= 0.8) [94.4%]",
    "C7_adjacent_merge_not_worse": "adjacent-pitch merge rate (pct_multi_semitone) <= v1's [0.82% -> 0.42%]",
    "C8_vibrato_like_split_le_2pct": "<= 2% of v1 vibrato-like notes split; informational if < 20 such notes [0.76%]",
    "C9_same_pitch_tiny_le_3pct": "new same-pitch tiny notes (category F, < 0.10 s) <= 3% of notes [1.6%]",
    "C10_no_invalid_from_valid_le_0.1pct": "clear-invalid notes built only from v1-valid notes <= 0.1% of notes [0]",
    "C11_every_song_has_notes": "every song yields >= 1 recommended note",
}


def main() -> int:
    OUT.mkdir(exist_ok=True)
    sel = json.load(open(E.REPO / "data" / "dataset_selections.json"))["songs"]
    songs = sorted(s for s, v in sel.items() if not v["axis2_ml"])
    assert len(songs) == 13, len(songs)
    E.songs_axis2 = lambda: songs          # every E/RA loader now sees only these 13
    E._SONGS = E.load_songs()
    assert sorted(E._SONGS) == songs

    verification = {"songs": songs, "v1_reproduction": E.verify_v1(songs)}

    cfg_by_name = {c["name"]: c for c in E.build_configs() + F.configs()}
    v1_rows = [r for s in songs for r in E.eval_song(s, cfg_by_name["v1"])]
    E._V1_OWNER = {s: np.full(E._SONGS[s]["times"].size, -1) for s in songs}
    for r in v1_rows:
        E._V1_OWNER[r["song_id"]][r["source_frames"]] = r["position"]
    likely = lambda r: r["B_likely"] or r["C_likely"] or r["D_likely"]
    E._V1_VALID = {(r["song_id"], r["position"]) for r in v1_rows if not likely(r)}
    clean = lambda r: (not likely(r)) and r["largest_time_gap_s"] <= 0.10 + 1e-9 and r["frame_count"] >= 15
    # identical subset definitions to evaluate_segmentation_v2.main; the
    # stable-center examples all come from axis2_ml songs, so that set is empty
    E._SUBSETS = {
        "v1_plausible_A": {(r["song_id"], r["position"]) for r in v1_rows if r["categories"] == "A"},
        "v1_unflagged": {(r["song_id"], r["position"]) for r in v1_rows if not r["flagged"] and not likely(r)},
        "vibrato_like": {(r["song_id"], r["position"]) for r in v1_rows
                         if clean(r) and r["frame_count"] >= 20 and r["slow_component_range_cents"] < 100
                         and r["full_iqr_cents"] >= 40 and r["n_center_crossings"] >= 4},
        "scoop": {(r["song_id"], r["position"]) for r in v1_rows if clean(r) and abs(r["onset_minus_core_cents"]) >= 80},
        "fall": {(r["song_id"], r["position"]) for r in v1_rows if clean(r) and abs(r["release_minus_core_cents"]) >= 80},
        "portamento_drift": {(r["song_id"], r["position"]) for r in v1_rows
                             if clean(r) and 100 <= r["slow_component_range_cents"] < 200},
        "stable_center_examples": set(),
    }
    verification["subset_sizes"] = {k: len(v) for k, v in E._SUBSETS.items()}
    json.dump(verification, open(OUT / "_verification.json", "w"), indent=1)

    rows_by_cfg, summaries = {}, []
    for name in CONFIGS:
        rows = [r for s in songs for r in E.eval_song(s, cfg_by_name[name])]
        rows_by_cfg[name] = rows
        summaries.append(E.summarize(cfg_by_name[name], rows))
    n_v1 = summaries[0]["total_notes"]
    for S in summaries:
        S["note_count_change_pct"] = 100 * (S["total_notes"] - n_v1) / n_v1
    with (OUT / "parameter_comparison_excluded.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        for S in summaries:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in S.items()})

    per_song = []
    for s in songs:
        a = [r for r in rows_by_cfg["v1"] if r["song_id"] == s]
        b = [r for r in rows_by_cfg[RECOMMENDED] if r["song_id"] == s]
        per_song.append(dict(
            song_id=s, exclusion_reason=sel[s].get("axis2_exclusion_reason", ""),
            v1_notes=len(a), v2_notes=len(b), note_change_pct=100 * (len(b) - len(a)) / len(a),
            v1_clear=sum(r["any_clear_invalid"] for r in a), v2_clear=sum(r["any_clear_invalid"] for r in b),
            v1_likely=sum(bool(likely(r)) for r in a), v2_likely=sum(bool(likely(r)) for r in b),
            v1_gap_gt0_5=sum(r["largest_time_gap_s"] > 0.5 + 1e-9 for r in a),
            v2_gap_gt0_5=sum(r["largest_time_gap_s"] > 0.5 + 1e-9 for r in b),
            v1_max_gap_s=max(r["largest_time_gap_s"] for r in a), v2_max_gap_s=max(r["largest_time_gap_s"] for r in b),
            v1_iqr_gt400=sum(r["full_iqr_cents"] > 400 for r in a), v2_iqr_gt400=sum(r["full_iqr_cents"] > 400 for r in b),
            v1_tiny_lt0_10=sum(r["duration_s"] < 0.10 - 1e-9 for r in a),
            v2_tiny_lt0_10=sum(r["duration_s"] < 0.10 - 1e-9 for r in b),
        ))
    with (OUT / "per_song.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_song[0].keys()))
        w.writeheader()
        for r in per_song:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    # follow-up analysis (fix accounting, tiny-note mechanisms, vibrato chains)
    E.OUT = OUT
    RA.HYBRID = RECOMMENDED
    RA.main()
    ra = json.load(open(OUT / f"_resume_analysis_{RECOMMENDED}.json"))

    S1 = summaries[0]
    SR = next(S for S in summaries if S["name"] == RECOMMENDED)
    vib = ra["vibrato"]["hybrid_splits_of_v1_vibrato_like"]
    vib_total = vib.get("vibrato_like_total", 0)
    vib_split = vib.get("vibrato_like_split", 0)
    f_tiny = ra["tiny_notes"]["lt0.1"]["categories"].get("F_same_pitch_as_neighbor", [0, 0.0])[0]
    from_valid = ra["fix_accounting"].get("hybrid_clear_from_only_valid_v1", 0)
    rel = 100 * (S1["pct_clear_invalid"] - SR["pct_clear_invalid"]) / S1["pct_clear_invalid"] if S1["pct_clear_invalid"] else float("nan")
    results = {
        "C1_v1_reproduced_exactly": (not verification["v1_reproduction"]["mismatching_songs"],
                                     verification["v1_reproduction"]),
        "C2_no_gap_over_0.5s": (SR["notes_gap_gt0.5"] == 0, SR["notes_gap_gt0.5"]),
        "C3_clear_invalid_relative_reduction_ge_75pct": (rel >= 75.0, {"v1_pct": S1["pct_clear_invalid"],
                                                                        "v2_pct": SR["pct_clear_invalid"],
                                                                        "relative_reduction_pct": rel}),
        "C4_no_song_clear_invalid_increase": (all(r["v2_clear"] <= r["v1_clear"] for r in per_song),
                                              [r["song_id"] for r in per_song if r["v2_clear"] > r["v1_clear"]]),
        "C5_likely_invalid_decreases": (SR["pct_likely_invalid_total"] < S1["pct_likely_invalid_total"],
                                        {"v1_pct": S1["pct_likely_invalid_total"], "v2_pct": SR["pct_likely_invalid_total"]}),
        "C6_valid_notes_preserved_ge_90pct": (SR["pct_v1valid_same"] >= 90.0, SR["pct_v1valid_same"]),
        "C7_adjacent_merge_not_worse": (SR["pct_multi_semitone"] <= S1["pct_multi_semitone"],
                                        {"v1_pct": S1["pct_multi_semitone"], "v2_pct": SR["pct_multi_semitone"]}),
        "C8_vibrato_like_split_le_2pct": ((vib_total < 20) or (100 * vib_split / vib_total <= 2.0),
                                          {"vibrato_like_total": vib_total, "split": vib_split,
                                           "pct": 100 * vib_split / vib_total if vib_total else None,
                                           "informational_only": vib_total < 20}),
        "C9_same_pitch_tiny_le_3pct": (100 * f_tiny / SR["total_notes"] <= 3.0,
                                       {"n": f_tiny, "pct_of_notes": 100 * f_tiny / SR["total_notes"]}),
        "C10_no_invalid_from_valid_le_0.1pct": (100 * from_valid / SR["total_notes"] <= 0.1,
                                                {"n": from_valid, "pct_of_notes": 100 * from_valid / SR["total_notes"]}),
        "C11_every_song_has_notes": (all(r["v2_notes"] >= 1 for r in per_song),
                                     {r["song_id"]: r["v2_notes"] for r in per_song}),
    }
    report = {"criteria": {k: {"rule": CRITERIA[k], "passed": bool(v[0]), "value": v[1]} for k, v in results.items()}}
    report["all_passed"] = all(v["passed"] for v in report["criteria"].values())
    json.dump(report, open(OUT / "_pass_criteria.json", "w"), indent=1, default=float)
    for k, v in report["criteria"].items():
        print(f"{'PASS' if v['passed'] else 'FAIL'}  {k}")
    print("ALL PASSED" if report["all_passed"] else "NOT ALL PASSED")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
