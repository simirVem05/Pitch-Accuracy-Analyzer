#!/usr/bin/env python3
"""
Per-note detail for one v2 configuration (by name from build_configs()):

  v1_vs_v2_notes.csv          one row per v1 note: outcome under v2 and the
                              overlapping v2 notes
  v2_notes.csv                one row per v2 note: metrics + audit categories
  representative_examples.csv v1 and v2 segments for Stapleton #319, Starboy #2,
                              the tail audit's 64 extreme examples, and the
                              stable-center audit's example notes
  _detail_<config>.json       target-estimator comparison and headline traces

Usage: python detail_outputs.py <config_name>
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter

import numpy as np

import evaluate_segmentation_v2 as E
import followup_gap_reference as F
from note_segmentation import hz_to_midi_float

OUT = E.OUT
FIELDS = ["song_id", "position", "start_s", "end_s", "duration_s", "frame_count", "target_midi",
          "median_cents", "core_median_cents", "full_iqr_cents", "p5_p95_range_cents",
          "max_adjacent_jump_cents", "largest_time_gap_s", "n_islands", "median_confidence",
          "target_frame_share", "slow_component_range_cents", "plateau_centers", "categories",
          "any_clear_invalid", "B_clear", "C_clear", "D_clear", "B_likely", "C_likely", "D_likely",
          "classification_reason"]


def fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else v


def main() -> int:
    name = sys.argv[1]
    cfg = next(c for c in E.build_configs() + F.configs() if c["name"] == name)
    E._SONGS = E.load_songs()
    splits = json.load(open(E.REPO / "data/ml_dataset/axis2_v1/splits.json"))
    split_of = {s: k for k in ("train", "validation", "test") for s in splits[k]}

    v1, v2 = {}, {}
    for s in E._SONGS:
        v1[s] = E.eval_song(s, dict(name="v1", kind="v1"))
        v2[s] = E.eval_song(s, cfg)

    # ---- v2_notes.csv
    with (OUT / "v2_notes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config"] + FIELDS + ["split", "v1_positions"])
        for s in E._SONGS:
            own1 = np.full(E._SONGS[s]["times"].size, -1)
            for r in v1[s]:
                own1[r["source_frames"]] = r["position"]
            for r in v2[s]:
                ids = sorted(set(own1[r["source_frames"]].tolist()) - {-1})
                w.writerow([name] + [fmt(r[k]) for k in FIELDS] + [split_of.get(s, ""), ";".join(map(str, ids))])

    # ---- v1_vs_v2_notes.csv
    outcome_counts = Counter()
    with (OUT / "v1_vs_v2_notes.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "split", "v1_index", "v1_start_s", "v1_end_s", "v1_duration_s", "v1_frames",
                    "v1_target_midi", "v1_iqr_cents", "v1_largest_gap_s", "v1_categories", "v1_clear_invalid",
                    "v1_likely_invalid", "outcome", "n_substantive_pieces", "n_overlapping_v2", "v2_indices",
                    "v2_frames_from_this_note", "v2_targets", "v2_max_iqr_cents", "v2_categories",
                    "v2_any_clear_invalid", "config"])
        for s in E._SONGS:
            own2 = np.full(E._SONGS[s]["times"].size, -1)
            for r in v2[s]:
                own2[r["source_frames"]] = r["position"]
            for r in v1[s]:
                fr = r["source_frames"]
                ids, cnt = np.unique(own2[fr], return_counts=True)
                keep = ids != -1
                ids, cnt = ids[keep], cnt[keep]
                pieces = int(np.sum((cnt >= 3) & (cnt >= 0.15 * fr.size)))
                if ids.size == 0:
                    o = "dropped"
                else:
                    j = int(np.argmax(cnt))
                    big = v2[s][int(ids[j])]
                    iou = cnt[j] / (fr.size + big["frame_count"] - cnt[j])
                    o = ("same" if iou >= 0.8 else "split" if pieces >= 2
                         else "merged" if cnt[j] >= 0.85 * fr.size else "partial")
                outcome_counts[o] += 1
                vs = [v2[s][int(i)] for i in ids]
                likely = r["B_likely"] or r["C_likely"] or r["D_likely"]
                w.writerow([s, split_of.get(s, ""), r["position"], fmt(r["start_s"]), fmt(r["end_s"]),
                            fmt(r["duration_s"]), r["frame_count"], r["target_midi"], fmt(r["full_iqr_cents"]),
                            fmt(r["largest_time_gap_s"]), r["categories"], r["any_clear_invalid"], likely, o,
                            pieces, ids.size, ";".join(map(str, ids.tolist())), ";".join(map(str, cnt.tolist())),
                            ";".join(str(v["target_midi"]) for v in vs),
                            fmt(max((v["full_iqr_cents"] for v in vs), default=float("nan"))),
                            ";".join(v["categories"] or "-" for v in vs),
                            ";".join(str(v["any_clear_invalid"]) for v in vs), name])

    # ---- representative_examples.csv
    picks = {("chris_stapleton_tennessee_whiskey_official_audio", 319): "stapleton_319",
             ("the_weeknd_starboy_audio_ft_daft_punk", 2): "starboy_2"}
    for row in csv.DictReader(open(E.TAIL / "extreme_examples.csv")):
        picks.setdefault((row["song_id"], int(row["source_note_index"])), "tail_" + row["selection"])
    for row in csv.DictReader(open(E.REPO / "ml/audits/axis2_stable_center/example_notes.csv")):
        picks.setdefault((row["song_id"], int(float(row["source_note_index"]))), "stable_center_" + row["group"])
    traces = {}
    with (OUT / "representative_examples.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["example", "song_id", "v1_index", "version", "segment_index"] + FIELDS[2:] + ["config"])
        for (s, k), label in picks.items():
            r1 = v1[s][k]
            own2 = np.full(E._SONGS[s]["times"].size, -1)
            for r in v2[s]:
                own2[r["source_frames"]] = r["position"]
            ids = sorted(set(own2[r1["source_frames"]].tolist()) - {-1})
            w.writerow([label, s, k, "v1", k] + [fmt(r1[f]) for f in FIELDS[2:]] + [name])
            for i in ids:
                r = v2[s][i]
                w.writerow([label, s, k, "v2", i] + [fmt(r[f]) for f in FIELDS[2:]] + [name])
            if label in ("stapleton_319", "starboy_2"):
                traces[label] = {
                    "v1": {f: fmt(r1[f]) for f in FIELDS[2:]},
                    "v2": [{"segment_index": i, **{f: fmt(v2[s][i][f]) for f in FIELDS[2:]},
                            "frames_from_v1_note": int(np.sum(own2[r1["source_frames"]] == i))} for i in ids],
                    "v1_frames_dropped_by_v2": int(np.sum(own2[r1["source_frames"]] == -1)),
                }

    # ---- target estimator comparison on the same v2 segments
    agree = rep_core = rep_modal = n = 0
    for s in E._SONGS:
        for r in v2[s]:
            fi = r["source_frames"]
            hz = E._SONGS[s]["f0"][fi]
            midi = np.asarray(hz_to_midi_float(hz, E._SONGS[s]["tuning"]), float)
            core = int(np.round(np.median(E._core_slice(midi))))
            notes, counts = np.unique(np.round(midi).astype(int), return_counts=True)
            best = notes[counts == counts.max()]
            modal = int(best[np.argmin(np.abs(best - np.median(E._core_slice(midi))))])
            share = lambda m: float(np.mean(np.abs(100 * (midi - m)) <= 50))
            n += 1
            agree += core == modal
            rep_core += share(core) >= 0.5
            rep_modal += share(modal) >= 0.5
    detail = {
        "config": cfg,
        "v1_outcomes": dict(outcome_counts),
        "target_estimator_comparison": {
            "notes": n, "core_median_equals_modal_pct": 100 * agree / n,
            "core_median_target_represented_pct": 100 * rep_core / n,
            "modal_target_represented_pct": 100 * rep_modal / n,
            "represented_definition": ">= 50% of the segment's frames within +-50 c of the target",
        },
        "traces": traces,
    }
    json.dump(detail, open(OUT / f"_detail_{name}.json", "w"), indent=1, default=str)
    print(json.dumps({k: detail[k] for k in ("v1_outcomes", "target_estimator_comparison")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
