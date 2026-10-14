#!/usr/bin/env python3
"""
Follow-up analysis for the recommended hybrid (gap_v1) configuration, written
after the parameter sweep (evaluate_segmentation_v2.py) and the per-note detail
dump (detail_outputs.py) had already completed. Reuses both; reruns no sweep.

Answers the questions the sweep summary cannot:
  * which v1 problems the hybrid fixes, and how much valid v1 material it touches
  * what created each hybrid boundary (hard gap / short-gap rule / v1 commit)
  * where the new very short notes come from
  * how often consecutive notes alternate by one semitone (vibrato over-splitting)
  * whether the target-MIDI rule leaves unrepresented targets

Reads only harmony.npz / notes.npz via evaluate_segmentation_v2. Writes only
into this directory:
  _resume_analysis_<config>.json   every number quoted in segmentation_v2_summary.md
  tiny_notes_<config>.csv          one row per new hybrid note shorter than 0.10 s

Usage: python resume_analysis.py [hybrid_config_name]
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

HYBRID = sys.argv[1] if len(sys.argv) > 1 else "gapv1local_g0.5_s100"
PURE = "v2_g0.5_s70_p3"

HARD_GAP_S = 0.50
CONTIG_S = 0.03          # > 1.5 hops = an unvoiced gap of at least one frame
LOCAL_FRAMES = 10        # contiguous frames either side of a boundary for its local pitch step
SAME_PITCH_CENTS = 50    # local medians this close on both sides = the same pitch resumed
STEP_CENTS = 80          # local step this large = a genuine pitch change (tail audit REGION_SEP)
STABLE_RANGE_CENTS = 100  # p5-p95 within this = one held pitch
LOW_CONF = 0.70


def likely(r):
    return bool(r["B_likely"] or r["C_likely"] or r["D_likely"])


def ambiguous(r):
    return r["categories"] == "E" and bool(r["flagged"])


def owners(rows, n):
    own = np.full(n, -1)
    for r in rows:
        own[r["source_frames"]] = r["position"]
    return own


def abs_cents(song, frames):
    d = E._SONGS[song]
    return 100.0 * np.asarray(hz_to_midi_float(d["f0"][frames], d["tuning"]), float)


def contiguous_tail(times, frames, from_end):
    """Last (or first) <= LOCAL_FRAMES frames that are gap-free."""
    f = frames[::-1] if from_end else frames
    t = times[f]
    k = 1
    while k < min(f.size, LOCAL_FRAMES) and abs(t[k] - t[k - 1]) <= CONTIG_S:
        k += 1
    out = f[:k]
    return out[::-1] if from_end else out


def boundary(song, a, b):
    """Describe the boundary between consecutive notes a -> b (rows)."""
    t = E._SONGS[song]["times"]
    gap = float(t[b["source_frames"][0]] - t[a["source_frames"][-1]])
    left = contiguous_tail(t, a["source_frames"], True)
    right = contiguous_tail(t, b["source_frames"], False)
    step = float(np.median(abs_cents(song, right)) - np.median(abs_cents(song, left)))
    kind = "hard" if gap > HARD_GAP_S + 1e-9 else "short" if gap > CONTIG_S else "contig"
    return dict(gap_s=gap, kind=kind, step_cents=step)


def iou_best(rows_a, own_b, size_b):
    """For each note in rows_a, the best frame IoU with any note of the other segmentation."""
    out = []
    for r in rows_a:
        ids, cnt = np.unique(own_b[r["source_frames"]], return_counts=True)
        keep = ids != -1
        ids, cnt = ids[keep], cnt[keep]
        if not ids.size:
            out.append((0.0, -1))
            continue
        j = int(np.argmax(cnt))
        out.append((float(cnt[j] / (r["frame_count"] + size_b[int(ids[j])] - cnt[j])), int(ids[j])))
    return out


def v1_outcome(r1, own2, size2):
    """Same mapping rule as evaluate_segmentation_v2.summarize."""
    fr = r1["source_frames"]
    ids, cnt = np.unique(own2[fr], return_counts=True)
    keep = ids != -1
    ids, cnt = ids[keep], cnt[keep]
    pieces = int(np.sum((cnt >= 3) & (cnt >= 0.15 * fr.size)))
    if ids.size == 0:
        return "dropped", pieces, ids
    j = int(np.argmax(cnt))
    iou = cnt[j] / (fr.size + size2[int(ids[j])] - cnt[j])
    o = ("same" if iou >= 0.8 else "split" if pieces >= 2
         else "merged" if cnt[j] >= 0.85 * fr.size else "partial")
    return o, pieces, ids


def alternating_chains(song, rows):
    """
    Runs of >= 3 consecutive gap-free notes whose targets alternate between two
    values one semitone apart (A B A ...), each note <= 0.30 s. This is what v1's
    commit rule does to wide vibrato; a fast trill would look the same, so the
    count is an upper bound on vibrato over-splitting.
    """
    t = E._SONGS[song]["times"]
    chains, cur = [], [0] if rows else []
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        ok = (t[b["source_frames"][0]] - t[a["source_frames"][-1]] <= CONTIG_S
              and abs(a["target_midi"] - b["target_midi"]) == 1
              and a["duration_s"] <= 0.30 and b["duration_s"] <= 0.30
              and (len(cur) < 2 or rows[cur[-2]]["target_midi"] == b["target_midi"]))
        if ok:
            cur.append(i)
        else:
            if len(cur) >= 3:
                chains.append(cur)
            cur = [i]
    if len(cur) >= 3:
        chains.append(cur)
    return chains


def main() -> int:
    E._SONGS = E.load_songs()
    cfgs = {c["name"]: c for c in E.build_configs() + F.configs()}
    hyb_cfg = cfgs[HYBRID]
    v1, hy, pu, hy_modal = {}, {}, {}, {}
    for s in E._SONGS:
        v1[s] = E.eval_song(s, cfgs["v1"])
        hy[s] = E.eval_song(s, hyb_cfg)
        pu[s] = E.eval_song(s, cfgs[PURE])
        hy_modal[s] = E.eval_song(s, dict(hyb_cfg, name=HYBRID + "_modal", target_estimator="modal"))

    allrows = lambda d: [r for s in d for r in d[s]]
    R1, RH, RP = allrows(v1), allrows(hy), allrows(pu)
    out = {"configs": {"hybrid": hyb_cfg, "pure": cfgs[PURE]}}

    # ---- consistency with the saved sweep and detail CSVs (only when this
    # run is over the same songs the sweep covered, i.e. writes beside it)
    sweep_paths = (E.OUT / "parameter_comparison.csv",
                   E.OUT / "followup" / "parameter_comparison_gap_reference.csv")
    if all(p.exists() for p in sweep_paths) and (E.OUT / "v2_notes.csv").exists():
        sweep = {r["name"]: r for path in sweep_paths for r in csv.DictReader(open(path))}
        n_saved_v2 = sum(1 for _ in csv.DictReader(open(E.OUT / "v2_notes.csv")))
        saved_cfgs = {r["config"] for r in csv.DictReader(open(E.OUT / "v2_notes.csv"))}
        out["consistency"] = {
            "v1_notes_recomputed": len(R1), "v1_notes_sweep": int(sweep["v1"]["total_notes"]),
            "hybrid_notes_recomputed": len(RH), "hybrid_notes_sweep": int(sweep[HYBRID]["total_notes"]),
            "pure_notes_recomputed": len(RP), "pure_notes_sweep": int(sweep[PURE]["total_notes"]),
            "hybrid_clear_recomputed_pct": 100 * np.mean([r["any_clear_invalid"] for r in RH]),
            "hybrid_clear_sweep_pct": float(sweep[HYBRID]["pct_clear_invalid"]),
            "v2_notes_csv_rows": n_saved_v2, "v2_notes_csv_configs": sorted(saved_cfgs),
        }
    else:
        out["consistency"] = {"skipped": f"no sweep outputs in {E.OUT}",
                              "songs": sorted(E._SONGS), "v1_notes_recomputed": len(R1),
                              "hybrid_notes_recomputed": len(RH), "pure_notes_recomputed": len(RP)}

    # ---- section 2/6 extras: counts behind the sweep percentages
    def counts(rows):
        N = len(rows)
        dur = np.array([r["duration_s"] for r in rows])
        gap = np.array([r["largest_time_gap_s"] for r in rows])
        iqr = np.array([r["full_iqr_cents"] for r in rows])
        c = {"total": N,
             "clear": int(sum(r["any_clear_invalid"] for r in rows)),
             "likely": int(sum(likely(r) for r in rows)),
             "ambiguous_E": int(sum(ambiguous(r) for r in rows)),
             "B_clear": int(sum(r["B_clear"] for r in rows)),
             "C_clear": int(sum(r["C_clear"] for r in rows)),
             "D_clear": int(sum(r["D_clear"] for r in rows))}
        for thr in (100, 200, 400):
            c[f"iqr_gt{thr}"] = int((iqr > thr).sum())
        for thr in (0.10, 0.25, 0.50, 1.0):
            c[f"gap_gt{thr}"] = int((gap > thr + 1e-9).sum())
        for thr in (0.06, 0.08, 0.10, 0.15):
            c[f"dur_lt{thr}"] = int((dur < thr - 1e-9).sum())
        c["max_gap_s"] = float(gap.max())
        return c
    out["counts"] = {"v1": counts(R1), "hybrid": counts(RH), "pure": counts(RP)}

    # remaining hybrid clear-invalid: which rule, and with what gap
    rem = [r for r in RH if r["any_clear_invalid"]]
    out["hybrid_remaining_clear"] = {
        "n": len(rem),
        "by_rule": Counter("".join(k[0] for k in ("B_clear", "C_clear", "D_clear") if r[k]) for r in rem),
        "D_clear_with_gap_0.25_0.50": int(sum(r["D_clear"] and r["largest_time_gap_s"] > 0.25 for r in rem)),
        "C_clear_octave_jump": int(sum(r["C_clear"] and r["n_octave_jumps_consecutive"] > 0 for r in rem)),
    }

    # ---- section 3/4: what the hybrid changes
    fix = Counter()
    pieces_hist = {"all": Counter(), "v1_valid": Counter(), "v1_likely": Counter(), "v1_clear": Counter()}
    outcomes = {"all": Counter(), "v1_valid": Counter(), "v1_likely": Counter(), "v1_clear": Counter()}
    valid_changed_split_bounds = Counter()
    vib_split_bounds = Counter()
    for s in E._SONGS:
        n = E._SONGS[s]["times"].size
        own1, own2 = owners(v1[s], n), owners(hy[s], n)
        size2 = {r["position"]: r["frame_count"] for r in hy[s]}
        for r1 in v1[s]:
            o, pieces, ids = v1_outcome(r1, own2, size2)
            groups = ["all", "v1_clear" if r1["any_clear_invalid"] else None,
                      "v1_likely" if likely(r1) else "v1_valid"]
            for g in filter(None, groups):
                outcomes[g][o] += 1
                pieces_hist[g][min(pieces, 4)] += 1
            v2s = [hy[s][int(i)] for i in ids]
            any_clear = any(v["any_clear_invalid"] for v in v2s)
            any_likely = any(likely(v) for v in v2s)
            if r1["any_clear_invalid"]:
                fix["v1_clear_total"] += 1
                fix["v1_clear_fixed" if not any_clear else "v1_clear_remaining"] += 1
                fix["v1_clear_fixed_to_valid" if not any_likely else "v1_clear_still_likely"] += 1
            if likely(r1):
                fix["v1_likely_total"] += 1
                fix["v1_likely_fixed" if not any_likely else "v1_likely_remaining"] += 1
            else:
                fix["v1_valid_total"] += 1
                if o != "same":
                    fix["v1_valid_changed"] += 1
                if any_clear:
                    fix["v1_valid_touching_clear_v2"] += 1
                if any_likely:
                    fix["v1_valid_touching_likely_v2"] += 1
                if o == "split":
                    # what kind of boundary cut this valid v1 note
                    inside = [v for v in v2s]
                    inside.sort(key=lambda v: v["position"])
                    for a, b in zip(inside, inside[1:]):
                        bd = boundary(s, a, b)
                        tag = bd["kind"] + ("_samepitch" if abs(bd["step_cents"]) < SAME_PITCH_CENTS
                                            else "_step" if abs(bd["step_cents"]) >= STEP_CENTS else "_mid")
                        valid_changed_split_bounds[tag] += 1
            if o != "same":
                fix["changed_total"] += 1
                fix["changed_v1_likely"] += likely(r1)
                fix["changed_v1_clear"] += bool(r1["any_clear_invalid"])
        # newly created invalid: hybrid clear notes whose v1 sources were all valid
        likely1 = {r["position"]: likely(r) for r in v1[s]}
        for r2 in hy[s]:
            src = set(own1[r2["source_frames"]].tolist()) - {-1}
            if r2["any_clear_invalid"] and src and not any(likely1[p] for p in src):
                fix["hybrid_clear_from_only_valid_v1"] += 1
            if likely(r2) and src and not any(likely1[p] for p in src):
                fix["hybrid_likely_from_only_valid_v1"] += 1
    out["fix_accounting"] = dict(fix)
    out["v1_outcomes"] = {g: dict(c) for g, c in outcomes.items()}
    out["v1_pieces_hist"] = {g: {str(k): v for k, v in sorted(c.items())} for g, c in pieces_hist.items()}
    out["v1_valid_split_boundaries"] = dict(valid_changed_split_bounds)

    # reverse mapping: hybrid notes holding >= 2 v1 notes nearly whole (sweep definition)
    merges = Counter()
    for s in E._SONGS:
        n = E._SONGS[s]["times"].size
        own1 = owners(v1[s], n)
        size1 = Counter(own1[own1 >= 0].tolist())
        for r2 in hy[s]:
            ids, cnt = np.unique(own1[r2["source_frames"]], return_counts=True)
            whole = sum(1 for i, c in zip(ids, cnt) if i != -1 and c >= 3 and c >= 0.5 * size1[int(i)])
            merges[min(whole, 3)] += 1
    out["hybrid_notes_by_whole_v1_contained"] = {str(k): v for k, v in sorted(merges.items())}

    # ---- every hybrid boundary: what made it, and does it separate the same pitch?
    bstats = Counter()
    same_pitch_short = []
    for s in E._SONGS:
        n = E._SONGS[s]["times"].size
        own1 = owners(v1[s], n)
        for a, b in zip(hy[s], hy[s][1:]):
            bd = boundary(s, a, b)
            had_v1 = own1[a["source_frames"][-1]] != own1[b["source_frames"][0]]
            sp = abs(bd["step_cents"]) < SAME_PITCH_CENTS
            key = f"{bd['kind']}|{'v1_had' if had_v1 else 'new'}|{'samepitch' if sp else 'pitchchange'}"
            bstats[key] += 1
            if bd["kind"] == "short" and not had_v1 and sp:
                same_pitch_short.append(bd["gap_s"])
    out["hybrid_boundaries"] = dict(sorted(bstats.items()))
    out["new_short_gap_samepitch_boundaries"] = {
        "n": len(same_pitch_short),
        "gap_s_quartiles": [float(x) for x in np.percentile(same_pitch_short, [25, 50, 75])] if same_pitch_short else [],
    }

    # ---- section 5: new tiny notes
    tiny_rows = []
    for s in E._SONGS:
        n = E._SONGS[s]["times"].size
        own1 = owners(v1[s], n)
        size1 = {r["position"]: r["frame_count"] for r in v1[s]}
        best = iou_best(hy[s], own1, size1)
        for i, r2 in enumerate(hy[s]):
            if r2["duration_s"] >= 0.10 - 1e-9 or best[i][0] >= 0.8:
                continue
            sides = []
            if i > 0:
                sides.append(("left", boundary(s, hy[s][i - 1], r2), hy[s][i - 1]))
            if i + 1 < len(hy[s]):
                sides.append(("right", boundary(s, r2, hy[s][i + 1]), hy[s][i + 1]))
            src = sorted(set(own1[r2["source_frames"]].tolist()) - {-1})
            src_likely = any(likely(v1[s][p]) for p in src)
            src_clear = any(v1[s][p]["any_clear_invalid"] for p in src)
            kinds = [bd["kind"] for _, bd, _ in sides] + ["edge"] * (2 - len(sides))
            near = [(bd, nb) for _, bd, nb in sides if bd["kind"] != "hard"]
            stable = r2["p5_p95_range_cents"] <= STABLE_RANGE_CENTS
            # an island: nothing gap-free on either side, and at least one side
            # is a hard gap (or the edge of the track)
            isolated = "contig" not in kinds and ("hard" in kinds or "edge" in kinds)
            same_as_neighbor = any(abs(bd["step_cents"]) < SAME_PITCH_CENTS for bd, _ in near)
            if isolated:
                cat = "C_hardgap_isolated_fragment"
            elif same_as_neighbor:
                cat = "F_same_pitch_as_neighbor"
            elif r2["median_confidence"] < LOW_CONF:
                cat = "E_low_confidence"
            elif stable and any(bd["kind"] == "short" and abs(bd["step_cents"]) >= STEP_CENTS for bd, _ in near):
                cat = "A_short_pitch_after_gap"
            elif stable and any(abs(bd["step_cents"]) >= STEP_CENTS for bd, _ in near):
                cat = "B_adjacent_note_exposed"
            elif not stable and any(bd["kind"] == "contig" for bd, _ in near):
                cat = "D_v1_restart_glide_fragment"
            else:
                cat = "G_ambiguous"
            tiny_rows.append(dict(
                song_id=s, position=r2["position"], start_s=r2["start_s"], end_s=r2["end_s"],
                duration_s=r2["duration_s"], frame_count=r2["frame_count"], target_midi=r2["target_midi"],
                p5_p95_range_cents=r2["p5_p95_range_cents"], median_confidence=r2["median_confidence"],
                left_kind=sides[0][1]["kind"] if sides and sides[0][0] == "left" else "edge",
                left_gap_s=sides[0][1]["gap_s"] if sides and sides[0][0] == "left" else "",
                left_step_cents=sides[0][1]["step_cents"] if sides and sides[0][0] == "left" else "",
                right_kind=sides[-1][1]["kind"] if sides and sides[-1][0] == "right" else "edge",
                right_gap_s=sides[-1][1]["gap_s"] if sides and sides[-1][0] == "right" else "",
                right_step_cents=sides[-1][1]["step_cents"] if sides and sides[-1][0] == "right" else "",
                v1_source_notes=";".join(map(str, src)), v1_source_likely_invalid=src_likely,
                v1_source_clear_invalid=src_clear, best_v1_iou=best[i][0],
                hybrid_clear_invalid=r2["any_clear_invalid"], category=cat))
    with (E.OUT / f"tiny_notes_{HYBRID}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tiny_rows[0].keys()))
        w.writeheader()
        for r in tiny_rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    tiny = {}
    for thr in (0.06, 0.08, 0.10):
        sub = [r for r in tiny_rows if r["duration_s"] < thr - 1e-9]
        c = Counter(r["category"] for r in sub)
        tiny[f"lt{thr}"] = {"n_new": len(sub),
                            "categories": {k: [v, 100 * v / len(sub)] for k, v in sorted(c.items())},
                            "from_v1_likely_invalid_pct": 100 * np.mean([r["v1_source_likely_invalid"] for r in sub]),
                            "from_v1_clear_invalid_pct": 100 * np.mean([r["v1_source_clear_invalid"] for r in sub])}
    tiny["v1_tiny_lt0.10_total"] = out["counts"]["v1"]["dur_lt0.1"]
    tiny["hybrid_tiny_lt0.10_total"] = out["counts"]["hybrid"]["dur_lt0.1"]
    tiny["hybrid_tiny_lt0.10_matching_a_v1_note"] = tiny["hybrid_tiny_lt0.10_total"] - tiny["lt0.1"]["n_new"]
    out["tiny_notes"] = tiny

    # ---- section 8: target rule
    def represented(rows):
        return 100 * np.mean([r["target_frame_share"] >= 0.5 for r in rows])
    unrep = [(s, i) for s in E._SONGS for i, r in enumerate(hy[s]) if r["target_frame_share"] < 0.5]
    fixed_by_modal = sum(hy_modal[s][i]["target_frame_share"] >= 0.5 for s, i in unrep)
    changed_by_modal = sum(hy_modal[s][i]["target_midi"] != hy[s][i]["target_midi"] for s in E._SONGS
                           for i in range(len(hy[s])))
    broken_by_modal = sum(hy[s][i]["target_frame_share"] >= 0.5 > hy_modal[s][i]["target_frame_share"]
                          for s in E._SONGS for i in range(len(hy[s])))
    # where core-median targets go unrepresented, why: wide note vs wrong semitone
    why = Counter()
    for s, i in unrep:
        r = hy[s][i]
        midi = abs_cents(s, r["source_frames"]) / 100.0
        top_share = np.bincount(np.round(midi).astype(int) - int(np.round(midi).min())).max() / midi.size
        why["no_semitone_holds_half" if top_share < 0.5 else "another_semitone_holds_half"] += 1
    long_ = lambda rows: [r for r in rows if r["frame_count"] >= 10]
    out["target_rule"] = {
        "represented_definition": ">= 50% of frames within +-50 c of target",
        "v1_core_median_pct": represented(R1), "hybrid_core_median_pct": represented(RH),
        "hybrid_modal_pct": represented(allrows(hy_modal)),
        "v1_core_median_pct_ge10frames": represented(long_(R1)),
        "hybrid_core_median_pct_ge10frames": represented(long_(RH)),
        "hybrid_unrepresented_n": len(unrep), "fixed_by_modal": int(fixed_by_modal),
        "targets_changed_by_modal": int(changed_by_modal), "made_unrepresented_by_modal": int(broken_by_modal),
        "unrepresented_reason": dict(why),
        "modal_clear_invalid_pct": 100 * np.mean([r["any_clear_invalid"] for r in allrows(hy_modal)]),
    }

    # ---- section 9: vibrato over-splitting
    vib = {}
    for name, d in (("v1", v1), ("hybrid", hy), ("pure", pu)):
        ch = [c for s in d for c in alternating_chains(s, d[s])]
        notes_in = sum(len(c) for c in ch)
        N = sum(len(d[s]) for s in d)
        vib[name] = {"chains": len(ch), "notes_in_chains": notes_in, "pct_notes": 100 * notes_in / N,
                     "chains_ge5_notes": sum(len(c) >= 5 for c in ch)}
    # the sweep's vibrato_like subset that the hybrid splits: what cut them
    for s in E._SONGS:
        n = E._SONGS[s]["times"].size
        own2 = owners(hy[s], n)
        size2 = {r["position"]: r["frame_count"] for r in hy[s]}
        for r1 in v1[s]:
            clean = (not likely(r1)) and r1["largest_time_gap_s"] <= 0.10 + 1e-9 and r1["frame_count"] >= 15
            if not (clean and r1["frame_count"] >= 20 and r1["slow_component_range_cents"] < 100
                    and r1["full_iqr_cents"] >= 40 and r1["n_center_crossings"] >= 4):
                continue
            o, _, ids = v1_outcome(r1, own2, size2)
            vib_split_bounds["vibrato_like_total"] += 1
            if o == "split":
                vs = sorted((hy[s][int(i)] for i in ids), key=lambda v: v["position"])
                for a, b in zip(vs, vs[1:]):
                    bd = boundary(s, a, b)
                    vib_split_bounds[f"{bd['kind']}_{'samepitch' if abs(bd['step_cents']) < SAME_PITCH_CENTS else 'step'}"] += 1
                vib_split_bounds["vibrato_like_split"] += 1
    vib["hybrid_splits_of_v1_vibrato_like"] = dict(vib_split_bounds)
    out["vibrato"] = vib

    json.dump(out, open(E.OUT / f"_resume_analysis_{HYBRID}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("consistency", "fix_accounting", "tiny_notes", "target_rule", "vibrato")},
                     indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
