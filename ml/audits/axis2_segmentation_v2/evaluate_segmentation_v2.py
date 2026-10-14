#!/usr/bin/env python3
"""
Shadow-mode evaluation of note segmentation v2 against v1.

Source of truth: data/main_data/<song>/harmony.npz (the original audio no
longer exists). Every configuration re-segments all 50 axis2_ml songs from the
saved vocal track, in memory. Nothing is written outside this directory.

Per-note metrics and the invalidity classifier are those of
ml/audits/axis2_segmentation_tail/ — the metric code below is copied verbatim
from that audit and the classifier is imported from it, and both are asserted
to reproduce the tail audit's per-note results on v1 before anything else runs,
so v1 and v2 numbers are directly comparable.

Reads: dataset_selections.json, main_data/<song>/{harmony,notes}.npz,
tail-audit CSV, stable-center example list. Never reads ML labels.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TAIL = REPO / "ml" / "audits" / "axis2_segmentation_tail"
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(TAIL))

from note_segmentation import (  # noqa: E402
    _core_slice,
    cents_diff,
    midi_to_hz,
    segment_notes_with_frame_indices as v1_segment,
)
from note_segmentation_v2 import SegmentationV2Config, segment_notes_v2_with_frame_indices as v2_segment  # noqa: E402
from segmentation_tail_audit import (  # noqa: E402
    CLEAR_REGION_FRAMES,
    FLAG_RULES,
    GAP_LIKELY_S,
    HOP_S,
    SMOOTH_WIN,
    classify,
    excursions,
    pitch_regions,
    plateaus,
    rolling_median,
)

MAIN = REPO / "data" / "main_data"

# ---------------------------------------------------------------- configs

HARD_GAPS = (0.25, 0.35, 0.50, 0.75)
SEPARATIONS = (70, 80, 100, 120)
PERSISTENCE = (3, 4, 5)


def build_configs():
    cfgs = [dict(name="v1", kind="v1")]
    for g in HARD_GAPS:
        cfgs.append(dict(name=f"v1+hardgap{g}", kind="v1_hardgap", hard_gap_s=g))
    for g in HARD_GAPS:
        for s in SEPARATIONS:
            for y in PERSISTENCE:
                cfgs.append(dict(name=f"v2_g{g}_s{s}_p{y}", kind="v2", hard_gap_s=g,
                                 separation_cents=s, persistence_frames=y))
    # hybrids: gap rules + v1's exact-semitone commit (+ sustained path)
    for g in HARD_GAPS:
        for sp in SEPARATIONS:
            cfgs.append(dict(name=f"gapv1_g{g}_s{sp}", kind="v2", mode="gap_v1", hard_gap_s=g,
                             separation_cents=sp, persistence_frames=4))
    for g in (0.35, 0.50):
        for sp in SEPARATIONS:
            for y in PERSISTENCE:
                cfgs.append(dict(name=f"gapv1sus_g{g}_s{sp}_p{y}", kind="v2", mode="gap_v1_sustained",
                                 hard_gap_s=g, separation_cents=sp, persistence_frames=y))
    # one-at-a-time sensitivity around a central configuration
    base = dict(kind="v2", hard_gap_s=0.35, separation_cents=100, persistence_frames=4)
    for r in (0, 2, 4, 6):
        cfgs.append(dict(base, name=f"sens_return{r}", return_window_frames=r))
    for w in (15, 25):
        cfgs.append(dict(base, name=f"sens_center{w}", center_window_frames=w))
    cfgs.append(dict(base, name="sens_target_modal", target_estimator="modal"))
    return cfgs


def v2_config(cfg):
    keys = SegmentationV2Config.__dataclass_fields__.keys()
    return SegmentationV2Config(**{k: v for k, v in cfg.items() if k in keys})


# ---------------------------------------------------------------- data

_SONGS = None
_V1_OWNER = None     # song -> frame -> v1 note position (or -1)
_SUBSETS = None      # subset name -> set of (song, v1 position)
_V1_VALID = None     # set of (song, v1 position) with no likely B/C/D in v1


def songs_axis2():
    sel = json.load(open(REPO / "data" / "dataset_selections.json"))["songs"]
    return sorted(s for s, v in sel.items() if v["axis2_ml"])


def load_songs():
    out = {}
    for s in songs_axis2():
        h = np.load(MAIN / s / "harmony.npz", allow_pickle=False)
        out[s] = dict(
            times=h["vocal_frame_times_s"], f0=h["vocal_f0_hz"],
            conf=h["vocal_crepe_confidence"], tuning=float(h["tuning_offset_semitones"]),
        )
    return out


def segment(song, cfg):
    """List of (source frame indices, target MIDI) for one song and config."""
    d = _SONGS[song]
    if cfg["kind"] == "v1":
        segs, idx = v1_segment(d["times"], d["f0"], d["tuning"])
    elif cfg["kind"] == "v1_hardgap":
        # ablation: v1's own logic run separately inside each hard-gap region
        vp = np.flatnonzero(~np.isnan(d["f0"]))
        cuts = np.flatnonzero(np.diff(d["times"][vp]) > cfg["hard_gap_s"]) + 1
        segs, idx = [], []
        for part in np.split(vp, cuts):
            a, b = part[0], part[-1] + 1
            s_, i_ = v1_segment(d["times"][a:b], d["f0"][a:b], d["tuning"])
            segs += s_
            idx += [x + a for x in i_]
    else:
        segs, idx = v2_segment(d["times"], d["f0"], d["tuning"], v2_config(cfg))
    return [(i, int(s["target_note"])) for s, i in zip(segs, idx)]


# ---------------------------------------------------------------- metrics


def pct(a, q):
    return float(np.percentile(a, q)) if a.size else float("nan")


def note_metrics(song, fi, target):
    d = _SONGS[song]
    t = d["times"][fi]
    c = cents_diff(d["f0"][fi], midi_to_hz(target, d["tuning"]))
    conf = d["conf"][fi]
    # ---- verbatim from segmentation_tail_audit.py (fields the classifier uses)
    nf = fi.size
    dur = float(t[-1] - t[0])
    dc = np.abs(np.diff(c))
    dt = np.diff(t)
    fgap = np.diff(fi)
    q25, q75 = np.percentile(c, [25, 75])
    iqr = float(q75 - q25)
    cut = np.flatnonzero(fgap > 1) + 1
    isl = np.split(np.arange(nf), cut)
    isl_med = [float(np.median(c[x])) for x in isl]
    isl_len = [x.size for x in isl]
    regs = pitch_regions(c)
    centers = [r[2] for r in regs]
    reg_lens = [r[1] - r[0] for r in regs]
    reg_spread = float(max(centers) - min(centers)) if len(regs) > 1 else 0.0
    big = [r for r in regs if r[1] - r[0] >= CLEAR_REGION_FRAMES]
    big_spread = (max(r[2] for r in big) - min(r[2] for r in big)) if len(big) > 1 else 0.0
    slow = rolling_median(c, SMOOTH_WIN) if nf >= 3 else c
    slow_range = float(np.ptp(slow))
    plats = plateaus(c)
    pc_ = [p[2] for p in plats]
    plat_spread = float(max(pc_) - min(pc_)) if len(plats) > 1 else 0.0
    plat4 = [p for p in plats if p[1] - p[0] >= 4]
    plat4_spread = float(max(p[2] for p in plat4) - min(p[2] for p in plat4)) if len(plat4) > 1 else 0.0
    consec = fgap == 1
    jump_consec = float(dc[consec].max()) if consec.any() else 0.0
    exc, badmask = excursions(c)
    octave_jumps_consec = int(np.sum((dc >= 1000) & (dc <= 1400) & (fgap == 1)))
    if badmask.any() and (~badmask).sum() >= 4:
        range_clean = float(np.ptp(c[~badmask]))
    else:
        range_clean = float(np.ptp(c))
    isl_steps = [abs(isl_med[i + 1] - isl_med[i]) for i in range(len(isl) - 1)
                 if isl_len[i] >= 3 and isl_len[i + 1] >= 3]
    max_isl_step = max(isl_steps, default=0.0)
    octave_like_isl = any(abs(s - 1200) <= 150 for s in isl_steps)
    gap_pitch_steps = [abs(isl_med[i + 1] - isl_med[i]) for i in range(len(isl) - 1)
                       if fgap[np.flatnonzero(fgap > 1)[i]] * HOP_S > GAP_LIKELY_S]
    # ---- end verbatim
    core = float(np.median(_core_slice(c)))
    m = max(1, int(round(0.2 * nf)))
    return dict(
        song_id=song, start_s=float(t[0]), end_s=float(t[-1]), duration_s=dur, frame_count=nf,
        target_midi=target, median_cents=float(np.median(c)), core_median_cents=core,
        full_iqr_cents=iqr, p5_p95_range_cents=float(np.subtract(*np.percentile(c, [95, 5]))),
        full_range_cents=float(np.ptp(c)),
        max_adjacent_jump_cents=float(dc.max()) if dc.size else 0.0,
        largest_time_gap_s=float(dt.max()) if dt.size else 0.0,
        n_islands=len(isl), island_medians=";".join(f"{x:.0f}" for x in isl_med),
        max_cross_gap_pitch_step_cents=max(gap_pitch_steps, default=0.0),
        max_island_step_cents=max_isl_step, octave_like_island_step=octave_like_isl,
        median_confidence=float(np.median(conf)),
        apparent_pitch_region_count=len(regs), region_centers=";".join(f"{x:.0f}" for x in centers),
        region_lengths=";".join(map(str, reg_lens)), region_spread_cents=reg_spread,
        clear_region_spread_cents=float(big_spread), slow_component_range_cents=slow_range,
        plateau_centers=";".join(f"{x:.0f}" for x in pc_),
        plateau_lengths=";".join(str(p[1] - p[0]) for p in plats),
        plateau_spread_cents=plat_spread, plateau4_spread_cents=plat4_spread,
        max_jump_consecutive_frames=jump_consec, n_short_excursions=len(exc),
        n_excursion_frames=int(badmask.sum()), n_octave_jumps_consecutive=octave_jumps_consec,
        range_without_excursions=range_clean,
        # extra descriptors (not used by the classifier)
        target_frame_share=float(np.mean(np.abs(c) <= 50.0)),
        onset_minus_core_cents=float(np.median(c[:m]) - core),
        release_minus_core_cents=float(np.median(c[-m:]) - core),
        n_center_crossings=int(np.sum(np.diff(np.sign(c - np.median(c))) != 0)),
    )


def eval_song(song, cfg, keep_rows=False):
    segs = segment(song, cfg)
    rows = []
    for pos, (fi, target) in enumerate(segs):
        r = note_metrics(song, fi, target)
        classify(r)
        r["position"] = pos
        r["source_frames"] = fi
        rows.append(r)
    return rows


# ---------------------------------------------------------------- summary


def summarize(cfg, rows):
    N = len(rows)
    A = lambda k: np.array([r[k] for r in rows], float)
    dur, nfr = A("duration_s"), A("frame_count")
    gap, iqr = A("largest_time_gap_s"), A("full_iqr_cents")
    eps = 1e-9
    S = dict(name=cfg["name"], kind=cfg["kind"],
             hard_gap_s=cfg.get("hard_gap_s", ""), separation_cents=cfg.get("separation_cents", ""),
             persistence_frames=cfg.get("persistence_frames", ""),
             mode=cfg.get("mode", "sustained" if cfg["kind"] == "v2" else cfg["kind"]))
    S.update(total_notes=N, assigned_frames=int(nfr.sum()),
             median_duration_s=float(np.median(dur)), p90_duration_s=pct(dur, 90),
             p95_duration_s=pct(dur, 95), p99_duration_s=pct(dur, 99), max_duration_s=float(dur.max()),
             median_frames=float(np.median(nfr)),
             notes_gt2s=int((dur > 2).sum()), notes_gt5s=int((dur > 5).sum()), notes_gt10s=int((dur > 10).sum()))
    for thr in (100, 200, 400):
        S[f"pct_iqr_gt{thr}"] = 100 * float((iqr > thr).mean())
    S["pct_p5p95_gt200"] = 100 * float((A("p5_p95_range_cents") > 200).mean())
    for thr in (100, 200, 300):
        S[f"pct_jump_gt{thr}"] = 100 * float((A("max_adjacent_jump_cents") > thr).mean())
    for thr in (0.10, 0.25, 0.50, 1.0):
        S[f"notes_gap_gt{thr}"] = int((gap > thr + eps).sum())
    S["max_internal_gap_s"] = float(gap.max())
    for thr in (0.06, 0.08, 0.10, 0.15):
        S[f"pct_dur_lt{thr}"] = 100 * float((dur < thr - eps).mean())
    S["notes_dur_lt0.10"] = int((dur < 0.10 - eps).sum())

    B = lambda k: np.array([bool(r[k]) for r in rows])
    likely = B("B_likely") | B("C_likely") | B("D_likely")
    clear = B("any_clear_invalid")
    flagged = B("flagged")
    cat = np.array([r["categories"] for r in rows])
    S.update(
        pct_clear_invalid=100 * clear.mean(),
        pct_likely_invalid_total=100 * likely.mean(),
        pct_likely_not_clear=100 * (likely & ~clear).mean(),
        pct_ambiguous_E=100 * ((cat == "E") & flagged).mean(),
        pct_plausible_A=100 * (cat == "A").mean(),
        pct_unflagged=100 * (~flagged & ~likely).mean(),
        pct_B_clear=100 * B("B_clear").mean(), pct_D_clear=100 * B("D_clear").mean(),
        pct_C_likely=100 * B("C_likely").mean(), pct_C_clear=100 * B("C_clear").mean(),
        pct_B_likely=100 * B("B_likely").mean(), pct_D_likely=100 * B("D_likely").mean(),
        n_clear_invalid=int(clear.sum()), n_likely_invalid_total=int(likely.sum()),
        # semitone-scale multi-pitch units the tail rules are too coarse to see:
        # two raw >=4-frame plateaus >= 80 c apart AND a slow component that
        # moves >= 80 c (vibrato barely moves the ~180 ms rolling median)
        pct_multi_semitone=100 * float(((A("plateau4_spread_cents") >= 80) & (A("slow_component_range_cents") >= 80)).mean()),
        pct_target_represented=100 * float((A("target_frame_share") >= 0.5).mean()),
    )

    # v1 -> config mapping by shared source frames.
    # A v2 note is a substantive piece of a v1 note if it holds >= 3 of its
    # frames and >= 15% of them; shifting 1-2 boundary frames is not a split.
    # Outcome per v1 note:
    #   same    best-matching v2 note has frame IoU >= 0.8
    #   split   >= 2 substantive pieces
    #   merged  one v2 note holds >= 85% of it but is much larger (IoU < 0.8)
    #   dropped no frames assigned (v2 note < 3 frames)
    #   partial anything else
    v2_owner = {s: np.full(_SONGS[s]["times"].size, -1) for s in _SONGS}
    v2_size = {}
    for r in rows:
        v2_owner[r["song_id"]][r["source_frames"]] = r["position"]
        v2_size[(r["song_id"], r["position"])] = r["frame_count"]
    pieces_hist, outcome = Counter(), Counter()
    valid_out, invalid_out = Counter(), Counter()
    subset_out = {k: Counter() for k in _SUBSETS}
    for s, own in _V1_OWNER.items():
        for pos in range(own.max() + 1):
            fr = np.flatnonzero(own == pos)
            ids, cnt = np.unique(v2_owner[s][fr], return_counts=True)
            keep = ids != -1
            ids, cnt = ids[keep], cnt[keep]
            pieces = int(np.sum((cnt >= 3) & (cnt >= 0.15 * fr.size)))
            pieces_hist[min(pieces, 4)] += 1
            if ids.size == 0:
                o = "dropped"
            else:
                j = int(np.argmax(cnt))
                iou = cnt[j] / (fr.size + v2_size[(s, int(ids[j]))] - cnt[j])
                if iou >= 0.8:
                    o = "same"
                elif pieces >= 2:
                    o = "split"
                elif cnt[j] >= 0.85 * fr.size:
                    o = "merged"
                else:
                    o = "partial"
            outcome[o] += 1
            key = (s, pos)
            (valid_out if key in _V1_VALID else invalid_out)[o] += 1
            for name, members in _SUBSETS.items():
                if key in members:
                    subset_out[name][o] += 1
    tot = sum(outcome.values())
    for k, lab in [(0, "0"), (1, "1"), (2, "2"), (3, "3"), (4, "4plus")]:
        S[f"pct_v1_to_{lab}"] = 100 * pieces_hist[k] / tot
    for o in ("same", "split", "merged", "partial", "dropped"):
        S[f"pct_v1_{o}"] = 100 * outcome[o] / tot
    S["fragmentation_rate"] = S["pct_v1_split"]
    nv, ni = sum(valid_out.values()), sum(invalid_out.values())
    for o in ("same", "split", "merged"):
        S[f"pct_v1valid_{o}"] = 100 * valid_out[o] / nv
        S[f"pct_v1invalid_{o}"] = 100 * invalid_out[o] / ni
    # reverse: v2 notes that contain >= 2 v1 notes nearly whole
    v1_size = {(s, p): int(np.sum(_V1_OWNER[s] == p)) for s in _V1_OWNER for p in range(_V1_OWNER[s].max() + 1)}
    merges = 0
    for r in rows:
        ids, cnt = np.unique(_V1_OWNER[r["song_id"]][r["source_frames"]], return_counts=True)
        whole = [1 for i, c in zip(ids, cnt) if i != -1 and c >= 3 and c >= 0.5 * v1_size[(r["song_id"], int(i))]]
        merges += len(whole) >= 2
    S["pct_notes_merging_2plus_v1"] = 100 * merges / N
    for name, h in subset_out.items():
        t_ = sum(h.values())
        for o in ("same", "split", "merged"):
            S[f"{name}_{o}"] = 100 * h[o] / t_ if t_ else float("nan")
    S["note_count_change_pct"] = float("nan")
    return S


def run_config(cfg):
    rows = []
    for s in _SONGS:
        rows += eval_song(s, cfg)
    return summarize(cfg, rows)


def init_worker(songs, v1_owner, subsets, v1_valid):
    global _SONGS, _V1_OWNER, _SUBSETS, _V1_VALID
    _SONGS, _V1_OWNER, _SUBSETS, _V1_VALID = songs, v1_owner, subsets, v1_valid


# ---------------------------------------------------------------- main


def verify_v1(songs):
    """v1 on the saved track == canonical notes.npz, exactly."""
    report = {"songs": 0, "notes": 0, "mismatching_songs": []}
    for s in songs:
        n = np.load(MAIN / s / "notes.npz", allow_pickle=False)
        segs, idx = v1_segment(_SONGS[s]["times"], _SONGS[s]["f0"], _SONGS[s]["tuning"])
        off, src = n["contour_offsets"], n["contour_source_frame_index"]
        ok = len(segs) == n["note_index"].size and all(
            np.array_equal(idx[i], src[off[i]:off[i + 1]]) for i in range(len(segs)))
        ok = ok and np.array_equal([g["target_note"] for g in segs], n["target_midi_note"])
        ok = ok and np.array_equal(np.asarray([g["core_median_cents"] for g in segs]), n["core_median_cents_deviation"])
        ok = ok and np.array_equal(np.asarray([g["median_cents"] for g in segs]), n["median_cents_deviation"])
        ok = ok and np.array_equal(np.asarray([g["start"] for g in segs]), n["start_time_s"])
        ok = ok and np.array_equal(np.asarray([g["end"] for g in segs]), n["end_time_s"])
        ok = ok and np.array_equal(_SONGS[s]["times"][src], n["contour_time_s"])
        ok = ok and np.array_equal(_SONGS[s]["f0"][src], n["contour_f0_hz"])
        ok = ok and np.array_equal(_SONGS[s]["conf"][src], n["contour_crepe_confidence"])
        # the cents contour the audit metrics use
        tgt = np.repeat(n["target_midi_note"], np.diff(off))
        cents = 1200.0 * np.log2(_SONGS[s]["f0"][src] / np.array([midi_to_hz(m, _SONGS[s]["tuning"]) for m in tgt]))
        ok = ok and np.allclose(cents, n["contour_cents_deviation"], atol=1e-9, rtol=0)
        report["songs"] += 1
        report["notes"] += len(segs)
        if not ok:
            report["mismatching_songs"].append(s)
    return report


def main() -> int:
    global _SONGS, _V1_OWNER, _SUBSETS, _V1_VALID
    songs = songs_axis2()
    assert len(songs) == 50
    _SONGS = load_songs()
    verification = {"v1_reproduction": verify_v1(songs)}
    assert not verification["v1_reproduction"]["mismatching_songs"]

    # v1 per-note rows; must match the tail audit's classification exactly
    v1_rows = []
    for s in songs:
        v1_rows += eval_song(s, dict(name="v1", kind="v1"))
    tail = {(r["song_id"], int(r["source_note_index"])): r
            for r in csv.DictReader(open(TAIL / "segmentation_tail_audit.csv"))}
    mism = 0
    for r in v1_rows:
        t = tail[(r["song_id"], r["position"])]
        same = (r["categories"] == t["categories"] and str(r["any_clear_invalid"]) == t["any_clear_invalid"]
                and all(str(r[k]) == t[k] for k in ("B_clear", "C_clear", "D_clear", "B_likely", "C_likely", "D_likely"))
                and np.isclose(r["full_iqr_cents"], float(t["full_iqr_cents"]), rtol=1e-5, atol=1e-6))
        mism += not same
    verification["tail_classifier_reproduction"] = {"notes": len(v1_rows), "mismatches": mism}
    assert mism == 0, mism

    _V1_OWNER = {s: np.full(_SONGS[s]["times"].size, -1) for s in songs}
    for r in v1_rows:
        _V1_OWNER[r["song_id"]][r["source_frames"]] = r["position"]
    likely = lambda r: r["B_likely"] or r["C_likely"] or r["D_likely"]
    _V1_VALID = {(r["song_id"], r["position"]) for r in v1_rows if not likely(r)}
    clean = lambda r: (not likely(r)) and r["largest_time_gap_s"] <= 0.10 + 1e-9 and r["frame_count"] >= 15
    stable_ex = {(row["song_id"], int(float(row["source_note_index"])))
                 for row in csv.DictReader(open(REPO / "ml/audits/axis2_stable_center/example_notes.csv"))}
    _SUBSETS = {
        "v1_plausible_A": {(r["song_id"], r["position"]) for r in v1_rows if r["categories"] == "A"},
        "v1_unflagged": {(r["song_id"], r["position"]) for r in v1_rows if not r["flagged"] and not likely(r)},
        "vibrato_like": {(r["song_id"], r["position"]) for r in v1_rows
                         if clean(r) and r["frame_count"] >= 20 and r["slow_component_range_cents"] < 100
                         and r["full_iqr_cents"] >= 40 and r["n_center_crossings"] >= 4},
        "scoop": {(r["song_id"], r["position"]) for r in v1_rows if clean(r) and abs(r["onset_minus_core_cents"]) >= 80},
        "fall": {(r["song_id"], r["position"]) for r in v1_rows if clean(r) and abs(r["release_minus_core_cents"]) >= 80},
        "portamento_drift": {(r["song_id"], r["position"]) for r in v1_rows
                             if clean(r) and 100 <= r["slow_component_range_cents"] < 200},
        "stable_center_examples": stable_ex,
    }
    verification["subset_sizes"] = {k: len(v) for k, v in _SUBSETS.items()}
    print(json.dumps(verification, indent=1))

    cfgs = build_configs()
    with Pool(10, initializer=init_worker, initargs=(_SONGS, _V1_OWNER, _SUBSETS, _V1_VALID)) as pool:
        results = pool.map(run_config, cfgs)
    n_v1 = results[0]["total_notes"]
    for r in results:
        r["note_count_change_pct"] = 100 * (r["total_notes"] - n_v1) / n_v1
    with (OUT / "parameter_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        w.writeheader()
        for r in results:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(verification, open(OUT / "_verification.json", "w"), indent=1)
    print(f"{len(results)} configurations written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
