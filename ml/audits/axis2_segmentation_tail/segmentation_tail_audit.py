#!/usr/bin/env python3
"""
Read-only audit of the extreme tail of Axis 2 clean notes.

Question: are the very wide contours (IQR > 100 c, > 400 c, 15 s "notes") real
expressive single notes, or artifacts of segmentation / CREPE?

Reads only:
  data/dataset_selections.json            (axis2_ml membership)
  data/main_data/<song>/notes.npz         (canonical clean notes)
  data/main_data/<song>/harmony.npz       (full saved vocal track + tuning)
  data/ml_dataset/axis2_v1/manifest.csv   (split membership only; the
                                           injected_shift_cents column is never read)

Writes only into this directory. Nothing is filtered, re-segmented in place, or
re-extracted. The production segmentation loop is re-run on the saved full track
in an instrumented copy, and the copy is asserted to reproduce production's
boundaries exactly before any of its "why did this merge" output is used.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend"))
from note_segmentation import (  # noqa: E402
    DRIFT_CENTS,
    TRIGGER_FRAMES,
    cents_diff,
    hz_to_midi_float,
    midi_to_hz,
    segment_notes_with_frame_indices,
)

OUT = Path(__file__).resolve().parent
MAIN = REPO / "data" / "main_data"
MANIFEST = REPO / "data" / "ml_dataset" / "axis2_v1" / "manifest.csv"
VARIANTS_PER_NOTE = 5
HOP_S = 0.02

# Diagnostic constants. None of these is a proposed production threshold.
SMOOTH_WIN = 9          # ~180 ms rolling median: about one vibrato cycle at 5.5 Hz
REGION_MIN_FRAMES = 6   # a "sustained region" must hold one semitone level >= 120 ms
REGION_SEP_CENTS = 80   # region centers this far apart count as different pitches
CLEAR_REGION_FRAMES = 8     # clear B: two regions each >= 160 ms ...
CLEAR_REGION_SEP = 100      # ... at least a full semitone apart
OUTLIER_DEV_CENTS = 200     # frame this far from its local median = excursion frame
EXCURSION_MAX_FRAMES = 3    # excursions this short that return = CREPE-style blip
GAP_LIKELY_S = 0.25
GAP_CLEAR_S = 0.50
PLATEAU_MIN_FRAMES = 3      # 60 ms: shorter than the 4-frame commit rule can ever split
SLOW_RANGE_CENTS = 200      # range of the ~180 ms rolling median; vibrato barely moves it
PLATEAU_SEP_CENTS = 150


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    h = win // 2
    return np.array([np.median(x[max(0, i - h): i + h + 1]) for i in range(x.size)])


def runs(values: np.ndarray):
    """(start, end_exclusive, value) for each maximal run of equal values."""
    out, i = [], 0
    while i < values.size:
        j = i
        while j < values.size and values[j] == values[i]:
            j += 1
        out.append((i, j, values[i]))
        i = j
    return out


def pitch_regions(cents: np.ndarray):
    """
    Sustained pitch regions: runs where the ~180 ms rolling median stays on one
    semitone level for >= REGION_MIN_FRAMES frames. Adjacent kept runs whose raw
    centers are < REGION_SEP_CENTS apart are merged (vibrato straddling a level
    boundary). Returns list of (start, end, center_cents).
    """
    if cents.size < REGION_MIN_FRAMES:
        return [(0, cents.size, float(np.median(cents)))]
    sm = rolling_median(cents, SMOOTH_WIN)
    level = np.round(sm / 100.0).astype(int)
    kept = [(a, b) for a, b, _ in runs(level) if b - a >= REGION_MIN_FRAMES]
    regions = []
    for a, b in kept:
        c = float(np.median(cents[a:b]))
        if regions and abs(c - regions[-1][2]) < REGION_SEP_CENTS:
            a0 = regions[-1][0]
            regions[-1] = (a0, b, float(np.median(cents[a0:b])))
        else:
            regions.append((a, b, c))
    return regions


def plateaus(cents: np.ndarray):
    """
    Short pitch plateaus on the RAW contour: maximal runs of >= PLATEAU_MIN_FRAMES
    consecutive frames on one semitone level, adjacent same-level runs merged.
    Catches melisma notes (60-100 ms) that the smoothed region detector blurs.
    Vibrato also produces level runs, so this is only used together with the
    slow-component range, which vibrato does not move.
    """
    level = np.round(cents / 100.0).astype(int)
    out = []
    for a, b, v in runs(level):
        if b - a >= PLATEAU_MIN_FRAMES:
            c = float(np.median(cents[a:b]))
            if out and out[-1][3] == v:
                out[-1] = (out[-1][0], b, float(np.median(cents[out[-1][0]:b])), v)
            else:
                out.append((a, b, c, v))
    return [(a, b, c) for a, b, c, _ in out]


def excursions(cents: np.ndarray):
    """
    Short excursions: maximal runs of frames > OUTLIER_DEV_CENTS from the local
    median that last <= EXCURSION_MAX_FRAMES and have non-excursion frames on
    both sides (i.e. the contour came back).
    """
    if cents.size < 5:
        return [], np.zeros(cents.size, bool)
    dev = np.abs(cents - rolling_median(cents, SMOOTH_WIN))
    bad = dev > OUTLIER_DEV_CENTS
    ex = []
    for a, b, v in runs(bad):
        if v and (b - a) <= EXCURSION_MAX_FRAMES and a > 0 and b < cents.size:
            ex.append((a, b, float(np.max(dev[a:b]))))
    return ex, bad


def instrumented_segment(times, freqs, tuning):
    """
    Exact copy of segment_notes_with_frame_indices' state machine, recording why
    every off-note candidate run was abandoned and the current note at every
    voiced frame. Returns (segment position ranges, per-position log).
    """
    voiced = ~np.isnan(freqs)
    vp = np.flatnonzero(voiced)
    cur = int(np.round(hz_to_midi_float(freqs[vp[0]], tuning)))
    cand, cstart, ccount = None, None, 0
    seg_start = 0
    bounds = []
    log = {"cur": np.empty(vp.size, int), "drift": np.zeros(vp.size), "event": [""] * vp.size}
    log["cur"][0] = cur

    def abandon(pos, reason):
        if cand is not None:
            log["event"][pos] = f"abort:{reason}:{ccount}"

    for pos in range(1, vp.size):
        i = vp[pos]
        drift = abs(float(cents_diff(np.array([freqs[i]]), midi_to_hz(cur, tuning))[0]))
        log["cur"][pos] = cur
        log["drift"][pos] = drift
        if drift < DRIFT_CENTS:
            abandon(pos, "returned_within_40c")
            cand, cstart, ccount = None, None, 0
            continue
        prop = int(np.round(hz_to_midi_float(freqs[i], tuning)))
        if prop == cur:
            abandon(pos, "rounds_to_current_note")
            cand, cstart, ccount = None, None, 0
            continue
        if cand != prop:
            abandon(pos, "proposal_changed")
            cand, cstart, ccount = prop, pos, 1
        else:
            ccount += 1
        if ccount > TRIGGER_FRAMES:
            bounds.append((seg_start, cstart))
            seg_start = cstart
            cur = cand
            log["event"][pos] = "commit"
            cand, cstart, ccount = None, None, 0
    bounds.append((seg_start, vp.size))
    return vp, bounds, log


def pct(a, q):
    return float(np.percentile(a, q)) if a.size else float("nan")


def load_splits():
    split = {}
    with MANIFEST.open() as fh:
        for row in csv.DictReader(fh):
            if row["variant"] == "clean":
                split[(row["song_id"], int(row["source_note_index"]))] = (row["split"], int(row["num_frames"]))
    return split


def artist_of(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return stem.split(" - ")[0].strip() if " - " in stem else "unknown"


def main() -> int:
    sel = json.load(open(REPO / "data" / "dataset_selections.json"))["songs"]
    songs = sorted(s for s, v in sel.items() if v["axis2_ml"])
    splits = load_splits()

    rows, contours = [], {}
    abort_totals = Counter()

    for song in songs:
        n = np.load(MAIN / song / "notes.npz", allow_pickle=False)
        h = np.load(MAIN / song / "harmony.npz", allow_pickle=False)
        tuning = float(h["tuning_offset_semitones"])
        ftimes, ff0 = h["vocal_frame_times_s"], h["vocal_f0_hz"]

        # Production must reproduce the saved notes, and the instrumented copy
        # must reproduce production, or nothing below is trustworthy.
        _, prod_idx = segment_notes_with_frame_indices(ftimes, ff0, tuning)
        vp, bounds, log = instrumented_segment(ftimes, ff0, tuning)
        off, src = n["contour_offsets"], n["contour_source_frame_index"]
        inst_idx = [vp[a:b] for a, b in bounds]
        inst_idx = [x for x in inst_idx if x.size >= 3]
        assert len(prod_idx) == len(n["note_index"]) == len(inst_idx), song
        pos_of = {int(f): p for p, f in enumerate(vp)}

        meta = json.load(open(MAIN / song / "metadata.json"))
        artist = artist_of(meta["original_filename"])

        for k in range(len(n["note_index"])):
            sl = slice(off[k], off[k + 1])
            fi, t = src[sl], n["contour_time_s"][sl]
            c, conf = n["contour_cents_deviation"][sl], n["contour_crepe_confidence"][sl]
            assert np.array_equal(fi, prod_idx[k]) and np.array_equal(fi, inst_idx[k])
            note_idx = int(n["note_index"][k])
            split, mf = splits[(song, note_idx)]
            assert mf == fi.size

            nf = fi.size
            dur = float(n["duration_s"][k])
            dc = np.abs(np.diff(c))
            dt = np.diff(t)
            fgap = np.diff(fi)
            span = int(fi[-1] - fi[0] + 1)
            q25, q75 = np.percentile(c, [25, 75])
            iqr = float(q75 - q25)

            # islands = runs of consecutive source frames
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
            jump_gap = float(dc[~consec].max()) if (~consec).any() else 0.0
            # CREPE confidence on the unvoiced frames inside the largest gap
            if fgap.size and fgap.max() > 1:
                g = int(np.argmax(fgap))
                gap_conf = h["vocal_crepe_confidence"][fi[g] + 1: fi[g + 1]]
                gap_conf_med, gap_conf_max = float(np.median(gap_conf)), float(gap_conf.max())
            else:
                gap_conf_med = gap_conf_max = float("nan")

            exc, badmask = excursions(c)
            octave_jumps = int(np.sum((dc >= 1000) & (dc <= 1400)))
            octave_jumps_consec = int(np.sum((dc >= 1000) & (dc <= 1400) & (fgap == 1)))
            if badmask.any() and (~badmask).sum() >= 4:
                q25c, q75c = np.percentile(c[~badmask], [25, 75])
                iqr_clean = float(q75c - q25c)
                range_clean = float(np.ptp(c[~badmask]))
            else:
                iqr_clean, range_clean = iqr, float(np.ptp(c))

            j = int(np.argmax(dc)) if dc.size else 0
            # segmentation log over this note's voiced positions
            p0, p1 = pos_of[int(fi[0])], pos_of[int(fi[-1])] + 1
            ev = [e for e in log["event"][p0 + 1: p1] if e.startswith("abort")]
            aborts = Counter(e.split(":")[1] for e in ev)
            abort_totals.update(aborts)
            max_cand_run = max([int(e.split(":")[2]) for e in ev], default=0)
            off_frames = int(np.sum(log["drift"][p0 + 1: p1] >= DRIFT_CENTS))
            # drift of the first frame after the largest gap vs the note being tracked
            gap_pos = int(np.argmax(fgap)) + 1 if fgap.size else 0
            post_gap_drift = float(log["drift"][p0 + gap_pos]) if gap_pos else float("nan")
            # gaps whose two sides sit at different pitch centers
            # pitch step across ANY unvoiced break, between islands of >= 3 frames
            isl_steps = [abs(isl_med[i + 1] - isl_med[i]) for i in range(len(isl) - 1)
                         if isl_len[i] >= 3 and isl_len[i + 1] >= 3]
            max_isl_step = max(isl_steps, default=0.0)
            octave_like_isl = any(abs(s - 1200) <= 150 for s in isl_steps)
            gap_pitch_steps = [abs(isl_med[i + 1] - isl_med[i]) for i in range(len(isl) - 1)
                               if fgap[np.flatnonzero(fgap > 1)[i]] * HOP_S > GAP_LIKELY_S]

            r = dict(
                song_id=song, artist=artist, split=split, source_note_index=note_idx,
                start_s=float(n["start_time_s"][k]), end_s=float(n["end_time_s"][k]),
                duration_s=dur, frame_count=nf,
                frame_density=nf / dur if dur > 0 else float("nan"),
                target_midi=int(n["target_midi_note"][k]),
                median_cents=float(n["median_cents_deviation"][k]),
                core_median_cents=float(n["core_median_cents_deviation"][k]),
                full_iqr_cents=iqr,
                p5_p95_range_cents=float(np.subtract(*np.percentile(c, [95, 5]))),
                full_range_cents=float(np.ptp(c)),
                max_adjacent_jump_cents=float(dc.max()) if dc.size else 0.0,
                median_adjacent_jump_cents=float(np.median(dc)) if dc.size else 0.0,
                p90_adjacent_jump_cents=pct(dc, 90),
                n_jumps_gt50=int(np.sum(dc > 50)), n_jumps_gt100=int(np.sum(dc > 100)),
                n_jumps_gt200=int(np.sum(dc > 200)), n_jumps_gt300=int(np.sum(dc > 300)),
                largest_time_gap_s=float(dt.max()) if dt.size else 0.0,
                median_time_gap_s=float(np.median(dt)) if dt.size else 0.0,
                n_nonconsecutive_gaps=int(np.sum(fgap > 1)),
                frames_consecutive=bool(np.all(fgap == 1)),
                voiced_time_s=nf * HOP_S, elapsed_span_s=span * HOP_S,
                voiced_to_elapsed_ratio=nf / span,
                n_islands=len(isl), island_medians=";".join(f"{m:.0f}" for m in isl_med),
                island_lengths=";".join(map(str, isl_len)),
                max_cross_gap_pitch_step_cents=max(gap_pitch_steps, default=0.0),
                max_island_step_cents=max_isl_step, octave_like_island_step=octave_like_isl,
                median_confidence=float(np.median(conf)), min_confidence=float(conf.min()),
                extreme_jump_confidence=float(min(conf[j], conf[j + 1])) if dc.size else float("nan"),
                confidence_before_jump=float(conf[j]) if dc.size else float("nan"),
                confidence_after_jump=float(conf[j + 1]) if dc.size else float("nan"),
                apparent_pitch_region_count=len(regs),
                region_centers=";".join(f"{x:.0f}" for x in centers),
                region_lengths=";".join(map(str, reg_lens)),
                region_spread_cents=reg_spread, clear_region_spread_cents=float(big_spread),
                slow_component_range_cents=slow_range,
                n_plateaus=len(plats), plateau_centers=";".join(f"{x:.0f}" for x in pc_),
                plateau_lengths=";".join(str(p[1] - p[0]) for p in plats),
                plateau_spread_cents=plat_spread, plateau4_spread_cents=plat4_spread,
                max_jump_consecutive_frames=jump_consec, max_jump_across_gap=jump_gap,
                largest_gap_median_confidence=gap_conf_med, largest_gap_max_confidence=gap_conf_max,
                n_short_excursions=len(exc), n_excursion_frames=int(badmask.sum()),
                n_octave_jumps=octave_jumps, n_octave_jumps_consecutive=octave_jumps_consec,
                iqr_without_excursions=iqr_clean, range_without_excursions=range_clean,
                seg_off_note_frames=off_frames, seg_max_candidate_run=max_cand_run,
                seg_abort_returned=aborts["returned_within_40c"],
                seg_abort_rounds_to_current=aborts["rounds_to_current_note"],
                seg_abort_proposal_changed=aborts["proposal_changed"],
                post_largest_gap_drift_cents=post_gap_drift,
            )
            rows.append(r)
            contours[(song, note_idx)] = (t, c, conf, fi)

    for r in rows:
        classify(r)

    write_csv(OUT / "segmentation_tail_audit.csv", rows)
    write_examples(rows, contours)
    json.dump(summary(rows, abort_totals), open(OUT / "_summary_stats.json", "w"), indent=1)
    print(f"{len(rows)} notes, {sum(r['flagged'] for r in rows)} flagged")
    return 0


FLAG_RULES = [
    ("iqr_gt100", lambda r: r["full_iqr_cents"] > 100),
    ("iqr_gt200", lambda r: r["full_iqr_cents"] > 200),
    ("iqr_gt400", lambda r: r["full_iqr_cents"] > 400),
    ("p5p95_gt200", lambda r: r["p5_p95_range_cents"] > 200),
    ("jump_gt100", lambda r: r["max_adjacent_jump_cents"] > 100),
    ("jump_gt200", lambda r: r["max_adjacent_jump_cents"] > 200),
    ("gap_gt0.10", lambda r: r["largest_time_gap_s"] > 0.10 + 1e-9),
    ("gap_gt0.25", lambda r: r["largest_time_gap_s"] > 0.25 + 1e-9),
    ("dur_gt2", lambda r: r["duration_s"] > 2),
    ("dur_gt5", lambda r: r["duration_s"] > 5),
    ("dur_gt10", lambda r: r["duration_s"] > 10),
]


def classify(r):
    """
    Transparent rule-based diagnostic categories. Multi-label. 'clear' variants
    are the stricter subsets used for the exclusion scenarios.
    """
    r["flags"] = ";".join(name for name, f in FLAG_RULES if f(r))
    r["flagged"] = bool(r["flags"])
    cats, why = [], []

    # D: the note bridges a long unvoiced gap.
    d = r["largest_time_gap_s"] > GAP_LIKELY_S + 1e-9
    d_clear = r["largest_time_gap_s"] > GAP_CLEAR_S + 1e-9 or (
        d and r["max_cross_gap_pitch_step_cents"] >= CLEAR_REGION_SEP)
    if d:
        cats.append("D")
        why.append(f"{r['largest_time_gap_s']:.2f}s unvoiced gap inside note "
                   f"({r['n_islands']} voiced islands, medians {r['island_medians']} c)")

    # C: octave jumps between consecutive frames, or short returning excursions
    # that account for most of the note's range.
    exc_dominates = (r["n_short_excursions"] > 0 and r["full_range_cents"] > 0 and
                     r["range_without_excursions"] < 0.5 * r["full_range_cents"])
    c = r["n_octave_jumps_consecutive"] > 0 or exc_dominates or r["octave_like_island_step"]
    c_clear = r["n_octave_jumps_consecutive"] > 0 or (
        exc_dominates and r["n_excursion_frames"] <= 3 and r["max_jump_consecutive_frames"] > 300)
    if c:
        cats.append("C")
        why.append(f"{r['n_short_excursions']} short excursion(s), {r['n_octave_jumps_consecutive']} "
                   f"consecutive-frame octave jump(s), octave-sized step across a break: "
                   f"{r['octave_like_island_step']}; range {r['full_range_cents']:.0f}->"
                   f"{r['range_without_excursions']:.0f} c without excursions")

    # B: two or more pitch centers. Either sustained (smoothed) regions, or
    # short raw plateaus (melisma) that also move the slow component, which
    # vibrato does not.
    b_sus = r["apparent_pitch_region_count"] >= 2 and r["region_spread_cents"] >= REGION_SEP_CENTS
    b_fast = (r["slow_component_range_cents"] >= SLOW_RANGE_CENTS
              and r["plateau_spread_cents"] >= PLATEAU_SEP_CENTS)
    b_break = r["max_island_step_cents"] >= 300
    b = b_sus or b_fast or b_break
    b_clear = r["clear_region_spread_cents"] >= CLEAR_REGION_SEP or b_break or (
        r["slow_component_range_cents"] >= 300 and r["plateau4_spread_cents"] >= 200)
    if b:
        cats.append("B")
        why.append(f"pitch plateaus at {r['plateau_centers']} c (lengths {r['plateau_lengths']} frames), "
                   f"slow-component range {r['slow_component_range_cents']:.0f} c, "
                   f"max step across an unvoiced break {r['max_island_step_cents']:.0f} c")

    if r["flagged"] and not cats:
        if r["slow_component_range_cents"] < SLOW_RANGE_CENTS and r["full_iqr_cents"] <= 200:
            cats.append("A")
            why.append("single sustained region, continuous contour (vibrato/scoop/drift scale)")
        else:
            cats.append("E")
            why.append(f"wide (IQR {r['full_iqr_cents']:.0f} c, max jump {r['max_adjacent_jump_cents']:.0f} c) "
                       "but no sustained second region, gap, or short excursion")

    r["categories"] = "".join(cats) if r["flagged"] else ""
    r["category_list"] = json.dumps(cats if r["flagged"] else [])
    r["classification_reason"] = " | ".join(why) if r["flagged"] else ""
    r["B_likely"], r["C_likely"], r["D_likely"] = b, c, d
    r["B_clear"], r["C_clear"], r["D_clear"] = b_clear, c_clear, d_clear
    r["any_clear_invalid"] = b_clear or c_clear or d_clear


def write_csv(path, rows, cols=None):
    cols = cols or list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


EXAMPLE_COLS = ["selection", "song_id", "source_note_index", "split", "start_s", "end_s", "duration_s",
                "frame_count", "target_midi", "median_cents", "full_iqr_cents", "p5_p95_range_cents",
                "max_adjacent_jump_cents", "largest_time_gap_s", "n_islands", "median_confidence",
                "apparent_pitch_region_count", "region_centers", "seg_max_candidate_run",
                "seg_abort_returned", "seg_abort_rounds_to_current", "seg_abort_proposal_changed",
                "post_largest_gap_drift_cents", "categories", "classification_reason"]


def write_examples(rows, contours):
    picks = {}
    for label, key in [("top_iqr", "full_iqr_cents"), ("top_jump", "max_adjacent_jump_cents"),
                       ("top_gap", "largest_time_gap_s"), ("top_duration", "duration_s")]:
        for r in sorted(rows, key=lambda r: -r[key])[:20]:
            k = (r["song_id"], r["source_note_index"])
            picks.setdefault(k, dict(r, selection=[]))["selection"].append(label)
    ex = sorted(picks.values(), key=lambda r: (r["song_id"], r["source_note_index"]))
    for r in ex:
        r["selection"] = "+".join(r["selection"])
    write_csv(OUT / "extreme_examples.csv", ex, EXAMPLE_COLS)
    with (OUT / "extreme_contours.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["song_id", "source_note_index", "frame_position", "source_frame_index",
                    "time_s", "cents_deviation", "crepe_confidence"])
        for r in ex:
            t, c, conf, fi = contours[(r["song_id"], r["source_note_index"])]
            for p in range(t.size):
                w.writerow([r["song_id"], r["source_note_index"], p, int(fi[p]),
                            f"{t[p]:.4f}", f"{c[p]:.3f}", f"{conf[p]:.4f}"])
    print(f"{len(ex)} deduplicated examples")


def summary(rows, abort_totals):
    N = len(rows)
    a = lambda k, sub=rows: np.array([r[k] for r in sub], float)
    flagged = [r for r in rows if r["flagged"]]
    S = {"n_notes": N, "n_flagged": len(flagged), "abort_reason_totals": dict(abort_totals)}
    S["flag_counts"] = {name: sum(f(r) for r in rows) for name, f in FLAG_RULES}

    def dist(x):
        return {q: float(np.percentile(x, q)) for q in (0, 5, 10, 25, 50, 75, 90, 95, 99, 100)}
    for k in ["largest_time_gap_s", "n_nonconsecutive_gaps", "voiced_to_elapsed_ratio",
              "max_adjacent_jump_cents", "p90_adjacent_jump_cents", "median_adjacent_jump_cents",
              "n_jumps_gt50", "n_jumps_gt100", "n_jumps_gt200", "n_jumps_gt300",
              "apparent_pitch_region_count", "full_iqr_cents", "duration_s", "frame_count"]:
        S.setdefault("dist_flagged", {})[k] = dist(a(k, flagged))
        S.setdefault("dist_all", {})[k] = dist(a(k))
    S["frames_consecutive_all"] = sum(r["frames_consecutive"] for r in rows)
    S["frames_consecutive_flagged"] = sum(r["frames_consecutive"] for r in flagged)

    cat = Counter()
    for r in flagged:
        for ch in json.loads(r["category_list"]):
            cat[ch] += 1
    S["category_counts"] = {k: {"count": v, "pct_flagged": 100 * v / len(flagged), "pct_all": 100 * v / N}
                            for k, v in sorted(cat.items())}
    S["category_combos"] = dict(Counter(r["categories"] for r in flagged).most_common())
    for k in ["B_likely", "C_likely", "D_likely", "B_clear", "C_clear", "D_clear", "any_clear_invalid"]:
        S.setdefault("all_note_rates", {})[k] = {"count": sum(r[k] for r in rows),
                                                  "pct_all": 100 * sum(r[k] for r in rows) / N}

    for thr in (100, 400):
        sub = [r for r in rows if r["full_iqr_cents"] > thr]
        S[f"iqr_gt{thr}_breakdown"] = {
            "n": len(sub),
            "combos": dict(Counter(r["categories"] for r in sub).most_common()),
            "any_clear_invalid": sum(r["any_clear_invalid"] for r in sub),
            "B_clear": sum(r["B_clear"] for r in sub), "C_clear": sum(r["C_clear"] for r in sub),
            "D_clear": sum(r["D_clear"] for r in sub),
            "B_likely": sum(r["B_likely"] for r in sub), "D_likely": sum(r["D_likely"] for r in sub),
            "C_likely": sum(r["C_likely"] for r in sub),
            "slow_range_ge200": sum(r["slow_component_range_cents"] >= 200 for r in sub),
            "dominant_abort": dict(Counter(max(("returned", r["seg_abort_returned"]),
                                               ("rounds", r["seg_abort_rounds_to_current"]),
                                               ("changed", r["seg_abort_proposal_changed"]),
                                               key=lambda x: x[1])[0] for r in sub)),
        }

    scen = {"S1_clear_B": lambda r: r["B_clear"],
            "S2_clear_B_or_D": lambda r: r["B_clear"] or r["D_clear"],
            "S3_clear_B_C_D": lambda r: r["any_clear_invalid"],
            "S3b_any_likely_BCD": lambda r: r["B_likely"] or r["C_likely"] or r["D_likely"]}
    split_tot = Counter(r["split"] for r in rows)
    for name, f in scen.items():
        sub = [r for r in rows if f(r)]
        bys = Counter(r["split"] for r in sub)
        S.setdefault("scenarios", {})[name] = {
            "notes": len(sub), "examples": VARIANTS_PER_NOTE * len(sub), "pct_notes": 100 * len(sub) / N,
            "by_split": {s: {"notes": bys[s], "examples": VARIANTS_PER_NOTE * bys[s],
                             "pct_of_split": 100 * bys[s] / split_tot[s]} for s in ("train", "validation", "test")},
            "frames_removed": int(sum(r["frame_count"] for r in sub)),
        }
    S["total_frames"] = int(sum(r["frame_count"] for r in rows))

    dsub = [r for r in rows if r["D_likely"]]
    S["D_detail"] = {
        "n": len(dsub),
        "cross_gap_step_lt80": sum(r["max_cross_gap_pitch_step_cents"] < 80 for r in dsub),
        "cross_gap_step_ge100": sum(r["max_cross_gap_pitch_step_cents"] >= 100 for r in dsub),
        "gap_gt0.5": sum(r["largest_time_gap_s"] > 0.5 for r in dsub),
        "gap_gt1": sum(r["largest_time_gap_s"] > 1 for r in dsub),
        "gap_gt2": sum(r["largest_time_gap_s"] > 2 for r in dsub),
        "gap_median_conf": dist(a("largest_gap_median_confidence", dsub)),
        "gap_max_conf": dist(a("largest_gap_max_confidence", dsub)),
        "islands": dist(a("n_islands", dsub)),
    }
    dur, vt = a("duration_s"), a("voiced_time_s")
    for k in ["D_likely", "D_clear", "any_clear_invalid"]:
        m = np.array([r[k] for r in rows])
        S.setdefault("weight_share", {})[k] = {
            "pct_notes": 100 * m.mean(), "pct_duration_s_weight": 100 * dur[m].sum() / dur.sum(),
            "pct_voiced_time": 100 * vt[m].sum() / vt.sum()}
    S["jump_gt100_where"] = {
        "any": sum(r["max_adjacent_jump_cents"] > 100 for r in rows),
        "within_consecutive_frames": sum(r["max_jump_consecutive_frames"] > 100 for r in rows),
        "only_across_gaps": sum(r["max_jump_across_gap"] > 100 and r["max_jump_consecutive_frames"] <= 100
                                for r in rows)}
    unf = [r for r in rows if not r["flagged"]]
    S["unflagged_rule_hits"] = {k: sum(r[k] for r in unf) for k in ("B_likely", "C_likely", "D_likely")}
    low = [r for r in rows if r["full_iqr_cents"] <= 100]
    S["iqr_le100_rule_hits"] = {"n": len(low), **{k: sum(r[k] for r in low) for k in
                                                   ("B_likely", "B_clear", "C_likely", "C_clear")}}

    def bucketed(keyf, order):
        g = defaultdict(list)
        for r in rows:
            g[keyf(r)].append(r)
        return {b: {"n": len(g[b]),
                    "pct_clear_invalid": 100 * np.mean([r["any_clear_invalid"] for r in g[b]]),
                    "pct_B_clear": 100 * np.mean([r["B_clear"] for r in g[b]]),
                    "pct_D_clear": 100 * np.mean([r["D_clear"] for r in g[b]]),
                    "pct_C_clear": 100 * np.mean([r["C_clear"] for r in g[b]]),
                    "pct_iqr_gt400": 100 * np.mean([r["full_iqr_cents"] > 400 for r in g[b]])}
                for b in order if g[b]}

    def bk(x, edges, labels):
        for e, l in zip(edges, labels):
            if x < e:
                return l
        return labels[-1]
    dl = ["<0.1", "0.1-0.25", "0.25-0.5", "0.5-1", "1-2", "2-5", "5-10", ">=10"]
    S["by_duration"] = bucketed(lambda r: bk(r["duration_s"], [0.1, 0.25, 0.5, 1, 2, 5, 10], dl), dl)
    fl = ["3-4", "5-9", "10-19", "20-49", "50-99", ">=100"]
    S["by_frames"] = bucketed(lambda r: bk(r["frame_count"], [5, 10, 20, 50, 100], fl), fl)
    cl = ["<0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", ">=0.95"]
    S["by_confidence"] = bucketed(lambda r: bk(r["median_confidence"], [0.7, 0.8, 0.9, 0.95], cl), cl)
    songs = sorted({r["song_id"] for r in rows})
    S["by_song"] = bucketed(lambda r: r["song_id"], songs)
    for s in songs:
        S["by_song"][s]["split"] = next(r["split"] for r in rows if r["song_id"] == s)
    artists = sorted({r["artist"] for r in rows})
    S["by_artist"] = bucketed(lambda r: r["artist"], artists)
    return S


if __name__ == "__main__":
    raise SystemExit(main())
