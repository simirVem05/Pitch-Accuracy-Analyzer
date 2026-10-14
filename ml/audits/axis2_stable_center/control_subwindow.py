#!/usr/bin/env python3
"""
Control: is the middle-50% "improvement" real, or a sample-size artifact?

The IQR/MAD/std of a k-frame subwindow is mechanically smaller than that of the
n-frame whole note simply because k < n, regardless of where the subwindow sits.
So `middle50_IQR < full_IQR` is NOT by itself evidence that the middle is a
plateau.

The control: for each note, compare the middle-50% window against ALL other
contiguous windows of the SAME length from the same contour. If the middle is
genuinely a stable plateau, its dispersion should rank near the bottom of that
distribution. If the improvement is purely a length effect, the middle should
rank near the middle (~50th percentile) of same-length windows.

Read-only. Writes only into this directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[2]
CANONICAL_DIR = REPO_ROOT / "data" / "main_data"


def _v1_notes(song_dir: Path) -> Path:
    """This audit describes v1 notes: notes_v1.npz once the store is re-segmented (schema 2), notes.npz before."""
    p = song_dir / "notes_v1.npz"
    return p if p.exists() else song_dir / "notes.npz"
EXAMPLES = REPO_ROOT / "data" / "ml_dataset" / "axis2_v1" / "examples.npz"

MIN_REGION_FRAMES = 3
PCTS = (10, 25, 50, 75, 90)


def iqr(x):
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def main():
    d = np.load(EXAMPLES, allow_pickle=False)
    vn = [str(v) for v in d["variant_names"]]
    clean = np.flatnonzero(d["variant_index"] == vn.index("clean"))
    song_ids = [str(v) for v in d["song_ids"]]
    si, sni = d["song_index"], d["source_note_index"]

    from collections import defaultdict
    by_song = defaultdict(list)
    for i in clean:
        by_song[int(si[i])].append(int(i))

    pct_ranks, mid_iqrs, best_iqrs, worst_iqrs, med_same_len = [], [], [], [], []
    nframes, ratios_vs_median_window = [], []
    n_eval = 0

    for scode in sorted(by_song):
        sid = song_ids[scode]
        canon = np.load(_v1_notes(CANONICAL_DIR / sid), allow_pickle=False)
        c_off = canon["contour_offsets"]
        c_cents = canon["contour_cents_deviation"]
        c_time = canon["contour_time_s"]

        for i in by_song[scode]:
            ni = int(sni[i])
            lo, hi = int(c_off[ni]), int(c_off[ni + 1])
            cents = c_cents[lo:hi]
            t = c_time[lo:hi]
            n = cents.size
            span = float(t[-1] - t[0])
            if span <= 0:
                continue
            tn = (t - t[0]) / span
            m = (tn >= 0.25) & (tn < 0.75)
            k = int(m.sum())
            if k < MIN_REGION_FRAMES:
                continue
            # need at least one alternative window of the same length
            if n - k < 1:
                continue

            mid_iqr = iqr(cents[m])
            # all contiguous windows of length k
            windows = np.lib.stride_tricks.sliding_window_view(cents, k)
            q75 = np.percentile(windows, 75, axis=1)
            q25 = np.percentile(windows, 25, axis=1)
            all_iqr = q75 - q25

            # percentile rank of the middle window among same-length windows
            rank = float(np.count_nonzero(all_iqr < mid_iqr) / all_iqr.size)
            pct_ranks.append(100 * rank)
            mid_iqrs.append(mid_iqr)
            best_iqrs.append(float(all_iqr.min()))
            worst_iqrs.append(float(all_iqr.max()))
            mw = float(np.median(all_iqr))
            med_same_len.append(mw)
            ratios_vs_median_window.append((mid_iqr / mw) if mw > 0 else np.nan)
            nframes.append(n)
            n_eval += 1

    pct_ranks = np.array(pct_ranks)
    mid_iqrs = np.array(mid_iqrs)
    med_same_len = np.array(med_same_len)
    best_iqrs = np.array(best_iqrs)
    rvm = np.array(ratios_vs_median_window)
    nframes = np.array(nframes)

    def dist(x):
        x = x[np.isfinite(x)]
        o = {"n": int(x.size), "min": float(x.min())}
        for p in PCTS:
            o[f"p{p}"] = float(np.percentile(x, p))
        o["max"] = float(x.max())
        o["mean"] = float(x.mean())
        return o

    res = {
        "notes_evaluated": n_eval,
        "explanation": (
            "pct_rank = percentage of same-length contiguous windows in the same note "
            "whose IQR is SMALLER than the middle-50% window's IQR. 50 means the middle "
            "is typical for its length (no plateau effect). Near 0 means the middle is "
            "genuinely the most stable region."
        ),
        "pct_rank_of_middle_among_same_length_windows": dist(pct_ranks),
        "middle_iqr": dist(mid_iqrs),
        "median_same_length_window_iqr": dist(med_same_len),
        "best_same_length_window_iqr": dist(best_iqrs),
        "ratio_middle_to_median_same_length_window": dist(rvm),
        "pct_middle_better_than_median_same_length_window": float(
            100 * np.count_nonzero(rvm < 1.0) / np.isfinite(rvm).sum()
        ),
        "pct_middle_is_single_best_window": float(
            100 * np.count_nonzero(mid_iqrs <= best_iqrs + 1e-12) / mid_iqrs.size
        ),
    }

    # stratified by frame count
    res["by_frames"] = {}
    for name, lo, hi in [("5-6", 5, 6), ("7-10", 7, 10), ("11-20", 11, 20),
                         ("21-50", 21, 50), ("51+", 51, 10**9)]:
        m = (nframes >= lo) & (nframes <= hi)
        if m.sum() == 0:
            continue
        res["by_frames"][name] = {
            "n": int(m.sum()),
            "median_pct_rank": float(np.median(pct_ranks[m])),
            "median_mid_iqr": float(np.median(mid_iqrs[m])),
            "median_same_len_window_iqr": float(np.median(med_same_len[m])),
            "median_ratio_vs_same_len": float(np.nanmedian(rvm[m])),
            "pct_better_than_median_same_len": float(
                100 * np.count_nonzero(rvm[m] < 1.0) / np.isfinite(rvm[m]).sum()),
        }

    (OUT_DIR / "_control_subwindow.json").write_text(
        json.dumps(res, indent=2) + "\n", encoding="utf-8")

    P = print
    P(f"notes evaluated: {n_eval:,}")
    P()
    P("Percentile rank of middle-50% IQR among SAME-LENGTH windows in the same note")
    P("  (50 = middle is typical for its length; 0 = middle is the most stable region)")
    r = res["pct_rank_of_middle_among_same_length_windows"]
    P(f"  min {r['min']:.1f}  P10 {r['p10']:.1f}  P25 {r['p25']:.1f}  median {r['p50']:.1f}  "
      f"P75 {r['p75']:.1f}  P90 {r['p90']:.1f}  max {r['max']:.1f}  mean {r['mean']:.1f}")
    P()
    P(f"middle IQR vs MEDIAN same-length window IQR:")
    v = res["ratio_middle_to_median_same_length_window"]
    P(f"  ratio median {v['p50']:.4f}  P25 {v['p25']:.4f}  P75 {v['p75']:.4f}")
    P(f"  middle better than median same-length window: {res['pct_middle_better_than_median_same_length_window']:.1f}% of notes")
    P(f"  middle is the single most stable window: {res['pct_middle_is_single_best_window']:.1f}% of notes")
    P()
    P("BY FRAME COUNT")
    P(f"{'bucket':>8} {'n':>7} {'medPctRank':>11} {'midIQR':>8} {'sameLenIQR':>11} {'ratio':>7} {'%better':>8}")
    for k, v in res["by_frames"].items():
        P(f"{k:>8} {v['n']:>7,} {v['median_pct_rank']:>11.1f} {v['median_mid_iqr']:>8.2f} "
          f"{v['median_same_len_window_iqr']:>11.2f} {v['median_ratio_vs_same_len']:>7.3f} "
          f"{v['pct_better_than_median_same_len']:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
