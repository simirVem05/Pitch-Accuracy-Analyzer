#!/usr/bin/env python3
"""
Targeted interpretation check for the natural-offset experiment (TRAIN clean notes only).

Question: is the shape signal that predicts N tied to the semitone grid?

N is a rounding residual: target = round(core median MIDI), so N lies in [-50, +50].
Segmentation (v1's commit rule, kept inside each v2 gap piece) splits a note once
frames sit >= 40 c from the current target and round to a different semitone for
more than 3 consecutive frames. If that truncates contours relative to the grid,
the room a centered contour has above its center should shrink as N -> +50
(headroom 50 - N) and the room below should shrink as N -> -50. A purely musical
effect has no reason to bend at the grid boundary.

Reports, by N bin: mean centered P90 and P10 (and their sum, the asymmetry), mean
centered onset/release medians, fraction of frames beyond the grid boundary
(|uncentered cents| > 50), and the next-interval direction. Validation and test are
not read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "backend"))
from note_segmentation import _core_slice  # noqa: E402

BINS = [(-50, -40), (-40, -25), (-25, -10), (-10, 10), (10, 25), (25, 40), (40, 50.0001)]


def main() -> int:
    with np.load(REPO / "data/ml_dataset/axis2_v2/examples.npz", allow_pickle=False) as z:
        sp, vi, off = z["split_index"], z["variant_index"], z["contour_offsets"]
        cf, si, ni, ids = z["cents_flat"], z["song_index"], z["source_note_index"], z["song_ids"]
        idx = np.flatnonzero((sp == 0) & (vi == 0))          # train clean only
        midi = z["target_midi"]
    ctx = {}
    rows = []
    for i in idx:
        c = cf[off[i]:off[i + 1]]
        n_ = float(np.median(_core_slice(c)))
        cc = c - n_
        n = c.size
        k = max(1, int(np.floor(0.2 * n)))
        s = int(si[i])
        if s not in ctx:
            with np.load(REPO / "data/main_data" / str(ids[s]) / "notes.npz", allow_pickle=False) as nn:
                ctx[s] = nn["target_midi_note"].astype(int)
        m = ctx[s]
        j = int(ni[i])
        nxt = (m[j + 1] - m[j]) if j + 1 < m.size else 0
        rows.append((n_, float(np.percentile(cc, 90)), float(np.percentile(cc, 10)), float(np.median(cc[:k])),
                     float(np.median(cc[n - k:])), float(np.mean(np.abs(c) > 50)), np.sign(nxt), n))
    a = np.array(rows)
    N = a[:, 0]
    out = {"population": "train clean notes", "notes": int(len(a)), "bins": []}
    for lo, hi in BINS:
        mk = (N >= lo) & (N < hi)
        b = a[mk]
        out["bins"].append({
            "N_range": [lo, round(hi)], "notes": int(mk.sum()),
            "mean_centered_p90": float(b[:, 1].mean()), "mean_centered_p10": float(b[:, 2].mean()),
            "mean_asymmetry_p90_plus_p10": float((b[:, 1] + b[:, 2]).mean()),
            "headroom_above_center_50_minus_N": float(50 - b[:, 0].mean()),
            "headroom_below_center_50_plus_N": float(50 + b[:, 0].mean()),
            "mean_centered_onset_median": float(b[:, 3].mean()), "mean_centered_release_median": float(b[:, 4].mean()),
            "fraction_frames_beyond_grid_50c": float(b[:, 5].mean()),
            "next_interval_up_fraction": float(np.mean(b[:, 6] > 0)), "next_interval_down_fraction": float(np.mean(b[:, 6] < 0)),
            "mean_frames": float(b[:, 7].mean()),
        })
    long_ = a[:, 7] >= 10
    out["correlations_with_N"] = {
        "centered_p90": float(np.corrcoef(N, a[:, 1])[0, 1]), "centered_p10": float(np.corrcoef(N, a[:, 2])[0, 1]),
        "asymmetry_p90_plus_p10": float(np.corrcoef(N, a[:, 1] + a[:, 2])[0, 1]),
        "centered_onset_median": float(np.corrcoef(N, a[:, 3])[0, 1]),
        "centered_release_median": float(np.corrcoef(N, a[:, 4])[0, 1]),
        "next_interval_sign": float(np.corrcoef(N, a[:, 6])[0, 1]),
        "asymmetry_notes_ge_10_frames": float(np.corrcoef(N[long_], (a[:, 1] + a[:, 2])[long_])[0, 1]),
    }
    (OUT / "grid_coupling_diagnostic.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
