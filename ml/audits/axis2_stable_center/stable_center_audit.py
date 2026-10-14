#!/usr/bin/env python3
"""
Read-only descriptive audit: do Axis 2 V1 clean notes have stable pitch centers?

Analyzes only `variant_index == clean` (22,241 notes). The four corrupted siblings
are a pure constant translation of the clean contour, so every shape statistic
computed here (IQR, MAD, std, drift, within-window fractions, region differences)
is mathematically identical for them and would add no information.

Normalized time comes from the canonical store's `contour_time_s`, not from frame
position. 34% of notes contain interleaved unvoiced frames, so consecutive stored
frames are NOT uniformly spaced; treating index as time would distort region
boundaries and every drift slope on a third of the corpus. `examples.npz` carries
no time array, so the canonical contours are joined on (song_id, source_note_index)
after verifying byte-equality of the cents and confidence contours.

Writes only into this directory. Never opens the dataset for writing.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "data" / "ml_dataset" / "axis2_v1"
EXAMPLES_PATH = DATASET_DIR / "examples.npz"
MANIFEST_CSV = DATASET_DIR / "manifest.csv"
CANONICAL_DIR = REPO_ROOT / "data" / "main_data"


def _v1_notes(song_dir: Path) -> Path:
    """This audit describes v1 notes: notes_v1.npz once the store is re-segmented (schema 2), notes.npz before."""
    p = song_dir / "notes_v1.npz"
    return p if p.exists() else song_dir / "notes.npz"
OUT_DIR = Path(__file__).resolve().parent

# A median needs a real interior to mean anything. Below 3 frames a "region
# median" is one or two samples and its IQR/MAD are degenerate, so region metrics
# are marked unavailable rather than invented.
MIN_REGION_FRAMES = 3
# A least-squares slope through 2 points is exact and meaningless; require 4.
MIN_SLOPE_FRAMES = 4

WINDOWS_CENTS = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
PERCENTILES = (10, 25, 50, 75, 90, 95)

# Ratio classification tolerance. A ratio within +-2% of 1.0 is called
# "approximately equal"; the raw ratio distribution is also reported so the
# conclusion does not rest on this cutoff.
RATIO_TOL = 0.02

FRAME_BUCKETS = [
    ("3-4", 3, 4),
    ("5-6", 5, 6),
    ("7-10", 7, 10),
    ("11-20", 11, 20),
    ("21-50", 21, 50),
    ("51+", 51, 10**9),
]
DURATION_BUCKETS = [
    ("<0.10s", -np.inf, 0.10),
    ("0.10-0.20s", 0.10, 0.20),
    ("0.20-0.40s", 0.20, 0.40),
    ("0.40-0.80s", 0.40, 0.80),
    (">=0.80s", 0.80, np.inf),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def mad(x: np.ndarray) -> float:
    """Median absolute deviation from the region's own median."""
    return float(np.median(np.abs(x - np.median(x))))


def p90_p10(x: np.ndarray) -> float:
    return float(np.percentile(x, 90) - np.percentile(x, 10))


def region_mask(tn: np.ndarray, lo: float, hi: float, include_hi: bool) -> np.ndarray:
    """
    Frames whose normalized time falls in [lo, hi). Uses true normalized time so
    unevenly spaced frames land in the correct region.
    """
    return (tn >= lo) & (tn <= hi) if include_hi else (tn >= lo) & (tn < hi)


def fit_drift(tn: np.ndarray, c: np.ndarray) -> tuple[Optional[float], Optional[float]]:
    """
    Least-squares slope of cents on normalized time, plus the fitted line's
    end-minus-start rise across the frames actually present.

    Slope is per unit normalized note duration. The rise uses the observed span
    of tn, so a region covering only part of the note is not extrapolated.
    """
    if c.size < MIN_SLOPE_FRAMES:
        return None, None
    span = float(tn[-1] - tn[0])
    if span <= 0 or float(np.ptp(tn)) == 0.0:
        return None, None
    slope, _intercept = np.polyfit(tn, c, 1)
    return float(slope), float(slope * span)


def within_fractions(c: np.ndarray, center: float) -> Dict[float, float]:
    dev = np.abs(c - center)
    return {w: float(np.count_nonzero(dev <= w) / dev.size) for w in WINDOWS_CENTS}


def dist(x: np.ndarray) -> Dict[str, Optional[float]]:
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size == 0:
        return {"n": 0}
    out: Dict[str, Optional[float]] = {"n": int(x.size), "min": float(x.min())}
    for p in PERCENTILES:
        out[f"p{p}"] = float(np.percentile(x, p))
    out["max"] = float(x.max())
    out["mean"] = float(x.mean())
    return out


def main() -> int:
    before = {p.name: sha256(p) for p in sorted(DATASET_DIR.iterdir()) if p.is_file()}

    d = np.load(EXAMPLES_PATH, allow_pickle=False)
    variant_names = [str(v) for v in d["variant_names"]]
    split_names = [str(v) for v in d["split_names"]]
    clean_code = variant_names.index("clean")

    clean_idx = np.flatnonzero(d["variant_index"] == clean_code)
    song_ids = [str(v) for v in d["song_ids"]]
    offsets = d["contour_offsets"]
    cents_flat = d["cents_flat"]
    conf_flat = d["confidence_flat"]
    song_index = d["song_index"]
    note_index = d["source_note_index"]
    split_index = d["split_index"]
    duration = d["duration_s"]
    start_s = d["start_s"]
    end_s = d["end_s"]
    target_midi = d["target_midi"]
    num_frames = d["num_frames"]
    stored_median = d["median_cents"]
    stored_core_median = d["core_median_cents"]
    injected = d["injected_shift_cents"]

    assert clean_idx.size == 22241, f"expected 22241 clean notes, got {clean_idx.size}"
    assert np.all(injected[clean_idx] == 0.0), "clean labels must be exactly 0.0"

    # Group by song so each canonical file opens once.
    by_song: Dict[int, List[int]] = defaultdict(list)
    for i in clean_idx:
        by_song[int(song_index[i])].append(int(i))

    rows: List[Dict] = []
    contour_mismatches = 0
    time_nonmonotonic = 0

    for scode in sorted(by_song):
        sid = song_ids[scode]
        canon = np.load(_v1_notes(CANONICAL_DIR / sid), allow_pickle=False)
        c_off = canon["contour_offsets"]
        c_cents = canon["contour_cents_deviation"]
        c_conf = canon["contour_crepe_confidence"]
        c_time = canon["contour_time_s"]

        for i in by_song[scode]:
            ni = int(note_index[i])
            lo, hi = int(offsets[i]), int(offsets[i + 1])
            cents = cents_flat[lo:hi]
            conf = conf_flat[lo:hi]

            clo, chi = int(c_off[ni]), int(c_off[ni + 1])
            k_cents = c_cents[clo:chi]
            k_conf = c_conf[clo:chi]
            t = c_time[clo:chi]

            # The join is only sound if the ML copy is verbatim. Verified per note.
            if k_cents.shape != cents.shape or not np.array_equal(k_cents, cents):
                contour_mismatches += 1
                continue
            if k_conf.shape != conf.shape or not np.array_equal(k_conf, conf):
                contour_mismatches += 1
                continue
            if not np.all(np.diff(t) > 0):
                time_nonmonotonic += 1
                continue

            n = cents.size
            span = float(t[-1] - t[0])
            # Single-frame span cannot be normalized; n>=3 by construction so span>0.
            tn = (t - t[0]) / span if span > 0 else np.zeros(n)

            # Regions by true normalized time. Last region includes its upper edge
            # so the final frame is never dropped.
            m_q1 = region_mask(tn, 0.0, 0.25, False)
            m_mid50 = region_mask(tn, 0.25, 0.75, False)
            m_q4 = region_mask(tn, 0.75, 1.0, True)
            m_t1 = region_mask(tn, 0.0, 1.0 / 3.0, False)
            m_t2 = region_mask(tn, 1.0 / 3.0, 2.0 / 3.0, False)
            m_t3 = region_mask(tn, 2.0 / 3.0, 1.0, True)

            def stat(mask: np.ndarray, minimum: int = MIN_REGION_FRAMES):
                sub = cents[mask]
                if sub.size < minimum:
                    return None, None, None, None, None, int(sub.size)
                return (
                    float(np.median(sub)),
                    iqr(sub),
                    mad(sub),
                    float(np.std(sub, ddof=0)),
                    p90_p10(sub),
                    int(sub.size),
                )

            full_med = float(np.median(cents))
            full_iqr = iqr(cents)
            full_mad = mad(cents)
            full_std = float(np.std(cents, ddof=0))
            full_range = p90_p10(cents)

            mid_med, mid_iqr, mid_mad, mid_std, mid_range, n_mid = stat(m_mid50)
            q1_med, _, _, _, _, n_q1 = stat(m_q1)
            q4_med, _, _, _, _, n_q4 = stat(m_q4)
            mt_med, mt_iqr, mt_mad, _, _, n_mt = stat(m_t2)
            t1_med, _, _, _, _, n_t1 = stat(m_t1)
            t3_med, _, _, _, _, n_t3 = stat(m_t3)

            full_slope, full_rise = fit_drift(tn, cents)
            if n_mid >= MIN_SLOPE_FRAMES:
                mid_slope, mid_rise = fit_drift(tn[m_mid50], cents[m_mid50])
            else:
                mid_slope, mid_rise = None, None

            row: Dict = {
                "song_id": sid,
                "source_note_index": ni,
                "split": split_names[int(split_index[i])],
                "target_midi": int(target_midi[i]),
                "duration_s": float(duration[i]),
                "start_s": float(start_s[i]),
                "end_s": float(end_s[i]),
                "num_frames": int(num_frames[i]),
                "time_span_s": span,
                # A gap means unvoiced frames were skipped inside the note.
                "has_time_gaps": int(bool(np.any(np.diff(t) > 0.0301))),
                "median_confidence": float(np.median(conf)),
                "stored_median_cents": float(stored_median[i]),
                "stored_core_median_cents": float(stored_core_median[i]),
                "full_median_cents": full_med,
                "full_iqr_cents": full_iqr,
                "full_mad_cents": full_mad,
                "full_std_cents": full_std,
                "full_p90_p10_cents": full_range,
                "full_slope_cents_per_dur": full_slope,
                "full_drift_cents": full_rise,
                "n_frames_first_quarter": n_q1,
                "n_frames_middle50": n_mid,
                "n_frames_last_quarter": n_q4,
                "n_frames_middle_third": n_mt,
                "middle50_median_cents": mid_med,
                "middle50_iqr_cents": mid_iqr,
                "middle50_mad_cents": mid_mad,
                "middle50_std_cents": mid_std,
                "middle50_p90_p10_cents": mid_range,
                "middle50_slope_cents_per_dur": mid_slope,
                "middle50_drift_cents": mid_rise,
                "middle_third_median_cents": mt_med,
                "middle_third_iqr_cents": mt_iqr,
                "middle_third_mad_cents": mt_mad,
                "first_quarter_median_cents": q1_med,
                "last_quarter_median_cents": q4_med,
                "first_third_median_cents": t1_med,
                "last_third_median_cents": t3_med,
            }

            row["abs_full_minus_middle50"] = (
                abs(full_med - mid_med) if mid_med is not None else None
            )
            row["onset_to_center"] = (
                mid_med - q1_med if (mid_med is not None and q1_med is not None) else None
            )
            row["release_from_center"] = (
                q4_med - mid_med if (mid_med is not None and q4_med is not None) else None
            )
            row["abs_onset_to_center"] = (
                abs(row["onset_to_center"]) if row["onset_to_center"] is not None else None
            )
            row["abs_release_from_center"] = (
                abs(row["release_from_center"])
                if row["release_from_center"] is not None
                else None
            )
            row["abs_middle50_drift_cents"] = (
                abs(mid_rise) if mid_rise is not None else None
            )
            row["abs_full_drift_cents"] = abs(full_rise) if full_rise is not None else None

            # Ratios only where the full-note denominator is nonzero.
            for name, mv, fv in (
                ("iqr", mid_iqr, full_iqr),
                ("mad", mid_mad, full_mad),
                ("std", mid_std, full_std),
            ):
                row[f"ratio_{name}"] = (
                    (mv / fv) if (mv is not None and fv is not None and fv > 0) else None
                )

            if mid_med is not None:
                fr = within_fractions(cents[m_mid50], mid_med)
                for w, v in fr.items():
                    row[f"mid_frac_within_{int(w)}c"] = v
                fr_full = within_fractions(cents, full_med)
                for w, v in fr_full.items():
                    row[f"full_frac_within_{int(w)}c"] = v
            else:
                for w in WINDOWS_CENTS:
                    row[f"mid_frac_within_{int(w)}c"] = None
                    row[f"full_frac_within_{int(w)}c"] = None

            rows.append(row)

    assert contour_mismatches == 0, f"{contour_mismatches} contours diverged from canonical"
    assert time_nonmonotonic == 0, f"{time_nonmonotonic} non-monotonic time contours"
    assert len(rows) == 22241, f"expected 22241 rows, got {len(rows)}"

    fieldnames = list(rows[0].keys())
    csv_path = OUT_DIR / "stable_center_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})

    after = {p.name: sha256(p) for p in sorted(DATASET_DIR.iterdir()) if p.is_file()}
    integrity = {
        "dataset_files_before": before,
        "dataset_files_after": after,
        "unchanged": before == after,
    }

    stats = {
        "clean_notes_analyzed": len(rows),
        "contour_mismatches_vs_canonical": contour_mismatches,
        "time_nonmonotonic": time_nonmonotonic,
        "integrity": integrity,
    }
    (OUT_DIR / "_audit_rows.json").write_text(
        json.dumps({"n": len(rows)}, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "_integrity.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"rows written: {len(rows)}  -> {csv_path.relative_to(REPO_ROOT)}")
    print(f"dataset unchanged: {integrity['unchanged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
