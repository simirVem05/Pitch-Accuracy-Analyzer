from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np


def _pick_deviation_cents(seg: Dict) -> float:
    for k in ("core_median_cents", "median_cents"):
        v = seg.get(k, None)
        try:
            v = float(v)
            if np.isfinite(v): return v
        except: continue
    return 0.0


def _intonation_score_from_cents(abs_cents: float) -> float:
    """Perception Curve optimized for modern R&B stylistic drift."""
    d = float(abs_cents)
    if d <= 25.0: return 1.0 - (0.10 / 25.0) * d 
    if d <= 45.0: return 0.90 - (0.25 / 20.0) * (d - 25.0) 
    return float(0.65 * np.exp(-(d - 45.0) / 40.0))


def score_segment(seg: Dict) -> float:
    d = abs(_pick_deviation_cents(seg))
    s_int = _intonation_score_from_cents(d)
    cls = str(seg.get("classification", "unknown")).strip().lower()

    if cls in {"diatonic", "chromatic", "blue", "passing", "neighbor", "leading"}:
        final = s_int
    else:
        # Soft penalty for unknown notes
        pc_dist = int(seg.get("pc_distance_to_allowed", 3))
        base = 0.70 + (s_int - 0.90) * 2.0 if s_int >= 0.90 else (0.50 + (s_int - 0.65) * 0.8 if s_int >= 0.65 else (s_int / 0.65) * 0.5)
        final = base - (0.02 * pc_dist)

    # 0.05 'Musical Floor' to avoid erratic 0% spikes
    return float(np.clip(final, 0.05, 1.0))


def score_segments(segments: List[Dict], score_scale: str = "fraction") -> Tuple[List[Dict], List[Tuple[float, Optional[float]]]]:
    """Generates smoothed on-key scores and inserts nulls for silences."""
    if not segments: return [], []

    # 1. Compute individual segment scores
    for seg in segments:
        s = score_segment(seg)
        seg["on_key_score"] = s if score_scale == "fraction" else s * 100.0

    # 2. Extract raw data for smoothing
    raw_times = np.array([float(s["start"]) for s in segments])
    raw_scores = np.array([float(s["on_key_score"]) for s in segments])
    
    # 3. Apply 1-second sliding median filter
    smoothed_scores = []
    window = 0.5 # +/- 0.5s
    for i, t in enumerate(raw_times):
        indices = np.where((raw_times >= t - window) & (raw_times <= t + window))[0]
        smoothed_scores.append(float(np.median(raw_scores[indices])))

    # 4. Assemble tuples with Gap-Break Logic
    final_tuples: List[Tuple[float, Optional[float]]] = []
    for i in range(len(segments)):
        seg = segments[i]
        
        # Insert break if gap > 0.1 seconds
        if i > 0:
            prev_end = float(segments[i-1]["end"])
            if (seg["start"] - prev_end) > 0.1:
                # Add a point at the end of the silence to 'lift the pen'
                final_tuples.append((float(seg["start"] - 0.001), None))

        final_tuples.append((float(seg["start"]), smoothed_scores[i]))

    return segments, final_tuples