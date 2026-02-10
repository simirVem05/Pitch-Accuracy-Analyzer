from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


def _pick_deviation_cents(seg: Dict) -> float:
    # Prefer core median if available
    for k in ("core_median_cents", "median_cents"):
        v = seg.get(k, None)
        try:
            v = float(v)
        except Exception:
            continue
        if np.isfinite(v):
            return v
    return 0.0


def _intonation_score_from_cents(abs_cents: float) -> float:
    """
    Your spec:
      0–20c => high zone -> 0.80–1.00
      20–30c => mediocre -> 0.60–0.80
      30+c => low -> <0.60 decay
    """
    d = float(abs_cents)

    if d <= 20.0:
        return 1.0 - (0.20 / 20.0) * d  # 1.0 -> 0.8

    if d <= 30.0:
        return 0.80 - (0.20 / 10.0) * (d - 20.0)  # 0.8 -> 0.6

    return float(0.60 * np.exp(-(d - 30.0) / 43.0))


def intonation_tightness_score(seg: Dict) -> float:
    d = abs(_pick_deviation_cents(seg))
    return float(np.clip(_intonation_score_from_cents(d), 0.0, 1.0))


def is_musically_correct(classification: str) -> bool:
    c = (classification or "").strip().lower()
    return c in {"diatonic", "chromatic", "blue", "passing", "neighbor", "leading"}


def _musically_incorrect_score(s_int: float, pc_dist: int) -> float:
    """
    Your rule for musically incorrect notes:
      - mediocre intonation => ~0.50–0.70
      - high intonation => ~0.70–0.90
      - low intonation => <=0.50
    plus mild pitch-class distance reduction.
    """
    s_int = float(np.clip(s_int, 0.0, 1.0))
    pc_dist = int(np.clip(pc_dist, 0, 6))

    if s_int >= 0.80:
        base = 0.70 + (s_int - 0.80) * (0.20 / 0.20)  # -> 0.90
    elif s_int >= 0.60:
        base = 0.50 + (s_int - 0.60) * (0.20 / 0.20)  # -> 0.70
    else:
        base = (s_int / 0.60) * 0.50  # -> 0.50 at 0.60

    base -= 0.02 * pc_dist
    return float(np.clip(base, 0.0, 1.0))


def score_segment(seg: Dict) -> float:
    s_int = intonation_tightness_score(seg)
    cls = str(seg.get("classification", "unknown")).strip().lower()

    if is_musically_correct(cls):
        return s_int

    pc_dist = int(seg.get("pc_distance_to_allowed", 3))
    return _musically_incorrect_score(s_int, pc_dist)


def score_segments(segments: List[Dict], score_scale: str = "fraction") -> Tuple[List[Dict], List[Tuple[float, float]]]:
    tuples: List[Tuple[float, float]] = []
    for seg in segments:
        s = score_segment(seg)
        seg["on_key_score"] = s if score_scale == "fraction" else s * 100.0
        tuples.append((float(seg["start"]), float(seg["on_key_score"])))
    return segments, tuples
