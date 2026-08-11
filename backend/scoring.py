from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


SILENCE_BREAK_S = 0.1
SMOOTHING_HALF_WINDOW_S = 0.5

SCORE_FLOOR = 0.05


def _deviation_cents(seg: Dict) -> float:
    for key in ("core_median_cents", "median_cents"):
        value = seg.get(key)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            return value
    return 0.0


def intonation_score_from_cents(abs_cents: float) -> float:
    """
    Perceptual curve, gentler than linear near zero because small deviations are
    inaudible and expressive singing in these genres routinely sits tens of cents
    off an equal-tempered grid.

    Deviation is measured against a tuning-corrected target, so this no longer
    absorbs a constant offset from a mistuned backing track. It remains a 12-TET
    grid and so still penalizes intentionally microtonal notes.
    """
    d = float(abs_cents)
    if d <= 25.0:
        return 1.0 - (0.10 / 25.0) * d
    if d <= 45.0:
        return 0.90 - (0.25 / 20.0) * (d - 25.0)
    return float(0.65 * np.exp(-(d - 45.0) / 40.0))


def score_intonation(segments: List[Dict]) -> List[Dict]:
    for seg in segments:
        deviation = abs(_deviation_cents(seg))
        seg["abs_cents_deviation"] = float(deviation)
        seg["intonation_score"] = float(
            np.clip(intonation_score_from_cents(deviation), SCORE_FLOOR, 1.0)
        )
    return segments


def build_graph_points(segments: List[Dict]) -> List[Tuple[float, Optional[float]]]:
    """
    Time series of smoothed intonation score, with a null inserted across each
    silence so the chart lifts the pen instead of interpolating through a rest.

    Each note emits both its start and its end so sustained notes have width;
    emitting only starts made the final note infinitely thin.
    """
    if not segments:
        return []

    starts = np.array([float(s["start"]) for s in segments], dtype=float)
    raw = np.array([float(s["intonation_score"]) for s in segments], dtype=float)

    smoothed = np.empty_like(raw)
    for i, t in enumerate(starts):
        window = (starts >= t - SMOOTHING_HALF_WINDOW_S) & (starts <= t + SMOOTHING_HALF_WINDOW_S)
        smoothed[i] = float(np.median(raw[window]))

    points: List[Tuple[float, Optional[float]]] = []
    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg["end"])

        if i > 0:
            gap = start - float(segments[i - 1]["end"])
            if gap > SILENCE_BREAK_S:
                points.append((max(start - 1e-3, float(segments[i - 1]["end"])), None))

        value = float(smoothed[i])
        points.append((start, value))
        if end > start:
            points.append((end, value))

    return points
