from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

from .config import Genre


@dataclass(frozen=True)
class PerformanceSummary:
    overall_score: float
    median_score: float
    pct_frames_above_90: float
    pct_frames_above_70: float
    median_abs_dev_cents: float
    p95_abs_dev_cents: float
    vibrato_pct: float
    voiced_pct: float


def summarize(
    time_s: np.ndarray,
    score_final: np.ndarray,
    deviation_cents: np.ndarray,
    is_vibrato: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float,
) -> PerformanceSummary:
    voiced = np.isfinite(score_final)
    if voiced.sum() == 0:
        return PerformanceSummary(
            overall_score=0.0,
            median_score=0.0,
            pct_frames_above_90=0.0,
            pct_frames_above_70=0.0,
            median_abs_dev_cents=float("nan"),
            p95_abs_dev_cents=float("nan"),
            vibrato_pct=0.0,
            voiced_pct=0.0,
        )
    s = score_final[voiced]
    dev = np.abs(deviation_cents[voiced])

    return PerformanceSummary(
        overall_score=float(np.nanmean(s)),
        median_score=float(np.nanmedian(s)),
        pct_frames_above_90=float(np.mean(s >= 90.0) * 100.0),
        pct_frames_above_70=float(np.mean(s >= 70.0) * 100.0),
        median_abs_dev_cents=float(np.nanmedian(dev)),
        p95_abs_dev_cents=float(np.nanpercentile(dev, 95)),
        vibrato_pct=float(np.mean(is_vibrato[voiced]) * 100.0),
        voiced_pct=float(np.mean(confidence >= confidence_threshold) * 100.0),
    )


def build_gemini_prompt(
    declared_key: str,
    genre: Genre,
    summary: PerformanceSummary,
    notable_sections: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Returns a prompt string you can send to Gemini.
    """
    sections = notable_sections or {}
    return f"""
You are a vocal coach and music producer. Evaluate a singer's pitch performance with musical nuance.

Context:
- Declared key: {declared_key}
- Genre: {genre}

Objective metrics:
- Overall on-key score (0-100): {summary.overall_score:.1f}
- Median score: {summary.median_score:.1f}
- Frames >= 90%: {summary.pct_frames_above_90:.1f}%
- Frames >= 70%: {summary.pct_frames_above_70:.1f}%
- Median absolute deviation: {summary.median_abs_dev_cents:.1f} cents
- 95th percentile absolute deviation: {summary.p95_abs_dev_cents:.1f} cents
- Vibrato detected in: {summary.vibrato_pct:.1f}% of voiced frames
- Voiced confidence >= threshold: {summary.voiced_pct:.1f}%

Notable sections (if provided):
{sections}

Write a concise report with:
1) Overall assessment (tone: supportive, professional)
2) Strengths (intonation stability, control, stylistic fit)
3) Weaknesses (where pitch center drifts, wide vibrato, inconsistent targets)
4) 3 actionable practice tips
5) If genre allows chromaticism (blue notes/passing), acknowledge stylistic intent and avoid over-penalizing it.
""".strip()
