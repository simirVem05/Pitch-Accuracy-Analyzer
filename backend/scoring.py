from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .music_theory import AllowedPitchClasses


@dataclass(frozen=True)
class ScoreResult:
    deviation_cents: np.ndarray
    validity_weight: np.ndarray
    score_raw: np.ndarray            # 0..100 from cents curve
    score_final: np.ndarray          # 0..100 weighted


def cents_score_curve(dev_cents: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    dev = np.abs(dev_cents)
    s = 100.0 * np.exp(-((dev / sigma) ** alpha))
    s[~np.isfinite(dev_cents)] = np.nan
    return s


def compute_validity_weight(
    target_pc: np.ndarray,
    allowed: AllowedPitchClasses,
    is_passing_like: np.ndarray,
    strictness: float,
    w_valid: float,
    w_genre: float,
    w_passing: float,
    w_invalid: float,
) -> np.ndarray:
    """
    Weights are “base weights”. We then interpolate them with strictness:
    - strictness=0 => closer to 1.0
    - strictness=1 => use given weights fully
    """
    n = len(target_pc)
    w = np.full(n, np.nan, dtype=float)

    for i, pc in enumerate(target_pc):
        if pc < 0:
            continue
        if pc in allowed.diatonic:
            base = w_valid
        elif pc in allowed.genre_allowed:
            base = w_genre
        elif pc in allowed.passing_allowed:
            base = w_passing
        else:
            base = w_invalid

        # If the segment looks like passing (short) and pc is in passing_allowed,
        # give it the passing weight even if also genre_allowed.
        if is_passing_like[i] and (pc in allowed.passing_allowed):
            base = min(base, w_passing)

        # strictness interpolation: relax toward 1.0 when strictness is low
        w[i] = (1.0 - strictness) * 1.0 + strictness * base

    return w


def compute_scores(
    sung_cents: np.ndarray,
    target_cents: np.ndarray,
    target_pc: np.ndarray,
    allowed: AllowedPitchClasses,
    note_duration_frames: np.ndarray,
    min_note_frames_for_non_passing: int,
    sigma: float,
    alpha: float,
    strictness: float,
    w_valid: float,
    w_genre: float,
    w_passing: float,
    w_invalid: float,
) -> ScoreResult:
    dev = sung_cents - target_cents
    raw = cents_score_curve(dev, sigma=sigma, alpha=alpha)

    # Passing-like: short notes
    is_passing_like = (note_duration_frames > 0) & (note_duration_frames < min_note_frames_for_non_passing)

    vw = compute_validity_weight(
        target_pc=target_pc,
        allowed=allowed,
        is_passing_like=is_passing_like,
        strictness=strictness,
        w_valid=w_valid,
        w_genre=w_genre,
        w_passing=w_passing,
        w_invalid=w_invalid,
    )

    final = raw * vw
    # For unvoiced frames, keep NaN (frontend can gap)
    final[~np.isfinite(raw)] = np.nan
    return ScoreResult(deviation_cents=dev, validity_weight=vw, score_raw=raw, score_final=final)
