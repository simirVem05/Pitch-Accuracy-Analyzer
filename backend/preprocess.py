from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import crepe
import librosa
import numpy as np
from scipy import signal


CREPE_SR = 16000


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = CREPE_SR
    step_size_ms: int = 20
    highpass_hz: float = 85.0
    butter_order: int = 5

    model_capacity: str = "full"
    viterbi: bool = True

    conf_threshold: float = 0.60

    # Plausible sung-pitch range; separation bleed often lands outside it.
    min_f0_hz: float = 65.0
    max_f0_hz: float = 1200.0

    min_true_run: int = 5
    max_gap_fill: int = 1


@dataclass(frozen=True)
class PitchTrack:
    time: np.ndarray
    frequency: np.ndarray   # NaN where unvoiced
    confidence: np.ndarray
    coverage: float         # fraction of frames confidently voiced


def butter_highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 85.0,
    order: int = 5,
) -> np.ndarray:
    if cutoff_hz <= 0 or y.size == 0:
        return y
    nyq = 0.5 * sr
    b, a = signal.butter(order, cutoff_hz / nyq, btype="highpass", analog=False)
    return signal.filtfilt(b, a, y).astype(np.float32)


def _run_length_filter(mask: np.ndarray, min_len: int) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if mask.size == 0 or min_len <= 1:
        return mask.copy()

    out = np.zeros_like(mask, dtype=bool)
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_len:
            out[i:j] = True
        i = j
    return out


def _fill_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if mask.size == 0 or max_gap <= 0:
        return mask.copy()

    out = mask.copy()
    n = out.size
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and (not out[j]):
            j += 1

        if (i - 1 >= 0) and out[i - 1] and (j < n) and out[j] and (j - i) <= max_gap:
            out[i:j] = True

        i = j
    return out


def smooth_voicing_mask(
    initial_mask: np.ndarray,
    min_true_run: int = 5,
    max_gap_fill: int = 1,
) -> np.ndarray:
    m = _fill_small_gaps(initial_mask, max_gap_fill)
    m = _run_length_filter(m, min_true_run)
    return _fill_small_gaps(m, max_gap_fill)


def detect_pitch(
    vocals: np.ndarray,
    sample_rate: int,
    config: Optional[PreprocessConfig] = None,
) -> PitchTrack:
    """
    CREPE f0 on an isolated vocal, with unvoiced frames masked to NaN.

    Confidence gating is load-bearing here rather than cosmetic. The vocals stem
    contains lead plus any backing, harmonies and doubles, and CREPE is
    monophonic — given layered input it follows whichever voice is loudest and
    can jump between lead and harmony mid-phrase. Dropping ambiguous frames
    scores fewer notes instead of scoring them wrong; `coverage` reports how much
    was skipped so the omission stays visible.
    """
    cfg = config or PreprocessConfig()

    y = vocals if sample_rate == cfg.sample_rate else librosa.resample(
        vocals, orig_sr=sample_rate, target_sr=cfg.sample_rate
    )
    y = np.ascontiguousarray(y, dtype=np.float32)
    y = butter_highpass_filter(y, sr=cfg.sample_rate, cutoff_hz=cfg.highpass_hz, order=cfg.butter_order)

    time, frequency, confidence, _activation = crepe.predict(
        y,
        cfg.sample_rate,
        step_size=cfg.step_size_ms,
        viterbi=cfg.viterbi,
        model_capacity=cfg.model_capacity,
        verbose=0,
    )

    time = np.asarray(time, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)

    voiced_raw = (
        (confidence >= cfg.conf_threshold)
        & (frequency >= cfg.min_f0_hz)
        & (frequency <= cfg.max_f0_hz)
    )
    voiced = smooth_voicing_mask(voiced_raw, cfg.min_true_run, cfg.max_gap_fill)

    frequency_masked = frequency.copy()
    frequency_masked[~voiced] = np.nan

    coverage = float(np.mean(voiced)) if voiced.size else 0.0

    return PitchTrack(
        time=time,
        frequency=frequency_masked,
        confidence=confidence,
        coverage=coverage,
    )
