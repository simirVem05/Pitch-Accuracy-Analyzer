from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import librosa
from scipy import signal
import crepe


@dataclass(frozen=True)
class PreprocessConfig:
    sample_rate: int = 16000
    step_size_ms: int = 20
    highpass_hz: float = 85.0
    butter_order: int = 5

    # Unvoiced masking thresholds
    conf_threshold: float = 0.15
    rms_threshold: float = 0.0  # RMS == 0 treated as unvoiced

    # Mask smoothing behavior
    min_true_run: int = 5       # keep runs of at least 5 frames
    max_gap_fill: int = 1       # fill gaps up to this many frames (e.g., 1 frame)


def load_audio_multiformat(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio in many formats using librosa, resample to `sample_rate`,
    return mono float32 waveform in [-1, 1].
    """
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)
    return y, sr


def butter_highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 85.0,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter to remove sub-vocal rumble.
    """
    if cutoff_hz <= 0:
        return y

    nyq = 0.5 * sr
    norm_cut = cutoff_hz / nyq
    b, a = signal.butter(order, norm_cut, btype="highpass", analog=False)
    return signal.filtfilt(b, a, y).astype(np.float32)


def _run_length_filter(mask: np.ndarray, min_len: int) -> np.ndarray:
    """
    Keep only True runs of length >= min_len.
    """
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
    """
    Fill small False gaps inside True regions.
    """
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

        gap_len = j - i
        left_true = (i - 1 >= 0) and out[i - 1]
        right_true = (j < n) and out[j]

        if left_true and right_true and gap_len <= max_gap:
            out[i:j] = True

        i = j
    return out


def smooth_voicing_mask(
    initial_mask: np.ndarray,
    min_true_run: int = 5,
    max_gap_fill: int = 1,
) -> np.ndarray:
    """
    1) Fill tiny gaps
    2) Keep only voiced runs of at least min_true_run
    3) Fill gaps again
    """
    m = _fill_small_gaps(initial_mask, max_gap_fill)
    m = _run_length_filter(m, min_true_run)
    m = _fill_small_gaps(m, max_gap_fill)
    return m


def compute_rms_aligned(
    y: np.ndarray,
    sr: int,
    step_size_ms: int,
    frame_length: int = 1024,
    center: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Compute RMS per frame aligned with CREPE hops.
    """
    hop_length = int(round(sr * step_size_ms / 1000.0))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=center)[0]
    return rms.astype(np.float32), hop_length


def _align_1d_length(x: np.ndarray, n: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Pad or trim a 1D array to length n.
    """
    x = np.asarray(x)
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    pad = np.full((n - x.size,), pad_value, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def preprocess_audio_and_detect_pitch(
    audio_path: str,
    config: Optional[PreprocessConfig] = None,
    *,
    viterbi: bool = True,
    model_capacity: str = "tiny",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess + CREPE pitch detection.

    Returns:
      time (N,)
      frequency (N,) with np.nan where unvoiced
      confidence (N,)
      activation (N, K) with np.nan rows where unvoiced
    """
    cfg = config or PreprocessConfig()

    # 1) Load & resample
    y, sr = load_audio_multiformat(audio_path, sample_rate=cfg.sample_rate)

    # 2) High-pass filter
    y_hp = butter_highpass_filter(y, sr=sr, cutoff_hz=cfg.highpass_hz, order=cfg.butter_order)

    # 3) CREPE pitch detection @ step_size_ms
    time, frequency, confidence, activation = crepe.predict(
        y_hp,
        sr,
        step_size=cfg.step_size_ms,
        viterbi=viterbi,
        model_capacity=model_capacity,
        verbose=0,
    )

    time = np.asarray(time, dtype=np.float32)
    frequency = np.asarray(frequency, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)
    activation = np.asarray(activation, dtype=np.float32)

    n_frames = time.shape[0]

    # 4) RMS aligned
    rms, _hop = compute_rms_aligned(y_hp, sr=sr, step_size_ms=cfg.step_size_ms, center=True)
    rms = _align_1d_length(rms, n_frames, pad_value=0.0)

    # 5) Unvoiced mask: low confidence OR rms == 0
    voicing_mask_raw = (confidence >= cfg.conf_threshold) & (rms > cfg.rms_threshold)

    # 6) Smooth mask
    voicing_mask = smooth_voicing_mask(
        voicing_mask_raw,
        min_true_run=cfg.min_true_run,
        max_gap_fill=cfg.max_gap_fill,
    )

    # 7) Apply mask
    unvoiced = ~voicing_mask
    frequency_masked = frequency.copy()
    activation_masked = activation.copy()

    frequency_masked[unvoiced] = np.nan
    activation_masked[unvoiced, :] = np.nan

    return time, frequency_masked, confidence, activation_masked
