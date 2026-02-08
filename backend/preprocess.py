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
    max_gap_fill: int = 1       # fill gaps up to this many frames


def load_audio_multiformat(
    file_path: str,
    sample_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Load audio in many formats using librosa,
    resample to `sample_rate`, and return mono float32 waveform.
    """
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32), sr


def butter_highpass_filter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = 85.0,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter.
    """
    if cutoff_hz <= 0:
        return y

    nyq = 0.5 * sr
    norm_cut = cutoff_hz / nyq
    b, a = signal.butter(order, norm_cut, btype="highpass")
    return signal.filtfilt(b, a, y).astype(np.float32)


def _run_length_filter(mask: np.ndarray, min_len: int) -> np.ndarray:
    """
    Keep only True runs of length >= min_len.
    """
    out = np.zeros_like(mask, dtype=bool)
    i = 0
    n = mask.size

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
    out = mask.copy()
    i = 0
    n = out.size

    while i < n:
        if out[i]:
            i += 1
            continue

        j = i
        while j < n and not out[j]:
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
    min_true_run: int,
    max_gap_fill: int,
) -> np.ndarray:
    """
    Smooth voicing mask by:
    1) filling tiny gaps
    2) removing short runs
    3) filling gaps again
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
) -> np.ndarray:
    """
    Compute RMS aligned to CREPE frame hops.
    """
    hop_length = int(round(sr * step_size_ms / 1000.0))
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    return rms.astype(np.float32)


def _align_1d_length(x: np.ndarray, n: int) -> np.ndarray:
    """
    Trim or pad a 1D array to length n.
    """
    if x.size > n:
        return x[:n]
    if x.size < n:
        return np.pad(x, (0, n - x.size))
    return x


def preprocess(
    audio_path: str,
    config: Optional[PreprocessConfig] = None,
    *,
    viterbi: bool = False,
    model_capacity: str = "full",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess audio and run CREPE pitch tracking.

    Returns:
        time        (n_frames,)
        frequency   (n_frames,) Hz (NaN where unvoiced)
        confidence  (n_frames,)
        activation  (n_frames, 360) (NaN rows where unvoiced)
    """
    cfg = config or PreprocessConfig()

    # Load & filter
    y, sr = load_audio_multiformat(audio_path, cfg.sample_rate)
    y = butter_highpass_filter(y, sr, cfg.highpass_hz, cfg.butter_order)

    # CREPE pitch detection
    time, frequency, confidence, activation = crepe.predict(
        y,
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

    # RMS + voicing mask
    rms = compute_rms_aligned(y, sr, cfg.step_size_ms)
    rms = _align_1d_length(rms, time.size)

    voicing_raw = (confidence >= cfg.conf_threshold) & (rms > cfg.rms_threshold)
    voicing = smooth_voicing_mask(
        voicing_raw,
        cfg.min_true_run,
        cfg.max_gap_fill,
    )

    # Apply mask
    frequency[~voicing] = np.nan
    activation[~voicing, :] = np.nan

    return time, frequency, confidence, activation