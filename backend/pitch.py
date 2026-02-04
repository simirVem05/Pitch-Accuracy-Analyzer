from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.signal

# CREPE requires tensorflow + crepe installed
import crepe  # type: ignore


@dataclass(frozen=True)
class PitchTrack:
    time_s: np.ndarray
    f0_hz: np.ndarray
    confidence: np.ndarray


def extract_pitch_crepe(
    y: np.ndarray,
    sr: int,
    step_size_ms: int = 10,
    model_capacity: str = "full",
    viterbi: bool = True,
) -> PitchTrack:
    time_s, f0_hz, conf, _activation = crepe.predict(
        y,
        sr,
        step_size=step_size_ms,
        model_capacity=model_capacity,
        viterbi=viterbi,
        verbose=0,
    )
    # Ensure 1D float arrays
    return PitchTrack(
        time_s=np.asarray(time_s, dtype=float).reshape(-1),
        f0_hz=np.asarray(f0_hz, dtype=float).reshape(-1),
        confidence=np.asarray(conf, dtype=float).reshape(-1),
    )


def filter_by_confidence(track: PitchTrack, threshold: float) -> PitchTrack:
    f0 = track.f0_hz.copy()
    # Mark low-confidence as NaN (unvoiced)
    f0[track.confidence < threshold] = np.nan
    return PitchTrack(track.time_s, f0, track.confidence)


def median_smooth_f0(f0_hz: np.ndarray, sr_hop: float, window_ms: int) -> np.ndarray:
    """
    sr_hop: frames per second (e.g., 100 for 10ms hop).
    """
    f0 = f0_hz.copy()
    win = max(1, int(round((window_ms / 1000.0) * sr_hop)))
    if win % 2 == 0:
        win += 1
    # Use nanmedian by applying filter to interpolated version
    idx = np.arange(len(f0))
    good = np.isfinite(f0)
    if good.sum() < 3:
        return f0
    interp = np.interp(idx, idx[good], f0[good])
    sm = scipy.signal.medfilt(interp, kernel_size=win)
    # Preserve NaN where originally unvoiced
    sm[~good] = np.nan
    return sm
