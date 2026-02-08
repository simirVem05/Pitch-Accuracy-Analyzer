from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np

# Import your preprocessing pipeline
# preprocess.py must be in the same folder or installed as a module
from preprocess import preprocess, PreprocessConfig


# Spec from your doc:
# - If pitch drifts by < 40 cents, stay in current NoteSeg
# - Only if it stays in a new note area for > 3 frames, trigger a new NoteSeg
DRIFT_CENTS = 40.0
TRIGGER_FRAMES = 3


def hz_to_midi(hz: float) -> int:
    # Note = 12 * log2(f/440) + 69  (then nearest MIDI note)
    return int(np.round(12.0 * np.log2(hz / 440.0) + 69.0))


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def cents_diff(artist_hz: np.ndarray, target_hz: float) -> np.ndarray:
    """
    Vectorized cents difference.
    artist_hz may include np.nan (unvoiced); output will be np.nan there.
    """
    return 1200.0 * np.log2(artist_hz / target_hz)


def segment_notes(times: np.ndarray, freqs: np.ndarray) -> List[Dict]:
    """
    Segment frame-level f0 into note segments using the spec hysteresis rules.

    Parameters
    ----------
    times : np.ndarray
        Frame timestamps in seconds (same length as freqs)
    freqs : np.ndarray
        Frame f0 in Hz. Use np.nan for unvoiced frames.

    Returns
    -------
    List[Dict]
        Each dict: {start, end, midi, target_hz, median_cents}
    """
    times = np.asarray(times, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)

    if times.shape != freqs.shape:
        raise ValueError("times and freqs must have same shape")

    n = len(times)
    if n == 0:
        return []

    voiced = ~np.isnan(freqs)
    if not np.any(voiced):
        return []

    # First voiced frame initializes the current segment
    i0 = int(np.argmax(voiced))
    current_midi = hz_to_midi(float(freqs[i0]))
    seg_start = i0

    # Candidate new note tracking (hysteresis trigger)
    cand_midi: Optional[int] = None
    cand_count = 0

    segments: List[Dict] = []

    def finalize_segment(end_idx: int):
        nonlocal segments, seg_start, current_midi

        seg_freqs = freqs[seg_start : end_idx + 1]
        seg_times = times[seg_start : end_idx + 1]

        target = midi_to_hz(current_midi)
        seg_cents = cents_diff(seg_freqs, target)  # vectorized
        med_cents = float(np.nanmedian(seg_cents))

        segments.append(
            {
                "start": float(seg_times[0]),
                "end": float(seg_times[-1]),
                "midi": int(current_midi),
                "target_hz": float(target),
                "median_cents": med_cents,
            }
        )

    for i in range(i0 + 1, n):
        if np.isnan(freqs[i]):
            # Ignore unvoiced frames for segmentation (per your current approach)
            continue

        target_hz_cur = midi_to_hz(current_midi)
        drift = abs(float(cents_diff(np.asarray([freqs[i]]), target_hz_cur)[0]))

        if drift <= DRIFT_CENTS:
            # Still close enough to current note; reset candidate
            cand_midi = None
            cand_count = 0
            continue

        proposed = hz_to_midi(float(freqs[i]))

        if proposed == current_midi:
            cand_midi = None
            cand_count = 0
            continue

        # Hysteresis: must persist > TRIGGER_FRAMES in the new note area
        if cand_midi is None or cand_midi != proposed:
            cand_midi = proposed
            cand_count = 1
        else:
            cand_count += 1

        if cand_count > TRIGGER_FRAMES:
            # End current segment at the frame just BEFORE the run began
            end_current = i - cand_count
            if end_current >= seg_start:
                finalize_segment(end_current)

            # Start new segment at the beginning of the candidate run
            seg_start = i - cand_count + 1
            current_midi = cand_midi

            # Reset candidate tracking
            cand_midi = None
            cand_count = 0

    # Finalize last segment to last voiced frame at/after seg_start
    last_voiced_idx = None
    for j in range(n - 1, seg_start - 1, -1):
        if not np.isnan(freqs[j]):
            last_voiced_idx = j
            break

    if last_voiced_idx is not None and last_voiced_idx >= seg_start:
        finalize_segment(last_voiced_idx)

    return segments


def segment_notes_from_audio(
    audio_path: str,
    config: Optional[PreprocessConfig] = None,
    *,
    viterbi: bool = False,
    model_capacity: str = "full",
) -> Tuple[List[Dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience wrapper that:
      1) calls preprocess.py to get time/frequency/confidence/activation
      2) runs segmentation on time+frequency
    Returns:
      segments, time, frequency, confidence, activation
    """
    time, frequency, confidence, activation = preprocess(
        audio_path,
        config=config,
        viterbi=viterbi,
        model_capacity=model_capacity,
    )
    segments = segment_notes(time, frequency)
    return segments, time, frequency, confidence, activation