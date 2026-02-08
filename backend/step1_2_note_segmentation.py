from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


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


def cents_diff(artist_hz: float, target_hz: float) -> float:
    return 1200.0 * np.log2(artist_hz / target_hz)


def segment_notes(times: np.ndarray, freqs: np.ndarray) -> List[Dict]:
    """
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
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    if times.shape != freqs.shape:
        raise ValueError("times and freqs must have same shape")

    n = len(times)
    if n == 0:
        return []

    # valid voiced indices
    voiced = ~np.isnan(freqs)

    # If everything is unvoiced, nothing to segment
    if not np.any(voiced):
        return []

    # Find first voiced frame to initialize
    i0 = int(np.argmax(voiced))
    current_midi = hz_to_midi(freqs[i0])
    seg_start = i0

    # Candidate new note tracking (hysteresis trigger)
    cand_midi: Optional[int] = None
    cand_count = 0

    segments: List[Dict] = []

    def finalize_segment(end_idx: int):
        nonlocal segments, seg_start, current_midi
        seg_freqs = freqs[seg_start : end_idx + 1]
        seg_times = times[seg_start : end_idx + 1]

        # compute median cents vs target
        target = midi_to_hz(current_midi)
        seg_cents = cents_diff(seg_freqs, target)
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
            # For Step 1.2 we’ll just ignore unvoiced frames here.
            # (Later you can decide whether unvoiced splits segments.)
            continue

        target_hz_cur = midi_to_hz(current_midi)
        drift = abs(cents_diff(freqs[i], target_hz_cur))

        if drift <= DRIFT_CENTS:
            # still close enough to current note; reset candidate
            cand_midi = None
            cand_count = 0
            continue

        # outside drift window => propose new note based on nearest MIDI
        proposed = hz_to_midi(freqs[i])

        # if proposed is same note (rare rounding edge), treat as staying
        if proposed == current_midi:
            cand_midi = None
            cand_count = 0
            continue

        # hysteresis: must persist > 3 frames in the new note area
        if cand_midi is None or cand_midi != proposed:
            cand_midi = proposed
            cand_count = 1
        else:
            cand_count += 1

        if cand_count > TRIGGER_FRAMES:
            # switch notes:
            # end current segment at the frame just BEFORE the run began
            end_current = i - cand_count
            if end_current >= seg_start:
                finalize_segment(end_current)

            # start new segment at the beginning of the candidate run
            seg_start = i - cand_count + 1
            current_midi = cand_midi

            # reset candidate tracking
            cand_midi = None
            cand_count = 0

    # finalize last segment to last voiced frame
    # find last voiced index at/after seg_start
    last_voiced_idx = None
    for j in range(n - 1, seg_start - 1, -1):
        if not np.isnan(freqs[j]):
            last_voiced_idx = j
            break

    if last_voiced_idx is not None and last_voiced_idx >= seg_start:
        finalize_segment(last_voiced_idx)

    return segments
