from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


DRIFT_CENTS = 40.0
TRIGGER_FRAMES = 3

CORE_TRIM_FRACTION = 0.20
MIN_VOICED_FOR_CORE = 10

MIN_SEGMENT_VOICED_FRAMES = 3


def hz_to_midi_float(hz: np.ndarray | float, tuning_semitones: float = 0.0) -> np.ndarray | float:
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=float) / 440.0) - tuning_semitones


def midi_to_hz(midi: float, tuning_semitones: float = 0.0) -> float:
    return float(440.0 * (2.0 ** ((float(midi) + tuning_semitones - 69.0) / 12.0)))


def cents_diff(artist_hz: np.ndarray, target_hz: float) -> np.ndarray:
    return 1200.0 * np.log2(np.asarray(artist_hz, dtype=float) / float(target_hz))


@dataclass
class NoteSeg:
    start: float
    end: float
    target_note: int
    median_cents: float
    core_median_cents: float
    n_voiced_frames: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["pitch_class"] = int(self.target_note) % 12
        d["duration_s"] = float(self.end - self.start)
        return d


def _core_slice(cents_all: np.ndarray) -> np.ndarray:
    n = cents_all.size
    if n < MIN_VOICED_FOR_CORE:
        return cents_all
    trim = int(np.floor(n * CORE_TRIM_FRACTION))
    lo, hi = trim, n - trim
    return cents_all if hi <= lo + 1 else cents_all[lo:hi]


def _finalize_segment(
    seg_times: np.ndarray,
    seg_freqs: np.ndarray,
    tuning_semitones: float,
) -> Optional[NoteSeg]:
    """
    Build a segment whose target note comes from the middle of the note.

    The target is derived from the core-trimmed median rather than the first
    voiced frame: onsets are exactly where scoops live, so anchoring there can
    pick the wrong semitone and bias every deviation measured against it.
    """
    voiced_idx = np.flatnonzero(~np.isnan(seg_freqs))
    if voiced_idx.size < MIN_SEGMENT_VOICED_FRAMES:
        return None

    voiced_hz = seg_freqs[voiced_idx]
    midi_float = np.asarray(hz_to_midi_float(voiced_hz, tuning_semitones), dtype=float)

    core_midi = _core_slice(midi_float)
    target_note = int(np.round(np.median(core_midi)))
    target_hz = midi_to_hz(target_note, tuning_semitones)

    cents_all = cents_diff(voiced_hz, target_hz)

    return NoteSeg(
        start=float(seg_times[voiced_idx[0]]),
        end=float(seg_times[voiced_idx[-1]]),
        target_note=target_note,
        median_cents=float(np.median(cents_all)),
        core_median_cents=float(np.median(_core_slice(cents_all))),
        n_voiced_frames=int(voiced_idx.size),
    )


def segment_notes_with_frame_indices(
    times: np.ndarray,
    freqs: np.ndarray,
    tuning_semitones: float = 0.0,
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Group voiced frames into notes and retain each note's source frame indices.

    A new note is only committed after enough consecutive *voiced* frames sit far
    from the current target and agree on a different semitone, which prevents
    vibrato and jitter from splitting one note into many.

    The indices are exposed for consumers that need frame-level contours. Keeping
    them here ensures those consumers use exactly the same note boundaries and
    target-note calculation as production scoring.
    """
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    voiced = ~np.isnan(freqs)
    if not np.any(voiced):
        return [], []

    voiced_positions = np.flatnonzero(voiced)
    segments: List[NoteSeg] = []
    segment_frame_indices: List[np.ndarray] = []

    seg_start_pos = 0
    current_note = int(np.round(hz_to_midi_float(freqs[voiced_positions[0]], tuning_semitones)))

    cand_note: Optional[int] = None
    cand_start_pos: Optional[int] = None
    cand_count = 0

    def flush(end_pos_exclusive: int) -> None:
        idx = voiced_positions[seg_start_pos:end_pos_exclusive]
        if idx.size == 0:
            return
        seg = _finalize_segment(times[idx[0] : idx[-1] + 1], freqs[idx[0] : idx[-1] + 1], tuning_semitones)
        if seg is not None:
            segments.append(seg)
            segment_frame_indices.append(idx.copy())

    for pos in range(1, voiced_positions.size):
        i = voiced_positions[pos]
        target_hz = midi_to_hz(current_note, tuning_semitones)
        drift = abs(float(cents_diff(np.array([freqs[i]]), target_hz)[0]))

        if drift < DRIFT_CENTS:
            cand_note, cand_start_pos, cand_count = None, None, 0
            continue

        proposed = int(np.round(hz_to_midi_float(freqs[i], tuning_semitones)))
        if proposed == current_note:
            cand_note, cand_start_pos, cand_count = None, None, 0
            continue

        if cand_note != proposed:
            cand_note, cand_start_pos, cand_count = proposed, pos, 1
        else:
            cand_count += 1

        if cand_count > TRIGGER_FRAMES:
            # Boundary is tracked in voiced-frame positions, so interleaved
            # unvoiced frames cannot displace it.
            flush(cand_start_pos)
            seg_start_pos = cand_start_pos
            current_note = cand_note
            cand_note, cand_start_pos, cand_count = None, None, 0

    flush(voiced_positions.size)

    return [s.to_dict() for s in segments], segment_frame_indices


def segment_notes(
    times: np.ndarray,
    freqs: np.ndarray,
    tuning_semitones: float = 0.0,
) -> List[Dict]:
    """Production note segmentation without the optional frame-index detail."""
    segments, _frame_indices = segment_notes_with_frame_indices(times, freqs, tuning_semitones)
    return segments
