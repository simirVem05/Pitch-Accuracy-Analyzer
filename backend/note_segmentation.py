from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# 1.2 segmentation rule constants:
DRIFT_CENTS = 40.0
TRIGGER_FRAMES = 3

# Core-trim for stable pitch scoring:
# We score from the "core" of the voiced frames to avoid scoops/releases.
CORE_TRIM_FRACTION = 0.20  # trim 20% of voiced frames from each end (middle 60% remains)
MIN_VOICED_FOR_CORE = 10   # need enough voiced frames to justify trimming


# --- Frequency <-> MIDI ---
def hz_to_midi(hz: float) -> int:
    return int(np.round(12.0 * np.log2(hz / 440.0) + 69.0))


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def cents_diff(artist_hz: np.ndarray, target_hz: float) -> np.ndarray:
    return 1200.0 * np.log2(artist_hz / target_hz)


# --- Pitch-class utilities ---
NOTE_NAME_TO_PC = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "E#": 5, "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}


def tonic_to_pitch_class(tonic: str) -> int:
    if tonic not in NOTE_NAME_TO_PC:
        raise ValueError(f"Invalid tonic: {tonic}")
    return NOTE_NAME_TO_PC[tonic]


MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
NAT_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


GENRE_EXTRA_INTERVALS = {
    "hip hop": {1, 6, 10},
    "pop": {8, 6},
    "alternative pop": {1, 3},
    "r&b": {6, 1},
    "rock": {3, 10, 6},
    "country": {2, 3, 10},
    "jazz": {8, 3},
}


def build_allowed_pitch_classes(
    tonic_pc: int,
    scale: str,
    genre: str,
) -> Tuple[Set[int], Set[int], Set[int]]:
    scale_l = scale.lower()
    genre_l = genre.lower()

    if scale_l in {"major", "ionian"}:
        diatonic_offsets = MAJOR_SCALE
        is_minor = False
    else:
        diatonic_offsets = NAT_MINOR_SCALE
        is_minor = True

    diatonic_pitch_classes = {(tonic_pc + x) % 12 for x in diatonic_offsets}
    allowed = set(diatonic_pitch_classes)

    extras = set(GENRE_EXTRA_INTERVALS.get(genre_l, set()))

    # stylistic additions in minor (kept from your earlier logic)
    if is_minor and genre_l in {"pop", "r&b"}:
        extras.add(11)  # natural 7
    if is_minor and genre_l == "jazz":
        extras.add(9)   # natural 13

    for semis in extras:
        allowed.add((tonic_pc + semis) % 12)

    blue_offsets = {3, 6, 10}
    blue_pitch_classes = {(tonic_pc + x) % 12 for x in blue_offsets}

    return allowed, diatonic_pitch_classes, blue_pitch_classes


# --- NoteSeg object ---
@dataclass
class NoteSeg:
    start: float
    end: float
    target_note: int
    median_cents: float
    core_median_cents: float
    n_voiced_frames: int

    classification: str = "unknown"
    on_key_score: Optional[float] = None
    vibrato: Optional[Dict[str, Optional[bool]]] = None
    portamento: Optional[Dict[str, Optional[bool]]] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["target_hz"] = float(midi_to_hz(self.target_note))
        d["pitch_class"] = self.target_note % 12
        d["duration_s"] = float(self.end - self.start)
        return d


def _last_voiced_index(freqs: np.ndarray, start: int, end: int) -> Optional[int]:
    """Return last i in [start, end] where freqs[i] is voiced."""
    end = min(end, len(freqs) - 1)
    start = max(0, start)
    for i in range(end, start - 1, -1):
        if not np.isnan(freqs[i]):
            return i
    return None


def _compute_core_median_cents(seg_freqs: np.ndarray, target_hz: float) -> Tuple[float, float, int]:
    """
    Returns:
      (median_cents_all, core_median_cents, n_voiced_frames)
    """
    voiced_idx = np.flatnonzero(~np.isnan(seg_freqs))
    n_voiced = int(voiced_idx.size)
    if n_voiced == 0:
        return 0.0, 0.0, 0

    cents_all = cents_diff(seg_freqs[voiced_idx], target_hz)
    med_all = float(np.median(cents_all))

    # Core median: trim edges of the voiced run (avoid scoops/releases)
    if n_voiced >= MIN_VOICED_FOR_CORE:
        trim = int(np.floor(n_voiced * CORE_TRIM_FRACTION))
        lo = trim
        hi = n_voiced - trim
        if hi <= lo + 1:
            core = cents_all
        else:
            core = cents_all[lo:hi]
        med_core = float(np.median(core))
    else:
        med_core = med_all

    return med_all, med_core, n_voiced


def segment_notes(times: np.ndarray, freqs: np.ndarray) -> List[NoteSeg]:
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    voiced = ~np.isnan(freqs)
    if not np.any(voiced):
        return []

    i0 = int(np.argmax(voiced))
    current_note = hz_to_midi(freqs[i0])
    seg_start = i0

    cand_note: Optional[int] = None
    cand_count = 0
    segments: List[NoteSeg] = []

    def finalize(end_idx: int):
        nonlocal segments, seg_start, current_note

        # Ensure we end on a voiced frame (important!)
        last_v = _last_voiced_index(freqs, seg_start, end_idx)
        if last_v is None:
            return

        seg_freqs = freqs[seg_start:last_v + 1]
        seg_times = times[seg_start:last_v + 1]

        target_hz = midi_to_hz(current_note)
        med_all, med_core, n_voiced_frames = _compute_core_median_cents(seg_freqs, target_hz)

        segments.append(
            NoteSeg(
                start=float(seg_times[0]),
                end=float(seg_times[-1]),   # now guaranteed voiced end
                target_note=int(current_note),
                median_cents=float(med_all),
                core_median_cents=float(med_core),
                n_voiced_frames=int(n_voiced_frames),
                vibrato={"present": None},
                portamento={"present": None},
            )
        )

    for i in range(i0 + 1, len(freqs)):
        if np.isnan(freqs[i]):
            continue

        drift = abs(cents_diff(np.array([freqs[i]]), midi_to_hz(current_note))[0])
        if drift < DRIFT_CENTS:
            cand_note = None
            cand_count = 0
            continue

        proposed = hz_to_midi(freqs[i])
        if proposed == current_note:
            cand_note = None
            cand_count = 0
            continue

        if cand_note != proposed:
            cand_note = proposed
            cand_count = 1
        else:
            cand_count += 1

        if cand_count > TRIGGER_FRAMES:
            # End current segment right before the candidate run started
            finalize(i - cand_count)
            seg_start = i - cand_count + 1
            current_note = cand_note
            cand_note = None
            cand_count = 0

    # Finalize last segment to last voiced frame in the entire track
    last_v = _last_voiced_index(freqs, seg_start, len(freqs) - 1)
    if last_v is not None:
        finalize(last_v)

    return segments


def segment_and_build_allowed(
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    tonic: str,
    scale: str,
    genre: str,
) -> Tuple[
    List[Dict],
    Set[int],
    Set[int],
    Set[int],
    int,
]:
    """
    Returns:
        segments_dicts,
        allowed_pitch_classes,
        diatonic_pitch_classes,
        blue_pitch_classes,
        tonic_pitch_class
    """
    tonic_pc = tonic_to_pitch_class(tonic)
    allowed, diatonic, blue = build_allowed_pitch_classes(tonic_pc, scale, genre)

    segments = segment_notes(times, freqs)
    segments_dicts = [s.to_dict() for s in segments]

    return segments_dicts, allowed, diatonic, blue, tonic_pc
