from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

# 1.2 segmentation rule constants:
DRIFT_CENTS = 40.0          # if pitch drifts by < 40 cents, stay in current NoteSeg :contentReference[oaicite:6]{index=6}
TRIGGER_FRAMES = 3          # only if it stays in new note area for > 3 frames, switch :contentReference[oaicite:7]{index=7}


# --- Frequency <-> MIDI (per spec) ---
def hz_to_midi(hz: float) -> int:
    # Note = 12 * log2(f/440) + 69  (nearest MIDI) :contentReference[oaicite:8]{index=8}
    return int(np.round(12.0 * np.log2(hz / 440.0) + 69.0))


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def cents_diff(artist_hz: np.ndarray, target_hz: float) -> np.ndarray:
    # Cents = 1200 * log2(artist_frequency / target_frequency) :contentReference[oaicite:9]{index=9}
    # If artist_hz includes np.nan, output will be np.nan there.
    return 1200.0 * np.log2(artist_hz / target_hz)


# --- Allowed pitch classes by key+scale+genre (built in 1.2, used in 1.3) ---
GENRE_EXTRA_INTERVALS = {
    # intervals are semitone offsets relative to tonic pitch class
    "hip hop": {1, 6, 10},             # b2, b5, b7 :contentReference[oaicite:10]{index=10}
    "pop": {8, 6, 11},                 # b6, #4, nat7 (minor context handled by caller) :contentReference[oaicite:11]{index=11}
    "alternative pop": {1, 3},         # b2, borrowed b3 (plus "chromatic passing tones" later in 1.3) :contentReference[oaicite:12]{index=12}
    "r&b": {6, 1, 11},                 # #11(= #4), b9(= b2), maj7 :contentReference[oaicite:13]{index=13}
    "rock": {3, 10, 6},                # b3, b7, b5 :contentReference[oaicite:14]{index=14}
    "country": {3, 10, 2, 3},          # slides (handled later), include b3, b7, 2 and b3 :contentReference[oaicite:15]{index=15}
    "jazz": {8, 3, 9},                 # #5, #9, nat13(= 6) — simplified to useful semitone sets :contentReference[oaicite:16]{index=16}
}

MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
NAT_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


def build_allowed_pitch_classes(tonic_pc: int, scale: str, genre: str) -> Set[int]:
    """
    Returns pitch classes (0-11) allowed for key+genre.
    1.2 requires building these sets; classification happens in 1.3. :contentReference[oaicite:17]{index=17}
    """
    scale_l = scale.strip().lower()
    genre_l = genre.strip().lower()

    if scale_l in {"major", "ionian"}:
        diatonic = MAJOR_SCALE
    else:
        diatonic = NAT_MINOR_SCALE

    allowed = {(tonic_pc + x) % 12 for x in diatonic}

    extras = GENRE_EXTRA_INTERVALS.get(genre_l, set())
    for semis in extras:
        allowed.add((tonic_pc + semis) % 12)

    return allowed


# --- NoteSeg object (initialized per 1.2; filled later in 1.3+) ---
@dataclass
class NoteSeg:
    start: float
    end: float
    target_note: int          # MIDI number of closest note :contentReference[oaicite:18]{index=18}
    median_cents: float       # initialized here :contentReference[oaicite:19]{index=19}

    # placeholders initialized now; computed later:
    classification: str = "unknown"     # :contentReference[oaicite:20]{index=20}
    on_key_score: Optional[float] = None  # :contentReference[oaicite:21]{index=21}
    vibrato: Optional[Dict[str, Optional[bool]]] = None  # :contentReference[oaicite:22]{index=22}
    portamento: Optional[Dict[str, Optional[bool]]] = None  # :contentReference[oaicite:23]{index=23}

    def to_dict(self) -> Dict:
        d = asdict(self)
        # keep both target_hz and target_note handy (nice for debugging/plots)
        d["target_hz"] = float(midi_to_hz(self.target_note))
        return d


def segment_notes(times: np.ndarray, freqs: np.ndarray) -> List[Dict]:
    """
    1.2 Note Segmentation and Targeting:
    - group continuous frames by same target note
    - hysteresis: <40 cents stays; must persist >3 frames to switch :contentReference[oaicite:24]{index=24}
    - initialize NoteSeg objects with start, end, median_cents (and placeholder fields) :contentReference[oaicite:25]{index=25}
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

    # init at first voiced frame
    i0 = int(np.argmax(voiced))
    current_note = hz_to_midi(float(freqs[i0]))
    seg_start = i0

    cand_note: Optional[int] = None
    cand_count = 0

    out: List[NoteSeg] = []

    def finalize(end_idx: int) -> None:
        nonlocal out, seg_start, current_note

        seg_freqs = freqs[seg_start : end_idx + 1]
        seg_times = times[seg_start : end_idx + 1]

        target_hz = midi_to_hz(current_note)
        seg_cents = cents_diff(seg_freqs, target_hz)
        med_cents = float(np.nanmedian(seg_cents))

        out.append(
            NoteSeg(
                start=float(seg_times[0]),
                end=float(seg_times[-1]),
                target_note=int(current_note),
                median_cents=med_cents,
                classification="unknown",
                on_key_score=None,
                vibrato={"present": None, "good_vibrato": None},
                portamento={"present": None, "good_portamento": None},
            )
        )

    for i in range(i0 + 1, n):
        if np.isnan(freqs[i]):
            # skip unvoiced frames for segmentation boundaries
            continue

        # drift from current target in cents
        drift = abs(float(cents_diff(np.asarray([freqs[i]]), midi_to_hz(current_note))[0]))

        if drift < DRIFT_CENTS:
            # stay in current note area
            cand_note = None
            cand_count = 0
            continue

        proposed = hz_to_midi(float(freqs[i]))
        if proposed == current_note:
            cand_note = None
            cand_count = 0
            continue

        # hysteresis counting: must persist > TRIGGER_FRAMES to switch
        if cand_note is None or cand_note != proposed:
            cand_note = proposed
            cand_count = 1
        else:
            cand_count += 1

        if cand_count > TRIGGER_FRAMES:
            # end current segment right before candidate run started
            end_current = i - cand_count
            if end_current >= seg_start:
                finalize(end_current)

            # start new segment at beginning of candidate run
            seg_start = i - cand_count + 1
            current_note = cand_note

            cand_note = None
            cand_count = 0

    # finalize last segment through last voiced frame
    last_voiced = None
    for j in range(n - 1, seg_start - 1, -1):
        if not np.isnan(freqs[j]):
            last_voiced = j
            break

    if last_voiced is not None and last_voiced >= seg_start:
        finalize(last_voiced)

    return [seg.to_dict() for seg in out]
