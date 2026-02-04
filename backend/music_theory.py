from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import math
import numpy as np
from music21 import key as m21key
from music21 import pitch as m21pitch
from music21 import scale as m21scale

from .config import Genre


PITCH_CLASS_NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def hz_to_cents(f_hz: float, a4_hz: float = 440.0) -> float:
    """
    Convert frequency to cents relative to A4 tuning.
    Uses MIDI mapping anchored to A4=69.
    """
    if not (f_hz and f_hz > 0.0 and math.isfinite(f_hz)):
        return float("nan")
    midi = 69.0 + 12.0 * math.log2(f_hz / a4_hz)
    return 100.0 * midi  # cents where 1 semitone = 100 cents


def cents_to_hz(cents: float, a4_hz: float = 440.0) -> float:
    if not math.isfinite(cents):
        return float("nan")
    midi = cents / 100.0
    return a4_hz * (2.0 ** ((midi - 69.0) / 12.0))


def midi_to_pitch_class(midi: float) -> int:
    return int(round(midi)) % 12


def cents_to_midi(cents: float) -> float:
    return cents / 100.0


def parse_key(declared_key: str) -> m21key.Key:
    """
    Robust key parser.
    Accepts:
      - "B minor", "C major"
      - "Bb minor", "E♭ major", "F# minor"
      - "Bmin", "Cmaj"
      - "B", "C#" (defaults to major)
    """
    s = declared_key.strip().replace("♭", "b").replace("♯", "#")
    s = " ".join(s.split())  # normalize whitespace

    parts = s.split(" ")
    if len(parts) == 1:
        tonic = parts[0]
        mode = "major"
        # handle shorthand
        if tonic.lower().endswith("min"):
            tonic = tonic[:-3]
            mode = "minor"
        elif tonic.lower().endswith("maj"):
            tonic = tonic[:-3]
            mode = "major"
        return m21key.Key(tonic, mode)

    tonic = parts[0]
    mode = parts[1].lower()

    # allow common aliases
    if mode in ("minor", "min"):
        mode = "minor"
    elif mode in ("major", "maj"):
        mode = "major"
    else:
        # fallback: if user typed something weird, assume major
        mode = "major"

    return m21key.Key(tonic, mode)


@dataclass(frozen=True)
class AllowedPitchClasses:
    diatonic: Set[int]
    genre_allowed: Set[int]
    passing_allowed: Set[int]  # used as approach/passing tones (lighter penalty)


def _scale_pitch_classes(k: m21key.Key) -> Set[int]:
    # Use music21 scale implied by key
    sc = k.getScale()
    pcs = set()
    for p in sc.getPitches("C1", "C2"):
        pcs.add(int(p.pitchClass))
    return pcs


def _pc_from_degree(k: m21key.Key, degree_1_indexed: int) -> int:
    # degree: 1..7
    sc = k.getScale()
    p = sc.pitchFromDegree(degree_1_indexed)
    return int(p.pitchClass)


def _transpose_pc(pc: int, semitones: int) -> int:
    return (pc + semitones) % 12


def build_allowed_pitch_classes(k: m21key.Key, genre: Genre) -> AllowedPitchClasses:
    """
    Produces three sets:
    - diatonic: always valid
    - genre_allowed: non-diatonic but stylistically plausible
    - passing_allowed: chromatic approach tones (usually short)
    """
    diatonic = _scale_pitch_classes(k)
    genre_allowed: Set[int] = set()
    passing_allowed: Set[int] = set()

    tonic_pc = int(k.tonic.pitchClass)
    # Scale degrees (1..7) in this key
    deg = {i: _pc_from_degree(k, i) for i in range(1, 8)}
    dominant_pc = deg[5]

    # Generic passing/approach tones: +/- 1 semitone around diatonic
    for pc in diatonic:
        passing_allowed.add(_transpose_pc(pc, -1))
        passing_allowed.add(_transpose_pc(pc, +1))
    passing_allowed -= diatonic  # only non-diatonic in passing list

    if genre == "pop":
        # Modal mixture in major contexts (b3, b6) + secondary-dominant-ish leading tones
        # Without chord progression, we approximate by allowing b3/b6 in major and #4 as Lydian color.
        if k.mode == "major":
            # b3 relative to tonic
            genre_allowed.add(_transpose_pc(tonic_pc, 3))  # minor 3rd above tonic pitch class? careful: this is +3 semitones
            genre_allowed.add(_transpose_pc(tonic_pc, 8))  # b6
        genre_allowed.add(_transpose_pc(tonic_pc, 6))      # tritone/#4 color sometimes in pop
        genre_allowed |= passing_allowed

    elif genre == "rock":
        # Blues inflection b3, b7 plus chromatic approaches
        genre_allowed.add(_transpose_pc(tonic_pc, 3))  # b3
        genre_allowed.add(_transpose_pc(tonic_pc, 10)) # b7
        genre_allowed |= passing_allowed

    elif genre == "blues":
        # Blue notes: b3, b5, b7 (+ microtonal tendency handled by scoring width)
        genre_allowed.add(_transpose_pc(tonic_pc, 3))  # b3
        genre_allowed.add(_transpose_pc(tonic_pc, 6))  # b5
        genre_allowed.add(_transpose_pc(tonic_pc, 10)) # b7
        genre_allowed |= passing_allowed

    elif genre == "jazz":
        # Chromatic passing tones always
        genre_allowed |= passing_allowed

        # Altered extensions typically on V (dominant): b9, #9, #11, b13 relative to dominant root
        # Map: dominant root + 1 (b9), +3 (#9), +6 (#11), +8 (b13)
        genre_allowed.add(_transpose_pc(dominant_pc, 1))  # b9
        genre_allowed.add(_transpose_pc(dominant_pc, 3))  # #9
        genre_allowed.add(_transpose_pc(dominant_pc, 6))  # #11
        genre_allowed.add(_transpose_pc(dominant_pc, 8))  # b13

        # Bebop passing tones: add chromatic between 5-6 and 2-3-ish (approx)
        # Approx: allow chromatic between diatonic neighbors
        genre_allowed |= passing_allowed

    elif genre == "rnb_soul":
        # Blue notes b3, b7 + pentatonic embellishment + slides
        genre_allowed.add(_transpose_pc(tonic_pc, 3))   # b3
        genre_allowed.add(_transpose_pc(tonic_pc, 10))  # b7
        genre_allowed |= passing_allowed

    elif genre == "hiphop_rap":
        # Similar to R&B for melodic hooks, but generally more forgiving
        genre_allowed.add(_transpose_pc(tonic_pc, 3))   # b3
        genre_allowed.add(_transpose_pc(tonic_pc, 10))  # b7
        genre_allowed |= passing_allowed

    elif genre == "classical":
        # Mostly diatonic; allow functional chromaticism:
        # - leading tone of harmonic/melodic minor is handled by key mode if user chooses it,
        # but also allow secondary-dominant leading tones via passing tones very lightly.
        genre_allowed |= set()  # keep minimal
        # Keep passing tones available but treat as passing-only (not fully genre_allowed)
        # (So they incur smaller penalty only if short)
        pass

    # Ensure sets disjoint-ish; passing is subset of genre for most genres except classical
    if genre != "classical":
        passing_allowed = passing_allowed

    # Avoid accidental full chromatic for very small scales
    genre_allowed -= diatonic
    passing_allowed -= diatonic

    return AllowedPitchClasses(diatonic=diatonic, genre_allowed=genre_allowed, passing_allowed=passing_allowed)


def pc_to_name(pc: int) -> str:
    return PITCH_CLASS_NAMES_SHARP[pc % 12]
