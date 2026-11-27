"""
Synthetic accompaniment audio with known harmony, rendered through the production
chroma path. Used to calibrate and test chord estimation without real-song labels.

Synthetic timbres (summed harmonic partials) are far simpler than a real `other`
stem; results here bound what the estimator can do under ideal-ish conditions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import librosa
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend"))
import harmony as H  # noqa: E402

SR = H.CHROMA_SR
HOP = H.CHROMA_HOP
TIMBRES = {
    "saw": [1.0 / k for k in range(1, 9)],
    "soft": [0.5 ** (k - 1) for k in range(1, 7)],
    "organ": [1.0, 0.8, 0.6, 0.45, 0.3, 0.2],
}


def production_cens(y: np.ndarray) -> np.ndarray:
    """The exact chroma call in backend/harmony.py:build_harmonic_context (sr already 22050)."""
    y = np.ascontiguousarray(y, dtype=np.float32)
    tuning = H.estimate_tuning_semitones(y, SR)
    return np.asarray(librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP, tuning=tuning), dtype=float)


def frame_times(n: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n), sr=SR, hop_length=HOP)


def tone(midi: float, dur: float, timbre: str, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(round(dur * SR))) / SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    y = np.zeros_like(t)
    for k, a in enumerate(TIMBRES[timbre], start=1):
        if f0 * k < SR / 2 - 500:
            y += a * np.sin(2 * np.pi * f0 * k * t)
    env = np.ones_like(t)
    na, nr = min(len(t), int(0.01 * SR)), min(len(t), int(0.05 * SR))
    env[:na] = np.linspace(0, 1, na)
    env[len(t) - nr:] *= np.linspace(1, 0, nr)
    return amp * y * env


def voicing(pcs: Sequence[int], rng: np.random.Generator, low: int = 48, high: int = 60) -> List[int]:
    """Random voicing: a base MIDI in [low, high], each pitch class at its nearest MIDI >= base, optional doubled lowest note."""
    base = int(rng.integers(low, high + 1))
    notes = sorted(base + ((pc - base) % 12) for pc in pcs)
    if rng.random() < 0.5:
        notes.append(notes[0] + 12)
    return notes


def chord_audio(midis: Sequence[int], dur: float, timbre: str, rng: np.random.Generator) -> np.ndarray:
    y = sum(tone(m, dur, timbre, amp=float(rng.uniform(0.7, 1.0))) for m in midis)
    return y / max(np.max(np.abs(y)), 1e-9) * 0.5


def white(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0, 1, n)


def pink(n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.fft.rfft(rng.normal(0, 1, n))
    f = np.arange(X.size)
    f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n)
    return y / np.std(y)


def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = np.mean(y ** 2)
    return y + white(len(y), rng) * np.sqrt(p / (10 ** (snr_db / 10)))


def progression(chords: Sequence[frozenset], beats_each: Sequence[int], tempo: float, timbre: str,
                rng: np.random.Generator, snr_db: float = 30.0) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float, frozenset]]]:
    """Render a chord progression on a regular beat grid. Returns audio, beat times and (start, end, set) segments."""
    beat = 60.0 / tempo
    parts, segs, t = [], [], 0.0
    for s, nb in zip(chords, beats_each):
        d = nb * beat
        parts.append(chord_audio(voicing(sorted(s), rng), d, timbre, rng))
        segs.append((t, t + d, s))
        t += d
    y = np.concatenate(parts)
    y = add_noise(y, snr_db, rng)
    beats = np.arange(0.0, t + 1e-9, beat)
    return y, beats, segs
