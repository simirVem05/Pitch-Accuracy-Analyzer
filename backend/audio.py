from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import librosa


@dataclass(frozen=True)
class AudioData:
    y: np.ndarray
    sr: int
    duration_s: float


def load_audio_mono(path: str, target_sr: int) -> AudioData:
    # librosa loads float32 in [-1, 1]
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = librosa.util.normalize(y)
    duration_s = float(len(y) / sr)
    return AudioData(y=y, sr=sr, duration_s=duration_s)
