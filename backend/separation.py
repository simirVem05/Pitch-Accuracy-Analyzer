from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


VOCAL_STEM = "vocals"
HARMONY_STEMS = ("other", "bass")


@dataclass(frozen=True)
class Stems:
    """Mono float32 waveforms at `sample_rate`."""

    vocals: np.ndarray
    other: np.ndarray
    bass: np.ndarray
    percussive: np.ndarray
    sample_rate: int

    @property
    def harmony(self) -> np.ndarray:
        """Pitched accompaniment: the grid the singer is aiming at."""
        return self.other + self.bass

    @property
    def duration_s(self) -> float:
        return float(len(self.vocals)) / float(self.sample_rate)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _to_mono(wav: torch.Tensor) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav.mean(dim=0)
    return wav.detach().to("cpu", torch.float32).numpy()


def separate(audio_path: str, model_name: Optional[str] = None) -> Stems:
    """
    Split a full mix into vocals / harmony / percussive.

    `harmony` is other+bass summed — the pitched accompaniment that defines the
    grid the singer is aiming at. Drums are excluded from it because broadband
    transients smear energy across all 12 chroma bins and flatten the harmonic
    profile; they are returned separately only because beat tracking wants them.
    """
    from demucs import api

    model_name = model_name or os.getenv("DEMUCS_MODEL", "htdemucs")
    device = os.getenv("DEMUCS_DEVICE") or _pick_device()

    separator = api.Separator(model=model_name, device=device, progress=False)
    _origin, sources = separator.separate_audio_file(audio_path)

    missing = [s for s in (VOCAL_STEM, *HARMONY_STEMS) if s not in sources]
    if missing:
        raise RuntimeError(
            f"Demucs model '{model_name}' did not return required stems: {missing}. "
            f"Got: {sorted(sources)}"
        )

    return Stems(
        vocals=_to_mono(sources[VOCAL_STEM]),
        other=_to_mono(sources["other"]),
        bass=_to_mono(sources["bass"]),
        percussive=_to_mono(sources["drums"]) if "drums" in sources else np.zeros(0, dtype=np.float32),
        sample_rate=int(separator.samplerate),
    )
