from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Genre = Literal["pop", "rock", "blues", "jazz", "rnb_soul", "hiphop_rap", "classical"]


@dataclass(frozen=True)
class AnalyzerConfig:
    # Audio / CREPE
    sample_rate: int = 16000          # CREPE expects 16kHz
    step_size_ms: int = 20            # CREPE hop in ms (10ms default)
    model_capacity: str = "tiny"      # tiny|small|medium|large|full
    viterbi: bool = True              # CREPE viterbi smoothing
    confidence_threshold: float = 0.25 # drop unvoiced/low-conf frames

    # Tuning
    a4_hz: float = 440.0

    # Pitch cleanup / smoothing
    median_filter_ms: int = 50
    max_cents_jump_per_frame: float = 150.0  # for continuity penalties

    # Note segmentation
    min_note_duration_ms: int = 80
    max_transition_ms: int = 80

    # Vibrato detection
    vibrato_window_s: float = 0.6
    vibrato_hz_min: float = 4.0
    vibrato_hz_max: float = 8.0
    vibrato_depth_cents_min: float = 15.0
    vibrato_depth_cents_max: float = 90.0
    vibrato_strength_threshold: float = 0.35

    # Scoring curve: score = 100 * exp(-(abs(dev)/sigma)^alpha)
    # Fitted to your desired anchor points: 10c->~90%, 25c->~70%, 50c->~40%
    score_sigma_cents: float = 53.5
    score_alpha: float = 1.35

    # Musical validity weights (scaled later by strictness)
    weight_valid: float = 1.0
    weight_genre_allowed: float = 0.90
    weight_passing: float = 0.95
    weight_invalid: float = 0.60

    # Strictness: 0 = forgiving, 1 = strict
    strictness: float = 0.6
