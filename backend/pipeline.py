from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

from .config import AnalyzerConfig, Genre
from .audio import load_audio_mono
from .pitch import extract_pitch_crepe, filter_by_confidence, median_smooth_f0
from .music_theory import parse_key, hz_to_cents, build_allowed_pitch_classes
from .vibrato import detect_vibrato
from .target import infer_targets
from .scoring import compute_scores
from .report import summarize, build_gemini_prompt


@dataclass(frozen=True)
class AnalysisResult:
    graph: Dict[str, Any]
    summary: Dict[str, Any]
    gemini_prompt: str


def analyze_vocals(
    wav_path: str,
    declared_key: str,
    genre: Genre,
    config: AnalyzerConfig = AnalyzerConfig(),
) -> AnalysisResult:
    audio = load_audio_mono(wav_path, target_sr=config.sample_rate)

    track = extract_pitch_crepe(
        audio.y,
        audio.sr,
        step_size_ms=config.step_size_ms,
        model_capacity=config.model_capacity,
        viterbi=config.viterbi,
    )
    track = filter_by_confidence(track, threshold=config.confidence_threshold)

    hop_hz = 1000.0 / config.step_size_ms  # frames/sec
    f0_sm = median_smooth_f0(track.f0_hz, sr_hop=hop_hz, window_ms=config.median_filter_ms)

    sung_cents = np.array([hz_to_cents(f, config.a4_hz) for f in f0_sm], dtype=float)

    k = parse_key(declared_key)
    allowed = build_allowed_pitch_classes(k, genre=genre)

    # Vibrato detection first (so target can reference vibrato centers)
    vib = detect_vibrato(
        time_s=track.time_s,
        pitch_cents=sung_cents,
        window_s=config.vibrato_window_s,
        hz_min=config.vibrato_hz_min,
        hz_max=config.vibrato_hz_max,
        depth_min=config.vibrato_depth_cents_min,
        depth_max=config.vibrato_depth_cents_max,
        strength_threshold=config.vibrato_strength_threshold,
    )

    # Infer targets
    target = infer_targets(
        time_s=track.time_s,
        f0_hz=f0_sm,
        allowed=allowed,
        a4_hz=config.a4_hz,
        max_jump_cents_per_frame=config.max_cents_jump_per_frame,
    )

    # If vibrato, treat the “target cents” as vibrato center for scoring
    target_cents_for_scoring = target.target_cents.copy()
    vib_center = vib.center_cents.copy()
    vib_mask = vib.is_vibrato & np.isfinite(vib_center) & np.isfinite(sung_cents)
    target_cents_for_scoring[vib_mask] = vib_center[vib_mask]

    # Frames threshold for passing classification
    min_note_frames = max(1, int(round((config.min_note_duration_ms / 1000.0) * hop_hz)))

    scores = compute_scores(
        sung_cents=sung_cents,
        target_cents=target_cents_for_scoring,
        target_pc=target.target_pc,
        allowed=allowed,
        note_duration_frames=target.note_duration_frames,
        min_note_frames_for_non_passing=min_note_frames,
        sigma=config.score_sigma_cents,
        alpha=config.score_alpha,
        strictness=config.strictness,
        w_valid=config.weight_valid,
        w_genre=config.weight_genre_allowed,
        w_passing=config.weight_passing,
        w_invalid=config.weight_invalid,
    )

    summ = summarize(
        time_s=track.time_s,
        score_final=scores.score_final,
        deviation_cents=scores.deviation_cents,
        is_vibrato=vib.is_vibrato,
        confidence=track.confidence,
        confidence_threshold=config.confidence_threshold,
    )

    # Build Gemini prompt (actual API call happens elsewhere later)
    gemini_prompt = build_gemini_prompt(
        declared_key=declared_key,
        genre=genre,
        summary=summ,
        notable_sections=None,
    )

    def _nan_to_none(x: np.ndarray):
        x = np.asarray(x, dtype=float)
        return np.where(np.isfinite(x), x, None).tolist()

    graph = {
        "time_s": track.time_s.tolist(),
        "score_pct": _nan_to_none(scores.score_final),
        "score_raw_pct": _nan_to_none(scores.score_raw),
        "f0_hz": _nan_to_none(f0_sm),
        "target_hz": _nan_to_none(target.target_hz),
        "deviation_cents": _nan_to_none(scores.deviation_cents),
        "confidence": track.confidence.tolist(),
        "vibrato_mask": vib.is_vibrato.astype(int).tolist(),
        "vibrato_strength": vib.strength.tolist(),
    }

    summary_dict = {
        "overall_score": summ.overall_score,
        "median_score": summ.median_score,
        "pct_frames_above_90": summ.pct_frames_above_90,
        "pct_frames_above_70": summ.pct_frames_above_70,
        "median_abs_dev_cents": summ.median_abs_dev_cents,
        "p95_abs_dev_cents": summ.p95_abs_dev_cents,
        "vibrato_pct": summ.vibrato_pct,
        "voiced_pct": summ.voiced_pct,
    }

    return AnalysisResult(graph=graph, summary=summary_dict, gemini_prompt=gemini_prompt)
