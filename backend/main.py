from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from harmony import build_harmonic_context, score_key_compliance
from note_segmentation import segment_notes
from preprocess import detect_pitch
from scoring import build_graph_points, score_intonation
from separation import separate


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return x


def _duration_weighted_mean(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(np.sum(w))
    if total <= 0:
        return float(np.mean(v))
    return float(np.sum(v * w) / total)


def compute_metrics(
    segments: List[Dict[str, Any]],
    voiced_coverage: float,
    tuning_cents: float,
    tempo_bpm: float,
) -> Dict[str, Any]:
    """
    Aggregate both axes, weighted by note duration so a grace note does not count
    as much as a sustained one.
    """
    scored_for_key = [s for s in segments if s.get("key_compliance") is not None]

    key_values = [float(s["key_compliance"]) for s in scored_for_key]
    key_weights = [float(s["duration_s"]) for s in scored_for_key]

    int_values = [float(s["intonation_score"]) for s in segments]
    int_weights = [float(s["duration_s"]) for s in segments]

    deviations = np.asarray([float(s["abs_cents_deviation"]) for s in segments], dtype=float)

    return {
        "key_compliance": _duration_weighted_mean(key_values, key_weights) if scored_for_key else None,
        "intonation_accuracy": _duration_weighted_mean(int_values, int_weights),
        "median_cents_deviation": float(np.median(deviations)) if deviations.size else 0.0,
        "total_notes_analyzed": int(len(segments)),
        "notes_scored_for_key": int(len(scored_for_key)),
        "voiced_coverage": float(voiced_coverage),
        "tuning_offset_cents": float(tuning_cents),
        "tempo_bpm": float(tempo_bpm),
        "has_accompaniment": bool(scored_for_key),
    }


def build_gemini_prompt(metrics: Dict[str, Any]) -> str:
    if metrics["key_compliance"] is None:
        key_section = """1. KEY COMPLIANCE - NOT MEASURED. No instrumental was found in this upload, so there was
   no harmonic reference to compare the sung notes against. Say plainly that note choice could
   not be assessed and that uploading the full song (instrumental plus vocals) would enable it.
   Do not guess at it or infer it from the intonation figure."""
    else:
        key_section = f"""1. KEY COMPLIANCE ({metrics['key_compliance']:.1%}) - whether the notes they chose fit the song's
   own harmony. Derived by comparing each sung note against the pitch-class content of the
   separated instrumental at that moment, allowing about one beat of slack for notes sung
   slightly before or after a chord change. A low value can mean adventurous or borderline note
   choices, not necessarily mistakes."""

    return f"""
You are a supportive vocal coach. Write a concise, constructive report (6-10 sentences) for a singer.

The analysis measured two INDEPENDENT things. Discuss them separately and never average them:

{key_section}
2. INTONATION ACCURACY ({metrics['intonation_accuracy']:.1%}) - whether those notes were sung
   cleanly. Median absolute deviation was {metrics['median_cents_deviation']:.1f} cents, measured
   against a target corrected for this song's tuning offset of {metrics['tuning_offset_cents']:+.1f} cents.

Context:
- Notes analyzed: {metrics['total_notes_analyzed']} (of which {metrics['notes_scored_for_key']} had usable harmonic context)
- Confidently scorable vocal: {metrics['voiced_coverage']:.0%} of the track
- Detected tempo: {metrics['tempo_bpm']:.0f} BPM

Requirements:
- Address the axes separately, then briefly explain what their combination suggests.
  Strong intonation with weaker key compliance suggests confident execution of adventurous
  choices; the reverse suggests safe choices needing tighter pitch control.
- Give practical advice: note centers, phrasing, breath support, ear training.
- Do NOT claim the singer is overall sharp or flat - only unsigned deviations were measured.
- Do NOT comment on vibrato or portamento; they were not measured.
- If coverage is below about 70%, note that parts of the vocal could not be confidently
  analyzed, which happens with layered harmonies or heavy production.
""".strip()


def generate_report(metrics: Dict[str, Any]) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Report could not be generated: missing GEMINI_API_KEY in environment."

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(build_gemini_prompt(metrics))
        return (response.text or "").strip() or "Report generated, but the response was empty."
    except Exception as e:
        return f"Report unavailable ({type(e).__name__}: {e})"


def run_backend(
    audio_path: str,
    *,
    with_report: bool = True,
) -> Tuple[List[Tuple[float, Optional[float]]], Dict[str, Any], str]:
    """
    Full-mix analysis. Returns (graph_points, metrics, report).

    The two axes are independent measurements sharing two prerequisites: stem
    separation and note segmentation. Both prerequisites are sequential and
    dominate runtime, so the axes are separate functions for testability rather
    than for concurrency.
    """
    stems = separate(audio_path)

    context = build_harmonic_context(
        stems.other, stems.percussive, stems.sample_rate, vocals=stems.vocals
    )

    track = detect_pitch(stems.vocals, stems.sample_rate)

    segments = segment_notes(track.time, track.frequency, context.tuning_semitones)
    if not segments:
        raise ValueError(
            "No sung notes were detected. The vocal may be absent, too quiet, or "
            "too heavily processed to track."
        )

    segments = score_key_compliance(segments, context)
    segments = score_intonation(segments)

    graph_points = build_graph_points(segments)
    metrics = compute_metrics(
        segments,
        voiced_coverage=track.coverage,
        tuning_cents=context.tuning_cents,
        tempo_bpm=context.tempo_bpm,
    )
    report = generate_report(metrics) if with_report else ""

    return graph_points, metrics, report


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_path> [--no-report]")
        print("Example: python main.py sample_songs/dont.mp3")
        return 2

    audio_path = sys.argv[1]
    with_report = "--no-report" not in sys.argv[2:]

    graph_points, metrics, report = run_backend(audio_path, with_report=with_report)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(_to_jsonable(metrics), indent=4), encoding="utf-8")
    (out_dir / "graph_points.json").write_text(json.dumps(_to_jsonable(graph_points), indent=4), encoding="utf-8")
    if report:
        (out_dir / "report.txt").write_text(report, encoding="utf-8")

    key = metrics["key_compliance"]
    print("ANALYSIS COMPLETE")
    print(f"  Key compliance      {key:.1%}" if key is not None else "  Key compliance      n/a (no instrumental found)")
    print(f"  Intonation accuracy {metrics['intonation_accuracy']:.1%}")
    print(f"  Coverage            {metrics['voiced_coverage']:.0%}")
    print(f"  Notes               {metrics['total_notes_analyzed']} ({metrics['notes_scored_for_key']} with harmony)")
    print(f"  Tuning offset       {metrics['tuning_offset_cents']:+.1f} cents")
    print(f"  Tempo               {metrics['tempo_bpm']:.0f} BPM")
    print(f"  -> {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
