# main.py
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv

from preprocess import preprocess_audio_and_detect_pitch
from note_segmentation import segment_and_build_allowed
from note_classification import run_step_1_3
from scoring import score_segments


def _normalize_genre(genre: str) -> str:
    g = genre.strip().lower()
    if g in {"rnb", "r&b", "r and b"}:
        return "r&b"
    return g


def _parse_key(key_str: str) -> Tuple[str, str]:
    """
    Accepts: "B minor", "D major", "F# minor", "Bb major"
    Returns: (tonic, scale) where scale is "major" or "minor"
    """
    s = key_str.strip()
    parts = s.split()
    if len(parts) < 2:
        raise ValueError('Key must look like "B minor" or "D major".')

    tonic = parts[0].strip()
    mode = parts[1].strip().lower()

    if mode in {"maj", "major", "ionian"}:
        scale = "major"
    elif mode in {"min", "minor", "aeolian"}:
        scale = "minor"
    else:
        raise ValueError(f'Unrecognized mode "{parts[1]}". Use major/minor.')

    return tonic, scale


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    return x


def compute_metrics(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Note-count based ratios (as you chose).
    Buckets:
      high >= 0.80
      mediocre 0.60–0.80
      low < 0.60
    """
    if not segments:
        return {
            "median_cents_deviation": 0.0,
            "high_ratio": 0.0,
            "mediocre_ratio": 0.0,
            "low_ratio": 0.0,
            "total_notes_analyzed": 0,
            "vibrato_detected_count": 0,
            "portamento_detected_count": 0,
        }

    scores = np.array([float(s.get("on_key_score", 0.0)) for s in segments], dtype=float)

    devs = []
    for s in segments:
        v = s.get("core_median_cents", s.get("median_cents", 0.0))
        try:
            devs.append(abs(float(v)))
        except Exception:
            devs.append(0.0)
    devs = np.array(devs, dtype=float)

    total = int(len(scores))
    high = int(np.sum(scores >= 0.80))
    mediocre = int(np.sum((scores >= 0.60) & (scores < 0.80)))
    low = int(np.sum(scores < 0.60))

    vib_present = 0
    port_present = 0
    for s in segments:
        vib = s.get("vibrato") or {}
        por = s.get("portamento") or {}
        if bool(vib.get("present")):
            vib_present += 1
        if bool(por.get("present")):
            port_present += 1

    return {
        "median_cents_deviation": float(np.median(devs)) if devs.size else 0.0,
        "high_ratio": float(high / total) if total else 0.0,
        "mediocre_ratio": float(mediocre / total) if total else 0.0,
        "low_ratio": float(low / total) if total else 0.0,
        "total_notes_analyzed": total,
        "vibrato_detected_count": int(vib_present),
        "portamento_detected_count": int(port_present),
    }


def build_gemini_prompt(metrics: Dict[str, Any], key: str, genre: str) -> str:
    return f"""
You are a supportive vocal coach. Write a concise, constructive report (6–10 sentences) for a singer.

Context:
- Genre: {genre}
- Key selected by the user: {key}

Pitch metrics (IMPORTANT: these are score buckets, not sharp/flat):
- Median absolute cents deviation: {metrics.get("median_cents_deviation")}
- High-score ratio (>= 80%): {metrics.get("high_ratio")}
- Mediocre-score ratio (60–80%): {metrics.get("mediocre_ratio")}
- Low-score ratio (< 60%): {metrics.get("low_ratio")}
- Notes analyzed: {metrics.get("total_notes_analyzed")}
- Vibrato detected count: {metrics.get("vibrato_detected_count")}
- Portamento detected count: {metrics.get("portamento_detected_count")}

Requirements:
- Practical advice (intonation consistency, note centers, phrasing, breath support).
- Mention vibrato/portamento as stylistic observations only (do not judge good/bad).
- Do NOT claim the singer is sharp or flat overall (no signed cents stats are provided).
""".strip()


def call_gemini_25_flash(prompt: str) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Report could not be generated: missing GEMINI_API_KEY in environment."

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(prompt)
        return (resp.text or "").strip() or "Report generated, but the response was empty."
    except Exception as e:
        return f"API Error: {e}"


def _run_preprocess(audio_path: str):
    """
    Calls preprocess_audio_and_detect_pitch using ONLY the kwargs it supports.
    This prevents mismatch errors when you refactor preprocess.py.
    """
    sig = inspect.signature(preprocess_audio_and_detect_pitch)
    params = sig.parameters

    kwargs = {}
    # Prefer audio_path kw if it exists
    if "audio_path" in params:
        kwargs["audio_path"] = audio_path
    else:
        # Fallback: first positional arg
        return preprocess_audio_and_detect_pitch(audio_path)

    return preprocess_audio_and_detect_pitch(**kwargs)


def run_backend(audio_path: str, key: str, genre: str) -> Tuple[List[Tuple[float, float]], Dict[str, Any], str]:
    """
    Orchestrates the backend pipeline.
    Returns:
      graph_tuples, metrics_dict, report_text
    """
    genre_norm = _normalize_genre(genre)
    tonic, scale = _parse_key(key)

    # 1) Preprocess + CREPE
    # Expected return: time, frequency, confidence, activation
    time, frequency, confidence, activation = _run_preprocess(audio_path)

    # 2) Note segmentation + allowed pitch classes
    segments, allowed_pcs, diatonic_pcs, blue_pcs, tonic_pc = segment_and_build_allowed(
        time,
        frequency,
        tonic=tonic,
        scale=scale,
        genre=genre_norm,
    )

    # 3) Classification + technique presence detection
    segments, _ = run_step_1_3(
        segments,
        time,
        frequency,
        allowed_pitch_classes=allowed_pcs,
        diatonic_pitch_classes=diatonic_pcs,
        blue_pitch_classes=blue_pcs,
    )

    # 4) Scoring -> tuples
    segments, tuples = score_segments(segments, score_scale="fraction")

    # 5) Metrics + Report
    metrics = compute_metrics(segments)
    prompt = build_gemini_prompt(metrics, key=key, genre=genre_norm)
    report = call_gemini_25_flash(prompt)

    return tuples, metrics, report


def main() -> int:
    if len(sys.argv) < 4:
        print('Usage: python main.py <audio_path> "<Key like B minor>" <genre>')
        print('Example: python main.py sample_songs/dont.mp3 "B minor" rnb')
        return 2

    audio_path = sys.argv[1]
    key = sys.argv[2]
    genre = sys.argv[3]

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    tuples, metrics, report = run_backend(audio_path, key, genre)

    (out_dir / "metrics.json").write_text(json.dumps(_to_jsonable(metrics), indent=4), encoding="utf-8")
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    (out_dir / "graph_tuples.json").write_text(json.dumps(_to_jsonable(tuples), indent=4), encoding="utf-8")

    print("✅ BACKEND COMPLETE")
    print(f"- outputs/metrics.json")
    print(f"- outputs/report.txt")
    print(f"- outputs/graph_tuples.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
