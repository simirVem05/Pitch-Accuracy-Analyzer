from __future__ import annotations

import argparse
import json

from .pipeline import analyze_vocals
from .config import AnalyzerConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Pitch Accuracy Analyzer (core backend, no API server)")
    p.add_argument("wav_path", type=str, help="Path to a cappella WAV file")
    p.add_argument("--key", required=True, type=str, help="Declared key e.g. 'C major', 'E♭ minor'")
    p.add_argument("--genre", required=True, type=str,
                   choices=["pop","rock","blues","jazz","rnb_soul","hiphop_rap","classical"])
    p.add_argument("--strictness", type=float, default=0.6, help="0..1")
    p.add_argument("--confidence", type=float, default=0.5, help="CREPE confidence threshold")

    args = p.parse_args()

    cfg = AnalyzerConfig(strictness=float(args.strictness), confidence_threshold=float(args.confidence))
    result = analyze_vocals(args.wav_path, args.key, args.genre, cfg)

    print(json.dumps({
        "summary": result.summary,
        "gemini_prompt": result.gemini_prompt,
        "graph": result.graph,
    }, indent=2))


if __name__ == "__main__":
    main()
