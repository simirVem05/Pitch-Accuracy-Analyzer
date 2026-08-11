"""
Label-free evaluation harness for Axis 1 (key compliance).

Released vocals are by definition in-key, so the system should rate the notes
actually sung above the same notes relabelled to a wrong pitch class. That gives
a falsifiable target without any human annotation.

Perturbation is symbolic — only the integer `pitch_class` of each note segment is
changed. Axis 1 reads nothing but pitch class and time bounds, so relabelling is
a complete simulation of "what if a different note were sung here," and it avoids
re-running separation and pitch tracking (which would change segmentation and
confound the result).

Usage:
    python evaluate.py                    # all songs in test_songs/
    python evaluate.py path/to/song.mp3   # one file
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from harmony import build_harmonic_context, score_key_compliance
from note_segmentation import segment_notes
from preprocess import detect_pitch
from separation import Stems, separate


CACHE_DIR = Path("/tmp/pitch_eval_cache")
TEST_DIR = Path("test_songs")
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a"}

PERTURB_FRACTION = 0.20
PERTURB_SEED = 0xC0FFEE


@dataclass
class Prepared:
    name: str
    stems: Stems
    segments: List[Dict]


def _cached_stems(path: Path) -> Stems:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{path.stem}.npz"
    if cache.exists():
        d = np.load(cache)
        return Stems(
            vocals=d["vocals"],
            other=d["other"],
            bass=d["bass"],
            percussive=d["percussive"],
            sample_rate=int(d["sr"]),
        )
    stems = separate(str(path))
    np.savez(
        cache,
        vocals=stems.vocals,
        other=stems.other,
        bass=stems.bass,
        percussive=stems.percussive,
        sr=stems.sample_rate,
    )
    return stems


def prepare(path: Path) -> Optional[Prepared]:
    stems = _cached_stems(path)
    ctx = build_harmonic_context(stems.harmony, stems.percussive, stems.sample_rate, vocals=stems.vocals)
    track = detect_pitch(stems.vocals, stems.sample_rate)
    segments = segment_notes(track.time, track.frequency, ctx.tuning_semitones)
    if not segments:
        return None
    return Prepared(name=path.stem, stems=stems, segments=segments)


def _mean_rank(segments: List[Dict], ctx, shifts: np.ndarray) -> float:
    fake = []
    for seg, shift in zip(segments, shifts):
        copy = dict(seg)
        copy["pitch_class"] = (int(seg["pitch_class"]) + int(shift)) % 12
        fake.append(copy)
    scored = score_key_compliance(fake, ctx)
    values = [s["harmonic_rank"] for s in scored if s.get("harmonic_rank") is not None]
    return float(np.mean(values)) if values else float("nan")


def transposition_test(prep: Prepared, ctx) -> Dict[str, float]:
    """Global transposition: the true key should outrank all 11 wrong ones."""
    n = len(prep.segments)
    by_shift = {
        shift: _mean_rank(prep.segments, ctx, np.full(n, shift, dtype=int))
        for shift in range(12)
    }
    real = by_shift[0]
    wrong = [v for s, v in by_shift.items() if s != 0]
    return {
        "real": real,
        "wrong_mean": float(np.mean(wrong)),
        "wrong_best": float(np.max(wrong)),
        "separation": real - float(np.mean(wrong)),
        "margin_vs_best": real - float(np.max(wrong)),
        "placement": 1 + sum(1 for v in wrong if v > real),
        "by_shift": by_shift,
    }


def perturbation_test(prep: Prepared, ctx) -> Dict[str, float]:
    """
    Scatter wrong notes through an otherwise-correct performance.

    Closer to what a real user sounds like than a global key change, and it tests
    a different property: whether individual wrong notes are penalized at all,
    rather than whether a whole wrong key can be identified.
    """
    rng = np.random.default_rng(PERTURB_SEED)
    n = len(prep.segments)
    baseline = _mean_rank(prep.segments, ctx, np.zeros(n, dtype=int))

    deltas = []
    for _ in range(5):
        shifts = np.zeros(n, dtype=int)
        chosen = rng.choice(n, size=max(1, int(n * PERTURB_FRACTION)), replace=False)
        shifts[chosen] = rng.choice([-2, -1, 1, 2], size=chosen.size)
        deltas.append(baseline - _mean_rank(prep.segments, ctx, shifts))

    return {
        "baseline": baseline,
        "mean_drop": float(np.mean(deltas)),
        "min_drop": float(np.min(deltas)),
    }


def bleed_test(prep: Prepared) -> Dict[str, float]:
    """
    Is the signal real harmony, or the vocal's own leakage into `other`?

    Separation is imperfect, so the singer's note bleeds into the accompaniment
    stem — where it would make every sung note look "supported" by harmony that
    is really just their own voice. Bass sits mostly below the vocal range and so
    carries far less bleed. If separation survives on bass alone, the signal is
    harmonic; if it collapses, part of it was circular.
    """
    out: Dict[str, float] = {}
    for label, harmony in (
        ("other+bass", prep.stems.harmony),
        ("other_only", prep.stems.other),
        ("bass_only", prep.stems.bass),
    ):
        ctx = build_harmonic_context(
            harmony, prep.stems.percussive, prep.stems.sample_rate, vocals=prep.stems.vocals
        )
        result = transposition_test(prep, ctx)
        out[f"{label}_sep"] = result["separation"]
        out[f"{label}_place"] = float(result["placement"])
    return out


def main() -> int:
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        paths = sorted(p for p in TEST_DIR.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)

    if not paths:
        print(f"No audio found in {TEST_DIR}/")
        return 2

    rows = []
    for path in paths:
        prep = prepare(path)
        if prep is None:
            print(f"  skipped {path.name}: no notes detected")
            continue
        ctx = build_harmonic_context(
            prep.stems.harmony, prep.stems.percussive, prep.stems.sample_rate, vocals=prep.stems.vocals
        )
        rows.append((prep, ctx, transposition_test(prep, ctx), perturbation_test(prep, ctx), bleed_test(prep)))
        print(f"  done {path.stem}")

    print()
    print("=" * 104)
    print("TRANSPOSITION TEST — can the true key be identified? (placement 1 of 12 is ideal)")
    print("=" * 104)
    print(f"{'song':<34}{'notes':>7}{'real':>8}{'wrong':>8}{'best':>8}{'sep':>9}{'margin':>9}{'place':>8}")
    for prep, _ctx, t, _p, _b in rows:
        print(
            f"{prep.name[:34]:<34}{len(prep.segments):>7}{t['real']:>8.3f}{t['wrong_mean']:>8.3f}"
            f"{t['wrong_best']:>8.3f}{t['separation']:>+9.3f}{t['margin_vs_best']:>+9.3f}"
            f"{t['placement']:>6}/12"
        )
    places = [t["placement"] for _, _, t, _, _ in rows]
    print(f"{'MEAN':<34}{'':>7}{'':>8}{'':>8}{'':>8}"
          f"{np.mean([t['separation'] for _, _, t, _, _ in rows]):>+9.3f}"
          f"{np.mean([t['margin_vs_best'] for _, _, t, _, _ in rows]):>+9.3f}"
          f"{np.mean(places):>6.1f}/12")
    print(f"  true key ranked #1 on {sum(1 for p in places if p == 1)}/{len(places)} songs")

    print()
    print("=" * 104)
    print(f"PERTURBATION TEST — are scattered wrong notes ({PERTURB_FRACTION:.0%}) penalized?")
    print("=" * 104)
    print(f"{'song':<34}{'baseline':>10}{'mean drop':>12}{'min drop':>12}")
    for prep, _ctx, _t, p, _b in rows:
        print(f"{prep.name[:34]:<34}{p['baseline']:>10.3f}{p['mean_drop']:>+12.4f}{p['min_drop']:>+12.4f}")

    print()
    print("=" * 104)
    print("BLEED TEST — does separation survive without the vocal-contaminated `other` stem?")
    print("=" * 104)
    print(f"{'song':<34}{'other+bass':>14}{'other only':>14}{'bass only':>14}")
    for prep, _ctx, _t, _p, b in rows:
        print(
            f"{prep.name[:34]:<34}"
            f"{b['other+bass_sep']:>+9.3f} #{b['other+bass_place']:>2.0f}"
            f"{b['other_only_sep']:>+9.3f} #{b['other_only_place']:>2.0f}"
            f"{b['bass_only_sep']:>+9.3f} #{b['bass_only_place']:>2.0f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
