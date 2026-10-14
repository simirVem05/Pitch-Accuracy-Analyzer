#!/usr/bin/env python3
"""
Synthetic behavioral checks for segmentation v2 (and v1, for contrast).

Each case is a hand-built f0 track at 20 ms with a known intended segmentation.
These are pass/fail assertions on behavior the design depends on; they need no
audio and no dataset. Run before trusting the real-data sweep.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "backend"))
from note_segmentation import segment_notes_with_frame_indices as v1  # noqa: E402
from note_segmentation_v2 import SegmentationV2Config, segment_notes_v2_with_frame_indices as v2  # noqa: E402

HOP = 0.02


def hz(midi_cents):
    return 440.0 * 2 ** ((np.asarray(midi_cents, float) / 100.0 - 69.0) / 12.0)


def track(*parts):
    """parts: arrays of absolute cents (MIDI*100) or ints = number of NaN frames."""
    f = []
    for part in parts:
        f.extend([np.nan] * part if isinstance(part, int) else list(hz(part)))
    f = np.array(f)
    return np.arange(f.size) * HOP, f


def frames(n, cents):
    return np.full(n, float(cents))


def vib(n, center, extent, rate=5.5):
    return center + extent * np.sin(2 * np.pi * rate * np.arange(n) * HOP)


CASES = [
    # name, track, expected v2 note count
    ("steady note", track(frames(50, 6000)), 1),
    ("vibrato +-100c @5.5Hz, 2 s", track(vib(100, 6000, 100)), 1),
    ("vibrato +-150c @5.5Hz, 2 s", track(vib(100, 6000, 150)), 1),
    ("vibrato +-120c @4.5Hz, 2 s", track(vib(100, 6000, 120, rate=4.5)), 1),
    ("vibrato +-100c @5.5Hz, 0.4 s (short note)", track(vib(20, 6000, 100)), 1),
    ("vibrato straddling semitone boundary (center +50c)", track(vib(100, 6050, 80)), 1),
    ("slow drift 0 -> +80c over 2 s", track(np.linspace(6000, 6080, 100)), 1),
    ("onset scoop -200 -> 0 over 4 frames, then hold", track(np.r_[np.linspace(5800, 6000, 4), frames(30, 6000)]), 1),
    ("release fall 0 -> -900 over 8 frames", track(np.r_[frames(30, 6000), np.linspace(6000, 5100, 8)]), 1),
    ("two notes, step +200c, no gap", track(frames(20, 6000), frames(20, 6200)), 2),
    ("two notes joined by 6-frame glide", track(frames(20, 6000), np.linspace(6000, 6300, 6), frames(20, 6300)), 2),
    ("same pitch re-sung after 0.2 s gap", track(frames(20, 6000), 10, frames(20, 6005)), 1),
    ("different pitch after 0.2 s gap", track(frames(20, 6000), 10, frames(20, 6300)), 2),
    ("same pitch after 10 s gap (hard gap)", track(frames(20, 6000), 500, frames(20, 6000)), 2),
    ("sliding fragment after 0.2 s gap", track(frames(20, 6000), 10, np.linspace(5450, 5300, 6)), 2),
    ("single-frame +700c blip", track(np.r_[frames(15, 6000), [6700], frames(15, 6000)]), 1),
    ("vibrato +-120c @4Hz, 2 s", track(vib(100, 6000, 120, rate=4.0)), 1),
    ("note then +200c neighbor for 8 frames then back", track(frames(20, 6000), frames(8, 6200), frames(20, 6000)), None),
    ("note then +200c neighbor for 15 frames then back", track(frames(20, 6000), frames(15, 6200), frames(20, 6000)), 3),
    ("vibrato +-100c with a 0.1 s dropout mid-note", track(vib(50, 6000, 100), 5, vib(50, 6000, 100)[7:]), 1),
    ("semitone step +100c, no gap", track(frames(20, 6000), frames(20, 6100)), 2),
    ("+300c region 6 frames, short gap, back to 0", track(frames(20, 6000), frames(6, 6300), 8, frames(20, 6000)), 3),
    ("3-frame melisma notes 0,+200,+400,+200", track(frames(3, 6000), frames(3, 6200), frames(3, 6400), frames(3, 6200)), None),
]


MODES = ("sustained", "gap_v1", "gap_v1_sustained", "recommended")

# the configuration recommended in segmentation_v2_summary.md
RECOMMENDED = dict(mode="gap_v1", hard_gap_s=0.5, separation_cents=100, gap_before="local")


def main() -> int:
    cfgs = {m: SegmentationV2Config(mode=m) for m in MODES[:-1]}
    cfgs["recommended"] = SegmentationV2Config(**RECOMMENDED)
    print(f"base config: {cfgs['sustained']}")
    print(f"{'case':55s} {'v1':>3s} " + " ".join(f"{m[:14]:>14s}" for m in MODES) + "  want")
    fails = Counter()
    for name, (t, f), want in CASES:
        n1 = len(v1(t, f, 0.0)[0])
        cells = []
        for m in MODES:
            n2 = len(v2(t, f, 0.0, cfgs[m])[0])
            bad = want is not None and n2 != want
            fails[m] += bad
            cells.append(f"{n2:>12d}{' X' if bad else '  '}")
        print(f"{name:55s} {n1:3d} " + " ".join(cells) + f"  {want if want is not None else 'n/a'}")
    for m in MODES:
        print(f"{m}: {'all checks passed' if not fails[m] else str(fails[m]) + ' FAILED'}")
    # the shipped default must pass everything
    return 1 if fails[SegmentationV2Config().mode] else 0


if __name__ == "__main__":
    raise SystemExit(main())
