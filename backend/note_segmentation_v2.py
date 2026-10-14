"""
Note segmentation v2. Not used by the production backend (main.py / api.py /
evaluate.py still segment with v1); used to build the canonical Axis 2 feature
store via scripts/resegment_main_dataset.py with CANONICAL_CONFIG below.

v1 (`note_segmentation.py`) has two measured failure modes (see
ml/audits/axis2_segmentation_tail/):

  * It never looks at time. Voiced frames are walked by position only, so a
    10 s rest is treated exactly like a 20 ms one, and 96.7% of clearly invalid
    v1 notes contain an unvoiced break.
  * A boundary commits only after 4 consecutive frames round to the SAME new
    semitone. Material that slides (glides, fragments after a break, fast runs)
    changes its rounded semitone almost every frame, so the counter keeps
    restarting and the material is absorbed into the previous note.

v2 replaces both decisions and keeps everything else — inputs, voiced-only
frames, the 3-frame minimum, the output dict schema — identical to v1, so the
two can be compared frame for frame on the same saved track.

Inputs are exactly what `harmony.npz` stores: frame times, f0 with NaN marking
unvoiced frames, and the accompaniment tuning offset. No audio, no CREPE
re-run, and nothing from Axis 1 (chroma, harmonic fit) is consulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from note_segmentation import (
    MIN_SEGMENT_VOICED_FRAMES,
    NoteSeg,
    _core_slice,
    cents_diff,
    hz_to_midi_float,
    midi_to_hz,
    segment_notes_with_frame_indices as v1_segment_with_frame_indices,
)


@dataclass(frozen=True)
class SegmentationV2Config:
    # Any elapsed gap longer than this between consecutive voiced frames ends the
    # note unconditionally. Measured as the difference of frame timestamps, so a
    # gap-free run has gaps of exactly one hop (0.02 s).
    hard_gap_s: float = 0.35

    # A new pitch region must sit more than this far from the current note's
    # robust center. Must be below 100: an exact semitone step is 100 c, and
    # with a strict comparison a 100 c setting cannot commit one.
    separation_cents: float = 80.0

    # ...and hold for this many consecutive voiced frames, all on the same side
    # of the center and spanning no more than `separation_cents` themselves.
    # The spread condition is what keeps a glide from fragmenting: a slide never
    # sits still long enough to count as a new region.
    persistence_frames: int = 4

    # Trailing window (voiced frames) for the current note's center. 1 s spans
    # 4-7 vibrato cycles, so the partial cycle at the window edge biases the
    # median only slightly; a 300 ms window was tried first and let +-150 c
    # vibrato split (the median of 1.65 cycles leans toward one half-cycle).
    # Slow drift is still followed, with ~0.5 s lag.
    center_window_frames: int = 50

    # Vibrato is distinguished from a new note by coming back. A departure is
    # rejected if, within this many contiguous frames after the persistence
    # run, the pitch comes at least halfway back toward the old center. Wide
    # vibrato (+-120-150 c) spends 3-4 frames beyond an 80 c separation per
    # half-cycle, exactly like a short new note; what differs is that it swings
    # back within a fraction of a cycle. Kept short so that a melodic A-B-A
    # figure whose B lasts >= persistence + window frames still splits: an
    # earlier 6-frame window with a "within sep/2 of the center" test erased
    # ~27% of v1 boundaries, mostly real returns to a previous pitch.
    return_window_frames: int = 3

    # Contiguous frames after a short gap whose median is compared with the
    # note's center. 10 frames is about one vibrato cycle; 5 was tried first and
    # a half-cycle median on either side could differ by the vibrato extent.
    gap_context_frames: int = 10

    # What the pitch after a short gap is compared against:
    #   "center_window" — median of up to `center_window_frames` since the last
    #                     split. In the gap_v1 modes nothing but a gap split
    #                     advances that start, so the window spans earlier v1
    #                     notes and a held pitch resuming after a 60-160 ms
    #                     dropout can read as a change (836 such boundaries).
    #   "local"         — median of the gap-free run just before the gap, capped
    #                     at `gap_context_frames`, mirroring the "after" side.
    gap_before: str = "center_window"

    # How the completed segment's target MIDI note is chosen:
    #   "core_median" — v1's estimator: rounded median of the middle 60%.
    #   "modal"       — most common rounded semitone; ties go to the one nearest
    #                   the core median.
    target_estimator: str = "core_median"

    # Which boundary rules run inside each hard-gap region:
    #   "sustained"         short-gap rule + sustained-departure commit only
    #   "gap_v1"            short-gap rule, then v1's exact-semitone commit
    #   "gap_v1_sustained"  as gap_v1, then sustained departure as an extra path
    mode: str = "sustained"



# The configuration adopted for data/main_data (schema 2) and the Axis 2 ML
# dataset, recommended in ml/audits/axis2_segmentation_v2/segmentation_v2_summary.md
# as `gapv1local_g0.5_s100`. The dataclass defaults above are the sweep's base
# configuration and stay unchanged, so every recorded sweep result reproduces.
CANONICAL_CONFIG = SegmentationV2Config(
    mode="gap_v1",
    hard_gap_s=0.5,
    separation_cents=100,
    gap_before="local",
    gap_context_frames=10,
    target_estimator="core_median",
)


def _robust_center(p: np.ndarray) -> float:
    return float(np.median(p))


def _has_stable_run(p: np.ndarray, length: int, spread: float) -> bool:
    """True if some `length` consecutive values span no more than `spread`."""
    if p.size < length:
        return False
    for i in range(p.size - length + 1):
        w = p[i : i + length]
        if float(w.max() - w.min()) <= spread:
            return True
    return False


def _contiguous_end(t: np.ndarray, k: int, limit: int, short_gap: float) -> int:
    """End (exclusive) of the gap-free run starting at k, capped at k + limit."""
    e = min(t.size, k + limit)
    gaps = np.flatnonzero(np.diff(t[k:e]) > short_gap)
    return k + int(gaps[0]) + 1 if gaps.size else e


def _contiguous_start(t: np.ndarray, k: int, floor: int, short_gap: float) -> int:
    """Start of the gap-free run ending at k - 1, no earlier than `floor`."""
    gaps = np.flatnonzero(np.diff(t[floor:k]) > short_gap)
    return floor + int(gaps[-1]) + 1 if gaps.size else floor


def _split_region(
    p: np.ndarray,
    t: np.ndarray,
    hop_s: float,
    cfg: SegmentationV2Config,
    use_gap_rule: bool = True,
    use_sustained: bool = True,
) -> List[Tuple[int, int]]:
    """
    Split one hard-gap-free region of voiced frames. `p` is pitch in absolute
    tuning-corrected cents (MIDI * 100), `t` the frame times. Returns
    [start, end) ranges into the region.
    """
    n = p.size
    sep = cfg.separation_cents
    y = cfg.persistence_frames
    ctx = cfg.gap_context_frames
    short_gap = 1.5 * hop_s

    bounds: List[Tuple[int, int]] = []
    note_start = 0
    k = 1
    while k < n:
        # Short unvoiced gap: split only if the pitch resumes somewhere else.
        # Both sides are medians over several frames, so one noisy frame cannot
        # decide it.
        if use_gap_rule and t[k] - t[k - 1] > short_gap:
            if cfg.gap_before == "local":
                before = _robust_center(p[_contiguous_start(t, k, max(note_start, k - ctx), short_gap) : k])
            elif cfg.gap_before == "center_window":
                before = _robust_center(p[max(note_start, k - cfg.center_window_frames) : k])
            else:
                raise ValueError(f"unknown gap_before: {cfg.gap_before!r}")
            after = _robust_center(p[k : _contiguous_end(t, k, ctx, short_gap)])
            if abs(after - before) > sep:
                bounds.append((note_start, k))
                note_start = k
                k += 1
                continue

        # Sustained departure from the current note's robust center.
        if use_sustained and k + y <= n and k - note_start >= MIN_SEGMENT_VOICED_FRAMES:
            run_t = t[k : k + y]
            if np.all(np.diff(run_t) <= short_gap):
                center = _robust_center(p[max(note_start, k - cfg.center_window_frames) : k])
                run = p[k : k + y] - center
                same_side = bool(np.all(run > sep) or np.all(run < -sep))
                stable = float(run.max() - run.min()) <= sep
                # Only frames contiguous with the run: what happens after an
                # unvoiced gap is the gap rule's decision, not evidence of vibrato.
                after = p[k + y : _contiguous_end(t, k + y, cfg.return_window_frames, short_gap)] if k + y < n else p[:0]
                departure = float(np.median(run))
                returns = bool(np.any(np.abs(after - center) <= 0.5 * abs(departure)))
                # The note being left must have actually settled somewhere.
                # Otherwise it was an onset scoop or a slide into this region,
                # and belongs to the note that follows rather than being cut
                # off as a fragment of its own.
                if same_side and stable and not returns and _has_stable_run(p[note_start:k], y, sep):
                    bounds.append((note_start, k))
                    note_start = k
                    k += 1
                    continue
        k += 1

    bounds.append((note_start, n))
    return bounds


def _target_note(midi_float: np.ndarray, estimator: str) -> int:
    core = _core_slice(midi_float)
    core_target = int(np.round(np.median(core)))
    if estimator == "core_median":
        return core_target
    if estimator == "modal":
        notes, counts = np.unique(np.round(midi_float).astype(int), return_counts=True)
        best = notes[counts == counts.max()]
        return int(best[np.argmin(np.abs(best - np.median(core)))])
    raise ValueError(f"unknown target_estimator: {estimator!r}")


def _finalize(
    times: np.ndarray,
    hz: np.ndarray,
    tuning_semitones: float,
    estimator: str,
) -> NoteSeg:
    midi_float = np.asarray(hz_to_midi_float(hz, tuning_semitones), dtype=float)
    target = _target_note(midi_float, estimator)
    cents_all = cents_diff(hz, midi_to_hz(target, tuning_semitones))
    return NoteSeg(
        start=float(times[0]),
        end=float(times[-1]),
        target_note=target,
        median_cents=float(np.median(cents_all)),
        core_median_cents=float(np.median(_core_slice(cents_all))),
        n_voiced_frames=int(hz.size),
    )


def segment_notes_v2_with_frame_indices(
    times: np.ndarray,
    freqs: np.ndarray,
    tuning_semitones: float = 0.0,
    config: Optional[SegmentationV2Config] = None,
) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Same contract as v1's `segment_notes_with_frame_indices`: NaN in `freqs`
    means unvoiced, returns (segment dicts, per-segment source frame indices).
    """
    cfg = config or SegmentationV2Config()
    times = np.asarray(times, dtype=float)
    freqs = np.asarray(freqs, dtype=float)

    vp = np.flatnonzero(~np.isnan(freqs))
    if vp.size == 0:
        return [], []

    hop_s = float(np.median(np.diff(times))) if times.size > 1 else 0.02
    t = times[vp]
    p = 100.0 * np.asarray(hz_to_midi_float(freqs[vp], tuning_semitones), dtype=float)

    # Hard gaps first: nothing may span one, whatever the pitch does.
    cuts = np.flatnonzero(np.diff(t) > cfg.hard_gap_s) + 1
    region_edges = np.concatenate(([0], cuts, [vp.size]))

    pieces: List[np.ndarray] = []  # voiced-position index arrays
    for r0, r1 in zip(region_edges[:-1], region_edges[1:]):
        if cfg.mode == "sustained":
            for a, b in _split_region(p[r0:r1], t[r0:r1], hop_s, cfg):
                pieces.append(np.arange(r0 + a, r0 + b))
            continue
        # Hybrid modes: gap rules first, then v1's own commit logic inside each
        # gap-delimited piece, so v1's sensitivity to semitone steps is kept.
        for a, b in _split_region(p[r0:r1], t[r0:r1], hop_s, cfg, use_sustained=False):
            lo, hi = vp[r0 + a], vp[r0 + b - 1] + 1
            _segs, v1_idx = v1_segment_with_frame_indices(times[lo:hi], freqs[lo:hi], tuning_semitones)
            for idx_raw in v1_idx:
                pos = np.searchsorted(vp, idx_raw + lo)
                if cfg.mode == "gap_v1":
                    pieces.append(pos)
                elif cfg.mode == "gap_v1_sustained":
                    # extra commit path: sustained departures v1 cannot see
                    for c, d in _split_region(p[pos], t[pos], hop_s, cfg, use_gap_rule=False):
                        pieces.append(pos[c:d])
                else:
                    raise ValueError(f"unknown mode: {cfg.mode!r}")

    segments: List[Dict] = []
    frame_indices: List[np.ndarray] = []
    for pos in pieces:
        if pos.size < MIN_SEGMENT_VOICED_FRAMES:
            continue
        idx = vp[pos]
        seg = _finalize(times[idx], freqs[idx], tuning_semitones, cfg.target_estimator)
        segments.append(seg.to_dict())
        frame_indices.append(idx.copy())
    return segments, frame_indices


def segment_notes_v2(
    times: np.ndarray,
    freqs: np.ndarray,
    tuning_semitones: float = 0.0,
    config: Optional[SegmentationV2Config] = None,
) -> List[Dict]:
    segments, _frame_indices = segment_notes_v2_with_frame_indices(times, freqs, tuning_semitones, config)
    return segments
