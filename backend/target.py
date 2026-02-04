from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from .music_theory import cents_to_hz, hz_to_cents, cents_to_midi, midi_to_pitch_class, AllowedPitchClasses


@dataclass(frozen=True)
class TargetInference:
    target_cents: np.ndarray
    target_hz: np.ndarray
    target_pc: np.ndarray           # pitch class per frame (-1 if unvoiced)
    note_id: np.ndarray             # segment id per frame (-1 if unvoiced)
    note_duration_frames: np.ndarray


def _candidate_pcs(allowed: AllowedPitchClasses) -> List[int]:
    # candidates include diatonic + genre_allowed + passing_allowed
    pcs = set(allowed.diatonic) | set(allowed.genre_allowed) | set(allowed.passing_allowed)
    return sorted(pcs)


def infer_targets(
    time_s: np.ndarray,
    f0_hz: np.ndarray,
    allowed: AllowedPitchClasses,
    a4_hz: float,
    max_jump_cents_per_frame: float,
) -> TargetInference:
    """
    Framewise target inference with simple continuity:
    - For each frame, consider nearest pitch in allowed pitch classes across octaves near the sung pitch.
    - Dynamic programming to discourage big jumps.
    """
    n = len(f0_hz)
    sung_cents = np.array([hz_to_cents(f, a4_hz) for f in f0_hz], dtype=float)
    voiced = np.isfinite(sung_cents)

    target_cents = np.full(n, np.nan, dtype=float)
    target_pc = np.full(n, -1, dtype=int)

    pcs = _candidate_pcs(allowed)
    if not pcs or voiced.sum() < 3:
        return TargetInference(target_cents, np.full(n, np.nan), target_pc, np.full(n, -1), np.zeros(n, dtype=int))

    # Precompute candidate targets per frame: (cents_value, pc)
    # We generate candidates within +/- 1200 cents around sung pitch (one octave), across nearby octaves.
    candidates: List[List[Tuple[float, int]]] = [[] for _ in range(n)]
    for i in range(n):
        if not voiced[i]:
            continue
        midi = cents_to_midi(sung_cents[i])
        base_oct = int(np.floor(midi / 12.0))
        # Search octaves around the current estimate
        for oct_shift in (-1, 0, 1):
            octv = base_oct + oct_shift
            for pc in pcs:
                cand_midi = 12 * octv + pc
                cand_cents = 100.0 * cand_midi
                if abs(cand_cents - sung_cents[i]) <= 600:  # within half octave
                    candidates[i].append((cand_cents, pc))
        # Fallback: ensure at least one candidate (nearest pc at current octave)
        if not candidates[i]:
            best = None
            for pc in pcs:
                cand_midi = 12 * base_oct + pc
                cand_cents = 100.0 * cand_midi
                d = abs(cand_cents - sung_cents[i])
                if best is None or d < best[0]:
                    best = (d, cand_cents, pc)
            if best:
                candidates[i].append((best[1], best[2]))

    # Dynamic programming
    BIG = 1e18
    dp = [None] * n
    back = [None] * n

    # Initialize
    if candidates[0]:
        dp0 = np.array([abs(c - sung_cents[0]) for (c, _pc) in candidates[0]], dtype=float)
        dp[0] = dp0
        back[0] = np.full(len(dp0), -1, dtype=int)

    for i in range(1, n):
        if not candidates[i]:
            dp[i] = None
            back[i] = None
            continue
        cur = candidates[i]
        prev = candidates[i - 1]
        if dp[i - 1] is None or not prev:
            # Start fresh (no continuity)
            dp[i] = np.array([abs(c - sung_cents[i]) for (c, _pc) in cur], dtype=float)
            back[i] = np.full(len(cur), -1, dtype=int)
            continue

        prev_cost = dp[i - 1]
        new_cost = np.full(len(cur), BIG, dtype=float)
        new_back = np.full(len(cur), -1, dtype=int)

        for j, (c_cur, _pc_cur) in enumerate(cur):
            # best transition from prev candidates
            best_val = BIG
            best_k = -1
            for k, (c_prev, _pc_prev) in enumerate(prev):
                jump = abs(c_cur - c_prev)
                # soft penalty: if jump > max_jump, increase cost more steeply
                penalty = (jump / max_jump_cents_per_frame) ** 2
                val = prev_cost[k] + abs(c_cur - sung_cents[i]) + 15.0 * penalty
                if val < best_val:
                    best_val = val
                    best_k = k
            new_cost[j] = best_val
            new_back[j] = best_k

        dp[i] = new_cost
        back[i] = new_back

    # Backtrack best path
    # Find last frame with dp
    last = None
    for i in range(n - 1, -1, -1):
        if dp[i] is not None and candidates[i]:
            last = i
            break
    if last is None:
        return TargetInference(target_cents, np.full(n, np.nan), target_pc, np.full(n, -1), np.zeros(n, dtype=int))

    j = int(np.argmin(dp[last]))
    for i in range(last, -1, -1):
        if dp[i] is None or not candidates[i]:
            continue
        c, pc = candidates[i][j]
        target_cents[i] = c
        target_pc[i] = pc
        j_prev = int(back[i][j])
        if j_prev < 0:
            # when discontinuity breaks, keep nearest for remaining earlier voiced frames
            # (we’ll just stop following)
            break
        j = j_prev

    # Convert
    target_hz = np.array([cents_to_hz(c, a4_hz) if np.isfinite(c) else np.nan for c in target_cents], dtype=float)

    # Note segmentation: group contiguous same pitch class and nearby cents
    note_id = np.full(n, -1, dtype=int)
    note_duration = np.zeros(n, dtype=int)
    current = -1
    seg_start = 0

    def close_enough(c1: float, c2: float) -> bool:
        if not (np.isfinite(c1) and np.isfinite(c2)):
            return False
        return abs(c1 - c2) <= 70.0  # within ~semitone-ish window for same note center

    for i in range(n):
        if not np.isfinite(target_cents[i]):
            continue
        if current < 0:
            current = 0
            seg_start = i
            note_id[i] = current
            continue
        prev_i = i - 1
        if prev_i >= 0 and (target_pc[i] == target_pc[prev_i]) and close_enough(target_cents[i], target_cents[prev_i]):
            note_id[i] = current
        else:
            # close previous segment
            current += 1
            seg_start = i
            note_id[i] = current

    # Fill durations
    if current >= 0:
        for nid in range(current + 1):
            idx = np.where(note_id == nid)[0]
            if idx.size:
                note_duration[idx] = int(idx.size)

    return TargetInference(
        target_cents=target_cents,
        target_hz=target_hz,
        target_pc=target_pc,
        note_id=note_id,
        note_duration_frames=note_duration,
    )
