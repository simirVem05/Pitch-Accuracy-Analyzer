from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


# ---------------------------
# Detection-only thresholds
# ---------------------------

# Vibrato presence (keep reasonably strict so "present" means something)
VIBRATO_PRESENT_MIN_HZ = 4.0
VIBRATO_PRESENT_MAX_HZ = 8.5
VIBRATO_PRESENT_MIN_DURATION_S = 0.25
VIBRATO_PRESENT_MIN_PROMINENCE = 1.8
VIBRATO_PRESENT_MIN_WIDTH_CENTS = 8.0

# Portamento presence: duration window + minimum slide + "linear-ish"
PORTAMENTO_MIN_DUR_S = 0.08
PORTAMENTO_MAX_DUR_S = 0.70
PORTAMENTO_MIN_SLIDE_ABS_CENTS = 120.0
PORTAMENTO_MIN_R2 = 0.60

# Segment endpoint stability check
ENDPOINT_STABLE_WINDOW_FRAMES = 4
ENDPOINT_STABLE_MAX_PP_CENTS = 25.0  # stable if small peak-to-peak deviation

NEAR_ALLOWED_CENTS = 65.0
SHORT_ORNAMENT_MAX_S = 0.25


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def cents_series(freqs_hz: np.ndarray, target_hz: float) -> np.ndarray:
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    out = np.full_like(freqs_hz, np.nan, dtype=np.float64)
    m = ~np.isnan(freqs_hz)
    if np.any(m) and target_hz > 0:
        out[m] = 1200.0 * np.log2(freqs_hz[m] / target_hz)
    return out


def semitone_dist_pc(a_pc: int, b_pc: int) -> int:
    d = (a_pc - b_pc) % 12
    return min(d, 12 - d)


def _indices_for_segment(times: np.ndarray, seg: Dict[str, Any]) -> Tuple[int, int]:
    start_t = float(seg["start"])
    end_t = float(seg["end"])
    i0 = int(np.searchsorted(times, start_t, side="left"))
    i1 = int(np.searchsorted(times, end_t, side="right")) - 1
    i0 = max(0, min(i0, len(times) - 1))
    i1 = max(0, min(i1, len(times) - 1))
    if i1 < i0:
        i1 = i0
    return i0, i1


def _linear_fit_r2(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 3:
        return 0.0
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) + 1e-12
    return 1.0 - ss_res / ss_tot


# ---------------------------
# Classification (same idea)
# ---------------------------

def classify_segments(
    segments: List[Dict[str, Any]],
    *,
    allowed_pitch_classes: Set[int],
    diatonic_pitch_classes: Set[int],
    blue_pitch_classes: Set[int],
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    pcs = [int(s.get("pitch_class", int(s["target_note"]) % 12)) for s in segments]

    for i, seg in enumerate(segments):
        pc = pcs[i]
        seg["fits_key_strict"] = bool(pc in allowed_pitch_classes)

        if pc in diatonic_pitch_classes:
            seg["classification"] = "diatonic"
        elif pc in blue_pitch_classes:
            seg["classification"] = "blue"
        elif pc in allowed_pitch_classes:
            seg["classification"] = "chromatic"
        else:
            # near-miss chromaticism around allowed pcs
            em = float(seg["target_note"]) + (float(seg.get("median_cents", 0.0)) / 100.0)
            is_near = False
            for m in {int(np.floor(em)) - 1, int(np.floor(em)), int(np.floor(em)) + 1}:
                if (m % 12) in allowed_pitch_classes and abs((em - m) * 100.0) <= NEAR_ALLOWED_CENTS:
                    is_near = True
                    break
            seg["classification"] = "chromatic" if is_near else "dissonant"

        # distance-to-allowed for scoring penalty logic elsewhere
        if pc in allowed_pitch_classes:
            seg["pc_distance_to_allowed"] = 0
        else:
            seg["pc_distance_to_allowed"] = min(semitone_dist_pc(pc, apc) for apc in allowed_pitch_classes) if allowed_pitch_classes else 6

    # Contextual patterns (only for dissonant -> passing/neighbor/leading)
    for i, seg in enumerate(segments):
        if seg.get("classification") != "dissonant":
            continue
        if i == 0 or i == len(segments) - 1:
            seg["classification"] = "unknown"
            continue

        dur = float(seg["end"]) - float(seg["start"])
        prev_p, cur_p, next_p = pcs[i - 1], pcs[i], pcs[i + 1]
        p_ok = prev_p in allowed_pitch_classes
        n_ok = next_p in allowed_pitch_classes

        if p_ok and n_ok:
            if dur <= SHORT_ORNAMENT_MAX_S or prev_p != next_p:
                seg["classification"] = "passing"
            else:
                seg["classification"] = "neighbor"
        elif n_ok and semitone_dist_pc(cur_p, next_p) == 1:
            seg["classification"] = "leading"

    return segments


# ---------------------------
# Vibrato presence only
# ---------------------------

def detect_vibrato(segments: List[Dict[str, Any]], times: np.ndarray, freqs: np.ndarray) -> List[Dict[str, Any]]:
    for seg in segments:
        i0, i1 = _indices_for_segment(times, seg)
        t = times[i0 : i1 + 1]
        f = freqs[i0 : i1 + 1]

        target_hz = midi_to_hz(int(seg["target_note"]))
        cents = cents_series(f, target_hz)
        m = ~np.isnan(cents)

        seg["vibrato"] = {"present": False}

        if np.sum(m) < 10:
            continue

        t = t[m].astype(np.float64)
        x = cents[m].astype(np.float64)

        dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        if dur < VIBRATO_PRESENT_MIN_DURATION_S:
            continue

        dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        if dt <= 0:
            continue

        x0 = x - np.mean(x)
        xw = x0 * np.hanning(len(x0))

        f_fft = np.fft.rfftfreq(len(xw), d=dt)
        mag = np.abs(np.fft.rfft(xw))

        band = (f_fft >= VIBRATO_PRESENT_MIN_HZ) & (f_fft <= VIBRATO_PRESENT_MAX_HZ)
        if not np.any(band):
            continue

        prom = float(np.max(mag[band]) / (np.median(mag[band]) + 1e-9))
        width_pp = float(np.percentile(x0, 95) - np.percentile(x0, 5))

        seg["vibrato"] = {
            "present": bool(
                (prom >= VIBRATO_PRESENT_MIN_PROMINENCE)
                and (width_pp >= VIBRATO_PRESENT_MIN_WIDTH_CENTS)
            ),
            "rate_hz": float(f_fft[band][int(np.argmax(mag[band]))]),
            "width_pp_cents": width_pp,
            "prominence": prom,
        }

    return segments


def _stable_endpoint_ok(times: np.ndarray, freqs: np.ndarray, seg: Dict[str, Any], where: str) -> bool:
    """
    Endpoint stability: last/first small window of frames in a segment
    should be relatively stable in cents around the segment target.
    """
    i0, i1 = _indices_for_segment(times, seg)
    if where == "end":
        lo = max(i0, i1 - ENDPOINT_STABLE_WINDOW_FRAMES + 1)
        hi = i1
    else:
        lo = i0
        hi = min(i1, i0 + ENDPOINT_STABLE_WINDOW_FRAMES - 1)

    f = freqs[lo : hi + 1]
    if np.sum(~np.isnan(f)) < max(2, ENDPOINT_STABLE_WINDOW_FRAMES // 2):
        return False

    target_hz = midi_to_hz(int(seg["target_note"]))
    cents = cents_series(f, target_hz)
    cents = cents[np.isfinite(cents)]
    if cents.size < 2:
        return False

    pp = float(np.percentile(cents, 95) - np.percentile(cents, 5))
    return pp <= ENDPOINT_STABLE_MAX_PP_CENTS


# ---------------------------
# Portamento presence only
# ---------------------------

def detect_portamento(segments: List[Dict[str, Any]], times: np.ndarray, freqs: np.ndarray) -> List[Dict[str, Any]]:
    # Default
    for seg in segments:
        seg["portamento"] = {"present": False}

    for i in range(len(segments) - 1):
        seg_a = segments[i]
        seg_b = segments[i + 1]

        # Endpoint validity: both ends should be stable notes
        if not _stable_endpoint_ok(times, freqs, seg_a, "end"):
            continue
        if not _stable_endpoint_ok(times, freqs, seg_b, "start"):
            continue

        # Analyze a small bridge window between segments
        a1 = _indices_for_segment(times, seg_a)[1]
        b0 = _indices_for_segment(times, seg_b)[0]
        w_start = max(0, a1 - 3)
        w_end = min(len(times) - 1, b0 + 3)
        if w_end <= w_start:
            continue

        t = times[w_start : w_end + 1]
        f = freqs[w_start : w_end + 1]
        m = ~np.isnan(f)
        if np.sum(m) < 6:
            continue

        t = t[m].astype(np.float64)
        f = f[m].astype(np.float64)

        dur = float(t[-1] - t[0]) if len(t) > 1 else 0.0
        if not (PORTAMENTO_MIN_DUR_S <= dur <= PORTAMENTO_MAX_DUR_S):
            continue

        # Slide magnitude relative to seg_a target
        a_target_hz = midi_to_hz(int(seg_a["target_note"]))
        cents = 1200.0 * np.log2(f / a_target_hz)
        slide_pp = float(np.abs(np.percentile(cents, 95) - np.percentile(cents, 5)))
        if slide_pp < PORTAMENTO_MIN_SLIDE_ABS_CENTS:
            continue

        # "linear-ish" check
        t0 = t - t[0]
        r2 = _linear_fit_r2(t0, cents)

        seg_a["portamento"] = {
            "present": bool(r2 >= PORTAMENTO_MIN_R2),
            "duration_s": dur,
            "slide_pp_cents": slide_pp,
            "smoothness_r2": float(r2),
        }

    return segments


def run_step_1_3(
    segments: List[Dict[str, Any]],
    time: np.ndarray,
    frequency: np.ndarray,
    *,
    allowed_pitch_classes: Set[int],
    diatonic_pitch_classes: Set[int],
    blue_pitch_classes: Set[int],
) -> Tuple[List[Dict[str, Any]], None]:
    segments = classify_segments(
        segments,
        allowed_pitch_classes=allowed_pitch_classes,
        diatonic_pitch_classes=diatonic_pitch_classes,
        blue_pitch_classes=blue_pitch_classes,
    )
    segments = detect_vibrato(segments, time, frequency)
    segments = detect_portamento(segments, time, frequency)
    return segments, None
