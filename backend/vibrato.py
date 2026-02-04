from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import scipy.signal


@dataclass(frozen=True)
class VibratoResult:
    is_vibrato: np.ndarray          # bool mask per frame
    center_cents: np.ndarray        # vibrato center per frame (NaN where not voiced)
    strength: np.ndarray            # 0..1-ish heuristic


def detect_vibrato(
    time_s: np.ndarray,
    pitch_cents: np.ndarray,
    window_s: float,
    hz_min: float,
    hz_max: float,
    depth_min: float,
    depth_max: float,
    strength_threshold: float,
) -> VibratoResult:
    """
    Sliding-window vibrato detection:
    - detrend pitch (remove slow drift) => residual
    - FFT peak in [hz_min, hz_max]
    - depth from residual std / peak amplitude
    """
    n = len(pitch_cents)
    is_vib = np.zeros(n, dtype=bool)
    center = np.full(n, np.nan, dtype=float)
    strength = np.zeros(n, dtype=float)

    # Need uniform hop
    if n < 5:
        return VibratoResult(is_vib, center, strength)

    dt = np.nanmedian(np.diff(time_s))
    if not np.isfinite(dt) or dt <= 0:
        return VibratoResult(is_vib, center, strength)

    fs = 1.0 / dt
    win = max(5, int(round(window_s * fs)))
    if win % 2 == 0:
        win += 1

    # Interpolate NaNs to run filters, but keep NaN mask for reliability
    idx = np.arange(n)
    good = np.isfinite(pitch_cents)
    if good.sum() < win:
        return VibratoResult(is_vib, center, strength)

    interp = np.interp(idx, idx[good], pitch_cents[good])

    # Center estimate: moving average (slow trend)
    trend = scipy.signal.savgol_filter(interp, window_length=win, polyorder=2, mode="interp")
    residual = interp - trend

    # Sliding FFT on residual
    step = max(1, win // 4)
    freqs = np.fft.rfftfreq(win, d=dt)

    band = (freqs >= hz_min) & (freqs <= hz_max)
    if band.sum() == 0:
        return VibratoResult(is_vib, center, strength)

    for start in range(0, n - win + 1, step):
        end = start + win
        seg = residual[start:end]
        seg_center = trend[start:end]

        # Reject windows with too many originally-unvoiced frames
        seg_good = good[start:end]
        if seg_good.mean() < 0.7:
            continue

        # Windowing reduces leakage
        w = np.hanning(win)
        spec = np.abs(np.fft.rfft(seg * w))

        band_spec = spec[band]
        peak = float(band_spec.max()) if band_spec.size else 0.0
        total = float(spec.sum()) + 1e-9
        rel = peak / total  # heuristic “vibrato-ness”

        # Depth estimate
        depth = float(np.nanstd(seg))
        depth_ok = (depth >= depth_min) and (depth <= depth_max)

        s = rel
        strength[start:end] = np.maximum(strength[start:end], s)

        if depth_ok and s >= strength_threshold:
            is_vib[start:end] = True
            center[start:end] = seg_center

    # Ensure center is NaN where not voiced
    center[~good] = np.nan
    return VibratoResult(is_vibrato=is_vib, center_cents=center, strength=strength)
