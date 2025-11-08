import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from music21 import scale
from dataclasses import dataclass
from scipy.ndimage import binary_closing, binary_dilation

# inputs - vocal file and key
AUDIO_FILE = "glimpse_of_us.mp3"
TARGET_TONIC = "Ab"
TARGET_MODE = "major"


# preemphasis (slightly reduce lower frequencies and slightly boost higher frequencies)
USE_PREEMPH = True
# HPSS - harmonic percussive source separation (separates harmonic and percussive components)
USE_HPSS = True


# time between the start of one frame and the next
HOP_MS = 12
# range of notes for pYIN to detect
FMIN = "C2"
FMAX = "C7"


# main confidence level for a voiced frame to be kept
CONF_KEEP = 0.35
# confidence level for falsettos and soft singing
CONF_WEAK = 0.20
# fill in small gaps less than about 50 ms between vocals by 4 frames on both sides
DILATE_FRAMES = 4
# confidence level for finding global pitch offset
TUNE_THRESH = 0.85


# ignore the bottom 10% of RMS energy values
RMS_BOTTOM_QUANT = 0.10
# dont analyze a region unless it has 2 consecutive voiced frames
MIN_RUN_KEEP = 2
# to estimate global pitch offset, we need at least 3 consecutive, confident, voiced frames
MIN_RUN_TUNE = 3


# a passing note can't be more than 0.12 seconds
PASSING_MAX_SEC = 0.12
ALLOW_BLUE_NOTES = True
CONTEXT_ENABLE = True
# a contextual chromatic note can't be more than 0.25 seconds
CTX_MAX_SEC = 0.25
# a contextual chromatic note must appear at least twice to be considered
CTX_MIN_COUNT = 2
# allow a few modal mixture notes
INCLUDE_STATIC_MIX = True  # b6 in major; raised 6/7 in minor


# compliance curve
WINDOW_SEC = 2.0
MIN_VOICED_FRAC = 0.35
VOICED_GAP_SEC = 0.08


ZERO_AT_CENTS = 60.0
TIGHT_BONUS_CE = 20.0
STABILITY_SD_CE = 50.0

# leniency around note edges / transitions
ONOFF_GUARD_SEC = 0.060    # ignore first/last 60 ms of a note for stability
CHANGE_GUARD_FRAMES = 2        # ignore a couple frames around semitone changes

# vibrato/falsetto
VIB_MIN_HZ = 4.0
VIB_MAX_HZ = 8.0
VIB_MIN_FRAMES = 8           # need at least this many frames inside a note
VIB_WIDTH_OK_CE = 65.0        # treat width ≤ this as musical, not sloppy
VIB_TIGHT_BOOST = 1.15        # multiply tightness when vibrato is musical (cap 100)
VIB_RELAX_ADD = 15.0        # add this (cents) to stability cutoff for vibrato notes

FALSETTO_SEMITONES_ABOVE_MED = 5.0   # above singer's median register -> likely falsetto
FALSETTO_LOW_RMS_Q = 0.35  # and with low RMS relative to singer's distribution
FAL_TIGHT_BOOST = 1.10  # gentle boost for soft high notes
FAL_RELAX_ADD = 10.0  # relax stability cutoff slightly

# plot
HIGHLIGHT_PERFECT  = True
PLOT_OUTLIER_MARKS = False

# helpers
def load_audio_any(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    if USE_PREEMPH:
        y = librosa.effects.preemphasis(y)
    if USE_HPSS:
        y_harm, _ = librosa.effects.hpss(y)
        y = y_harm
    return y, sr

def min_run(m, k=3):
    out = np.zeros_like(m, dtype=bool)
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if (j - i) >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out

def wrap_half(x):  # (-0.5, 0.5]
    return ((x + 0.5) % 1.0) - 0.5

def hz_to_midi_safe(f):
    f = np.asarray(f)
    out = np.full_like(f, np.nan, dtype=float)
    good = np.isfinite(f) & (f > 0)
    out[good] = librosa.hz_to_midi(f[good])
    return out

def cents_to_percent_cosine(c_abs, zero_at=60.0):
    r = np.clip(c_abs / zero_at, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * r)) * 100.0

def median_filter_1d(x, k=3):
    if k <= 1 or x.size == 0:
        return x.copy()
    k = int(k) + (1 - int(k) % 2)  # make odd
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i+k])
    return out

def build_allowed_pitch_classes(tonic_name, mode_name, allow_blue=True, include_static_mix=True):
    if mode_name.lower() == "major":
        sc = scale.MajorScale(tonic_name); allow_blues = True
        root_pc = sc.getTonic().pitchClass
        base = {sc.pitchFromDegree(d).pitchClass for d in range(1, 8)}
        extra = set()
        if allow_blue and allow_blues:
            extra |= {(root_pc + 3) % 12, (root_pc + 6) % 12, (root_pc + 10) % 12}  # b3,b5,b7
        if include_static_mix:
            extra |= {(root_pc + 8) % 12}  # b6
        return base, extra, (base | extra)
    elif mode_name.lower() == "minor":
        sc = scale.MinorScale(tonic_name)
        root_pc = sc.getTonic().pitchClass
        base = {sc.pitchFromDegree(d).pitchClass for d in range(1, 8)}
        extra = set()
        if include_static_mix:
            extra |= {(root_pc + 9) % 12, (root_pc + 11) % 12}  # raised 6,7
        return base, extra, (base | extra)
    else:
        raise ValueError("TARGET_MODE must be 'major' or 'minor'.")

@dataclass
class NoteSeg:
    t0: float
    t1: float
    midi_target: float
    cents_med: float
    duration: float
    pc: int
    klass: str = "unlabeled"  # 'scale'|'blue'|'context'|'passing'|'out'

def segment_notes(times, midi_corr, nearest_midi, keep_mask):
    notes = []
    n = len(times); i = 0
    hop = times[1] - times[0] if len(times) > 1 else 0.01
    while i < n:
        if keep_mask[i]:
            j = i + 1
            target = nearest_midi[i]
            idx = [i]
            while j < n and keep_mask[j] and nearest_midi[j] == target:
                idx.append(j); j += 1
            cents = 100.0 * (midi_corr[idx] - target)
            t0, t1 = times[idx[0]], times[idx[-1]]
            dur = (t1 - t0) + hop
            pc = int(round(target)) % 12
            notes.append(NoteSeg(t0, t1, target, float(np.median(cents)), dur, pc))
            i = j
        else:
            i += 1
    return notes

def _min_cyclic_dist(a, b):
    d = abs(int(a) - int(b)) % 12
    return min(d, 12 - d)

def _stepwise(a, b):
    if a is None or b is None: return False
    return _min_cyclic_dist(a, b) in (1, 2)

def classify_notes(notes, base, blue_or_mix, context_allow=None, passing_max_sec=0.12):
    context_allow = context_allow or set()
    pcs = [n.pc for n in notes]
    for k, n in enumerate(notes):
        if n.pc in base:
            n.klass = "scale"; continue
        if n.pc in context_allow:
            n.klass = "context"; continue
        if n.pc in blue_or_mix and n.pc not in base:
            n.klass = "blue"; continue
        prev_pc = pcs[k-1] if k > 0 else None
        next_pc = pcs[k+1] if k+1 < len(pcs) else None
        if n.duration <= passing_max_sec and _stepwise(prev_pc, next_pc):
            n.klass = "passing"
        else:
            n.klass = "out"
    return notes

def discover_context_colors(notes, base, root_pc, max_sec=0.25, min_count=2):
    counts = {}
    pcs = [n.pc for n in notes]
    for k, n in enumerate(notes):
        if n.pc in base or n.duration > max_sec:
            continue
        prev_pc = pcs[k-1] if k > 0 else None
        next_pc = pcs[k+1] if k+1 < len(pcs) else None
        ok_prev = prev_pc is not None and _stepwise(n.pc, prev_pc) and (prev_pc in base)
        ok_next = next_pc is not None and _stepwise(n.pc, next_pc) and (next_pc in base)
        if ok_prev or ok_next:
            counts[n.pc] = counts.get(n.pc, 0) + 1
        if _stepwise(n.pc, root_pc) or _stepwise(n.pc, (root_pc + 7) % 12):
            if ok_prev or ok_next:
                counts[n.pc] = counts.get(n.pc, 0) + 1
    return {pc for pc, c in counts.items() if c >= min_count}

# adaptive detectors
def detect_vibrato_params(cents_series, hop_sec, f_lo=4.0, f_hi=8.0):
    x = cents_series[np.isfinite(cents_series)]
    if x.size < VIB_MIN_FRAMES:
        return False, 0.0, 0.0
    # detrend (remove DC / slow drift)
    x = x - np.median(x)
    # FFT
    N = len(x)
    # next pow2 fft for speed/robustness
    nfft = 1 << (N - 1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=hop_sec)
    # band 4–8 Hz
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return False, 0.0, 0.0
    power = np.abs(X)**2
    idx = np.argmax(power[band])
    peak_idx = np.where(band)[0][idx]
    vib_hz = float(freqs[peak_idx])
    # approx width as 2*RMS (≈peak-to-peak/√2). Scaled to cents.
    width = float(np.std(x) * 2.0)
    is_musical = (f_lo <= vib_hz <= f_hi) and (width <= VIB_WIDTH_OK_CE)
    return is_musical, vib_hz, width

def detect_falsetto_note(note_midis, note_rms, global_median_midi, rms_quantile_thresh=0.35, semitone_above=5.0):
    if note_midis.size == 0 or note_rms.size == 0:
        return False
    med = np.median(note_midis[np.isfinite(note_midis)])
    hi = (med - global_median_midi) >= semitone_above
    low_power = (np.median(note_rms) <= np.quantile(note_rms, rms_quantile_thresh))
    return bool(hi and low_power)

# analysis
def analyze_key_compliance_over_time(audio_path, tonic, mode, window_sec=2.0):
    y, sr = load_audio_any(audio_path)
    hop_length = max(1, round(sr * (HOP_MS / 1000.0)))
    hop_sec = hop_length / sr

    f0, vflag, vprob = librosa.pyin(
        y, fmin=librosa.note_to_hz(FMIN), fmax=librosa.note_to_hz(FMAX), hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # energy gate
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True).flatten()
    thr = np.quantile(rms, RMS_BOTTOM_QUANT)
    rms_mask = rms > thr

    # voiced masks
    finite = np.isfinite(f0)
    strong = (vflag == True) & (vprob >= CONF_KEEP) & finite & rms_mask
    weak   = (vflag == True) & (vprob >= CONF_WEAK) & finite & rms_mask
    any_f0 = finite & rms_mask
    base_mask = strong | weak | any_f0
    if np.any(base_mask):
        base_mask = binary_dilation(base_mask, iterations=DILATE_FRAMES)
    mask_acc  = min_run(base_mask, k=MIN_RUN_KEEP)
    mask_tune = min_run((vflag == True) & (vprob >= TUNE_THRESH) & finite & rms_mask, k=MIN_RUN_TUNE)

    # global tuning
    tuning_offset = 0.0
    if np.any(mask_tune):
        midi_vals = hz_to_midi_safe(f0[mask_tune])
        dev = wrap_half(midi_vals - np.round(midi_vals))
        med = float(np.median(dev))
        if abs(med) <= 0.25 and not np.isclose(abs(med), 0.5, atol=1e-2):
            tuning_offset = med

    # nearest chromatic
    midi_full = hz_to_midi_safe(f0) - tuning_offset
    chrom_midis = np.arange(36, 97, dtype=float)  # C2..C7
    chrom_hz    = librosa.midi_to_hz(chrom_midis)

    nearest_full = np.full_like(midi_full, np.nan, dtype=float)
    if np.any(mask_acc):
        mc = midi_full[mask_acc]
        f0_corr_hz = librosa.midi_to_hz(mc)
        cents_mat  = 1200.0 * np.log2(f0_corr_hz[:, None] / chrom_hz[None, :])
        idx = np.argmin(np.abs(cents_mat), axis=1)
        nearest_full[mask_acc] = chrom_midis[idx]

    # segment + context
    base, static_extra, _ = build_allowed_pitch_classes(
        tonic, mode, allow_blue=ALLOW_BLUE_NOTES, include_static_mix=INCLUDE_STATIC_MIX
    )
    notes = segment_notes(times, midi_full, nearest_full, mask_acc)

    context_allow = set()
    if CONTEXT_ENABLE and notes:
        sc = scale.MajorScale(tonic) if mode.lower() == "major" else scale.MinorScale(tonic)
        root_pc = sc.getTonic().pitchClass
        tmp_notes = [NoteSeg(n.t0, n.t1, n.midi_target, n.cents_med, n.duration, n.pc, n.klass) for n in notes]
        tmp_notes = classify_notes(tmp_notes, base, static_extra, context_allow=set(), passing_max_sec=PASSING_MAX_SEC)
        context_allow = discover_context_colors(tmp_notes, base, root_pc, max_sec=CTX_MAX_SEC, min_count=CTX_MIN_COUNT)

    notes = classify_notes(notes, base, static_extra, context_allow=context_allow, passing_max_sec=PASSING_MAX_SEC)

    # framewise compliance
    N = len(times)
    frame_ok   = np.full(N, np.nan, dtype=float)
    ok_classes = {"scale", "blue", "context", "passing"}
    hop = hop_sec

    for n in notes:
        a = int(np.floor(n.t0 / hop)); b = int(np.ceil(n.t1 / hop)) + 1
        a = max(0, a); b = min(N, b)
        frame_ok[a:b] = 1.0 if n.klass in ok_classes else 0.0

    # per-frame cents error to nearest chromatic
    cents_err = np.full(N, np.nan, dtype=float)
    m = np.isfinite(nearest_full) & np.isfinite(midi_full)
    cents_err[m] = 100.0 * (midi_full[m] - nearest_full[m])

    # smooth jitter
    if np.any(np.isfinite(cents_err)):
        ce_tmp = cents_err.copy()
        ce_tmp[np.isnan(ce_tmp)] = 0.0
        cents_err = median_filter_1d(ce_tmp, k=3)
        cents_err[np.isnan(cents_err)] = np.nan

    # baseline tightness
    tight = np.full(N, np.nan, dtype=float)
    mm = np.isfinite(cents_err)
    if np.any(mm):
        ce = np.abs(cents_err[mm])
        tight[mm] = cents_to_percent_cosine(ce, zero_at=ZERO_AT_CENTS)
        tight[mm] += 5.0 * (ce <= TIGHT_BONUS_CE)
        tight[mm] = np.clip(tight[mm], 0.0, 100.0)

    # stability + adaptive boosts
    stability_weight = np.ones(N, dtype=float)

    # guards
    onoff_guard_frames = int(round(ONOFF_GUARD_SEC / hop))
    change_guard = int(CHANGE_GUARD_FRAMES)

    # semitone change mask
    change_mask = np.zeros(N, dtype=bool)
    nf = np.where(np.isfinite(nearest_full), nearest_full, -9999)
    change_idx = np.where(np.abs(np.diff(nf)) >= 0.5)[0]
    for ix in change_idx:
        a = max(0, ix - change_guard)
        b = min(N, ix + 1 + change_guard)
        change_mask[a:b] = True

    # singer's median register for falsetto heuristic
    singer_median_midi = float(np.nanmedian(midi_full))

    # per-note adaptation
    for n in notes:
        a = int(np.floor(n.t0 / hop)); b = int(np.ceil(n.t1 / hop)) + 1
        a = max(0, a); b = min(N, b)
        if b <= a: 
            continue

        # core region for SD
        aa = min(b, a + onoff_guard_frames)
        bb = max(a, b - onoff_guard_frames)
        core = np.arange(aa, bb)
        core = core[(core >= 0) & (core < N)]
        if core.size == 0:
            continue

        ce_note = cents_err[a:b]
        ce_core = cents_err[core]
        ce_core = ce_core[np.isfinite(ce_core)]

        # note-local arrays
        midi_note = midi_full[a:b]
        rms_note  = rms[a:b]

        # detect musical vibrato on the core
        is_vib, vib_hz, vib_width = detect_vibrato_params(ce_core, hop_sec, VIB_MIN_HZ, VIB_MAX_HZ)

        # falsetto heuristic
        is_fal = detect_falsetto_note(midi_note, rms_note, singer_median_midi,
                                      rms_quantile_thresh=FALSETTO_LOW_RMS_Q,
                                      semitone_above=FALSETTO_SEMITONES_ABOVE_MED)

        # compute SD for stability penalty (on core only)
        if ce_core.size:
            sd = float(np.std(ce_core))
            cutoff = STABILITY_SD_CE
            if is_vib:
                cutoff = max(cutoff, min(VIB_WIDTH_OK_CE, vib_width + VIB_RELAX_ADD))
            if is_fal:
                cutoff = cutoff + FAL_RELAX_ADD

            if sd > cutoff:
                w = max(0.0, 1.0 - ((sd - cutoff) / cutoff) ** 2)  # soft knee
                stability_weight[a:b] *= w

        # relax across immediate semitone changes
        stability_weight[a:b] = np.where(change_mask[a:b], 1.0, stability_weight[a:b])

        # tightness boosts (no penalty for musical vibrato / soft falsetto)
        if np.any(np.isfinite(tight[a:b])):
            boost = 1.0
            if is_vib:
                boost *= VIB_TIGHT_BOOST
            if is_fal:
                boost *= FAL_TIGHT_BOOST
            tight[a:b] = np.clip(tight[a:b] * boost, 0.0, 100.0)

    tight_weighted = np.where(np.isfinite(tight), tight * stability_weight, np.nan)

    # rolling averages with hard gaps
    win = int(max(1, round(window_sec / hop)))
    valid = np.isfinite(frame_ok).astype(float)
    valsC = np.nan_to_num(frame_ok, nan=0.0)
    valsT = np.nan_to_num(tight_weighted / 100.0, nan=0.0)

    if VOICED_GAP_SEC > 0:
        k = int(round(VOICED_GAP_SEC / hop))
        if k > 0:
            valid = binary_closing(valid.astype(bool), structure=np.ones(k, dtype=bool)).astype(float)

    kernel = np.ones(win, dtype=float)
    numC = np.convolve(valsC * valid, kernel, mode="same")
    numT = np.convolve(valsT * valid, kernel, mode="same")
    den  = np.convolve(valid,       kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        compliance = 100.0 * (numC / den)
        tightness  = 100.0 * (numT / den)

    min_voiced = MIN_VOICED_FRAC * win
    compliance[den < min_voiced] = np.nan
    tightness[den  < min_voiced] = np.nan

    combined = (compliance * tightness) / 100.0
    return times, compliance, tightness, combined, notes

# main
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), AUDIO_FILE)
    times, compliance, tightness, combined, notes = analyze_key_compliance_over_time(
        path, TARGET_TONIC, TARGET_MODE, WINDOW_SEC
    )

    plt.figure(figsize=(12, 5.2))
    plt.axhline(100.0, color="k", linestyle="--", linewidth=0.8, alpha=0.35, zorder=1)
    maskedC = np.ma.masked_invalid(compliance)
    maskedT = np.ma.masked_invalid(tightness)
    maskedX = np.ma.masked_invalid(combined)

    plt.plot(times, maskedX, linewidth=2.2, zorder=4, label="Performance score")
    plt.plot(times, maskedC, linewidth=1.4, alpha=0.65, zorder=3, label="Key compliance")
    plt.plot(times, maskedT, linewidth=1.2, alpha=0.55, zorder=2, label="Intonation tightness")

    if HIGHLIGHT_PERFECT:
        perf_mask = np.isfinite(combined) & (combined >= 99.5)
        if np.any(perf_mask):
            plt.scatter(times[perf_mask], combined[perf_mask],
                        s=12, color="#0050ff", edgecolors="none", alpha=0.9, zorder=5, label="≈100%")

    plt.ylim(0, 105)
    plt.xlabel("Time (s)")
    plt.ylabel("Score (%)")
    plt.title(f"Performance of {AUDIO_FILE} — {TARGET_TONIC.upper()} {TARGET_MODE.capitalize()}")
    plt.grid(alpha=0.30)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
