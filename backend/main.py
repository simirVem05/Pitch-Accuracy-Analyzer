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
USE_PREEMPH = False
# HPSS - harmonic percussive source separation (separates harmonic and percussive components)
USE_HPSS = False


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
# dont analyze a region unless it has at least 2 consecutive voiced frames
MIN_RUN_KEEP = 2
# to estimate global pitch offset, we need at least 3 consecutive, confident, voiced frames
MIN_RUN_TUNE = 3


# a passing note can't be more than 0.12 seconds
PASSING_MAX_SEC = 0.12
ALLOW_BLUE_NOTES = True
# enable detection of contextual chromatic notes
CONTEXT_ENABLE = True
# a contextual chromatic note can't be more than 0.25 seconds
CTX_MAX_SEC = 0.25
# a contextual chromatic note must appear at least twice to be considered
CTX_MIN_COUNT = 2
# allow a few modal mixture notes
INCLUDE_STATIC_MIX = True  # b6 in major; raised 6/7 in minor


# duration of the rolling window
WINDOW_SEC = 1.0
# minimum percentage of vocal activity in a window for it to be plotted
MIN_VOICED_FRAC = 0.45
# if a gap between vocals is less than this duration it gets filled in on the graph
VOICED_GAP_SEC = 0.06


# if a singer is 60 cents off their accuracy is 0%
ZERO_AT_CENTS = 60.0
# if a singer is 20 cents off, they get a pitch accuracy bonus
TIGHT_BONUS_CE = 20.0
# a singer will be penalized if the standard deviation of the pitch variance of a note segment is more than 50 cents
STABILITY_SD_CE = 50.0


# small breaks in singing that are less than 0.060 seconds wont cause a note to split up into multiple ones
ONOFF_GUARD_SEC = 0.060
# 2 consecutive frames are required to consider that the note has changed
CHANGE_GUARD_FRAMES = 2


# vibrato must have a frequency of at least 4 cycles per second
VIB_MIN_HZ = 4.0
# vibrato can't have a frequency of over 7 cycles per second
VIB_MAX_HZ = 7.0
# there needs to be at least 20 consecutive frames for vibrato to be considered
VIB_MIN_FRAMES = 20
# the vibrato width cannot be more than 65 cents
VIB_WIDTH_OK_CE = 65.0
# intonation tightness is boosted by 15% if an artist does good vibrato
VIB_TIGHT_BOOST = 1.15
# intonaton tightness is boosted up a fixed number of 15 for expressive singing that somewhat resembles vibrato
VIB_RELAX_ADD = 15.0

# minimum number of semitones a note has to be above the median pitch to be considered falsetto
FALSETTO_SEMITONES_ABOVE_MED = 5.0
# if a note is less than or in the 35th percentile of the RMS distribution we can consider it falsetto
FALSETTO_LOW_RMS_Q = 0.35
# good falsetto is boosted by 10 percent
FAL_TIGHT_BOOST = 1.10
# a boost of 10 awarded for partial falsetto
FAL_RELAX_ADD = 10.0

# when performance is perfect or near-perfect, blue scatter dots are added
HIGHLIGHT_PERFECT  = True
# notes classified as 'out' will have a vertical line when they are sung on the graph
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

# leaves out any run of confident notes that are less than 3 frames
def min_run(vocal_flag, k=3):
    out = np.zeros_like(vocal_flag, dtype=bool)
    i, n = 0, len(vocal_flag)
    while i < n:
        if vocal_flag[i]:
            j = i
            while j < n and vocal_flag[j]:
                j += 1
            if (j - i) >= k:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out

# allows us to only run pYIN on loud regions
def pyin_loud_regions(y, hop_length, fmin_hz, fmax_hz, rms_bottom_quant, frame_length):
    rms = librosa.feature.rms(y=y, frame_length=frame_length, 
                              hop_length=hop_length, center=True
                              ).flatten()
    
    thr = np.quantile(rms, rms_bottom_quant)
    rms_mask = rms > thr

    rms_mask = min_run(rms_mask, k=3)
    n = len(rms)
    f0_full = np.full(n, np.nan, dtype=float)
    vflag_full = np.zeros(n, dtype=bool)
    vprob_full = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        if rms_mask[i]:
            j = i + 1
            while j < n and rms_mask[j]:
                j += 1
            
            start_sample = i * hop_length
            end_sample = min(len(y), j * hop_length)
            if end_sample <= start_sample:
                i = j
                continue

            y_seg = y[start_sample:end_sample]
            
            f0_seg, vflag_seg, vprob_seg = librosa.pyin(
                y_seg,
                fmin=fmin_hz,
                fmax=fmax_hz,
                hop_length=hop_length
            )

            length = min(len(f0_seg), j - i)
            f0_full[i:i+length] = f0_seg[:length]
            vflag_full[i:i+length] = vflag_seg[:length]
            vprob_full[i:i+length] = vprob_seg[:length]

            i = j
        else:
            i += 1
    
    return f0_full, vflag_full, vprob_full, rms, rms_mask

# keeps x in (-0.5, 0.5)
def wrap_half(x):
    return ((x + 0.5) % 1.0) - 0.5

# outputs a boolean array filled with True when a frequency is finite and positive
def hz_to_midi_safe(frequencies_hz):
    frequencies_hz = np.asarray(frequencies_hz)
    out = np.full_like(frequencies_hz, np.nan, dtype=float)
    good = np.isfinite(frequencies_hz) & (frequencies_hz > 0)
    out[good] = librosa.hz_to_midi(frequencies_hz[good])
    return out

# converts the deviation from a note in cents to a percentage using a cosine function
def cents_to_percent_cosine(c_abs, zero_at=60.0):
    r = np.clip(c_abs / zero_at, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * r)) * 100.0 # y = (1 + x) / 2 = (1 + cos(pi * r)) / 2 
    # returns y * 100

# smooths out sudden dips and spikes in a 1d array
def median_filter_1d(x, k=3):
    if k <= 1 or x.size == 0:
        return x.copy()
    k = int(k) + (1 - int(k) % 2) # make odd
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i+k])
    return out

# constructs the allowed pitch classes(diatonic and chromatic) for a given scale
def build_allowed_pitch_classes(tonic_name, mode_name, allow_blue=True, include_static_mix=True):
    if mode_name.lower() == "major":
        sc = scale.MajorScale(tonic_name); allow_blues = True
        root_pc = sc.getTonic().pitchClass
        base = {sc.pitchFromDegree(d).pitchClass for d in range(1, 8)} # retrieves the pitch class of every diatonic note in the scale using scale degrees
        extra = set()
        if allow_blue and allow_blues:
            extra |= {(root_pc + 3) % 12, (root_pc + 6) % 12, (root_pc + 10) % 12}  # adds b3, b5, b7
        if include_static_mix:
            extra |= {(root_pc + 8) % 12}  # adds b6
        return base, extra, (base | extra)
    elif mode_name.lower() == "minor":
        sc = scale.MinorScale(tonic_name)
        root_pc = sc.getTonic().pitchClass
        base = {sc.pitchFromDegree(d).pitchClass for d in range(1, 8)}
        extra = set()
        if include_static_mix:
            extra |= {(root_pc + 9) % 12, (root_pc + 11) % 12}  # raised 6, 7 from major scale
        return base, extra, (base | extra)
    else:
        raise ValueError("TARGET_MODE must be 'major' or 'minor'.")

# each NoteSeg object represents a note that is sung by the artist
@dataclass
class NoteSeg:
    t0: float # start of the note in seconds
    t1: float # end of the note in seconds
    midi_target: float # nearest chromatic MIDI note after global tuning offset
    cents_med: float # the median of the deviation from the target note in cents
    duration: float # duration of the note
    pc: int # pitch class of the target note
    klass: str = "unlabeled"  # 'scale'|'blue'|'context'|'passing'|'out'

# return a list of NoteSeg objects that each represent a note the artist sang
def segment_notes(times, midi_corr, nearest_midi, keep_mask):
    notes = [] # for NoteSeg objects
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

# returns the smallest semitone distance between 2 pitch classes
def _min_cyclic_dist(a, b):
    if not(0 <= a <= 11 and 0 <= b <= 11):
        raise ValueError(f"Pitch classes must be 0-11, got a={a} and b={b}")
    d = abs(int(a) - int(b))
    return min(d, 12 - d)

# returns a boolean telling us if two pitch classes are within 1 or 2 semitone or not
def _stepwise(a, b):
    if a is None or b is None: return False
    return _min_cyclic_dist(a, b) in (1, 2)

# classifies each note in notes, the list of NoteSeg objects
def classify_notes(notes, base, blue_or_mix, context_allow=None, passing_max_sec=0.12):
    context_allow = context_allow or set()
    pcs = [n.pc for n in notes] # target pitch class for every NoteSeg object
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

# returns pitch classes of context color notes, which must be non-diatonic, <= 0.25 sec, stepwise with a diatonic note, and appear at least twice
def discover_context_colors(notes, base, root_pc, max_sec=0.25, min_count=2):
    counts = {}
    pcs = [n.pc for n in notes]
    for k, n in enumerate(notes):
        if n.pc in base or n.duration > max_sec: # max_sec is the max duration for a note to be considered a context color
            continue # skip diatonic notes
        prev_pc = pcs[k-1] if k > 0 else None
        next_pc = pcs[k+1] if k+1 < len(pcs) else None
        ok_prev = prev_pc is not None and _stepwise(n.pc, prev_pc) and (prev_pc in base)
        ok_next = next_pc is not None and _stepwise(n.pc, next_pc) and (next_pc in base)
        if ok_prev or ok_next:
            counts[n.pc] = counts.get(n.pc, 0) + 1
        if _stepwise(n.pc, root_pc) or _stepwise(n.pc, (root_pc + 7) % 12): # stepwise with the tonic or dominant 5th
            if ok_prev or ok_next:
                counts[n.pc] = counts.get(n.pc, 0) + 1
    return {pc for pc, c in counts.items() if c >= min_count}

# returns whether a note is sung in vibrato, and if it is, returns the vibrato rate in Hz, and its width in cents
def detect_vibrato_params(cents_series, hop_sec, f_lo=4.0, f_hi=8.0):
    x = cents_series[np.isfinite(cents_series)]
    if x.size < VIB_MIN_FRAMES:
        return False, 0.0, 0.0
    # detrend
    x = x - np.median(x)
    # FFT
    N = len(x)
    # next pow2 fft for speed/robustness
    nfft = 1 << (N - 1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=hop_sec)
    # band 4–7 Hz
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return False, 0.0, 0.0
    power = np.abs(X)**2
    idx = np.argmax(power[band])
    peak_idx = np.where(band)[0][idx]
    vib_hz = float(freqs[peak_idx])
    # approx width as 2*RMS, scaled to cents
    width = float(np.std(x) * 2.0)
    is_musical = (f_lo <= vib_hz <= f_hi) and (width <= VIB_WIDTH_OK_CE)
    return is_musical, vib_hz, width

# checks whether a note is in falsetto or not by seeing if it is 5 semitones higher than the median pitch and in the 35th rms percentile or smaller
def detect_falsetto_note(note_midis, note_rms, global_median_midi, rms_quantile_thresh=0.35, semitone_above=5.0):
    if note_midis.size == 0 or note_rms.size == 0:
        return False
    med = np.median(note_midis[np.isfinite(note_midis)])
    hi = (med - global_median_midi) >= semitone_above
    low_power = (np.median(note_rms) <= np.quantile(note_rms, rms_quantile_thresh))
    return bool(hi and low_power)

# analysis
def analyze_key_compliance_over_time(audio_path, tonic, mode, window_sec=2.0):
    # y is an array of amplitudes
    y, sr = load_audio_any(audio_path)
    # hop_length = number of samples between the start of a frame and the start of the next
    hop_length = max(1, round(sr * (HOP_MS / 1000.0)))
    # the duration of the start of one frame and the start of the next
    hop_sec = hop_length / sr

    # retrieve our array of fundamental frequencies, only for values that are
    # in the top 90% of rms energy values, and the same goes for vflag and vprob
    f0, vflag, vprob, rms, rms_mask = pyin_loud_regions(
        y, hop_length, librosa.note_to_hz(FMIN), 
        librosa.note_to_hz(FMAX), 0.15, 2048)
    times = np.arange(len(f0)) * hop_sec

    # voiced masks
    finite = np.isfinite(f0)
    strong = (vflag == True) & (vprob >= CONF_KEEP) & finite & rms_mask
    weak = (vflag == True) & (vprob >= CONF_WEAK) & finite & rms_mask
    any_f0 = finite & rms_mask
    base_mask = strong | weak | any_f0
    if np.any(base_mask):
        base_mask = binary_dilation(base_mask, iterations=DILATE_FRAMES)
    mask_acc = min_run(base_mask, k=MIN_RUN_KEEP)
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
    chrom_hz = librosa.midi_to_hz(chrom_midis)

    nearest_full = np.full_like(midi_full, np.nan, dtype=float)
    if np.any(mask_acc):
        mc = midi_full[mask_acc]
        f0_corr_hz = librosa.midi_to_hz(mc)
        cents_mat = 1200.0 * np.log2(f0_corr_hz[:, None] / chrom_hz[None, :])
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
    frame_ok = np.full(N, np.nan, dtype=float)
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
    den = np.convolve(valid, kernel, mode="same")

    with np.errstate(invalid="ignore", divide="ignore"):
        compliance = 100.0 * (numC / den)
        tightness = 100.0 * (numT / den)

    min_voiced = MIN_VOICED_FRAC * win
    compliance[den < min_voiced] = np.nan
    tightness[den < min_voiced] = np.nan

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
    plt.title(f"Performance of {AUDIO_FILE} — {TARGET_TONIC} {TARGET_MODE.capitalize()}")
    plt.grid(alpha=0.30)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()
