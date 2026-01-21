import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
from music21 import scale
from scipy.ndimage import binary_closing, binary_dilation

from processing import pyin_loud_regions
from processing import min_run
from processing import hz_to_midi_safe
from note_classification import build_allowed_pitch_classes
from note_classification import segment_notes
from note_classification import classify_notes
from note_classification import discover_context_colors
from processing import median_filter_1d
from processing import cents_to_percent_cosine
from singing_analysis import detect_falsetto_note
from singing_analysis import detect_vibrato_params
from note_classification import NoteSeg
from report import get_report
from note_classification import generate_json

# inputs - vocal file and key
AUDIO_FILE = "sample_songs/dont.mp3"
TARGET_TONIC = "B"
TARGET_MODE = "minor"


# time between the start of one frame and the next
HOP_MS = 12


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


# keeps x in (-0.5, 0.5)
def wrap_half(x):
    return ((x + 0.5) % 1.0) - 0.5

# analysis
def analyze_key_compliance_over_time(audio_path, tonic, mode, window_sec=2.0):
    # y is an array of amplitudes
    y, sr = librosa.load(path, sr=None, mono=True)
    # hop_length = number of samples between the start of a frame and the start of the next
    hop_length = max(1, round(sr * (HOP_MS / 1000.0)))
    # the duration between the start of one frame and the start of the next
    hop_sec = hop_length / sr

    # retrieve our array of fundamental frequencies, only for values that are
    # in the top 90% of rms energy values, and the same goes for vflag and vprob
    f0, vflag, vprob, rms, rms_mask = pyin_loud_regions(
        y, hop_length, librosa.note_to_hz('C2'), 
        librosa.note_to_hz('C7'), 0.15, 2048, sr)
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
    rms_valid = rms[rms_mask]

    # global tuning
    tuning_offset = 0.0
    if np.any(mask_tune):
        midi_vals = hz_to_midi_safe(f0[mask_tune])
        deviations = wrap_half(midi_vals - np.round(midi_vals))
        median = float(np.median(deviations))
        if abs(median) <= 0.25:
            tuning_offset = median

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
        start = int(np.floor(n.t0 / hop))
        end = int(np.ceil(n.t1 / hop)) + 1
        start = max(0, start)
        end = min(N, end)
        if n.klass in ok_classes:
            frame_ok[start:end] = 1.0
        else:
            frame_ok[start:end] = 0.0

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
        tight[mm] = cents_to_percent_cosine(ce, zero_at=ZERO_AT_CENTS, power=2.5)
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
                                      rms_valid, rms_quantile_thresh=FALSETTO_LOW_RMS_Q,
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

    json = generate_json(notes, TARGET_TONIC, TARGET_MODE)
    report = get_report(json)

    return times, compliance, tightness, combined, notes, midi_full, nearest_full, report

# main
if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), AUDIO_FILE)
    times, compliance, tightness, combined, notes, midi_full, nearest_full, report = analyze_key_compliance_over_time(
        path, TARGET_TONIC, TARGET_MODE, WINDOW_SEC
    )

    report_filename = "vocal_performance_report.txt"
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report)
    print("Report successfully generated.")

    # performance score graph
    plt.figure(figsize=(12, 5.2))
    plt.axhline(100.0, color="k", linestyle="--", linewidth=0.8, alpha=0.35, zorder=1)
    maskedX = np.ma.masked_invalid(combined)

    plt.plot(times, maskedX, color='#1f77b4', linewidth=2.5, zorder=4, label="Overall Performance")
    plt.fill_between(times, maskedX, color='#1f77b4', alpha=0.1)

    if HIGHLIGHT_PERFECT:
        perf_mask = np.isfinite(combined) & (combined >= 99.5)
        if np.any(perf_mask):
            plt.scatter(times[perf_mask], combined[perf_mask],
                        s=12, color="#0050ff", edgecolors="none", alpha=0.9, zorder=5, label="≈100%")

    plt.ylim(0, 105)
    plt.xlabel("Time (s)")
    plt.ylabel("Score (%)")
    plt.title(f"Performance Score: {AUDIO_FILE}")
    plt.grid(alpha=0.30, linestyle=':')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()