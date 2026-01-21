import numpy as np
import matplotlib.pyplot as plt
from music21 import scale

# there needs to be at least 20 consecutive frames for vibrato to be considered
VIB_MIN_FRAMES = 20

# the vibrato width cannot be more than 65 cents
VIB_WIDTH_OK_CE = 65.0


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
def detect_falsetto_note(note_midis, note_rms, global_median_midi, global_rms_dist, rms_quantile_thresh=0.35, semitone_above=5.0):
    if note_midis.size == 0 or note_rms.size == 0:
        return False
    
    median_pitch = np.median(note_midis[np.isfinite(note_midis)])
    hi = (median_pitch - global_median_midi) >= semitone_above
    rms_cutoff = np.quantile(global_rms_dist, rms_quantile_thresh)
    low_power = np.median(note_rms) <= rms_cutoff

    return bool(hi and low_power)