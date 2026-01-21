import numpy as np
import librosa
from scipy.signal import butter, sosfilt

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

def highpass_filter(y, sr, cutoff=85.0, order=5):
    sos = butter(order, cutoff, btype='high', fs=sr, output='sos')
    filtered_y = sosfilt(sos, y)
    return filtered_y

# allows us to only run pYIN on loud regions
def pyin_loud_regions(y, hop_length, fmin_hz, fmax_hz, rms_bottom_quant, frame_length, sr):
    filtered_y = highpass_filter(y, sr, 85.0, 5)
    
    rms = librosa.feature.rms(y=filtered_y, frame_length=frame_length, 
                              hop_length=hop_length, center=True
                              ).flatten()
    
    rms_no_zeros = rms[rms > 0]
    if rms_no_zeros.size > 0:
        thresh = np.quantile(rms_no_zeros, rms_bottom_quant)
    else:
        thresh = 0.0
    rms_mask = rms > thresh

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

# outputs an array of MIDI notes when the frequency that note is equivalent to is finite and positive
def hz_to_midi_safe(frequencies_hz):
    frequencies_hz = np.asarray(frequencies_hz)
    out = np.full_like(frequencies_hz, np.nan, dtype=float)
    good = np.isfinite(frequencies_hz) & (frequencies_hz > 0)
    out[good] = librosa.hz_to_midi(frequencies_hz[good])
    return out

# converts the deviation from a note in cents to a percentage using a cosine function
def cents_to_percent_cosine(c_abs, zero_at=65.0, power=2.5):
    r = np.clip(c_abs / zero_at, 0.0, 1.0)
    r_steep = r ** power
    score = 0.5 * (1.0 + np.cos(np.pi * r_steep)) * 100.0
    return score

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