from dataclasses import dataclass
import librosa
import os
import crepe
from scipy.signal import butter, sosfilt
import numpy as np
from music21 import scale

AUDIO_FILE = "sample_songs/dont.mp3"
TONIC = "B"
MODE = "minor"
GENRE = "rnb"

STEP_SIZE_MS = 20
HOP_LENGTH = int(16000 * (STEP_SIZE_MS / 1000))
FRAME_LENGTH = 1024
VITERBI = True

RMS_BOTTOM_QUANT = 0.10
MIN_RUN_FRAMES = 5
CONFIDENCE_LEVEL = 0.20
TUNE_CONFIDENCE = 0.5

def min_run(mask, k):
    n = len(mask)
    output = np.full(n, False)
    
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= k:
                output[i:j] = True
            i = j
        else:
            i += 1
    
    return output

def highpass_filter(audio, sr, cutoff=85):
    sos = butter(10, cutoff, btype="highpass", fs=sr, output='sos')
    filtered_audio = sosfilt(sos, audio)
    return filtered_audio

def audio_transform(input_path):
    filename, filetype = os.path.splitext(input_path)
    filetype = filetype.lower()

    audio, sr = librosa.load(input_path, sr=16000)
    
    transformed = highpass_filter(audio, sr)
    return transformed, sr

def crepe_predict(audio_path, rms_bottom_quant, hop_length, step_size_ms, frame_length, min_run_frames,
    confidence_level, viterbi):
    audio, sr = audio_transform(audio_path)

    time, frequency, confidence, activation = crepe.predict(
        audio, sr, viterbi=viterbi, step_size=step_size_ms, model_capacity='small'
    )

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length, center=True
    ).flatten()
    
    if len(rms) != len(confidence):
        n = min(len(rms), len(confidence))
        rms = rms[:n]
        time = time[:n]
        frequency = frequency[:n]
        confidence = confidence[:n]
        activation = activation[:n]

    rms_nonzero = rms[rms > 0]
    if rms_nonzero.size > 0:
        threshold = np.quantile(rms_nonzero, rms_bottom_quant)
    else:
        print("Audio is completely silent.")
        return time, frequency, confidence, activation

    valid_mask = (rms > threshold) & (confidence > confidence_level) & (np.isfinite(frequency))
    valid_mask = min_run(valid_mask, min_run_frames)
    
    frequency = np.where(valid_mask, frequency, np.nan)
    confidence = np.where(valid_mask, confidence, 0.0)
    activation = np.where(valid_mask[:, np.newaxis], activation, np.nan)
    
    return time, frequency, confidence, activation

def hz_to_midi(frequencies):
    frequencies = np.asarray(frequencies)
    out = np.full_like(frequencies, np.nan, dtype=float)
    valid = np.isfinite(frequencies) and frequencies > 0
    out[valid] = librosa.hz_to_midi(frequencies[valid])

def calculate_global_offset(frequency, tune_thresh):
    if len(frequency) == 0:
        print("There are no frequencies.")
        return 0.0

    midis = librosa.hz_to_midi(frequency)
    cents_deviations = (midis - np.round(midis)) * 100
    median = float(np.median(cents_deviations))
    
    return median

GENRE_CONFIGS = {
    "pop": {
        "major": [], 
        "minor": []
    },
    "alt-pop": {
        "major": [3, 8, 10], # b3, b6, b7
        "minor": [9, 11] # natural 6, natural 7
    },
    "mel-rap": {
        "major": [3, 10], # b3, b7
        "minor": [11] # natural 7
    },
    "rnb": {
        "major": [3, 6, 8, 10], # b3, b5, b6, b7
        "minor": [6, 9, 11] # b5, natural 6, natural 7
    },
    "rock": {
        "major": [3, 6, 10], # b3, b5, b7
        "minor": [6, 9] # b5, natural 6
    },
    "country": {
        "major": [3, 10], # b3, b7
        "minor": [9, 11] # natural 6, natural 7
    },
    "jazz": {
        "major": [3, 6, 10], # b3, b5, b7
        "minor": [6, 9, 11] # natural 6, natural 7
    },
    "electronic": {
        "major": [10],
        "minor": [1, 11]
    }
}

def build_pitch_classes(tonic, mode, genre):
    if mode.lower() == "major":
        sc = scale.MajorScale(tonic)
    else:
        sc = scale.MinorScale(tonic)
    
    pitch_classes = {sc.pitchFromDegree(d).pitchClass for d in range(1, 8)}

    tonic_pc = sc.getTonic().pitchClass
    offsets = GENRE_CONFIGS.get(genre, {}).get(mode.lower(), [])

    for offset in offsets:
        pitch_classes.add((tonic_pc + offset) % 12)
    
    return pitch_classes

@dataclass
class Note:
    t0: float
    t1: float
    midi_target: float
    cents_med: float
    pc_target: float
    p_class: str = "unlabeled"

if __name__ == '__main__':
    time, frequency, confidence, activation = crepe_predict(
        AUDIO_FILE, RMS_BOTTOM_QUANT, HOP_LENGTH, STEP_SIZE_MS, FRAME_LENGTH,
        MIN_RUN_FRAMES, CONFIDENCE_LEVEL, VITERBI
    )

    tuning_offset = calculate_global_offset(frequency, TUNE_CONFIDENCE)
    
    midi_full = hz_to_midi(frequency) - tuning_offset
    chrom_midis = np.arange(36, 97, dtype=float)

    print("CREPE ran successfully ✅")
    print(f"Frames: {len(time)}")
    print(f"Activation shape: {activation.shape}")
    voiced = np.isfinite(frequency)
    print(f"Voiced frames: {voiced.sum()} / {len(frequency)}")
    print(f"Mean confidence (all): {np.mean(confidence):.3f}")
    if voiced.any():
        print(f"Mean f0 (voiced): {np.nanmean(frequency):.2f} Hz")