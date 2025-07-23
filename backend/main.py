import os
from pydub import AudioSegment
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np

file_path = os.path.join(os.path.dirname(__file__), "vocals.mp3")
audio = AudioSegment.from_mp3(file_path)


wav_path = os.path.join(os.path.dirname(__file__), "vocals.wav")
audio.export(wav_path, format="wav")
print("Exported to vocals.wav ✅")

y, sr = sf.read(wav_path)
print("Waveform shape:", y.shape)
print("Sample rate:", sr)





if y.ndim > 1:
    y = np.mean(y, axis=1)


f0, voiced_flag, voiced_probs = librosa.pyin(
    y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
)


print("f0 shape:", f0.shape)
print("First 10 estimated pitches (Hz):", f0[:10])


plt.figure(figsize=(12, 6))


plt.subplot(2, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title("Waveform")
times = librosa.times_like(f0, sr=sr)


plt.subplot(2, 1, 2)
plt.plot(times, f0, label='Estimated pitch (Hz)', color='b')
plt.title("Pitch Track (librosa.pyin)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.legend()
plt.tight_layout()
plt.show()