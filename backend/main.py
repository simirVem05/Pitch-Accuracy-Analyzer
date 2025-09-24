import os
from pydub import AudioSegment
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
from music21 import scale, pitch
import math

# mp3 to wav conversion
file_path = os.path.join(os.path.dirname(__file__), "wildThoughts.mp3")
audio = AudioSegment.from_mp3(file_path)
wav_path = os.path.join(os.path.dirname(__file__), "wildThoughts.wav")
audio.export(wav_path, format="wav")
print("Exported to wildThoughts.wav ")

# loading the wav file with its og sample rate as well as an array of amplitudes(y)
y, sr = librosa.load(wav_path, sr=None)

# regardless of the sample rate the target hop length will be about 8 ms
hop_ms = 8 # desired time spacing between starts of each interval
hop_length = round(sr * (hop_ms / 1000)) # desired number of samples in a time interval

# creating an array of fundamental frequencies(f0) in hz for every time interval
f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz('C2'),
    fmax=librosa.note_to_hz('C7'),
    hop_length=hop_length
)

user_tonic = input("Enter tonic: ").strip()
user_mode = input("Enter mode: ").strip().lower()

# creating the appropriate scale object based on user input
if user_mode == "major":
    sc = scale.MajorScale(user_tonic)
elif user_mode == "minor":
    sc = scale.MinorScale(user_tonic)
else:
    raise ValueError("Mode must be major or minor")

# creating a list of notes from C2-C7 that are in the user's chosen key
notes_in_key = []
# looping through every midi key (note) from C2-C7
for midi in range(36, 97):
    # creating a Pitch object to represent the current midi key
    p = pitch.Pitch()
    p.midi = midi
    # looping through every note in the key and comparing notes, ignoring octave difference
    for deg in range(1, 8):
        # comparing the Pitch object p with d to see if they have equivalent pitch
        d = sc.pitchFromDegree(deg)
        d.octave = p.octave
        if d.pitchClass == p.pitchClass:
            notes_in_key.append(p.nameWithOctave)
print(notes_in_key)

# creating a list of the frequencies of all notes in key from C2-C7 
note_freqs = []
for note in notes_in_key:
    p = pitch.Pitch(note)
    note_freqs.append(p.frequency)

# a method to convert cents to a percentage
def cents_to_percent(c, zero_at=30.0):
    percentage = 1.0 - (abs(c) / zero_at)
    return max(0.0, percentage) * 100

# a list that holds the percentages of how on-key the artist is on every note
percentages = []
# iterating through the array of fundamental frequencies f0
for index, f1 in enumerate(f0):
    if voiced_prob[index] >= 0.8 and voiced_flag[index] and np.isfinite(f1):
        # finding the minimum difference in frequency between the note the singer sang and one of the notes in note_freqs
        min_cents = float("inf")
        # comparing the current fundamental frequency f1 to every frequency f2 in note_freqs (frequencies of every note in the key)
        for f2 in note_freqs:
            cents = 1200 * math.log2(f1 / f2)
            if abs(cents) < min_cents:
                min_cents = abs(cents)
            # converting the diff in frequencies to cents
        percentages.append(cents_to_percent(min_cents))

# creating a numpy array that has np.nan for unvoiced frames and percentages from the list percentages for voiced frames
percentages_arr = np.array(percentages, dtype=float)
mask = (voiced_flag == True) & (voiced_prob >= 0.8) & np.isfinite(f0)
percentages_full = np.full_like(f0, np.nan, dtype=float)
# for every value in mask which is true, we are changing the value to that of percentages_arr
percentages_full[mask] = percentages_arr

times_all = librosa.times_like(f0, sr=sr, hop_length=hop_length)

plt.figure()
plt.plot(times_all, percentages_full)
plt.xlabel("Time (Seconds)")
plt.ylabel("On-Key Percentage (%)")
plt.title("Pitch Accuracy Over Time")
plt.show()