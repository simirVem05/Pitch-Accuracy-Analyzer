🎤 Pitch Accuracy Analyzer

A musically aware pitch analysis system that evaluates how well vocals fit a target key and how accurately notes are sung, designed for modern pop, R&B, hip-hop, and mainstream indie artists.

This project goes beyond traditional “on-pitch vs off-pitch” metrics by incorporating musical context, intonation tolerance, and expressive singing behavior (e.g. vibrato, passing tones), producing feedback that aligns with how humans actually perceive good singing.

🚀 Features

Key Compliance Analysis

Determines whether sung notes belong to:

Diatonic scale tones

Blue notes (♭3, ♭5, ♭7)

Contextually valid chromatic notes

Passing tones

Penalizes notes that are musically out of context rather than merely “non-diatonic”.

Intonation Tightness Scoring

Measures how closely a singer hits each intended pitch (in cents).

Uses a perceptual cosine mapping rather than linear error.

Accounts for vibrato, portamento, and natural pitch movement.

Combined Performance Score

A time-varying metric that combines:

Performance Score = Key Compliance × Intonation Tightness


Separates note choice from note execution.

Context-Aware Pitch Classification

Automatically detects:

Passing notes

Repeated chromatic color tones

Borrowed scale tones (modal mixture)

Avoids falsely penalizing musically valid melodic movement.

Expressive Singing Handling

Detects musical vibrato via frequency-domain analysis.

Uses falsetto heuristics to avoid over-penalizing register changes.

Relaxes stability penalties when expressive techniques are detected.

True Silence Handling

Gaps are shown as gaps on the graph (no misleading “0%” values).

Uses RMS energy gating and morphological cleanup to isolate vocal regions.

Optimized Pitch Tracking

pYIN is run only on high-energy vocal regions instead of the entire track.

Significantly improves performance on long audio files.

🧠 How It Works (High-Level)

Audio Loading & Preprocessing

Optional pre-emphasis and harmonic–percussive separation.

Audio is framed using a consistent hop size (≈ 8–12 ms).

Energy Gating

RMS energy is computed per frame.

Quiet frames are discarded early to avoid unnecessary pitch tracking.

Pitch Extraction

Fundamental frequency (F0) is extracted using librosa pYIN.

A global tuning offset is estimated and corrected.

Note Segmentation

Consecutive frames quantizing to the same pitch are grouped into notes.

Musical Classification

Each note is labeled as:

scale, blue, context, passing, or out

Contextual chromatic notes are discovered automatically.

Intonation Analysis

Per-frame pitch deviation (in cents) is computed.

Stability, vibrato, and falsetto are evaluated per note.

Rolling Window Scoring

Scores are smoothed over a sliding time window.

Windows without sufficient vocals are masked out.

Visualization

Produces a time-aligned plot of:

Key compliance

Intonation tightness

Combined performance score

📊 Output

Matplotlib graph showing performance over time with real gaps

Optional markers for near-perfect performance regions

Designed for easy integration into a frontend (React dashboard)

🛠 Tech Stack

Backend

Python

librosa (audio + pitch tracking)

music21 (music theory & pitch classes)

numpy, scipy

matplotlib

Frontend (in progress)

React

Vite

🎯 Intended Use

This tool is optimized for artists with smooth, relatively stable vocal styles, common in:

Pop

R&B

Hip-hop

Mainstream indie

It intentionally trades off perfect accuracy for high musical relevance, meaning:

Expressive but controlled singing is rewarded.

Genuinely off-pitch singing is penalized.

Extremely expressive or classical styles are not the primary target.

⚠️ Limitations

Not designed for highly ornamented or operatic singing styles.

Performance depends on acapella quality.

Very heavy vibrato or extreme pitch bends may still be under-represented.

▶️ Running the Project
python main.py


Place your acapella file in the backend directory and update:

AUDIO_FILE = "your_file_here.mp3"
TARGET_TONIC = "C"
TARGET_MODE = "major"

📌 Future Work

Real-time API for frontend integration

Artist-friendly feedback summaries

Adaptive genre-specific tuning profiles

Faster pitch extraction alternatives