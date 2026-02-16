Pitch Accuracy Analyzer

A perception-first vocal pitch analysis system designed for modern artists.

Pitch Accuracy Analyzer evaluates vocal intonation using neural pitch detection, music theory–aware classification, and a perceptual scoring curve optimized for expressive genres like R&B, Pop, and Hip Hop.

It is not a basic tuner. It is a full audio signal processing pipeline with musical context modeling and AI-generated coaching feedback.

What This Project Does

Given a vocal recording, selected key, and genre, the system:

Extracts pitch using a neural pitch detection model (CREPE).

Segments the audio into stable melodic note regions.

Computes pitch deviation using trimmed median logic.

Evaluates musical correctness relative to key and genre.

Applies a perceptual intonation scoring curve.

Detects stylistic techniques such as vibrato and portamento.

Generates:

A time-series performance graph

Quantitative summary metrics

A professional vocal coaching report using Gemini

Why This Is Different

Most pitch analyzers:

Evaluate frame-by-frame pitch

Punish expressive slides

Use hard thresholds

Ignore musical context

This system is perception-first:

Uses hysteresis-based note segmentation

Computes core pitch centers (trims expressive edges)

Applies genre-aware pitch class allowances

Smooths visualization to reflect what listeners actually perceive

Treats vibrato and portamento as stylistic features, not mistakes

Architecture Overview

Backend (Python):

preprocess.py
Audio loading, high-pass filtering, CREPE pitch extraction, voicing mask.

note_segmentation.py
Converts frequency to MIDI, groups frames into NoteSegs using hysteresis, computes core median cents deviation.

note_classification.py
Key-aware pitch classification, contextual rescue logic, vibrato and portamento detection.

scoring.py
Perceptual intonation curve, soft penalties, sliding median smoothing, silence break logic.

main.py
Orchestrates full pipeline and generates metrics and Gemini report.

api.py
FastAPI wrapper exposing an /analyze endpoint.

Frontend (React + Recharts):

Upload form for vocal file

Key and genre selection

Apple-inspired minimalist UI

Step-based graph visualization

Clean performance dashboard

Core Technical Concepts
1. Neural Pitch Detection

Model: CREPE

Sample rate: 16 kHz

Step size: 20 ms

Confidence-based voicing mask

2. Hysteresis-Based Note Segmentation

Frames are grouped into notes only when pitch remains within 40 cents of a target MIDI note, and new pitch centers must persist for multiple frames before switching.

This prevents jitter and artificial note flipping.

3. Core Pitch Deviation

For notes longer than 10 frames:

Trim first 20 percent

Trim last 20 percent

Compute median over middle 60 percent

This avoids penalizing stylistic scoops and releases.

4. Perceptual Intonation Curve

Absolute cents deviation → score (0.0–1.0)

0–25 cents: 0.90–1.00 (Pro Zone)

25–45 cents: 0.65–0.90 (Mediocre Zone)

45+ cents: exponential decay

Score floor: 0.05

This matches modern listening tolerance.

5. Musical Context Awareness

Segments are classified as:

Diatonic

Blue

Chromatic

Dissonant

Contextual rescue logic detects passing tones, neighbor tones, and leading tones to avoid over-penalizing intentional tension.

6. Stylization Detection

Vibrato:

FFT band energy detection in 3.5–9.5 Hz range

Portamento:

Slide magnitude >= 140 cents

Linear fit R squared threshold

These are reported but not judged as good or bad.

7. Visualization Refinement

1-second sliding median smoothing

Silence gap detection inserts null values

Step-based graph to represent stable pitch centers

API Usage

POST /analyze

Form data:

file: audio file

key: string (example "B minor")

genre: string (example "rnb")

Returns:

{
"metrics": {...},
"graph_tuples": [...],
"report": "..."
}

Running the Project
Backend

Create virtual environment:

python -m venv venv
source venv/bin/activate (Mac/Linux)
venv\Scripts\activate (Windows)

Install dependencies:

pip install -r requirements.txt

Run server:

uvicorn api:app --reload

Open:
http://127.0.0.1:8000/docs

Frontend

npm install
npm run dev

Example Output

Metrics:

Median cents deviation

Percentage of high-scoring notes

Percentage of mediocre notes

Percentage of low notes

Vibrato count

Portamento count

Visualization:

Time-series step graph

No artificial diagonal connections during silence

Report:

Gemini-generated vocal coaching feedback based only on quantitative metrics.

Design Philosophy

This system is built around three principles:

Perception First
Score what listeners hear, not raw frame math.

Musical Context Matters
A non-diatonic note is not automatically wrong.

Expressiveness Is Not a Mistake
Slides and vibrato are stylistic tools, not tuning errors.

Future Improvements

Duration-weighted scoring metrics

Adaptive confidence threshold

Real-time streaming mode

Signed cents bias detection (sharp vs flat tendencies)

Multi-take comparison

Tech Stack

Python
NumPy
Librosa
SciPy
CREPE
FastAPI
React
Recharts
Gemini API

Author

Built as a perception-first vocal analysis system combining digital signal processing, music theory modeling, and AI-generated feedback.
