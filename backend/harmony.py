from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np


CHROMA_SR = 22050
CHROMA_HOP = 512

# Thresholds on the combined local/global support score.
#
# Calibrated against the only available anchor: commercially released tracks,
# whose vocals are by definition in-key. Measured across the five test songs,
# combined support is 0.68-0.82 for the real vocal and ~0.55 for the same notes
# transposed to a wrong pitch class. This band maps released recordings to
# 72-100% and wrong keys to 17-28%.
#
# A working calibration on five songs, not a fit to labelled data.
RANK_LOW = 0.50
RANK_HIGH = 0.75

DEFAULT_SLACK_S = 0.25

# Weight of the global (whole-song key membership) term relative to the local
# (this-moment chord support) term, combined as a geometric mean.
#
# Local support alone cannot distinguish two very different cases: a note outside
# the song's key entirely, and an in-key note sitting over a chord that happens
# not to contain it — a passing tone, suspension or anticipation, which is
# musically correct but locally unsupported. The global term separates them,
# because a genuinely wrong note is low on BOTH terms while a passing tone is
# high on one.
#
# A geometric mean is used rather than a weighted sum so the terms gate rather
# than compensate: strong key membership cannot fully rescue a note the current
# harmony rejects outright.
#
# Measured on the five test songs: this raised "true key ranks first" from 4/5 to
# 5/5 and improved sensitivity to scattered wrong notes by ~25%. The 0.20-0.30
# range performs equivalently, so this sits mid-plateau rather than on a peak.
GLOBAL_WEIGHT = 0.25

# An a cappella upload still yields a nominal "harmony" stem, but it holds only
# separation residue of the voice itself — scoring against it would compare the
# vocal to its own leakage and return a confident, meaningless number. Measured
# on the bundled samples: real mixes sit at harmony/vocal RMS ~0.18, a cappellas
# at ~0.005-0.009, so this threshold has more than an order of magnitude of room.
MIN_HARMONY_TO_VOCAL_RMS = 0.04


@dataclass(frozen=True)
class HarmonicContext:
    """Tuning-aligned pitch-class prominence for the accompaniment."""

    rank: np.ndarray            # (12, n_frames), per-frame rank in [0, 1]
    global_rank: np.ndarray     # (12,), whole-song pitch-class rank in [0, 1]
    salience: np.ndarray        # (12, n_frames), raw CENS chroma
    frame_times: np.ndarray     # (n_frames,)
    hop_s: float
    tuning_semitones: float     # fractional-semitone offset from A440
    beat_times: np.ndarray      # (n_beats,), may be empty
    tempo_bpm: float
    slack_s: float              # one beat, or DEFAULT_SLACK_S if untracked
    has_accompaniment: bool

    @property
    def tuning_cents(self) -> float:
        return float(self.tuning_semitones * 100.0)


def estimate_tuning_semitones(y: np.ndarray, sr: int) -> float:
    """
    Fractional-semitone offset of the accompaniment from the A440 grid.

    Estimated from the instrumental rather than the voice: fixed-pitch
    instruments define the grid the singer is aiming at. An unremoved offset
    shifts every cents measurement uniformly and smears energy between chroma
    bins, so this feeds both axes.
    """
    if y.size == 0:
        return 0.0
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    return float(tuning) if np.isfinite(tuning) else 0.0


def _to_rank(chroma: np.ndarray) -> np.ndarray:
    """
    Convert per-frame salience to per-frame rank in [0, 1].

    Absolute salience is dominated by level and by how much of the octave a
    given arrangement fills, so a raw threshold is not comparable across
    sections of a song. Rank asks the question that actually matters — is this
    pitch class prominent *relative to the other eleven right now* — and
    measured roughly twice the separation between real and transposed notes.
    """
    if chroma.shape[1] == 0:
        return chroma
    return np.argsort(np.argsort(chroma, axis=0), axis=0).astype(float) / float(chroma.shape[0] - 1)


def _track_beats(y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    if y.size < sr:
        return np.zeros(0, dtype=float), 0.0
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    except Exception:
        return np.zeros(0, dtype=float), 0.0
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    return np.asarray(beats, dtype=float), tempo


def build_harmonic_context(
    harmony: np.ndarray,
    percussive: np.ndarray,
    sample_rate: int,
    vocals: Optional[np.ndarray] = None,
) -> HarmonicContext:
    """
    Pitch-class content of the accompaniment, aligned to its own tuning.

    Chroma is used rather than per-stem pitch tracking because the accompaniment
    is irreducibly polyphonic: no separator yields individual instruments, and a
    monophonic tracker would pick one arbitrary note per frame. Chroma never
    commits to a single note, so chords need no special handling. CENS is used
    over plain CQT chroma because its local normalization gave measurably better
    contrast between prominent and absent pitch classes.

    Pass the `other` stem alone, not other+bass. Bass measures 2.8-4.6x louder
    than `other` in the test mixes, so summing the waveforms lets root notes
    dominate the chromagram and drown out the chord voicings that actually
    distinguish one harmony from another. Measured cost of including bass: the
    margin over the best wrong key falls from +0.043 to +0.014 and the true key
    stops ranking first on half the songs (see evaluate.py).
    """
    y = (
        librosa.resample(harmony, orig_sr=sample_rate, target_sr=CHROMA_SR)
        if sample_rate != CHROMA_SR
        else harmony
    )
    y = np.ascontiguousarray(y, dtype=np.float32)

    tuning = estimate_tuning_semitones(y, CHROMA_SR)

    salience = np.asarray(
        librosa.feature.chroma_cens(y=y, sr=CHROMA_SR, hop_length=CHROMA_HOP, tuning=tuning),
        dtype=float,
    )
    frame_times = librosa.frames_to_time(
        np.arange(salience.shape[1]), sr=CHROMA_SR, hop_length=CHROMA_HOP
    )
    hop_s = float(CHROMA_HOP) / float(CHROMA_SR)

    beat_source = percussive if percussive.size else harmony
    beat_y = (
        librosa.resample(beat_source, orig_sr=sample_rate, target_sr=CHROMA_SR)
        if sample_rate != CHROMA_SR
        else beat_source
    )
    beat_times, tempo = _track_beats(np.ascontiguousarray(beat_y, dtype=np.float32), CHROMA_SR)

    has_accompaniment = True
    if vocals is not None and vocals.size:
        vocal_rms = float(np.sqrt(np.mean(np.square(vocals, dtype=np.float64))))
        harmony_rms = float(np.sqrt(np.mean(np.square(harmony, dtype=np.float64))))
        if vocal_rms > 0:
            has_accompaniment = (harmony_rms / vocal_rms) >= MIN_HARMONY_TO_VOCAL_RMS

    global_profile = salience.mean(axis=1) if salience.shape[1] else np.zeros(12)
    global_rank = np.argsort(np.argsort(global_profile)).astype(float) / 11.0

    return HarmonicContext(
        rank=_to_rank(salience),
        global_rank=global_rank,
        salience=salience,
        frame_times=frame_times,
        hop_s=hop_s,
        tuning_semitones=tuning,
        beat_times=beat_times,
        tempo_bpm=tempo,
        slack_s=float(60.0 / tempo) if tempo > 0 else DEFAULT_SLACK_S,
        has_accompaniment=has_accompaniment,
    )


def _rank_to_compliance(rank: float) -> float:
    if rank <= RANK_LOW:
        return 0.0
    if rank >= RANK_HIGH:
        return 1.0
    return float((rank - RANK_LOW) / (RANK_HIGH - RANK_LOW))


def score_key_compliance(segments: List[Dict], ctx: HarmonicContext) -> List[Dict]:
    """
    Score each note against the accompaniment's pitch-class prominence.

    The note's own span is measured, then the same span shifted one beat earlier
    and one beat later; the best of the three wins. Shifting a fixed-width window
    rather than widening it is what makes the slack meaningful — a widened window
    dilutes a short note into several beats of unrelated harmony and raises the
    score for every pitch class equally, which forgives wrong notes just as much
    as anticipated ones. Shifting instead asks the intended question: was this
    note supported by the chord just before or just after it? That is what
    accommodates anticipations and melisma across chord changes.

    Chroma is octave-blind, so this measures pitch-class fit only and cannot
    detect a note sung in the wrong octave.
    """
    n_frames = ctx.rank.shape[1]
    if n_frames == 0 or not ctx.has_accompaniment:
        for seg in segments:
            seg["harmonic_rank"] = None
            seg["key_compliance"] = None
        return segments

    slack_frames = int(round(ctx.slack_s / ctx.hop_s)) if ctx.hop_s > 0 else 0

    for seg in segments:
        pc = int(seg["pitch_class"])
        i0 = int(np.searchsorted(ctx.frame_times, float(seg["start"]), side="left"))
        i1 = int(np.searchsorted(ctx.frame_times, float(seg["end"]), side="right"))
        i0 = min(max(i0, 0), max(n_frames - 1, 0))
        i1 = max(i1, i0 + 1)

        if i0 >= n_frames:
            seg["harmonic_rank"] = None
            seg["key_compliance"] = None
            continue

        width = min(i1, n_frames) - i0
        local = float(np.mean(ctx.rank[pc, i0 : i0 + width]))

        for offset in (-slack_frames, slack_frames):
            if offset == 0:
                continue
            a = i0 + offset
            b = a + width
            if a >= 0 and b <= n_frames:
                local = max(local, float(np.mean(ctx.rank[pc, a:b])))

        global_support = float(ctx.global_rank[pc])
        combined = float(
            (local ** (1.0 - GLOBAL_WEIGHT)) * (max(global_support, 1e-6) ** GLOBAL_WEIGHT)
        )

        seg["local_support"] = local
        seg["global_support"] = global_support
        seg["harmonic_rank"] = combined
        seg["key_compliance"] = _rank_to_compliance(combined)

    return segments
