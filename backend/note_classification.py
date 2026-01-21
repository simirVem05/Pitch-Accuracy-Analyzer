import librosa
import numpy as np
from music21 import scale
from dataclasses import dataclass
import json

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

def generate_json(notes, tonic, mode):
    compliant_notes = [n for n in notes if n.klass in {'scale', 'blue', 'context', 'passing'}]
    overall_compliance = (len(compliant_notes) / len(notes)) * 100 if notes else 0
    deviations = [n.cents_med for n in notes]
    overall_deviance = np.median(np.abs(deviations))

    report = {
        "metadata": {
            "target_key": f"{tonic} {mode}",
            "total_notes_detected": len(notes),
            "global_key_compliance_rate": f"{round(overall_compliance, 1)}%",
            "global_intonation_tightness": f"{overall_deviance}"
        },
        "note_details": []
    }

    for n in notes:
        note_name = librosa.midi_to_note(int(round(n.midi_target)))

        note_data = {
            "start_time_sec": round(n.t0, 2),
            "duration_sec": round(n.duration, 3),
            "target_note_name": note_name,
            "target_midi": float(n.midi_target),
            "classification": n.klass,
            "is_musically_valid": n.klass in {'scale', 'blue', 'note', 'context', 'passing'},
            "median_cents_deviation": round(n.cents_med, 1),
            "pitch_tendency": "sharp" if n.cents_med > 5 else "flat" if n.cents_med < -5 else "perfect"
        }
        report["note_details"].append(note_data)
    
    return json.dumps(report, indent=2)