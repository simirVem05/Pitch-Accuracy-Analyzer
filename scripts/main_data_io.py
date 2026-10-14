"""
Audio-free I/O shared by the canonical feature-store tools.

`build_main_dataset.py` (audio -> features) and `resegment_main_dataset.py`
(harmony.npz -> notes) must build `notes.npz` identically, so the note-array
construction, atomic writers, and validation live here rather than in either
script. This module imports only numpy and the backend's pure segmentation
helpers: no Demucs, no CREPE, no TensorFlow or torch.

Schema versions of data/main_data/<song>/:
  1  notes.npz produced by v1 segmentation directly from audio
  2  notes.npz re-segmented from harmony.npz by a recorded segmentation
     config; v1 notes preserved alongside as notes_v1.npz. The notes.npz array
     layout is identical in both; schema 2 adds provenance to metadata.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from note_segmentation import cents_diff, midi_to_hz  # noqa: E402


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
KNOWN_SCHEMA_VERSIONS = (1, 2)

# Metadata a schema-2 song must carry so every note can be traced to the exact
# segmentation code, config, and source track that produced it.
SCHEMA2_METADATA_KEYS = (
    "segmentation_version",
    "segmentation_algorithm",
    "segmentation_config",
    "segmentation_source",
    "harmony_npz_sha256",
    "notes_npz_content_sha256",
    "v1_notes_path",
    "v1_notes_sha256",
    "v1_notes_content_sha256",
    "v1_number_of_segmented_notes",
    "resegmentation_git_commit",
    "resegmentation_timestamp_utc",
)


class ValidationError(ValueError):
    """Raised when generated dataset files do not satisfy schema invariants."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False) as tmp:
        json.dump(_jsonable(dict(payload)), tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".npz", dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def npz_content_sha256(source: Path | Mapping[str, np.ndarray]) -> str:
    """
    Hash of an npz's arrays (names, dtypes, shapes, bytes), independent of the
    zip container. np.savez stamps each zip entry with the write time, so two
    byte-different files can hold identical arrays; this is the hash that
    decides whether two outputs are the same.
    """
    def digest(arrays: Mapping[str, np.ndarray]) -> str:
        h = hashlib.sha256()
        for key in sorted(arrays):
            a = np.ascontiguousarray(arrays[key])
            h.update(key.encode())
            h.update(a.dtype.str.encode())
            h.update(repr(a.shape).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    if isinstance(source, Path):
        with np.load(source, allow_pickle=False) as data:
            return digest({k: data[k] for k in data.files})
    return digest(source)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _midi_note_name(midi_note: int) -> str:
    return f"{NOTE_NAMES[int(midi_note) % 12]}{int(midi_note) // 12 - 1}"


def _require_keys(container: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(container.keys()))
    if missing:
        raise ValidationError(f"{label} is missing required fields: {', '.join(missing)}")


def _note_arrays(
    segments: Sequence[Mapping[str, Any]],
    frame_indices: Sequence[np.ndarray],
    track: Any,
    tuning_semitones: float,
) -> Dict[str, np.ndarray]:
    """`track` needs `.time`, `.frequency` (NaN = unvoiced) and `.confidence`, as on PitchTrack."""
    offsets = np.zeros(len(segments) + 1, dtype=np.int64)
    times: List[np.ndarray] = []
    f0_values: List[np.ndarray] = []
    cents_values: List[np.ndarray] = []
    confidence_values: List[np.ndarray] = []
    source_indices: List[np.ndarray] = []

    for note_index, (segment, indices) in enumerate(zip(segments, frame_indices)):
        indices = np.asarray(indices, dtype=np.int64)
        f0 = np.asarray(track.frequency[indices], dtype=np.float64)
        if indices.size == 0 or not np.all(np.isfinite(f0)):
            raise ValidationError(f"Note {note_index} contains missing or invalid voiced frames")
        target_hz = midi_to_hz(int(segment["target_note"]), tuning_semitones)
        cents = np.asarray(cents_diff(f0, target_hz), dtype=np.float64)

        source_indices.append(indices)
        times.append(np.asarray(track.time[indices], dtype=np.float64))
        f0_values.append(f0)
        cents_values.append(cents)
        confidence_values.append(np.asarray(track.confidence[indices], dtype=np.float64))
        offsets[note_index + 1] = offsets[note_index] + indices.size

    concatenate = lambda values, dtype: np.concatenate(values).astype(dtype, copy=False) if values else np.empty(0, dtype=dtype)
    target_midi = np.asarray([int(segment["target_note"]) for segment in segments], dtype=np.int16)

    return {
        "note_index": np.arange(len(segments), dtype=np.int32),
        "target_midi_note": target_midi,
        "target_note_name": np.asarray([_midi_note_name(note) for note in target_midi], dtype="U8"),
        "pitch_class": np.asarray([int(segment["pitch_class"]) for segment in segments], dtype=np.int8),
        "start_time_s": np.asarray([float(segment["start"]) for segment in segments], dtype=np.float64),
        "end_time_s": np.asarray([float(segment["end"]) for segment in segments], dtype=np.float64),
        "duration_s": np.asarray([float(segment["duration_s"]) for segment in segments], dtype=np.float64),
        "median_cents_deviation": np.asarray([float(segment["median_cents"]) for segment in segments], dtype=np.float64),
        "core_median_cents_deviation": np.asarray(
            [float(segment["core_median_cents"]) for segment in segments], dtype=np.float64
        ),
        "voiced_frame_count": np.asarray([int(segment["n_voiced_frames"]) for segment in segments], dtype=np.int32),
        "contour_offsets": offsets,
        "contour_source_frame_index": concatenate(source_indices, np.int64),
        "contour_time_s": concatenate(times, np.float64),
        "contour_f0_hz": concatenate(f0_values, np.float64),
        "contour_cents_deviation": concatenate(cents_values, np.float64),
        "contour_crepe_confidence": concatenate(confidence_values, np.float64),
    }


def validate_song_output(
    song_dir: Path,
    expected_source_sha256: str | None = None,
    *,
    accepted_schema_versions: Sequence[int] = KNOWN_SCHEMA_VERSIONS,
    harmony_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Re-open one song's files and check every schema invariant.

    `harmony_path` overrides `song_dir / "harmony.npz"`, so a staged song
    directory (notes + metadata only) can be validated against the canonical
    harmony file it was derived from.
    """
    notes_path = song_dir / "notes.npz"
    harmony_path = harmony_path or (song_dir / "harmony.npz")
    metadata_path = song_dir / "metadata.json"
    for path in (notes_path, harmony_path, metadata_path):
        if not path.is_file():
            raise ValidationError(f"Missing required file: {path}")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Invalid metadata JSON: {exc}") from exc
    _require_keys(
        metadata,
        ("dataset_schema_version", "song_id", "original_filename", "source_sha256", "number_of_segmented_notes"),
        "metadata",
    )
    schema = metadata["dataset_schema_version"]
    if schema not in accepted_schema_versions:
        raise ValidationError(
            f"Schema {schema} is not one of the accepted schema versions {tuple(accepted_schema_versions)}"
        )
    if expected_source_sha256 and metadata["source_sha256"] != expected_source_sha256:
        raise ValidationError("Source SHA-256 does not match the generated metadata")

    with np.load(notes_path, allow_pickle=False) as notes:
        _require_keys(
            notes,
            (
                "note_index", "target_midi_note", "pitch_class", "start_time_s", "end_time_s", "duration_s",
                "contour_offsets", "contour_time_s", "contour_f0_hz", "contour_cents_deviation",
                "contour_crepe_confidence", "median_cents_deviation", "core_median_cents_deviation",
            ),
            "notes.npz",
        )
        note_count = int(notes["note_index"].size)
        if note_count < 1:
            raise ValidationError("No segmented notes were saved")
        if note_count != int(metadata["number_of_segmented_notes"]):
            raise ValidationError("Note count differs between notes.npz and metadata.json")
        for key in ("target_midi_note", "pitch_class", "start_time_s", "end_time_s", "duration_s"):
            if notes[key].size != note_count:
                raise ValidationError(f"{key} does not align with the note count")
        if not np.all(np.isfinite(notes["duration_s"])) or not np.all(notes["duration_s"] > 0):
            raise ValidationError("Every note duration must be finite and positive")

        offsets = np.asarray(notes["contour_offsets"], dtype=np.int64)
        if offsets.shape != (note_count + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) <= 0):
            raise ValidationError("Contour offsets must define one non-empty contour per note")
        contour_length = int(offsets[-1])
        for key in (
            "contour_time_s", "contour_f0_hz", "contour_cents_deviation", "contour_crepe_confidence"
        ):
            if notes[key].size != contour_length:
                raise ValidationError(f"{key} length does not match contour_offsets")
        if contour_length == 0:
            raise ValidationError("Saved note contours are unexpectedly empty")
        if not np.all(np.isfinite(notes["contour_f0_hz"])) or np.any(notes["contour_f0_hz"] <= 0):
            raise ValidationError("Note F0 contours contain invalid values")
        if not np.all(np.isfinite(notes["contour_cents_deviation"])):
            raise ValidationError("Note cents contours contain invalid values")
        confidence = notes["contour_crepe_confidence"]
        if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
            raise ValidationError("Note confidence contours fall outside [0, 1]")

        if schema >= 2:
            # Every contour frame must point back at the matching frame of the
            # full vocal track it was segmented from.
            if "contour_source_frame_index" not in notes.files:
                raise ValidationError("notes.npz is missing contour_source_frame_index")
            src = np.asarray(notes["contour_source_frame_index"], dtype=np.int64)
            if src.size != contour_length:
                raise ValidationError("contour_source_frame_index length does not match contour_offsets")
            with np.load(harmony_path, allow_pickle=False) as h:
                vt, vf, vc = h["vocal_frame_times_s"], h["vocal_f0_hz"], h["vocal_crepe_confidence"]
                if src.min() < 0 or src.max() >= vt.size:
                    raise ValidationError("contour_source_frame_index points outside the vocal track")
                if not (np.array_equal(vt[src], notes["contour_time_s"])
                        and np.array_equal(vf[src], notes["contour_f0_hz"])
                        and np.array_equal(vc[src], notes["contour_crepe_confidence"])):
                    raise ValidationError("Note contours do not match their source frames in harmony.npz")
            notes_hash = npz_content_sha256({k: notes[k] for k in notes.files})

    if schema >= 2:
        _require_keys(metadata, SCHEMA2_METADATA_KEYS, "schema-2 metadata")
        if metadata["harmony_npz_sha256"] != _sha256(harmony_path):
            raise ValidationError("harmony.npz differs from the file recorded in metadata.json")
        if metadata["notes_npz_content_sha256"] != notes_hash:
            raise ValidationError("notes.npz content differs from the hash recorded in metadata.json")

    with np.load(harmony_path, allow_pickle=False) as harmony:
        _require_keys(
            harmony,
            (
                "other_chroma_cens", "chroma_frame_times_s", "global_chroma_profile", "beat_times_s",
                "vocal_frame_times_s", "vocal_f0_hz", "vocal_crepe_confidence",
                "bass_frame_times_s", "bass_f0_hz", "bass_crepe_confidence",
            ),
            "harmony.npz",
        )
        chroma = harmony["other_chroma_cens"]
        if chroma.ndim != 2 or chroma.shape[0] != 12 or chroma.shape[1] == 0:
            raise ValidationError("Other-stem chroma must have shape (12, nonzero frames)")
        if harmony["chroma_frame_times_s"].shape != (chroma.shape[1],):
            raise ValidationError("Chroma timestamps do not align with chroma frames")
        if harmony["global_chroma_profile"].shape != (12,):
            raise ValidationError("Global chroma profile must have 12 values")
        for prefix in ("vocal", "bass"):
            lengths = {
                harmony[f"{prefix}_frame_times_s"].size,
                harmony[f"{prefix}_f0_hz"].size,
                harmony[f"{prefix}_crepe_confidence"].size,
            }
            if len(lengths) != 1 or next(iter(lengths)) == 0:
                raise ValidationError(f"{prefix.title()} CREPE arrays are empty or misaligned")
            confidence = harmony[f"{prefix}_crepe_confidence"]
            if not np.all(np.isfinite(confidence)) or np.any((confidence < 0) | (confidence > 1)):
                raise ValidationError(f"{prefix.title()} confidence values fall outside [0, 1]")

    return {"song_id": metadata["song_id"], "note_count": note_count, "status": "valid", "schema": schema}
