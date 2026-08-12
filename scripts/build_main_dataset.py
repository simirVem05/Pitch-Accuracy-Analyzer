#!/usr/bin/env python3
"""Build reusable, score-free features from the production audio pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = REPO_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from harmony import CHROMA_HOP, CHROMA_SR, build_harmonic_context  # noqa: E402
from note_segmentation import (  # noqa: E402
    cents_diff,
    midi_to_hz,
    segment_notes_with_frame_indices,
)
from preprocess import PreprocessConfig, PitchTrack, detect_pitch  # noqa: E402
from separation import Stems, separate  # noqa: E402


DATASET_SCHEMA_VERSION = 1
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "test_songs"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "main_data"
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif"}
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

LOGGER = logging.getLogger("build_main_dataset")


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


def _song_slug(filename: str) -> str:
    stem = Path(filename).stem
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_stem.lower()).strip("_")
    return slug or "song"


def _midi_note_name(midi_note: int) -> str:
    return f"{NOTE_NAMES[int(midi_note) % 12]}{int(midi_note) // 12 - 1}"


def _load_manifest(output_dir: Path) -> Dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.exists():
        return {"dataset_schema_version": DATASET_SCHEMA_VERSION, "songs": []}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read existing manifest {path}: {exc}") from exc
    if not isinstance(manifest.get("songs"), list):
        raise ValidationError(f"Manifest {path} has no songs list")
    return manifest


def _save_manifest(output_dir: Path, manifest: Dict[str, Any]) -> None:
    manifest["dataset_schema_version"] = DATASET_SCHEMA_VERSION
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["songs"] = sorted(manifest.get("songs", []), key=lambda item: item.get("song_id", ""))
    _atomic_json(output_dir / "manifest.json", manifest)


def _upsert_manifest_record(manifest: Dict[str, Any], record: Dict[str, Any]) -> None:
    songs = manifest.setdefault("songs", [])
    for index, existing in enumerate(songs):
        if existing.get("song_id") == record.get("song_id"):
            songs[index] = record
            return
    songs.append(record)


def _choose_song_id(path: Path, source_sha256: str, manifest: Mapping[str, Any], output_dir: Path) -> str:
    for record in manifest.get("songs", []):
        if record.get("source_sha256") == source_sha256 and record.get("song_id"):
            return str(record["song_id"])

    base = _song_slug(path.name)
    metadata_path = output_dir / base / "metadata.json"
    if not metadata_path.exists():
        return base
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"{base}_{source_sha256[:8]}"
    if existing.get("source_sha256") == source_sha256 or existing.get("original_filename") == path.name:
        return base
    return f"{base}_{source_sha256[:8]}"


def _note_arrays(
    segments: Sequence[Mapping[str, Any]],
    frame_indices: Sequence[np.ndarray],
    track: PitchTrack,
    tuning_semitones: float,
) -> Dict[str, np.ndarray]:
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


def _harmony_arrays(context: Any, vocal_track: PitchTrack, bass_track: PitchTrack) -> Dict[str, np.ndarray]:
    return {
        "other_chroma_cens": np.asarray(context.salience, dtype=np.float32),
        "other_chroma_rank": np.asarray(context.rank, dtype=np.float32),
        "chroma_frame_times_s": np.asarray(context.frame_times, dtype=np.float64),
        "global_chroma_profile": np.asarray(context.global_profile, dtype=np.float64),
        "global_chroma_rank": np.asarray(context.global_rank, dtype=np.float64),
        "beat_times_s": np.asarray(context.beat_times, dtype=np.float64),
        "estimated_tempo_bpm": np.asarray(context.tempo_bpm, dtype=np.float64),
        "tuning_offset_semitones": np.asarray(context.tuning_semitones, dtype=np.float64),
        "tuning_offset_cents": np.asarray(context.tuning_cents, dtype=np.float64),
        "vocal_frame_times_s": np.asarray(vocal_track.time, dtype=np.float64),
        "vocal_f0_hz": np.asarray(vocal_track.frequency, dtype=np.float64),
        "vocal_crepe_confidence": np.asarray(vocal_track.confidence, dtype=np.float64),
        "vocal_voiced_mask": np.asarray(np.isfinite(vocal_track.frequency), dtype=np.bool_),
        "bass_frame_times_s": np.asarray(bass_track.time, dtype=np.float64),
        "bass_f0_hz": np.asarray(bass_track.frequency, dtype=np.float64),
        "bass_crepe_confidence": np.asarray(bass_track.confidence, dtype=np.float64),
        "bass_voiced_mask": np.asarray(np.isfinite(bass_track.frequency), dtype=np.bool_),
    }


def _metadata(
    path: Path,
    song_id: str,
    source_sha256: str,
    stems: Stems,
    context: Any,
    vocal_track: PitchTrack,
    bass_track: PitchTrack,
    note_count: int,
    config: PreprocessConfig,
    git_commit: str | None,
) -> Dict[str, Any]:
    vocal_voiced = int(np.count_nonzero(np.isfinite(vocal_track.frequency)))
    bass_voiced = int(np.count_nonzero(np.isfinite(bass_track.frequency)))
    return {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "original_filename": path.name,
        "source_relative_path": str(path.relative_to(REPO_ROOT)),
        "source_sha256": source_sha256,
        "song_id": song_id,
        "duration_s": stems.duration_s,
        "demucs_model": os.getenv("DEMUCS_MODEL", "htdemucs"),
        "demucs_device": os.getenv("DEMUCS_DEVICE", "auto"),
        "crepe_model_capacity": config.model_capacity,
        "crepe_step_size_ms": config.step_size_ms,
        "crepe_sample_rate_hz": config.sample_rate,
        "source_stem_sample_rate_hz": stems.sample_rate,
        "chroma_sample_rate_hz": CHROMA_SR,
        "chroma_hop_length_samples": CHROMA_HOP,
        "crepe_confidence_threshold": config.conf_threshold,
        "crepe_viterbi": config.viterbi,
        "pitch_range_hz": [config.min_f0_hz, config.max_f0_hz],
        "tuning_offset_semitones": context.tuning_semitones,
        "tuning_offset_cents": context.tuning_cents,
        "estimated_tempo_bpm": context.tempo_bpm,
        "number_of_segmented_notes": note_count,
        "vocal_total_frame_count": int(vocal_track.frequency.size),
        "vocal_voiced_frame_count": vocal_voiced,
        "vocal_unvoiced_frame_count": int(vocal_track.frequency.size - vocal_voiced),
        "vocal_coverage": vocal_track.coverage,
        "bass_total_frame_count": int(bass_track.frequency.size),
        "bass_voiced_frame_count": bass_voiced,
        "bass_coverage": bass_track.coverage,
        "other_chromagram_only": True,
        "processing_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
    }


def _require_keys(container: Mapping[str, Any], required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(container.keys()))
    if missing:
        raise ValidationError(f"{label} is missing required fields: {', '.join(missing)}")


def validate_song_output(song_dir: Path, expected_source_sha256: str | None = None) -> Dict[str, Any]:
    notes_path = song_dir / "notes.npz"
    harmony_path = song_dir / "harmony.npz"
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
    if metadata["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValidationError(
            f"Schema {metadata['dataset_schema_version']} does not match current schema {DATASET_SCHEMA_VERSION}"
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

    return {"song_id": metadata["song_id"], "note_count": note_count, "status": "valid"}


def _existing_output_is_current(song_dir: Path, source_sha256: str) -> Tuple[bool, int]:
    try:
        result = validate_song_output(song_dir, expected_source_sha256=source_sha256)
    except (OSError, ValueError, ValidationError):
        return False, 0
    return True, int(result["note_count"])


def _process_song(
    path: Path,
    song_id: str,
    source_sha256: str,
    output_dir: Path,
    git_commit: str | None,
) -> Dict[str, Any]:
    config = PreprocessConfig()
    stems = separate(str(path))

    # Production intentionally builds chroma from `other` only; bass stays an
    # independent contour and is never mixed into this harmonic representation.
    context = build_harmonic_context(stems.other, stems.percussive, stems.sample_rate, vocals=stems.vocals)
    vocal_track = detect_pitch(stems.vocals, stems.sample_rate, config=config)
    segments, frame_indices = segment_notes_with_frame_indices(
        vocal_track.time, vocal_track.frequency, context.tuning_semitones
    )
    if not segments:
        raise ValidationError("Production segmentation detected no vocal notes")

    # This deliberately reuses the production CREPE configuration. It is a useful
    # exploratory root-motion feature, not a claim that the vocal-tuned gate is an
    # optimal bass tracker.
    bass_track = detect_pitch(stems.bass, stems.sample_rate, config=config)

    notes = _note_arrays(segments, frame_indices, vocal_track, context.tuning_semitones)
    harmony = _harmony_arrays(context, vocal_track, bass_track)
    metadata = _metadata(
        path, song_id, source_sha256, stems, context, vocal_track, bass_track,
        len(segments), config, git_commit,
    )

    song_dir = output_dir / song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    _atomic_npz(song_dir / "notes.npz", notes)
    _atomic_npz(song_dir / "harmony.npz", harmony)
    _atomic_json(song_dir / "metadata.json", metadata)

    validation = validate_song_output(song_dir, expected_source_sha256=source_sha256)
    return {
        "song_id": song_id,
        "source_filename": path.name,
        "source_relative_path": str(path.relative_to(REPO_ROOT)),
        "source_sha256": source_sha256,
        "number_of_notes": validation["note_count"],
        "processing_status": "processed",
        "validation_status": "passed",
        "notes_path": str((song_dir / "notes.npz").relative_to(REPO_ROOT)),
        "harmony_path": str((song_dir / "harmony.npz").relative_to(REPO_ROOT)),
        "metadata_path": str((song_dir / "metadata.json").relative_to(REPO_ROOT)),
        "processed_at_utc": metadata["processing_timestamp_utc"],
    }


def _audio_files(input_dir: Path) -> List[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(
        (path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES),
        key=lambda path: path.name,
    )


def validate_dataset(output_dir: Path) -> Tuple[int, int]:
    manifest = _load_manifest(output_dir)
    checked = 0
    total_notes = 0
    failures: List[str] = []
    for record in manifest.get("songs", []):
        if record.get("processing_status") == "failed":
            continue
        song_id = record.get("song_id")
        if not song_id:
            failures.append("manifest record without song_id")
            continue
        try:
            result = validate_song_output(output_dir / song_id, record.get("source_sha256"))
        except (OSError, ValueError, ValidationError) as exc:
            failures.append(f"{song_id}: {exc}")
            continue
        checked += 1
        total_notes += int(result["note_count"])
    if failures:
        raise ValidationError("Dataset validation failed:\n  " + "\n  ".join(failures))
    return checked, total_notes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing source audio")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Dataset output directory")
    parser.add_argument("--limit", type=int, default=None, help="Process up to N sorted songs without valid output")
    parser.add_argument("--force", action="store_true", help="Reprocess valid current-schema outputs")
    parser.add_argument("--validate-only", action="store_true", help="Validate existing outputs without processing audio")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be a positive integer")

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        checked, total_notes = validate_dataset(output_dir)
        print(f"Validated: {checked}")
        print(f"Total notes: {total_notes:,}")
        print(f"Output: {output_dir.relative_to(REPO_ROOT)}/")
        return 0

    manifest = _load_manifest(output_dir)
    paths = _audio_files(input_dir)
    if not args.force:
        paths = [
            path
            for path in paths
            if not _existing_output_is_current(
                output_dir / _choose_song_id(path, source_sha256 := _sha256(path), manifest, output_dir),
                source_sha256,
            )[0]
        ]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"No unprocessed supported audio files found in {input_dir}")

    git_commit = _git_commit()
    processed = skipped = failed = total_notes = 0

    for position, path in enumerate(paths, start=1):
        print(f"[{position}/{len(paths)}] {path.name}", flush=True)
        source_sha256 = ""
        song_id = _song_slug(path.name)
        try:
            source_sha256 = _sha256(path)
            song_id = _choose_song_id(path, source_sha256, manifest, output_dir)
            current, existing_notes = _existing_output_is_current(output_dir / song_id, source_sha256)
            if current and not args.force:
                skipped += 1
                total_notes += existing_notes
                record = {
                    "song_id": song_id,
                    "source_filename": path.name,
                    "source_relative_path": str(path.relative_to(REPO_ROOT)),
                    "source_sha256": source_sha256,
                    "number_of_notes": existing_notes,
                    "processing_status": "skipped",
                    "validation_status": "passed",
                    "notes_path": str((output_dir / song_id / "notes.npz").relative_to(REPO_ROOT)),
                    "harmony_path": str((output_dir / song_id / "harmony.npz").relative_to(REPO_ROOT)),
                    "metadata_path": str((output_dir / song_id / "metadata.json").relative_to(REPO_ROOT)),
                }
                print(f"  skipped: valid schema {DATASET_SCHEMA_VERSION} output ({existing_notes} notes)", flush=True)
            else:
                record = _process_song(path, song_id, source_sha256, output_dir, git_commit)
                count = int(record["number_of_notes"])
                processed += 1
                total_notes += count
                print(f"  processed: {count} notes", flush=True)
        except Exception as exc:
            failed += 1
            LOGGER.exception("Failed to process %s", path.name)
            record = {
                "song_id": song_id,
                "source_filename": path.name,
                "source_relative_path": str(path.relative_to(REPO_ROOT)),
                "source_sha256": source_sha256 or None,
                "number_of_notes": 0,
                "processing_status": "failed",
                "validation_status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"  failed: {type(exc).__name__}: {exc}", flush=True)

        _upsert_manifest_record(manifest, record)
        _save_manifest(output_dir, manifest)

    print()
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Total notes: {total_notes:,}")
    print(f"Output: {output_dir.relative_to(REPO_ROOT)}/")
    return 1 if failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
