#!/usr/bin/env python3
"""
Re-segment the canonical feature store from its saved vocal tracks.

The source audio no longer exists. Every input segmentation needs is already in
data/main_data/<song>/harmony.npz (vocal_frame_times_s, vocal_f0_hz with NaN =
unvoiced, vocal_crepe_confidence, tuning_offset_semitones), and v1 notes were
shown to be exactly reproducible from it. This script never reads audio and
never runs Demucs or CREPE, and it never writes harmony.npz.

Subcommands, in migration order:

  stage      Segment every manifest song into a staging directory (notes.npz +
             metadata.json per song, plus manifest.json) and validate it.
             `--segmentation v2` builds the schema-2 store with CANONICAL_CONFIG;
             `--segmentation v1` rebuilds v1 notes for the reproduction check.
  compare-v1 Check a v1 staging directory against the canonical v1 notes
             (notes_v1.npz once migrated, notes.npz before) array-for-array
             and byte-for-byte.
  backup     Copy data/main_data to a new backup directory, hash every file,
             re-read and re-hash the copy, and record protected-file hashes.
  promote    Move a validated v2 staging directory into data/main_data:
             notes.npz -> notes_v1.npz and metadata.json -> metadata_v1.json
             (verified against the backup), then staged files via os.replace,
             then the manifest last. Resumable: songs already promoted with the
             same content are verified and skipped.
  validate   Post-migration checks over the whole store.

Determinism: segmentation is a pure function of harmony.npz and the config,
np.savez stamps zip entries with a fixed date, and JSON is written with
sorted keys, so two `stage` runs with the same --timestamp are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_data_io import (  # noqa: E402
    REPO_ROOT,
    ValidationError,
    _atomic_json,
    _atomic_npz,
    _git_commit,
    _note_arrays,
    _sha256,
    npz_content_sha256,
    validate_song_output,
)

from note_segmentation import segment_notes_with_frame_indices  # noqa: E402
from note_segmentation_v2 import CANONICAL_CONFIG, segment_notes_v2_with_frame_indices  # noqa: E402


MAIN_DIR = REPO_ROOT / "data" / "main_data"
SELECTIONS = REPO_ROOT / "data" / "dataset_selections.json"
ML_V1_DIR = REPO_ROOT / "data" / "ml_dataset" / "axis2_v1"
SCHEMA_VERSION = 2
SEGMENTATION_ALGORITHM = "backend/note_segmentation_v2.py:segment_notes_v2_with_frame_indices"
SEGMENTATION_SOURCE = "harmony.npz: vocal_frame_times_s, vocal_f0_hz, vocal_crepe_confidence, tuning_offset_semitones"
CODE_FILES = ("backend/note_segmentation.py", "backend/note_segmentation_v2.py",
              "scripts/main_data_io.py", "scripts/resegment_main_dataset.py")
# Files a promote/validate run checks for byte-identity against the backup.
PROTECTED_PER_SONG = ("harmony.npz",)


# ---------------------------------------------------------------- helpers


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_songs(main_dir: Path) -> List[str]:
    songs = [r["song_id"] for r in _load_json(main_dir / "manifest.json")["songs"]]
    if len(songs) != len(set(songs)):
        raise ValidationError("manifest.json lists a song twice")
    return sorted(songs)


def _v1_paths(song_dir: Path) -> Tuple[Path, Path]:
    """The v1 notes/metadata: the preserved sidecars once migrated, the live files before."""
    if (song_dir / "notes_v1.npz").exists():
        return song_dir / "notes_v1.npz", song_dir / "metadata_v1.json"
    meta = _load_json(song_dir / "metadata.json")
    if meta["dataset_schema_version"] != 1:
        raise ValidationError(f"{song_dir.name}: schema {meta['dataset_schema_version']} but no notes_v1.npz")
    return song_dir / "notes.npz", song_dir / "metadata.json"


def _track(harmony_path: Path) -> Tuple[SimpleNamespace, float]:
    with np.load(harmony_path, allow_pickle=False) as h:
        track = SimpleNamespace(
            time=np.asarray(h["vocal_frame_times_s"], dtype=np.float64),
            frequency=np.asarray(h["vocal_f0_hz"], dtype=np.float64),
            confidence=np.asarray(h["vocal_crepe_confidence"], dtype=np.float64),
        )
        tuning = float(h["tuning_offset_semitones"])
        mask = np.asarray(h["vocal_voiced_mask"], dtype=bool)
    # NaN is the unvoiced marker every consumer tests; the explicit mask is
    # stored redundantly, so a disagreement means the track is not what it claims.
    if not np.array_equal(mask, np.isfinite(track.frequency)):
        raise ValidationError(f"{harmony_path}: vocal_voiced_mask disagrees with NaN in vocal_f0_hz")
    return track, tuning


def segment(harmony_path: Path, version: str) -> Tuple[List[Dict], List[np.ndarray], SimpleNamespace, float]:
    track, tuning = _track(harmony_path)
    if version == "v1":
        segs, idx = segment_notes_with_frame_indices(track.time, track.frequency, tuning)
    elif version == "v2":
        segs, idx = segment_notes_v2_with_frame_indices(track.time, track.frequency, tuning, CANONICAL_CONFIG)
    else:
        raise ValueError(version)
    if not segs:
        raise ValidationError(f"{harmony_path.parent.name}: segmentation produced no notes")
    return segs, idx, track, tuning


def _git_state() -> Dict[str, Any]:
    commit = _git_commit()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_FILES], cwd=REPO_ROOT,
                           capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": commit, "code_dirty": bool(dirty)}


def _code_hashes() -> Dict[str, str]:
    return {f: _sha256(REPO_ROOT / f) for f in CODE_FILES}


def _rel(p: Path) -> str:
    """Repo-relative path when inside the repo (as the manifest records), absolute otherwise."""
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _copy_atomic(src: Path, dst: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=dst.parent, prefix=dst.name + ".", suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    shutil.copy2(src, tmp_path)
    os.replace(tmp_path, dst)


# ---------------------------------------------------------------- stage


def stage(args: argparse.Namespace) -> int:
    main_dir, out = args.main_dir.resolve(), args.staging.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; staging never overwrites. Choose a new directory.")
    git = _git_state()
    if args.segmentation == "v2" and (git["code_dirty"] or not git["commit"]) and not args.allow_dirty:
        raise SystemExit("Segmentation code has uncommitted changes, so provenance would not point at real "
                         "code. Commit first, or pass --allow-dirty for a throwaway staging run.")
    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    songs = _manifest_songs(main_dir)
    out.mkdir(parents=True)
    report: Dict[str, Any] = {"segmentation": args.segmentation, "main_dir": str(main_dir), "timestamp_utc": timestamp,
                              "git": git, "code_sha256": _code_hashes(), "songs": {}}
    if args.segmentation == "v2":
        report["segmentation_config"] = asdict(CANONICAL_CONFIG)

    manifest = _load_json(main_dir / "manifest.json")
    records = {r["song_id"]: dict(r) for r in manifest["songs"]}
    for i, song in enumerate(songs, 1):
        song_dir = main_dir / song
        harmony = song_dir / "harmony.npz"
        segs, idx, track, tuning = segment(harmony, args.segmentation)
        notes = _note_arrays(segs, idx, track, tuning)
        dst = out / song
        _atomic_npz(dst / "notes.npz", notes)
        content = npz_content_sha256(notes)
        entry = {"number_of_notes": len(segs), "notes_npz_content_sha256": content,
                 "harmony_npz_sha256": _sha256(harmony)}

        if args.segmentation == "v2":
            v1_notes, v1_meta = _v1_paths(song_dir)
            base = _load_json(v1_meta)
            if base["dataset_schema_version"] != 1:
                raise ValidationError(f"{song}: v1 metadata is not schema 1")
            if abs(float(base["tuning_offset_semitones"]) - tuning) > 1e-12:
                raise ValidationError(f"{song}: tuning in metadata and harmony.npz disagree")
            meta = dict(base)
            meta.update({
                "dataset_schema_version": SCHEMA_VERSION,
                "number_of_segmented_notes": len(segs),
                "segmentation_version": "v2",
                "segmentation_algorithm": SEGMENTATION_ALGORITHM,
                "segmentation_config": asdict(CANONICAL_CONFIG),
                "segmentation_source": SEGMENTATION_SOURCE,
                "segmentation_code_sha256": report["code_sha256"],
                "harmony_npz_sha256": entry["harmony_npz_sha256"],
                "notes_npz_content_sha256": content,
                "v1_notes_path": "notes_v1.npz",
                "v1_metadata_path": "metadata_v1.json",
                "v1_notes_sha256": _sha256(v1_notes),
                "v1_notes_content_sha256": npz_content_sha256(v1_notes),
                "v1_number_of_segmented_notes": int(base["number_of_segmented_notes"]),
                "resegmentation_git_commit": git["commit"],
                "resegmentation_timestamp_utc": timestamp,
            })
            _atomic_json(dst / "metadata.json", meta)
            validate_song_output(dst, accepted_schema_versions=(SCHEMA_VERSION,), harmony_path=harmony)
            entry.update(v1_number_of_notes=meta["v1_number_of_segmented_notes"],
                         v1_notes_sha256=meta["v1_notes_sha256"],
                         metadata_sha256=_sha256(dst / "metadata.json"))
            rec = records[song]
            rec.update(number_of_notes=len(segs), validation_status="passed", segmentation_version="v2",
                       notes_v1_path=_rel(song_dir / "notes_v1.npz"),
                       metadata_v1_path=_rel(song_dir / "metadata_v1.json"),
                       resegmented_at_utc=timestamp)
        entry["notes_sha256"] = _sha256(dst / "notes.npz")
        report["songs"][song] = entry
        print(f"[{i}/{len(songs)}] {song}: {len(segs)} notes", flush=True)

    if args.segmentation == "v2":
        manifest = dict(manifest)
        manifest.update(dataset_schema_version=SCHEMA_VERSION, updated_at_utc=timestamp,
                        segmentation_version="v2", segmentation_config=asdict(CANONICAL_CONFIG),
                        songs=[records[s] for s in songs])
        _atomic_json(out / "manifest.json", manifest)
        report["manifest_sha256"] = _sha256(out / "manifest.json")
    report["total_notes"] = sum(e["number_of_notes"] for e in report["songs"].values())
    _atomic_json(out / "_stage_report.json", report)
    print(f"staged {len(songs)} songs, {report['total_notes']:,} notes -> {out}")
    return 0


# ---------------------------------------------------------------- compare-v1


def compare_v1(args: argparse.Namespace) -> int:
    main_dir, staging = args.main_dir.resolve(), args.staging.resolve()
    rep = _load_json(staging / "_stage_report.json")
    if rep["segmentation"] != "v1":
        raise SystemExit("compare-v1 needs a --segmentation v1 staging directory")
    songs = _manifest_songs(main_dir)
    result = {"songs": len(songs), "notes": 0, "array_mismatches": [], "byte_identical": 0, "byte_different": []}
    for song in songs:
        canon, _ = _v1_paths(main_dir / song)
        staged = staging / song / "notes.npz"
        with np.load(canon, allow_pickle=False) as a, np.load(staged, allow_pickle=False) as b:
            if sorted(a.files) != sorted(b.files):
                result["array_mismatches"].append(f"{song}: keys differ")
                continue
            for k in a.files:
                if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
                    result["array_mismatches"].append(f"{song}:{k}")
            result["notes"] += int(a["note_index"].size)
        if _sha256(canon) == _sha256(staged):
            result["byte_identical"] += 1
        else:
            result["byte_different"].append(song)
    print(json.dumps(result, indent=1))
    ok = not result["array_mismatches"]
    print("V1 REPRODUCED EXACTLY" if ok else "V1 MISMATCH")
    return 0 if ok else 1


# ---------------------------------------------------------------- backup


def _tree_hashes(root: Path) -> Dict[str, str]:
    return {str(p.relative_to(root)): _sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _protected_hashes(main_dir: Path) -> Dict[str, str]:
    out = {"data/dataset_selections.json": _sha256(SELECTIONS)}
    for p in sorted(ML_V1_DIR.rglob("*")):
        if p.is_file():
            out[_rel(p)] = _sha256(p)
    for song in _manifest_songs(main_dir):
        for name in PROTECTED_PER_SONG:
            p = main_dir / song / name
            out[_rel(p)] = _sha256(p)
    return out


def backup(args: argparse.Namespace) -> int:
    main_dir, dest = args.main_dir.resolve(), args.backup_dir.resolve()
    if dest.exists():
        raise SystemExit(f"{dest} already exists; refusing to write into an existing backup")
    original = _tree_hashes(main_dir)
    shutil.copytree(main_dir, dest, copy_function=shutil.copy2)
    copied = _tree_hashes(dest)
    if copied != original:
        missing = sorted(set(original) - set(copied))
        changed = sorted(k for k in original if k in copied and copied[k] != original[k])
        raise SystemExit(f"backup verification FAILED: missing {missing[:5]}, changed {changed[:5]}")
    # readable, not just byte-equal: open every npz and json in the copy
    for rel in copied:
        p = dest / rel
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as d:
                for k in d.files:
                    d[k]
        elif p.suffix == ".json":
            _load_json(p)
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(main_dir),
              "file_count": len(copied), "files_sha256": copied, "verified": True,
              "protected_sha256": _protected_hashes(main_dir)}
    _atomic_json(dest.parent / f"{dest.name}.backup_manifest.json", record)
    print(f"backup verified: {len(copied)} files -> {dest}")
    print(f"record: {dest.parent / (dest.name + '.backup_manifest.json')}")
    return 0


# ---------------------------------------------------------------- promote


def promote(args: argparse.Namespace) -> int:
    main_dir, staging, bdir = args.main_dir.resolve(), args.staging.resolve(), args.backup_dir.resolve()
    rep = _load_json(staging / "_stage_report.json")
    brec_path = bdir.parent / f"{bdir.name}.backup_manifest.json"
    brec = _load_json(brec_path)
    if rep["segmentation"] != "v2" or not brec.get("verified"):
        raise SystemExit("promote needs a v2 staging directory and a verified backup")
    songs = _manifest_songs(main_dir)
    if sorted(rep["songs"]) != songs:
        raise SystemExit("staged song set differs from the manifest")
    if _git_state()["commit"] != rep["git"]["commit"]:
        raise SystemExit("HEAD moved since staging; re-stage so provenance names the code being promoted")
    # the backup must still be a backup of what is on disk now
    bh = brec["files_sha256"]
    for f, h in brec["protected_sha256"].items():
        if _sha256(REPO_ROOT / f) != h:
            raise SystemExit(f"protected file changed since backup: {f}")

    for i, song in enumerate(songs, 1):
        sd, st, entry = main_dir / song, staging / song, rep["songs"][song]
        validate_song_output(st, accepted_schema_versions=(SCHEMA_VERSION,), harmony_path=sd / "harmony.npz")
        meta_now = _load_json(sd / "metadata.json")
        if (meta_now["dataset_schema_version"] == SCHEMA_VERSION and (sd / "notes_v1.npz").exists()
                and _sha256(sd / "notes.npz") == entry["notes_sha256"]
                and _sha256(sd / "metadata.json") == entry["metadata_sha256"]):
            print(f"[{i}/{len(songs)}] {song}: already promoted, verified", flush=True)
            continue
        # preserve v1 exactly once; a sidecar from an interrupted run must match the backup
        for live, side in (("notes.npz", "notes_v1.npz"), ("metadata.json", "metadata_v1.json")):
            want = bh[f"{song}/{live}"]
            if (sd / side).exists():
                if _sha256(sd / side) != want:
                    raise SystemExit(f"{song}/{side} exists but does not match the backup; stopping")
            else:
                if _sha256(sd / live) != want:
                    raise SystemExit(f"{song}/{live} changed since the backup; stopping")
                _copy_atomic(sd / live, sd / side)
                if _sha256(sd / side) != want:
                    raise SystemExit(f"{song}/{side} copy verification failed; stopping")
        # notes first, metadata second: a song reads as schema 2 only once its notes are in place
        _copy_atomic(st / "notes.npz", sd / "notes.npz")
        _copy_atomic(st / "metadata.json", sd / "metadata.json")
        validate_song_output(sd, accepted_schema_versions=(SCHEMA_VERSION,))
        print(f"[{i}/{len(songs)}] {song}: promoted ({entry['v1_number_of_notes']} -> {entry['number_of_notes']} notes)", flush=True)

    _copy_atomic(staging / "manifest.json", main_dir / "manifest.json")
    print("manifest promoted")
    return validate(args)


# ---------------------------------------------------------------- validate


def validate(args: argparse.Namespace) -> int:
    main_dir = args.main_dir.resolve()
    manifest = _load_json(main_dir / "manifest.json")
    songs = _manifest_songs(main_dir)
    dirs = sorted(p.name for p in main_dir.iterdir() if p.is_dir())
    checks: Dict[str, Any] = {}
    fails: List[str] = []

    def check(name, ok, detail=None):
        checks[name] = {"passed": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)

    check("manifest_schema_2", manifest.get("dataset_schema_version") == SCHEMA_VERSION, manifest.get("dataset_schema_version"))
    check("manifest_config_canonical", manifest.get("segmentation_config") == asdict(CANONICAL_CONFIG))
    check("song_dirs_match_manifest", dirs == songs, {"dirs": len(dirs), "manifest": len(songs)})
    sel = _load_json(SELECTIONS)["songs"]
    check("songs_match_selections", sorted(sel) == songs, len(sel))

    recs = {r["song_id"]: r for r in manifest["songs"]}
    total = total_v1 = 0
    per_song_fail = []
    for song in songs:
        sd = main_dir / song
        try:
            res = validate_song_output(sd, accepted_schema_versions=(SCHEMA_VERSION,))
            meta = _load_json(sd / "metadata.json")
            n = res["note_count"]
            assert recs[song]["number_of_notes"] == n == meta["number_of_segmented_notes"], "counts disagree"
            assert meta["segmentation_config"] == asdict(CANONICAL_CONFIG), "config differs"
            assert meta["segmentation_version"] == "v2" and meta["segmentation_algorithm"] == SEGMENTATION_ALGORITHM
            assert subprocess.run(["git", "cat-file", "-e", f"{meta['resegmentation_git_commit']}^{{commit}}"],
                                  cwd=REPO_ROOT).returncode == 0, "provenance commit not in repo"
            # v1 sidecars present and exactly what the metadata says they are
            assert _sha256(sd / "notes_v1.npz") == meta["v1_notes_sha256"], "notes_v1 hash"
            assert npz_content_sha256(sd / "notes_v1.npz") == meta["v1_notes_content_sha256"], "notes_v1 content"
            v1meta = _load_json(sd / "metadata_v1.json")
            assert v1meta["dataset_schema_version"] == 1, "metadata_v1 schema"
            assert v1meta["number_of_segmented_notes"] == meta["v1_number_of_segmented_notes"], "v1 count"
            with np.load(sd / "notes_v1.npz", allow_pickle=False) as v:
                assert v["note_index"].size == meta["v1_number_of_segmented_notes"], "v1 notes length"
            # provenance is accurate: re-segmenting harmony.npz now reproduces the stored notes exactly
            segs, idx, track, tuning = segment(sd / "harmony.npz", "v2")
            assert npz_content_sha256(_note_arrays(segs, idx, track, tuning)) == meta["notes_npz_content_sha256"], \
                "re-segmentation differs from stored notes"
            # and the v1 sidecar is still what v1 segmentation produces
            segs1, idx1, _, _ = segment(sd / "harmony.npz", "v1")
            assert npz_content_sha256(_note_arrays(segs1, idx1, track, tuning)) == meta["v1_notes_content_sha256"], \
                "v1 re-segmentation differs from notes_v1.npz"
            total += n
            total_v1 += meta["v1_number_of_segmented_notes"]
        except (AssertionError, ValidationError, OSError, KeyError) as exc:
            per_song_fail.append(f"{song}: {exc}")
    check("every_song_valid_and_consistent", not per_song_fail, per_song_fail[:10])
    check("totals", True, {"v2_notes": total, "v1_notes": total_v1})

    if args.backup_dir:
        bdir = args.backup_dir.resolve()
        brec = _load_json(bdir.parent / f"{bdir.name}.backup_manifest.json")
        changed = [f for f, h in brec["protected_sha256"].items() if _sha256(REPO_ROOT / f) != h]
        check("protected_files_byte_identical", not changed, {"checked": len(brec["protected_sha256"]), "changed": changed})
        bh = brec["files_sha256"]
        moved = [s for s in songs if _sha256(main_dir / s / "notes_v1.npz") != bh[f"{s}/notes.npz"]
                 or _sha256(main_dir / s / "metadata_v1.json") != bh[f"{s}/metadata.json"]]
        check("v1_sidecars_equal_backup", not moved, moved[:5])
        check("backup_still_intact", _tree_hashes(bdir) == bh)

    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "checks": checks, "all_passed": not fails}
    print(json.dumps(report, indent=1))
    if args.report:
        _atomic_json(args.report.resolve(), report)
    return 0 if not fails else 1


# ---------------------------------------------------------------- cli


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--main-dir", type=Path, default=MAIN_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--segmentation", choices=("v1", "v2"), required=True)
    s.add_argument("--staging", type=Path, required=True)
    s.add_argument("--timestamp", help="fixed ISO timestamp, for byte-reproducible staging")
    s.add_argument("--allow-dirty", action="store_true")
    c = sub.add_parser("compare-v1")
    c.add_argument("--staging", type=Path, required=True)
    b = sub.add_parser("backup")
    b.add_argument("--backup-dir", type=Path, required=True)
    pr = sub.add_parser("promote")
    pr.add_argument("--staging", type=Path, required=True)
    pr.add_argument("--backup-dir", type=Path, required=True)
    pr.add_argument("--report", type=Path)
    v = sub.add_parser("validate")
    v.add_argument("--backup-dir", type=Path)
    v.add_argument("--report", type=Path)
    args = p.parse_args()
    return {"stage": stage, "compare-v1": compare_v1, "backup": backup, "promote": promote, "validate": validate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
