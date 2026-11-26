#!/usr/bin/env python3
"""
Build the Axis 2 synthetic pitch-center corruption dataset from canonical notes.

The builder that produced data/ml_dataset/axis2_v1/ was never committed. This
re-implements it from that dataset's README and summary.json, and is checked
by rebuilding axis2_v1 from the v1 notes and comparing the result with the
published files (`--reproduce-check`). Only after that passes is it used to
build axis2_v2 from the re-segmented store.

Per clean note (song_id, note_index), five examples:
  variant 0  clean        shift 0.0, canonical cents contour verbatim
  variant k  corruption_k shift drawn per band (5-15, 15-25, 25-35, 35-50 c),
                          exactly two positive and two negative, applied to
                          every cents frame; confidence, timing, target MIDI
                          unchanged. median/core median recomputed from the
                          shifted contour with production _core_slice.
Shifts come from a per-note RNG seeded by
sha256(f"{global_seed}:{song_id}:{note_index}")[:8], so they do not depend on
iteration order. Splits are song-level and reused verbatim from --splits-from.

--corruption-policy target_preserving (axis2_v3) keeps the target MIDI fixed and
requires every corrupted core median to stay on that target:
-49.999 < N + S < 49.999, N = the clean note's production core median. It reuses
the exact fixed_two_two draws (same magnitude per band, same seeded sign) and
only flips the sign of a shift that would cross the boundary. Magnitudes are
never changed; a note that cannot satisfy a band in either direction is an
error, never a silent fallback. The default policy (fixed_two_two) is the
axis2_v1/axis2_v2 behaviour and is unchanged.

Never modifies data/main_data, the source split file, or any existing dataset:
output is built in a temporary sibling directory, validated, then renamed into
place, and an existing output directory is never overwritten.

Usage:
  build_axis2_ml_dataset.py --dataset-name axis2_v1 --notes-file notes_v1.npz \\
      --out /tmp/axis2_v1_rebuild --reproduce-check data/ml_dataset/axis2_v1
  build_axis2_ml_dataset.py --dataset-name axis2_v2 --notes-file notes.npz \\
      --out data/ml_dataset/axis2_v2
  build_axis2_ml_dataset.py --dataset-name axis2_v3 --notes-file notes.npz \\
      --corruption-policy target_preserving --out data/ml_dataset/axis2_v3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_data_io import REPO_ROOT, _sha256, npz_content_sha256  # noqa: E402

from note_segmentation import _core_slice  # noqa: E402


MAIN_DIR = REPO_ROOT / "data" / "main_data"
SELECTIONS = REPO_ROOT / "data" / "dataset_selections.json"
DEFAULT_SPLITS = REPO_ROOT / "data" / "ml_dataset" / "axis2_v1" / "splits.json"
GLOBAL_SEED = 20260824
BANDS = ((5.0, 15.0), (15.0, 25.0), (25.0, 35.0), (35.0, 50.0))
VARIANTS = ("clean", "corruption_1", "corruption_2", "corruption_3", "corruption_4")
SPLITS = ("train", "validation", "test")
MANIFEST_COLUMNS = ("example_id", "clean_note_id", "song_id", "source_filename", "source_note_index", "split",
                    "variant", "variant_index", "target_midi", "duration_s", "start_s", "end_s", "num_frames",
                    "injected_shift_cents", "median_cents", "core_median_cents")
POLICIES = ("fixed_two_two", "target_preserving")
# A corrupted core median must stay strictly inside this many cents of the
# unchanged target MIDI. Beyond 50 c the sung pitch is nearer the adjacent
# semitone, so the label would no longer describe intonation around the note
# the singer is targeting; the 0.001 c margin keeps float noise off the boundary.
TARGET_PRESERVING_LIMIT_CENTS = 49.999
FEASIBILITY = ("both", "positive_only", "negative_only")


def note_shifts(song_id: str, note_index: int) -> np.ndarray:
    """The four signed corruption shifts of one note, in band order."""
    digest = hashlib.sha256(f"{GLOBAL_SEED}:{song_id}:{note_index}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    magnitudes = rng.uniform([lo for lo, _ in BANDS], [hi for _, hi in BANDS])
    signs = -np.ones(len(BANDS))
    signs[rng.choice(len(BANDS), 2, replace=False)] = 1.0
    return magnitudes * signs


def _on_target(x: float) -> bool:
    return -TARGET_PRESERVING_LIMIT_CENTS < x < TARGET_PRESERVING_LIMIT_CENTS


def target_preserving_shifts(song_id: str, note_index: int, clean_core: float) -> tuple:
    """
    The fixed_two_two draws, with a sign flipped only where it would push the
    core median off the target: keep the drawn sign if N + S stays on target,
    otherwise use -S. Returns (shifts, per-band feasibility, per-band flipped).
    """
    base = note_shifts(song_id, note_index)
    shifts, feas, flipped = np.empty_like(base), [], []
    for b, s in enumerate(base):
        m = abs(float(s))
        pos, neg = _on_target(clean_core + m), _on_target(clean_core - m)
        if not (pos or neg):
            raise SystemExit(f"{song_id}:{note_index} band {VARIANTS[b + 1]}: |S|={m!r} with clean core {clean_core!r} "
                             f"leaves the target in both directions; refusing to change the magnitude")
        feas.append("both" if pos and neg else "positive_only" if pos else "negative_only")
        keep = pos if s > 0 else neg
        shifts[b] = s if keep else -s
        flipped.append(not keep)
    return shifts, feas, flipped


def _structurally_valid(cents: np.ndarray, conf: np.ndarray, midi: int, duration: float) -> bool:
    return (cents.size > 0 and cents.size == conf.size and bool(np.all(np.isfinite(cents)))
            and bool(np.all(np.isfinite(conf))) and bool(np.all((conf >= 0) & (conf <= 1)))
            and 0 <= midi <= 127 and np.isfinite(duration) and duration > 0)


def build(args: argparse.Namespace, out: Path) -> Dict[str, Any]:
    sel = json.loads(SELECTIONS.read_text(encoding="utf-8"))["songs"]
    songs = sorted(s for s, v in sel.items() if v["axis2_ml"])
    split_src = json.loads(args.splits_from.read_text(encoding="utf-8"))
    split_of = {s: k for k in SPLITS for s in split_src[k]}
    if sorted(split_of) != songs or sum(len(split_src[k]) for k in SPLITS) != len(songs):
        raise SystemExit("split file does not partition the axis2_ml songs exactly")

    cols: Dict[str, List] = {k: [] for k in ("song_index", "source_note_index", "variant_index", "split_index",
                                              "target_midi", "duration_s", "start_s", "end_s", "injected_shift_cents",
                                              "median_cents", "core_median_cents", "num_frames")}
    cents_parts, conf_parts, lengths = [], [], []
    rows, filenames = [], []
    note_counts = {k: 0 for k in SPLITS}
    schemas, filtered, source_hashes = set(), 0, {}
    preserving = args.corruption_policy == "target_preserving"
    feas_counts = {VARIANTS[b + 1]: {f: 0 for f in FEASIBILITY} for b in range(len(BANDS))}
    flip_counts = {VARIANTS[b + 1]: 0 for b in range(len(BANDS))}
    for si, song in enumerate(songs):
        sd = args.main_dir / song
        meta = json.loads((sd / ("metadata_v1.json" if args.notes_file == "notes_v1.npz" else "metadata.json"))
                          .read_text(encoding="utf-8"))
        schemas.add(int(meta["dataset_schema_version"]))
        filenames.append(meta["original_filename"])
        npath = sd / args.notes_file
        source_hashes[song] = {"notes_sha256": _sha256(npath), "notes_content_sha256": npz_content_sha256(npath)}
        split = split_of[song]
        with np.load(npath, allow_pickle=False) as n:
            off = n["contour_offsets"]
            for i in range(int(n["note_index"].size)):
                cents = np.asarray(n["contour_cents_deviation"][off[i]:off[i + 1]], dtype=np.float64)
                conf = np.asarray(n["contour_crepe_confidence"][off[i]:off[i + 1]], dtype=np.float64)
                midi, dur = int(n["target_midi_note"][i]), float(n["duration_s"][i])
                if not _structurally_valid(cents, conf, midi, dur):
                    filtered += 1
                    continue
                note_index = int(n["note_index"][i])
                start, end = float(n["start_time_s"][i]), float(n["end_time_s"][i])
                note_counts[split] += 1
                if preserving:
                    clean_core = float(np.median(_core_slice(cents)))
                    note_shift_values, feas, flipped = target_preserving_shifts(song, note_index, clean_core)
                    for b in range(len(BANDS)):
                        feas_counts[VARIANTS[b + 1]][feas[b]] += 1
                        flip_counts[VARIANTS[b + 1]] += int(flipped[b])
                else:
                    note_shift_values = note_shifts(song, note_index)
                for v, shift in enumerate((0.0, *note_shift_values)):
                    c = cents + shift
                    med, core = float(np.median(c)), float(np.median(_core_slice(c)))
                    for k, val in (("song_index", si), ("source_note_index", note_index), ("variant_index", v),
                                   ("split_index", SPLITS.index(split)), ("target_midi", midi),
                                   ("duration_s", dur), ("start_s", start), ("end_s", end),
                                   ("injected_shift_cents", shift), ("median_cents", med),
                                   ("core_median_cents", core), ("num_frames", c.size)):
                        cols[k].append(val)
                    cents_parts.append(c)
                    conf_parts.append(conf)
                    lengths.append(c.size)
                    rows.append((f"{song}:{note_index}:{VARIANTS[v]}", f"{song}:{note_index}", song,
                                 meta["original_filename"], str(note_index), split, VARIANTS[v], str(v), str(midi),
                                 f"{dur:.6f}", f"{start:.6f}", f"{end:.6f}", str(c.size), f"{shift:.6f}",
                                 f"{med:.6f}", f"{core:.6f}"))
        print(f"[{si + 1}/{len(songs)}] {song}", flush=True)
    if len(schemas) != 1:
        raise SystemExit(f"notes come from mixed feature-store schemas {sorted(schemas)}")
    schema = schemas.pop()

    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    arrays = {
        "song_index": np.asarray(cols["song_index"], np.int32),
        "source_note_index": np.asarray(cols["source_note_index"], np.int32),
        "variant_index": np.asarray(cols["variant_index"], np.int8),
        "split_index": np.asarray(cols["split_index"], np.int8),
        "target_midi": np.asarray(cols["target_midi"], np.int16),
        "duration_s": np.asarray(cols["duration_s"], np.float64),
        "start_s": np.asarray(cols["start_s"], np.float64),
        "end_s": np.asarray(cols["end_s"], np.float64),
        "injected_shift_cents": np.asarray(cols["injected_shift_cents"], np.float64),
        "median_cents": np.asarray(cols["median_cents"], np.float64),
        "core_median_cents": np.asarray(cols["core_median_cents"], np.float64),
        "num_frames": np.asarray(cols["num_frames"], np.int32),
        "contour_offsets": offsets,
        "cents_flat": np.concatenate(cents_parts).astype(np.float64),
        "confidence_flat": np.concatenate(conf_parts).astype(np.float64),
        "song_ids": np.asarray(songs, dtype="U120"),
        "song_source_filenames": np.asarray(filenames, dtype="U200"),
        "song_split_index": np.asarray([SPLITS.index(split_of[s]) for s in songs], np.int8),
        "variant_names": np.asarray(VARIANTS, dtype="U16"),
        "split_names": np.asarray(SPLITS, dtype="U12"),
    }
    np.savez_compressed(out / "examples.npz", **arrays)
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MANIFEST_COLUMNS)
        w.writerows(rows)

    n_ex = {k: 5 * note_counts[k] for k in SPLITS}
    splits_out = {"split_strategy": split_src["split_strategy"],
                  "split_locked_before_corruption": split_src["split_locked_before_corruption"]}
    splits_out.update({k: sorted(split_src[k]) for k in SPLITS})
    splits_out.update({f"{k}_song_count": len(split_src[k]) for k in SPLITS})
    splits_out.update({f"{k}_clean_note_count": note_counts[k] for k in SPLITS})
    splits_out.update({f"{k}_example_count": n_ex[k] for k in SPLITS})
    (out / "splits.json").write_text(json.dumps(splits_out, indent=2) + "\n", encoding="utf-8")

    shifts = arrays["injected_shift_cents"][arrays["variant_index"] > 0]
    total_notes = sum(note_counts.values())
    summary = {
        "schema_version": 1,
        "dataset_name": args.dataset_name,
        "created_at_utc": args.created_at or datetime.now(timezone.utc).isoformat(),
        "global_seed": GLOBAL_SEED,
        "seed_derivation": "sha256(f'{global_seed}:{clean_note_id}')[:8] -> np.random.default_rng",
        "source_of_truth": ("data/main_data/ (immutable canonical feature store)" if schema == 1 else
                            f"data/main_data/<song>/{args.notes_file} (canonical feature store, schema {schema})"),
        "membership_source": "data/dataset_selections.json (axis2_ml == true)",
        "song_count": len(songs),
        "clean_note_count": total_notes,
        "variants_per_note": len(VARIANTS),
        "total_example_count": 5 * total_notes,
    }
    summary.update({f"{k}_song_count": len(split_src[k]) for k in SPLITS})
    summary.update({f"{k}_clean_note_count": note_counts[k] for k in SPLITS})
    summary.update({f"{k}_example_count": n_ex[k] for k in SPLITS})
    summary.update({
        "corruption_model": "constant pitch-center displacement applied to every cents frame",
        "corruption_magnitude_bands_cents": {VARIANTS[i + 1]: list(b) for i, b in enumerate(BANDS)},
        "corruption_magnitude_sampling": "continuous uniform within each band, per note",
        "corruption_sign_policy": "exactly 2 positive and 2 negative per clean note; which severity bands receive which sign is randomized per note",
        "label_field": "injected_shift_cents",
        "label_semantics": "signed synthetic pitch-center displacement in cents; clean == 0.0 exactly",
        "clean_definition": "zero synthetic corruption; canonical contour preserved verbatim and NOT recentered",
        "confidence_policy": "CREPE confidence is a continuous model input, never a filtering threshold, and is identical across all five variants of a note",
        "target_midi_policy": "fixed from the clean canonical note; never recomputed after shifting",
        "notes_filtered": filtered,
        "filtering_policy": "no confidence, cents, duration, vibrato, portamento, scoop, or pitch-variation filtering; structural validity only",
        "total_frames": int(arrays["cents_flat"].size),
        "shift_statistics_cents": {
            "corrupted_example_count": int(shifts.size),
            "positive_count": int((shifts > 0).sum()), "negative_count": int((shifts < 0).sum()),
            "min": float(shifts.min()), "max": float(shifts.max()),
            "min_abs": float(np.abs(shifts).min()), "max_abs": float(np.abs(shifts).max()),
            "mean_abs": float(np.abs(shifts).mean()), "median_abs": float(np.median(np.abs(shifts))),
        },
        "scope": "V1 models pitch-center corruption only; no vibrato, timing, or timbral corruption",
    })
    if preserving:
        cores = arrays["core_median_cents"]
        summary.update({
            "corruption_policy": "target_preserving",
            "corruption_sign_policy": (
                "per band, the fixed_two_two draw (same magnitude, same seeded sign); the sign is flipped only when "
                f"clean core + shift would leave (-{TARGET_PRESERVING_LIMIT_CENTS}, +{TARGET_PRESERVING_LIMIT_CENTS}); "
                "magnitudes are never changed, so 2+/2- per note is no longer guaranteed"),
            "target_preserving_constraint": f"-{TARGET_PRESERVING_LIMIT_CENTS} < N + S < {TARGET_PRESERVING_LIMIT_CENTS}, "
                                            "N = clean production core median relative to the unchanged target MIDI",
            "target_preserving_limit_cents": TARGET_PRESERVING_LIMIT_CENTS,
            "sign_feasibility_by_band": feas_counts,
            "sign_flipped_from_fixed_two_two_by_band": flip_counts,
            "max_abs_core_median_cents": float(np.abs(cores).max()),
            "max_abs_core_median_cents_corrupted": float(np.abs(cores[arrays["variant_index"] > 0]).max()),
        })
    if schema >= 2:
        # provenance only a re-segmented source has; absent from axis2_v1 by construction
        seg = json.loads((args.main_dir / songs[0] / "metadata.json").read_text(encoding="utf-8"))
        summary.update({
            "source_feature_store_schema_version": schema,
            "source_notes_file": args.notes_file,
            "segmentation_version": seg["segmentation_version"],
            "segmentation_config": seg["segmentation_config"],
            "split_assignment_source": str(args.splits_from.resolve().relative_to(REPO_ROOT)),
            "source_notes_sha256": source_hashes,
            "builder": "scripts/build_axis2_ml_dataset.py",
        })
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def validate(out: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Re-open the written files and re-derive everything independently."""
    with np.load(out / "examples.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}   # NpzFile decompresses on every access
    n = d["song_index"].size
    off = d["contour_offsets"]
    checks = {}
    checks["lengths_aligned"] = all(d[k].size == n for k in ("source_note_index", "variant_index", "split_index",
                                                            "target_midi", "injected_shift_cents", "num_frames"))
    checks["offsets_valid"] = bool(off[0] == 0 and off[-1] == d["cents_flat"].size == d["confidence_flat"].size
                                   and np.all(np.diff(off) == d["num_frames"]) and np.all(d["num_frames"] > 0))
    checks["five_variants_per_note"] = bool(n % 5 == 0 and np.array_equal(d["variant_index"], np.tile(np.arange(5), n // 5)))
    sh = d["injected_shift_cents"].reshape(-1, 5)
    checks["clean_is_zero"] = bool(np.all(sh[:, 0] == 0.0))
    preserving = args.corruption_policy == "target_preserving"
    if not preserving:
        checks["two_pos_two_neg"] = bool(np.all((sh[:, 1:] > 0).sum(1) == 2) and np.all((sh[:, 1:] < 0).sum(1) == 2))
    mags = np.abs(sh[:, 1:])
    checks["magnitudes_in_bands"] = bool(all(np.all((mags[:, i] >= lo) & (mags[:, i] <= hi)) for i, (lo, hi) in enumerate(BANDS)))
    songs = list(d["song_ids"])
    seed_draws = np.array([note_shifts(songs[s], k) for s, k in zip(d["song_index"][::5], d["source_note_index"][::5])])
    if not preserving:
        checks["shifts_rederived_from_seed"] = bool(np.max(np.abs(seed_draws - sh[:, 1:])) <= 1e-12)
    else:
        # Clean cores recomputed from the stored clean contour (itself checked against the canonical notes below).
        clean_core = np.array([float(np.median(_core_slice(d["cents_flat"][off[r]:off[r + 1]])))
                               for r in range(0, n, 5)])
        rederived = np.array([target_preserving_shifts(songs[s], k, c)[0] for s, k, c
                              in zip(d["song_index"][::5], d["source_note_index"][::5], clean_core)])
        checks["shifts_rederived_from_seed"] = bool(np.max(np.abs(rederived - sh[:, 1:])) <= 1e-12)
        checks["magnitudes_equal_fixed_two_two_draws"] = bool(np.array_equal(mags, np.abs(seed_draws)))
        checks["signs_changed_only_where_required"] = bool(np.all(
            (np.sign(sh[:, 1:]) == np.sign(seed_draws)) | ~np.vectorize(_on_target)(clean_core[:, None] + seed_draws)))
        recomputed = np.array([float(np.median(_core_slice(d["cents_flat"][off[r]:off[r + 1]]))) for r in range(n)])
        checks["core_medians_recomputed"] = bool(np.max(np.abs(recomputed - d["core_median_cents"])) <= 1e-9)
        expected = np.repeat(clean_core, 5) + d["injected_shift_cents"]
        checks["core_equals_clean_core_plus_shift"] = bool(np.max(np.abs(recomputed - expected)) <= 1e-9)
        corrupted = d["variant_index"] > 0
        checks["corrupted_cores_on_target"] = bool(np.all(np.abs(expected[corrupted]) < TARGET_PRESERVING_LIMIT_CENTS)
                                                   and np.all(np.abs(recomputed[corrupted]) < TARGET_PRESERVING_LIMIT_CENTS + 1e-9))
        checks["all_cores_below_50"] = bool(np.all(np.abs(recomputed) < 50.0))
        tm = d["target_midi"].reshape(-1, 5)
        checks["target_midi_identical_across_variants"] = bool(np.all(tm == tm[:, :1]))
    # per-note: clean contour verbatim from canonical notes; corrupted = clean + shift; confidence unchanged
    ok_contour = True
    for si, song in enumerate(songs):
        with np.load(args.main_dir / song / args.notes_file, allow_pickle=False) as z:
            nn = {k: z[k] for k in z.files}
        no = nn["contour_offsets"]
        rows = np.flatnonzero((d["song_index"] == si) & (d["variant_index"] == 0))
        if rows.size != nn["note_index"].size:
            ok_contour = False
            continue
        for r in rows:
            k = int(d["source_note_index"][r])
            canon = nn["contour_cents_deviation"][no[k]:no[k + 1]]
            cconf = nn["contour_crepe_confidence"][no[k]:no[k + 1]]
            for v in range(5):
                lo, hi = off[r + v], off[r + v + 1]
                if not (np.array_equal(d["cents_flat"][lo:hi], canon + sh[r // 5, v])
                        and np.array_equal(d["confidence_flat"][lo:hi], cconf)
                        and d["target_midi"][r + v] == nn["target_midi_note"][k]):
                    ok_contour = False
    checks["contours_match_canonical_notes"] = ok_contour
    sp = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    sets = [set(sp[k]) for k in SPLITS]
    checks["splits_disjoint_song_level"] = bool(not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
                                               and np.array_equal(d["split_index"], d["song_split_index"][d["song_index"]]))
    with (out / "manifest.csv").open(newline="", encoding="utf-8") as fh:
        checks["manifest_rows"] = sum(1 for _ in csv.reader(fh)) - 1 == n
    return checks


def reproduce_check(out: Path, ref: Path) -> Dict[str, Any]:
    res = {}
    a, b = np.load(out / "examples.npz", allow_pickle=False), np.load(ref / "examples.npz", allow_pickle=False)
    res["examples_keys_equal"] = sorted(a.files) == sorted(b.files)
    res["examples_array_mismatches"] = [k for k in b.files if k not in a.files or a[k].dtype != b[k].dtype
                                        or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k])]
    a.close(); b.close()
    res["examples_npz_byte_identical"] = _sha256(out / "examples.npz") == _sha256(ref / "examples.npz")
    res["manifest_csv_byte_identical"] = _sha256(out / "manifest.csv") == _sha256(ref / "manifest.csv")
    res["splits_json_byte_identical"] = _sha256(out / "splits.json") == _sha256(ref / "splits.json")
    s_new = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    s_ref = json.loads((ref / "summary.json").read_text(encoding="utf-8"))
    res["summary_differences_excluding_created_at"] = sorted(
        k for k in set(s_new) | set(s_ref) if k != "created_at_utc" and s_new.get(k) != s_ref.get(k))
    res["passed"] = (not res["examples_array_mismatches"] and res["examples_keys_equal"]
                     and res["manifest_csv_byte_identical"] and res["splits_json_byte_identical"]
                     and not res["summary_differences_excluding_created_at"])
    return res


README_V2 = """# Axis 2 V2 — synthetic pitch-center corruption dataset (re-segmented notes)

Same construction as `../axis2_v1/` (read its README for label semantics, the
clean/corruption definitions, splits, and the leakage caveats; all of it applies
unchanged). Built by `scripts/build_axis2_ml_dataset.py`, the same code that
rebuilds `axis2_v1` exactly from the v1 notes.

**What differs from axis2_v1: only the notes.** Clean notes come from
`data/main_data/<song>/notes.npz` at feature-store schema 2, segmented from the
saved vocal tracks by `note_segmentation_v2` with the canonical config recorded
in `summary.json` (`segmentation_config`). See
`ml/audits/axis2_segmentation_v2/segmentation_v2_summary.md` for why: v1 bridged
unvoiced gaps of any length, leaving 12.1% of notes clearly invalid.

- Song membership: unchanged (`dataset_selections.json`, `axis2_ml == true`, 50 songs).
- Split assignment: reused verbatim from `axis2_v1/splits.json` (song-level,
  vocalist-disjoint, locked before corruption); only note and example counts change.
- Corruption bands, sign policy, seed scheme, and global seed: unchanged. Seeds key
  on `song_id:note_index`, and note indices differ from v1, so individual draws
  differ from v1 even for notes whose contour is identical.
- No note filtering beyond structural validity, exactly as in v1.

Counts are in `summary.json` and `splits.json`; the source notes' hashes are in
`summary.json` (`source_notes_sha256`). The v1 baselines in `ml/results/axis2_v1/`
were computed on axis2_v1 and do not describe this dataset.
"""

README_V3 = """# Axis 2 V3 — target-preserving synthetic pitch-center corruption

Same clean notes, splits, array schema, seed scheme and corruption magnitudes as
`../axis2_v2/` (read its README and `../axis2_v1/README.md` for label semantics
and leakage caveats). Built by `scripts/build_axis2_ml_dataset.py
--corruption-policy target_preserving`.

**What differs from axis2_v2: only the signs of some corruption shifts.**
Axis 2 measures intonation around the note the singer is targeting, and the
target MIDI is never recomputed after shifting. In axis2_v2 a shift could carry
the core median past +-50 c (e.g. clean +30 c, shift +35 c -> +65 c), a pitch
nearer the adjacent semitone than the labelled target. Here every corruption
satisfies

    -49.999 < N + S < 49.999,   N = clean production core median (cents vs target)

Each band reuses axis2_v2's exact draw for that note (same magnitude, same seeded
sign); the sign is flipped only where the drawn direction would leave the target.
Magnitudes are never shrunk, so the 35-50 c band keeps genuine 35-50 c errors.

Consequences:
- 2 positive / 2 negative per note is no longer guaranteed (global balance is in
  `summary.json`, `shift_statistics_cents`).
- The sign of a large corruption is now partly predictable from where the clean
  note sits: a note far sharp of its target can only be corrupted flat in the
  upper bands. This is a property of the target-preserving definition, not a leak
  of the label, but models should be compared on axis2_v3 with that in mind.
- Feasibility and flip counts per band are in `summary.json`
  (`sign_feasibility_by_band`, `sign_flipped_from_fixed_two_two_by_band`).

Results computed on axis2_v1 or axis2_v2 do not describe this dataset.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--notes-file", choices=("notes.npz", "notes_v1.npz"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits-from", type=Path, default=DEFAULT_SPLITS)
    p.add_argument("--main-dir", type=Path, default=MAIN_DIR,
                   help="feature store to read notes from (a staging directory before promotion)")
    p.add_argument("--corruption-policy", choices=POLICIES, default="fixed_two_two",
                   help="fixed_two_two: axis2_v1/v2 (2+/2- per note); target_preserving: axis2_v3 (|N+S| < 49.999)")
    p.add_argument("--created-at", help="fixed ISO timestamp for reproducible output")
    p.add_argument("--reproduce-check", type=Path, help="compare the build with this existing dataset")
    args = p.parse_args()

    args.main_dir = args.main_dir.resolve()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to overwrite a dataset")
    tmp = out.parent / f".{out.name}.building-{os.getpid()}"
    tmp.mkdir(parents=True)
    try:
        build(args, tmp)
        if args.corruption_policy == "target_preserving":
            (tmp / "README.md").write_text(README_V3, encoding="utf-8")
        elif args.dataset_name != "axis2_v1":
            (tmp / "README.md").write_text(README_V2, encoding="utf-8")
        checks = validate(tmp, args)
        report = {"validation": checks, "validation_passed": all(checks.values())}
        if args.reproduce_check:
            report["reproduce_check"] = reproduce_check(tmp, args.reproduce_check.resolve())
        print(json.dumps(report, indent=1))
        if not report["validation_passed"] or (args.reproduce_check and not report["reproduce_check"]["passed"]):
            raise SystemExit("dataset checks FAILED; nothing was moved into place")
        (tmp / "_build_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
