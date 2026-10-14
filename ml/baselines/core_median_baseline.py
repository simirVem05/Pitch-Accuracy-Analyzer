#!/usr/bin/env python3
"""
Core-median baseline (Baseline 3) for Axis 2: predict the production core median
of each example's own cents contour.

    prediction = median(_core_slice(own cents contour))

`_core_slice` is imported from backend/note_segmentation.py, the same helper
production uses for `core_median_cents` and for choosing a note's target MIDI.
It is index-based over the note's voiced frames: notes with fewer than
MIN_VOICED_FOR_CORE frames are used whole; otherwise floor(CORE_TRIM_FRACTION * n)
frames are dropped from each end (the middle ~60%). Nothing is re-implemented.

Like Baseline 2, this assumes a professional reference note is naturally centered
at 0 cents; what survives is the natural offset of the note's core. The only
difference from Baseline 2 is which frames enter the median, so the comparison
isolates the value of trimming onsets and releases.

Anti-leakage is structural, exactly as in median_cents_baseline.py:
`_predict_core_median_from_own_contour` is handed only the flat contour store and
the indices to score. Sibling grouping is built afterwards, only to verify the
constant-shift identity.

The test split is sealed: no test contour is read for prediction and no test
metric is computed.

  python core_median_baseline.py                              # axis2_v2 (default)
  python core_median_baseline.py --dataset-version axis2_v1
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from median_cents_baseline import (
    GROUP_TOLERANCE_CENTS,
    _by_song,
    _evaluate,
    _group_sanity_checks,
    _predict_from_own_contour,
    _song_extremes,
    _variant_identity_check,
)
from zero_baseline import (
    DATASET_VERSIONS,
    EXPECTED_SPLIT_ORDER,
    EXPECTED_VARIANT_ORDER,
    REPO_ROOT,
    DatasetContractError,
    _metrics,
    _resolve_lookup,
    _tail,
    _verify_counts,
    dataset_paths,
)

sys.path.insert(0, str(REPO_ROOT / "backend"))
import note_segmentation  # noqa: E402
from note_segmentation import CORE_TRIM_FRACTION, MIN_VOICED_FOR_CORE, _core_slice  # noqa: E402


DEFAULT_DATASET_VERSION = "axis2_v2"
EVALUATED_SPLITS = ("train", "validation")
SEALED_SPLIT = "test"
STORED_CORE_TOLERANCE_CENTS = 1e-9
IDENTICAL_TOLERANCE_CENTS = 1e-9

# Frame-count buckets. 9 | 10 is MIN_VOICED_FOR_CORE: below it the core is the whole contour.
FRAME_BUCKETS = ((3, 4), (5, 6), (7, 9), (10, 20), (21, 50), (51, None))
DURATION_BUCKETS_S = ((None, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.80), (0.80, None))
DIFF_THRESHOLDS_CENTS = (5.0, 10.0, 20.0, 50.0)
EXTREME_COUNT = 20


def _predict_core_median_from_own_contour(
    cents_flat: np.ndarray,
    contour_offsets: np.ndarray,
    example_indices: np.ndarray,
) -> np.ndarray:
    """
    Production core median of each example's own cents contour.

    Same deliberately narrow signature as the full-median predictor: no targets,
    no variant codes, no sibling index.
    """
    predictions = np.empty(example_indices.size, dtype=np.float64)
    for position, index in enumerate(example_indices):
        start = int(contour_offsets[index])
        end = int(contour_offsets[index + 1])
        predictions[position] = np.median(_core_slice(cents_flat[start:end]))
    return predictions


def _core_definition() -> Dict[str, Any]:
    return {
        "helper": "backend/note_segmentation.py:_core_slice",
        "helper_source": inspect.getsource(_core_slice),
        "min_voiced_for_core": MIN_VOICED_FOR_CORE,
        "core_trim_fraction": CORE_TRIM_FRACTION,
        "rule": (f"index-based over the note's voiced frames (gaps already removed): if n < {MIN_VOICED_FOR_CORE} "
                 f"the whole contour; else drop floor({CORE_TRIM_FRACTION} * n) frames from each end, keeping "
                 f"the middle ~{100 * (1 - 2 * CORE_TRIM_FRACTION):.0f}% (fallback to whole if fewer than 2 frames "
                 "would remain, which cannot happen for n >= 10)"),
        "trim_is_noop_when": f"num_frames < {MIN_VOICED_FOR_CORE}",
        "production_uses": ["notes.npz core_median_cents_deviation", "target MIDI selection in _finalize_segment",
                            "examples.npz core_median_cents"],
        "module_file": str(Path(note_segmentation.__file__).resolve().relative_to(REPO_ROOT)),
    }


def _verify_stored(predictions: np.ndarray, stored: np.ndarray) -> Dict[str, Any]:
    diff = np.abs(predictions - stored)
    out = {
        "compared_example_count": int(diff.size),
        "max_abs_difference_cents": float(diff.max()) if diff.size else 0.0,
        "median_abs_difference_cents": float(np.median(diff)) if diff.size else 0.0,
        "mismatches_above_tolerance": int(np.count_nonzero(diff > STORED_CORE_TOLERANCE_CENTS)),
        "tolerance_cents": STORED_CORE_TOLERANCE_CENTS,
    }
    out["matches_stored_core_median_cents"] = out["mismatches_above_tolerance"] == 0
    if not out["matches_stored_core_median_cents"]:
        raise DatasetContractError(f"Recomputed core medians disagree with stored core_median_cents: {out}")
    return out


def _in(value: np.ndarray, lo, hi, upper_inclusive: bool) -> np.ndarray:
    m = np.ones(value.shape, dtype=bool)
    if lo is not None:
        m &= value >= lo
    if hi is not None:
        m &= (value <= hi) if upper_inclusive else (value < hi)
    return m


def _pair_metrics(full_err: np.ndarray, core_err: np.ndarray) -> Dict[str, Any]:
    """Full vs core on the same examples. Errors are signed (prediction - target)."""
    n = int(full_err.size)
    if not n:
        return {"note_count": 0}
    f_mae, c_mae = float(np.mean(np.abs(full_err))), float(np.mean(np.abs(core_err)))
    f_rmse, c_rmse = float(np.sqrt(np.mean(full_err ** 2))), float(np.sqrt(np.mean(core_err ** 2)))
    d = np.abs(core_err - full_err)
    return {
        "note_count": n,
        "full_median_mae": f_mae, "core_median_mae": c_mae,
        "mae_improvement": f_mae - c_mae, "percent_mae_improvement": 100.0 * (f_mae - c_mae) / f_mae if f_mae else None,
        "full_median_rmse": f_rmse, "core_median_rmse": c_rmse,
        "rmse_improvement": f_rmse - c_rmse, "percent_rmse_improvement": 100.0 * (f_rmse - c_rmse) / f_rmse if f_rmse else None,
        "fraction_full_equals_core": float(np.mean(d <= IDENTICAL_TOLERANCE_CENTS)),
        "median_abs_core_minus_full_cents": float(np.median(d)),
    }


def _improvement(old: float, new: float) -> Dict[str, Any]:
    return {"full_median": old, "core_median": new, "absolute_improvement": old - new,
            "percent_improvement": 100.0 * (old - new) / old if old else None}


def compute(version: str, zero_results: Mapping[str, Any], full_results: Mapping[str, Any],
            result_paths: Mapping[str, str]) -> Dict[str, Any]:
    paths = dataset_paths(version)
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    for label, res in (("zero", zero_results), ("full-median", full_results)):
        if res["dataset"]["dataset_name"] != version or res.get("test_evaluated") is not False:
            raise DatasetContractError(f"{label} results are not a sealed-test run on {version}")

    with np.load(paths["examples"], allow_pickle=False) as data:
        split_codes = _resolve_lookup(data["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
        variant_codes = _resolve_lookup(data["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")
        split_index = np.asarray(data["split_index"])
        variant_index = np.asarray(data["variant_index"])
        song_index = np.asarray(data["song_index"])
        source_note_index = np.asarray(data["source_note_index"])
        song_ids = np.asarray(data["song_ids"])
        target_all = np.asarray(data["injected_shift_cents"], dtype=np.float64)
        stored_core_all = np.asarray(data["core_median_cents"], dtype=np.float64)
        num_frames_all = np.asarray(data["num_frames"], dtype=np.int64)
        duration_all = np.asarray(data["duration_s"], dtype=np.float64)
        cents_flat = np.asarray(data["cents_flat"], dtype=np.float64)
        contour_offsets = np.asarray(data["contour_offsets"], dtype=np.int64)

    _verify_counts(split_index, split_codes, splits)
    if summary.get("dataset_name") != version:
        raise DatasetContractError(f"{paths['summary']} names {summary.get('dataset_name')!r}, not {version!r}")
    if not np.array_equal(np.diff(contour_offsets), num_frames_all):
        raise DatasetContractError("num_frames disagrees with contour_offsets")

    # Test indices are never placed in this set, so no test contour is read.
    evaluated_indices = np.flatnonzero(np.isin(split_index, [split_codes[s] for s in EVALUATED_SPLITS]))

    predictions = _predict_core_median_from_own_contour(cents_flat, contour_offsets, evaluated_indices)
    targets = target_all[evaluated_indices]
    errors = predictions - targets
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise DatasetContractError("Predictions or targets contain non-finite values")
    stored_check = _verify_stored(predictions, stored_core_all[evaluated_indices])

    # Full-median predictions on the same examples, for note-level comparison only.
    # Same function Baseline 2 uses; its aggregate must reproduce the saved Baseline 2 results.
    full_pred = _predict_from_own_contour(cents_flat, contour_offsets, evaluated_indices)
    full_err = full_pred - targets

    ev_split, ev_var = split_index[evaluated_indices], variant_index[evaluated_indices]
    ev_song, ev_note = song_index[evaluated_indices], source_note_index[evaluated_indices]
    ev_frames, ev_dur = num_frames_all[evaluated_indices], duration_all[evaluated_indices]

    full_repro = {}
    for s in EVALUATED_SPLITS:
        mine = _metrics(targets[ev_split == split_codes[s]], full_pred[ev_split == split_codes[s]])
        saved = full_results[s]["overall"]
        full_repro[s] = {k: {"recomputed": mine[k], "saved": saved[k]} for k in mine}
        if any(not np.isclose(mine[k], saved[k], rtol=1e-12, atol=1e-12) for k in mine):
            raise DatasetContractError(f"full-median recomputation does not reproduce saved Baseline 2 on {s}")

    results: Dict[str, Any] = {
        "experiment": f"{version}_core_median_baseline",
        "prediction_rule": "production_core_median_of_own_cents_contour",
        "features_used": ["cents_contour"],
        "test_evaluated": False,
        "dataset": {
            "path": str(paths["examples"].relative_to(REPO_ROOT)),
            "dataset_name": summary.get("dataset_name"),
            "schema_version": summary.get("schema_version"),
            "label_field": summary.get("label_field"),
            "split_encoding": split_codes,
            "variant_encoding": variant_codes,
            "sealed_split": SEALED_SPLIT,
            "sealed_split_metadata_only": {
                "example_count": int(splits[f"{SEALED_SPLIT}_example_count"]),
                "clean_note_count": int(splits[f"{SEALED_SPLIT}_clean_note_count"]),
            },
        },
        "core_definition": _core_definition(),
        "verification": {
            "prediction_inputs": ["own_example_cents_contour"],
            "sibling_variants_used_for_prediction": False,
            "clean_reference_used_for_prediction": False,
            "song_or_artist_used_for_prediction": False,
            "harmonic_fit_or_chroma_used": False,
            "predicted_example_count": int(evaluated_indices.size),
            "predictions_all_finite": True,
            "targets_all_finite": True,
            "independent_core_median_vs_stored_core_median_cents": stored_check,
            "full_median_recomputation_matches_saved_baseline_2": full_repro,
        },
    }

    for s in EVALUATED_SPLITS:
        m = ev_split == split_codes[s]
        section = _evaluate(targets[m], predictions[m], ev_var[m], variant_codes)
        results[s] = section

    results["group_sanity_checks"] = _group_sanity_checks(errors, ev_song, ev_note)
    results["variant_identity_checks"] = {
        s: _variant_identity_check(errors[ev_split == split_codes[s]], ev_var[ev_split == split_codes[s]],
                                   ev_song[ev_split == split_codes[s]], ev_note[ev_split == split_codes[s]], variant_codes)
        for s in EVALUATED_SPLITS
    }

    # ---- validation detail on clean notes (one row per note; under the verified
    # identity every variant of a note carries the same error)
    val = ev_split == split_codes["validation"]
    vm = val
    vc = val & (ev_var == variant_codes["clean"])
    results["validation"]["by_song"] = _by_song(targets[vm], predictions[vm], ev_song[vm], ev_var[vm], song_ids,
                                                variant_codes["clean"])
    results["validation"]["song_extremes"] = _song_extremes(results["validation"]["by_song"])
    fe, ce = full_err[vc], errors[vc]
    nf, du = ev_frames[vc], ev_dur[vc]
    results["validation"]["by_frame_count"] = {
        (f"{lo}+" if hi is None else f"{lo}-{hi}"): _pair_metrics(fe[_in(nf, lo, hi, True)], ce[_in(nf, lo, hi, True)])
        for lo, hi in FRAME_BUCKETS
    }
    results["validation"]["by_duration"] = {
        (f"<{hi:.2f}s" if lo is None else f">={lo:.2f}s" if hi is None else f"{lo:.2f}-{hi:.2f}s"):
            _pair_metrics(fe[_in(du, lo, hi, False)], ce[_in(du, lo, hi, False)])
        for lo, hi in DURATION_BUCKETS_S
    }
    noop_val = float(np.mean(nf < MIN_VOICED_FOR_CORE))
    tr_c = (ev_split == split_codes["train"]) & (ev_var == variant_codes["clean"])
    results["validation"]["by_frame_count"]["_note"] = (
        f"9|10 boundary = MIN_VOICED_FOR_CORE; below it core == full contour, so full and core medians are equal by construction")

    # ---- comparisons
    fv, cv = full_results["validation"], results["validation"]
    ft, ct = fv["tail"], cv["tail"]
    keys_q = ("p90", "p95", "p99", "p99_9")
    results["comparison_to_full_median"] = {
        "full_median_results_path": result_paths["full"],
        "split": "validation",
        "mae": _improvement(fv["overall"]["mae"], cv["overall"]["mae"]),
        "rmse": _improvement(fv["overall"]["rmse"], cv["overall"]["rmse"]),
        "median_absolute_error": _improvement(fv["overall"]["median_absolute_error"], cv["overall"]["median_absolute_error"]),
        "mean_signed_error": {"full_median": fv["overall"]["mean_signed_error"], "core_median": cv["overall"]["mean_signed_error"]},
        **{f"{q}_abs_error": _improvement(ft["absolute_error_quantiles"][q], ct["absolute_error_quantiles"][q]) for q in keys_q},
        "max_abs_error": _improvement(ft["max_absolute_error"], ct["max_absolute_error"]),
        "exceedances": {k: {"full_median": ft["absolute_error_exceeding"][k], "core_median": ct["absolute_error_exceeding"][k]}
                        for k in ct["absolute_error_exceeding"]},
        "by_variant_mae": {n: _improvement(fv["by_variant"][n]["mae"], cv["by_variant"][n]["mae"]) for n in EXPECTED_VARIANT_ORDER},
        "train": {"mae": _improvement(full_results["train"]["overall"]["mae"], results["train"]["overall"]["mae"]),
                  "rmse": _improvement(full_results["train"]["overall"]["rmse"], results["train"]["overall"]["rmse"])},
    }
    zv = zero_results["validation"]
    results["comparison_to_zero_baseline"] = {
        "zero_results_path": result_paths["zero"],
        "split": "validation",
        "ladder": {
            name: {"mae": r["overall"]["mae"], "rmse": r["overall"]["rmse"],
                   "median_absolute_error": r["overall"]["median_absolute_error"]}
            for name, r in (("zero", zv), ("full_median", fv), ("core_median", cv))
        },
        "core_vs_zero": {m: {"zero": zv["overall"][m], "core_median": cv["overall"][m],
                             "absolute_improvement": zv["overall"][m] - cv["overall"][m],
                             "percent_improvement": 100.0 * (zv["overall"][m] - cv["overall"][m]) / zv["overall"][m]}
                         for m in ("mae", "rmse", "median_absolute_error")},
        "by_variant": {n: {"zero_mae": zv["by_variant"][n]["mae"], "full_median_mae": fv["by_variant"][n]["mae"],
                           "core_median_mae": cv["by_variant"][n]["mae"], "zero_rmse": zv["by_variant"][n]["rmse"],
                           "full_median_rmse": fv["by_variant"][n]["rmse"], "core_median_rmse": cv["by_variant"][n]["rmse"]}
                       for n in EXPECTED_VARIANT_ORDER},
    }

    # ---- when do full and core differ, and who wins (clean validation notes)
    d = np.abs(ce - fe)
    fa, ca = np.abs(fe), np.abs(ce)
    delta = ca - fa
    improved, worsened = delta < -IDENTICAL_TOLERANCE_CENTS, delta > IDENTICAL_TOLERANCE_CENTS
    rows = []
    vci = np.flatnonzero(vc)
    for j, gi in enumerate(vci):
        rows.append({"song_id": str(song_ids[ev_song[gi]]), "source_note_index": int(ev_note[gi]),
                     "frame_count": int(nf[j]), "duration_s": float(du[j]),
                     "full_median": float(fe[j]), "core_median": float(ce[j]),
                     "full_abs_error": float(fa[j]), "core_abs_error": float(ca[j]), "delta_core_minus_full": float(delta[j])})
    order = np.argsort(delta, kind="stable")
    results["full_vs_core_difference_analysis"] = {
        "population": "clean validation notes (targets are 0, so errors equal the medians)",
        "note_count": int(d.size),
        "abs_core_minus_full_cents": {
            **{f"p{q:g}": float(np.percentile(d, q)) for q in (50, 75, 90, 95, 99)}, "max": float(d.max()),
            "percent_identical_within_1e-9": 100.0 * float(np.mean(d <= IDENTICAL_TOLERANCE_CENTS)),
            **{f"percent_gt_{t:g}_cents": 100.0 * float(np.mean(d > t)) for t in DIFF_THRESHOLDS_CENTS},
        },
        "trim_noop": {
            "validation_clean_notes_percent": 100.0 * noop_val,
            "validation_clean_notes_count": int(np.count_nonzero(nf < MIN_VOICED_FOR_CORE)),
            "train_clean_notes_count": int(np.count_nonzero(ev_frames[tr_c] < MIN_VOICED_FOR_CORE)),
            "train_clean_notes_percent": 100.0 * float(np.mean(ev_frames[tr_c] < MIN_VOICED_FOR_CORE)),
            "trimmed_notes_with_identical_median_percent_of_trimmed":
                100.0 * float(np.mean(d[nf >= MIN_VOICED_FOR_CORE] <= IDENTICAL_TOLERANCE_CENTS)),
        },
        "per_note_outcome": {
            "percent_improved": 100.0 * float(improved.mean()),
            "percent_unchanged_within_1e-9": 100.0 * float((~improved & ~worsened).mean()),
            "percent_worsened": 100.0 * float(worsened.mean()),
            "median_improvement_among_improved_cents": float(np.median(-delta[improved])) if improved.any() else None,
            "median_degradation_among_worsened_cents": float(np.median(delta[worsened])) if worsened.any() else None,
            "improved_count": int(improved.sum()), "worsened_count": int(worsened.sum()),
        },
        "biggest_improvements": [rows[i] for i in order[:EXTREME_COUNT]],
        "biggest_degradations": [rows[i] for i in order[::-1][:EXTREME_COUNT]],
    }

    # ---- coupling between this estimator and the dataset's target definition.
    # A note's target MIDI is round(median(_core_slice(midi))) (_finalize_segment)
    # and its cents are measured from that target, so the clean core median is a
    # rounding residual: |core median| <= 50 c by construction. Measure how much of
    # the gain over the full median is that cap, and how much survives without it.
    band = np.abs(fe) <= 50.0
    clip = np.clip(fe, -50.0, 50.0)
    results["target_coupling_analysis"] = {
        "explanation": ("target_midi = round(median(_core_slice(midi_float))), cents = 100*(midi_float - target); "
                        "so median(_core_slice(clean cents)) lies in [-50, +50] c for every note by construction. "
                        "The full median has no such bound. Errors above 50 c are therefore impossible for this "
                        "baseline, and its tail improvement over the full median is partly structural."),
        "clean_validation_notes": int(ce.size),
        "max_abs_core_median_cents": float(np.max(np.abs(ce))),
        "notes_with_abs_core_median_gt_50": int(np.count_nonzero(np.abs(ce) > 50.0)),
        "notes_with_abs_full_median_gt_50": int(np.count_nonzero(~band)),
        "percent_notes_with_abs_full_median_gt_50": 100.0 * float(np.mean(~band)),
        "notes_with_abs_full_median_le_50": _pair_metrics(fe[band], ce[band]),
        "notes_with_abs_full_median_gt_50": _pair_metrics(fe[~band], ce[~band]),
        "full_median_clipped_to_plus_minus_50": {
            "mae": float(np.mean(np.abs(clip))), "rmse": float(np.sqrt(np.mean(clip ** 2))),
            "note": "full-median errors capped at the same +-50 c bound the core median has by construction; "
                    "a descriptive reference, not a baseline",
        },
        "core_median": {"mae": float(np.mean(np.abs(ce))), "rmse": float(np.sqrt(np.mean(ce ** 2)))},
        "full_median": {"mae": float(np.mean(np.abs(fe))), "rmse": float(np.sqrt(np.mean(fe ** 2)))},
    }

    # ---- per song, full vs core
    per_song = {}
    for code in np.unique(ev_song[vm]):
        m = vm & (ev_song == code)
        f_, c_ = full_err[m], errors[m]
        per_song[str(song_ids[code])] = {
            "clean_note_count": int(np.count_nonzero(m & (ev_var == variant_codes["clean"]))),
            "full_median_mae": float(np.mean(np.abs(f_))), "core_median_mae": float(np.mean(np.abs(c_))),
            "mae_change": float(np.mean(np.abs(c_)) - np.mean(np.abs(f_))),
            "full_median_rmse": float(np.sqrt(np.mean(f_ ** 2))), "core_median_rmse": float(np.sqrt(np.mean(c_ ** 2))),
            "rmse_change": float(np.sqrt(np.mean(c_ ** 2)) - np.sqrt(np.mean(f_ ** 2))),
        }
    by_change = sorted(per_song, key=lambda s: per_song[s]["mae_change"])
    results["validation"]["by_song_full_vs_core"] = per_song
    results["validation"]["by_song_full_vs_core_extremes"] = {
        "largest_mae_improvement": {"song_id": by_change[0], **per_song[by_change[0]]},
        "largest_mae_degradation_or_smallest_improvement": {"song_id": by_change[-1], **per_song[by_change[-1]]},
        "songs_improved_mae": sum(per_song[s]["mae_change"] < 0 for s in per_song),
        "songs_worsened_mae": sum(per_song[s]["mae_change"] > 0 for s in per_song),
    }

    tr, va = results["train"]["overall"], results["validation"]["overall"]
    results["train_vs_validation"] = {
        "train_mae": tr["mae"], "validation_mae": va["mae"], "mae_gap_validation_minus_train": va["mae"] - tr["mae"],
        "train_rmse": tr["rmse"], "validation_rmse": va["rmse"], "rmse_gap_validation_minus_train": va["rmse"] - tr["rmse"],
        "interpretation": "no learned parameters: the gap reflects which songs and notes each split holds, not overfitting",
    }
    results["interpretation_caveat"] = (
        "The production core trim is symmetric. An improvement shows trimming helps; it does not show that "
        "symmetric trimming is optimal (earlier audits found releases deviate from note centers more than onsets). "
        "Because the dataset's target MIDI is itself chosen from the core median, this baseline's errors are "
        "bounded by +-50 c by construction; see target_coupling_analysis.")
    return results


def _comparison(results: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "experiment": results["experiment"] + "_comparison",
        "test_evaluated": False,
        "dataset": results["dataset"]["dataset_name"],
        "core_definition_rule": results["core_definition"]["rule"],
        "validation_ladder": results["comparison_to_zero_baseline"]["ladder"],
        "core_vs_zero": results["comparison_to_zero_baseline"]["core_vs_zero"],
        "core_vs_full_median": results["comparison_to_full_median"],
        "by_frame_count": results["validation"]["by_frame_count"],
        "by_duration": results["validation"]["by_duration"],
        "full_vs_core_difference_analysis": {k: v for k, v in results["full_vs_core_difference_analysis"].items()
                                             if k not in ("biggest_improvements", "biggest_degradations")},
        "by_song_full_vs_core": results["validation"]["by_song_full_vs_core"],
        "by_song_full_vs_core_extremes": results["validation"]["by_song_full_vs_core_extremes"],
        "train_vs_validation": results["train_vs_validation"],
        "target_coupling_analysis": results["target_coupling_analysis"],
        "sanity": {"group_sanity_checks": results["group_sanity_checks"],
                   "variant_identity_all_identical": {s: v["all_variants_identical_within_tolerance"]
                                                      for s, v in results["variant_identity_checks"].items()},
                   "stored_core_median_check": results["verification"]["independent_core_median_vs_stored_core_median_cents"]},
        "interpretation_caveat": results["interpretation_caveat"],
    }


def _markdown(r: Mapping[str, Any]) -> str:
    f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
    lad = r["comparison_to_zero_baseline"]["ladder"]
    cf = r["comparison_to_full_median"]
    L = [f"# Baseline 3 (core median) on {r['dataset']['dataset_name']}: validation", "",
         "Generated by `ml/baselines/core_median_baseline.py`. Train and validation only; **test not evaluated**.", "",
         f"Core rule: {r['core_definition']['rule']}.", "",
         "| validation | MAE | RMSE | median abs error |", "|---|---|---|---|"]
    for n in ("zero", "full_median", "core_median"):
        L.append(f"| {n} | {f(lad[n]['mae'])} | {f(lad[n]['rmse'])} | {f(lad[n]['median_absolute_error'])} |")
    L += ["", "| full → core (validation) | full median | core median | improvement | % |", "|---|---|---|---|---|"]
    for k in ("mae", "rmse", "median_absolute_error", "p90_abs_error", "p95_abs_error", "p99_abs_error", "p99_9_abs_error", "max_abs_error"):
        x = cf[k]
        L.append(f"| {k} | {f(x['full_median'])} | {f(x['core_median'])} | {x['absolute_improvement']:+.4f} | {f(x['percent_improvement'], 2)} |")
    L += ["", "| errors above | full median | core median |", "|---|---|---|"]
    for k, x in cf["exceedances"].items():
        L.append(f"| {k} | {x['full_median']['count']} ({x['full_median']['percent']:.2f}%) | {x['core_median']['count']} ({x['core_median']['percent']:.2f}%) |")
    for title, key in (("By frame count (clean validation notes)", "by_frame_count"), ("By duration (clean validation notes)", "by_duration")):
        L += ["", f"**{title}**", "", "| bucket | notes | full MAE | core MAE | Δ MAE | % | full RMSE | core RMSE | identical | median abs(core−full) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for b, x in r["validation"][key].items():
            if b.startswith("_") or not x.get("note_count"):
                continue
            L.append(f"| {b} | {x['note_count']} | {f(x['full_median_mae'])} | {f(x['core_median_mae'])} | {x['mae_improvement']:+.4f} | "
                     f"{f(x['percent_mae_improvement'], 2)} | {f(x['full_median_rmse'])} | {f(x['core_median_rmse'])} | "
                     f"{100 * x['fraction_full_equals_core']:.1f}% | {f(x['median_abs_core_minus_full_cents'], 3)} |")
    L += ["", "**By song (validation; estimator fit to each recording, not singer quality)**", "",
          "| song | notes | full MAE | core MAE | Δ MAE | full RMSE | core RMSE | Δ RMSE |", "|---|---|---|---|---|---|---|---|"]
    for s, x in sorted(r["validation"]["by_song_full_vs_core"].items(), key=lambda kv: kv[1]["mae_change"]):
        L.append(f"| {s} | {x['clean_note_count']} | {f(x['full_median_mae'])} | {f(x['core_median_mae'])} | {x['mae_change']:+.4f} | "
                 f"{f(x['full_median_rmse'])} | {f(x['core_median_rmse'])} | {x['rmse_change']:+.4f} |")
    tc = r["target_coupling_analysis"]
    L += ["", "**Target coupling.** " + tc["explanation"], "",
          "| clean validation notes | n | full MAE | core MAE | full RMSE | core RMSE |", "|---|---|---|---|---|---|"]
    for k, label in (("notes_with_abs_full_median_le_50", "|full median| <= 50 c"), ("notes_with_abs_full_median_gt_50", "|full median| > 50 c")):
        x = tc[k]
        L.append(f"| {label} | {x['note_count']} | {f(x['full_median_mae'])} | {f(x['core_median_mae'])} | {f(x['full_median_rmse'])} | {f(x['core_median_rmse'])} |")
    c = tc["full_median_clipped_to_plus_minus_50"]
    L.append(f"| full median clipped to ±50 c (reference) | {tc['clean_validation_notes']} | {f(c['mae'])} | — | {f(c['rmse'])} | — |")
    L += ["", f"_{r['interpretation_caveat']}_", ""]
    return "\n".join(L)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-version", choices=DATASET_VERSIONS, default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--results-dir", type=Path, help="override ml/results/<dataset-version>/")
    args = parser.parse_args()
    results_dir = (args.results_dir or dataset_paths(args.dataset_version)["results_dir"]).resolve()
    rel = lambda p: str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
    zero_path, full_path = results_dir / "zero_baseline.json", results_dir / "median_cents_baseline.json"
    results = compute(args.dataset_version, json.loads(zero_path.read_text(encoding="utf-8")),
                      json.loads(full_path.read_text(encoding="utf-8")), {"zero": rel(zero_path), "full": rel(full_path)})

    out = results_dir / "core_median_baseline.json"
    for p in (out, results_dir / "core_median_comparison.json", results_dir / "core_median_comparison.md"):
        if p.exists():
            raise SystemExit(f"{p} already exists; refusing to overwrite")
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (results_dir / "core_median_comparison.json").write_text(json.dumps(_comparison(results), indent=2) + "\n", encoding="utf-8")
    (results_dir / "core_median_comparison.md").write_text(_markdown(results), encoding="utf-8")

    tr, va = results["train"]["overall"], results["validation"]["overall"]
    cf = results["comparison_to_full_median"]
    print(f"{args.dataset_version} — Core-Median Baseline")
    print(f"TRAIN       n={tr['example_count']:,}  MAE {tr['mae']:.6f}  RMSE {tr['rmse']:.6f}")
    print(f"VALIDATION  n={va['example_count']:,}  MAE {va['mae']:.6f}  RMSE {va['rmse']:.6f}")
    print(f"vs full median: MAE {cf['mae']['absolute_improvement']:+.6f} ({cf['mae']['percent_improvement']:+.3f}%)  "
          f"RMSE {cf['rmse']['absolute_improvement']:+.6f} ({cf['rmse']['percent_improvement']:+.3f}%)")
    g = results["group_sanity_checks"]
    print(f"groups {g['note_groups_checked']:,}  max spread {g['max_within_note_error_spread_cents']:.3e}  violations {g['groups_violating_tolerance']}")
    print("Test evaluated: NO")
    print(f"Results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
