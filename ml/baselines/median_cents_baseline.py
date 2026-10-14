#!/usr/bin/env python3
"""
Median-cents baseline for Axis 2: predict each example's own contour median.

The rule is `prediction = median(own cents contour)`, which assumes the observed
median deviation *is* the injected synthetic shift — i.e. that a professional
reference note is naturally centered at 0 cents. That assumption is the point of
the experiment, not a bug to fix here: whatever error survives is the natural
offset of the released performance.

Anti-leakage is structural rather than promised. `_predict_from_own_contour`
receives only the flat contour store and the example indices to score; it is
handed no labels, no variant codes, and no sibling grouping, so a
`corrupted_median - clean_median` shortcut is not expressible inside it. Sibling
grouping is built afterwards, and only to verify a mathematical property of the
constant-shift corruption.

The test split is sealed: no test contour median is computed and no test metric
is reported.

Runs against any dataset version with the same encoding; reads that version's
zero-baseline results for the comparison section, so run zero_baseline.py first:
  python median_cents_baseline.py                              # axis2_v1
  python median_cents_baseline.py --dataset-version axis2_v2   # axis2_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from zero_baseline import (
    DATASET_VERSIONS,
    DEFAULT_DATASET_VERSION,
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

EVALUATED_SPLITS = ("train", "validation")
SEALED_SPLIT = "test"

# median(clean + S) - S reproduces median(clean) only up to float rounding, so the
# within-note spread is expected at ~1e-14 rather than exactly zero.
GROUP_TOLERANCE_CENTS = 1e-9
STORED_MEDIAN_TOLERANCE_CENTS = 1e-9

VARIANTS_PER_NOTE = 5


def _predict_from_own_contour(
    cents_flat: np.ndarray,
    contour_offsets: np.ndarray,
    example_indices: np.ndarray,
) -> np.ndarray:
    """
    Median of each example's own cents contour.

    Deliberately narrow signature: no targets, no variant codes, no sibling
    index. Each prediction reads exactly one contiguous contour slice.
    """
    predictions = np.empty(example_indices.size, dtype=np.float64)
    for position, index in enumerate(example_indices):
        start = int(contour_offsets[index])
        end = int(contour_offsets[index + 1])
        predictions[position] = np.median(cents_flat[start:end])
    return predictions


def _verify_against_stored_median(predictions: np.ndarray, stored: np.ndarray) -> Dict[str, Any]:
    max_difference = float(np.max(np.abs(predictions - stored))) if predictions.size else 0.0
    if max_difference > STORED_MEDIAN_TOLERANCE_CENTS:
        raise DatasetContractError(
            f"Independently computed contour medians diverge from stored median_cents by "
            f"{max_difference:.6g} cents, above the {STORED_MEDIAN_TOLERANCE_CENTS:g} tolerance"
        )
    return {
        "compared_example_count": int(predictions.size),
        "max_abs_difference_cents": max_difference,
        "tolerance_cents": STORED_MEDIAN_TOLERANCE_CENTS,
        "matches_stored_median_cents": True,
    }


def _evaluate(
    targets: np.ndarray,
    predictions: np.ndarray,
    variant_index: np.ndarray,
    variant_codes: Mapping[str, int],
) -> Dict[str, Any]:
    by_variant = {
        name: _metrics(targets[variant_index == code], predictions[variant_index == code])
        for name, code in variant_codes.items()
    }

    corrupted = variant_index != variant_codes["clean"]
    by_sign = {}
    for label, sign_mask in (
        ("positive", corrupted & (targets > 0)),
        ("negative", corrupted & (targets < 0)),
    ):
        by_sign[label] = _metrics(targets[sign_mask], predictions[sign_mask])

    return {
        "overall": _metrics(targets, predictions),
        "by_variant": by_variant,
        "by_sign": by_sign,
        "tail": _tail(np.abs(predictions - targets)),
    }


def _by_song(
    targets: np.ndarray,
    predictions: np.ndarray,
    song_index: np.ndarray,
    variant_index: np.ndarray,
    song_ids: np.ndarray,
    clean_code: int,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for code in np.unique(song_index):
        mask = song_index == code
        metrics = _metrics(targets[mask], predictions[mask])
        absolute_error = np.abs(predictions[mask] - targets[mask])
        metrics["p95_absolute_error"] = float(np.percentile(absolute_error, 95))
        metrics["p99_absolute_error"] = float(np.percentile(absolute_error, 99))
        metrics["max_absolute_error"] = float(np.max(absolute_error))

        clean_mask = mask & (variant_index == clean_code)
        clean_error = predictions[clean_mask] - targets[clean_mask]
        metrics["clean_mae"] = float(np.mean(np.abs(clean_error))) if clean_error.size else None
        metrics["clean_example_count"] = int(clean_error.size)
        metrics["clean_note_count"] = int(clean_error.size)

        out[str(song_ids[code])] = metrics
    return out


def _song_extremes(by_song: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Which recordings this naive estimator fits best and worst; not a ranking of singers."""
    by = lambda key: sorted(by_song, key=lambda song: by_song[song][key])
    return {
        "best_by_mae": {"song_id": by("mae")[0], "mae": by_song[by("mae")[0]]["mae"]},
        "worst_by_mae": {"song_id": by("mae")[-1], "mae": by_song[by("mae")[-1]]["mae"]},
        "best_by_rmse": {"song_id": by("rmse")[0], "rmse": by_song[by("rmse")[0]]["rmse"]},
        "worst_by_rmse": {"song_id": by("rmse")[-1], "rmse": by_song[by("rmse")[-1]]["rmse"]},
    }


def _variant_identity_check(
    errors: np.ndarray,
    variant_index: np.ndarray,
    song_index: np.ndarray,
    source_note_index: np.ndarray,
    variant_codes: Mapping[str, int],
) -> Dict[str, Any]:
    """
    Post-hoc: is each corruption variant's error vector, note for note, the clean
    variant's? Under a pure constant shift it must be, which makes every
    per-variant error distribution identical.
    """
    def ordered(code: int) -> np.ndarray:
        m = variant_index == code
        order = np.lexsort((source_note_index[m], song_index[m]))
        return errors[m][order]

    clean = ordered(variant_codes["clean"])
    out = {}
    for name, code in variant_codes.items():
        if name == "clean":
            continue
        other = ordered(code)
        out[name] = {
            "max_abs_difference_from_clean_cents": float(np.max(np.abs(other - clean))) if other.size == clean.size else None,
            "same_note_count": bool(other.size == clean.size),
        }
    out["all_variants_identical_within_tolerance"] = all(
        v["same_note_count"] and v["max_abs_difference_from_clean_cents"] <= GROUP_TOLERANCE_CENTS
        for k, v in out.items() if isinstance(v, dict)
    )
    out["tolerance_cents"] = GROUP_TOLERANCE_CENTS
    return out


def _group_sanity_checks(
    errors: np.ndarray,
    song_index: np.ndarray,
    source_note_index: np.ndarray,
) -> Dict[str, Any]:
    """
    Verify the constant-shift property, AFTER predictions were made independently.

    Because corruption adds one constant to every frame, the signed error of all
    five variants of a note must equal that note's natural median cents. Any real
    spread would mean the corruption was not a pure constant translation.
    """
    keys = np.stack([song_index.astype(np.int64), source_note_index.astype(np.int64)])
    _unique, inverse, counts = np.unique(keys, axis=1, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).ravel()
    group_count = counts.size

    group_max = np.full(group_count, -np.inf)
    group_min = np.full(group_count, np.inf)
    np.maximum.at(group_max, inverse, errors)
    np.minimum.at(group_min, inverse, errors)
    spread = group_max - group_min

    return {
        "note_groups_checked": int(group_count),
        "expected_members_per_group": VARIANTS_PER_NOTE,
        "groups_with_unexpected_member_count": int(np.count_nonzero(counts != VARIANTS_PER_NOTE)),
        "tolerance_cents": GROUP_TOLERANCE_CENTS,
        "max_within_note_error_spread_cents": float(np.max(spread)),
        "median_within_note_error_spread_cents": float(np.median(spread)),
        "groups_violating_tolerance": int(np.count_nonzero(spread > GROUP_TOLERANCE_CENTS)),
        "grouping_used_for_prediction": False,
    }


def _improvement(zero_mae: float | None, median_mae: float | None) -> Dict[str, Any]:
    if zero_mae is None or median_mae is None:
        return {"zero_mae": zero_mae, "median_cents_mae": median_mae}
    absolute = zero_mae - median_mae
    return {
        "zero_mae": zero_mae,
        "median_cents_mae": median_mae,
        "absolute_mae_improvement": absolute,
        "percent_mae_improvement": (absolute / zero_mae * 100.0) if zero_mae != 0 else None,
    }


def compute(
    version: str = DEFAULT_DATASET_VERSION,
    zero_results: Mapping[str, Any] | None = None,
    zero_results_path: str | None = None,
) -> Dict[str, Any]:
    """All median-cents results for one dataset version. Writes nothing."""
    paths = dataset_paths(version)
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if zero_results is None:
        zero_path = paths["results_dir"] / "zero_baseline.json"
        zero_results = json.loads(zero_path.read_text(encoding="utf-8"))
        zero_results_path = str(zero_path.relative_to(REPO_ROOT))
    if zero_results["dataset"]["dataset_name"] != version:
        raise DatasetContractError("zero-baseline results belong to a different dataset version")

    with np.load(paths["examples"], allow_pickle=False) as data:
        split_codes = _resolve_lookup(data["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
        variant_codes = _resolve_lookup(data["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")

        split_index = np.asarray(data["split_index"])
        variant_index = np.asarray(data["variant_index"])
        song_index = np.asarray(data["song_index"])
        source_note_index = np.asarray(data["source_note_index"])
        song_ids = np.asarray(data["song_ids"])
        target_all = np.asarray(data["injected_shift_cents"], dtype=np.float64)
        stored_median_all = np.asarray(data["median_cents"], dtype=np.float64)
        cents_flat = np.asarray(data["cents_flat"], dtype=np.float64)
        contour_offsets = np.asarray(data["contour_offsets"], dtype=np.int64)

    _verify_counts(split_index, split_codes, splits)
    if summary.get("dataset_name") != version:
        raise DatasetContractError(f"{paths['summary']} names dataset {summary.get('dataset_name')!r}, not {version!r}")

    # Test indices are never placed in this set, so no test contour is ever read.
    evaluated_codes = [split_codes[name] for name in EVALUATED_SPLITS]
    evaluated_indices = np.flatnonzero(np.isin(split_index, evaluated_codes))

    predictions = _predict_from_own_contour(cents_flat, contour_offsets, evaluated_indices)
    targets = target_all[evaluated_indices]
    errors = predictions - targets

    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise DatasetContractError("Predictions or targets contain non-finite values")

    median_verification = _verify_against_stored_median(
        predictions, stored_median_all[evaluated_indices]
    )

    eval_split = split_index[evaluated_indices]
    eval_variant = variant_index[evaluated_indices]
    eval_song = song_index[evaluated_indices]
    eval_note = source_note_index[evaluated_indices]

    results: Dict[str, Any] = {
        "experiment": f"{version}_median_cents_baseline",
        "prediction_rule": "predict_median_cents_contour",
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
        },
        "verification": {
            "prediction_inputs": ["own_example_cents_contour"],
            "sibling_variants_used_for_prediction": False,
            "clean_reference_used_for_prediction": False,
            "predictions_all_finite": True,
            "targets_all_finite": True,
            "independent_median_vs_stored_median_cents": median_verification,
        },
    }

    for name in EVALUATED_SPLITS:
        mask = eval_split == split_codes[name]
        section = _evaluate(targets[mask], predictions[mask], eval_variant[mask], variant_codes)
        if name == "validation":
            section["by_song"] = _by_song(
                targets[mask],
                predictions[mask],
                eval_song[mask],
                eval_variant[mask],
                song_ids,
                variant_codes["clean"],
            )
            section["song_extremes"] = _song_extremes(section["by_song"])
        results[name] = section

    results["group_sanity_checks"] = _group_sanity_checks(errors, eval_song, eval_note)
    results["variant_identity_checks"] = {
        name: _variant_identity_check(
            errors[eval_split == split_codes[name]], eval_variant[eval_split == split_codes[name]],
            eval_song[eval_split == split_codes[name]], eval_note[eval_split == split_codes[name]], variant_codes,
        )
        for name in EVALUATED_SPLITS
    }

    zero_validation = zero_results["validation"]
    comparison: Dict[str, Any] = {
        "zero_baseline_results_path": zero_results_path,
        "split": "validation",
        "overall": _improvement(
            zero_validation["overall"]["mae"], results["validation"]["overall"]["mae"]
        ),
        "by_variant": {
            name: _improvement(
                zero_validation["by_variant"][name]["mae"],
                results["validation"]["by_variant"][name]["mae"],
            )
            for name in EXPECTED_VARIANT_ORDER
        },
    }
    results["comparison_to_zero_baseline"] = comparison
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-version", choices=DATASET_VERSIONS, default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--results-dir", type=Path, help="override ml/results/<dataset-version>/")
    args = parser.parse_args()

    results_dir = (args.results_dir or dataset_paths(args.dataset_version)["results_dir"]).resolve()
    zero_path = results_dir / "zero_baseline.json"
    zero_label = str(zero_path.relative_to(REPO_ROOT)) if zero_path.is_relative_to(REPO_ROOT) else str(zero_path)
    results = compute(args.dataset_version, json.loads(zero_path.read_text(encoding="utf-8")), zero_label)
    comparison = results["comparison_to_zero_baseline"]
    results_path = results_dir / "median_cents_baseline.json"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    train = results["train"]["overall"]
    validation = results["validation"]["overall"]
    clean = results["validation"]["by_variant"]["clean"]
    overall_comparison = comparison["overall"]

    print(f"{args.dataset_version} — Median-Cents Baseline")
    print()
    print("TRAIN")
    print(f"Examples: {train['example_count']:,}")
    print(f"MAE: {train['mae']:.6f}")
    print(f"RMSE: {train['rmse']:.6f}")
    print()
    print("VALIDATION")
    print(f"Examples: {validation['example_count']:,}")
    print(f"MAE: {validation['mae']:.6f}")
    print(f"RMSE: {validation['rmse']:.6f}")
    print()
    print("Validation by variant:")
    for name in EXPECTED_VARIANT_ORDER:
        variant = results["validation"]["by_variant"][name]
        print(
            f"{name}: MAE {variant['mae']:.6f}  RMSE {variant['rmse']:.6f}  "
            f"signed {variant['mean_signed_error']:+.6f}  n={variant['example_count']:,}"
        )
    print()
    print("Validation clean detail (natural offset of professional references):")
    print(
        f"MAE {clean['mae']:.6f}  RMSE {clean['rmse']:.6f}  "
        f"signed {clean['mean_signed_error']:+.6f}  median abs {clean['median_absolute_error']:.6f}"
    )
    print()
    print("Validation by corruption sign:")
    for label in ("positive", "negative"):
        sign = results["validation"]["by_sign"][label]
        print(
            f"{label}: MAE {sign['mae']:.6f}  RMSE {sign['rmse']:.6f}  "
            f"signed {sign['mean_signed_error']:+.6f}  n={sign['example_count']:,}"
        )
    print()
    print("Comparison to zero baseline (validation):")
    print(f"Zero baseline MAE:   {overall_comparison['zero_mae']:.6f}")
    print(f"Median-cents MAE:    {overall_comparison['median_cents_mae']:.6f}")
    print(f"Absolute improvement: {overall_comparison['absolute_mae_improvement']:+.6f} cents")
    print(f"Percent improvement:  {overall_comparison['percent_mae_improvement']:+.4f}%")
    print()
    group = results["group_sanity_checks"]
    print("Group sanity (constant-shift property, computed after prediction):")
    print(f"Note groups checked: {group['note_groups_checked']:,}")
    print(f"Max within-note error spread: {group['max_within_note_error_spread_cents']:.3e} cents")
    print(f"Groups violating tolerance: {group['groups_violating_tolerance']}")
    print()
    print("Test evaluated: NO")
    print()
    print("Results written to:")
    print(f"{results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
