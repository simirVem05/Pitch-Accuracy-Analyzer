#!/usr/bin/env python3
"""
Median-cents baseline for Axis 2 V1: predict each example's own contour median.

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
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from zero_baseline import (
    EXPECTED_SPLIT_ORDER,
    EXPECTED_VARIANT_ORDER,
    REPO_ROOT,
    SPLITS_PATH,
    SUMMARY_PATH,
    DatasetContractError,
    _metrics,
    _resolve_lookup,
    _verify_counts,
)
from zero_baseline import EXAMPLES_PATH


RESULTS_PATH = REPO_ROOT / "ml" / "results" / "axis2_v1" / "median_cents_baseline.json"
ZERO_BASELINE_PATH = REPO_ROOT / "ml" / "results" / "axis2_v1" / "zero_baseline.json"

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
        metrics.pop("median_absolute_error")

        clean_mask = mask & (variant_index == clean_code)
        clean_error = predictions[clean_mask] - targets[clean_mask]
        metrics["clean_mae"] = float(np.mean(np.abs(clean_error))) if clean_error.size else None
        metrics["clean_example_count"] = int(clean_error.size)

        out[str(song_ids[code])] = metrics
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


def main() -> int:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    zero_results = json.loads(ZERO_BASELINE_PATH.read_text(encoding="utf-8"))

    with np.load(EXAMPLES_PATH, allow_pickle=False) as data:
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
        "experiment": "axis2_v1_median_cents_baseline",
        "prediction_rule": "predict_median_cents_contour",
        "features_used": ["cents_contour"],
        "test_evaluated": False,
        "dataset": {
            "path": str(EXAMPLES_PATH.relative_to(REPO_ROOT)),
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
        results[name] = section

    results["group_sanity_checks"] = _group_sanity_checks(errors, eval_song, eval_note)

    zero_validation = zero_results["validation"]
    comparison: Dict[str, Any] = {
        "zero_baseline_results_path": str(ZERO_BASELINE_PATH.relative_to(REPO_ROOT)),
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

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    train = results["train"]["overall"]
    validation = results["validation"]["overall"]
    clean = results["validation"]["by_variant"]["clean"]
    overall_comparison = comparison["overall"]

    print("Axis 2 V1 — Median-Cents Baseline")
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
    print(f"{RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
