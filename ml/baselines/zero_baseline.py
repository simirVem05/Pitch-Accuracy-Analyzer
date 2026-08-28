#!/usr/bin/env python3
"""
Zero baseline for Axis 2 V1: predict 0.0 cents of injected shift for every example.

This is the no-information floor that a learned model must beat. It reads no
features at all — not the cents contour, not CREPE confidence, not duration,
target MIDI, song, or variant. Because the prediction is identically zero, the
absolute error of every example is exactly `abs(injected_shift_cents)`, which
makes the resulting MAE a direct readout of the label distribution.

The test split is sealed: it is never sliced and no test metric is computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "data" / "ml_dataset" / "axis2_v1"
EXAMPLES_PATH = DATASET_DIR / "examples.npz"
SPLITS_PATH = DATASET_DIR / "splits.json"
SUMMARY_PATH = DATASET_DIR / "summary.json"
RESULTS_PATH = REPO_ROOT / "ml" / "results" / "axis2_v1" / "zero_baseline.json"

EVALUATED_SPLITS = ("train", "validation")
SEALED_SPLIT = "test"

EXPECTED_SPLIT_ORDER = ("train", "validation", "test")
EXPECTED_VARIANT_ORDER = ("clean", "corruption_1", "corruption_2", "corruption_3", "corruption_4")


class DatasetContractError(RuntimeError):
    """Raised when the dataset does not match the encoding this baseline assumes."""


def _resolve_lookup(names: np.ndarray, expected: tuple[str, ...], label: str) -> Dict[str, int]:
    """
    Map name -> stored integer code using the dataset's own lookup array.

    The codes are read from the file rather than hardcoded, so a future rebuild
    that reorders them fails loudly here instead of silently mislabelling every
    metric downstream.
    """
    actual = tuple(str(name) for name in names)
    if actual != expected:
        raise DatasetContractError(f"{label} lookup is {actual}, expected {expected}")
    return {name: index for index, name in enumerate(actual)}


def _verify_counts(split_index: np.ndarray, split_codes: Mapping[str, int], splits: Mapping[str, Any]) -> None:
    for name, code in split_codes.items():
        stored = splits.get(f"{name}_example_count")
        if stored is None:
            raise DatasetContractError(f"splits.json has no {name}_example_count")
        counted = int(np.count_nonzero(split_index == code))
        if counted != int(stored):
            raise DatasetContractError(
                f"{name} split holds {counted} examples but splits.json declares {stored}"
            )


def _metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    error = predictions - targets
    absolute_error = np.abs(error)
    return {
        "example_count": int(targets.size),
        "mae": float(np.mean(absolute_error)) if targets.size else None,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if targets.size else None,
        "mean_signed_error": float(np.mean(error)) if targets.size else None,
        "median_absolute_error": float(np.median(absolute_error)) if targets.size else None,
    }


def _sign_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """Signed-error breakdown, so a sharp bias cannot hide inside an absolute mean."""
    out = _metrics(targets, predictions)
    out.pop("median_absolute_error")
    return out


def _check_invariants(targets: np.ndarray, predictions: np.ndarray, is_clean: np.ndarray) -> Dict[str, Any]:
    absolute_error = np.abs(predictions - targets)
    identity_residual = float(np.max(np.abs(absolute_error - np.abs(targets)))) if targets.size else 0.0

    checks = {
        "all_predictions_exactly_zero": bool(np.all(predictions == 0.0)),
        "all_clean_targets_exactly_zero": bool(np.all(targets[is_clean] == 0.0)),
        "predictions_all_finite": bool(np.all(np.isfinite(predictions))),
        "targets_all_finite": bool(np.all(np.isfinite(targets))),
        "max_abs_error_minus_abs_target": identity_residual,
        "abs_error_equals_abs_target": identity_residual == 0.0,
    }
    failed = [name for name, value in checks.items() if isinstance(value, bool) and not value]
    if failed:
        raise DatasetContractError(f"Baseline invariants failed: {', '.join(failed)}")
    return checks


def _evaluate_split(
    targets: np.ndarray,
    variant_index: np.ndarray,
    variant_codes: Mapping[str, int],
) -> Dict[str, Any]:
    predictions = np.zeros_like(targets)
    is_clean = variant_index == variant_codes["clean"]

    by_variant = {
        name: _metrics(targets[variant_index == code], predictions[variant_index == code])
        for name, code in variant_codes.items()
    }

    corrupted = ~is_clean
    by_sign = {
        "positive": _sign_metrics(
            targets[corrupted & (targets > 0)], predictions[corrupted & (targets > 0)]
        ),
        "negative": _sign_metrics(
            targets[corrupted & (targets < 0)], predictions[corrupted & (targets < 0)]
        ),
    }

    return {
        "overall": _metrics(targets, predictions),
        "by_variant": by_variant,
        "by_sign": by_sign,
        "sanity_checks": _check_invariants(targets, predictions, is_clean),
    }


def _by_song(targets: np.ndarray, song_index: np.ndarray, song_ids: np.ndarray) -> Dict[str, Any]:
    predictions = np.zeros_like(targets)
    out: Dict[str, Any] = {}
    for code in np.unique(song_index):
        mask = song_index == code
        song_id = str(song_ids[code])
        metrics = _metrics(targets[mask], predictions[mask])
        metrics.pop("median_absolute_error")
        out[song_id] = metrics
    return out


def main() -> int:
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    with np.load(EXAMPLES_PATH, allow_pickle=False) as data:
        split_codes = _resolve_lookup(data["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
        variant_codes = _resolve_lookup(data["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")

        split_index = np.asarray(data["split_index"])
        variant_index = np.asarray(data["variant_index"])
        song_index = np.asarray(data["song_index"])
        song_ids = np.asarray(data["song_ids"])
        target = np.asarray(data["injected_shift_cents"], dtype=np.float64)

    _verify_counts(split_index, split_codes, splits)

    results: Dict[str, Any] = {
        "experiment": "axis2_v1_zero_baseline",
        "prediction_rule": "always_predict_zero_cents",
        "test_evaluated": False,
        "dataset": {
            "path": str(EXAMPLES_PATH.relative_to(REPO_ROOT)),
            "dataset_name": summary.get("dataset_name"),
            "schema_version": summary.get("schema_version"),
            "label_field": summary.get("label_field"),
            "total_example_count": int(summary.get("total_example_count", target.size)),
            "split_encoding": split_codes,
            "variant_encoding": variant_codes,
            "sealed_split": SEALED_SPLIT,
        },
        "features_used": [],
    }

    for name in EVALUATED_SPLITS:
        mask = split_index == split_codes[name]
        section = _evaluate_split(target[mask], variant_index[mask], variant_codes)
        if name == "validation":
            section["by_song"] = _by_song(target[mask], song_index[mask], song_ids)
        results[name] = section

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    train = results["train"]["overall"]
    validation = results["validation"]["overall"]

    print("Axis 2 V1 — Zero Baseline")
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
        print(f"{name}: MAE {variant['mae']:.6f}  RMSE {variant['rmse']:.6f}  n={variant['example_count']:,}")
    print()
    print("Test evaluated: NO")
    print()
    print("Results written to:")
    print(f"{RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
