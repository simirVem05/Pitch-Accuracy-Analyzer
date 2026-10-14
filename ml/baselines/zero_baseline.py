#!/usr/bin/env python3
"""
Zero baseline for Axis 2: predict 0.0 cents of injected shift for every example.

This is the no-information floor that a learned model must beat. It reads no
features at all — not the cents contour, not CREPE confidence, not duration,
target MIDI, song, or variant. Because the prediction is identically zero, the
absolute error of every example is exactly `abs(injected_shift_cents)`, which
makes the resulting MAE a direct readout of the label distribution.

The test split is sealed: it is never sliced and no test metric is computed.

Runs against any dataset version with the same encoding:
  python zero_baseline.py                              # axis2_v1 -> ml/results/axis2_v1/
  python zero_baseline.py --dataset-version axis2_v2   # axis2_v2 -> ml/results/axis2_v2/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_VERSION = "axis2_v1"
DATASET_VERSIONS = ("axis2_v1", "axis2_v2")

EVALUATED_SPLITS = ("train", "validation")
SEALED_SPLIT = "test"

EXPECTED_SPLIT_ORDER = ("train", "validation", "test")
EXPECTED_VARIANT_ORDER = ("clean", "corruption_1", "corruption_2", "corruption_3", "corruption_4")


# Descriptive tail of the absolute-error distribution. Added for the v1/v2
# comparison; these do not enter any prediction.
TAIL_QUANTILES = (50.0, 75.0, 90.0, 95.0, 99.0, 99.5, 99.9)
TAIL_THRESHOLDS_CENTS = (25.0, 50.0, 100.0, 200.0, 400.0)


class DatasetContractError(RuntimeError):
    """Raised when the dataset does not match the encoding this baseline assumes."""


def dataset_paths(version: str) -> Dict[str, Path]:
    if version not in DATASET_VERSIONS:
        raise DatasetContractError(f"unknown dataset version {version!r}; expected one of {DATASET_VERSIONS}")
    dataset_dir = REPO_ROOT / "data" / "ml_dataset" / version
    return {
        "dataset_dir": dataset_dir,
        "examples": dataset_dir / "examples.npz",
        "splits": dataset_dir / "splits.json",
        "summary": dataset_dir / "summary.json",
        "results_dir": REPO_ROOT / "ml" / "results" / version,
    }


def _tail(absolute_error: np.ndarray) -> Dict[str, Any]:
    n = int(absolute_error.size)
    if not n:
        return {"example_count": 0}
    return {
        "example_count": n,
        "absolute_error_quantiles": {
            f"p{q:g}".replace(".", "_"): float(np.percentile(absolute_error, q)) for q in TAIL_QUANTILES
        },
        "max_absolute_error": float(np.max(absolute_error)),
        "absolute_error_exceeding": {
            f"gt_{t:g}_cents": {
                "count": int(np.count_nonzero(absolute_error > t)),
                "percent": 100.0 * float(np.count_nonzero(absolute_error > t)) / n,
            }
            for t in TAIL_THRESHOLDS_CENTS
        },
    }


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
        "tail": _tail(np.abs(predictions - targets)),
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


def compute(version: str = DEFAULT_DATASET_VERSION) -> Dict[str, Any]:
    """All zero-baseline results for one dataset version. Writes nothing."""
    paths = dataset_paths(version)
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    with np.load(paths["examples"], allow_pickle=False) as data:
        split_codes = _resolve_lookup(data["split_names"], EXPECTED_SPLIT_ORDER, "split_names")
        variant_codes = _resolve_lookup(data["variant_names"], EXPECTED_VARIANT_ORDER, "variant_names")

        split_index = np.asarray(data["split_index"])
        variant_index = np.asarray(data["variant_index"])
        song_index = np.asarray(data["song_index"])
        song_ids = np.asarray(data["song_ids"])
        target = np.asarray(data["injected_shift_cents"], dtype=np.float64)

    _verify_counts(split_index, split_codes, splits)
    if summary.get("dataset_name") != version:
        raise DatasetContractError(f"{paths['summary']} names dataset {summary.get('dataset_name')!r}, not {version!r}")

    results: Dict[str, Any] = {
        "experiment": f"{version}_zero_baseline",
        "prediction_rule": "always_predict_zero_cents",
        "test_evaluated": False,
        "dataset": {
            "path": str(paths["examples"].relative_to(REPO_ROOT)),
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
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-version", choices=DATASET_VERSIONS, default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--results-dir", type=Path, help="override ml/results/<dataset-version>/")
    args = parser.parse_args()

    results = compute(args.dataset_version)
    results_path = (args.results_dir or dataset_paths(args.dataset_version)["results_dir"]).resolve() / "zero_baseline.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    train = results["train"]["overall"]
    validation = results["validation"]["overall"]

    print(f"{args.dataset_version} — Zero Baseline")
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
    print(f"{results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
