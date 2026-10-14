#!/usr/bin/env python3
"""
Compare Baselines 1 (zero) and 2 (median-cents) between axis2_v1 and axis2_v2.

1. Recomputes both baselines on axis2_v1 in memory with the current code and
   checks every value in the historical ml/results/axis2_v1/*.json against it,
   so the refactor is shown not to have changed v1 behaviour. The historical
   files are only read. Their tail statistics did not exist yet, so the v1 tail
   numbers used below come from this recomputation.
2. Recomputes axis2_v2 in memory and checks it equals ml/results/axis2_v2/*.json.
3. Writes ml/results/axis2_v2/v1_vs_v2_baseline_comparison.{json,md}.

Train and validation only. The test split is never scored by either baseline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import median_cents_baseline as MB
import zero_baseline as ZB

REPO = ZB.REPO_ROOT
V1_DIR = REPO / "ml" / "results" / "axis2_v1"
V2_DIR = REPO / "ml" / "results" / "axis2_v2"
VARIANTS = ZB.EXPECTED_VARIANT_ORDER
REL_TOL = 1e-12
ABS_TOL = 1e-12


def _diff(reference: Any, candidate: Any, path: str, out: List[str]) -> int:
    """Every leaf of `reference` must exist in `candidate` with an equal value. Returns leaves compared."""
    if isinstance(reference, dict):
        if not isinstance(candidate, dict):
            out.append(f"{path}: not a dict")
            return 0
        n = 0
        for k, v in reference.items():
            if k not in candidate:
                out.append(f"{path}/{k}: missing")
                continue
            n += _diff(v, candidate[k], f"{path}/{k}", out)
        return n
    if isinstance(reference, list):
        if not isinstance(candidate, list) or len(reference) != len(candidate):
            out.append(f"{path}: list differs")
            return 0
        return sum(_diff(a, b, f"{path}[{i}]", out) for i, (a, b) in enumerate(zip(reference, candidate)))
    if isinstance(reference, float) and isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
        if not math.isclose(reference, candidate, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            out.append(f"{path}: {reference!r} != {candidate!r}")
        return 1
    if reference != candidate:
        out.append(f"{path}: {reference!r} != {candidate!r}")
    return 1


def _added_keys(reference: Any, candidate: Any, path: str = "") -> List[str]:
    if not (isinstance(reference, dict) and isinstance(candidate, dict)):
        return []
    added = [f"{path}/{k}" for k in candidate if k not in reference]
    for k in reference:
        if k in candidate:
            added += _added_keys(reference[k], candidate[k], f"{path}/{k}")
    return added


def _change(v1: float, v2: float) -> Dict[str, Any]:
    return {"v1": v1, "v2": v2, "change": v2 - v1, "percent_change": (100.0 * (v2 - v1) / v1) if v1 else None}


def _zero_vs_median(zero: Dict[str, Any], med: Dict[str, Any]) -> Dict[str, Any]:
    def one(z: Dict[str, Any], m: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "zero_mae": z["mae"], "median_mae": m["mae"],
            "mae_improvement": z["mae"] - m["mae"], "percent_mae_improvement": 100.0 * (z["mae"] - m["mae"]) / z["mae"] if z["mae"] else None,
            "zero_rmse": z["rmse"], "median_rmse": m["rmse"],
            "rmse_change": m["rmse"] - z["rmse"], "percent_rmse_change": 100.0 * (m["rmse"] - z["rmse"]) / z["rmse"] if z["rmse"] else None,
            "median_better_mae": m["mae"] < z["mae"], "median_better_rmse": m["rmse"] < z["rmse"],
        }
    v = zero["validation"], med["validation"]
    return {
        "overall": one(v[0]["overall"], v[1]["overall"]),
        "by_variant": {name: one(v[0]["by_variant"][name], v[1]["by_variant"][name]) for name in VARIANTS},
    }


def _tail_row(tail: Dict[str, Any]) -> Dict[str, Any]:
    q = tail["absolute_error_quantiles"]
    return {**q, "max": tail["max_absolute_error"],
            **{k: v["count"] for k, v in tail["absolute_error_exceeding"].items()},
            **{f"{k}_percent": v["percent"] for k, v in tail["absolute_error_exceeding"].items()},
            "example_count": tail["example_count"]}


def main() -> int:
    # ---- 1. v1 reproduction against the historical files
    z1 = ZB.compute("axis2_v1")
    m1 = MB.compute("axis2_v1", z1, "ml/results/axis2_v1/zero_baseline.json")
    reproduction = {}
    for label, fresh, path in (("zero_baseline", z1, V1_DIR / "zero_baseline.json"),
                               ("median_cents_baseline", m1, V1_DIR / "median_cents_baseline.json")):
        hist = json.loads(path.read_text(encoding="utf-8"))
        mismatches: List[str] = []
        leaves = _diff(hist, fresh, "", mismatches)
        reproduction[label] = {
            "historical_file": str(path.relative_to(REPO)),
            "historical_leaves_compared": leaves,
            "mismatches": mismatches,
            "reproduced": not mismatches,
            "tolerance": {"rel": REL_TOL, "abs": ABS_TOL},
            "keys_added_by_current_code": _added_keys(hist, fresh),
        }
    if not all(r["reproduced"] for r in reproduction.values()):
        raise SystemExit(f"v1 reproduction FAILED: {json.dumps(reproduction, indent=1)[:2000]}")

    # ---- 2. v2 files equal a fresh in-memory computation
    z2 = ZB.compute("axis2_v2")
    m2 = MB.compute("axis2_v2", z2, "ml/results/axis2_v2/zero_baseline.json")
    for label, fresh in (("zero_baseline", z2), ("median_cents_baseline", m2)):
        saved = json.loads((V2_DIR / f"{label}.json").read_text(encoding="utf-8"))
        mm: List[str] = []
        _diff(saved, fresh, "", mm)
        _diff(fresh, saved, "", mm)
        if mm:
            raise SystemExit(f"{label}: saved v2 results differ from recomputation: {mm[:5]}")

    for r in (z1, m1, z2, m2):
        assert r["test_evaluated"] is False

    # ---- 3. comparison
    def ov(res, split="validation"):
        return res[split]["overall"]

    validation_table = {
        "zero_mae": _change(ov(z1)["mae"], ov(z2)["mae"]),
        "zero_rmse": _change(ov(z1)["rmse"], ov(z2)["rmse"]),
        "median_mae": _change(ov(m1)["mae"], ov(m2)["mae"]),
        "median_rmse": _change(ov(m1)["rmse"], ov(m2)["rmse"]),
        "median_median_absolute_error": _change(ov(m1)["median_absolute_error"], ov(m2)["median_absolute_error"]),
        "median_mean_signed_error": _change(ov(m1)["mean_signed_error"], ov(m2)["mean_signed_error"]),
    }
    t1, t2 = m1["validation"]["tail"], m2["validation"]["tail"]
    for q in ("p90", "p95", "p99", "p99_9"):
        validation_table[f"median_{q}_abs_error"] = _change(t1["absolute_error_quantiles"][q], t2["absolute_error_quantiles"][q])
    validation_table["median_max_abs_error"] = _change(t1["max_absolute_error"], t2["max_absolute_error"])

    severity = {name: _change(m1["validation"]["by_variant"][name]["mae"], m2["validation"]["by_variant"][name]["mae"])
                for name in VARIANTS}
    severity_rmse = {name: _change(m1["validation"]["by_variant"][name]["rmse"], m2["validation"]["by_variant"][name]["rmse"])
                     for name in VARIANTS}

    tails = {
        version: {
            split: {"zero": _tail_row(z[split]["tail"]), "median_cents": _tail_row(m[split]["tail"])}
            for split in ("train", "validation")
        }
        for version, z, m in (("axis2_v1", z1, m1), ("axis2_v2", z2, m2))
    }
    exceed = {
        f"gt_{t}_cents": {
            "v1_count": t1["absolute_error_exceeding"][f"gt_{t}_cents"]["count"],
            "v2_count": t2["absolute_error_exceeding"][f"gt_{t}_cents"]["count"],
            "v1_percent": t1["absolute_error_exceeding"][f"gt_{t}_cents"]["percent"],
            "v2_percent": t2["absolute_error_exceeding"][f"gt_{t}_cents"]["percent"],
        }
        for t in ("25", "50", "100", "200", "400")
    }

    def train_val_gap(m):
        tr, va = m["train"]["overall"], m["validation"]["overall"]
        return {"train_mae": tr["mae"], "validation_mae": va["mae"], "mae_gap_validation_minus_train": va["mae"] - tr["mae"],
                "train_rmse": tr["rmse"], "validation_rmse": va["rmse"], "rmse_gap_validation_minus_train": va["rmse"] - tr["rmse"],
                "interpretation": "no learned parameters: the gap reflects which songs and notes each split holds, not overfitting"}

    comparison = {
        "description": "Baselines 1 and 2, axis2_v1 (segmentation v1) vs axis2_v2 (segmentation v2). Train/validation only.",
        "test_evaluated": False,
        "datasets": {
            v: {"clean_notes": {s: r[s]["by_variant"]["clean"]["example_count"] for s in ("train", "validation")},
                "examples": {s: r[s]["overall"]["example_count"] for s in ("train", "validation")}}
            for v, r in (("axis2_v1", z1), ("axis2_v2", z2))
        },
        "v1_reproduction": reproduction,
        "v1_tail_source": "recomputed from data/ml_dataset/axis2_v1 with the current (reproduction-verified) code; the historical v1 JSON predates tail statistics",
        "validation": validation_table,
        "median_by_severity_validation_mae": severity,
        "median_by_severity_validation_rmse": severity_rmse,
        "median_by_sign_validation": {
            v: {s: m["validation"]["by_sign"][s] for s in ("positive", "negative")} for v, m in (("axis2_v1", m1), ("axis2_v2", m2))
        },
        "median_validation_exceedances": exceed,
        "tails": tails,
        "zero_vs_median_validation": {"axis2_v1": _zero_vs_median(z1, m1), "axis2_v2": _zero_vs_median(z2, m2)},
        "median_train_vs_validation": {"axis2_v1": train_val_gap(m1), "axis2_v2": train_val_gap(m2)},
        "median_sanity": {
            v: {"group_sanity_checks": m["group_sanity_checks"], "variant_identity_checks": m["variant_identity_checks"]}
            for v, m in (("axis2_v1", m1), ("axis2_v2", m2))
        },
        "median_song_extremes_validation": {v: m["validation"]["song_extremes"] for v, m in (("axis2_v1", m1), ("axis2_v2", m2))},
    }
    (V2_DIR / "v1_vs_v2_baseline_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    (V2_DIR / "v1_vs_v2_baseline_comparison.md").write_text(_markdown(comparison, m2), encoding="utf-8")
    print(json.dumps({k: comparison[k] for k in ("validation", "median_validation_exceedances")}, indent=1))
    print("v1 reproduction:", {k: (v["reproduced"], v["historical_leaves_compared"]) for k, v in reproduction.items()})
    return 0


def _markdown(c: Dict[str, Any], m2: Dict[str, Any]) -> str:
    f = lambda x, d=4: "—" if x is None else f"{x:.{d}f}"
    L = ["# Baselines 1–2: axis2_v1 vs axis2_v2 (validation)", "",
         "Generated by `ml/baselines/compare_dataset_versions.py`. Train and validation only; **test not evaluated**.", "",
         "| validation | v1 | v2 | change | % |", "|---|---|---|---|---|"]
    names = {"zero_mae": "Zero MAE", "zero_rmse": "Zero RMSE", "median_mae": "Median MAE", "median_rmse": "Median RMSE",
             "median_median_absolute_error": "Median median abs error", "median_mean_signed_error": "Median mean signed error",
             "median_p90_abs_error": "Median P90 abs error", "median_p95_abs_error": "Median P95 abs error",
             "median_p99_abs_error": "Median P99 abs error", "median_p99_9_abs_error": "Median P99.9 abs error",
             "median_max_abs_error": "Median max abs error"}
    for k, label in names.items():
        r = c["validation"][k]
        L.append(f"| {label} | {f(r['v1'])} | {f(r['v2'])} | {r['change']:+.4f} | {f(r['percent_change'], 2)} |")
    L += ["", "**Median-cents validation absolute error above threshold**", "", "| | v1 | v2 |", "|---|---|---|"]
    for k, r in c["median_validation_exceedances"].items():
        L.append(f"| {k.replace('gt_', '> ').replace('_cents', ' ¢')} | {r['v1_count']} ({r['v1_percent']:.2f}%) | {r['v2_count']} ({r['v2_percent']:.2f}%) |")
    L += ["", "**Zero vs median within each version (validation)**", "", "| | zero MAE | median MAE | zero RMSE | median RMSE |", "|---|---|---|---|---|"]
    for v in ("axis2_v1", "axis2_v2"):
        o = c["zero_vs_median_validation"][v]["overall"]
        L.append(f"| {v} | {f(o['zero_mae'])} | {f(o['median_mae'])} | {f(o['zero_rmse'])} | {f(o['median_rmse'])} |")
    L += ["", "**Median-cents validation by song, axis2_v2** (fit of this naive estimator to each recording's extracted note centers; not a ranking of singers)", "",
          "| song | notes | MAE | RMSE | median abs | signed | P95 | P99 | max |", "|---|---|---|---|---|---|---|---|---|"]
    for song, r in sorted(m2["validation"]["by_song"].items(), key=lambda kv: kv[1]["mae"]):
        L.append(f"| {song} | {r['clean_note_count']} | {f(r['mae'])} | {f(r['rmse'])} | {f(r['median_absolute_error'])} | "
                 f"{r['mean_signed_error']:+.3f} | {f(r['p95_absolute_error'], 2)} | {f(r['p99_absolute_error'], 2)} | {f(r['max_absolute_error'], 2)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
