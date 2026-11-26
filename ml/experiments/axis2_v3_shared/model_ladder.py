#!/usr/bin/env python3
"""
Assemble the axis2_v3 model ladder from the stored result files. Evaluates nothing.

Reads the V3 (and, for the historical table, V2) results of the zero, full-median,
core-median and Ridge baselines and of the natural-offset, fixed-hybrid and
dynamic-hybrid experiments, and writes

  ml/results/axis2_v3/axis2_v3_model_ladder.json
  ml/results/axis2_v3/axis2_v3_model_ladder.md

Two numbers are computed from saved artifacts rather than read: the maximum
|observed production core| over V3 train + validation (from examples.npz), and the
V2 dynamic-vs-fixed gain restricted to |observed core| <= 50 (from the stored V2
gate validation predictions). No test row is read.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "ml" / "results" / "axis2_v3"
VARIANTS = ("clean", "corruption_1", "corruption_2", "corruption_3", "corruption_4")
LEARNED = ("ridge", "natural", "fixed", "magnitude", "dynamic")
LABEL = {"zero": "Zero", "median": "Full median", "core": "Core median", "ridge": "Ridge", "natural": "Natural offset",
         "fixed": "Fixed hybrid", "magnitude": "Magnitude gate", "dynamic": "Dynamic hybrid"}


def J(p):
    return json.loads((REPO / p).read_text())


def baseline_row(v):
    o, t = v["overall"], v["tail"]
    q = t["absolute_error_quantiles"]
    return {"mae": o["mae"], "rmse": o["rmse"], "median_absolute_error": o["median_absolute_error"], "mean_signed_error": o["mean_signed_error"],
            "p90": q["p90"], "p95": q["p95"], "p99": q["p99"], "p99.5": q["p99_5"], "p99.9": q["p99_9"], "max": t["max_absolute_error"],
            "gt_25": t["absolute_error_exceeding"]["gt_25_cents"], "gt_50": t["absolute_error_exceeding"]["gt_50_cents"],
            "by_variant_mae": {n: v["by_variant"][n]["mae"] for n in VARIANTS}, "slope": None}


def learned_row(dv, k):
    o = dv["overall"][k]
    return {"mae": o["mae"], "rmse": o["rmse"], "median_absolute_error": o["median_absolute_error"], "mean_signed_error": o["mean_signed_error"],
            **{q: o["quantiles"][q] for q in ("p90", "p95", "p99", "p99.5", "p99.9")}, "max": o["max"],
            "gt_25": {"count": o["exceeding"]["gt_25_cents"], "percent": 100 * o["exceeding"]["gt_25_cents"] / o["count"]},
            "gt_50": {"count": o["exceeding"]["gt_50_cents"], "percent": 100 * o["exceeding"]["gt_50_cents"] / o["count"]},
            "by_variant_mae": {n: dv["by_variant"][n][k]["mae"] for n in VARIANTS},
            "by_variant_rmse": {n: dv["by_variant"][n][k]["rmse"] for n in VARIANTS},
            "slope": dv["prediction_slope"][k]}


def ladder(version: str):
    rd = f"ml/results/{version}"
    ed = "axis2_v2" if version == "axis2_v2" else "axis2_v3"
    dyn = J(f"ml/experiments/{ed}_dynamic_hybrid/dynamic_hybrid_results.json")
    rows = {"zero": baseline_row(J(f"{rd}/zero_baseline.json")["validation"]),
            "median": baseline_row(J(f"{rd}/median_cents_baseline.json")["validation"]),
            "core": baseline_row(J(f"{rd}/core_median_baseline.json")["validation"])}
    rows.update({k: learned_row(dyn["validation"], k) for k in LEARNED})
    return rows, dyn


def main() -> int:
    v3, dyn3 = ladder("axis2_v3")
    v2, dyn2 = ladder("axis2_v2")
    for name, d in (("v3", dyn3), ("v2", dyn2)):
        if d["test_evaluated"] is not False:
            raise RuntimeError(f"{name} dynamic result is not test-sealed")
    rid3, nat3, hyb3 = J("ml/results/axis2_v3/ridge_baseline.json"), J("ml/experiments/axis2_v3_natural_offset/natural_offset_results.json"), \
        J("ml/experiments/axis2_v3_hybrid/hybrid_results.json")
    hyb2 = J("ml/experiments/axis2_v2_hybrid/hybrid_results.json")
    files_sealed = {p: J(p)["test_evaluated"] is False for p in (
        "ml/results/axis2_v3/zero_baseline.json", "ml/results/axis2_v3/median_cents_baseline.json", "ml/results/axis2_v3/core_median_baseline.json",
        "ml/results/axis2_v3/ridge_baseline.json", "ml/experiments/axis2_v3_natural_offset/natural_offset_results.json",
        "ml/experiments/axis2_v3_hybrid/hybrid_results.json", "ml/experiments/axis2_v3_dynamic_hybrid/dynamic_hybrid_results.json")}

    # max |observed core| on V3 train + validation only
    with np.load(REPO / "data/ml_dataset/axis2_v3/examples.npz", allow_pickle=False) as z:
        split, core = z["split_index"], z["core_median_cents"]
        names = [str(s) for s in z["split_names"]]
    tv = np.isin(split, [names.index("train"), names.index("validation")])
    core_check = {"max_abs_observed_core_train_validation": float(np.abs(core[tv]).max()),
                  "examples_abs_core_ge_50_train_validation": int(np.sum(np.abs(core[tv]) >= 50)), "below_50": bool(np.abs(core[tv]).max() < 50)}

    # V2 dynamic-vs-fixed gain on the |core| <= 50 subset (the part that can exist under V3's definition)
    rows = list(csv.DictReader((REPO / "ml/experiments/axis2_v2_dynamic_hybrid/gate_validation_predictions.csv").open()))
    y = np.array([float(r["true_shift"]) for r in rows]); c = np.array([float(r["observed_core_median"]) for r in rows])
    fx = np.abs(np.array([float(r["fixed_hybrid_prediction"]) for r in rows]) - y)
    dy = np.abs(np.array([float(r["dynamic_gate_prediction"]) for r in rows]) - y)
    lo = np.abs(c) <= 50
    v2_split = {"all": {"fixed": float(fx.mean()), "dynamic": float(dy.mean()), "gain": float(fx.mean() - dy.mean())},
                "abs_core_le_50": {"n": int(lo.sum()), "fixed": float(fx[lo].mean()), "dynamic": float(dy[lo].mean()),
                                   "gain": float(fx[lo].mean() - dy[lo].mean())},
                "share_of_gain_from_abs_core_gt_50": float((fx - dy)[~lo].sum() / (fx - dy).sum())}

    best = min(LEARNED + ("zero", "median", "core"), key=lambda k: v3[k]["mae"])
    best_rmse = min(LEARNED + ("zero", "median", "core"), key=lambda k: v3[k]["rmse"])
    pq = dyn3["validation"]["primary_question"]
    dv = dyn3["validation"]
    res = {
        "dataset": "axis2_v3", "test_evaluated": False, "source": "assembled from stored result files; evaluates nothing",
        "all_component_results_test_sealed": all(files_sealed.values()), "component_results": files_sealed,
        "validation_ladder": v3, "best_mae_model": best, "best_rmse_model": best_rmse,
        "best_vs_ridge": {"mae": v3["ridge"]["mae"] - v3[best]["mae"], "percent": 100 * (v3["ridge"]["mae"] - v3[best]["mae"]) / v3["ridge"]["mae"],
                          "rmse": v3["ridge"]["rmse"] - v3[best]["rmse"], "rmse_percent": 100 * (v3["ridge"]["rmse"] - v3[best]["rmse"]) / v3["ridge"]["rmse"]},
        "dynamic_vs_fixed": pq["dynamic_gate"], "magnitude_vs_fixed": pq["magnitude_gate"],
        "magnitude_share_of_dynamic_gain": pq["magnitude_gate"]["mae_improvement"] / pq["dynamic_gate"]["mae_improvement"],
        "fixed_weight": {"v2_ridge": hyb2["blend"]["selected_weight_ridge"], "v3_ridge": hyb3["blend"]["selected_weight_ridge"]},
        "ridge": {"frozen_alpha": rid3["hyperparameter_selection"]["selected_alpha"], "alpha_source": rid3["hyperparameter_selection"].get("alpha_frozen_from"),
                  "train": rid3["train"]["overall"], "cv_mae_at_frozen_alpha": rid3["hyperparameter_selection"]["selected_mean_cv_mae"],
                  "center_only_validation": rid3["center_only_ablation"]["validation"]["overall"],
                  "center_only_alpha": rid3["center_only_ablation"]["hyperparameter_selection"]["selected_alpha"]},
        "natural": {"clean_N_hat": nat3["natural_offset_validation"]["selected"],
                    "sibling_N_hat_spread": nat3["synthetic_shift_validation"]["structural_identity"]["max_within_note_N_hat_spread"]},
        "core_check": core_check,
        "dynamic_weights": {"distribution": dv["weight_distribution"]["dynamic"], "by_variant": dv["mean_ridge_weight_by_variant"]["dynamic"],
                            "by_abs_core": {k: x.get("dynamic_mean_ridge_weight") for k, x in dv["by_abs_observed_core_median"].items()},
                            "core_buckets": list(dv["by_abs_observed_core_median"])},
        "disagreement": dv["by_abs_ridge_minus_natural"], "note_groups": dv["note_groups"], "by_sign": dv["by_sign_corrupted"],
        "v2_vs_v3": {k: {"v2": v2[k]["mae"], "v3": v3[k]["mae"]} for k in ("zero", "median", "core", "ridge", "natural", "fixed", "magnitude", "dynamic")},
        "v2_vs_v3_slope": {k: {"v2": v2[k]["slope"]["slope"], "v3": v3[k]["slope"]["slope"]} for k in LEARNED},
        "v2_dynamic_gain_split": v2_split,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "axis2_v3_model_ladder.json").write_text(json.dumps(res, indent=2) + "\n")
    (OUT / "axis2_v3_model_ladder.md").write_text(md(res))
    print(json.dumps({"best": best, "mae": v3[best]["mae"], "sealed": res["all_component_results_test_sealed"]}, indent=1))
    return 0


def md(r) -> str:
    v = r["validation_ladder"]
    f = lambda x, d=3: "—" if x is None else f"{x:.{d}f}"  # noqa: E731
    order = ("zero", "median", "core", "ridge", "natural", "fixed", "magnitude", "dynamic")
    bm, br = r["best_mae_model"], r["best_rmse_model"]
    L = ["# Axis 2 V3 model ladder (validation)", "",
         "Assembled by `ml/experiments/axis2_v3_shared/model_ladder.py` from stored results. Frozen V2 model definitions; "
         f"fixed-hybrid weight and gate settings re-selected on V3 train OOF only. **test_evaluated: false.**", "",
         "## Main table", "", "| model | MAE | RMSE | median AE | mean signed | slope |", "|---|---|---|---|---|---|"]
    for k in order:
        x = v[k]
        mae = f"**{f(x['mae'], 4)}**" if k == bm else f(x["mae"], 4)
        rmse = f"**{f(x['rmse'], 4)}**" if k == br else f(x["rmse"], 4)
        L.append(f"| {LABEL[k]} | {mae} | {rmse} | {f(x['median_absolute_error'])} | {x['mean_signed_error']:+.3f} | "
                 f"{f(x['slope']['slope']) if x['slope'] else 'n/a'} |")
    L += ["", "## MAE by variant", "", "| model | clean | c1 | c2 | c3 | c4 |", "|---|---|---|---|---|---|"]
    for k in order:
        L.append(f"| {LABEL[k]} | " + " | ".join(f(v[k]["by_variant_mae"][n], 2) for n in VARIANTS) + " |")
    L += ["", "## Prediction slope (corrupted validation)", "", "| model | slope | intercept | R² |", "|---|---|---|---|"]
    for k in LEARNED:
        s = v[k]["slope"]
        L.append(f"| {LABEL[k]} | {f(s['slope'])} | {s['intercept']:+.3f} | {f(s['r2'])} |")
    L += ["", "## Sign (corrupted validation)", "", "| model | + n | + MAE | + RMSE | + signed | − n | − MAE | − RMSE | − signed |", "|---|---|---|---|---|---|---|---|---|"]
    for k in LEARNED:
        p_, n_ = r["by_sign"]["positive"][k], r["by_sign"]["negative"][k]
        L.append(f"| {LABEL[k]} | {p_['count']} | {f(p_['mae'])} | {f(p_['rmse'])} | {p_['mean_signed_error']:+.3f} | "
                 f"{n_['count']} | {f(n_['mae'])} | {f(n_['rmse'])} | {n_['mean_signed_error']:+.3f} |")
    L += ["", "## Tail", "", "| model | P90 | P95 | P99 | P99.5 | P99.9 | max | > 25 ¢ | > 50 ¢ |", "|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        x = v[k]
        L.append(f"| {LABEL[k]} | " + " | ".join(f(x[q], 2) for q in ("p90", "p95", "p99", "p99.5", "p99.9", "max"))
                 + f" | {x['gt_25']['count']} ({x['gt_25']['percent']:.2f}%) | {x['gt_50']['count']} ({x['gt_50']['percent']:.2f}%) |")
    w = r["dynamic_weights"]
    L += ["", "## Dynamic gate Ridge weight", "", "| mean | std | P10 | P25 | P50 | P75 | P90 | min | max |", "|---|---|---|---|---|---|---|---|---|",
          "| " + " | ".join(f(w["distribution"][q]) for q in ("mean", "std", "p10", "p25", "p50", "p75", "p90", "min", "max")) + " |", "",
          "| variant | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * 5, "| w̄ | " + " | ".join(f(w["by_variant"][n]) for n in VARIANTS) + " |", "",
          "| \\|core\\| | " + " | ".join(w["core_buckets"]) + " |", "|---|" + "---|" * len(w["core_buckets"]),
          "| w̄ | " + " | ".join(f(w["by_abs_core"][b]) for b in w["core_buckets"]) + " |", "",
          f"Max |observed core| on V3 train+validation: {r['core_check']['max_abs_observed_core_train_validation']:.4f} ¢ "
          f"(examples ≥ 50 ¢: {r['core_check']['examples_abs_core_ge_50_train_validation']}); no 50+ bucket exists.", "",
          "## Expert disagreement |ridge − natural|", "", "| bucket | n | Ridge | Natural | Fixed | Dynamic | w̄ dynamic |", "|---|---|---|---|---|---|---|"]
    for b, x in r["disagreement"].items():
        L.append(f"| {b} | {x['count']} | {f(x['ridge_mae'])} | {f(x['natural_mae'])} | {f(x['fixed_mae'])} | {f(x['dynamic_mae'])} | {f(x['dynamic_mean_ridge_weight'])} |")
    ng = r["note_groups"]
    a, b = ng["dynamic_vs_fixed"], ng["dynamic_vs_ridge"]
    L += ["", f"## Note groups ({ng['note_groups']} validation notes, tie within {ng['tie_tolerance_cents']} ¢)", "",
          f"- Dynamic vs fixed: dynamic better {a['dynamic_better_percent']:.1f}%, tie {a['tie_percent']:.1f}%, fixed better {a['fixed_better_percent']:.1f}%; "
          f"median gain {f(a['median_gain_when_dynamic_wins'])} ¢, median loss {f(a['median_loss_when_fixed_wins'])} ¢.",
          f"- Dynamic vs Ridge: dynamic better {b['dynamic_better_percent']:.1f}%, tie {b['tie_percent']:.1f}%, Ridge better {b['ridge_better_percent']:.1f}%.", "",
          "## V2 → V3 (validation MAE; the task changed, so this is not a ranking of datasets)", "", "| model | V2 | V3 |", "|---|---|---|"]
    for k, x in r["v2_vs_v3"].items():
        L.append(f"| {LABEL[k]} | {f(x['v2'], 4)} | {f(x['v3'], 4)} |")
    s2 = r["v2_dynamic_gain_split"]
    L += ["", f"Fixed-hybrid weight: V2 {r['fixed_weight']['v2_ridge']:.2f} Ridge / {1 - r['fixed_weight']['v2_ridge']:.2f} natural; "
              f"V3 {r['fixed_weight']['v3_ridge']:.2f} / {1 - r['fixed_weight']['v3_ridge']:.2f}.",
          "Slopes V2 → V3: " + ", ".join(f"{LABEL[k]} {x['v2']:.3f} → {x['v3']:.3f}" for k, x in r["v2_vs_v3_slope"].items()) + ".", "",
          f"V2 dynamic-vs-fixed gain: {s2['all']['gain']:.4f} ¢ overall, of which {100 * s2['share_of_gain_from_abs_core_gt_50']:.1f}% came from "
          f"|core| > 50; on the |core| ≤ 50 subset it was {s2['abs_core_le_50']['gain']:.4f} ¢. V3 dynamic-vs-fixed gain: "
          f"{r['dynamic_vs_fixed']['mae_improvement']:.4f} ¢ ({r['dynamic_vs_fixed']['mae_improvement_percent']:.2f}%).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
