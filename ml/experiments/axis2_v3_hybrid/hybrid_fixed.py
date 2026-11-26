#!/usr/bin/env python3
"""
Fixed convex blend of the two frozen experts, dataset-parameterized (controlled rerun).

    hybrid = w * ridge + (1 - w) * natural,   w one global scalar in {0.00, 0.01, ..., 1.00}

Same procedure as ml/experiments/axis2_v2_hybrid/hybrid_ridge_natural.py, whose
helpers, buckets, and report writer are imported. Only the dataset is a
parameter. The experts are frozen (V2-selected definitions); w is RE-SELECTED on
the given dataset's train out-of-fold predictions (GroupKFold(5) by song, one
partition shared by both experts), tie -> nearest 0.5. Validation is predicted
once, after w is fixed.

Freeze checks against the given dataset's own expert results: per-fold CV MAEs,
validation MAE/RMSE, and the natural model's per-example validation predictions.

  python hybrid_fixed.py --dataset axis2_v2 --out /tmp/x \\
      --ridge-results ml/results/axis2_v2/ridge_baseline.json \\
      --natural-results ml/experiments/axis2_v2_natural_offset/natural_offset_results.json \\
      --natural-predictions ml/experiments/axis2_v2_natural_offset/validation_predictions.csv
  (axis2_v3: the same with ml/results/axis2_v3/ and ml/experiments/axis2_v3_natural_offset/)

The test split is sealed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "axis2_v3_shared"))

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import frozen_experts as FE  # noqa: E402

H, NAT = FE.H, FE.NAT
V2_WEIGHT_RIDGE = 0.56   # context only; never used as the blend weight here


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=FE.DATASETS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ridge-results", type=Path, required=True)
    p.add_argument("--natural-results", type=Path, required=True)
    p.add_argument("--natural-predictions", type=Path, required=True)
    args = p.parse_args()
    t0 = time.time()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in H.OUTPUTS:
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; refusing to overwrite")
    protected = FE.protected_paths()
    hashes_before = NAT._hash_tree(protected)

    ridge_cfg, nat_cfg = H.frozen_configs()
    refs = FE.expert_references(args.ridge_results, args.natural_results, ridge_cfg["alpha"], nat_cfg)
    d = FE.load(args.dataset)
    E = FE.prepare_experts(d, ridge_cfg, nat_cfg, refs)
    D, vc = d["D"], d["variant_codes"]
    tr, va = d["tr"], d["va"]
    y_tr, y_va, clean_tr = E["y_tr"], E["y_va"], E["clean_tr"]
    groups_tr, fold_of = E["groups_tr"], E["fold_of"]
    ridge_oof, natural_oof = E["ridge_oof"], E["natural_oof"]
    ridge_va, natural_va = E["ridge_va"], E["natural_va"]

    # ---- weight sweep on train OOF
    corr_tr = ~clean_tr
    sweep = []
    for w in H.WEIGHTS:
        pr = w * ridge_oof + (1.0 - w) * natural_oof
        m = H.metrics(pr - y_tr)
        sweep.append({"weight_ridge": w, "weight_natural": round(1.0 - w, 2), "oof_mae": m["mae"], "oof_rmse": m["rmse"],
                      "median_absolute_error": m["median_absolute_error"], "mean_signed_error": m["mean_signed_error"],
                      "prediction_slope": NAT.slope_stats(y_tr[corr_tr], pr[corr_tr])["slope"]})
    best_mae = min(r["oof_mae"] for r in sweep)
    tied = [r for r in sweep if r["oof_mae"] - best_mae <= H.TIE_TOLERANCE_CENTS]
    chosen = min(tied, key=lambda r: (abs(r["weight_ridge"] - 0.5), r["weight_ridge"]))
    w = chosen["weight_ridge"]
    print(f"[{args.dataset}] selected w_ridge = {w}", flush=True)
    with (out / "weight_sweep.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sweep[0].keys()))
        wr.writeheader()
        for r in sweep:
            wr.writerow({k: (f"{v:.6f}" if isinstance(v, float) and k not in ("weight_ridge", "weight_natural") else v) for k, v in r.items()})
    hybrid_oof = w * ridge_oof + (1.0 - w) * natural_oof
    oof = {name: H.metrics(pr - y_tr) | {"slope_corrupted": NAT.slope_stats(y_tr[corr_tr], pr[corr_tr])["slope"]}
           for name, pr in (("ridge", ridge_oof), ("natural", natural_oof), ("hybrid", hybrid_oof))}

    hybrid_va = w * ridge_va + (1.0 - w) * natural_va
    kv = FE.keys(d, va)
    if len(set(kv)) != len(kv):
        raise RuntimeError("duplicate validation key")
    stored_nat = {}
    with args.natural_predictions.open() as fh:
        for row in csv.DictReader(fh):
            stored_nat[(row["song_id"], int(row["source_note_index"]), row["variant"])] = float(row["predicted_shift"])
    if stored_nat.keys() != set(kv):
        raise RuntimeError("validation keys differ from the natural model's stored predictions")
    nat_csv_diff = max(abs(stored_nat[k] - natural_va[j]) for j, k in enumerate(kv))
    if nat_csv_diff > FE.CSV_TOLERANCE:
        raise RuntimeError(f"natural validation predictions do not reproduce the stored CSV ({nat_csv_diff})")

    preds = {"ridge": ridge_va, "natural": natural_va, "hybrid": hybrid_va}
    errs = {k: pr - y_va for k, pr in preds.items()}
    var_va = E["var_va"]
    corr_va = var_va != vc["clean"]
    clean_va = ~corr_va
    mag = np.abs(y_va)
    duration = d["duration"]

    def per_estimator(mask):
        return {k: H.metrics(e[mask]) for k, e in errs.items()}

    overall = {k: H.metrics(e) | {"tail": H.tail(e)} for k, e in errs.items()}
    by_variant = {v: per_estimator(var_va == c) for v, c in vc.items()}
    by_sign = {"positive": per_estimator(corr_va & (y_va > 0)), "negative": per_estimator(corr_va & (y_va < 0))}
    slopes = {k: NAT.slope_stats(y_va[corr_va], pr[corr_va]) for k, pr in preds.items()}
    by_magnitude = {lab: per_estimator(H.in_band(mag, lo, hi)) for lab, lo, hi in H.MAGNITUDE_BANDS}
    by_fine = {lab: per_estimator(corr_va & H.in_band(mag, lo, hi)) for lab, lo, hi in H.FINE_BANDS}
    nframes = D["num_frames"][va].astype(np.float64)
    by_frames = {lab: per_estimator(clean_va & H.in_range(nframes, lo, hi)) for lab, lo, hi in H.FRAME_BUCKETS}
    by_duration = {lab: per_estimator(clean_va & H.in_range(duration[va], lo, hi)) for lab, lo, hi in H.DURATION_BUCKETS}

    gkey = D["song_index"][va].astype(np.int64) * 1_000_000 + D["source_note_index"][va].astype(np.int64)
    uniq, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    if not np.all(counts == 5):
        raise RuntimeError("a validation note group does not have exactly five variants")
    gmae = {k: np.bincount(inv, weights=np.abs(e)) / counts for k, e in errs.items()}
    gain = gmae["ridge"] - gmae["hybrid"]
    wins, losses = gain > H.NOTE_TIE_CENTS, gain < -H.NOTE_TIE_CENTS
    ties = ~(wins | losses)
    note_groups = {
        "note_groups": int(uniq.size), "tie_tolerance_cents": H.NOTE_TIE_CENTS,
        "metric": "per note: mean absolute error over its five variants",
        "hybrid_wins_percent": 100.0 * float(wins.mean()), "tied_percent": 100.0 * float(ties.mean()),
        "ridge_wins_percent": 100.0 * float(losses.mean()),
        "median_improvement_when_hybrid_wins": float(np.median(gain[wins])) if wins.any() else None,
        "median_degradation_when_hybrid_loses": float(np.median(-gain[losses])) if losses.any() else None,
        "mean_note_mae": {k: float(v.mean()) for k, v in gmae.items()},
        "mean_per_note_gain_hybrid_vs_ridge": float(gain.mean()),
        "natural_vs_ridge": {"natural_wins_percent": 100.0 * float(np.mean(gmae["ridge"] - gmae["natural"] > H.NOTE_TIE_CENTS)),
                             "ridge_wins_percent": 100.0 * float(np.mean(gmae["natural"] - gmae["ridge"] > H.NOTE_TIE_CENTS))},
    }
    rmae, hmae = overall["ridge"]["mae"], overall["hybrid"]["mae"]
    rrmse, hrmse = overall["ridge"]["rmse"], overall["hybrid"]["rmse"]
    primary = {"ridge_mae_benchmark": rmae, "hybrid_mae": hmae, "hybrid_beats_ridge_mae": hmae < rmae,
               "mae_improvement": rmae - hmae, "mae_improvement_percent": 100.0 * (rmae - hmae) / rmae,
               "ridge_rmse": rrmse, "hybrid_rmse": hrmse,
               "rmse_improvement": rrmse - hrmse, "rmse_improvement_percent": 100.0 * (rrmse - hrmse) / rrmse}
    comparison = {k: {"mae": overall[k]["mae"], "rmse": overall[k]["rmse"], "median_absolute_error": overall[k]["median_absolute_error"],
                      "p90": overall[k]["tail"]["quantiles"]["p90"], "p95": overall[k]["tail"]["quantiles"]["p95"],
                      "p99": overall[k]["tail"]["quantiles"]["p99"], "max": overall[k]["tail"]["max"],
                      "prediction_slope": slopes[k]["slope"],
                      "by_variant_mae": {v: by_variant[v][k]["mae"] for v in FE.EXPECTED_VARIANT_ORDER}}
                  for k in H.ESTIMATORS}

    with (out / "validation_predictions.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["song_id", "source_note_index", "variant", "frame_count", "duration_s", "injected_shift",
                     "ridge_prediction", "natural_prediction", "hybrid_prediction"])
        for j, i in enumerate(va):
            wr.writerow([kv[j][0], kv[j][1], kv[j][2], int(D["num_frames"][i]), f"{duration[i]:.4f}", f"{y_va[j]:.6f}",
                         f"{ridge_va[j]:.6f}", f"{natural_va[j]:.6f}", f"{hybrid_va[j]:.6f}"])

    hashes_after = NAT._hash_tree(protected)
    fr = E["freeze"]
    results = {
        "experiment": f"{args.dataset}_ridge_natural_fixed_blend",
        "test_evaluated": False,
        "dataset": FE.rel(d["dir"]),
        "components": {"ridge": {k: v for k, v in ridge_cfg.items() if not k.startswith("stored_")},
                       "natural_offset": {k: v for k, v in nat_cfg.items() if not k.startswith("stored_") and k != "columns"},
                       "frozen": "definitions selected on axis2_v2; refit on this dataset's train split; not re-tuned"},
        "blend": {"formula": "w * ridge + (1-w) * natural", "selected_weight_ridge": w,
                  "selected_weight_natural": round(1.0 - w, 2), "selection_metric": "train_oof_mae",
                  "grid": "w = 0.00, 0.01, ..., 1.00 (101 values)", "global_scalar": True,
                  "depends_on": "nothing (no per-note, per-variant, per-song, or magnitude-dependent weighting)",
                  "v2_weight_ridge_context_only": V2_WEIGHT_RIDGE},
        "oof_weight_selection": {
            "method": "GroupKFold(5) by song over all train examples; one song partition shared by both components; "
                      "Ridge refit per fold on all variants of the fit songs, natural model refit per fold on the fit songs' clean notes",
            "folds": [{"fold": k, "heldout_songs": int(np.unique(groups_tr[fold_of == k]).size),
                       "heldout_examples": int(np.sum(fold_of == k)), "fit_examples": int(np.sum(fold_of != k)),
                       "natural_fit_clean_notes": int(np.sum((fold_of != k) & clean_tr))} for k in range(FE.CV_FOLDS)],
            "natural_native_folds_same_song_partition": True,
            "oof_metrics": oof, "selected": chosen,
            "tied_within_tolerance": [r["weight_ridge"] for r in tied], "tie_tolerance_cents": H.TIE_TOLERANCE_CENTS,
            "tie_rule": "nearest to 0.5",
            "slope_definition": "OLS of OOF prediction on injected shift, corrupted train examples only",
        },
        "validation": {"overall": overall, "by_variant": by_variant, "by_sign": by_sign,
                       "by_true_magnitude_band": by_magnitude, "by_true_magnitude_5c_corrupted_only": by_fine,
                       "by_frame_count_clean": by_frames, "by_duration_clean": by_duration,
                       "prediction_slope": {"definition": "OLS predicted_shift ~ injected_shift on corrupted validation examples", **slopes},
                       "notes": {"magnitude_bands": "the bands coincide with the variant definitions, so they equal by_variant; "
                                                    "the 5-cent bins are the finer post-hoc view",
                                 "post_hoc_only": "magnitude and length breakdowns were computed after w was fixed and did not inform it"}},
        "comparison": comparison,
        "primary_question": primary,
        "note_group_comparison": note_groups,
        "integrity": {
            "component_cv_reproduced": {"ridge_max_fold_mae_diff": fr["ridge_max_fold_mae_diff"],
                                        "natural_max_fold_mae_diff": fr["natural_max_fold_mae_diff"]},
            "component_validation_reproduced": {"ridge_max_mae_rmse_diff": fr["ridge_max_validation_mae_rmse_diff"],
                                                "natural_max_mae_rmse_diff": fr["natural_max_validation_mae_rmse_diff"],
                                                "natural_max_per_example_diff_vs_stored_csv": nat_csv_diff},
            "reference_results": fr["reference_results"],
            "oof_alignment": {"examples": int(tr.size)}, "validation_alignment": {"examples": int(va.size)},
            "validation_used_for_weight_selection": False, "validation_prediction_passes": 1,
            "test_examples_featurized": 0, "test_song_note_files_opened": FE.test_songs_opened(d),
            "sealed_test_metadata_only": {"examples": d["splits"]["test_example_count"], "clean_notes": d["splits"]["test_clean_note_count"]},
            "protected_paths": [FE.rel(x) for x in protected],
            "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
            "protected_unchanged": hashes_before == hashes_after, "runtime_s": time.time() - t0,
        },
    }
    if results["integrity"]["test_song_note_files_opened"]:
        raise RuntimeError("a test song's notes were opened")
    (out / "hybrid_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    md = H.summary_md(results).replace("# Axis 2 V2: fixed Ridge + natural-offset blend",
                                       f"# Axis 2 ({args.dataset}): fixed Ridge + natural-offset blend", 1)
    md = md.replace("Generated by `hybrid_ridge_natural.py`.", "Generated by `axis2_v3_hybrid/hybrid_fixed.py` (V2 procedure, frozen experts).", 1)
    (out / "hybrid_summary.md").write_text(md)
    print(json.dumps({"w_ridge": w, "primary": primary, "protected_unchanged": results["integrity"]["protected_unchanged"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
