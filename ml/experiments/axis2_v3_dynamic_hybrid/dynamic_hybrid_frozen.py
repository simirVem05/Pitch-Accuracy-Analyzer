#!/usr/bin/env python3
"""
Dynamic per-example blend of the two frozen experts, dataset-parameterized (controlled rerun).

    prediction = w(x) * ridge + (1 - w(x)) * natural,   w(x) = sigmoid(gate(x)) in [0, 1]

Same procedure as ml/experiments/axis2_v2_dynamic_hybrid/dynamic_hybrid.py. Its
gate class, training loop, gate feature list (20), architecture
(Linear(20,16)-GELU-Linear(16,8)-GELU-Linear(8,1)-sigmoid, 481 parameters),
magnitude gate, hyperparameter grid, seeds, and report writer are imported, not
copied. Only the dataset and the matching expert / fixed-hybrid results are
parameters. Gates are trained on THIS dataset's train out-of-fold expert
predictions (GroupKFold(5) by song); weight decay and epochs are re-selected on
that pooled OOF MAE, exactly as in V2. Validation is predicted once.

The fixed hybrid compared against is the one selected on the same dataset
(--hybrid-results), and its per-example validation predictions are reproduced.

  python dynamic_hybrid_frozen.py --dataset axis2_v2 --out /tmp/x \\
      --ridge-results ml/results/axis2_v2/ridge_baseline.json \\
      --natural-results ml/experiments/axis2_v2_natural_offset/natural_offset_results.json \\
      --hybrid-results ml/experiments/axis2_v2_hybrid/hybrid_results.json \\
      --hybrid-predictions ml/experiments/axis2_v2_hybrid/validation_predictions.csv

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
from typing import Any, Dict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import frozen_experts as FE  # noqa: E402
import dynamic_hybrid as DH  # noqa: E402  (V2 module; on sys.path via frozen_experts)

H, NAT = FE.H, FE.NAT
EXTRA_PROTECTED = ("ml/results/axis2_v3", "ml/experiments/axis2_v3_natural_offset", "ml/experiments/axis2_v3_hybrid")
V3_CORE_BUCKETS = (("0-10", 0, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30-40", 30, 40), ("40-49.999", 40, None))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=FE.DATASETS)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ridge-results", type=Path, required=True)
    p.add_argument("--natural-results", type=Path, required=True)
    p.add_argument("--hybrid-results", type=Path, required=True)
    p.add_argument("--hybrid-predictions", type=Path, required=True)
    p.add_argument("--rewrite-summary", action="store_true",
                   help="only regenerate dynamic_hybrid_summary.md from the existing results JSON (no training, no evaluation)")
    args = p.parse_args()
    if args.rewrite_summary:
        write_summary(json.loads((args.out.resolve() / "dynamic_hybrid_results.json").read_text()), args.out.resolve())
        return 0
    t0 = time.time()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in DH.OUTPUTS:
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; refusing to overwrite")
    for f in DH.GATE_FEATURES:
        if any(bad in f for bad in DH.FORBIDDEN_GATE_INPUTS):
            raise RuntimeError(f"forbidden gate input {f}")
    protected = FE.protected_paths(EXTRA_PROTECTED)
    hashes_before = NAT._hash_tree(protected)
    v3 = args.dataset == "axis2_v3"

    ridge_cfg, nat_cfg = H.frozen_configs()
    refs = FE.expert_references(args.ridge_results, args.natural_results, ridge_cfg["alpha"], nat_cfg)
    hyb = json.loads(args.hybrid_results.read_text())
    if hyb["test_evaluated"] is not False:
        raise RuntimeError("fixed hybrid result is not test-sealed")
    W_FIXED = float(hyb["blend"]["selected_weight_ridge"])
    stored_fixed = {k: hyb["validation"]["overall"]["hybrid"][k] for k in ("mae", "rmse")}
    stored_fixed_oof = hyb["oof_weight_selection"]["oof_metrics"]["hybrid"]["mae"]

    d = FE.load(args.dataset)
    E = FE.prepare_experts(d, ridge_cfg, nat_cfg, refs)
    D, vc = d["D"], d["variant_codes"]
    tr, va = d["tr"], d["va"]
    y_tr, y_va, clean_tr, var_tr, var_va = E["y_tr"], E["y_va"], E["clean_tr"], E["var_tr"], E["var_va"]
    fold_of, groups_tr = E["fold_of"], E["groups_tr"]
    ridge_oof, natural_oof, ridge_va, natural_va = E["ridge_oof"], E["natural_oof"], E["ridge_va"], E["natural_va"]
    core_tr, core_va, Xfull_tr, Xfull_va = E["core_tr"], E["core_va"], E["Xfull_tr"], E["Xfull_va"]

    fixed_oof = W_FIXED * ridge_oof + (1.0 - W_FIXED) * natural_oof
    fixed_oof_diff = abs(float(np.mean(np.abs(fixed_oof - y_tr))) - stored_fixed_oof)
    if fixed_oof_diff > FE.FREEZE_TOLERANCE:
        raise RuntimeError(f"fixed-hybrid OOF MAE does not reproduce ({fixed_oof_diff})")

    G_tr = DH.gate_inputs(ridge_oof, natural_oof, core_tr, Xfull_tr)
    mag_idx = [DH.GATE_FEATURES.index(f) for f in DH.MAGNITUDE_FEATURES]
    gk_tr = D["song_index"][tr].astype(np.int64) * 1_000_000 + D["source_note_index"][tr].astype(np.int64)
    meta_cols = [DH.GATE_FEATURES.index(f) for f in DH.META_FEATURES]
    M = G_tr[np.lexsort((var_tr, gk_tr))][:, meta_cols].reshape(-1, 5, len(meta_cols))
    meta_variant_spread = float(np.max(np.abs(M - M[:, :1, :])))
    if meta_variant_spread != 0.0:
        raise RuntimeError("a metadata gate input differs between variants of one note")

    # ---- gate selection on train OOF rows only (identical to V2)
    gate_specs = {"magnitude": mag_idx, "dynamic": list(range(len(DH.GATE_FEATURES)))}
    selection: Dict[str, Any] = {}
    for kind, cols in gate_specs.items():
        curves, traces = {}, {}
        for wd in DH.WEIGHT_DECAYS:
            print(f"[{args.dataset}] gate CV: {kind} wd={wd}", flush=True)
            tr_w = np.empty((DH.MAX_EPOCHS, tr.size), np.float32)
            for k in range(FE.CV_FOLDS):
                fit, held = fold_of != k, fold_of == k
                _, _, trace = DH.train_gate(kind, G_tr[fit][:, cols], ridge_oof[fit], natural_oof[fit], y_tr[fit], wd,
                                            DH.MAX_EPOCHS, DH.SEED + k, X_eval=G_tr[held][:, cols])
                tr_w[:, held] = trace
            pred = tr_w.astype(np.float64) * ridge_oof + (1.0 - tr_w.astype(np.float64)) * natural_oof
            curves[wd] = np.mean(np.abs(pred - y_tr), axis=1)
            traces[wd] = tr_w
        cands = [(float(curves[wd][e]), e + 1, wd) for wd in DH.WEIGHT_DECAYS for e in range(DH.MAX_EPOCHS)]
        best_mae, best_ep, best_wd = min(cands)
        w_sel = traces[best_wd][best_ep - 1].astype(np.float64)
        p_sel = w_sel * ridge_oof + (1.0 - w_sel) * natural_oof
        selection[kind] = {
            "selected_weight_decay": best_wd, "selected_epochs": best_ep, "oof_mae": best_mae,
            "oof_metrics": DH.short(p_sel - y_tr) | {"slope_corrupted": NAT.slope_stats(y_tr[~clean_tr], p_sel[~clean_tr])["slope"]},
            "oof_mae_by_weight_decay_at_best_epoch": {str(wd): float(curves[wd].min()) for wd in DH.WEIGHT_DECAYS},
            "oof_best_epoch_by_weight_decay": {str(wd): int(np.argmin(curves[wd]) + 1) for wd in DH.WEIGHT_DECAYS},
            "oof_mae_curve": {str(wd): [float(x) for x in curves[wd]] for wd in DH.WEIGHT_DECAYS},
            "oof_weight_distribution": DH.wdist(w_sel),
            "note": "epochs and weight decay were selected on this same pooled OOF MAE, so it is mildly optimistic "
                    f"(best of {len(cands)} candidates)",
        }
        seed_maes = [best_mae]
        for sd in DH.SEED_CHECK:
            ww = np.empty(tr.size)
            for k in range(FE.CV_FOLDS):
                fit, held = fold_of != k, fold_of == k
                m, sc, _ = DH.train_gate(kind, G_tr[fit][:, cols], ridge_oof[fit], natural_oof[fit], y_tr[fit], best_wd, best_ep, sd + k)
                ww[held] = DH.apply_gate(m, sc, G_tr[held][:, cols])
            seed_maes.append(float(np.mean(np.abs(ww * ridge_oof + (1 - ww) * natural_oof - y_tr))))
        selection[kind]["seed_check_oof_mae"] = {"seeds": [DH.SEED, *DH.SEED_CHECK], "oof_mae": seed_maes, "std": float(np.std(seed_maes)),
                                                 "note": "selected config re-run at fixed epochs with other seeds; report only"}
        print(f"  selected wd={best_wd} epochs={best_ep} OOF MAE={best_mae:.4f}", flush=True)

    # ---- final gates; validation predicted once
    fixed_va = W_FIXED * ridge_va + (1.0 - W_FIXED) * natural_va
    G_va = DH.gate_inputs(ridge_va, natural_va, core_va, Xfull_va)
    final_gates, final_scalers, w_va = {}, {}, {}
    for kind, cols in gate_specs.items():
        s = selection[kind]
        m, sc, _ = DH.train_gate(kind, G_tr[:, cols], ridge_oof, natural_oof, y_tr, s["selected_weight_decay"], s["selected_epochs"],
                                 DH.SEED + FE.CV_FOLDS)
        final_gates[kind], final_scalers[kind] = m, sc
        w_va[kind] = DH.apply_gate(m, sc, G_va[:, cols])
        w_in = DH.apply_gate(m, sc, G_tr[:, cols])
        s["final_fit"] = {"rows": int(tr.size), "source": "all train OOF expert predictions", "epochs": s["selected_epochs"],
                          "weight_decay": s["selected_weight_decay"], "seed": DH.SEED + FE.CV_FOLDS,
                          "train_oof_mae_in_sample": float(np.mean(np.abs(w_in * ridge_oof + (1 - w_in) * natural_oof - y_tr)))}

    preds = {"ridge": ridge_va, "natural": natural_va, "fixed": fixed_va,
             "magnitude": w_va["magnitude"] * ridge_va + (1.0 - w_va["magnitude"]) * natural_va,
             "dynamic": w_va["dynamic"] * ridge_va + (1.0 - w_va["dynamic"]) * natural_va}
    weights = {"ridge": np.ones(va.size), "natural": np.zeros(va.size), "fixed": np.full(va.size, W_FIXED),
               "magnitude": w_va["magnitude"], "dynamic": w_va["dynamic"]}
    errs = {k: pr - y_va for k, pr in preds.items()}

    fixed_val_diff = max(abs(DH.short(errs["fixed"])[k] - stored_fixed[k]) for k in ("mae", "rmse"))
    stored_rows = {}
    with args.hybrid_predictions.open() as fh:
        for row in csv.DictReader(fh):
            stored_rows[(row["song_id"], int(row["source_note_index"]), row["variant"])] = (
                float(row["ridge_prediction"]), float(row["natural_prediction"]), float(row["hybrid_prediction"]), float(row["injected_shift"]))
    kv = FE.keys(d, va)
    if len(set(kv)) != len(kv) or set(kv) != stored_rows.keys():
        raise RuntimeError("validation keys differ from the stored fixed-hybrid predictions")
    csv_diff = max(max(abs(stored_rows[k][0] - ridge_va[j]), abs(stored_rows[k][1] - natural_va[j]),
                       abs(stored_rows[k][2] - fixed_va[j]), abs(stored_rows[k][3] - y_va[j])) for j, k in enumerate(kv))
    if fixed_val_diff > FE.FREEZE_TOLERANCE or csv_diff > FE.CSV_TOLERANCE:
        raise RuntimeError(f"fixed hybrid does not reproduce: {fixed_val_diff} csv {csv_diff}")

    # ---- validation analysis (post hoc)
    corr_va = var_va != vc["clean"]
    absdiff = np.abs(ridge_va - natural_va)
    abscore = np.abs(core_va)
    overall = {k: DH.metrics(e) for k, e in errs.items()}
    fx = overall["fixed"]
    benchmark = fx["mae"]

    def vs_fixed(m):
        r = {"mae": m["mae"], "rmse": m["rmse"],
             "mae_improvement": fx["mae"] - m["mae"], "mae_improvement_percent": 100 * (fx["mae"] - m["mae"]) / fx["mae"],
             "rmse_improvement": fx["rmse"] - m["rmse"], "rmse_improvement_percent": 100 * (fx["rmse"] - m["rmse"]) / fx["rmse"]}
        if v3:
            r["beats_fixed_hybrid_mae"] = m["mae"] < benchmark
        else:
            r["beats_fixed_mae_benchmark_12_5828"] = m["mae"] < DH.FIXED_BENCHMARK["mae"]
        return r

    primary = {"fixed_hybrid": {"mae": fx["mae"], "rmse": fx["rmse"]}, "dynamic_gate": vs_fixed(overall["dynamic"]),
               "magnitude_gate": vs_fixed(overall["magnitude"]), "positive_improvement_means": "gate better than fixed"}
    if v3:
        primary["fixed_hybrid"]["selected_weight_ridge"] = W_FIXED
    by_variant = {v: {k: DH.short(errs[k][var_va == c]) for k in DH.MODELS} for v, c in vc.items()}
    slopes = {k: NAT.slope_stats(y_va[corr_va], preds[k][corr_va]) for k in DH.MODELS}
    by_sign = {sgn: {k: DH.short(errs[k][mask]) | {"mean_ridge_weight": float(weights[k][mask].mean())} for k in DH.MODELS}
               for sgn, mask in (("positive", corr_va & (y_va > 0)), ("negative", corr_va & (y_va < 0)))}
    weight_dist = {k: DH.wdist(weights[k]) for k in ("magnitude", "dynamic")}
    w_by_variant = {k: {v: float(weights[k][var_va == c].mean()) for v, c in vc.items()} for k in ("magnitude", "dynamic")}

    def bucket_table(values, buckets):
        res = {}
        for lab, lo, hi in buckets:
            m = DH.bucket(values, lo, hi)
            res[lab] = {"count": int(m.sum()),
                        **({f"{k}_mae": float(np.mean(np.abs(errs[k][m]))) for k in DH.MODELS} if m.any() else {}),
                        **({f"{k}_mean_ridge_weight": float(weights[k][m].mean()) for k in ("magnitude", "dynamic")} if m.any() else {})}
        return res

    core_buckets = DH.CORE_BUCKETS
    n_core_ge_50 = int(np.sum(abscore >= 50.0))
    if v3:
        if n_core_ge_50 or np.any(abscore >= 49.999):
            raise RuntimeError("axis2_v3 has validation examples at or beyond the target boundary")
        core_buckets = V3_CORE_BUCKETS
    by_core = bucket_table(abscore, core_buckets)
    by_disagree = bucket_table(absdiff, DH.DISAGREE_BUCKETS)

    gkey = D["song_index"][va].astype(np.int64) * 1_000_000 + D["source_note_index"][va].astype(np.int64)
    uniq, inv, counts = np.unique(gkey, return_inverse=True, return_counts=True)
    if not np.all(counts == 5):
        raise RuntimeError("a validation note group does not have exactly five variants")
    gmae = {k: np.bincount(inv, weights=np.abs(e)) / counts for k, e in errs.items()}

    def group_cmp(a, b):
        gain = gmae[b] - gmae[a]
        win, loss = gain > DH.NOTE_TIE_CENTS, gain < -DH.NOTE_TIE_CENTS
        return {f"{a}_better_percent": 100 * float(win.mean()), "tie_percent": 100 * float((~win & ~loss).mean()),
                f"{b}_better_percent": 100 * float(loss.mean()),
                f"median_gain_when_{a}_wins": float(np.median(gain[win])) if win.any() else None,
                f"median_loss_when_{b}_wins": float(np.median(-gain[loss])) if loss.any() else None, "mean_gain": float(gain.mean())}

    note_groups = {"note_groups": int(uniq.size), "tie_tolerance_cents": DH.NOTE_TIE_CENTS,
                   "metric": "per note: mean absolute error over its five variants",
                   "dynamic_vs_fixed": group_cmp("dynamic", "fixed"), "dynamic_vs_ridge": group_cmp("dynamic", "ridge"),
                   "magnitude_vs_fixed": group_cmp("magnitude", "fixed"), "mean_note_mae": {k: float(v.mean()) for k, v in gmae.items()}}

    # ---- outputs (same files and columns as V2)
    with (out / "gate_validation_predictions.csv").open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["song_id", "source_note_index", "variant", "true_shift", "observed_core_median", "ridge_prediction", "natural_prediction",
                     "fixed_hybrid_prediction", "magnitude_gate_prediction", "magnitude_ridge_weight", "dynamic_gate_prediction",
                     "dynamic_ridge_weight"])
        for j, k in enumerate(kv):
            wr.writerow([k[0], k[1], k[2], f"{y_va[j]:.6f}", f"{core_va[j]:.6f}", f"{ridge_va[j]:.6f}", f"{natural_va[j]:.6f}",
                         f"{fixed_va[j]:.6f}", f"{preds['magnitude'][j]:.6f}", f"{w_va['magnitude'][j]:.6f}",
                         f"{preds['dynamic'][j]:.6f}", f"{w_va['dynamic'][j]:.6f}"])
    with (out / "gate_weight_analysis.csv").open("w", newline="") as fh:
        cols_out = ["analysis", "bucket", "count", *[f"{k}_mae" for k in DH.MODELS], *[f"{k}_rmse" for k in DH.MODELS],
                    "magnitude_mean_ridge_weight", "dynamic_mean_ridge_weight"]
        wr = csv.DictWriter(fh, fieldnames=cols_out)
        wr.writeheader()
        groups = [("variant", lab, var_va == c) for lab, c in vc.items()]
        groups += [("abs_observed_core_median", lab, DH.bucket(abscore, lo, hi)) for lab, lo, hi in core_buckets]
        groups += [("abs_ridge_minus_natural", lab, DH.bucket(absdiff, lo, hi)) for lab, lo, hi in DH.DISAGREE_BUCKETS]
        groups += [("corrupted_sign", "positive", corr_va & (y_va > 0)), ("corrupted_sign", "negative", corr_va & (y_va < 0)),
                   ("all", "validation", np.ones(va.size, bool))]
        for analysis, lab, m in groups:
            row = {"analysis": analysis, "bucket": lab, "count": int(m.sum())}
            if m.any():
                for k in DH.MODELS:
                    row[f"{k}_mae"] = f"{np.mean(np.abs(errs[k][m])):.6f}"
                    row[f"{k}_rmse"] = f"{np.sqrt(np.mean(errs[k][m] ** 2)):.6f}"
                row["magnitude_mean_ridge_weight"] = f"{weights['magnitude'][m].mean():.6f}"
                row["dynamic_mean_ridge_weight"] = f"{weights['dynamic'][m].mean():.6f}"
            wr.writerow(row)

    mag_lin = final_gates["magnitude"].net
    a_std, b_std = float(mag_lin.bias.item()), float(mag_lin.weight.item())
    mu, sd = float(final_scalers["magnitude"]["mean"][0]), float(final_scalers["magnitude"]["std"][0])
    magnitude_params = {"formula": "w_ridge = sigmoid(a + b * |observed_core_median|)", "a": a_std - b_std * mu / sd, "b_per_cent": b_std / sd,
                        "w_at_abs_core": {str(c): float(1 / (1 + np.exp(-(a_std + b_std * (c - mu) / sd)))) for c in (0, 10, 20, 30, 40, 50, 75)}}
    torch.save({k: {"state_dict": final_gates[k].state_dict(), "scaler_mean": torch.from_numpy(final_scalers[k]["mean"]),
                    "scaler_std": torch.from_numpy(final_scalers[k]["std"]), "features": [DH.GATE_FEATURES[i] for i in gate_specs[k]]}
                for k in gate_specs}, out / "gate_model.pt")
    gate_config = {
        "dynamic": {"features": list(DH.GATE_FEATURES),
                    "feature_sources": {**{f: "computed from expert predictions / observed core median" for f in DH.PRED_FEATURES},
                                        **{f: f"natural-offset extractor column '{c}'" for f, c in DH.META_FEATURES.items()}},
                    "architecture": f"Linear({len(DH.GATE_FEATURES)},{DH.HIDDEN[0]}) -> GELU -> Linear({DH.HIDDEN[0]},{DH.HIDDEN[1]}) -> GELU -> Linear({DH.HIDDEN[1]},1) -> sigmoid",
                    "parameters": DH.n_params(final_gates["dynamic"])},
        "magnitude": {"features": list(DH.MAGNITUDE_FEATURES), "architecture": "Linear(1,1) -> sigmoid",
                      "parameters": DH.n_params(final_gates["magnitude"]), "fitted": magnitude_params},
        "training": {"loss": "L1 of w*ridge + (1-w)*natural against injected shift", "optimizer": "Adam", "learning_rate": DH.LEARNING_RATE,
                     "weight_decay_grid": list(DH.WEIGHT_DECAYS), "batch_size": DH.BATCH_SIZE, "max_epochs": DH.MAX_EPOCHS,
                     "input_standardisation": "z-score fit on the gate's own fit rows", "dtype": "float64", "seed": DH.SEED,
                     "selection": "pooled OOF MAE over the shared song folds; argmin over (weight decay, epoch)",
                     "selected": {k: {"weight_decay": selection[k]["selected_weight_decay"], "epochs": selection[k]["selected_epochs"]}
                                  for k in gate_specs}},
        "fixed": {"weight_ridge": W_FIXED, "source": FE.rel(args.hybrid_results)},
    }
    (out / "gate_config.json").write_text(json.dumps(gate_config, indent=2) + "\n")

    hashes_after = NAT._hash_tree(protected)
    fr = E["freeze"]
    results = {
        "experiment": f"{args.dataset}_ridge_natural_dynamic_gate",
        "test_evaluated": False,
        "dataset": FE.rel(d["dir"]),
        "question": "does a learned per-example blend weight beat the fixed blend selected on the same dataset?",
        "experts": {"ridge": {k: v for k, v in ridge_cfg.items() if not k.startswith("stored_")},
                    "natural_offset": {k: v for k, v in nat_cfg.items() if not k.startswith("stored_") and k != "columns"},
                    "frozen": True, "third_expert": None, "sequence_model_used": False},
        "gates": gate_config,
        "oof_protocol": {
            "method": "GroupKFold(5) by song over all train examples; one partition shared by both experts and by the gate CV; "
                      "Ridge refit per fold on all variants of the fit songs, natural HGB refit per fold on the fit songs' clean notes",
            "folds": [{"fold": k, "heldout_songs": int(np.unique(groups_tr[fold_of == k]).size), "heldout_examples": int(np.sum(fold_of == k))}
                      for k in range(FE.CV_FOLDS)],
            "gate_cv": "for each fold k, a gate is fit on the OOF rows of the other four folds and predicts fold k's OOF rows",
            "final_gate": "refit on all train OOF rows with the selected weight decay and epoch count; applied once to validation "
                          "expert predictions from experts refit on all of train",
            "oof_mae": {"ridge": float(np.mean(np.abs(ridge_oof - y_tr))), "natural": float(np.mean(np.abs(natural_oof - y_tr))),
                        "fixed": float(np.mean(np.abs(fixed_oof - y_tr))), "magnitude": selection["magnitude"]["oof_mae"],
                        "dynamic": selection["dynamic"]["oof_mae"]},
            "selection": selection,
        },
        "validation": {"overall": overall, "primary_question": primary, "by_variant": by_variant,
                       "prediction_slope": {"definition": "OLS predicted_shift ~ injected_shift, corrupted validation examples", **slopes},
                       "weight_distribution": weight_dist, "mean_ridge_weight_by_variant": w_by_variant,
                       "by_abs_observed_core_median": by_core, "by_abs_ridge_minus_natural": by_disagree,
                       "by_sign_corrupted": by_sign, "note_groups": note_groups,
                       "post_hoc_only": "variant, sign, magnitude and disagreement breakdowns were computed after the gates were fixed"},
        "integrity": {
            "expert_cv_reproduced": {"ridge_max_fold_mae_diff": fr["ridge_max_fold_mae_diff"], "natural_max_fold_mae_diff": fr["natural_max_fold_mae_diff"],
                                     "fixed_hybrid_oof_mae_diff": fixed_oof_diff},
            "validation_reproduced": {"ridge_max_mae_rmse_diff": fr["ridge_max_validation_mae_rmse_diff"],
                                      "natural_max_mae_rmse_diff": fr["natural_max_validation_mae_rmse_diff"],
                                      "fixed_max_mae_rmse_diff": fixed_val_diff, "max_per_example_diff_vs_stored_fixed_hybrid_csv": csv_diff},
            "reference_results": {**fr["reference_results"], "fixed_hybrid": FE.rel(args.hybrid_results)},
            "gate_metadata_inputs_identical_across_variants_max_diff": meta_variant_spread,
            "forbidden_gate_inputs_checked": list(DH.FORBIDDEN_GATE_INPUTS),
            "validation_examples_abs_core_ge_50": n_core_ge_50,
            "validation_used_for_gate_selection": False, "validation_prediction_passes": 1,
            "test_examples_featurized": 0, "test_song_note_files_opened": FE.test_songs_opened(d),
            "sealed_test_metadata_only": {"examples": d["splits"]["test_example_count"], "clean_notes": d["splits"]["test_clean_note_count"]},
            "protected_paths": [FE.rel(x) for x in protected],
            "protected_hashes_before": hashes_before, "protected_hashes_after": hashes_after,
            "protected_unchanged": hashes_before == hashes_after, "runtime_s": time.time() - t0,
        },
    }
    if results["integrity"]["test_song_note_files_opened"]:
        raise RuntimeError("a test song's notes were opened")
    (out / "dynamic_hybrid_results.json").write_text(json.dumps(results, indent=2, default=float) + "\n")

    write_summary(results, out)
    print(json.dumps({"primary": primary, "protected_unchanged": results["integrity"]["protected_unchanged"]}, indent=1))
    return 0


def write_summary(results: Dict[str, Any], out: Path) -> None:
    """V2's report writer; on V3 only the fixed-blend label and benchmark wording are adapted."""
    dataset = results["experiment"].split("_ridge_natural")[0]
    v3 = dataset == "axis2_v3"
    w_fixed = float(results["gates"]["fixed"]["weight_ridge"])
    benchmark = float(results["validation"]["primary_question"]["fixed_hybrid"]["mae"])
    view = json.loads(json.dumps(results, default=float))
    if v3:
        for k in ("dynamic_gate", "magnitude_gate"):
            view["validation"]["primary_question"][k]["beats_fixed_mae_benchmark_12_5828"] = view["validation"]["primary_question"][k]["beats_fixed_hybrid_mae"]
    md = DH.summary_md(view)
    md = md.replace("# Axis 2 V2: dynamic Ridge / natural-offset gate", f"# Axis 2 ({dataset}): dynamic Ridge / natural-offset gate", 1)
    md = md.replace("Generated by `dynamic_hybrid.py`.", "Generated by `axis2_v3_dynamic_hybrid/dynamic_hybrid_frozen.py` (V2 procedure, frozen experts).", 1)
    if v3:
        fixed_label = f"Fixed {round(100 * w_fixed)}/{round(100 * (1 - w_fixed))}"
        md = md.replace(DH.LABEL["fixed"], fixed_label)
        md = md.replace("Beats 12.5828:", f"Beats the {dataset} fixed hybrid ({benchmark:.4f}):")
        md = md.replace(f"- Fixed: w = {w_fixed:.2f}.", f"- Fixed: w = {w_fixed:.2f} (selected on {dataset} train OOF).")
    (out / "dynamic_hybrid_summary.md").write_text(md)


if __name__ == "__main__":
    raise SystemExit(main())
